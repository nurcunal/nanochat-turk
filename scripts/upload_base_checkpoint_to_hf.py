"""
Upload a nanochat base-model checkpoint bundle to a Hugging Face model repo.

This uploads the raw nanochat checkpoint format, not a Transformers-compatible
conversion. It preserves the model checkpoint, optional optimizer shards,
tokenizer artifacts, report/eval outputs, logs, and provenance files.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Upload a nanochat base checkpoint to Hugging Face Hub")
    parser.add_argument("--repo-id", required=True, help="Hugging Face model repo id, e.g. user/model-name")
    parser.add_argument("--base-dir", default=os.environ.get("NANOCHAT_BASE_DIR", ""), help="nanochat base dir")
    parser.add_argument("--model-tag", default=os.environ.get("MODEL_TAG", ""), help="checkpoint model tag")
    parser.add_argument("--tokenizer-name", default=os.environ.get("NANOCHAT_TOKENIZER_NAME", ""))
    parser.add_argument("--step", default="latest", help="checkpoint step integer or 'latest'")
    parser.add_argument("--job-id", default=os.environ.get("TRAIN_JOBID", os.environ.get("SLURM_JOB_ID", "")))
    parser.add_argument("--cetvel-job-id", default=os.environ.get("CETVEL_JOBID", ""))
    parser.add_argument("--repo-prefix", default="", help="optional subdirectory in the HF repo")
    parser.add_argument("--private", action="store_true", help="create repo as private if it does not exist")
    parser.add_argument("--no-optimizer", action="store_true", help="do not upload optimizer shards")
    parser.add_argument("--dry-run", action="store_true", help="print files that would be uploaded and exit")
    parser.add_argument("--family-recipe", default="", help="sealed d32 family recipe; activates strict three-final upload mode")
    parser.add_argument("--preflight-receipt", default="")
    parser.add_argument("--attention-probe", default="")
    parser.add_argument("--wd-proxy-approval", default="")
    parser.add_argument("--static-launcher-gate", default="")
    parser.add_argument("--signal-resume-gate", default="")
    parser.add_argument("--production-gate", default="")
    parser.add_argument("--cluster-launch-receipt", default="")
    parser.add_argument("--lineage-dir", default="")
    parser.add_argument("--source-plan", default="")
    parser.add_argument("--backend-calibration", default="")
    parser.add_argument("--backend-resource-report", default="")
    parser.add_argument("--mixture-quality-approval", default="")
    parser.add_argument("--resource-approval", default="")
    parser.add_argument("--production-pack-plan", default="")
    parser.add_argument("--data-prep-storage-sample", default="")
    parser.add_argument("--writer-probe", default="")
    parser.add_argument("--data-prep-storage-gate", default="")
    parser.add_argument("--final-quality-approval", default="")
    parser.add_argument(
        "--family-final-optimizer-policy",
        choices=("include", "omit"),
        help="required in family mode; stable forks are always uploaded fully",
    )
    return parser.parse_args()


def run_command(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def repo_path(*parts):
    clean = [str(part).strip("/") for part in parts if str(part).strip("/")]
    return "/".join(clean)


def require_file(path, label):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def resolve_step(checkpoint_dir, requested):
    if requested != "latest":
        return int(requested)
    steps = []
    for path in checkpoint_dir.glob("model_*.pt"):
        match = re.match(r"model_(\d+)\.pt$", path.name)
        if match:
            steps.append(int(match.group(1)))
    if not steps:
        raise FileNotFoundError(f"No model_*.pt files found in {checkpoint_dir}")
    return max(steps)


def add_existing(files, local_path, path_in_repo):
    if local_path.is_file():
        files.append((local_path, path_in_repo))


def add_tree(files, local_dir, path_in_repo):
    if not local_dir.is_dir():
        return
    for path in sorted(local_dir.rglob("*")):
        if path.is_file():
            files.append((path, repo_path(path_in_repo, path.relative_to(local_dir))))


def _select_verified_tokenizer_uploads(
    tokenizer_root,
    *,
    expected_sha256,
    expected_name,
    expected_vocab_size,
    path_in_repo="tokenizer",
):
    """Verify and select only the sealed production-tokenizer inventory."""

    from nanochat.strict_tokenizer import EXPECTED_FILES, verify_tokenizer_package

    root = Path(tokenizer_root)
    manifest_path = root / "package_manifest.json"
    package = verify_tokenizer_package(
        manifest_path,
        expected_sha256=expected_sha256,
        expected_name=expected_name,
        expected_vocab_size=expected_vocab_size,
    )
    # verify_tokenizer_package requires an exact on-disk inventory (apart from
    # package_manifest.json) and validates every expected path/role. Build the
    # upload list from that closed allowlist instead of recursively walking the
    # directory, so an operator note, stale package, or secret can never become
    # an accidental Hub payload.
    selected = [(manifest_path, repo_path(path_in_repo, manifest_path.name))]
    selected.extend(
        (root / name, repo_path(path_in_repo, name))
        for name in sorted(EXPECTED_FILES)
    )
    return package, selected


def _snapshot_verified_tokenizer_uploads(
    tokenizer_root,
    snapshot_root,
    *,
    expected_sha256,
    expected_name,
    expected_vocab_size,
    path_in_repo="tokenizer",
):
    """Copy the six verified small files into a private upload snapshot."""

    source_package, source_files = _select_verified_tokenizer_uploads(
        tokenizer_root,
        expected_sha256=expected_sha256,
        expected_name=expected_name,
        expected_vocab_size=expected_vocab_size,
        path_in_repo=path_in_repo,
    )
    destination = Path(snapshot_root)
    if destination.exists():
        raise FileExistsError(f"tokenizer upload snapshot already exists: {destination}")
    destination.mkdir(parents=True)
    for source, _remote in source_files:
        target = destination / source.name
        target.write_bytes(source.read_bytes())
        target.chmod(0o400)
    snapshot_package, snapshot_files = _select_verified_tokenizer_uploads(
        destination,
        expected_sha256=expected_sha256,
        expected_name=expected_name,
        expected_vocab_size=expected_vocab_size,
        path_in_repo=path_in_repo,
    )
    if snapshot_package.canonical_sha256 != source_package.canonical_sha256:
        raise ValueError("tokenizer upload snapshot differs from verified source package")
    return snapshot_package, snapshot_files


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_entry(path, path_in_repo):
    stat = path.stat()
    return {
        "path_in_repo": path_in_repo,
        "local_path": str(path),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": sha256_file(path),
    }


def _metric_sort_key(item):
    key, _ = item
    preferred = [
        "acc",
        "acc_norm",
        "exact_match",
        "f1",
        "bleu",
        "chrf",
        "rougeL",
        "rouge1",
        "ter",
        "perplexity",
        "bpb",
    ]
    base = str(key).split(",")[0]
    try:
        return preferred.index(base)
    except ValueError:
        return len(preferred)


def summarize_cetvel(base_dir):
    cetvel_root = base_dir / "cetvel_out"
    if not cetvel_root.is_dir():
        return "No CETVEL output directory was found at upload time.\n"

    result_files = sorted(cetvel_root.glob("*/cetvel_*_results.json"))
    if not result_files:
        return f"CETVEL output directory exists, but no `cetvel_*_results.json` files were found under `{cetvel_root}`.\n"

    sections = []
    for result_file in result_files:
        try:
            with result_file.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            sections.append(f"- Could not parse `{result_file.relative_to(cetvel_root)}`: {exc}")
            continue

        suite = result_file.parent.name
        rows = []
        for task_name, task_metrics in sorted(payload.get("results", {}).items()):
            if not isinstance(task_metrics, dict):
                continue
            numeric = [(k, v) for k, v in task_metrics.items() if isinstance(v, (int, float))]
            if not numeric:
                continue
            metric_name, metric_value = sorted(numeric, key=_metric_sort_key)[0]
            rows.append((task_name, metric_name, metric_value))

        if not rows:
            sections.append(f"### CETVEL `{suite}`\n\nResults file: `{result_file.relative_to(base_dir)}`\n\nNo numeric task metrics were parsed.\n")
            continue

        lines = [
            f"### CETVEL `{suite}`",
            "",
            f"Results file: `{result_file.relative_to(base_dir)}`",
            "",
            "| Task | Metric | Value |",
            "|---|---:|---:|",
        ]
        for task_name, metric_name, metric_value in rows:
            lines.append(f"| `{task_name}` | `{metric_name}` | {metric_value:.6g} |")
        sections.append("\n".join(lines))

    return "\n\n".join(sections) + "\n"


def build_model_card(args, base_dir, checkpoint_dir, tokenizer_dir, step, files):
    git_commit = run_command(["git", "rev-parse", "HEAD"])
    git_branch = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    git_dirty = bool(run_command(["git", "status", "--porcelain"]))

    meta_path = checkpoint_dir / f"meta_{step:06d}.json"
    meta = {}
    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

    model_config = meta.get("model_config", {})
    user_config = meta.get("user_config", {})
    total_batch_size = meta.get("total_batch_size", user_config.get("total_batch_size", "unknown"))

    return f"""---
language:
- tr
tags:
- nanochat
- turkish
- pytorch
- raw-checkpoint
pipeline_tag: text-generation
library_name: pytorch
---

# nanochat Turkish `{args.model_tag}` Raw Checkpoint

This repository stores a raw nanochat checkpoint bundle. It is intended for
restoring or evaluating this repository's `nanochat` implementation. It is not
yet converted to the Hugging Face Transformers `from_pretrained` format.

## Checkpoint

- Model tag: `{args.model_tag}`
- Step: `{step}`
- Tokenizer: `{args.tokenizer_name}`
- Base dir on UHeM: `{base_dir}`
- Checkpoint dir on UHeM: `{checkpoint_dir}`
- Tokenizer dir on UHeM: `{tokenizer_dir}`
- Training job id: `{args.job_id or "unknown"}`
- CETVEL job id: `{args.cetvel_job_id or "unknown"}`

## Model Config

```json
{json.dumps(model_config, indent=2, sort_keys=True)}
```

## Training Config Highlights

- Depth: `{user_config.get("depth", model_config.get("n_layer", "unknown"))}`
- Vocab size: `{model_config.get("vocab_size", "unknown")}`
- Sequence length: `{model_config.get("sequence_len", user_config.get("max_seq_len", "unknown"))}`
- Device batch size: `{meta.get("device_batch_size", user_config.get("device_batch_size", "unknown"))}`
- Total batch size: `{total_batch_size}`
- Window pattern: `{model_config.get("window_pattern", user_config.get("window_pattern", "unknown"))}`

## Contents

The important files are:

- `checkpoint/model_{step:06d}.pt`
- `checkpoint/meta_{step:06d}.json`
- `checkpoint/optim_{step:06d}_rank*.pt` if optimizer shards were uploaded
- `tokenizer/tokenizer.pkl`
- `tokenizer/tokenizer_config.json`
- `tokenizer/token_bytes.pt`
- `report/` and `logs/` when available
- `cetvel_out/` when CETVEL has completed
- `provenance/upload_manifest.json`

## CETVEL

{summarize_cetvel(base_dir)}

## Provenance

- Git branch: `{git_branch or "unknown"}`
- Git commit: `{git_commit or "unknown"}`
- Git dirty at upload time: `{git_dirty}`
- Uploaded at: `{datetime.now(timezone.utc).isoformat()}`

## Caveat

This is a raw research checkpoint. Use the source code in this repository to
load it, or convert it separately before expecting Transformers-compatible
loading.
"""


def _family_model_card(recipe, optimizer_policy, manifest):
    finals = recipe["checkpoints"]["finals"]
    final_lines = "\n".join(
        f"- `{item['label']}`: `{item['model_tag']}` at update {item['final_step']:,} "
        f"({item['scheduled_tokens']:,} scheduled tokens)"
        for item in finals
    )
    return f"""---
language:
- tr
tags:
- nanochat
- turkish
- pytorch
- raw-checkpoint
pipeline_tag: text-generation
library_name: pytorch
---

# Turkish Nanochat d32 WSD Family

This private-first artifact contains the immutable 12x, 20x, and 40x cooled
base-model checkpoints from one shared stable WSD lineage. It is a raw Nanochat
checkpoint family, not a Transformers `from_pretrained` conversion.

{final_lines}

The tokenizer is `{recipe['artifacts']['tokenizer_name']}`; training data and validation
provenance are Turkish-only and code-corpus-free. Stable fork checkpoints are
always retained with full optimizer/loader/RNG state. Final optimizer policy:
`{optimizer_policy}`. When omitted, the upload manifest explicitly binds the
fully verified source transaction and lists every omitted role.

Family recipe SHA-256: `{manifest['family_recipe_sha256']}`
Training code revision: `{manifest['code_revision']}`
Manual final-model quality approval SHA-256: `{manifest.get('final_quality_approval_sha256', 'not supplied')}`
"""


def _add_strict_transaction(files, manifest, step_dir, remote_root, *, full):
    completion_path = require_file(step_dir / "completion.json", "strict completion")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Strict completion has no file inventory: {completion_path}")
    selected_roles = []
    omitted_roles = []
    for record in records:
        role = record.get("role")
        path = require_file(step_dir / record["path"], f"strict {role} payload")
        if full or role in {"model", "meta"}:
            files.append((path, repo_path(remote_root, path.name)))
            selected_roles.append(role)
        else:
            omitted_roles.append(role)
    completion_remote = (
        repo_path(remote_root, "completion.json")
        if full
        else repo_path(remote_root, "provenance", "source_completion.json")
    )
    files.append((completion_path, completion_remote))
    return {
        "source_step_dir": str(step_dir),
        "source_completion_sha256": manifest["canonical_sha256"],
        "retention_class": "full_resumable" if full else "model_metadata_only",
        "selected_roles": selected_roles,
        "omitted_roles": omitted_roles,
    }


def _verified_relative_evidence(root, record, label):
    """Resolve one hash-bound evidence record without following symlink escapes."""

    if not isinstance(record, dict):
        raise ValueError(f"{label} evidence record is malformed")
    relative = Path(str(record.get("path") or ""))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} evidence path is unsafe")
    root = Path(root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"{label} evidence root is unsafe or missing")
    unresolved = root / relative
    resolved = unresolved.resolve()
    if root not in resolved.parents:
        raise ValueError(f"{label} evidence escapes its receipt tree")
    current = unresolved
    while current != root:
        if current.is_symlink():
            raise ValueError(f"{label} evidence path is symlinked")
        current = current.parent
    require_file(resolved, f"{label} evidence")
    expected_size = record.get("size_bytes")
    expected_sha = record.get("sha256")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 0
        or not isinstance(expected_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None
        or resolved.stat().st_size != expected_size
        or sha256_file(resolved) != expected_sha
    ):
        raise ValueError(f"{label} evidence content drift")
    return resolved, relative


def _select_archived_mixture_audit_uploads(mixture, mixture_path, remote_parent):
    """Select the bounded, human-reviewed audit evidence needed off-cluster."""

    from nanochat.experiment_manifest import load_json_strict, verify_manifest_hash

    bundle = mixture.get("evidence_bundle")
    if not isinstance(bundle, dict) or bundle.get("schema_version") != "1.0":
        raise ValueError("mixture-quality approval evidence bundle is malformed")
    root_relative = Path(str(bundle.get("root") or ""))
    if (
        not str(bundle.get("root") or "")
        or root_relative.is_absolute()
        or ".." in root_relative.parts
    ):
        raise ValueError("mixture-quality evidence root is unsafe")
    approval_parent = Path(mixture_path).resolve().parent
    audit_root = (approval_parent / root_relative).resolve()
    if audit_root != approval_parent and approval_parent not in audit_root.parents:
        raise ValueError("mixture-quality evidence root escapes approval tree")
    remote_audit_root = repo_path(
        remote_parent,
        "" if root_relative.as_posix() == "." else root_relative.as_posix(),
    )

    report_path, report_relative = _verified_relative_evidence(
        audit_root, bundle.get("report"), "mixture-quality audit report"
    )
    report = load_json_strict(report_path)
    report_sha = verify_manifest_hash(report)
    if report_sha != mixture.get("sample_quality_audit_sha256"):
        raise ValueError("mixture-quality audit report differs from its approval")

    records = [(report_path, repo_path(remote_audit_root, report_relative))]
    input_artifacts = report.get("input_artifacts")
    example_sampling = report.get("example_sampling")
    example_files = (
        example_sampling.get("files") if isinstance(example_sampling, dict) else None
    )
    if not isinstance(input_artifacts, dict) or not isinstance(example_files, dict):
        raise ValueError("mixture-quality audit evidence inventory is malformed")
    for name in ("cluster_receipt", "object_launch_receipt", "bucket_launch_receipt"):
        path, relative = _verified_relative_evidence(
            audit_root, input_artifacts.get(name), f"sample audit {name}"
        )
        records.append((path, repo_path(remote_audit_root, relative)))
    for decision in ("accepted", "rejected"):
        decision_files = example_files.get(decision)
        if not isinstance(decision_files, dict):
            raise ValueError(f"sample audit {decision} example inventory is malformed")
        for representation in ("jsonl", "plaintext"):
            path, relative = _verified_relative_evidence(
                audit_root,
                decision_files.get(representation),
                f"sample audit {decision} {representation}",
            )
            records.append((path, repo_path(remote_audit_root, relative)))
    if len({remote for _path, remote in records}) != len(records):
        raise ValueError("mixture-quality audit evidence contains duplicate paths")
    return records


def _verify_family_data_controls(
    args,
    *,
    recipe,
    recipe_sha,
    preflight,
    repo_root,
    expected_chain,
):
    """Verify and select the exact v3 data-control chain used by preflight."""

    from nanochat.experiment_manifest import (
        canonical_json,
        load_json_strict,
        verify_manifest_hash,
    )
    from nanochat.turkish_backend import (
        MIXTURE_QUALITY_APPROVAL_KIND,
        RESOURCE_APPROVAL_KIND,
        RESOURCE_REPORT_KIND,
        validate_backend_calibration,
        validate_mixture_quality_approval,
        validate_resource_approval,
        validate_resource_projection,
        validate_source_plan,
    )
    from nanochat.turkish_corpus import load_corpus_policy
    from scripts.d32_family_workflow import (
        DATA_PREP_PACK_PLAN_KIND,
        DATA_PREP_STORAGE_SAMPLE_KIND,
        DATA_PREP_WRITER_PROBE_KIND,
        _load_receipt,
        _validate_data_prep_storage_sample,
        _validate_data_prep_storage_gate_receipt,
        _validate_production_pack_plan,
        _validate_recipe_policy_identity,
        _validate_storage_approval_evidence,
        _validate_writer_probe,
    )

    policy_path = (repo_root / recipe["artifacts"]["mixture_config"]).resolve()
    policy = load_corpus_policy(policy_path)
    _validate_recipe_policy_identity(recipe, policy)
    policy_sha = hashlib.sha256(canonical_json(policy).encode("utf-8")).hexdigest()

    source_plan_path = require_file(Path(args.source_plan).resolve(), "source plan")
    calibration_path = require_file(
        Path(args.backend_calibration).resolve(), "backend calibration"
    )
    source_plan = load_json_strict(source_plan_path)
    calibration = load_json_strict(calibration_path)
    validate_source_plan(source_plan, policy)
    validate_backend_calibration(calibration, policy)
    source_plan_sha = verify_manifest_hash(source_plan)
    calibration_sha = verify_manifest_hash(calibration)

    mixture_path = Path(args.mixture_quality_approval).resolve()
    mixture, mixture_sha = _load_receipt(
        mixture_path, MIXTURE_QUALITY_APPROVAL_KIND
    )
    validate_mixture_quality_approval(
        mixture,
        policy=policy,
        plan=source_plan,
        calibration=calibration,
        approval_path=mixture_path,
    )
    resource_path = Path(args.resource_approval).resolve()
    resource, resource_sha = _load_receipt(resource_path, RESOURCE_APPROVAL_KIND)
    validate_resource_approval(
        resource,
        plan=source_plan,
        policy=policy,
        calibration=calibration,
        approval_path=resource_path,
    )
    if resource.get("mixture_quality_approval_sha256") != mixture_sha:
        raise ValueError("resource approval does not bind the supplied quality approval")

    backend_report_path = Path(args.backend_resource_report).resolve()
    backend_report, backend_report_sha = _load_receipt(
        backend_report_path, RESOURCE_REPORT_KIND
    )
    validate_resource_projection(backend_report, plan=source_plan)
    if (
        backend_report_sha != resource.get("resource_report_sha256")
        or backend_report.get("policy_sha256") != policy_sha
        or backend_report.get("source_plan_sha256") != source_plan_sha
        or backend_report.get("calibration_sha256") != calibration_sha
        or backend_report.get("sample_cluster_receipt_sha256")
        != mixture.get("sample_cluster_receipt_sha256")
    ):
        raise ValueError("backend resource report does not form the approval chain")

    pack_path = Path(args.production_pack_plan).resolve()
    pack_plan, pack_sha = _load_receipt(pack_path, DATA_PREP_PACK_PLAN_KIND)
    _validate_production_pack_plan(
        pack_plan,
        recipe=recipe,
        recipe_sha=recipe_sha,
        policy_sha=policy_sha,
        source_plan=source_plan,
        source_plan_sha=source_plan_sha,
    )
    storage_sample_path = Path(args.data_prep_storage_sample).resolve()
    storage_sample, storage_sample_sha = _load_receipt(
        storage_sample_path, DATA_PREP_STORAGE_SAMPLE_KIND
    )
    _validate_data_prep_storage_sample(
        storage_sample, recipe=recipe, recipe_sha=recipe_sha
    )
    _validate_storage_approval_evidence(
        storage_sample,
        measurement_path=storage_sample_path,
        policy_path=policy_path,
    )
    writer_path = Path(args.writer_probe).resolve()
    writer_probe, writer_sha = _load_receipt(
        writer_path, DATA_PREP_WRITER_PROBE_KIND
    )
    _validate_writer_probe(
        writer_probe,
        recipe=recipe,
        recipe_sha=recipe_sha,
        policy_sha=policy_sha,
        source_plan_sha=source_plan_sha,
        calibration_sha=calibration_sha,
        backend_report_sha=backend_report_sha,
        cluster_sha=mixture["sample_cluster_receipt_sha256"],
        sample_documents=storage_sample["sample_documents"],
        estimated_total_documents=storage_sample["estimated_total_documents"],
    )
    storage_path = Path(args.data_prep_storage_gate).resolve()
    storage_gate, storage_sha = _load_receipt(
        storage_path, "d32_data_prep_storage_gate"
    )
    _validate_data_prep_storage_gate_receipt(
        storage_gate, recipe=recipe, recipe_sha=recipe_sha
    )

    expected = {
        "policy_sha256": policy_sha,
        "source_plan_sha256": source_plan_sha,
        "calibration_sha256": calibration_sha,
        "backend_resource_report_sha256": backend_report_sha,
        "resource_approval_sha256": resource_sha,
        "mixture_quality_approval_sha256": mixture_sha,
        "sample_quality_audit_sha256": mixture["sample_quality_audit_sha256"],
        "production_pack_plan_sha256": pack_sha,
    }
    if preflight.get("data_preparation_provenance") != expected:
        raise ValueError("upload data controls differ from production preflight")
    if (
        preflight.get("data_preparation_storage_gate_sha256") != storage_sha
        or storage_gate.get("policy_sha256") != policy_sha
        or storage_gate.get("source_plan_sha256") != source_plan_sha
        or storage_gate.get("calibration_sha256") != calibration_sha
        or storage_gate.get("backend_resource_report_sha256") != backend_report_sha
        or storage_gate.get("resource_approval_sha256") != resource_sha
        or storage_gate.get("mixture_quality_approval_sha256") != mixture_sha
        or storage_gate.get("production_pack_plan_sha256") != pack_sha
        or storage_gate.get("sample_measurement_sha256") != storage_sample_sha
        or storage_gate.get("writer_probe_sha256") != writer_sha
        or storage_sample.get("policy_sha256") != policy_sha
        or storage_sample.get("source_plan_sha256") != source_plan_sha
        or storage_sample.get("calibration_sha256") != calibration_sha
        or storage_sample.get("backend_resource_report_sha256") != backend_report_sha
        or storage_sample.get("resource_approval_sha256") != resource_sha
        or storage_sample.get("mixture_quality_approval_sha256") != mixture_sha
        or storage_sample.get("sample_quality_audit_sha256")
        != mixture["sample_quality_audit_sha256"]
        or storage_sample.get("sample_cluster_receipt_sha256")
        != mixture["sample_cluster_receipt_sha256"]
        or storage_sample.get("production_pack_plan_sha256") != pack_sha
        or storage_sample.get("writer_probe_sha256") != writer_sha
        or expected_chain.get("data_prep_storage_gate_sha256") != storage_sha
        or expected_chain.get("production_pack_plan_sha256") != pack_sha
        or expected_chain.get("resource_approval_sha256") != resource_sha
        or expected_chain.get("mixture_quality_approval_sha256") != mixture_sha
    ):
        raise ValueError("upload data controls do not form the preflight production chain")

    archive_root = storage_sample_path.parent.resolve()
    archive_remote_root = repo_path("provenance", "data_controls")
    approval_evidence = storage_sample.get("approval_evidence")
    if not isinstance(approval_evidence, dict):
        raise ValueError("storage sample approval-evidence inventory is malformed")
    primary = {
        "source_plan": source_plan_path,
        "calibration": calibration_path,
        "backend_resource_report": backend_report_path,
        "resource_approval": resource_path,
        "mixture_quality_approval": mixture_path,
    }
    uploads = []
    primary_remotes = {}
    for key, supplied_path in primary.items():
        evidence_path, relative = _verified_relative_evidence(
            archive_root, approval_evidence.get(key), f"storage sample {key}"
        )
        if evidence_path != supplied_path:
            raise ValueError(f"supplied {key} is not the storage sample evidence file")
        remote = repo_path(archive_remote_root, relative)
        primary_remotes[key] = remote
        uploads.append((supplied_path, remote))

    resource_bundle = resource.get("evidence_bundle")
    if not isinstance(resource_bundle, dict):
        raise ValueError("resource approval evidence bundle is malformed")
    resource_remote_parent = Path(primary_remotes["resource_approval"]).parent.as_posix()
    for key, supplied_path in (
        ("resource_report", backend_report_path),
        ("mixture_quality_approval", mixture_path),
    ):
        evidence_path, relative = _verified_relative_evidence(
            resource_path.parent, resource_bundle.get(key), f"resource approval {key}"
        )
        expected_remote = repo_path(resource_remote_parent, relative)
        supplied_remote = primary_remotes[
            "backend_resource_report" if key == "resource_report" else key
        ]
        if evidence_path != supplied_path or expected_remote != supplied_remote:
            raise ValueError(
                f"resource approval {key} link would not be portable in the upload"
            )

    uploads.extend(
        (
            (pack_path, repo_path(archive_remote_root, "production_source_pack_plan.json")),
            (
                storage_sample_path,
                repo_path(archive_remote_root, "d32_data_prep_storage_sample.json"),
            ),
            (writer_path, repo_path(archive_remote_root, "post_cluster_writer_probe.json")),
            (storage_path, repo_path(archive_remote_root, "data_prep_storage_gate.json")),
        )
    )
    mixture_remote_parent = Path(
        primary_remotes["mixture_quality_approval"]
    ).parent.as_posix()
    uploads.extend(
        _select_archived_mixture_audit_uploads(
            mixture,
            mixture_path,
            mixture_remote_parent,
        )
    )
    return expected | {
        "backend_resource_report_sha256": backend_report_sha,
        "data_prep_storage_sample_sha256": storage_sample_sha,
        "writer_probe_sha256": writer_sha,
        "data_prep_storage_gate_sha256": storage_sha,
    }, uploads


def main_family(args):
    if not args.family_final_optimizer_policy:
        raise ValueError("--family-final-optimizer-policy=include|omit is required")
    required_args = {
        "base-dir": args.base_dir,
        "preflight-receipt": args.preflight_receipt,
        "attention-probe": args.attention_probe,
        "wd-proxy-approval": args.wd_proxy_approval,
        "static-launcher-gate": args.static_launcher_gate,
        "signal-resume-gate": args.signal_resume_gate,
        "production-gate": args.production_gate,
        "cluster-launch-receipt": args.cluster_launch_receipt,
        "lineage-dir": args.lineage_dir,
        "source-plan": getattr(args, "source_plan", ""),
        "backend-calibration": getattr(args, "backend_calibration", ""),
        "backend-resource-report": getattr(args, "backend_resource_report", ""),
        "mixture-quality-approval": getattr(args, "mixture_quality_approval", ""),
        "resource-approval": getattr(args, "resource_approval", ""),
        "production-pack-plan": getattr(args, "production_pack_plan", ""),
        "data-prep-storage-sample": getattr(args, "data_prep_storage_sample", ""),
        "writer-probe": getattr(args, "writer_probe", ""),
        "data-prep-storage-gate": getattr(args, "data_prep_storage_gate", ""),
        "final-quality-approval": getattr(args, "final_quality_approval", ""),
    }
    missing = [name for name, value in required_args.items() if not value]
    if missing:
        raise ValueError("family upload is missing: " + ", ".join(missing))

    from nanochat.strict_checkpoint import inspect_strict_checkpoint
    from nanochat.experiment_manifest import seal_manifest, verify_manifest_hash
    from nanochat.tokenizer_quality import validate_tokenizer_quality_gate
    from scripts.d32_family_workflow import (
        FINAL_MODEL_PUBLICATION_APPROVAL_KIND,
        _load_receipt,
        _verify_attention_probe,
        _verify_gate_and_preflight,
        _verify_signal_resume_gate,
        _verify_static_launcher_gate,
        collect_final_evaluation_evidence,
        load_recipe,
        validate_final_model_publication_approval,
    )

    repo_root = Path.cwd().resolve()
    base_dir = Path(args.base_dir).expanduser().resolve()
    recipe_path = Path(args.family_recipe).resolve()
    recipe, recipe_sha = load_recipe(recipe_path)
    preflight, preflight_sha = _load_receipt(
        Path(args.preflight_receipt), "d32_family_preflight_receipt"
    )
    cluster_launch, cluster_launch_sha = _load_receipt(
        Path(args.cluster_launch_receipt),
        "turkish_packed_production_cluster_launch_receipt",
    )
    if (
        preflight.get("production_cluster_launch_receipt_sha256")
        != cluster_launch_sha
        or cluster_launch.get("cluster_completed") is not True
    ):
        raise ValueError("upload cluster launch differs from production preflight")
    _probe, probe_sha = _verify_attention_probe(
        Path(args.attention_probe),
        recipe=recipe,
        recipe_sha=recipe_sha,
        preflight_sha=preflight_sha,
        code_revision=preflight["code"]["git_commit"],
    )
    _static_gate, static_gate_sha = _verify_static_launcher_gate(
        Path(args.static_launcher_gate),
        recipe=recipe,
        recipe_sha=recipe_sha,
        preflight=preflight,
        preflight_sha=preflight_sha,
    )
    _signal_gate, signal_gate_sha = _verify_signal_resume_gate(
        Path(args.signal_resume_gate),
        recipe=recipe,
        recipe_sha=recipe_sha,
        preflight=preflight,
        preflight_sha=preflight_sha,
    )
    (
        _preflight,
        _preflight_sha,
        gate,
        gate_sha,
        approval,
        approval_sha,
    ) = _verify_gate_and_preflight(
        recipe,
        recipe_sha,
        Path(args.preflight_receipt),
        Path(args.production_gate),
        Path(args.wd_proxy_approval),
        Path(args.attention_probe),
    )
    if gate.get("passed") is not True or gate.get(
        "authorized_production_world_size"
    ) not in {8, 16}:
        raise ValueError("family upload requires a passed ws8/ws16 topology gate")
    if gate.get("signal_resume_gate_sha256") != signal_gate_sha:
        raise ValueError("family upload signal/resume gate differs from production")
    smoke_root = Path(args.production_gate).resolve().parent
    smoke_receipts = []
    for filename, hash_field, expected_world_size in (
        ("smoke_ws8.json", "smoke_8gpu_sha256", 8),
        ("smoke_ws16.json", "smoke_16gpu_sha256", 16),
    ):
        expected_sha = gate.get(hash_field)
        if expected_sha is None:
            if expected_world_size == 8:
                raise ValueError("production gate does not bind the required ws8 smoke")
            continue
        smoke_path = smoke_root / filename
        smoke, smoke_sha = _load_receipt(
            smoke_path, "d32_distributed_smoke_receipt"
        )
        if smoke_sha != expected_sha or smoke.get("world_size") != expected_world_size:
            raise ValueError(f"{filename} differs from the production topology gate")
        smoke_receipts.append(smoke_path)
    revision = run_command(["git", "rev-parse", "HEAD"])
    if revision != preflight["code"]["git_commit"]:
        raise ValueError("upload code revision differs from production preflight")
    if run_command(["git", "status", "--porcelain"]):
        raise ValueError("family upload requires the same clean Git worktree as training")

    files = []
    retention = {"stable_forks": [], "finals": []}
    checkpoint_manifests = {}
    trunk_tag = recipe["checkpoints"]["trunk_model_tag"]
    for fork in recipe["checkpoints"]["stable_forks"]:
        step = int(fork["step"])
        step_dir = base_dir / "base_checkpoints" / trunk_tag / f"strict_{step:06d}"
        manifest = inspect_strict_checkpoint(step_dir.parent, step)
        checkpoint_manifests[(trunk_tag, step)] = (
            manifest,
            verify_manifest_hash(manifest),
        )
        retention["stable_forks"].append(
            {
                "model_tag": trunk_tag,
                "step": step,
                **_add_strict_transaction(
                    files,
                    manifest,
                    step_dir,
                    repo_path("stable_forks", f"step_{step:06d}"),
                    full=True,
                ),
            }
        )

    include_final_state = args.family_final_optimizer_policy == "include"
    for final in recipe["checkpoints"]["finals"]:
        label = final["label"]
        model_tag = final["model_tag"]
        step = int(final["final_step"])
        step_dir = base_dir / "base_checkpoints" / model_tag / f"strict_{step:06d}"
        manifest = inspect_strict_checkpoint(step_dir.parent, step)
        checkpoint_manifests[(model_tag, step)] = (
            manifest,
            verify_manifest_hash(manifest),
        )
        retention["finals"].append(
            {
                "label": label,
                "model_tag": model_tag,
                "step": step,
                **_add_strict_transaction(
                    files,
                    manifest,
                    step_dir,
                    repo_path("finals", label),
                    full=include_final_state,
                ),
            }
        )

    lineage_dir = Path(args.lineage_dir).resolve()
    lineage_hashes = {}
    lineage_records = {}
    for stage in recipe["stages"]:
        path = lineage_dir / f"{stage['id']}.json"
        receipt, digest = _load_receipt(path, "d32_checkpoint_lineage_receipt")
        stage_id = stage["id"]
        target_key = (str(stage["model_tag"]), int(stage["target_step"]))
        target_manifest, target_sha = checkpoint_manifests[target_key]
        target_identity = target_manifest.get("identity")
        if not isinstance(target_identity, dict):
            raise ValueError(f"target checkpoint identity is malformed: {stage_id}")
        protocol = target_identity.get("protocol")
        if not isinstance(protocol, dict):
            raise ValueError(f"target checkpoint protocol is malformed: {stage_id}")
        exposure_key = (
            f"{stage['exposure_plan_family']}_ws"
            f"{gate['authorized_production_world_size']}_seed42"
        )
        exposure_sha = preflight["corpus"]["training_exposure_plans"][
            exposure_key
        ]["sha256"]
        expected_run_id = (
            f"{recipe['family_id']}_trunk"
            if stage["kind"] == "trunk"
            else f"{recipe['family_id']}_{stage_id}"
        )
        expected_recipe_scope = (
            "production_trunk" if stage["kind"] == "trunk" else stage_id
        )
        expected_retention_class = (
            "full_resumable_stable_fork"
            if stage["kind"] == "trunk"
            else (
                "cooled_final_full_resumable_retained"
                if recipe["family_id"] == "tr_d32_general_bpe32k_v3"
                else "cooled_final_full_transaction_pending_explicit_export_policy"
            )
        )
        source_step = stage.get("source_step")
        expected_source = None
        if source_step is not None:
            source_key = (
                str(stage.get("source_model_tag", stage["model_tag"])),
                int(source_step),
            )
            source_manifest, source_sha = checkpoint_manifests[source_key]
            source_identity = source_manifest.get("identity")
            if not isinstance(source_identity, dict):
                raise ValueError(f"source checkpoint identity is malformed: {stage_id}")
            expected_source = {
                "model_tag": source_key[0],
                "step": source_key[1],
                "checkpoint_sha256": source_sha,
                "run_id": source_identity.get("run_id"),
            }
        if (
            receipt.get("family_id") != recipe["family_id"]
            or receipt.get("stage_id") != stage_id
            or receipt.get("recipe_sha256") != recipe_sha
            or receipt.get("preflight_receipt_sha256") != preflight_sha
            or receipt.get("production_gate_sha256") != gate_sha
            or receipt.get("wsd_proxy_approval_sha256") != approval_sha
            or receipt.get("production_world_size")
            != gate["authorized_production_world_size"]
            or receipt.get("exposure_plan_key") != exposure_key
            or receipt.get("source") != expected_source
            or receipt.get("target", {}).get("model_tag") != target_key[0]
            or receipt.get("target", {}).get("step") != target_key[1]
            or receipt.get("target", {}).get("checkpoint_sha256") != target_sha
            or receipt.get("target", {}).get("retention_class")
            != expected_retention_class
            or target_manifest.get("expected_world_size")
            != gate["authorized_production_world_size"]
            or target_identity.get("run_id") != expected_run_id
            or target_identity.get("study_manifest_sha256") != recipe_sha
            or target_identity.get("tokenizer_artifact_sha256")
            != preflight["tokenizer"]["package_manifest_sha256"]
            or target_identity.get("exposure_plan_sha256") != exposure_sha
            or protocol.get("protocol_version") != "d32_wsd_strict_v1"
            or protocol.get("run_kind") != "production"
            or protocol.get("recipe_scope") != expected_recipe_scope
            or protocol.get("model_tag") != target_key[0]
            or protocol.get("num_iterations") != int(stage["num_iterations"])
        ):
            raise ValueError(f"lineage receipt mismatch: {path}")
        lineage_hashes[stage["id"]] = digest
        lineage_records[stage["id"]] = (receipt, digest)
        files.append((path, repo_path("provenance", "lineage", path.name)))

    final_evaluations = collect_final_evaluation_evidence(
        recipe=recipe,
        recipe_sha=recipe_sha,
        preflight=preflight,
        preflight_sha=preflight_sha,
        gate=gate,
        gate_sha=gate_sha,
        base_dir=base_dir,
        lineage_dir=lineage_dir,
        checkpoint_records=checkpoint_manifests,
        lineage_records=lineage_records,
    )
    final_quality_path = Path(args.final_quality_approval).resolve()
    final_quality, _final_quality_sha = _load_receipt(
        final_quality_path, FINAL_MODEL_PUBLICATION_APPROVAL_KIND
    )
    final_quality_sha = validate_final_model_publication_approval(
        final_quality,
        recipe=recipe,
        recipe_sha=recipe_sha,
        preflight_sha=preflight_sha,
        gate_sha=gate_sha,
        expected_evidence=final_evaluations,
        require_accepted=True,
    )
    files.append(
        (
            final_quality_path,
            repo_path("provenance", "control", "final_quality_approval.json"),
        )
    )

    corpus_root = Path(preflight["corpus"]["root"])
    expected_chain = preflight["corpus"].get("production_chain")
    if (
        not isinstance(expected_chain, dict)
        or expected_chain.get("data_prep_storage_gate_sha256")
        != preflight["data_preparation_storage_gate_sha256"]
        or expected_chain.get("cluster_launch_receipt_sha256")
        != cluster_launch_sha
        or expected_chain.get("sample_cluster_receipt_sha256")
        != cluster_launch.get("sample_cluster_receipt_sha256")
    ):
        raise ValueError("production preflight has malformed data lineage")
    data_control_hashes, data_control_uploads = _verify_family_data_controls(
        args,
        recipe=recipe,
        recipe_sha=recipe_sha,
        preflight=preflight,
        repo_root=repo_root,
        expected_chain=expected_chain,
    )
    files.extend(data_control_uploads)
    corpus_manifest, corpus_manifest_sha = _load_receipt(
        corpus_root / recipe["artifacts"]["corpus_manifest"],
        "turkish_pretrain_corpus",
    )
    parent_pool, parent_pool_sha = _load_receipt(
        corpus_root / "parent_pool_manifest.json", "turkish_pretrain_corpus"
    )
    qa_approval, qa_approval_sha = _load_receipt(
        corpus_root / "qa" / "qa_approval.json", "turkish_pretrain_qa_approval"
    )
    parent_ownership, parent_ownership_sha = _load_receipt(
        corpus_root / "parent_pool_ownership.json", "turkish_run_owned_filtered_pool"
    )
    qa_report, qa_report_sha = _load_receipt(
        corpus_root / "qa" / "qa_report.json",
        "turkish_pretrain_stratified_qa_report",
    )
    packing_report, packing_report_sha = _load_receipt(
        corpus_root / "packing_preflight" / "packing_preflight_report.json",
        "turkish_packing_preflight_report",
    )
    packing_approval, packing_approval_sha = _load_receipt(
        corpus_root / "packing_preflight" / "packing_preflight_approval.json",
        "turkish_packing_preflight_approval",
    )
    final_dataset_path = require_file(
        corpus_root / recipe["artifacts"]["nanochat_dataset_manifest"],
        "final dataset manifest",
    )
    final_dataset = json.loads(final_dataset_path.read_text(encoding="utf-8"))
    final_dataset_sha = verify_manifest_hash(final_dataset)
    if (
        corpus_manifest_sha != preflight["corpus"]["manifest_sha256"]
        or parent_pool_sha != preflight["corpus"]["parent_pool_manifest_sha256"]
        or qa_approval_sha != preflight["corpus"]["qa_approval_sha256"]
        or final_dataset_sha != preflight["corpus"]["dataset_manifest_sha256"]
        or corpus_manifest.get("production_chain") != expected_chain
        or parent_pool.get("production_chain") != expected_chain
        or corpus_manifest.get("parent_pool_manifest_sha256") != parent_pool_sha
        or final_dataset.get("metadata", {}).get("parent_pool_manifest_sha256")
        != parent_pool_sha
        or final_dataset.get("metadata", {}).get("qa_approval_sha256")
        != qa_approval_sha
        or final_dataset.get("metadata", {}).get("production_chain")
        != expected_chain
        or qa_approval.get("decision") != "accepted"
        or parent_ownership.get("pool_manifest_sha256") != parent_pool_sha
        or qa_approval.get("qa_report_sha256") != qa_report_sha
        or corpus_manifest.get("quality_assurance", {}).get("report_sha256")
        != qa_report_sha
        or corpus_manifest.get("quality_assurance", {}).get("approval_sha256")
        != qa_approval_sha
        or corpus_manifest.get("packing_preflight", {}).get("report_sha256")
        != packing_report_sha
        or corpus_manifest.get("packing_preflight", {}).get("approval_sha256")
        != packing_approval_sha
        or packing_approval.get("packing_report_sha256") != packing_report_sha
    ):
        raise ValueError("final corpus differs from exact preflight pool/QA lineage")

    tokenizer_name = preflight["tokenizer"]["name"]
    tokenizer_root = Path(preflight["tokenizer"]["root"])
    verified_tokenizer, tokenizer_uploads = _select_verified_tokenizer_uploads(
        tokenizer_root,
        expected_sha256=preflight["tokenizer"]["package_manifest_sha256"],
        expected_name=tokenizer_name,
        expected_vocab_size=recipe["model"]["vocab_size"],
    )
    files.extend(tokenizer_uploads)
    tokenizer_control_root = base_dir / "control" / "tokenizer" / tokenizer_name
    quality_root = tokenizer_control_root / "quality"
    quality_report, quality_approval = validate_tokenizer_quality_gate(
        quality_root,
        expected_package_sha256=preflight["tokenizer"]["package_manifest_sha256"],
        expected_training_receipt_sha256=verified_tokenizer.manifest[
            "training_receipt_sha256"
        ],
        expected_production_chain=expected_chain,
    )
    for name in ("quality_report.json", "quality_approval.json"):
        path = require_file(quality_root / name, f"tokenizer quality evidence {name}")
        files.append((path, repo_path("provenance", "tokenizer", "quality", name)))

    sample_root = tokenizer_control_root / "sample"
    training_receipt = _load_receipt(
        tokenizer_root / "training_receipt.json", "turkish_raw_bpe_training_receipt"
    )[0]
    tokenizer_package, tokenizer_package_sha = _load_receipt(
        tokenizer_root / "package_manifest.json", "turkish_raw_bpe_tokenizer_package"
    )
    sample_manifest, sample_sha = _load_receipt(
        sample_root / "tokenizer_sample_manifest.json", "turkish_raw_bpe_training_sample"
    )
    dataset_manifest = require_file(
        sample_root / "fineweb2_manifest.json", "tokenizer sample dataset manifest"
    )
    dataset_payload = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    dataset_sha = verify_manifest_hash(dataset_payload)
    if (
        training_receipt.get("sample_manifest_sha256") != sample_sha
        or sample_manifest.get("nanochat_dataset_manifest_sha256") != dataset_sha
        or tokenizer_package_sha
        != preflight["tokenizer"]["package_manifest_sha256"]
        or any(
            receipt.get("production_chain") != expected_chain
            or receipt.get("parent_corpus_manifest_sha256") != parent_pool_sha
            or receipt.get("qa_approval_sha256") != qa_approval_sha
            for receipt in (
                training_receipt,
                tokenizer_package,
                sample_manifest,
                quality_report,
                quality_approval,
            )
        )
        or dataset_payload.get("metadata", {}).get("production_chain")
        != expected_chain
        or dataset_payload.get("metadata", {}).get(
            "parent_corpus_manifest_sha256"
        )
        != parent_pool_sha
        or dataset_payload.get("metadata", {}).get("qa_approval_sha256")
        != qa_approval_sha
    ):
        raise ValueError("tokenizer sample provenance differs from its training receipt")
    for path in (sample_root / "tokenizer_sample_manifest.json", dataset_manifest):
        files.append(
            (path, repo_path("provenance", "tokenizer", "sample", path.name))
        )
    source_receipt_path = require_file(
        corpus_root / recipe["artifacts"]["source_receipt"],
        "family source receipt",
    )
    source_receipt, source_receipt_sha = _load_receipt(
        source_receipt_path, "turkish_pretrain_source_receipt"
    )
    if source_receipt_sha != preflight["corpus"].get("source_receipt_sha256"):
        raise ValueError("family source receipt differs from production preflight")
    derived_sources = source_receipt.get("derived_sources")
    if not isinstance(derived_sources, dict):
        raise ValueError("family source receipt has malformed derived-source provenance")
    provenance_names = {
        recipe["artifacts"]["corpus_manifest"],
        recipe["artifacts"]["nanochat_dataset_manifest"],
        recipe["artifacts"]["source_receipt"],
        recipe["artifacts"]["validation_exposure_manifest"],
        recipe["artifacts"]["exposure_plan_index"],
        recipe["artifacts"]["packing_capacity_receipt"],
        *recipe["artifacts"]["training_exposure_manifests"].values(),
    }
    for name in sorted(provenance_names):
        path = require_file(corpus_root / name, f"corpus provenance {name}")
        files.append((path, repo_path("provenance", "data", name)))
    files.append(
        (
            corpus_root / "parent_pool_manifest.json",
            repo_path("provenance", "data", "parent_pool_manifest.json"),
        )
    )
    files.append(
        (
            corpus_root / "parent_pool_ownership.json",
            repo_path("provenance", "data", "parent_pool_ownership.json"),
        )
    )
    files.append(
        (
            corpus_root / "qa" / "qa_report.json",
            repo_path("provenance", "data", "qa", "qa_report.json"),
        )
    )
    for record in qa_report["examples"].values():
        path = require_file(
            corpus_root / "qa" / record["path"], "archived QA reviewed example"
        )
        if path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError("archived QA reviewed example differs from report")
        files.append(
            (path, repo_path("provenance", "data", "qa", path.name))
        )
    for name in (
        "packing_preflight_report.json",
        "packing_preflight_approval.json",
    ):
        files.append(
            (
                corpus_root / "packing_preflight" / name,
                repo_path("provenance", "data", "packing_preflight", name),
            )
        )
    files.append(
        (
            corpus_root / "qa" / "qa_approval.json",
            repo_path("provenance", "data", "qa", "qa_approval.json"),
        )
    )
    macocu_manifest = recipe["artifacts"].get("macocu_preparation_manifest")
    if macocu_manifest:
        path = require_file(base_dir / macocu_manifest, "MaCoCu preparation manifest")
        manifest, manifest_sha = _load_receipt(
            path, "turkish_macocu_genre_preparation"
        )
        from nanochat.turkish_backend import validate_macocu_preparation_manifest
        from nanochat.turkish_corpus import load_corpus_policy

        data_policy = load_corpus_policy(
            repo_root / recipe["artifacts"]["mixture_config"]
        )
        validate_macocu_preparation_manifest(
            manifest,
            data_policy,
            path.parent,
            verify_files=True,
        )
        source_macocu = derived_sources.get("macocu_genre_tr")
        preflight_macocu = preflight["corpus"].get(
            "macocu_preparation_manifest"
        )
        if (
            not isinstance(source_macocu, dict)
            or not isinstance(preflight_macocu, dict)
            or source_macocu.get("manifest_sha256") != manifest_sha
            or preflight_macocu.get("sha256") != manifest_sha
            or Path(str(preflight_macocu.get("path", ""))).resolve()
            != path.resolve()
            or manifest.get("canonical_sha256") != manifest_sha
        ):
            raise ValueError(
                "MaCoCu preparation manifest differs from source receipt/preflight"
            )
        files.append(
            (
                path,
                repo_path("provenance", "data", "macocu_preparation_manifest.json"),
            )
        )
    elif derived_sources:
        raise ValueError("v1 family source receipt unexpectedly contains derived data")
    from scripts.d32_family_workflow import _verify_anchor_preparation_binding

    for source_id, artifact_key in (
        ("mot_tr_v1_11", "mot_preparation_manifest"),
        ("parlamint_tr_v5_0", "parlamint_preparation_manifest"),
    ):
        relative = recipe["artifacts"].get(artifact_key)
        if relative is None:
            if source_id in derived_sources:
                raise ValueError(
                    f"family source receipt binds {source_id} without {artifact_key}"
                )
            continue
        path = require_file(base_dir / relative, f"{source_id} preparation manifest")
        record = _verify_anchor_preparation_binding(
            path,
            source_id=source_id,
            derived_sources=derived_sources,
        )
        if preflight["corpus"].get(artifact_key) != record:
            raise ValueError(
                f"{source_id} preparation manifest differs from source receipt/preflight"
            )
        files.append(
            (
                path,
                repo_path("provenance", "data", f"{artifact_key}.json"),
            )
        )
    control_files = [
        recipe_path,
        Path(args.preflight_receipt),
        Path(args.attention_probe),
        Path(args.wd_proxy_approval),
        Path(args.static_launcher_gate),
        Path(args.signal_resume_gate),
        Path(args.production_gate),
        Path(args.cluster_launch_receipt),
        *smoke_receipts,
        repo_root / recipe["artifacts"]["mixture_config"],
        repo_root / "pyproject.toml",
        repo_root / "uv.lock",
    ]
    for path in control_files:
        require_file(path, "family control/provenance file")
        files.append((path, repo_path("provenance", "control", path.name)))
    code_paths = set(recipe["code_provenance"]["core_scope"])
    code_paths.update(
        {
            "scripts/d32_family_workflow.py",
            "scripts/d32_attention_probe.py",
            "scripts/d32_static_launch_probe.py",
            "scripts/d32_wsd_train.py",
            "scripts/build_turkish_pretrain_corpus.py",
            "scripts/audit_turkish_backend_sample.py",
            "scripts/review_turkish_tokenizer_quality.py",
            "scripts/train_turkish_raw_bpe.py",
            "scripts/turkish_packed_sample.py",
            "scripts/turkish_packed_production.py",
            "scripts/turkish_data_backend.py",
            "scripts/upload_base_checkpoint_to_hf.py",
            "nanochat/experiment_manifest.py",
            "nanochat/exposure.py",
            "nanochat/strict_checkpoint.py",
            "nanochat/strict_dataloader.py",
            "nanochat/strict_eval.py",
            "nanochat/strict_runtime.py",
            "nanochat/strict_tokenizer.py",
            "nanochat/packing_capacity.py",
            "nanochat/tokenizer_quality.py",
            "nanochat/training_log.py",
            "nanochat/turkish_backend.py",
            "nanochat/turkish_corpus.py",
            "nanochat/wsd.py",
            "configs/pretrain/fineweb2_tur_Latn.yml",
            "configs/pretrain/glotlid_calibration_tr_v1.jsonl",
            "environments/turkish-data/README.md",
            "environments/turkish-data/environment.json",
            "environments/turkish-data/pyproject.toml",
            "environments/turkish-data/uv.lock",
            "schemas/artifact-manifest.schema.json",
            "schemas/dataset-manifest.schema.json",
            "runs/uhem_d32_train_node.sh",
            "runs/uhem_d32_srun_env.sh",
            "runs/uhem_d32_prepare_training_env.sh",
            "runs/uhem_d32_attention_probe.sbatch",
            "runs/uhem_d32_proxy_arm.sh",
            "runs/uhem_d32_proxy.sbatch",
            "runs/uhem_d32_static_launcher_probe.sbatch",
            "runs/uhem_d32_static_launcher_finalize.sbatch",
            "runs/uhem_d32_signal_resume_smoke.sbatch",
            "runs/uhem_d32_signal_resume_finalize.sbatch",
            "runs/uhem_d32_smoke.sbatch",
            "runs/uhem_d32_smoke_finalize.sbatch",
            "runs/uhem_d32_smoke_gate.sbatch",
            "runs/uhem_d32_production.sbatch",
            "runs/uhem_d32_stage_finalize.sbatch",
            "runs/uhem_d32_family_upload.sbatch",
            "runs/uhem_d32_data_prep_storage_sample.sbatch",
            "runs/uhem_d32_data_prep_writer_probe.sbatch",
            "runs/uhem_submit_d32_family.sh",
            "runs/uhem_turkish_data_objects.sbatch",
            "runs/uhem_turkish_data_objects_packed_sample.sbatch",
            "runs/uhem_turkish_data_objects_packed_production.sbatch",
            "runs/uhem_turkish_data_buckets_packed_sample.sbatch",
            "runs/uhem_turkish_data_buckets_packed_production.sbatch",
            "runs/uhem_turkish_data_buckets.sbatch",
            "runs/uhem_turkish_data_cluster.sbatch",
            "runs/uhem_turkish_data_bootstrap.sbatch",
            "runs/uhem_turkish_data_prepare_macocu.sbatch",
            "runs/uhem_turkish_anchor_fetch_v3.sbatch",
            "runs/uhem_turkish_anchor_prepare_v3.sbatch",
            "runs/uhem_turkish_sample_quality_audit.sbatch",
            "runs/uhem_turkish_prepare_data_env.sbatch",
            "runs/uhem_turkish_corpus_finalize.sbatch",
            "runs/uhem_turkish_packing_preflight.sbatch",
            "runs/uhem_turkish_production_pool.sbatch",
            "runs/uhem_turkish_tokenizer_sample.sbatch",
            "runs/uhem_turkish_tokenizer_train.sbatch",
            "runs/uhem_turkish_tokenizer_quality.sbatch",
        }
    )
    for relative in sorted(code_paths):
        path = require_file(repo_root / relative, f"reviewed source {relative}")
        files.append((path, repo_path("provenance", "source", relative)))
    add_tree(files, base_dir / "metrics" / "d32_family", repo_path("metrics", "family"))
    add_tree(files, base_dir / "metrics" / "d32_smoke", repo_path("metrics", "smoke"))
    add_tree(files, base_dir / "metrics" / "d32_proxy", repo_path("metrics", "proxy"))

    remote_paths = [remote for _local, remote in files]
    if len(remote_paths) != len(set(remote_paths)):
        raise ValueError("family upload selected duplicate remote paths")
    upload_manifest = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "d32_family_huggingface_upload_manifest",
            "repo_id": args.repo_id,
            "visibility_requested": "private" if args.private else "existing_or_public",
            "family_id": recipe["family_id"],
            "family_recipe_sha256": recipe_sha,
            "preflight_receipt_sha256": preflight_sha,
            "attention_probe_sha256": probe_sha,
            "static_launcher_gate_sha256": static_gate_sha,
            "signal_resume_gate_sha256": signal_gate_sha,
            "wsd_proxy_approval_sha256": approval_sha,
            "production_gate_sha256": gate_sha,
            "final_quality_approval_sha256": final_quality_sha,
            "data_preparation_provenance": data_control_hashes,
            "code_revision": revision,
            "final_optimizer_policy": args.family_final_optimizer_policy,
            "retention": retention,
            "lineage_receipts": lineage_hashes,
            "files": [file_entry(path, remote) for path, remote in files],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "canonical_sha256": None,
        }
    )
    print(f"Verified family upload files: {len(files)}")
    print(f"Final optimizer policy: {args.family_final_optimizer_policy}")
    for local_path, path_in_repo in files:
        print(f"{local_path} -> {path_in_repo}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        card_path = tmp_path / "README.md"
        manifest_path = tmp_path / "upload_manifest.json"
        card_path.write_text(
            _family_model_card(recipe, args.family_final_optimizer_policy, upload_manifest),
            encoding="utf-8",
        )
        manifest_path.write_text(
            json.dumps(upload_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        generated = [
            (card_path, "README.md"),
            (manifest_path, repo_path("provenance", "upload_manifest.json")),
        ]
        if args.dry_run:
            print("Dry run: remote state was not changed.")
            return
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise SystemExit(
                "Missing huggingface_hub. Install with: uv pip install -U huggingface_hub"
            ) from exc
        snapshot_tokenizer, snapshot_tokenizer_uploads = (
            _snapshot_verified_tokenizer_uploads(
                tokenizer_root,
                tmp_path / "verified_tokenizer_upload_snapshot",
                expected_sha256=preflight["tokenizer"]["package_manifest_sha256"],
                expected_name=tokenizer_name,
                expected_vocab_size=recipe["model"]["vocab_size"],
            )
        )
        snapshot_by_remote = {
            remote: local for local, remote in snapshot_tokenizer_uploads
        }
        if (
            snapshot_tokenizer.canonical_sha256
            != verified_tokenizer.canonical_sha256
            or set(snapshot_by_remote) != {remote for _local, remote in tokenizer_uploads}
        ):
            raise ValueError("tokenizer upload snapshot drifted after assembly")
        upload_files = [
            (snapshot_by_remote.get(remote, local), remote) for local, remote in files
        ]
        api = HfApi()
        api.create_repo(
            repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True
        )
        for local_path, path_in_repo in generated + upload_files:
            print(f"Uploading {local_path} -> {path_in_repo}", flush=True)
            api.upload_file(
                repo_id=args.repo_id,
                repo_type="model",
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                commit_message=f"Upload {path_in_repo}",
            )
    print(f"Upload complete: https://huggingface.co/{args.repo_id}")


def main():
    args = parse_args()
    if args.family_recipe:
        return main_family(args)
    repo_root = Path.cwd()
    base_dir = Path(args.base_dir or (Path.home() / "nanochat-turk-d20-bpe32k"))
    model_tag = args.model_tag or "tr_d20_bpe_32768_chinchilla20"
    args.model_tag = model_tag

    checkpoint_dir = base_dir / "base_checkpoints" / model_tag
    step = resolve_step(checkpoint_dir, args.step)

    step_tag = f"{step:06d}"
    require_file(checkpoint_dir / f"model_{step_tag}.pt", "model checkpoint")
    meta_path = require_file(checkpoint_dir / f"meta_{step_tag}.json", "checkpoint metadata")
    with meta_path.open("r", encoding="utf-8") as f:
        checkpoint_meta = json.load(f)

    recorded_tokenizer_name = checkpoint_meta.get("tokenizer_name", "")
    if recorded_tokenizer_name:
        if args.tokenizer_name and args.tokenizer_name != recorded_tokenizer_name:
            raise ValueError(
                f"Checkpoint metadata says tokenizer_name={recorded_tokenizer_name!r}, "
                f"but upload was requested with --tokenizer-name={args.tokenizer_name!r}."
            )
        args.tokenizer_name = recorded_tokenizer_name
    elif not args.tokenizer_name:
        raise ValueError(
            "Checkpoint metadata does not record tokenizer_name. Pass --tokenizer-name "
            "explicitly for this older checkpoint."
        )

    tokenizer_dir = base_dir / "tokenizers" / args.tokenizer_name
    require_file(tokenizer_dir / "tokenizer.pkl", "tokenizer.pkl")
    require_file(tokenizer_dir / "tokenizer_config.json", "tokenizer_config.json")
    require_file(tokenizer_dir / "token_bytes.pt", "token_bytes.pt")

    prefix = args.repo_prefix
    files = []
    files.append((checkpoint_dir / f"model_{step_tag}.pt", repo_path(prefix, "checkpoint", f"model_{step_tag}.pt")))
    files.append((checkpoint_dir / f"meta_{step_tag}.json", repo_path(prefix, "checkpoint", f"meta_{step_tag}.json")))

    if not args.no_optimizer:
        optimizers = sorted(checkpoint_dir.glob(f"optim_{step_tag}_rank*.pt"))
        if not optimizers:
            raise FileNotFoundError(f"No optimizer shards found for step {step} in {checkpoint_dir}")
        for path in optimizers:
            files.append((path, repo_path(prefix, "checkpoint", path.name)))

    for name in ("tokenizer.pkl", "tokenizer_config.json", "token_bytes.pt"):
        add_existing(files, tokenizer_dir / name, repo_path(prefix, "tokenizer", name))

    add_tree(files, base_dir / "report", repo_path(prefix, "report"))
    add_tree(files, base_dir / "base_eval", repo_path(prefix, "base_eval"))
    add_tree(files, base_dir / "cetvel_out", repo_path(prefix, "cetvel_out"))

    add_existing(files, repo_root / "report.md", repo_path(prefix, "report.md"))
    if args.job_id:
        add_existing(files, repo_root / f"nanochat-tr-d20-bpe32k-{args.job_id}.out", repo_path(prefix, "logs", f"nanochat-tr-d20-bpe32k-{args.job_id}.out"))
        add_existing(files, repo_root / f"nanochat-tr-d20-bpe32k-{args.job_id}.err", repo_path(prefix, "logs", f"nanochat-tr-d20-bpe32k-{args.job_id}.err"))
        add_tree(files, repo_root / "logs" / f"job{args.job_id}", repo_path(prefix, "logs", f"job{args.job_id}"))
    if args.cetvel_job_id:
        add_existing(files, repo_root / f"nanochat-cetvel-full-{args.cetvel_job_id}.out", repo_path(prefix, "logs", f"nanochat-cetvel-full-{args.cetvel_job_id}.out"))
        add_existing(files, repo_root / f"nanochat-cetvel-full-{args.cetvel_job_id}.err", repo_path(prefix, "logs", f"nanochat-cetvel-full-{args.cetvel_job_id}.err"))
    add_existing(files, repo_root / "train-wandb-run-id.txt", repo_path(prefix, "provenance", "train-wandb-run-id.txt"))

    for name in (
        "README.md",
        "pyproject.toml",
        "uv.lock",
        "runs/turkish_foundation.sh",
        "runs/uhem_nakane_a100x4_d20_bpe32k.sbatch",
        "scripts/base_train.py",
        "scripts/base_eval.py",
        "nanochat/gpt.py",
        "nanochat/tokenizer.py",
        "nanochat/checkpoint_manager.py",
    ):
        add_existing(files, repo_root / name, repo_path(prefix, "provenance", "source", name))

    manifest = {
        "repo_id": args.repo_id,
        "repo_prefix": prefix,
        "base_dir": str(base_dir),
        "model_tag": model_tag,
        "step": step,
        "tokenizer_name": args.tokenizer_name,
        "checkpoint_dir": str(checkpoint_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "job_id": args.job_id,
        "cetvel_job_id": args.cetvel_job_id,
        "uploaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": run_command(["git", "rev-parse", "HEAD"]),
        "git_branch": run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(run_command(["git", "status", "--porcelain"])),
        "files": [file_entry(path, path_in_repo) for path, path_in_repo in files],
    }

    print(f"Resolved step: {step}")
    print(f"Files selected for upload: {len(files)}")
    for local_path, path_in_repo in files:
        print(f"{local_path} -> {path_in_repo}")
    if args.dry_run:
        return

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Missing huggingface_hub. Install with: uv pip install -U huggingface_hub") from exc

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        card_path = tmp_path / "README.md"
        manifest_path = tmp_path / "upload_manifest.json"
        card_path.write_text(build_model_card(args, base_dir, checkpoint_dir, tokenizer_dir, step, files), encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        generated = [
            (card_path, repo_path(prefix, "README.md") if prefix else "README.md"),
            (manifest_path, repo_path(prefix, "provenance", "upload_manifest.json")),
        ]

        for local_path, path_in_repo in generated + files:
            print(f"Uploading {local_path} -> {path_in_repo}", flush=True)
            api.upload_file(
                repo_id=args.repo_id,
                repo_type="model",
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                commit_message=f"Upload {path_in_repo}",
            )

    print(f"Upload complete: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
