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
    }
    missing = [name for name, value in required_args.items() if not value]
    if missing:
        raise ValueError("family upload is missing: " + ", ".join(missing))

    from nanochat.strict_checkpoint import inspect_strict_checkpoint
    from nanochat.experiment_manifest import seal_manifest, verify_manifest_hash
    from nanochat.tokenizer_quality import validate_tokenizer_quality_gate
    from scripts.d32_family_workflow import (
        _load_receipt,
        _verify_attention_probe,
        _verify_gate_and_preflight,
        _verify_signal_resume_gate,
        _verify_static_launcher_gate,
        load_recipe,
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
        files.append((path, repo_path("provenance", "lineage", path.name)))

    corpus_root = Path(preflight["corpus"]["root"])
    expected_chain = preflight["corpus"].get("production_chain")
    if (
        not isinstance(expected_chain, dict)
        or expected_chain.get("data_prep_storage_gate_sha256")
        != preflight["data_preparation_storage_gate_sha256"]
        or expected_chain.get("cluster_launch_receipt_sha256")
        != cluster_launch_sha
    ):
        raise ValueError("production preflight has malformed data lineage")
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

    tokenizer_root = Path(preflight["tokenizer"]["root"])
    add_tree(files, tokenizer_root, repo_path("tokenizer"))
    tokenizer_name = preflight["tokenizer"]["name"]
    tokenizer_control_root = base_dir / "control" / "tokenizer" / tokenizer_name
    quality_root = tokenizer_control_root / "quality"
    quality_report, quality_approval = validate_tokenizer_quality_gate(
        quality_root,
        expected_package_sha256=preflight["tokenizer"]["package_manifest_sha256"],
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
        if path.stat().st_size != record["size_bytes"] or file_sha256(path) != record["sha256"]:
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
    control_files = [
        recipe_path,
        Path(args.preflight_receipt),
        Path(args.attention_probe),
        Path(args.wd_proxy_approval),
        Path(args.static_launcher_gate),
        Path(args.signal_resume_gate),
        Path(args.production_gate),
        Path(args.cluster_launch_receipt),
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
        api = HfApi()
        api.create_repo(
            repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True
        )
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
