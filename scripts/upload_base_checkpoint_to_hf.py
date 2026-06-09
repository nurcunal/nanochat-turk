"""
Upload a nanochat base-model checkpoint bundle to a Hugging Face model repo.

This uploads the raw nanochat checkpoint format, not a Transformers-compatible
conversion. It preserves the model checkpoint, optional optimizer shards,
tokenizer artifacts, report/eval outputs, logs, and provenance files.
"""

import argparse
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


def file_entry(path, path_in_repo):
    stat = path.stat()
    return {
        "path_in_repo": path_in_repo,
        "local_path": str(path),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
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


def main():
    args = parse_args()
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
