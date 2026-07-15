"""Publish a trained nanochat tokenizer bundle.

The bundle is intentionally tokenizer-only: it contains the nanochat tokenizer
files, tokenizer metrics when available, segmentation provenance, checksums, and
a short README. It can be copied to a local directory for GitHub archival and/or
uploaded to a Hugging Face Hub repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_TOKENIZER_FILES = ("tokenizer.pkl", "tokenizer_config.json", "token_bytes.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a trained nanochat tokenizer artifact")
    parser.add_argument("--base-dir", default=os.environ.get("NANOCHAT_BASE_DIR", ""))
    parser.add_argument("--tokenizer-name", default=os.environ.get("NANOCHAT_TOKENIZER_NAME", ""))
    parser.add_argument("--repo-id", default=os.environ.get("HF_TOKENIZER_REPO_ID", ""))
    parser.add_argument("--repo-type", default=os.environ.get("HF_TOKENIZER_REPO_TYPE", "model"))
    parser.add_argument("--repo-prefix", default=os.environ.get("HF_TOKENIZER_REPO_PREFIX", ""))
    parser.add_argument("--local-output-dir", default=os.environ.get("TOKENIZER_PUBLISH_DIR", ""))
    parser.add_argument("--metrics-path", default=os.environ.get("TOKENIZER_METRICS_PATH", ""))
    parser.add_argument("--segmentation-manifest", default=os.environ.get("SEGMENTATION_MANIFEST", ""))
    parser.add_argument("--segmented-dataset-manifest", default=os.environ.get("SEGMENTED_DATASET_MANIFEST", ""))
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--require-metrics", action="store_true")
    parser.add_argument("--require-segmentation-manifest", action="store_true")
    parser.add_argument("--require-segmented-dataset-manifest", action="store_true")
    parser.add_argument("--no-hf-upload", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_command(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {"value": payload}


def maybe_add(
    files: list[tuple[Path, Path]],
    path: Path,
    bundle_path: str,
    *,
    required: bool = False,
    label: str = "",
) -> None:
    if path.is_file():
        files.append((path, Path(bundle_path)))
    elif required:
        raise FileNotFoundError(f"Missing {label or bundle_path}: {path}")


def repo_path(prefix: str, path: Path) -> str:
    clean_prefix = prefix.strip("/")
    clean_path = str(path).strip("/")
    return f"{clean_prefix}/{clean_path}" if clean_prefix else clean_path


def write_readme(
    path: Path,
    *,
    tokenizer_name: str,
    tokenizer_config: dict[str, Any],
    metrics: dict[str, Any] | None,
    manifest: dict[str, Any],
) -> None:
    implementation = str(tokenizer_config.get("implementation", "bpe"))
    method_tag = {
        "bpe": "bpe",
        "morphbpe": "morphbpe",
        "preseg_bpe": "preseg-bpe",
    }.get(implementation, implementation.replace("_", "-"))
    metrics_lines = ""
    if metrics:
        preferred = [
            "docs",
            "bytes",
            "tokens",
            "tokens_per_byte",
            "bytes_per_token",
            "chars_per_token",
            "words_per_token",
            "encode_docs_per_sec",
        ]
        rows = []
        for key in preferred:
            if key in metrics:
                rows.append(f"- `{key}`: `{metrics[key]}`")
        if rows:
            metrics_lines = "\n".join(rows)
        else:
            metrics_lines = "Metrics JSON is included under `metrics/raw_metrics.json`."
    else:
        metrics_lines = "No tokenizer metrics JSON was bundled."

    path.write_text(
        f"""---
language:
- tr
tags:
- nanochat
- turkish
- tokenizer
- {method_tag}
library_name: tiktoken
---

# `{tokenizer_name}`

This artifact stores a trained nanochat tokenizer bundle for the Turkish
MorphBPE ablation study. It is a raw nanochat/tiktoken tokenizer artifact, not a
Transformers `AutoTokenizer` export.

## Tokenizer Config

```json
{json.dumps(tokenizer_config, ensure_ascii=False, indent=2, sort_keys=True)}
```

## Included Files

- `tokenizer.pkl`
- `tokenizer_config.json`
- `token_bytes.pt`
- `metrics/raw_metrics.json` when available
- `provenance/segmentation_manifest.json` when available
- `provenance/segmented_dataset_manifest.json` when available
- `provenance/publish_manifest.json`

## Metrics

{metrics_lines}

## Provenance

- Git commit: `{manifest.get("git_commit") or "unknown"}`
- Git branch: `{manifest.get("git_branch") or "unknown"}`
- Uploaded/generated at: `{manifest["generated_at_utc"]}`
- Source tokenizer dir: `{manifest["tokenizer_dir"]}`

## Loading

Use this repository's `nanochat.tokenizer.RustBPETokenizer.from_directory(...)`
or set:

```bash
export NANOCHAT_BASE_DIR=/path/to/base-dir
export NANOCHAT_TOKENIZER_NAME={tokenizer_name}
```
""",
        encoding="utf-8",
    )


def copy_bundle(files: list[tuple[Path, Path]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for local_path, bundle_path in files:
        target = output_dir / bundle_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, target)


def main() -> None:
    args = parse_args()
    if not args.base_dir:
        raise ValueError("--base-dir or NANOCHAT_BASE_DIR is required")
    if not args.tokenizer_name:
        raise ValueError("--tokenizer-name or NANOCHAT_TOKENIZER_NAME is required")
    if args.no_hf_upload and not args.local_output_dir:
        raise ValueError("Nothing to do: pass --local-output-dir or enable HF upload")
    if not args.no_hf_upload and not args.repo_id:
        raise ValueError("--repo-id or HF_TOKENIZER_REPO_ID is required unless --no-hf-upload is set")

    base_dir = Path(args.base_dir).expanduser().resolve()
    tokenizer_dir = base_dir / "tokenizers" / args.tokenizer_name
    for name in REQUIRED_TOKENIZER_FILES:
        require_file(tokenizer_dir / name, name)

    tokenizer_config_path = tokenizer_dir / "tokenizer_config.json"
    tokenizer_config = load_json(tokenizer_config_path)

    metrics_path = Path(args.metrics_path).expanduser() if args.metrics_path else base_dir / f"{args.tokenizer_name}_raw_metrics.json"
    segmentation_manifest = Path(args.segmentation_manifest).expanduser() if args.segmentation_manifest else Path(
        tokenizer_config.get("data_dir", "")
    ) / "manifest.json"
    segmented_dataset_manifest = (
        Path(args.segmented_dataset_manifest).expanduser()
        if args.segmented_dataset_manifest
        else Path(tokenizer_config.get("data_dir", "")) / "fineweb2_manifest.json"
    )

    files: list[tuple[Path, Path]] = []
    for name in REQUIRED_TOKENIZER_FILES:
        files.append((tokenizer_dir / name, Path(name)))
    maybe_add(
        files,
        metrics_path,
        "metrics/raw_metrics.json",
        required=args.require_metrics,
        label="tokenizer metrics",
    )
    maybe_add(
        files,
        segmentation_manifest,
        "provenance/segmentation_manifest.json",
        required=args.require_segmentation_manifest,
        label="segmentation manifest",
    )
    maybe_add(
        files,
        segmented_dataset_manifest,
        "provenance/segmented_dataset_manifest.json",
        required=args.require_segmented_dataset_manifest,
        label="segmented dataset manifest",
    )

    metrics = load_json(metrics_path) if metrics_path.is_file() else None
    generated_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "tokenizer_name": args.tokenizer_name,
        "base_dir": str(base_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "tokenizer_config": tokenizer_config,
        "metrics_path": str(metrics_path) if metrics_path.is_file() else "",
        "segmentation_manifest": str(segmentation_manifest) if segmentation_manifest.is_file() else "",
        "segmented_dataset_manifest": str(segmented_dataset_manifest) if segmented_dataset_manifest.is_file() else "",
        "repo_id": args.repo_id,
        "repo_type": args.repo_type,
        "repo_prefix": args.repo_prefix or args.tokenizer_name,
        "generated_at_utc": generated_at,
        "git_commit": run_command(["git", "rev-parse", "HEAD"]),
        "git_branch": run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(run_command(["git", "status", "--porcelain"])),
        "files": [],
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        readme_path = tmp_path / "README.md"
        manifest_path = tmp_path / "publish_manifest.json"
        write_readme(
            readme_path,
            tokenizer_name=args.tokenizer_name,
            tokenizer_config=tokenizer_config,
            metrics=metrics,
            manifest=manifest,
        )
        generated_files = [
            (readme_path, Path("README.md")),
            (manifest_path, Path("provenance/publish_manifest.json")),
        ]
        all_files = files + generated_files
        manifest["files"] = [
            {
                "bundle_path": str(bundle_path),
                "source_path": str(local_path),
                "size_bytes": local_path.stat().st_size,
                "sha256": sha256_file(local_path),
            }
            for local_path, bundle_path in all_files
            if local_path != manifest_path
        ]
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        manifest["files"].append(
            {
                "bundle_path": "provenance/publish_manifest.json",
                "source_path": str(manifest_path),
                "size_bytes": manifest_path.stat().st_size,
                "sha256": sha256_file(manifest_path),
            }
        )

        bundle_dir = tmp_path / "bundle"
        copy_bundle(all_files, bundle_dir)

        print(f"Tokenizer: {args.tokenizer_name}")
        print(f"Bundle files: {len(all_files)}")
        for local_path, bundle_path in all_files:
            print(f"{local_path} -> {bundle_path}")

        if args.local_output_dir:
            local_output_dir = Path(args.local_output_dir).expanduser()
            print(f"Writing local bundle to {local_output_dir}")
            if not args.dry_run:
                copy_bundle(all_files, local_output_dir)

        if args.no_hf_upload:
            return

        print(f"Uploading to Hugging Face: {args.repo_id}/{args.repo_prefix or args.tokenizer_name}")
        if args.dry_run:
            return

        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise SystemExit("Missing huggingface_hub. Install with: uv pip install -U huggingface_hub") from exc

        api = HfApi()
        api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            folder_path=str(bundle_dir),
            path_in_repo=(args.repo_prefix or args.tokenizer_name).strip("/"),
            commit_message=f"Upload tokenizer {args.tokenizer_name}",
        )
        print(f"Upload complete: https://huggingface.co/{args.repo_id}/tree/main/{args.repo_prefix or args.tokenizer_name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
