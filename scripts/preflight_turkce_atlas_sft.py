"""Verify prepared Türkçe Atlas data and the runtime selected for nanochat SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from nanochat.sft_data import build_custom_sft_data_identity, sha256_file
from tasks.customjson import CustomJSON


SPECIAL_TOKEN_PATTERN = re.compile(r"<\|[^|\r\n]+\|>")
KNOWN_INCOMPLETE_RESPONSES = {"S:"}


def _pair_digest(user: str, assistant: str) -> bytes:
    digest = hashlib.sha256()
    for value in (user, assistant):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.digest()


def _audit_prepared_split(path: Path) -> tuple[set[bytes], Counter[bytes]]:
    # CustomJSON's lazy index validates every JSON line and the full role sequence.
    dataset = CustomJSON(str(path), lazy=True)
    prompt_digests: set[bytes] = set()
    pair_digests: Counter[bytes] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            messages = CustomJSON._parse_line(line, line_number)
            roles = [message["role"] for message in messages]
            if roles != ["user", "assistant"]:
                raise ValueError(f"{path}:{line_number}: expected user -> assistant, got {roles}")
            user = messages[0]["content"]
            assistant = messages[1]["content"]
            if not user.strip() or not assistant.strip():
                raise ValueError(f"{path}:{line_number}: empty or whitespace-only content")
            if assistant in KNOWN_INCOMPLETE_RESPONSES:
                raise ValueError(f"{path}:{line_number}: known incomplete response {assistant!r}")
            if SPECIAL_TOKEN_PATTERN.search(user) or SPECIAL_TOKEN_PATTERN.search(assistant):
                raise ValueError(f"{path}:{line_number}: embedded nanochat-style special token")
            prompt_digests.add(hashlib.sha256(user.encode("utf-8")).digest())
            pair_digests[_pair_digest(user, assistant)] += 1
    if len(dataset) != sum(pair_digests.values()):
        raise RuntimeError(f"{path}: CustomJSON and audit row counts differ")
    return prompt_digests, pair_digests


def _audit_source(path: Path) -> tuple[Counter[bytes], Counter[str]]:
    pairs: Counter[bytes] = Counter()
    removed: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            messages = payload.get("messages") if isinstance(payload, dict) else None
            roles = [message.get("role") for message in messages] if isinstance(messages, list) else []
            if roles != ["system", "user", "assistant"]:
                raise ValueError(f"{path}:{line_number}: unexpected source schema")
            user = messages[1]["content"]
            assistant = messages[2]["content"]
            if assistant in KNOWN_INCOMPLETE_RESPONSES:
                removed[assistant] += 1
                continue
            pairs[_pair_digest(user, assistant)] += 1
    return pairs, removed


def _tokenizer_identity(tokenizer_dir: Path) -> dict:
    tokenizer_dir = tokenizer_dir.expanduser().resolve()
    files = {}
    for filename in ("tokenizer.pkl", "tokenizer.json", "tokenizer_config.json", "token_bytes.pt"):
        path = tokenizer_dir / filename
        if path.is_file():
            files[filename] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    if len({"tokenizer.pkl", "tokenizer.json"} & files.keys()) != 1:
        raise ValueError(f"{tokenizer_dir}: expected exactly one tokenizer model file")
    for required in ("tokenizer_config.json", "token_bytes.pt"):
        if required not in files:
            raise FileNotFoundError(tokenizer_dir / required)
    return {"directory": str(tokenizer_dir), "files": files}


def _verify_tokenizer(manifest_tokenizer: dict | None, tokenizer_identity: dict) -> None:
    if not manifest_tokenizer:
        raise ValueError("data manifest does not record tokenizer identity")
    files = tokenizer_identity["files"]
    model_file = files.get("tokenizer.pkl") or files.get("tokenizer.json")
    checks = (
        ("tokenizer_sha256", model_file["sha256"]),
        ("config_sha256", files["tokenizer_config.json"]["sha256"]),
        ("token_bytes_sha256", files["token_bytes.pt"]["sha256"]),
    )
    for field, actual in checks:
        expected = manifest_tokenizer.get(field)
        if not expected:
            raise ValueError(f"data manifest tokenizer identity is missing {field}")
        if actual != expected:
            raise ValueError(f"tokenizer {field} mismatch: {actual} != {expected}")


def _verify_checkpoint(
    checkpoint_dir: Path,
    step: int,
    world_size: int,
    expected_model_tag: str,
    expected_total_batch_size: int,
) -> dict:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    required = [
        checkpoint_dir / f"model_{step:06d}.pt",
        checkpoint_dir / f"meta_{step:06d}.json",
        *(checkpoint_dir / f"optim_{step:06d}_rank{rank}.pt" for rank in range(world_size)),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing base checkpoint files: {missing}")
    meta_path = checkpoint_dir / f"meta_{step:06d}.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("step") != step:
        raise ValueError(f"base checkpoint step mismatch: {meta.get('step')} != {step}")
    recorded_model_tag = meta.get("model_tag") or meta.get("user_config", {}).get(
        "model_tag"
    )
    if recorded_model_tag != expected_model_tag:
        raise ValueError(
            f"base checkpoint model_tag mismatch: {recorded_model_tag} != {expected_model_tag}"
        )
    model_config = meta.get("model_config", {})
    expected_config = {
        "sequence_len": 2048,
        "vocab_size": 32768,
        "n_layer": 20,
        "n_head": 10,
        "n_kv_head": 10,
        "n_embd": 1280,
    }
    for key, expected in expected_config.items():
        if model_config.get(key) != expected:
            raise ValueError(
                f"base checkpoint model_config.{key} mismatch: "
                f"{model_config.get(key)} != {expected}"
            )
    if meta.get("total_batch_size") != expected_total_batch_size:
        raise ValueError(
            "base checkpoint total_batch_size mismatch: "
            f"{meta.get('total_batch_size')} != {expected_total_batch_size}"
        )
    return {
        "directory": str(checkpoint_dir),
        "step": step,
        "model_tag": expected_model_tag,
        "metadata_sha256": sha256_file(meta_path),
        "model_bytes": required[0].stat().st_size,
        "optimizer_shard_bytes": [path.stat().st_size for path in required[2:]],
        "model_config": expected_config,
        "pretrain_total_batch_size": expected_total_batch_size,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path)
    parser.add_argument("--base-checkpoint-dir", type=Path)
    parser.add_argument("--model-step", type=int, default=17100)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument(
        "--expected-model-tag", default="tr_d20_bpe_32768_chinchilla20"
    )
    parser.add_argument("--expected-total-batch-size", type=int, default=1048576)
    parser.add_argument("--disk-path", type=Path)
    parser.add_argument("--min-free-gb", type=float, default=10.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    data_identity = build_custom_sft_data_identity(
        data_dir / "train.jsonl",
        data_dir / "validation.jsonl",
        data_dir / "manifest.json",
    )
    train_prompts, train_pairs = _audit_prepared_split(data_dir / "train.jsonl")
    validation_prompts, validation_pairs = _audit_prepared_split(
        data_dir / "validation.jsonl"
    )
    overlap = train_prompts & validation_prompts
    if overlap:
        raise ValueError(f"{len(overlap)} exact user prompt groups cross train/validation")

    source_audit = None
    if args.source_jsonl:
        source_pairs, removed = _audit_source(args.source_jsonl.expanduser().resolve())
        if source_pairs != train_pairs + validation_pairs:
            raise ValueError("filtered source/output dialogue-pair multisets differ")
        source_audit = {
            "retained_pair_multiset_matches": True,
            "removed_counts": dict(sorted(removed.items())),
        }

    tokenizer_identity = _tokenizer_identity(args.tokenizer_dir)
    _verify_tokenizer(data_identity.get("tokenizer"), tokenizer_identity)

    checkpoint_identity = None
    if args.base_checkpoint_dir:
        checkpoint_identity = _verify_checkpoint(
            args.base_checkpoint_dir,
            args.model_step,
            args.world_size,
            args.expected_model_tag,
            args.expected_total_batch_size,
        )

    disk_identity = None
    if args.disk_path:
        disk_path = args.disk_path.expanduser().resolve()
        usage = shutil.disk_usage(disk_path)
        free_gb = usage.free / 1_000_000_000
        if free_gb < args.min_free_gb:
            raise RuntimeError(
                f"insufficient free disk at {disk_path}: {free_gb:.2f} GB < {args.min_free_gb:.2f} GB"
            )
        disk_identity = {"path": str(disk_path), "free_gb": round(free_gb, 3)}

    print(
        json.dumps(
            {
                "status": "ok",
                "data": data_identity,
                "exact_prompt_groups_crossing_splits": 0,
                "source_audit": source_audit,
                "tokenizer": tokenizer_identity,
                "base_checkpoint": checkpoint_identity,
                "disk": disk_identity,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
