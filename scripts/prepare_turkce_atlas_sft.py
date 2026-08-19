"""Prepare Türkçe Atlas for nanochat SFT without rewriting any dialogue text.

The source dataset stores one ``system -> user -> assistant`` conversation per
JSON object. Nanochat represents chat data as ``{"messages": [...]}`` and folds
system text into the first user turn. Because Atlas repeats the same long system
message in every row, this preparer removes that shared preamble by default,
preserves every user/assistant string exactly, and writes deterministic grouped
train/validation splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


FORMAT_VERSION = 1
EXPECTED_ROLES = ["system", "user", "assistant"]
KNOWN_INCOMPLETE_ASSISTANT_RESPONSES = {"S:"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_source_line(raw_line: bytes, line_number: int) -> list[dict[str, str]]:
    try:
        payload = json.loads(raw_line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"line {line_number}: invalid UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"messages"}:
        raise ValueError(f"line {line_number}: expected an object containing only 'messages'")
    messages = payload["messages"]
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError(f"line {line_number}: expected exactly three messages")
    roles = [message.get("role") if isinstance(message, dict) else None for message in messages]
    if roles != EXPECTED_ROLES:
        raise ValueError(f"line {line_number}: expected roles {EXPECTED_ROLES}, got {roles}")
    for index, message in enumerate(messages):
        if set(message) != {"role", "content"}:
            raise ValueError(f"line {line_number}: message {index} must contain only role/content")
        if not isinstance(message["content"], str) or not message["content"]:
            raise ValueError(f"line {line_number}: message {index} content must be a non-empty string")
    return messages


def _seeded_group_key(user_digest: bytes, seed: int) -> bytes:
    return hashlib.sha256(str(seed).encode("ascii") + b"\0" + user_digest).digest()


def _choose_validation_groups(
    group_sizes: dict[bytes, int], validation_size: int, seed: int
) -> tuple[set[bytes], int]:
    if validation_size <= 0:
        raise ValueError("validation_size must be positive")
    total_rows = sum(group_sizes.values())
    if validation_size >= total_rows:
        raise ValueError("validation_size must be smaller than the dataset")
    ordered_groups = sorted(group_sizes, key=lambda digest: _seeded_group_key(digest, seed))
    selected: set[bytes] = set()
    selected_rows = 0
    for digest in ordered_groups:
        selected.add(digest)
        selected_rows += group_sizes[digest]
        if selected_rows >= validation_size:
            break
    return selected, selected_rows


def _quantile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _new_token_accumulator() -> dict[str, list]:
    return {"lengths": [], "supervised": [], "kept_supervised": []}


def _finalize_token_accumulator(accumulator: dict[str, list], max_seq_len: int) -> dict:
    lengths = accumulator["lengths"]
    supervised = accumulator["supervised"]
    kept = accumulator["kept_supervised"]
    return {
        "rows": len(lengths),
        "formatted_tokens": sum(lengths),
        "supervised_assistant_tokens": sum(supervised),
        "formatted_length": {
            "mean": round(sum(lengths) / len(lengths), 4),
            "median": _quantile(lengths, 0.50),
            "p95": _quantile(lengths, 0.95),
            "p99": _quantile(lengths, 0.99),
            "max": max(lengths),
        },
        "rows_over_max_seq_len": sum(length > max_seq_len for length in lengths),
        "rows_losing_assistant_tokens": sum(a > b for a, b in zip(supervised, kept)),
        "assistant_tokens_lost_at_max_seq_len": sum(a - b for a, b in zip(supervised, kept)),
    }


def prepare_atlas(
    input_path: Path,
    output_dir: Path,
    *,
    validation_size: int = 5000,
    split_seed: int = 20260819,
    drop_shared_system: bool = True,
    drop_known_incomplete: bool = True,
    tokenizer_dir: Path | None = None,
    max_seq_len: int = 2048,
) -> dict:
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    source_digest = hashlib.sha256()
    system_counts: Counter[str] = Counter()
    source_user_groups: set[bytes] = set()
    user_group_sizes: dict[bytes, int] = {}
    source_row_count = 0
    removed_response_counts: Counter[str] = Counter()

    with input_path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            source_digest.update(raw_line)
            if not raw_line.strip():
                continue
            messages = _parse_source_line(raw_line, line_number)
            source_row_count += 1
            system_counts[messages[0]["content"]] += 1
            user_digest = hashlib.sha256(messages[1]["content"].encode("utf-8")).digest()
            source_user_groups.add(user_digest)
            assistant_content = messages[2]["content"]
            if drop_known_incomplete and assistant_content in KNOWN_INCOMPLETE_ASSISTANT_RESPONSES:
                removed_response_counts[assistant_content] += 1
                continue
            user_group_sizes[user_digest] = user_group_sizes.get(user_digest, 0) + 1

    if source_row_count == 0:
        raise ValueError("source dataset is empty")
    retained_row_count = source_row_count - sum(removed_response_counts.values())
    if retained_row_count == 0:
        raise ValueError("all source rows were removed by the configured filters")
    if drop_shared_system and len(system_counts) != 1:
        raise ValueError(
            "refusing to drop system messages because the source does not contain exactly one shared prompt"
        )

    validation_groups, actual_validation_size = _choose_validation_groups(
        user_group_sizes, validation_size, split_seed
    )

    tokenizer = None
    tokenizer_metadata = None
    if tokenizer_dir is not None:
        from nanochat.tokenizer import load_tokenizer_from_directory

        tokenizer_dir = tokenizer_dir.resolve()
        tokenizer = load_tokenizer_from_directory(str(tokenizer_dir))
        tokenizer_metadata = {
            "directory": str(tokenizer_dir),
            "vocab_size": tokenizer.get_vocab_size(),
            "tokenizer_sha256": _sha256_file(tokenizer_dir / "tokenizer.pkl"),
            "config_sha256": _sha256_file(tokenizer_dir / "tokenizer_config.json"),
            "token_bytes_sha256": _sha256_file(tokenizer_dir / "token_bytes.pt"),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    system_path = output_dir / "source_system_prompt.txt"
    temporary_paths = {
        train_path: train_path.with_suffix(".jsonl.tmp"),
        validation_path: validation_path.with_suffix(".jsonl.tmp"),
        system_path: system_path.with_suffix(".txt.tmp"),
    }
    output_hashes = {"train": hashlib.sha256(), "validation": hashlib.sha256()}
    output_bytes = {"train": 0, "validation": 0}
    output_rows = {"train": 0, "validation": 0}
    token_stats = {"train": _new_token_accumulator(), "validation": _new_token_accumulator()}

    try:
        shared_system = system_counts.most_common(1)[0][0]
        temporary_paths[system_path].write_bytes(shared_system.encode("utf-8"))
        with temporary_paths[train_path].open("wb") as train_file, temporary_paths[
            validation_path
        ].open("wb") as validation_file, input_path.open("rb") as source:
            for line_number, raw_line in enumerate(source, start=1):
                if not raw_line.strip():
                    continue
                messages = _parse_source_line(raw_line, line_number)
                if (
                    drop_known_incomplete
                    and messages[2]["content"] in KNOWN_INCOMPLETE_ASSISTANT_RESPONSES
                ):
                    continue
                output_messages = messages[1:] if drop_shared_system else messages
                conversation = {"messages": output_messages}
                encoded = (
                    json.dumps(output_messages, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                user_digest = hashlib.sha256(messages[1]["content"].encode("utf-8")).digest()
                split = "validation" if user_digest in validation_groups else "train"
                output_file = validation_file if split == "validation" else train_file
                output_file.write(encoded)
                output_hashes[split].update(encoded)
                output_bytes[split] += len(encoded)
                output_rows[split] += 1

                if tokenizer is not None:
                    ids, mask = tokenizer.render_conversation(conversation, max_tokens=10**9)
                    supervised_count = sum(mask)
                    token_stats[split]["lengths"].append(len(ids))
                    token_stats[split]["supervised"].append(supervised_count)
                    token_stats[split]["kept_supervised"].append(sum(mask[:max_seq_len]))

        for final_path, temporary_path in temporary_paths.items():
            os.replace(temporary_path, final_path)
    except Exception:
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)
        raise

    if sum(output_rows.values()) != retained_row_count:
        raise RuntimeError("output row count does not match retained source row count")

    system_bytes = shared_system.encode("utf-8")
    manifest = {
        "format_version": FORMAT_VERSION,
        "operation": "format/filter only; retained user and assistant strings are preserved",
        "source": {
            "path": str(input_path),
            "sha256": source_digest.hexdigest(),
            "rows": source_row_count,
            "unique_exact_user_prompts": len(source_user_groups),
        },
        "normalization": {
            "output_schema": [{"role": "user|assistant", "content": "string"}],
            "drop_shared_system": drop_shared_system,
            "rows_removed": sum(removed_response_counts.values()),
            "user_or_assistant_text_rewritten": False,
            "filters": {
                "drop_known_incomplete_assistant_responses": drop_known_incomplete,
                "exact_assistant_responses": sorted(KNOWN_INCOMPLETE_ASSISTANT_RESPONSES),
                "removed_counts": dict(sorted(removed_response_counts.items())),
                "reason": "dataset card identifies these exact responses as incomplete examples",
            },
            "source_system_prompt": {
                "path": system_path.name,
                "sha256": hashlib.sha256(system_bytes).hexdigest(),
                "characters": len(shared_system),
                "utf8_bytes": len(system_bytes),
                "source_occurrences": system_counts[shared_system],
                "retained_in_output": not drop_shared_system,
            },
        },
        "split": {
            "strategy": "seeded SHA-256 groups over exact user content",
            "seed": split_seed,
            "requested_validation_rows": validation_size,
            "actual_validation_rows": actual_validation_size,
            "exact_user_prompt_groups_crossing_splits": 0,
        },
        "outputs": {
            "train": {
                "path": train_path.name,
                "rows": output_rows["train"],
                "bytes": output_bytes["train"],
                "sha256": output_hashes["train"].hexdigest(),
            },
            "validation": {
                "path": validation_path.name,
                "rows": output_rows["validation"],
                "bytes": output_bytes["validation"],
                "sha256": output_hashes["validation"].hexdigest(),
            },
        },
        "tokenizer": tokenizer_metadata,
        "max_seq_len": max_seq_len,
    }
    if tokenizer is not None:
        manifest["outputs"]["train"]["tokens"] = _finalize_token_accumulator(
            token_stats["train"], max_seq_len
        )
        manifest["outputs"]["validation"]["tokens"] = _finalize_token_accumulator(
            token_stats["validation"], max_seq_len
        )

    manifest_path = output_dir / "manifest.json"
    manifest_tmp = output_dir / "manifest.json.tmp"
    manifest_tmp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(manifest_tmp, manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Atlas train.chat.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--split-seed", type=int, default=20260819)
    parser.add_argument(
        "--keep-shared-system",
        action="store_true",
        help="retain Atlas's identical system message in every conversation",
    )
    parser.add_argument(
        "--keep-known-incomplete",
        action="store_true",
        help="retain exact assistant responses identified by the Atlas card as incomplete",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("artifacts/tokenizers/bpe_32768"),
        help="exact tokenizer artifact used for compatibility statistics",
    )
    parser.add_argument("--max-seq-len", type=int, default=2048)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = prepare_atlas(
        args.input,
        args.output_dir,
        validation_size=args.validation_size,
        split_seed=args.split_seed,
        drop_shared_system=not args.keep_shared_system,
        drop_known_incomplete=not args.keep_known_incomplete,
        tokenizer_dir=args.tokenizer_dir,
        max_seq_len=args.max_seq_len,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
