"""Train and seal the d32 family's raw Turkish 32,768-token BPE package."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from pathlib import Path

import pyarrow.parquet as pq
import torch

from nanochat.experiment_manifest import (
    file_sha256,
    load_json_strict,
    seal_manifest,
    validate_dataset_manifest,
    verify_file_inventory,
    verify_manifest_hash,
    write_json_atomic,
)
from nanochat.tokenizer import SPECIAL_TOKENS, SPLIT_PATTERN, RustBPETokenizer
from nanochat.tokenizer_quality import build_tokenizer_quality_report
from nanochat.turkish_corpus import TOKENIZER_NAME, VOCAB_SIZE, load_corpus_policy


PACKAGE_KIND = "turkish_raw_bpe_tokenizer_package"
PINNED_TOK_TRAIN_ITERATOR_SHA256 = (
    "206d8c89554ceeb4de7afe22e53786806d567e1c4f5493352b2170c0ac174a29"
)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/pretrain/tr_d32_turkish_general_v2.json"),
    )
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quality-output-dir", type=Path, required=True)
    parser.add_argument("--baseline-tokenizer-dir", type=Path)
    return parser


def _validate_training_sample(sample_dir: Path, policy: dict) -> tuple[dict, dict]:
    sample = load_json_strict(sample_dir / "tokenizer_sample_manifest.json")
    verify_manifest_hash(sample)
    if sample.get("kind") != "turkish_raw_bpe_training_sample":
        raise ValueError("unexpected tokenizer sample manifest")
    if sample.get("name") != TOKENIZER_NAME or sample.get("vocab_size") != VOCAB_SIZE:
        raise ValueError("tokenizer sample identity drift")
    requested = policy["tokenizer_training"]["max_chars"]
    realized = sample.get("characters", 0)
    document_cap = policy["tokenizer_training"]["max_chars_per_document"]
    if not isinstance(realized, int) or not requested < realized <= requested + document_cap:
        raise ValueError(
            "production tokenizer sample must match pinned post-yield threshold overshoot"
        )
    if (
        sample.get("requested_max_characters") != requested
        or sample.get("terminal_overshoot_characters") != realized - requested
        or sample.get("stop_rule") != policy["tokenizer_training"]["stop_rule"]
    ):
        raise ValueError("tokenizer sample threshold/overshoot receipt drift")
    if not isinstance(sample.get("qa_approval_sha256"), str):
        raise ValueError("production tokenizer sample is not bound to manual corpus QA approval")
    if sample.get("trainer_visible_characters") != sample.get("characters"):
        raise ValueError("tokenizer sample trainer-visible character accounting drift")
    if sample.get("max_chars_per_document") != policy["tokenizer_training"][
        "max_chars_per_document"
    ]:
        raise ValueError("tokenizer sample document cap drift")
    dataset = load_json_strict(sample_dir / "fineweb2_manifest.json")
    validate_dataset_manifest(dataset, profile="strict")
    verify_manifest_hash(dataset)
    if dataset.get("validation_file") != "validation.parquet":
        raise ValueError("tokenizer sample validation identity drift")
    if dataset.get("metadata", {}).get("sample_scope") != "post_filter_train_only":
        raise ValueError("tokenizer sample is not declared train-only")
    if sample.get("nanochat_dataset_manifest_sha256") != dataset["canonical_sha256"]:
        raise ValueError("sample and Nanochat dataset manifests differ")
    verify_file_inventory(sample_dir, dataset["ordered_files"])
    return sample, dataset


def _capped_threshold_iterator(documents, *, max_chars: int, doc_cap: int):
    cumulative = 0
    for document in documents:
        text = document[:doc_cap]
        cumulative += len(text)
        yield text
        if cumulative > max_chars:
            return


def run_pinned_iterator_parity_fixture() -> dict[str, object]:
    """Compare our stop helper with the byte-exact 92d63d4 tok_train loop."""

    repository = Path(__file__).resolve().parents[1]
    try:
        source = subprocess.check_output(
            [
                "git",
                "-C",
                str(repository),
                "show",
                "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd:scripts/tok_train.py",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("pinned tok_train source is unavailable") from exc
    node = next(
        (
            value
            for value in ast.parse(source).body
            if isinstance(value, ast.FunctionDef) and value.name == "text_iterator"
        ),
        None,
    )
    if node is None:
        raise ValueError("pinned tok_train text_iterator is absent")
    function_source = ast.get_source_segment(source, node)
    if not isinstance(function_source, str):
        raise ValueError("cannot extract pinned tok_train iterator source")
    source_hash = hashlib.sha256(function_source.encode("utf-8")).hexdigest()
    if source_hash != PINNED_TOK_TRAIN_ITERATOR_SHA256:
        raise ValueError("pinned tok_train iterator source hash drift")
    batches = [["abcdef", "xy"], ["1234567", "sonraki"]]
    namespace = {
        "args": SimpleNamespace(max_chars=10, doc_cap=5),
        "parquets_iter_batched": lambda *, split: iter(batches),
    }
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]),
            filename="92d63d4:scripts/tok_train.py",
            mode="exec",
        ),
        namespace,
    )
    pinned = list(namespace["text_iterator"]())
    local = list(
        _capped_threshold_iterator(
            (document for batch in batches for document in batch),
            max_chars=10,
            doc_cap=5,
        )
    )
    if pinned != local or sum(map(len, local)) <= 10 or sum(map(len, local)) > 15:
        raise ValueError("local tokenizer threshold semantics differ from pinned upstream")
    return {
        "passed": True,
        "upstream_commit": "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd",
        "upstream_iterator_source_sha256": source_hash,
        "fixture_documents": [document for batch in batches for document in batch],
        "yielded_documents": local,
        "requested_max_characters": 10,
        "realized_characters": sum(map(len, local)),
        "terminal_overshoot_characters": sum(map(len, local)) - 10,
    }


def _training_texts(
    sample_dir: Path,
    dataset: dict,
    *,
    document_cap: int,
    character_limit: int,
    stats: dict[str, int],
):
    validation = dataset["validation_file"]
    consumed = 0
    for record in dataset["ordered_files"]:
        if record["path"] == validation:
            continue
        parquet = pq.ParquetFile(sample_dir / record["path"])
        for row_group_index in range(parquet.num_row_groups):
            values = parquet.read_row_group(
                row_group_index, columns=[dataset["text_column"]]
            ).column(dataset["text_column"])
            for value in values.to_pylist():
                text = "" if value is None else str(value)
                if len(text) > document_cap:
                    raise ValueError(
                        "sealed tokenizer sample contains a row above its pre-applied document cap"
                    )
                if consumed + len(text) > character_limit:
                    raise ValueError("sealed tokenizer sample exceeds its declared character count")
                consumed += len(text)
                stats["documents"] += 1
                stats["characters"] += len(text)
                yield text
    if consumed != character_limit:
        raise ValueError(
            "strict tokenizer Parquet exposure differs from sample receipt: "
            f"expected={character_limit}, consumed={consumed}"
        )


def _token_byte_lengths(tokenizer: RustBPETokenizer) -> torch.Tensor:
    special_ids = {
        int(tokenizer.encode_special(token)) for token in tokenizer.get_special_tokens()
    }
    lengths = [
        0
        if token_id in special_ids
        else len(tokenizer.enc.decode_single_token_bytes(token_id))
        for token_id in range(tokenizer.get_vocab_size())
    ]
    return torch.tensor(lengths, dtype=torch.int32, device="cpu")


def _torch_save_atomic(value: object, destination: Path) -> None:
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _validate_tokenizer(tokenizer: RustBPETokenizer) -> dict[str, object]:
    if tokenizer.get_vocab_size() != VOCAB_SIZE:
        raise ValueError(
            f"raw BPE vocabulary must be exactly {VOCAB_SIZE}, got {tokenizer.get_vocab_size()}"
        )
    if tuple(tokenizer.get_special_tokens()) != tuple(SPECIAL_TOKENS):
        # tiktoken returns a set; compare the actual ID policy separately below.
        if set(tokenizer.get_special_tokens()) != set(SPECIAL_TOKENS):
            raise ValueError("Nanochat special-token set drift")
    lexical = VOCAB_SIZE - len(SPECIAL_TOKENS)
    special_ids = {token: tokenizer.encode_special(token) for token in SPECIAL_TOKENS}
    if special_ids != {token: lexical + index for index, token in enumerate(SPECIAL_TOKENS)}:
        raise ValueError("Nanochat special-token ID ordering drift")
    mergeable = tokenizer.enc._mergeable_ranks
    missing_bytes = [value for value in range(256) if bytes([value]) not in mergeable]
    if missing_bytes:
        raise ValueError(f"byte alphabet is incomplete: {missing_bytes}")
    probes = [
        "İstanbul'da bugün hava nasıl? Türkçe ğüşiöç karakterleri.",
        "Merhaba! Nasılsın? Ben iyiyim; teşekkür ederim. 😊",
        "Ankara—İzmir 2026: %42,5 / 1.234,56 TL",
        "Kırgızca değil; Türkçe bir Unicode sınamasıdır: â, î, û.",
    ]
    for probe in probes:
        if tokenizer.decode(tokenizer.encode(probe)) != probe:
            raise ValueError(f"Unicode/Turkish round trip failed: {probe!r}")
    return {
        "exact_vocab_size": VOCAB_SIZE,
        "lexical_vocab_size": lexical,
        "special_token_ids": special_ids,
        "all_256_bytes_representable": True,
        "unicode_roundtrip_probes": len(probes),
        "split_pattern_sha256": hashlib.sha256(SPLIT_PATTERN.encode("utf-8")).hexdigest(),
    }


def _rebuild_package(output_dir: Path, receipt: dict) -> dict:
    roles = {
        "tokenizer.pkl": "tokenizer",
        "tokenizer_config.json": "runtime_config",
        "token_bytes.pt": "token_byte_lengths",
        "training_receipt.json": "training_receipt",
    }
    files = []
    for name, role in roles.items():
        path = output_dir / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing/unsafe tokenizer payload: {path}")
        files.append(
            {
                "path": name,
                "role": role,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    package = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": PACKAGE_KIND,
            "name": TOKENIZER_NAME,
            "vocab_size": VOCAB_SIZE,
            "implementation": "raw_byte_bpe",
            "training_receipt_sha256": receipt["canonical_sha256"],
            "files": files,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(output_dir / "package_manifest.json", package)
    return package


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        policy = load_corpus_policy(args.policy)
        sample, dataset = _validate_training_sample(args.sample_dir, policy)
        iterator_parity = run_pinned_iterator_parity_fixture()
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError(f"refusing non-empty tokenizer output: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        iterator_stats = {"documents": 0, "characters": 0}
        training_iterator = _training_texts(
            args.sample_dir,
            dataset,
            document_cap=int(policy["tokenizer_training"]["max_chars_per_document"]),
            character_limit=int(sample["characters"]),
            stats=iterator_stats,
        )
        started = time.monotonic()
        tokenizer = RustBPETokenizer.train_from_iterator(training_iterator, VOCAB_SIZE)
        train_time_seconds = time.monotonic() - started
        if iterator_stats["characters"] != sample["characters"]:
            raise ValueError("raw-BPE trainer character accounting differs from sample")
        if iterator_stats["documents"] != sample["documents"]:
            raise ValueError("raw-BPE trainer document accounting differs from sample")
        config = {
            "schema_version": "1.0",
            "name": TOKENIZER_NAME,
            "implementation": "bpe",
            "algorithm": "raw_byte_bpe",
            "vocab_size": VOCAB_SIZE,
            "reproducibility": {
                "ordered_training_iterator": True,
                "trainer_seed_api": "not_exposed_by_RustBPETokenizer.train_from_iterator",
                "seed_claimed": False,
            },
            "corpus_name": policy["name"],
            "max_chars": sample["requested_max_characters"],
            "realized_training_characters": sample["characters"],
            "terminal_overshoot_characters": sample[
                "terminal_overshoot_characters"
            ],
            "stop_rule": sample["stop_rule"],
            "doc_cap": sample["max_chars_per_document"],
            "iterator_stats": dict(iterator_stats),
            "sample_manifest_sha256": sample["canonical_sha256"],
            "dataset_manifest_sha256": dataset["canonical_sha256"],
            "parent_corpus_manifest_sha256": sample[
                "parent_corpus_manifest_sha256"
            ],
            "nanochat_upstream_revision": policy["tokenizer_training"][
                "nanochat_upstream_revision"
            ],
            "special_tokens": list(SPECIAL_TOKENS),
            "split_pattern": SPLIT_PATTERN,
            "requires_runtime_segmentation": False,
            "decode_strip": "",
            "pinned_iterator_parity": iterator_parity,
        }
        validation = _validate_tokenizer(tokenizer)
        tokenizer.save(str(args.output_dir))
        write_json_atomic(args.output_dir / "tokenizer_config.json", config)
        _torch_save_atomic(_token_byte_lengths(tokenizer), args.output_dir / "token_bytes.pt")
        receipt = seal_manifest(
            {
                "schema_version": "1.0",
                "kind": "turkish_raw_bpe_training_receipt",
                "name": TOKENIZER_NAME,
                "vocab_size": VOCAB_SIZE,
                "algorithm": "raw_byte_bpe",
                "reproducibility": config["reproducibility"],
                "git_commit": _git_commit(),
                "nanochat_upstream_revision": policy["tokenizer_training"][
                    "nanochat_upstream_revision"
                ],
                "sample_manifest_sha256": sample["canonical_sha256"],
                "dataset_manifest_sha256": dataset["canonical_sha256"],
                "parent_corpus_manifest_sha256": sample[
                    "parent_corpus_manifest_sha256"
                ],
                "training_characters": sample["characters"],
                "sample_characters": sample["characters"],
                "iterator_characters": iterator_stats["characters"],
                "requested_max_characters": sample["requested_max_characters"],
                "terminal_overshoot_characters": sample[
                    "terminal_overshoot_characters"
                ],
                "stop_rule": sample["stop_rule"],
                "pinned_iterator_parity": iterator_parity,
                "training_documents": sample["documents"],
                "iterator_documents": iterator_stats["documents"],
                "max_chars_per_document": sample["max_chars_per_document"],
                "qa_approval_sha256": sample.get("qa_approval_sha256"),
                "train_time_seconds": train_time_seconds,
                "validation": validation,
                "payload_sha256": {
                    "tokenizer.pkl": file_sha256(args.output_dir / "tokenizer.pkl"),
                    "tokenizer_config.json": file_sha256(
                        args.output_dir / "tokenizer_config.json"
                    ),
                    "token_bytes.pt": file_sha256(args.output_dir / "token_bytes.pt"),
                },
                "canonical_sha256": None,
            }
        )
        write_json_atomic(args.output_dir / "training_receipt.json", receipt)
        package = _rebuild_package(args.output_dir, receipt)
        quality = build_tokenizer_quality_report(
            args.output_dir,
            args.sample_dir,
            args.quality_output_dir,
            baseline_tokenizer_dir=args.baseline_tokenizer_dir,
        )
        print(
            json.dumps(
                {
                    "tokenizer": TOKENIZER_NAME,
                    "vocab_size": VOCAB_SIZE,
                    "output_dir": str(args.output_dir),
                    "package_sha256": package["canonical_sha256"],
                    "training_receipt_sha256": receipt["canonical_sha256"],
                    "quality_report_sha256": quality["canonical_sha256"],
                    "quality_approval_required": True,
                    "validation": validation,
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        return 0
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
