"""Train and seal the d32 family's raw Turkish 32,768-token BPE package."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from types import SimpleNamespace
from pathlib import Path

import pyarrow.parquet as pq
import torch
import tiktoken

from nanochat.experiment_manifest import (
    canonical_json,
    file_sha256,
    load_json_strict,
    seal_manifest,
    validate_dataset_manifest,
    verify_file_inventory,
    verify_manifest_hash,
    write_json_atomic,
)
from nanochat.tokenizer import SPECIAL_TOKENS, SPLIT_PATTERN, RustBPETokenizer
from nanochat.tokenizer_quality import (
    build_tokenizer_quality_report,
    validate_pinned_baseline_tokenizer,
)
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
    parser.add_argument("--baseline-tokenizer-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-preflight-only",
        action="store_true",
        help="verify the pinned baseline and exit before touching output directories",
    )
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
    policy_sha = hashlib.sha256(canonical_json(policy).encode("utf-8")).hexdigest()
    production_chain = sample.get("production_chain")
    required_chain = {
        "cluster_launch_receipt_sha256",
        "production_pack_plan_sha256",
        "resource_approval_sha256",
        "mixture_quality_approval_sha256",
        "data_prep_storage_gate_sha256",
        "sample_cluster_receipt_sha256",
    }
    if (
        sample.get("policy_sha256") != policy_sha
        or not isinstance(production_chain, dict)
        or set(production_chain) != required_chain
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in production_chain.values()
        )
    ):
        raise ValueError("production tokenizer sample lineage drift")
    if sample.get("trainer_visible_characters") != sample.get("characters"):
        raise ValueError("tokenizer sample trainer-visible character accounting drift")
    if sample.get("max_chars_per_document") != policy["tokenizer_training"][
        "max_chars_per_document"
    ]:
        raise ValueError("tokenizer sample document cap drift")
    traversal = sample.get("representative_traversal")
    expected_mixtures = {bucket["id"] for bucket in policy["mixture"]}
    if (
        not isinstance(traversal, dict)
        or traversal.get("algorithm")
        != "weighted_deficit_stable_rowgroup_shuffle_v2"
        or traversal.get("seed") != policy["tokenizer_training"]["sampling_seed"]
        or traversal.get("pool_manifest_sha256")
        != sample["parent_corpus_manifest_sha256"]
        or traversal.get("split") != "train"
        or traversal.get("row_group_schedule_covers_full_pool") is not True
        or {
            item.get("mixture_id")
            for item in traversal.get("by_mixture", [])
            if isinstance(item, dict)
        }
        != expected_mixtures
    ):
        raise ValueError("tokenizer sample representative traversal receipt drift")
    distribution = sample.get("sample_distribution")
    if (
        not isinstance(distribution, dict)
        or distribution.get("documents") != sample.get("documents")
        or distribution.get("characters") != sample.get("characters")
    ):
        raise ValueError("tokenizer sample distribution totals drift")
    for dimension, id_field in (
        ("by_mixture", "mixture_id"),
        ("by_source", "source_id"),
        ("by_register", "register_bucket"),
    ):
        rows = distribution.get(dimension)
        if not isinstance(rows, list) or not rows or any(
            not isinstance(item, dict)
            or not isinstance(item.get(id_field), str)
            or not item[id_field]
            or isinstance(item.get("documents"), bool)
            or not isinstance(item.get("documents"), int)
            or item["documents"] <= 0
            or isinstance(item.get("characters"), bool)
            or not isinstance(item.get("characters"), int)
            or item["characters"] <= 0
            or isinstance(item.get("document_share"), bool)
            or not isinstance(item.get("document_share"), (int, float))
            or isinstance(item.get("character_share"), bool)
            or not isinstance(item.get("character_share"), (int, float))
            for item in rows or ()
        ):
            raise ValueError(f"tokenizer sample {dimension} rows drift")
        if (
            len({item[id_field] for item in rows}) != len(rows)
            or sum(item["documents"] for item in rows)
            != sample["documents"]
            or sum(item["characters"] for item in rows)
            != sample["characters"]
            or not math.isclose(
                sum(float(item["document_share"]) for item in rows),
                1.0,
                rel_tol=0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                sum(float(item["character_share"]) for item in rows),
                1.0,
                rel_tol=0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"tokenizer sample {dimension} counters/shares drift")
    holdout = sample.get("quality_holdout")
    if not isinstance(holdout, dict):
        raise ValueError("tokenizer held-out selection receipt is missing")
    for key in (
        "selected_documents",
        "minimum_documents",
        "selected_utf8_bytes",
        "minimum_utf8_bytes",
    ):
        if isinstance(holdout.get(key), bool) or not isinstance(holdout.get(key), int):
            raise ValueError(f"tokenizer held-out {key} counter drift")
    if (
        holdout.get("split") != "val"
        or holdout.get("algorithm")
        != policy["tokenizer_training"]["holdout"]["selection"]
        or holdout.get("seed") != policy["tokenizer_training"]["holdout"]["seed"]
        or holdout.get("complete_available_stratum_coverage") is not True
        or holdout.get("available_strata") != holdout.get("selected_strata")
        or holdout.get("target_documents_per_available_stratum")
        != policy["tokenizer_training"]["holdout"][
            "target_documents_per_available_stratum"
        ]
        or (
            holdout["selected_documents"] < holdout["minimum_documents"]
            and holdout["selected_utf8_bytes"] < holdout["minimum_utf8_bytes"]
        )
        or sample.get("quality_gate_policy")
        != policy["tokenizer_training"]["quality_gate"]
        or sample.get("baseline_tokenizer")
        != policy["tokenizer_training"]["baseline"]
    ):
        raise ValueError("tokenizer held-out selection/quality policy drift")
    for item in holdout.get("strata", []):
        if (
            not isinstance(item, dict)
            or item.get("target_documents")
            != holdout["target_documents_per_available_stratum"]
            or item.get("coverage_floor_documents")
            != min(
                int(item.get("eligible_documents", -1)),
                holdout["target_documents_per_available_stratum"],
            )
            or int(item.get("selected_documents", -1))
            < int(item.get("coverage_floor_documents", 0))
        ):
            raise ValueError("tokenizer held-out per-stratum coverage floor drift")
    dataset = load_json_strict(sample_dir / "fineweb2_manifest.json")
    validate_dataset_manifest(dataset, profile="strict")
    verify_manifest_hash(dataset)
    if dataset.get("validation_file") != "validation.parquet":
        raise ValueError("tokenizer sample validation identity drift")
    if dataset.get("metadata", {}).get("sample_scope") != "post_filter_train_only":
        raise ValueError("tokenizer sample is not declared train-only")
    if (
        dataset.get("metadata", {}).get("quality_holdout_scope")
        != "post_filter_val_only"
        or dataset.get("metadata", {}).get("quality_holdout") != holdout
    ):
        raise ValueError("tokenizer held-out dataset metadata drift")
    if sample.get("nanochat_dataset_manifest_sha256") != dataset["canonical_sha256"]:
        raise ValueError("sample and Nanochat dataset manifests differ")
    if (
        dataset.get("metadata", {}).get("policy_sha256") != policy_sha
        or dataset.get("metadata", {}).get("production_chain") != production_chain
    ):
        raise ValueError("tokenizer sample dataset lineage drift")
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
    stats: dict[str, object],
):
    validation = dataset["validation_file"]
    consumed = 0
    for record in dataset["ordered_files"]:
        if record["path"] == validation:
            continue
        parquet = pq.ParquetFile(sample_dir / record["path"])
        for row_group_index in range(parquet.num_row_groups):
            rows = parquet.read_row_group(
                row_group_index,
                columns=[
                    dataset["text_column"],
                    "mixture_id",
                    "source_id",
                    "register_bucket",
                ],
            ).to_pylist()
            for row in rows:
                value = row[dataset["text_column"]]
                text = "" if value is None else str(value)
                if len(text) > document_cap:
                    raise ValueError(
                        "sealed tokenizer sample contains a row above its pre-applied document cap"
                    )
                if consumed + len(text) > character_limit:
                    raise ValueError("sealed tokenizer sample exceeds its declared character count")
                consumed += len(text)
                stats["documents"] = int(stats["documents"]) + 1
                stats["characters"] = int(stats["characters"]) + len(text)
                for prefix, value in (
                    ("mixture", str(row["mixture_id"])),
                    ("source", str(row["source_id"])),
                    (
                        "register",
                        str(row.get("register_bucket") or "not_applicable"),
                    ),
                ):
                    stats[f"{prefix}_documents"][value] += 1
                    stats[f"{prefix}_characters"][value] += len(text)
                yield text
    if consumed != character_limit:
        raise ValueError(
            "strict tokenizer Parquet exposure differs from sample receipt: "
            f"expected={character_limit}, consumed={consumed}"
        )


def _distribution_rows(
    documents: Mapping[str, int],
    characters: Mapping[str, int],
    *,
    id_field: str,
) -> list[dict[str, object]]:
    total_documents = sum(documents.values())
    total_characters = sum(characters.values())
    if total_documents <= 0 or total_characters <= 0:
        raise ValueError("trainer-visible sample distribution is empty")
    return [
        {
            id_field: key,
            "documents": int(documents[key]),
            "characters": int(characters[key]),
            "document_share": documents[key] / total_documents,
            "character_share": characters[key] / total_characters,
        }
        for key in sorted(set(documents) | set(characters))
    ]


def _realized_sample_distribution(stats: Mapping[str, object]) -> dict[str, object]:
    return {
        "documents": int(stats["documents"]),
        "characters": int(stats["characters"]),
        "by_mixture": _distribution_rows(
            stats["mixture_documents"],
            stats["mixture_characters"],
            id_field="mixture_id",
        ),
        "by_source": _distribution_rows(
            stats["source_documents"],
            stats["source_characters"],
            id_field="source_id",
        ),
        "by_register": _distribution_rows(
            stats["register_documents"],
            stats["register_characters"],
            id_field="register_bucket",
        ),
    }


def _validate_realized_sample_distribution(
    stats: Mapping[str, object], declared: Mapping[str, object]
) -> dict[str, object]:
    realized = _realized_sample_distribution(stats)
    if realized != dict(declared):
        raise ValueError(
            "trainer-visible sample distribution differs from sealed sample receipt"
        )
    return realized


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


def _export_and_verify_canonical_tiktoken(
    tokenizer: RustBPETokenizer,
    output_dir: Path,
    token_byte_lengths: torch.Tensor,
) -> dict[str, object]:
    """Export canonical rank-sorted tiktoken BPE bytes and reconstruct runtime IDs."""

    mergeable = dict(tokenizer.enc._mergeable_ranks)
    lexical = VOCAB_SIZE - len(SPECIAL_TOKENS)
    if len(mergeable) != lexical or set(mergeable.values()) != set(range(lexical)):
        raise ValueError("mergeable ranks are not a dense lexical token-ID range")
    export_path = output_dir / "tokenizer.tiktoken"
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_dir,
            prefix=".tokenizer.tiktoken.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            for token_bytes, rank in sorted(mergeable.items(), key=lambda item: item[1]):
                handle.write(base64.b64encode(token_bytes))
                handle.write(b" ")
                handle.write(str(rank).encode("ascii"))
                handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, export_path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)

    reconstructed: dict[bytes, int] = {}
    for line_number, raw_line in enumerate(export_path.read_bytes().splitlines(), 1):
        parts = raw_line.split(b" ")
        if len(parts) != 2:
            raise ValueError(f"non-canonical tokenizer.tiktoken line {line_number}")
        try:
            token_bytes = base64.b64decode(parts[0], validate=True)
            rank = int(parts[1].decode("ascii"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"invalid tokenizer.tiktoken line {line_number}"
            ) from exc
        if token_bytes in reconstructed or rank != line_number - 1:
            raise ValueError("tokenizer.tiktoken ranks are duplicated or non-canonical")
        reconstructed[token_bytes] = rank
    if reconstructed != mergeable:
        raise ValueError("tokenizer.tiktoken reconstruction differs from trained ranks")
    special_tokens = {
        token: lexical + index for index, token in enumerate(SPECIAL_TOKENS)
    }
    rebuilt = tiktoken.Encoding(
        name=TOKENIZER_NAME,
        pat_str=SPLIT_PATTERN,
        mergeable_ranks=reconstructed,
        special_tokens=special_tokens,
    )
    if rebuilt.n_vocab != VOCAB_SIZE:
        raise ValueError("canonical tokenizer reconstruction vocabulary drift")
    for rank in range(lexical):
        expected = tokenizer.enc.decode_single_token_bytes(rank)
        actual = rebuilt.decode_single_token_bytes(rank)
        if expected != actual:
            raise ValueError(f"canonical tokenizer rank/ID drift at {rank}")
        if int(token_byte_lengths[rank].item()) != len(actual):
            raise ValueError(f"token-byte table reconstruction drift at {rank}")
    for token, token_id in special_tokens.items():
        if rebuilt.encode_single_token(token) != token_id:
            raise ValueError(f"canonical special-token ID drift for {token}")
        if int(token_byte_lengths[token_id].item()) != 0:
            raise ValueError(f"special token-byte length must be zero for {token}")
    probes = (
        "İstanbul'da bugün hava nasıl?",
        "TÜRKİYE Türkiye türkiye; IĞDIR Iğdır ığdır.",
        "sorumluluklarımızdakilerdenmişsinizcesine",
        "Ankara—İzmir 2026: %42,5 😊",
    )
    for probe in probes:
        original_ids = tokenizer.encode(probe)
        rebuilt_ids = rebuilt.encode_ordinary(probe)
        if rebuilt_ids != original_ids or rebuilt.decode(rebuilt_ids) != probe:
            raise ValueError(f"canonical tokenizer probe reconstruction failed: {probe!r}")
    return {
        "path": export_path.name,
        "format": "tiktoken_bpe_base64_token_space_decimal_rank_newline",
        "sha256": file_sha256(export_path),
        "lexical_ranks": lexical,
        "dense_rank_id_identity_verified": True,
        "special_token_id_order_verified": True,
        "probe_id_sequences_verified": len(probes),
        "token_byte_lengths_reconstructed": VOCAB_SIZE,
    }


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
        "tokenizer.tiktoken": "canonical_tiktoken_export",
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
            "policy_sha256": receipt["policy_sha256"],
            "production_chain": receipt["production_chain"],
            "parent_corpus_manifest_sha256": receipt[
                "parent_corpus_manifest_sha256"
            ],
            "qa_approval_sha256": receipt["qa_approval_sha256"],
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
        _baseline_tokenizer, baseline_identity = validate_pinned_baseline_tokenizer(
            args.baseline_tokenizer_dir,
            policy["tokenizer_training"]["baseline"],
        )
        if args.baseline_preflight_only:
            print(json.dumps(baseline_identity, ensure_ascii=False, sort_keys=True))
            return 0
        sample, dataset = _validate_training_sample(args.sample_dir, policy)
        iterator_parity = run_pinned_iterator_parity_fixture()
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError(f"refusing non-empty tokenizer output: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        iterator_stats: dict[str, object] = {
            "documents": 0,
            "characters": 0,
            "mixture_documents": Counter(),
            "mixture_characters": Counter(),
            "source_documents": Counter(),
            "source_characters": Counter(),
            "register_documents": Counter(),
            "register_characters": Counter(),
        }
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
        realized_distribution = _validate_realized_sample_distribution(
            iterator_stats, sample["sample_distribution"]
        )
        public_iterator_stats = {
            "documents": int(iterator_stats["documents"]),
            "characters": int(iterator_stats["characters"]),
        }
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
            "iterator_stats": public_iterator_stats,
            "representative_traversal": sample["representative_traversal"],
            "sample_distribution": realized_distribution,
            "quality_holdout": sample["quality_holdout"],
            "quality_gate_policy": policy["tokenizer_training"]["quality_gate"],
            "baseline_tokenizer": policy["tokenizer_training"]["baseline"],
            "baseline_identity": baseline_identity,
            "sample_manifest_sha256": sample["canonical_sha256"],
            "dataset_manifest_sha256": dataset["canonical_sha256"],
            "parent_corpus_manifest_sha256": sample[
                "parent_corpus_manifest_sha256"
            ],
            "policy_sha256": sample["policy_sha256"],
            "production_chain": sample["production_chain"],
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
        token_byte_lengths = _token_byte_lengths(tokenizer)
        _torch_save_atomic(token_byte_lengths, args.output_dir / "token_bytes.pt")
        canonical_export = _export_and_verify_canonical_tiktoken(
            tokenizer, args.output_dir, token_byte_lengths
        )
        validation["canonical_export"] = canonical_export
        config["canonical_export"] = canonical_export
        write_json_atomic(args.output_dir / "tokenizer_config.json", config)
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
                "policy_sha256": sample["policy_sha256"],
                "production_chain": sample["production_chain"],
                "training_characters": sample["characters"],
                "sample_characters": sample["characters"],
                "iterator_characters": public_iterator_stats["characters"],
                "requested_max_characters": sample["requested_max_characters"],
                "terminal_overshoot_characters": sample[
                    "terminal_overshoot_characters"
                ],
                "stop_rule": sample["stop_rule"],
                "pinned_iterator_parity": iterator_parity,
                "training_documents": sample["documents"],
                "iterator_documents": public_iterator_stats["documents"],
                "max_chars_per_document": sample["max_chars_per_document"],
                "representative_traversal": sample["representative_traversal"],
                "sample_distribution": realized_distribution,
                "quality_holdout": sample["quality_holdout"],
                "quality_gate_policy": policy["tokenizer_training"]["quality_gate"],
                "baseline_tokenizer": policy["tokenizer_training"]["baseline"],
                "baseline_identity": baseline_identity,
                "qa_approval_sha256": sample.get("qa_approval_sha256"),
                "train_time_seconds": train_time_seconds,
                "validation": validation,
                "payload_sha256": {
                    "tokenizer.pkl": file_sha256(args.output_dir / "tokenizer.pkl"),
                    "tokenizer.tiktoken": file_sha256(
                        args.output_dir / "tokenizer.tiktoken"
                    ),
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
