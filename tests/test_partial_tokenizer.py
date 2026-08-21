from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest
import tiktoken
import torch

from nanochat.experiment_manifest import (
    canonical_json,
    file_sha256,
    seal_manifest,
    write_json_atomic,
)
from nanochat.partial_tokenizer import (
    EXPECTED_FILES,
    EXPECTED_RANKS,
    INCOMPLETE_GATES,
    PARTIAL_PACKAGE_KIND,
    PARTIAL_RECEIPT_KIND,
    PINNED_NAME,
    PINNED_PACKAGE_SHA256,
    PINNED_RECEIPT_SHA256,
    PartialTokenizerPackageError,
    load_verified_partial_token_bytes,
    load_verified_partial_tokenizer,
    verify_partial_tokenizer_package,
)
from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS, SPLIT_PATTERN


VOCAB_SIZE = 32768
LEXICAL_SIZE = VOCAB_SIZE - len(SPECIAL_TOKENS)
PAYLOAD_FILES = EXPECTED_FILES[:-1]
MISSING_RECEIPTS = {112, 119, 126}


def _synthetic_tokenizer() -> RustBPETokenizer:
    lexical_tokens = [bytes([value]) for value in range(256)]
    for first in range(256):
        for second in range(256):
            if len(lexical_tokens) == LEXICAL_SIZE:
                break
            lexical_tokens.append(bytes((first, second)))
        if len(lexical_tokens) == LEXICAL_SIZE:
            break
    ranks = {token: rank for rank, token in enumerate(lexical_tokens)}
    encoding = tiktoken.Encoding(
        name="partial_fixture",
        pat_str=SPLIT_PATTERN,
        mergeable_ranks=ranks,
        special_tokens={
            token: LEXICAL_SIZE + index
            for index, token in enumerate(SPECIAL_TOKENS)
        },
    )
    return RustBPETokenizer(encoding, "<|bos|>")


def _write_canonical_export(root: Path, tokenizer: RustBPETokenizer) -> dict:
    ordered = sorted(tokenizer.enc._mergeable_ranks.items(), key=lambda item: item[1])
    raw = b"".join(
        base64.b64encode(token) + b" " + str(rank).encode("ascii") + b"\n"
        for token, rank in ordered
    )
    path = root / "tokenizer.tiktoken"
    path.write_bytes(raw)
    return {
        "path": path.name,
        "format": "tiktoken_bpe_base64_token_space_decimal_rank_newline",
        "sha256": file_sha256(path),
        "lexical_ranks": LEXICAL_SIZE,
        "dense_rank_id_identity_verified": True,
        "special_token_id_order_verified": True,
        "probe_id_sequences_verified": 4,
        "token_byte_lengths_reconstructed": VOCAB_SIZE,
    }


def _file_records(root: Path, names: tuple[str, ...]) -> list[dict]:
    return [
        {
            "path": name,
            "size_bytes": (root / name).stat().st_size,
            "sha256": file_sha256(root / name),
        }
        for name in names
    ]


def _input_inventory() -> list[dict]:
    result = []
    for index, rank in enumerate(EXPECTED_RANKS):
        receipt_present = rank not in MISSING_RECEIPTS
        result.append(
            {
                "rank": rank,
                "source_id": "fixture_source",
                "path": f"/fixture/objects/{rank:05d}/candidates.parquet",
                "size_bytes": 1000 + index,
                "sha256": hashlib.sha256(f"candidate:{rank}".encode()).hexdigest(),
                "rows": 10 + index,
                "row_groups": 1,
                "object_receipt_present": receipt_present,
                "object_receipt_sha256": (
                    hashlib.sha256(f"receipt:{rank}".encode()).hexdigest()
                    if receipt_present
                    else None
                ),
            }
        )
    return result


def _write_fixture(root: Path) -> tuple[Path, dict, dict]:
    root.mkdir()
    tokenizer = _synthetic_tokenizer()
    tokenizer.save(str(root))
    token_bytes = torch.tensor(
        [
            len(tokenizer.decode_single_token_bytes(token_id))
            for token_id in range(LEXICAL_SIZE)
        ]
        + [0] * len(SPECIAL_TOKENS),
        dtype=torch.int32,
    )
    torch.save(token_bytes, root / "token_bytes.pt")
    canonical_export = _write_canonical_export(root, tokenizer)
    training_characters = 2_000_000_123
    training_documents = 800_000
    distribution = [
        {
            "mixture_id": "fixture_general",
            "target_share": 1.0,
            "documents": training_documents,
            "characters": training_characters,
            "realized_share": 1.0,
        }
    ]
    config = {
        "schema_version": "1.0",
        "name": PINNED_NAME,
        "implementation": "raw_byte_bpe",
        "vocab_size": VOCAB_SIZE,
        "split_pattern": SPLIT_PATTERN,
        "special_tokens": list(SPECIAL_TOKENS),
        "max_chars": 2_000_000_000,
        "realized_training_characters": training_characters,
        "doc_cap": 10_000,
        "sample_distribution": distribution,
        "production_eligible": False,
        "requires_runtime_segmentation": False,
        "decode_strip": "",
        "canonical_export": canonical_export,
    }
    write_json_atomic(root / "tokenizer_config.json", config)
    inventory = _input_inventory()
    inventory_sha = hashlib.sha256(
        canonical_json(inventory).encode("utf-8")
    ).hexdigest()
    validation = {
        "exact_vocab_size": VOCAB_SIZE,
        "lexical_vocab_size": LEXICAL_SIZE,
        "special_token_ids": {
            token: LEXICAL_SIZE + index
            for index, token in enumerate(SPECIAL_TOKENS)
        },
        "all_256_bytes_representable": True,
        "unicode_roundtrip_probes": 4,
        "split_pattern_sha256": hashlib.sha256(
            SPLIT_PATTERN.encode("utf-8")
        ).hexdigest(),
        "canonical_export": canonical_export,
    }
    receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": PARTIAL_RECEIPT_KIND,
            "name": PINNED_NAME,
            "vocab_size": VOCAB_SIZE,
            "algorithm": "raw_byte_bpe",
            "expected_ranks": list(EXPECTED_RANKS),
            "input_inventory": inventory,
            "input_inventory_sha256": inventory_sha,
            "object_receipts_present": 15,
            "object_receipts_missing": sorted(MISSING_RECEIPTS),
            "training_documents": training_documents,
            "training_characters": training_characters,
            "requested_max_characters": 2_000_000_000,
            "terminal_overshoot_characters": 123,
            "max_chars_per_document": 10_000,
            "sample_distribution": distribution,
            "validation": validation,
            "payload": _file_records(root, PAYLOAD_FILES),
            "production_eligible": False,
            "global_near_dedup_completed": False,
            "sample_cluster_completed": False,
            "manual_corpus_qa_completed": False,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(root / "partial_training_receipt.json", receipt)
    package = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": PARTIAL_PACKAGE_KIND,
            "name": PINNED_NAME,
            "vocab_size": VOCAB_SIZE,
            "training_receipt_sha256": receipt["canonical_sha256"],
            "input_inventory_sha256": inventory_sha,
            "production_eligible": False,
            "files": _file_records(root, EXPECTED_FILES),
            "canonical_sha256": None,
        }
    )
    path = root / "package_manifest.json"
    write_json_atomic(path, package)
    return path, package, receipt


def _reseal(root: Path) -> tuple[dict, dict]:
    receipt = json.loads(
        (root / "partial_training_receipt.json").read_text(encoding="utf-8")
    )
    inventory_sha = hashlib.sha256(
        canonical_json(receipt["input_inventory"]).encode("utf-8")
    ).hexdigest()
    receipt["input_inventory_sha256"] = inventory_sha
    receipt["payload"] = _file_records(root, PAYLOAD_FILES)
    receipt["canonical_sha256"] = None
    receipt = seal_manifest(receipt)
    write_json_atomic(root / "partial_training_receipt.json", receipt)
    package = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": PARTIAL_PACKAGE_KIND,
            "name": PINNED_NAME,
            "vocab_size": VOCAB_SIZE,
            "training_receipt_sha256": receipt["canonical_sha256"],
            "input_inventory_sha256": inventory_sha,
            "production_eligible": False,
            "files": _file_records(root, EXPECTED_FILES),
            "canonical_sha256": None,
        }
    )
    write_json_atomic(root / "package_manifest.json", package)
    return package, receipt


def _verify_fixture(path: Path, package: dict, receipt: dict):
    return verify_partial_tokenizer_package(
        path,
        expected_sha256=package["canonical_sha256"],
        expected_training_receipt_sha256=receipt["canonical_sha256"],
    )


def test_partial_verifier_accepts_synthetic_fixture_and_loads_runtime(
    tmp_path: Path,
) -> None:
    path, package, receipt = _write_fixture(tmp_path / "tokenizer")
    verified = _verify_fixture(path, package, receipt)
    assert verified.canonical_sha256 == package["canonical_sha256"]
    assert verified.receipt_sha256 == receipt["canonical_sha256"]
    assert verified.receipt["expected_ranks"] == list(EXPECTED_RANKS)
    assert all(verified.receipt[gate] is False for gate in INCOMPLETE_GATES)
    assert load_verified_partial_tokenizer(verified).get_vocab_size() == VOCAB_SIZE
    token_bytes = load_verified_partial_token_bytes(verified)
    assert token_bytes.shape == (VOCAB_SIZE,)
    assert token_bytes.dtype == torch.int32
    assert token_bytes[-len(SPECIAL_TOKENS) :].tolist() == [0] * len(SPECIAL_TOKENS)


def test_partial_verifier_defaults_pin_exact_uhem_identities(tmp_path: Path) -> None:
    path, _package, _receipt = _write_fixture(tmp_path / "tokenizer")
    assert PINNED_PACKAGE_SHA256 == (
        "909bfa20516c79b7349d3e35aacd655ef584aac431c055966366cf6e1545d871"
    )
    assert PINNED_RECEIPT_SHA256 == (
        "5014766c50fee069fde94806a9eb82de9c42e78d8d55632c4e37f2ebed94c445"
    )
    with pytest.raises(PartialTokenizerPackageError, match="package SHA-256"):
        verify_partial_tokenizer_package(path)


@pytest.mark.parametrize("gate", INCOMPLETE_GATES)
def test_partial_verifier_requires_every_incomplete_gate_false(
    tmp_path: Path, gate: str
) -> None:
    path, _package, _receipt = _write_fixture(tmp_path / "tokenizer")
    receipt_path = path.parent / "partial_training_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt[gate] = True
    write_json_atomic(receipt_path, receipt)
    package, receipt = _reseal(path.parent)
    with pytest.raises(PartialTokenizerPackageError, match=f"{gate}=false"):
        _verify_fixture(path, package, receipt)


def test_partial_verifier_rejects_rank_and_exact_file_inventory_drift(
    tmp_path: Path,
) -> None:
    path, _package, _receipt = _write_fixture(tmp_path / "tokenizer")
    receipt_path = path.parent / "partial_training_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["expected_ranks"][-1] = 211
    write_json_atomic(receipt_path, receipt)
    package, receipt = _reseal(path.parent)
    with pytest.raises(PartialTokenizerPackageError, match="training receipt drifted"):
        _verify_fixture(path, package, receipt)

    clean_path, clean_package, clean_receipt = _write_fixture(tmp_path / "clean")
    (clean_path.parent / "unrecorded.txt").write_text("drift\n", encoding="utf-8")
    with pytest.raises(PartialTokenizerPackageError, match="unrecorded files"):
        _verify_fixture(clean_path, clean_package, clean_receipt)


def test_partial_verifier_reconstructs_canonical_export_and_token_bytes(
    tmp_path: Path,
) -> None:
    path, _package, _receipt = _write_fixture(tmp_path / "tokenizer")
    export = path.parent / "tokenizer.tiktoken"
    lines = export.read_bytes().splitlines()
    lines[0] = lines[0].split(b" ")[0] + b" 1"
    export.write_bytes(b"\n".join(lines) + b"\n")
    config_path = path.parent / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["canonical_export"]["sha256"] = file_sha256(export)
    write_json_atomic(config_path, config)
    receipt_path = path.parent / "partial_training_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["validation"]["canonical_export"] = config["canonical_export"]
    write_json_atomic(receipt_path, receipt)
    package, receipt = _reseal(path.parent)
    with pytest.raises(PartialTokenizerPackageError, match="dense and canonical"):
        _verify_fixture(path, package, receipt)

    token_path, _package, _receipt = _write_fixture(tmp_path / "token-bytes")
    lengths = torch.load(
        token_path.parent / "token_bytes.pt", map_location="cpu", weights_only=True
    )
    lengths[0] += 1
    torch.save(lengths, token_path.parent / "token_bytes.pt")
    package, receipt = _reseal(token_path.parent)
    with pytest.raises(PartialTokenizerPackageError, match="token-byte table content"):
        _verify_fixture(token_path, package, receipt)


def test_partial_verifier_checks_byte_coverage_and_dense_special_ids(
    tmp_path: Path,
) -> None:
    path, _package, _receipt = _write_fixture(tmp_path / "tokenizer")
    receipt_path = path.parent / "partial_training_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["validation"]["all_256_bytes_representable"] = False
    write_json_atomic(receipt_path, receipt)
    package, receipt = _reseal(path.parent)
    with pytest.raises(PartialTokenizerPackageError, match="validation/canonical"):
        _verify_fixture(path, package, receipt)

    special_path, _package, _receipt = _write_fixture(tmp_path / "specials")
    tokenizer = RustBPETokenizer.from_directory(str(special_path.parent))
    specials = dict(tokenizer.enc._special_tokens)
    first, second = SPECIAL_TOKENS[:2]
    specials[first], specials[second] = specials[second], specials[first]
    wrong = tiktoken.Encoding(
        name="wrong_specials",
        pat_str=SPLIT_PATTERN,
        mergeable_ranks=dict(tokenizer.enc._mergeable_ranks),
        special_tokens=specials,
    )
    RustBPETokenizer(wrong, first).save(str(special_path.parent))
    package, receipt = _reseal(special_path.parent)
    with pytest.raises(PartialTokenizerPackageError, match="special-ID policy"):
        _verify_fixture(special_path, package, receipt)
