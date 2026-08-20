from __future__ import annotations

import copy
import hashlib
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import tiktoken

import scripts.train_turkish_raw_bpe as trainer
from nanochat.experiment_manifest import (
    canonical_json,
    file_sha256,
    load_json_strict,
    seal_manifest,
    write_json_atomic,
)
from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS, SPLIT_PATTERN
from nanochat.tokenizer_quality import (
    _FIXED_TURKISH_PROBES,
    _validation_rows,
    evaluate_tokenizer_quality_gate,
    seal_tokenizer_quality_approval,
    validate_pinned_baseline_tokenizer,
    validate_tokenizer_quality_gate,
)
from nanochat.turkish_corpus import (
    FragmentWriter,
    TOKENIZER_BASELINE_V1,
    TOKENIZER_QUALITY_GATE_V1,
    TurkishCorpusError,
    load_corpus_policy,
    write_pool_ownership_manifest,
    write_tokenizer_sample,
)


def test_production_baseline_inventory_hash_is_canonical() -> None:
    assert hashlib.sha256(
        canonical_json(TOKENIZER_BASELINE_V1["files"]).encode("utf-8")
    ).hexdigest() == TOKENIZER_BASELINE_V1["payload_inventory_sha256"]
    assert TOKENIZER_BASELINE_V1["token_byte_table_semantics"] == (
        "legacy_decoded_utf8_replacement_v1"
    )
    assert TOKENIZER_BASELINE_V1["raw_byte_length_mismatch_count"] == 159


def _row(
    text: str,
    index: int,
    *,
    source: str,
    mixture: str,
    register: str,
) -> dict:
    return {
        "text": text,
        "source_id": source,
        "mixture_id": mixture,
        "document_id": f"doc-{index:04d}",
        "url": "",
        "cluster_id": f"{index:064x}",
        "shuffle_key": hashlib.sha256(f"shuffle-{index}".encode()).hexdigest(),
        "quality_score": 1.0,
        "register_bucket": register,
    }


def _write_pool_file(
    root: Path,
    relative: str,
    rows: list[dict],
    *,
    split: str,
    mixture: str,
) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=FragmentWriter._schema),
        path,
        compression="zstd",
        row_group_size=1,
    )
    parquet = pq.ParquetFile(path)
    return {
        "path": relative,
        "split": split,
        "mixture_id": mixture,
        "rows": len(rows),
        "row_groups": parquet.num_row_groups,
        "shuffle_bucket_min": 0,
        "shuffle_bucket_max": 255,
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _fixture_pool(tmp_path: Path) -> tuple[Path, dict]:
    policy = copy.deepcopy(
        load_corpus_policy("configs/pretrain/tr_d32_turkish_general_v2.json")
    )
    policy["tokenizer_training"]["holdout"]["min_documents"] = 4
    policy["tokenizer_training"]["holdout"]["min_utf8_bytes"] = 1_000_000
    root = tmp_path / "pool"
    root.mkdir()
    files = [
        _write_pool_file(
            root,
            "pool/train/fineweb2_hq_general/a.parquet",
            [
                _row(
                    f"Gündelik Türkçe konuşma örneği {index}; bugün nasılsınız?",
                    index,
                    source="fineweb2_hq_tr",
                    mixture="fineweb2_hq_general",
                    register="NA",
                )
                for index in range(1, 7)
            ],
            split="train",
            mixture="fineweb2_hq_general",
        ),
        _write_pool_file(
            root,
            "pool/train/macocu_conversation/b.parquet",
            [
                _row(
                    f"Forumdaki arkadaş {index} şöyle dedi: Bence yarın görüşelim.",
                    100 + index,
                    source="macocu_genre_tr",
                    mixture="macocu_conversation",
                    register="Forum",
                )
                for index in range(1, 7)
            ],
            split="train",
            mixture="macocu_conversation",
        ),
        _write_pool_file(
            root,
            "pool/val/fineweb2_hq_general/c.parquet",
            [
                _row(
                    f"Doğrulama metni {index}: İstanbul'da hayat bugün hareketli.",
                    200 + index,
                    source="fineweb2_hq_tr",
                    mixture="fineweb2_hq_general",
                    register="NA",
                )
                for index in range(1, 4)
            ],
            split="val",
            mixture="fineweb2_hq_general",
        ),
        _write_pool_file(
            root,
            "pool/val/macocu_conversation/d.parquet",
            [
                _row(
                    "Forum doğrulaması: Sizce bu akşam ne pişirelim?",
                    301,
                    source="macocu_genre_tr",
                    mixture="macocu_conversation",
                    register="Forum",
                ),
                _row(
                    "Görüş yazısı: Toplu taşıma daha sık çalışmalı.",
                    302,
                    source="macocu_genre_tr",
                    mixture="macocu_conversation",
                    register="OP",
                ),
                _row(
                    "Bir başka forum yanıtı: Katılıyorum, iyi fikir.",
                    303,
                    source="macocu_genre_tr",
                    mixture="macocu_conversation",
                    register="Forum",
                ),
            ],
            split="val",
            mixture="macocu_conversation",
        ),
    ]
    policy_sha = hashlib.sha256(canonical_json(policy).encode()).hexdigest()
    manifest = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "turkish_pretrain_corpus",
            "name": policy["name"],
            "stage": "filtered_pool",
            "backend_scope": "reference_smoke_only",
            "policy_sha256": policy_sha,
            "files": sorted(files, key=lambda item: item["path"]),
            "canonical_sha256": None,
        }
    )
    write_json_atomic(root / "corpus_manifest.json", manifest)
    write_pool_ownership_manifest(root, manifest)
    return root, policy


def test_sample_traverses_stable_full_pool_schedule_and_seals_stratified_holdout(
    tmp_path: Path,
) -> None:
    pool, policy = _fixture_pool(tmp_path)
    outputs = []
    for name in ("left", "right"):
        output = tmp_path / name
        receipt = write_tokenizer_sample(
            pool,
            policy,
            output,
            git_commit="a" * 40,
            max_chars=180,
            allow_reference_pool=True,
        )
        outputs.append((output, receipt))
    left, right = outputs
    assert left[1]["representative_traversal"] == right[1]["representative_traversal"]
    assert left[1]["sample_distribution"] == right[1]["sample_distribution"]
    assert left[1]["quality_holdout"] == right[1]["quality_holdout"]
    assert (left[0] / "train-00000.parquet").read_bytes() == (
        right[0] / "train-00000.parquet"
    ).read_bytes()
    traversal = left[1]["representative_traversal"]
    assert traversal["algorithm"] == "weighted_deficit_stable_rowgroup_shuffle_v2"
    assert traversal["row_group_schedule_covers_full_pool"] is True
    eligible = sum(item["eligible_row_groups"] for item in traversal["by_mixture"])
    assert eligible == 12
    distribution = left[1]["sample_distribution"]
    assert sum(item["documents"] for item in distribution["by_source"]) == distribution[
        "documents"
    ]
    assert sum(item["character_share"] for item in distribution["by_mixture"]) == pytest.approx(
        1.0
    )
    holdout = left[1]["quality_holdout"]
    assert holdout["split"] == "val"
    assert holdout["selected_documents"] >= 4
    assert holdout["available_strata"] == holdout["selected_strata"] == 3
    assert holdout["complete_available_stratum_coverage"] is True
    assert holdout["target_documents_per_available_stratum"] == 32
    assert holdout["selected_documents"] == 6
    assert all(
        item["selected_documents"] == item["coverage_floor_documents"]
        == item["eligible_documents"]
        for item in holdout["strata"]
    )


def test_quality_holdout_rejects_dataset_substitution_and_path_escape(
    tmp_path: Path,
) -> None:
    pool, policy = _fixture_pool(tmp_path)
    sample_dir = tmp_path / "sample"
    write_tokenizer_sample(
        pool,
        policy,
        sample_dir,
        git_commit="a" * 40,
        max_chars=180,
        allow_reference_pool=True,
    )
    dataset_path = sample_dir / "fineweb2_manifest.json"
    original_dataset = load_json_strict(dataset_path)
    substituted = copy.deepcopy(original_dataset)
    substituted["metadata"]["quality_holdout"]["seed"] = "substituted"
    substituted["canonical_sha256"] = None
    write_json_atomic(dataset_path, seal_manifest(substituted))
    with pytest.raises(TurkishCorpusError, match="not bound"):
        _validation_rows(sample_dir)

    write_json_atomic(dataset_path, original_dataset)
    escaped = copy.deepcopy(original_dataset)
    validation_record = next(
        item
        for item in escaped["ordered_files"]
        if item["path"] == escaped["validation_file"]
    )
    external = tmp_path / "external-validation.parquet"
    shutil.copy2(sample_dir / escaped["validation_file"], external)
    escaped_path = "../external-validation.parquet"
    validation_record["path"] = escaped_path
    validation_record["size_bytes"] = external.stat().st_size
    validation_record["sha256"] = file_sha256(external)
    escaped["validation_file"] = escaped_path
    escaped["canonical_sha256"] = None
    escaped = seal_manifest(escaped)
    write_json_atomic(dataset_path, escaped)
    sample_path = sample_dir / "tokenizer_sample_manifest.json"
    sample = load_json_strict(sample_path)
    sample["nanochat_dataset_manifest_sha256"] = escaped["canonical_sha256"]
    sample["canonical_sha256"] = None
    write_json_atomic(sample_path, seal_manifest(sample))
    with pytest.raises((TurkishCorpusError, ValueError), match="path|relative|escape"):
        _validation_rows(sample_dir)


def _quality_metrics(*, roundtrip_failures: int = 0, utilization: float = 0.9) -> dict:
    metrics = {
        "roundtrip": {"failures": roundtrip_failures},
        "vocabulary_utilization": {"fraction": utilization},
        "token_byte_use": {
            "single_byte_token_use_fraction": 0.2,
            "non_ascii_token_use_fraction": 0.1,
            "invalid_utf8_token_use_fraction": 0.01,
        },
        "efficiency": {"bytes_per_token": 4.0, "characters_per_token": 3.0},
        "fertility": {
            "contextual_tokens_per_word": 1.5,
            "isolated_tokens_per_word": 1.2,
        },
    }
    return {
        "overall": copy.deepcopy(metrics),
        "strata": [
            {
                "mixture_id": "mix",
                "source_id": "source",
                "register_bucket": "register",
                "metrics": copy.deepcopy(metrics),
            }
        ],
        "fixed_turkish_probes": {
            "suite": "turkish_apostrophe_casing_long_suffix_v1",
            "suite_sha256": (
                "60c963ff62721a24f1a88064f2354a47420e87385c61d81fb8e4869e48d70593"
            ),
            "passed": True,
            "cases": [
                {
                    "id": probe_id,
                    "text": text,
                    "text_sha256": hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                    "roundtrip": True,
                    "bytes_per_token": 4.0,
                    "contextual_tokens_per_word": 1.5,
                    "isolated_tokens_per_word": 1.2,
                }
                for probe_id, text in _FIXED_TURKISH_PROBES
            ],
        },
    }


def _gate_policy() -> dict:
    return copy.deepcopy(TOKENIZER_QUALITY_GATE_V1)


def test_quality_gate_is_fail_closed_for_roundtrip_and_vocab_use() -> None:
    baseline = _quality_metrics()
    passing = evaluate_tokenizer_quality_gate(_quality_metrics(), baseline, _gate_policy())
    assert passing["passed"] is True
    failing = evaluate_tokenizer_quality_gate(
        _quality_metrics(roundtrip_failures=1, utilization=0.1),
        baseline,
        _gate_policy(),
    )
    assert failing["passed"] is False
    assert "current_roundtrip_zero" in failing["failures"]
    assert "vocabulary_utilization" in failing["failures"]


def test_quality_gate_rejects_turkish_probe_regression() -> None:
    baseline = _quality_metrics()
    current = _quality_metrics()
    current["fixed_turkish_probes"]["cases"][0]["bytes_per_token"] = 1.0
    current["fixed_turkish_probes"]["cases"][0][
        "contextual_tokens_per_word"
    ] = 3.0
    result = evaluate_tokenizer_quality_gate(current, baseline, _gate_policy())
    assert result["passed"] is False
    assert "probe_efficiency:ascii_apostrophe" in result["failures"]
    assert (
        "probe_fertility_contextual_tokens_per_word:ascii_apostrophe"
        in result["failures"]
    )


@pytest.mark.parametrize("mutation", ("duplicate", "hash", "text", "roundtrip", "suite"))
def test_quality_gate_requires_exact_frozen_probe_evidence(mutation: str) -> None:
    baseline = _quality_metrics()
    current = _quality_metrics()
    cases = current["fixed_turkish_probes"]["cases"]
    if mutation == "duplicate":
        cases[-1] = copy.deepcopy(cases[0])
    elif mutation == "hash":
        cases[0]["text_sha256"] = "0" * 64
    elif mutation == "text":
        cases[0]["text"] = "başka metin"
    elif mutation == "roundtrip":
        cases[0]["roundtrip"] = False
    else:
        current["fixed_turkish_probes"]["suite_sha256"] = "0" * 64
    result = evaluate_tokenizer_quality_gate(current, baseline, _gate_policy())
    assert result["passed"] is False
    expected_failure = (
        "fixed_turkish_probes"
        if mutation == "suite"
        else "fixed_turkish_probe_case_identity"
    )
    assert expected_failure in result["failures"]


def _mock_verified_tokenizer(
    monkeypatch: pytest.MonkeyPatch, *, training_receipt_sha256: str = "8" * 64
) -> None:
    monkeypatch.setattr(
        "nanochat.tokenizer_quality.verify_tokenizer_package",
        lambda *_args, **_kwargs: SimpleNamespace(
            manifest={"training_receipt_sha256": training_receipt_sha256}
        ),
    )


def test_manual_acceptance_rejects_missing_or_failed_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_verified_tokenizer(monkeypatch)
    metrics = _quality_metrics()
    policy = _gate_policy()
    automatic = evaluate_tokenizer_quality_gate(metrics, metrics, policy)
    report = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "turkish_tokenizer_heldout_quality",
            "tokenizer_package_sha256": "1" * 64,
            "training_receipt_sha256": "8" * 64,
            "heldout_validation": {"sha256": "2" * 64},
            "policy_sha256": "3" * 64,
            "production_chain": {"fixture": "4" * 64},
            "parent_corpus_manifest_sha256": "5" * 64,
            "qa_approval_sha256": "6" * 64,
            "metrics": metrics,
            "quality_gate_policy": policy,
            "baseline_comparison": {
                "available": False,
                "required": True,
                "metrics": metrics,
                "comparison_gate_passed": automatic["passed"],
            },
            "automatic_gate": automatic,
            "automated_gate_passed": automatic["passed"],
            "canonical_sha256": None,
        }
    )
    write_json_atomic(tmp_path / "quality_report.json", report)
    with pytest.raises(TurkishCorpusError, match="missing baseline"):
        seal_tokenizer_quality_approval(
            tmp_path,
            tokenizer_dir=tmp_path / "tokenizer",
            reviewer="reviewer",
            reviewed_at_utc="2026-08-20T12:00:00Z",
            decision="accepted",
        )
    rejected = seal_tokenizer_quality_approval(
        tmp_path,
        tokenizer_dir=tmp_path / "tokenizer",
        reviewer="reviewer",
        reviewed_at_utc="2026-08-20T12:00:00Z",
        decision="rejected",
    )
    assert rejected["decision"] == "rejected"


def test_manual_acceptance_recomputes_gate_from_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_verified_tokenizer(monkeypatch)
    metrics = _quality_metrics()
    policy = _gate_policy()
    automatic = evaluate_tokenizer_quality_gate(metrics, metrics, policy)
    forged_metrics = copy.deepcopy(metrics)
    forged_metrics["overall"]["vocabulary_utilization"]["fraction"] = 0.01
    report = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "turkish_tokenizer_heldout_quality",
            "tokenizer_package_sha256": "1" * 64,
            "training_receipt_sha256": "8" * 64,
            "heldout_validation": {"sha256": "2" * 64},
            "policy_sha256": "3" * 64,
            "production_chain": {"fixture": "4" * 64},
            "parent_corpus_manifest_sha256": "5" * 64,
            "qa_approval_sha256": "6" * 64,
            "metrics": forged_metrics,
            "quality_gate_policy": policy,
            "baseline_comparison": {
                "available": True,
                "required": True,
                "identity": {"sha256": "7" * 64},
                "metrics": metrics,
                "comparison_gate_passed": True,
            },
            # These stale pass flags are deliberately retained after degrading metrics.
            "automatic_gate": automatic,
            "automated_gate_passed": True,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(tmp_path / "quality_report.json", report)
    with pytest.raises(TurkishCorpusError, match="does not recompute"):
        seal_tokenizer_quality_approval(
            tmp_path,
            tokenizer_dir=tmp_path / "tokenizer",
            reviewer="reviewer",
            reviewed_at_utc="2026-08-20T12:00:00Z",
            decision="accepted",
        )


def _write_passing_quality_report(
    root: Path, *, policy: dict | None = None
) -> dict:
    metrics = _quality_metrics()
    gate_policy = copy.deepcopy(policy or _gate_policy())
    automatic = evaluate_tokenizer_quality_gate(metrics, metrics, gate_policy)
    report = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "turkish_tokenizer_heldout_quality",
            "tokenizer_package_sha256": "1" * 64,
            "training_receipt_sha256": "8" * 64,
            "heldout_validation": {"sha256": "2" * 64},
            "policy_sha256": "3" * 64,
            "production_chain": {"fixture": "4" * 64},
            "parent_corpus_manifest_sha256": "5" * 64,
            "qa_approval_sha256": "6" * 64,
            "metrics": metrics,
            "quality_gate_policy": gate_policy,
            "baseline_comparison": {
                "available": True,
                "required": True,
                "identity": {"sha256": "7" * 64},
                "metrics": metrics,
                "comparison_gate_passed": automatic["passed"],
            },
            "automatic_gate": automatic,
            "automated_gate_passed": automatic["passed"],
            "manual_acceptance_required": True,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(root / "quality_report.json", report)
    return report


def test_quality_approval_requires_exact_frozen_gate_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_verified_tokenizer(monkeypatch)
    policy = _gate_policy()
    policy["min_vocab_utilization_fraction"] = 0.70
    _write_passing_quality_report(tmp_path, policy=policy)
    with pytest.raises(TurkishCorpusError, match="not frozen"):
        seal_tokenizer_quality_approval(
            tmp_path,
            tokenizer_dir=tmp_path / "tokenizer",
            reviewer="reviewer",
            reviewed_at_utc="2026-08-20T12:00:00Z",
            decision="accepted",
        )


def test_quality_approval_and_downstream_bind_verified_training_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mismatch = tmp_path / "mismatch"
    mismatch.mkdir()
    _write_passing_quality_report(mismatch)
    _mock_verified_tokenizer(monkeypatch, training_receipt_sha256="9" * 64)
    with pytest.raises(TurkishCorpusError, match="training receipt binding drift"):
        seal_tokenizer_quality_approval(
            mismatch,
            tokenizer_dir=tmp_path / "tokenizer",
            reviewer="reviewer",
            reviewed_at_utc="2026-08-20T12:00:00Z",
            decision="accepted",
        )
    _mock_verified_tokenizer(monkeypatch)
    _write_passing_quality_report(tmp_path)
    approval = seal_tokenizer_quality_approval(
        tmp_path,
        tokenizer_dir=tmp_path / "tokenizer",
        reviewer="reviewer",
        reviewed_at_utc="2026-08-20T12:00:00Z",
        decision="accepted",
    )
    assert approval["training_receipt_sha256"] == "8" * 64
    validate_tokenizer_quality_gate(
        tmp_path,
        expected_package_sha256="1" * 64,
        expected_training_receipt_sha256="8" * 64,
        expected_production_chain={"fixture": "4" * 64},
    )
    with pytest.raises(TurkishCorpusError, match="absent, failed, or stale"):
        validate_tokenizer_quality_gate(
            tmp_path,
            expected_package_sha256="1" * 64,
            expected_training_receipt_sha256="9" * 64,
            expected_production_chain={"fixture": "4" * 64},
        )


def test_trainer_recomputes_realized_sample_distribution() -> None:
    stats = {
        "documents": 2,
        "characters": 11,
        "mixture_documents": Counter({"mix": 2}),
        "mixture_characters": Counter({"mix": 11}),
        "source_documents": Counter({"source": 2}),
        "source_characters": Counter({"source": 11}),
        "register_documents": Counter({"Forum": 1, "NA": 1}),
        "register_characters": Counter({"Forum": 6, "NA": 5}),
    }
    realized = trainer._realized_sample_distribution(stats)
    assert trainer._validate_realized_sample_distribution(stats, realized) == realized
    tampered = copy.deepcopy(realized)
    tampered["by_source"][0]["characters"] = 10
    with pytest.raises(ValueError, match="trainer-visible sample distribution"):
        trainer._validate_realized_sample_distribution(stats, tampered)


def _write_baseline_fixture(
    root: Path,
    *,
    split_pattern: str = SPLIT_PATTERN,
    swap_special_ids: bool = False,
) -> tuple[dict, RustBPETokenizer]:
    texts = [
        "Merhaba dünya, İstanbul'da bugün nasılsın?",
        "TÜRKİYE Türkiye türkiye IĞDIR Iğdır ığdır",
        "sorumluluklarımızdakilerdenmişsinizcesine",
    ] * 50
    trained = RustBPETokenizer.train_from_iterator(iter(texts), 300)
    lexical = 300 - len(SPECIAL_TOKENS)
    specials = {
        token: lexical + index for index, token in enumerate(SPECIAL_TOKENS)
    }
    if swap_special_ids:
        specials[SPECIAL_TOKENS[0]], specials[SPECIAL_TOKENS[1]] = (
            specials[SPECIAL_TOKENS[1]],
            specials[SPECIAL_TOKENS[0]],
        )
    encoding = tiktoken.Encoding(
        name="fixture_baseline",
        pat_str=split_pattern,
        mergeable_ranks=dict(trained.enc._mergeable_ranks),
        special_tokens=specials,
    )
    tokenizer = RustBPETokenizer(encoding, "<|bos|>")
    root.mkdir()
    tokenizer.save(str(root))
    write_json_atomic(
        root / "tokenizer_config.json",
        {"name": "fixture_baseline", "vocab_size": 300},
    )
    legacy_lengths = torch.tensor(
        [
            len(tokenizer.decode([token_id]).encode("utf-8"))
            if token_id < lexical
            else 0
            for token_id in range(300)
        ],
        dtype=torch.int32,
    )
    raw_lengths = torch.tensor(
        [
            len(tokenizer.decode_single_token_bytes(token_id))
            if token_id < lexical
            else 0
            for token_id in range(300)
        ],
        dtype=torch.int32,
    )
    raw_mismatch_count = int((legacy_lengths != raw_lengths).sum().item())
    torch.save(legacy_lengths, root / "token_bytes.pt")
    files = [
        {
            "path": name,
            "size_bytes": (root / name).stat().st_size,
            "sha256": file_sha256(root / name),
        }
        for name in ("token_bytes.pt", "tokenizer.pkl", "tokenizer_config.json")
    ]
    contract = {
        "name": "fixture_baseline",
        "vocab_size": 300,
        "split_pattern": "nanochat_gpt4_style_numbers_1_or_2_v1",
        "special_token_policy": "nanochat_9_v1",
        "token_byte_table_semantics": "legacy_decoded_utf8_replacement_v1",
        "raw_byte_length_mismatch_count": raw_mismatch_count,
        "files": files,
        "payload_inventory_sha256": hashlib.sha256(
            canonical_json(files).encode("utf-8")
        ).hexdigest(),
    }
    return contract, tokenizer


def test_pinned_baseline_is_fail_closed_and_semantically_comparable(
    tmp_path: Path,
) -> None:
    valid_root = tmp_path / "valid"
    contract, _tokenizer = _write_baseline_fixture(valid_root)
    _loaded, identity = validate_pinned_baseline_tokenizer(valid_root, contract)
    assert identity["sha256"] == contract["payload_inventory_sha256"]
    assert identity["token_byte_table_semantics"] == (
        "legacy_decoded_utf8_replacement_v1"
    )
    assert identity["raw_byte_length_mismatch_count"] == contract[
        "raw_byte_length_mismatch_count"
    ] > 0
    assert identity["raw_byte_length_semantics_compatible"] is False
    assert identity["raw_byte_length_mismatch_examples"]

    wrong_inventory = copy.deepcopy(contract)
    wrong_inventory["payload_inventory_sha256"] = "0" * 64
    with pytest.raises(TurkishCorpusError, match="inventory hash"):
        validate_pinned_baseline_tokenizer(valid_root, wrong_inventory)

    write_json_atomic(valid_root / "package_manifest.json", {})
    with pytest.raises(TurkishCorpusError, match="strict verification failed"):
        validate_pinned_baseline_tokenizer(valid_root, contract)

    wrong_pattern_root = tmp_path / "wrong-pattern"
    wrong_pattern, _tokenizer = _write_baseline_fixture(
        wrong_pattern_root, split_pattern=r"(?s).+"
    )
    with pytest.raises(TurkishCorpusError, match="not comparable"):
        validate_pinned_baseline_tokenizer(wrong_pattern_root, wrong_pattern)

    wrong_special_root = tmp_path / "wrong-special"
    wrong_special, _tokenizer = _write_baseline_fixture(
        wrong_special_root, swap_special_ids=True
    )
    with pytest.raises(TurkishCorpusError, match="not comparable"):
        validate_pinned_baseline_tokenizer(wrong_special_root, wrong_special)


def test_canonical_tiktoken_export_reconstructs_rank_ids_and_token_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    texts = [
        "Merhaba dünya, İstanbul'da bugün nasılsın?",
        "TÜRKİYE Türkiye türkiye IĞDIR Iğdır ığdır",
        "sorumluluklarımızdakilerdenmişsinizcesine",
    ] * 50
    tokenizer = RustBPETokenizer.train_from_iterator(iter(texts), 300)
    monkeypatch.setattr(trainer, "VOCAB_SIZE", 300)
    lengths = trainer._token_byte_lengths(tokenizer)
    metadata = trainer._export_and_verify_canonical_tiktoken(
        tokenizer, tmp_path, lengths
    )
    assert metadata["dense_rank_id_identity_verified"] is True
    assert metadata["probe_id_sequences_verified"] == 4
    assert len((tmp_path / "tokenizer.tiktoken").read_text().splitlines()) == 291
