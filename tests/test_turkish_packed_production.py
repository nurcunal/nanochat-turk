from __future__ import annotations

import copy
from pathlib import Path

import pytest

from nanochat.experiment_manifest import seal_manifest, write_json_atomic
from nanochat.turkish_corpus import TurkishCorpusError
from scripts import turkish_packed_production as production


STRICT_CONTENT_POLICY = dict(production.PRODUCTION_TEXT_INTEGRITY_LIMITS)


def _sealed(**values):
    return seal_manifest({**values, "canonical_sha256": None})


def _fixture(tmp_path: Path):
    hashes = {name: name[0] * 64 for name in ("recipe", "policy", "source", "calibration")}
    hashes.update(
        {
            "pack": "e" * 64,
            "resource": "f" * 64,
            "mixture": "a" * 64,
            "gate": "b" * 64,
            "sample_cluster": "d" * 64,
        }
    )
    inputs = {
        "recipe": {"family_id": "tr_d32_fixture"},
        "recipe_sha": hashes["recipe"],
        "policy_sha": hashes["policy"],
        "source_plan_sha": hashes["source"],
        "calibration_sha": hashes["calibration"],
        "pack_plan_sha": hashes["pack"],
        "resource_approval_sha": hashes["resource"],
        "mixture_approval_sha": hashes["mixture"],
        "storage_gate_sha": hashes["gate"],
        "sample_cluster_receipt_sha": hashes["sample_cluster"],
        "storage_gate": {
            "work_dir": str(tmp_path),
            "work_dir_filesystem_device": tmp_path.stat().st_dev,
        },
    }
    launch = _sealed(
        schema_version="2.0",
        kind=production.CLUSTER_LAUNCH_KIND,
        **production._receipt_bindings(inputs),
        cluster_completed=True,
    )
    launch_path = tmp_path / "cluster_launch.json"
    write_json_atomic(launch_path, launch)
    chain = {
        "cluster_launch_receipt_sha256": launch["canonical_sha256"],
        "production_pack_plan_sha256": hashes["pack"],
        "resource_approval_sha256": hashes["resource"],
        "mixture_quality_approval_sha256": hashes["mixture"],
        "data_prep_storage_gate_sha256": hashes["gate"],
        "sample_cluster_receipt_sha256": hashes["sample_cluster"],
    }
    pool = tmp_path / "pool"
    (pool / "qa").mkdir(parents=True)
    pool_manifest = _sealed(
        schema_version="1.0",
        kind="turkish_pretrain_corpus",
        stage="filtered_pool",
        policy_sha256=hashes["policy"],
        production_chain=chain,
    )
    qa = _sealed(
        schema_version="1.0",
        kind="turkish_pretrain_qa_approval",
        decision="accepted",
        pool_manifest_sha256=pool_manifest["canonical_sha256"],
    )
    write_json_atomic(pool / "corpus_manifest.json", pool_manifest)
    write_json_atomic(pool / "qa" / "qa_approval.json", qa)
    sample = tmp_path / "sample"
    sample.mkdir()
    sample_manifest = _sealed(
        schema_version="1.0",
        kind="turkish_raw_bpe_training_sample",
        policy_sha256=hashes["policy"],
        production_chain=chain,
        parent_corpus_manifest_sha256=pool_manifest["canonical_sha256"],
        qa_approval_sha256=qa["canonical_sha256"],
    )
    write_json_atomic(sample / "tokenizer_sample_manifest.json", sample_manifest)
    return inputs, launch_path, pool, sample


def test_downstream_lineage_accepts_exact_pool_and_sample(tmp_path: Path) -> None:
    inputs, launch, pool, sample = _fixture(tmp_path)
    result = production.validate_downstream_lineage(
        inputs,
        launch,
        pool_dir=pool,
        tokenizer_sample_dir=sample,
    )
    assert result["production_chain"]["cluster_launch_receipt_sha256"]
    assert result["checked"]["pool_qa_approval_sha256"]


def test_downstream_lineage_rejects_resealed_wrong_cluster_launch(tmp_path: Path) -> None:
    inputs, launch, pool, sample = _fixture(tmp_path)
    wrong_launch = tmp_path / "wrong_launch.json"
    payload = copy.deepcopy(production.load_json_strict(launch))
    payload["extra"] = "different canonical identity"
    write_json_atomic(wrong_launch, seal_manifest(payload))
    with pytest.raises(TurkishCorpusError, match="production lineage drift"):
        production.validate_downstream_lineage(
            inputs,
            wrong_launch,
            pool_dir=pool,
            tokenizer_sample_dir=sample,
        )


def test_downstream_lineage_rejects_resealed_sample_parent_tamper(tmp_path: Path) -> None:
    inputs, launch, pool, sample = _fixture(tmp_path)
    payload = production.load_json_strict(sample / "tokenizer_sample_manifest.json")
    payload["parent_corpus_manifest_sha256"] = "0" * 64
    write_json_atomic(
        sample / "tokenizer_sample_manifest.json", seal_manifest(payload)
    )
    with pytest.raises(TurkishCorpusError, match="parent-pool/QA"):
        production.validate_downstream_lineage(
            inputs,
            launch,
            pool_dir=pool,
            tokenizer_sample_dir=sample,
        )


def test_gated_write_dir_rejects_path_outside_tree(tmp_path: Path) -> None:
    inputs, _launch, _pool, _sample = _fixture(tmp_path)
    inside = tmp_path / "inside"
    inside.mkdir()
    assert production.validate_gated_write_dir(inputs, inside) == str(inside.resolve())
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    with pytest.raises(TurkishCorpusError, match="outside"):
        production.validate_gated_write_dir(inputs, outside)


@pytest.mark.parametrize("source_id", ["finepdfs_edu_tr", "fineweb2_tr"])
def test_production_source_gate_rejects_disqualified_sources_but_not_historical_inventory(
    source_id: str,
) -> None:
    policy = {
        "sources": [
            {"id": "finepdfs_edu_tr"},
            {"id": "fineweb2_tr"},
            {"id": "fineweb2_hq_tr", "text_origin": "born_digital_text"},
        ],
        "mixture": [
            {"id": "general", "source_id": source_id},
        ],
        "content_policy": STRICT_CONTENT_POLICY,
    }
    with pytest.raises(TurkishCorpusError, match="manually disqualified"):
        production._validate_production_source_eligibility(policy)

    # Merely preserving a disqualified source in an immutable audit policy is
    # allowed; the production gate rejects it only if the mixture selects it.
    policy["mixture"] = [
        {"id": "general", "source_id": "fineweb2_hq_tr"},
    ]
    production._validate_production_source_eligibility(policy)


@pytest.mark.parametrize(
    "text_origin",
    [
        None,
        "ocr",
        "pdf_extraction",
        "mixed",
        "unknown",
        "Born_Digital_Text",
        ["born_digital_text"],
        {"kind": "born_digital_text"},
    ],
)
def test_production_source_gate_rejects_non_native_or_undeclared_text_origin(
    text_origin: object,
) -> None:
    source = {"id": "candidate"}
    if text_origin is not None:
        source["text_origin"] = text_origin
    policy = {
        "sources": [source],
        "mixture": [{"id": "general", "source_id": "candidate"}],
        "content_policy": STRICT_CONTENT_POLICY,
    }
    with pytest.raises(TurkishCorpusError, match="PDF-extracted, OCR-derived"):
        production._validate_production_source_eligibility(policy)


@pytest.mark.parametrize("text_origin", ["born_digital_text", "structured_text"])
def test_production_source_gate_accepts_explicit_native_text_origin(
    text_origin: str,
) -> None:
    production._validate_production_source_eligibility(
        {
            "sources": [{"id": "candidate", "text_origin": text_origin}],
            "mixture": [{"id": "general", "source_id": "candidate"}],
            "content_policy": STRICT_CONTENT_POLICY,
        }
    )


def test_production_source_gate_rejects_selected_source_absent_from_inventory() -> None:
    with pytest.raises(TurkishCorpusError, match="absent from its inventory"):
        production._validate_production_source_eligibility(
            {
                "sources": [
                    {"id": "native", "text_origin": "born_digital_text"}
                ],
                "mixture": [{"id": "general", "source_id": "missing"}],
                "content_policy": STRICT_CONTENT_POLICY,
            }
        )


@pytest.mark.parametrize(
    "mixture",
    [None, {}, "candidate", [], [{}], [{"source_id": ""}], [{"source_id": []}]],
)
def test_production_source_gate_rejects_malformed_mixture(mixture: object) -> None:
    with pytest.raises(TurkishCorpusError, match="mixture"):
        production._validate_production_source_eligibility(
            {
                "sources": [
                    {"id": "candidate", "text_origin": "born_digital_text"}
                ],
                "mixture": mixture,
                "content_policy": STRICT_CONTENT_POLICY,
            }
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("max_unicode_replacement_characters", None),
        ("max_unicode_replacement_characters", 1),
        ("max_mojibake_sequence_hits", -1),
        ("max_c1_control_characters", False),
        ("max_unicode_surrogate_characters", "0"),
    ],
)
def test_production_source_gate_requires_zero_text_integrity_limits(
    key: str, value: object
) -> None:
    content_policy = dict(STRICT_CONTENT_POLICY)
    if value is None:
        content_policy.pop(key)
    else:
        content_policy[key] = value
    with pytest.raises(TurkishCorpusError, match=key):
        production._validate_production_source_eligibility(
            {
                "sources": [
                    {"id": "candidate", "text_origin": "born_digital_text"}
                ],
                "mixture": [{"id": "general", "source_id": "candidate"}],
                "content_policy": content_policy,
            }
        )
