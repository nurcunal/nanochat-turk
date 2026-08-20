from __future__ import annotations

import copy
from pathlib import Path

import pytest

from nanochat.experiment_manifest import seal_manifest, write_json_atomic
from nanochat.turkish_corpus import TurkishCorpusError
from scripts import turkish_packed_production as production


def _sealed(**values):
    return seal_manifest({**values, "canonical_sha256": None})


def _fixture(tmp_path: Path):
    hashes = {name: name[0] * 64 for name in ("recipe", "policy", "source", "calibration")}
    hashes.update({"pack": "e" * 64, "resource": "f" * 64, "mixture": "a" * 64, "gate": "b" * 64})
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
        "storage_gate": {
            "work_dir": str(tmp_path),
            "work_dir_filesystem_device": tmp_path.stat().st_dev,
        },
    }
    launch = _sealed(
        schema_version="1.0",
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
