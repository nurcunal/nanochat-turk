from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from nanochat.strict_runtime import (
    FAMILY_ID_V3,
    StrictTrainingError,
    _validate_anchor_preflight_binding,
    capacity_authorized_positions,
    capacity_world_gate_record,
    family_artifact_contract,
    validate_family_recipe,
)
from nanochat.experiment_manifest import file_sha256, seal_manifest, write_json_atomic
from nanochat import packing_capacity, turkish_anchor_preparation
from scripts import d32_family_workflow as workflow
from scripts.turkish_data_backend import _prepared_source_manifests


ROOT = Path(__file__).resolve().parents[1]
RECIPE_V3 = ROOT / "configs/pretrain/tr_d32_turkish_general_wsd_v3.json"
POLICY_V2 = ROOT / "configs/pretrain/tr_d32_turkish_general_v2.json"
POLICY_V3 = ROOT / "configs/pretrain/tr_d32_turkish_general_v3.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _compact_repeat_capacity_receipt() -> dict:
    worlds = {}
    for world_size, accumulation in ((8, 32), (16, 16)):
        worlds[str(world_size)] = {
            "world_size": world_size,
            "device_batch_sequences": 4,
            "max_seq_len": 2048,
            "gradient_accumulation_steps": accumulation,
            "repetition_tier": "preferred",
            "whole_pool_repetition_only": True,
            "source_specific_repetition": False,
            "horizons": {
                "s40_margin": {
                    "scheduled_token_positions": 32_640 * 2_097_152,
                    "max_loaded_epoch": 2,
                    "max_consumed_epoch": 2,
                    "epoch5_loaded_including_prefetch": False,
                }
            },
        }
    return seal_manifest(
        {
            "kind": "turkish_bestfit_repeat_capacity_receipt",
            "simulation": {
                "implementation_file_sha256": file_sha256(
                    ROOT / "nanochat" / "packing_capacity.py"
                ),
                "worlds": worlds,
            },
            "canonical_sha256": None,
        }
    )


def test_v3_recipe_is_an_exact_supported_runtime_family() -> None:
    recipe = _load(RECIPE_V3)
    workflow.validate_recipe(recipe)
    validate_family_recipe(recipe)
    assert recipe["family_id"] == FAMILY_ID_V3


def test_v3_artifact_namespace_includes_all_prepared_sources() -> None:
    contract = family_artifact_contract(FAMILY_ID_V3)
    assert contract["corpus_id"] == "tr_general_clean_v3"
    assert contract["tokenizer_name"] == "tr_general_raw_bpe_32k_v3"
    assert contract["macocu_preparation_manifest"] == (
        "source_data/macocu_genre_tr_v1/manifest.json"
    )
    assert contract["mot_preparation_manifest"] == (
        "source_data/mot_tr_v1_11/manifest.json"
    )
    assert contract["parlamint_preparation_manifest"] == (
        "source_data/parlamint_tr_v5_0/manifest.json"
    )


def test_v3_cannot_reuse_v2_policy_identity() -> None:
    recipe = _load(RECIPE_V3)
    v2_policy = _load(POLICY_V2)
    with pytest.raises(workflow.FamilyWorkflowError, match="policy name"):
        workflow._validate_recipe_policy_identity(recipe, v2_policy)

    v3_policy = _load(POLICY_V3)
    workflow._validate_recipe_policy_identity(recipe, v3_policy)
    drifted = copy.deepcopy(v3_policy)
    drifted["tokenizer_training"]["name"] = "tr_general_raw_bpe_32k_v2"
    with pytest.raises(workflow.FamilyWorkflowError, match="tokenizer name"):
        workflow._validate_recipe_policy_identity(recipe, drifted)


def test_prepared_source_cli_requires_exact_unique_v3_source_ids() -> None:
    parsed = _prepared_source_manifests(
        ["mot_tr_v1_11=/data/mot", "parlamint_tr_v5_0=/data/parla/manifest.json"]
    )
    assert parsed == {
        "mot_tr_v1_11": Path("/data/mot"),
        "parlamint_tr_v5_0": Path("/data/parla/manifest.json"),
    }
    with pytest.raises(ValueError, match="duplicate"):
        _prepared_source_manifests(
            ["mot_tr_v1_11=/data/a", "mot_tr_v1_11=/data/b"]
        )
    with pytest.raises(ValueError, match="must be"):
        _prepared_source_manifests(["unknown=/data/source"])


def test_current_cli_defaults_select_v3_without_removing_explicit_v2() -> None:
    assert workflow.DEFAULT_RECIPE == Path(
        "configs/pretrain/tr_d32_turkish_general_wsd_v3.json"
    )
    assert workflow.DEFAULT_POLICY == Path(
        "configs/pretrain/tr_d32_turkish_general_v3.json"
    )
    v2_recipe = _load(
        ROOT / "configs/pretrain/tr_d32_turkish_general_wsd_v2.json"
    )
    workflow.validate_recipe(v2_recipe)
    validate_family_recipe(v2_recipe)


def test_runtime_anchor_binding_requires_exact_manifest_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    acceptance = seal_manifest({"canonical_sha256": None})
    acquisition = seal_manifest({"canonical_sha256": None})
    manifest = seal_manifest(
        {
            "source_id": "mot_tr_v1_11",
            "preparer_version": "turkish_native_text_anchors_v1",
            "production_acceptance": {
                "stage": "accepted_production",
                "receipt": acceptance,
            },
            "acquisition_receipt": acquisition,
            "artifacts": {
                "data": {
                    "logical_jsonl_sha256": "a" * 64,
                    "totals": {"rows": 10},
                }
            },
            "clean": {"documents": 10},
            "canonical_sha256": None,
        }
    )
    manifest_path = tmp_path / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    monkeypatch.setattr(
        turkish_anchor_preparation,
        "validate_anchor_preparation",
        lambda *_args, **_kwargs: manifest,
    )
    derived = {
        "mot_tr_v1_11": {
            "manifest_uri": manifest_path.resolve().as_uri(),
            "manifest_sha256": manifest["canonical_sha256"],
            "source_id": "mot_tr_v1_11",
            "preparer_version": "turkish_native_text_anchors_v1",
            "production_acceptance": {
                "stage": "accepted_production",
                "receipt_sha256": acceptance["canonical_sha256"],
            },
            "acquisition_receipt_sha256": acquisition["canonical_sha256"],
            "clean": {"documents": 10},
            "data_artifact": {
                "logical_jsonl_sha256": "a" * 64,
                "totals": {"rows": 10},
            },
            "downstream_admission": {
                "preparer_automatically_admits_training": False,
                "backend_turkish_no_code_audit_required": True,
            },
        }
    }
    record = {
        "path": str(manifest_path.resolve()),
        "sha256": manifest["canonical_sha256"],
    }
    _validate_anchor_preflight_binding(
        manifest_path,
        source_id="mot_tr_v1_11",
        derived_sources=derived,
        preflight_record=record,
    )
    drifted = copy.deepcopy(derived)
    drifted["mot_tr_v1_11"]["downstream_admission"] = {}
    with pytest.raises(StrictTrainingError, match="binding drifted"):
        _validate_anchor_preflight_binding(
            manifest_path,
            source_id="mot_tr_v1_11",
            derived_sources=drifted,
            preflight_record=record,
        )


def test_v3_capacity_record_is_compact_and_preflight_safe() -> None:
    receipt = _compact_repeat_capacity_receipt()
    selected = capacity_world_gate_record(receipt, 8)
    assert selected["capacity_mode"] == "whole_pool_repeat_v3"
    assert selected["repetition_tier"] == "preferred"
    assert "rank_evidence" not in selected
    assert capacity_authorized_positions(selected) == 32_640 * 2_097_152

    preflight = {
        "corpus": {
            "packing_capacity_receipt": {
                "sha256": receipt["canonical_sha256"],
                "gate_passed": True,
                "worlds": {"8": selected},
            }
        }
    }
    digest, observed = workflow._preflight_capacity_world(preflight, 8)
    assert digest == receipt["canonical_sha256"]
    assert observed == selected


def test_workflow_v3_capacity_verifier_uses_hardened_public_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _compact_repeat_capacity_receipt()
    receipt_path = tmp_path / "capacity.json"
    write_json_atomic(receipt_path, receipt)
    calls = []

    def validate(candidate, **kwargs):
        calls.append((candidate, kwargs))
        return {
            "canonical_sha256": receipt["canonical_sha256"],
            "repetition_tier": "preferred",
            "gate_passed": True,
            "cleanup_authorized": True,
            "approval_required": False,
            "approval_satisfied": True,
        }

    monkeypatch.setattr(
        packing_capacity, "validate_repetition_capacity_receipt", validate
    )
    observed, digest = workflow._verify_packing_capacity_receipt(
        receipt_path,
        dataset_sha256="a" * 64,
        tokenizer_sha256="b" * 64,
        implementation_path=ROOT / "nanochat" / "packing_capacity.py",
        family_id=FAMILY_ID_V3,
    )
    assert observed == receipt
    assert digest == receipt["canonical_sha256"]
    assert calls[0][1] == {
        "dataset_manifest_sha256": "a" * 64,
        "tokenizer_package_sha256": "b" * 64,
        "manual_repetition_risk_approval": None,
    }

    monkeypatch.setattr(
        packing_capacity,
        "validate_repetition_capacity_receipt",
        lambda *_args, **_kwargs: {
            "canonical_sha256": receipt["canonical_sha256"],
            "repetition_tier": "manual_risk",
            "gate_passed": True,
            "cleanup_authorized": True,
            "approval_required": True,
            "approval_satisfied": True,
        },
    )
    with pytest.raises(workflow.FamilyWorkflowError, match="preferred tier"):
        workflow._verify_packing_capacity_receipt(
            receipt_path,
            dataset_sha256="a" * 64,
            tokenizer_sha256="b" * 64,
            implementation_path=ROOT / "nanochat" / "packing_capacity.py",
            family_id=FAMILY_ID_V3,
        )
