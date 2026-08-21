from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from nanochat import packing_capacity
from nanochat.experiment_manifest import file_sha256, seal_manifest, write_json_atomic
from nanochat.strict_runtime import (
    FAMILY_ID,
    FAMILY_ID_V2,
    FAMILY_ID_V3,
    FAMILY_ID_V4,
    family_artifact_contract,
    validate_family_recipe,
)
from nanochat.turkish_corpus import load_corpus_policy
from scripts import d32_family_workflow as workflow
from scripts import build_turkish_pretrain_corpus as corpus_cli
from scripts import train_turkish_raw_bpe as tokenizer_cli
from scripts import turkish_data_backend as backend_cli
from scripts import upload_base_checkpoint_to_hf as uploader


ROOT = Path(__file__).resolve().parents[1]
RECIPE_V3 = ROOT / "configs" / "pretrain" / "tr_d32_turkish_general_wsd_v3.json"
RECIPE_V4 = ROOT / "configs" / "pretrain" / "tr_d32_turkish_general_wsd_v4.json"
POLICY_V4 = ROOT / "configs" / "pretrain" / "tr_d32_turkish_general_v4.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace_version(value, old: str, new: str):
    if isinstance(value, dict):
        return {key: _replace_version(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_version(item, old, new) for item in value]
    if isinstance(value, str):
        return value.replace(old, new)
    return value


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


def test_v4_recipe_is_sealed_supported_and_only_reidentifies_v3_recipe() -> None:
    recipe_v4, digest = workflow.load_recipe(RECIPE_V4)
    validate_family_recipe(recipe_v4)
    assert recipe_v4["family_id"] == FAMILY_ID_V4
    assert digest == recipe_v4["canonical_sha256"]

    normalized = _replace_version(copy.deepcopy(recipe_v4), "_v4", "_v3")
    recipe_v3 = _load(RECIPE_V3)
    normalized["canonical_sha256"] = recipe_v3["canonical_sha256"]
    assert normalized == recipe_v3


def test_v4_artifact_namespace_is_v3_equivalent_with_fresh_identities() -> None:
    contract_v3 = family_artifact_contract(FAMILY_ID_V3)
    contract_v4 = family_artifact_contract(FAMILY_ID_V4)
    assert _replace_version(contract_v4, "_v4", "_v3") == contract_v3
    assert contract_v4["mixture_config"].endswith("tr_d32_turkish_general_v4.json")
    assert contract_v4["corpus_id"] == "tr_general_clean_v4"
    assert contract_v4["corpus_root"] == "pretrain_data/tr_general_clean_v4"
    assert contract_v4["tokenizer_name"] == "tr_general_raw_bpe_32k_v4"
    assert contract_v4["tokenizer_root"] == "tokenizers/tr_general_raw_bpe_32k_v4"


def test_v4_defaults_and_policy_identity_are_consistent() -> None:
    recipe = _load(RECIPE_V4)
    policy = load_corpus_policy(POLICY_V4)
    assert workflow.DEFAULT_RECIPE == Path(
        "configs/pretrain/tr_d32_turkish_general_wsd_v4.json"
    )
    assert workflow.DEFAULT_POLICY == Path(
        "configs/pretrain/tr_d32_turkish_general_v4.json"
    )
    assert corpus_cli.DEFAULT_POLICY == workflow.DEFAULT_POLICY
    assert backend_cli.DEFAULT_POLICY == workflow.DEFAULT_POLICY
    assert tokenizer_cli.DEFAULT_POLICY == workflow.DEFAULT_POLICY
    assert tokenizer_cli.build_parser().get_default("policy") == workflow.DEFAULT_POLICY
    workflow._validate_recipe_policy_identity(recipe, policy)


def test_v4_requires_full_finals_and_manual_publication_approval() -> None:
    recipe = _load(RECIPE_V4)
    assert {item["retention"] for item in recipe["checkpoints"]["finals"]} == {
        "full_resumable"
    }
    assert recipe["storage"]["full_cooled_final_transactions_at_peak"] == 3
    assert recipe["publication"]["require_manual_final_quality_approval"] is True

    drifted = copy.deepcopy(recipe)
    drifted["checkpoints"]["finals"][0]["retention"] = "metadata_only"
    with pytest.raises(workflow.FamilyWorkflowError, match=r"finals\[0\] drifted"):
        workflow.validate_recipe(drifted)

    drifted = copy.deepcopy(recipe)
    drifted["publication"]["require_manual_final_quality_approval"] = False
    with pytest.raises(workflow.FamilyWorkflowError, match="manual final-model"):
        workflow.validate_recipe(drifted)


def test_v4_capacity_verifier_uses_repeat_capacity_gate(
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
        family_id=FAMILY_ID_V4,
    )
    assert observed == receipt
    assert digest == receipt["canonical_sha256"]
    assert calls[0][1] == {
        "dataset_manifest_sha256": "a" * 64,
        "tokenizer_package_sha256": "b" * 64,
        "manual_repetition_risk_approval": None,
    }


def test_uploader_retains_v3_and_adds_v4_family_inventory() -> None:
    assert uploader._family_uses_full_final_transactions(FAMILY_ID_V3)
    assert uploader._family_uses_full_final_transactions(FAMILY_ID_V4)
    assert uploader._family_paths_inventory(FAMILY_ID_V3) == (
        "runs/uhem_d32_v3_paths.sh"
    )
    assert uploader._family_paths_inventory(FAMILY_ID_V4) == (
        "runs/uhem_d32_v4_paths.sh"
    )
    assert uploader._family_paths_inventory(FAMILY_ID) is None
    assert uploader._family_paths_inventory(FAMILY_ID_V2) is None
    with pytest.raises(ValueError, match="unsupported family"):
        uploader._family_paths_inventory("tr_d32_general_bpe32k_unknown")
