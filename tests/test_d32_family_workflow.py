from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from nanochat.experiment_manifest import seal_manifest, write_json_atomic
from scripts import d32_family_workflow as workflow


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "configs" / "pretrain" / "tr_d32_turkish_general_wsd_v1.json"


def test_recipe_has_no_broad_core_patch_allowlist() -> None:
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    workflow.validate_recipe(recipe)
    code = recipe["code_provenance"]
    assert "core_patch_allowlist" not in code
    assert set(code["core_scope"]) == set(code["exact_file_sha256"])
    assert {
        "nanochat/checkpoint_manager.py",
        "nanochat/dataloader.py",
        "nanochat/dataset.py",
        "nanochat/loss_eval.py",
        "nanochat/tokenizer.py",
        "scripts/base_train.py",
    }.issubset(code["exact_file_sha256"])


@pytest.mark.parametrize(
    ("stage_index", "field", "value"),
    [
        (0, "kind", "cooldown_fork"),
        (1, "source_model_tag", "wrong_parent"),
        (2, "source_step", 8639),
        (3, "target_step", 16001),
        (4, "num_iterations", 32000),
        (5, "cooldown_start_step", 28799),
    ],
)
def test_recipe_rejects_stage_lineage_drift(
    stage_index: int, field: str, value: object
) -> None:
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    recipe["stages"][stage_index][field] = value
    with pytest.raises(workflow.FamilyWorkflowError, match="lineage"):
        workflow.validate_recipe(recipe)


@pytest.mark.parametrize(
    ("section", "index", "field", "value"),
    [
        ("stable_forks", 0, "retention", "model_only"),
        ("finals", 1, "model_tag", "wrong_model"),
    ],
)
def test_recipe_rejects_checkpoint_contract_drift(
    section: str, index: int, field: str, value: object
) -> None:
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    recipe["checkpoints"][section][index][field] = value
    with pytest.raises(workflow.FamilyWorkflowError, match="drifted"):
        workflow.validate_recipe(recipe)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("raw_estimated_aggregate_a100_hours_range", [1000, 1900]),
        ("raw_estimated_cpu_saat_range", [16000, 30400]),
        ("estimated_cpu_saat_range_with_15_percent_reserve", [18400, 34960]),
    ],
)
def test_recipe_rejects_coherently_lowered_cost_envelope(
    field: str, value: list[int]
) -> None:
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    recipe["uhem_budget"][field] = value
    with pytest.raises(workflow.FamilyWorkflowError, match="envelope"):
        workflow.validate_recipe(recipe)


def test_measured_smoke_cost_projection_is_exact_and_fail_closed() -> None:
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    projection = workflow._measured_cost_projection(
        recipe,
        selected_smoke_sha256="a" * 64,
        world_size=8,
        nodes=2,
        measured_positions_per_second=100_000.0,
    )
    assert projection["full_shared_updates"] == 34_560
    assert projection["full_scheduled_positions"] == 72_477_573_120
    assert projection["raw_training_cpu_saat_ceiling"] == 25_770
    assert projection["reserved_training_cpu_saat"] == 29_636
    assert projection["projected_total_package_cpu_saat"] == 33_636
    assert projection["passed"] is True

    too_slow = workflow._measured_cost_projection(
        recipe,
        selected_smoke_sha256="b" * 64,
        world_size=8,
        nodes=2,
        measured_positions_per_second=50_000.0,
    )
    assert too_slow["projected_total_package_cpu_saat"] > 40_000
    assert too_slow["passed"] is False


def test_uhem_quota_cross_checks_raw_usage_seconds(monkeypatch, tmp_path: Path) -> None:
    output = "nakane|nunal|cpu=16279700|cpu=124440|cpu=0|7466432\n"
    monkeypatch.setattr(workflow.subprocess, "check_output", lambda *a, **k: output)
    remaining, _digest, audit = workflow._live_uhem_cpu_saat(
        tmp_path, "nakane", "nunal"
    )
    assert remaining == pytest.approx((16_279_700 - 124_440) / 60)
    assert audit["raw_usage_tres_seconds"] == 7_466_432
    assert audit["raw_usage_equivalent_cpu_minutes"] == pytest.approx(
        7_466_432 / 60
    )


def test_uhem_quota_rejects_raw_usage_disagreement(monkeypatch, tmp_path: Path) -> None:
    output = "nakane|nunal|cpu=16279700|cpu=124440|cpu=0|6000000\n"
    monkeypatch.setattr(workflow.subprocess, "check_output", lambda *a, **k: output)
    with pytest.raises(workflow.FamilyWorkflowError, match="RawUsage/60"):
        workflow._live_uhem_cpu_saat(tmp_path, "nakane", "nunal")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1.5 TiB", int(1.5 * 1024**4)), ("2 TB", 2 * 1000**4), ("unlimited", None)],
)
def test_storage_unit_parser(raw: str, expected: int | None) -> None:
    assert workflow._parse_storage_bytes(raw, "test") == expected


def test_beegfs_unlimited_quota_uses_physical_free(monkeypatch, tmp_path: Path) -> None:
    output = "uid,used size,hard limit size\n4500,2 TiB,unlimited\n"
    monkeypatch.setattr(workflow.subprocess, "check_output", lambda *a, **k: output)
    monkeypatch.setattr(workflow, "_disk_free_bytes", lambda _path: 123_456)
    free, audit = workflow._live_beegfs_storage(
        tmp_path, uid=4500, storage_pool_id=1, path=tmp_path
    )
    assert free == 123_456
    assert audit["hard_quota_unlimited"] is True
    assert audit["used_bytes"] == 2 * 1024**4


def _sealed_sha(label: str) -> str:
    return (label.encode("utf-8").hex() + "0" * 64)[:64]


def _protocol_fixture(
    recipe: dict,
    *,
    run_kind: str,
    recipe_scope: str,
    model_tag: str,
    depth: int,
    model_dim: int,
    world_size: int,
    device_batch_size: int,
    total_batch_size: int,
    num_iterations: int,
    seed: int,
    exposure_plan: dict,
) -> tuple[dict, dict, dict]:
    preflight = {
        "code": {
            "git_commit": "1" * 40,
            "uv_lock_sha256": "2" * 64,
            "core_file_sha256": recipe["code_provenance"]["exact_file_sha256"],
        },
        "corpus": {
            "dataset_manifest_sha256": "3" * 64,
            "validation_exposure_manifest_sha256": "4" * 64,
            "validation_payload_bytes": 1234,
            "validation_documents": 12,
        },
        "tokenizer": {"package_manifest_sha256": "5" * 64},
    }
    attention = {
        "selection_reason": "fixture_sdpa_fallback",
        "decision": "accepted_sdpa_L_fallback",
        "module_detection": {
            "selected_backend_after_probe": "sdpa",
            "selected_window_pattern": "L",
        },
    }
    scaling = (
        recipe["model"]["scaling_parameters"]
        if depth == 32
        else (
            recipe["weight_decay_proxy_ablation"]["screen_stage"]
            if depth == 12
            else recipe["weight_decay_proxy_ablation"]["confirmation_stage"]
        )["scaling_parameters"]
    )
    total_parameters = recipe["model"]["total_parameters"] if depth == 32 else scaling + 1
    logical_rows = 7
    padded_runtime = (
        -(-logical_rows // (device_batch_size * world_size))
        * device_batch_size
        * 2048
        * world_size
    )
    lr_scale = (total_batch_size / 524_288) ** 0.5
    protocol = {
        "protocol_version": "d32_wsd_strict_v1",
        "recipe_scope": recipe_scope,
        "run_kind": run_kind,
        "code": {
            "git_revision": "1" * 40,
            "upstream_base_revision": recipe["code_provenance"]["upstream_base_revision"],
            "environment_lock_sha256": "2" * 64,
            "exact_core_sha256": recipe["code_provenance"]["exact_file_sha256"],
        },
        "model_config": {
            "sequence_len": 2048,
            "vocab_size": 32768,
            "n_layer": depth,
            "n_head": model_dim // 128,
            "n_kv_head": model_dim // 128,
            "n_embd": model_dim,
            "window_pattern": "L",
        },
        "architecture_cli": {
            "depth": depth,
            "aspect_ratio": 64,
            "head_dim": 128,
            "max_seq_len": 2048,
            "window_pattern": "L",
            "total_parameters": total_parameters,
            "scaling_parameters": scaling,
        },
        "tokenizer": {
            "name": recipe["artifacts"]["tokenizer_name"],
            "artifact_sha256": "5" * 64,
            "vocab_size": 32768,
        },
        "source_dataset_manifest_sha256": "3" * 64,
        "packing_capacity": {"receipt_sha256": None, "selected_topology": None},
        "topology": None,
        "validation": {
            "manifest_sha256": "4" * 64,
            "full_manifest": True,
            "packing_policy": "whole_document_no_crop_rows_before_rank_sharding",
            "bos_boundary_targets_masked": True,
            "padding_targets_masked": True,
            "target_tokens": 100,
            "payload_bytes": 1234,
            "documents": 12,
            "logical_rows": logical_rows,
            "row_layout_sha256": "6" * 64,
            "padded_token_positions_world1": logical_rows * 2048,
            "padded_token_positions_runtime_world": padded_runtime,
            "eval_every_updates": 250,
            "eval_tokens_cli_unused": -1,
        },
        "precision": {"compute_dtype": "torch.bfloat16", "fp8_enabled": False},
        "attention": {
            "backend": "sdpa",
            "window_pattern": "L",
            "probe_sha256": "7" * 64,
            "selection_reason": attention["selection_reason"],
            "decision": attention["decision"],
            "live_fa3_kernel_inventory_sha256": None,
        },
        "checkpointing": {"transactional": True, "save_every_updates": -1},
        "preemption": {
            "signals": ["SIGUSR1", "SIGTERM"],
            "checkpoint_boundary": "next_optimizer_safe_update",
            "exit_code": 75,
        },
        "model_tag": model_tag,
        "data_order": "bestfit",
        "data_order_authority": (
            "sealed_dataset_manifest_materialization_order_with_"
            "upstream_row_group_rank_sharding"
        ),
        "exposure_plan": exposure_plan,
        "seed": seed,
        "world_size": world_size,
        "device_batch_size": device_batch_size,
        "total_batch_size": total_batch_size,
        "num_iterations": num_iterations,
        "optimizer": {
            "gradient_clip_norm": 0.0,
            "learning_rates": {
                "embedding": 0.3 * lr_scale,
                "unembedding": 0.008 * lr_scale,
                "matrix": 0.02 * lr_scale,
                "scalar": 0.5 * lr_scale,
            },
        },
        "schedule": {},
        "parent": None,
    }
    return protocol, preflight, attention


@pytest.mark.parametrize(
    ("run_kind", "scope", "depth", "width", "world", "device_batch", "batch", "steps"),
    [
        ("proxy", "proxy_d12", 12, 768, 1, 16, 524_288, 4200),
        ("signal_smoke", "signal_smoke_ws4", 32, 2048, 4, 4, 2_097_152, 6),
    ],
)
def test_current_nested_trainer_protocol_is_accepted(
    run_kind: str,
    scope: str,
    depth: int,
    width: int,
    world: int,
    device_batch: int,
    batch: int,
    steps: int,
) -> None:
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    exposure = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "test_exposure",
            "world_size": world,
            "seed": 42,
            "horizon": {"unit": "token_positions", "value": steps * batch},
            "canonical_sha256": None,
        }
    )
    protocol, preflight, attention = _protocol_fixture(
        recipe,
        run_kind=run_kind,
        recipe_scope=scope,
        model_tag=f"fixture_{run_kind}",
        depth=depth,
        model_dim=width,
        world_size=world,
        device_batch_size=device_batch,
        total_batch_size=batch,
        num_iterations=steps,
        seed=42,
        exposure_plan=exposure,
    )
    workflow._verify_frozen_protocol(
        protocol,
        recipe=recipe,
        preflight=preflight,
        attention_probe=attention,
        attention_probe_sha256="7" * 64,
        label="fixture",
        run_kind=run_kind,
        recipe_scope=scope,
        model_tag=f"fixture_{run_kind}",
        exposure_plan_sha256=exposure["canonical_sha256"],
        depth=depth,
        model_dim=width,
        world_size=world,
        device_batch_size=device_batch,
        total_batch_size=batch,
        num_iterations=steps,
        eval_every_updates=250,
        seed=42,
    )


def test_legacy_flat_protocol_is_rejected() -> None:
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    with pytest.raises(workflow.FamilyWorkflowError, match="version"):
        workflow._verify_frozen_protocol(
            {
                "protocol_version": "strict_pretrain_v1",
                "optimizer": {"gradient_clip_norm": 0.0},
            },
            recipe=recipe,
            preflight={},
            attention_probe={},
            attention_probe_sha256="7" * 64,
            label="legacy",
            run_kind="smoke",
            recipe_scope="smoke_ws8",
            model_tag="legacy",
            exposure_plan_sha256="8" * 64,
            depth=32,
            model_dim=2048,
            world_size=8,
            device_batch_size=4,
            total_batch_size=2_097_152,
            num_iterations=100,
            eval_every_updates=250,
            seed=42,
        )


def test_smoke_protocol_binds_capacity_without_production_topology(
    tmp_path: Path,
) -> None:
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    world = 8
    steps = 100
    batch = 2_097_152
    exposure = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "test_exposure",
            "world_size": world,
            "seed": 42,
            "horizon": {"unit": "token_positions", "value": steps * batch},
            "canonical_sha256": None,
        }
    )
    protocol, preflight, attention = _protocol_fixture(
        recipe,
        run_kind="smoke",
        recipe_scope="smoke_ws8",
        model_tag="fixture_smoke",
        depth=32,
        model_dim=2048,
        world_size=world,
        device_batch_size=4,
        total_batch_size=batch,
        num_iterations=steps,
        seed=42,
        exposure_plan=exposure,
    )
    selected_capacity = {
        "world_size": world,
        "passes_40x_no_wrap_with_margin": True,
        "safe_global_scheduled_positions": 68_451_041_280,
    }
    capacity = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "turkish_bestfit_capacity_receipt",
            "simulation": {"worlds": {str(world): selected_capacity}},
            "canonical_sha256": None,
        }
    )
    capacity_path = tmp_path / "packing_capacity_receipt.json"
    write_json_atomic(capacity_path, capacity)
    preflight["corpus"]["packing_capacity_receipt"] = {
        "path": str(capacity_path),
        "sha256": capacity["canonical_sha256"],
    }
    protocol["packing_capacity"] = {
        "receipt_sha256": capacity["canonical_sha256"],
        "selected_topology": selected_capacity,
    }

    workflow._verify_frozen_protocol(
        protocol,
        recipe=recipe,
        preflight=preflight,
        attention_probe=attention,
        attention_probe_sha256="7" * 64,
        label="smoke fixture",
        run_kind="smoke",
        recipe_scope="smoke_ws8",
        model_tag="fixture_smoke",
        exposure_plan_sha256=exposure["canonical_sha256"],
        depth=32,
        model_dim=2048,
        world_size=world,
        device_batch_size=4,
        total_batch_size=batch,
        num_iterations=steps,
        eval_every_updates=250,
        seed=42,
    )


def test_production_protocol_binds_capacity_and_topology_gate(tmp_path: Path) -> None:
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    world = 8
    steps = 28_800
    batch = 2_097_152
    exposure = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "test_exposure",
            "world_size": world,
            "seed": 42,
            "horizon": {"unit": "token_positions", "value": steps * batch},
            "canonical_sha256": None,
        }
    )
    protocol, preflight, attention = _protocol_fixture(
        recipe,
        run_kind="production",
        recipe_scope="production_trunk",
        model_tag=recipe["checkpoints"]["trunk_model_tag"],
        depth=32,
        model_dim=2048,
        world_size=world,
        device_batch_size=4,
        total_batch_size=batch,
        num_iterations=steps,
        seed=42,
        exposure_plan=exposure,
    )
    selected_capacity = {
        "world_size": world,
        "passes_40x_no_wrap_with_margin": True,
        "safe_global_scheduled_positions": 68_451_041_280,
    }
    capacity = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "turkish_bestfit_capacity_receipt",
            "simulation": {"worlds": {str(world): selected_capacity}},
            "canonical_sha256": None,
        }
    )
    capacity_path = tmp_path / "packing_capacity_receipt.json"
    write_json_atomic(capacity_path, capacity)
    preflight["corpus"]["packing_capacity_receipt"] = {
        "path": str(capacity_path),
        "sha256": capacity["canonical_sha256"],
    }
    gate_sha = "8" * 64
    gate = {
        "authorized_production_nodes": 2,
        "selection_reason": "fixture_ws8_fallback",
    }
    protocol["packing_capacity"] = {
        "receipt_sha256": capacity["canonical_sha256"],
        "selected_topology": selected_capacity,
    }
    protocol["topology"] = {
        "gate_sha256": gate_sha,
        "authorized_world_size": world,
        "authorized_nodes": 2,
        "selection_reason": gate["selection_reason"],
        "require_single_world_size_for_entire_lineage": True,
    }
    workflow._verify_frozen_protocol(
        protocol,
        recipe=recipe,
        preflight=preflight,
        attention_probe=attention,
        attention_probe_sha256="7" * 64,
        label="production fixture",
        run_kind="production",
        recipe_scope="production_trunk",
        model_tag=recipe["checkpoints"]["trunk_model_tag"],
        exposure_plan_sha256=exposure["canonical_sha256"],
        depth=32,
        model_dim=2048,
        world_size=world,
        device_batch_size=4,
        total_batch_size=batch,
        num_iterations=steps,
        eval_every_updates=250,
        seed=42,
        production_gate=gate,
        production_gate_sha256=gate_sha,
    )


def test_seal_preemption_passes_each_gate_artifact_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Keep the production requeue path aligned with the shared gate verifier."""

    recipe = {"storage": {"maximum_retained_preemption_transactions": 6}}
    recipe_sha = "1" * 64
    paths = {
        "preflight_receipt": tmp_path / "preflight.json",
        "gate": tmp_path / "topology.json",
        "wd_proxy_approval": tmp_path / "wd.json",
        "attention_probe": tmp_path / "attention.json",
    }
    observed: tuple[object, ...] | None = None

    def fake_verify_gate_and_preflight(
        actual_recipe: object,
        actual_recipe_sha: object,
        preflight_path: object,
        gate_path: object,
        proxy_approval_path: object,
        attention_probe_path: object,
    ) -> tuple[dict, str, dict, str, dict, str]:
        nonlocal observed
        observed = (
            actual_recipe,
            actual_recipe_sha,
            preflight_path,
            gate_path,
            proxy_approval_path,
            attention_probe_path,
        )
        return {}, "2" * 64, {}, "3" * 64, {}, "4" * 64

    class StopAfterGateVerification(RuntimeError):
        pass

    monkeypatch.setattr(workflow, "load_recipe", lambda _path: (recipe, recipe_sha))
    monkeypatch.setattr(
        workflow, "_verify_gate_and_preflight", fake_verify_gate_and_preflight
    )
    monkeypatch.setattr(
        workflow,
        "_stage_by_id",
        lambda *_args: (_ for _ in ()).throw(StopAfterGateVerification()),
    )
    args = argparse.Namespace(
        recipe=tmp_path / "recipe.json",
        slurm_restart_count=0,
        stage="trunk_to_s12_fork",
        signal_resume_gate=tmp_path / "signal.json",
        base_dir=tmp_path,
        launch_receipt=tmp_path / "launch.json",
        slurm_job_id="123",
        output=tmp_path / "receipt.json",
        **paths,
    )

    with pytest.raises(StopAfterGateVerification):
        workflow.command_seal_preemption(args)

    assert observed == (
        recipe,
        recipe_sha,
        paths["preflight_receipt"],
        paths["gate"],
        paths["wd_proxy_approval"],
        paths["attention_probe"],
    )


@pytest.mark.parametrize(
    ("launcher", "expected_kind"),
    [
        ("slurm_srun_direct_python_env_v1", "d32_static_srun_rank_exit"),
        ("slurm_batch_direct_python_env_v1", "d32_batch_direct_rank_exit"),
    ],
)
def test_rank_exit_receipt_records_truthful_launcher_kind(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    launcher: str,
    expected_kind: str,
) -> None:
    recipe = {"family_id": "tr_d32_general_bpe32k_v1"}
    monkeypatch.setattr(workflow, "load_recipe", lambda _path: (recipe, "1" * 64))
    output = tmp_path / "rank.json"
    workflow.command_record_rank_exit(
        argparse.Namespace(
            recipe=tmp_path / "recipe.json",
            run_id="test_run",
            phase="proxy_train" if "batch" in launcher else "smoke",
            slurm_job_id="123",
            slurm_step_id="batch" if "batch" in launcher else "0",
            node="a100-node",
            rank=0,
            local_rank=0,
            world_size=1 if "batch" in launcher else 4,
            exit_code=0,
            launcher=launcher,
            output=output,
        )
    )
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["kind"] == expected_kind
    assert receipt["launcher"] == launcher
