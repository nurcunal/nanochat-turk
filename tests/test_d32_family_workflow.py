from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from nanochat.experiment_manifest import seal_manifest, write_json_atomic
from nanochat.strict_runtime import (
    StrictTrainingError,
    family_artifact_contract,
    validate_family_recipe,
)
from nanochat.turkish_corpus import D32_EXPOSURE_MATRIX_V1
from scripts import d32_family_workflow as workflow


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "configs" / "pretrain" / "tr_d32_turkish_general_wsd_v1.json"
RECIPE_V2 = ROOT / "configs" / "pretrain" / "tr_d32_turkish_general_wsd_v2.json"


@pytest.mark.parametrize("recipe_path", [RECIPE, RECIPE_V2])
def test_v1_and_v2_are_exact_supported_runtime_families(recipe_path: Path) -> None:
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    workflow.validate_recipe(recipe)
    validate_family_recipe(recipe)


def test_v2_cannot_reuse_v1_artifact_namespace() -> None:
    recipe = json.loads(RECIPE_V2.read_text(encoding="utf-8"))
    recipe["artifacts"]["corpus_id"] = "tr_general_clean_v1"
    recipe["artifacts"]["corpus_root"] = "pretrain_data/tr_general_clean_v1"
    recipe["artifacts"]["mixture_config"] = (
        "configs/pretrain/tr_d32_turkish_general_v1.json"
    )
    recipe["artifacts"].pop("macocu_preparation_manifest")
    with pytest.raises(workflow.FamilyWorkflowError, match="per-family"):
        workflow.validate_recipe(recipe)
    with pytest.raises(StrictTrainingError, match="exact family identity"):
        validate_family_recipe(recipe)


def test_d32_exposure_matrix_exactly_matches_family_artifact_contract() -> None:
    matrix_keys = {item[0] for item in D32_EXPOSURE_MATRIX_V1}
    contract = family_artifact_contract("tr_d32_general_bpe32k_v2")
    exposure_artifacts = contract["training_exposure_manifests"]

    assert matrix_keys == set(exposure_artifacts)
    assert exposure_artifacts["smoke_ws8"] == "training_exposure_smoke_ws8.json"
    assert exposure_artifacts["smoke_ws16"] == "training_exposure_smoke_ws16.json"


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


def test_beegfs_live_uhem_duplicate_hard_schema_is_parsed_positionally(
    monkeypatch, tmp_path: Path
) -> None:
    output = (
        "name,id,size,hard,files,hard\n"
        "nunal,4500,2791507521536,unlimited,873812,unlimited\n"
    )
    monkeypatch.setattr(workflow.subprocess, "check_output", lambda *a, **k: output)
    monkeypatch.setattr(workflow, "_disk_free_bytes", lambda _path: 987_654)

    free, audit = workflow._live_beegfs_storage(
        tmp_path, uid=4500, storage_pool_id=1, path=tmp_path
    )

    assert free == 987_654
    assert audit["used_bytes"] == 2_791_507_521_536
    assert audit["hard_quota_unlimited"] is True
    assert audit["csv_schema"] == "uhem_name_id_size_hard_files_hard_v1"


def test_beegfs_rejects_missing_requested_uid(monkeypatch, tmp_path: Path) -> None:
    output = "name,id,size,hard,files,hard\nother,9999,1,unlimited,1,unlimited\n"
    monkeypatch.setattr(workflow.subprocess, "check_output", lambda *a, **k: output)
    with pytest.raises(workflow.FamilyWorkflowError, match="requested UID"):
        workflow._live_beegfs_storage(
            tmp_path, uid=4500, storage_pool_id=1, path=tmp_path
        )


def _data_prep_allocation(job_id: str, *, stage: str, elapsed: int = 10) -> dict:
    cpu_time = elapsed * workflow.CPU2DQ_BILLABLE_CPUS
    return {
        "job_id": job_id,
        "job_id_raw": job_id,
        "stage": stage,
        "state": "COMPLETED",
        "partition": "cpu2dq",
        "elapsed_raw_seconds": elapsed,
        "alloc_cpus": workflow.CPU2DQ_BILLABLE_CPUS,
        "cpu_time_raw_seconds": cpu_time,
        "billed_cpu_saat": cpu_time / 3600.0,
        "evidence_receipt_sha256s": ["e" * 64],
        "sacct_output_sha256": "f" * 64,
    }


def _data_prep_measurement(recipe: dict, recipe_sha: str) -> dict:
    components = {
        name: {
            "sample_measured_bytes": 50 + index,
            "projected_peak_bytes_before_safety": 100 + index,
            "projection_basis": f"fixture_{name}",
            "evidence_sha256s": [f"{index + 1:064x}"],
        }
        for index, name in enumerate(workflow.DATA_PREP_STORAGE_COMPONENTS)
    }
    allocations = [
        _data_prep_allocation("100", stage="bootstrap"),
        _data_prep_allocation("101", stage="packed_object_sample", elapsed=20),
    ]
    total_cpu_time = sum(item["cpu_time_raw_seconds"] for item in allocations)
    total_billed = sum(item["billed_cpu_saat"] for item in allocations)
    historical = _data_prep_allocation("99", stage="macocu_genre_preparation")
    historical["evidence_receipt_sha256s"] = ["9" * 64]
    future_values = {
        "production_backend": 10.0,
        "production_pool_materialization": 20.0,
        **{
            name: float(value)
            for name, value in workflow.DATA_PREP_FIXED_CPU2DQ_CEILINGS.items()
        },
    }
    future_components = {
        name: {
            "projected_cpu_saat_before_safety": future_values[name],
            "projection_basis": f"fixture_{name}",
            "evidence_sha256s": [f"{index + 10:064x}"],
        }
        for index, name in enumerate(workflow.DATA_PREP_FUTURE_CPU_COMPONENTS)
    }
    future_cpu = sum(
        item["projected_cpu_saat_before_safety"]
        for item in future_components.values()
    )
    return seal_manifest(
        {
            "schema_version": "3.0",
            "kind": workflow.DATA_PREP_STORAGE_SAMPLE_KIND,
            "family_id": recipe["family_id"],
            "recipe_sha256": recipe_sha,
            "policy_sha256": "1" * 64,
            "source_plan_sha256": "2" * 64,
            "calibration_sha256": "3" * 64,
            "backend_resource_report_sha256": "4" * 64,
            "resource_approval_sha256": "a" * 64,
            "mixture_quality_approval_sha256": "b" * 64,
            "sample_quality_audit_sha256": "c" * 64,
            "sample_cluster_receipt_sha256": "d" * 64,
            "sample_lane_plan_sha256": "5" * 64,
            "production_pack_plan_sha256": "6" * 64,
            "writer_probe_sha256": "7" * 64,
            "macocu_preparation_manifest_sha256": "9" * 64,
            "sample_documents": 100_000,
            "estimated_total_documents": 1_000_000,
            "components": components,
            "sample_allocations": allocations,
            "sample_allocation_totals": {
                "unique_allocations": len(allocations),
                "cpu_time_raw_seconds": total_cpu_time,
                "billed_cpu_saat": total_billed,
                "accounting_role": (
                    "already_consumed_measurement_evidence_not_future_quota"
                ),
            },
            "historical_one_time_preparations": [
                {
                    "preparation_id": "macocu_genre_tr_v1",
                    "manifest_sha256": "9" * 64,
                    "allocation": historical,
                    "accounting_status": (
                        "already_consumed_excluded_from_future_projection"
                    ),
                    "future_projected_cpu_saat": 0,
                }
            ],
            "future_resource_projection": {
                "components": future_components,
                "allocation_details": {
                    "production_backend": {
                        "node_count": 1,
                        "projected_node_wall_seconds_before_safety": {"0": 10.0},
                        "projected_packed_object_cpu_saat_before_safety": (
                            10 * workflow.CPU2DQ_BILLABLE_CPUS / 3600
                        ),
                        "projected_packed_bucket_node_wall_seconds_before_safety": 10.0,
                        "projected_minhash_bucket_cpu_saat_before_safety": (
                            10 * workflow.CPU2DQ_BILLABLE_CPUS / 3600
                        ),
                        "projected_priority_cluster_cpu_saat_before_safety": (
                            10 - 20 * workflow.CPU2DQ_BILLABLE_CPUS / 3600
                        ),
                        "sample_priority_cluster_peak_rss_bytes": 1024**3,
                        "projected_priority_cluster_peak_rss_bytes_before_safety": (
                            2 * 1024**3
                        ),
                        "sample_priority_cluster_edge_participating_documents": 10,
                        "projected_priority_cluster_edge_participating_documents": 20.0,
                        "projected_backend_cpu_saat_before_safety": 10.0,
                    },
                    "production_pool_materialization": {
                        "allocation_contract": (
                            "one_exclusive_128cpu_cpu2dq_node_scaled_by_candidate_documents"
                        ),
                        "sample_elapsed_wall_seconds": 10.0,
                        "document_scale": 2.0,
                        "projected_cpu_saat_before_safety": 20.0,
                    },
                    **{
                        name: {
                            "allocation_contract": (
                                "one_exclusive_128cpu_cpu2dq_node"
                            ),
                            "maximum_wall_hours": (
                                value / workflow.CPU2DQ_BILLABLE_CPUS
                            ),
                            "projected_cpu_saat_before_safety": float(value),
                            "submission_must_not_exceed_ceiling": True,
                        }
                        for name, value in (
                            workflow.DATA_PREP_FIXED_CPU2DQ_CEILINGS.items()
                        )
                    },
                },
                "projected_cpu_saat_before_safety": future_cpu,
                "safety_factor_applied": False,
                "excluded_historical_and_sample_allocations": True,
            },
            "canonical_sha256": None,
        }
    )


def test_data_prep_gate_applies_safety_once_to_explicit_future_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recipe, recipe_sha = workflow.load_recipe(RECIPE_V2)
    measurement = _data_prep_measurement(recipe, recipe_sha)
    measurement_path = tmp_path / "measurement.json"
    output_path = tmp_path / "gate.json"
    write_json_atomic(measurement_path, measurement)
    monkeypatch.setattr(
        workflow,
        "_live_beegfs_storage",
        lambda *_args, **_kwargs: (10**15, {"fixture": True}),
    )
    monkeypatch.setattr(
        workflow,
        "_live_uhem_cpu_saat",
        lambda *_args, **_kwargs: (100_000.0, "c" * 64, {"fixture": True}),
    )
    monkeypatch.setattr(
        workflow, "_validate_storage_approval_evidence", lambda *_args, **_kwargs: None
    )

    workflow.command_data_prep_storage_gate(
        argparse.Namespace(
            recipe=RECIPE_V2,
            policy=ROOT / "configs" / "pretrain" / "tr_d32_turkish_general_v2.json",
            sample_measurement=measurement_path,
            work_dir=tmp_path,
            output=output_path,
        )
    )

    gate = json.loads(output_path.read_text(encoding="utf-8"))
    projected_bytes = sum(100 + index for index in range(7))
    assert gate["projected_peak_bytes_before_safety"] == projected_bytes
    assert gate["projected_peak_bytes_with_safety"] == math.ceil(
        projected_bytes * 1.35
    )
    assert gate["safety_factor_application_count"] == 1
    assert gate["projected_data_preparation_cpu_saat_before_safety"] == 13_854.0
    assert gate["projected_data_preparation_cpu_saat_with_safety"] == 18_703
    assert gate["total_project_operational_ceiling_cpu_saat"] == 58_703
    assert gate["sample_allocation_totals"]["billed_cpu_saat"] > 1
    assert gate["historical_one_time_preparations"][0][
        "future_projected_cpu_saat"
    ] == 0


def test_storage_gate_reopens_actual_approval_evidence(tmp_path: Path) -> None:
    recipe, recipe_sha = workflow.load_recipe(RECIPE_V2)
    measurement = _data_prep_measurement(recipe, recipe_sha)
    missing = {
        "path": "missing.json",
        "size_bytes": 1,
        "sha256": "a" * 64,
    }
    measurement["approval_evidence"] = {
        "schema_version": "1.0",
        "source_plan": missing,
        "calibration": missing,
        "backend_resource_report": missing,
        "resource_approval": missing,
        "mixture_quality_approval": missing,
    }
    measurement = seal_manifest(measurement)
    path = tmp_path / "measurement.json"
    write_json_atomic(path, measurement)

    with pytest.raises(workflow.FamilyWorkflowError, match="missing|escapes"):
        workflow._validate_storage_approval_evidence(
            measurement,
            measurement_path=path,
            policy_path=ROOT
            / "configs"
            / "pretrain"
            / "tr_d32_turkish_general_v2.json",
        )


def test_data_prep_sample_rejects_duplicate_allocations_and_double_safety() -> None:
    recipe, recipe_sha = workflow.load_recipe(RECIPE_V2)
    measurement = _data_prep_measurement(recipe, recipe_sha)
    workflow._validate_data_prep_storage_sample(
        measurement, recipe=recipe, recipe_sha=recipe_sha
    )

    duplicate = copy.deepcopy(measurement)
    duplicate["sample_allocations"][1]["job_id_raw"] = duplicate[
        "sample_allocations"
    ][0]["job_id_raw"]
    duplicate = seal_manifest(duplicate)
    with pytest.raises(workflow.FamilyWorkflowError, match="duplicate allocation"):
        workflow._validate_data_prep_storage_sample(
            duplicate, recipe=recipe, recipe_sha=recipe_sha
        )

    double_safety = copy.deepcopy(measurement)
    double_safety["future_resource_projection"]["safety_factor_applied"] = True
    double_safety = seal_manifest(double_safety)
    with pytest.raises(workflow.FamilyWorkflowError, match="pre-safety"):
        workflow._validate_data_prep_storage_sample(
            double_safety, recipe=recipe, recipe_sha=recipe_sha
        )


def test_production_pack_plan_bills_thirty_two_workers_as_one_node() -> None:
    source_plan = {
        "objects": [
            {"rank": rank, "source_id": "source", "size_bytes": 100}
            for rank in range(64)
        ]
    }
    lanes = workflow._production_pack_plan_lanes(source_plan, 1)
    assert len(lanes) == 32
    assert all(len(lane["object_ranks"]) == 2 for lane in lanes)
    pack_plan = {
        "node_count": 1,
        "lanes": lanes,
    }
    report = {
        "source_projections": {
            "source": {
                "full_input_bytes": 6_400,
                "projected_download_wall_seconds": 64.0,
                "projected_score_lid_wall_seconds": 256.0,
                "projected_minhash_signature_wall_seconds": 320.0,
            }
        },
        "projection": {
            "stage_billed_cpu_saat_before_safety_factor": {
                "download": 64 * 128 / 3600,
                "score_lid": 256 * 128 / 3600,
                "minhash_signature": 320 * 128 / 3600,
                "minhash_buckets": 2.0,
                "priority_cluster_quality_format": 3.0,
            },
            "signature_bytes": 14_000,
            "cluster_scaling": {
                "sample_peak_rss_bytes": 1024**3,
                "projected_peak_rss_bytes": 2 * 1024**3,
                "sample_edge_participating_documents": 10,
                "projected_edge_participating_documents": 20.0,
            },
        },
    }
    bucket_receipts = {
        rank: {
            "input_signature_bytes": 1_000,
            "telemetry": {"wall_seconds": 1.0 + rank / 10},
        }
        for rank in range(14)
    }

    total, audit = workflow._packed_production_backend_cpu_projection(
        source_plan=source_plan,
        pack_plan=pack_plan,
        backend_report=report,
        sample_bucket_receipts=bucket_receipts,
    )

    assert audit["projected_node_wall_seconds_before_safety"] == {"0": 20.0}
    assert audit["projected_packed_object_cpu_saat_before_safety"] == pytest.approx(
        20 * 128 / 3600
    )
    assert audit["projected_signature_bytes_per_bucket_before_safety"] == 1_000
    expected_bucket_wall = 2.3
    expected_bucket_cpu = expected_bucket_wall * 128 / 3600
    assert total == pytest.approx(20 * 128 / 3600 + expected_bucket_cpu + 3.0)


def test_sacct_allocation_uses_exact_array_task_and_full_node_billing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = "123_4|900_4|COMPLETED|cpu2dq|10|128|1280|\n"
    monkeypatch.setattr(workflow.subprocess, "check_output", lambda *a, **k: output)

    allocation = workflow._live_completed_cpu2dq_allocation(
        tmp_path,
        job_id="123_4",
        stage="minhash_bucket_004",
        evidence_receipt_sha256s=["a" * 64],
    )

    assert allocation["job_id"] == "123_4"
    assert allocation["job_id_raw"] == "900_4"
    assert allocation["cpu_time_raw_seconds"] == 1280
    assert allocation["billed_cpu_saat"] == pytest.approx(1280 / 3600)


def test_post_cluster_writer_probe_generator_emits_valid_bounded_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import nanochat.turkish_backend as backend
    import nanochat.turkish_corpus as corpus

    run_dir = tmp_path / "sample"
    run_dir.mkdir()
    cluster_path = run_dir / "backend_output" / "00000.parquet"
    cluster_path.parent.mkdir()
    rows = [
        {
            "source_id": "source",
            "dedup_keep": True,
            "source_lid_label": "tur",
            "source_lid_probability": 0.99,
            "text": f"Türkçe deneme metni {index}",
            "url": "https://example.test/",
            "dedup_cluster_id": f"{index + 1:064x}",
            "document_id": f"doc-{index}",
            "quality_score": 0.9,
        }
        for index in range(2)
    ]
    pq.write_table(pa.Table.from_pylist(rows), cluster_path)
    cluster = {
        "canonical_sha256": "c" * 64,
        "output_files": [
            {
                "path": "backend_output/00000.parquet",
                "size_bytes": cluster_path.stat().st_size,
            }
        ],
    }
    objects = {0: {"candidate_file": {"rows": 2}}}
    report = seal_manifest(
        {
            "policy_sha256": "p" * 64,
            "source_plan_sha256": "s" * 64,
            "calibration_sha256": "a" * 64,
            "sample_selection": {"ranks": [0]},
            "projection": {"safety_factor": 1.0, "candidate_documents": 20.0},
            "automated_gate_passed": True,
            "canonical_sha256": None,
        }
    )
    report_path = tmp_path / "backend-report.json"
    write_json_atomic(report_path, report)
    recipe = {"family_id": "tr_d32_general_bpe32k_v2"}
    policy = {
        "sources": [
            {"id": "source", "adapter": {"turkish_values": ["tur"]}}
        ],
        "content_policy": {},
        "splits": {"seed": 42},
        "materialization": {
            "rows_per_fragment": 1,
            "shuffle_buckets": 4,
            "max_buffered_rows": 4,
            "rows_per_output_file": 4,
        },
    }
    monkeypatch.setattr(
        workflow,
        "_load_data_prep_inputs",
        lambda **_kwargs: (
            recipe,
            "r" * 64,
            policy,
            "p" * 64,
            {"objects": []},
            "s" * 64,
            {},
            "a" * 64,
        ),
    )
    monkeypatch.setattr(backend, "validate_resource_projection", lambda value: value["canonical_sha256"])
    monkeypatch.setattr(
        workflow,
        "_load_sample_receipt_inventory",
        lambda *_args, **_kwargs: (objects, {}, cluster),
    )
    monkeypatch.setattr(corpus, "_production_lid_ok", lambda *_args: True)
    monkeypatch.setattr(corpus, "_production_quality_ok", lambda *_args: True)
    monkeypatch.setattr(
        corpus,
        "audit_document",
        lambda text, **_kwargs: SimpleNamespace(accepted=True, normalized_text=text),
    )
    monkeypatch.setattr(
        corpus, "select_mixture_bucket", lambda *_args: ("general", 0.5)
    )
    monkeypatch.setattr(corpus, "assign_split", lambda *_args: "train")
    monkeypatch.setattr(corpus, "dominant_register", lambda *_args: "general")
    monkeypatch.setattr(
        corpus,
        "stable_shuffle_key",
        lambda document_id, _seed: (document_id.encode().hex() + "0" * 64)[:64],
    )
    output = tmp_path / "writer-probe.json"

    workflow.command_seal_data_prep_writer_probe(
        argparse.Namespace(
            recipe=tmp_path / "recipe.json",
            policy=tmp_path / "policy.json",
            source_plan=tmp_path / "plan.json",
            calibration=tmp_path / "calibration.json",
            sample_run_dir=run_dir,
            backend_resource_report=report_path,
            scratch_dir=tmp_path,
            output=output,
        )
    )

    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["kind"] == workflow.DATA_PREP_WRITER_PROBE_KIND
    assert receipt["sample"]["candidate_documents"] == 2
    assert receipt["sample"]["accepted_documents"] == 2
    assert receipt["sample"]["train_parquet_bytes"] > 0
    assert receipt["sample"]["temporary_peak_bytes"] >= 1_073_741_824
    assert receipt["projection"]["document_scale"] == 10.0
    assert not list(tmp_path.glob("d32-writer-probe-*"))


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
