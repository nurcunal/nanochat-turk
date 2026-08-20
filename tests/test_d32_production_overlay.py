from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import weakref
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch

import nanochat.strict_checkpoint as strict_checkpoints
from nanochat.experiment_manifest import (
    file_sha256,
    seal_manifest,
    write_json_atomic,
)
from nanochat.strict_checkpoint import (
    CheckpointIntegrityError,
    StrictCheckpointPayload,
    build_strict_checkpoint_identity,
    load_strict_checkpoint,
    save_strict_checkpoint,
    strict_checkpoint_dir,
)
from nanochat.gpt import GPT, GPTConfig
from nanochat.strict_dataloader import (
    StatefulBestFitLoader,
    StatefulSequentialDocumentLoader,
    build_validation_rows,
    measure_validation_coverage,
    verify_strict_dataset,
)
from nanochat.strict_runtime import (
    StrictTrainingError,
    derive_seed_plan,
    validate_attention_probe_receipt,
    validate_bestfit_capacity_receipt,
    validate_family_recipe,
    validate_preflight_artifact_bindings,
    validate_production_topology_gate,
    validate_recipe_invocation,
)
from nanochat.strict_tokenizer import (
    TokenizerPackageError,
    verify_tokenizer_package,
)
from nanochat.tokenizer import SPECIAL_TOKENS, SPLIT_PATTERN
from scripts.d32_wsd_train import (
    _parse_args,
    _restore_and_release_checkpoint_payload,
    validate_cli,
)
from strict_loader_fixtures import (
    STUDY_HASH,
    TOKENIZER_HASH,
    strict_dataset,
    strict_training_plan,
    strict_validation_manifest,
)


EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _identity(step: int = 3) -> dict:
    return build_strict_checkpoint_identity(
        study_id="tr-family",
        run_id="run-1",
        study_manifest_sha256="1" * 64,
        run_sha256="2" * 64,
        tokenizer_artifact_sha256="3" * 64,
        exposure_plan_sha256="4" * 64,
        optimizer_audit={"policy": "nanochat_muon_adamw_wsd"},
        curve_log_state={
            "event_count": step,
            "last_event_sha256": "5" * 64 if step else None,
            "last_updates_completed": step,
            "file_sha256": EMPTY_SHA256,
        },
    )


def _proxy_cli() -> list[str]:
    return [
        "--strict-mode",
        "--strict-run-kind=proxy",
        "--no-fp8",
        "--attention-backend=sdpa",
        "--depth=12",
        "--aspect-ratio=64",
        "--head-dim=128",
        "--max-seq-len=2048",
        "--window-pattern=L",
        "--num-iterations=4200",
        "--target-flops=-1",
        "--target-param-data-ratio=-1",
        "--target-param-count=scaling",
        "--horizon-unit=token_positions",
        f"--horizon-value={4200 * 524288}",
        "--device-batch-size=16",
        "--total-batch-size=524288",
        "--optimizer=muon_adamw",
        "--lr-schedule=nanochat_linear",
        "--grad-clip=0",
        "--eval-every=100",
        "--core-metric-every=-1",
        "--sample-every=-1",
        "--save-every=-1",
        "--stop-at-step=4200",
        "--model-tag=proxy_d12_control",
        "--study-id=tr_d32_general_bpe32k_v1",
        "--run-id=proxy_d12_control_seed42",
        "--study-manifest=recipe.json",
        "--preflight-receipt=preflight.json",
        "--attention-probe=attention.json",
        "--metrics-dir=metrics",
        "--tokenizer-manifest=package_manifest.json",
        "--exposure-plan=exposure.json",
        "--validation-manifest=validation.json",
        "--data-dir=data",
        "--code-revision=" + "a" * 40,
        "--seed=42",
        "--data-order=bestfit",
    ]


def test_dedicated_cli_accepts_launcher_surface_and_optional_separator() -> None:
    direct = _parse_args(_proxy_cli())
    separated = _parse_args(["--", *_proxy_cli()])
    validate_cli(direct)
    validate_cli(separated)
    assert vars(direct) == vars(separated)
    assert direct.fp8 is False
    assert direct.grad_clip == 0.0


def test_distributed_smoke_cli_requires_explicit_capacity_receipt() -> None:
    cli = [
        value.replace("--strict-run-kind=proxy", "--strict-run-kind=smoke")
        .replace("--lr-schedule=nanochat_linear", "--lr-schedule=wsd")
        for value in _proxy_cli()
    ]
    cli.extend(
        [
            "--wsd-recipe-version=tr_d32_wsd_wd_proxy_v1",
            "--wsd-base-weight-decay=0.1",
            "--wsd-weight-decay-cooldown=constant",
            "--wsd-proxy-approval=approval.json",
        ]
    )
    with pytest.raises(StrictTrainingError, match="packing-capacity-receipt"):
        validate_cli(_parse_args(cli))
    cli.append("--packing-capacity-receipt=data/packing_capacity_receipt.json")
    validate_cli(_parse_args(cli))


def test_dedicated_trainer_has_no_research_path_imports() -> None:
    source = Path("scripts/d32_wsd_train.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = {
        "nanochat.checkpoint_manager",
        "nanochat.continuation",
        "nanochat.dataloader",
        "nanochat.dataset",
        "nanochat.loss_eval",
        "nanochat.tokenizer_package",
        "nanochat.training_protocol",
        "scripts.base_train",
    }
    assert imported.isdisjoint(forbidden)
    assert not any(name.startswith("nanochat.morphology") for name in imported)


def test_family_workflow_has_no_research_tokenizer_dependency() -> None:
    source = Path("scripts/d32_family_workflow.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "nanochat.strict_tokenizer" in imported
    assert "nanochat.tokenizer_package" not in imported


def test_seed_plan_uses_literal_upstream_seed_on_every_rank() -> None:
    rank0 = derive_seed_plan(42, rank=0)
    rank1 = derive_seed_plan(42, rank=1)
    assert rank0.model_init == rank1.model_init == 42
    assert rank0.runtime == rank1.runtime == 42
    assert derive_seed_plan(42, rank=1) == rank1
    torch.manual_seed(rank0.model_init)
    actual = torch.rand(8)
    torch.manual_seed(42)
    expected = torch.rand(8)
    assert torch.equal(actual, expected)

    def construct(seed: int) -> dict[str, torch.Tensor]:
        torch.manual_seed(seed)
        with torch.device("meta"):
            model = GPT(
                GPTConfig(
                    sequence_len=8,
                    vocab_size=64,
                    n_layer=2,
                    n_head=2,
                    n_kv_head=2,
                    n_embd=16,
                    window_pattern="L",
                )
            )
        model.to_empty(device="cpu")
        model.init_weights()
        return {name: value.detach().clone() for name, value in model.state_dict().items()}

    pinned_literal = construct(42)
    strict_literal = construct(rank0.model_init)
    assert pinned_literal.keys() == strict_literal.keys()
    assert all(
        torch.equal(pinned_literal[name], strict_literal[name])
        for name in pinned_literal
    )


def test_preflight_artifact_binding_rejects_stale_and_corrupt_inputs(
    tmp_path: Path,
) -> None:
    recipe_sha = "1" * 64
    dataset_sha = "2" * 64
    tokenizer_sha = "3" * 64
    validation_sha = "4" * 64
    exposure_sha = "5" * 64
    code_revision = "a" * 40
    validation_path = tmp_path / "validation.parquet"
    validation_path.write_bytes(b"fixed-validation")
    capacity = seal_manifest(
        {
            "kind": "turkish_bestfit_capacity_receipt",
            "gate_passed": True,
            "canonical_sha256": None,
        }
    )
    capacity_path = tmp_path / "packing_capacity_receipt.json"
    write_json_atomic(capacity_path, capacity)
    corpus = seal_manifest(
        {
            "nanochat_dataset_manifest_sha256": dataset_sha,
            "tokenizer": {"package_sha256": tokenizer_sha},
            "packing_capacity": {"sha256": capacity["canonical_sha256"]},
            "canonical_sha256": None,
        }
    )
    write_json_atomic(tmp_path / "corpus_manifest.json", corpus)
    exposure_index = seal_manifest(
        {
            "kind": "d32_exposure_plan_index",
            "study_manifest_sha256": recipe_sha,
            "source_dataset_manifest_sha256": dataset_sha,
            "tokenizer_artifact_sha256": tokenizer_sha,
            "packing_capacity_receipt_sha256": capacity["canonical_sha256"],
            "validation": {"sha256": validation_sha},
            "plans": [{"sha256": exposure_sha}],
            "canonical_sha256": None,
        }
    )
    write_json_atomic(tmp_path / "exposure_plan_index.json", exposure_index)
    preflight = seal_manifest(
        {
            "kind": "d32_family_preflight_receipt",
            "family_id": "tr_d32_general_bpe32k_v1",
            "recipe": {"canonical_sha256": recipe_sha},
            "code": {
                "git_commit": code_revision,
                "pyproject_sha256": "6" * 64,
                "uv_lock_sha256": "7" * 64,
                "uv_version": "0.11.29",
                "python_version": "3.12.4",
                "environment_sync_mode": "uv_sync_frozen_extra_gpu",
            },
            "tokenizer": {"package_manifest_sha256": tokenizer_sha},
            "corpus": {
                "root": str(tmp_path.resolve()),
                "manifest_sha256": corpus["canonical_sha256"],
                "dataset_manifest_sha256": dataset_sha,
                "validation_exposure_manifest_sha256": validation_sha,
                "validation_payload_bytes": 16,
                "validation_documents": 2,
                "exposure_plan_index_sha256": exposure_index["canonical_sha256"],
                "training_exposure_plans": {
                    "fixture": {"sha256": exposure_sha}
                },
                "packing_capacity_receipt": {
                    "path": str(capacity_path.resolve()),
                    "sha256": capacity["canonical_sha256"],
                    "gate_passed": True,
                },
                "validation_file": "validation.parquet",
                "validation_file_size_bytes": validation_path.stat().st_size,
                "validation_file_sha256": file_sha256(validation_path),
            },
            "canonical_sha256": None,
        }
    )
    preflight_path = tmp_path / "preflight.json"
    write_json_atomic(preflight_path, preflight)
    kwargs = {
        "recipe": {
            "family_id": "tr_d32_general_bpe32k_v1",
            "code_provenance": {
                "training_environment": {
                    "pyproject_sha256": "6" * 64,
                    "uv_lock_sha256": "7" * 64,
                    "uv_version": "0.11.29",
                    "python_version": "3.12.4",
                    "sync_mode": "uv_sync_frozen_extra_gpu",
                }
            },
        },
        "recipe_sha256": recipe_sha,
        "code_revision": code_revision,
        "data_dir": tmp_path,
        "tokenizer_sha256": tokenizer_sha,
        "dataset_manifest": {"validation_file": "validation.parquet"},
        "dataset_sha256": dataset_sha,
        "validation_manifest": {
            "selection": {
                "realized_payload_bytes": 16,
                "realized_documents": 2,
            }
        },
        "validation_sha256": validation_sha,
        "exposure_plan_sha256": exposure_sha,
    }
    _loaded, loaded_sha, capacity_sha = validate_preflight_artifact_bindings(
        preflight_path, **kwargs
    )
    assert loaded_sha == preflight["canonical_sha256"]
    assert capacity_sha == capacity["canonical_sha256"]

    stale = json.loads(json.dumps(preflight))
    stale["tokenizer"]["package_manifest_sha256"] = "f" * 64
    stale["canonical_sha256"] = None
    stale = seal_manifest(stale)
    stale_path = tmp_path / "stale_preflight.json"
    write_json_atomic(stale_path, stale)
    with pytest.raises(StrictTrainingError, match="another tokenizer"):
        validate_preflight_artifact_bindings(stale_path, **kwargs)

    corrupted = json.loads(json.dumps(capacity))
    corrupted["gate_passed"] = False
    write_json_atomic(capacity_path, corrupted)
    with pytest.raises(StrictTrainingError, match="invalid packing capacity"):
        validate_preflight_artifact_bindings(preflight_path, **kwargs)


def test_attention_receipt_requires_exact_d32_construction_and_dtypes(
    tmp_path: Path,
) -> None:
    recipe_sha = "1" * 64
    code_revision = "a" * 40
    flash_sha = "2" * 64
    recipe = {
        "family_id": "tr_d32_general_bpe32k_v1",
        "attention_backend_gate": {
            "required_gpu_family": "A100",
            "preferred_backend": "fa3",
            "preferred_window_pattern": "SSSL",
            "fallback_backend": "sdpa",
            "fallback_window_pattern": "L",
        },
        "code_provenance": {
            "exact_file_sha256": {"nanochat/flash_attention.py": flash_sha}
        },
    }
    preflight = seal_manifest(
        {
            "kind": "d32_family_preflight_receipt",
            "recipe": {"canonical_sha256": recipe_sha},
            "code": {"git_commit": code_revision},
            "canonical_sha256": None,
        }
    )
    preflight_path = tmp_path / "preflight.json"
    write_json_atomic(preflight_path, preflight)
    full_model = {
        "backend": "sdpa",
        "window_pattern": "L",
        "config": {
            "depth": 32,
            "model_dim": 2048,
            "num_heads": 16,
            "num_kv_heads": 16,
            "head_dim": 128,
            "max_seq_len": 2048,
            "vocab_size": 32768,
        },
        "construction": "meta_then_to_empty_cuda_then_literal_seed42_init_weights",
        "initialization_seed": 42,
        "batch_sequences": 1,
        "sequence_length": 2048,
        "compute_dtype": "torch.bfloat16",
        "parameter_dtype_inventory": {
            "torch.bfloat16": {
                "tensor_count": 17,
                "element_count": 1_140_850_688,
            },
            "torch.float32": {
                "tensor_count": 214,
                "element_count": 1_677_724_762,
            },
        },
        "buffer_dtype_inventory": {
            "torch.float32": {"tensor_count": 2, "element_count": 2_621_440}
        },
        "output_finite": True,
        "loss_finite": True,
        "gradients_present": True,
        "gradients_finite": True,
        "gradient_tensor_count": 231,
        "peak_cuda_memory_bytes": 1,
    }
    probe = seal_manifest(
        {
            "kind": "d32_attention_backend_probe",
            "family_id": recipe["family_id"],
            "recipe_sha256": recipe_sha,
            "preflight_receipt_sha256": preflight["canonical_sha256"],
            "code_revision": code_revision,
            "world_size": 1,
            "gpu": {"name": "NVIDIA A100-SXM4-80GB"},
            "module_detection": {
                "selected_backend_after_probe": "sdpa",
                "selected_window_pattern": "L",
                "flash_attention_file_sha256": flash_sha,
            },
            "selection_reason": "fa3 unavailable; accepted reviewed fallback",
            "decision": "accepted_sdpa_L_fallback",
            "selected_d32_model_forward_backward": full_model,
            "pattern_benchmarks": {"L": {}, "SSSL": {}},
            "canonical_sha256": None,
        }
    )
    probe_path = tmp_path / "probe.json"
    write_json_atomic(probe_path, probe)
    _loaded, digest = validate_attention_probe_receipt(
        probe_path,
        preflight_path,
        recipe=recipe,
        recipe_sha256=recipe_sha,
        code_revision=code_revision,
        attention_backend="sdpa",
        window_pattern="L",
    )
    assert digest == probe["canonical_sha256"]

    stale = json.loads(json.dumps(probe))
    stale["selected_d32_model_forward_backward"]["construction"] = "direct_cuda"
    stale["canonical_sha256"] = None
    stale = seal_manifest(stale)
    write_json_atomic(probe_path, stale)
    with pytest.raises(StrictTrainingError, match="construction"):
        validate_attention_probe_receipt(
            probe_path,
            preflight_path,
            recipe=recipe,
            recipe_sha256=recipe_sha,
            code_revision=code_revision,
            attention_backend="sdpa",
            window_pattern="L",
        )


def test_capacity_receipt_binds_explicit_path_and_40x_safe_horizon(
    tmp_path: Path,
) -> None:
    dataset_sha = "1" * 64
    tokenizer_sha = "2" * 64
    recipe_sha = "3" * 64
    exposure_sha = "4" * 64
    global_batch = 2_097_152
    safe_positions = 32_640 * global_batch

    def world_record(world_size: int) -> dict:
        accumulation = global_batch // (world_size * 4 * 2048)
        requested = 32_640 * accumulation
        return {
            "world_size": world_size,
            "device_batch_sequences": 4,
            "max_seq_len": 2048,
            "buffer_size": 1000,
            "preserve_document_tails": False,
            "row_capacity": 2049,
            "rank_sharding": "parquet_row_group_index_mod_world_size",
            "gradient_accumulation_steps": accumulation,
            "requested_microbatches_per_rank": requested,
            "completed_microbatches_by_rank": [requested] * world_size,
            "first_wrap_before_microbatch_by_rank": [None] * world_size,
            "required_optimizer_steps": 32_000,
            "safety_margin_fraction": 0.02,
            "required_optimizer_steps_with_margin": 32_640,
            "required_positions_with_margin": safe_positions,
            "passes_40x_no_wrap_with_margin": True,
            "first_wrap_observation": "right_censored_at_required_horizon",
            "common_prefix_scheduled_positions": safe_positions,
            "safe_global_scheduled_positions": safe_positions,
            "safe_global_scheduled_positions_semantics": (
                "right_censored_proven_lower_bound_at_required_horizon"
            ),
            "aggregate_scope": "exact_common_required_horizon_all_ranks",
        }

    receipt = seal_manifest(
        {
            "kind": "turkish_bestfit_capacity_receipt",
            "dataset_manifest_sha256": dataset_sha,
            "tokenizer_package_sha256": tokenizer_sha,
            "mix_gate_evaluated_on_common_horizon": True,
            "mix_gate_passed": True,
            "no_wrap_gate_passed": True,
            "gate_passed": True,
            "cleanup_authorized": True,
            "recommendation_requires_fresh_simulation": True,
            "simulation": {
                "implementation": "nanochat_upstream_bos_bestfit_crop_capacity_v2",
                "implementation_file_sha256": file_sha256(
                    Path("nanochat/packing_capacity.py")
                ),
                "all_worlds_pass": True,
                "upstream_contract": {
                    "nanochat_revision": "92d63d4e",
                    "encode_call": (
                        "tokenizer.encode(doc_batch, prepend=bos_token, num_threads=4)"
                    ),
                    "tokenizer_batch_size": 128,
                    "tokenizer_threads": 4,
                    "refill_buffer_size": 1000,
                    "tie_breaks": "first_largest_fit_else_first_shortest",
                    "cropped_tail_policy": "discard",
                    "rank_sharding": "row_group_index_mod_world_size",
                },
                "fixture_parity": {
                    "passed": True,
                    "upstream_commit": (
                        "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"
                    ),
                },
                "worlds": {"8": world_record(8), "16": world_record(16)},
            },
            "canonical_sha256": None,
        }
    )
    capacity_path = tmp_path / "packing_capacity_receipt.json"
    write_json_atomic(capacity_path, receipt)
    no_crop = {
        "policy": "whole_document_no_crop",
        "max_payload_tokens": 2048,
        "max_encoded_tokens_with_bos": 2049,
        "oversized_document_action": "excluded_before_exposure_selection",
    }
    corpus = seal_manifest(
        {
            "nanochat_dataset_manifest_sha256": dataset_sha,
            "tokenizer": {"package_sha256": tokenizer_sha},
            "packing_capacity": {
                "sha256": receipt["canonical_sha256"],
                "all_worlds_pass": True,
            },
            "validation_whole_document_no_crop": no_crop,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(tmp_path / "corpus_manifest.json", corpus)
    index = seal_manifest(
        {
            "kind": "d32_exposure_plan_index",
            "study_manifest_sha256": recipe_sha,
            "source_dataset_manifest_sha256": dataset_sha,
            "tokenizer_artifact_sha256": tokenizer_sha,
            "packing_capacity_receipt_sha256": receipt["canonical_sha256"],
            "plans": [{"sha256": exposure_sha}],
            "canonical_sha256": None,
        }
    )
    write_json_atomic(tmp_path / "exposure_plan_index.json", index)
    _loaded, digest, selected = validate_bestfit_capacity_receipt(
        capacity_path,
        data_dir=tmp_path,
        dataset_manifest={"metadata": {"validation_policy": no_crop}},
        dataset_sha256=dataset_sha,
        tokenizer_sha256=tokenizer_sha,
        recipe_sha256=recipe_sha,
        exposure_plan_sha256=exposure_sha,
        world_size=8,
        required_token_positions=32_000 * global_batch,
        batch_sequences=4,
        sequence_length=2048,
        global_batch_tokens=global_batch,
    )
    assert digest == receipt["canonical_sha256"]
    assert selected["safe_global_scheduled_positions"] == safe_positions
    with pytest.raises(StrictTrainingError, match="must name"):
        validate_bestfit_capacity_receipt(
            tmp_path / "copy.json",
            data_dir=tmp_path,
            dataset_manifest={"metadata": {"validation_policy": no_crop}},
            dataset_sha256=dataset_sha,
            tokenizer_sha256=tokenizer_sha,
            recipe_sha256=recipe_sha,
            exposure_plan_sha256=exposure_sha,
            world_size=8,
            required_token_positions=32_000 * global_batch,
            batch_sequences=4,
            sequence_length=2048,
            global_batch_tokens=global_batch,
        )


def test_topology_gate_is_bound_to_selected_capacity_world(tmp_path: Path) -> None:
    recipe_sha = "1" * 64
    attention_sha = "2" * 64
    approval_sha = "3" * 64
    capacity_sha = "4" * 64
    safe_positions = 68_451_041_280
    preflight = seal_manifest(
        {
            "kind": "d32_family_preflight_receipt",
            "recipe": {"canonical_sha256": recipe_sha},
            "corpus": {
                "packing_capacity_receipt": {
                    "sha256": capacity_sha,
                    "worlds": {
                        "8": {
                            "passes_40x_no_wrap_with_margin": True,
                            "safe_global_scheduled_positions": safe_positions,
                        }
                    },
                }
            },
            "canonical_sha256": None,
        }
    )
    preflight_path = tmp_path / "preflight.json"
    write_json_atomic(preflight_path, preflight)
    gate = seal_manifest(
        {
            "kind": "d32_production_topology_gate",
            "family_id": "tr_d32_general_bpe32k_v1",
            "recipe_sha256": recipe_sha,
            "preflight_receipt_sha256": preflight["canonical_sha256"],
            "attention_probe_sha256": attention_sha,
            "wsd_proxy_approval_sha256": approval_sha,
            "accepted_base_weight_decay": 0.1,
            "accepted_weight_decay_cooldown_policy": "constant",
            "passed": True,
            "authorized_production_world_size": 8,
            "authorized_production_nodes": 2,
            "packing_capacity_receipt_sha256": capacity_sha,
            "authorized_packing_capacity_world_size": 8,
            "authorized_safe_global_scheduled_positions": safe_positions,
            "require_single_world_size_for_entire_lineage": True,
            "required_speedup": 1.7,
            "signal_resume_gate_sha256": "5" * 64,
            "smoke_8gpu_sha256": "6" * 64,
            "smoke_16gpu_sha256": None,
            "throughput_8gpu": 100_000.0,
            "throughput_16gpu": None,
            "preferred_topology_accepted": False,
            "selection_reason": "use measured ws8 fallback",
            "cost_projection": {
                "version": "measured_smoke_v1",
                "selected_smoke_sha256": "6" * 64,
                "world_size": 8,
                "nodes": 2,
                "global_batch_tokens": 2_097_152,
                "full_shared_updates": 34_560,
                "full_scheduled_positions": 72_477_573_120,
                "measured_positions_per_second": 100_000.0,
                "billing_cpu_saat_per_node_hour": 64,
                "reserve_fraction": 0.15,
                "raw_training_cpu_saat_ceiling": 25_770,
                "reserved_training_cpu_saat": 29_636,
                "proxy_smoke_allowance_cpu_saat": 4_000,
                "projected_total_package_cpu_saat": 33_636,
                "operational_ceiling_cpu_saat": 40_000,
                "passed": True,
            },
            "storage_calibration": {"safety_factor": 1.25},
            "canonical_sha256": None,
        }
    )
    gate_path = tmp_path / "gate.json"
    write_json_atomic(gate_path, gate)
    recipe = {
        "family_id": "tr_d32_general_bpe32k_v1",
        "distributed_gate": {
            "minimum_8_to_16_gpu_speedup": 1.7,
            "minimum_parallel_efficiency": 0.85,
        },
        "training": {"global_batch_tokens": 2_097_152},
        "stages": [{"source_step": None, "target_step": 34_560}],
        "uhem_budget": {
            "cpu_saat_per_4gpu_node_hour": 64,
            "proxy_and_smoke_reserve_cpu_saat": 4_000,
            "operational_ceiling_cpu_saat": 40_000,
        },
    }
    selected = {"safe_global_scheduled_positions": safe_positions}
    _loaded, digest = validate_production_topology_gate(
        gate_path,
        preflight_path,
        recipe=recipe,
        recipe_sha256=recipe_sha,
        attention_probe_sha256=attention_sha,
        proxy_approval_sha256=approval_sha,
        accepted_base_weight_decay=0.1,
        accepted_weight_decay_cooldown_policy="constant",
        world_size=8,
        packing_capacity_receipt_sha256=capacity_sha,
        selected_capacity=selected,
    )
    assert digest == gate["canonical_sha256"]
    tampered_gate = dict(gate)
    tampered_gate["cost_projection"] = dict(gate["cost_projection"])
    tampered_gate["cost_projection"]["raw_training_cpu_saat_ceiling"] -= 1
    write_json_atomic(gate_path, seal_manifest(tampered_gate))
    with pytest.raises(StrictTrainingError, match="measured-cost"):
        validate_production_topology_gate(
            gate_path,
            preflight_path,
            recipe=recipe,
            recipe_sha256=recipe_sha,
            attention_probe_sha256=attention_sha,
            proxy_approval_sha256=approval_sha,
            accepted_base_weight_decay=0.1,
            accepted_weight_decay_cooldown_policy="constant",
            world_size=8,
            packing_capacity_receipt_sha256=capacity_sha,
            selected_capacity=selected,
        )
    write_json_atomic(gate_path, gate)
    with pytest.raises(StrictTrainingError, match="capacity"):
        validate_production_topology_gate(
            gate_path,
            preflight_path,
            recipe=recipe,
            recipe_sha256=recipe_sha,
            attention_probe_sha256=attention_sha,
            proxy_approval_sha256=approval_sha,
            accepted_base_weight_decay=0.1,
            accepted_weight_decay_cooldown_policy="constant",
            world_size=8,
            packing_capacity_receipt_sha256="f" * 64,
            selected_capacity=selected,
        )


def test_additive_checkpoint_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    identity = _identity()
    barriers: list[int] = []
    manifest = save_strict_checkpoint(
        tmp_path,
        3,
        {"weight": torch.tensor([1.0])},
        {"optimizer": torch.tensor([2.0])},
        {"step": 3, "updates_completed": 3},
        loader_state={"cursor": 7},
        rng_state={"torch": torch.tensor([8])},
        rank=0,
        expected_world_size=1,
        identity=identity,
        updates_completed=3,
        barrier=lambda: barriers.append(1),
    )
    assert barriers == [1, 1]
    assert manifest["updates_completed"] == 3
    loaded = load_strict_checkpoint(
        tmp_path,
        3,
        "cpu",
        rank=0,
        expected_world_size=1,
        expected_identity=identity,
        expected_updates_completed=3,
    )
    assert torch.equal(loaded.model_data["weight"], torch.tensor([1.0]))
    assert loaded.loader_state == {"cursor": 7}

    path = strict_checkpoint_dir(tmp_path, 3) / "rank_00000_optimizer.pt"
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(CheckpointIntegrityError, match="hash mismatch"):
        strict_checkpoints.inspect_strict_checkpoint(tmp_path, 3)


def test_trainer_releases_duplicate_model_payload_before_compile() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    references: dict[str, weakref.ReferenceType] = {}

    def load_payload() -> StrictCheckpointPayload:
        weight = torch.tensor([[7.0]])
        payload = StrictCheckpointPayload(
            step=1,
            updates_completed=1,
            model_data={"weight": weight},
            optimizer_data=optimizer.state_dict(),
            loader_state={"cursor": 9},
            rng_state={"seed_plan": {"fixture": True}},
            meta_data={"updates_completed": 1},
            manifest={"canonical_sha256": "1" * 64},
        )
        references["weight"] = weakref.ref(weight)
        references["payload"] = weakref.ref(payload)
        return payload

    meta, loader, rng, evidence = _restore_and_release_checkpoint_payload(
        load_payload,
        model=model,
        optimizer=optimizer,
        load_model=True,
        device=torch.device("cpu"),
    )
    assert torch.equal(model.weight, torch.tensor([[7.0]]))
    assert meta == {"updates_completed": 1}
    assert loader == {"cursor": 9}
    assert rng == {"seed_plan": {"fixture": True}}
    assert references["payload"]() is None
    assert references["weight"]() is None
    assert evidence == {
        "model_tensor_count_released": 1,
        "model_tensor_bytes_released": 4,
        "payload_container_released": 1,
    }


def test_additive_bestfit_loader_resumes_exactly(tmp_path: Path) -> None:
    dataset, _validation, tokenizer = strict_dataset(
        tmp_path, ("abcdefg", "ab", "çd", "a")
    )
    contract = verify_strict_dataset(tmp_path)
    kwargs = {
        "data_dir": tmp_path,
        "token_bytes": tokenizer.token_bytes(),
        "study_sha256": STUDY_HASH,
        "tokenizer_sha256": TOKENIZER_HASH,
        "device": "cpu",
        "rank": 0,
        "world_size": 1,
        "buffer_size": 3,
        "dataset_contract": contract,
        "exposure_plan": strict_training_plan(dataset),
    }
    uninterrupted = StatefulBestFitLoader(tokenizer, 1, 4, **kwargs)
    _x, _y, checkpoint_state = next(uninterrupted)
    expected_x, expected_y, expected_state = next(uninterrupted)
    resumed = StatefulBestFitLoader(
        tokenizer,
        1,
        4,
        resume_state=checkpoint_state,
        restore_rng=False,
        **kwargs,
    )
    actual_x, actual_y, actual_state = next(resumed)
    assert torch.equal(actual_x, expected_x)
    assert torch.equal(actual_y, expected_y)
    for field in ("position", "buffer", "totals", "next_batch_index"):
        assert actual_state[field] == expected_state[field]
    assert actual_state["resume_lineage"] == [checkpoint_state["canonical_sha256"]]


@pytest.mark.parametrize(("rank", "world_size"), [(0, 1), (0, 2), (1, 2)])
def test_bestfit_batch_trace_matches_pinned_upstream(
    tmp_path: Path, rank: int, world_size: int
) -> None:
    """Execute the pinned functions and compare the uninterrupted token trace."""

    dataset, _validation, tokenizer = strict_dataset(
        tmp_path,
        ("abcdefg", "ab", "çd", "a", "bcdef", "d"),
    )
    revision = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"
    source = subprocess.check_output(
        ["git", "show", f"{revision}:nanochat/dataloader.py"], text=True
    )
    tree = ast.parse(source)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name
        in {
            "_document_batches",
            "tokenizing_distributed_data_loader_with_state_bos_bestfit",
        }
    ]
    assert [node.name for node in selected] == [
        "_document_batches",
        "tokenizing_distributed_data_loader_with_state_bos_bestfit",
    ]
    train_path = str(tmp_path / "train.parquet")
    validation_path = str(tmp_path / "validation.parquet")
    namespace = {
        "torch": torch,
        "pq": pq,
        "get_dist_info": lambda: (
            world_size > 1,
            rank,
            rank,
            world_size,
        ),
        "list_parquet_files": lambda warn_on_legacy=False: [
            train_path,
            validation_path,
        ],
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<pinned>", "exec"), namespace)
    upstream = namespace["tokenizing_distributed_data_loader_with_state_bos_bestfit"](
        tokenizer,
        1,
        4,
        "train",
        tokenizer_threads=4,
        tokenizer_batch_size=2,
        device="cpu",
        buffer_size=3,
    )
    strict = StatefulBestFitLoader(
        tokenizer,
        1,
        4,
        data_dir=tmp_path,
        token_bytes=tokenizer.token_bytes(),
        study_sha256=STUDY_HASH,
        tokenizer_sha256=TOKENIZER_HASH,
        exposure_plan=strict_training_plan(dataset, world_size=world_size),
        device="cpu",
        rank=rank,
        world_size=world_size,
        tokenizer_batch_size=2,
        tokenizer_threads=4,
        buffer_size=3,
        dataset_contract=verify_strict_dataset(tmp_path),
    )
    for _ in range(8):
        expected_x, expected_y, _approximate_state = next(upstream)
        actual_x, actual_y, _exact_state = next(strict)
        assert torch.equal(actual_x, expected_x)
        assert torch.equal(actual_y, expected_y)
    if world_size == 1:
        assert strict.totals.documents_cropped > 0
        assert strict.totals.discarded_tokens > 0


def test_validation_coverage_is_exactly_measured_and_bound(tmp_path: Path) -> None:
    dataset, _unused_exposure, tokenizer = strict_dataset(tmp_path, ("abcd", "çab"))
    validation = strict_validation_manifest(dataset)
    contract = verify_strict_dataset(tmp_path)
    coverage = measure_validation_coverage(
        tokenizer,
        exposure_manifest=validation,
        data_dir=tmp_path,
        token_bytes=tokenizer.token_bytes(),
        study_sha256=STUDY_HASH,
        tokenizer_sha256=TOKENIZER_HASH,
        dataset_contract=contract,
        sequence_length=3,
    )
    assert coverage["payload_bytes"] == validation["selection"]["realized_payload_bytes"]
    assert coverage["documents"] == validation["selection"]["realized_documents"]
    assert coverage["target_tokens"] > 0
    assert coverage["padded_token_positions_world1"] >= coverage["target_tokens"]


@pytest.mark.parametrize("world_size", [1, 2, 8, 16])
def test_validation_rows_and_context_loss_are_topology_invariant(
    tmp_path: Path, world_size: int
) -> None:
    validation_texts = ("ab", "c", "de", "a")
    dataset, _unused, tokenizer = strict_dataset(
        tmp_path,
        ("abcde",),
        validation_texts=validation_texts,
    )
    validation = strict_validation_manifest(dataset, validation_texts)
    contract = verify_strict_dataset(tmp_path)
    rows = build_validation_rows(
        tokenizer,
        exposure_manifest=validation,
        data_dir=tmp_path,
        token_bytes=tokenizer.token_bytes(),
        study_sha256=STUDY_HASH,
        tokenizer_sha256=TOKENIZER_HASH,
        sequence_length=4,
        dataset_contract=contract,
    )
    observed: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    aggregate_loss = 0.0
    aggregate_targets = 0
    for rank in range(world_size):
        loader = StatefulSequentialDocumentLoader(
            tokenizer,
            1,
            4,
            exposure_manifest=validation,
            data_dir=tmp_path,
            token_bytes=tokenizer.token_bytes(),
            study_sha256=STUDY_HASH,
            tokenizer_sha256=TOKENIZER_HASH,
            device="cpu",
            rank=rank,
            world_size=world_size,
            dataset_contract=contract,
            prepared_rows=rows,
        )
        for batch_index, (inputs, targets, _state) in enumerate(loader):
            row_index = batch_index * world_size + rank
            valid = targets >= 0
            if row_index < len(rows.inputs):
                observed[row_index] = (inputs[0].clone(), targets[0].clone())
            # A deterministic context-sensitive per-target loss fixture.
            prefix = inputs[0].to(torch.float64).cumsum(0)
            safe_targets = torch.where(valid[0], targets[0], 0).to(torch.float64)
            aggregate_loss += float(
                torch.where(valid[0], prefix * 0.01 + safe_targets * 0.001, 0.0)
                .sum()
                .item()
            )
            aggregate_targets += int(valid.sum().item())
    assert sorted(observed) == list(range(len(rows.inputs)))
    for index, (inputs, targets) in observed.items():
        assert torch.equal(inputs, rows.inputs[index])
        assert torch.equal(targets, rows.targets[index])
    reference_prefix = rows.inputs.to(torch.float64).cumsum(1)
    reference_valid = rows.targets >= 0
    reference_targets = torch.where(reference_valid, rows.targets, 0).to(torch.float64)
    reference_loss = float(
        torch.where(
            reference_valid,
            reference_prefix * 0.01 + reference_targets * 0.001,
            0.0,
        )
        .sum()
        .item()
    )
    assert aggregate_targets == rows.target_tokens
    assert aggregate_loss == pytest.approx(reference_loss, rel=0, abs=1e-12)


def _write_production_tokenizer_package(root: Path) -> tuple[Path, dict]:
    root.mkdir()
    (root / "tokenizer.pkl").write_bytes(b"fixture")
    torch.save(torch.ones(32768, dtype=torch.int16), root / "token_bytes.pt")
    parity = {
        "passed": True,
        "upstream_commit": "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd",
        "upstream_iterator_source_sha256": (
            "206d8c89554ceeb4de7afe22e53786806d567e1c4f5493352b2170c0ac174a29"
        ),
        "fixture_documents": ["abcdef", "xy", "1234567", "sonraki"],
        "yielded_documents": ["abcde", "xy", "12345"],
        "requested_max_characters": 10,
        "realized_characters": 12,
        "terminal_overshoot_characters": 2,
    }
    requested = 2_000_000_000
    overshoot = 123
    realized = requested + overshoot
    documents = 250_000
    stop_rule = (
        "yield_full_capped_document_then_stop_when_cumulative_characters_"
        "strictly_exceed_threshold"
    )
    config = {
        "schema_version": "1.0",
        "name": "tr_general_raw_bpe_32k_v1",
        "implementation": "bpe",
        "algorithm": "raw_byte_bpe",
        "vocab_size": 32768,
        "special_tokens": list(SPECIAL_TOKENS),
        "split_pattern": SPLIT_PATTERN,
        "requires_runtime_segmentation": False,
        "max_chars": requested,
        "realized_training_characters": realized,
        "terminal_overshoot_characters": overshoot,
        "stop_rule": stop_rule,
        "doc_cap": 10_000,
        "iterator_stats": {"documents": documents, "characters": realized},
        "nanochat_upstream_revision": "92d63d4e",
        "pinned_iterator_parity": parity,
    }
    write_json_atomic(root / "tokenizer_config.json", config)
    receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "turkish_raw_bpe_training_receipt",
            "name": config["name"],
            "vocab_size": config["vocab_size"],
            "nanochat_upstream_revision": "92d63d4e",
            "training_characters": realized,
            "sample_characters": realized,
            "iterator_characters": realized,
            "requested_max_characters": requested,
            "terminal_overshoot_characters": overshoot,
            "stop_rule": stop_rule,
            "pinned_iterator_parity": parity,
            "training_documents": documents,
            "iterator_documents": documents,
            "max_chars_per_document": 10_000,
            "validation": {
                "exact_vocab_size": config["vocab_size"],
                "all_256_bytes_representable": True,
                "unicode_roundtrip_probes": 8,
            },
            "canonical_sha256": None,
        }
    )
    write_json_atomic(root / "training_receipt.json", receipt)
    roles = {
        "tokenizer.pkl": "tokenizer",
        "tokenizer_config.json": "runtime_config",
        "token_bytes.pt": "token_byte_lengths",
        "training_receipt.json": "training_receipt",
    }
    records = []
    for name in sorted(roles):
        path = root / name
        records.append(
            {
                "path": name,
                "role": roles[name],
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    manifest = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "turkish_raw_bpe_tokenizer_package",
            "training_receipt_sha256": receipt["canonical_sha256"],
            "files": records,
            "canonical_sha256": None,
        }
    )
    path = root / "package_manifest.json"
    write_json_atomic(path, manifest)
    return path, manifest


def test_narrow_tokenizer_verifier_accepts_only_exact_production_package(
    tmp_path: Path,
) -> None:
    path, manifest = _write_production_tokenizer_package(tmp_path / "tokenizer")
    verified = verify_tokenizer_package(
        path,
        expected_sha256=manifest["canonical_sha256"],
        expected_name="tr_general_raw_bpe_32k_v1",
        expected_vocab_size=32768,
    )
    assert verified.canonical_sha256 == manifest["canonical_sha256"]
    (path.parent / "tokenizer.pkl").write_bytes(b"tampered")
    with pytest.raises(TokenizerPackageError, match="inventory"):
        verify_tokenizer_package(path)


def test_recipe_proxy_control_is_bound_to_exact_upstream_schedule() -> None:
    recipe = json.loads(
        Path("configs/pretrain/tr_d32_turkish_general_wsd_v1.json").read_text(
            encoding="utf-8"
        )
    )
    validate_family_recipe(recipe)
    binding = validate_recipe_invocation(
        recipe,
        run_kind="proxy",
        depth=12,
        model_config={
            "sequence_len": 2048,
            "vocab_size": 32768,
            "n_layer": 12,
            "n_head": 6,
            "n_kv_head": 6,
            "n_embd": 768,
            "window_pattern": "L",
        },
        total_parameters=0,
        scaling_parameters=110100912,
        world_size=1,
        device_batch_size=16,
        total_batch_size=524288,
        num_iterations=4200,
        stop_at_step=4200,
        eval_every=100,
        seed=42,
        model_tag="proxy_d12_control",
        lr_schedule="nanochat_linear",
        effective_weight_decay=0.28,
        weight_decay_cooldown_policy="cosine_full_horizon",
        cooldown_start_step=None,
    )
    assert binding["proxy_candidate_id"] == "upstream_92d63d4e_control"
