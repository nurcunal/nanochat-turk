"""Fail-closed operational controls for the Turkish d32 WSD model family.

This module does not submit jobs and does not train models.  It validates the
sealed family recipe and immutable local artifacts, measures the production-
identical 8/16-GPU smokes, authorizes 16-GPU execution only after the declared
speedup gate, and seals checkpoint-lineage receipts after each stage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import io
import json
import math
import os
import re
import shutil
import shlex
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanochat.experiment_manifest import (
    ManifestValidationError,
    canonical_json,
    file_sha256,
    load_json_strict,
    seal_manifest,
    validate_dataset_manifest,
    verify_file_inventory,
    verify_manifest_hash,
    write_json_atomic,
)
from nanochat.strict_runtime import (
    FAMILY_ARTIFACT_CONTRACTS,
    FAMILY_ID_V3,
    StrictTrainingError,
    capacity_authorized_positions,
    capacity_world_gate_record,
    family_artifact_contract,
    validate_production_topology_gate,
)
from nanochat.training_log import read_training_log


DEFAULT_RECIPE = Path("configs/pretrain/tr_d32_turkish_general_wsd_v3.json")
DEFAULT_POLICY = Path("configs/pretrain/tr_d32_turkish_general_v3.json")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

DATA_PREP_STORAGE_SAMPLE_KIND = "d32_data_prep_storage_sample"
DATA_PREP_PACK_PLAN_KIND = "d32_data_prep_production_pack_plan"
DATA_PREP_WRITER_PROBE_KIND = "d32_data_prep_post_cluster_writer_probe"
FINAL_MODEL_PUBLICATION_APPROVAL_KIND = "d32_final_model_publication_approval"
MIXTURE_QUALITY_APPROVAL_KIND = (
    "turkish_bounded_backend_sample_quality_approval"
)
DATA_PREP_STORAGE_COMPONENTS = (
    "source_downloads",
    "filtered_text",
    "minhash_signatures",
    "minhash_buckets",
    "cluster_assignments",
    "tokenized_output",
    "temporary_merge_space",
)
DATA_PREP_FUTURE_CPU_COMPONENTS = (
    "production_backend",
    "production_pool_materialization",
    "tokenizer_sample",
    "tokenizer_training",
    "tokenizer_quality",
    "packing_preflight",
    "final_corpus_materialization_and_capacity",
)
CPU2DQ_BILLABLE_CPUS = 128
PRODUCTION_WORKERS_PER_NODE = 32
PRODUCTION_CPUS_PER_WORKER = 4
DATA_PREP_FIXED_CPU2DQ_CEILINGS = {
    "tokenizer_sample": 12 * CPU2DQ_BILLABLE_CPUS,
    "tokenizer_training": 24 * CPU2DQ_BILLABLE_CPUS,
    "tokenizer_quality": 12 * CPU2DQ_BILLABLE_CPUS,
    "packing_preflight": 12 * CPU2DQ_BILLABLE_CPUS,
    "final_corpus_materialization_and_capacity": 48 * CPU2DQ_BILLABLE_CPUS,
}
SLURM_ALLOCATION_ID_RE = re.compile(r"^[0-9]+(?:_[0-9]+)?$")


class FamilyWorkflowError(ValueError):
    """Raised when an operational prerequisite is missing or inconsistent."""


def _fail(message: str) -> None:
    raise FamilyWorkflowError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, list):
        _fail(f"{name} must be an array")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{name} must be a non-negative integer")
    return value


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        _fail(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{name} must be a positive finite number")
    return result


def _nonnegative_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        _fail(f"{name} must be a non-negative finite number")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{name} must be a non-negative finite number")
    return result


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        _fail(f"{name} must be a lowercase SHA-256")
    return value


def _git_commit(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", value) is None:
        _fail(f"{name} must be a full lowercase Git object ID")
    return value


def _safe_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or SAFE_ID_RE.fullmatch(value) is None:
        _fail(f"{name} must match {SAFE_ID_RE.pattern}")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        _fail(
            f"{name} keys drifted; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = load_json_strict(path)
    except (ManifestValidationError, OSError, ValueError) as exc:
        raise FamilyWorkflowError(f"cannot load {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{label} must contain a JSON object: {path}")
    return value


def _verify_sealed(path: Path, label: str, expected_sha256: str | None = None) -> tuple[dict[str, Any], str]:
    value = _load_object(path, label)
    try:
        actual = verify_manifest_hash(value)
    except ValueError as exc:
        raise FamilyWorkflowError(f"{label} is not correctly self-hashed: {path}: {exc}") from exc
    if expected_sha256 is not None and not hmac.compare_digest(
        actual, _sha256(expected_sha256, f"expected {label} SHA-256")
    ):
        _fail(f"{label} SHA-256 mismatch: expected {expected_sha256}, found {actual}")
    return value, actual


def validate_recipe(recipe: Mapping[str, Any]) -> None:
    """Check the exact agreed d32 family invariants, not just JSON shape."""

    if recipe.get("schema_version") != "1.0":
        _fail("recipe schema_version must equal '1.0'")
    family_id = recipe.get("family_id")
    if family_id not in FAMILY_ARTIFACT_CONTRACTS:
        _fail("unexpected family_id")

    code = _mapping(recipe.get("code_provenance"), "code_provenance")
    _git_commit(code.get("upstream_base_revision"), "code_provenance.upstream_base_revision")
    if code.get("require_clean_git") is not True:
        _fail("code provenance must require a clean Git worktree")
    if code.get("require_upstream_base_ancestor") is not True:
        _fail("code provenance must require the pinned upstream base as an ancestor")
    scope = _sequence(code.get("core_scope"), "core_scope")
    exact_files = _mapping(code.get("exact_file_sha256"), "exact_file_sha256")
    expected_exact_files = {
        "nanochat/gpt.py": "8cdedc2e418b9d4dc45884e07f59904fcdec066bb3d76664b2adf2d415f704e3",
        "nanochat/optim.py": "7c9b714c6f76e3a9d50fc530044b2767d0cd89391237e609548f1cd730afed8f",
        "nanochat/flash_attention.py": "a10b0b10ad91893fcf9c9b19c96bd3468a2721cf156dcdbeeeecf05f97f22c2f",
        "nanochat/engine.py": "76636fc24228e72afdafe4747ff8e52a6cc60a5ec61c97205061f3bfed0a86fd",
        "nanochat/common.py": "b843db6e769db18d3364e42cb3742a9537a3d4cf43e5f098293babc7ddeb44bf",
        "nanochat/checkpoint_manager.py": "aa63399ddaf2a0412dc099287bf0dc879f9cb0cce00f861db50eeec58746f3ba",
        "nanochat/dataloader.py": "5cc72d7207931f112d685ba8e04c112e1a4ab7756dbbb29b95bdb4908a21864d",
        "nanochat/dataset.py": "3ec2ad5987875e3d7f3ec72f08d78089fe8a3b20090448b798c21c4efb93d18a",
        "nanochat/loss_eval.py": "00faad1e0ae8912022f79ee4bf583c4f9b4c058e4523c5674144648c49229fd6",
        "nanochat/tokenizer.py": "e6f12cb88fead3e6d0d45a7dd2af4192280712d92bfde83e2399cc4b3a0860fb",
        "scripts/base_train.py": "87395b8078491d5088e7e8100866162b494638097215759b46d8ceb5d2f0fdcc",
    }
    if exact_files != expected_exact_files:
        _fail("exact training-core file hashes drifted from the reviewed revision")
    if set(scope) != set(exact_files):
        _fail("core_scope must exactly equal the immutable upstream file-hash set")
    if "core_patch_allowlist" in code:
        _fail("broad core patch allowlists are forbidden; strict extensions must be additive")
    expected_training_environment = {
        "manager": "uv",
        "uv_version": "0.11.29",
        "python_version": "3.12.4",
        "uhem_python_module": "Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0",
        "sync_mode": "uv_sync_frozen_extra_gpu",
        "sync_command": "uv sync --frozen --extra gpu --python $(command -v python3)",
        "pyproject_sha256": "c91fdd03ae9705565572eee31924d4c0bca24bf5431a8eabff4c061882f94929",
        "uv_lock_sha256": "de7891b832854162111208644ddb72685069ce8128e2bed9dbb7993aa6af5861",
        "relative_exclude_newer_lock_requires_frozen": True,
    }
    if code.get("training_environment") != expected_training_environment:
        _fail("training environment drifted from the pinned upstream frozen lock")

    language = _mapping(recipe.get("language_policy"), "language_policy")
    if language.get("allowed_languages") != ["tr"]:
        _fail("the family recipe must allow Turkish only")
    for forbidden in (
        "allow_code_corpora",
        "allow_synthetic_instruction_data",
        "allow_translated_filler",
    ):
        if language.get(forbidden) is not False:
            _fail(f"language_policy.{forbidden} must be false")

    artifacts = _mapping(recipe.get("artifacts"), "artifacts")
    if dict(artifacts) != family_artifact_contract(str(family_id)):
        _fail("artifacts differ from the exact per-family contract")

    model = _mapping(recipe.get("model"), "model")
    _require_exact_keys(
        model,
        {
            "depth",
            "aspect_ratio",
            "model_dim",
            "head_dim",
            "num_heads",
            "num_kv_heads",
            "max_seq_len",
            "window_pattern",
            "attention_backend",
            "vocab_size",
            "total_parameters",
            "scaling_parameters",
            "parameter_ratio_convention",
        },
        "model",
    )
    expected_model = {
        "depth": 32,
        "aspect_ratio": 64,
        "model_dim": 2048,
        "head_dim": 128,
        "num_heads": 16,
        "num_kv_heads": 16,
        "max_seq_len": 2048,
        "window_pattern": "selected_by_a100_probe",
        "attention_backend": "selected_by_a100_probe",
        "vocab_size": 32768,
        "total_parameters": 2818575450,
        "scaling_parameters": 1677724672,
        "parameter_ratio_convention": "nanochat_scaling",
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            _fail(f"model.{key} must equal {expected!r}")

    training = _mapping(recipe.get("training"), "training")
    _require_exact_keys(
        training,
        {
            "optimizer",
            "precision",
            "fp8_enabled",
            "global_batch_tokens",
            "device_batch_sequences",
            "warmup_steps",
            "gradient_clip_norm",
            "lr_schedule",
            "cooldown_fraction",
            "cooldown_final_lr_fraction",
            "optimizer_hyperparameters",
            "seed",
            "data_order",
            "target_param_count",
            "target_param_data_ratio",
            "evaluation",
            "strict_transactional_checkpoints",
            "fixed_validation",
            "save_every_steps",
        },
        "training",
    )
    exact_training = {
        "optimizer": "muon_adamw",
        "precision": "bfloat16",
        "fp8_enabled": False,
        "global_batch_tokens": 2_097_152,
        "device_batch_sequences": 4,
        "warmup_steps": 40,
        "gradient_clip_norm": 0.0,
        "lr_schedule": "wsd",
        "cooldown_fraction": 0.1,
        "cooldown_final_lr_fraction": 0.0,
        "seed": 42,
        "data_order": "bestfit",
        "target_param_count": "scaling",
        "target_param_data_ratio": -1.0,
        "strict_transactional_checkpoints": True,
        "fixed_validation": True,
        "save_every_steps": -1,
    }
    for key, expected in exact_training.items():
        if training.get(key) != expected:
            _fail(f"training.{key} must equal {expected!r}")
    evaluation = _mapping(training.get("evaluation"), "training.evaluation")
    expected_evaluation = {
        "eval_every_updates": 250,
        "eval_tokens_cli_unused": -1,
        "fixed_validation_full_manifest": True,
        "core_metric_every_updates": -1,
        "core_metric_max_per_task": 500,
        "sample_every_updates": -1,
    }
    if evaluation != expected_evaluation:
        _fail("training.evaluation drifted from the frozen full-manifest validation policy")
    optimizer = _mapping(
        training.get("optimizer_hyperparameters"),
        "training.optimizer_hyperparameters",
    )
    _require_exact_keys(
        optimizer,
        {
            "embedding_lr",
            "unembedding_lr",
            "matrix_lr",
            "scalar_lr",
            "global_batch_lr_multiplier",
            "adam_model_width_multiplier",
            "derived_initial_group_lrs",
            "stable_weight_decay",
            "weight_decay_derivation",
            "weight_decay_status",
            "momentum_warmup_start",
            "momentum_warmup_end",
            "momentum_warmup_steps",
            "stable_muon_momentum",
            "cooldown_final_muon_momentum",
            "weight_decay_cooldown_policy",
        },
        "training.optimizer_hyperparameters",
    )
    exact_optimizer = {
        "embedding_lr": 0.3,
        "unembedding_lr": 0.008,
        "matrix_lr": 0.02,
        "scalar_lr": 0.5,
        "global_batch_lr_multiplier": 2.0,
        "adam_model_width_multiplier": 0.6123724356957945,
        "stable_weight_decay": 0.03675007690415606,
        "weight_decay_derivation": "legacy_d12_update_normalized_proxy_for_d32_scaling_params_and_2097152_token_batch",
        "weight_decay_status": "proxy_ablation_required_before_production",
        "momentum_warmup_start": 0.85,
        "momentum_warmup_end": 0.97,
        "momentum_warmup_steps": 400,
        "stable_muon_momentum": 0.97,
        "cooldown_final_muon_momentum": 0.9,
        "weight_decay_cooldown_policy": "selected_by_proxy_acceptance",
    }
    for key, expected in exact_optimizer.items():
        if optimizer.get(key) != expected:
            _fail(f"optimizer_hyperparameters.{key} must equal {expected!r}")
    exact_group_lrs = {
        "lm_head": 0.009797959,
        "token_embedding": 0.367423461,
        "value_embeddings": 0.1837117305,
        "residual_scalars": 0.01,
        "x0_scalars": 1.0,
        "smear": 0.2,
        "muon_matrices": 0.04,
    }
    if optimizer.get("derived_initial_group_lrs") != exact_group_lrs:
        _fail("optimizer_hyperparameters.derived_initial_group_lrs drifted")

    checkpoints = _mapping(recipe.get("checkpoints"), "checkpoints")
    forks = _sequence(checkpoints.get("stable_forks"), "checkpoints.stable_forks")
    finals = _sequence(checkpoints.get("finals"), "checkpoints.finals")
    if checkpoints.get("trunk_model_tag") != f"{family_id}_trunk":
        _fail("checkpoints.trunk_model_tag drifted")
    expected_forks = (
        {
            "scale": 10.8,
            "step": 8640,
            "scheduled_tokens": 8640 * training["global_batch_tokens"],
            "retention": "full_resumable",
        },
        {
            "scale": 18.0,
            "step": 14400,
            "scheduled_tokens": 14400 * training["global_batch_tokens"],
            "retention": "full_resumable",
        },
        {
            "scale": 36.0,
            "step": 28800,
            "scheduled_tokens": 28800 * training["global_batch_tokens"],
            "retention": "full_resumable",
        },
    )
    final_retention = (
        "full_resumable"
        if family_id == FAMILY_ID_V3
        else "model_metadata_provenance_required_optimizer_explicit"
    )
    expected_finals = (
        {
            "label": "s12",
            "model_tag": f"{family_id}_s12",
            "parent_step": 8640,
            "cooldown_start_step": 8640,
            "final_step": 9600,
            "cooldown_steps": 960,
            "scheduled_tokens": 9600 * training["global_batch_tokens"],
            "retention": final_retention,
        },
        {
            "label": "s20",
            "model_tag": f"{family_id}_s20",
            "parent_step": 14400,
            "cooldown_start_step": 14400,
            "final_step": 16000,
            "cooldown_steps": 1600,
            "scheduled_tokens": 16000 * training["global_batch_tokens"],
            "retention": final_retention,
        },
        {
            "label": "s40",
            "model_tag": f"{family_id}_s40",
            "parent_step": 28800,
            "cooldown_start_step": 28800,
            "final_step": 32000,
            "cooldown_steps": 3200,
            "scheduled_tokens": 32000 * training["global_batch_tokens"],
            "retention": final_retention,
        },
    )
    if len(forks) != len(expected_forks) or len(finals) != len(expected_finals):
        _fail("the recipe must contain exactly three stable forks and three finals")
    for index, expected in enumerate(expected_forks):
        fork = _mapping(forks[index], f"stable_forks[{index}]")
        if dict(fork) != expected:
            _fail(f"stable_forks[{index}] drifted from the reviewed boundary")
    for index, expected in enumerate(expected_finals):
        final = _mapping(finals[index], f"finals[{index}]")
        if dict(final) != expected:
            _fail(f"finals[{index}] drifted from the reviewed cooldown boundary")

    stages = _sequence(recipe.get("stages"), "stages")
    expected_stages = (
        {
            "id": "trunk_to_s12_fork",
            "kind": "trunk",
            "model_tag": f"{family_id}_trunk",
            "exposure_plan_family": "trunk",
            "source_step": None,
            "target_step": 8640,
            "num_iterations": 28800,
            "cooldown_start_step": -1,
        },
        {
            "id": "s12_cooldown",
            "kind": "cooldown_fork",
            "model_tag": f"{family_id}_s12",
            "exposure_plan_family": "s12",
            "source_model_tag": f"{family_id}_trunk",
            "source_step": 8640,
            "target_step": 9600,
            "num_iterations": 9600,
            "cooldown_start_step": 8640,
        },
        {
            "id": "trunk_to_s20_fork",
            "kind": "trunk",
            "model_tag": f"{family_id}_trunk",
            "exposure_plan_family": "trunk",
            "source_step": 8640,
            "target_step": 14400,
            "num_iterations": 28800,
            "cooldown_start_step": -1,
        },
        {
            "id": "s20_cooldown",
            "kind": "cooldown_fork",
            "model_tag": f"{family_id}_s20",
            "exposure_plan_family": "s20",
            "source_model_tag": f"{family_id}_trunk",
            "source_step": 14400,
            "target_step": 16000,
            "num_iterations": 16000,
            "cooldown_start_step": 14400,
        },
        {
            "id": "trunk_to_s40_fork",
            "kind": "trunk",
            "model_tag": f"{family_id}_trunk",
            "exposure_plan_family": "trunk",
            "source_step": 14400,
            "target_step": 28800,
            "num_iterations": 28800,
            "cooldown_start_step": -1,
        },
        {
            "id": "s40_cooldown",
            "kind": "cooldown_fork",
            "model_tag": f"{family_id}_s40",
            "exposure_plan_family": "s40",
            "source_model_tag": f"{family_id}_trunk",
            "source_step": 28800,
            "target_step": 32000,
            "num_iterations": 32000,
            "cooldown_start_step": 28800,
        },
    )
    if len(stages) != len(expected_stages):
        _fail("stages must contain exactly the six reviewed lineage stages")
    for index, expected in enumerate(expected_stages):
        stage = _mapping(stages[index], f"stages[{index}]")
        if dict(stage) != expected:
            _fail(f"stages[{index}] drifted from the reviewed sequential lineage")

    gate = _mapping(recipe.get("distributed_gate"), "distributed_gate")
    if gate.get("smoke_node_order") != [2, 4]:
        _fail("smoke_node_order must be [2, 4]")
    if gate.get("minimum_8_to_16_gpu_speedup") != 1.7:
        _fail("minimum 8-to-16-GPU speedup must equal 1.7")
    expected_topology_gate = {
        "partition": "a100x4q",
        "gpus_per_node": 4,
        "static_launcher": "slurm_srun_direct_python_env_v1",
        "static_launcher_probe_nodes": 1,
        "static_launcher_probe_world_size": 4,
        "require_clean_static_launcher_probe_before_distributed_smokes": True,
        "signal_resume_probe_world_size": 4,
        "signal_resume_probe_updates": 6,
        "signal_resume_probe_after_completed_update": 1,
        "require_signal_resume_gate_before_distributed_smokes": True,
        "smoke_node_order": [2, 4],
        "smoke_updates": 100,
        "forced_resume_step": 50,
        "benchmark_first_update": 21,
        "benchmark_last_update": 100,
        "maximum_aggregate_loader_fraction": 0.35,
        "maximum_p95_loader_fraction": 0.6,
        "minimum_8_to_16_gpu_speedup": 1.7,
        "minimum_parallel_efficiency": 0.85,
        "preferred_production_nodes": 4,
        "preferred_production_world_size": 16,
        "fallback_production_nodes": 2,
        "fallback_production_world_size": 8,
        "selection_policy": (
            "use_16_only_when_clean_8_and_16_gpu_smokes_exist_and_speedup_is_at_least_1.7_"
            "otherwise_use_8"
        ),
        "require_single_world_size_for_entire_lineage": True,
    }
    if gate != expected_topology_gate:
        _fail("distributed topology gate drifted from the reviewed 8-GPU fallback policy")

    attention_gate = _mapping(
        recipe.get("attention_backend_gate"), "attention_backend_gate"
    )
    expected_attention_gate = {
        "required_before_proxy_and_production": True,
        "probe_world_size": 1,
        "required_gpu_family": "A100",
        "selection_policy": (
            "use_pinned_upstream_auto_fa3_with_SSSL_after_actual_d32_bf16_finite_smoke_"
            "else_sdpa_with_L"
        ),
        "preferred_backend": "fa3",
        "preferred_window_pattern": "SSSL",
        "fallback_backend": "sdpa",
        "fallback_window_pattern": "L",
        "require_forward_backward_finite": True,
        "require_upstream_auto_detection_for_fa3": True,
        "require_actual_d32_model_forward_backward": True,
        "diagnostic_fa3_sdpa_comparison_is_non_decisional": True,
        "fallback_reason_required": True,
        "benchmark_both_window_patterns": True,
        "record_kernel_cache_inventory": True,
    }
    if attention_gate != expected_attention_gate:
        _fail("attention backend gate drifted from the reviewed SDPA-on-A100 policy")

    proxy = _mapping(
        recipe.get("weight_decay_proxy_ablation"),
        "weight_decay_proxy_ablation",
    )
    if proxy.get("required_before_production") is not True:
        _fail("weight-decay proxy acceptance must be required before production")
    if proxy.get("recipe_version") != "tr_d32_wsd_wd_proxy_v1":
        _fail("unexpected weight-decay proxy recipe version")
    if proxy.get("weight_decay_transfer_rule") != "nanochat_width_batch_v1":
        _fail("unexpected weight-decay scale-transfer rule")
    if proxy.get("production_scaling_parameters") != 1_677_724_672:
        _fail("proxy production scaling-parameter binding drifted")
    if proxy.get("production_global_batch_tokens") != 2_097_152:
        _fail("proxy production batch binding drifted")
    if proxy.get("two_stage_transfer_gate") is not True:
        _fail("weight-decay proxy must use the d12-to-d20 transfer gate")
    screen = _mapping(proxy.get("screen_stage"), "weight_decay_proxy_ablation.screen_stage")
    confirmation = _mapping(
        proxy.get("confirmation_stage"),
        "weight_decay_proxy_ablation.confirmation_stage",
    )
    expected_screen = {
        "model_depth": 12,
        "model_dim": 768,
        "world_size": 1,
        "device_batch_sequences": 16,
        "global_batch_tokens": 524_288,
        "target_scaling_ratio": 20,
        "scaling_parameters": 110_100_912,
        "scheduled_tokens": 2_202_009_600,
        "updates": 4200,
        "validation_every_updates": 100,
        "eval_tokens_cli_unused": -1,
        "fixed_validation_full_manifest": True,
        "final_validation_points": 5,
        "seeds": [42, 314159],
        "advance_top_wsd_recipes": 2,
    }
    for key, expected in expected_screen.items():
        if screen.get(key) != expected:
            _fail(f"weight_decay_proxy_ablation.screen_stage.{key} must equal {expected!r}")
    expected_confirmation = {
        "model_depth": 20,
        "model_dim": 1280,
        "world_size": 1,
        "device_batch_sequences": 4,
        "global_batch_tokens": 1_048_576,
        "target_scaling_ratio": 12,
        "scaling_parameters": 435_160_240,
        "scheduled_tokens": 5_221_908_480,
        "updates": 4980,
        "validation_every_updates": 100,
        "eval_tokens_cli_unused": -1,
        "fixed_validation_full_manifest": True,
        "final_validation_points": 5,
        "seeds": [42],
        "arms": "pinned_upstream_control_plus_two_wsd_screen_winners",
    }
    for key, expected in expected_confirmation.items():
        if confirmation.get(key) != expected:
            _fail(
                f"weight_decay_proxy_ablation.confirmation_stage.{key} "
                f"must equal {expected!r}"
            )
    candidates = _sequence(proxy.get("candidates"), "weight_decay_proxy_ablation.candidates")
    expected_candidates = {
        "upstream_92d63d4e_control": (0.03675007690415606, 1.0, "nanochat_linear", "cosine_full_horizon", False),
        "legacy_transferred": (0.03675007690415606, 1.0, "wsd", "linear_to_zero", True),
        "half_transferred_constant_cooldown_wd": (0.01837503845207803, 0.5, "wsd", "constant", True),
        "half_transferred_linear_cooldown_wd": (0.01837503845207803, 0.5, "wsd", "linear_to_zero", True),
        "no_weight_decay": (0.0, 0.0, "wsd", "linear_to_zero", True),
    }
    candidate_map = {
        candidate.get("id"): candidate
        for candidate in candidates
        if isinstance(candidate, Mapping)
    }
    if set(candidate_map) != set(expected_candidates):
        _fail("weight-decay proxy candidate ID set drifted")
    stage_specs = {"d12": screen, "d20": confirmation}
    for candidate_id, expected in expected_candidates.items():
        candidate = _mapping(candidate_map[candidate_id], f"candidate {candidate_id}")
        production_wd, ratio, schedule_name, cooldown_policy, eligible = expected
        expected_fields = {
            "production_base_weight_decay": production_wd,
            "proxy_ratio_to_upstream_effective_weight_decay": ratio,
            "proxy_scale_rule": "nanochat_width_batch_v1",
            "schedule": schedule_name,
            "cooldown_weight_decay": cooldown_policy,
            "eligible_for_production": eligible,
        }
        for key, value in expected_fields.items():
            if candidate.get(key) != value:
                _fail(f"candidate {candidate_id}.{key} must equal {value!r}")
        if candidate_id == "upstream_92d63d4e_control":
            if candidate.get("input_weight_decay") != 0.28:
                _fail("upstream proxy control must bind input_weight_decay=0.28")
        elif "input_weight_decay" in candidate:
            _fail(f"WSD candidate {candidate_id} must not use legacy input_weight_decay")
        stage_values = _mapping(
            candidate.get("stage_effective_weight_decay"),
            f"candidate {candidate_id}.stage_effective_weight_decay",
        )
        if set(stage_values) != set(stage_specs):
            _fail(f"candidate {candidate_id} stage-effective WD keys drifted")
        for stage_id, stage_spec in stage_specs.items():
            derived = production_wd * math.sqrt(
                stage_spec["global_batch_tokens"]
                / proxy["production_global_batch_tokens"]
            ) * (
                proxy["production_scaling_parameters"]
                / stage_spec["scaling_parameters"]
            )
            if not math.isclose(
                float(stage_values[stage_id]), derived, rel_tol=0.0, abs_tol=1e-15
            ):
                _fail(
                    f"candidate {candidate_id} {stage_id} effective WD is not "
                    "the exact nanochat_width_batch_v1 transfer"
                )
    acceptance = _mapping(proxy.get("acceptance_rule"), "proxy acceptance_rule")
    if acceptance.get("maximum_wsd_vs_upstream_control_bpb") != 0.002:
        _fail("proxy acceptance must reject WSD recipes materially worse than upstream")

    storage = _mapping(recipe.get("storage"), "storage")
    if storage.get("never_auto_delete_existing_artifacts") is not True:
        _fail("the storage policy must never auto-delete existing artifacts")
    expected_live_storage = {
        "uid": 4500,
        "storage_pool_id": 1,
        "query": "beegfs-ctl_--getquota_--uid_4500_--storagepoolid=1_--csv",
        "allow_unlimited_hard_quota": True,
        "effective_free_policy": "minimum_of_physical_free_and_finite_user_quota_remaining",
    }
    if storage.get("uhem_live_quota") != expected_live_storage:
        _fail("UHeM live storage quota policy drifted")
    data_prep_gate = _mapping(
        storage.get("data_preparation_peak_gate"),
        "storage.data_preparation_peak_gate",
    )
    expected_data_prep_gate = {
        "required": True,
        "minimum_sample_documents": 100_000,
        "extrapolation_safety_factor": 1.35,
        "required_measured_components": [
            "source_downloads",
            "filtered_text",
            "minhash_signatures",
            "minhash_buckets",
            "cluster_assignments",
            "tokenized_output",
            "temporary_merge_space",
        ],
        "allow_verified_uhem_scratch": True,
        "never_assume_intermediate_bytes_are_negligible": True,
        "require_billed_cpu_sample_extrapolation": True,
        "never_auto_delete_existing_artifacts": True,
    }
    if data_prep_gate != expected_data_prep_gate:
        _fail("data-preparation storage gate drifted")
    if storage.get("estimated_cooled_final_model_bundle_bytes") != 12 * 1024**3:
        _fail("cooled-final estimate must leave room above raw d32 model weights")
    if family_id == FAMILY_ID_V3 and (
        storage.get("cooled_final_model_bundles_retained") != 0
        or storage.get("full_cooled_final_transactions_at_peak") != 3
    ):
        _fail("v3 storage must retain all three cooled finals as full transactions")
    if storage.get("smoke_measurement_safety_factor") != 1.25:
        _fail("smoke checkpoint storage safety factor must equal 1.25")
    required = _positive_int(
        storage.get("required_free_bytes_at_training_preflight"),
        "storage.required_free_bytes_at_training_preflight",
    )
    recomputed = (
        _positive_int(
            storage.get("estimated_full_resumable_transaction_bytes"),
            "estimated_full_resumable_transaction_bytes",
        )
        * (
            _positive_int(
                storage.get("full_resumable_stable_forks_retained"),
                "full_resumable_stable_forks_retained",
            )
            + _positive_int(
                storage.get("full_cooled_final_transactions_at_peak"),
                "full_cooled_final_transactions_at_peak",
            )
            + _positive_int(
                storage.get("atomic_write_transient_transactions"),
                "atomic_write_transient_transactions",
            )
            + _positive_int(
                storage.get("maximum_retained_preemption_transactions"),
                "maximum_retained_preemption_transactions",
            )
        )
        + _positive_int(
            storage.get("estimated_cooled_final_model_bundle_bytes"),
            "estimated_cooled_final_model_bundle_bytes",
        )
        * _nonnegative_int(
            storage.get("cooled_final_model_bundles_retained"),
            "cooled_final_model_bundles_retained",
        )
        + _positive_int(storage.get("estimated_logs_and_evaluations_bytes"), "estimated_logs_and_evaluations_bytes")
        + _positive_int(
            storage.get("estimated_proxy_and_smoke_retained_bytes"),
            "estimated_proxy_and_smoke_retained_bytes",
        )
        + _positive_int(storage.get("minimum_free_headroom_bytes"), "minimum_free_headroom_bytes")
    )
    if required != recomputed:
        _fail("storage.required_free_bytes_at_training_preflight arithmetic mismatch")
    end_to_end = (
        required
        + _positive_int(storage.get("estimated_corpus_bytes"), "estimated_corpus_bytes")
        + _positive_int(
            storage.get("estimated_tokenizer_and_receipts_bytes"),
            "estimated_tokenizer_and_receipts_bytes",
        )
    )
    if storage.get("estimated_end_to_end_peak_project_bytes") != end_to_end:
        _fail("storage.estimated_end_to_end_peak_project_bytes arithmetic mismatch")

    budget = _mapping(recipe.get("uhem_budget"), "uhem_budget")
    if budget.get("account") != "nakane" or budget.get("user") != "nunal":
        _fail("UHeM account/user binding drifted")
    if budget.get("quota_query") != (
        "sshare_-n_-P_-A_nakane_-u_nunal_-o_"
        "Account,User,GrpTRESMins,GrpTRESRaw,TRESRunMins,RawUsage"
    ):
        _fail("UHeM quota query must read limit, raw use, and running reservation")
    expected_rates = {
        "cpu_saat_per_4gpu_node_hour": 64,
        "cpu_saat_per_fully_utilized_a100_gpu_hour": 16,
        "cpu_saat_per_4node_wall_hour": 256,
    }
    for key, expected in expected_rates.items():
        if budget.get(key) != expected:
            _fail(f"uhem_budget.{key} must equal {expected}")
    exact_physical_tokens = 34_560 * 2_097_152
    if budget.get("shared_lineage_scheduled_token_work") != exact_physical_tokens:
        _fail("UHeM cost must be based on exact batch-aligned shared-lineage work")
    exact_scaling_work = exact_physical_tokens / 1_677_724_672
    if abs(float(budget.get("shared_lineage_scaling_work", -1)) - exact_scaling_work) > 1e-12:
        _fail("shared-lineage scaling work arithmetic mismatch")
    calibration = _mapping(budget.get("calibration"), "uhem_budget.calibration")
    expected_flops = {"d20_flops_per_token": 3_240_107_184, "d32_flops_per_token": 11_676_960_912}
    for key, expected in expected_flops.items():
        if calibration.get(key) != expected:
            _fail(f"uhem_budget.calibration.{key} must equal {expected}")
    flops_ratio = expected_flops["d32_flops_per_token"] / expected_flops["d20_flops_per_token"]
    if abs(float(calibration.get("d32_to_d20_scaling_compute_ratio", -1)) - flops_ratio) > 1e-9:
        _fail("d32/d20 FLOP-per-token calibration ratio mismatch")
    projected = float(calibration.get("source_sustained_tokens_per_second", -1)) / flops_ratio
    if abs(float(calibration.get("projected_d32_8gpu_tokens_per_second", -1)) - projected) > 1:
        _fail("projected d32 throughput is inconsistent with exact FLOP/token ratio")
    proxy_cost = _mapping(budget.get("proxy_cost_model"), "uhem_budget.proxy_cost_model")
    expected_proxy_cost = {
        "allocation": "one_4xa100_node_with_up_to_four_concurrent_single_gpu_arms",
        "partition": "a100x4q",
        "allocated_a100_gpus": 4,
        "screen_packed_waves": 3,
        "confirmation_packed_waves": 1,
        "forbid_single_gpu_partition_fallback": True,
        "d12_flops_per_token": 887_098_032,
        "screen_runs": 10,
        "screen_aggregate_tokens": 22_020_096_000,
        "screen_projected_raw_cpu_saat": 621,
        "confirmation_runs": 3,
        "confirmation_aggregate_tokens": 15_665_725_440,
        "confirmation_projected_raw_cpu_saat": 1794,
        "full_manifest_evaluation_overhead_allowance_cpu_saat": 1000,
        "kernel_and_distributed_smokes_projected_raw_cpu_saat": 200,
        "estimated_total_cpu_saat_range_with_queue_runtime_overhead": [3000, 4000],
        "single_gpu_partition_fallback_estimate_cpu_saat": 7456,
        "require_four_way_packing": True,
    }
    if proxy_cost != expected_proxy_cost:
        _fail("proxy/smoke cost model drifted from the reviewed four-way-packed estimate")
    gpu_hour_range = budget.get("raw_estimated_aggregate_a100_hours_range")
    cpu_range = budget.get("raw_estimated_cpu_saat_range")
    planned_range = budget.get("estimated_cpu_saat_range_with_15_percent_reserve")
    for name, value in (
        ("raw_estimated_aggregate_a100_hours_range", gpu_hour_range),
        ("raw_estimated_cpu_saat_range", cpu_range),
        ("estimated_cpu_saat_range_with_15_percent_reserve", planned_range),
    ):
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
            or value[0] <= 0
            or value[1] < value[0]
        ):
            _fail(f"uhem_budget.{name} must be an increasing positive two-value range")
    throughput_range = calibration.get("projected_d32_8gpu_tokens_per_second_range")
    if throughput_range != [97_300, 131_600]:
        _fail("d32 ws8 throughput envelope drifted from the reviewed calibration")
    shared_positions = int(budget["shared_lineage_scheduled_token_work"])
    minimum_speedup = float(
        recipe["distributed_gate"]["minimum_8_to_16_gpu_speedup"]
    )
    expected_gpu_hours = [
        math.ceil(shared_positions / throughput_range[1] / 3600 * 8),
        math.ceil(
            shared_positions
            / (throughput_range[0] * minimum_speedup)
            / 3600
            * 16
        ),
    ]
    expected_cpu_saat = [value * 16 for value in expected_gpu_hours]
    expected_reserved_cpu_saat = [
        math.ceil(value * 1.15) for value in expected_cpu_saat
    ]
    if gpu_hour_range != expected_gpu_hours:
        _fail("aggregate A100-hour envelope is not derived from tokens/throughput")
    if cpu_range != expected_cpu_saat:
        _fail("CPU-saat envelope is not exact aggregate A100-hours times 16")
    if planned_range != expected_reserved_cpu_saat:
        _fail("reserved CPU-saat envelope is not the exact 15% rounded-up plan")
    for index in range(2):
        if abs(float(gpu_hour_range[index]) * 16 - float(cpu_range[index])) > 20:
            _fail("UHeM CPU-saat range must equal aggregate A100-hour range times 16")
        if float(planned_range[index]) < float(cpu_range[index]) * 1.15 - 1:
            _fail("planned UHeM CPU-saat range does not include a full 15% reserve")
    ceiling = int(budget.get("operational_ceiling_cpu_saat", -1))
    expected_ceiling_scope = (
        "training_plus_weight_decay_proxy_plus_kernel_and_distributed_smokes_only; "
        "data_preparation_is_separately_sample_extrapolated_and_added"
    )
    if budget.get("operational_ceiling_scope") != expected_ceiling_scope:
        _fail("UHeM operational ceiling scope must explicitly exclude data preparation")
    if ceiling < float(planned_range[1]) + int(budget.get("proxy_and_smoke_reserve_cpu_saat", -1)):
        _fail("operational CPU-saat ceiling does not cover training, proxy, and smokes")
    if family_id == FAMILY_ID_V3:
        publication = _mapping(recipe.get("publication"), "publication")
        if publication.get("require_manual_final_quality_approval") is not True:
            _fail("v3 publication must require manual final-model quality approval")


def load_recipe(path: Path, *, require_sealed: bool = True) -> tuple[dict[str, Any], str]:
    recipe = _load_object(path, "family recipe")
    validate_recipe(recipe)
    try:
        digest = verify_manifest_hash(recipe, required=require_sealed)
    except ValueError as exc:
        raise FamilyWorkflowError(f"family recipe hash is invalid: {exc}") from exc
    return recipe, digest


def _git_output(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=repo_root, text=True, stderr=subprocess.STDOUT
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise FamilyWorkflowError(f"git {' '.join(args)} failed: {exc.output.strip()}") from exc


def _manifest_inventory_if_present(root: Path, manifest: Mapping[str, Any], label: str) -> None:
    records = manifest.get("ordered_files")
    location = "ordered_files"
    if records is None:
        records = manifest.get("files")
        location = "files"
    if records is None:
        return
    if not isinstance(records, list) or not records:
        _fail(f"{label}.{location} must be a non-empty array")
    require_role = all(isinstance(record, Mapping) and "role" in record for record in records)
    try:
        verify_file_inventory(
            root,
            records,
            require_exact=False,
            require_role=require_role,
            location=location,
        )
    except (ManifestValidationError, OSError, ValueError) as exc:
        raise FamilyWorkflowError(f"{label} file inventory failed: {exc}") from exc


def _directory_size(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            _fail(f"artifact directories must not contain symlinks: {path}")
        if path.is_file():
            total += path.stat().st_size
    return total


def _checkpoint_storage_observation(
    manifest: Mapping[str, Any], step_dir: Path
) -> dict[str, int]:
    records = _sequence(manifest.get("files"), "strict checkpoint files")
    declared_payload = 0
    model_bundle = 0
    for index, value in enumerate(records):
        record = _mapping(value, f"strict checkpoint files[{index}]")
        size = _positive_int(record.get("size_bytes"), f"strict checkpoint files[{index}].size_bytes")
        declared_payload += size
        if record.get("role") in {"model", "meta"}:
            model_bundle += size
    completion = step_dir / "completion.json"
    if not completion.is_file() or completion.is_symlink():
        _fail(f"strict checkpoint completion manifest is missing or unsafe: {completion}")
    completion_bytes = completion.stat().st_size
    actual = _directory_size(step_dir)
    if actual != declared_payload + completion_bytes:
        _fail("strict checkpoint storage inventory does not match on-disk bytes")
    return {
        "full_transaction_bytes": actual,
        "declared_payload_bytes": declared_payload,
        "model_metadata_completion_bytes": model_bundle + completion_bytes,
    }


def _preflight_capacity_world(
    preflight: Mapping[str, Any], world_size: int
) -> tuple[str, Mapping[str, Any]]:
    capacity = _mapping(
        _mapping(preflight.get("corpus"), "preflight corpus").get(
            "packing_capacity_receipt"
        ),
        "preflight packing capacity",
    )
    digest = _sha256(capacity.get("sha256"), "preflight packing-capacity SHA-256")
    if capacity.get("gate_passed") is not True:
        _fail("preflight packing-capacity gate did not pass")
    worlds = _mapping(capacity.get("worlds"), "preflight packing-capacity worlds")
    world = _mapping(worlds.get(str(world_size)), f"preflight packing capacity ws{world_size}")
    repeat_mode = world.get("capacity_mode") == "whole_pool_repeat_v3"
    if repeat_mode:
        if (
            world.get("repetition_tier") != "preferred"
            or world.get("whole_pool_repetition_only") is not True
            or world.get("source_specific_repetition") is not False
            or world.get("epoch5_loaded_including_prefetch") is not False
        ):
            _fail(f"preflight repetition capacity did not pass for ws{world_size}")
    elif world.get("passes_40x_no_wrap_with_margin") is not True:
        _fail(f"preflight packing capacity did not pass for ws{world_size}")
    try:
        safe = capacity_authorized_positions(world)
    except StrictTrainingError as exc:
        raise FamilyWorkflowError(
            f"preflight packing capacity ws{world_size} lacks an authorized horizon"
        ) from exc
    required = _positive_int(
        world.get("required_positions_with_margin"),
        f"preflight packing capacity ws{world_size} required positions",
    )
    if safe < required:
        _fail(f"preflight packing capacity ws{world_size} is below its required margin")
    return digest, world


def _disk_free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not probe.exists():
        _fail(f"cannot find an existing filesystem ancestor for {path}")
    return shutil.disk_usage(probe).free


def _parse_storage_bytes(value: str, name: str) -> int | None:
    cleaned = value.strip().replace(",", "")
    if cleaned.lower() == "unlimited":
        return None
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([KMGTPE]?i?B)?", cleaned, re.I)
    if match is None:
        _fail(f"cannot parse {name} from BeeGFS quota output: {value!r}")
    number = float(match.group(1))
    unit = (match.group(2) or "B").upper()
    powers = {
        "B": 0,
        "KB": 1,
        "MB": 2,
        "GB": 3,
        "TB": 4,
        "PB": 5,
        "EB": 6,
        "KIB": 1,
        "MIB": 2,
        "GIB": 3,
        "TIB": 4,
        "PIB": 5,
        "EIB": 6,
    }
    base = 1024 if "I" in unit else 1000
    return int(number * base ** powers[unit])


def _live_beegfs_storage(
    repo_root: Path, *, uid: int, storage_pool_id: int, path: Path
) -> tuple[int, dict[str, Any]]:
    command = [
        "beegfs-ctl",
        "--getquota",
        "--uid",
        str(uid),
        f"--storagepoolid={storage_pool_id}",
        "--csv",
    ]
    try:
        output = subprocess.check_output(
            command, cwd=repo_root, text=True, stderr=subprocess.STDOUT
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        details = getattr(exc, "output", "")
        raise FamilyWorkflowError(
            "cannot query live UHeM BeeGFS user quota"
            + (f": {str(details).strip()}" if details else "")
        ) from exc
    rows = list(csv.reader(io.StringIO(output)))
    if len(rows) < 2:
        _fail("BeeGFS quota CSV has no data row")
    raw_header = [cell.strip().lower() for cell in rows[0]]
    data_rows = [
        row
        for row in rows[1:]
        if row and not all(not cell.strip() for cell in row)
    ]
    if not data_rows:
        _fail("BeeGFS quota CSV has no usable data row")
    # UHeM's verified BeeGFS build emits duplicate `hard` headings: the first
    # size/hard pair is byte quota and the second files/hard pair is inode
    # quota. Preserve the positional schema instead of collapsing duplicate
    # headings into a dictionary.
    if raw_header == ["name", "id", "size", "hard", "files", "hard"]:
        candidates = [row for row in data_rows if len(row) == 6]
        if not candidates:
            _fail("UHeM BeeGFS quota CSV row does not have six columns")
        matching_rows = [row for row in candidates if row[1].strip() == str(uid)]
        if len(matching_rows) != 1:
            _fail("UHeM BeeGFS quota CSV must contain exactly one requested UID row")
        selected_row = matching_rows[0]
        used_raw = selected_row[2]
        hard_raw = selected_row[3]
        csv_schema = "uhem_name_id_size_hard_files_hard_v1"
    else:
        header = [
            re.sub(r"[^a-z0-9]+", "_", cell).strip("_")
            for cell in raw_header
        ]
        indexed_rows = [
            list(enumerate(row + [""] * max(0, len(header) - len(row))))
            for row in data_rows
        ]
        matching_rows = [
            row
            for row in indexed_rows
            if str(uid) in {value.strip() for _index, value in row}
        ]
        if len(matching_rows) != 1:
            _fail("BeeGFS quota CSV must contain exactly one requested UID row")
        selected = matching_rows[0]

        def find_field(*needles: str) -> str:
            matches = [
                value
                for index, value in selected
                if index < len(header)
                and all(needle in header[index] for needle in needles)
                and "inode" not in header[index]
            ]
            if len(matches) != 1:
                _fail(
                    "BeeGFS quota CSV does not expose exactly one "
                    + "/".join(needles)
                    + " size field"
                )
            return matches[0]

        used_raw = find_field("used")
        hard_raw = find_field("hard")
        csv_schema = "descriptive_used_and_hard_size_headers_v1"

    used = _parse_storage_bytes(used_raw, "BeeGFS used bytes")
    hard = _parse_storage_bytes(hard_raw, "BeeGFS hard quota")
    if used is None:
        _fail("BeeGFS used-byte value cannot be unlimited")
    physical_free = _disk_free_bytes(path)
    finite_remaining = None if hard is None else max(0, hard - used)
    effective_free = (
        physical_free
        if finite_remaining is None
        else min(physical_free, finite_remaining)
    )
    return effective_free, {
        "uid": uid,
        "storage_pool_id": storage_pool_id,
        "used_bytes": used,
        "hard_quota_bytes": hard,
        "hard_quota_unlimited": hard is None,
        "csv_schema": csv_schema,
        "finite_user_quota_remaining_bytes": finite_remaining,
        "physical_filesystem_free_bytes": physical_free,
        "effective_free_bytes": effective_free,
        "beegfs_quota_output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
    }


def _live_uhem_cpu_saat(
    repo_root: Path, account: str, user: str
) -> tuple[float, str, dict[str, Any]]:
    command = [
        "sshare",
        "-n",
        "-P",
        "-A",
        account,
        "-u",
        user,
        "-o",
        "Account,User,GrpTRESMins,GrpTRESRaw,TRESRunMins,RawUsage",
    ]
    try:
        output = subprocess.check_output(
            command, cwd=repo_root, text=True, stderr=subprocess.STDOUT
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        details = getattr(exc, "output", "")
        raise FamilyWorkflowError(
            "cannot query live UHeM CPU-saat quota with sshare"
            + (f": {str(details).strip()}" if details else "")
        ) from exc
    def cpu_minutes(field: str) -> int | None:
        match = re.search(r"(?:^|,)cpu=(\d+)(?:,|$)", field)
        return int(match.group(1)) if match else None

    def raw_usage_seconds(field: str) -> float:
        try:
            value = float(field.strip())
        except ValueError as exc:
            raise FamilyWorkflowError(
                f"cannot parse sshare RawUsage TRES-seconds: {field!r}"
            ) from exc
        if not math.isfinite(value) or value < 0:
            _fail(f"invalid sshare RawUsage TRES-seconds: {field!r}")
        return value

    rows = []
    for raw_line in output.splitlines():
        if not raw_line.strip():
            continue
        fields = raw_line.split("|")
        if len(fields) < 6:
            _fail(f"unexpected sshare row: {raw_line!r}")
        row_account, row_user = fields[0].strip(), fields[1].strip()
        if row_account != account or row_user not in {"", user}:
            continue
        rows.append(
            {
                "user": row_user,
                "limit": cpu_minutes(fields[2]),
                "used": cpu_minutes(fields[3]),
                "running": cpu_minutes(fields[4]),
                "raw_usage_seconds": raw_usage_seconds(fields[5]),
            }
        )
    if not rows:
        _fail("live UHeM sshare output contains no matching account/user row")
    limits = [int(row["limit"]) for row in rows if row["limit"] is not None]
    if not limits:
        _fail("live UHeM sshare output contains no inherited or direct CPU-minute limit")
    # The user row may inherit a blank limit from its account row.  Use the
    # smallest declared limit, and conservatively subtract the largest account
    # or user raw/running counters so shared-account use is never understated.
    positive_limits = [value for value in limits if value > 0]
    if not positive_limits:
        _fail("live UHeM CPU-minute limit is zero")
    for row in rows:
        if row["used"] is None:
            _fail("live UHeM sshare row is missing GrpTRESRaw cpu minutes")
        raw_minutes = float(row["raw_usage_seconds"]) / 60.0
        # UHeM exposes RawUsage in TRES-seconds while GrpTRESRaw is rendered in
        # whole CPU minutes.  Permit only the one-minute display-rounding gap;
        # a larger disagreement means the quota cannot be interpreted safely.
        if abs(float(row["used"]) - raw_minutes) > 1.0:
            _fail(
                "live UHeM GrpTRESRaw cpu minutes disagree with RawUsage/60 "
                f"for user {row['user']!r}: {row['used']} versus {raw_minutes}"
            )
    limit = min(positive_limits)
    used_row = max(rows, key=lambda row: int(row["used"]))
    used = int(used_row["used"])
    running = max([int(row["running"] or 0) for row in rows], default=0)
    remaining_minutes = limit - used - running
    if remaining_minutes <= 0:
        _fail("live UHeM remaining CPU-minute quota is zero")
    audit = {
        "limit_cpu_minutes": limit,
        "used_raw_cpu_minutes": used,
        "raw_usage_tres_seconds": float(used_row["raw_usage_seconds"]),
        "raw_usage_equivalent_cpu_minutes": float(used_row["raw_usage_seconds"]) / 60.0,
        "raw_usage_rounding_delta_cpu_minutes": (
            float(used_row["raw_usage_seconds"]) / 60.0 - used
        ),
        "reserved_running_cpu_minutes": running,
        "remaining_cpu_minutes": remaining_minutes,
    }
    remaining = remaining_minutes / 60.0
    return remaining, hashlib.sha256(output.encode("utf-8")).hexdigest(), audit


def _resolved_artifact_paths(recipe: Mapping[str, Any], base_dir: Path) -> dict[str, Path]:
    artifacts = _mapping(recipe["artifacts"], "artifacts")
    corpus_root = base_dir / str(artifacts["corpus_root"])
    tokenizer_root = base_dir / str(artifacts["tokenizer_root"])
    resolved = {
        "corpus_root": corpus_root,
        "corpus_manifest": corpus_root / str(artifacts["corpus_manifest"]),
        "dataset_manifest": corpus_root / str(artifacts["nanochat_dataset_manifest"]),
        "source_receipt": corpus_root / str(artifacts["source_receipt"]),
        "packing_capacity": corpus_root / str(artifacts["packing_capacity_receipt"]),
        "validation_exposure": corpus_root / str(artifacts["validation_exposure_manifest"]),
        "exposure_plan_index": corpus_root / str(artifacts["exposure_plan_index"]),
        "tokenizer_root": tokenizer_root,
        "tokenizer_package": tokenizer_root / str(artifacts["tokenizer_package_manifest"]),
    }
    if "macocu_preparation_manifest" in artifacts:
        resolved["macocu_preparation"] = base_dir / str(
            artifacts["macocu_preparation_manifest"]
        )
    for source_id, artifact_key, resolved_key in (
        ("mot_tr_v1_11", "mot_preparation_manifest", "mot_preparation"),
        (
            "parlamint_tr_v5_0",
            "parlamint_preparation_manifest",
            "parlamint_preparation",
        ),
    ):
        if artifact_key in artifacts:
            resolved[resolved_key] = base_dir / str(artifacts[artifact_key])
    for key, relative in _mapping(
        artifacts["training_exposure_manifests"],
        "artifacts.training_exposure_manifests",
    ).items():
        resolved[f"exposure:{key}"] = corpus_root / str(relative)
    return resolved


def _verify_anchor_preparation_binding(
    path: Path,
    *,
    source_id: str,
    derived_sources: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify one accepted native-text preparation and its source provenance."""

    if path.name != "manifest.json" or path.is_symlink() or not path.is_file():
        _fail(f"{source_id} preparation manifest is missing or unsafe")
    try:
        from nanochat.turkish_anchor_preparation import validate_anchor_preparation

        manifest = validate_anchor_preparation(path.parent, verify_files=True)
        digest = verify_manifest_hash(manifest)
    except (OSError, ValueError) as exc:
        raise FamilyWorkflowError(
            f"invalid accepted preparation for {source_id}: {exc}"
        ) from exc
    acceptance = _mapping(
        _mapping(
            manifest.get("production_acceptance"),
            f"{source_id} production acceptance",
        ).get("receipt"),
        f"{source_id} production acceptance receipt",
    )
    data = _mapping(
        _mapping(manifest.get("artifacts"), f"{source_id} artifacts").get("data"),
        f"{source_id} data artifact",
    )
    expected_provenance = {
        "manifest_uri": path.resolve().as_uri(),
        "manifest_sha256": digest,
        "source_id": source_id,
        "preparer_version": manifest.get("preparer_version"),
        "production_acceptance": {
            "stage": "accepted_production",
            "receipt_sha256": acceptance.get("canonical_sha256"),
        },
        "acquisition_receipt_sha256": _mapping(
            manifest.get("acquisition_receipt"),
            f"{source_id} acquisition receipt",
        ).get("canonical_sha256"),
        "clean": manifest.get("clean"),
        "data_artifact": {
            "logical_jsonl_sha256": data.get("logical_jsonl_sha256"),
            "totals": data.get("totals"),
        },
        "downstream_admission": {
            "preparer_automatically_admits_training": False,
            "backend_turkish_no_code_audit_required": True,
        },
    }
    if manifest.get("source_id") != source_id:
        _fail(f"{source_id} preparation manifest names another source")
    if derived_sources.get(source_id) != expected_provenance:
        _fail(f"{source_id} preparation/source-receipt binding drifted")
    return {"path": str(path.resolve()), "sha256": digest}


def _verify_packing_capacity_receipt(
    path: Path,
    *,
    dataset_sha256: str,
    tokenizer_sha256: str,
    implementation_path: Path,
    family_id: str,
) -> tuple[dict[str, Any], str]:
    """Verify the family-specific capacity proof for both topologies."""

    if family_id == FAMILY_ID_V3:
        receipt, digest = _load_receipt(
            path, "turkish_bestfit_repeat_capacity_receipt"
        )
        if not implementation_path.is_file() or implementation_path.is_symlink():
            _fail("packing-capacity simulator implementation is missing or unsafe")
        try:
            from nanochat.packing_capacity import (
                validate_repetition_capacity_receipt,
            )

            summary = validate_repetition_capacity_receipt(
                receipt,
                dataset_manifest_sha256=dataset_sha256,
                tokenizer_package_sha256=tokenizer_sha256,
                # The v3 production family currently accepts the preferred
                # tier only. A manual-risk receipt therefore remains closed
                # until a separate sealed approval is added to the contract.
                manual_repetition_risk_approval=None,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FamilyWorkflowError(
                f"invalid v3 repetition capacity receipt: {exc}"
            ) from exc
        if (
            summary.get("canonical_sha256") != digest
            or summary.get("repetition_tier") != "preferred"
            or summary.get("gate_passed") is not True
            or summary.get("cleanup_authorized") is not True
            or summary.get("approval_required") is not False
            or summary.get("approval_satisfied") is not True
        ):
            _fail("v3 repetition capacity gate did not pass at the preferred tier")
        simulation = _mapping(
            receipt.get("simulation"), "repetition packing-capacity simulation"
        )
        if simulation.get("implementation_file_sha256") != file_sha256(
            implementation_path
        ):
            _fail("packing-capacity simulator source differs from the receipt")
        worlds = _mapping(
            simulation.get("worlds"), "repetition packing-capacity worlds"
        )
        if set(worlds) != {"8", "16"}:
            _fail("repetition capacity receipt must cover exactly ws8 and ws16")
        for world_size in (8, 16):
            selected = capacity_world_gate_record(receipt, world_size)
            if (
                selected.get("world_size") != world_size
                or selected.get("device_batch_sequences") != 4
                or selected.get("max_seq_len") != 2048
                or selected.get("gradient_accumulation_steps")
                != 2_097_152 // (world_size * 4 * 2048)
                or capacity_authorized_positions(selected)
                < 32_640 * 2_097_152
            ):
                _fail(f"repetition capacity ws{world_size} topology drifted")
        return receipt, digest

    receipt, digest = _load_receipt(path, "turkish_bestfit_capacity_receipt")
    expected_top = {
        "dataset_manifest_sha256": dataset_sha256,
        "tokenizer_package_sha256": tokenizer_sha256,
        "mix_gate_evaluated_on_common_horizon": True,
        "mix_gate_passed": True,
        "no_wrap_gate_passed": True,
        "gate_passed": True,
        "cleanup_authorized": True,
    }
    for field, expected in expected_top.items():
        if receipt.get(field) != expected:
            _fail(f"packing-capacity receipt {field} mismatch")
    simulation = _mapping(receipt.get("simulation"), "packing-capacity simulation")
    if simulation.get("implementation") != "nanochat_upstream_bos_bestfit_crop_capacity_v2":
        _fail("packing-capacity simulator identity mismatch")
    if not implementation_path.is_file() or implementation_path.is_symlink():
        _fail("packing-capacity simulator implementation is missing or unsafe")
    if simulation.get("implementation_file_sha256") != file_sha256(implementation_path):
        _fail("packing-capacity simulator source differs from the receipt")
    contract = _mapping(simulation.get("upstream_contract"), "packing upstream contract")
    expected_contract = {
        "nanochat_revision": "92d63d4e",
        "encode_call": "tokenizer.encode(doc_batch, prepend=bos_token, num_threads=4)",
        "tokenizer_batch_size": 128,
        "tokenizer_threads": 4,
        "refill_buffer_size": 1000,
        "tie_breaks": "first_largest_fit_else_first_shortest",
        "cropped_tail_policy": "discard",
        "rank_sharding": "row_group_index_mod_world_size",
    }
    if contract != expected_contract:
        _fail("packing-capacity upstream loader contract drifted")
    parity = _mapping(simulation.get("fixture_parity"), "packing fixture parity")
    if (
        parity.get("passed") is not True
        or parity.get("upstream_commit")
        != "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"
        or parity.get("upstream_loader_source_sha256")
        != "ed1d4997e3c407f242fbbcfa627f2987f8f34a3a0c340d792c4e10a202981990"
        or parity.get("actual_output_sha256") != parity.get("simulated_output_sha256")
    ):
        _fail("packing-capacity upstream fixture parity is incomplete")
    for field in ("fixture_sha256", "actual_output_sha256", "simulated_output_sha256"):
        _sha256(parity.get(field), f"packing fixture {field}")
    if simulation.get("all_worlds_pass") is not True:
        _fail("packing-capacity simulation did not pass every selectable topology")
    worlds = _mapping(simulation.get("worlds"), "packing-capacity worlds")
    if set(worlds) != {"8", "16"}:
        _fail("packing-capacity receipt must cover exactly ws8 and ws16")
    required_steps = 32_000
    required_steps_with_margin = 32_640
    global_batch = 2_097_152
    required_positions = required_steps_with_margin * global_batch
    for world_size in (8, 16):
        world = _mapping(worlds[str(world_size)], f"packing capacity ws{world_size}")
        expected_world = {
            "world_size": world_size,
            "device_batch_sequences": 4,
            "max_seq_len": 2048,
            "buffer_size": 1000,
            "preserve_document_tails": False,
            "row_capacity": 2049,
            "rank_sharding": "parquet_row_group_index_mod_world_size",
            "gradient_accumulation_steps": global_batch // (world_size * 4 * 2048),
            "required_optimizer_steps": required_steps,
            "safety_margin_fraction": 0.02,
            "required_optimizer_steps_with_margin": required_steps_with_margin,
            "required_positions_with_margin": required_positions,
            "passes_40x_no_wrap_with_margin": True,
            "first_wrap_observation": "right_censored_at_required_horizon",
            "safe_global_scheduled_positions_semantics": (
                "right_censored_proven_lower_bound_at_required_horizon"
            ),
            "aggregate_scope": "exact_common_required_horizon_all_ranks",
        }
        for field, expected in expected_world.items():
            if world.get(field) != expected:
                _fail(f"packing capacity ws{world_size}.{field} mismatch")
        if int(world.get("safe_global_scheduled_positions", -1)) < required_positions:
            _fail(f"packing capacity ws{world_size} is below the 40x margin")
        completed = _sequence(
            world.get("completed_microbatches_by_rank"),
            f"packing capacity ws{world_size} rank completions",
        )
        requested = _positive_int(
            world.get("requested_microbatches_per_rank"),
            f"packing capacity ws{world_size} requested microbatches",
        )
        if len(completed) != world_size or any(
            isinstance(value, bool) or not isinstance(value, int) or value < requested
            for value in completed
        ):
            _fail(f"packing capacity ws{world_size} lacks complete per-rank coverage")
        first_wrap = _sequence(
            world.get("first_wrap_before_microbatch_by_rank"),
            f"packing capacity ws{world_size} first-wrap observations",
        )
        if len(first_wrap) != world_size or any(value is not None for value in first_wrap):
            _fail(f"packing capacity ws{world_size} observed an epoch wrap")
    return receipt, digest


def command_validate_recipe(args: argparse.Namespace) -> None:
    recipe, digest = load_recipe(args.recipe, require_sealed=not args.allow_unsealed)
    print(json.dumps({"family_id": recipe["family_id"], "canonical_sha256": digest}, sort_keys=True))


def _data_prep_policy_sha256(policy: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(policy).encode("utf-8")).hexdigest()


def _validate_recipe_policy_identity(
    recipe: Mapping[str, Any], policy: Mapping[str, Any]
) -> None:
    artifacts = _mapping(recipe.get("artifacts"), "recipe artifacts")
    tokenizer = _mapping(policy.get("tokenizer_training"), "policy tokenizer_training")
    if policy.get("name") != artifacts.get("corpus_id"):
        _fail("corpus policy name differs from the selected family recipe")
    if tokenizer.get("name") != artifacts.get("tokenizer_name"):
        _fail("corpus policy tokenizer name differs from the selected family recipe")
    if tokenizer.get("vocab_size") != recipe.get("model", {}).get("vocab_size"):
        _fail("corpus policy tokenizer vocabulary differs from the family model")


def _load_data_prep_inputs(
    *,
    recipe_path: Path,
    policy_path: Path,
    source_plan_path: Path,
    calibration_path: Path,
) -> tuple[
    dict[str, Any],
    str,
    dict[str, Any],
    str,
    dict[str, Any],
    str,
    dict[str, Any],
    str,
]:
    """Load and validate the shared data-preparation provenance spine."""

    recipe, recipe_sha = load_recipe(recipe_path)
    try:
        from nanochat.turkish_backend import (
            validate_backend_calibration,
            validate_source_plan,
        )
        from nanochat.turkish_corpus import load_corpus_policy

        policy = load_corpus_policy(policy_path)
        _validate_recipe_policy_identity(recipe, policy)
        source_plan = _load_object(source_plan_path, "source plan")
        calibration = _load_object(calibration_path, "backend calibration")
        validate_source_plan(source_plan, policy)
        validate_backend_calibration(calibration, policy)
    except (OSError, ValueError) as exc:
        raise FamilyWorkflowError(f"invalid data-preparation provenance: {exc}") from exc
    policy_sha = _data_prep_policy_sha256(policy)
    source_plan_sha = _sha256(source_plan.get("canonical_sha256"), "source-plan SHA-256")
    calibration_sha = _sha256(
        calibration.get("canonical_sha256"), "backend-calibration SHA-256"
    )
    return (
        recipe,
        recipe_sha,
        policy,
        policy_sha,
        source_plan,
        source_plan_sha,
        calibration,
        calibration_sha,
    )


def _production_pack_plan_lanes(
    plan: Mapping[str, Any], node_count: int
) -> list[dict[str, Any]]:
    """Balance objects over 32 serial 4-CPU workers on each cpu2dq node."""

    if node_count <= 0:
        _fail("production pack-plan node count must be positive")
    lane_count = node_count * PRODUCTION_WORKERS_PER_NODE
    objects = _sequence(plan.get("objects"), "source-plan objects")
    if lane_count > len(objects):
        _fail("production pack-plan cannot have more lanes than source objects")
    lanes = [
        {
            "lane_id": lane_id,
            "node_index": lane_id // PRODUCTION_WORKERS_PER_NODE,
            "node_local_lane_id": lane_id % PRODUCTION_WORKERS_PER_NODE,
            "object_ranks": [],
            "total_input_bytes": 0,
        }
        for lane_id in range(lane_count)
    ]
    def object_rank(item: Mapping[str, Any]) -> int:
        rank = item.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            _fail("source object rank must be a non-negative integer")
        return rank

    ordered = sorted(
        objects,
        key=lambda item: (
            -_positive_int(item.get("size_bytes"), "source object size"),
            object_rank(item),
        ),
    )
    for item in ordered:
        lane = min(lanes, key=lambda value: (value["total_input_bytes"], value["lane_id"]))
        lane["object_ranks"].append(int(item["rank"]))
        lane["total_input_bytes"] += int(item["size_bytes"])
    size_by_rank = {int(item["rank"]): int(item["size_bytes"]) for item in objects}
    for lane in lanes:
        lane["object_ranks"].sort()
        lane["maximum_staged_raw_bytes"] = max(
            size_by_rank[rank] for rank in lane["object_ranks"]
        )
    return lanes


def _validate_production_pack_plan(
    pack_plan: Mapping[str, Any],
    *,
    recipe: Mapping[str, Any],
    recipe_sha: str,
    policy_sha: str,
    source_plan: Mapping[str, Any],
    source_plan_sha: str,
) -> int:
    verify_manifest_hash(pack_plan)
    if pack_plan.get("schema_version") != "1.0" or pack_plan.get("kind") != DATA_PREP_PACK_PLAN_KIND:
        _fail("unexpected data-preparation production pack plan")
    expected_bindings = {
        "family_id": recipe["family_id"],
        "recipe_sha256": recipe_sha,
        "policy_sha256": policy_sha,
        "source_plan_sha256": source_plan_sha,
    }
    for field, expected in expected_bindings.items():
        if pack_plan.get(field) != expected:
            _fail(f"production pack-plan {field} binding mismatch")
    expected_contract = {
        "allocation": "one_exclusive_128cpu_cpu2dq_allocation_per_node",
        "workers_per_node": PRODUCTION_WORKERS_PER_NODE,
        "cpus_per_worker": PRODUCTION_CPUS_PER_WORKER,
        "lane_concurrency": "thirty_two_worker_lanes_share_each_node_all_nodes_concurrent",
        "staging": "at_most_one_source_object_per_lane_at_a_time",
        "rank_execution": "ascending_rank_order_within_each_lane",
        "production_launch_must_consume_exact_plan": True,
    }
    if pack_plan.get("execution_contract") != expected_contract:
        _fail("production pack-plan execution contract drifted")
    expected_bucket_contract = {
        "allocation": "one_exclusive_128cpu_cpu2dq_node",
        "bucket_tasks": 14,
        "cpus_per_task": 8,
        "all_bucket_tasks_concurrent": True,
        "production_launch_must_consume_exact_plan": True,
    }
    if pack_plan.get("minhash_bucket_execution_contract") != expected_bucket_contract:
        _fail("production pack-plan MinHash bucket execution contract drifted")
    objects = _sequence(source_plan.get("objects"), "source-plan objects")
    size_by_rank = {int(item["rank"]): int(item["size_bytes"]) for item in objects}
    lanes = _sequence(pack_plan.get("lanes"), "production pack-plan lanes")
    if not lanes:
        _fail("production pack plan has no lanes")
    node_count = _positive_int(pack_plan.get("node_count"), "production node count")
    if (
        pack_plan.get("workers_per_node") != PRODUCTION_WORKERS_PER_NODE
        or pack_plan.get("cpus_per_worker") != PRODUCTION_CPUS_PER_WORKER
        or pack_plan.get("lane_count")
        != node_count * PRODUCTION_WORKERS_PER_NODE
        or len(lanes) != node_count * PRODUCTION_WORKERS_PER_NODE
    ):
        _fail("production pack-plan node/worker/lane geometry drifted")
    seen: list[int] = []
    maxima: list[int] = []
    for expected_lane_id, raw_lane in enumerate(lanes):
        lane = _mapping(raw_lane, f"production pack lane {expected_lane_id}")
        _require_exact_keys(
            lane,
            {
                "lane_id",
                "node_index",
                "node_local_lane_id",
                "object_ranks",
                "total_input_bytes",
                "maximum_staged_raw_bytes",
            },
            f"production pack lane {expected_lane_id}",
        )
        if lane.get("lane_id") != expected_lane_id:
            _fail("production pack-plan lane IDs must be contiguous")
        if (
            lane.get("node_index")
            != expected_lane_id // PRODUCTION_WORKERS_PER_NODE
            or lane.get("node_local_lane_id")
            != expected_lane_id % PRODUCTION_WORKERS_PER_NODE
        ):
            _fail("production pack-plan lane-to-node ownership drifted")
        ranks = list(_sequence(lane.get("object_ranks"), "production lane ranks"))
        if not ranks or ranks != sorted(ranks) or any(
            isinstance(rank, bool) or not isinstance(rank, int) or rank not in size_by_rank
            for rank in ranks
        ):
            _fail("production pack-plan lane ranks are invalid or unsorted")
        if lane.get("total_input_bytes") != sum(size_by_rank[rank] for rank in ranks):
            _fail("production pack-plan lane input-byte arithmetic mismatch")
        lane_maximum = max(size_by_rank[rank] for rank in ranks)
        if lane.get("maximum_staged_raw_bytes") != lane_maximum:
            _fail("production pack-plan lane staged-byte arithmetic mismatch")
        seen.extend(ranks)
        maxima.append(lane_maximum)
    if sorted(seen) != list(range(len(objects))) or len(seen) != len(set(seen)):
        _fail("production pack plan must cover every source rank exactly once")
    projected_peak = sum(maxima)
    if pack_plan.get("projected_peak_staged_raw_bytes") != projected_peak:
        _fail("production pack-plan projected staged-byte peak mismatch")
    return projected_peak


def _packed_production_backend_cpu_projection(
    *,
    source_plan: Mapping[str, Any],
    pack_plan: Mapping[str, Any],
    backend_report: Mapping[str, Any],
    sample_bucket_receipts: Mapping[int, Mapping[str, Any]],
) -> tuple[float, dict[str, Any]]:
    """Bill packed object workers by node wall, then other stages once."""

    source_projections = _mapping(
        backend_report.get("source_projections"), "backend source projections"
    )
    per_byte_wall: dict[str, float] = {}
    for source_id, raw in source_projections.items():
        projection = _mapping(raw, f"backend source projection {source_id}")
        full_bytes = _positive_number(
            projection.get("full_input_bytes"), f"{source_id} full input bytes"
        )
        wall = sum(
            _nonnegative_number(
                projection.get(field), f"{source_id} projected {field}"
            )
            for field in (
                "projected_download_wall_seconds",
                "projected_score_lid_wall_seconds",
                "projected_minhash_signature_wall_seconds",
            )
        )
        per_byte_wall[str(source_id)] = wall / full_bytes
    objects = _sequence(source_plan.get("objects"), "source-plan objects")
    object_wall: dict[int, float] = {}
    for item in objects:
        rank = int(item["rank"])
        source_id = str(item["source_id"])
        if source_id not in per_byte_wall:
            _fail(f"backend report lacks a projection for source {source_id}")
        object_wall[rank] = int(item["size_bytes"]) * per_byte_wall[source_id]
    node_lane_wall: dict[int, list[float]] = {
        node: [] for node in range(int(pack_plan["node_count"]))
    }
    for raw_lane in pack_plan["lanes"]:
        lane = _mapping(raw_lane, "production pack lane")
        lane_wall = sum(object_wall[int(rank)] for rank in lane["object_ranks"])
        node_lane_wall[int(lane["node_index"])].append(lane_wall)
    node_wall = {
        str(node): max(walls) for node, walls in sorted(node_lane_wall.items())
    }
    packed_object_cpu = sum(node_wall.values()) * CPU2DQ_BILLABLE_CPUS / 3600.0
    report_projection = _mapping(
        backend_report.get("projection"), "backend report projection"
    )
    stage_cpu = _mapping(
        report_projection.get("stage_billed_cpu_saat_before_safety_factor"),
        "backend stage CPU projection",
    )
    # The report's aggregate bucket stage wall is diagnostic here: all fourteen
    # disjoint bucket ranks share one future node allocation. Project each task
    # independently and bill the slowest concurrent task once at 128 CPUs.
    _nonnegative_number(
        stage_cpu.get("minhash_buckets"), "aggregate MinHash bucket diagnostic"
    )
    projected_signature_bytes = _positive_number(
        report_projection.get("signature_bytes"), "projected signature bytes"
    )
    if set(sample_bucket_receipts) != set(range(14)):
        _fail("packed bucket CPU projection requires all fourteen sample receipts")
    # The backend report's signature_bytes is the sum over all fourteen MinHash
    # bands. Each concurrent bucket task owns exactly one band, so extrapolate a
    # task from one fourteenth of that total rather than charging every task for
    # the complete signature inventory.
    projected_signature_bytes_per_bucket = projected_signature_bytes / 14
    projected_bucket_wall = {
        str(rank): _positive_number(
            _mapping(receipt.get("telemetry"), "sample bucket telemetry").get(
                "wall_seconds"
            ),
            "sample bucket wall seconds",
        )
        * projected_signature_bytes_per_bucket
        / _positive_number(
            receipt.get("input_signature_bytes"), "sample bucket signature bytes"
        )
        for rank, receipt in sorted(sample_bucket_receipts.items())
    }
    packed_bucket_node_wall = max(projected_bucket_wall.values())
    bucket_cpu = packed_bucket_node_wall * CPU2DQ_BILLABLE_CPUS / 3600.0
    cluster_cpu = _nonnegative_number(
        stage_cpu.get("priority_cluster_quality_format"),
        "projected priority-cluster CPU-saat",
    )
    cluster_scaling = _mapping(
        report_projection.get("cluster_scaling"),
        "backend projected cluster scaling",
    )
    sample_cluster_peak_rss = _positive_int(
        cluster_scaling.get("sample_peak_rss_bytes"),
        "sample cluster peak RSS",
    )
    projected_cluster_peak_rss = _positive_number(
        cluster_scaling.get("projected_peak_rss_bytes"),
        "projected cluster peak RSS",
    )
    total = packed_object_cpu + bucket_cpu + cluster_cpu
    if total <= 0:
        _fail("packed production backend CPU projection must be positive")
    return total, {
        "object_worker_contract": "thirty_two_4cpu_lanes_per_exclusive_128cpu_node",
        "node_count": int(pack_plan["node_count"]),
        "projected_node_wall_seconds_before_safety": node_wall,
        "projected_packed_object_cpu_saat_before_safety": packed_object_cpu,
        "projected_signature_bytes_per_bucket_before_safety": (
            projected_signature_bytes_per_bucket
        ),
        "projected_bucket_task_wall_seconds_before_safety": projected_bucket_wall,
        "projected_packed_bucket_node_wall_seconds_before_safety": (
            packed_bucket_node_wall
        ),
        "projected_minhash_bucket_cpu_saat_before_safety": bucket_cpu,
        "projected_priority_cluster_cpu_saat_before_safety": cluster_cpu,
        "sample_priority_cluster_peak_rss_bytes": sample_cluster_peak_rss,
        "projected_priority_cluster_peak_rss_bytes_before_safety": (
            projected_cluster_peak_rss
        ),
        "sample_priority_cluster_edge_participating_documents": (
            _nonnegative_int(
                cluster_scaling.get("sample_edge_participating_documents"),
                "sample cluster edge-participating documents",
            )
        ),
        "projected_priority_cluster_edge_participating_documents": (
            _nonnegative_number(
                cluster_scaling.get("projected_edge_participating_documents"),
                "projected cluster edge-participating documents",
            )
        ),
        "projected_backend_cpu_saat_before_safety": total,
    }


def command_seal_data_prep_pack_plan(args: argparse.Namespace) -> None:
    if args.output.exists():
        _fail(f"refusing to overwrite production pack plan: {args.output}")
    (
        recipe,
        recipe_sha,
        _policy,
        policy_sha,
        source_plan,
        source_plan_sha,
        _calibration,
        _calibration_sha,
    ) = _load_data_prep_inputs(
        recipe_path=args.recipe,
        policy_path=args.policy,
        source_plan_path=args.source_plan,
        calibration_path=args.calibration,
    )
    lanes = _production_pack_plan_lanes(source_plan, args.nodes)
    receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": DATA_PREP_PACK_PLAN_KIND,
            "family_id": recipe["family_id"],
            "recipe_sha256": recipe_sha,
            "policy_sha256": policy_sha,
            "source_plan_sha256": source_plan_sha,
            "execution_contract": {
                "allocation": "one_exclusive_128cpu_cpu2dq_allocation_per_node",
                "workers_per_node": PRODUCTION_WORKERS_PER_NODE,
                "cpus_per_worker": PRODUCTION_CPUS_PER_WORKER,
                "lane_concurrency": (
                    "thirty_two_worker_lanes_share_each_node_all_nodes_concurrent"
                ),
                "staging": "at_most_one_source_object_per_lane_at_a_time",
                "rank_execution": "ascending_rank_order_within_each_lane",
                "production_launch_must_consume_exact_plan": True,
            },
            "minhash_bucket_execution_contract": {
                "allocation": "one_exclusive_128cpu_cpu2dq_node",
                "bucket_tasks": 14,
                "cpus_per_task": 8,
                "all_bucket_tasks_concurrent": True,
                "production_launch_must_consume_exact_plan": True,
            },
            "node_count": args.nodes,
            "workers_per_node": PRODUCTION_WORKERS_PER_NODE,
            "cpus_per_worker": PRODUCTION_CPUS_PER_WORKER,
            "lane_count": len(lanes),
            "lanes": lanes,
            "projected_peak_staged_raw_bytes": sum(
                int(lane["maximum_staged_raw_bytes"]) for lane in lanes
            ),
            "canonical_sha256": None,
        }
    )
    _validate_production_pack_plan(
        receipt,
        recipe=recipe,
        recipe_sha=recipe_sha,
        policy_sha=policy_sha,
        source_plan=source_plan,
        source_plan_sha=source_plan_sha,
    )
    write_json_atomic(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))


def command_seal_mixture_quality_approval(args: argparse.Namespace) -> None:
    """Seal an explicit human decision over the bounded QA report and examples."""

    if args.output.exists():
        _fail(f"refusing to overwrite mixture-quality approval: {args.output}")
    try:
        from nanochat.turkish_backend import (
            _MAX_AUDIT_REPORT_BYTES,
            _read_bounded_regular_file_snapshot,
            select_resource_sample_ranks,
            validate_backend_calibration,
            validate_sample_quality_audit_bundle,
            validate_source_plan,
        )
        from nanochat.turkish_corpus import load_corpus_policy

        policy = load_corpus_policy(args.policy)
        source_plan = _load_object(args.source_plan, "source plan")
        calibration = _load_object(args.calibration, "backend calibration")
        validate_source_plan(source_plan, policy)
        validate_backend_calibration(calibration, policy)
    except (OSError, ValueError) as exc:
        raise FamilyWorkflowError(f"invalid mixture-quality provenance: {exc}") from exc
    policy_sha = _data_prep_policy_sha256(policy)
    source_plan_sha = _sha256(
        source_plan.get("canonical_sha256"), "source-plan SHA-256"
    )
    calibration_sha = _sha256(
        calibration.get("canonical_sha256"), "calibration SHA-256"
    )
    output_parent = args.output.expanduser().resolve().parent
    audit_root = args.audit_report.expanduser().resolve().parent
    if audit_root != output_parent and output_parent not in audit_root.parents:
        _fail("quality-audit evidence must remain inside the approval directory tree")
    report_snapshot = _read_bounded_regular_file_snapshot(
        args.audit_report.expanduser().resolve(),
        label="bounded sample quality audit",
        max_bytes=_MAX_AUDIT_REPORT_BYTES,
    )
    report_record = {
        "path": args.audit_report.expanduser().resolve().relative_to(audit_root).as_posix(),
        "size_bytes": len(report_snapshot),
        "sha256": hashlib.sha256(report_snapshot).hexdigest(),
    }
    try:
        verified_audit, verified_audit_sha = validate_sample_quality_audit_bundle(
            audit_root,
            report_record,
            policy=policy,
            plan=source_plan,
            calibration=calibration,
            require_complete_accepted_coverage=args.decision == "accepted",
            report_snapshot=report_snapshot,
        )
    except ValueError as exc:
        raise FamilyWorkflowError(f"invalid sample-quality evidence bundle: {exc}") from exc
    audit = verified_audit
    audit_sha = verified_audit_sha
    cluster_sha = _sha256(
        audit.get("sample_cluster_receipt_sha256"),
        "quality-audit sample-cluster SHA-256",
    )
    if audit.get("cluster_receipt_sha256") != cluster_sha:
        _fail("quality-audit cluster aliases disagree")
    coverage = _mapping(audit.get("coverage"), "quality-audit coverage")
    expected_mixtures = sorted(str(item["id"]) for item in policy["mixture"])
    expected_ranks = select_resource_sample_ranks(source_plan)
    expected_hplt_bins = sorted(
        {
            int(source_plan["objects"][rank]["wds_bin"])
            for rank in expected_ranks
            if source_plan["objects"][rank]["source_id"] == "hplt3_tr"
        }
    )
    coverage_complete = (
        coverage.get("expected_mixtures") == expected_mixtures
        and coverage.get("mixtures_without_accepted_rows") == []
        and coverage.get("mixtures_with_accepted_rows") == expected_mixtures
        and coverage.get("expected_source_ranks") == expected_ranks
        and coverage.get("source_ranks_without_accepted_rows") == []
        and coverage.get("source_ranks_without_accepted_examples") == []
        and coverage.get("expected_hplt_wds_bins") == expected_hplt_bins
        and coverage.get("hplt_wds_bins_without_accepted_rows") == []
        and coverage.get("hplt_wds_bins_without_accepted_examples") == []
    )
    if args.decision == "accepted" and not coverage_complete:
        _fail("cannot accept a quality audit without accepted-row mixture coverage")
    if not args.reviewer.strip() or re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", args.reviewed_at_utc
    ) is None:
        _fail("mixture-quality approval requires reviewer and RFC3339 UTC timestamp")

    example_sampling = _mapping(
        audit.get("example_sampling"), "quality-audit example sampling"
    )
    files = _mapping(example_sampling.get("files"), "quality-audit example files")
    reviewed_files: dict[str, dict[str, Any]] = {}
    for decision in ("accepted", "rejected"):
        record = _mapping(files.get(decision), f"quality-audit {decision} examples")
        rows = _nonnegative_int(record.get("rows"), f"{decision} example rows")
        reviewed_files[decision] = {
            "rows": rows,
            "jsonl": dict(
                _mapping(
                    record.get("jsonl"),
                    f"quality-audit {decision} JSONL",
                )
            ),
            "plaintext": dict(
                _mapping(
                    record.get("plaintext"),
                    f"quality-audit {decision} plaintext",
                )
            ),
        }
    receipt = seal_manifest(
        {
            "schema_version": "3.0",
            "kind": MIXTURE_QUALITY_APPROVAL_KIND,
            "sample_quality_audit_sha256": audit_sha,
            "policy_sha256": policy_sha,
            "source_plan_sha256": source_plan_sha,
            "calibration_sha256": calibration_sha,
            "cluster_receipt_sha256": cluster_sha,
            "sample_cluster_receipt_sha256": cluster_sha,
            "evidence_bundle": {
                "schema_version": "1.0",
                "root": audit_root.relative_to(output_parent).as_posix(),
                "report": report_record,
            },
            "reviewed_example_files": reviewed_files,
            "coverage_complete": coverage_complete,
            "automatic_decision": False,
            "review_confirmation": (
                "bounded_strata_and_accepted_rejected_examples_reviewed"
            ),
            "reviewer": args.reviewer.strip(),
            "reviewed_at_utc": args.reviewed_at_utc,
            "decision": args.decision,
            "notes": args.notes,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))


def _live_completed_cpu2dq_allocation(
    repo_root: Path,
    *,
    job_id: str,
    stage: str,
    evidence_receipt_sha256s: Sequence[str],
) -> dict[str, Any]:
    """Return one exact completed allocation, never an array parent or job step."""

    if SLURM_ALLOCATION_ID_RE.fullmatch(job_id) is None:
        _fail(f"invalid Slurm allocation ID: {job_id!r}")
    evidence = [
        _sha256(value, f"{stage} evidence receipt SHA-256")
        for value in evidence_receipt_sha256s
    ]
    if not evidence or len(evidence) != len(set(evidence)):
        _fail(f"{stage} allocation must bind unique evidence receipts")
    command = [
        "sacct",
        "-n",
        "-X",
        "-P",
        "-j",
        job_id,
        "-o",
        "JobID,JobIDRaw,State,Partition,ElapsedRaw,AllocCPUS,CPUTimeRAW",
    ]
    try:
        output = subprocess.check_output(
            command, cwd=repo_root, text=True, stderr=subprocess.STDOUT
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        details = getattr(exc, "output", "")
        raise FamilyWorkflowError(
            "cannot verify completed cpu2dq sample allocation with sacct"
            + (f": {str(details).strip()}" if details else "")
        ) from exc
    rows: list[list[str]] = []
    for raw_line in output.splitlines():
        if not raw_line.strip():
            continue
        fields = [field.strip() for field in raw_line.split("|")]
        if len(fields) < 7:
            _fail(f"unexpected sacct allocation row: {raw_line!r}")
        if fields[0] == job_id:
            rows.append(fields[:7])
    if len(rows) != 1:
        _fail(f"sacct returned {len(rows)} exact allocation rows for {job_id}")
    row_job, job_id_raw, state, partition, elapsed_text, cpus_text, cpu_time_text = rows[0]
    if row_job != job_id or SLURM_ALLOCATION_ID_RE.fullmatch(job_id_raw) is None:
        _fail("sacct allocation identity is invalid")
    if state != "COMPLETED" or partition != "cpu2dq":
        _fail(
            f"allocation {job_id} must be COMPLETED on cpu2dq, got {state}/{partition}"
        )
    try:
        elapsed = int(elapsed_text)
        alloc_cpus = int(cpus_text)
        cpu_time = int(cpu_time_text)
    except ValueError as exc:
        raise FamilyWorkflowError(f"cannot parse sacct accounting for {job_id}") from exc
    if elapsed <= 0 or alloc_cpus != CPU2DQ_BILLABLE_CPUS:
        _fail(f"allocation {job_id} does not bind one full 128-CPU cpu2dq node")
    if cpu_time != elapsed * alloc_cpus:
        _fail(f"allocation {job_id} CPUTimeRAW arithmetic mismatch")
    return {
        "job_id": job_id,
        "job_id_raw": job_id_raw,
        "stage": stage,
        "state": state,
        "partition": partition,
        "elapsed_raw_seconds": elapsed,
        "alloc_cpus": alloc_cpus,
        "cpu_time_raw_seconds": cpu_time,
        "billed_cpu_saat": cpu_time / 3600.0,
        "evidence_receipt_sha256s": evidence,
        "sacct_output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
    }


def _load_sample_receipt_inventory(
    sample_run_dir: Path,
    *,
    object_ranks: Sequence[int],
    backend_report: Mapping[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
    objects: dict[int, dict[str, Any]] = {}
    for rank in object_ranks:
        receipt, _digest = _verify_sealed(
            sample_run_dir / "objects" / f"{rank:05d}" / "object_receipt.json",
            f"sample object receipt {rank}",
        )
        if receipt.get("rank") != rank or receipt.get("sample_mode") is not True:
            _fail(f"sample object receipt {rank} identity mismatch")
        objects[rank] = receipt
    report_object_hashes = list(
        _sequence(
            backend_report.get("sample_object_receipt_sha256"),
            "backend report sample object receipts",
        )
    )
    if report_object_hashes != [objects[rank]["canonical_sha256"] for rank in object_ranks]:
        _fail("backend report/sample object receipt inventory mismatch")
    buckets: dict[int, dict[str, Any]] = {}
    for rank in range(14):
        receipt, _digest = _verify_sealed(
            sample_run_dir / "bucket_receipts" / f"{rank:05d}.json",
            f"sample bucket receipt {rank}",
        )
        if receipt.get("rank") != rank or receipt.get("sample_mode") is not True:
            _fail(f"sample bucket receipt {rank} identity mismatch")
        buckets[rank] = receipt
    report_bucket_hashes = list(
        _sequence(
            backend_report.get("sample_bucket_receipt_sha256"),
            "backend report sample bucket receipts",
        )
    )
    if report_bucket_hashes != [buckets[rank]["canonical_sha256"] for rank in range(14)]:
        _fail("backend report/sample bucket receipt inventory mismatch")
    cluster, _cluster_sha = _verify_sealed(
        sample_run_dir / "cluster_receipt.json", "sample cluster receipt"
    )
    if cluster.get("sample_mode") is not True:
        _fail("sample cluster receipt is not a sample-mode receipt")
    if backend_report.get("sample_cluster_receipt_sha256") != cluster["canonical_sha256"]:
        _fail("backend report/sample cluster receipt binding mismatch")
    return objects, buckets, cluster


def _validate_writer_probe(
    probe: Mapping[str, Any],
    *,
    recipe: Mapping[str, Any],
    recipe_sha: str,
    policy_sha: str,
    source_plan_sha: str,
    calibration_sha: str,
    backend_report_sha: str,
    cluster_sha: str,
    sample_documents: int,
    estimated_total_documents: int,
) -> tuple[dict[str, dict[str, Any]], float]:
    verify_manifest_hash(probe)
    if probe.get("schema_version") != "1.0" or probe.get("kind") != DATA_PREP_WRITER_PROBE_KIND:
        _fail("unexpected post-cluster writer probe")
    bindings = {
        "family_id": recipe["family_id"],
        "recipe_sha256": recipe_sha,
        "policy_sha256": policy_sha,
        "source_plan_sha256": source_plan_sha,
        "calibration_sha256": calibration_sha,
        "backend_resource_report_sha256": backend_report_sha,
        "cluster_receipt_sha256": cluster_sha,
    }
    for field, expected in bindings.items():
        if probe.get(field) != expected:
            _fail(f"writer probe {field} binding mismatch")
    sample = _mapping(probe.get("sample"), "writer probe sample")
    _require_exact_keys(
        sample,
        {
            "candidate_documents",
            "accepted_documents",
            "pool_parquet_bytes",
            "train_parquet_bytes",
            "temporary_peak_bytes",
            "elapsed_wall_seconds",
            "process_cpu_seconds",
            "source_output_documents",
            "mixture_output_documents",
        },
        "writer probe sample",
    )
    if sample.get("candidate_documents") != sample_documents:
        _fail("writer probe candidate-document count mismatch")
    accepted = _positive_int(sample.get("accepted_documents"), "writer accepted documents")
    if accepted > sample_documents:
        _fail("writer probe accepted more documents than it received")
    pool_bytes = _positive_int(sample.get("pool_parquet_bytes"), "writer pool bytes")
    train_bytes = _positive_int(sample.get("train_parquet_bytes"), "writer train bytes")
    temporary_bytes = _positive_int(
        sample.get("temporary_peak_bytes"), "writer temporary peak bytes"
    )
    if train_bytes > pool_bytes:
        _fail("writer probe train bytes exceed its complete pool bytes")
    elapsed = _positive_number(sample.get("elapsed_wall_seconds"), "writer elapsed wall")
    _nonnegative_number(sample.get("process_cpu_seconds"), "writer process CPU")
    for label in ("source_output_documents", "mixture_output_documents"):
        counts = _mapping(sample.get(label), f"writer {label}")
        if not counts or any(
            not isinstance(key, str)
            or not key
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for key, value in counts.items()
        ):
            _fail(f"writer {label} must contain positive document counts")
        if sum(counts.values()) != accepted:
            _fail(f"writer {label} does not sum to accepted_documents")
    scale = estimated_total_documents / sample_documents
    projected_tokenized = math.ceil(train_bytes * scale)
    projected_temporary = temporary_bytes
    projected_cpu = elapsed * scale * CPU2DQ_BILLABLE_CPUS / 3600.0
    expected_projection = {
        "target_candidate_documents": estimated_total_documents,
        "document_scale": scale,
        "tokenized_output_bytes_before_safety": projected_tokenized,
        "tokenized_output_projection_basis": "linear_by_candidate_documents",
        "temporary_merge_bytes_before_safety": projected_temporary,
        "temporary_merge_projection_basis": "measured_bounded_fixed_peak",
        "billable_cpus_per_job": CPU2DQ_BILLABLE_CPUS,
        "materialization_and_finalization_cpu_saat_before_safety": projected_cpu,
        "cpu_projection_basis": (
            "sample_elapsed_wall_linear_by_candidate_documents_times_128_cpu2dq_cpus"
        ),
        "safety_factor_applied": False,
    }
    projection = _mapping(probe.get("projection"), "writer probe projection")
    if set(projection) != set(expected_projection):
        _fail("writer probe projection keys drifted")
    for field, expected in expected_projection.items():
        actual = projection.get(field)
        if isinstance(expected, float):
            if (
                isinstance(actual, bool)
                or not isinstance(actual, (int, float))
                or not math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-9)
            ):
                _fail(f"writer probe {field} arithmetic mismatch")
        elif actual != expected:
            _fail(f"writer probe {field} mismatch")
    evidence = [probe["canonical_sha256"], cluster_sha]
    components = {
        "tokenized_output": {
            "sample_measured_bytes": train_bytes,
            "projected_peak_bytes_before_safety": projected_tokenized,
            "projection_basis": "post_cluster_writer_probe_linear_by_candidate_documents",
            "evidence_sha256s": evidence,
        },
        "temporary_merge_space": {
            "sample_measured_bytes": temporary_bytes,
            "projected_peak_bytes_before_safety": projected_temporary,
            "projection_basis": "post_cluster_writer_probe_measured_bounded_fixed_peak",
            "evidence_sha256s": evidence,
        },
    }
    return components, projected_cpu


def command_seal_data_prep_writer_probe(args: argparse.Namespace) -> None:
    """Run the production pool writer over the bounded post-cluster sample."""

    if args.output.exists():
        _fail(f"refusing to overwrite post-cluster writer probe: {args.output}")
    (
        recipe,
        recipe_sha,
        policy,
        policy_sha,
        source_plan,
        source_plan_sha,
        _calibration,
        calibration_sha,
    ) = _load_data_prep_inputs(
        recipe_path=args.recipe,
        policy_path=args.policy,
        source_plan_path=args.source_plan,
        calibration_path=args.calibration,
    )
    try:
        import pyarrow.parquet as pq

        from nanochat.turkish_backend import validate_resource_projection
        from nanochat.turkish_corpus import (
            FragmentWriter,
            _production_lid_ok,
            _production_quality_ok,
            assign_split,
            audit_document,
            canonical_text_hash,
            dominant_register,
            select_mixture_bucket,
            stable_shuffle_key,
        )
    except ImportError as exc:
        raise FamilyWorkflowError("Turkish data writer environment is unavailable") from exc

    backend_report = _load_object(args.backend_resource_report, "backend resource report")
    try:
        backend_report_sha = validate_resource_projection(
            backend_report, plan=source_plan
        )
    except ValueError as exc:
        raise FamilyWorkflowError(f"invalid backend resource report: {exc}") from exc
    for field, expected in {
        "policy_sha256": policy_sha,
        "source_plan_sha256": source_plan_sha,
        "calibration_sha256": calibration_sha,
    }.items():
        if backend_report.get(field) != expected:
            _fail(f"backend resource report {field} binding mismatch")
    report_projection = _mapping(
        backend_report.get("projection"), "backend resource projection"
    )
    if report_projection.get("safety_factor") != 1.0:
        _fail("writer probe requires a pre-safety backend report")
    if backend_report.get("automated_gate_passed") is not True:
        _fail("backend resource report failed its automated storage gate")
    expected_ranks = list(
        _sequence(
            _mapping(backend_report.get("sample_selection"), "sample selection").get(
                "ranks"
            ),
            "backend report sample ranks",
        )
    )
    sample_run_dir = args.sample_run_dir.expanduser().resolve()
    objects, _buckets, cluster = _load_sample_receipt_inventory(
        sample_run_dir,
        object_ranks=expected_ranks,
        backend_report=backend_report,
    )
    sample_documents = sum(int(item["candidate_file"]["rows"]) for item in objects.values())
    estimated_total_documents = math.ceil(
        _positive_number(
            report_projection.get("candidate_documents"),
            "projected candidate documents",
        )
    )
    if sample_documents <= 0 or estimated_total_documents < sample_documents:
        _fail("writer probe document horizon is invalid")
    scratch_root = args.scratch_dir.expanduser().resolve()
    if not scratch_root.is_dir() or scratch_root.is_symlink():
        _fail("writer probe scratch directory must exist and not be a symlink")
    conservative_probe_scratch = (
        2 * sum(int(item["size_bytes"]) for item in cluster["output_files"])
        + 1_073_741_824
    )
    if shutil.disk_usage(scratch_root).free < conservative_probe_scratch:
        _fail("writer probe scratch has insufficient bounded free space")

    source_policies = {source["id"]: source for source in policy["sources"]}
    source_counts: Counter[str] = Counter()
    mixture_counts: Counter[str] = Counter()
    accepted = 0
    observed_candidates = 0
    start_wall = time.monotonic()
    start_cpu = time.process_time()
    with tempfile.TemporaryDirectory(prefix="d32-writer-probe-", dir=scratch_root) as raw:
        probe_root = Path(raw)
        writer = FragmentWriter(
            probe_root,
            rows_per_fragment=int(policy["materialization"]["rows_per_fragment"]),
            buckets=int(policy["materialization"]["shuffle_buckets"]),
            max_buffered_rows=int(policy["materialization"]["max_buffered_rows"]),
            rows_per_output_file=int(
                policy["materialization"]["rows_per_output_file"]
            ),
        )
        for file_record in cluster["output_files"]:
            path = sample_run_dir / str(file_record["path"])
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=2048):
                for record in batch.to_pylist():
                    observed_candidates += 1
                    source_id = str(record.get("source_id") or "")
                    if source_id not in source_policies:
                        continue
                    if record.get("dedup_keep") is not True:
                        continue
                    if not _production_lid_ok(record, policy) or not _production_quality_ok(
                        record
                    ):
                        continue
                    source_policy = source_policies[source_id]
                    adapter = source_policy["adapter"]
                    source_label = record.get("source_lid_label")
                    source_probability = record.get("source_lid_probability")
                    source_lid_ok = source_label in set(
                        adapter.get("turkish_values", ["tur", "tur_Latn"])
                    )
                    if source_probability is not None:
                        try:
                            source_lid_ok = source_lid_ok and float(
                                source_probability
                            ) >= float(adapter.get("source_lid_min_probability", 0.0))
                        except (TypeError, ValueError):
                            source_lid_ok = False
                    audit = audit_document(
                        record.get("text"),
                        url=str(record.get("url") or ""),
                        source_lid_ok=source_lid_ok,
                        content_policy=policy["content_policy"],
                    )
                    if not audit.accepted:
                        continue
                    candidate = select_mixture_bucket(source_id, record, policy)
                    if candidate is None:
                        continue
                    cluster_id = str(record.get("dedup_cluster_id") or "")
                    if SHA256_RE.fullmatch(cluster_id) is None:
                        _fail("writer probe encountered an invalid dedup cluster ID")
                    mixture_id, selector_quality = candidate
                    document_id = str(
                        record.get("document_id")
                        or canonical_text_hash(audit.normalized_text)
                    )
                    split = assign_split(cluster_id, policy["splits"])
                    writer.add(
                        split,
                        mixture_id,
                        {
                            "text": audit.normalized_text,
                            "source_id": source_id,
                            "mixture_id": mixture_id,
                            "document_id": document_id,
                            "url": str(record.get("url") or ""),
                            "cluster_id": cluster_id,
                            "shuffle_key": stable_shuffle_key(
                                document_id, policy["splits"]["seed"]
                            ),
                            "quality_score": max(
                                float(record.get("quality_score") or 0.0),
                                selector_quality,
                            ),
                            "register_bucket": dominant_register(record),
                        },
                    )
                    accepted += 1
                    source_counts[source_id] += 1
                    mixture_counts[mixture_id] += 1
        files = writer.close()
        elapsed_wall = time.monotonic() - start_wall
        process_cpu = time.process_time() - start_cpu
        if observed_candidates != sample_documents or accepted <= 0:
            _fail("writer probe candidate/accepted document accounting mismatch")
        pool_bytes = sum(int(item["size_bytes"]) for item in files)
        train_files = [item for item in files if item["split"] == "train"]
        train_bytes = sum(int(item["size_bytes"]) for item in train_files)
        if pool_bytes <= 0 or train_bytes <= 0:
            _fail("writer probe emitted no usable pool/train Parquet bytes")
        maximum_train_fragment_by_mixture = {
            mixture_id: max(
                (
                    int(item["size_bytes"])
                    for item in train_files
                    if item["mixture_id"] == mixture_id
                ),
                default=0,
            )
            for mixture_id in mixture_counts
        }
        eval_bytes = sum(
            int(item["size_bytes"])
            for item in files
            if item["split"] in {"val", "test"}
        )
        temporary_peak = (
            sum(maximum_train_fragment_by_mixture.values())
            + math.ceil(eval_bytes * 1.5)
            + 1_073_741_824
        )

    scale = estimated_total_documents / sample_documents
    projected_cpu = elapsed_wall * scale * CPU2DQ_BILLABLE_CPUS / 3600.0
    receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": DATA_PREP_WRITER_PROBE_KIND,
            "family_id": recipe["family_id"],
            "recipe_sha256": recipe_sha,
            "policy_sha256": policy_sha,
            "source_plan_sha256": source_plan_sha,
            "calibration_sha256": calibration_sha,
            "backend_resource_report_sha256": backend_report_sha,
            "cluster_receipt_sha256": cluster["canonical_sha256"],
            "sample": {
                "candidate_documents": sample_documents,
                "accepted_documents": accepted,
                "pool_parquet_bytes": pool_bytes,
                "train_parquet_bytes": train_bytes,
                "temporary_peak_bytes": temporary_peak,
                "elapsed_wall_seconds": elapsed_wall,
                "process_cpu_seconds": process_cpu,
                "source_output_documents": dict(sorted(source_counts.items())),
                "mixture_output_documents": dict(sorted(mixture_counts.items())),
            },
            "projection": {
                "target_candidate_documents": estimated_total_documents,
                "document_scale": scale,
                "tokenized_output_bytes_before_safety": math.ceil(train_bytes * scale),
                "tokenized_output_projection_basis": "linear_by_candidate_documents",
                "temporary_merge_bytes_before_safety": temporary_peak,
                "temporary_merge_projection_basis": "measured_bounded_fixed_peak",
                "billable_cpus_per_job": CPU2DQ_BILLABLE_CPUS,
                "materialization_and_finalization_cpu_saat_before_safety": (
                    projected_cpu
                ),
                "cpu_projection_basis": (
                    "sample_elapsed_wall_linear_by_candidate_documents_times_128_cpu2dq_cpus"
                ),
                "safety_factor_applied": False,
            },
            "canonical_sha256": None,
        }
    )
    _validate_writer_probe(
        receipt,
        recipe=recipe,
        recipe_sha=recipe_sha,
        policy_sha=policy_sha,
        source_plan_sha=source_plan_sha,
        calibration_sha=calibration_sha,
        backend_report_sha=backend_report_sha,
        cluster_sha=cluster["canonical_sha256"],
        sample_documents=sample_documents,
        estimated_total_documents=estimated_total_documents,
    )
    write_json_atomic(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))


def _validate_sample_lane_plan(
    lane_plan: Mapping[str, Any],
    *,
    policy_sha: str,
    source_plan_sha: str,
    calibration_sha: str,
    expected_ranks: Sequence[int],
) -> tuple[dict[int, list[int]], int, int]:
    """Validate the packed sample allocation-to-object-rank ownership map."""

    verify_manifest_hash(lane_plan)
    lane_count = _positive_int(lane_plan.get("lane_count"), "sample lane count")
    cpus_per_lane = _positive_int(
        lane_plan.get("cpus_per_lane"), "sample CPUs per lane"
    )
    if (
        lane_plan.get("schema_version") != "1.0"
        or lane_plan.get("kind") != "turkish_packed_resource_sample_lane_plan"
        or lane_plan.get("sample_mode") is not True
        or lane_plan.get("assignment_algorithm") != "sorted_rank_round_robin_v1"
        or lane_count != 32
        or cpus_per_lane != 4
        or lane_count * cpus_per_lane != CPU2DQ_BILLABLE_CPUS
    ):
        _fail("unexpected packed resource-sample lane plan")
    bindings = {
        "policy_sha256": policy_sha,
        "source_plan_sha256": source_plan_sha,
        "calibration_sha256": calibration_sha,
    }
    for field, expected in bindings.items():
        if lane_plan.get(field) != expected:
            _fail(f"sample lane-plan {field} binding mismatch")
    raw_lanes = _sequence(lane_plan.get("lanes"), "sample lane-plan lanes")
    if len(raw_lanes) != lane_count:
        _fail("sample lane-plan lane inventory does not match its geometry")
    lanes: dict[int, list[int]] = {}
    seen: list[int] = []
    for expected_lane_id, raw in enumerate(raw_lanes):
        lane = _mapping(raw, f"sample lane {expected_lane_id}")
        lane_id = lane.get("lane_id")
        ranks_value = lane.get("object_ranks", lane.get("ranks"))
        if lane_id != expected_lane_id:
            _fail("sample lane IDs must be contiguous")
        ranks = list(_sequence(ranks_value, f"sample lane {lane_id} ranks"))
        if ranks != sorted(ranks) or any(
            isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
            for rank in ranks
        ):
            _fail("sample lane ranks are invalid or unsorted")
        lanes[int(lane_id)] = ranks
        seen.extend(ranks)
    if sorted(seen) != sorted(expected_ranks) or len(seen) != len(set(seen)):
        _fail("sample lane plan must cover every sampled source rank exactly once")
    rank_binding = _mapping(
        lane_plan.get("resource_sample_ranks"), "sample lane-plan rank binding"
    )
    _sha256(rank_binding.get("file_sha256"), "resource sample-ranks file SHA-256")
    declared_ranks = _sequence(
        rank_binding.get("ranks"), "sample lane-plan declared ranks"
    )
    if list(declared_ranks) != sorted(expected_ranks):
        _fail("sample lane-plan declared rank inventory mismatch")
    return lanes, lane_count, cpus_per_lane


def _validate_packed_sample_launch_receipt(
    receipt: Mapping[str, Any],
    *,
    job_id: str,
    lane_plan_sha: str,
    policy_sha: str,
    source_plan_sha: str,
    calibration_sha: str,
    objects: Mapping[int, Mapping[str, Any]],
    lane_count: int,
    cpus_per_lane: int,
) -> str:
    digest = verify_manifest_hash(receipt)
    if (
        receipt.get("schema_version") != "1.0"
        or receipt.get("kind") != "turkish_packed_resource_sample_launch_receipt"
        or receipt.get("sample_mode") is not True
        or receipt.get("all_lanes_completed") is not True
    ):
        _fail("unexpected packed object-sample launch receipt")
    bindings = {
        "lane_plan_sha256": lane_plan_sha,
        "policy_sha256": policy_sha,
        "source_plan_sha256": source_plan_sha,
        "calibration_sha256": calibration_sha,
    }
    for field, expected in bindings.items():
        if receipt.get(field) != expected:
            _fail(f"packed object-sample launch {field} binding mismatch")
    allocation = _mapping(receipt.get("allocation"), "packed sample allocation")
    if allocation != {
        "slurm_job_id": job_id,
        "slurm_step_id": allocation.get("slurm_step_id"),
        "slurm_node": allocation.get("slurm_node"),
        "nodes": 1,
        "tasks": lane_count,
        "cpus_per_task": cpus_per_lane,
        "allocated_cpus": 128,
    }:
        _fail("packed object-sample launch allocation geometry mismatch")
    if (
        not isinstance(allocation.get("slurm_step_id"), str)
        or not allocation["slurm_step_id"]
        or not isinstance(allocation.get("slurm_node"), str)
        or not allocation["slurm_node"]
    ):
        _fail("packed object-sample launch lacks Slurm step/node identity")
    lane_records = _sequence(receipt.get("lane_receipts"), "packed lane receipts")
    if len(lane_records) != lane_count:
        _fail("packed object-sample launch lane inventory drifted")
    for lane_id, raw in enumerate(lane_records):
        record = _mapping(raw, f"packed lane record {lane_id}")
        if record.get("lane_id") != lane_id:
            _fail("packed object-sample lane receipt order drifted")
        _sha256(record.get("canonical_sha256"), "packed lane receipt SHA-256")
    object_records = _sequence(
        receipt.get("object_receipts"), "packed launch object receipts"
    )
    if [record.get("rank") for record in object_records if isinstance(record, Mapping)] != sorted(
        objects
    ):
        _fail("packed launch object-rank inventory mismatch")
    for raw in object_records:
        record = _mapping(raw, "packed launch object record")
        rank = int(record["rank"])
        if record.get("canonical_sha256") != objects[rank]["canonical_sha256"]:
            _fail(f"packed launch object receipt {rank} hash mismatch")
    return digest


def _validate_packed_bucket_launch_receipt(
    receipt: Mapping[str, Any],
    *,
    job_id: str,
    object_launch_sha: str,
    policy_sha: str,
    source_plan_sha: str,
    calibration_sha: str,
    buckets: Mapping[int, Mapping[str, Any]],
) -> str:
    digest = verify_manifest_hash(receipt)
    if (
        receipt.get("schema_version") != "1.0"
        or receipt.get("kind") != "turkish_packed_sample_bucket_launch_receipt"
        or receipt.get("sample_mode") is not True
        or receipt.get("all_buckets_completed") is not True
    ):
        _fail("unexpected packed bucket-sample launch receipt")
    bindings = {
        "policy_sha256": policy_sha,
        "source_plan_sha256": source_plan_sha,
        "calibration_sha256": calibration_sha,
        "object_sample_launch_receipt_sha256": object_launch_sha,
    }
    for field, expected in bindings.items():
        if receipt.get(field) != expected:
            _fail(f"packed bucket-sample launch {field} binding mismatch")
    if receipt.get("assignment") != {
        "algorithm": "slurm_procid_equals_minhash_bucket_rank_v1",
        "bucket_ranks": list(range(14)),
        "world_size": 14,
    }:
        _fail("packed bucket-sample assignment drifted")
    allocation = _mapping(receipt.get("allocation"), "packed bucket allocation")
    if allocation != {
        "slurm_job_id": job_id,
        "slurm_step_id": allocation.get("slurm_step_id"),
        "slurm_node": allocation.get("slurm_node"),
        "nodes": 1,
        "tasks": 14,
        "cpus_per_task": 8,
        "allocated_cpus": 112,
    }:
        _fail("packed bucket-sample launch allocation geometry mismatch")
    if (
        not isinstance(allocation.get("slurm_step_id"), str)
        or not allocation["slurm_step_id"]
        or not isinstance(allocation.get("slurm_node"), str)
        or not allocation["slurm_node"]
    ):
        _fail("packed bucket-sample launch lacks Slurm step/node identity")
    task_records = _sequence(receipt.get("task_receipts"), "packed bucket tasks")
    backend_records = _sequence(
        receipt.get("backend_bucket_receipts"), "packed backend bucket receipts"
    )
    if len(task_records) != 14 or len(backend_records) != 14:
        _fail("packed bucket launch must bind fourteen task/backend receipts")
    for rank in range(14):
        task = _mapping(task_records[rank], f"packed bucket task record {rank}")
        backend = _mapping(
            backend_records[rank], f"packed backend bucket record {rank}"
        )
        if task.get("bucket_rank") != rank or backend.get("bucket_rank") != rank:
            _fail("packed bucket receipt order drifted")
        _sha256(task.get("canonical_sha256"), "packed bucket task SHA-256")
        if backend.get("canonical_sha256") != buckets[rank]["canonical_sha256"]:
            _fail(f"packed backend bucket receipt {rank} hash mismatch")
    return digest


def _validate_data_prep_storage_sample(
    measurement: Mapping[str, Any],
    *,
    recipe: Mapping[str, Any] | None = None,
    recipe_sha: str | None = None,
) -> str:
    digest = verify_manifest_hash(measurement)
    if (
        measurement.get("schema_version") != "3.0"
        or measurement.get("kind") != DATA_PREP_STORAGE_SAMPLE_KIND
    ):
        _fail("data-preparation storage sample has the wrong kind/version")
    if recipe is not None:
        if measurement.get("family_id") != recipe["family_id"]:
            _fail("data-preparation sample family binding mismatch")
        if measurement.get("recipe_sha256") != recipe_sha:
            _fail("data-preparation sample recipe binding mismatch")
    for field in (
        "recipe_sha256",
        "policy_sha256",
        "source_plan_sha256",
        "calibration_sha256",
        "backend_resource_report_sha256",
        "resource_approval_sha256",
        "mixture_quality_approval_sha256",
        "sample_quality_audit_sha256",
        "sample_cluster_receipt_sha256",
        "sample_lane_plan_sha256",
        "production_pack_plan_sha256",
        "writer_probe_sha256",
        "macocu_preparation_manifest_sha256",
    ):
        _sha256(measurement.get(field), f"data-preparation sample {field}")
    sample_documents = _positive_int(
        measurement.get("sample_documents"), "sample_documents"
    )
    total_documents = _positive_int(
        measurement.get("estimated_total_documents"), "estimated_total_documents"
    )
    if total_documents < sample_documents:
        _fail("estimated total documents cannot be smaller than the sample")

    components = _mapping(measurement.get("components"), "sample components")
    if set(components) != set(DATA_PREP_STORAGE_COMPONENTS):
        _fail("data-preparation sample storage component set mismatch")
    for name in DATA_PREP_STORAGE_COMPONENTS:
        record = _mapping(components[name], f"sample component {name}")
        _require_exact_keys(
            record,
            {
                "sample_measured_bytes",
                "projected_peak_bytes_before_safety",
                "projection_basis",
                "evidence_sha256s",
            },
            f"sample component {name}",
        )
        _nonnegative_int(record.get("sample_measured_bytes"), f"{name}.sample bytes")
        _nonnegative_int(
            record.get("projected_peak_bytes_before_safety"),
            f"{name}.projected bytes",
        )
        if not isinstance(record.get("projection_basis"), str) or not record[
            "projection_basis"
        ]:
            _fail(f"{name}.projection_basis must be non-empty")
        evidence = _sequence(record.get("evidence_sha256s"), f"{name} evidence")
        if not evidence or len(evidence) != len(set(evidence)):
            _fail(f"{name} evidence SHA-256 inventory is empty or duplicated")
        for value in evidence:
            _sha256(value, f"{name} evidence SHA-256")

    allocations = _sequence(
        measurement.get("sample_allocations"), "sample allocation ledger"
    )
    if not allocations:
        _fail("sample allocation ledger is empty")
    job_ids: set[str] = set()
    raw_ids: set[str] = set()
    total_billed = 0.0
    total_cpu_time = 0
    expected_allocation_keys = {
        "job_id",
        "job_id_raw",
        "stage",
        "state",
        "partition",
        "elapsed_raw_seconds",
        "alloc_cpus",
        "cpu_time_raw_seconds",
        "billed_cpu_saat",
        "evidence_receipt_sha256s",
        "sacct_output_sha256",
    }
    for index, raw in enumerate(allocations):
        allocation = _mapping(raw, f"sample allocation {index}")
        _require_exact_keys(
            allocation, expected_allocation_keys, f"sample allocation {index}"
        )
        job_id = allocation.get("job_id")
        job_id_raw = allocation.get("job_id_raw")
        if (
            not isinstance(job_id, str)
            or SLURM_ALLOCATION_ID_RE.fullmatch(job_id) is None
            or not isinstance(job_id_raw, str)
            or SLURM_ALLOCATION_ID_RE.fullmatch(job_id_raw) is None
        ):
            _fail("sample allocation contains an invalid Slurm identity")
        if job_id in job_ids or job_id_raw in raw_ids:
            _fail("sample allocation ledger contains a duplicate allocation ID")
        job_ids.add(job_id)
        raw_ids.add(job_id_raw)
        _safe_id(allocation.get("stage"), "sample allocation stage")
        if allocation.get("state") != "COMPLETED" or allocation.get("partition") != "cpu2dq":
            _fail("sample allocation must be a completed cpu2dq allocation")
        elapsed = _positive_int(
            allocation.get("elapsed_raw_seconds"), "allocation elapsed seconds"
        )
        if allocation.get("alloc_cpus") != CPU2DQ_BILLABLE_CPUS:
            _fail("sample allocation does not bind 128 billed CPUs")
        cpu_time = _positive_int(
            allocation.get("cpu_time_raw_seconds"), "allocation CPUTimeRAW"
        )
        if cpu_time != elapsed * CPU2DQ_BILLABLE_CPUS:
            _fail("sample allocation CPUTimeRAW arithmetic mismatch")
        billed = _positive_number(
            allocation.get("billed_cpu_saat"), "sample allocation billed CPU-saat"
        )
        if not math.isclose(billed, cpu_time / 3600.0, rel_tol=1e-12, abs_tol=1e-9):
            _fail("sample allocation billed CPU-saat arithmetic mismatch")
        evidence = _sequence(
            allocation.get("evidence_receipt_sha256s"), "allocation evidence"
        )
        if not evidence or len(evidence) != len(set(evidence)):
            _fail("sample allocation evidence is empty or duplicated")
        for value in evidence:
            _sha256(value, "sample allocation evidence SHA-256")
        _sha256(allocation.get("sacct_output_sha256"), "sacct output SHA-256")
        total_billed += billed
        total_cpu_time += cpu_time
    totals = _mapping(
        measurement.get("sample_allocation_totals"), "sample allocation totals"
    )
    expected_totals = {
        "unique_allocations": len(allocations),
        "cpu_time_raw_seconds": total_cpu_time,
        "billed_cpu_saat": total_billed,
        "accounting_role": "already_consumed_measurement_evidence_not_future_quota",
    }
    if set(totals) != set(expected_totals):
        _fail("sample allocation total keys drifted")
    for field, expected in expected_totals.items():
        actual = totals.get(field)
        if isinstance(expected, float):
            if not math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-9):
                _fail("sample allocation billed total arithmetic mismatch")
        elif actual != expected:
            _fail(f"sample allocation total {field} mismatch")

    historical = _sequence(
        measurement.get("historical_one_time_preparations"),
        "historical one-time preparations",
    )
    if len(historical) != 1:
        _fail("exactly one historical MaCoCu preparation is required")
    macocu = _mapping(historical[0], "historical MaCoCu preparation")
    _require_exact_keys(
        macocu,
        {
            "preparation_id",
            "manifest_sha256",
            "allocation",
            "accounting_status",
            "future_projected_cpu_saat",
        },
        "historical MaCoCu preparation",
    )
    if (
        macocu.get("preparation_id") != "macocu_genre_tr_v1"
        or macocu.get("manifest_sha256")
        != measurement.get("macocu_preparation_manifest_sha256")
        or macocu.get("accounting_status")
        != "already_consumed_excluded_from_future_projection"
        or macocu.get("future_projected_cpu_saat") != 0
    ):
        _fail("historical MaCoCu accounting contract drifted")
    historical_allocation = _mapping(macocu.get("allocation"), "MaCoCu allocation")
    _require_exact_keys(
        historical_allocation,
        expected_allocation_keys,
        "historical MaCoCu allocation",
    )
    if historical_allocation.get("job_id_raw") in raw_ids:
        _fail("historical preparation allocation duplicates the sample ledger")
    if (
        not isinstance(historical_allocation.get("job_id"), str)
        or SLURM_ALLOCATION_ID_RE.fullmatch(historical_allocation["job_id"])
        is None
        or not isinstance(historical_allocation.get("job_id_raw"), str)
        or SLURM_ALLOCATION_ID_RE.fullmatch(historical_allocation["job_id_raw"])
        is None
        or historical_allocation.get("stage") != "macocu_genre_preparation"
    ):
        _fail("historical MaCoCu allocation identity drifted")
    # Reuse the exact allocation arithmetic validator by checking its essential fields.
    hist_elapsed = _positive_int(
        historical_allocation.get("elapsed_raw_seconds"), "MaCoCu elapsed seconds"
    )
    hist_cpu_time = _positive_int(
        historical_allocation.get("cpu_time_raw_seconds"), "MaCoCu CPUTimeRAW"
    )
    if (
        historical_allocation.get("state") != "COMPLETED"
        or historical_allocation.get("partition") != "cpu2dq"
        or historical_allocation.get("alloc_cpus") != CPU2DQ_BILLABLE_CPUS
        or hist_cpu_time != hist_elapsed * CPU2DQ_BILLABLE_CPUS
        or not math.isclose(
            float(historical_allocation.get("billed_cpu_saat", -1)),
            hist_cpu_time / 3600.0,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
    ):
        _fail("historical MaCoCu allocation accounting drifted")
    evidence = _sequence(
        historical_allocation.get("evidence_receipt_sha256s"),
        "historical MaCoCu evidence",
    )
    if evidence != [measurement.get("macocu_preparation_manifest_sha256")]:
        _fail("historical MaCoCu allocation must bind only its manifest")
    _sha256(historical_allocation.get("sacct_output_sha256"), "MaCoCu sacct SHA-256")

    future = _mapping(
        measurement.get("future_resource_projection"), "future resource projection"
    )
    _require_exact_keys(
        future,
        {
            "components",
            "allocation_details",
            "projected_cpu_saat_before_safety",
            "safety_factor_applied",
            "excluded_historical_and_sample_allocations",
        },
        "future resource projection",
    )
    future_components = _mapping(
        future.get("components"), "future resource projection components"
    )
    if set(future_components) != set(DATA_PREP_FUTURE_CPU_COMPONENTS):
        _fail("future resource projection component set mismatch")
    allocation_details = _mapping(
        future.get("allocation_details"), "future allocation details"
    )
    if set(allocation_details) != set(DATA_PREP_FUTURE_CPU_COMPONENTS):
        _fail("future allocation-detail component set mismatch")
    for name in DATA_PREP_FUTURE_CPU_COMPONENTS:
        if not _mapping(allocation_details[name], f"future allocation details {name}"):
            _fail("future allocation details must not be empty")
    for name, ceiling in DATA_PREP_FIXED_CPU2DQ_CEILINGS.items():
        expected_details = {
            "allocation_contract": "one_exclusive_128cpu_cpu2dq_node",
            "maximum_wall_hours": ceiling / CPU2DQ_BILLABLE_CPUS,
            "projected_cpu_saat_before_safety": float(ceiling),
            "submission_must_not_exceed_ceiling": True,
        }
        if allocation_details.get(name) != expected_details:
            _fail(f"future {name} allocation ceiling drifted")
    projected_cpu = 0.0
    for name in DATA_PREP_FUTURE_CPU_COMPONENTS:
        record = _mapping(future_components[name], f"future CPU component {name}")
        _require_exact_keys(
            record,
            {
                "projected_cpu_saat_before_safety",
                "projection_basis",
                "evidence_sha256s",
            },
            f"future CPU component {name}",
        )
        component_cpu = _positive_number(
            record.get("projected_cpu_saat_before_safety"),
            f"future CPU component {name}",
        )
        projected_cpu += component_cpu
        if name in DATA_PREP_FIXED_CPU2DQ_CEILINGS and not math.isclose(
            component_cpu,
            float(DATA_PREP_FIXED_CPU2DQ_CEILINGS[name]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            _fail(f"future {name} CPU ceiling drifted")
        if not isinstance(record.get("projection_basis"), str) or not record[
            "projection_basis"
        ]:
            _fail("future CPU projection basis is empty")
        component_evidence = _sequence(
            record.get("evidence_sha256s"), f"future CPU {name} evidence"
        )
        if not component_evidence or len(component_evidence) != len(
            set(component_evidence)
        ):
            _fail("future CPU evidence is empty or duplicated")
        for value in component_evidence:
            _sha256(value, f"future CPU {name} evidence SHA-256")
    if not math.isclose(
        float(future.get("projected_cpu_saat_before_safety", -1)),
        projected_cpu,
        rel_tol=1e-12,
        abs_tol=1e-9,
    ):
        _fail("future projected CPU-saat arithmetic mismatch")
    if (
        future.get("safety_factor_applied") is not False
        or future.get("excluded_historical_and_sample_allocations") is not True
    ):
        _fail("future resource projection must exclude history and be pre-safety")
    return digest


def _evidence_record_under(root: Path, path_value: Path, label: str) -> dict[str, Any]:
    source = path_value.expanduser()
    if source.is_symlink() or not source.is_file():
        _fail(f"{label} evidence is unsafe or missing")
    root = root.resolve()
    path = source.resolve()
    if path.parent != root and root not in path.parents:
        _fail(f"{label} evidence must remain inside the receipt directory tree")
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _load_evidence_under(
    root: Path, raw: Any, label: str
) -> tuple[Path, dict[str, Any], bytes]:
    from nanochat.turkish_backend import (
        _MAX_AUDIT_REPORT_BYTES,
        _read_bounded_regular_file_snapshot,
    )

    record = dict(_mapping(raw, f"{label} evidence record"))
    relative = Path(str(record.get("path") or ""))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        _fail(f"{label} evidence path is unsafe")
    unresolved = root.resolve() / relative
    path = unresolved.resolve()
    if root.resolve() not in path.parents:
        _fail(f"{label} evidence is missing or escapes its receipt tree")
    current = unresolved
    while current != root.resolve():
        if current.is_symlink():
            _fail(f"{label} evidence path is symlinked")
        current = current.parent
    if (
        isinstance(record.get("size_bytes"), bool)
        or not isinstance(record.get("size_bytes"), int)
        or record["size_bytes"] < 0
        or record["size_bytes"] > _MAX_AUDIT_REPORT_BYTES
    ):
        _fail(f"{label} evidence content drift")
    try:
        snapshot = _read_bounded_regular_file_snapshot(
            path, label=label, max_bytes=_MAX_AUDIT_REPORT_BYTES
        )
    except ValueError as exc:
        raise FamilyWorkflowError(str(exc)) from exc
    if (
        len(snapshot) != record["size_bytes"]
        or hashlib.sha256(snapshot).hexdigest()
        != _sha256(record.get("sha256"), f"{label} SHA-256")
    ):
        _fail(f"{label} evidence content drift")
    return path, record, snapshot


def _validate_storage_approval_evidence(
    measurement: Mapping[str, Any],
    *,
    measurement_path: Path,
    policy_path: Path,
) -> None:
    """Re-open the real approvals/audit bundle before sealing the storage gate."""

    try:
        from nanochat.turkish_backend import (
            _load_json_snapshot,
            validate_backend_calibration,
            validate_mixture_quality_approval,
            validate_resource_approval,
            validate_resource_projection,
            validate_source_plan,
        )
        from nanochat.turkish_corpus import load_corpus_policy
    except ImportError as exc:
        raise FamilyWorkflowError("Turkish data environment is unavailable") from exc
    policy = load_corpus_policy(policy_path)
    root = measurement_path.expanduser().resolve().parent
    bundle = _mapping(
        measurement.get("approval_evidence"), "storage sample approval_evidence"
    )
    if bundle.get("schema_version") != "1.0":
        _fail("storage sample approval evidence version drift")
    required = {
        "schema_version",
        "source_plan",
        "calibration",
        "backend_resource_report",
        "resource_approval",
        "mixture_quality_approval",
    }
    if set(bundle) != required:
        _fail("storage sample approval evidence inventory drift")
    source_plan_path, _, source_plan_raw = _load_evidence_under(
        root, bundle["source_plan"], "source plan"
    )
    calibration_path, _, calibration_raw = _load_evidence_under(
        root, bundle["calibration"], "calibration"
    )
    report_path, _, report_raw = _load_evidence_under(
        root, bundle["backend_resource_report"], "backend resource report"
    )
    resource_path, _, resource_raw = _load_evidence_under(
        root, bundle["resource_approval"], "resource approval"
    )
    mixture_path, _, mixture_raw = _load_evidence_under(
        root, bundle["mixture_quality_approval"], "mixture-quality approval"
    )
    source_plan = _load_json_snapshot(source_plan_raw, "storage evidence source plan")
    calibration = _load_json_snapshot(
        calibration_raw, "storage evidence calibration"
    )
    if not isinstance(source_plan, Mapping) or not isinstance(calibration, Mapping):
        _fail("storage source-plan/calibration evidence must be JSON objects")
    validate_source_plan(source_plan, policy)
    validate_backend_calibration(calibration, policy)
    report = _load_json_snapshot(report_raw, "storage evidence backend report")
    report_sha = validate_resource_projection(report, plan=source_plan)
    mixture = _load_json_snapshot(mixture_raw, "storage evidence mixture approval")
    mixture_sha = validate_mixture_quality_approval(
        mixture,
        policy=policy,
        plan=source_plan,
        calibration=calibration,
        approval_path=mixture_path,
    )
    resource = _load_json_snapshot(resource_raw, "storage evidence resource approval")
    validate_resource_approval(
        resource,
        plan=source_plan,
        policy=policy,
        calibration=calibration,
        approval_path=resource_path,
    )
    exact = {
        "policy_sha256": _data_prep_policy_sha256(policy),
        "source_plan_sha256": source_plan["canonical_sha256"],
        "calibration_sha256": calibration["canonical_sha256"],
        "backend_resource_report_sha256": report_sha,
        "resource_approval_sha256": resource["canonical_sha256"],
        "mixture_quality_approval_sha256": mixture_sha,
        "sample_quality_audit_sha256": mixture["sample_quality_audit_sha256"],
        "sample_cluster_receipt_sha256": mixture[
            "sample_cluster_receipt_sha256"
        ],
    }
    sample_cluster_sha = exact["sample_cluster_receipt_sha256"]
    if (
        report.get("sample_cluster_receipt_sha256") != sample_cluster_sha
        or resource.get("sample_cluster_receipt_sha256") != sample_cluster_sha
    ):
        _fail("storage sample approval chain crosses sample-cluster lineages")
    for field, expected in exact.items():
        if measurement.get(field) != expected:
            _fail(f"storage sample actual evidence {field} binding mismatch")


def command_seal_data_prep_storage_sample(args: argparse.Namespace) -> None:
    if args.output.exists():
        _fail(f"refusing to overwrite data-preparation storage sample: {args.output}")
    (
        recipe,
        recipe_sha,
        policy,
        policy_sha,
        source_plan,
        source_plan_sha,
        calibration,
        calibration_sha,
    ) = _load_data_prep_inputs(
        recipe_path=args.recipe,
        policy_path=args.policy,
        source_plan_path=args.source_plan,
        calibration_path=args.calibration,
    )
    try:
        from nanochat.turkish_backend import (
            select_resource_sample_ranks,
            validate_macocu_preparation_manifest,
            validate_mixture_quality_approval,
            validate_resource_approval,
            validate_resource_projection,
        )
        from scripts.turkish_packed_sample import (
            _validate_bucket_task_receipt,
            _validate_lane_receipt,
            validate_lane_plan,
        )
    except ImportError as exc:
        raise FamilyWorkflowError("Turkish data environment is unavailable") from exc

    backend_report = _load_object(args.backend_resource_report, "backend resource report")
    try:
        backend_report_sha = validate_resource_projection(
            backend_report, plan=source_plan
        )
    except ValueError as exc:
        raise FamilyWorkflowError(f"invalid backend resource report: {exc}") from exc
    expected_report_bindings = {
        "policy_sha256": policy_sha,
        "source_plan_sha256": source_plan_sha,
        "calibration_sha256": calibration_sha,
    }
    for field, expected in expected_report_bindings.items():
        if backend_report.get(field) != expected:
            _fail(f"backend resource report {field} binding mismatch")
    report_projection = _mapping(
        backend_report.get("projection"), "backend resource projection"
    )
    if report_projection.get("safety_factor") != 1.0:
        _fail("backend resource report must be pre-safety; use --safety-factor=1")
    if backend_report.get("automated_gate_passed") is not True:
        _fail("backend resource report failed its automated storage gate")

    mixture_quality_approval, mixture_quality_approval_sha = _verify_sealed(
        args.mixture_quality_approval, "mixture-quality approval"
    )
    try:
        validate_mixture_quality_approval(
            mixture_quality_approval,
            policy=policy,
            plan=source_plan,
            calibration=calibration,
            approval_path=args.mixture_quality_approval,
        )
    except ValueError as exc:
        raise FamilyWorkflowError(f"invalid mixture-quality approval: {exc}") from exc
    resource_approval, resource_approval_sha = _verify_sealed(
        args.resource_approval, "backend resource approval"
    )
    try:
        validate_resource_approval(
            resource_approval,
            plan=source_plan,
            policy=policy,
            calibration=calibration,
            approval_path=args.resource_approval,
        )
    except ValueError as exc:
        raise FamilyWorkflowError(f"invalid backend resource approval: {exc}") from exc
    if (
        resource_approval.get("resource_report_sha256") != backend_report_sha
        or resource_approval.get("mixture_quality_approval_sha256")
        != mixture_quality_approval_sha
        or resource_approval.get("sample_cluster_receipt_sha256")
        != backend_report.get("sample_cluster_receipt_sha256")
        or mixture_quality_approval.get("sample_cluster_receipt_sha256")
        != backend_report.get("sample_cluster_receipt_sha256")
    ):
        _fail("resource approval does not bind the supplied report/quality approval")

    macocu, macocu_sha = _verify_sealed(
        args.macocu_manifest, "MaCoCu preparation manifest"
    )
    try:
        validate_macocu_preparation_manifest(
            macocu, policy, args.macocu_manifest.parent, verify_files=False
        )
    except ValueError as exc:
        raise FamilyWorkflowError(f"invalid MaCoCu preparation manifest: {exc}") from exc
    derived = _mapping(source_plan.get("derived_sources"), "source-plan derived sources")
    macocu_binding = _mapping(
        derived.get("macocu_genre_tr"), "source-plan MaCoCu binding"
    )
    if macocu_binding.get("manifest_sha256") != macocu_sha:
        _fail("source plan is not bound to the supplied MaCoCu preparation")

    sample_selection = _mapping(
        backend_report.get("sample_selection"), "sample selection"
    )
    expected_object_ranks = list(
        _sequence(sample_selection.get("ranks"), "backend report sample ranks")
    )
    selected_by_current_code = select_resource_sample_ranks(source_plan)
    expected_hplt_objects = [
        {
            "rank": item["rank"],
            "wds_bin": item["wds_bin"],
            "size_bytes": item["size_bytes"],
            "uri": item["uri"],
        }
        for item in source_plan["objects"]
        if item["source_id"] == "hplt3_tr"
        and item["rank"] in selected_by_current_code
    ]
    if (
        expected_object_ranks != selected_by_current_code
        or sample_selection.get("covers_every_source") is not True
        or sample_selection.get("size_based_selection") is not True
        or sample_selection.get("hplt_selected_objects") != expected_hplt_objects
    ):
        _fail("backend resource report sample selection drifted from current code")
    lane_plan, lane_plan_sha = _verify_sealed(
        args.sample_lane_plan, "packed resource-sample lane plan"
    )
    try:
        validate_lane_plan(
            lane_plan,
            policy=policy,
            source_plan=source_plan,
            calibration=calibration,
            sample_ranks_path=args.sample_ranks,
        )
    except ValueError as exc:
        raise FamilyWorkflowError(f"invalid packed resource-sample lane plan: {exc}") from exc
    sample_lanes, sample_lane_count, sample_cpus_per_lane = _validate_sample_lane_plan(
        lane_plan,
        policy_sha=policy_sha,
        source_plan_sha=source_plan_sha,
        calibration_sha=calibration_sha,
        expected_ranks=expected_object_ranks,
    )
    sample_run_dir = args.sample_run_dir.expanduser().resolve()
    if not sample_run_dir.is_dir() or sample_run_dir.is_symlink():
        _fail("sample run directory must exist and not be a symlink")
    objects, buckets, cluster = _load_sample_receipt_inventory(
        sample_run_dir,
        object_ranks=expected_object_ranks,
        backend_report=backend_report,
    )
    object_hashes = [
        objects[rank]["canonical_sha256"] for rank in expected_object_ranks
    ]
    bucket_hashes = [buckets[rank]["canonical_sha256"] for rank in range(14)]
    packed_launch, _packed_launch_sha = _verify_sealed(
        args.sample_object_launch_receipt, "packed object-sample launch receipt"
    )
    packed_launch_sha = _validate_packed_sample_launch_receipt(
        packed_launch,
        job_id=args.sample_object_job_id,
        lane_plan_sha=lane_plan_sha,
        policy_sha=policy_sha,
        source_plan_sha=source_plan_sha,
        calibration_sha=calibration_sha,
        objects=objects,
        lane_count=sample_lane_count,
        cpus_per_lane=sample_cpus_per_lane,
    )
    for lane_id, record in enumerate(packed_launch["lane_receipts"]):
        lane_receipt_path = args.sample_object_launch_receipt.parent / record["path"]
        lane_receipt = _load_object(
            lane_receipt_path, f"packed object-sample lane receipt {lane_id}"
        )
        try:
            lane_receipt_sha = _validate_lane_receipt(
                lane_receipt,
                lane_id=lane_id,
                lane_plan=lane_plan,
                source_plan=source_plan,
                calibration=calibration,
                run_root=sample_run_dir,
                job_id=args.sample_object_job_id,
            )
        except ValueError as exc:
            raise FamilyWorkflowError(
                f"invalid packed object-sample lane receipt {lane_id}: {exc}"
            ) from exc
        if lane_receipt_sha != record["canonical_sha256"]:
            _fail(f"packed object-sample lane receipt {lane_id} hash mismatch")
    packed_bucket_launch, _packed_bucket_launch_sha = _verify_sealed(
        args.sample_bucket_launch_receipt, "packed bucket-sample launch receipt"
    )
    packed_bucket_launch_sha = _validate_packed_bucket_launch_receipt(
        packed_bucket_launch,
        job_id=args.sample_bucket_job_id,
        object_launch_sha=packed_launch_sha,
        policy_sha=policy_sha,
        source_plan_sha=source_plan_sha,
        calibration_sha=calibration_sha,
        buckets=buckets,
    )
    for rank, record in enumerate(packed_bucket_launch["task_receipts"]):
        task_receipt_path = args.sample_bucket_launch_receipt.parent / record["path"]
        task_receipt = _load_object(
            task_receipt_path, f"packed bucket-sample task receipt {rank}"
        )
        try:
            task_sha, backend_sha = _validate_bucket_task_receipt(
                task_receipt,
                rank=rank,
                source_plan=source_plan,
                calibration=calibration,
                object_launch_sha256=packed_launch_sha,
                object_receipt_hashes=object_hashes,
                run_root=sample_run_dir,
                job_id=args.sample_bucket_job_id,
            )
        except ValueError as exc:
            raise FamilyWorkflowError(
                f"invalid packed bucket-sample task receipt {rank}: {exc}"
            ) from exc
        if (
            task_sha != record["canonical_sha256"]
            or backend_sha != buckets[rank]["canonical_sha256"]
        ):
            _fail(f"packed bucket-sample task {rank} receipt hash mismatch")
    sample_documents = sum(
        _positive_int(
            _mapping(receipt.get("candidate_file"), "candidate file").get("rows"),
            "sample candidate rows",
        )
        for receipt in objects.values()
    )
    projected_documents_float = _positive_number(
        report_projection.get("candidate_documents"), "projected candidate documents"
    )
    estimated_total_documents = math.ceil(projected_documents_float)
    if estimated_total_documents < sample_documents:
        _fail("backend report projects fewer documents than its sample")

    pack_plan, pack_plan_sha = _verify_sealed(
        args.production_pack_plan, "production source pack plan"
    )
    source_download_peak = _validate_production_pack_plan(
        pack_plan,
        recipe=recipe,
        recipe_sha=recipe_sha,
        policy_sha=policy_sha,
        source_plan=source_plan,
        source_plan_sha=source_plan_sha,
    )
    writer_probe, writer_probe_sha = _verify_sealed(
        args.writer_probe, "post-cluster writer probe"
    )
    writer_components, writer_projected_cpu = _validate_writer_probe(
        writer_probe,
        recipe=recipe,
        recipe_sha=recipe_sha,
        policy_sha=policy_sha,
        source_plan_sha=source_plan_sha,
        calibration_sha=calibration_sha,
        backend_report_sha=backend_report_sha,
        cluster_sha=cluster["canonical_sha256"],
        sample_documents=sample_documents,
        estimated_total_documents=estimated_total_documents,
    )

    sample_source_peak = sum(
        max(
            _positive_int(
                _mapping(objects[rank].get("raw_object"), "raw object").get(
                    "size_bytes"
                ),
                "sample raw object bytes",
            )
            for rank in ranks
        )
        for ranks in sample_lanes.values()
        if ranks
    )
    components: dict[str, dict[str, Any]] = {
        "source_downloads": {
            "sample_measured_bytes": sample_source_peak,
            "projected_peak_bytes_before_safety": source_download_peak,
            "projection_basis": "explicit_serial_per_lane_production_pack_plan",
            "evidence_sha256s": [lane_plan_sha, pack_plan_sha, *object_hashes],
        },
        "filtered_text": {
            "sample_measured_bytes": sum(
                int(receipt["candidate_file"]["size_bytes"])
                for receipt in objects.values()
            ),
            "projected_peak_bytes_before_safety": math.ceil(
                _nonnegative_number(
                    report_projection.get("candidate_bytes"),
                    "projected candidate bytes",
                )
            ),
            "projection_basis": "backend_report_source_stratified_candidate_projection",
            "evidence_sha256s": [backend_report_sha, *object_hashes],
        },
        "minhash_signatures": {
            "sample_measured_bytes": sum(
                int(item["size_bytes"])
                for receipt in objects.values()
                for item in receipt["signature_files"]
            ),
            "projected_peak_bytes_before_safety": math.ceil(
                _nonnegative_number(
                    report_projection.get("signature_bytes"),
                    "projected signature bytes",
                )
            ),
            "projection_basis": "backend_report_projected_candidates_times_signature_layout",
            "evidence_sha256s": [backend_report_sha, *object_hashes],
        },
        "minhash_buckets": {
            "sample_measured_bytes": sum(
                int(receipt["output"]["size_bytes"]) for receipt in buckets.values()
            ),
            "projected_peak_bytes_before_safety": math.ceil(
                _nonnegative_number(
                    report_projection.get("duplicate_edge_bytes"),
                    "projected duplicate-edge bytes",
                )
            ),
            "projection_basis": "backend_report_sample_edge_density_projection",
            "evidence_sha256s": [backend_report_sha, *bucket_hashes],
        },
        "cluster_assignments": {
            "sample_measured_bytes": sum(
                int(item["size_bytes"]) for item in cluster["output_files"]
            ),
            "projected_peak_bytes_before_safety": math.ceil(
                _nonnegative_number(
                    report_projection.get("backend_output_bytes"),
                    "projected backend output bytes",
                )
            ),
            "projection_basis": "backend_report_sample_cluster_output_projection",
            "evidence_sha256s": [backend_report_sha, cluster["canonical_sha256"]],
        },
        **writer_components,
    }

    allocations = [
        _live_completed_cpu2dq_allocation(
            REPO_ROOT,
            job_id=args.bootstrap_job_id,
            stage="bootstrap",
            evidence_receipt_sha256s=[source_plan_sha, calibration_sha, lane_plan_sha],
        )
    ]
    allocations.append(
        _live_completed_cpu2dq_allocation(
            REPO_ROOT,
            job_id=args.sample_object_job_id,
            stage="packed_object_sample",
            evidence_receipt_sha256s=[packed_launch_sha, *object_hashes],
        )
    )
    allocations.append(
        _live_completed_cpu2dq_allocation(
            REPO_ROOT,
            job_id=args.sample_bucket_job_id,
            stage="packed_minhash_bucket_sample",
            evidence_receipt_sha256s=[packed_bucket_launch_sha, *bucket_hashes],
        )
    )
    allocations.extend(
        [
            _live_completed_cpu2dq_allocation(
                REPO_ROOT,
                job_id=args.sample_cluster_job_id,
                stage="priority_cluster_quality_format",
                evidence_receipt_sha256s=[cluster["canonical_sha256"]],
            ),
            _live_completed_cpu2dq_allocation(
                REPO_ROOT,
                job_id=args.sample_quality_audit_job_id,
                stage="bounded_sample_quality_audit",
                evidence_receipt_sha256s=[
                    mixture_quality_approval["sample_quality_audit_sha256"]
                ],
            ),
            _live_completed_cpu2dq_allocation(
                REPO_ROOT,
                job_id=args.writer_probe_job_id,
                stage="post_cluster_writer_probe",
                evidence_receipt_sha256s=[writer_probe_sha],
            ),
        ]
    )
    if len({item["job_id_raw"] for item in allocations}) != len(allocations):
        _fail("sample allocation ledger contains a duplicate Slurm allocation")
    macocu_allocation = _live_completed_cpu2dq_allocation(
        REPO_ROOT,
        job_id=args.macocu_job_id,
        stage="macocu_genre_preparation",
        evidence_receipt_sha256s=[macocu_sha],
    )
    if macocu_allocation["job_id_raw"] in {
        item["job_id_raw"] for item in allocations
    }:
        _fail("MaCoCu preparation allocation duplicates the sample ledger")

    backend_projected_cpu, backend_allocation_projection = (
        _packed_production_backend_cpu_projection(
            source_plan=source_plan,
            pack_plan=pack_plan,
            backend_report=backend_report,
            sample_bucket_receipts=buckets,
        )
    )
    future_components = {
        "production_backend": {
            "projected_cpu_saat_before_safety": backend_projected_cpu,
            "projection_basis": (
                "packed_object_node_wall_plus_once_billed_bucket_and_cluster_allocations"
            ),
            "evidence_sha256s": [backend_report_sha, pack_plan_sha],
        },
        "production_pool_materialization": {
            "projected_cpu_saat_before_safety": writer_projected_cpu,
            "projection_basis": (
                "post_cluster_writer_probe_wall_linear_by_candidate_documents"
            ),
            "evidence_sha256s": [writer_probe_sha, cluster["canonical_sha256"]],
        },
    }
    for stage, ceiling in DATA_PREP_FIXED_CPU2DQ_CEILINGS.items():
        future_components[stage] = {
            "projected_cpu_saat_before_safety": float(ceiling),
            "projection_basis": (
                "explicit_one_node_cpu2dq_submission_walltime_ceiling"
            ),
            "evidence_sha256s": [recipe_sha, policy_sha],
        }
    fixed_allocation_details = {
        stage: {
            "allocation_contract": "one_exclusive_128cpu_cpu2dq_node",
            "maximum_wall_hours": ceiling / CPU2DQ_BILLABLE_CPUS,
            "projected_cpu_saat_before_safety": float(ceiling),
            "submission_must_not_exceed_ceiling": True,
        }
        for stage, ceiling in DATA_PREP_FIXED_CPU2DQ_CEILINGS.items()
    }
    total_sample_cpu_time = sum(int(item["cpu_time_raw_seconds"]) for item in allocations)
    total_sample_billed = sum(float(item["billed_cpu_saat"]) for item in allocations)
    receipt = seal_manifest(
        {
            "schema_version": "3.0",
            "kind": DATA_PREP_STORAGE_SAMPLE_KIND,
            "family_id": recipe["family_id"],
            "recipe_sha256": recipe_sha,
            "policy_sha256": policy_sha,
            "source_plan_sha256": source_plan_sha,
            "calibration_sha256": calibration_sha,
            "backend_resource_report_sha256": backend_report_sha,
            "resource_approval_sha256": resource_approval_sha,
            "mixture_quality_approval_sha256": mixture_quality_approval_sha,
            "sample_quality_audit_sha256": mixture_quality_approval[
                "sample_quality_audit_sha256"
            ],
            "sample_cluster_receipt_sha256": cluster["canonical_sha256"],
            "approval_evidence": {
                "schema_version": "1.0",
                "source_plan": _evidence_record_under(
                    args.output.expanduser().resolve().parent,
                    args.source_plan,
                    "source plan",
                ),
                "calibration": _evidence_record_under(
                    args.output.expanduser().resolve().parent,
                    args.calibration,
                    "calibration",
                ),
                "backend_resource_report": _evidence_record_under(
                    args.output.expanduser().resolve().parent,
                    args.backend_resource_report,
                    "backend resource report",
                ),
                "resource_approval": _evidence_record_under(
                    args.output.expanduser().resolve().parent,
                    args.resource_approval,
                    "resource approval",
                ),
                "mixture_quality_approval": _evidence_record_under(
                    args.output.expanduser().resolve().parent,
                    args.mixture_quality_approval,
                    "mixture-quality approval",
                ),
            },
            "sample_lane_plan_sha256": lane_plan_sha,
            "production_pack_plan_sha256": pack_plan_sha,
            "writer_probe_sha256": writer_probe_sha,
            "macocu_preparation_manifest_sha256": macocu_sha,
            "sample_documents": sample_documents,
            "estimated_total_documents": estimated_total_documents,
            "components": components,
            "sample_allocations": allocations,
            "sample_allocation_totals": {
                "unique_allocations": len(allocations),
                "cpu_time_raw_seconds": total_sample_cpu_time,
                "billed_cpu_saat": total_sample_billed,
                "accounting_role": (
                    "already_consumed_measurement_evidence_not_future_quota"
                ),
            },
            "historical_one_time_preparations": [
                {
                    "preparation_id": "macocu_genre_tr_v1",
                    "manifest_sha256": macocu_sha,
                    "allocation": macocu_allocation,
                    "accounting_status": (
                        "already_consumed_excluded_from_future_projection"
                    ),
                    "future_projected_cpu_saat": 0,
                }
            ],
            "future_resource_projection": {
                "components": future_components,
                "allocation_details": {
                    "production_backend": backend_allocation_projection,
                    "production_pool_materialization": {
                        "allocation_contract": (
                            "one_exclusive_128cpu_cpu2dq_node_scaled_by_candidate_documents"
                        ),
                        "sample_elapsed_wall_seconds": writer_probe["sample"][
                            "elapsed_wall_seconds"
                        ],
                        "document_scale": writer_probe["projection"]["document_scale"],
                        "projected_cpu_saat_before_safety": writer_projected_cpu,
                    },
                    **fixed_allocation_details,
                },
                "projected_cpu_saat_before_safety": sum(
                    float(item["projected_cpu_saat_before_safety"])
                    for item in future_components.values()
                ),
                "safety_factor_applied": False,
                "excluded_historical_and_sample_allocations": True,
            },
            "canonical_sha256": None,
        }
    )
    _validate_data_prep_storage_sample(
        receipt, recipe=recipe, recipe_sha=recipe_sha
    )
    write_json_atomic(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))


def command_data_prep_storage_gate(args: argparse.Namespace) -> None:
    """Gate the explicit future peak; historical/sample allocations stay excluded."""

    recipe, recipe_sha = load_recipe(args.recipe)
    measurement, measurement_sha = _verify_sealed(
        args.sample_measurement, "data-preparation storage sample"
    )
    _validate_data_prep_storage_sample(
        measurement, recipe=recipe, recipe_sha=recipe_sha
    )
    _validate_storage_approval_evidence(
        measurement,
        measurement_path=args.sample_measurement,
        policy_path=args.policy,
    )
    policy = recipe["storage"]["data_preparation_peak_gate"]
    sample_documents = _positive_int(
        measurement.get("sample_documents"), "sample_documents"
    )
    total_documents = _positive_int(
        measurement.get("estimated_total_documents"), "estimated_total_documents"
    )
    if sample_documents < int(policy["minimum_sample_documents"]):
        _fail("data-preparation storage sample is too small")
    if total_documents < sample_documents:
        _fail("estimated total documents cannot be smaller than the sample")
    components = _mapping(measurement.get("components"), "sample components")
    expected_components = set(policy["required_measured_components"])
    if expected_components != set(DATA_PREP_STORAGE_COMPONENTS) or set(
        components
    ) != expected_components:
        _fail(
            "data-preparation sample component set mismatch; "
            f"missing={sorted(expected_components - set(components))}, "
            f"unexpected={sorted(set(components) - expected_components)}"
        )
    projections: dict[str, dict[str, Any]] = {}
    projected_peak = 0
    for name in sorted(expected_components):
        record = _mapping(components[name], f"sample component {name}")
        projected = _nonnegative_int(
            record["projected_peak_bytes_before_safety"],
            f"{name}.projected_peak_bytes_before_safety",
        )
        projections[name] = dict(record)
        projected_peak += projected
    future_projection = _mapping(
        measurement.get("future_resource_projection"), "future resource projection"
    )
    if future_projection.get("safety_factor_applied") is not False:
        _fail("future resource projection already contains a safety factor")
    projected_cpu_saat = _positive_number(
        future_projection.get("projected_cpu_saat_before_safety"),
        "future projected data-preparation CPU-saat",
    )
    safety_factor = float(policy["extrapolation_safety_factor"])
    projected_with_safety = math.ceil(projected_peak * safety_factor)
    projected_cpu_saat_with_safety = math.ceil(projected_cpu_saat * safety_factor)
    required_free = projected_with_safety + int(recipe["storage"]["minimum_free_headroom_bytes"])
    work_dir = args.work_dir.expanduser().resolve()
    if not work_dir.is_dir() or work_dir.is_symlink():
        _fail(f"data-preparation work directory must already exist and not be a symlink: {work_dir}")
    live_storage_policy = recipe["storage"]["uhem_live_quota"]
    free, live_storage_audit = _live_beegfs_storage(
        REPO_ROOT,
        uid=int(live_storage_policy["uid"]),
        storage_pool_id=int(live_storage_policy["storage_pool_id"]),
        path=work_dir,
    )
    if free < required_free:
        _fail(
            "data-preparation peak storage gate failed: "
            f"need {required_free} free bytes, found {free}; no existing artifact may be deleted"
        )
    budget = recipe["uhem_budget"]
    remaining_cpu_saat, quota_output_sha, quota_audit = _live_uhem_cpu_saat(
        REPO_ROOT, str(budget["account"]), str(budget["user"])
    )
    training_ceiling = int(budget["operational_ceiling_cpu_saat"])
    total_project_ceiling = projected_cpu_saat_with_safety + training_ceiling
    if remaining_cpu_saat < total_project_ceiling:
        _fail(
            "data-preparation plus training quota gate failed: "
            f"need {total_project_ceiling} CPU-saat, found {remaining_cpu_saat:.2f}"
        )
    receipt = seal_manifest(
        {
            "schema_version": "3.0",
            "kind": "d32_data_prep_storage_gate",
            "family_id": recipe["family_id"],
            "recipe_sha256": recipe_sha,
            "sample_measurement_sha256": measurement_sha,
            "policy_sha256": measurement["policy_sha256"],
            "source_plan_sha256": measurement["source_plan_sha256"],
            "calibration_sha256": measurement["calibration_sha256"],
            "backend_resource_report_sha256": measurement[
                "backend_resource_report_sha256"
            ],
            "resource_approval_sha256": measurement[
                "resource_approval_sha256"
            ],
            "mixture_quality_approval_sha256": measurement[
                "mixture_quality_approval_sha256"
            ],
            "sample_quality_audit_sha256": measurement[
                "sample_quality_audit_sha256"
            ],
            "sample_cluster_receipt_sha256": measurement[
                "sample_cluster_receipt_sha256"
            ],
            "production_pack_plan_sha256": measurement[
                "production_pack_plan_sha256"
            ],
            "writer_probe_sha256": measurement["writer_probe_sha256"],
            "sample_documents": sample_documents,
            "estimated_total_documents": total_documents,
            "component_projections": projections,
            "sample_allocation_totals": measurement["sample_allocation_totals"],
            "historical_one_time_preparations": measurement[
                "historical_one_time_preparations"
            ],
            "future_resource_projection": future_projection,
            "projected_peak_bytes_before_safety": projected_peak,
            "safety_factor": safety_factor,
            "safety_factor_application_count": 1,
            "projected_peak_bytes_with_safety": projected_with_safety,
            "projected_data_preparation_cpu_saat_before_safety": projected_cpu_saat,
            "projected_data_preparation_cpu_saat_with_safety": projected_cpu_saat_with_safety,
            "training_proxy_smoke_operational_ceiling_cpu_saat": training_ceiling,
            "total_project_operational_ceiling_cpu_saat": total_project_ceiling,
            "required_free_bytes_including_headroom": required_free,
            "work_dir": str(work_dir),
            "work_dir_filesystem_device": work_dir.stat().st_dev,
            "observed_free_bytes": free,
            "live_storage": live_storage_audit,
            "uhem_quota": {
                "remaining_cpu_saat": remaining_cpu_saat,
                "sshare_output_sha256": quota_output_sha,
                **quota_audit,
            },
            "never_auto_delete_existing_artifacts": True,
            "canonical_sha256": None,
        }
    )
    _validate_data_prep_storage_gate_receipt(
        receipt, recipe=recipe, recipe_sha=recipe_sha
    )
    write_json_atomic(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))


def _validate_data_prep_storage_gate_receipt(
    gate: Mapping[str, Any], *, recipe: Mapping[str, Any], recipe_sha: str
) -> int:
    """Recompute every post-safety storage/future-CPU gate invariant."""

    verify_manifest_hash(gate)
    if (
        gate.get("schema_version") != "3.0"
        or gate.get("kind") != "d32_data_prep_storage_gate"
        or gate.get("family_id") != recipe["family_id"]
        or gate.get("recipe_sha256") != recipe_sha
        or gate.get("never_auto_delete_existing_artifacts") is not True
    ):
        _fail("data-preparation storage gate identity drifted")
    for field in (
        "sample_measurement_sha256",
        "policy_sha256",
        "source_plan_sha256",
        "calibration_sha256",
        "backend_resource_report_sha256",
        "resource_approval_sha256",
        "mixture_quality_approval_sha256",
        "sample_quality_audit_sha256",
        "sample_cluster_receipt_sha256",
        "production_pack_plan_sha256",
        "writer_probe_sha256",
    ):
        _sha256(gate.get(field), f"data-preparation gate {field}")
    safety = float(recipe["storage"]["data_preparation_peak_gate"]["extrapolation_safety_factor"])
    if gate.get("safety_factor") != safety or gate.get("safety_factor_application_count") != 1:
        _fail("data-preparation storage gate safety-factor drifted")
    components = _mapping(gate.get("component_projections"), "gate storage components")
    if set(components) != set(DATA_PREP_STORAGE_COMPONENTS):
        _fail("data-preparation storage component inventory drifted")
    projected_storage = sum(
        _nonnegative_int(
            _mapping(components[name], f"gate component {name}").get(
                "projected_peak_bytes_before_safety"
            ),
            f"gate component {name} projected bytes",
        )
        for name in DATA_PREP_STORAGE_COMPONENTS
    )
    storage_with_safety = math.ceil(projected_storage * safety)
    required_free = storage_with_safety + int(recipe["storage"]["minimum_free_headroom_bytes"])
    if (
        gate.get("projected_peak_bytes_before_safety") != projected_storage
        or gate.get("projected_peak_bytes_with_safety") != storage_with_safety
        or gate.get("required_free_bytes_including_headroom") != required_free
        or _nonnegative_int(gate.get("observed_free_bytes"), "gate observed free bytes")
        < required_free
    ):
        _fail("data-preparation storage gate byte arithmetic drifted")
    future = _mapping(gate.get("future_resource_projection"), "gate future projection")
    if (
        future.get("safety_factor_applied") is not False
        or future.get("excluded_historical_and_sample_allocations") is not True
    ):
        _fail("data-preparation future projection accounting drifted")
    future_components = _mapping(future.get("components"), "gate future components")
    details = _mapping(future.get("allocation_details"), "gate allocation details")
    if set(future_components) != set(DATA_PREP_FUTURE_CPU_COMPONENTS) or set(details) != set(
        DATA_PREP_FUTURE_CPU_COMPONENTS
    ):
        _fail("data-preparation future CPU component inventory drifted")
    projected_cpu = 0.0
    for name in DATA_PREP_FUTURE_CPU_COMPONENTS:
        component = _mapping(future_components[name], f"gate future component {name}")
        value = _positive_number(
            component.get("projected_cpu_saat_before_safety"),
            f"gate future component {name} CPU-saat",
        )
        projected_cpu += value
        if name in DATA_PREP_FIXED_CPU2DQ_CEILINGS:
            ceiling = DATA_PREP_FIXED_CPU2DQ_CEILINGS[name]
            expected_details = {
                "allocation_contract": "one_exclusive_128cpu_cpu2dq_node",
                "maximum_wall_hours": ceiling / CPU2DQ_BILLABLE_CPUS,
                "projected_cpu_saat_before_safety": float(ceiling),
                "submission_must_not_exceed_ceiling": True,
            }
            if value != float(ceiling) or details.get(name) != expected_details:
                _fail(f"data-preparation gate {name} ceiling drifted")
    backend_details = _mapping(
        details.get("production_backend"), "gate production backend allocation details"
    )
    node_count = _positive_int(
        backend_details.get("node_count"), "gate production object node count"
    )
    node_walls = _mapping(
        backend_details.get("projected_node_wall_seconds_before_safety"),
        "gate production object node walls",
    )
    if set(node_walls) != {str(index) for index in range(node_count)}:
        _fail("gate production object node-wall inventory drifted")
    object_cpu = (
        sum(
            _positive_number(value, "gate production object node wall")
            for value in node_walls.values()
        )
        * CPU2DQ_BILLABLE_CPUS
        / 3600.0
    )
    bucket_wall = _positive_number(
        backend_details.get(
            "projected_packed_bucket_node_wall_seconds_before_safety"
        ),
        "gate production bucket node wall",
    )
    bucket_cpu = bucket_wall * CPU2DQ_BILLABLE_CPUS / 3600.0
    cluster_cpu = _positive_number(
        backend_details.get("projected_priority_cluster_cpu_saat_before_safety"),
        "gate production cluster CPU-saat",
    )
    backend_cpu = object_cpu + bucket_cpu + cluster_cpu
    for actual, expected, label in (
        (
            backend_details.get("projected_packed_object_cpu_saat_before_safety"),
            object_cpu,
            "object CPU",
        ),
        (
            backend_details.get("projected_minhash_bucket_cpu_saat_before_safety"),
            bucket_cpu,
            "bucket CPU",
        ),
        (
            backend_details.get("projected_backend_cpu_saat_before_safety"),
            backend_cpu,
            "backend CPU",
        ),
        (
            future_components["production_backend"].get(
                "projected_cpu_saat_before_safety"
            ),
            backend_cpu,
            "backend component CPU",
        ),
    ):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool) or not math.isclose(
            float(actual), expected, rel_tol=1e-12, abs_tol=1e-9
        ):
            _fail(f"data-preparation gate production {label} arithmetic drifted")
    cluster_wall = cluster_cpu * 3600.0 / CPU2DQ_BILLABLE_CPUS
    pool_cpu = _positive_number(
        _mapping(
            future_components["production_pool_materialization"],
            "gate production pool component",
        ).get("projected_cpu_saat_before_safety"),
        "gate production pool CPU-saat",
    )
    pool_details = _mapping(
        details.get("production_pool_materialization"),
        "gate production pool allocation details",
    )
    if (
        pool_details.get("allocation_contract")
        != "one_exclusive_128cpu_cpu2dq_node_scaled_by_candidate_documents"
        or not math.isclose(
            float(pool_details.get("projected_cpu_saat_before_safety", -1)),
            pool_cpu,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
    ):
        _fail("data-preparation gate production pool allocation drifted")
    pool_wall = pool_cpu * 3600.0 / CPU2DQ_BILLABLE_CPUS
    cluster_memory_limit = 192 * 1024**3
    sample_cluster_peak_rss = _positive_int(
        backend_details.get("sample_priority_cluster_peak_rss_bytes"),
        "gate sample priority-cluster peak RSS",
    )
    projected_cluster_peak_rss = _positive_number(
        backend_details.get(
            "projected_priority_cluster_peak_rss_bytes_before_safety"
        ),
        "gate projected priority-cluster peak RSS",
    )
    if (
        max(float(value) for value in node_walls.values()) * safety > 172_800
        or bucket_wall * safety > 86_400
        or cluster_wall * safety > 172_800
        or pool_wall * safety > 172_800
        or sample_cluster_peak_rss >= cluster_memory_limit
        or projected_cluster_peak_rss * safety >= cluster_memory_limit
    ):
        _fail("data-preparation gate exceeds a production backend walltime/RSS limit")
    cpu_with_safety = math.ceil(projected_cpu * safety)
    training_ceiling = int(recipe["uhem_budget"]["operational_ceiling_cpu_saat"])
    total_ceiling = cpu_with_safety + training_ceiling
    if (
        not math.isclose(
            float(future.get("projected_cpu_saat_before_safety", -1)),
            projected_cpu,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
        or not math.isclose(
            float(gate.get("projected_data_preparation_cpu_saat_before_safety", -1)),
            projected_cpu,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
        or gate.get("projected_data_preparation_cpu_saat_with_safety")
        != cpu_with_safety
        or gate.get("training_proxy_smoke_operational_ceiling_cpu_saat")
        != training_ceiling
        or gate.get("total_project_operational_ceiling_cpu_saat") != total_ceiling
    ):
        _fail("data-preparation gate CPU arithmetic drifted")
    quota = _mapping(gate.get("uhem_quota"), "gate UHeM quota")
    if _positive_number(quota.get("remaining_cpu_saat"), "gate remaining CPU-saat") < total_ceiling:
        _fail("data-preparation gate sealed insufficient CPU quota")
    return cpu_with_safety


def command_preflight(args: argparse.Namespace) -> None:
    recipe, recipe_sha = load_recipe(args.recipe)
    try:
        from nanochat.turkish_backend import (
            validate_backend_calibration,
            validate_mixture_quality_approval,
            validate_resource_approval,
            validate_source_plan,
        )
        from nanochat.turkish_corpus import load_corpus_policy

        policy = load_corpus_policy(args.policy)
        _validate_recipe_policy_identity(recipe, policy)
        source_plan = _load_object(args.source_plan, "source plan")
        calibration = _load_object(args.calibration, "backend calibration")
        validate_source_plan(source_plan, policy)
        validate_backend_calibration(calibration, policy)
    except (OSError, ValueError) as exc:
        raise FamilyWorkflowError(f"invalid preflight data provenance: {exc}") from exc
    policy_sha = _data_prep_policy_sha256(policy)
    source_plan_sha = _sha256(
        source_plan.get("canonical_sha256"), "preflight source-plan SHA-256"
    )
    calibration_sha = _sha256(
        calibration.get("canonical_sha256"), "preflight calibration SHA-256"
    )
    mixture_quality_approval, mixture_quality_approval_sha = _verify_sealed(
        args.mixture_quality_approval, "preflight mixture-quality approval"
    )
    try:
        validate_mixture_quality_approval(
            mixture_quality_approval,
            policy=policy,
            plan=source_plan,
            calibration=calibration,
            approval_path=args.mixture_quality_approval,
        )
    except ValueError as exc:
        raise FamilyWorkflowError(f"invalid mixture-quality approval: {exc}") from exc
    resource_approval, resource_approval_sha = _verify_sealed(
        args.resource_approval, "preflight backend resource approval"
    )
    try:
        validate_resource_approval(
            resource_approval,
            plan=source_plan,
            policy=policy,
            calibration=calibration,
            approval_path=args.resource_approval,
        )
    except ValueError as exc:
        raise FamilyWorkflowError(f"invalid backend resource approval: {exc}") from exc
    if (
        resource_approval.get("mixture_quality_approval_sha256")
        != mixture_quality_approval_sha
    ):
        _fail("resource approval does not bind the current mixture-quality approval")
    data_prep_gate, data_prep_gate_sha = _load_receipt(
        args.data_prep_storage_gate, "d32_data_prep_storage_gate"
    )
    data_prep_cpu = _validate_data_prep_storage_gate_receipt(
        data_prep_gate, recipe=recipe, recipe_sha=recipe_sha
    )
    if data_prep_gate.get("recipe_sha256") != recipe_sha:
        _fail("data-preparation storage gate was created for a different recipe")
    if data_prep_gate.get("schema_version") != "3.0":
        _fail("data-preparation storage gate schema is stale")
    expected_data_bindings = {
        "policy_sha256": policy_sha,
        "source_plan_sha256": source_plan_sha,
        "calibration_sha256": calibration_sha,
        "resource_approval_sha256": resource_approval_sha,
        "mixture_quality_approval_sha256": mixture_quality_approval_sha,
        "sample_quality_audit_sha256": mixture_quality_approval[
            "sample_quality_audit_sha256"
        ],
        "sample_cluster_receipt_sha256": mixture_quality_approval[
            "sample_cluster_receipt_sha256"
        ],
        "backend_resource_report_sha256": resource_approval[
            "resource_report_sha256"
        ],
    }
    for field, expected in expected_data_bindings.items():
        if data_prep_gate.get(field) != expected:
            _fail(f"data-preparation gate {field} binding mismatch")
    cluster_launch, cluster_launch_sha = _verify_sealed(
        args.cluster_launch_receipt, "production cluster launch receipt"
    )
    expected_cluster_launch = {
        "schema_version": "2.0",
        "kind": "turkish_packed_production_cluster_launch_receipt",
        "family_id": recipe["family_id"],
        "recipe_sha256": recipe_sha,
        "policy_sha256": policy_sha,
        "source_plan_sha256": source_plan_sha,
        "calibration_sha256": calibration_sha,
        "production_pack_plan_sha256": data_prep_gate[
            "production_pack_plan_sha256"
        ],
        "resource_approval_sha256": resource_approval_sha,
        "mixture_quality_approval_sha256": mixture_quality_approval_sha,
        "data_prep_storage_gate_sha256": data_prep_gate_sha,
        "sample_cluster_receipt_sha256": data_prep_gate[
            "sample_cluster_receipt_sha256"
        ],
        "cluster_completed": True,
    }
    for field, expected in expected_cluster_launch.items():
        if cluster_launch.get(field) != expected:
            _fail(f"production cluster launch {field} binding mismatch")
    if data_prep_gate.get("never_auto_delete_existing_artifacts") is not True:
        _fail("data-preparation storage gate does not preserve existing artifacts")
    data_prep_policy = recipe["storage"]["data_preparation_peak_gate"]
    safety_factor = float(data_prep_policy["extrapolation_safety_factor"])
    if (
        data_prep_gate.get("safety_factor") != safety_factor
        or data_prep_gate.get("safety_factor_application_count") != 1
    ):
        _fail("data-preparation gate safety-factor contract drifted")
    gate_components = _mapping(
        data_prep_gate.get("component_projections"),
        "data-preparation gate component projections",
    )
    if set(gate_components) != set(DATA_PREP_STORAGE_COMPONENTS):
        _fail("data-preparation gate component inventory drifted")
    projected_peak_before = sum(
        _nonnegative_int(
            _mapping(gate_components[name], f"data-preparation gate {name}").get(
                "projected_peak_bytes_before_safety"
            ),
            f"data-preparation gate {name} projected peak",
        )
        for name in DATA_PREP_STORAGE_COMPONENTS
    )
    if (
        data_prep_gate.get("projected_peak_bytes_before_safety")
        != projected_peak_before
        or data_prep_gate.get("projected_peak_bytes_with_safety")
        != math.ceil(projected_peak_before * safety_factor)
    ):
        _fail("data-preparation gate storage safety arithmetic mismatch")
    future_projection = _mapping(
        data_prep_gate.get("future_resource_projection"),
        "data-preparation gate future resource projection",
    )
    if (
        future_projection.get("safety_factor_applied") is not False
        or future_projection.get("excluded_historical_and_sample_allocations")
        is not True
    ):
        _fail("data-preparation gate future-work accounting drifted")
    projected_cpu_before = _positive_number(
        future_projection.get("projected_cpu_saat_before_safety"),
        "data-preparation gate future CPU-saat",
    )
    if (
        not math.isclose(
            float(
                data_prep_gate.get(
                    "projected_data_preparation_cpu_saat_before_safety", -1
                )
            ),
            projected_cpu_before,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
        or data_prep_gate.get("projected_data_preparation_cpu_saat_with_safety")
        != math.ceil(projected_cpu_before * safety_factor)
    ):
        _fail("data-preparation gate CPU safety arithmetic mismatch")
    data_prep_cpu = _positive_int(data_prep_cpu, "data-preparation projected CPU-saat")
    expected_total_ceiling = (
        data_prep_cpu + int(recipe["uhem_budget"]["operational_ceiling_cpu_saat"])
    )
    if data_prep_gate.get("total_project_operational_ceiling_cpu_saat") != expected_total_ceiling:
        _fail("data-preparation gate total project CPU-saat arithmetic mismatch")
    base_dir = args.base_dir.expanduser().resolve()
    if not base_dir.is_dir():
        _fail(f"base directory does not exist: {base_dir}")
    paths = _resolved_artifact_paths(recipe, base_dir)
    artifacts = recipe["artifacts"]

    mixture_path = (args.repo_root / str(artifacts["mixture_config"])).resolve()
    if not mixture_path.is_file() or mixture_path.is_symlink():
        _fail(f"mixture config is missing or unsafe: {mixture_path}")
    mixture = _load_object(mixture_path, "mixture config")
    try:
        from nanochat.turkish_corpus import validate_corpus_policy

        validate_corpus_policy(mixture)
    except Exception as exc:
        raise FamilyWorkflowError(f"corpus policy verification failed: {exc}") from exc
    corpus_id = str(artifacts["corpus_id"])
    if mixture.get("name") != corpus_id:
        _fail("mixture config names another family corpus")
    mixture_sha = file_sha256(mixture_path)
    policy_sha = hashlib.sha256(canonical_json(mixture).encode("utf-8")).hexdigest()

    corpus, corpus_sha = _verify_sealed(
        paths["corpus_manifest"], "corpus manifest", args.corpus_manifest_sha256
    )
    _manifest_inventory_if_present(paths["corpus_root"], corpus, "corpus manifest")
    dataset, dataset_sha = _verify_sealed(
        paths["dataset_manifest"], "Nanochat dataset manifest", args.dataset_manifest_sha256
    )
    try:
        validate_dataset_manifest(dataset, profile="strict")
        verify_file_inventory(
            paths["corpus_root"], dataset["ordered_files"], require_exact=False
        )
    except (ManifestValidationError, OSError, ValueError) as exc:
        raise FamilyWorkflowError(f"strict dataset manifest verification failed: {exc}") from exc
    if (
        dataset.get("dataset", {}).get("repo_id") != f"local-composite/{corpus_id}"
        or dataset.get("metadata", {}).get("corpus_name") != corpus_id
    ):
        _fail("Nanochat dataset manifest names another family corpus")
    validation_relative = dataset.get("validation_file")
    if not isinstance(validation_relative, str) or not validation_relative:
        _fail("strict dataset manifest does not name a validation_file")
    validation_path = paths["corpus_root"] / validation_relative
    if not validation_path.is_file() or validation_path.is_symlink():
        _fail(f"fixed validation file is missing or unsafe: {validation_path}")

    source_receipt, source_sha = _verify_sealed(
        paths["source_receipt"], "source receipt", args.source_receipt_sha256
    )
    try:
        from nanochat.turkish_corpus import validate_source_receipt

        validate_source_receipt(source_receipt, mixture)
    except Exception as exc:
        raise FamilyWorkflowError(f"source receipt verification failed: {exc}") from exc
    expected_corpus_fields = {
        "schema_version": "1.0",
        "kind": "turkish_pretrain_corpus",
        "name": corpus_id,
        "stage": "final_interleaved",
        "policy_sha256": policy_sha,
        "source_receipt_sha256": source_sha,
        "nanochat_dataset_manifest_sha256": dataset_sha,
        "language": "tur_Latn",
        "code_allowed": False,
    }
    for field, expected in expected_corpus_fields.items():
        if corpus.get(field) != expected:
            _fail(f"corpus manifest {field} differs from the family contract")
    production_chain = _mapping(
        corpus.get("production_chain"), "corpus production chain"
    )
    parent_pool, parent_pool_sha = _verify_sealed(
        paths["corpus_root"] / "parent_pool_manifest.json",
        "archived parent pool manifest",
    )
    archived_qa_approval, archived_qa_sha = _verify_sealed(
        paths["corpus_root"] / "qa" / "qa_approval.json",
        "archived parent pool QA approval",
    )
    expected_chain_bindings = {
        "cluster_launch_receipt_sha256": cluster_launch_sha,
        "production_pack_plan_sha256": data_prep_gate[
            "production_pack_plan_sha256"
        ],
        "resource_approval_sha256": resource_approval_sha,
        "mixture_quality_approval_sha256": mixture_quality_approval_sha,
        "data_prep_storage_gate_sha256": data_prep_gate_sha,
        "sample_cluster_receipt_sha256": data_prep_gate[
            "sample_cluster_receipt_sha256"
        ],
    }
    if (
        set(production_chain) != set(expected_chain_bindings)
        or any(
            production_chain.get(field) != expected
            for field, expected in expected_chain_bindings.items()
        )
        or source_receipt.get("production_chain") != production_chain
        or dataset.get("metadata", {}).get("production_chain")
        != production_chain
        or parent_pool.get("production_chain") != production_chain
        or parent_pool.get("policy_sha256") != policy_sha
        or corpus.get("parent_pool_manifest_sha256") != parent_pool_sha
        or dataset.get("metadata", {}).get("parent_pool_manifest_sha256")
        != parent_pool_sha
        or dataset.get("metadata", {}).get("qa_approval_sha256")
        != archived_qa_sha
        or _mapping(corpus.get("quality_assurance"), "corpus quality assurance").get(
            "approval_sha256"
        )
        != archived_qa_sha
        or archived_qa_approval.get("decision") != "accepted"
    ):
        _fail("final corpus lineage does not descend from the current data gate")
    corpus_tokenizer = _mapping(corpus.get("tokenizer"), "corpus tokenizer")

    validation_exposure, validation_exposure_sha = _verify_sealed(
        paths["validation_exposure"],
        "validation exposure manifest",
        args.validation_exposure_sha256,
    )
    exposure_index, exposure_index_sha = _verify_sealed(
        paths["exposure_plan_index"],
        "exposure plan index",
        args.exposure_plan_index_sha256,
    )
    if (
        exposure_index.get("schema_version") != "1.0"
        or exposure_index.get("kind") != "d32_exposure_plan_index"
        or exposure_index.get("family_id") != recipe["family_id"]
    ):
        _fail("exposure plan index family identity mismatch")

    package, package_sha = _verify_sealed(
        paths["tokenizer_package"], "tokenizer package", args.tokenizer_package_sha256
    )
    try:
        from nanochat.strict_tokenizer import verify_tokenizer_package

        verified_package = verify_tokenizer_package(
            paths["tokenizer_package"],
            expected_sha256=package_sha,
            expected_name=recipe["artifacts"]["tokenizer_name"],
            expected_vocab_size=recipe["model"]["vocab_size"],
        )
    except Exception as exc:
        raise FamilyWorkflowError(f"tokenizer package verification failed: {exc}") from exc
    if verified_package.manifest != package:
        _fail("tokenizer package changed while it was being verified")
    if (
        corpus_tokenizer.get("name") != artifacts["tokenizer_name"]
        or corpus_tokenizer.get("vocab_size") != recipe["model"]["vocab_size"]
        or corpus_tokenizer.get("package_sha256") != package_sha
        or package.get("policy_sha256") != policy_sha
        or package.get("production_chain") != production_chain
        or package.get("parent_corpus_manifest_sha256") != parent_pool_sha
        or package.get("qa_approval_sha256") != archived_qa_sha
    ):
        _fail("corpus manifest tokenizer binding differs from the family contract")

    macocu_record: dict[str, Any] | None = None
    macocu_path = paths.get("macocu_preparation")
    derived_sources = _mapping(
        source_receipt.get("derived_sources"), "source receipt derived_sources"
    )
    if macocu_path is None:
        if derived_sources:
            _fail("v1 source receipt unexpectedly inventories derived data")
    else:
        macocu_manifest, macocu_sha = _verify_sealed(
            macocu_path, "MaCoCu preparation manifest"
        )
        try:
            from nanochat.turkish_backend import validate_macocu_preparation_manifest

            validate_macocu_preparation_manifest(
                macocu_manifest, mixture, macocu_path.parent
            )
        except Exception as exc:
            raise FamilyWorkflowError(
                f"MaCoCu preparation verification failed: {exc}"
            ) from exc
        source_macocu = _mapping(
            derived_sources.get("macocu_genre_tr"),
            "source receipt MaCoCu provenance",
        )
        if (
            source_macocu.get("manifest_sha256") != macocu_sha
            or source_macocu.get("manifest_uri") != macocu_path.resolve().as_uri()
            or source_macocu.get("upstream") != macocu_manifest.get("upstream")
        ):
            _fail("source receipt is not bound to the exact MaCoCu preparation")
        macocu_record = {"path": str(macocu_path.resolve()), "sha256": macocu_sha}

    anchor_records: dict[str, dict[str, Any]] = {}
    for source_id, path_key, artifact_key in (
        ("mot_tr_v1_11", "mot_preparation", "mot_preparation_manifest"),
        (
            "parlamint_tr_v5_0",
            "parlamint_preparation",
            "parlamint_preparation_manifest",
        ),
    ):
        anchor_path = paths.get(path_key)
        if anchor_path is None:
            if source_id in derived_sources:
                _fail(f"family without {artifact_key} unexpectedly binds {source_id}")
            continue
        anchor_records[artifact_key] = _verify_anchor_preparation_binding(
            anchor_path,
            source_id=source_id,
            derived_sources=derived_sources,
        )

    packing_capacity, packing_capacity_sha = _verify_packing_capacity_receipt(
        paths["packing_capacity"],
        dataset_sha256=dataset_sha,
        tokenizer_sha256=package_sha,
        implementation_path=args.repo_root / "nanochat" / "packing_capacity.py",
        family_id=recipe["family_id"],
    )
    expected_corpus_capacity = {
        "path": paths["packing_capacity"].relative_to(paths["corpus_root"]).as_posix(),
        "sha256": packing_capacity_sha,
        "all_worlds_pass": True,
        "cleanup_authorized": True,
    }
    if corpus.get("packing_capacity") != expected_corpus_capacity:
        _fail("corpus manifest does not bind the passing packing-capacity receipt")
    if exposure_index.get("packing_capacity_receipt_sha256") != packing_capacity_sha:
        _fail("exposure plan index does not bind the packing-capacity receipt")

    try:
        from nanochat.exposure import (
            validate_exposure_manifest,
            validate_training_exposure_plan,
        )

        validate_exposure_manifest(
            validation_exposure, source_dataset_manifest=dataset
        )
    except Exception as exc:
        raise FamilyWorkflowError(f"validation exposure verification failed: {exc}") from exc
    if validation_exposure.get("mode") != "validation":
        _fail("validation exposure manifest must use validation mode")
    validation_selection = _mapping(
        validation_exposure.get("selection"), "validation exposure selection"
    )
    validation_payload_bytes = _positive_int(
        validation_selection.get("realized_payload_bytes"),
        "validation exposure realized_payload_bytes",
    )
    validation_documents = _positive_int(
        validation_selection.get("realized_documents"),
        "validation exposure realized_documents",
    )

    expected_plan_specs = {
        "proxy_d12_seed42_ws1": (1, 42, 4200, 524_288),
        "proxy_d12_seed314159_ws1": (1, 314159, 4200, 524_288),
        "proxy_d20_seed42_ws1": (1, 42, 4980, 1_048_576),
        "signal_smoke_ws4_seed42": (4, 42, 6, 2_097_152),
        "smoke_ws8": (8, 42, 100, 2_097_152),
        "smoke_ws16": (16, 42, 100, 2_097_152),
        "trunk_ws8_seed42": (8, 42, 28800, 2_097_152),
        "s12_ws8_seed42": (8, 42, 9600, 2_097_152),
        "s20_ws8_seed42": (8, 42, 16000, 2_097_152),
        "s40_ws8_seed42": (8, 42, 32000, 2_097_152),
        "trunk_ws16_seed42": (16, 42, 28800, 2_097_152),
        "s12_ws16_seed42": (16, 42, 9600, 2_097_152),
        "s20_ws16_seed42": (16, 42, 16000, 2_097_152),
        "s40_ws16_seed42": (16, 42, 32000, 2_097_152),
    }
    index_plans = exposure_index.get("plans")
    if not isinstance(index_plans, list):
        _fail("exposure plan index plans must be an array")
    by_key = {
        record.get("key"): record
        for record in index_plans
        if isinstance(record, Mapping)
    }
    if set(by_key) != set(expected_plan_specs):
        _fail(
            "exposure plan index key set mismatch; "
            f"expected={sorted(expected_plan_specs)}, found={sorted(str(key) for key in by_key)}"
        )
    verified_plans: dict[str, dict[str, Any]] = {}
    for key, (world_size, seed, optimizer_steps, global_batch_tokens) in expected_plan_specs.items():
        record = _mapping(by_key[key], f"exposure index plan {key}")
        expected_path = paths[f"exposure:{key}"]
        if record.get("path") != expected_path.relative_to(paths["corpus_root"]).as_posix():
            _fail(f"exposure index path mismatch for {key}")
        plan, plan_sha = _verify_sealed(expected_path, f"training exposure plan {key}")
        if record.get("sha256") != plan_sha:
            _fail(f"exposure index hash mismatch for {key}")
        try:
            validate_training_exposure_plan(plan)
        except Exception as exc:
            raise FamilyWorkflowError(f"training exposure plan {key} is invalid: {exc}") from exc
        token_positions = optimizer_steps * global_batch_tokens
        expected_fields = {
            "world_size": world_size,
            "seed": seed,
            "optimizer_steps": optimizer_steps,
            "token_positions": token_positions,
            "global_batch_tokens": global_batch_tokens,
        }
        for field, expected in expected_fields.items():
            if record.get(field) != expected:
                _fail(f"exposure index {key}.{field} must equal {expected}")
        if plan.get("world_size") != world_size or plan.get("seed") != seed:
            _fail(f"training exposure plan {key} runtime binding mismatch")
        if plan.get("horizon") != {"unit": "token_positions", "value": token_positions}:
            _fail(f"training exposure plan {key} token horizon mismatch")
        if plan.get("derived", {}).get("optimizer_steps") != optimizer_steps:
            _fail(f"training exposure plan {key} optimizer-step mismatch")
        if plan.get("source_dataset_manifest_sha256") != dataset_sha:
            _fail(f"training exposure plan {key} dataset binding mismatch")
        if plan.get("tokenizer_sha256") != package_sha:
            _fail(f"training exposure plan {key} tokenizer binding mismatch")
        if plan.get("study_sha256") != recipe_sha:
            _fail(f"training exposure plan {key} recipe binding mismatch")
        verified_plans[key] = {
            "path": str(expected_path),
            "sha256": plan_sha,
            **expected_fields,
        }
    if exposure_index.get("study_manifest_sha256") != recipe_sha:
        _fail("exposure plan index family-recipe binding mismatch")
    if exposure_index.get("source_dataset_manifest_sha256") != dataset_sha:
        _fail("exposure plan index dataset binding mismatch")
    if exposure_index.get("tokenizer_artifact_sha256") != package_sha:
        _fail("exposure plan index tokenizer binding mismatch")
    expected_validation_record = {
        "path": paths["validation_exposure"].relative_to(paths["corpus_root"]).as_posix(),
        "sha256": validation_exposure_sha,
    }
    if exposure_index.get("validation") != expected_validation_record:
        _fail("exposure plan index validation binding mismatch")

    expected_mix_weights = {
        str(item["id"]): float(item["weight"])
        for item in _sequence(mixture.get("mixture"), "mixture config weights")
        if isinstance(item, Mapping)
    }
    if packing_capacity.get("intended_mixture_weights") != expected_mix_weights:
        _fail("packing-capacity mix gate used different intended mixture weights")
    training_environment = _mapping(
        recipe["code_provenance"].get("training_environment"),
        "code_provenance.training_environment",
    )
    pyproject_path = args.repo_root / "pyproject.toml"
    lock_path = args.repo_root / "uv.lock"
    if not pyproject_path.is_file() or not lock_path.is_file():
        _fail("pinned root pyproject.toml or uv.lock is missing")
    pyproject_sha = file_sha256(pyproject_path)
    lock_sha = file_sha256(lock_path)
    if pyproject_sha != training_environment["pyproject_sha256"]:
        _fail("root pyproject.toml differs from the reviewed upstream environment")
    if lock_sha != training_environment["uv_lock_sha256"]:
        _fail("root uv.lock differs from the reviewed upstream environment")
    expected_python = str(training_environment["python_version"])
    actual_python = ".".join(str(value) for value in sys.version_info[:3])
    if actual_python != expected_python:
        _fail(
            f"preflight Python must be {expected_python}, found {actual_python}"
        )
    try:
        uv_version_output = subprocess.check_output(
            ["uv", "--version"],
            cwd=args.repo_root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise FamilyWorkflowError("cannot verify the pinned uv binary") from exc
    uv_fields = uv_version_output.split()
    actual_uv_version = uv_fields[1] if len(uv_fields) >= 2 else ""
    if actual_uv_version != training_environment["uv_version"]:
        _fail(
            f"preflight uv must be {training_environment['uv_version']}, "
            f"found {actual_uv_version or uv_version_output!r}"
        )

    commit = _git_output(args.repo_root, "rev-parse", "HEAD")
    if SHA256_RE.fullmatch(commit) is None and re.fullmatch(r"^[0-9a-f]{40}$", commit) is None:
        _fail("Git HEAD is not a full immutable revision")
    dirty = bool(_git_output(args.repo_root, "status", "--porcelain"))
    if dirty and not args.allow_dirty:
        _fail("production/smoke preflight requires a clean Git worktree")
    code_policy = _mapping(recipe["code_provenance"], "code_provenance")
    upstream_revision = str(code_policy["upstream_base_revision"])
    try:
        subprocess.check_call(
            ["git", "cat-file", "-e", f"{upstream_revision}^{{commit}}"],
            cwd=args.repo_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        raise FamilyWorkflowError(
            f"pinned upstream base revision is unavailable locally: {upstream_revision}"
        ) from exc
    if code_policy.get("require_upstream_base_ancestor"):
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", upstream_revision, commit],
            cwd=args.repo_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if ancestor.returncode != 0:
            _fail(
                "the production revision does not descend from pinned upstream base "
                f"{upstream_revision}"
            )
    changed_core = set(
        line
        for line in _git_output(
            args.repo_root,
            "diff",
            "--name-only",
            upstream_revision,
            "--",
            *[str(path) for path in code_policy["core_scope"]],
        ).splitlines()
        if line
    )
    if changed_core:
        _fail(
            "production core differs from the pinned upstream base; only additive strict "
            "modules are permitted: " + ", ".join(sorted(changed_core))
        )
    core_hashes: dict[str, str] = {}
    for relative in code_policy["core_scope"]:
        core_path = args.repo_root / str(relative)
        if not core_path.is_file():
            _fail(f"reviewed training-core file is missing: {core_path}")
        core_hashes[str(relative)] = file_sha256(core_path)
    for relative, expected_hash in code_policy["exact_file_sha256"].items():
        actual_hash = core_hashes.get(relative)
        if actual_hash is None:
            _fail(f"exact-hash-pinned core file is outside core_scope: {relative}")
        if not hmac.compare_digest(actual_hash, _sha256(expected_hash, f"exact hash for {relative}")):
            _fail(
                f"exact-hash-pinned core file drifted: {relative}; "
                f"expected {expected_hash}, found {actual_hash}"
            )

    corpus_bytes = _directory_size(paths["corpus_root"])
    tokenizer_bytes = _directory_size(paths["tokenizer_root"])
    storage_policy = recipe["storage"]
    live_storage = storage_policy["uhem_live_quota"]
    free_bytes, live_storage_audit = _live_beegfs_storage(
        args.repo_root,
        uid=int(live_storage["uid"]),
        storage_pool_id=int(live_storage["storage_pool_id"]),
        path=base_dir,
    )
    required_free = int(storage_policy["required_free_bytes_at_training_preflight"])
    if free_bytes < required_free:
        _fail(
            "insufficient free storage: "
            f"need {required_free} bytes for retained checkpoints/transient writes/logs/headroom, "
            f"found {free_bytes}; existing artifacts will not be deleted"
        )

    budget = recipe["uhem_budget"]
    remaining_cpu_saat, quota_output_sha, quota_audit = _live_uhem_cpu_saat(
        args.repo_root, str(budget["account"]), str(budget["user"])
    )
    required_cpu_saat = float(budget["operational_ceiling_cpu_saat"])
    if remaining_cpu_saat < required_cpu_saat:
        _fail(
            "insufficient live UHeM quota: "
            f"need at least {required_cpu_saat:.0f} CPU-saat, found {remaining_cpu_saat:.2f}"
        )

    receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "d32_family_preflight_receipt",
            "family_id": recipe["family_id"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "base_dir": str(base_dir),
            "recipe": {"path": str(args.recipe), "canonical_sha256": recipe_sha},
            "mixture_config": {
                "path": str(mixture_path),
                "sha256": mixture_sha,
                "policy_sha256": policy_sha,
                "corpus_name": corpus_id,
            },
            "data_preparation_storage_gate_sha256": data_prep_gate_sha,
            "production_cluster_launch_receipt_sha256": cluster_launch_sha,
            "data_preparation_provenance": {
                "policy_sha256": policy_sha,
                "source_plan_sha256": source_plan_sha,
                "calibration_sha256": calibration_sha,
                "backend_resource_report_sha256": resource_approval[
                    "resource_report_sha256"
                ],
                "resource_approval_sha256": resource_approval_sha,
                "mixture_quality_approval_sha256": (
                    mixture_quality_approval_sha
                ),
                "sample_quality_audit_sha256": mixture_quality_approval[
                    "sample_quality_audit_sha256"
                ],
                "production_pack_plan_sha256": data_prep_gate[
                    "production_pack_plan_sha256"
                ],
            },
            "data_preparation_cpu_saat_with_safety": data_prep_cpu,
            "total_project_operational_ceiling_cpu_saat": expected_total_ceiling,
            "corpus": {
                "root": str(paths["corpus_root"]),
                "name": corpus_id,
                "manifest_sha256": corpus_sha,
                "parent_pool_manifest_sha256": parent_pool_sha,
                "qa_approval_sha256": archived_qa_sha,
                "production_chain": dict(production_chain),
                "dataset_manifest_sha256": dataset_sha,
                "source_receipt_sha256": source_sha,
                "validation_exposure_manifest_sha256": validation_exposure_sha,
                "validation_payload_bytes": validation_payload_bytes,
                "validation_documents": validation_documents,
                "exposure_plan_index_sha256": exposure_index_sha,
                "training_exposure_plans": verified_plans,
                "packing_capacity_receipt": {
                    "path": str(paths["packing_capacity"]),
                    "sha256": packing_capacity_sha,
                    "gate_passed": True,
                    "worlds": {
                        world: capacity_world_gate_record(
                            packing_capacity, int(world)
                        )
                        for world in ("8", "16")
                    },
                },
                "validation_file": validation_relative,
                "validation_file_size_bytes": validation_path.stat().st_size,
                "validation_file_sha256": file_sha256(validation_path),
                "actual_bytes": corpus_bytes,
                **(
                    {"macocu_preparation_manifest": macocu_record}
                    if macocu_record is not None
                    else {}
                ),
                **anchor_records,
            },
            "tokenizer": {
                "root": str(paths["tokenizer_root"]),
                "name": recipe["artifacts"]["tokenizer_name"],
                "vocab_size": recipe["model"]["vocab_size"],
                "package_manifest_sha256": package_sha,
                "actual_bytes": tokenizer_bytes,
            },
            "code": {
                "git_commit": commit,
                "git_dirty": dirty,
                "upstream_base_revision": upstream_revision,
                "changed_core_paths": sorted(changed_core),
                "core_file_sha256": core_hashes,
                "pyproject_sha256": pyproject_sha,
                "uv_lock_sha256": lock_sha,
                "uv_version": actual_uv_version,
                "python_version": actual_python,
                "environment_sync_mode": training_environment["sync_mode"],
            },
            "storage": {
                **live_storage_audit,
                "required_free_bytes": required_free,
                "never_auto_delete_existing_artifacts": True,
            },
            "uhem_quota": {
                "account": budget["account"],
                "user": budget["user"],
                "remaining_cpu_saat": remaining_cpu_saat,
                "required_operational_ceiling_cpu_saat": required_cpu_saat,
                "sshare_output_sha256": quota_output_sha,
                **quota_audit,
            },
            "canonical_sha256": None,
        }
    )
    write_json_atomic(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))


def _load_receipt(path: Path, expected_kind: str) -> tuple[dict[str, Any], str]:
    value, digest = _verify_sealed(path, expected_kind)
    if value.get("kind") != expected_kind:
        _fail(f"{path} is not a {expected_kind}")
    return value, digest


def _verify_attention_probe(
    path: Path,
    *,
    recipe: Mapping[str, Any],
    recipe_sha: str,
    preflight_sha: str,
    code_revision: str,
) -> tuple[dict[str, Any], str]:
    probe, digest = _load_receipt(path, "d32_attention_backend_probe")
    expected = {
        "family_id": recipe["family_id"],
        "recipe_sha256": recipe_sha,
        "preflight_receipt_sha256": preflight_sha,
        "code_revision": code_revision,
        "world_size": 1,
    }
    for key, value in expected.items():
        if probe.get(key) != value:
            _fail(f"attention probe {key} mismatch")
    gate = recipe["attention_backend_gate"]
    gpu = _mapping(probe.get("gpu"), "attention probe GPU")
    if str(gpu.get("name", "")).upper().find(str(gate["required_gpu_family"])) < 0:
        _fail("attention probe did not run on the required A100 family")
    detection = _mapping(
        probe.get("module_detection"), "attention probe module_detection"
    )
    selected_backend = detection.get("selected_backend_after_probe")
    selected_pattern = detection.get("selected_window_pattern")
    selected_pair = (selected_backend, selected_pattern)
    preferred_pair = (gate["preferred_backend"], gate["preferred_window_pattern"])
    fallback_pair = (gate["fallback_backend"], gate["fallback_window_pattern"])
    if selected_pair not in {preferred_pair, fallback_pair}:
        _fail("attention probe selected a backend/window pair outside the recipe policy")
    expected_flash_hash = recipe["code_provenance"]["exact_file_sha256"][
        "nanochat/flash_attention.py"
    ]
    if detection.get("flash_attention_file_sha256") != expected_flash_hash:
        _fail("attention probe used an unreviewed flash_attention.py")
    check = _mapping(
        probe.get("selected_d32_model_forward_backward"),
        "attention probe selected d32 model correctness",
    )
    expected_d32_config = {
        "depth": 32,
        "model_dim": 2048,
        "num_heads": 16,
        "num_kv_heads": 16,
        "head_dim": 128,
        "max_seq_len": 2048,
        "vocab_size": 32768,
    }
    expected_check = {
        "backend": selected_backend,
        "window_pattern": selected_pattern,
        "config": expected_d32_config,
        "construction": "meta_then_to_empty_cuda_then_literal_seed42_init_weights",
        "initialization_seed": 42,
        "batch_sequences": 1,
        "sequence_length": 2048,
        "compute_dtype": "torch.bfloat16",
    }
    for field, value in expected_check.items():
        if check.get(field) != value:
            _fail(f"selected d32 attention smoke {field} mismatch")
    for field in ("output_finite", "loss_finite", "gradients_present", "gradients_finite"):
        if check.get(field) is not True:
            _fail(f"selected d32 attention smoke failed {field}")
    if not isinstance(check.get("gradient_tensor_count"), int) or check[
        "gradient_tensor_count"
    ] <= 0:
        _fail("selected d32 attention smoke did not record gradient tensors")
    if not isinstance(check.get("peak_cuda_memory_bytes"), int) or check[
        "peak_cuda_memory_bytes"
    ] <= 0:
        _fail("selected d32 attention smoke did not record peak CUDA memory")
    parameter_dtypes = _mapping(
        check.get("parameter_dtype_inventory"),
        "selected d32 attention parameter dtype inventory",
    )
    if not {"torch.bfloat16", "torch.float32"}.issubset(parameter_dtypes):
        _fail("selected d32 attention smoke lacks the expected BF16/FP32 parameters")
    parameter_elements = 0
    for dtype, raw_record in parameter_dtypes.items():
        record = _mapping(raw_record, f"selected d32 parameter dtype {dtype}")
        _positive_int(record.get("tensor_count"), f"selected d32 {dtype} tensor count")
        parameter_elements += _positive_int(
            record.get("element_count"), f"selected d32 {dtype} element count"
        )
    if parameter_elements != recipe["model"]["total_parameters"]:
        _fail("selected d32 attention parameter inventory has the wrong total")
    buffer_dtypes = _mapping(
        check.get("buffer_dtype_inventory"),
        "selected d32 attention buffer dtype inventory",
    )
    if "torch.bfloat16" not in buffer_dtypes:
        _fail("selected d32 attention smoke lacks BF16 rotary-buffer evidence")
    reason = probe.get("selection_reason")
    if not isinstance(reason, str) or not reason:
        _fail("attention probe does not record a selection/fallback reason")
    if selected_pair == preferred_pair:
        if probe.get("decision") != "accepted_fa3_SSSL":
            _fail("preferred FA3+SSSL selection has the wrong decision")
        if detection.get("HAS_FA3_at_import") is not True or detection.get(
            "USE_FA3_at_import"
        ) is not True:
            _fail("FA3 may be selected only when pinned upstream auto-detection selected it")
        if probe.get("fa3_actual_d32_smoke_passed") is not True:
            _fail("FA3+SSSL did not pass the actual-d32 finite smoke")
    else:
        if probe.get("decision") != "accepted_sdpa_L_fallback":
            _fail("SDPA+L fallback selection has the wrong decision")
        if probe.get("fa3_actual_d32_smoke_passed") is True:
            _fail("probe fell back to SDPA despite a passing upstream-auto FA3 d32 smoke")
    if probe.get("fa3_sdpa_comparison_decisional") is not False:
        _fail("diagnostic FA3-vs-SDPA comparisons must not decide production selection")
    benchmarks = _mapping(probe.get("pattern_benchmarks"), "pattern benchmarks")
    if set(benchmarks) != {"L", "SSSL"}:
        _fail("attention probe lacks the L-versus-SSSL comparison")
    for pattern, result in benchmarks.items():
        record = _mapping(result, f"pattern benchmark {pattern}")
        if record.get("pattern") != pattern or float(record.get("median_seconds", 0)) <= 0:
            _fail(f"attention probe benchmark {pattern} is invalid")
    return probe, digest


def _production_identity(
    recipe_sha: str,
    preflight: Mapping[str, Any],
    *,
    attention_probe_sha256: str,
    proxy_approval_sha256: str,
    accepted_base_weight_decay: float,
    accepted_weight_decay_cooldown_policy: str,
) -> dict[str, Any]:
    return {
        "recipe_sha256": recipe_sha,
        "mixture_config_sha256": preflight["mixture_config"]["sha256"],
        "corpus_manifest_sha256": preflight["corpus"]["manifest_sha256"],
        "dataset_manifest_sha256": preflight["corpus"]["dataset_manifest_sha256"],
        "validation_file_sha256": preflight["corpus"]["validation_file_sha256"],
        "tokenizer_package_sha256": preflight["tokenizer"]["package_manifest_sha256"],
        "code_revision": preflight["code"]["git_commit"],
        "attention_probe_sha256": attention_probe_sha256,
        "wsd_proxy_approval_sha256": proxy_approval_sha256,
        "wsd_base_weight_decay": accepted_base_weight_decay,
        "wsd_weight_decay_cooldown": accepted_weight_decay_cooldown_policy,
        "depth": 32,
        "global_batch_tokens": 2_097_152,
        "device_batch_sequences": 4,
        "max_seq_len": 2048,
        "optimizer": "muon_adamw",
        "lr_schedule": "wsd",
        "gradient_clip_norm": 0.0,
        "target_param_count": "scaling",
        "target_param_data_ratio": -1.0,
    }


def command_attention_env(args: argparse.Namespace) -> None:
    recipe, recipe_sha = load_recipe(args.recipe)
    preflight, preflight_sha = _load_receipt(
        args.preflight_receipt, "d32_family_preflight_receipt"
    )
    if preflight["recipe"]["canonical_sha256"] != recipe_sha:
        _fail("attention environment preflight recipe mismatch")
    probe, probe_sha = _verify_attention_probe(
        args.attention_probe,
        recipe=recipe,
        recipe_sha=recipe_sha,
        preflight_sha=preflight_sha,
        code_revision=preflight["code"]["git_commit"],
    )
    detection = _mapping(probe["module_detection"], "attention probe detection")
    values = {
        "ATTENTION_BACKEND": str(detection["selected_backend_after_probe"]),
        "WINDOW_PATTERN": str(detection["selected_window_pattern"]),
        "ATTENTION_PROBE_SHA256": probe_sha,
        "CODE_REVISION": str(preflight["code"]["git_commit"]),
    }
    for key, value in values.items():
        print(f"{key}={shlex.quote(value)}")


def _verify_frozen_optimizer_protocol(
    protocol: Mapping[str, Any],
    *,
    label: str,
) -> Mapping[str, Any]:
    """Verify settings that must never be inherited from mutable CLI defaults."""

    optimizer = _mapping(protocol.get("optimizer"), f"{label} optimizer")
    if optimizer.get("gradient_clip_norm") != 0.0:
        _fail(f"{label} must bind gradient_clip_norm=0.0")
    return optimizer


def _verify_frozen_protocol(
    protocol: Mapping[str, Any],
    *,
    recipe: Mapping[str, Any],
    preflight: Mapping[str, Any],
    attention_probe: Mapping[str, Any],
    attention_probe_sha256: str,
    label: str,
    run_kind: str,
    recipe_scope: str,
    model_tag: str,
    exposure_plan_sha256: str,
    depth: int,
    model_dim: int,
    world_size: int,
    device_batch_size: int,
    total_batch_size: int,
    num_iterations: int,
    eval_every_updates: int,
    seed: int,
    production_gate: Mapping[str, Any] | None = None,
    production_gate_sha256: str | None = None,
) -> Mapping[str, Any]:
    optimizer = _verify_frozen_optimizer_protocol(protocol, label=label)
    if protocol.get("protocol_version") != "d32_wsd_strict_v1":
        _fail(f"{label} strict protocol version mismatch")
    exact_scalars = {
        "run_kind": run_kind,
        "recipe_scope": recipe_scope,
        "model_tag": model_tag,
        "source_dataset_manifest_sha256": preflight["corpus"]["dataset_manifest_sha256"],
        "data_order": recipe["training"]["data_order"],
        "data_order_authority": (
            "sealed_dataset_manifest_materialization_order_with_"
            "upstream_row_group_rank_sharding"
        ),
        "seed": seed,
        "world_size": world_size,
        "device_batch_size": device_batch_size,
        "total_batch_size": total_batch_size,
        "num_iterations": num_iterations,
    }
    for key, expected in exact_scalars.items():
        if protocol.get(key) != expected:
            _fail(f"{label} {key} must equal {expected!r}")
    code = _mapping(protocol.get("code"), f"{label} code provenance")
    expected_code = {
        "git_revision": preflight["code"]["git_commit"],
        "upstream_base_revision": recipe["code_provenance"]["upstream_base_revision"],
        "environment_lock_sha256": preflight["code"]["uv_lock_sha256"],
        "exact_core_sha256": preflight["code"]["core_file_sha256"],
    }
    if code != expected_code:
        _fail(f"{label} code provenance differs from the sealed preflight")
    selected = _mapping(
        attention_probe.get("module_detection"), f"{label} attention selection"
    )
    selected_backend = selected.get("selected_backend_after_probe")
    selected_pattern = selected.get("selected_window_pattern")
    architecture = _mapping(protocol.get("architecture_cli"), f"{label} architecture")
    expected_architecture = {
        "depth": depth,
        "aspect_ratio": 64,
        "head_dim": 128,
        "max_seq_len": 2048,
        "window_pattern": selected_pattern,
    }
    for field, expected in expected_architecture.items():
        if architecture.get(field) != expected:
            _fail(f"{label} architecture CLI {field} differs from frozen recipe")
    total_parameters = _positive_int(
        architecture.get("total_parameters"), f"{label} total parameters"
    )
    scaling_parameters = _positive_int(
        architecture.get("scaling_parameters"), f"{label} scaling parameters"
    )
    if depth == recipe["model"]["depth"]:
        if total_parameters != recipe["model"]["total_parameters"]:
            _fail(f"{label} total parameter count differs from the d32 recipe")
        if scaling_parameters != recipe["model"]["scaling_parameters"]:
            _fail(f"{label} scaling parameter count differs from the d32 recipe")
    else:
        proxy_stage = (
            recipe["weight_decay_proxy_ablation"]["screen_stage"]
            if depth == 12
            else recipe["weight_decay_proxy_ablation"]["confirmation_stage"]
        )
        if scaling_parameters != proxy_stage["scaling_parameters"]:
            _fail(f"{label} proxy scaling parameter count differs from the recipe")
    model_config = _mapping(protocol.get("model_config"), f"{label} model_config")
    expected_model_config = {
        "sequence_len": 2048,
        "vocab_size": 32768,
        "n_layer": depth,
        "n_head": model_dim // 128,
        "n_kv_head": model_dim // 128,
        "n_embd": model_dim,
        "window_pattern": selected_pattern,
    }
    if model_config != expected_model_config:
        _fail(f"{label} realized model config differs from frozen MHA architecture")
    tokenizer = _mapping(protocol.get("tokenizer"), f"{label} tokenizer")
    expected_tokenizer = {
        "name": recipe["artifacts"]["tokenizer_name"],
        "artifact_sha256": preflight["tokenizer"]["package_manifest_sha256"],
        "vocab_size": 32768,
    }
    if tokenizer != expected_tokenizer:
        _fail(f"{label} tokenizer binding mismatch")
    precision = _mapping(protocol.get("precision"), f"{label} precision")
    if precision != {"compute_dtype": "torch.bfloat16", "fp8_enabled": False}:
        _fail(f"{label} must use BF16 with FP8 explicitly disabled")
    attention = _mapping(protocol.get("attention"), f"{label} attention")
    expected_attention = {
        "backend": selected_backend,
        "window_pattern": selected_pattern,
        "probe_sha256": attention_probe_sha256,
        "selection_reason": attention_probe["selection_reason"],
        "decision": attention_probe["decision"],
        "live_fa3_kernel_inventory_sha256": (
            _mapping(
                selected.get("fa3_kernel_inventory"),
                f"{label} FA3 kernel inventory",
            ).get("inventory_sha256")
            if selected_backend == "fa3"
            else None
        ),
    }
    if attention != expected_attention:
        _fail(f"{label} attention identity differs from the sealed A100 probe")
    validation = _mapping(protocol.get("validation"), f"{label} validation")
    expected_validation = {
        "manifest_sha256": preflight["corpus"]["validation_exposure_manifest_sha256"],
        "payload_bytes": preflight["corpus"]["validation_payload_bytes"],
        "documents": preflight["corpus"]["validation_documents"],
        "full_manifest": True,
        "packing_policy": "whole_document_no_crop_rows_before_rank_sharding",
        "bos_boundary_targets_masked": True,
        "padding_targets_masked": True,
        "eval_every_updates": eval_every_updates,
        "eval_tokens_cli_unused": recipe["training"]["evaluation"][
            "eval_tokens_cli_unused"
        ],
    }
    for field, expected in expected_validation.items():
        if validation.get(field) != expected:
            _fail(f"{label} fixed-validation {field} mismatch")
    target_tokens = _positive_int(
        validation.get("target_tokens"), f"{label} validation target tokens"
    )
    logical_rows = _positive_int(
        validation.get("logical_rows"), f"{label} validation logical rows"
    )
    padded_world1 = _positive_int(
        validation.get("padded_token_positions_world1"),
        f"{label} validation padded positions world1",
    )
    if padded_world1 != logical_rows * 2048 or target_tokens > padded_world1:
        _fail(f"{label} validation row/token arithmetic mismatch")
    expected_runtime_padding = (
        math.ceil(logical_rows / (device_batch_size * world_size))
        * device_batch_size
        * 2048
        * world_size
    )
    if validation.get("padded_token_positions_runtime_world") != expected_runtime_padding:
        _fail(f"{label} distributed validation padding arithmetic mismatch")
    _sha256(validation.get("row_layout_sha256"), f"{label} validation row layout")
    if set(validation) != {
        *expected_validation,
        "target_tokens",
        "logical_rows",
        "row_layout_sha256",
        "padded_token_positions_world1",
        "padded_token_positions_runtime_world",
    }:
        _fail(f"{label} validation protocol has missing or unexpected fields")
    packing_capacity = _mapping(
        protocol.get("packing_capacity"), f"{label} packing capacity"
    )
    topology = protocol.get("topology")
    if run_kind in {"production", "smoke"}:
        capacity_record = _mapping(
            preflight["corpus"].get("packing_capacity_receipt"),
            f"{label} preflight packing capacity",
        )
        expected_capacity_kind = (
            "turkish_bestfit_repeat_capacity_receipt"
            if recipe["family_id"] == FAMILY_ID_V3
            else "turkish_bestfit_capacity_receipt"
        )
        capacity_receipt, capacity_sha = _load_receipt(
            Path(str(capacity_record["path"])), expected_capacity_kind
        )
        if capacity_sha != capacity_record.get("sha256"):
            _fail(f"{label} packing-capacity file differs from preflight")
        try:
            selected_capacity = capacity_world_gate_record(
                capacity_receipt, world_size
            )
        except StrictTrainingError as exc:
            raise FamilyWorkflowError(
                f"{label} packing-capacity topology is invalid: {exc}"
            ) from exc
        if packing_capacity != {
            "receipt_sha256": capacity_sha,
            "selected_topology": selected_capacity,
        }:
            _fail(f"{label} selected packing capacity differs from the sealed receipt")
    else:
        if packing_capacity != {"receipt_sha256": None, "selected_topology": None}:
            _fail(f"{label} run claims an unsupported packing-capacity selection")

    if run_kind == "production":
        if production_gate is None or production_gate_sha256 is None:
            _fail(f"{label} production protocol requires its topology gate")
        expected_topology = {
            "gate_sha256": production_gate_sha256,
            "authorized_world_size": world_size,
            "authorized_nodes": production_gate["authorized_production_nodes"],
            "selection_reason": production_gate["selection_reason"],
            "require_single_world_size_for_entire_lineage": True,
        }
        if topology != expected_topology:
            _fail(f"{label} topology identity differs from the production gate")
    else:
        if topology is not None:
            _fail(f"{label} non-production run claims a production topology gate")
    checkpointing = _mapping(protocol.get("checkpointing"), f"{label} checkpointing")
    if checkpointing != {"transactional": True, "save_every_updates": -1}:
        _fail(f"{label} checkpoint policy mismatch")
    preemption = _mapping(protocol.get("preemption"), f"{label} preemption")
    if preemption != {
        "signals": ["SIGUSR1", "SIGTERM"],
        "checkpoint_boundary": "next_optimizer_safe_update",
        "exit_code": 75,
    }:
        _fail(f"{label} preemption protocol mismatch")
    exposure_plan = _mapping(protocol.get("exposure_plan"), f"{label} exposure plan")
    try:
        realized_exposure_sha = verify_manifest_hash(exposure_plan)
    except ValueError as exc:
        raise FamilyWorkflowError(f"{label} embedded exposure plan is invalid: {exc}") from exc
    if realized_exposure_sha != exposure_plan_sha256:
        _fail(f"{label} embedded exposure plan differs from its preflight hash")
    if exposure_plan.get("world_size") != world_size or exposure_plan.get("seed") != seed:
        _fail(f"{label} embedded exposure plan runtime binding mismatch")
    if exposure_plan.get("horizon") != {
        "unit": "token_positions",
        "value": num_iterations * total_batch_size,
    }:
        _fail(f"{label} embedded exposure horizon mismatch")
    expected_lr_scale = math.sqrt(total_batch_size / 524_288)
    expected_lrs = {
        "embedding": 0.3 * expected_lr_scale,
        "unembedding": 0.008 * expected_lr_scale,
        "matrix": 0.02 * expected_lr_scale,
        "scalar": 0.5 * expected_lr_scale,
    }
    actual_lrs = _mapping(optimizer.get("learning_rates"), f"{label} learning rates")
    if set(actual_lrs) != set(expected_lrs) or any(
        not math.isclose(
            float(actual_lrs[key]), expected, rel_tol=1e-15, abs_tol=1e-15
        )
        for key, expected in expected_lrs.items()
    ):
        _fail(f"{label} batch-scaled optimizer learning rates drifted")
    return optimizer


def _verify_proxy_acceptance(
    path: Path,
    *,
    recipe: Mapping[str, Any],
    recipe_sha: str,
    preflight: Mapping[str, Any],
    attention_probe_sha256: str,
) -> tuple[dict[str, Any], str]:
    approval, digest = _load_receipt(path, "wsd_proxy_acceptance")
    proxy = recipe["weight_decay_proxy_ablation"]
    expected = {
        "decision": "accepted",
        "recipe_version": proxy["recipe_version"],
        "study_manifest_sha256": recipe_sha,
        "tokenizer_artifact_sha256": preflight["tokenizer"]["package_manifest_sha256"],
        "source_dataset_manifest_sha256": preflight["corpus"]["dataset_manifest_sha256"],
        "trainer_code_revision": preflight["code"]["git_commit"],
        "gradient_clip_norm": 0.0,
        "attention_probe_sha256": attention_probe_sha256,
        "weight_decay_transfer_rule": proxy["weight_decay_transfer_rule"],
        "production_scaling_parameters": proxy["production_scaling_parameters"],
        "production_global_batch_tokens": proxy["production_global_batch_tokens"],
    }
    for key, value in expected.items():
        if approval.get(key) != value:
            _fail(f"weight-decay proxy approval {key} mismatch")
    accepted = approval.get("accepted_base_weight_decay")
    candidates = {
        candidate["id"]: candidate
        for candidate in proxy["candidates"]
    }
    accepted_policy = approval.get("accepted_weight_decay_cooldown_policy")
    if not any(
        candidate["production_base_weight_decay"] == accepted
        and candidate["cooldown_weight_decay"] == accepted_policy
        and candidate["eligible_for_production"] is True
        for candidate in candidates.values()
    ):
        _fail("proxy approval selected a weight decay outside the reviewed candidates")
    results = approval.get("candidate_results")
    if not isinstance(results, list) or {
        result.get("id") for result in results if isinstance(result, Mapping)
    } != set(candidates):
        _fail("proxy approval candidate_results do not cover the reviewed candidates")
    return approval, digest


def command_seal_proxy_run(args: argparse.Namespace) -> None:
    recipe, recipe_sha = load_recipe(args.recipe)
    preflight, preflight_sha = _load_receipt(
        args.preflight_receipt, "d32_family_preflight_receipt"
    )
    if preflight["recipe"]["canonical_sha256"] != recipe_sha:
        _fail("proxy run preflight recipe mismatch")
    _attention_probe, attention_probe_sha = _verify_attention_probe(
        args.attention_probe,
        recipe=recipe,
        recipe_sha=recipe_sha,
        preflight_sha=preflight_sha,
        code_revision=preflight["code"]["git_commit"],
    )
    proxy = recipe["weight_decay_proxy_ablation"]
    proxy_stage_name = "screen_stage" if args.model_depth == 12 else "confirmation_stage"
    proxy_stage = _mapping(proxy[proxy_stage_name], proxy_stage_name)
    if proxy_stage["model_depth"] != args.model_depth:
        _fail("proxy stage/model depth mismatch")
    candidate_map = {
        candidate["id"]: candidate
        for candidate in proxy["candidates"]
    }
    if args.candidate_id not in candidate_map:
        _fail(f"unknown weight-decay proxy candidate: {args.candidate_id}")
    if args.seed not in proxy_stage["seeds"]:
        _fail(f"unreviewed weight-decay proxy seed: {args.seed}")
    candidate = candidate_map[args.candidate_id]
    expected_plan_key = f"proxy_d{args.model_depth}_seed{args.seed}_ws1"
    expected_plan = preflight["corpus"]["training_exposure_plans"].get(expected_plan_key)
    if not isinstance(expected_plan, Mapping):
        _fail(f"preflight receipt lacks {expected_plan_key}")

    final_step = int(proxy_stage["updates"])
    from nanochat.strict_checkpoint import inspect_strict_checkpoint

    try:
        checkpoint = inspect_strict_checkpoint(args.checkpoint_root, final_step)
    except Exception as exc:
        raise FamilyWorkflowError(f"proxy final checkpoint verification failed: {exc}") from exc
    checkpoint_sha = verify_manifest_hash(checkpoint)
    if checkpoint.get("expected_world_size") != proxy_stage["world_size"]:
        _fail("proxy checkpoint world size mismatch")
    identity = _mapping(checkpoint.get("identity"), "proxy checkpoint identity")
    expected_proxy_run_id = (
        f"{recipe['family_id']}_proxy_d{args.model_depth}_"
        f"{args.candidate_id}_seed{args.seed}"
    )
    rank_exit, rank_exit_sha = _load_receipt(
        args.rank_exit_receipt, "d32_batch_direct_rank_exit"
    )
    expected_rank_exit = {
        "family_id": recipe["family_id"],
        "recipe_sha256": recipe_sha,
        "run_id": expected_proxy_run_id,
        "phase": "proxy_train",
        "slurm_job_id": args.slurm_job_id,
        "rank": 0,
        "local_rank": 0,
        "world_size": 1,
        "launcher": "slurm_batch_direct_python_env_v1",
        "child_exit_code": 0,
        "termination": "clean",
    }
    for field, value in expected_rank_exit.items():
        if rank_exit.get(field) != value:
            _fail(f"proxy rank-exit receipt {field} mismatch")
    if identity.get("study_id") != recipe["family_id"] or identity.get(
        "run_id"
    ) != expected_proxy_run_id:
        _fail("proxy checkpoint study/run identity mismatch")
    protocol = _mapping(identity.get("protocol"), "proxy checkpoint protocol")
    architecture = _mapping(protocol.get("architecture_cli"), "proxy checkpoint architecture")
    if architecture.get("depth") != args.model_depth:
        _fail("proxy checkpoint model depth mismatch")
    optimizer = _verify_frozen_protocol(
        protocol,
        recipe=recipe,
        preflight=preflight,
        attention_probe=_attention_probe,
        attention_probe_sha256=attention_probe_sha,
        label="proxy checkpoint protocol",
        run_kind="proxy",
        recipe_scope=f"proxy_d{args.model_depth}",
        model_tag=expected_proxy_run_id,
        exposure_plan_sha256=str(expected_plan["sha256"]),
        depth=args.model_depth,
        model_dim=int(proxy_stage["model_dim"]),
        world_size=int(proxy_stage["world_size"]),
        device_batch_size=int(proxy_stage["device_batch_sequences"]),
        total_batch_size=int(proxy_stage["global_batch_tokens"]),
        num_iterations=final_step,
        eval_every_updates=int(proxy_stage["validation_every_updates"]),
        seed=args.seed,
    )
    stage_key = f"d{args.model_depth}"
    effective_weight_decay = float(candidate["stage_effective_weight_decay"][stage_key])
    if not math.isclose(
        float(optimizer.get("muon_base_weight_decay", float("nan"))),
        effective_weight_decay,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        _fail("proxy checkpoint effective base weight decay differs from candidate rule")
    schedule = _mapping(protocol.get("schedule"), "proxy checkpoint schedule")
    if schedule.get("name") != candidate["schedule"]:
        _fail("proxy checkpoint schedule differs from candidate")
    if candidate["schedule"] == "wsd":
        expected_policy = candidate["cooldown_weight_decay"]
        if schedule.get("recipe_version") != proxy["recipe_version"]:
            _fail("proxy checkpoint WSD recipe version mismatch")
        if schedule.get("warmup_steps") != 40 or schedule.get("momentum_warmup_steps") != 400:
            _fail("proxy checkpoint WSD warmup policy mismatch")
        if not math.isclose(
            float(schedule.get("stable_muon_weight_decay", float("nan"))),
            effective_weight_decay,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            _fail("proxy checkpoint schedule effective WD mismatch")
        if schedule.get("proxy_approval_sha256") is not None:
            _fail("proxy WSD run unexpectedly claims a production approval")
        if schedule.get("weight_decay_cooldown_policy") != expected_policy:
            _fail("proxy checkpoint WSD cooldown policy differs from candidate")
        if schedule.get("cooldown_start_step") != final_step - final_step // 10:
            _fail("proxy checkpoint does not use an exact 10% WSD cooldown")
    if identity.get("study_manifest_sha256") != recipe_sha:
        _fail("proxy checkpoint recipe hash mismatch")
    if identity.get("tokenizer_artifact_sha256") != preflight["tokenizer"]["package_manifest_sha256"]:
        _fail("proxy checkpoint tokenizer hash mismatch")
    if identity.get("exposure_plan_sha256") != expected_plan["sha256"]:
        _fail("proxy checkpoint exposure-plan hash mismatch")

    events, state = read_training_log(args.curve_log)
    validation_points = [
        event
        for event in events
        if event.get("event_type") == "validation"
        and int(event["updates_completed"]) <= final_step
    ]
    count = int(proxy_stage["final_validation_points"])
    if len(validation_points) < count:
        _fail(f"proxy run has fewer than {count} fixed-validation points")
    selected = validation_points[-count:]
    every = int(proxy_stage["validation_every_updates"])
    all_expected_steps = list(range(0, final_step + 1, every))
    if not all_expected_steps or all_expected_steps[-1] != final_step:
        all_expected_steps.append(final_step)
    expected_steps = all_expected_steps[-count:]
    actual_steps = [int(event["updates_completed"]) for event in selected]
    if actual_steps != expected_steps:
        _fail(
            f"proxy final validation steps must equal {expected_steps}; found {actual_steps}"
        )
    bpbs = [float(event["metrics"]["val/bpb"]) for event in selected]
    if any(not (0.0 < value < 100.0) for value in bpbs):
        _fail("proxy validation contains a non-finite or implausible BPB")
    maximum_range = float(proxy["acceptance_rule"]["maximum_last5_bpb_range"])
    maximum_regression = float(
        proxy["acceptance_rule"]["maximum_final_minus_best_bpb"]
    )
    if max(bpbs) - min(bpbs) > maximum_range:
        _fail("proxy final fixed-validation BPB range exceeds the stability gate")
    if bpbs[-1] - min(bpbs) > maximum_regression:
        _fail("proxy final fixed-validation BPB regressed beyond the stability gate")
    receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "wsd_proxy_run_receipt",
            "family_id": recipe["family_id"],
            "recipe_version": proxy["recipe_version"],
            "candidate_id": args.candidate_id,
            "model_depth": args.model_depth,
            "schedule": candidate["schedule"],
            "input_weight_decay": candidate.get("input_weight_decay"),
            "production_base_weight_decay": candidate["production_base_weight_decay"],
            "effective_base_weight_decay": effective_weight_decay,
            "weight_decay_transfer_rule": proxy["weight_decay_transfer_rule"],
            "stage_scaling_parameters": proxy_stage["scaling_parameters"],
            "stage_global_batch_tokens": proxy_stage["global_batch_tokens"],
            "weight_decay_cooldown_policy": candidate["cooldown_weight_decay"],
            "eligible_for_production": candidate["eligible_for_production"],
            "seed": args.seed,
            "world_size": proxy_stage["world_size"],
            "optimizer_steps": final_step,
            "preflight_receipt_sha256": preflight_sha,
            "study_manifest_sha256": recipe_sha,
            "tokenizer_artifact_sha256": preflight["tokenizer"]["package_manifest_sha256"],
            "source_dataset_manifest_sha256": preflight["corpus"]["dataset_manifest_sha256"],
            "trainer_code_revision": preflight["code"]["git_commit"],
            "gradient_clip_norm": 0.0,
            "attention_probe_sha256": attention_probe_sha,
            "exposure_plan_sha256": expected_plan["sha256"],
            "checkpoint_sha256": checkpoint_sha,
            "curve_log_sha256": file_sha256(args.curve_log),
            "rank_exit_receipt_sha256": rank_exit_sha,
            "curve_log_terminal_event_sha256": state.last_event_sha256,
            "final_validation_steps": actual_steps,
            "final_validation_bpb": bpbs,
            "mean_final_validation_bpb": sum(bpbs) / len(bpbs),
            "slurm_job_id": args.slurm_job_id,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))


def command_proxy_env(args: argparse.Namespace) -> None:
    recipe, _recipe_sha = load_recipe(args.recipe)
    proxy = recipe["weight_decay_proxy_ablation"]
    candidates = {candidate["id"]: candidate for candidate in proxy["candidates"]}
    candidate = candidates.get(args.candidate_id)
    if not isinstance(candidate, Mapping):
        _fail(f"unknown proxy candidate: {args.candidate_id}")
    stage = proxy["screen_stage"] if args.model_depth == 12 else proxy["confirmation_stage"]
    if args.seed not in stage["seeds"]:
        _fail(f"seed {args.seed} is not frozen for d{args.model_depth}")
    stage_key = f"d{args.model_depth}"
    run_id = (
        f"{recipe['family_id']}_proxy_{stage_key}_{args.candidate_id}_seed{args.seed}"
    )
    values = {
        "RUN_ID": run_id,
        "MODEL_TAG": run_id,
        "DEPTH": str(args.model_depth),
        "MODEL_DIM": str(stage["model_dim"]),
        "DEVICE_BATCH_SIZE": str(stage["device_batch_sequences"]),
        "TOTAL_BATCH_SIZE": str(stage["global_batch_tokens"]),
        "NUM_ITERATIONS": str(stage["updates"]),
        "STOP_AT_STEP": str(stage["updates"]),
        "SEED": str(args.seed),
        "EVAL_EVERY": str(stage["validation_every_updates"]),
        "EXPOSURE_PLAN_KEY": f"proxy_{stage_key}_seed{args.seed}_ws1",
        "LR_SCHEDULE": str(candidate["schedule"]),
        "EFFECTIVE_BASE_WEIGHT_DECAY": str(
            candidate["stage_effective_weight_decay"][stage_key]
        ),
        "INPUT_WEIGHT_DECAY": str(candidate.get("input_weight_decay", 0.28)),
        "WSD_WEIGHT_DECAY_COOLDOWN": str(candidate["cooldown_weight_decay"]),
        "WSD_COOLDOWN_START_STEP": (
            str(int(stage["updates"]) - int(stage["updates"]) // 10)
            if candidate["schedule"] == "wsd"
            else "-1"
        ),
    }
    for key, value in values.items():
        print(f"{key}={shlex.quote(value)}")


def _proxy_run_cells(
    paths: Sequence[Path],
    *,
    expected_depth: int,
    expected_cells: set[tuple[str, int]],
    candidates: Mapping[str, Mapping[str, Any]],
    proxy: Mapping[str, Any],
    recipe_sha: str,
    preflight: Mapping[str, Any],
    preflight_sha: str,
    attention_probe_sha: str,
) -> dict[tuple[str, int], tuple[dict[str, Any], str]]:
    cells: dict[tuple[str, int], tuple[dict[str, Any], str]] = {}
    for path in paths:
        receipt, digest = _load_receipt(path, "wsd_proxy_run_receipt")
        key = (str(receipt.get("candidate_id")), int(receipt.get("seed", -1)))
        if key in cells:
            _fail(f"duplicate proxy cell receipt: {key}")
        expected_common = {
            "preflight_receipt_sha256": preflight_sha,
            "study_manifest_sha256": recipe_sha,
            "tokenizer_artifact_sha256": preflight["tokenizer"]["package_manifest_sha256"],
            "source_dataset_manifest_sha256": preflight["corpus"]["dataset_manifest_sha256"],
            "trainer_code_revision": preflight["code"]["git_commit"],
            "gradient_clip_norm": 0.0,
            "attention_probe_sha256": attention_probe_sha,
            "recipe_version": proxy["recipe_version"],
            "model_depth": expected_depth,
        }
        for field, expected in expected_common.items():
            if receipt.get(field) != expected:
                _fail(f"proxy cell {key} {field} mismatch")
        candidate = candidates.get(key[0])
        if not isinstance(candidate, Mapping):
            _fail(f"proxy cell {key} uses an unknown candidate")
        expected_candidate = {
            "production_base_weight_decay": candidate["production_base_weight_decay"],
            "effective_base_weight_decay": candidate["stage_effective_weight_decay"][f"d{expected_depth}"],
            "weight_decay_transfer_rule": proxy["weight_decay_transfer_rule"],
            "schedule": candidate["schedule"],
            "weight_decay_cooldown_policy": candidate["cooldown_weight_decay"],
        }
        for field, expected in expected_candidate.items():
            if receipt.get(field) != expected:
                _fail(f"proxy cell {key} {field} mismatch")
        cells[key] = (receipt, digest)
    if set(cells) != expected_cells:
        _fail(
            "proxy receipt matrix is incomplete; "
            f"missing={sorted(expected_cells - set(cells))}, "
            f"extra={sorted(set(cells) - expected_cells)}"
        )
    return cells


def command_screen_proxy(args: argparse.Namespace) -> None:
    recipe, recipe_sha = load_recipe(args.recipe)
    preflight, preflight_sha = _load_receipt(
        args.preflight_receipt, "d32_family_preflight_receipt"
    )
    proxy = recipe["weight_decay_proxy_ablation"]
    _attention_probe, attention_probe_sha = _verify_attention_probe(
        args.attention_probe,
        recipe=recipe,
        recipe_sha=recipe_sha,
        preflight_sha=preflight_sha,
        code_revision=preflight["code"]["git_commit"],
    )
    candidates = {
        candidate["id"]: candidate
        for candidate in proxy["candidates"]
    }
    screen = proxy["screen_stage"]
    expected_cells = {
        (candidate_id, int(seed))
        for candidate_id in candidates
        for seed in screen["seeds"]
    }
    cells = _proxy_run_cells(
        args.run_receipt,
        expected_depth=12,
        expected_cells=expected_cells,
        candidates=candidates,
        proxy=proxy,
        recipe_sha=recipe_sha,
        preflight=preflight,
        preflight_sha=preflight_sha,
        attention_probe_sha=attention_probe_sha,
    )

    candidate_results: list[dict[str, Any]] = []
    for candidate_id, candidate in candidates.items():
        seed_results = []
        all_bpbs: list[float] = []
        for seed in screen["seeds"]:
            receipt, digest = cells[(candidate_id, int(seed))]
            values = [float(value) for value in receipt["final_validation_bpb"]]
            all_bpbs.extend(values)
            seed_results.append(
                {
                    "seed": int(seed),
                    "run_receipt_sha256": digest,
                    "mean_final_validation_bpb": sum(values) / len(values),
                }
            )
        candidate_results.append(
            {
                "id": candidate_id,
                "schedule": candidate["schedule"],
                "production_base_weight_decay": candidate["production_base_weight_decay"],
                "weight_decay_cooldown_policy": candidate["cooldown_weight_decay"],
                "weight_decay_transfer_rule": proxy["weight_decay_transfer_rule"],
                "stage_effective_weight_decays": [
                    {
                        "stage_id": stage_id,
                        "scaling_parameters": stage_spec["scaling_parameters"],
                        "global_batch_tokens": stage_spec["global_batch_tokens"],
                        "effective_base_weight_decay": candidate[
                            "stage_effective_weight_decay"
                        ][stage_id],
                    }
                    for stage_id, stage_spec in (
                        ("d12", proxy["screen_stage"]),
                        ("d20", proxy["confirmation_stage"]),
                    )
                ],
                "eligible_for_production": candidate["eligible_for_production"],
                "seed_results": seed_results,
                "d12_screen_primary_metric": sum(all_bpbs) / len(all_bpbs),
            }
        )
    selectable = [
        result for result in candidate_results if result["eligible_for_production"]
    ]
    if not selectable:
        _fail("proxy recipe has no production-eligible candidate")
    ranked = sorted(
        selectable,
        key=lambda result: (
            result["d12_screen_primary_metric"],
            result["production_base_weight_decay"],
            result["id"],
        ),
    )
    advance_count = int(screen["advance_top_wsd_recipes"])
    advanced_ids = [result["id"] for result in ranked[:advance_count]]
    screening = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "wsd_proxy_screening",
            "decision": "advance_to_d20",
            "recipe_version": proxy["recipe_version"],
            "study_manifest_sha256": recipe_sha,
            "tokenizer_artifact_sha256": preflight["tokenizer"]["package_manifest_sha256"],
            "source_dataset_manifest_sha256": preflight["corpus"]["dataset_manifest_sha256"],
            "trainer_code_revision": preflight["code"]["git_commit"],
            "gradient_clip_norm": 0.0,
            "attention_probe_sha256": attention_probe_sha,
            "upstream_control_id": "upstream_92d63d4e_control",
            "advanced_candidate_ids": advanced_ids,
            "acceptance_rule": proxy["acceptance_rule"],
            "candidate_results": candidate_results,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(args.output, screening)
    print(json.dumps(screening, sort_keys=True))


def command_accept_proxy(args: argparse.Namespace) -> None:
    recipe, recipe_sha = load_recipe(args.recipe)
    preflight, preflight_sha = _load_receipt(
        args.preflight_receipt, "d32_family_preflight_receipt"
    )
    screening, screening_sha = _load_receipt(
        args.screening_receipt, "wsd_proxy_screening"
    )
    proxy = recipe["weight_decay_proxy_ablation"]
    _attention_probe, attention_probe_sha = _verify_attention_probe(
        args.attention_probe,
        recipe=recipe,
        recipe_sha=recipe_sha,
        preflight_sha=preflight_sha,
        code_revision=preflight["code"]["git_commit"],
    )
    candidates = {candidate["id"]: candidate for candidate in proxy["candidates"]}
    expected_screening = {
        "recipe_version": proxy["recipe_version"],
        "study_manifest_sha256": recipe_sha,
        "tokenizer_artifact_sha256": preflight["tokenizer"]["package_manifest_sha256"],
        "source_dataset_manifest_sha256": preflight["corpus"]["dataset_manifest_sha256"],
        "trainer_code_revision": preflight["code"]["git_commit"],
        "gradient_clip_norm": 0.0,
        "attention_probe_sha256": attention_probe_sha,
        "decision": "advance_to_d20",
    }
    for field, expected in expected_screening.items():
        if screening.get(field) != expected:
            _fail(f"d12 screening receipt {field} mismatch")
    advanced = screening.get("advanced_candidate_ids")
    if (
        not isinstance(advanced, list)
        or len(advanced) != proxy["screen_stage"]["advance_top_wsd_recipes"]
        or len(set(advanced)) != len(advanced)
        or any(
            candidate_id not in candidates
            or not candidates[candidate_id]["eligible_for_production"]
            for candidate_id in advanced
        )
    ):
        _fail("d12 screening advanced-candidate set is invalid")
    control_id = screening.get("upstream_control_id")
    confirmation_ids = [str(control_id), *[str(value) for value in advanced]]
    confirmation = proxy["confirmation_stage"]
    expected_cells = {
        (candidate_id, int(seed))
        for candidate_id in confirmation_ids
        for seed in confirmation["seeds"]
    }
    cells = _proxy_run_cells(
        args.run_receipt,
        expected_depth=20,
        expected_cells=expected_cells,
        candidates=candidates,
        proxy=proxy,
        recipe_sha=recipe_sha,
        preflight=preflight,
        preflight_sha=preflight_sha,
        attention_probe_sha=attention_probe_sha,
    )
    d20_results: dict[str, dict[str, Any]] = {}
    for candidate_id in confirmation_ids:
        all_bpbs: list[float] = []
        seed_results = []
        for seed in confirmation["seeds"]:
            receipt, digest = cells[(candidate_id, int(seed))]
            values = [float(value) for value in receipt["final_validation_bpb"]]
            all_bpbs.extend(values)
            seed_results.append(
                {
                    "seed": int(seed),
                    "run_receipt_sha256": digest,
                    "mean_final_validation_bpb": sum(values) / len(values),
                }
            )
        d20_results[candidate_id] = {
            "seed_results": seed_results,
            "d20_confirmation_primary_metric": sum(all_bpbs) / len(all_bpbs),
        }
    selectable = [
        {
            "id": candidate_id,
            **candidates[candidate_id],
            **d20_results[candidate_id],
        }
        for candidate_id in advanced
    ]
    best_metric = min(
        result["d20_confirmation_primary_metric"] for result in selectable
    )
    upstream_metric = d20_results[str(control_id)]["d20_confirmation_primary_metric"]
    maximum_vs_upstream = float(
        proxy["acceptance_rule"]["maximum_wsd_vs_upstream_control_bpb"]
    )
    if best_metric > upstream_metric + maximum_vs_upstream:
        _fail(
            "no confirmed WSD candidate matches the pinned upstream control within "
            f"{maximum_vs_upstream:.6f} BPB"
        )
    tolerance = float(proxy["acceptance_rule"]["tie_tolerance_bpb"])
    eligible = [
        result
        for result in selectable
        if result["d20_confirmation_primary_metric"] <= best_metric + tolerance
    ]
    selected = min(
        eligible,
        key=lambda result: (result["production_base_weight_decay"], result["id"]),
    )
    candidate_results = []
    screen_by_id = {
        result["id"]: result for result in screening["candidate_results"]
    }
    for candidate_id in candidates:
        result = dict(screen_by_id[candidate_id])
        result["d20_confirmation"] = d20_results.get(candidate_id)
        candidate_results.append(result)
    acceptance = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "wsd_proxy_acceptance",
            "decision": "accepted",
            "recipe_version": proxy["recipe_version"],
            "study_manifest_sha256": recipe_sha,
            "tokenizer_artifact_sha256": preflight["tokenizer"]["package_manifest_sha256"],
            "source_dataset_manifest_sha256": preflight["corpus"]["dataset_manifest_sha256"],
            "trainer_code_revision": preflight["code"]["git_commit"],
            "gradient_clip_norm": 0.0,
            "attention_probe_sha256": attention_probe_sha,
            "weight_decay_transfer_rule": proxy["weight_decay_transfer_rule"],
            "production_scaling_parameters": proxy["production_scaling_parameters"],
            "production_global_batch_tokens": proxy["production_global_batch_tokens"],
            "screening_receipt_sha256": screening_sha,
            "accepted_candidate_id": selected["id"],
            "accepted_base_weight_decay": selected["production_base_weight_decay"],
            "accepted_weight_decay_cooldown_policy": selected[
                "cooldown_weight_decay"
            ],
            "acceptance_rule": proxy["acceptance_rule"],
            "candidate_results": candidate_results,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(args.output, acceptance)
    print(json.dumps(acceptance, sort_keys=True))


def command_record_rank_exit(args: argparse.Namespace) -> None:
    """Seal one trainer child-process exit after the child has terminated."""

    recipe, recipe_sha = load_recipe(args.recipe)
    if args.world_size not in {1, 4, 8, 16}:
        _fail("rank-exit world size must be one of 1, 4, 8, or 16")
    if not 0 <= args.rank < args.world_size:
        _fail("rank-exit global rank is outside the world")
    if not 0 <= args.local_rank < 4:
        _fail("rank-exit local rank must be in 0..3")
    if args.world_size > 1 and args.world_size % 4:
        _fail("multi-GPU rank-exit world size must use complete 4-GPU nodes")
    if not args.node or any(value in args.node for value in ("/", "\n", "\r")):
        _fail("rank-exit node name is empty or unsafe")
    if not args.slurm_job_id or re.fullmatch(r"[0-9]+", args.slurm_job_id) is None:
        _fail("rank-exit requires a numeric Slurm job ID")
    if not args.slurm_step_id or any(
        value in args.slurm_step_id for value in ("/", "\n", "\r")
    ):
        _fail("rank-exit requires a safe Slurm step ID")
    _safe_id(args.run_id, "rank-exit run ID")
    _safe_id(args.phase, "rank-exit phase")
    if args.exit_code not in {0, 75}:
        _fail("rank-exit records only clean exit 0 or collective preemption exit 75")
    receipt_kind = {
        "slurm_srun_direct_python_env_v1": "d32_static_srun_rank_exit",
        "slurm_batch_direct_python_env_v1": "d32_batch_direct_rank_exit",
    }[args.launcher]
    receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": receipt_kind,
            "family_id": recipe["family_id"],
            "recipe_sha256": recipe_sha,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "launcher": args.launcher,
            "run_id": args.run_id,
            "phase": args.phase,
            "slurm_job_id": args.slurm_job_id,
            "slurm_step_id": args.slurm_step_id,
            "node": args.node,
            "rank": args.rank,
            "local_rank": args.local_rank,
            "world_size": args.world_size,
            "child_exit_code": args.exit_code,
            "termination": "clean" if args.exit_code == 0 else "collective_preemption",
            "canonical_sha256": None,
        }
    )
    write_json_atomic(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))


def _aggregate_rank_receipts(
    *,
    recipe: Mapping[str, Any],
    recipe_sha: str,
    receipt_dir: Path,
    expected_kind: str,
    run_id: str,
    phase: str,
    slurm_job_id: str,
    world_size: int,
    nodes: int,
    expected_exit_code: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if world_size != nodes * int(recipe["distributed_gate"]["gpus_per_node"]):
        _fail("launcher receipt world size does not equal nodes times GPUs per node")
    expected_names = {f"rank_{rank:05d}.json" for rank in range(world_size)}
    if not receipt_dir.is_dir() or receipt_dir.is_symlink():
        _fail(f"launcher rank receipt directory is missing or unsafe: {receipt_dir}")
    actual_names = {
        path.name
        for path in receipt_dir.iterdir()
        if path.is_file() and path.suffix == ".json"
    }
    if actual_names != expected_names:
        _fail(
            "launcher rank receipt set is incomplete or contaminated; "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )
    receipts: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for rank in range(world_size):
        path = receipt_dir / f"rank_{rank:05d}.json"
        receipt, digest = _load_receipt(path, expected_kind)
        expected = {
            "family_id": recipe["family_id"],
            "recipe_sha256": recipe_sha,
            "run_id": run_id,
            "phase": phase,
            "slurm_job_id": slurm_job_id,
            "rank": rank,
            "world_size": world_size,
        }
        for field, value in expected.items():
            if receipt.get(field) != value:
                _fail(f"launcher rank {rank} {field} mismatch")
        local_rank = receipt.get("local_rank")
        if isinstance(local_rank, bool) or not isinstance(local_rank, int):
            _fail(f"launcher rank {rank} has an invalid local rank")
        node = receipt.get("node")
        if not isinstance(node, str) or not node:
            _fail(f"launcher rank {rank} has an invalid node")
        if expected_exit_code is not None and receipt.get("child_exit_code") != expected_exit_code:
            _fail(f"launcher rank {rank} did not exit with {expected_exit_code}")
        receipts.append(receipt)
        inventory.append(
            {
                "rank": rank,
                "local_rank": local_rank,
                "node": node,
                "sha256": digest,
                "path": path.name,
            }
        )
    by_node: dict[str, list[int]] = {}
    slurm_step_ids: set[str] = set()
    for receipt in receipts:
        by_node.setdefault(str(receipt["node"]), []).append(int(receipt["local_rank"]))
        slurm_step_ids.add(str(receipt.get("slurm_step_id", "")))
    if len(slurm_step_ids) != 1 or re.fullmatch(r"[0-9]+", next(iter(slurm_step_ids))) is None:
        _fail("launcher rank receipts do not share one numeric Slurm srun step ID")
    if len(by_node) != nodes:
        _fail(f"launcher receipts cover {len(by_node)} nodes, expected {nodes}")
    expected_local_ranks = list(range(int(recipe["distributed_gate"]["gpus_per_node"])))
    for node, local_ranks in by_node.items():
        if sorted(local_ranks) != expected_local_ranks:
            _fail(
                f"launcher node {node} local-rank coverage mismatch: {sorted(local_ranks)}"
            )
    node_inventory = [
        {"node": node, "local_ranks": sorted(local_ranks)}
        for node, local_ranks in sorted(by_node.items())
    ]
    return inventory, node_inventory


def command_seal_static_launch(args: argparse.Namespace) -> None:
    recipe, recipe_sha = load_recipe(args.recipe)
    inventory, node_inventory = _aggregate_rank_receipts(
        recipe=recipe,
        recipe_sha=recipe_sha,
        receipt_dir=args.receipt_dir,
        expected_kind="d32_static_srun_rank_exit",
        run_id=args.run_id,
        phase=args.phase,
        slurm_job_id=args.slurm_job_id,
        world_size=args.world_size,
        nodes=args.nodes,
        expected_exit_code=args.expected_exit_code,
    )
    if args.srun_exit_code != 0:
        _fail("srun must exit zero after every wrapper records its direct child result")
    clean = args.expected_exit_code == 0
    receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "d32_static_srun_launch_receipt",
            "family_id": recipe["family_id"],
            "recipe_sha256": recipe_sha,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "launcher": "slurm_srun_direct_python_env_v1",
            "elastic_rendezvous": False,
            "run_id": args.run_id,
            "phase": args.phase,
            "slurm_job_id": args.slurm_job_id,
            "world_size": args.world_size,
            "nodes": args.nodes,
            "gpus_per_node": recipe["distributed_gate"]["gpus_per_node"],
            "srun_exit_code": args.srun_exit_code,
            "rank_exit_code": args.expected_exit_code,
            "termination": "clean" if clean else "collective_preemption",
            "all_rank_exit_codes_zero": clean,
            "rank_receipts": inventory,
            "node_inventory": node_inventory,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))


def command_seal_static_probe(args: argparse.Namespace) -> None:
    recipe, recipe_sha = load_recipe(args.recipe)
    inventory, node_inventory = _aggregate_rank_receipts(
        recipe=recipe,
        recipe_sha=recipe_sha,
        receipt_dir=args.receipt_dir,
        expected_kind="d32_static_srun_probe_rank",
        run_id=args.run_id,
        phase=args.phase,
        slurm_job_id=args.slurm_job_id,
        world_size=args.world_size,
        nodes=args.nodes,
        expected_exit_code=0,
    )
    if args.srun_exit_code != 0:
        _fail("a static distributed probe cannot be sealed unless srun exited zero")
    for item in inventory:
        rank_receipt, _digest = _load_receipt(
            args.receipt_dir / str(item["path"]), "d32_static_srun_probe_rank"
        )
        if rank_receipt.get("visible_device_count") != 1:
            _fail("static probe rank did not see exactly its one assigned GPU")
        if rank_receipt.get("device_index") != 0 or rank_receipt.get("torch_local_rank") != 0:
            _fail("static probe rank did not use remapped local CUDA device zero")
        collective = _mapping(rank_receipt.get("collective"), "static probe collective")
        expected_sum = args.world_size * (args.world_size - 1) / 2
        expected_collective = {
            "backend": "nccl",
            "all_reduce_expected": expected_sum,
            "all_reduce_observed": expected_sum,
            "all_gather_world_size": args.world_size,
            "final_barrier_completed": True,
            "process_group_destroyed_before_receipt": True,
        }
        if collective != expected_collective:
            _fail("static probe collective or clean-shutdown evidence is incomplete")
    receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "d32_static_srun_probe_receipt",
            "family_id": recipe["family_id"],
            "recipe_sha256": recipe_sha,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "launcher": "slurm_srun_direct_python_env_v1",
            "elastic_rendezvous": False,
            "run_id": args.run_id,
            "phase": args.phase,
            "slurm_job_id": args.slurm_job_id,
            "world_size": args.world_size,
            "nodes": args.nodes,
            "gpus_per_node": recipe["distributed_gate"]["gpus_per_node"],
            "srun_exit_code": args.srun_exit_code,
            "all_rank_exit_codes_zero": True,
            "nccl_collective_passed": True,
            "all_process_groups_destroyed_before_rank_receipts": True,
            "rank_receipts": inventory,
            "node_inventory": node_inventory,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))


def _verify_static_probe_receipt(
    path: Path,
    *,
    recipe: Mapping[str, Any],
    recipe_sha: str,
    run_id: str,
    slurm_job_id: str,
    world_size: int,
    nodes: int,
) -> tuple[dict[str, Any], str]:
    receipt, digest = _load_receipt(path, "d32_static_srun_probe_receipt")
    expected = {
        "family_id": recipe["family_id"],
        "recipe_sha256": recipe_sha,
        "launcher": "slurm_srun_direct_python_env_v1",
        "elastic_rendezvous": False,
        "run_id": run_id,
        "phase": "static_nccl_probe",
        "slurm_job_id": slurm_job_id,
        "world_size": world_size,
        "nodes": nodes,
        "gpus_per_node": recipe["distributed_gate"]["gpus_per_node"],
        "srun_exit_code": 0,
        "all_rank_exit_codes_zero": True,
        "nccl_collective_passed": True,
        "all_process_groups_destroyed_before_rank_receipts": True,
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            _fail(f"static NCCL probe receipt {field} mismatch")
    return receipt, digest


def command_finalize_static_probe(args: argparse.Namespace) -> None:
    recipe, recipe_sha = load_recipe(args.recipe)
    preflight, preflight_sha = _load_receipt(
        args.preflight_receipt, "d32_family_preflight_receipt"
    )
    if preflight["recipe"]["canonical_sha256"] != recipe_sha:
        _fail("static launcher probe preflight recipe mismatch")
    world_size = args.nodes * int(recipe["distributed_gate"]["gpus_per_node"])
    run_id = f"{recipe['family_id']}_launcher_probe_ws{world_size}"
    _probe, probe_sha = _verify_static_probe_receipt(
        args.raw_probe_receipt,
        recipe=recipe,
        recipe_sha=recipe_sha,
        run_id=run_id,
        slurm_job_id=args.slurm_job_id,
        world_size=world_size,
        nodes=args.nodes,
    )
    slurm_completion = _live_slurm_completed_job(
        REPO_ROOT, job_id=args.slurm_job_id, expected_nodes=args.nodes
    )
    receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "d32_static_launcher_gate",
            "family_id": recipe["family_id"],
            "recipe_sha256": recipe_sha,
            "preflight_receipt_sha256": preflight_sha,
            "code_revision": preflight["code"]["git_commit"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "launcher": "slurm_srun_direct_python_env_v1",
            "elastic_rendezvous": False,
            "probe_world_size": world_size,
            "probe_nodes": args.nodes,
            "raw_probe_receipt_sha256": probe_sha,
            "slurm_completion": slurm_completion,
            "passed": True,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))


def _verify_static_launcher_gate(
    path: Path,
    *,
    recipe: Mapping[str, Any],
    recipe_sha: str,
    preflight: Mapping[str, Any],
    preflight_sha: str,
) -> tuple[dict[str, Any], str]:
    gate, digest = _load_receipt(path, "d32_static_launcher_gate")
    expected = {
        "family_id": recipe["family_id"],
        "recipe_sha256": recipe_sha,
        "preflight_receipt_sha256": preflight_sha,
        "code_revision": preflight["code"]["git_commit"],
        "launcher": "slurm_srun_direct_python_env_v1",
        "elastic_rendezvous": False,
        "probe_world_size": 4,
        "probe_nodes": 1,
        "passed": True,
    }
    for field, value in expected.items():
        if gate.get(field) != value:
            _fail(f"static launcher gate {field} mismatch")
    completion = _mapping(gate.get("slurm_completion"), "static launcher Slurm completion")
    if completion.get("state") != "COMPLETED" or completion.get("exit_code") != "0:0":
        _fail("static launcher gate does not prove a clean Slurm completion")
    return gate, digest


def command_verify_static_launcher_gate(args: argparse.Namespace) -> None:
    recipe, recipe_sha = load_recipe(args.recipe)
    preflight, preflight_sha = _load_receipt(
        args.preflight_receipt, "d32_family_preflight_receipt"
    )
    _gate, digest = _verify_static_launcher_gate(
        args.gate,
        recipe=recipe,
        recipe_sha=recipe_sha,
        preflight=preflight,
        preflight_sha=preflight_sha,
    )
    print(json.dumps({"passed": True, "static_launcher_gate_sha256": digest}, sort_keys=True))


def _signal_smoke_identity(recipe: Mapping[str, Any]) -> tuple[str, str, int]:
    world_size = int(recipe["distributed_gate"]["signal_resume_probe_world_size"])
    return (
        f"{recipe['family_id']}_signal_smoke_ws{world_size}",
        f"{recipe['family_id']}_signal_smoke_ws{world_size}",
        world_size,
    )


def command_signal_smoke_env(args: argparse.Namespace) -> None:
    recipe, _recipe_sha = load_recipe(args.recipe)
    run_id, model_tag, _world_size = _signal_smoke_identity(recipe)
    final_step = int(recipe["distributed_gate"]["signal_resume_probe_updates"])
    base_dir = args.base_dir.expanduser().resolve()
    checkpoint_root = base_dir / "base_checkpoints" / model_tag
    candidates = []
    if checkpoint_root.is_dir():
        for child in checkpoint_root.glob("strict_*"):
            match = re.fullmatch(r"strict_(\d{6,})", child.name)
            if match is None or not (child / "completion.json").is_file():
                continue
            step = int(match.group(1))
            if 0 < step < final_step:
                checkpoint, _sha = _checkpoint_manifest(base_dir, model_tag, step)
                if checkpoint["identity"].get("run_id") != run_id:
                    _fail("signal-smoke checkpoint run ID mismatch")
                candidates.append(step)
    resume_step = max(candidates) if candidates else None
    print(f"RUN_ID={shlex.quote(run_id)}")
    print(f"MODEL_TAG={shlex.quote(model_tag)}")
    print(f"RESUME_FROM_STEP={shlex.quote('' if resume_step is None else str(resume_step))}")


def command_record_signal_request(args: argparse.Namespace) -> None:
    recipe, recipe_sha = load_recipe(args.recipe)
    run_id, _model_tag, world_size = _signal_smoke_identity(recipe)
    events, state = read_training_log(args.curve_log)
    threshold = int(recipe["distributed_gate"]["signal_resume_probe_after_completed_update"])
    completed = max(
        [
            int(event["updates_completed"])
            for event in events
            if event.get("event_type") == "train_update"
        ],
        default=-1,
    )
    if completed < threshold:
        _fail("signal request was attempted before the declared completed-update boundary")
    if args.slurm_restart_count != 0:
        _fail("bounded signal request must occur only on the initial Slurm attempt")
    receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "d32_signal_request_receipt",
            "family_id": recipe["family_id"],
            "recipe_sha256": recipe_sha,
            "run_id": run_id,
            "slurm_job_id": args.slurm_job_id,
            "slurm_step_id": args.slurm_step_id,
            "slurm_restart_count": args.slurm_restart_count,
            "signal": "SIGUSR1",
            "delivery": "scancel_signal_to_exact_srun_step",
            "world_size": world_size,
            "observed_updates_completed": completed,
            "minimum_updates_completed": threshold,
            "curve_log_sha256": file_sha256(args.curve_log),
            "curve_log_last_event_sha256": state.last_event_sha256,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "canonical_sha256": None,
        }
    )
    write_json_atomic(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))


def command_seal_signal_preemption(args: argparse.Namespace) -> None:
    recipe, recipe_sha = load_recipe(args.recipe)
    run_id, model_tag, world_size = _signal_smoke_identity(recipe)
    nodes = world_size // int(recipe["distributed_gate"]["gpus_per_node"])
    signal_request, signal_request_sha = _load_receipt(
        args.signal_request, "d32_signal_request_receipt"
    )
    for field, value in {
        "family_id": recipe["family_id"],
        "recipe_sha256": recipe_sha,
        "run_id": run_id,
        "slurm_job_id": args.slurm_job_id,
        "slurm_step_id": f"{args.slurm_job_id}.0",
        "slurm_restart_count": 0,
        "signal": "SIGUSR1",
        "delivery": "scancel_signal_to_exact_srun_step",
        "world_size": world_size,
    }.items():
        if signal_request.get(field) != value:
            _fail(f"signal request {field} mismatch")
    threshold = int(recipe["distributed_gate"]["signal_resume_probe_after_completed_update"])
    if signal_request.get("minimum_updates_completed") != threshold or int(
        signal_request.get("observed_updates_completed", -1)
    ) < threshold:
        _fail("signal request does not prove the bounded completed-update threshold")
    _launch, launch_sha = _verify_static_launch_receipt(
        args.launch_receipt,
        recipe=recipe,
        recipe_sha=recipe_sha,
        run_id=run_id,
        phase="signal_smoke_preemption",
        slurm_job_id=args.slurm_job_id,
        world_size=world_size,
        nodes=nodes,
        expected_child_exit_code=75,
    )
    final_step = int(recipe["distributed_gate"]["signal_resume_probe_updates"])
    base_dir = args.base_dir.expanduser().resolve()
    checkpoint_root = base_dir / "base_checkpoints" / model_tag
    candidates = []
    if checkpoint_root.is_dir():
        for child in checkpoint_root.glob("strict_*"):
            match = re.fullmatch(r"strict_(\d{6,})", child.name)
            if match and (child / "completion.json").is_file():
                step = int(match.group(1))
                if 0 < step < final_step:
                    candidates.append(step)
    if not candidates:
        _fail("signal smoke exited 75 without a new complete checkpoint")
    checkpoint_step = max(candidates)
    checkpoint, checkpoint_sha = _checkpoint_manifest(base_dir, model_tag, checkpoint_step)
    if checkpoint.get("expected_world_size") != world_size:
        _fail("signal-smoke preemption checkpoint world size mismatch")
    identity = _mapping(checkpoint.get("identity"), "signal-smoke checkpoint identity")
    if identity.get("run_id") != run_id or identity.get("study_manifest_sha256") != recipe_sha:
        _fail("signal-smoke preemption checkpoint identity mismatch")
    meta = _load_object(
        checkpoint_root / f"strict_{checkpoint_step:06d}" / "meta.json",
        "signal-smoke preemption metadata",
    )
    preemption = _mapping(meta.get("preemption"), "signal-smoke preemption metadata")
    if preemption.get("signal") != "SIGUSR1" or preemption.get("exit_code") != 75:
        _fail("signal-smoke checkpoint does not prove SIGUSR1/exit-75 handling")
    receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "d32_signal_preemption_receipt",
            "family_id": recipe["family_id"],
            "recipe_sha256": recipe_sha,
            "run_id": run_id,
            "slurm_job_id": args.slurm_job_id,
            "world_size": world_size,
            "signal_request_sha256": signal_request_sha,
            "launch_receipt_sha256": launch_sha,
            "checkpoint_step": checkpoint_step,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_preemption": dict(preemption),
            "requeue_authorized": True,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "canonical_sha256": None,
        }
    )
    write_json_atomic(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))


def command_finalize_signal_smoke(args: argparse.Namespace) -> None:
    recipe, recipe_sha = load_recipe(args.recipe)
    preflight, preflight_sha = _load_receipt(
        args.preflight_receipt, "d32_family_preflight_receipt"
    )
    if preflight["recipe"]["canonical_sha256"] != recipe_sha:
        _fail("signal-smoke preflight recipe mismatch")
    _static_gate, static_gate_sha = _verify_static_launcher_gate(
        args.static_launcher_gate,
        recipe=recipe,
        recipe_sha=recipe_sha,
        preflight=preflight,
        preflight_sha=preflight_sha,
    )
    _probe, probe_sha = _verify_attention_probe(
        args.attention_probe,
        recipe=recipe,
        recipe_sha=recipe_sha,
        preflight_sha=preflight_sha,
        code_revision=preflight["code"]["git_commit"],
    )
    _approval, approval_sha = _verify_proxy_acceptance(
        args.wd_proxy_approval,
        recipe=recipe,
        recipe_sha=recipe_sha,
        preflight=preflight,
        attention_probe_sha256=probe_sha,
    )
    run_id, model_tag, world_size = _signal_smoke_identity(recipe)
    nodes = world_size // int(recipe["distributed_gate"]["gpus_per_node"])
    slurm_completion = _live_slurm_completed_job(
        REPO_ROOT, job_id=args.slurm_job_id, expected_nodes=nodes
    )
    preemption, preemption_sha = _load_receipt(
        args.preemption_receipt, "d32_signal_preemption_receipt"
    )
    for field, value in {
        "family_id": recipe["family_id"],
        "recipe_sha256": recipe_sha,
        "run_id": run_id,
        "slurm_job_id": args.slurm_job_id,
        "world_size": world_size,
        "requeue_authorized": True,
    }.items():
        if preemption.get(field) != value:
            _fail(f"signal preemption receipt {field} mismatch")
    _launch, final_launch_sha = _verify_static_launch_receipt(
        args.final_launch_receipt,
        recipe=recipe,
        recipe_sha=recipe_sha,
        run_id=run_id,
        phase="signal_smoke_resume",
        slurm_job_id=args.slurm_job_id,
        world_size=world_size,
        nodes=nodes,
    )
    final_step = int(recipe["distributed_gate"]["signal_resume_probe_updates"])
    base_dir = args.base_dir.expanduser().resolve()
    checkpoint, checkpoint_sha = _checkpoint_manifest(base_dir, model_tag, final_step)
    if checkpoint.get("expected_world_size") != world_size:
        _fail("signal-smoke final checkpoint world size mismatch")
    identity = _mapping(checkpoint.get("identity"), "signal-smoke final identity")
    plan = preflight["corpus"]["training_exposure_plans"]["signal_smoke_ws4_seed42"]
    if (
        identity.get("run_id") != run_id
        or identity.get("study_manifest_sha256") != recipe_sha
        or identity.get("exposure_plan_sha256") != plan["sha256"]
    ):
        _fail("signal-smoke final checkpoint identity mismatch")
    protocol = _mapping(identity.get("protocol"), "signal-smoke final protocol")
    optimizer = _verify_frozen_protocol(
        protocol,
        recipe=recipe,
        preflight=preflight,
        attention_probe=_probe,
        attention_probe_sha256=probe_sha,
        label="signal-smoke final protocol",
        run_kind="signal_smoke",
        recipe_scope=f"signal_smoke_ws{world_size}",
        model_tag=model_tag,
        exposure_plan_sha256=str(plan["sha256"]),
        depth=32,
        model_dim=2048,
        world_size=world_size,
        device_batch_size=recipe["training"]["device_batch_sequences"],
        total_batch_size=recipe["training"]["global_batch_tokens"],
        num_iterations=final_step,
        eval_every_updates=recipe["training"]["evaluation"]["eval_every_updates"],
        seed=recipe["training"]["seed"],
    )
    if optimizer.get("muon_base_weight_decay") != approval[
        "accepted_base_weight_decay"
    ]:
        _fail("signal-smoke base weight decay differs from proxy approval")
    schedule = _mapping(protocol.get("schedule"), "signal-smoke schedule")
    if (
        schedule.get("name") != "wsd"
        or schedule.get("cooldown_start_step") is not None
        or schedule.get("proxy_approval_sha256") != approval_sha
        or schedule.get("weight_decay_cooldown_policy")
        != approval["accepted_weight_decay_cooldown_policy"]
    ):
        _fail("signal-smoke WSD schedule differs from the approved stable phase")
    meta = _load_object(
        base_dir / "base_checkpoints" / model_tag / f"strict_{final_step:06d}" / "meta.json",
        "signal-smoke final metadata",
    )
    resumed_from = _mapping(
        meta.get("user_config"), "signal-smoke final user_config"
    ).get("resume_from_step")
    if resumed_from != preemption.get("checkpoint_step"):
        _fail("signal-smoke final checkpoint does not prove resume from preemption")
    receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "d32_signal_resume_gate",
            "family_id": recipe["family_id"],
            "recipe_sha256": recipe_sha,
            "preflight_receipt_sha256": preflight_sha,
            "static_launcher_gate_sha256": static_gate_sha,
            "attention_probe_sha256": probe_sha,
            "wsd_proxy_approval_sha256": approval_sha,
            "run_id": run_id,
            "world_size": world_size,
            "slurm_job_id": args.slurm_job_id,
            "slurm_completion": slurm_completion,
            "preemption_receipt_sha256": preemption_sha,
            "resumed_from_step": resumed_from,
            "final_step": final_step,
            "final_checkpoint_sha256": checkpoint_sha,
            "final_launch_receipt_sha256": final_launch_sha,
            "passed": True,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "canonical_sha256": None,
        }
    )
    write_json_atomic(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))


def _verify_signal_resume_gate(
    path: Path,
    *,
    recipe: Mapping[str, Any],
    recipe_sha: str,
    preflight: Mapping[str, Any],
    preflight_sha: str,
) -> tuple[dict[str, Any], str]:
    gate, digest = _load_receipt(path, "d32_signal_resume_gate")
    expected = {
        "family_id": recipe["family_id"],
        "recipe_sha256": recipe_sha,
        "preflight_receipt_sha256": preflight_sha,
        "world_size": recipe["distributed_gate"]["signal_resume_probe_world_size"],
        "final_step": recipe["distributed_gate"]["signal_resume_probe_updates"],
        "passed": True,
    }
    for field, value in expected.items():
        if gate.get(field) != value:
            _fail(f"signal/resume gate {field} mismatch")
    completion = _mapping(gate.get("slurm_completion"), "signal/resume Slurm completion")
    if completion.get("state") != "COMPLETED" or completion.get("exit_code") != "0:0":
        _fail("signal/resume gate lacks a clean final Slurm completion")
    return gate, digest


def command_verify_signal_resume_gate(args: argparse.Namespace) -> None:
    recipe, recipe_sha = load_recipe(args.recipe)
    preflight, preflight_sha = _load_receipt(
        args.preflight_receipt, "d32_family_preflight_receipt"
    )
    _gate, digest = _verify_signal_resume_gate(
        args.gate,
        recipe=recipe,
        recipe_sha=recipe_sha,
        preflight=preflight,
        preflight_sha=preflight_sha,
    )
    print(json.dumps({"passed": True, "signal_resume_gate_sha256": digest}, sort_keys=True))


def _verify_static_launch_receipt(
    path: Path,
    *,
    recipe: Mapping[str, Any],
    recipe_sha: str,
    run_id: str,
    phase: str,
    slurm_job_id: str,
    world_size: int,
    nodes: int,
    expected_child_exit_code: int = 0,
) -> tuple[dict[str, Any], str]:
    receipt, digest = _load_receipt(path, "d32_static_srun_launch_receipt")
    expected = {
        "family_id": recipe["family_id"],
        "recipe_sha256": recipe_sha,
        "launcher": "slurm_srun_direct_python_env_v1",
        "elastic_rendezvous": False,
        "run_id": run_id,
        "phase": phase,
        "slurm_job_id": slurm_job_id,
        "world_size": world_size,
        "nodes": nodes,
        "gpus_per_node": recipe["distributed_gate"]["gpus_per_node"],
        "srun_exit_code": 0,
        "rank_exit_code": expected_child_exit_code,
        "termination": (
            "clean" if expected_child_exit_code == 0 else "collective_preemption"
        ),
        "all_rank_exit_codes_zero": expected_child_exit_code == 0,
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            _fail(f"static srun launch receipt {field} mismatch")
    ranks = _sequence(receipt.get("rank_receipts"), "static launch rank receipts")
    if [entry.get("rank") for entry in ranks if isinstance(entry, Mapping)] != list(
        range(world_size)
    ):
        _fail("static launch receipt does not cover every rank exactly once")
    nodes_seen = _sequence(receipt.get("node_inventory"), "static launch node inventory")
    if len(nodes_seen) != nodes:
        _fail("static launch receipt node count mismatch")
    return receipt, digest


def _live_slurm_completed_job(
    repo_root: Path, *, job_id: str, expected_nodes: int
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9]+", job_id) is None:
        _fail("Slurm completion verification requires a numeric job ID")
    command = [
        "sacct",
        "-n",
        "-X",
        "-P",
        "-j",
        job_id,
        "-o",
        "JobIDRaw,State,ExitCode,NNodes",
    ]
    try:
        output = subprocess.check_output(
            command, cwd=repo_root, text=True, stderr=subprocess.STDOUT
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        details = getattr(exc, "output", "")
        raise FamilyWorkflowError(
            "cannot verify the completed Slurm allocation with sacct"
            + (f": {str(details).strip()}" if details else "")
        ) from exc
    matching = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) < 4:
            _fail(f"unexpected sacct completion row: {line!r}")
        if fields[0].strip() == job_id:
            matching.append(tuple(field.strip() for field in fields[:4]))
    if len(matching) != 1:
        _fail(f"sacct returned {len(matching)} exact allocation rows for job {job_id}")
    _row_job, state, exit_code, nodes_text = matching[0]
    if state != "COMPLETED" or exit_code != "0:0":
        _fail(
            f"Slurm job {job_id} was not a clean success: state={state!r}, exit={exit_code!r}"
        )
    try:
        nodes = int(nodes_text)
    except ValueError as exc:
        raise FamilyWorkflowError(f"cannot parse Slurm NNodes value: {nodes_text!r}") from exc
    if nodes != expected_nodes:
        _fail(f"Slurm job {job_id} used {nodes} nodes, expected {expected_nodes}")
    return {
        "job_id": job_id,
        "state": state,
        "exit_code": exit_code,
        "nodes": nodes,
        "sacct_output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
    }


def command_seal_smoke(args: argparse.Namespace) -> None:
    recipe, recipe_sha = load_recipe(args.recipe)
    preflight, preflight_sha = _load_receipt(
        args.preflight_receipt, "d32_family_preflight_receipt"
    )
    if preflight["recipe"]["canonical_sha256"] != recipe_sha:
        _fail("preflight receipt was created for a different family recipe")
    _launcher_gate, launcher_gate_sha = _verify_static_launcher_gate(
        args.static_launcher_gate,
        recipe=recipe,
        recipe_sha=recipe_sha,
        preflight=preflight,
        preflight_sha=preflight_sha,
    )
    _signal_gate, signal_gate_sha = _verify_signal_resume_gate(
        args.signal_resume_gate,
        recipe=recipe,
        recipe_sha=recipe_sha,
        preflight=preflight,
        preflight_sha=preflight_sha,
    )
    _attention_probe, attention_probe_sha = _verify_attention_probe(
        args.attention_probe,
        recipe=recipe,
        recipe_sha=recipe_sha,
        preflight_sha=preflight_sha,
        code_revision=preflight["code"]["git_commit"],
    )
    proxy_approval, proxy_approval_sha = _verify_proxy_acceptance(
        args.wd_proxy_approval,
        recipe=recipe,
        recipe_sha=recipe_sha,
        preflight=preflight,
        attention_probe_sha256=attention_probe_sha,
    )
    expected_nodes = recipe["distributed_gate"]["smoke_node_order"]
    if args.nodes not in expected_nodes:
        _fail(f"smoke nodes must be one of {expected_nodes}")
    gpus = args.nodes * int(recipe["distributed_gate"]["gpus_per_node"])
    packing_capacity_sha, packing_capacity_world = _preflight_capacity_world(
        preflight, gpus
    )
    smoke_horizon_positions = (
        int(recipe["distributed_gate"]["smoke_updates"])
        * int(recipe["training"]["global_batch_tokens"])
    )
    if capacity_authorized_positions(packing_capacity_world) < smoke_horizon_positions:
        _fail(f"ws{gpus} packing capacity is below the smoke horizon")
    run_id = f"{recipe['family_id']}_smoke_ws{gpus}"
    slurm_completion = _live_slurm_completed_job(
        REPO_ROOT, job_id=args.slurm_job_id, expected_nodes=args.nodes
    )
    _static_probe, static_probe_sha = _verify_static_probe_receipt(
        args.static_probe_receipt,
        recipe=recipe,
        recipe_sha=recipe_sha,
        run_id=run_id,
        slurm_job_id=args.slurm_job_id,
        world_size=gpus,
        nodes=args.nodes,
    )
    expected_launch_phases = ("smoke_initial_50", "smoke_resume_100")
    if len(args.launch_receipt) != len(expected_launch_phases):
        _fail("smoke finalization requires exactly the initial and resumed launch receipts")
    launch_inventory = []
    for path, phase in zip(args.launch_receipt, expected_launch_phases, strict=True):
        _launch, launch_sha = _verify_static_launch_receipt(
            path,
            recipe=recipe,
            recipe_sha=recipe_sha,
            run_id=run_id,
            phase=phase,
            slurm_job_id=args.slurm_job_id,
            world_size=gpus,
            nodes=args.nodes,
        )
        launch_inventory.append({"phase": phase, "sha256": launch_sha, "path": str(path)})

    from nanochat.strict_checkpoint import inspect_strict_checkpoint

    forced_resume_step = int(recipe["distributed_gate"]["forced_resume_step"])
    final_step = int(recipe["distributed_gate"]["smoke_updates"])
    try:
        resume_checkpoint = inspect_strict_checkpoint(args.checkpoint_root, forced_resume_step)
        final_checkpoint = inspect_strict_checkpoint(args.checkpoint_root, final_step)
    except Exception as exc:
        raise FamilyWorkflowError(
            "smoke must contain verified strict checkpoints at the forced-resume and final boundaries: "
            f"{exc}"
        ) from exc
    resume_checkpoint_sha = verify_manifest_hash(resume_checkpoint)
    final_checkpoint_sha = verify_manifest_hash(final_checkpoint)
    resume_step_dir = args.checkpoint_root / f"strict_{forced_resume_step:06d}"
    final_step_dir = args.checkpoint_root / f"strict_{final_step:06d}"
    storage_observations = {
        "forced_resume": _checkpoint_storage_observation(
            resume_checkpoint, resume_step_dir
        ),
        "final": _checkpoint_storage_observation(final_checkpoint, final_step_dir),
    }
    final_meta = _load_object(
        args.checkpoint_root / f"strict_{final_step:06d}" / "meta.json",
        "final smoke checkpoint metadata",
    )
    final_user_config = _mapping(final_meta.get("user_config"), "final smoke user_config")
    if final_user_config.get("resume_from_step") != forced_resume_step:
        _fail(
            "final smoke checkpoint does not prove the required forced resume from "
            f"step {forced_resume_step}"
        )
    final_identity = _mapping(final_checkpoint.get("identity"), "final smoke identity")
    smoke_plan_key = f"smoke_ws{gpus}"
    expected_identity_fields = {
        "study_id": recipe["family_id"],
        "run_id": run_id,
        "study_manifest_sha256": recipe_sha,
        "tokenizer_artifact_sha256": preflight["tokenizer"]["package_manifest_sha256"],
        "exposure_plan_sha256": preflight["corpus"]["training_exposure_plans"][
            smoke_plan_key
        ]["sha256"],
    }
    for field, expected in expected_identity_fields.items():
        if final_identity.get(field) != expected:
            _fail(f"final smoke identity {field} mismatch")
    final_protocol = _mapping(final_identity.get("protocol"), "final smoke protocol")
    _verify_frozen_protocol(
        final_protocol,
        recipe=recipe,
        preflight=preflight,
        attention_probe=_attention_probe,
        attention_probe_sha256=attention_probe_sha,
        label="final smoke protocol",
        run_kind="smoke",
        recipe_scope=f"smoke_ws{gpus}",
        model_tag=run_id,
        exposure_plan_sha256=str(
            preflight["corpus"]["training_exposure_plans"][smoke_plan_key]["sha256"]
        ),
        depth=32,
        model_dim=2048,
        world_size=gpus,
        device_batch_size=recipe["training"]["device_batch_sequences"],
        total_batch_size=recipe["training"]["global_batch_tokens"],
        num_iterations=final_step,
        eval_every_updates=recipe["training"]["evaluation"]["eval_every_updates"],
        seed=recipe["training"]["seed"],
    )
    smoke_optimizer = _mapping(final_protocol.get("optimizer"), "final smoke optimizer")
    if smoke_optimizer.get("muon_base_weight_decay") != proxy_approval[
        "accepted_base_weight_decay"
    ]:
        _fail("smoke checkpoint base weight decay differs from proxy approval")
    smoke_schedule = _mapping(final_protocol.get("schedule"), "final smoke schedule")
    expected_smoke_schedule = {
        "name": "wsd",
        "recipe_version": recipe["weight_decay_proxy_ablation"]["recipe_version"],
        "cooldown_start_step": None,
        "weight_decay_cooldown_policy": proxy_approval[
            "accepted_weight_decay_cooldown_policy"
        ],
        "proxy_approval_sha256": proxy_approval_sha,
    }
    for field, expected in expected_smoke_schedule.items():
        if smoke_schedule.get(field) != expected:
            _fail(f"final smoke schedule {field} differs from production contract")

    events, state = read_training_log(args.curve_log)
    first = int(recipe["distributed_gate"]["benchmark_first_update"])
    last = int(recipe["distributed_gate"]["benchmark_last_update"])
    selected = [
        event
        for event in events
        if event.get("event_type") == "train_update"
        and first <= int(event["updates_completed"]) <= last
    ]
    updates = [int(event["updates_completed"]) for event in selected]
    if updates != list(range(first, last + 1)):
        _fail(
            f"smoke curve must contain contiguous measured updates {first}..{last}; found {updates}"
        )
    durations = [float(event["metrics"]["train/duration_seconds"]) for event in selected]
    positions = [int(event["metrics"]["train/scheduled_positions"]) for event in selected]
    loader_seconds = [
        float(event["metrics"]["train/loader_seconds"]) for event in selected
    ]
    loader_throughputs = [
        float(event["metrics"]["train/loader_scheduled_positions_per_second"])
        for event in selected
    ]
    loader_fractions = [
        float(event["metrics"]["train/loader_fraction_of_update"])
        for event in selected
    ]
    expected_batch = int(recipe["training"]["global_batch_tokens"])
    if any(value != expected_batch for value in positions):
        _fail("smoke did not use the production global batch on every measured update")
    if any(not math.isfinite(value) or value <= 0 for value in durations):
        _fail("smoke curve contains a non-positive or non-finite update duration")
    if any(
        not math.isfinite(seconds)
        or seconds < 0
        or not math.isfinite(rate)
        or rate <= 0
        or not math.isfinite(fraction)
        or fraction < 0
        or fraction > 1
        for seconds, rate, fraction in zip(
            loader_seconds, loader_throughputs, loader_fractions, strict=True
        )
    ):
        _fail("smoke curve contains invalid loader performance measurements")
    total_seconds = sum(durations)
    total_positions = sum(positions)
    throughput = total_positions / total_seconds
    aggregate_loader_fraction = sum(loader_seconds) / total_seconds
    sorted_loader_fractions = sorted(loader_fractions)
    p95_loader_fraction = sorted_loader_fractions[
        max(0, math.ceil(0.95 * len(sorted_loader_fractions)) - 1)
    ]
    if aggregate_loader_fraction > float(
        recipe["distributed_gate"]["maximum_aggregate_loader_fraction"]
    ):
        _fail("smoke loader occupies too much aggregate update time")
    if p95_loader_fraction > float(
        recipe["distributed_gate"]["maximum_p95_loader_fraction"]
    ):
        _fail("smoke loader p95 fraction exceeds the production gate")
    identity = _production_identity(
        recipe_sha,
        preflight,
        attention_probe_sha256=attention_probe_sha,
        proxy_approval_sha256=proxy_approval_sha,
        accepted_base_weight_decay=float(proxy_approval["accepted_base_weight_decay"]),
        accepted_weight_decay_cooldown_policy=str(
            proxy_approval["accepted_weight_decay_cooldown_policy"]
        ),
    )
    identity_sha = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "d32_distributed_smoke_receipt",
            "family_id": recipe["family_id"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "slurm_job_id": args.slurm_job_id,
            "slurm_completion": slurm_completion,
            "static_nccl_probe_sha256": static_probe_sha,
            "static_launcher_gate_sha256": launcher_gate_sha,
            "signal_resume_gate_sha256": signal_gate_sha,
            "packing_capacity_receipt_sha256": packing_capacity_sha,
            "packing_capacity_world_size": gpus,
            "packing_capacity_safe_global_scheduled_positions": int(
                capacity_authorized_positions(packing_capacity_world)
            ),
            "static_srun_launches": launch_inventory,
            "nodes": args.nodes,
            "gpus_per_node": recipe["distributed_gate"]["gpus_per_node"],
            "world_size": gpus,
            "measured_first_update": first,
            "measured_last_update": last,
            "measured_updates": len(selected),
            "forced_resume": {
                "step": forced_resume_step,
                "checkpoint_sha256": resume_checkpoint_sha,
                "final_step": final_step,
                "final_checkpoint_sha256": final_checkpoint_sha,
                "verified_from_final_metadata": True,
            },
            "scheduled_positions": total_positions,
            "duration_seconds": total_seconds,
            "scheduled_positions_per_second": throughput,
            "loader_performance": {
                "aggregate_loader_seconds": sum(loader_seconds),
                "aggregate_loader_fraction": aggregate_loader_fraction,
                "p95_loader_fraction": p95_loader_fraction,
                "minimum_scheduled_positions_per_second": min(loader_throughputs),
                "median_scheduled_positions_per_second": sorted(loader_throughputs)[
                    len(loader_throughputs) // 2
                ],
                "maximum_aggregate_loader_fraction": recipe["distributed_gate"][
                    "maximum_aggregate_loader_fraction"
                ],
                "maximum_p95_loader_fraction": recipe["distributed_gate"][
                    "maximum_p95_loader_fraction"
                ],
                "passed": True,
            },
            "checkpoint_storage": storage_observations,
            "curve_log": {
                "path": str(args.curve_log),
                "sha256": file_sha256(args.curve_log),
                "event_count": state.event_count,
                "last_event_sha256": state.last_event_sha256,
            },
            "preflight_receipt_sha256": preflight_sha,
            "production_identity": identity,
            "production_identity_sha256": identity_sha,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))


def command_optimizer_env(args: argparse.Namespace) -> None:
    recipe, recipe_sha = load_recipe(args.recipe)
    preflight, preflight_sha = _load_receipt(
        args.preflight_receipt, "d32_family_preflight_receipt"
    )
    if preflight["recipe"]["canonical_sha256"] != recipe_sha:
        _fail("optimizer environment preflight recipe mismatch")
    _probe, probe_sha = _verify_attention_probe(
        args.attention_probe,
        recipe=recipe,
        recipe_sha=recipe_sha,
        preflight_sha=preflight_sha,
        code_revision=preflight["code"]["git_commit"],
    )
    approval, approval_sha = _verify_proxy_acceptance(
        args.wd_proxy_approval,
        recipe=recipe,
        recipe_sha=recipe_sha,
        preflight=preflight,
        attention_probe_sha256=probe_sha,
    )
    values = {
        "EFFECTIVE_BASE_WEIGHT_DECAY": str(approval["accepted_base_weight_decay"]),
        "WSD_WEIGHT_DECAY_COOLDOWN": str(
            approval["accepted_weight_decay_cooldown_policy"]
        ),
        "WD_PROXY_APPROVAL_SHA256": approval_sha,
        "ATTENTION_PROBE_SHA256": probe_sha,
        "ATTENTION_BACKEND": str(
            _mapping(_probe["module_detection"], "attention probe detection")[
                "selected_backend_after_probe"
            ]
        ),
        "WINDOW_PATTERN": str(
            _mapping(_probe["module_detection"], "attention probe detection")[
                "selected_window_pattern"
            ]
        ),
        "CODE_REVISION": str(preflight["code"]["git_commit"]),
    }
    for key, value in values.items():
        print(f"{key}={shlex.quote(value)}")


def _training_cost_for_positions(
    *,
    scheduled_positions: int,
    measured_positions_per_second: float,
    nodes: int,
    cpu_saat_per_node_hour: int,
    reserve_fraction: float,
) -> tuple[int, int]:
    """Return raw and reserved CPU-saat ceilings from measured throughput."""

    if scheduled_positions < 0:
        _fail("scheduled positions for cost projection cannot be negative")
    if (
        not math.isfinite(measured_positions_per_second)
        or measured_positions_per_second <= 0
    ):
        _fail("measured throughput for cost projection must be positive and finite")
    if nodes <= 0 or cpu_saat_per_node_hour <= 0:
        _fail("cost projection node count and billing rate must be positive")
    if not math.isfinite(reserve_fraction) or reserve_fraction < 0:
        _fail("cost projection reserve fraction must be non-negative and finite")
    raw = (
        scheduled_positions
        / measured_positions_per_second
        / 3600.0
        * nodes
        * cpu_saat_per_node_hour
    )
    return math.ceil(raw), math.ceil(raw * (1.0 + reserve_fraction))


def _measured_cost_projection(
    recipe: Mapping[str, Any],
    *,
    selected_smoke_sha256: str,
    world_size: int,
    nodes: int,
    measured_positions_per_second: float,
) -> dict[str, Any]:
    """Derive the production admission cost from the selected d32 smoke."""

    _sha256(selected_smoke_sha256, "selected smoke SHA-256")
    gpus_per_node = int(recipe["distributed_gate"]["gpus_per_node"])
    if world_size not in {8, 16} or nodes * gpus_per_node != world_size:
        _fail("measured cost projection topology is invalid")
    stage_updates = sum(
        int(stage["target_step"]) - int(stage.get("source_step") or 0)
        for stage in recipe["stages"]
    )
    budget = recipe["uhem_budget"]
    full_positions = stage_updates * int(recipe["training"]["global_batch_tokens"])
    if (
        stage_updates != 34_560
        or full_positions != int(budget["shared_lineage_scheduled_token_work"])
    ):
        _fail("measured cost projection shared-lineage arithmetic drifted")
    billing_rate = int(budget["cpu_saat_per_4gpu_node_hour"])
    reserve_fraction = 0.15
    raw_cpu_saat, reserved_cpu_saat = _training_cost_for_positions(
        scheduled_positions=full_positions,
        measured_positions_per_second=measured_positions_per_second,
        nodes=nodes,
        cpu_saat_per_node_hour=billing_rate,
        reserve_fraction=reserve_fraction,
    )
    allowance = int(budget["proxy_and_smoke_reserve_cpu_saat"])
    projected_total = reserved_cpu_saat + allowance
    ceiling = int(budget["operational_ceiling_cpu_saat"])
    return {
        "version": "measured_smoke_v1",
        "selected_smoke_sha256": selected_smoke_sha256,
        "world_size": world_size,
        "nodes": nodes,
        "global_batch_tokens": int(recipe["training"]["global_batch_tokens"]),
        "full_shared_updates": stage_updates,
        "full_scheduled_positions": full_positions,
        "measured_positions_per_second": measured_positions_per_second,
        "billing_cpu_saat_per_node_hour": billing_rate,
        "reserve_fraction": reserve_fraction,
        "raw_training_cpu_saat_ceiling": raw_cpu_saat,
        "reserved_training_cpu_saat": reserved_cpu_saat,
        "proxy_smoke_allowance_cpu_saat": allowance,
        "projected_total_package_cpu_saat": projected_total,
        "operational_ceiling_cpu_saat": ceiling,
        "passed": projected_total <= ceiling,
    }


def command_compare_smokes(args: argparse.Namespace) -> None:
    recipe, recipe_sha = load_recipe(args.recipe)
    expected_smoke8 = args.output.parent / "smoke_ws8.json"
    if args.smoke_8gpu.resolve() != expected_smoke8.resolve() or args.smoke_8gpu.is_symlink():
        _fail("topology gate requires the fixed sibling smoke_ws8.json receipt")
    expected_smoke16 = args.output.parent / "smoke_ws16.json"
    if args.smoke_16gpu is not None:
        if (
            args.smoke_16gpu.resolve() != expected_smoke16.resolve()
            or args.smoke_16gpu.is_symlink()
        ):
            _fail("topology gate requires the fixed sibling smoke_ws16.json receipt")
    elif expected_smoke16.exists() or expected_smoke16.is_symlink():
        _fail("fixed smoke_ws16.json exists but was omitted from topology comparison")
    smoke8, smoke8_sha = _load_receipt(args.smoke_8gpu, "d32_distributed_smoke_receipt")
    if smoke8.get("world_size") != 8:
        _fail("topology gate requires a clean 8-GPU fallback receipt")
    smoke16 = None
    smoke16_sha = None
    if args.smoke_16gpu is not None:
        smoke16, smoke16_sha = _load_receipt(
            args.smoke_16gpu, "d32_distributed_smoke_receipt"
        )
        if smoke16.get("world_size") != 16:
            _fail("the optional preferred-topology receipt must be a 16-GPU smoke")
        if smoke8.get("production_identity_sha256") != smoke16.get(
            "production_identity_sha256"
        ):
            _fail("8- and 16-GPU smokes do not have the same production identity")
    identity = _mapping(smoke8.get("production_identity"), "production_identity")
    if identity.get("recipe_sha256") != recipe_sha:
        _fail("smoke receipts were produced for a different recipe")
    if smoke16 is not None and smoke8.get("preflight_receipt_sha256") != smoke16.get(
        "preflight_receipt_sha256"
    ):
        _fail("8- and 16-GPU smokes were not authorized by the same preflight receipt")
    if smoke16 is not None and smoke8.get("signal_resume_gate_sha256") != smoke16.get(
        "signal_resume_gate_sha256"
    ):
        _fail("8- and 16-GPU smokes used different signal/resume gates")
    launcher_gate_sha = _sha256(
        smoke8.get("static_launcher_gate_sha256"),
        "8-GPU smoke static-launcher-gate SHA-256",
    )
    if smoke16 is not None and smoke16.get("static_launcher_gate_sha256") != launcher_gate_sha:
        _fail("8- and 16-GPU smokes used different static launcher gates")
    capacity_sha = _sha256(
        smoke8.get("packing_capacity_receipt_sha256"),
        "8-GPU smoke packing-capacity SHA-256",
    )
    if smoke8.get("packing_capacity_world_size") != 8:
        _fail("8-GPU smoke did not bind the ws8 packing-capacity record")
    if smoke16 is not None:
        if smoke16.get("packing_capacity_receipt_sha256") != capacity_sha:
            _fail("8- and 16-GPU smokes used different packing-capacity receipts")
        if smoke16.get("packing_capacity_world_size") != 16:
            _fail("16-GPU smoke did not bind the ws16 packing-capacity record")
    throughput8 = _positive_number(
        smoke8.get("scheduled_positions_per_second"), "8-GPU smoke throughput"
    )
    throughput16 = (
        None
        if smoke16 is None
        else _positive_number(
            smoke16.get("scheduled_positions_per_second"), "16-GPU smoke throughput"
        )
    )
    speedup = None if throughput16 is None else throughput16 / throughput8
    threshold = float(recipe["distributed_gate"]["minimum_8_to_16_gpu_speedup"])
    preferred_accepted = speedup is not None and speedup >= threshold
    selected_world_size = 16 if preferred_accepted else 8
    selected_nodes = selected_world_size // int(recipe["distributed_gate"]["gpus_per_node"])
    if smoke16 is None:
        selection_reason = "no_clean_16gpu_smoke_receipt_supplied_use_8gpu_fallback"
    elif preferred_accepted:
        selection_reason = "clean_16gpu_smoke_meets_minimum_1.7_speedup"
    else:
        selection_reason = "clean_16gpu_smoke_below_minimum_1.7_speedup_use_8gpu_fallback"
    selected_smoke_sha = smoke16_sha if selected_world_size == 16 else smoke8_sha
    selected_throughput = throughput16 if selected_world_size == 16 else throughput8
    assert selected_smoke_sha is not None and selected_throughput is not None
    cost_projection = _measured_cost_projection(
        recipe,
        selected_smoke_sha256=selected_smoke_sha,
        world_size=selected_world_size,
        nodes=selected_nodes,
        measured_positions_per_second=selected_throughput,
    )
    if cost_projection["passed"] is not True:
        _fail(
            "measured d32 throughput projects the training package above the "
            "reviewed 40,000 CPU-saat ceiling"
        )
    smoke_storage = []
    smoke_pairs = [("8gpu", smoke8)]
    if smoke16 is not None:
        smoke_pairs.append(("16gpu", smoke16))
    for label, smoke in smoke_pairs:
        storage_record = _mapping(smoke.get("checkpoint_storage"), f"{label} storage")
        for boundary in ("forced_resume", "final"):
            smoke_storage.append(
                _mapping(storage_record.get(boundary), f"{label} {boundary} storage")
            )
    measured_full = max(int(record["full_transaction_bytes"]) for record in smoke_storage)
    measured_model = max(
        int(record["model_metadata_completion_bytes"]) for record in smoke_storage
    )
    storage_factor = float(recipe["storage"]["smoke_measurement_safety_factor"])
    calibrated_full = math.ceil(measured_full * storage_factor)
    calibrated_model = math.ceil(measured_model * storage_factor)
    gate = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "d32_production_topology_gate",
            "family_id": recipe["family_id"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "recipe_sha256": recipe_sha,
            "preflight_receipt_sha256": smoke8["preflight_receipt_sha256"],
            "attention_probe_sha256": identity["attention_probe_sha256"],
            "wsd_proxy_approval_sha256": identity["wsd_proxy_approval_sha256"],
            "signal_resume_gate_sha256": smoke8["signal_resume_gate_sha256"],
            "packing_capacity_receipt_sha256": capacity_sha,
            "authorized_packing_capacity_world_size": selected_world_size,
            "authorized_safe_global_scheduled_positions": (
                smoke8["packing_capacity_safe_global_scheduled_positions"]
                if selected_world_size == 8
                else smoke16["packing_capacity_safe_global_scheduled_positions"]
            ),
            "accepted_base_weight_decay": identity["wsd_base_weight_decay"],
            "accepted_weight_decay_cooldown_policy": identity[
                "wsd_weight_decay_cooldown"
            ],
            "smoke_8gpu_sha256": smoke8_sha,
            "smoke_16gpu_sha256": smoke16_sha,
            "throughput_8gpu": throughput8,
            "throughput_16gpu": throughput16,
            "speedup_8_to_16": speedup,
            "parallel_efficiency": None if speedup is None else speedup / 2.0,
            "cost_projection": cost_projection,
            "storage_calibration": {
                "safety_factor": storage_factor,
                "maximum_measured_full_transaction_bytes": measured_full,
                "maximum_measured_model_bundle_bytes": measured_model,
                "calibrated_full_transaction_bytes": calibrated_full,
                "calibrated_model_bundle_bytes": calibrated_model,
            },
            "required_speedup": threshold,
            "preferred_topology_accepted": preferred_accepted,
            "selection_reason": selection_reason,
            "passed": True,
            "authorized_production_nodes": selected_nodes,
            "authorized_production_world_size": selected_world_size,
            "require_single_world_size_for_entire_lineage": True,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(args.output, gate)
    print(json.dumps(gate, sort_keys=True))


def _stage_by_id(recipe: Mapping[str, Any], stage_id: str) -> Mapping[str, Any]:
    for value in recipe["stages"]:
        if isinstance(value, Mapping) and value.get("id") == stage_id:
            return value
    _fail(f"unknown family stage: {stage_id}")


def _stage_run_id(recipe: Mapping[str, Any], stage: Mapping[str, Any]) -> str:
    return (
        f"{recipe['family_id']}_trunk"
        if stage["kind"] == "trunk"
        else f"{recipe['family_id']}_{stage['id']}"
    )


def _verified_stage_resume_step(
    recipe: Mapping[str, Any], base_dir: Path, stage: Mapping[str, Any]
) -> int | None:
    """Return the latest verified in-stage checkpoint used by env and cost gates."""

    checkpoint_root = base_dir / "base_checkpoints" / str(stage["model_tag"])
    minimum_step = int(stage.get("source_step") or 0)
    target_step = int(stage["target_step"])
    expected_run_id = _stage_run_id(recipe, stage)
    candidates: list[int] = []
    if checkpoint_root.is_dir():
        for child in checkpoint_root.glob("strict_*"):
            match = re.fullmatch(r"strict_(\d{6,})", child.name)
            if match is None or not (child / "completion.json").is_file():
                continue
            step = int(match.group(1))
            if minimum_step <= step < target_step:
                checkpoint, _sha = _checkpoint_manifest(
                    base_dir, str(stage["model_tag"]), step
                )
                identity = _mapping(
                    checkpoint.get("identity"), "stage resume checkpoint identity"
                )
                if identity.get("run_id") != expected_run_id:
                    _fail("stage resume checkpoint run ID differs from the recipe stage")
                candidates.append(step)
    return max(candidates) if candidates else None


def _checkpoint_manifest(base_dir: Path, model_tag: str, step: int) -> tuple[dict[str, Any], str]:
    from nanochat.strict_checkpoint import inspect_strict_checkpoint

    checkpoint_root = base_dir / "base_checkpoints" / model_tag
    try:
        manifest = inspect_strict_checkpoint(checkpoint_root, step)
    except Exception as exc:
        raise FamilyWorkflowError(
            f"strict checkpoint verification failed for {model_tag}@{step}: {exc}"
        ) from exc
    digest = verify_manifest_hash(manifest)
    return manifest, digest


def collect_final_evaluation_evidence(
    *,
    recipe: Mapping[str, Any],
    recipe_sha: str,
    preflight: Mapping[str, Any],
    preflight_sha: str,
    gate: Mapping[str, Any],
    gate_sha: str,
    base_dir: Path,
    lineage_dir: Path,
    checkpoint_records: Mapping[tuple[str, int], tuple[Mapping[str, Any], str]] | None = None,
    lineage_records: Mapping[str, tuple[Mapping[str, Any], str]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Collect exact, finite full-validation evidence for all cooled finals."""

    if (
        preflight.get("recipe", {}).get("canonical_sha256") != recipe_sha
        or gate.get("recipe_sha256") != recipe_sha
        or gate.get("preflight_receipt_sha256") != preflight_sha
        or gate.get("passed") is not True
    ):
        _fail("final-quality evidence uses an invalid preflight/topology chain")
    by_target = {
        (str(stage["model_tag"]), int(stage["target_step"])): stage
        for stage in recipe["stages"]
        if stage["kind"] == "cooldown_fork"
    }
    finals = _sequence(recipe["checkpoints"].get("finals"), "checkpoint finals")
    evidence: dict[str, dict[str, Any]] = {}
    for final_value in finals:
        final = _mapping(final_value, "cooled final")
        label = str(final["label"])
        model_tag = str(final["model_tag"])
        step = int(final["final_step"])
        stage = by_target.get((model_tag, step))
        if stage is None:
            _fail(f"cooled final {label} has no exact recipe stage")
        key = (model_tag, step)
        if checkpoint_records is None:
            checkpoint, checkpoint_sha = _checkpoint_manifest(base_dir, model_tag, step)
        else:
            checkpoint_record = checkpoint_records.get(key)
            if checkpoint_record is None:
                _fail(f"final-quality evidence lacks checkpoint {model_tag}@{step}")
            checkpoint, checkpoint_sha = checkpoint_record
            if verify_manifest_hash(checkpoint) != checkpoint_sha:
                _fail(f"final-quality checkpoint hash drifted for {label}")
        lineage_path = lineage_dir / f"{stage['id']}.json"
        if lineage_records is None:
            lineage, lineage_sha = _load_receipt(
                lineage_path, "d32_checkpoint_lineage_receipt"
            )
        else:
            lineage_record = lineage_records.get(str(stage["id"]))
            if lineage_record is None:
                _fail(f"final-quality evidence lacks lineage for {label}")
            lineage, lineage_sha = lineage_record
            if verify_manifest_hash(lineage) != lineage_sha:
                _fail(f"final-quality lineage hash drifted for {label}")
        if (
            lineage.get("family_id") != recipe["family_id"]
            or lineage.get("stage_id") != stage["id"]
            or lineage.get("recipe_sha256") != recipe_sha
            or lineage.get("preflight_receipt_sha256") != preflight_sha
            or lineage.get("production_gate_sha256") != gate_sha
            or lineage.get("target", {}).get("model_tag") != model_tag
            or lineage.get("target", {}).get("step") != step
            or lineage.get("target", {}).get("checkpoint_sha256") != checkpoint_sha
            or lineage.get("target", {}).get("retention_class")
            != "cooled_final_full_resumable_retained"
        ):
            _fail(f"final-quality lineage binding drifted for {label}")

        identity = _mapping(checkpoint.get("identity"), f"{label} identity")
        protocol = _mapping(identity.get("protocol"), f"{label} protocol")
        validation = _mapping(protocol.get("validation"), f"{label} validation")
        curve_log = _mapping(identity.get("curve_log"), f"{label} curve log")
        run_id = str(identity.get("run_id", ""))
        if (
            protocol.get("run_kind") != "production"
            or protocol.get("recipe_scope") != stage["id"]
            or protocol.get("model_tag") != model_tag
            or protocol.get("num_iterations") != int(stage["num_iterations"])
            or run_id != f"{recipe['family_id']}_{stage['id']}"
            or validation.get("manifest_sha256")
            != preflight["corpus"]["validation_exposure_manifest_sha256"]
            or validation.get("full_manifest") is not True
            or validation.get("eval_tokens_cli_unused") != -1
            or curve_log.get("last_updates_completed") != step
        ):
            _fail(f"final-quality fixed-validation protocol drifted for {label}")

        file_records = _sequence(checkpoint.get("files"), f"{label} checkpoint files")
        meta_records = [record for record in file_records if record.get("role") == "meta"]
        if len(meta_records) != 1 or meta_records[0].get("path") != "meta.json":
            _fail(f"final-quality checkpoint metadata inventory drifted for {label}")
        meta_record = _mapping(meta_records[0], f"{label} metadata record")
        step_dir = base_dir / "base_checkpoints" / model_tag / f"strict_{step:06d}"
        meta_path = step_dir / "meta.json"
        meta = _load_object(meta_path, f"{label} final checkpoint metadata")
        if (
            meta.get("step") != step
            or meta.get("updates_completed") != step
            or meta.get("strict_run_contract_sha256") != identity.get("run_sha256")
            or meta_path.stat().st_size != meta_record.get("size_bytes")
            or file_sha256(meta_path) != meta_record.get("sha256")
        ):
            _fail(f"final-quality checkpoint metadata drifted for {label}")
        coverage = _mapping(meta.get("validation_coverage"), f"{label} validation coverage")
        expected_coverage = {
            field: validation[field]
            for field in (
                "target_tokens",
                "payload_bytes",
                "documents",
                "logical_rows",
                "padded_token_positions_world1",
                "row_layout_sha256",
            )
        }
        if dict(coverage) != expected_coverage:
            _fail(f"final-quality checkpoint validation coverage drifted for {label}")
        final_bpb = meta.get("val_bpb")
        minimum_bpb = _mapping(meta.get("loop_state"), f"{label} loop state").get(
            "min_val_bpb"
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in (final_bpb, minimum_bpb)
        ) or float(minimum_bpb) > float(final_bpb):
            _fail(f"final-quality BPB evidence is invalid for {label}")
        curve_path = base_dir / "metrics" / "d32_family" / run_id / "training_curve.jsonl"
        events, curve_state = read_training_log(
            curve_path,
            expected_study_id=recipe["family_id"],
            expected_run_id=run_id,
        )
        expected_curve_state = {
            "event_count": curve_state.event_count,
            "last_event_sha256": curve_state.last_event_sha256,
            "last_updates_completed": curve_state.last_updates_completed,
            "recovered_truncated_bytes": curve_state.recovered_truncated_bytes,
            "file_sha256": file_sha256(curve_path),
        }
        if curve_log != expected_curve_state or not events:
            _fail(f"final-quality curve-log binding drifted for {label}")
        terminal = _mapping(events[-1], f"{label} terminal validation event")
        metrics = _mapping(terminal.get("metrics"), f"{label} terminal metrics")
        terminal_identities = _mapping(
            terminal.get("identities"), f"{label} terminal identities"
        )
        metric_keys = {
            "val/all_target_nats",
            "val/all_target_count",
            "val/payload_nats",
            "val/payload_target_count",
            "val/payload_bytes",
            "val/bpb",
        }
        if (
            terminal.get("event_type") != "validation"
            or terminal.get("updates_completed") != step
            or terminal_identities.get("validation_manifest_sha256")
            != validation["manifest_sha256"]
            or terminal_identities.get("run_sha256") != identity.get("run_sha256")
            or set(metrics) != metric_keys
        ):
            _fail(f"final-quality terminal validation event drifted for {label}")
        for name in (
            "val/all_target_count",
            "val/payload_target_count",
            "val/payload_bytes",
        ):
            _positive_int(metrics.get(name), f"{label} terminal {name}")
        for name in ("val/all_target_nats", "val/payload_nats", "val/bpb"):
            _positive_number(metrics.get(name), f"{label} terminal {name}")
        recomputed_bpb = float(metrics["val/payload_nats"]) / (
            math.log(2.0) * int(metrics["val/payload_bytes"])
        )
        if (
            metrics["val/all_target_count"] != coverage["target_tokens"]
            or metrics["val/payload_target_count"] != coverage["target_tokens"]
            or metrics["val/payload_bytes"] != coverage["payload_bytes"]
            or not math.isclose(
                float(metrics["val/bpb"]), recomputed_bpb, rel_tol=1e-12, abs_tol=1e-12
            )
            or not math.isclose(
                float(metrics["val/bpb"]), float(final_bpb), rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            _fail(f"final-quality terminal validation arithmetic drifted for {label}")
        evidence[label] = {
            "stage_id": str(stage["id"]),
            "model_tag": model_tag,
            "step": step,
            "checkpoint_sha256": checkpoint_sha,
            "lineage_receipt_sha256": lineage_sha,
            "checkpoint_meta_sha256": meta_record["sha256"],
            "validation_manifest_sha256": validation["manifest_sha256"],
            "full_fixed_validation": True,
            "validation_coverage": expected_coverage,
            "run_id": run_id,
            "run_sha256": identity["run_sha256"],
            "exposure_plan_sha256": identity["exposure_plan_sha256"],
            "curve_log": {
                "path": curve_path.relative_to(base_dir).as_posix(),
                "size_bytes": curve_path.stat().st_size,
                **dict(curve_log),
                "curve_log_state_sha256": identity["curve_log_state_sha256"],
            },
            "terminal_validation_event": {
                "event_index": terminal["event_index"],
                "event_sha256": terminal["event_sha256"],
                "updates_completed": step,
                **{name: metrics[name] for name in sorted(metric_keys)},
            },
            "final_validation_bpb": float(final_bpb),
            "minimum_validation_bpb": float(minimum_bpb),
        }
    expected_labels = {str(final["label"]) for final in finals}
    if set(evidence) != expected_labels or len(evidence) != 3:
        _fail("final-quality evidence must contain exactly all three cooled finals")
    return evidence


def validate_final_model_publication_approval(
    approval: Mapping[str, Any],
    *,
    recipe: Mapping[str, Any],
    recipe_sha: str,
    preflight_sha: str,
    gate_sha: str,
    expected_evidence: Mapping[str, Any],
    require_accepted: bool,
) -> str:
    """Validate one manual publication decision over exact final evaluations."""

    digest = verify_manifest_hash(approval)
    if (
        approval.get("schema_version") != "1.0"
        or approval.get("kind") != FINAL_MODEL_PUBLICATION_APPROVAL_KIND
        or approval.get("family_id") != recipe["family_id"]
        or approval.get("recipe_sha256") != recipe_sha
        or approval.get("preflight_receipt_sha256") != preflight_sha
        or approval.get("production_gate_sha256") != gate_sha
        or approval.get("automatic_decision") is not False
        or approval.get("review_confirmation")
        != "all_three_final_fixed_validation_results_and_checkpoint_lineage_reviewed"
        or approval.get("final_evaluations") != expected_evidence
        or approval.get("required_final_labels") != sorted(expected_evidence)
        or approval.get("automatic_evidence_validation_passed") is not True
        or approval.get("manual_acceptance_required") is not True
        or approval.get("quality_decision_policy")
        != "manual_review_no_automatic_numeric_threshold"
        or approval.get("decision") not in {"accepted", "rejected"}
        or not isinstance(approval.get("reviewer"), str)
        or not approval["reviewer"].strip()
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
            str(approval.get("reviewed_at_utc", "")),
        )
        is None
        or not isinstance(approval.get("notes"), str)
    ):
        _fail("final-model publication approval is malformed or stale")
    if require_accepted and approval.get("decision") != "accepted":
        _fail("family publication requires an accepted final-model quality decision")
    return digest


def command_seal_final_model_publication_approval(args: argparse.Namespace) -> None:
    if args.output.exists():
        _fail(f"refusing to overwrite final-model publication approval: {args.output}")
    recipe, recipe_sha = load_recipe(args.recipe)
    preflight, preflight_sha = _load_receipt(
        args.preflight_receipt, "d32_family_preflight_receipt"
    )
    gate, gate_sha = _load_receipt(args.gate, "d32_production_topology_gate")
    evidence = collect_final_evaluation_evidence(
        recipe=recipe,
        recipe_sha=recipe_sha,
        preflight=preflight,
        preflight_sha=preflight_sha,
        gate=gate,
        gate_sha=gate_sha,
        base_dir=args.base_dir.expanduser().resolve(),
        lineage_dir=args.lineage_dir.expanduser().resolve(),
    )
    if not args.reviewer.strip() or re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", args.reviewed_at_utc
    ) is None:
        _fail("final-model approval requires reviewer and RFC3339 UTC timestamp")
    approval = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": FINAL_MODEL_PUBLICATION_APPROVAL_KIND,
            "family_id": recipe["family_id"],
            "recipe_sha256": recipe_sha,
            "preflight_receipt_sha256": preflight_sha,
            "production_gate_sha256": gate_sha,
            "final_evaluations": evidence,
            "required_final_labels": sorted(evidence),
            "automatic_evidence_validation_passed": True,
            "manual_acceptance_required": True,
            "automatic_decision": False,
            "quality_decision_policy": "manual_review_no_automatic_numeric_threshold",
            "review_confirmation": (
                "all_three_final_fixed_validation_results_and_checkpoint_lineage_reviewed"
            ),
            "reviewer": args.reviewer.strip(),
            "reviewed_at_utc": args.reviewed_at_utc,
            "decision": args.decision,
            "notes": args.notes,
            "canonical_sha256": None,
        }
    )
    validate_final_model_publication_approval(
        approval,
        recipe=recipe,
        recipe_sha=recipe_sha,
        preflight_sha=preflight_sha,
        gate_sha=gate_sha,
        expected_evidence=evidence,
        require_accepted=False,
    )
    write_json_atomic(args.output, approval)
    print(json.dumps(approval, sort_keys=True))


def _verify_gate_and_preflight(
    recipe: Mapping[str, Any],
    recipe_sha: str,
    preflight_path: Path,
    gate_path: Path,
    proxy_approval_path: Path,
    attention_probe_path: Path,
) -> tuple[dict[str, Any], str, dict[str, Any], str, dict[str, Any], str]:
    preflight, preflight_sha = _load_receipt(
        preflight_path, "d32_family_preflight_receipt"
    )
    gate, gate_sha = _load_receipt(gate_path, "d32_production_topology_gate")
    if preflight["recipe"]["canonical_sha256"] != recipe_sha:
        _fail("preflight receipt recipe mismatch")
    if gate.get("recipe_sha256") != recipe_sha:
        _fail("production topology gate recipe mismatch")
    if gate.get("preflight_receipt_sha256") != preflight_sha:
        _fail("production topology gate was produced from a different preflight receipt")
    selected_world_size = gate.get("authorized_production_world_size")
    selected_nodes = gate.get("authorized_production_nodes")
    if gate.get("passed") is not True or selected_world_size not in {8, 16}:
        _fail("production topology gate did not authorize 8 or 16 GPUs")
    if selected_nodes != selected_world_size // int(recipe["distributed_gate"]["gpus_per_node"]):
        _fail("production topology gate node/world-size arithmetic mismatch")
    if gate.get("require_single_world_size_for_entire_lineage") is not True:
        _fail("production topology gate does not lock one world size for the lineage")
    if selected_world_size == 16:
        if gate.get("preferred_topology_accepted") is not True or float(
            gate.get("speedup_8_to_16", 0)
        ) < float(recipe["distributed_gate"]["minimum_8_to_16_gpu_speedup"]):
            _fail("16 GPUs were authorized without the required measured speedup")
    elif gate.get("preferred_topology_accepted") is not False:
        _fail("8-GPU fallback gate has an inconsistent preferred-topology decision")
    selected_smoke_sha = (
        gate.get("smoke_16gpu_sha256")
        if selected_world_size == 16
        else gate.get("smoke_8gpu_sha256")
    )
    selected_throughput = (
        gate.get("throughput_16gpu")
        if selected_world_size == 16
        else gate.get("throughput_8gpu")
    )
    try:
        expected_cost_projection = _measured_cost_projection(
            recipe,
            selected_smoke_sha256=str(selected_smoke_sha),
            world_size=int(selected_world_size),
            nodes=int(selected_nodes),
            measured_positions_per_second=float(selected_throughput),
        )
    except (TypeError, ValueError) as exc:
        raise FamilyWorkflowError(
            f"production topology gate cost inputs are invalid: {exc}"
        ) from exc
    if gate.get("cost_projection") != expected_cost_projection:
        _fail("production topology gate measured-cost projection drifted")
    if expected_cost_projection["passed"] is not True:
        _fail("production topology gate exceeds the reviewed CPU-saat ceiling")
    capacity_sha, capacity_world = _preflight_capacity_world(
        preflight, int(selected_world_size)
    )
    if gate.get("packing_capacity_receipt_sha256") != capacity_sha:
        _fail("production topology gate used a different packing-capacity receipt")
    if gate.get("authorized_packing_capacity_world_size") != selected_world_size:
        _fail("production topology gate selected a different capacity world")
    safe_positions = capacity_authorized_positions(capacity_world)
    if gate.get("authorized_safe_global_scheduled_positions") != safe_positions:
        _fail("production topology gate safe-position bound differs from preflight")
    maximum_horizon = max(
        int(final["scheduled_tokens"]) for final in recipe["checkpoints"]["finals"]
    )
    if safe_positions < maximum_horizon:
        _fail("selected production topology cannot reach the 40x horizon without wrap")
    _probe, attention_probe_sha = _verify_attention_probe(
        attention_probe_path,
        recipe=recipe,
        recipe_sha=recipe_sha,
        preflight_sha=preflight_sha,
        code_revision=preflight["code"]["git_commit"],
    )
    approval, approval_sha = _verify_proxy_acceptance(
        proxy_approval_path,
        recipe=recipe,
        recipe_sha=recipe_sha,
        preflight=preflight,
        attention_probe_sha256=attention_probe_sha,
    )
    if gate.get("attention_probe_sha256") != attention_probe_sha:
        _fail("production topology gate used a different attention-backend probe")
    if gate.get("wsd_proxy_approval_sha256") != approval_sha:
        _fail("production topology gate used a different proxy-approved optimizer")
    if gate.get("accepted_base_weight_decay") != approval["accepted_base_weight_decay"]:
        _fail("production topology gate base weight decay differs from proxy approval")
    if gate.get("accepted_weight_decay_cooldown_policy") != approval[
        "accepted_weight_decay_cooldown_policy"
    ]:
        _fail("production topology gate cooldown policy differs from proxy approval")
    try:
        strict_gate, strict_gate_sha = validate_production_topology_gate(
            gate_path,
            preflight_path,
            recipe=recipe,
            recipe_sha256=recipe_sha,
            attention_probe_sha256=attention_probe_sha,
            proxy_approval_sha256=approval_sha,
            accepted_base_weight_decay=float(approval["accepted_base_weight_decay"]),
            accepted_weight_decay_cooldown_policy=str(
                approval["accepted_weight_decay_cooldown_policy"]
            ),
            world_size=int(selected_world_size),
            packing_capacity_receipt_sha256=capacity_sha,
            selected_capacity=capacity_world,
        )
    except StrictTrainingError as exc:
        raise FamilyWorkflowError(
            f"production topology fixed-smoke verification failed: {exc}"
        ) from exc
    if strict_gate_sha != gate_sha or strict_gate != gate:
        _fail("production topology validator returned a different sealed gate")
    return preflight, preflight_sha, gate, gate_sha, approval, approval_sha


def _completed_target_count(recipe: Mapping[str, Any], base_dir: Path) -> int:
    count = 0
    seen: set[tuple[str, int]] = set()
    for stage in recipe["stages"]:
        key = (str(stage["model_tag"]), int(stage["target_step"]))
        if key in seen:
            continue
        seen.add(key)
        path = base_dir / "base_checkpoints" / key[0] / f"strict_{key[1]:06d}" / "completion.json"
        if path.is_file():
            _checkpoint_manifest(base_dir, key[0], key[1])
            count += 1
    return count


def command_preflight_stage(args: argparse.Namespace) -> None:
    recipe, recipe_sha = load_recipe(args.recipe)
    preflight, preflight_sha, _gate, gate_sha, _approval, _approval_sha = _verify_gate_and_preflight(
        recipe, recipe_sha, args.preflight_receipt, args.gate, args.wd_proxy_approval,
        args.attention_probe,
    )
    if args.world_size != _gate["authorized_production_world_size"]:
        _fail("stage world size differs from the sealed production topology decision")
    _signal_gate, signal_gate_sha = _verify_signal_resume_gate(
        args.signal_resume_gate,
        recipe=recipe,
        recipe_sha=recipe_sha,
        preflight=preflight,
        preflight_sha=preflight_sha,
    )
    if _gate.get("signal_resume_gate_sha256") != signal_gate_sha:
        _fail("production topology gate used a different signal/resume gate")
    stage = _stage_by_id(recipe, args.stage)
    production_world_size = int(_gate["authorized_production_world_size"])
    stage_ids = [value["id"] for value in recipe["stages"]]
    stage_index = stage_ids.index(args.stage)
    for prior_id in stage_ids[:stage_index]:
        prior_path = args.lineage_dir / f"{prior_id}.json"
        prior, _prior_sha = _load_receipt(
            prior_path, "d32_checkpoint_lineage_receipt"
        )
        expected_prior = {
            "family_id": recipe["family_id"],
            "stage_id": prior_id,
            "recipe_sha256": recipe_sha,
            "preflight_receipt_sha256": preflight_sha,
            "production_gate_sha256": gate_sha,
            "wsd_proxy_approval_sha256": _approval_sha,
            "production_world_size": args.world_size,
        }
        for field, expected in expected_prior.items():
            if prior.get(field) != expected:
                _fail(f"prior lineage receipt {prior_id} {field} mismatch")
    base_dir = args.base_dir.expanduser().resolve()
    if str(base_dir) != preflight["base_dir"]:
        _fail("stage base directory differs from the preflight receipt")
    target_dir = (
        base_dir
        / "base_checkpoints"
        / str(stage["model_tag"])
        / f"strict_{int(stage['target_step']):06d}"
    )
    if target_dir.exists():
        _fail(f"refusing to overwrite an existing target checkpoint: {target_dir}")
    if stage.get("source_step") is not None:
        source_tag = str(stage.get("source_model_tag", stage["model_tag"]))
        _checkpoint_manifest(base_dir, source_tag, int(stage["source_step"]))

    completed = _completed_target_count(recipe, base_dir)
    storage = recipe["storage"]
    # Free space already reflects completed artifacts. Estimate only future
    # retained targets, one atomic-write copy, logs/evals and immutable safety
    # headroom. V3 keeps every future cooled final fully resumable.
    incomplete = [
        stage_value
        for stage_value in recipe["stages"]
        if not (
            base_dir
            / "base_checkpoints"
            / str(stage_value["model_tag"])
            / f"strict_{int(stage_value['target_step']):06d}"
            / "completion.json"
        ).is_file()
    ]
    future_stable = sum(stage_value["kind"] == "trunk" for stage_value in incomplete)
    future_finals = sum(stage_value["kind"] == "cooldown_fork" for stage_value in incomplete)
    calibration = _mapping(
        _gate.get("storage_calibration"), "production gate storage calibration"
    )
    if calibration.get("safety_factor") != storage["smoke_measurement_safety_factor"]:
        _fail("production gate storage safety factor differs from recipe")
    full_bytes = max(
        int(storage["estimated_full_resumable_transaction_bytes"]),
        _positive_int(
            calibration.get("calibrated_full_transaction_bytes"),
            "calibrated full transaction bytes",
        ),
    )
    model_bytes = max(
        int(storage["estimated_cooled_final_model_bundle_bytes"]),
        _positive_int(
            calibration.get("calibrated_model_bundle_bytes"),
            "calibrated model bundle bytes",
        ),
    )
    if (
        int(storage["cooled_final_model_bundles_retained"]) == 0
        and int(storage["full_cooled_final_transactions_at_peak"]) == 3
    ):
        final_bytes = full_bytes * future_finals
        full_final_transient_bytes = 0
    else:
        final_bytes = model_bytes * future_finals
        full_final_transient_bytes = (
            full_bytes * int(storage["full_cooled_final_transactions_at_peak"])
        )
    required_free = (
        full_bytes * future_stable
        + final_bytes
        + full_final_transient_bytes
        + full_bytes * int(storage["atomic_write_transient_transactions"])
        + full_bytes * int(storage["maximum_retained_preemption_transactions"])
        + int(storage["estimated_logs_and_evaluations_bytes"])
        + int(storage["minimum_free_headroom_bytes"])
    )
    live_storage_policy = storage["uhem_live_quota"]
    free, live_storage_audit = _live_beegfs_storage(
        REPO_ROOT,
        uid=int(live_storage_policy["uid"]),
        storage_pool_id=int(live_storage_policy["storage_pool_id"]),
        path=base_dir,
    )
    if free < required_free:
        _fail(
            f"insufficient free storage for stage {args.stage}: need {required_free}, found {free}; "
            "the workflow never auto-deletes existing or prior-model artifacts"
        )
    current_resume_step = _verified_stage_resume_step(recipe, base_dir, stage)
    current_source_step = int(stage.get("source_step") or 0)
    current_progress_step = max(
        current_source_step,
        current_source_step if current_resume_step is None else current_resume_step,
    )
    remaining_updates = int(stage["target_step"]) - current_progress_step
    if remaining_updates <= 0:
        _fail("current production stage has no positive uncompleted update range")
    for future_stage in recipe["stages"][stage_index + 1 :]:
        future_target = (
            base_dir
            / "base_checkpoints"
            / str(future_stage["model_tag"])
            / f"strict_{int(future_stage['target_step']):06d}"
            / "completion.json"
        )
        if future_target.is_file():
            _fail("a future stage target exists before its reviewed lineage turn")
        future_delta = int(future_stage["target_step"]) - int(
            future_stage.get("source_step") or 0
        )
        if future_delta <= 0:
            _fail("future production stage has a non-positive update range")
        remaining_updates += future_delta
    remaining_positions = remaining_updates * int(
        recipe["training"]["global_batch_tokens"]
    )
    measured_cost = _mapping(
        _gate.get("cost_projection"), "production gate cost projection"
    )
    raw_remaining_cpu_saat, reserved_remaining_cpu_saat = (
        _training_cost_for_positions(
            scheduled_positions=remaining_positions,
            measured_positions_per_second=float(
                measured_cost["measured_positions_per_second"]
            ),
            nodes=int(_gate["authorized_production_nodes"]),
            cpu_saat_per_node_hour=int(
                measured_cost["billing_cpu_saat_per_node_hour"]
            ),
            reserve_fraction=float(measured_cost["reserve_fraction"]),
        )
    )
    budget = recipe["uhem_budget"]
    live_cpu_saat, quota_output_sha, quota_audit = _live_uhem_cpu_saat(
        REPO_ROOT, str(budget["account"]), str(budget["user"])
    )
    quota_floor = float(reserved_remaining_cpu_saat)
    if live_cpu_saat < quota_floor:
        _fail(
            f"live UHeM quota fell below the measured remaining-work safety floor: "
            f"need {quota_floor:.0f} CPU-saat, found {live_cpu_saat:.2f}"
        )
    result = {
        "stage": args.stage,
        "preflight_receipt_sha256": preflight_sha,
        "gate_sha256": gate_sha,
        "completed_transactions": completed,
        "remaining_transactions_including_this_stage": len(incomplete),
        "filesystem_free_bytes": free,
        "live_storage": live_storage_audit,
        "required_free_bytes": required_free,
        "full_transaction_bytes_used": full_bytes,
        "model_bundle_bytes_used": model_bytes,
        "live_remaining_cpu_saat": live_cpu_saat,
        "required_cpu_saat_safety_floor": quota_floor,
        "remaining_cost_projection": {
            "current_resume_step": current_resume_step,
            "remaining_updates": remaining_updates,
            "remaining_scheduled_positions": remaining_positions,
            "measured_positions_per_second": measured_cost[
                "measured_positions_per_second"
            ],
            "nodes": _gate["authorized_production_nodes"],
            "billing_cpu_saat_per_node_hour": measured_cost[
                "billing_cpu_saat_per_node_hour"
            ],
            "reserve_fraction": measured_cost["reserve_fraction"],
            "raw_remaining_cpu_saat_ceiling": raw_remaining_cpu_saat,
            "reserved_remaining_cpu_saat": reserved_remaining_cpu_saat,
        },
        "quota_output_sha256": quota_output_sha,
        "live_quota": quota_audit,
    }
    print(json.dumps(result, sort_keys=True))


def command_stage_env(args: argparse.Namespace) -> None:
    recipe, recipe_sha = load_recipe(args.recipe)
    gate, _gate_sha = _load_receipt(args.gate, "d32_production_topology_gate")
    if gate.get("recipe_sha256") != recipe_sha or gate.get("passed") is not True:
        _fail("stage environment topology gate is invalid or belongs to another recipe")
    production_world_size = gate.get("authorized_production_world_size")
    production_nodes = gate.get("authorized_production_nodes")
    if production_world_size not in {8, 16} or production_nodes != production_world_size // 4:
        _fail("stage environment topology selection is invalid")
    stage = _stage_by_id(recipe, args.stage)
    base_dir = args.base_dir.expanduser().resolve()
    resume_step = _verified_stage_resume_step(recipe, base_dir, stage)
    parent_sha = ""
    parent_dir = ""
    if stage["kind"] == "cooldown_fork":
        source_tag = str(stage["source_model_tag"])
        source_step = int(stage["source_step"])
        _parent, parent_sha = _checkpoint_manifest(base_dir, source_tag, source_step)
        parent_dir = str(base_dir / "base_checkpoints" / source_tag)
    values: dict[str, str] = {
        "FAMILY_ID": str(recipe["family_id"]),
        "STAGE_ID": str(stage["id"]),
        "STAGE_KIND": str(stage["kind"]),
        "MODEL_TAG": str(stage["model_tag"]),
        "SOURCE_MODEL_TAG": str(stage.get("source_model_tag", stage["model_tag"])),
        "SOURCE_STEP": "" if stage.get("source_step") is None else str(stage["source_step"]),
        "TARGET_STEP": str(stage["target_step"]),
        "NUM_ITERATIONS": str(stage["num_iterations"]),
        "WSD_COOLDOWN_START_STEP": str(stage["cooldown_start_step"]),
        "EXPOSURE_PLAN_KEY": (
            f"{stage['exposure_plan_family']}_ws{production_world_size}_seed42"
        ),
        "PRODUCTION_WORLD_SIZE": str(production_world_size),
        "PRODUCTION_NODES": str(production_nodes),
        "RESUME_FROM_STEP": "" if resume_step is None else str(resume_step),
        "PARENT_CHECKPOINT_DIR": parent_dir,
        "PARENT_CHECKPOINT_SHA256": parent_sha,
        "RUN_ID": f"{recipe['family_id']}_{'trunk' if stage['kind'] == 'trunk' else stage['id']}",
    }
    for key, value in values.items():
        if "\n" in value or "\r" in value:
            _fail(f"unsafe newline in stage environment value {key}")
        print(f"{key}={shlex.quote(value)}")


def command_seal_preemption(args: argparse.Namespace) -> None:
    """Authorize requeue only after unanimous rank-75 exits and a verified checkpoint."""

    recipe, recipe_sha = load_recipe(args.recipe)
    maximum_requeues = int(recipe["storage"]["maximum_retained_preemption_transactions"])
    if args.slurm_restart_count < 0 or args.slurm_restart_count >= maximum_requeues:
        _fail(
            "production preemption exceeds the reviewed retained-transaction/requeue cap"
        )
    preflight, preflight_sha, gate, gate_sha, _approval, approval_sha = (
        _verify_gate_and_preflight(
            recipe,
            recipe_sha,
            args.preflight_receipt,
            args.gate,
            args.wd_proxy_approval,
            args.attention_probe,
        )
    )
    stage = _stage_by_id(recipe, args.stage)
    _signal_gate, signal_gate_sha = _verify_signal_resume_gate(
        args.signal_resume_gate,
        recipe=recipe,
        recipe_sha=recipe_sha,
        preflight=preflight,
        preflight_sha=preflight_sha,
    )
    if gate.get("signal_resume_gate_sha256") != signal_gate_sha:
        _fail("preemption stage uses a different signal/resume gate")
    attention_probe, attention_probe_sha = _verify_attention_probe(
        args.attention_probe,
        recipe=recipe,
        recipe_sha=recipe_sha,
        preflight_sha=preflight_sha,
        code_revision=preflight["code"]["git_commit"],
    )
    world_size = int(gate["authorized_production_world_size"])
    nodes = int(gate["authorized_production_nodes"])
    run_id = (
        f"{recipe['family_id']}_trunk"
        if stage["kind"] == "trunk"
        else f"{recipe['family_id']}_{stage['id']}"
    )
    _launch, launch_sha = _verify_static_launch_receipt(
        args.launch_receipt,
        recipe=recipe,
        recipe_sha=recipe_sha,
        run_id=run_id,
        phase=f"production_{stage['id']}",
        slurm_job_id=args.slurm_job_id,
        world_size=world_size,
        nodes=nodes,
        expected_child_exit_code=75,
    )
    base_dir = args.base_dir.expanduser().resolve()
    checkpoint_root = base_dir / "base_checkpoints" / str(stage["model_tag"])
    source_step = int(stage.get("source_step") or 0)
    target_step = int(stage["target_step"])
    candidates: list[int] = []
    if checkpoint_root.is_dir():
        for child in checkpoint_root.glob("strict_*"):
            match = re.fullmatch(r"strict_(\d{6,})", child.name)
            if match is None or not (child / "completion.json").is_file():
                continue
            step = int(match.group(1))
            if source_step <= step < target_step:
                candidates.append(step)
    selected_tag = str(stage["model_tag"])
    used_stage_source = False
    if candidates:
        checkpoint_step = max(candidates)
        checkpoint, checkpoint_sha = _checkpoint_manifest(
            base_dir, selected_tag, checkpoint_step
        )
        identity = _mapping(checkpoint.get("identity"), "preemption checkpoint identity")
        if identity.get("run_id") != run_id:
            _fail("preemption checkpoint run ID differs from the interrupted stage")
    elif stage.get("source_step") is not None:
        selected_tag = str(stage.get("source_model_tag", stage["model_tag"]))
        checkpoint_step = int(stage["source_step"])
        checkpoint, checkpoint_sha = _checkpoint_manifest(
            base_dir, selected_tag, checkpoint_step
        )
        used_stage_source = True
    else:
        _fail("collective rank-75 exit has no complete checkpoint from which to requeue")
    if int(checkpoint.get("expected_world_size", -1)) != world_size:
        _fail("preemption checkpoint world size differs from the locked topology")
    if checkpoint["identity"].get("study_manifest_sha256") != recipe_sha:
        _fail("preemption checkpoint recipe hash mismatch")
    preemption_metadata = None
    meta_path = (
        base_dir
        / "base_checkpoints"
        / selected_tag
        / f"strict_{checkpoint_step:06d}"
        / "meta.json"
    )
    meta = _load_object(meta_path, "preemption checkpoint metadata")
    if not used_stage_source:
        value = meta.get("preemption")
        if value is None:
            _fail("new preemption checkpoint lacks signal/exit metadata")
        preemption_metadata = dict(_mapping(value, "checkpoint preemption metadata"))
        if preemption_metadata.get("exit_code") != 75 or preemption_metadata.get(
            "signal"
        ) not in {"SIGUSR1", "SIGTERM"}:
            _fail("checkpoint preemption metadata is invalid")
        protocol = _mapping(
            checkpoint["identity"].get("protocol"), "preemption checkpoint protocol"
        )
        exposure_plan_key = (
            f"{stage['exposure_plan_family']}_ws{world_size}_seed42"
        )
        _verify_frozen_protocol(
            protocol,
            recipe=recipe,
            preflight=preflight,
            attention_probe=attention_probe,
            attention_probe_sha256=attention_probe_sha,
            label="preemption checkpoint protocol",
            run_kind="production",
            recipe_scope=(
                "production_trunk" if stage["kind"] == "trunk" else str(stage["id"])
            ),
            model_tag=str(stage["model_tag"]),
            exposure_plan_sha256=str(
                preflight["corpus"]["training_exposure_plans"][exposure_plan_key][
                    "sha256"
                ]
            ),
            depth=32,
            model_dim=2048,
            world_size=world_size,
            device_batch_size=recipe["training"]["device_batch_sequences"],
            total_batch_size=recipe["training"]["global_batch_tokens"],
            num_iterations=int(stage["num_iterations"]),
            eval_every_updates=recipe["training"]["evaluation"][
                "eval_every_updates"
            ],
            seed=recipe["training"]["seed"],
            production_gate=gate,
            production_gate_sha256=gate_sha,
        )
    receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "d32_collective_preemption_receipt",
            "family_id": recipe["family_id"],
            "recipe_sha256": recipe_sha,
            "preflight_receipt_sha256": preflight_sha,
            "production_gate_sha256": gate_sha,
            "wsd_proxy_approval_sha256": approval_sha,
            "stage_id": stage["id"],
            "run_id": run_id,
            "slurm_job_id": args.slurm_job_id,
            "slurm_restart_count": args.slurm_restart_count,
            "world_size": world_size,
            "nodes": nodes,
            "rank_exit_code": 75,
            "static_launch_receipt_sha256": launch_sha,
            "resume_checkpoint": {
                "model_tag": selected_tag,
                "step": checkpoint_step,
                "checkpoint_sha256": checkpoint_sha,
                "is_declared_stage_source": used_stage_source,
                "preemption_metadata": preemption_metadata,
            },
            "requeue_authorized": True,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))


def command_seal_stage(args: argparse.Namespace) -> None:
    recipe, recipe_sha = load_recipe(args.recipe)
    preflight, preflight_sha, _gate, gate_sha, approval, approval_sha = _verify_gate_and_preflight(
        recipe, recipe_sha, args.preflight_receipt, args.gate, args.wd_proxy_approval,
        args.attention_probe,
    )
    production_world_size = int(_gate["authorized_production_world_size"])
    _signal_gate, signal_gate_sha = _verify_signal_resume_gate(
        args.signal_resume_gate,
        recipe=recipe,
        recipe_sha=recipe_sha,
        preflight=preflight,
        preflight_sha=preflight_sha,
    )
    if _gate.get("signal_resume_gate_sha256") != signal_gate_sha:
        _fail("stage finalization uses a different signal/resume gate")
    attention_probe, attention_probe_sha = _verify_attention_probe(
        args.attention_probe,
        recipe=recipe,
        recipe_sha=recipe_sha,
        preflight_sha=preflight_sha,
        code_revision=preflight["code"]["git_commit"],
    )
    if _gate.get("attention_probe_sha256") != attention_probe_sha:
        _fail("stage finalization uses a different attention probe than the topology gate")
    stage = _stage_by_id(recipe, args.stage)
    production_nodes = int(_gate["authorized_production_nodes"])
    expected_run_id = (
        f"{recipe['family_id']}_trunk"
        if stage["kind"] == "trunk"
        else f"{recipe['family_id']}_{stage['id']}"
    )
    slurm_completion = _live_slurm_completed_job(
        REPO_ROOT, job_id=args.slurm_job_id, expected_nodes=production_nodes
    )
    _launch, launch_sha = _verify_static_launch_receipt(
        args.launch_receipt,
        recipe=recipe,
        recipe_sha=recipe_sha,
        run_id=expected_run_id,
        phase=f"production_{stage['id']}",
        slurm_job_id=args.slurm_job_id,
        world_size=production_world_size,
        nodes=production_nodes,
    )
    base_dir = args.base_dir.expanduser().resolve()
    target_manifest, target_sha = _checkpoint_manifest(
        base_dir, str(stage["model_tag"]), int(stage["target_step"])
    )
    source: dict[str, Any] | None = None
    if stage.get("source_step") is not None:
        source_tag = str(stage.get("source_model_tag", stage["model_tag"]))
        _source_manifest, source_sha = _checkpoint_manifest(
            base_dir, source_tag, int(stage["source_step"])
        )
        source = {
            "model_tag": source_tag,
            "step": int(stage["source_step"]),
            "checkpoint_sha256": source_sha,
            "run_id": _mapping(
                _source_manifest.get("identity"), "source checkpoint identity"
            ).get("run_id"),
        }
    identity = _mapping(target_manifest.get("identity"), "target checkpoint identity")
    if identity.get("tokenizer_artifact_sha256") != preflight["tokenizer"]["package_manifest_sha256"]:
        _fail("target checkpoint tokenizer hash differs from preflight")
    if identity.get("study_manifest_sha256") != recipe_sha:
        _fail("target checkpoint family-recipe hash differs from preflight")
    if int(target_manifest.get("expected_world_size", -1)) != production_world_size:
        _fail("target production checkpoint world size differs from the topology gate")
    if identity.get("run_id") != expected_run_id:
        _fail("target checkpoint run identity differs from the stage contract")
    exposure_plan_key = (
        f"{stage['exposure_plan_family']}_ws{production_world_size}_seed42"
    )
    expected_exposure = preflight["corpus"]["training_exposure_plans"][
        exposure_plan_key
    ]["sha256"]
    if identity.get("exposure_plan_sha256") != expected_exposure:
        _fail("target checkpoint exposure-plan hash differs from the stage contract")
    protocol = _mapping(identity.get("protocol"), "target checkpoint protocol")
    if protocol.get("num_iterations") != int(stage["num_iterations"]):
        _fail("target checkpoint training horizon differs from stage contract")
    optimizer = _verify_frozen_protocol(
        protocol,
        recipe=recipe,
        preflight=preflight,
        attention_probe=attention_probe,
        attention_probe_sha256=attention_probe_sha,
        label="target checkpoint protocol",
        run_kind="production",
        recipe_scope=(
            "production_trunk" if stage["kind"] == "trunk" else str(stage["id"])
        ),
        model_tag=str(stage["model_tag"]),
        exposure_plan_sha256=str(expected_exposure),
        depth=32,
        model_dim=2048,
        world_size=production_world_size,
        device_batch_size=recipe["training"]["device_batch_sequences"],
        total_batch_size=recipe["training"]["global_batch_tokens"],
        num_iterations=int(stage["num_iterations"]),
        eval_every_updates=recipe["training"]["evaluation"]["eval_every_updates"],
        seed=recipe["training"]["seed"],
        production_gate=_gate,
        production_gate_sha256=gate_sha,
    )
    if optimizer.get("muon_base_weight_decay") != approval["accepted_base_weight_decay"]:
        _fail("target checkpoint WSD base weight decay differs from proxy approval")
    schedule = _mapping(protocol.get("schedule"), "target checkpoint schedule")
    expected_cooldown = (
        None if int(stage["cooldown_start_step"]) < 0 else int(stage["cooldown_start_step"])
    )
    if schedule.get("name") != "wsd" or schedule.get("cooldown_start_step") != expected_cooldown:
        _fail("target checkpoint WSD phase differs from stage contract")
    if schedule.get("recipe_version") != recipe["weight_decay_proxy_ablation"]["recipe_version"]:
        _fail("target checkpoint WSD recipe version differs from family contract")
    if schedule.get("proxy_approval_sha256") != approval_sha:
        _fail("target checkpoint does not bind the accepted WSD proxy receipt")
    if schedule.get("warmup_steps") != 40 or schedule.get("momentum_warmup_steps") != 400:
        _fail("target checkpoint WSD warmup policy mismatch")
    if schedule.get("weight_decay_cooldown_policy") != approval[
        "accepted_weight_decay_cooldown_policy"
    ]:
        _fail("target checkpoint cooldown policy differs from proxy approval")
    if schedule.get("stable_muon_weight_decay") != approval[
        "accepted_base_weight_decay"
    ]:
        _fail("target checkpoint schedule base weight decay differs from approval")
    if schedule.get("momentum") != {"initial": 0.85, "stable": 0.97, "final": 0.9}:
        _fail("target checkpoint WSD momentum policy mismatch")
    expected_cooldown_fraction = None if expected_cooldown is None else 0.1
    expected_terminal_lr = 1.0 if expected_cooldown is None else 0.0
    if (
        schedule.get("cooldown_fraction") != expected_cooldown_fraction
        or schedule.get("terminal_lr_multiplier") != expected_terminal_lr
    ):
        _fail("target checkpoint WSD cooldown/terminal-LR policy mismatch")
    checkpoint_parent = protocol.get("parent")
    if source is None:
        if checkpoint_parent is not None:
            _fail("fresh trunk checkpoint unexpectedly records a parent")
    elif stage["kind"] == "cooldown_fork":
        expected_parent = {
            "checkpoint_sha256": source["checkpoint_sha256"],
            "run_id": source["run_id"],
            "step": source["step"],
        }
        if checkpoint_parent != expected_parent:
            _fail("cooldown checkpoint parent lineage differs from stable fork")
    receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "d32_checkpoint_lineage_receipt",
            "family_id": recipe["family_id"],
            "stage_id": stage["id"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "recipe_sha256": recipe_sha,
            "preflight_receipt_sha256": preflight_sha,
            "production_gate_sha256": gate_sha,
            "production_world_size": production_world_size,
            "exposure_plan_key": exposure_plan_key,
            "wsd_proxy_approval_sha256": approval_sha,
            "wsd_base_weight_decay": approval["accepted_base_weight_decay"],
            "wsd_weight_decay_cooldown_policy": approval[
                "accepted_weight_decay_cooldown_policy"
            ],
            "source": source,
            "target": {
                "model_tag": stage["model_tag"],
                "step": int(stage["target_step"]),
                "checkpoint_sha256": target_sha,
                "retention_class": (
                    "full_resumable_stable_fork"
                    if stage["kind"] == "trunk"
                    else (
                        "cooled_final_full_resumable_retained"
                        if recipe["family_id"] == FAMILY_ID_V3
                        else "cooled_final_full_transaction_pending_explicit_export_policy"
                    )
                ),
            },
            "slurm_job_id": args.slurm_job_id,
            "slurm_completion": slurm_completion,
            "static_srun_launch_receipt_sha256": launch_sha,
            "signal_resume_gate_sha256": signal_gate_sha,
            "code_revision": preflight["code"]["git_commit"],
            "canonical_sha256": None,
        }
    )
    write_json_atomic(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-recipe")
    validate.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    validate.add_argument("--allow-unsealed", action="store_true")
    validate.set_defaults(func=command_validate_recipe)

    quality_approval = subparsers.add_parser("seal-mixture-quality-approval")
    quality_approval.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY,
    )
    quality_approval.add_argument("--source-plan", type=Path, required=True)
    quality_approval.add_argument("--calibration", type=Path, required=True)
    quality_approval.add_argument("--audit-report", type=Path, required=True)
    quality_approval.add_argument("--reviewer", required=True)
    quality_approval.add_argument("--reviewed-at-utc", required=True)
    quality_approval.add_argument(
        "--decision", choices=("accepted", "rejected"), required=True
    )
    quality_approval.add_argument("--notes", default="")
    quality_approval.add_argument("--output", type=Path, required=True)
    quality_approval.set_defaults(func=command_seal_mixture_quality_approval)

    pack_plan = subparsers.add_parser("seal-data-prep-pack-plan")
    pack_plan.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    pack_plan.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY,
    )
    pack_plan.add_argument("--source-plan", type=Path, required=True)
    pack_plan.add_argument("--calibration", type=Path, required=True)
    pack_plan.add_argument("--nodes", type=int, required=True)
    pack_plan.add_argument("--output", type=Path, required=True)
    pack_plan.set_defaults(func=command_seal_data_prep_pack_plan)

    writer_probe = subparsers.add_parser("seal-data-prep-writer-probe")
    writer_probe.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    writer_probe.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY,
    )
    writer_probe.add_argument("--source-plan", type=Path, required=True)
    writer_probe.add_argument("--calibration", type=Path, required=True)
    writer_probe.add_argument("--sample-run-dir", type=Path, required=True)
    writer_probe.add_argument("--backend-resource-report", type=Path, required=True)
    writer_probe.add_argument("--scratch-dir", type=Path, required=True)
    writer_probe.add_argument("--output", type=Path, required=True)
    writer_probe.set_defaults(func=command_seal_data_prep_writer_probe)

    data_prep_sample = subparsers.add_parser("seal-data-prep-storage-sample")
    data_prep_sample.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    data_prep_sample.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY,
    )
    data_prep_sample.add_argument("--source-plan", type=Path, required=True)
    data_prep_sample.add_argument("--calibration", type=Path, required=True)
    data_prep_sample.add_argument("--sample-ranks", type=Path, required=True)
    data_prep_sample.add_argument("--sample-lane-plan", type=Path, required=True)
    data_prep_sample.add_argument("--sample-run-dir", type=Path, required=True)
    data_prep_sample.add_argument(
        "--sample-object-launch-receipt", type=Path, required=True
    )
    data_prep_sample.add_argument(
        "--sample-bucket-launch-receipt", type=Path, required=True
    )
    data_prep_sample.add_argument(
        "--backend-resource-report", type=Path, required=True
    )
    data_prep_sample.add_argument("--resource-approval", type=Path, required=True)
    data_prep_sample.add_argument(
        "--mixture-quality-approval", type=Path, required=True
    )
    data_prep_sample.add_argument("--macocu-manifest", type=Path, required=True)
    data_prep_sample.add_argument("--production-pack-plan", type=Path, required=True)
    data_prep_sample.add_argument("--writer-probe", type=Path, required=True)
    data_prep_sample.add_argument("--macocu-job-id", required=True)
    data_prep_sample.add_argument("--bootstrap-job-id", required=True)
    data_prep_sample.add_argument("--sample-object-job-id", required=True)
    data_prep_sample.add_argument("--sample-bucket-job-id", required=True)
    data_prep_sample.add_argument("--sample-cluster-job-id", required=True)
    data_prep_sample.add_argument("--sample-quality-audit-job-id", required=True)
    data_prep_sample.add_argument("--writer-probe-job-id", required=True)
    data_prep_sample.add_argument("--output", type=Path, required=True)
    data_prep_sample.set_defaults(func=command_seal_data_prep_storage_sample)

    data_prep = subparsers.add_parser("data-prep-storage-gate")
    data_prep.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    data_prep.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY,
    )
    data_prep.add_argument("--sample-measurement", type=Path, required=True)
    data_prep.add_argument("--work-dir", type=Path, required=True)
    data_prep.add_argument("--output", type=Path, required=True)
    data_prep.set_defaults(func=command_data_prep_storage_gate)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    preflight.add_argument("--repo-root", type=Path, default=Path.cwd())
    preflight.add_argument("--base-dir", type=Path, required=True)
    preflight.add_argument("--data-prep-storage-gate", type=Path, required=True)
    preflight.add_argument("--cluster-launch-receipt", type=Path, required=True)
    preflight.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY,
    )
    preflight.add_argument("--source-plan", type=Path, required=True)
    preflight.add_argument("--calibration", type=Path, required=True)
    preflight.add_argument("--resource-approval", type=Path, required=True)
    preflight.add_argument(
        "--mixture-quality-approval", type=Path, required=True
    )
    preflight.add_argument("--corpus-manifest-sha256", required=True)
    preflight.add_argument("--dataset-manifest-sha256", required=True)
    preflight.add_argument("--source-receipt-sha256", required=True)
    preflight.add_argument("--tokenizer-package-sha256", required=True)
    preflight.add_argument("--validation-exposure-sha256", required=True)
    preflight.add_argument("--exposure-plan-index-sha256", required=True)
    preflight.add_argument("--output", type=Path, required=True)
    preflight.add_argument("--allow-dirty", action="store_true")
    preflight.set_defaults(func=command_preflight)

    proxy_run = subparsers.add_parser("seal-proxy-run")
    proxy_run.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    proxy_run.add_argument("--preflight-receipt", type=Path, required=True)
    proxy_run.add_argument("--attention-probe", type=Path, required=True)
    proxy_run.add_argument("--model-depth", type=int, choices=(12, 20), required=True)
    proxy_run.add_argument("--candidate-id", required=True)
    proxy_run.add_argument("--seed", type=int, required=True)
    proxy_run.add_argument("--curve-log", type=Path, required=True)
    proxy_run.add_argument("--checkpoint-root", type=Path, required=True)
    proxy_run.add_argument("--rank-exit-receipt", type=Path, required=True)
    proxy_run.add_argument("--slurm-job-id", default="")
    proxy_run.add_argument("--output", type=Path, required=True)
    proxy_run.set_defaults(func=command_seal_proxy_run)

    proxy_env = subparsers.add_parser("proxy-env")
    proxy_env.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    proxy_env.add_argument("--model-depth", type=int, choices=(12, 20), required=True)
    proxy_env.add_argument("--candidate-id", required=True)
    proxy_env.add_argument("--seed", type=int, required=True)
    proxy_env.set_defaults(func=command_proxy_env)

    proxy_screen = subparsers.add_parser("screen-proxy")
    proxy_screen.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    proxy_screen.add_argument("--preflight-receipt", type=Path, required=True)
    proxy_screen.add_argument("--attention-probe", type=Path, required=True)
    proxy_screen.add_argument(
        "--run-receipt", type=Path, action="append", required=True
    )
    proxy_screen.add_argument("--output", type=Path, required=True)
    proxy_screen.set_defaults(func=command_screen_proxy)

    proxy_accept = subparsers.add_parser("accept-proxy")
    proxy_accept.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    proxy_accept.add_argument("--preflight-receipt", type=Path, required=True)
    proxy_accept.add_argument("--attention-probe", type=Path, required=True)
    proxy_accept.add_argument("--screening-receipt", type=Path, required=True)
    proxy_accept.add_argument(
        "--run-receipt", type=Path, action="append", required=True
    )
    proxy_accept.add_argument("--output", type=Path, required=True)
    proxy_accept.set_defaults(func=command_accept_proxy)

    smoke = subparsers.add_parser("seal-smoke")
    smoke.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    smoke.add_argument("--preflight-receipt", type=Path, required=True)
    smoke.add_argument("--attention-probe", type=Path, required=True)
    smoke.add_argument("--nodes", type=int, choices=(2, 4), required=True)
    smoke.add_argument("--curve-log", type=Path, required=True)
    smoke.add_argument("--checkpoint-root", type=Path, required=True)
    smoke.add_argument("--static-probe-receipt", type=Path, required=True)
    smoke.add_argument("--static-launcher-gate", type=Path, required=True)
    smoke.add_argument("--signal-resume-gate", type=Path, required=True)
    smoke.add_argument("--launch-receipt", type=Path, action="append", required=True)
    smoke.add_argument("--wd-proxy-approval", type=Path, required=True)
    smoke.add_argument("--slurm-job-id", default="")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.set_defaults(func=command_seal_smoke)

    optimizer_env = subparsers.add_parser("optimizer-env")
    optimizer_env.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    optimizer_env.add_argument("--preflight-receipt", type=Path, required=True)
    optimizer_env.add_argument("--attention-probe", type=Path, required=True)
    optimizer_env.add_argument("--wd-proxy-approval", type=Path, required=True)
    optimizer_env.set_defaults(func=command_optimizer_env)

    attention_env = subparsers.add_parser("attention-env")
    attention_env.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    attention_env.add_argument("--preflight-receipt", type=Path, required=True)
    attention_env.add_argument("--attention-probe", type=Path, required=True)
    attention_env.set_defaults(func=command_attention_env)

    rank_exit = subparsers.add_parser("record-rank-exit")
    rank_exit.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    rank_exit.add_argument("--run-id", required=True)
    rank_exit.add_argument("--phase", required=True)
    rank_exit.add_argument("--slurm-job-id", required=True)
    rank_exit.add_argument("--slurm-step-id", required=True)
    rank_exit.add_argument("--node", required=True)
    rank_exit.add_argument("--rank", type=int, required=True)
    rank_exit.add_argument("--local-rank", type=int, required=True)
    rank_exit.add_argument("--world-size", type=int, required=True)
    rank_exit.add_argument("--exit-code", type=int, required=True)
    rank_exit.add_argument(
        "--launcher",
        choices=(
            "slurm_srun_direct_python_env_v1",
            "slurm_batch_direct_python_env_v1",
        ),
        required=True,
    )
    rank_exit.add_argument("--output", type=Path, required=True)
    rank_exit.set_defaults(func=command_record_rank_exit)

    static_launch = subparsers.add_parser("seal-static-launch")
    static_launch.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    static_launch.add_argument("--receipt-dir", type=Path, required=True)
    static_launch.add_argument("--run-id", required=True)
    static_launch.add_argument("--phase", required=True)
    static_launch.add_argument("--slurm-job-id", required=True)
    static_launch.add_argument("--world-size", type=int, required=True)
    static_launch.add_argument("--nodes", type=int, required=True)
    static_launch.add_argument("--srun-exit-code", type=int, required=True)
    static_launch.add_argument("--expected-exit-code", type=int, choices=(0, 75), required=True)
    static_launch.add_argument("--output", type=Path, required=True)
    static_launch.set_defaults(func=command_seal_static_launch)

    static_probe = subparsers.add_parser("seal-static-probe")
    static_probe.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    static_probe.add_argument("--receipt-dir", type=Path, required=True)
    static_probe.add_argument("--run-id", required=True)
    static_probe.add_argument("--phase", required=True)
    static_probe.add_argument("--slurm-job-id", required=True)
    static_probe.add_argument("--world-size", type=int, required=True)
    static_probe.add_argument("--nodes", type=int, required=True)
    static_probe.add_argument("--srun-exit-code", type=int, required=True)
    static_probe.add_argument("--output", type=Path, required=True)
    static_probe.set_defaults(func=command_seal_static_probe)

    finalize_static_probe = subparsers.add_parser("finalize-static-probe")
    finalize_static_probe.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    finalize_static_probe.add_argument("--preflight-receipt", type=Path, required=True)
    finalize_static_probe.add_argument("--raw-probe-receipt", type=Path, required=True)
    finalize_static_probe.add_argument("--slurm-job-id", required=True)
    finalize_static_probe.add_argument("--nodes", type=int, choices=(1,), required=True)
    finalize_static_probe.add_argument("--output", type=Path, required=True)
    finalize_static_probe.set_defaults(func=command_finalize_static_probe)

    verify_static_probe = subparsers.add_parser("verify-static-launcher-gate")
    verify_static_probe.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    verify_static_probe.add_argument("--preflight-receipt", type=Path, required=True)
    verify_static_probe.add_argument("--gate", type=Path, required=True)
    verify_static_probe.set_defaults(func=command_verify_static_launcher_gate)

    signal_env = subparsers.add_parser("signal-smoke-env")
    signal_env.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    signal_env.add_argument("--base-dir", type=Path, required=True)
    signal_env.set_defaults(func=command_signal_smoke_env)

    signal_request = subparsers.add_parser("record-signal-request")
    signal_request.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    signal_request.add_argument("--curve-log", type=Path, required=True)
    signal_request.add_argument("--slurm-job-id", required=True)
    signal_request.add_argument("--slurm-step-id", required=True)
    signal_request.add_argument("--slurm-restart-count", type=int, required=True)
    signal_request.add_argument("--output", type=Path, required=True)
    signal_request.set_defaults(func=command_record_signal_request)

    signal_preemption = subparsers.add_parser("seal-signal-preemption")
    signal_preemption.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    signal_preemption.add_argument("--base-dir", type=Path, required=True)
    signal_preemption.add_argument("--signal-request", type=Path, required=True)
    signal_preemption.add_argument("--launch-receipt", type=Path, required=True)
    signal_preemption.add_argument("--slurm-job-id", required=True)
    signal_preemption.add_argument("--output", type=Path, required=True)
    signal_preemption.set_defaults(func=command_seal_signal_preemption)

    finalize_signal = subparsers.add_parser("finalize-signal-smoke")
    finalize_signal.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    finalize_signal.add_argument("--preflight-receipt", type=Path, required=True)
    finalize_signal.add_argument("--static-launcher-gate", type=Path, required=True)
    finalize_signal.add_argument("--attention-probe", type=Path, required=True)
    finalize_signal.add_argument("--wd-proxy-approval", type=Path, required=True)
    finalize_signal.add_argument("--base-dir", type=Path, required=True)
    finalize_signal.add_argument("--preemption-receipt", type=Path, required=True)
    finalize_signal.add_argument("--final-launch-receipt", type=Path, required=True)
    finalize_signal.add_argument("--slurm-job-id", required=True)
    finalize_signal.add_argument("--output", type=Path, required=True)
    finalize_signal.set_defaults(func=command_finalize_signal_smoke)

    verify_signal = subparsers.add_parser("verify-signal-resume-gate")
    verify_signal.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    verify_signal.add_argument("--preflight-receipt", type=Path, required=True)
    verify_signal.add_argument("--gate", type=Path, required=True)
    verify_signal.set_defaults(func=command_verify_signal_resume_gate)

    compare = subparsers.add_parser("compare-smokes")
    compare.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    compare.add_argument("--smoke-8gpu", type=Path, required=True)
    compare.add_argument("--smoke-16gpu", type=Path)
    compare.add_argument("--output", type=Path, required=True)
    compare.set_defaults(func=command_compare_smokes)

    stage_env = subparsers.add_parser("stage-env")
    stage_env.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    stage_env.add_argument("--stage", required=True)
    stage_env.add_argument("--base-dir", type=Path, required=True)
    stage_env.add_argument("--gate", type=Path, required=True)
    stage_env.set_defaults(func=command_stage_env)

    preemption = subparsers.add_parser("seal-preemption")
    preemption.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    preemption.add_argument("--preflight-receipt", type=Path, required=True)
    preemption.add_argument("--gate", type=Path, required=True)
    preemption.add_argument("--wd-proxy-approval", type=Path, required=True)
    preemption.add_argument("--attention-probe", type=Path, required=True)
    preemption.add_argument("--signal-resume-gate", type=Path, required=True)
    preemption.add_argument("--base-dir", type=Path, required=True)
    preemption.add_argument("--stage", required=True)
    preemption.add_argument("--launch-receipt", type=Path, required=True)
    preemption.add_argument("--slurm-job-id", required=True)
    preemption.add_argument("--slurm-restart-count", type=int, required=True)
    preemption.add_argument("--output", type=Path, required=True)
    preemption.set_defaults(func=command_seal_preemption)

    preflight_stage = subparsers.add_parser("preflight-stage")
    preflight_stage.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    preflight_stage.add_argument("--preflight-receipt", type=Path, required=True)
    preflight_stage.add_argument("--gate", type=Path, required=True)
    preflight_stage.add_argument("--wd-proxy-approval", type=Path, required=True)
    preflight_stage.add_argument("--attention-probe", type=Path, required=True)
    preflight_stage.add_argument("--signal-resume-gate", type=Path, required=True)
    preflight_stage.add_argument("--base-dir", type=Path, required=True)
    preflight_stage.add_argument("--lineage-dir", type=Path, required=True)
    preflight_stage.add_argument("--stage", required=True)
    preflight_stage.add_argument("--world-size", type=int, required=True)
    preflight_stage.set_defaults(func=command_preflight_stage)

    seal_stage = subparsers.add_parser("seal-stage")
    seal_stage.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    seal_stage.add_argument("--preflight-receipt", type=Path, required=True)
    seal_stage.add_argument("--gate", type=Path, required=True)
    seal_stage.add_argument("--wd-proxy-approval", type=Path, required=True)
    seal_stage.add_argument("--attention-probe", type=Path, required=True)
    seal_stage.add_argument("--signal-resume-gate", type=Path, required=True)
    seal_stage.add_argument("--base-dir", type=Path, required=True)
    seal_stage.add_argument("--stage", required=True)
    seal_stage.add_argument("--slurm-job-id", default="")
    seal_stage.add_argument("--launch-receipt", type=Path, required=True)
    seal_stage.add_argument("--output", type=Path, required=True)
    seal_stage.set_defaults(func=command_seal_stage)

    final_quality = subparsers.add_parser("seal-final-quality-approval")
    final_quality.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    final_quality.add_argument("--preflight-receipt", type=Path, required=True)
    final_quality.add_argument("--gate", type=Path, required=True)
    final_quality.add_argument("--base-dir", type=Path, required=True)
    final_quality.add_argument("--lineage-dir", type=Path, required=True)
    final_quality.add_argument("--reviewer", required=True)
    final_quality.add_argument("--reviewed-at-utc", required=True)
    final_quality.add_argument(
        "--decision", choices=("accepted", "rejected"), required=True
    )
    final_quality.add_argument("--notes", default="")
    final_quality.add_argument("--output", type=Path, required=True)
    final_quality.set_defaults(func=command_seal_final_model_publication_approval)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FamilyWorkflowError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
