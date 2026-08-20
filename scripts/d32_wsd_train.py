"""Dedicated strict Muon/WSD trainer for the Turkish d32 model family.

This is intentionally separate from ``scripts.base_train``.  It keeps the
pinned upstream model and optimizer unchanged while adding only the production
contracts needed for the d12/d20 proxy studies and the shared d32 trunk with
s12/s20/s40 cooldown forks.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import signal
import sys
import time
import weakref
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist
import wandb

import nanochat.flash_attention as attention_runtime
from nanochat.common import (
    COMPUTE_DTYPE,
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    get_peak_flops,
    is_ddp_initialized,
    print0,
    print_banner,
)
from nanochat.experiment_manifest import (
    canonical_json_bytes,
    load_json_strict,
    verify_manifest_hash,
)
from nanochat.gpt import GPT, GPTConfig
from nanochat.strict_checkpoint import (
    build_strict_checkpoint_identity,
    capture_rank_rng_state,
    inspect_strict_checkpoint,
    load_strict_checkpoint,
    restore_rank_rng_state,
    save_strict_checkpoint,
)
from nanochat.strict_dataloader import (
    StatefulBestFitLoader,
    StatefulSequentialDocumentLoader,
    build_validation_rows,
    verify_strict_dataset,
)
from nanochat.strict_eval import evaluate_loss
from nanochat.strict_runtime import (
    PREEMPTION_EXIT_CODE,
    StrictTrainingError,
    derive_seed_plan,
    load_artifact_bindings,
    nanochat_effective_weight_decay,
    validate_attention_probe_receipt,
    validate_bestfit_capacity_receipt,
    validate_preflight_artifact_bindings,
    validate_proxy_approval,
    validate_production_topology_gate,
    validate_recipe_invocation,
    verify_code_provenance,
    verify_live_fa3_kernel_inventory,
)
from nanochat.strict_tokenizer import (
    TokenizerPackageError,
    load_verified_token_bytes,
    load_verified_tokenizer,
    verify_tokenizer_package,
)
from nanochat.training_log import (
    CanonicalTrainingLog,
    checkpoint_curve_log_state,
    reconcile_training_log_to_checkpoint,
)
from nanochat.wsd import (
    nanochat_linear_schedule_values,
    validate_wsd_schedule,
    wsd_schedule_values,
)


MODEL_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
pending_preemption_signal = 0


def _record_preemption_signal(signum, _frame) -> None:
    global pending_preemption_signal
    pending_preemption_signal = int(signum)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-mode", action="store_true")
    parser.add_argument(
        "--strict-run-kind",
        choices=("production", "proxy", "smoke", "signal_smoke"),
        default="production",
    )
    parser.add_argument("--run", default="dummy")
    parser.add_argument("--wandb-every", type=int, default=100)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--study-manifest", required=True)
    parser.add_argument("--preflight-receipt", required=True)
    parser.add_argument("--attention-probe", required=True)
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--tokenizer-manifest", required=True)
    parser.add_argument("--exposure-plan", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--packing-capacity-receipt", default="")
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device-type", default="")
    parser.add_argument(
        "--attention-backend", choices=("sdpa", "fa3"), required=True
    )
    parser.add_argument(
        "--fp8", action=argparse.BooleanOptionalAction, default=False
    )

    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--aspect-ratio", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--window-pattern", default="L")

    parser.add_argument("--num-iterations", type=int, required=True)
    parser.add_argument("--target-flops", type=float, default=-1.0)
    parser.add_argument("--target-param-data-ratio", type=float, default=-1.0)
    parser.add_argument(
        "--target-param-count", choices=("scaling",), default="scaling"
    )
    parser.add_argument(
        "--horizon-unit", choices=("token_positions",), default="token_positions"
    )
    parser.add_argument("--horizon-value", type=int, required=True)
    parser.add_argument("--data-order", choices=("bestfit",), default="bestfit")

    parser.add_argument("--device-batch-size", type=int, required=True)
    parser.add_argument("--total-batch-size", type=int, required=True)
    parser.add_argument("--embedding-lr", type=float, default=0.3)
    parser.add_argument("--unembedding-lr", type=float, default=0.008)
    parser.add_argument("--matrix-lr", type=float, default=0.02)
    parser.add_argument("--scalar-lr", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.28)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--warmdown-ratio", type=float, default=0.65)
    parser.add_argument("--final-lr-frac", type=float, default=0.05)
    parser.add_argument("--optimizer", choices=("muon_adamw",), default="muon_adamw")
    parser.add_argument(
        "--lr-schedule", choices=("nanochat_linear", "wsd"), required=True
    )
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--wsd-recipe-version", default="")
    parser.add_argument("--wsd-base-weight-decay", type=float, default=-1.0)
    parser.add_argument(
        "--wsd-weight-decay-cooldown",
        choices=("", "constant", "linear_to_zero"),
        default="",
    )
    parser.add_argument("--wsd-proxy-approval", default="")
    parser.add_argument("--production-gate", default="")
    parser.add_argument("--wsd-cooldown-start-step", type=int, default=-1)
    parser.add_argument("--wsd-cooldown-fraction", type=float, default=0.10)

    parser.add_argument("--eval-every", type=int, required=True)
    parser.add_argument(
        "--eval-tokens",
        type=int,
        default=-1,
        help="Unused compatibility sentinel; strict evaluation exhausts the frozen manifest.",
    )
    parser.add_argument("--core-metric-every", type=int, default=-1)
    parser.add_argument("--core-metric-max-per-task", type=int, default=500)
    parser.add_argument("--sample-every", type=int, default=-1)
    parser.add_argument("--save-every", type=int, default=-1)

    parser.add_argument("--stop-at-step", type=int, required=True)
    parser.add_argument("--resume-from-step", type=int, default=-1)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--parent-checkpoint-dir", default="")
    parser.add_argument("--parent-checkpoint-step", type=int, default=-1)
    parser.add_argument("--parent-checkpoint-sha256", default="")
    return parser


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    values = list(sys.argv[1:] if argv is None else argv)
    # Tolerate the separator used by some torchrun wrappers.  Launchers should
    # normally omit it because this module is already the training entry point.
    if values[:1] == ["--"]:
        values = values[1:]
    return build_parser().parse_args(values)


def validate_cli(args: argparse.Namespace) -> None:
    if not args.strict_mode:
        raise StrictTrainingError("the dedicated trainer requires --strict-mode")
    if args.fp8:
        raise StrictTrainingError("the d32 family is BF16-only; FP8 is forbidden")
    if args.optimizer != "muon_adamw" or args.grad_clip != 0.0:
        raise StrictTrainingError("the d32 family requires upstream MuonAdamW and grad clip 0")
    if args.target_flops > 0 or args.target_param_data_ratio != -1.0:
        raise StrictTrainingError("the strict horizon must be supplied explicitly")
    if args.horizon_value != args.num_iterations * args.total_batch_size:
        raise StrictTrainingError("horizon-value must equal iterations times global batch")
    if min(
        args.depth,
        args.aspect_ratio,
        args.head_dim,
        args.max_seq_len,
        args.num_iterations,
        args.device_batch_size,
        args.total_batch_size,
        args.eval_every,
        args.stop_at_step,
    ) <= 0:
        raise StrictTrainingError("training dimensions and horizons must be positive")
    if args.eval_tokens != -1:
        raise StrictTrainingError(
            "strict evaluation exhausts the frozen manifest; --eval-tokens must be -1"
        )
    if args.stop_at_step > args.num_iterations:
        raise StrictTrainingError("stop-at-step exceeds the schedule horizon")
    if args.resume_from_step >= 0 and not args.resume_from_step < args.stop_at_step:
        raise StrictTrainingError("resume step must precede stop-at-step")
    if args.save_every == 0 or args.save_every < -1:
        raise StrictTrainingError("save-every must be -1 or positive")
    if args.core_metric_every != -1 or args.sample_every != -1:
        raise StrictTrainingError("CORE and sampling are excluded from production training")
    if args.data_order != "bestfit":
        raise StrictTrainingError("the d32 lane requires upstream bestfit data packing")
    if (args.attention_backend, args.window_pattern) not in {
        ("fa3", "SSSL"),
        ("sdpa", "L"),
    }:
        raise StrictTrainingError("backend/window pair is outside the reviewed probe policy")
    if (args.aspect_ratio, args.head_dim, args.max_seq_len) != (64, 128, 2048):
        raise StrictTrainingError("architecture flags differ from the reviewed family")
    if (
        args.embedding_lr,
        args.unembedding_lr,
        args.matrix_lr,
        args.scalar_lr,
        args.warmup_steps,
    ) != (0.3, 0.008, 0.02, 0.5, 40):
        raise StrictTrainingError("optimizer base hyperparameters drifted")
    if args.warmdown_ratio != 0.65 or args.final_lr_frac != 0.05:
        raise StrictTrainingError("upstream-control schedule hyperparameters drifted")
    if args.wsd_cooldown_fraction != 0.10:
        raise StrictTrainingError("WSD cooldown fraction must be exactly 0.10")
    if args.save_every != -1:
        raise StrictTrainingError("reviewed family runs save only declared boundaries")
    for name, value in (
        ("model-tag", args.model_tag),
        ("study-id", args.study_id),
        ("run-id", args.run_id),
    ):
        if not MODEL_TAG_RE.fullmatch(value):
            raise StrictTrainingError(f"{name} must be a safe lowercase identifier")
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", args.code_revision):
        raise StrictTrainingError("code-revision must be a full lowercase Git object ID")
    parent_fields = (
        bool(args.parent_checkpoint_dir),
        args.parent_checkpoint_step >= 0,
        bool(args.parent_checkpoint_sha256),
    )
    if any(parent_fields) and not all(parent_fields):
        raise StrictTrainingError("all parent checkpoint fields must be supplied together")
    if all(parent_fields) and args.strict_run_kind != "production":
        raise StrictTrainingError("only production cooldowns may fork a parent")
    if args.strict_run_kind in {"production", "smoke", "signal_smoke"} and not args.wsd_proxy_approval:
        raise StrictTrainingError("d32 production and smokes require an accepted WSD proxy receipt")
    if args.strict_run_kind == "production" and not args.production_gate:
        raise StrictTrainingError("production requires a sealed topology gate")
    if args.strict_run_kind != "production" and args.production_gate:
        raise StrictTrainingError("only production may consume a topology gate")
    capacity_required = args.strict_run_kind in {"production", "smoke"}
    if capacity_required and not args.packing_capacity_receipt:
        raise StrictTrainingError(
            "production and distributed smoke require --packing-capacity-receipt"
        )
    if not capacity_required and args.packing_capacity_receipt:
        raise StrictTrainingError(
            "proxy and signal smoke must not select an unsupported capacity topology"
        )
    if args.lr_schedule == "wsd":
        if (
            not args.wsd_recipe_version
            or args.wsd_base_weight_decay < 0
            or not args.wsd_weight_decay_cooldown
        ):
            raise StrictTrainingError("WSD requires explicit recipe, base WD, and cooldown policy")
    elif args.strict_run_kind != "proxy":
        raise StrictTrainingError("only proxy controls may use nanochat_linear")


def _broadcast_object(value: Any, *, source: int = 0) -> Any:
    if not is_ddp_initialized():
        return value
    values = [value]
    dist.broadcast_object_list(values, src=source)
    return values[0]


def _collective_signal(device: torch.device) -> int:
    observed = torch.tensor(
        pending_preemption_signal, dtype=torch.int32, device=device
    )
    if is_ddp_initialized():
        dist.all_reduce(observed, op=dist.ReduceOp.MAX)
    return int(observed.item())


def _restore_and_release_checkpoint_payload(
    load_payload: Callable[[], Any],
    *,
    model: GPT,
    optimizer: Any,
    load_model: bool,
    device: torch.device,
) -> tuple[Any, Any, Any, dict[str, int]]:
    """Restore state while proving the duplicate model payload was released.

    ``assign=False`` preserves optimizer parameter identity and copies rank-0
    weights into the already materialized model.  The deserialized model state
    must then die before loader construction and ``torch.compile``; otherwise a
    full extra d32 state dict can remain resident on the rank-0 GPU.
    """

    payload = load_payload()
    payload_ref = weakref.ref(payload)
    model_tensor_refs: tuple[weakref.ReferenceType[torch.Tensor], ...] = ()
    model_payload_bytes = 0
    if load_model:
        if not isinstance(payload.model_data, Mapping):
            raise StrictTrainingError("strict checkpoint model payload is malformed")
        model_tensors = tuple(
            value for value in payload.model_data.values() if torch.is_tensor(value)
        )
        model_tensor_refs = tuple(weakref.ref(value) for value in model_tensors)
        model_payload_bytes = sum(
            int(value.numel() * value.element_size()) for value in model_tensors
        )
        model.load_state_dict(payload.model_data, strict=True, assign=False)
        del model_tensors
    optimizer.load_state_dict(payload.optimizer_data)
    meta_data = payload.meta_data
    loader_state = payload.loader_state
    rng_state = payload.rng_state
    model_tensor_count = len(model_tensor_refs)
    del payload
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if payload_ref() is not None:
        raise StrictTrainingError("strict checkpoint payload container remained live")
    if any(reference() is not None for reference in model_tensor_refs):
        raise StrictTrainingError(
            "deserialized checkpoint model tensors remained live after assign=False restore"
        )
    return meta_data, loader_state, rng_state, {
        "model_tensor_count_released": model_tensor_count,
        "model_tensor_bytes_released": model_payload_bytes,
        "payload_container_released": 1,
    }


def _scaling_parameters(model: GPT) -> int:
    counts = model.num_scaling_params()
    return counts["transformer_matrices"] + counts["lm_head"]


def _model_meta(args: argparse.Namespace, vocab_size: int, depth: int) -> GPT:
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=model_dim // args.head_dim,
        n_kv_head=model_dim // args.head_dim,
        n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
    with torch.device("meta"):
        return GPT(config)


def _optimizer_audit(optimizer, *, args, batch_lr_scale, base_weight_decay):
    groups = []
    for index, group in enumerate(optimizer.param_groups):
        record: dict[str, Any] = {"index": index}
        for key, value in sorted(group.items()):
            if key == "params":
                record["parameter_count"] = sum(parameter.numel() for parameter in value)
            elif isinstance(value, tuple):
                record[key] = list(value)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                record[key] = value
        groups.append(record)
    return {
        "policy": f"nanochat_muon_adamw_{args.lr_schedule}",
        "implementation": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
        "gradient_clip_norm": 0.0,
        "learning_rates": {
            "embedding": args.embedding_lr * batch_lr_scale,
            "unembedding": args.unembedding_lr * batch_lr_scale,
            "matrix": args.matrix_lr * batch_lr_scale,
            "scalar": args.scalar_lr * batch_lr_scale,
        },
        "muon_base_weight_decay": base_weight_decay,
        "parameter_group_count": len(groups),
        "parameter_groups": groups,
        "pytorch_ddp": False,
    }


def _validate_d32_group_lrs(optimizer, recipe: Mapping[str, Any]) -> None:
    expected = recipe["training"]["optimizer_hyperparameters"][
        "derived_initial_group_lrs"
    ]
    first = [group["initial_lr"] for group in optimizer.param_groups[:6]]
    named = (
        expected["lm_head"],
        expected["token_embedding"],
        expected["value_embeddings"],
        expected["residual_scalars"],
        expected["x0_scalars"],
        expected["smear"],
    )
    if len(first) != len(named) or any(
        not math.isclose(actual, target, rel_tol=1e-7, abs_tol=1e-12)
        for actual, target in zip(first, named, strict=True)
    ):
        raise StrictTrainingError("runtime AdamW group LRs differ from the recipe")
    if any(
        not math.isclose(
            group["initial_lr"], expected["muon_matrices"], rel_tol=1e-12
        )
        for group in optimizer.param_groups[6:]
    ):
        raise StrictTrainingError("runtime Muon group LR differs from the recipe")


def _assert_parent_compatible(
    parent_manifest: Mapping[str, Any],
    *,
    child_protocol: Mapping[str, Any],
    study_id: str,
    study_sha256: str,
) -> None:
    identity = parent_manifest.get("identity")
    parent = identity.get("protocol") if isinstance(identity, Mapping) else None
    if not isinstance(parent, Mapping):
        raise StrictTrainingError("stable parent lacks a strict protocol identity")
    if parent.get("run_kind") != "production":
        raise StrictTrainingError("cooldown parent is not a production checkpoint")
    parent_schedule = parent.get("schedule")
    if not isinstance(parent_schedule, Mapping) or parent_schedule.get(
        "cooldown_start_step"
    ) is not None:
        raise StrictTrainingError("cooldown parent is not a stable-only checkpoint")
    invariant_fields = (
        "protocol_version",
        "code",
        "model_config",
        "architecture_cli",
        "tokenizer",
        "source_dataset_manifest_sha256",
        "packing_capacity",
        "topology",
        "validation",
        "precision",
        "attention",
        "checkpointing",
        "preemption",
        "data_order",
        "data_order_authority",
        "seed",
        "world_size",
        "device_batch_size",
        "total_batch_size",
        "optimizer",
    )
    mismatches = [
        field
        for field in invariant_fields
        if parent.get(field) != child_protocol.get(field)
    ]
    child_schedule = child_protocol["schedule"]
    for field in (
        "name",
        "recipe_version",
        "warmup_steps",
        "momentum_warmup_steps",
        "stable_muon_weight_decay",
        "weight_decay_cooldown_policy",
        "momentum",
        "proxy_approval_sha256",
    ):
        if parent_schedule.get(field) != child_schedule.get(field):
            mismatches.append(f"schedule.{field}")
    if identity.get("study_id") != study_id:
        mismatches.append("study_id")
    if identity.get("study_manifest_sha256") != study_sha256:
        mismatches.append("study_manifest_sha256")
    if mismatches:
        raise StrictTrainingError(
            "stable parent differs at: " + ", ".join(sorted(set(mismatches)))
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    validate_cli(args)
    print_banner()
    signal.signal(signal.SIGUSR1, _record_preemption_signal)
    signal.signal(signal.SIGTERM, _record_preemption_signal)

    repo_root = Path(__file__).resolve().parents[1]
    recipe_preview = load_json_strict(args.study_manifest)
    if not isinstance(recipe_preview, dict):
        raise StrictTrainingError("study manifest must contain an object")
    recipe_preview_sha = verify_manifest_hash(recipe_preview)
    expected_tokenizer_name = recipe_preview["artifacts"]["tokenizer_name"]
    try:
        tokenizer_package = verify_tokenizer_package(
            args.tokenizer_manifest,
            expected_name=expected_tokenizer_name,
            expected_vocab_size=recipe_preview["model"]["vocab_size"],
        )
    except TokenizerPackageError as exc:
        raise StrictTrainingError(f"invalid production tokenizer package: {exc}") from exc
    expected_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    artifacts = load_artifact_bindings(
        study_manifest_path=args.study_manifest,
        data_dir=args.data_dir,
        exposure_plan_path=args.exposure_plan,
        validation_manifest_path=args.validation_manifest,
        tokenizer_sha256=tokenizer_package.canonical_sha256,
        seed=args.seed,
        world_size=expected_world_size,
        num_iterations=args.num_iterations,
        total_batch_size=args.total_batch_size,
    )
    if artifacts.recipe_sha256 != recipe_preview_sha:
        raise StrictTrainingError("study manifest changed during preflight")
    if args.study_id != artifacts.recipe["family_id"]:
        raise StrictTrainingError("study-id differs from the family recipe")
    _preflight, _preflight_sha256, preflight_capacity_sha256 = (
        validate_preflight_artifact_bindings(
            args.preflight_receipt,
            recipe=artifacts.recipe,
            recipe_sha256=artifacts.recipe_sha256,
            code_revision=args.code_revision,
            data_dir=args.data_dir,
            tokenizer_sha256=tokenizer_package.canonical_sha256,
            dataset_manifest=artifacts.dataset_manifest,
            dataset_sha256=artifacts.dataset_sha256,
            validation_manifest=artifacts.validation_manifest,
            validation_sha256=artifacts.validation_sha256,
            exposure_plan_sha256=artifacts.exposure_plan_sha256,
        )
    )
    expected_eval = artifacts.recipe["training"]["evaluation"]
    if expected_eval.get("eval_tokens_cli_unused") != -1:
        raise StrictTrainingError(
            "family recipe must mark --eval-tokens as an unused -1 sentinel"
        )
    if args.lr_schedule == "wsd" and args.wsd_recipe_version != artifacts.recipe[
        "weight_decay_proxy_ablation"
    ]["recipe_version"]:
        raise StrictTrainingError("WSD recipe version differs from the family recipe")
    code_audit = verify_code_provenance(
        repo_root,
        expected_revision=args.code_revision,
        recipe=artifacts.recipe,
    )
    attention_probe, attention_probe_sha256 = validate_attention_probe_receipt(
        args.attention_probe,
        args.preflight_receipt,
        recipe=artifacts.recipe,
        recipe_sha256=artifacts.recipe_sha256,
        code_revision=args.code_revision,
        attention_backend=args.attention_backend,
        window_pattern=args.window_pattern,
    )

    device_type = autodetect_device_type() if not args.device_type else args.device_type
    if device_type != "cuda" or COMPUTE_DTYPE != torch.bfloat16:
        raise StrictTrainingError("the production lane requires CUDA BF16")
    _ddp, rank, local_rank, world_size, device = compute_init(device_type)
    if world_size != expected_world_size:
        raise StrictTrainingError("distributed world size changed during initialization")
    master_process = rank == 0
    seed_plan = derive_seed_plan(args.seed, rank=rank)
    required_gpu_family = artifacts.recipe["attention_backend_gate"][
        "required_gpu_family"
    ]
    if required_gpu_family.upper() not in torch.cuda.get_device_name(device).upper():
        raise StrictTrainingError("runtime GPU family differs from the attention gate")

    if args.attention_backend == "sdpa":
        attention_runtime.USE_FA3 = False
    else:
        if not attention_runtime.HAS_FA3:
            raise StrictTrainingError("FA3 was requested but the pinned kernel is unavailable")
        attention_runtime.USE_FA3 = True
    actual_attention_backend = "fa3" if attention_runtime.USE_FA3 else "sdpa"
    if actual_attention_backend != args.attention_backend:
        raise StrictTrainingError("actual attention backend differs from the request")
    live_fa3_inventory_sha256 = verify_live_fa3_kernel_inventory(
        attention_probe,
        attention_runtime._fa3 if actual_attention_backend == "fa3" else None,
    )

    tokenizer = load_verified_tokenizer(tokenizer_package)
    token_bytes = load_verified_token_bytes(tokenizer_package, device=device)
    vocab_size = tokenizer.get_vocab_size()

    dataset_contract = (
        verify_strict_dataset(args.data_dir, verify_bytes=True)
        if master_process
        else None
    )
    dataset_contract = _broadcast_object(dataset_contract)
    if dataset_contract["manifest_sha256"] != artifacts.dataset_sha256:
        raise StrictTrainingError("byte-verified dataset differs from the exposure plan")
    capacity_receipt_sha256 = None
    selected_capacity = None
    if args.strict_run_kind in {"production", "smoke"}:
        _capacity_receipt, capacity_receipt_sha256, selected_capacity = (
            validate_bestfit_capacity_receipt(
                args.packing_capacity_receipt,
                data_dir=args.data_dir,
                dataset_manifest=artifacts.dataset_manifest,
                dataset_sha256=artifacts.dataset_sha256,
                tokenizer_sha256=tokenizer_package.canonical_sha256,
                recipe_sha256=artifacts.recipe_sha256,
                exposure_plan_sha256=artifacts.exposure_plan_sha256,
                world_size=world_size,
                required_token_positions=args.horizon_value,
                batch_sequences=args.device_batch_size,
                sequence_length=args.max_seq_len,
                global_batch_tokens=args.total_batch_size,
            )
        )
        if capacity_receipt_sha256 != preflight_capacity_sha256:
            raise StrictTrainingError(
                "runtime capacity receipt differs from the sealed preflight"
            )
    validation_rows = build_validation_rows(
        tokenizer,
        exposure_manifest=artifacts.validation_manifest,
        data_dir=args.data_dir,
        token_bytes=token_bytes,
        study_sha256=artifacts.recipe_sha256,
        tokenizer_sha256=tokenizer_package.canonical_sha256,
        dataset_contract=dataset_contract,
        sequence_length=args.max_seq_len,
    )
    validation_layouts = [validation_rows.layout_sha256]
    if is_ddp_initialized():
        validation_layouts = [None] * world_size
        dist.all_gather_object(validation_layouts, validation_rows.layout_sha256)
    if len(set(validation_layouts)) != 1:
        raise StrictTrainingError("validation row layout differs across ranks")
    validation_coverage = {
        "target_tokens": validation_rows.target_tokens,
        "payload_bytes": validation_rows.payload_bytes,
        "documents": validation_rows.documents,
        "logical_rows": len(validation_rows.inputs),
        "padded_token_positions_world1": (
            len(validation_rows.inputs) * args.max_seq_len
        ),
        "row_layout_sha256": validation_rows.layout_sha256,
    }

    random.seed(seed_plan.model_init)
    torch.manual_seed(seed_plan.model_init)
    torch.cuda.manual_seed(seed_plan.model_init)
    orig_model = _model_meta(args, vocab_size, args.depth)
    model_config = asdict(orig_model.config)
    orig_model.to_empty(device=device)
    orig_model.init_weights()
    param_counts = orig_model.num_scaling_params()
    total_parameters = param_counts["total"]
    scaling_parameters = _scaling_parameters(orig_model)
    flops_per_token = orig_model.estimate_flops()

    if args.lr_schedule == "wsd":
        effective_weight_decay = args.wsd_base_weight_decay
        cooldown_policy = args.wsd_weight_decay_cooldown
        cooldown_start = (
            None
            if args.wsd_cooldown_start_step < 0
            else args.wsd_cooldown_start_step
        )
        validate_wsd_schedule(
            end_step=args.num_iterations,
            warmup_steps=args.warmup_steps,
            momentum_warmup_steps=400,
            cooldown_start_step=cooldown_start,
            required_cooldown_fraction=(
                args.wsd_cooldown_fraction if cooldown_start is not None else None
            ),
        )
    else:
        d12_ref = _model_meta(args, vocab_size, 12)
        effective_weight_decay = nanochat_effective_weight_decay(
            args.weight_decay,
            scaling_parameters=scaling_parameters,
            global_batch_tokens=args.total_batch_size,
            d12_scaling_parameters=_scaling_parameters(d12_ref),
        )
        cooldown_policy = "cosine_full_horizon"
        cooldown_start = None

    batch_lr_scale = math.sqrt(args.total_batch_size / 524_288)
    optimizer = orig_model.setup_optimizer(
        unembedding_lr=args.unembedding_lr * batch_lr_scale,
        embedding_lr=args.embedding_lr * batch_lr_scale,
        scalar_lr=args.scalar_lr * batch_lr_scale,
        matrix_lr=args.matrix_lr * batch_lr_scale,
        weight_decay=effective_weight_decay,
    )
    optimizer_audit = _optimizer_audit(
        optimizer,
        args=args,
        batch_lr_scale=batch_lr_scale,
        base_weight_decay=effective_weight_decay,
    )
    recipe_binding = validate_recipe_invocation(
        artifacts.recipe,
        run_kind=args.strict_run_kind,
        depth=args.depth,
        model_config=model_config,
        total_parameters=total_parameters,
        scaling_parameters=scaling_parameters,
        world_size=world_size,
        device_batch_size=args.device_batch_size,
        total_batch_size=args.total_batch_size,
        num_iterations=args.num_iterations,
        stop_at_step=args.stop_at_step,
        eval_every=args.eval_every,
        seed=args.seed,
        model_tag=args.model_tag,
        lr_schedule=args.lr_schedule,
        effective_weight_decay=effective_weight_decay,
        weight_decay_cooldown_policy=cooldown_policy,
        cooldown_start_step=cooldown_start,
    )
    if args.strict_run_kind == "production":
        selected_stage = next(
            stage
            for stage in artifacts.recipe["stages"]
            if stage["id"] == recipe_binding["recipe_stage_id"]
        )
        source_boundary = selected_stage.get("source_step")
        if selected_stage["kind"] == "trunk":
            if args.parent_checkpoint_dir:
                raise StrictTrainingError("stable trunk stages cannot have parent lineage")
            if source_boundary is not None and args.resume_from_step < source_boundary:
                raise StrictTrainingError("trunk continuation must resume at or after its source fork")
        elif not args.parent_checkpoint_dir:
            raise StrictTrainingError("cooldown stages require stable-parent lineage")
        if args.resume_from_step >= 0 and args.resume_from_step < int(source_boundary or 0):
            raise StrictTrainingError("resume checkpoint predates the recipe stage source")
    elif args.strict_run_kind == "proxy" and args.stop_at_step != args.num_iterations:
        raise StrictTrainingError("proxy arms must run to their full declared horizon")
    if args.strict_run_kind in {"production", "smoke", "signal_smoke"}:
        _validate_d32_group_lrs(optimizer, artifacts.recipe)

    proxy_approval_sha256 = None
    if args.wsd_proxy_approval:
        _approval, proxy_approval_sha256 = validate_proxy_approval(
            args.wsd_proxy_approval,
            recipe=artifacts.recipe,
            recipe_sha256=artifacts.recipe_sha256,
            tokenizer_sha256=tokenizer_package.canonical_sha256,
            dataset_sha256=artifacts.dataset_sha256,
            code_revision=args.code_revision,
            attention_probe_sha256=attention_probe_sha256,
            production_scaling_parameters=artifacts.recipe["model"][
                "scaling_parameters"
            ],
            production_global_batch_tokens=artifacts.recipe["training"][
                "global_batch_tokens"
            ],
            accepted_base_weight_decay=(
                args.wsd_base_weight_decay
                if args.strict_run_kind != "proxy"
                else artifacts.recipe["training"]["optimizer_hyperparameters"][
                    "stable_weight_decay"
                ]
            ),
            accepted_weight_decay_cooldown_policy=(
                args.wsd_weight_decay_cooldown
                if args.strict_run_kind != "proxy"
                else artifacts.recipe["training"]["optimizer_hyperparameters"][
                    "weight_decay_cooldown_policy"
                ]
            ),
        )

    topology_gate = None
    topology_gate_sha256 = None
    if args.strict_run_kind == "production":
        assert proxy_approval_sha256 is not None
        topology_gate, topology_gate_sha256 = validate_production_topology_gate(
            args.production_gate,
            args.preflight_receipt,
            recipe=artifacts.recipe,
            recipe_sha256=artifacts.recipe_sha256,
            attention_probe_sha256=attention_probe_sha256,
            proxy_approval_sha256=proxy_approval_sha256,
            accepted_base_weight_decay=args.wsd_base_weight_decay,
            accepted_weight_decay_cooldown_policy=args.wsd_weight_decay_cooldown,
            world_size=world_size,
            packing_capacity_receipt_sha256=capacity_receipt_sha256,
            selected_capacity=selected_capacity,
        )

    checkpoint_dir = Path(get_base_dir()) / "base_checkpoints" / args.model_tag
    curve_log_path = Path(args.metrics_dir) / "training_curve.jsonl"
    resuming = args.resume_from_step >= 0
    parent_fields = bool(args.parent_checkpoint_dir)
    forking = parent_fields and not resuming
    loading_checkpoint = resuming or forking
    parent_manifest = None
    parent_lineage = None

    distributed_validation_positions = (
        math.ceil(
            validation_coverage["logical_rows"]
            / (args.device_batch_size * world_size)
        )
        * args.device_batch_size
        * args.max_seq_len
        * world_size
    )
    schedule_protocol = (
        {
            "name": "wsd",
            "recipe_version": args.wsd_recipe_version,
            "warmup_steps": args.warmup_steps,
            "momentum_warmup_steps": 400,
            "cooldown_start_step": cooldown_start,
            "cooldown_fraction": (
                args.wsd_cooldown_fraction if cooldown_start is not None else None
            ),
            "terminal_lr_multiplier": 0.0 if cooldown_start is not None else 1.0,
            "stable_muon_weight_decay": effective_weight_decay,
            "weight_decay_cooldown_policy": cooldown_policy,
            "momentum": {"initial": 0.85, "stable": 0.97, "final": 0.90},
            "proxy_approval_sha256": proxy_approval_sha256,
        }
        if args.lr_schedule == "wsd"
        else {
            "name": "nanochat_linear",
            "warmup_steps": args.warmup_steps,
            "warmdown_ratio": args.warmdown_ratio,
            "final_lr_fraction": args.final_lr_frac,
            "stable_muon_weight_decay": effective_weight_decay,
            "weight_decay_cooldown_policy": "cosine_full_horizon",
            "momentum_warmup_steps": 400,
            "momentum": {"initial": 0.85, "stable": 0.97, "final": 0.90},
            "proxy_approval_sha256": None,
        }
    )
    protocol: dict[str, Any] = {
        "protocol_version": "d32_wsd_strict_v1",
        "recipe_scope": recipe_binding["recipe_scope"],
        "run_kind": args.strict_run_kind,
        "code": code_audit,
        "model_config": model_config,
        "architecture_cli": {
            "depth": args.depth,
            "aspect_ratio": args.aspect_ratio,
            "head_dim": args.head_dim,
            "max_seq_len": args.max_seq_len,
            "window_pattern": args.window_pattern,
            "total_parameters": total_parameters,
            "scaling_parameters": scaling_parameters,
        },
        "tokenizer": {
            "name": tokenizer_package.config["name"],
            "artifact_sha256": tokenizer_package.canonical_sha256,
            "vocab_size": vocab_size,
        },
        "source_dataset_manifest_sha256": artifacts.dataset_sha256,
        "packing_capacity": {
            "receipt_sha256": capacity_receipt_sha256,
            "selected_topology": selected_capacity,
        },
        "topology": (
            None
            if topology_gate is None
            else {
                "gate_sha256": topology_gate_sha256,
                "authorized_world_size": topology_gate[
                    "authorized_production_world_size"
                ],
                "authorized_nodes": topology_gate["authorized_production_nodes"],
                "selection_reason": topology_gate["selection_reason"],
                "require_single_world_size_for_entire_lineage": topology_gate[
                    "require_single_world_size_for_entire_lineage"
                ],
            }
        ),
        "validation": {
            "manifest_sha256": artifacts.validation_sha256,
            "full_manifest": True,
            "packing_policy": "whole_document_no_crop_rows_before_rank_sharding",
            "bos_boundary_targets_masked": True,
            "padding_targets_masked": True,
            "target_tokens": validation_coverage["target_tokens"],
            "payload_bytes": validation_coverage["payload_bytes"],
            "documents": validation_coverage["documents"],
            "logical_rows": validation_coverage["logical_rows"],
            "row_layout_sha256": validation_coverage["row_layout_sha256"],
            "padded_token_positions_world1": validation_coverage[
                "padded_token_positions_world1"
            ],
            "padded_token_positions_runtime_world": distributed_validation_positions,
            "eval_every_updates": args.eval_every,
            "eval_tokens_cli_unused": args.eval_tokens,
        },
        "precision": {"compute_dtype": str(COMPUTE_DTYPE), "fp8_enabled": False},
        "attention": {
            "backend": actual_attention_backend,
            "window_pattern": args.window_pattern,
            "probe_sha256": attention_probe_sha256,
            "selection_reason": attention_probe["selection_reason"],
            "decision": attention_probe["decision"],
            "live_fa3_kernel_inventory_sha256": live_fa3_inventory_sha256,
        },
        "checkpointing": {
            "transactional": True,
            "save_every_updates": args.save_every,
        },
        "preemption": {
            "signals": ["SIGUSR1", "SIGTERM"],
            "checkpoint_boundary": "next_optimizer_safe_update",
            "exit_code": PREEMPTION_EXIT_CODE,
        },
        "model_tag": args.model_tag,
        "data_order": "bestfit",
        "data_order_authority": (
            "sealed_dataset_manifest_materialization_order_with_"
            "upstream_row_group_rank_sharding"
        ),
        # The full sealed plan is intentionally checkpoint-visible.  It lets a
        # cooldown fork prove that its new horizon reuses the exact parent data
        # prefix instead of merely comparing two opaque plan hashes.
        "exposure_plan": artifacts.exposure_plan,
        "seed": args.seed,
        "world_size": world_size,
        "device_batch_size": args.device_batch_size,
        "total_batch_size": args.total_batch_size,
        "num_iterations": args.num_iterations,
        "optimizer": optimizer_audit,
        "schedule": schedule_protocol,
        "parent": None,
    }

    def inspect_once(source_dir: str | Path, source_step: int):
        value = inspect_strict_checkpoint(source_dir, source_step) if master_process else None
        return _broadcast_object(value)

    parent_exposure_plan = None
    if parent_fields:
        parent_manifest = inspect_once(
            args.parent_checkpoint_dir, args.parent_checkpoint_step
        )
        if parent_manifest.get("canonical_sha256") != args.parent_checkpoint_sha256:
            raise StrictTrainingError("stable parent checkpoint SHA-256 mismatch")
        if cooldown_start != args.parent_checkpoint_step:
            raise StrictTrainingError("cooldown must start at its parent boundary")
        parent_lineage = {
            "checkpoint_sha256": parent_manifest["canonical_sha256"],
            "run_id": parent_manifest["identity"]["run_id"],
            "step": args.parent_checkpoint_step,
        }
        protocol["parent"] = parent_lineage
        _assert_parent_compatible(
            parent_manifest,
            child_protocol=protocol,
            study_id=args.study_id,
            study_sha256=artifacts.recipe_sha256,
        )
        parent_protocol = parent_manifest["identity"]["protocol"]
        parent_exposure_plan = parent_protocol.get("exposure_plan")
        if not isinstance(parent_exposure_plan, Mapping):
            raise StrictTrainingError("stable parent lacks its sealed exposure plan")
    elif args.strict_run_kind == "production" and cooldown_start is not None:
        raise StrictTrainingError("production cooldown requires explicit parent lineage")

    run_contract = {
        "study_id": args.study_id,
        "run_id": args.run_id,
        "study_manifest_sha256": artifacts.recipe_sha256,
        "exposure_plan_sha256": artifacts.exposure_plan_sha256,
        "protocol": protocol,
    }
    run_sha256 = hashlib.sha256(canonical_json_bytes(run_contract)).hexdigest()

    def checkpoint_identity(curve_state):
        return build_strict_checkpoint_identity(
            study_id=args.study_id,
            run_id=args.run_id,
            study_manifest_sha256=artifacts.recipe_sha256,
            run_sha256=run_sha256,
            tokenizer_artifact_sha256=tokenizer_package.canonical_sha256,
            exposure_plan_sha256=artifacts.exposure_plan_sha256,
            optimizer_audit=optimizer_audit,
            curve_log_state=curve_state,
            extra={"protocol": protocol},
        )

    source_manifest = None
    source_dir: str | Path | None = None
    source_step: int | None = None
    resume_recovered_bytes = 0
    if resuming:
        source_dir = checkpoint_dir
        source_step = args.resume_from_step
        source_manifest = inspect_once(source_dir, source_step)
        stored_curve = source_manifest["identity"]["curve_log"]
        expected_identity = checkpoint_identity(stored_curve)
        if source_manifest["identity"] != expected_identity:
            raise StrictTrainingError("resume checkpoint identity differs from this run")
        if master_process:
            reconciled = reconcile_training_log_to_checkpoint(
                curve_log_path,
                stored_curve,
                expected_study_id=args.study_id,
                expected_run_id=args.run_id,
            )
            resume_recovered_bytes = reconciled.recovered_truncated_bytes
        if is_ddp_initialized():
            dist.barrier()
    elif forking:
        source_dir = args.parent_checkpoint_dir
        source_step = args.parent_checkpoint_step
        source_manifest = parent_manifest
        expected_identity = source_manifest["identity"]

    meta_data = None
    loader_resume_state = None
    stored_rng = None
    payload_release_evidence = {
        "model_tensor_count_released": 0,
        "model_tensor_bytes_released": 0,
        "payload_container_released": 0,
    }
    if source_manifest is not None:
        (
            meta_data,
            loader_resume_state,
            stored_rng,
            payload_release_evidence,
        ) = _restore_and_release_checkpoint_payload(
            lambda: load_strict_checkpoint(
                source_dir,
                source_step,
                device,
                rank=rank,
                expected_world_size=world_size,
                expected_identity=expected_identity,
                expected_updates_completed=source_step,
                verified_manifest=source_manifest,
                load_model=master_process,
            ),
            model=orig_model,
            optimizer=optimizer,
            load_model=master_process,
            device=device,
        )
    if is_ddp_initialized():
        for parameter in orig_model.parameters():
            dist.broadcast(parameter.data, src=0)
        for buffer in orig_model.buffers():
            dist.broadcast(buffer.data, src=0)

    curve_log = None
    if master_process:
        curve_log = CanonicalTrainingLog(
            curve_log_path,
            study_id=args.study_id,
            run_id=args.run_id,
            resume=resuming,
        )
        if resuming:
            if curve_log.state.last_updates_completed != args.resume_from_step:
                raise StrictTrainingError("curve log and resume checkpoint disagree")
            curve_log.append(
                event_type="resume",
                updates_completed=args.resume_from_step,
                metrics={
                    "resume/recovered_log_bytes": resume_recovered_bytes,
                    "resume/model_payload_tensor_count_released": (
                        payload_release_evidence["model_tensor_count_released"]
                    ),
                    "resume/model_payload_bytes_released": (
                        payload_release_evidence["model_tensor_bytes_released"]
                    ),
                    "resume/payload_container_released": (
                        payload_release_evidence["payload_container_released"]
                    ),
                },
                identities={"checkpoint": source_manifest["canonical_sha256"]},
            )
        elif forking:
            curve_log.append(
                event_type="fork_start",
                updates_completed=args.parent_checkpoint_step,
                metrics={
                    "fork/parent_step": args.parent_checkpoint_step,
                    "fork/model_payload_tensor_count_released": (
                        payload_release_evidence["model_tensor_count_released"]
                    ),
                    "fork/model_payload_bytes_released": (
                        payload_release_evidence["model_tensor_bytes_released"]
                    ),
                    "fork/payload_container_released": (
                        payload_release_evidence["payload_container_released"]
                    ),
                },
                identities={"parent_checkpoint": args.parent_checkpoint_sha256},
            )
        else:
            curve_log.append(
                event_type="run_start",
                updates_completed=0,
                metrics={
                    "run/parameters": total_parameters,
                    "run/scaling_parameters": scaling_parameters,
                    "run/target_updates": args.num_iterations,
                    "run/target_token_positions": args.horizon_value,
                    "validation/target_tokens": validation_coverage["target_tokens"],
                    "validation/payload_bytes": validation_coverage["payload_bytes"],
                },
                identities={"run_sha256": run_sha256},
            )

    train_loader = StatefulBestFitLoader(
        tokenizer,
        args.device_batch_size,
        args.max_seq_len,
        data_dir=args.data_dir,
        token_bytes=token_bytes,
        study_sha256=artifacts.recipe_sha256,
        tokenizer_sha256=tokenizer_package.canonical_sha256,
        exposure_plan=artifacts.exposure_plan,
        parent_exposure_plan=parent_exposure_plan,
        device=device,
        rank=rank,
        world_size=world_size,
        dataset_contract=dataset_contract,
        resume_state=loader_resume_state,
        restore_rng=False,
    )
    dataloader_state = train_loader.state()
    if stored_rng is not None:
        if not isinstance(stored_rng, dict) or stored_rng.get("seed_plan") != seed_plan.to_dict():
            raise StrictTrainingError("checkpoint rank seed plan mismatch")
        restore_rank_rng_state(stored_rng.get("state"), device)

    def build_validation_loader():
        return StatefulSequentialDocumentLoader(
            tokenizer,
            args.device_batch_size,
            args.max_seq_len,
            exposure_manifest=artifacts.validation_manifest,
            data_dir=args.data_dir,
            token_bytes=token_bytes,
            study_sha256=artifacts.recipe_sha256,
            tokenizer_sha256=tokenizer_package.canonical_sha256,
            device=device,
            rank=rank,
            world_size=world_size,
            dataset_contract=dataset_contract,
            prepared_rows=validation_rows,
            restore_rng=False,
        )

    model = torch.compile(orig_model, dynamic=False)
    print0(
        f"Strict {args.strict_run_kind} | d{args.depth} | "
        f"params={total_parameters:,} scaling={scaling_parameters:,} | "
        f"world={world_size} | backend={actual_attention_backend}"
    )
    gpu_peak_flops = get_peak_flops(torch.cuda.get_device_name(0))
    synchronize = torch.cuda.synchronize
    tokens_per_microbatch = args.device_batch_size * args.max_seq_len * world_size
    if args.total_batch_size % tokens_per_microbatch:
        raise StrictTrainingError("global batch is not divisible by world microbatch")
    accumulation_steps = args.total_batch_size // tokens_per_microbatch

    use_dummy_wandb = args.run == "dummy" or not master_process
    wandb_run = (
        DummyWandb()
        if use_dummy_wandb
        else wandb.init(
            project=os.environ.get("WANDB_PROJECT", "nanochat"),
            name=args.run,
            config=vars(args),
        )
    )

    if loading_checkpoint:
        step = meta_data.get("updates_completed", meta_data.get("step"))
        expected_step = args.resume_from_step if resuming else args.parent_checkpoint_step
        if step != expected_step:
            raise StrictTrainingError("checkpoint metadata update counter mismatch")
        loop_state = meta_data["loop_state"]
        val_bpb = meta_data["val_bpb"]
        min_val_bpb = loop_state["min_val_bpb"]
        smooth_train_loss = loop_state["smooth_train_loss"]
        total_training_time = loop_state["total_training_time"]
    else:
        step = 0
        val_bpb = None
        min_val_bpb = float("inf")
        smooth_train_loss = 0.0
        total_training_time = 0.0

    def schedule_values(update_index: int):
        if args.lr_schedule == "wsd":
            return wsd_schedule_values(
                update_index,
                end_step=args.num_iterations,
                warmup_steps=args.warmup_steps,
                cooldown_start_step=cooldown_start,
                weight_decay_cooldown_policy=args.wsd_weight_decay_cooldown,
            )
        return nanochat_linear_schedule_values(
            update_index,
            end_step=args.num_iterations,
            warmup_steps=args.warmup_steps,
            warmdown_ratio=args.warmdown_ratio,
            final_lr_fraction=args.final_lr_frac,
        )

    initial_step = step
    preemption_exit_signal = 0
    mfu = 0.0
    while True:
        scheduled_last = step in {args.num_iterations, args.stop_at_step}
        observed_signal = _collective_signal(device)
        preemption_requested = bool(observed_signal) and not scheduled_last
        if preemption_requested:
            preemption_exit_signal = observed_signal
        last_step = scheduled_last or preemption_requested
        at_loaded_boundary = loading_checkpoint and step == initial_step
        if preemption_requested and curve_log is not None:
            curve_log.append(
                event_type="preemption_requested",
                updates_completed=step,
                metrics={
                    "preemption/signal_number": observed_signal,
                    "preemption/exit_code": PREEMPTION_EXIT_CODE,
                },
                identities={
                    "preemption_signal": signal.Signals(observed_signal).name,
                    "run_sha256": run_sha256,
                },
            )

        if (
            not preemption_requested
            and not at_loaded_boundary
            and (last_step or step % args.eval_every == 0)
        ):
            model.eval()
            evaluation = evaluate_loss(
                model, build_validation_loader(), None, token_bytes
            )
            if (
                evaluation.payload_target_count != validation_coverage["target_tokens"]
                or evaluation.all_target_count != validation_coverage["target_tokens"]
                or evaluation.payload_bytes != validation_coverage["payload_bytes"]
            ):
                raise StrictTrainingError(
                    "distributed validation coverage differs from the frozen contract"
                )
            val_bpb = evaluation.bpb
            min_val_bpb = min(min_val_bpb, val_bpb)
            print0(f"Update {step:05d} | validation bpb {val_bpb:.6f}")
            if curve_log is not None:
                curve_log.append(
                    event_type="validation",
                    updates_completed=step,
                    metrics={
                        "val/all_target_nats": evaluation.all_target_nats,
                        "val/all_target_count": evaluation.all_target_count,
                        "val/payload_nats": evaluation.payload_nats,
                        "val/payload_target_count": evaluation.payload_target_count,
                        "val/payload_bytes": evaluation.payload_bytes,
                        "val/bpb": evaluation.bpb,
                    },
                    identities={
                        "validation_manifest_sha256": artifacts.validation_sha256,
                        "run_sha256": run_sha256,
                    },
                )
            wandb_run.log({"step": step, "val/bpb": val_bpb})
            model.train()

        should_save = not at_loaded_boundary and (
            last_step
            or (
                step > 0
                and args.save_every > 0
                and step % args.save_every == 0
                and step != args.resume_from_step
            )
        )
        if should_save:
            checkpoint_meta = {
                "step": step,
                "updates_completed": step,
                "val_bpb": val_bpb,
                "model_config": model_config,
                "tokenizer_name": tokenizer_package.config["name"],
                "tokenizer_config": tokenizer_package.config,
                "user_config": vars(args),
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": args.total_batch_size,
                "strict_run_contract_sha256": run_sha256,
                "parent_lineage": parent_lineage,
                "validation_coverage": validation_coverage,
                "loop_state": {
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            }
            if preemption_requested:
                checkpoint_meta["preemption"] = {
                    "signal": signal.Signals(observed_signal).name,
                    "exit_code": PREEMPTION_EXIT_CODE,
                }
            curve_state = (
                checkpoint_curve_log_state(curve_log_path, curve_log.state)
                if master_process
                else None
            )
            curve_state = _broadcast_object(curve_state)

            def gather_rank_records(record):
                if not is_ddp_initialized():
                    return [record]
                records = [None] * world_size
                dist.all_gather_object(records, record)
                return records

            save_strict_checkpoint(
                checkpoint_dir,
                step,
                orig_model.state_dict() if master_process else None,
                optimizer.state_dict(),
                checkpoint_meta,
                loader_state=dataloader_state,
                rng_state={
                    "seed_plan": seed_plan.to_dict(),
                    "state": capture_rank_rng_state(device),
                },
                rank=rank,
                expected_world_size=world_size,
                identity=checkpoint_identity(curve_state),
                updates_completed=step,
                gather_rank_records=gather_rank_records,
                barrier=dist.barrier if is_ddp_initialized() else None,
            )

        if last_step:
            break

        synchronize()
        started = time.time()
        accumulated_loss = torch.zeros((), dtype=torch.float64, device=device)
        loader_seconds = 0.0
        for _micro_step in range(accumulation_steps):
            loader_started = time.perf_counter()
            inputs, targets, dataloader_state = next(train_loader)
            loader_seconds += time.perf_counter() - loader_started
            loss = model(inputs, targets)
            accumulated_loss += loss.detach().to(torch.float64)
            (loss / accumulation_steps).backward()
        mean_loss = accumulated_loss / accumulation_steps
        if is_ddp_initialized():
            dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
            mean_loss /= world_size
        if not bool(torch.isfinite(mean_loss).item()):
            raise FloatingPointError("non-finite globally reduced training loss")
        train_loss = float(mean_loss.item())
        controls = schedule_values(step)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * controls.lr_multiplier
            if group.get("kind") == "muon":
                group["momentum"] = controls.muon_momentum
                group["weight_decay"] = (
                    effective_weight_decay * controls.weight_decay_multiplier
                )
        optimizer.step()
        model.zero_grad(set_to_none=True)
        synchronize()
        duration = time.time() - started

        updates_completed = step + 1
        smooth_train_loss = 0.9 * smooth_train_loss + 0.1 * train_loss
        debiased_loss = smooth_train_loss / (1 - 0.9**updates_completed)
        if updates_completed > 10:
            total_training_time += duration
        tokens_per_second = int(args.total_batch_size / duration)
        loader_scheduled_positions_per_second = (
            args.total_batch_size / loader_seconds
        )
        loader_fraction_of_update = loader_seconds / duration
        mfu = (
            100
            * flops_per_token
            * args.total_batch_size
            / duration
            / (gpu_peak_flops * world_size)
        )
        position = dataloader_state["position"]
        print0(
            f"update {updates_completed:05d}/{args.num_iterations:05d} | "
            f"loss {debiased_loss:.6f} | lrm {controls.lr_multiplier:.4f} | "
            f"dt {duration * 1000:.1f}ms | tok/s {tokens_per_second:,} | "
            f"bf16_mfu {mfu:.2f} | epoch {position['epoch']} "
            f"file {position['file_index']} rg {position['row_group_index']} "
            f"row {position['row_index']}"
        )
        if curve_log is not None:
            curve_log.append(
                event_type="train_update",
                updates_completed=updates_completed,
                metrics={
                    "train/token_nll": train_loss,
                    "train/smoothed_token_nll": debiased_loss,
                    "train/lr_multiplier": controls.lr_multiplier,
                    "train/muon_momentum": controls.muon_momentum,
                    "train/muon_weight_decay": (
                        effective_weight_decay * controls.weight_decay_multiplier
                    ),
                    "train/duration_seconds": duration,
                    "train/tokens_per_second": tokens_per_second,
                    "train/mfu_percent": mfu,
                    "train/scheduled_positions": args.total_batch_size,
                    "train/loader_seconds": loader_seconds,
                    "train/loader_scheduled_positions_per_second": (
                        loader_scheduled_positions_per_second
                    ),
                    "train/loader_fraction_of_update": loader_fraction_of_update,
                    "exposure/source_bytes_loaded_rank": dataloader_state["totals"][
                        "source_bytes_loaded"
                    ],
                    "exposure/documents_loaded_rank": dataloader_state["totals"][
                        "documents_loaded"
                    ],
                    "exposure/documents_cropped_rank": dataloader_state["totals"][
                        "documents_cropped"
                    ],
                    "exposure/discarded_tokens_rank": dataloader_state["totals"][
                        "discarded_tokens"
                    ],
                    "exposure/discarded_bytes_rank": dataloader_state["totals"][
                        "discarded_bytes"
                    ],
                    "exposure/token_positions_completed": (
                        updates_completed * args.total_batch_size
                    ),
                },
                identities={"run_sha256": run_sha256},
            )
        if args.wandb_every > 0 and updates_completed % args.wandb_every == 0:
            wandb_run.log(
                {
                    "step": updates_completed,
                    "train/loss": debiased_loss,
                    "train/lrm": controls.lr_multiplier,
                    "train/dt": duration,
                    "train/tokens_per_second": tokens_per_second,
                    "train/mfu": mfu,
                    "train/loader_seconds": loader_seconds,
                    "train/loader_scheduled_positions_per_second": (
                        loader_scheduled_positions_per_second
                    ),
                    "train/loader_fraction_of_update": loader_fraction_of_update,
                }
            )
        first_update = step == 0 or (loading_checkpoint and step == initial_step)
        step = updates_completed
        if first_update:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif step % 5000 == 0:
            gc.collect()

    print0(f"Peak memory: {torch.cuda.max_memory_allocated() / 2**20:.2f} MiB")
    print0(f"Training time: {total_training_time / 60:.2f} min")
    if val_bpb is not None:
        print0(f"Minimum validation bpb: {min_val_bpb:.6f}")
    wandb_run.finish()
    # Keep every worker alive through the same final rendezvous.  In
    # particular, rank 0 must not leave early after serializing the model or
    # finishing W&B while peers are still inside the launcher process group.
    synchronize()
    if is_ddp_initialized():
        dist.barrier()
    compute_cleanup()
    if preemption_exit_signal:
        print0(
            f"Checkpointed signal {preemption_exit_signal}; exiting "
            f"{PREEMPTION_EXIT_CODE} for launcher requeue"
        )
        return PREEMPTION_EXIT_CODE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
