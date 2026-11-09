#!/usr/bin/env python3
"""Seal the one-A100 attention/backend probe for the Turkish d32 family."""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from nanochat.experiment_manifest import file_sha256, seal_manifest, write_json_atomic
from scripts.d32_family_workflow import _load_receipt, load_recipe


def _tree_inventory(root: Path, *, maximum_files: int = 4096) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    if root.is_file():
        paths = [root]
        base = root.parent
    elif root.is_dir():
        paths = sorted(path for path in root.rglob("*") if path.is_file())
        base = root
    else:
        return {"root": str(root), "files": [], "inventory_sha256": None}
    if len(paths) > maximum_files:
        raise RuntimeError(
            f"kernel inventory is unexpectedly broad ({len(paths)} files > {maximum_files})"
        )
    for path in paths:
        if path.is_symlink():
            raise RuntimeError(f"kernel inventory contains a symlink: {path}")
        records.append(
            {
                "path": path.relative_to(base).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"root": str(root), "files": records, "inventory_sha256": digest}


def _finite_backend_check(
    attention_runtime: Any, *, use_fa3: bool, window_size: tuple[int, int]
) -> dict[str, Any]:
    torch.manual_seed(20260820)
    shape = (1, 256, 8, 64)
    tensors = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    ]
    original = attention_runtime.USE_FA3
    try:
        attention_runtime.USE_FA3 = use_fa3
        output = attention_runtime.flash_attn.flash_attn_func(
            *tensors, causal=True, window_size=window_size
        )
        output.float().square().mean().backward()
    finally:
        attention_runtime.USE_FA3 = original
    return {
        "backend": "fa3" if use_fa3 else "sdpa",
        "window_size": list(window_size),
        "output_finite": bool(torch.isfinite(output).all().item()),
        "gradient_finite": all(
            tensor.grad is not None and bool(torch.isfinite(tensor.grad).all().item())
            for tensor in tensors
        ),
        "output_shape": list(output.shape),
    }


def _fa3_comparison(
    attention_runtime: Any, *, window_size: tuple[int, int]
) -> dict[str, Any] | None:
    if not attention_runtime.HAS_FA3:
        return None
    torch.manual_seed(20260820)
    shape = (1, 2048, 16, 128)
    source = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)]

    def run(use_fa3: bool) -> tuple[torch.Tensor, list[torch.Tensor]]:
        values = [tensor.detach().clone().requires_grad_(True) for tensor in source]
        attention_runtime.USE_FA3 = use_fa3
        output = attention_runtime.flash_attn.flash_attn_func(
            *values, causal=True, window_size=window_size
        )
        output.float().square().mean().backward()
        return output.detach().float(), [value.grad.detach().float() for value in values]

    original = attention_runtime.USE_FA3
    try:
        sdpa_output, sdpa_grads = run(False)
        fa3_output, fa3_grads = run(True)
    finally:
        attention_runtime.USE_FA3 = original
    denominator = sdpa_output.abs().max().clamp_min(1e-12)
    output_relative_max_error = float(
        ((fa3_output - sdpa_output).abs().max() / denominator).item()
    )
    gradient_relative_max_errors = []
    for fa3_grad, sdpa_grad in zip(fa3_grads, sdpa_grads, strict=True):
        denominator = sdpa_grad.abs().max().clamp_min(1e-12)
        gradient_relative_max_errors.append(
            float(((fa3_grad - sdpa_grad).abs().max() / denominator).item())
        )
    return {
        "window_size": list(window_size),
        "output_finite": bool(torch.isfinite(fa3_output).all().item()),
        "gradients_finite": all(bool(torch.isfinite(value).all().item()) for value in fa3_grads),
        "output_relative_max_error_vs_sdpa": output_relative_max_error,
        "gradient_relative_max_errors_vs_sdpa": gradient_relative_max_errors,
    }


def _d32_model_finite_check(
    attention_runtime: Any, *, backend: str, window_pattern: str
) -> dict[str, Any]:
    """Run a real d32/T2048 BF16 forward/backward before choosing a backend."""

    from nanochat.gpt import GPT, GPTConfig

    if (backend, window_pattern) not in {("fa3", "SSSL"), ("sdpa", "L")}:
        raise ValueError("unreviewed attention backend/window pair")
    attention_runtime.USE_FA3 = backend == "fa3"
    # Mirror the pinned trainer's construction path exactly: initialize the
    # meta model, materialize empty CUDA storage, then run the literal family
    # seed through GPT.init_weights(). `GPT(config).cuda()` is invalid here
    # because GPT.__init__ intentionally creates meta-only rotary buffers.
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    config = GPTConfig(
        sequence_len=2048,
        vocab_size=32768,
        n_layer=32,
        n_head=16,
        n_kv_head=16,
        n_embd=2048,
        window_pattern=window_pattern,
    )
    model = None
    inputs = None
    targets = None
    loss = None
    gradients = None
    try:
        with torch.device("meta"):
            model = GPT(config)
        model.to_empty(device=torch.device("cuda"))
        model.init_weights()
        model.train()

        def dtype_inventory(values) -> dict[str, dict[str, int]]:
            inventory: dict[str, dict[str, int]] = {}
            for value in values:
                record = inventory.setdefault(
                    str(value.dtype), {"tensor_count": 0, "element_count": 0}
                )
                record["tensor_count"] += 1
                record["element_count"] += value.numel()
            return dict(sorted(inventory.items()))

        parameter_dtypes = dtype_inventory(model.parameters())
        buffer_dtypes = dtype_inventory(model.buffers())
        inputs = torch.randint(
            0, config.vocab_size, (1, config.sequence_len), device="cuda"
        )
        targets = torch.randint(
            0, config.vocab_size, (1, config.sequence_len), device="cuda"
        )
        # Supplying targets exercises the exact training forward path, including
        # logits, softcap, and cross-entropy, rather than a synthetic kernel only.
        loss = model(inputs, targets)
        output_finite = bool(torch.isfinite(loss).all().item())
        loss_finite = output_finite
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        gradients_finite = bool(gradients) and all(
            bool(torch.isfinite(gradient).all().item()) for gradient in gradients
        )
        peak_memory = int(torch.cuda.max_memory_allocated())
        return {
            "backend": backend,
            "window_pattern": window_pattern,
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
            "parameter_dtype_inventory": parameter_dtypes,
            "buffer_dtype_inventory": buffer_dtypes,
            "output_finite": output_finite,
            "loss_finite": loss_finite,
            "gradients_present": bool(gradients),
            "gradients_finite": gradients_finite,
            "gradient_tensor_count": len(gradients),
            "peak_cuda_memory_bytes": peak_memory,
        }
    finally:
        del loss, inputs, targets, gradients, model
        gc.collect()
        torch.cuda.empty_cache()


def _benchmark_pattern(attention_runtime: Any, pattern: str, repeats: int) -> dict[str, Any]:
    if pattern not in {"L", "SSSL"}:
        raise ValueError(f"unsupported probe pattern: {pattern}")
    torch.manual_seed(20260821)
    sequence_length = 2048
    shape = (1, sequence_length, 16, 128)
    q, k, v = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    ]
    windows = [(-1, 0)] * 4 if pattern == "L" else [(sequence_length // 4, 0)] * 3 + [(-1, 0)]

    def iteration() -> None:
        for tensor in (q, k, v):
            tensor.grad = None
        total = torch.zeros((), device="cuda", dtype=torch.float32)
        for window in windows:
            output = attention_runtime.flash_attn.flash_attn_func(
                q, k, v, causal=True, window_size=window
            )
            total = total + output.float().square().mean()
        total.backward()

    for _ in range(2):
        iteration()
    torch.cuda.synchronize()
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        iteration()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
    durations_sorted = sorted(durations)
    median = durations_sorted[len(durations_sorted) // 2]
    return {
        "pattern": pattern,
        "sequence_length": sequence_length,
        "attention_layers_per_iteration": 4,
        "repeats": repeats,
        "durations_seconds": durations,
        "median_seconds": median,
        "scheduled_token_layers_per_second": 4 * sequence_length / median,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    recipe, recipe_sha = load_recipe(args.recipe)
    preflight, preflight_sha = _load_receipt(
        args.preflight_receipt, "d32_family_preflight_receipt"
    )
    if preflight["recipe"]["canonical_sha256"] != recipe_sha:
        raise RuntimeError("preflight/recipe mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("attention probe requires exactly one visible CUDA GPU")
    gpu_name = torch.cuda.get_device_name(0)
    if "A100" not in gpu_name.upper():
        raise RuntimeError(f"attention probe requires A100, found {gpu_name}")
    if args.repeats < 3:
        raise ValueError("--repeats must be at least 3")

    import nanochat.flash_attention as attention_runtime

    initial_has_fa3 = bool(attention_runtime.HAS_FA3)
    initial_use_fa3 = bool(attention_runtime.USE_FA3)
    fa3_model_check = None
    fa3_model_probe_error = None
    fa3_model_smoke_passed = False
    if initial_has_fa3 and initial_use_fa3:
        try:
            fa3_model_check = _d32_model_finite_check(
                attention_runtime, backend="fa3", window_pattern="SSSL"
            )
            fa3_model_smoke_passed = (
                fa3_model_check["output_finite"] is True
                and fa3_model_check["loss_finite"] is True
                and fa3_model_check["gradients_present"] is True
                and fa3_model_check["gradients_finite"] is True
            )
        except Exception as exc:  # upstream policy permits the SDPA fallback
            fa3_model_probe_error = f"{type(exc).__name__}: {exc}"[:500]

    if fa3_model_smoke_passed:
        selected_backend = "fa3"
        selected_pattern = "SSSL"
        selected_model_check = fa3_model_check
        selection_reason = "pinned_upstream_auto_fa3_passed_actual_d32_bf16_finite_smoke"
        decision = "accepted_fa3_SSSL"
        attention_runtime.USE_FA3 = True
    else:
        selected_backend = "sdpa"
        selected_pattern = "L"
        if not initial_has_fa3:
            selection_reason = "fa3_unavailable_fallback_to_sdpa_full_attention"
        elif not initial_use_fa3:
            selection_reason = "pinned_upstream_auto_did_not_select_fa3"
        else:
            selection_reason = "fa3_actual_d32_bf16_smoke_failed_fallback_to_sdpa_full_attention"
        decision = "accepted_sdpa_L_fallback"
        attention_runtime.USE_FA3 = False
        selected_model_check = _d32_model_finite_check(
            attention_runtime, backend="sdpa", window_pattern="L"
        )
        if not all(
            selected_model_check[field] is True
            for field in ("output_finite", "loss_finite", "gradients_present", "gradients_finite")
        ):
            raise RuntimeError("the actual-d32 SDPA+L fallback failed BF16 forward/backward")

    # This comparison is diagnostic only. Upstream auto-selection plus the
    # actual-model finite smoke above, not a locally invented error threshold,
    # decides production behavior.
    fa3_check = None
    fa3_diagnostic_error = None
    if initial_has_fa3:
        try:
            fa3_check = _fa3_comparison(attention_runtime, window_size=(512, 0))
        except Exception as exc:
            fa3_diagnostic_error = f"{type(exc).__name__}: {exc}"[:500]
    attention_runtime.USE_FA3 = selected_backend == "fa3"
    full = _benchmark_pattern(attention_runtime, "L", args.repeats)
    sliding = _benchmark_pattern(attention_runtime, "SSSL", args.repeats)

    kernel_module_file = None
    kernel_inventory = {"root": None, "files": [], "inventory_sha256": None}
    if initial_has_fa3 and attention_runtime._fa3 is not None:
        try:
            kernel_module_file = Path(inspect.getfile(attention_runtime._fa3)).resolve()
            kernel_inventory = _tree_inventory(kernel_module_file.parent)
        except (OSError, TypeError) as exc:
            raise RuntimeError(f"cannot inventory the available FA3 kernel: {exc}") from exc

    repo_root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    if revision != preflight["code"]["git_commit"]:
        raise RuntimeError("probe code revision differs from preflight")
    flash_file = repo_root / "nanochat" / "flash_attention.py"
    receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "d32_attention_backend_probe",
            "family_id": recipe["family_id"],
            "recipe_sha256": recipe_sha,
            "preflight_receipt_sha256": preflight_sha,
            "code_revision": revision,
            "world_size": 1,
            "gpu": {
                "name": gpu_name,
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            },
            "module_detection": {
                "HAS_FA3_at_import": initial_has_fa3,
                "USE_FA3_at_import": initial_use_fa3,
                "selected_backend_after_probe": selected_backend,
                "selected_window_pattern": selected_pattern,
                "flash_attention_file_sha256": file_sha256(flash_file),
                "fa3_kernel_identifier": "pinned_upstream_auto_loader" if initial_has_fa3 else None,
                "fa3_module_file": str(kernel_module_file) if kernel_module_file else None,
                "fa3_kernel_inventory": kernel_inventory,
                "cache_environment": {
                    key: os.environ.get(key)
                    for key in ("HF_HOME", "HF_HUB_CACHE", "TORCH_HOME")
                },
            },
            "selected_d32_model_forward_backward": selected_model_check,
            "fa3_d32_model_forward_backward": fa3_model_check,
            "fa3_comparison": fa3_check,
            "fa3_model_probe_error": fa3_model_probe_error,
            "fa3_diagnostic_error": fa3_diagnostic_error,
            "fa3_actual_d32_smoke_passed": fa3_model_smoke_passed,
            "fa3_sdpa_comparison_decisional": False,
            "pattern_benchmarks": {"L": full, "SSSL": sliding},
            "selection_reason": selection_reason,
            "decision": decision,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "canonical_sha256": None,
        }
    )
    write_json_atomic(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
