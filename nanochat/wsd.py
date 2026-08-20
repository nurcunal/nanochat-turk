"""Horizon-independent warmup/stable/decay schedules.

The stable prefix must be byte-for-byte reusable when a run is forked into
several cooldown horizons.  In particular, none of the values returned for the
stable phase depend on the eventual training horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence


class WSDScheduleError(ValueError):
    """Raised when a WSD schedule or shared-training plan is inconsistent."""


@dataclass(frozen=True)
class WSDScheduleValues:
    lr_multiplier: float
    muon_momentum: float
    weight_decay_multiplier: float
    phase: str


@dataclass(frozen=True)
class WSDMilestone:
    scale: int
    final_step: int
    stable_parent_step: int
    cooldown_steps: int
    scheduled_tokens: int


def nanochat_linear_schedule_values(
    step: int,
    *,
    end_step: int,
    warmup_steps: int = 40,
    warmdown_ratio: float = 0.65,
    final_lr_fraction: float = 0.05,
    momentum_warmup_steps: int = 400,
    momentum_initial: float = 0.85,
    momentum_peak: float = 0.97,
    momentum_final: float = 0.90,
) -> WSDScheduleValues:
    """Return the pinned upstream Nanochat LR, momentum, and WD controls.

    The branch order and boundary arithmetic deliberately mirror upstream
    ``scripts/base_train.py``.  Keeping the control in one tested function
    prevents the WSD implementation from silently changing the ablation
    baseline.
    """

    end = _require_int(end_step, "end_step", minimum=1)
    current = _require_int(step, "step")
    if current > end:
        raise WSDScheduleError("step must not exceed end_step")
    warmup = _require_int(warmup_steps, "warmup_steps", minimum=1)
    momentum_warmup = _require_int(
        momentum_warmup_steps, "momentum_warmup_steps", minimum=1
    )
    ratio = float(warmdown_ratio)
    if not 0.0 < ratio < 1.0:
        raise WSDScheduleError("warmdown_ratio must be between 0 and 1")
    final_lr = float(final_lr_fraction)
    if not 0.0 <= final_lr <= 1.0:
        raise WSDScheduleError("final_lr_fraction must be between 0 and 1")
    warmdown_steps = round(ratio * end)
    if warmdown_steps <= 0:
        raise WSDScheduleError("warmdown must contain at least one update")
    warmdown_start = end - warmdown_steps

    if current < warmup:
        lr_multiplier = (current + 1) / warmup
        phase = "warmup"
    elif current <= warmdown_start:
        lr_multiplier = 1.0
        phase = "stable"
    else:
        progress = (end - current) / warmdown_steps
        lr_multiplier = progress + (1.0 - progress) * final_lr
        phase = "warmdown"

    if current < momentum_warmup:
        fraction = current / momentum_warmup
        momentum = (
            (1.0 - fraction) * momentum_initial + fraction * momentum_peak
        )
    elif current >= warmdown_start:
        progress = (current - warmdown_start) / warmdown_steps
        momentum = (
            momentum_peak * (1.0 - progress) + momentum_final * progress
        )
    else:
        momentum = momentum_peak

    return WSDScheduleValues(
        lr_multiplier=lr_multiplier,
        muon_momentum=momentum,
        weight_decay_multiplier=0.5 * (1.0 + math.cos(math.pi * current / end)),
        phase=phase,
    )


def _require_int(value: int, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise WSDScheduleError(f"{name} must be an integer >= {minimum}")
    return value


def validate_wsd_schedule(
    *,
    end_step: int,
    warmup_steps: int,
    momentum_warmup_steps: int,
    cooldown_start_step: int | None,
    required_cooldown_fraction: float | None = None,
) -> None:
    """Validate one stable-only trunk or one stable+cooldown branch."""

    end = _require_int(end_step, "end_step", minimum=1)
    warmup = _require_int(warmup_steps, "warmup_steps", minimum=1)
    momentum_warmup = _require_int(
        momentum_warmup_steps, "momentum_warmup_steps", minimum=1
    )
    if max(warmup, momentum_warmup) >= end:
        raise WSDScheduleError("warmup must finish before the run horizon")
    if cooldown_start_step is None:
        return
    start = _require_int(cooldown_start_step, "cooldown_start_step", minimum=1)
    if start < max(warmup, momentum_warmup):
        raise WSDScheduleError("cooldown must start after both warmups")
    if start >= end:
        raise WSDScheduleError("cooldown must start before the run horizon")
    if required_cooldown_fraction is not None:
        fraction = float(required_cooldown_fraction)
        if not 0.0 < fraction < 1.0:
            raise WSDScheduleError("required_cooldown_fraction must be between 0 and 1")
        expected_steps = round(end * fraction)
        if end - start != expected_steps:
            raise WSDScheduleError(
                "cooldown length does not equal the required fraction of the horizon"
            )


def wsd_schedule_values(
    step: int,
    *,
    end_step: int,
    warmup_steps: int,
    cooldown_start_step: int | None,
    weight_decay_cooldown_policy: str,
    momentum_warmup_steps: int = 400,
    momentum_initial: float = 0.85,
    momentum_peak: float = 0.97,
    momentum_final: float = 0.90,
) -> WSDScheduleValues:
    """Return LR, Muon momentum, and WD multipliers at an update boundary.

    ``step`` is the number of optimizer updates already completed.  A cooldown
    spanning ``[start, end)`` therefore applies exactly ``end - start`` updates;
    its final nonzero update uses ``1 / (end - start)`` of the peak LR, while
    the terminal checkpoint boundary at ``end`` evaluates to zero.
    """

    if weight_decay_cooldown_policy not in {"constant", "linear_to_zero"}:
        raise WSDScheduleError(
            "weight_decay_cooldown_policy must be constant or linear_to_zero"
        )
    validate_wsd_schedule(
        end_step=end_step,
        warmup_steps=warmup_steps,
        momentum_warmup_steps=momentum_warmup_steps,
        cooldown_start_step=cooldown_start_step,
    )
    current = _require_int(step, "step")
    if current > end_step:
        raise WSDScheduleError("step must not exceed end_step")

    if current < warmup_steps:
        lr_multiplier = (current + 1) / warmup_steps
        lr_phase = "warmup"
    else:
        lr_multiplier = 1.0
        lr_phase = "stable"

    momentum_progress = min(current / momentum_warmup_steps, 1.0)
    momentum = (
        momentum_initial * (1.0 - momentum_progress)
        + momentum_peak * momentum_progress
    )
    weight_decay_multiplier = 1.0

    if cooldown_start_step is not None and current >= cooldown_start_step:
        cooldown_steps = end_step - cooldown_start_step
        remaining_fraction = max(0.0, (end_step - current) / cooldown_steps)
        completed_fraction = 1.0 - remaining_fraction
        lr_multiplier = remaining_fraction
        momentum = (
            momentum_peak * remaining_fraction
            + momentum_final * completed_fraction
        )
        weight_decay_multiplier = (
            remaining_fraction
            if weight_decay_cooldown_policy == "linear_to_zero"
            else 1.0
        )
        lr_phase = "cooldown"

    return WSDScheduleValues(
        lr_multiplier=lr_multiplier,
        muon_momentum=momentum,
        weight_decay_multiplier=weight_decay_multiplier,
        phase=lr_phase,
    )


def build_shared_wsd_milestones(
    *,
    scaling_parameters: int,
    global_batch_tokens: int,
    scales: Iterable[int] = (12, 20, 40),
    cooldown_fraction: float = 0.10,
) -> tuple[WSDMilestone, ...]:
    """Build exact fork/final boundaries for a shared WSD model family."""

    parameters = _require_int(
        scaling_parameters, "scaling_parameters", minimum=1
    )
    batch = _require_int(global_batch_tokens, "global_batch_tokens", minimum=1)
    if not 0.0 < cooldown_fraction < 1.0:
        raise WSDScheduleError("cooldown_fraction must be between 0 and 1")
    result: list[WSDMilestone] = []
    seen: set[int] = set()
    for raw_scale in scales:
        scale = _require_int(raw_scale, "scale", minimum=1)
        if scale in seen:
            raise WSDScheduleError("scales must be unique")
        seen.add(scale)
        target_tokens = parameters * scale
        final_step = target_tokens // batch
        if final_step <= 0:
            raise WSDScheduleError("scale is smaller than one global batch")
        # Nanochat defines its finite horizon in whole optimizer updates.  The
        # scheduled count is therefore the largest batch boundary not exceeding
        # the mathematical ratio target (the shortfall is always < one batch).
        scheduled_tokens = final_step * batch
        cooldown_steps = round(final_step * cooldown_fraction)
        stable_parent_step = final_step - cooldown_steps
        validate_wsd_schedule(
            end_step=final_step,
            warmup_steps=1,
            momentum_warmup_steps=1,
            cooldown_start_step=stable_parent_step,
            required_cooldown_fraction=cooldown_fraction,
        )
        result.append(
            WSDMilestone(
                scale=scale,
                final_step=final_step,
                stable_parent_step=stable_parent_step,
                cooldown_steps=cooldown_steps,
                scheduled_tokens=scheduled_tokens,
            )
        )
    result.sort(key=lambda item: item.scale)
    return tuple(result)


def transfer_weight_decay_between_stages(
    base_weight_decay: float,
    *,
    source_scaling_parameters: int,
    source_global_batch_tokens: int,
    target_scaling_parameters: int,
    target_global_batch_tokens: int,
) -> float:
    """Transfer Nanochat's WD coefficient across model-width/batch stages.

    Nanochat uses ``lambda ∝ sqrt(B) / N_scaling`` when the data/parameter
    ratio is held fixed.  This form transfers an already-derived coefficient
    (for example the production d32 value) to a proxy stage without confusing
    an absolute d32 coefficient for a d12/d20 coefficient.
    """

    base = float(base_weight_decay)
    if not math.isfinite(base) or base < 0.0:
        raise WSDScheduleError("base_weight_decay must be finite and non-negative")
    source_parameters = _require_int(
        source_scaling_parameters, "source_scaling_parameters", minimum=1
    )
    source_batch = _require_int(
        source_global_batch_tokens, "source_global_batch_tokens", minimum=1
    )
    target_parameters = _require_int(
        target_scaling_parameters, "target_scaling_parameters", minimum=1
    )
    target_batch = _require_int(
        target_global_batch_tokens, "target_global_batch_tokens", minimum=1
    )
    return (
        base
        * math.sqrt(target_batch / source_batch)
        * (source_parameters / target_parameters)
    )


def validate_weight_decay_proxy_candidates(
    candidate_results: Sequence[Mapping[str, object]],
    *,
    production_scaling_parameters: int,
    production_global_batch_tokens: int,
    accepted_candidate_id: str,
    accepted_base_weight_decay: float,
    accepted_weight_decay_cooldown_policy: str,
) -> None:
    """Validate a two-stage WD proxy result set and its production mapping."""

    production_parameters = _require_int(
        production_scaling_parameters,
        "production_scaling_parameters",
        minimum=1,
    )
    production_batch = _require_int(
        production_global_batch_tokens,
        "production_global_batch_tokens",
        minimum=1,
    )
    if not isinstance(candidate_results, Sequence) or isinstance(
        candidate_results, (str, bytes, bytearray)
    ) or not candidate_results:
        raise WSDScheduleError("candidate_results must be a nonempty array")
    candidate_by_id: dict[str, Mapping[str, object]] = {}
    for candidate in candidate_results:
        if not isinstance(candidate, Mapping):
            raise WSDScheduleError("each proxy candidate must be an object")
        candidate_id = candidate.get("id")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id in candidate_by_id
        ):
            raise WSDScheduleError("proxy candidate IDs must be unique strings")
        candidate_by_id[candidate_id] = candidate
    required = {
        "upstream_92d63d4e_control",
        "legacy_transferred",
        "no_weight_decay",
    }
    if not required.issubset(candidate_by_id):
        raise WSDScheduleError("proxy results lack required comparison arms")
    accepted = candidate_by_id.get(accepted_candidate_id)
    if accepted is None:
        raise WSDScheduleError("accepted_candidate_id is not in candidate_results")
    if accepted.get("eligible_for_production") is not True:
        raise WSDScheduleError("accepted proxy candidate is not production-eligible")

    policies: set[str] = set()
    for candidate_id, candidate in candidate_by_id.items():
        production_wd = candidate.get("production_base_weight_decay")
        if (
            isinstance(production_wd, bool)
            or not isinstance(production_wd, (int, float))
            or not math.isfinite(float(production_wd))
            or float(production_wd) < 0.0
        ):
            raise WSDScheduleError(
                f"proxy candidate {candidate_id} has invalid production WD"
            )
        policy = candidate.get("weight_decay_cooldown_policy")
        if policy in {"constant", "linear_to_zero"}:
            policies.add(policy)
        stages = candidate.get("stage_effective_weight_decays")
        if not isinstance(stages, Sequence) or isinstance(
            stages, (str, bytes, bytearray)
        ) or len(stages) != 2:
            raise WSDScheduleError(
                f"proxy candidate {candidate_id} must bind d12 and d20 WD"
            )
        stage_by_id: dict[str, Mapping[str, object]] = {}
        for stage in stages:
            if not isinstance(stage, Mapping):
                raise WSDScheduleError("proxy stage WD record must be an object")
            stage_id = stage.get("stage_id")
            if not isinstance(stage_id, str) or stage_id in stage_by_id:
                raise WSDScheduleError("proxy stage IDs must be unique strings")
            stage_by_id[stage_id] = stage
        if set(stage_by_id) != {"d12", "d20"}:
            raise WSDScheduleError(
                f"proxy candidate {candidate_id} stage IDs must be d12 and d20"
            )
        for stage_id, stage in stage_by_id.items():
            try:
                expected = transfer_weight_decay_between_stages(
                    float(production_wd),
                    source_scaling_parameters=production_parameters,
                    source_global_batch_tokens=production_batch,
                    target_scaling_parameters=stage.get("scaling_parameters"),
                    target_global_batch_tokens=stage.get("global_batch_tokens"),
                )
                actual = float(stage.get("effective_base_weight_decay"))
            except (TypeError, ValueError) as exc:
                raise WSDScheduleError(
                    f"proxy candidate {candidate_id}/{stage_id} has invalid transfer fields"
                ) from exc
            if not math.isfinite(actual) or not math.isclose(
                actual, expected, rel_tol=1e-12, abs_tol=1e-15
            ):
                raise WSDScheduleError(
                    f"proxy candidate {candidate_id}/{stage_id} violates the "
                    "Nanochat width/batch transfer rule"
                )
    if policies != {"constant", "linear_to_zero"}:
        raise WSDScheduleError(
            "proxy results must compare constant and linear-to-zero cooldown WD"
        )
    if (
        accepted.get("production_base_weight_decay")
        != accepted_base_weight_decay
        or accepted.get("weight_decay_cooldown_policy")
        != accepted_weight_decay_cooldown_policy
    ):
        raise WSDScheduleError(
            "accepted proxy candidate differs from the production binding"
        )


def cumulative_effective_decay(
    lr_multipliers: Iterable[float],
    weight_decay_multipliers: Iterable[float],
) -> float:
    """Return ``sum(lr_multiplier * wd_multiplier)`` deterministically.

    This is a schedule-level proxy, not an assertion about Muon's cautious
    parameter-wise mask.  It is useful for selecting proxy-ablation candidates
    when changing schedule shape without pretending the candidates are already
    empirically equivalent.
    """

    learning_rates = tuple(float(value) for value in lr_multipliers)
    weight_decays = tuple(float(value) for value in weight_decay_multipliers)
    if len(learning_rates) != len(weight_decays) or not learning_rates:
        raise WSDScheduleError(
            "LR and weight-decay multiplier sequences must have equal nonzero length"
        )
    if any(value < 0.0 for value in (*learning_rates, *weight_decays)):
        raise WSDScheduleError("schedule multipliers must be non-negative")
    return sum(
        lr_multiplier * weight_decay_multiplier
        for lr_multiplier, weight_decay_multiplier in zip(
            learning_rates, weight_decays, strict=True
        )
    )


def legacy_nanochat_effective_decay(
    *,
    end_step: int,
    warmup_steps: int = 40,
    warmdown_ratio: float = 0.65,
    final_lr_fraction: float = 0.05,
) -> float:
    """Schedule-level integrated decay of Nanochat's legacy linear/cosine path."""

    end = _require_int(end_step, "end_step", minimum=1)
    warmup = _require_int(warmup_steps, "warmup_steps", minimum=1)
    if warmup >= end:
        raise WSDScheduleError("warmup must finish before the run horizon")
    if not 0.0 < warmdown_ratio < 1.0:
        raise WSDScheduleError("warmdown_ratio must be between 0 and 1")
    if not 0.0 <= final_lr_fraction <= 1.0:
        raise WSDScheduleError("final_lr_fraction must be between 0 and 1")
    lr_values: list[float] = []
    wd_values: list[float] = []
    # Update indices are [0, end); the terminal checkpoint at end performs no
    # optimizer update and is intentionally excluded from the integral.
    for step in range(end):
        values = nanochat_linear_schedule_values(
            step,
            end_step=end,
            warmup_steps=warmup,
            warmdown_ratio=warmdown_ratio,
            final_lr_fraction=final_lr_fraction,
        )
        lr_values.append(values.lr_multiplier)
        wd_values.append(values.weight_decay_multiplier)
    return cumulative_effective_decay(lr_values, wd_values)


def wsd_effective_decay(
    *,
    end_step: int,
    warmup_steps: int = 40,
    cooldown_start_step: int | None,
    weight_decay_cooldown_policy: str,
    momentum_warmup_steps: int = 400,
) -> float:
    """Schedule-level integrated decay of this module's WSD path."""

    values = (
        wsd_schedule_values(
            step,
            end_step=end_step,
            warmup_steps=warmup_steps,
            cooldown_start_step=cooldown_start_step,
            weight_decay_cooldown_policy=weight_decay_cooldown_policy,
            momentum_warmup_steps=momentum_warmup_steps,
        )
        for step in range(end_step)
    )
    materialized = tuple(values)
    return cumulative_effective_decay(
        (value.lr_multiplier for value in materialized),
        (value.weight_decay_multiplier for value in materialized),
    )


def integrated_decay_matched_wsd_base(
    legacy_base_weight_decay: float,
    *,
    end_step: int,
    warmup_steps: int = 40,
    legacy_warmdown_ratio: float = 0.65,
    legacy_final_lr_fraction: float = 0.05,
    cooldown_start_step: int,
    weight_decay_cooldown_policy: str,
    momentum_warmup_steps: int = 400,
) -> float:
    """Return a WSD base WD with the same schedule-level integrated budget."""

    base = float(legacy_base_weight_decay)
    if base < 0.0:
        raise WSDScheduleError("legacy_base_weight_decay must be non-negative")
    legacy_budget = legacy_nanochat_effective_decay(
        end_step=end_step,
        warmup_steps=warmup_steps,
        warmdown_ratio=legacy_warmdown_ratio,
        final_lr_fraction=legacy_final_lr_fraction,
    )
    wsd_budget = wsd_effective_decay(
        end_step=end_step,
        warmup_steps=warmup_steps,
        cooldown_start_step=cooldown_start_step,
        weight_decay_cooldown_policy=weight_decay_cooldown_policy,
        momentum_warmup_steps=momentum_warmup_steps,
    )
    return base * legacy_budget / wsd_budget


__all__ = [
    "WSDMilestone",
    "WSDScheduleError",
    "WSDScheduleValues",
    "build_shared_wsd_milestones",
    "cumulative_effective_decay",
    "integrated_decay_matched_wsd_base",
    "legacy_nanochat_effective_decay",
    "nanochat_linear_schedule_values",
    "transfer_weight_decay_between_stages",
    "validate_weight_decay_proxy_candidates",
    "validate_wsd_schedule",
    "wsd_effective_decay",
    "wsd_schedule_values",
]
