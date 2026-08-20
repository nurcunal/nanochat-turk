from __future__ import annotations

import copy
import math

import pytest

from nanochat.wsd import (
    WSDScheduleError,
    build_shared_wsd_milestones,
    integrated_decay_matched_wsd_base,
    legacy_nanochat_effective_decay,
    nanochat_linear_schedule_values,
    transfer_weight_decay_between_stages,
    validate_weight_decay_proxy_candidates,
    validate_wsd_schedule,
    wsd_effective_decay,
    wsd_schedule_values,
)


SCALING_PARAMETERS = 1_677_724_672
GLOBAL_BATCH_TOKENS = 2_097_152


def test_d32_shared_plan_has_exact_12x_20x_40x_boundaries() -> None:
    milestones = build_shared_wsd_milestones(
        scaling_parameters=SCALING_PARAMETERS,
        global_batch_tokens=GLOBAL_BATCH_TOKENS,
    )
    assert [
        (
            item.scale,
            item.stable_parent_step,
            item.cooldown_steps,
            item.final_step,
            item.scheduled_tokens,
        )
        for item in milestones
    ] == [
        (12, 8_640, 960, 9_600, 20_132_659_200),
        (20, 14_400, 1_600, 16_000, 33_554_432_000),
        (40, 28_800, 3_200, 32_000, 67_108_864_000),
    ]


@pytest.mark.parametrize(
    ("end_step", "cooldown_start"),
    [(9_600, 8_640), (16_000, 14_400), (32_000, 28_800)],
)
@pytest.mark.parametrize("weight_decay_policy", ["constant", "linear_to_zero"])
def test_each_cooldown_branch_has_an_identical_stable_prefix(
    end_step: int, cooldown_start: int, weight_decay_policy: str
) -> None:
    probes = {0, 1, 39, 40, 399, 400, cooldown_start - 1}
    probes.update(range(0, cooldown_start, 137))
    for step in sorted(probes):
        trunk = wsd_schedule_values(
            step,
            end_step=28_800,
            warmup_steps=40,
            cooldown_start_step=None,
            weight_decay_cooldown_policy=weight_decay_policy,
        )
        branch = wsd_schedule_values(
            step,
            end_step=end_step,
            warmup_steps=40,
            cooldown_start_step=cooldown_start,
            weight_decay_cooldown_policy=weight_decay_policy,
        )
        assert branch == trunk


def test_wsd_terminal_and_boundary_values_are_exact() -> None:
    start = wsd_schedule_values(
        28_800,
        end_step=32_000,
        warmup_steps=40,
        cooldown_start_step=28_800,
        weight_decay_cooldown_policy="linear_to_zero",
    )
    last_update = wsd_schedule_values(
        31_999,
        end_step=32_000,
        warmup_steps=40,
        cooldown_start_step=28_800,
        weight_decay_cooldown_policy="linear_to_zero",
    )
    terminal = wsd_schedule_values(
        32_000,
        end_step=32_000,
        warmup_steps=40,
        cooldown_start_step=28_800,
        weight_decay_cooldown_policy="linear_to_zero",
    )
    assert start.lr_multiplier == 1.0
    assert start.muon_momentum == 0.97
    assert start.weight_decay_multiplier == 1.0
    assert last_update.lr_multiplier == pytest.approx(1 / 3_200)
    assert last_update.weight_decay_multiplier == pytest.approx(1 / 3_200)
    assert terminal.lr_multiplier == 0.0
    assert terminal.muon_momentum == 0.90
    assert terminal.weight_decay_multiplier == 0.0

    constant_terminal = wsd_schedule_values(
        32_000,
        end_step=32_000,
        warmup_steps=40,
        cooldown_start_step=28_800,
        weight_decay_cooldown_policy="constant",
    )
    assert constant_terminal.lr_multiplier == 0.0
    assert constant_terminal.muon_momentum == 0.90
    assert constant_terminal.weight_decay_multiplier == 1.0


def test_integrated_decay_match_is_deterministic_and_materially_lower() -> None:
    transferred = 0.03675007690415606
    legacy_budget = legacy_nanochat_effective_decay(end_step=32_000)
    wsd_budget = wsd_effective_decay(
        end_step=32_000,
        cooldown_start_step=28_800,
        weight_decay_cooldown_policy="linear_to_zero",
    )
    matched = integrated_decay_matched_wsd_base(
        transferred,
        end_step=32_000,
        cooldown_start_step=28_800,
        weight_decay_cooldown_policy="linear_to_zero",
    )
    assert legacy_budget == pytest.approx(14_486.022451699691)
    assert wsd_budget == pytest.approx(29_847.66671875)
    assert matched == pytest.approx(0.01783598175869708)
    assert matched * wsd_budget == pytest.approx(transferred * legacy_budget)
    assert matched < transferred * 0.5


def test_constant_wd_integrated_decay_match_is_deterministic() -> None:
    transferred = 0.03675007690415606
    budget = wsd_effective_decay(
        end_step=32_000,
        cooldown_start_step=28_800,
        weight_decay_cooldown_policy="constant",
    )
    matched = integrated_decay_matched_wsd_base(
        transferred,
        end_step=32_000,
        cooldown_start_step=28_800,
        weight_decay_cooldown_policy="constant",
    )
    assert budget == pytest.approx(30_381.0)
    assert matched == pytest.approx(0.017522874136312004)
    assert matched * budget == pytest.approx(
        transferred * legacy_nanochat_effective_decay(end_step=32_000)
    )


def test_nanochat_linear_control_matches_pinned_upstream_equations() -> None:
    end = 4_200
    warmup = 40
    warmdown_steps = round(0.65 * end)
    warmdown_start = end - warmdown_steps
    for step in range(end + 1):
        values = nanochat_linear_schedule_values(step, end_step=end)
        if step < warmup:
            expected_lr = (step + 1) / warmup
        elif step <= warmdown_start:
            expected_lr = 1.0
        else:
            progress = (end - step) / warmdown_steps
            expected_lr = progress * 1.0 + (1 - progress) * 0.05
        if step < 400:
            fraction = step / 400
            expected_momentum = (1 - fraction) * 0.85 + fraction * 0.97
        elif step >= warmdown_start:
            progress = (step - warmdown_start) / warmdown_steps
            expected_momentum = 0.97 * (1 - progress) + 0.90 * progress
        else:
            expected_momentum = 0.97
        expected_wd = 0.5 * (1 + math.cos(math.pi * step / end))
        assert values.lr_multiplier == expected_lr
        assert values.muon_momentum == expected_momentum
        assert values.weight_decay_multiplier == expected_wd


def test_weight_decay_transfer_uses_exact_nanochat_width_batch_rule() -> None:
    d12_scaling_parameters = 110_100_912
    d12_batch = 524_288
    d32_batch = 2_097_152
    d32 = transfer_weight_decay_between_stages(
        0.28,
        source_scaling_parameters=d12_scaling_parameters,
        source_global_batch_tokens=d12_batch,
        target_scaling_parameters=SCALING_PARAMETERS,
        target_global_batch_tokens=d32_batch,
    )
    assert d32 == pytest.approx(0.03675007690415606)
    d12_round_trip = transfer_weight_decay_between_stages(
        d32,
        source_scaling_parameters=SCALING_PARAMETERS,
        source_global_batch_tokens=d32_batch,
        target_scaling_parameters=d12_scaling_parameters,
        target_global_batch_tokens=d12_batch,
    )
    assert d12_round_trip == pytest.approx(0.28)
    assert transfer_weight_decay_between_stages(
        0.0,
        source_scaling_parameters=SCALING_PARAMETERS,
        source_global_batch_tokens=d32_batch,
        target_scaling_parameters=d12_scaling_parameters,
        target_global_batch_tokens=d12_batch,
    ) == 0.0


def _proxy_candidates() -> list[dict]:
    stages = {
        "d12": (110_100_912, 524_288),
        "d20": (435_160_240, 1_048_576),
    }

    def candidate(candidate_id: str, wd: float, policy: str) -> dict:
        return {
            "id": candidate_id,
            "production_base_weight_decay": wd,
            "weight_decay_cooldown_policy": policy,
            "eligible_for_production": candidate_id != "upstream_92d63d4e_control",
            "stage_effective_weight_decays": [
                {
                    "stage_id": stage_id,
                    "scaling_parameters": parameters,
                    "global_batch_tokens": batch,
                    "effective_base_weight_decay": (
                        transfer_weight_decay_between_stages(
                            wd,
                            source_scaling_parameters=SCALING_PARAMETERS,
                            source_global_batch_tokens=GLOBAL_BATCH_TOKENS,
                            target_scaling_parameters=parameters,
                            target_global_batch_tokens=batch,
                        )
                    ),
                }
                for stage_id, (parameters, batch) in stages.items()
            ],
        }

    return [
        candidate("upstream_92d63d4e_control", 0.03675007690415606, "cosine_full_horizon"),
        candidate("legacy_transferred", 0.03675007690415606, "linear_to_zero"),
        candidate("half_transferred_constant_cooldown_wd", 0.01837503845207803, "constant"),
        candidate("no_weight_decay", 0.0, "linear_to_zero"),
    ]


def test_proxy_candidate_validator_recomputes_every_stage_transfer() -> None:
    candidates = _proxy_candidates()
    validate_weight_decay_proxy_candidates(
        candidates,
        production_scaling_parameters=SCALING_PARAMETERS,
        production_global_batch_tokens=GLOBAL_BATCH_TOKENS,
        accepted_candidate_id="half_transferred_constant_cooldown_wd",
        accepted_base_weight_decay=0.01837503845207803,
        accepted_weight_decay_cooldown_policy="constant",
    )
    tampered = copy.deepcopy(candidates)
    tampered[1]["stage_effective_weight_decays"][0][
        "effective_base_weight_decay"
    ] += 1e-4
    with pytest.raises(WSDScheduleError, match="transfer rule"):
        validate_weight_decay_proxy_candidates(
            tampered,
            production_scaling_parameters=SCALING_PARAMETERS,
            production_global_batch_tokens=GLOBAL_BATCH_TOKENS,
            accepted_candidate_id="half_transferred_constant_cooldown_wd",
            accepted_base_weight_decay=0.01837503845207803,
            accepted_weight_decay_cooldown_policy="constant",
        )


def test_wsd_rejects_non_ten_percent_cooldown_when_required() -> None:
    with pytest.raises(WSDScheduleError, match="required fraction"):
        validate_wsd_schedule(
            end_step=32_000,
            warmup_steps=40,
            momentum_warmup_steps=400,
            cooldown_start_step=29_000,
            required_cooldown_fraction=0.10,
        )
