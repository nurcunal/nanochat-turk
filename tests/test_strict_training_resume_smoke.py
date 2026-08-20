from __future__ import annotations

import copy
import hashlib
import random
from pathlib import Path

import torch

from nanochat.strict_checkpoint import (
    build_strict_checkpoint_identity,
    capture_rank_rng_state,
    load_strict_checkpoint,
    restore_rank_rng_state,
    save_strict_checkpoint,
)
from nanochat.strict_dataloader import StatefulBestFitLoader
from nanochat.optim import MuonAdamW
from nanochat.wsd import wsd_schedule_values
from strict_loader_fixtures import (
    STUDY_HASH,
    TOKENIZER_HASH,
    strict_dataset,
    strict_training_plan,
)


class _TinyTrainingState(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adam_matrix = torch.nn.Parameter(
            torch.arange(8, dtype=torch.float32).reshape(4, 2) / 20
        )
        self.muon_a = torch.nn.Parameter(
            torch.arange(4, dtype=torch.float32).reshape(2, 2) / 10
        )
        self.muon_b = torch.nn.Parameter(
            torch.arange(4, 8, dtype=torch.float32).reshape(2, 2) / 10
        )


def _optimizer(model: _TinyTrainingState) -> MuonAdamW:
    return MuonAdamW(
        [
            {
                "kind": "adamw",
                "params": [model.adam_matrix],
                "lr": 0.003,
                "initial_lr": 0.003,
                "betas": (0.8, 0.96),
                "eps": 1e-10,
                "weight_decay": 0.01,
            },
            {
                "kind": "muon",
                "params": [model.muon_a, model.muon_b],
                "lr": 0.01,
                "initial_lr": 0.01,
                "momentum": 0.95,
                "ns_steps": 5,
                "beta2": 0.9,
                "weight_decay": 0.02,
            },
        ]
    )


def _nested_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right, strict=True):
            _nested_equal(left_value, right_value)
    else:
        assert left == right


def _train_update(model, optimizer, loader, update_index: int) -> dict:
    x, y, loader_state = next(loader)
    torch_noise = torch.rand(())
    python_noise = random.random()
    batch_signal = (x.float().sum() + y.float().sum()) / 1_000
    loss = (
        model.adam_matrix.square().sum() * (0.01 + batch_signal)
        + model.muon_a.square().sum() * (0.02 + torch_noise / 100)
        + model.muon_b.square().sum() * (0.03 + python_noise / 100)
    )
    loss.backward()
    schedule = wsd_schedule_values(
        update_index,
        end_step=10,
        warmup_steps=2,
        momentum_warmup_steps=3,
        cooldown_start_step=None,
        weight_decay_cooldown_policy="constant",
    )
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * schedule.lr_multiplier
        if group["kind"] == "muon":
            group["momentum"] = schedule.muon_momentum
            group["weight_decay"] = 0.02 * schedule.weight_decay_multiplier
    optimizer.step()
    model.zero_grad(set_to_none=True)
    return {
        "x": x.clone(),
        "y": y.clone(),
        "loader": loader_state,
        "loss": loss.detach().clone(),
        "torch_noise": torch_noise.clone(),
        "python_noise": python_noise,
    }


def _identity(updates: int) -> dict:
    empty_hash = hashlib.sha256(b"").hexdigest()
    return build_strict_checkpoint_identity(
        study_id="strict-resume-smoke",
        run_id="forced-boundary",
        study_manifest_sha256="1" * 64,
        run_sha256="2" * 64,
        tokenizer_artifact_sha256=TOKENIZER_HASH,
        exposure_plan_sha256="4" * 64,
        optimizer_audit={"implementation": "nanochat.optim.MuonAdamW"},
        curve_log_state={
            "event_count": updates,
            "last_event_sha256": "5" * 64 if updates else None,
            "last_updates_completed": updates,
            "file_sha256": empty_hash,
        },
    )


def test_forced_checkpoint_resume_reproduces_the_exact_next_update(
    tmp_path: Path,
) -> None:
    (tmp_path / "data").mkdir()
    dataset, _exposure, tokenizer = strict_dataset(
        tmp_path / "data", ("abcdef", "çab", "fed", "cab")
    )
    loader_kwargs = {
        "data_dir": tmp_path / "data",
        "token_bytes": tokenizer.token_bytes(),
        "study_sha256": STUDY_HASH,
        "tokenizer_sha256": TOKENIZER_HASH,
        "device": "cpu",
        "rank": 0,
        "world_size": 1,
        "buffer_size": 3,
        "exposure_plan": strict_training_plan(dataset),
    }

    random.seed(1_729)
    torch.manual_seed(2_718)
    model = _TinyTrainingState()
    optimizer = _optimizer(model)
    loader = StatefulBestFitLoader(tokenizer, 1, 4, **loader_kwargs)
    _train_update(model, optimizer, loader, update_index=0)

    boundary_state = loader.state()
    rank_rng = {
        "seed_plan": {"fixture": "rank-0"},
        "state": capture_rank_rng_state("cpu"),
    }
    identity = _identity(updates=1)
    save_strict_checkpoint(
        tmp_path / "checkpoints",
        1,
        model.state_dict(),
        optimizer.state_dict(),
        {
            "step": 1,
            "updates_completed": 1,
            "loop_state": {"smooth_train_loss": 0.25},
        },
        loader_state=boundary_state,
        rng_state=rank_rng,
        rank=0,
        expected_world_size=1,
        identity=identity,
        updates_completed=1,
    )

    expected = _train_update(model, optimizer, loader, update_index=1)
    expected_model = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())

    payload = load_strict_checkpoint(
        tmp_path / "checkpoints",
        1,
        "cpu",
        rank=0,
        expected_world_size=1,
        expected_identity=identity,
        expected_updates_completed=1,
    )
    resumed_model = _TinyTrainingState()
    resumed_model.load_state_dict(payload.model_data, strict=True, assign=False)
    resumed_optimizer = _optimizer(resumed_model)
    resumed_optimizer.load_state_dict(payload.optimizer_data)
    resumed_loader = StatefulBestFitLoader(
        tokenizer,
        1,
        4,
        resume_state=payload.loader_state,
        restore_rng=False,
        **loader_kwargs,
    )
    # This mirrors base_train's loader construction/state audit before the
    # checkpointed stochastic streams are restored.
    resumed_loader.state()
    assert payload.rng_state["seed_plan"] == {"fixture": "rank-0"}
    restore_rank_rng_state(payload.rng_state["state"], "cpu")
    actual = _train_update(
        resumed_model, resumed_optimizer, resumed_loader, update_index=1
    )

    assert torch.equal(actual["x"], expected["x"])
    assert torch.equal(actual["y"], expected["y"])
    assert torch.equal(actual["loss"], expected["loss"])
    assert torch.equal(actual["torch_noise"], expected["torch_noise"])
    assert actual["python_noise"] == expected["python_noise"]
    for field in ("position", "buffer", "totals", "next_batch_index"):
        assert actual["loader"][field] == expected["loader"][field]
    _nested_equal(resumed_model.state_dict(), expected_model)
    _nested_equal(resumed_optimizer.state_dict(), expected_optimizer)
    assert payload.meta_data["loop_state"] == {"smooth_train_loss": 0.25}
