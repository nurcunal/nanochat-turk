"""Fixed-manifest, tokenizer-normalized evaluation for strict WSD runs."""

import math
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class LossEvaluation:
    """Globally reduced sufficient statistics and derived validation metrics."""

    all_target_nats: float
    all_target_count: int
    payload_nats: float
    payload_target_count: int
    payload_bytes: int
    all_target_nll: float
    payload_token_nll: float
    bpb: float

    def to_dict(self):
        return asdict(self)


@torch.no_grad()
def evaluate_loss(model, batches, steps, token_bytes) -> LossEvaluation:
    """
    Instead of the naive 'mean loss', this function returns the bits per byte (bpb),
    which is a tokenization vocab size-independent metric, meaning you are still comparing
    apples:apples if you change the vocab size. The way this works is that instead of just
    calculating the average loss as usual, you calculate the sum loss, and independently
    also the sum bytes (of all the target tokens), and divide. This normalizes the loss by
    the number of bytes that the target tokens represent.

    The added complexity is so that:
    1) All "normal" tokens are normalized by the length of the token in bytes
    2) No special tokens (e.g. <|bos|>) are included in the metric - they are masked out.
    3) No actively masked tokens (using ignore_index of e.g. -1) are included in the metric.

    In addition to evaluate_loss, we need the token_bytes tensor:
    It is a 1D tensor of shape (vocab_size,), indicating the number of bytes for
    each token id, or 0 if the token is to not be counted (e.g. special tokens).
    """
    if steps is not None and (
        isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0
    ):
        raise ValueError("steps must be a positive integer or None for a finite loader")
    if token_bytes.ndim != 1:
        raise ValueError("token_bytes must be a one-dimensional tensor")
    if hasattr(model, "get_device"):
        device = model.get_device()
    elif hasattr(model, "module") and hasattr(model.module, "get_device"):
        device = model.module.get_device()
    else:
        try:
            device = next(model.parameters()).device
        except (AttributeError, StopIteration) as exc:
            raise ValueError("cannot determine evaluation model device") from exc
    accumulation_dtype = torch.float32 if device.type == "mps" else torch.float64
    totals = torch.zeros(5, dtype=accumulation_dtype, device=device)
    byte_table = token_bytes.to(device=device)
    batch_iter = iter(batches)
    batches_evaluated = 0
    while steps is None or batches_evaluated < steps:
        try:
            batch = next(batch_iter)
        except StopIteration:
            if steps is not None:
                raise ValueError(
                    "evaluation loader ended before the requested number of steps"
                )
            break
        batches_evaluated += 1
        x, y = batch[:2]
        loss2d = model(x, y, loss_reduction='none') # (B, T)
        losses = loss2d.reshape(-1).to(accumulation_dtype)
        targets = y.reshape(-1)
        valid = targets >= 0
        safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
        if valid.any() and int(safe_targets.max().item()) >= byte_table.numel():
            raise ValueError("evaluation target ID is outside token_bytes")
        lengths = torch.where(
            valid,
            byte_table[safe_targets],
            torch.zeros_like(targets, dtype=byte_table.dtype),
        )
        payload = valid & (lengths > 0)
        totals[0] += torch.where(valid, losses, 0.0).sum()  # all-target nats
        totals[1] += valid.sum()  # all-target count
        totals[2] += torch.where(payload, losses, 0.0).sum()  # payload-only nats
        totals[3] += payload.sum()  # payload target count
        totals[4] += lengths.sum()  # payload bytes

    if batches_evaluated == 0:
        raise ValueError("evaluation loader produced no batches")
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    values = totals.cpu().tolist()
    all_nats, all_count_f, payload_nats, payload_count_f, payload_bytes_f = values
    all_count = int(all_count_f)
    payload_count = int(payload_count_f)
    payload_bytes = int(payload_bytes_f)
    return LossEvaluation(
        all_target_nats=all_nats,
        all_target_count=all_count,
        payload_nats=payload_nats,
        payload_target_count=payload_count,
        payload_bytes=payload_bytes,
        all_target_nll=all_nats / all_count if all_count else float("inf"),
        payload_token_nll=(
            payload_nats / payload_count if payload_count else float("inf")
        ),
        bpb=(
            payload_nats / (math.log(2) * payload_bytes)
            if payload_bytes
            else float("inf")
        ),
    )


@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """Compatibility scalar wrapper around :func:`evaluate_loss`."""

    return evaluate_loss(model, batches, steps, token_bytes).bpb


__all__ = ["LossEvaluation", "evaluate_bpb", "evaluate_loss"]
