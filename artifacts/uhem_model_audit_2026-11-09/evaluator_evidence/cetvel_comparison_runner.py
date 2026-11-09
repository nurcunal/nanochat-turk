"""Run the frozen CETVEL evaluator with standard lm-eval scoring semantics.

The historical nanochat adapter had two protocol deviations that matter when
comparing against Hugging Face models:

* generation kwargs were interpreted as a list instead of a mapping, so task
  stop strings and task-level generation limits were ignored;
* continuation scoring used the separately-tokenized context length without
  first moving trailing whitespace to the continuation or verifying an exact
  token boundary.

This runner patches only those adapter methods at process start.  It leaves the
frozen CETVEL tasks, lm-eval checkout, checkpoint loader, and metric code intact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import runpy
import time

import torch

from nanochat.common import print0
from nanochat.lm_eval_nanochat import NanochatLM


class ContinuationBoundaryError(ValueError):
    """Raised when a request has no exact context/continuation token boundary."""


def _generation_controls(
    value: object, *, cli_max_tokens: int
) -> tuple[list[str], int]:
    """Normalize lm-eval generation kwargs without exceeding the CLI cap."""

    if isinstance(value, Mapping):
        raw_until = value.get("until", [])
        raw_max_tokens = value.get("max_gen_toks", cli_max_tokens)
    else:
        raw_until = value
        raw_max_tokens = cli_max_tokens

    if raw_until is None:
        until: list[str] = []
    elif isinstance(raw_until, str):
        until = [raw_until]
    elif isinstance(raw_until, Sequence):
        until = [item for item in raw_until if isinstance(item, str)]
    else:
        raise TypeError(f"unsupported CETVEL until value: {type(raw_until).__name__}")
    until = [item for item in until if item]

    if isinstance(raw_max_tokens, bool):
        raise TypeError("max_gen_toks must be an integer")
    try:
        task_max_tokens = int(raw_max_tokens)
    except (TypeError, ValueError) as exc:
        raise TypeError("max_gen_toks must be an integer") from exc
    if task_max_tokens <= 0:
        raise ValueError("max_gen_toks must be positive")
    return until, min(cli_max_tokens, task_max_tokens)


def _trim_at_first_stop(text: str, until: Sequence[str]) -> tuple[str, bool]:
    positions = [position for stop in until if (position := text.find(stop)) >= 0]
    if not positions:
        return text, False
    return text[: min(positions)], True


def _encode_loglikelihood_pair_fixed(
    self: NanochatLM, context: str, continuation: str
) -> tuple[list[int], int]:
    """Return the joint encoding and its exact continuation token boundary."""

    trailing_whitespace = len(context) - len(context.rstrip())
    if trailing_whitespace:
        continuation = context[-trailing_whitespace:] + continuation
        context = context[:-trailing_whitespace]

    context_ids = self._encode(context)
    full_ids = self._encode(context + continuation)
    if full_ids[: len(context_ids)] != context_ids:
        common_prefix_tokens = 0
        for context_id, full_id in zip(context_ids, full_ids):
            if context_id != full_id:
                break
            common_prefix_tokens += 1
        raise ContinuationBoundaryError(
            "exact conditional scoring is impossible because tokenization "
            "crossed the context/continuation boundary; "
            f"common_prefix_tokens={common_prefix_tokens}, "
            f"context_tokens={len(context_ids)}, full_tokens={len(full_ids)}"
        )

    continuation_start = len(context_ids)
    if continuation and continuation_start == len(full_ids):
        raise ContinuationBoundaryError(
            "a non-empty continuation produced no continuation tokens"
        )
    return full_ids, continuation_start


@torch.inference_mode()
def _loglikelihood_fixed(self: NanochatLM, requests) -> list[tuple[float, bool]]:
    outputs: list[tuple[float, bool]] = []
    maximum_joint_tokens = 0
    truncated_requests = 0
    trailing_whitespace_requests = 0
    for request in requests:
        context, continuation = request.args
        if context and context[-1].isspace():
            trailing_whitespace_requests += 1
        full_ids, continuation_start = _encode_loglikelihood_pair_fixed(
            self, context, continuation
        )
        maximum_joint_tokens = max(maximum_joint_tokens, len(full_ids))
        continuation_tokens = len(full_ids) - continuation_start
        if continuation_tokens > self.model_max_len:
            raise ValueError(
                "continuation exceeds the model scoring window: "
                f"continuation_tokens={continuation_tokens}, "
                f"model_max_len={self.model_max_len}"
            )

        # lm-eval permits max_length input positions and therefore keeps one
        # additional joint token for the shifted target sequence.
        drop = max(0, len(full_ids) - (self.model_max_len + 1))
        if drop > 0:
            truncated_requests += 1
            full_ids = full_ids[drop:]
            continuation_start = max(1, continuation_start - drop)

        if continuation_start >= len(full_ids):
            outputs.append((0.0, True))
            continue

        input_ids = torch.tensor(
            full_ids[:-1], dtype=torch.long, device=self.device
        ).unsqueeze(0)
        target_ids = torch.tensor(
            full_ids[1:], dtype=torch.long, device=self.device
        ).unsqueeze(0)
        logits = self.model(input_ids)
        logprobs = torch.log_softmax(logits, dim=-1)

        start_position = continuation_start - 1
        continuation_targets = target_ids[:, start_position:]
        continuation_logprobs = logprobs[:, start_position:]
        token_logprobs = continuation_logprobs.gather(
            -1, continuation_targets.unsqueeze(-1)
        ).squeeze(-1)
        total_logprob = float(token_logprobs.sum().item())

        greedy = continuation_logprobs.argmax(dim=-1)
        is_greedy = bool((greedy == continuation_targets).all().item())
        outputs.append((total_logprob, is_greedy))
    print0(
        "[CETVEL comparison loglikelihood] "
        f"requests={len(requests)} boundary_failures=0 "
        f"trailing_whitespace_requests={trailing_whitespace_requests} "
        f"truncated_requests={truncated_requests} "
        f"maximum_joint_tokens={maximum_joint_tokens}",
        flush=True,
    )
    return outputs


@torch.inference_mode()
def _generate_until_fixed(self: NanochatLM, requests) -> list[str]:
    outputs: list[str] = []
    progress_enabled = os.environ.get("CETVEL_GENERATION_PROGRESS", "0") == "1"
    progress_every = max(
        1, int(os.environ.get("CETVEL_GENERATION_PROGRESS_EVERY", "1"))
    )
    total = len(requests)
    started = time.monotonic()
    if progress_enabled:
        task_name = self._request_task_name(requests[0]) if total else "unknown"
        print0(
            f"[CETVEL comparison generation] START task={task_name} total={total}",
            flush=True,
        )

    for index, request in enumerate(requests, start=1):
        context, generation_kwargs = request.args
        until, max_tokens = _generation_controls(
            generation_kwargs, cli_max_tokens=self.max_gen_tokens
        )
        prompt_ids = self._encode(context)
        # Match HFLM: reserve the requested decoding budget inside the model's
        # context limit and left-truncate the prompt when necessary.
        max_prompt_tokens = max(1, self.model_max_len - max_tokens)
        if len(prompt_ids) > max_prompt_tokens:
            prompt_ids = prompt_ids[-max_prompt_tokens:]
        generated: list[int] = []
        stopped = False
        rendered = ""
        terminal_token_ids = {self.bos_id}
        try:
            terminal_token_ids.add(
                int(self.tokenizer.encode_special("<|assistant_end|>"))
            )
        except Exception:
            pass
        for token_column, _token_masks in self.engine.generate(
            prompt_ids,
            num_samples=1,
            max_tokens=max_tokens,
            temperature=0.0,
        ):
            token = int(token_column[0])
            if token in terminal_token_ids:
                stopped = True
                break
            generated.append(token)
            rendered = self.tokenizer.decode(generated)
            rendered, stopped = _trim_at_first_stop(rendered, until)
            if stopped:
                break
        if not stopped:
            rendered = self.tokenizer.decode(generated)
            rendered, stopped = _trim_at_first_stop(rendered, until)
        outputs.append(rendered)

        if progress_enabled and (index % progress_every == 0 or index == total):
            elapsed = max(1e-9, time.monotonic() - started)
            task_name = self._request_task_name(request)
            print0(
                "[CETVEL comparison generation] "
                f"task={task_name} done={index}/{total} "
                f"generated_tokens={len(generated)} stopped={int(stopped)} "
                f"elapsed={elapsed:.1f}s rate={index / elapsed:.2f}/s",
                flush=True,
            )
    return outputs


def main() -> None:
    evaluator = Path(os.environ["CETVEL_LEGACY_EVAL_SCRIPT"]).resolve()
    if not evaluator.is_file():
        raise FileNotFoundError(f"frozen CETVEL evaluator is missing: {evaluator}")

    NanochatLM.loglikelihood = _loglikelihood_fixed
    NanochatLM.generate_until = _generate_until_fixed
    if os.environ.get("CETVEL_FORCE_SDPA", "0") == "1":
        import nanochat.flash_attention as attention_runtime

        attention_runtime.USE_FA3 = False
        print0("CETVEL attention backend forced to SDPA", flush=True)
    print0(
        "CETVEL comparison patches active: exact continuation boundary, "
        "generation kwargs, stop trimming",
        flush=True,
    )
    runpy.run_path(str(evaluator), run_name="__main__")


if __name__ == "__main__":
    main()
