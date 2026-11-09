"""Run the historical CETVEL evaluator with corrected generation semantics.

The June 2026 adapter treated lm-eval's generation kwargs mapping as a list of
stop strings.  CETVEL therefore ignored each task's actual ``until`` strings.
This runner keeps the historical evaluator/runtime while replacing only
``NanochatLM.generate_until`` and, when requested, fixing attention to SDPA.
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


def _generation_controls(
    value: object, *, cli_max_tokens: int
) -> tuple[list[str], int]:
    """Normalize lm-eval generation kwargs without exceeding the CLI cap."""

    if isinstance(value, Mapping):
        raw_until = value.get("until", [])
        raw_max_tokens = value.get("max_gen_toks", cli_max_tokens)
    else:
        # Compatibility for the oldest lm-eval request representation.
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
            f"[CETVEL generation fixed] START task={task_name} total={total}",
            flush=True,
        )

    for index, request in enumerate(requests, start=1):
        context, generation_kwargs = request.args
        until, max_tokens = _generation_controls(
            generation_kwargs, cli_max_tokens=self.max_gen_tokens
        )
        prompt_ids = self._encode(context)
        generated: list[int] = []
        stopped = False
        rendered = ""
        for token_column, _token_masks in self.engine.generate(
            prompt_ids,
            num_samples=1,
            max_tokens=max_tokens,
            temperature=0.0,
        ):
            token = int(token_column[0])
            if token == self.bos_id:
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
                "[CETVEL generation fixed] "
                f"task={task_name} done={index}/{total} "
                f"generated_tokens={len(generated)} stopped={int(stopped)} "
                f"elapsed={elapsed:.1f}s rate={index / elapsed:.2f}/s",
                flush=True,
            )
    return outputs


def main() -> None:
    legacy_script = Path(os.environ["CETVEL_LEGACY_EVAL_SCRIPT"]).resolve()
    if not legacy_script.is_file():
        raise FileNotFoundError(f"historical CETVEL evaluator is missing: {legacy_script}")

    NanochatLM.generate_until = _generate_until_fixed
    if os.environ.get("CETVEL_FORCE_SDPA", "0") == "1":
        import nanochat.flash_attention as attention_runtime

        attention_runtime.USE_FA3 = False
        print0("CETVEL attention backend forced to SDPA", flush=True)
    print0("CETVEL generation kwargs/stop handling patch active", flush=True)
    runpy.run_path(str(legacy_script), run_name="__main__")


if __name__ == "__main__":
    main()
