"""Pure scoring/generation controls for the versioned nanochat lm-eval adapter."""

from collections.abc import Callable, Mapping, Sequence


EVALUATION_PROTOCOL_VERSION = "nanochat-lm-eval-v2-2026-11-09"


class ContinuationBoundaryError(ValueError):
    """A text continuation has no exact boundary in the joint tokenization."""


def generation_controls(value: object, *, cli_max_tokens: int) -> tuple[list[str], int]:
    """Read current lm-eval kwargs, retaining compatibility with legacy stop lists."""
    if cli_max_tokens <= 0:
        raise ValueError("max_gen_tokens must be positive")
    if isinstance(value, Mapping):
        raw_until = value.get("until", [])
        raw_max_tokens = value.get("max_gen_toks", cli_max_tokens)
    else:
        raw_until = value
        raw_max_tokens = cli_max_tokens
    if raw_until is None:
        until = []
    elif isinstance(raw_until, str):
        until = [raw_until]
    elif isinstance(raw_until, Sequence) and all(isinstance(s, str) for s in raw_until):
        until = list(raw_until)
    else:
        raise TypeError("until must be a string or sequence of strings")
    if isinstance(raw_max_tokens, bool) or not isinstance(raw_max_tokens, int):
        raise TypeError("max_gen_toks must be an integer")
    if raw_max_tokens <= 0:
        raise ValueError("max_gen_toks must be positive")
    return [s for s in until if s], min(cli_max_tokens, raw_max_tokens)


def trim_at_first_stop(text: str, until: Sequence[str]) -> tuple[str, bool]:
    positions = [position for stop in until if stop and (position := text.find(stop)) >= 0]
    if not positions:
        return text, False
    return text[: min(positions)], True


def encode_loglikelihood_pair(
    encode: Callable[[str], list[int]], context: str, continuation: str
) -> tuple[list[int], int]:
    """Move trailing context whitespace, then require an exact joint-token prefix.

    Reject cross-boundary merges instead of silently scoring an empty or wrong
    suffix. The encoder supplies the BOS token, including for empty contexts.
    """
    if not continuation:
        ids = encode(context)
        return ids, len(ids)
    trailing_whitespace = len(context) - len(context.rstrip())
    if trailing_whitespace:
        continuation = context[-trailing_whitespace:] + continuation
        context = context[:-trailing_whitespace]
    context_ids = encode(context)
    full_ids = encode(context + continuation)
    if full_ids[: len(context_ids)] != context_ids:
        raise ContinuationBoundaryError(
            "exact conditional scoring is impossible: tokenization crossed "
            "the context/continuation boundary"
        )
    continuation_start = len(context_ids)
    if continuation_start == len(full_ids):
        raise ContinuationBoundaryError("a non-empty continuation produced no continuation tokens")
    return full_ids, continuation_start
