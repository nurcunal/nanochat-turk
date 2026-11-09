"""MorphBPE training helpers.

Paper-faithful MorphBPE uses morphology only while learning BPE merges. The
saved tokenizer remains a standard raw-text BPE tokenizer, so inference does
not require runtime segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import regex

from nanochat.morphology.boundary import MORPHEME_BOUNDARY
from nanochat.tokenizer import SPLIT_PATTERN

DEFAULT_SPLIT_RE = regex.compile(SPLIT_PATTERN)


@dataclass
class MorphBPEIteratorStats:
    """Statistics for the MorphBPE training-stream transform."""

    docs: int = 0
    docs_with_boundary: int = 0
    visible_chars: int = 0
    regex_chunks: int = 0
    training_chunks: int = 0
    boundary_splits: int = 0


def strip_boundaries_with_offsets(
    text: str,
    boundary: str = MORPHEME_BOUNDARY,
) -> tuple[str, tuple[int, ...]]:
    """Remove boundary markers and return their visible-text char offsets."""

    if not boundary:
        return text, ()

    visible: list[str] = []
    offsets: list[int] = []
    i = 0
    while i < len(text):
        if text.startswith(boundary, i):
            offsets.append(len(visible))
            i += len(boundary)
        else:
            visible.append(text[i])
            i += 1
    return "".join(visible), tuple(offsets)


def iter_morphbpe_training_chunks(
    text: str,
    boundary: str = MORPHEME_BOUNDARY,
    pattern: str = SPLIT_PATTERN,
    stats: MorphBPEIteratorStats | None = None,
) -> Iterable[str]:
    """Yield BPE training chunks that never cross a marked morpheme boundary."""

    visible, boundary_offsets = strip_boundaries_with_offsets(text, boundary)
    if stats is not None:
        stats.docs += 1
        stats.visible_chars += len(visible)
        if boundary_offsets:
            stats.docs_with_boundary += 1

    boundary_set = set(boundary_offsets)
    split_re = DEFAULT_SPLIT_RE if pattern == SPLIT_PATTERN else regex.compile(pattern)
    for match in split_re.finditer(visible):
        chunk_start, chunk_end = match.span()
        if stats is not None:
            stats.regex_chunks += 1

        cuts = sorted(
            offset for offset in boundary_set
            if chunk_start < offset < chunk_end
        )
        cursor = chunk_start
        for cut in cuts:
            if cursor < cut:
                yield visible[cursor:cut]
                if stats is not None:
                    stats.training_chunks += 1
            cursor = cut
            if stats is not None:
                stats.boundary_splits += 1
        if cursor < chunk_end:
            yield visible[cursor:chunk_end]
            if stats is not None:
                stats.training_chunks += 1


def iter_morphbpe_training_stream(
    texts: Iterable[str],
    boundary: str = MORPHEME_BOUNDARY,
    stats: MorphBPEIteratorStats | None = None,
) -> Iterable[str]:
    """Transform boundary-marked documents into MorphBPE training chunks."""

    for text in texts:
        yield from iter_morphbpe_training_chunks(
            text,
            boundary=boundary,
            stats=stats,
        )
