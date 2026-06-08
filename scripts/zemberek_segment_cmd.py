#!/usr/bin/env python3
"""Batch command wrapper for zemberek-python surface segmentation.

The command reads one word per stdin line and writes one surface segmentation
per stdout line. It is intended for use through:

    ZEMBEREK_SEGMENT_CMD="python scripts/zemberek_segment_cmd.py"

zemberek-python currently works more reliably under Python 3.12 than Python
3.13, so use a compatible virtualenv if the default interpreter fails.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import warnings


warnings.filterwarnings("ignore")

with contextlib.redirect_stdout(sys.stderr):
    from zemberek import TurkishMorphology


def analysis_to_surface(analysis: object) -> str:
    text = str(analysis)
    if "]" in text:
        text = text.split("]", 1)[1].strip()
    if "+" not in text:
        return text

    pieces = []
    for chunk in text.split("+"):
        chunk = chunk.strip()
        if not chunk:
            continue
        pieces.append(chunk.split(":", 1)[0].strip() or chunk)
    return "+".join(pieces)


def main() -> None:
    logging.getLogger("zemberek").disabled = True
    logging.getLogger("zemberek.morphology").disabled = True
    logging.getLogger("zemberek.morphology.turkish_morphology").disabled = True
    with contextlib.redirect_stdout(sys.stderr):
        morphology = TurkishMorphology.create_with_defaults()

    for raw in sys.stdin:
        word = raw.strip()
        if not word:
            print()
            continue
        try:
            analysis = morphology.analyze(word)
            analyses = list(getattr(analysis, "analysis_results", analysis))
            print(analysis_to_surface(analyses[0]) if analyses else word)
        except Exception:
            print(word)


if __name__ == "__main__":
    main()
