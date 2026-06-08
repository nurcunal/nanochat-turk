#!/usr/bin/env python3
"""Batch command wrapper for the TurkishDelight morpheme segmenter.

The wrapper is intentionally thin: it loads the upstream TurkishDelight joint
model, reads one word per stdin line, and writes one JSON list of surface
morpheme pieces per stdout line. It is meant to be used through
``TDELIGHT_SEGMENT_CMD`` with ``scripts/morph_benchmark.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import unicodedata
from pathlib import Path
from typing import Any


DEFAULT_MODEL_NAME = (
    "Turkish-jointAll-MTAG_COMP=w_sum-MORPH_COMP=w_sum-POS_COMP=w_sum-"
    "COMP_ALPHA=0.1-trialmodel"
)


def _default_repo_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "dev-ignore/vendor/turkish-delight-nlp-api"


def _load_runtime(repo_dir: Path, model_name: str) -> tuple[Any, Any]:
    sys.path.insert(0, str(repo_dir))
    from turkishdelightnlp.models.joint.runtime import load_model, predict_morphemes

    model_path = repo_dir / "data" / model_name
    model_opt_path = repo_dir / "data" / f"{model_name}.params"
    if not model_path.exists() or not model_opt_path.exists():
        raise FileNotFoundError(
            "TurkishDelight model files were not found. Expected "
            f"{model_path} and {model_opt_path}."
        )

    with contextlib.redirect_stdout(sys.stderr):
        model = load_model(str(model_path), str(model_opt_path))
    return model, predict_morphemes


def _pieces_from_response(response: dict[str, Any], word: str) -> list[str]:
    morphemes = response.get("morphemes", {})
    if not isinstance(morphemes, dict):
        return []

    pieces = morphemes.get(word)
    if pieces is None and len(morphemes) == 1:
        pieces = next(iter(morphemes.values()))
    if isinstance(pieces, str):
        pieces = pieces.replace("+", " ").split()
    if not isinstance(pieces, list):
        return []

    normalized = _attach_leading_combining_marks(
        [str(piece) for piece in pieces if str(piece)]
    )
    return _repair_surface_pieces(word, normalized)


def _attach_leading_combining_marks(pieces: list[str]) -> list[str]:
    repaired: list[str] = []
    for piece in pieces:
        cursor = 0
        while cursor < len(piece) and unicodedata.combining(piece[cursor]):
            cursor += 1
        if cursor > 0 and repaired:
            repaired[-1] += piece[:cursor]
        rest = piece[cursor:]
        if rest:
            repaired.append(rest)
    return repaired


def _match_key(text: str) -> str:
    text = text.replace("\u2019", "'")
    return unicodedata.normalize("NFC", text).casefold()


def _repair_surface_pieces(word: str, pieces: list[str]) -> list[str]:
    if "".join(pieces) == word:
        return pieces

    repaired = []
    cursor = 0
    for piece in pieces:
        target = _match_key(piece)
        matched = False
        for end in range(cursor + 1, len(word) + 1):
            candidate = word[cursor:end]
            if _match_key(candidate) == target:
                repaired.append(candidate)
                cursor = end
                matched = True
                break
        if not matched:
            return pieces

    if cursor == len(word):
        return repaired
    return pieces


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-dir",
        default=os.environ.get("TDELIGHT_REPO_DIR", str(_default_repo_dir())),
        help="Path to a local clone of halecakir/turkish-delight-nlp-api.",
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("TDELIGHT_MODEL_NAME", DEFAULT_MODEL_NAME),
        help="TurkishDelight joint model basename inside the repo data directory.",
    )
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).expanduser().resolve()
    model, predict_morphemes = _load_runtime(repo_dir, args.model_name)

    for line in sys.stdin:
        word = line.strip()
        if not word:
            print(json.dumps([]), flush=True)
            continue

        response = predict_morphemes(model, word)
        pieces = _pieces_from_response(response, word)
        print(json.dumps(pieces, ensure_ascii=False), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
