"""Local no-API morphology judge for blind Turkish segmenter packs.

This script does not call an external LLM API. It encodes the blind judging
rubric used for the first Codex-local pass so the item-level decisions can be
regenerated and audited. It reads only the blind judge JSONL and must be scored
later with ``scripts/morph_judge_score.py`` plus the answer key.
"""

from __future__ import annotations

import argparse
import json
import math
import unicodedata
from pathlib import Path
from typing import Any


GOOD_SHORT_ROOTS = {
    "aç", "al", "as", "at", "az", "de", "di", "düş", "et", "geç", "gel", "git",
    "gör", "iç", "in", "kal", "kıl", "ol", "öl", "san", "sat", "seç",
    "sil", "sor", "sus", "tak", "tut", "var", "ver", "vur", "yap", "ye",
}
ATOMIC_SUFFIXES = {
    "ım", "im", "um", "üm", "ımız", "imiz", "umuz", "ümüz",
    "ın", "in", "un", "ün", "ınız", "iniz", "unuz", "ünüz",
    "ı", "i", "u", "ü", "sı", "si", "su", "sü",
    "lar", "ler", "ları", "leri", "larda", "lerde", "lardan", "lerden",
    "dan", "den", "tan", "ten", "da", "de", "ta", "te", "a", "e",
    "ya", "ye", "yı", "yi", "yu", "yü", "na", "ne", "nı", "ni", "nu", "nü",
    "nin", "nın", "nun", "nün", "ın", "in", "un", "ün",
    "ki", "daki", "deki", "taki", "teki",
    "cı", "ci", "cu", "cü", "çı", "çi", "çu", "çü",
    "lık", "lik", "luk", "lük", "sız", "siz", "suz", "süz",
    "lı", "li", "lu", "lü", "sal", "sel", "ca", "ce", "ça", "çe",
    "laş", "leş", "laştır", "leştir", "lan", "len", "la", "le",
    "ıl", "il", "ul", "ül", "ın", "in", "un", "ün", "nıl", "nil", "nul", "nül",
    "abil", "ebil", "abilir", "ebilir",
    "dı", "di", "du", "dü", "tı", "ti", "tu", "tü",
    "mış", "miş", "muş", "müş", "acak", "ecek", "yor", "ar", "er", "ır",
    "ir", "ur", "ür", "r", "maz", "mez", "ma", "me", "sa", "se",
    "malı", "meli", "mak", "mek", "makta", "mekte", "madan", "meden",
    "ınca", "ince", "unca", "ünce", "ken", "rken",
    "dık", "dik", "duk", "dük", "tık", "tik", "tuk", "tük",
    "dığı", "diği", "duğu", "düğü", "tığı", "tiği", "tuğu", "tüğü",
    "dığ", "diğ", "duğ", "düğ", "tığ", "tiğ", "tuğ", "tüğ",
    "an", "en", "yan", "yen", "mış", "miş", "muş", "müş",
    "dır", "dir", "dur", "dür", "tır", "tir", "tur", "tür",
    "ım", "im", "um", "üm", "sın", "sin", "sun", "sün",
    "ız", "iz", "uz", "üz", "sınız", "siniz", "sunuz", "sünüz",
    "larım", "lerim", "ların", "lerin", "larını", "lerini", "larına", "lerine",
    "larından", "lerinden", "ımızdan", "imizden", "umuzdan", "ümüzden",
}
LOW_VALUE_SINGLETONS = {"n", "y", "s"}
VALID_SINGLETON_SUFFIXES = set("aeıiuümnkrs")
BAD_ATOMIC_SPLITS = {
    ("ım", "ız"), ("im", "iz"), ("um", "uz"), ("üm", "üz"),
    ("sın", "ız"), ("sin", "iz"), ("sun", "uz"), ("sün", "üz"),
}
ROOT_FUSED_SUFFIXES = (
    "abil", "ebil", "ıl", "il", "ul", "ül", "laş", "leş", "laştır", "leştir",
)
HIGH_VALUE_SUFFIXES = {"rken": 1.65, "abil": 1.2, "ebil": 1.2}
TURKISH_LETTERS = set("abcçdefgğhıijklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ")


def norm(text: str) -> str:
    text = text.replace("\u2019", "'")
    return unicodedata.normalize("NFC", text).casefold()


def suffix_key(text: str) -> str:
    return norm(text).strip("'")


def is_noisy_or_foreign(word: str) -> bool:
    letters = [ch for ch in word if ch.isalpha()]
    if not letters:
        return True
    unusual = [ch for ch in letters if ch not in TURKISH_LETTERS]
    if len(unusual) / max(len(letters), 1) > 0.15:
        return True
    if any(ch.isdigit() for ch in word):
        return True
    if len(word) > 4 and word.isupper() and any(ch in "QWX" for ch in word):
        return True
    return False


def has_proper_marker(word: str) -> bool:
    return "'" in word or "\u2019" in word


def likely_inflected(word: str) -> bool:
    key = suffix_key(word)
    return any(key.endswith(suffix) and len(key) > len(suffix) + 2 for suffix in ATOMIC_SUFFIXES)


def suffix_piece_score(piece: str) -> tuple[float, bool]:
    key = suffix_key(piece)
    if not key:
        return -0.4, False
    if key in LOW_VALUE_SINGLETONS:
        return 0.15, True
    if key in HIGH_VALUE_SUFFIXES:
        return HIGH_VALUE_SUFFIXES[key], True
    if key in ATOMIC_SUFFIXES:
        return (0.45 if len(key) == 1 else 1.0), True

    # Dynamic decomposition gives partial credit for fused surface suffix groups,
    # but less than explicit morpheme boundaries.
    dp = [-math.inf] * (len(key) + 1)
    dp[0] = 0.0
    suffixes = sorted(ATOMIC_SUFFIXES, key=len, reverse=True)
    for i in range(len(key)):
        if dp[i] == -math.inf:
            continue
        for suffix in suffixes:
            if key.startswith(suffix, i):
                weight = 0.25 if len(suffix) == 1 else 0.55
                dp[i + len(suffix)] = max(dp[i + len(suffix)], dp[i] + weight)
    if dp[-1] != -math.inf:
        return dp[-1], True

    if len(key) == 1 and key not in VALID_SINGLETON_SUFFIXES:
        return -0.45, False
    if len(key) >= 5:
        return -0.9, False
    return -0.25, False


def candidate_score(word: str, pieces: list[str], is_identity: bool) -> tuple[float, str]:
    noisy = is_noisy_or_foreign(word) or has_proper_marker(word)
    if is_identity or len(pieces) == 1:
        if noisy:
            return 1.4, "identity/noisy_or_proper"
        if likely_inflected(word):
            return 0.1, "identity/misses_likely_suffix"
        return 0.7, "identity/ambiguous"

    root = suffix_key(pieces[0])
    score = 0.75
    unknown_suffixes = 0
    if len(root) <= 1:
        score -= 1.2
    elif len(root) <= 3 and root not in GOOD_SHORT_ROOTS:
        score -= 0.45
    if root in ATOMIC_SUFFIXES:
        score -= 0.9
    if len(pieces) > 2 and root.endswith(ROOT_FUSED_SUFFIXES):
        score -= 0.75

    suffix_keys = [suffix_key(piece) for piece in pieces[1:]]
    for piece in pieces[1:]:
        piece_score, known = suffix_piece_score(piece)
        score += piece_score
        if not known:
            unknown_suffixes += 1

    for left, right in zip(suffix_keys, suffix_keys[1:]):
        if (left, right) in BAD_ATOMIC_SPLITS:
            score -= 0.55
        if left == "n" and right in {"lar", "ler"}:
            score -= 1.5
        if left in {"lar", "ler"} and len(root) <= 3 and root not in GOOD_SHORT_ROOTS:
            score -= 1.0

    score -= 0.08 * max(len(pieces) - 2, 0)
    if noisy and unknown_suffixes:
        score -= 0.45
    if unknown_suffixes == 0:
        score += 0.25

    if unknown_suffixes:
        return score, "split/has_unknown_suffix_piece"
    return score, "split/plausible_suffix_boundaries"


def judge_row(row: dict[str, Any]) -> dict[str, Any]:
    word = str(row["word"])
    scored = []
    for candidate in row["candidates"]:
        label = str(candidate["label"])
        pieces = [str(piece) for piece in candidate.get("pieces", [])]
        is_identity = bool(candidate.get("is_identity", len(pieces) == 1))
        score, reason = candidate_score(word, pieces, is_identity)
        scored.append({
            "label": label,
            "score": score,
            "reason": reason,
            "segmentation": str(candidate.get("segmentation", "")),
        })

    scored.sort(key=lambda item: (-item["score"], item["label"]))
    best = scored[0]
    second_score = scored[1]["score"] if len(scored) > 1 else best["score"]
    acceptable = [
        item["label"]
        for item in scored
        if item["score"] >= best["score"] - 0.45
    ]
    # Always keep exact duplicate best segmentations acceptable.
    best_seg = best["segmentation"]
    for item in scored:
        if item["segmentation"] == best_seg and item["label"] not in acceptable:
            acceptable.append(item["label"])
    acceptable = sorted(acceptable)

    gap = best["score"] - second_score
    if gap >= 1.0:
        confidence = "high"
    elif gap >= 0.35:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "id": row["id"],
        "best_label": best["label"],
        "acceptable_labels": acceptable,
        "confidence": confidence,
        "notes": best["reason"],
    }


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-pack", required=True, help="Blind judge JSONL.")
    parser.add_argument("--output", required=True, help="Output judgment JSONL.")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in iter_jsonl(Path(args.judge_pack)):
            judgment = judge_row(row)
            f.write(json.dumps(judgment, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
