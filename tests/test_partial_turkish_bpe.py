from __future__ import annotations

from collections import Counter
from pathlib import Path

from scripts.train_partial_turkish_bpe import RowGroupRef, weighted_training_texts


def test_weighted_training_texts_respects_cap_and_train_split(monkeypatch) -> None:
    policy = {
        "mixture": [
            {"id": "a", "weight": 0.75},
            {"id": "b", "weight": 0.25},
        ]
    }
    schedule = {
        "a": [RowGroupRef(Path("a"), 1, 0, "a_source")],
        "b": [RowGroupRef(Path("b"), 2, 0, "b_source")],
    }

    def fake_rows(refs, mixture_id, policy, counters):
        del refs, policy
        for index in range(100):
            counters[f"encountered:{mixture_id}:train"] += 1
            yield "x" * 10, f"{index:064x}", 1, index

    monkeypatch.setattr(
        "scripts.train_partial_turkish_bpe._mixture_rows", fake_rows
    )
    stats = {
        "documents": 0,
        "characters": 0,
        "documents_by_mixture": Counter(),
        "characters_by_mixture": Counter(),
        "counters": Counter(),
        "sequence_hash": __import__("hashlib").sha256(),
    }
    texts = list(
        weighted_training_texts(
            schedule, policy, max_chars=100, doc_cap=10, stats=stats
        )
    )
    assert len(texts) == 11
    assert stats["characters"] == 110
    assert stats["documents"] == 11
    assert stats["characters_by_mixture"] == Counter({"a": 80, "b": 30})
