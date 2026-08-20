from __future__ import annotations

import inspect

from nanochat.tokenizer import RustBPETokenizer
from scripts.train_turkish_raw_bpe import (
    _capped_threshold_iterator,
    run_pinned_iterator_parity_fixture,
)


def test_ordered_raw_bpe_training_is_reproducible_without_false_seed_claim():
    texts = (
        ["Merhaba dünya, bugün nasılsın?"]
        + ["İstanbul Ankara İzmir Bursa Adana"]
        + ["Türkçe konuşmak güzel ve doğal."]
    ) * 20
    left = RustBPETokenizer.train_from_iterator(iter(texts), 300)
    right = RustBPETokenizer.train_from_iterator(iter(texts), 300)

    assert left.get_vocab_size() == 300
    assert right.get_vocab_size() == 300
    assert left.enc._mergeable_ranks == right.enc._mergeable_ranks
    assert left.decode(left.encode("İğde, üzüm, çilek ve şeftali.")) == (
        "İğde, üzüm, çilek ve şeftali."
    )
    # RustBPE's public trainer has no seed argument; manifests must describe
    # ordered-input reproducibility rather than inventing a seed guarantee.
    assert "seed" not in inspect.signature(
        RustBPETokenizer.train_from_iterator
    ).parameters


def test_full_capped_document_overshoot_matches_pinned_tok_train_loop():
    parity = run_pinned_iterator_parity_fixture()
    assert parity["passed"] is True
    assert parity["realized_characters"] == 12
    assert parity["terminal_overshoot_characters"] == 2
    assert list(
        _capped_threshold_iterator(
            ["abcdef", "xy", "1234567", "never reached"],
            max_chars=10,
            doc_cap=5,
        )
    ) == ["abcde", "xy", "12345"]
