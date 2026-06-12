import json

from nanochat.morphology import (
    MORPHEME_BOUNDARY,
    MorphBPEIteratorStats,
    iter_morphbpe_training_chunks,
    iter_morphbpe_training_stream,
    strip_morpheme_boundaries,
)
from nanochat.tokenizer import RustBPETokenizer
from scripts.tokenizer_metrics import (
    extract_segmented_words,
    morphological_alignment_stats,
    morphological_consistency_stats,
    sequence_edit_distance,
)


class _FakeMetricEncoding:
    def __init__(self, pieces):
        self.pieces = pieces

    def decode_single_token_bytes(self, token_id):
        return self.pieces[token_id]


class _FakeMetricTokenizer:
    def __init__(self, encoded_by_surface, pieces):
        self.encoded_by_surface = encoded_by_surface
        self.enc = _FakeMetricEncoding(pieces)

    def encode(self, text, num_threads=0):
        del num_threads
        if isinstance(text, str):
            return self.encoded_by_surface[text]
        return [self.encoded_by_surface[item] for item in text]


def test_morph_boundary_strip_helper():
    text = f"ev{MORPHEME_BOUNDARY}ler{MORPHEME_BOUNDARY}den"
    assert strip_morpheme_boundaries(text) == "evlerden"


def test_extract_segmented_words_for_morphology_metrics():
    text = f"ev{MORPHEME_BOUNDARY}ler eve git{MORPHEME_BOUNDARY}ti."
    assert extract_segmented_words(text) == [
        ("evler", ("ev", "ler")),
        ("eve", ("eve",)),
        ("gitti", ("git", "ti")),
    ]


def test_sequence_edit_distance_over_morpheme_token_pieces():
    assert sequence_edit_distance((b"ev", b"ler"), (b"ev", b"ler")) == 0
    assert sequence_edit_distance((b"ev", b"ler"), (b"evler",)) == 2
    assert sequence_edit_distance((b"ev", b"ler"), (b"ev", b"ler", b"den")) == 1


def test_morphology_alignment_metrics_exact_and_cross_morpheme_token():
    tokenizer = _FakeMetricTokenizer(
        {
            "evler": [1, 2],
            "evlerden": [3],
        },
        {
            1: b"ev",
            2: b"ler",
            3: b"evlerden",
        },
    )
    metrics = morphological_alignment_stats(
        tokenizer,
        [
            ("evler", ("ev", "ler")),
            ("evlerden", ("ev", "ler", "den")),
        ],
        num_threads=1,
    )

    assert metrics["sample_word_occurrences"] == 2
    assert metrics["exact_morpheme_sequence_rate"] == 0.5
    assert metrics["morphological_edit_distance"] == 1.5


def test_morphology_consistency_metrics_are_binary_shared_relations():
    tokenizer = _FakeMetricTokenizer(
        {
            "evler": [1, 2],
            "evden": [1, 3],
            "kitaplar": [4, 2],
        },
        {
            1: b"ev",
            2: b"ler",
            3: b"den",
            4: b"kitap",
        },
    )
    metrics = morphological_consistency_stats(
        tokenizer,
        [
            ("evler", ("ev", "ler")),
            ("evden", ("ev", "den")),
            ("kitaplar", ("kitap", "lar")),
        ],
        num_threads=1,
        max_words=0,
        n_clusters=1,
        pairs_per_cluster=10,
        resamples=1,
        seed=13,
    )

    assert metrics["precision_mean"] == 0.5
    assert metrics["recall_mean"] == 1.0
    assert round(metrics["f1_mean"], 4) == 0.6667


def test_rustbpe_decode_strips_morph_boundaries_after_reload(tmp_path):
    segmented = [
        f"ev{MORPHEME_BOUNDARY}ler{MORPHEME_BOUNDARY}den geldik",
        f"çalış{MORPHEME_BOUNDARY}ıyor{MORPHEME_BOUNDARY}um",
    ]
    tokenizer = RustBPETokenizer.train_from_iterator(
        iter(segmented),
        vocab_size=512,
        decode_strip=MORPHEME_BOUNDARY,
    )

    ids = tokenizer.encode(segmented[0])
    assert tokenizer.decode(ids) == "evlerden geldik"

    tokenizer.save(str(tmp_path))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({
            "implementation": "preseg_bpe",
            "decode_strip": MORPHEME_BOUNDARY,
        }),
        encoding="utf-8",
    )
    reloaded = RustBPETokenizer.from_directory(str(tmp_path))
    assert reloaded.decode(reloaded.encode(segmented[1])) == "çalışıyorum"


def test_morph_boundary_tokens_do_not_absorb_left_context():
    text = " ".join([f"ev{MORPHEME_BOUNDARY}ler{MORPHEME_BOUNDARY}den"] * 200)
    tokenizer = RustBPETokenizer.train_from_iterator(
        iter([text]),
        vocab_size=512,
        decode_strip=MORPHEME_BOUNDARY,
    )
    marker_bytes = MORPHEME_BOUNDARY.encode("utf-8")

    for token_id in tokenizer.enc.encode_ordinary(f"ev{MORPHEME_BOUNDARY}ler{MORPHEME_BOUNDARY}den"):
        piece = tokenizer.enc.decode_single_token_bytes(token_id)
        marker_at = piece.find(marker_bytes)
        assert marker_at in (-1, 0)


def test_morphbpe_training_chunks_split_on_boundaries():
    marked = f" ev{MORPHEME_BOUNDARY}ler{MORPHEME_BOUNDARY}den!"
    stats = MorphBPEIteratorStats()

    assert list(iter_morphbpe_training_chunks(marked, stats=stats)) == [
        " ev",
        "ler",
        "den",
        "!",
    ]
    assert stats.docs == 1
    assert stats.docs_with_boundary == 1
    assert stats.boundary_splits == 2
    assert stats.training_chunks == 4


def test_paper_morphbpe_trains_raw_text_tokenizer_without_cross_boundary_merge():
    marked_docs = [
        " ".join([f"ev{MORPHEME_BOUNDARY}ler{MORPHEME_BOUNDARY}den"] * 200)
    ]
    stats = MorphBPEIteratorStats()
    tokenizer = RustBPETokenizer.train_from_iterator(
        iter_morphbpe_training_stream(iter(marked_docs), stats=stats),
        vocab_size=512,
    )

    assert tokenizer.decode_strip == ""
    assert tokenizer.decode(tokenizer.encode("evlerden geldik")) == "evlerden geldik"
    assert stats.docs == 1
    assert stats.boundary_splits == 400
    assert b"evlerden" not in tokenizer.enc._mergeable_ranks
    assert b"lerden" not in tokenizer.enc._mergeable_ranks
