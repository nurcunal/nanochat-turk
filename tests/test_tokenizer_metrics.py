import pytest

from tokenizers import Tokenizer, decoders, models, normalizers

from scripts.tokenizer_metrics import (
    HuggingFaceTokenizerAdapter,
    empty_metric_accumulator,
    measure_docs_batch,
    merge_metric_accumulator,
    morpheme_preservation_stats,
    word_coverage_stats,
)


class ByteTokenizer:
    """Fixed literal byte pieces, including UTF-8 byte fallback."""

    def __init__(self, pieces_by_word):
        self.pieces = {}
        self.words = {}
        for word, pieces in pieces_by_word.items():
            ids = []
            for piece in pieces:
                token_id = len(self.pieces)
                self.pieces[token_id] = piece
                ids.append(token_id)
            self.words[word] = ids
        self.enc = self

    def encode(self, words, num_threads=0):
        return self.words[words] if isinstance(words, str) else [self.words[word] for word in words]

    def decode_single_token_bytes(self, token_id):
        return self.pieces[token_id]

    def decode(self, ids):
        return b"".join(self.pieces[token_id] for token_id in ids).decode("utf-8")


def test_character_only_coverage_is_unicode_and_occurrence_weighted():
    tokenizer = ByteTokenizer({
        "ev": [b"e", b"v"],
        "evler": [b"ev", b"ler"],
        "ç": ["ç".encode()],
        "iş": [b"i", b"\xc5", b"\x9f"],
    })
    result = word_coverage_stats(tokenizer, ["ev", "ev", "evler", "ç", "iş"], num_threads=1)
    assert result["one_token_word_rate"] == 1 / 5
    assert result["character_only_word_rate"] == 3 / 5
    assert result["single_byte_only_word_rate"] == 3 / 5
    assert result["multi_character_character_only_word_rate"] == 2 / 4
    assert result["lossless_surface_partition_word_rate"] == 1
    assert result["unk_applicable"] is False
    assert result["unk_token_count"] is None


def test_morpheme_preservation_distinguishes_splits_merges_and_matching_spans():
    tokenizer = ByteTokenizer({
        "evler": [b"ev", b"ler"],
        "evden": [b"evden"],
        "evimiz": [b"e", b"v", b"imiz"],
    })
    result = morpheme_preservation_stats(tokenizer, [
        ("evler", ("ev", "ler")),
        ("evden", ("ev", "den")),
        ("evimiz", ("ev", "imiz")),
    ], num_threads=1)
    assert result["morpheme_count"] == 6
    assert result["preserved_morpheme_count"] == 3
    assert result["morpheme_preservation_rate"] == 0.5
    assert result["boundary_precision"] == 2 / 3
    assert result["boundary_recall"] == 2 / 3
    assert result["boundary_f1"] == 2 / 3
    assert result["exact_boundary_word_rate"] == 1 / 3


def test_byte_fallback_boundary_inside_unicode_character_is_false_positive():
    tokenizer = ByteTokenizer({"işler": [b"i", b"\xc5", b"\x9f", b"ler"]})
    result = morpheme_preservation_stats(tokenizer, [("işler", ("iş", "ler"))], num_threads=1)
    assert result["boundary_precision"] == 1 / 3
    assert result["boundary_recall"] == 1
    assert result["morpheme_preservation_rate"] == 0.5


def wordpiece_adapter(lowercase=False):
    backend = Tokenizer(models.WordPiece(vocab={"[UNK]": 0, "ev": 1, "##ler": 2}, unk_token="[UNK]"))
    backend.decoder = decoders.WordPiece(prefix="##")
    if lowercase:
        backend.normalizer = normalizers.Lowercase()
    return HuggingFaceTokenizerAdapter(backend, name="test", model_id="test", implementation="wordpiece", batch_size=20)


def test_unknown_is_not_a_one_token_covered_word():
    tokenizer = wordpiece_adapter()
    result = word_coverage_stats(tokenizer, ["ev", "evler", "bilinmeyen"], num_threads=1)
    assert result["one_token_word_count"] == 1
    assert result["unk_token_count"] == 1
    assert result["unk_token_rate"] == 1 / 4
    assert result["unknown_free_word_rate"] == 2 / 3
    assert result["unalignable_word_count"] == 1
    batch, _ = measure_docs_batch(tokenizer, ["ev", "bilinmeyen"], input_has_morph_boundaries=False, morph_boundary="\ue000", num_threads=1)
    aggregate = empty_metric_accumulator()
    merge_metric_accumulator(aggregate, batch)
    assert aggregate["unk_token_count"] == 1
    assert aggregate["docs_with_unk"] == 1


def test_lossy_normalization_is_reported_and_excluded_from_morpheme_denominator():
    tokenizer = wordpiece_adapter(lowercase=True)
    result = word_coverage_stats(tokenizer, ["EV", "ev"], num_threads=1)
    assert result["one_token_word_count"] == 1
    assert result["unalignable_word_count"] == 1
    morphology = morpheme_preservation_stats(tokenizer, [("EV", ("EV",)), ("ev", ("ev",))], num_threads=1)
    assert morphology["sample_word_occurrences"] == 2
    assert morphology["scored_word_occurrences"] == 1
    assert morphology["unalignable_word_occurrences"] == 1
    assert morphology["morpheme_preservation_rate"] == 1


def test_empty_denominators_are_missing_not_fake_zero_scores():
    tokenizer = ByteTokenizer({})
    assert word_coverage_stats(tokenizer, [], num_threads=1)["one_token_word_rate"] is None
    morphology = morpheme_preservation_stats(tokenizer, [], num_threads=1)
    assert morphology["morpheme_preservation_rate"] is None
    assert morphology["boundary_f1"] is None


def test_malformed_reference_rejected():
    tokenizer = ByteTokenizer({"ev": [b"ev"]})
    with pytest.raises(ValueError, match="concatenate"):
        morpheme_preservation_stats(tokenizer, [("ev", ("eve",))], num_threads=1)
