"""
Tests for morphology segmentation helpers.

Run: python -m pytest tests/test_morphology_segmenters.py -v
"""

import sys

import pytest

from nanochat.morphology import (
    BatchCommandSegmenter,
    IdentitySegmenter,
    SegmentationError,
    TRMorphFlookupSegmenter,
    WordSegmentation,
    create_segmenter,
    iter_word_spans,
    parse_surface_pieces,
    segment_text,
    segment_text_spans,
)


def test_word_segmentation_requires_surface_reconstruction():
    seg = WordSegmentation.from_pieces("evlerden", ["ev", "ler", "den"], source="test")
    assert seg.pieces == ("ev", "ler", "den")
    assert [span.text for span in seg.spans(offset=10)] == ["ev", "ler", "den"]
    assert [(span.start, span.end) for span in seg.spans(offset=10)] == [(10, 12), (12, 15), (15, 18)]

    with pytest.raises(ValueError):
        WordSegmentation.from_pieces("evlerden", ["ev", "PL", "ABL"], source="bad")


def test_parse_surface_pieces_accepts_common_formats():
    assert parse_surface_pieces("ev+ler+den", "evlerden") == ("ev", "ler", "den")
    assert parse_surface_pieces('["ev", "ler", "den"]', "evlerden") == ("ev", "ler", "den")
    assert parse_surface_pieces('{"pieces": ["ev", "ler", "den"]}', "evlerden") == ("ev", "ler", "den")
    assert parse_surface_pieces("ev<N>+ler<A3pl>+den<Abl>", "evlerden") == ("ev", "ler", "den")
    assert parse_surface_pieces("evlerden -> ev+ler+den", "evlerden") == ("ev", "ler", "den")
    assert parse_surface_pieces("Bir", "bir") == ("bir",)
    assert parse_surface_pieces("Aras-ı-nda", "arasında") == ("aras", "ı", "nda")
    assert parse_surface_pieces("damla+A3sg+n+ı", "damlanı") == ("damla", "n", "ı")
    assert parse_surface_pieces("büyü+A3sg|lü+ydü+A3sg", "büyülüydü") == ("büyü", "lü", "ydü")

    with pytest.raises(SegmentationError):
        parse_surface_pieces("ev<N><A3sg>", "evlerden")


def test_identity_segmenter_preserves_text_and_spans():
    text = "Türkiye'de evlerden geldik."
    segmenter = IdentitySegmenter()

    assert list(iter_word_spans(text)) == [
        (0, 10, "Türkiye'de"),
        (11, 19, "evlerden"),
        (20, 26, "geldik"),
    ]
    assert segment_text(text, segmenter) == text

    spans = segment_text_spans(text, segmenter)
    assert [span.text for span in spans] == ["Türkiye'de", "evlerden", "geldik"]
    assert text[spans[1].start:spans[1].end] == "evlerden"


def test_batch_command_segmenter_parses_stdout_lines():
    code = (
        "import sys\n"
        "for line in sys.stdin:\n"
        "    word = line.strip()\n"
        "    if word == 'evlerden':\n"
        "        print('ev+ler+den')\n"
        "    elif word == 'geldik':\n"
        "        print('[\"gel\", \"dik\"]')\n"
        "    else:\n"
        "        print(word)\n"
    )
    segmenter = BatchCommandSegmenter([sys.executable, "-c", code], name="fake")
    segmentations = segmenter.segment_words(["evlerden", "geldik", "bugun"])

    assert [seg.pieces for seg in segmentations] == [
        ("ev", "ler", "den"),
        ("gel", "dik"),
        ("bugun",),
    ]
    assert segment_text("evlerden geldik", segmenter) == "ev ler den gel dik"


def test_batch_command_segmenter_falls_back_or_raises_on_bad_output():
    code = "import sys\nfor line in sys.stdin:\n    print('ev<N><A3sg>')\n"

    fallback_segmenter = BatchCommandSegmenter([sys.executable, "-c", code], name="fake")
    fallback = fallback_segmenter.segment_word("evlerden")
    assert fallback.pieces == ("evlerden",)
    assert fallback.fallback is True
    assert "fallback_reason" in fallback.metadata

    strict_segmenter = BatchCommandSegmenter([sys.executable, "-c", code], name="fake", strict=True)
    with pytest.raises(SegmentationError):
        strict_segmenter.segment_word("evlerden")


def test_trmorph_flookup_segmenter_parses_blocks(tmp_path):
    segment_fst = tmp_path / "segment.fst"
    segment_fst.write_text("fake", encoding="utf-8")
    fake_flookup = tmp_path / "fake_flookup.py"
    code = (
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "for word in sys.stdin.read().splitlines():\n"
        "    if word == 'evlerden':\n"
        "        print(word + '\\t' + 'ev+ler+den')\n"
        "        print(word + '\\t' + 'evler+den')\n"
        "    elif word == 'bilinmeyen':\n"
        "        print(word + '\\t?')\n"
        "    else:\n"
        "        print(word + '\\t' + word)\n"
        "    print()\n"
    )
    fake_flookup.write_text(code, encoding="utf-8")
    fake_flookup.chmod(0o755)
    segmenter = TRMorphFlookupSegmenter(
        segment_fst=str(segment_fst),
        flookup_cmd=str(fake_flookup),
        pick="most_segments",
    )

    out = segmenter.segment_words(["evlerden", "bilinmeyen"])
    assert out[0].pieces == ("ev", "ler", "den")
    assert out[1].pieces == ("bilinmeyen",)
    assert out[1].fallback is True


def test_trmorph_flookup_segmenter_caps_pathological_outputs(tmp_path):
    segment_fst = tmp_path / "segment.fst"
    segment_fst.write_text("fake", encoding="utf-8")
    fake_flookup = tmp_path / "fake_flookup.py"
    code = (
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "for word in sys.stdin.read().splitlines():\n"
        "    if word == 'patolojik':\n"
        "        for _ in range(5):\n"
        "            print(word + '\\t' + word)\n"
        "    elif word == 'evlerden':\n"
        "        print(word + '\\t' + 'ev+ler+den')\n"
        "    else:\n"
        "        print(word + '\\t' + word)\n"
        "    print()\n"
    )
    fake_flookup.write_text(code, encoding="utf-8")
    fake_flookup.chmod(0o755)
    segmenter = TRMorphFlookupSegmenter(
        segment_fst=str(segment_fst),
        flookup_cmd=str(fake_flookup),
        max_output_lines_per_word=3,
    )

    out = segmenter.segment_words(["patolojik", "evlerden"])
    assert out[0].pieces == ("patolojik",)
    assert out[0].fallback is True
    assert "analysis_output_overflow" in out[0].metadata["fallback_reason"]
    assert out[1].pieces == ("ev", "ler", "den")
    assert out[1].fallback is False


def test_create_segmenter_known_backends(monkeypatch):
    assert isinstance(create_segmenter("identity"), IdentitySegmenter)

    monkeypatch.setenv("TRMORPH_SEGMENT_CMD", f"{sys.executable} -c \"import sys; print(sys.stdin.read(), end='')\"")
    seg = create_segmenter("trmorph")
    assert seg.segment_word("evlerden").pieces == ("evlerden",)
