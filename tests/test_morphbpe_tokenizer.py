import json

from nanochat.morphology import MORPHEME_BOUNDARY, strip_morpheme_boundaries
from nanochat.tokenizer import RustBPETokenizer


def test_morph_boundary_strip_helper():
    text = f"ev{MORPHEME_BOUNDARY}ler{MORPHEME_BOUNDARY}den"
    assert strip_morpheme_boundaries(text) == "evlerden"


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
            "implementation": "morphbpe",
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
