"""Internal morpheme-boundary marker utilities.

MorphBPE tokenizers train on text with an internal boundary marker inserted
between surface morphemes. The marker is not part of user-visible text: tokenizer
decoding strips it so segmented text round-trips back to the original document.
"""

MORPHEME_BOUNDARY = "\ue000"


def strip_morpheme_boundaries(text: str, boundary: str = MORPHEME_BOUNDARY) -> str:
    """Remove internal morpheme-boundary markers from text."""

    if not boundary:
        return text
    return text.replace(boundary, "")


def display_boundary(boundary: str = MORPHEME_BOUNDARY) -> str:
    """Return a readable representation for configs/logs."""

    if not boundary:
        return ""
    return " ".join(f"U+{ord(ch):04X}" for ch in boundary)
