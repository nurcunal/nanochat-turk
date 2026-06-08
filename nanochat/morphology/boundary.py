"""Internal morpheme-boundary marker utilities.

The marker annotates precomputed surface-morpheme boundaries. Paper-faithful
MorphBPE uses it only while training merge constraints; the pre-segmented BPE
control uses it as an internal text marker that is stripped on decode.
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
