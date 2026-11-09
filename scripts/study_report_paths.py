"""Shared locations for the single public report and its supporting evidence."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "docs/study_results"
DATA_DIR = REPORT_DIR / "data"
EVIDENCE_DIR = REPORT_DIR / "provenance/fresh_tokenizer_metrics"
SECTION_DIR = ROOT / "dev-ignore/study-sections"
CONTENT_DIR = ROOT / "scripts/study_report_content"


def write_section(filename: str, text: str) -> None:
    """Save a generated section draft, not another public report."""
    if Path(filename).name != filename or not filename.endswith(".md"):
        raise ValueError("A section must be a plain Markdown filename")
    SECTION_DIR.mkdir(parents=True, exist_ok=True)
    (SECTION_DIR / filename).write_text(text)
