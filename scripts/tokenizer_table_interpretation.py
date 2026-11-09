"""Add table-local readings without changing measured cells or data exports.

Rank only within a shown local vocabulary tier. Comparisons use displayed
precision deliberately, so rounding ties do not become spurious winners.
"""
from __future__ import annotations

from collections import defaultdict

TIERS = {32768: "32k", 65536: "64k", 131072: "128k"}


def number(value: str) -> float | None:
    try:
        return float(value.replace(",", "").replace("%", "").strip())
    except ValueError:
        return None


def short_name(value: str) -> str:
    return value.replace("MorphBPE / ", "")


def joined(values: list[str]) -> str:
    return values[0] if len(values) == 1 else ", ".join(values[:-1]) + " and " + values[-1]


def column(headers: list[str], *terms: str) -> str | None:
    for term in terms:
        for header in headers:
            if term.lower() in header.lower():
                return header
    return None


def local_groups(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    groups = defaultdict(list)
    for row in rows:
        vocab = number(row.get("Vocab", row.get("Vocabulary", "")))
        if vocab in TIERS:
            groups[int(vocab)].append(row)
    return dict(groups)


def winners(rows: list[dict[str, str]], key: str, minimize: bool = False) -> tuple[str, ...]:
    values = [(row, number(row[key])) for row in rows if number(row[key]) is not None]
    best = (min if minimize else max)(value for _, value in values)
    return tuple(short_name(row["Tokenizer"]) for row, value in values if value == best)


def rank_sentence(rows: list[dict[str, str]], key: str | None, description: str, *, minimize: bool = False) -> str:
    if key is None:
        return ""
    outcomes = defaultdict(list)
    for tier, group in local_groups(rows).items():
        if len(group) > 1:
            outcomes[winners(group, key, minimize)].append(TIERS[tier])
    parts = []
    for names, tiers in outcomes.items():
        subject = joined(list(names))
        if len(names) > 1:
            subject += " (tied at displayed precision)"
        parts.append(f"{subject} at {joined(tiers)}")
    text = f"{description}: {'; '.join(parts)}." if parts else ""
    if text and "isolated fertility" in description.lower():
        marginal = []
        for tier, group in local_groups(rows).items():
            values = sorted(number(row[key]) for row in group)
            if len(values) > 1 and 0 < values[1] - values[0] <= 0.00100001:
                marginal.append(TIERS[tier])
        if marginal:
            text += f" The lead at {joined(marginal)} is at most 0.001 token/word at the displayed precision."
    return text


def combined_rank(rows: list[dict[str, str]], keys: list[str | None], description: str, *, minimize: bool = False) -> str:
    keys = [key for key in keys if key is not None]
    if not keys:
        return ""
    groups = local_groups(rows)
    if all(len({winners(group, key, minimize) for key in keys}) == 1 for group in groups.values()):
        return rank_sentence(rows, keys[0], description, minimize=minimize)
    return " ".join(rank_sentence(rows, key, ("Lowest " if minimize else "Highest ") + key, minimize=minimize) for key in keys)


def efficiency_exception(rows: list[dict[str, str]], headers: list[str]) -> str:
    isolated = column(headers, "isolated tokens/word", "isolated tok/word")
    coverage = column(headers, "one-token words", "whole word: one token")
    if isolated is None or coverage is None:
        return ""
    exceptions = []
    for tier, group in local_groups(rows).items():
        baseline = next((r for r in group if r["Tokenizer"] == "Raw BPE"), None)
        if baseline is None:
            continue
        for row in group:
            if number(row[isolated]) > number(baseline[isolated]) and number(row[coverage]) < number(baseline[coverage]):
                exceptions.append(f'{short_name(row["Tokenizer"])} at {TIERS[tier]}')
    return (f"{joined(exceptions)} has both higher isolated fertility and lower one-token coverage than same-size raw BPE, despite its morphology improvements." if exceptions else "")


def reading(headers: list[str], rows: list[dict[str, str]], scope: str) -> str:
    """Choose a compact, metric-specific interpretation for one exact table."""
    has = lambda *terms: column(headers, *terms)
    if has("Original JSON"):
        return "**Reading note:** These links identify the original evidence for each row; their order is not a performance ranking."
    if "Analyzer" in headers and "Forms/s" in headers:
        return "**Interpretation:** On the common hash-100k screen, TRmorph has the highest measured processing rate (9,223 forms/s), while TurkishDelight has the lowest fallback rate (0%). Identity is a no-segmentation control. These analyzer readiness results do not rank the finished tokenizers or their LLMs."
    if "Analyzer" in headers and "Best %" in headers:
        return "**Interpretation:** TRmorph leads both best selections (48.2%) and acceptability (51.0%); TurkishDelight is second on both. This supports TRmorph under this disagreement-focused scripted rubric, not a universal linguistic-quality or downstream-model win."
    if "Split" in headers and "Tokenizer" not in headers:
        if has("Tokens/word"):
            return "**Interpretation:** Raw BPE uses slightly more corpus tokens per word on validation than training, despite better isolated-word one-token coverage there. Both splits have zero roundtrip failures. This is one tokenizer on different text distributions, so there is no tokenizer-family winner in this table."
        return "**Interpretation:** These timings describe different corpus sizes and worker counts for the same tokenizer. Neither the shorter validation runtime nor its recorded tokens/s establishes a faster tokenizer; vocabulary-use rates are scope-dependent diagnostics."
    if has("Recorded tokens/word"):
        return "**Interpretation:** TURNA and VBART have similar recorded token counts and one-token rates, but lose exact roundtrip on 95.214% of documents; BERTurk fails on 99.292%. These values cannot establish a lossless compression or whole-word-coverage winner."
    if has("Recorded boundary crossed"):
        return "**Interpretation:** BERTurk has the highest recorded encoding rate, while the boundary figures belong to tokenizers that normalize or alter text. Neither column supports a fair morphology or speed ranking against the lossless BPE tokenizers."
    if rows and all(row.get("Tokenizer") in {"kumru_2b", "cosmos_turkish_gpt2"} for row in rows):
        if has("Tokens/word"):
            return "**Interpretation:** Cosmos uses fewer corpus tokens per word; Kumru has lower isolated fertility and higher one-token coverage, exact-morpheme agreement, boundary preservation and pairwise consistency F1. The preferred reference depends on the metric; their different vocabularies prevent treating this as a controlled tokenizer-family ablation."
        return "**Interpretation:** Cosmos compresses the corpus into more bytes per token. Kumru has lower morphology edit distance and stronger consistency precision/recall, with fewer crossed reference boundaries. This repeats the compression-versus-morphology trade-off rather than identifying one universal winner."
    if len(rows) == 1:
        return "**Interpretation:** This single Zemberek diagnostic demonstrates lossless roundtrip on its 10,000-document sample. Without other tokenizers on this same diagnostic scope, it cannot identify the best tokenizer, and its absent morphology reference cannot establish boundary quality."
    if has("Scored / excluded morphology words"):
        return "**Interpretation:** All twelve tokenizers tie on observed roundtrip success and score all 500,000 morphology words without exclusions; UNK is N/A for every byte-complete tokenizer. " + rank_sentence(rows, has("Multi-character words split only"), "Lowest multi-character character-only rate", minimize=True) + " Character-only and byte-only rates describe fragmentation, not independent evidence of linguistic quality."
    if has("Coverage unaligned"):
        return "**Interpretation:** All rows tie at zero roundtrip failures and zero alignment exclusions. " + rank_sentence(rows, has("Multi-char character-only"), "Lowest multi-character character-only rate", minimize=True) + " " + rank_sentence(rows, has("Encode M tok/s"), "Highest recorded encoding tokens/s") + " These timings include concurrent-process contention, and more tokens/s can reflect producing more tokens rather than processing raw text faster."
    if has("Workers × threads"):
        return "**Interpretation:** Among the local tokenizer rows, " + rank_sentence(rows, has("Encode M tok/s"), "highest recorded encoding tokens/s") + " All rows tie at zero observed roundtrip failures. These timings use tokenizer-dependent token counts, so they do not establish a controlled raw-text-speed ranking. High vocabulary usage alone is not evidence of better morphology."
    if has("Intact morphemes", "Morphemes intact"):
        return "**Interpretation:** " + " ".join(filter(None, [
            rank_sentence(rows, has("Corpus tokens/word", "Corpus tok/word"), "Lowest corpus fertility", minimize=True),
            rank_sentence(rows, has("Isolated tokens/word", "Isolated tok/word"), "Lowest isolated fertility", minimize=True),
            combined_rank(rows, [has("One-token words"), has("Intact morphemes", "Morphemes intact"), has("boundary P %"), has("boundary R %"), has("boundary F1 %")], "Highest one-token coverage, intact-morpheme preservation and boundary P/R/F1"),
            efficiency_exception(rows, headers),
        ])) + " These morphology rankings use the automatic TRmorph reference."
    if has("Morph consistency F1", "Consistency F1") and has("Isolated tokens/word"):
        message = "**Interpretation:** " + " ".join(filter(None, [
            rank_sentence(rows, has("Corpus tokens/word", "Tokens/word"), "Lowest corpus fertility", minimize=True),
            rank_sentence(rows, has("Isolated tokens/word"), "Lowest isolated fertility", minimize=True),
            rank_sentence(rows, has("Morph edit"), "Lowest morphology edit distance", minimize=True),
            combined_rank(rows, [has("One-token words", "Whole word: one token"), has("Exact morph sequence", "Exact morpheme sequence"), has("Boundaries preserved", "Preserved boundaries")], "Highest one-token coverage, exact-morpheme agreement and boundary preservation"),
            rank_sentence(rows, has("Morph consistency F1", "Consistency F1"), "Highest original pairwise consistency F1"),
            efficiency_exception(rows, headers),
        ]))
        if scope == "archived_sample":
            message += " The mixed 32k word caps make its word-level rankings descriptive rather than fully matched."
        return message
    if has("Bytes/token") and has("Morph edit"):
        return "**Interpretation:** " + " ".join(filter(None, [
            rank_sentence(rows, has("Bytes/token"), "Best corpus compression (most bytes/token)"),
            combined_rank(rows, [has("Morph edit distance", "Morph edit"), has("Normalized edit")], "Lowest raw and normalized morphology edit distance", minimize=True),
            rank_sentence(rows, has("Consistency P"), "Highest pairwise consistency precision"),
            rank_sentence(rows, has("Consistency R"), "Highest pairwise consistency recall"),
            rank_sentence(rows, has("Consistency F1"), "Highest original pairwise consistency F1"),
            rank_sentence(rows, has("Exact morph sequence"), "Highest exact-morpheme agreement"),
            rank_sentence(rows, has("Document boundaries preserved"), "Highest document-boundary preservation"),
            rank_sentence(rows, has("Boundary crossed"), "Lowest crossed-boundary rate", minimize=True),
        ])) + (" Recorded encoding rates are not controlled raw-text speed comparisons." if has("Encode M tokens/s") else " This pairwise consistency metric is distinct from literal boundary F1.")
    if has("Normalized morph edit"):
        return "**Interpretation:** " + " ".join(filter(None, [
            rank_sentence(rows, has("Normalized morph edit"), "Lowest normalized morphology edit distance", minimize=True),
            rank_sentence(rows, has("Consistency P"), "Highest pairwise consistency precision"),
            rank_sentence(rows, has("Consistency R"), "Highest pairwise consistency recall"),
            rank_sentence(rows, has("Crossed boundaries"), "Lowest crossed-boundary rate", minimize=True),
        ])) + " Lower resample SD is not a quality win or a confidence interval; it measures variability of the sampled consistency calculation."
    raise ValueError(f"No table interpretation for {scope}: {headers}")


def interpret_tables(lines: list[str], *, scope: str) -> list[str]:
    """Preserve every table byte and insert a reading immediately beneath it."""
    output, index = [], 0
    while index < len(lines):
        if not lines[index].startswith("|"):
            output.append(lines[index])
            index += 1
            continue
        table = []
        while index < len(lines) and lines[index].startswith("|"):
            table.append(lines[index])
            index += 1
        cells = lambda line: [cell.strip() for cell in line.strip("|").split("|")]
        headers = cells(table[0])
        rows = [dict(zip(headers, cells(line))) for line in table[2:]]
        output.extend(table)
        output.extend(["", reading(headers, rows, scope)])
    return output
