"""Presentation-only best-score highlighting with explicit comparison groups.

Callers choose comparable metrics and groups; arrows alone do not authorize a
comparison. Winners and ties use the displayed numeric precision. Missing,
descriptive, and singleton values never become a winner by default.
"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, InvalidOperation
import re
from typing import Callable, Hashable, Mapping, Sequence


def numeric_value(cell: str) -> Decimal | None:
    """Parse a complete displayed score, never a number inside a label."""
    value = cell.strip().removeprefix("**").removesuffix("**").replace(",", "")
    if value.endswith(("%", "×")):
        value = value[:-1].strip()
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    if re.fullmatch(number, value):
        try:
            result = Decimal(value)
            return result if result.is_finite() else None
        except InvalidOperation:
            return None
    ratio = re.fullmatch(rf"({number})\s*/\s*({number})", value)
    if ratio:
        numerator, denominator = map(Decimal, ratio.groups())
        return numerator / denominator if denominator else None
    clock = re.fullmatch(r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)", value)
    if clock:
        hours, minutes, seconds = clock.groups()
        return Decimal(hours or 0) * 3600 + Decimal(minutes) * 60 + Decimal(seconds)
    duration = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?", value)
    if duration and any(duration.groups()):
        hours, minutes, seconds = duration.groups()
        return Decimal(hours or 0) * 3600 + Decimal(minutes or 0) * 60 + Decimal(seconds or 0)
    return None


def _bold(cell: str) -> str:
    return cell if cell.startswith("**") and cell.endswith("**") else f"**{cell}**"


def highlight_columns(
    rows: list[list[str]],
    directions: Mapping[int, str],
    group_keys: Sequence[Hashable] | None = None,
) -> list[list[str]]:
    """Bold extrema down selected columns within each explicit row group."""
    if any(direction not in {"min", "max"} for direction in directions.values()):
        raise ValueError("Directions must be min or max")
    if group_keys is not None and len(group_keys) != len(rows):
        raise ValueError("Each data row needs exactly one comparison-group key")
    output = [row.copy() for row in rows]
    groups = defaultdict(list)
    for index in range(len(rows)):
        groups[group_keys[index] if group_keys is not None else None].append(index)
    for indices in groups.values():
        for column, direction in directions.items():
            values = [(i, numeric_value(rows[i][column])) for i in indices]
            measured = [(i, value) for i, value in values if value is not None]
            if len(measured) < 2:
                continue
            best = (min if direction == "min" else max)(value for _, value in measured)
            for index, value in measured:
                if value == best:
                    output[index][column] = _bold(rows[index][column])
    return output


def highlight_rows(rows: list[list[str]], columns: Sequence[int], direction: str = "max") -> list[list[str]]:
    """Bold the best model value across selected columns of each task row."""
    output = [row.copy() for row in rows]
    for index, row in enumerate(rows):
        highlighted = highlight_columns([[row[column]] for column in columns], {0: direction})
        for column, cell in zip(columns, highlighted):
            output[index][column] = cell[0]
    return output


def map_tables(
    text: str,
    transform: Callable[[list[str], list[list[str]]], list[list[str]]],
) -> str:
    """Transform ordinary pipe-table bodies while preserving surrounding text."""
    lines = text.splitlines()
    output = []
    index = 0

    def cells(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    while index < len(lines):
        if (index + 1 < len(lines) and lines[index].startswith("|")
                and lines[index + 1].startswith("|")
                and all(re.fullmatch(r":?-+:?", cell) for cell in cells(lines[index + 1]))):
            headers = cells(lines[index])
            end = index + 2
            rows = []
            while end < len(lines) and lines[end].startswith("|"):
                row = cells(lines[end])
                if len(row) != len(headers):
                    raise ValueError("Table row width differs from its header")
                rows.append(row)
                end += 1
            changed = transform(headers, rows)
            if len(changed) != len(rows) or any(len(row) != len(headers) for row in changed):
                raise ValueError("Highlighting must preserve table dimensions")
            output.extend(lines[index:index + 2])
            if changed == rows:
                output.extend(lines[index + 2:end])
            else:
                output.extend("| " + " | ".join(row) + " |" for row in changed)
            index = end
        else:
            output.append(lines[index])
            index += 1
    return "\n".join(output) + ("\n" if text.endswith("\n") else "")
