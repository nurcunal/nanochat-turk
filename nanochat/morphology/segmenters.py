"""Morphological segmentation backends for tokenizer experiments.

The invariant in this module is intentionally strict: a segmentation is only
accepted when its surface pieces concatenate back to the original word. Analyzer
outputs with lemmas, feature tags, or abstract morpheme names are not usable for
boundary-constrained tokenizer training until they are converted to surface
pieces.
"""

from __future__ import annotations

import json
import os
import re
import selectors
import shutil
import shlex
import subprocess
import time
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


WORD_RE = re.compile(r"[^\W\d_]+(?:['\u2019][^\W\d_]+)?", flags=re.UNICODE)
DEFAULT_SEPARATORS = ("+", "|", "\u00b7", "\u2022", "\u241f", "\t", " ")
TRMORPH_OVERFLOW_MARKER = "__TRMORPH_OUTPUT_OVERFLOW__"
ANALYSIS_TAGS = {
    "A1sg", "A2sg", "A3sg", "A1pl", "A2pl", "A3pl",
    "Pnon", "P1sg", "P2sg", "P3sg", "P1pl", "P2pl", "P3pl",
    "Nom", "Acc", "Dat", "Loc", "Abl", "Gen", "Ins", "Equ",
    "Past", "Narr", "Prog1", "Prog2", "Fut", "Aor", "Pres",
    "Neg", "Pos", "Able", "Pass", "Caus", "Reflex", "Recip",
    "Neces", "Opt", "Desr", "Cond", "Imp", "Cop", "Ques",
    "Noun", "Verb", "Adj", "Adv", "Conj", "Det", "Pron", "Num",
    "Postp", "Interj", "Punc", "Dup", "Zero", "Inf1", "Inf2",
    "Inf3", "PastPart", "FutPart", "NarrPart", "PresPart",
    "Agt", "ByDoingSo", "SinceDoingSo", "WithoutHavingDoneSo",
    "AsIf", "AfterDoingSo", "Adamantly", "When", "While",
    "Become", "Acquire", "Related",
}


class SegmentationError(RuntimeError):
    """Raised when a segmentation backend returns unusable output."""


class SegmenterUnavailable(RuntimeError):
    """Raised when a requested external segmentation backend is not configured."""


@dataclass(frozen=True)
class MorphemeSpan:
    """A morpheme surface span in the original text."""

    start: int
    end: int
    text: str
    word: str
    source: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"Invalid span offsets: {self.start}:{self.end}")
        if not self.text:
            raise ValueError("MorphemeSpan text cannot be empty")


@dataclass(frozen=True)
class WordSegmentation:
    """Surface segmentation for one word token."""

    word: str
    pieces: tuple[str, ...]
    source: str
    fallback: bool = False
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.word:
            raise ValueError("word cannot be empty")
        if not self.pieces:
            raise ValueError("pieces cannot be empty")
        if any(piece == "" for piece in self.pieces):
            raise ValueError(f"Empty morpheme piece in segmentation for {self.word!r}")
        if "".join(self.pieces) != self.word:
            raise ValueError(
                f"Surface pieces do not reconstruct {self.word!r}: {self.pieces!r}"
            )

    @classmethod
    def identity(
        cls,
        word: str,
        source: str = "identity",
        fallback: bool = False,
        metadata: Mapping[str, str] | None = None,
    ) -> "WordSegmentation":
        return cls(
            word=word,
            pieces=(word,),
            source=source,
            fallback=fallback,
            metadata=metadata or {},
        )

    @classmethod
    def from_pieces(
        cls,
        word: str,
        pieces: Iterable[str],
        source: str,
        fallback: bool = False,
        metadata: Mapping[str, str] | None = None,
    ) -> "WordSegmentation":
        return cls(
            word=word,
            pieces=tuple(pieces),
            source=source,
            fallback=fallback,
            metadata=metadata or {},
        )

    def spans(self, offset: int = 0) -> tuple[MorphemeSpan, ...]:
        spans = []
        cursor = offset
        for piece in self.pieces:
            end = cursor + len(piece)
            spans.append(MorphemeSpan(cursor, end, piece, self.word, self.source))
            cursor = end
        return tuple(spans)

    def delimited(self, delimiter: str = " ") -> str:
        return delimiter.join(self.pieces)


class Segmenter:
    """Base interface for surface morpheme segmenters."""

    name = "segmenter"

    def segment_words(self, words: Sequence[str]) -> list[WordSegmentation]:
        return [self.segment_word(word) for word in words]

    def segment_word(self, word: str) -> WordSegmentation:
        raise NotImplementedError


class IdentitySegmenter(Segmenter):
    """No-op segmenter used for baselines and fallback behavior."""

    name = "identity"

    def segment_word(self, word: str) -> WordSegmentation:
        return WordSegmentation.identity(word, source=self.name)


class BatchCommandSegmenter(Segmenter):
    """Segmenter backed by an external batch command.

    The command receives one word per stdin line and must emit one segmented
    surface word per stdout line. Output can be JSON list syntax or a single
    string using common morpheme separators such as ``+``, ``|``, tabs, or
    spaces. The parsed pieces must reconstruct the original input word.
    """

    def __init__(
        self,
        command: Sequence[str] | str | None,
        name: str = "command",
        timeout: float = 60.0,
        strict: bool = False,
    ):
        if isinstance(command, str):
            command = shlex.split(command)
        self.command = list(command or [])
        self.name = name
        self.timeout = timeout
        self.strict = strict

    def _unavailable(self) -> SegmenterUnavailable:
        return SegmenterUnavailable(
            f"{self.name} segmenter is not configured. Set a command explicitly "
            f"or via the backend environment variable."
        )

    def segment_word(self, word: str) -> WordSegmentation:
        return self.segment_words([word])[0]

    def segment_words(self, words: Sequence[str]) -> list[WordSegmentation]:
        if not words:
            return []
        if not self.command:
            raise self._unavailable()

        input_text = "\n".join(words) + "\n"
        try:
            proc = subprocess.run(
                self.command,
                input=input_text,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise SegmenterUnavailable(
                f"{self.name} command not found: {self.command[0]!r}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise SegmentationError(
                f"{self.name} command timed out after {self.timeout}s"
            ) from exc

        if proc.returncode != 0:
            raise SegmentationError(
                f"{self.name} command failed with exit code {proc.returncode}: "
                f"{proc.stderr.strip()}"
            )

        output_lines = proc.stdout.splitlines()
        if len(output_lines) < len(words):
            raise SegmentationError(
                f"{self.name} returned {len(output_lines)} lines for {len(words)} words"
            )

        segmentations: list[WordSegmentation] = []
        for word, line in zip(words, output_lines):
            try:
                pieces = parse_surface_pieces(line, word)
                segmentations.append(
                    WordSegmentation.from_pieces(word, pieces, source=self.name)
                )
            except SegmentationError as exc:
                if self.strict:
                    raise
                segmentations.append(
                    WordSegmentation.identity(
                        word,
                        source=self.name,
                        fallback=True,
                        metadata={"fallback_reason": str(exc), "raw_output": line},
                    )
                )
        return segmentations


class TRMorphFlookupSegmenter(Segmenter):
    """TRmorph segmenter backed by a compiled ``segment.fst`` and ``flookup``."""

    name = "trmorph"

    def __init__(
        self,
        segment_fst: str = "",
        trmorph_dir: str = "",
        flookup_cmd: str = "flookup",
        flookup_flags: Sequence[str] | str = (),
        pick: str = "fewest_segments",
        max_analyses_per_word: int = 128,
        max_output_lines_per_word: int = 2048,
        timeout: float = 120.0,
        strict: bool = False,
    ):
        self.segment_fst = segment_fst or (
            os.path.join(trmorph_dir, "segment.fst") if trmorph_dir else ""
        )
        self.flookup_cmd = flookup_cmd
        self.flookup_flags = (
            shlex.split(flookup_flags)
            if isinstance(flookup_flags, str)
            else list(flookup_flags)
        )
        self.pick = pick
        self.max_analyses_per_word = max_analyses_per_word
        self.max_output_lines_per_word = max_output_lines_per_word
        self.timeout = timeout
        self.strict = strict

    def _check_available(self) -> None:
        if not self.segment_fst:
            raise SegmenterUnavailable(
                "TRmorph requires TRMORPH_SEGMENT_FST, TRMORPH_DIR, or an "
                "explicit segment_fst."
            )
        if not os.path.isfile(self.segment_fst):
            raise SegmenterUnavailable(
                f"TRmorph segment.fst not found: {self.segment_fst}"
            )
        if shutil.which(self.flookup_cmd) is None and not os.path.exists(
            self.flookup_cmd
        ):
            raise SegmenterUnavailable(
                f"TRmorph flookup command not found: {self.flookup_cmd}"
            )

    def segment_word(self, word: str) -> WordSegmentation:
        return self.segment_words([word])[0]

    def segment_words(self, words: Sequence[str]) -> list[WordSegmentation]:
        if not words:
            return []
        self._check_available()
        try:
            analyses = _run_flookup_batch(
                words,
                flookup_cmd=self.flookup_cmd,
                flookup_flags=self.flookup_flags,
                segment_fst=self.segment_fst,
                timeout=self.timeout,
                max_analyses_per_word=self.max_analyses_per_word,
                max_output_lines_per_word=self.max_output_lines_per_word,
            )
        except SegmentationError as exc:
            if self.strict:
                raise
            if len(words) == 1:
                return [
                    WordSegmentation.identity(
                        words[0],
                        source=self.name,
                        fallback=True,
                        metadata={"fallback_reason": str(exc)},
                    )
                ]
            mid = len(words) // 2
            return self.segment_words(words[:mid]) + self.segment_words(words[mid:])

        out: list[WordSegmentation] = []
        for word, word_analyses in zip(words, analyses):
            if word_analyses == [TRMORPH_OVERFLOW_MARKER]:
                out.append(
                    WordSegmentation.identity(
                        word,
                        source=self.name,
                        fallback=True,
                        metadata={
                            "fallback_reason": (
                                "analysis_output_overflow:"
                                f"{self.max_output_lines_per_word}"
                            )
                        },
                    )
                )
                continue
            chosen = _pick_trmorph_analysis(word_analyses, pick=self.pick)
            if not chosen:
                out.append(
                    WordSegmentation.identity(
                        word,
                        source=self.name,
                        fallback=True,
                        metadata={"fallback_reason": "no_analysis"},
                    )
                )
                continue
            try:
                pieces = parse_surface_pieces(chosen, word)
                out.append(
                    WordSegmentation.from_pieces(word, pieces, source=self.name)
                )
            except SegmentationError as exc:
                if self.strict:
                    raise
                out.append(
                    WordSegmentation.identity(
                        word,
                        source=self.name,
                        fallback=True,
                        metadata={"fallback_reason": str(exc), "raw_output": chosen},
                    )
                )
        return out


class ZemberekSegmenter(Segmenter):
    """Zemberek Python-binding segmenter.

    This adapter is intentionally lazy: importing Zemberek is attempted only
    when the backend is used, because different bindings support different
    Python versions and packaging layouts.
    """

    name = "zemberek"

    def __init__(self, strict: bool = False):
        self.strict = strict
        self._morphology: Any | None = None
        self._cache: dict[str, WordSegmentation] = {}

    def _load_morphology(self) -> Any:
        if self._morphology is not None:
            return self._morphology
        errors = []
        for module_name in ("zemberek", "zemberek_python"):
            try:
                module = __import__(module_name, fromlist=["TurkishMorphology"])
                morphology_cls = getattr(module, "TurkishMorphology")
                self._morphology = morphology_cls.create_with_defaults()
                return self._morphology
            except Exception as exc:
                errors.append(f"{module_name}: {exc}")
        raise SegmenterUnavailable(
            "Zemberek Python backend is not available. Tried: " + "; ".join(errors)
        )

    def segment_word(self, word: str) -> WordSegmentation:
        return self.segment_words([word])[0]

    def segment_words(self, words: Sequence[str]) -> list[WordSegmentation]:
        if not words:
            return []
        morphology = self._load_morphology()
        out = []
        for word in words:
            cached = self._cache.get(word)
            if cached is not None:
                out.append(cached)
                continue
            segmentation = self._segment_one(morphology, word)
            self._cache[word] = segmentation
            out.append(segmentation)
        return out

    def _segment_one(self, morphology: Any, word: str) -> WordSegmentation:
        try:
            analysis = morphology.analyze(word)
            analyses = getattr(analysis, "analysis_results", analysis)
            if not analyses:
                return WordSegmentation.identity(
                    word,
                    source=self.name,
                    fallback=True,
                    metadata={"fallback_reason": "no_analysis"},
                )
            pieces = parse_surface_pieces(
                _zemberek_analysis_to_surface(analyses[0]), word
            )
            return WordSegmentation.from_pieces(word, pieces, source=self.name)
        except Exception as exc:
            if self.strict:
                raise SegmentationError(f"Zemberek failed for {word!r}: {exc}") from exc
            return WordSegmentation.identity(
                word,
                source=self.name,
                fallback=True,
                metadata={"fallback_reason": str(exc)},
            )


class TurkishDelightSegmenter(Segmenter):
    """TurkishDelightNLP REST segmenter."""

    name = "tdelight"

    def __init__(
        self,
        url: str = "",
        endpoint: str = "",
        api_token: str = "",
        response_field: str = "",
        timeout: float = 20.0,
        strict: bool = False,
    ):
        self.url = url.rstrip("/")
        self.endpoint = endpoint
        self.api_token = api_token
        self.response_field = response_field
        self.timeout = timeout
        self.strict = strict
        self._cache: dict[str, WordSegmentation] = {}

    def _endpoints(self) -> list[str]:
        if not self.url:
            raise SegmenterUnavailable(
                "TurkishDelightNLP requires TDELIGHT_URL or an explicit url."
            )
        if self.endpoint:
            endpoint = (
                self.endpoint
                if self.endpoint.startswith("http")
                else self.url + "/" + self.endpoint.lstrip("/")
            )
            return [endpoint]
        return [
            self.url + path
            for path in (
                "/morphological-segmentation",
                "/morphological_segmentation",
                "/morpheme-segmentation",
                "/morpheme_segmentation",
                "/morph-segmentation",
                "/segment",
                "/api/morphological-segmentation",
                "/api/morpheme-segmentation",
                "/api/segment",
            )
        ]

    def segment_word(self, word: str) -> WordSegmentation:
        return self.segment_words([word])[0]

    def segment_words(self, words: Sequence[str]) -> list[WordSegmentation]:
        if not words:
            return []
        endpoints = self._endpoints()
        out = []
        for word in words:
            cached = self._cache.get(word)
            if cached is not None:
                out.append(cached)
                continue
            segmentation = self._segment_one(word, endpoints)
            self._cache[word] = segmentation
            out.append(segmentation)
        return out

    def _segment_one(self, word: str, endpoints: Sequence[str]) -> WordSegmentation:
        last_error: Exception | None = None
        for endpoint in endpoints:
            try:
                response = _post_json(
                    endpoint,
                    {"sentence": word, "text": word, "word": word},
                    api_token=self.api_token,
                    timeout=self.timeout,
                )
                pieces = _extract_tdelight_pieces(
                    response, word, field=self.response_field
                )
                return WordSegmentation.from_pieces(word, pieces, source=self.name)
            except Exception as exc:
                last_error = exc
                continue
        if self.strict:
            raise SegmentationError(
                f"TurkishDelightNLP failed for {word!r}: {last_error}"
            )
        return WordSegmentation.identity(
            word,
            source=self.name,
            fallback=True,
            metadata={"fallback_reason": str(last_error)},
        )


def create_segmenter(
    name: str,
    command: Sequence[str] | str | None = None,
    strict: bool = False,
    timeout: float = 60.0,
) -> Segmenter:
    """Create a segmenter by backend name.

    External backends can be configured directly or through environment:

    - ``trmorph`` reads ``TRMORPH_SEGMENT_CMD``
      or ``TRMORPH_SEGMENT_FST``/``TRMORPH_DIR`` with ``flookup``
    - ``zemberek`` reads ``ZEMBEREK_SEGMENT_CMD``
      or uses a Python binding exposing ``TurkishMorphology``
    - ``tdelight`` reads ``TDELIGHT_SEGMENT_CMD``
      or ``TDELIGHT_URL`` for a REST server
    - ``command`` reads ``MORPH_SEGMENT_CMD`` unless command is provided
    """

    normalized = name.lower().replace("-", "_")
    if normalized in {"identity", "none", "raw", "bpe"}:
        return IdentitySegmenter()

    env_by_name = {
        "trmorph": "TRMORPH_SEGMENT_CMD",
        "zemberek": "ZEMBEREK_SEGMENT_CMD",
        "tdelight": "TDELIGHT_SEGMENT_CMD",
        "turkishdelight": "TDELIGHT_SEGMENT_CMD",
        "turkish_delight": "TDELIGHT_SEGMENT_CMD",
        "command": "MORPH_SEGMENT_CMD",
    }
    if normalized not in env_by_name:
        raise ValueError(f"Unknown morphology segmenter: {name!r}")

    backend_command = command or os.environ.get(env_by_name[normalized], "")
    if backend_command:
        return BatchCommandSegmenter(
            backend_command,
            name=normalized,
            timeout=timeout,
            strict=strict,
        )
    if normalized == "trmorph":
        return TRMorphFlookupSegmenter(
            segment_fst=os.environ.get("TRMORPH_SEGMENT_FST", ""),
            trmorph_dir=os.environ.get("TRMORPH_DIR", ""),
            flookup_cmd=os.environ.get("TRMORPH_FLOOKUP_CMD", "flookup"),
            flookup_flags=os.environ.get("TRMORPH_FLOOKUP_FLAGS", ""),
            pick=os.environ.get("TRMORPH_PICK", "fewest_segments"),
            max_analyses_per_word=int(
                os.environ.get("TRMORPH_MAX_ANALYSES_PER_WORD", "128")
            ),
            max_output_lines_per_word=int(
                os.environ.get("TRMORPH_MAX_OUTPUT_LINES_PER_WORD", "2048")
            ),
            timeout=timeout,
            strict=strict,
        )
    if normalized == "zemberek":
        return ZemberekSegmenter(strict=strict)
    if normalized in {"tdelight", "turkishdelight", "turkish_delight"}:
        return TurkishDelightSegmenter(
            url=os.environ.get("TDELIGHT_URL", "")
            or os.environ.get("TURKISHDELIGHT_URL", ""),
            endpoint=os.environ.get("TDELIGHT_ENDPOINT", ""),
            api_token=os.environ.get("TDELIGHT_TOKEN", ""),
            response_field=os.environ.get("TDELIGHT_RESPONSE_FIELD", ""),
            timeout=timeout,
            strict=strict,
        )
    return BatchCommandSegmenter("", name=normalized, timeout=timeout, strict=strict)


def _run_flookup_batch(
    words: Sequence[str],
    *,
    flookup_cmd: str,
    flookup_flags: Sequence[str] = (),
    segment_fst: str,
    timeout: float,
    max_analyses_per_word: int,
    max_output_lines_per_word: int,
) -> list[list[str]]:
    command = [*shlex.split(flookup_cmd), *flookup_flags, segment_fst]
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None

    input_bytes = ("\n".join(words) + "\n").encode("utf-8")
    try:
        proc.stdin.write(input_bytes)
        proc.stdin.close()
    except BrokenPipeError:
        pass

    analyses: list[list[str]] = []
    current: list[str] = []
    current_output_lines = 0
    current_overflow = False
    stderr_chunks: list[bytes] = []
    stdout_buffer = b""
    deadline = time.monotonic() + timeout

    def finish_current() -> None:
        nonlocal current, current_output_lines, current_overflow
        if current_overflow:
            analyses.append([TRMORPH_OVERFLOW_MARKER])
        else:
            analyses.append(current)
        current = []
        current_output_lines = 0
        current_overflow = False

    def handle_stdout_line(raw_line: bytes) -> None:
        nonlocal current_output_lines, current_overflow
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            finish_current()
            return

        current_output_lines += 1
        if current_output_lines > max_output_lines_per_word:
            current_overflow = True
            return
        if len(current) >= max_analyses_per_word:
            return

        if "\t" in line:
            _input, output = line.split("\t", 1)
            current.append(output.strip())
        else:
            parts = line.split(None, 1)
            if len(parts) == 2:
                current.append(parts[1].strip())
            else:
                current.append(line.strip())

    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
    selector.register(proc.stderr, selectors.EVENT_READ, "stderr")

    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            events = selector.select(timeout=remaining)
            if not events:
                raise subprocess.TimeoutExpired(command, timeout)

            for key, _mask in events:
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                if key.data == "stderr":
                    if sum(len(part) for part in stderr_chunks) < 32768:
                        stderr_chunks.append(chunk)
                    continue

                stdout_buffer += chunk
                lines = stdout_buffer.split(b"\n")
                stdout_buffer = lines.pop()
                for line in lines:
                    handle_stdout_line(line)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        proc.wait()
        raise SegmentationError(f"flookup timed out after {timeout}s") from exc
    finally:
        selector.close()

    if stdout_buffer:
        handle_stdout_line(stdout_buffer)
    if current or current_overflow or len(analyses) < len(words):
        finish_current()

    returncode = proc.wait()
    stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace").strip()
    if returncode != 0:
        raise SegmentationError(
            f"flookup failed with exit code {returncode}: {stderr}"
        )

    while len(analyses) < len(words):
        analyses.append([])
    return analyses[: len(words)]


def _pick_trmorph_analysis(analyses: Sequence[str], pick: str) -> str:
    cleaned = []
    seen = set()
    for analysis in analyses:
        value = (analysis or "").strip()
        if not value or value == "?":
            continue
        if value not in seen:
            seen.add(value)
            cleaned.append(value)
    if not cleaned:
        return ""
    if pick == "first":
        return cleaned[0]

    def score(value: str) -> tuple[int, int, str]:
        return (value.count("+") + 1, len(value), value)

    if pick == "most_segments":
        return sorted(
            cleaned, key=lambda value: (-score(value)[0], score(value)[1], value)
        )[0]
    if pick == "shortest":
        return sorted(cleaned, key=lambda value: (len(value), value.count("+"), value))[0]
    return sorted(cleaned, key=score)[0]


def _zemberek_analysis_to_surface(analysis: Any) -> str:
    text = str(analysis)
    if "]" in text:
        text = text.split("]", 1)[1].strip()
    if "+" not in text:
        return text
    pieces = []
    for chunk in text.split("+"):
        chunk = chunk.strip()
        if not chunk:
            continue
        pieces.append(chunk.split(":", 1)[0].strip() or chunk)
    return "+".join(pieces)


def _post_json(
    url: str,
    payload: Mapping[str, str],
    *,
    api_token: str,
    timeout: float,
) -> Mapping[str, Any]:
    data = json.dumps(dict(payload)).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_token:
        headers["token"] = api_token
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    parsed = json.loads(body)
    if not isinstance(parsed, Mapping):
        raise SegmentationError(f"Unexpected TurkishDelightNLP response: {parsed!r}")
    return parsed


def _extract_tdelight_pieces(
    response: Mapping[str, Any],
    word: str,
    *,
    field: str,
) -> tuple[str, ...]:
    if field:
        value = response.get(field)
        if isinstance(value, list):
            return _validate_surface_pieces(tuple(str(piece) for piece in value), word, repr(response))
        if isinstance(value, str):
            return parse_surface_pieces(value, word)

    morphemes = response.get("morphemes")
    if isinstance(morphemes, Mapping):
        value = morphemes.get(word)
        if isinstance(value, list):
            return _validate_surface_pieces(tuple(str(piece) for piece in value), word, repr(response))
        if isinstance(value, str):
            return parse_surface_pieces(value, word)

    for key in ("pieces", "segments", "segmentation", "result", "output", "text"):
        value = response.get(key)
        if isinstance(value, list):
            return _validate_surface_pieces(tuple(str(piece) for piece in value), word, repr(response))
        if isinstance(value, str):
            return parse_surface_pieces(value, word)

    raise SegmentationError(f"No TurkishDelightNLP pieces found for {word!r}: {response!r}")


def parse_surface_pieces(raw: str, word: str) -> tuple[str, ...]:
    """Parse one backend output line into surface morpheme pieces."""

    line = raw.strip()
    if not line:
        raise SegmentationError(f"Empty segmentation output for {word!r}")

    json_pieces = _parse_json_pieces(line)
    if json_pieces is not None:
        return _validate_surface_pieces(json_pieces, word, raw)

    candidates = [line]
    for prefix in (word + "\t", word + " ", word + " => ", word + " -> "):
        if line.startswith(prefix):
            candidates.append(line[len(prefix) :])

    for candidate in candidates:
        cleaned = _strip_analysis_noise(candidate)
        for separator in DEFAULT_SEPARATORS:
            if separator in cleaned:
                pieces = tuple(piece for piece in cleaned.split(separator) if piece)
                try:
                    return _validate_surface_pieces(pieces, word, raw)
                except SegmentationError:
                    tagless = _drop_analysis_tag_pieces(pieces)
                    if tagless != pieces:
                        try:
                            return _validate_surface_pieces(tagless, word, raw)
                        except SegmentationError:
                            pass
        try:
            return _validate_surface_pieces((cleaned,), word, raw)
        except SegmentationError:
            pass

    raise SegmentationError(f"Could not parse surface pieces for {word!r}: {raw!r}")


def _parse_json_pieces(line: str) -> tuple[str, ...] | None:
    if not (line.startswith("[") or line.startswith("{")):
        return None
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return None
    if isinstance(value, list):
        return tuple(str(piece) for piece in value)
    if isinstance(value, dict):
        pieces = value.get("pieces") or value.get("morphemes") or value.get("segments")
        if isinstance(pieces, list):
            return tuple(str(piece) for piece in pieces)
    return None


def _strip_analysis_noise(text: str) -> str:
    text = text.strip()
    # Common unknown markers from analyzers belong to the analysis, not surface.
    text = text.strip("?")
    # If a backend emits token<TAG>+suffix<TAG>, remove feature tags.
    text = re.sub(r"<[^>]+>", "", text)
    # Some segmenters mark suffixes as -ler. The hyphen is not surface text.
    text = re.sub(r"(?<=\S)-(?=\S)", "+", text)
    return text.strip()


def _validate_surface_pieces(pieces: Sequence[str], word: str, raw: str) -> tuple[str, ...]:
    normalized = tuple(piece.strip() for piece in pieces if piece.strip())
    if not normalized:
        raise SegmentationError(f"No pieces parsed for {word!r}: {raw!r}")
    if "".join(normalized) != word:
        repaired = _repair_case_only_surface_pieces(normalized, word)
        if repaired is not None:
            return repaired
        raise SegmentationError(
            f"Pieces do not reconstruct {word!r}: {normalized!r} from {raw!r}"
        )
    return normalized


def _drop_analysis_tag_pieces(pieces: Sequence[str]) -> tuple[str, ...]:
    out = []
    for piece in pieces:
        for part in str(piece).split("|"):
            surface = part.split(":", 1)[0].strip()
            if not surface or surface in ANALYSIS_TAGS:
                continue
            out.append(surface)
    return tuple(out)


def _repair_case_only_surface_pieces(
    pieces: Sequence[str],
    word: str,
) -> tuple[str, ...] | None:
    joined = "".join(pieces)
    if len(joined) != len(word) or joined.casefold() != word.casefold():
        return None

    repaired = []
    cursor = 0
    for piece in pieces:
        end = cursor + len(piece)
        repaired.append(word[cursor:end])
        cursor = end
    return tuple(repaired)


def iter_word_spans(text: str) -> Iterable[tuple[int, int, str]]:
    """Yield word-like spans that should be considered for morphology."""

    for match in WORD_RE.finditer(text):
        yield match.start(), match.end(), match.group(0)


def segment_text(text: str, segmenter: Segmenter, delimiter: str = " ") -> str:
    """Return text with each word replaced by delimiter-joined morpheme pieces."""

    spans = list(iter_word_spans(text))
    if not spans:
        return text

    words = [word for _, _, word in spans]
    segmentations = segmenter.segment_words(words)

    chunks: list[str] = []
    cursor = 0
    for (start, end, _word), segmentation in zip(spans, segmentations):
        chunks.append(text[cursor:start])
        chunks.append(segmentation.delimited(delimiter))
        cursor = end
    chunks.append(text[cursor:])
    return "".join(chunks)


def segment_text_spans(text: str, segmenter: Segmenter) -> list[MorphemeSpan]:
    """Return all morpheme spans for word-like tokens in text."""

    spans = list(iter_word_spans(text))
    if not spans:
        return []

    segmentations = segmenter.segment_words([word for _, _, word in spans])
    out: list[MorphemeSpan] = []
    for (start, _end, _word), segmentation in zip(spans, segmentations):
        out.extend(segmentation.spans(offset=start))
    return out
