"""
Measure tokenizer compression, fertility, boundary behavior, and throughput.

Tokenizer-only metrics cannot report model BPB; true BPB depends on a trained
model's loss. This script reports the tokenizer-side quantities we can measure
before pretraining: bytes/token, tokens/word fertility, word fragmentation,
round-trip safety, encode speed, vocabulary shape, and MorphBPE boundary
crossing rates when boundary-marked text is supplied.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import re
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from nanochat.dataset import DATA_DIR, list_parquet_files, parquets_iter_batched
from nanochat.morphology import MORPHEME_BOUNDARY, strip_morpheme_boundaries
from nanochat.morphology.morphbpe import strip_boundaries_with_offsets
from nanochat.tokenizer import (
    RustBPETokenizer,
    get_tokenizer,
    get_tokenizer_dir,
    get_tokenizer_name,
)
from tokenizers import BertWordPieceTokenizer
from tokenizers import Tokenizer as HFTokenizer


WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
SEGMENTED_WORD_RE = re.compile(
    rf"[\w{re.escape(MORPHEME_BOUNDARY)}]+",
    flags=re.UNICODE,
)

_WORKER_TOKENIZER = None
_WORKER_TEXT_COLUMN = "text"
_WORKER_INPUT_HAS_MORPH_BOUNDARIES = False
_WORKER_MORPH_BOUNDARY = MORPHEME_BOUNDARY
_WORKER_NUM_THREADS = 1


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] * (hi - pos) + xs[hi] * (pos - lo))


def distribution_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0,
        }
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def _value_at_rank(counts: Counter[int], rank: int) -> int:
    seen = 0
    for value in sorted(counts):
        seen += counts[value]
        if rank < seen:
            return value
    return max(counts) if counts else 0


def percentile_from_counts(counts: Counter[int], q: float) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    if total == 1:
        return float(_value_at_rank(counts, 0))
    pos = (total - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(_value_at_rank(counts, lo))
    lo_value = _value_at_rank(counts, lo)
    hi_value = _value_at_rank(counts, hi)
    return float(lo_value * (hi - pos) + hi_value * (pos - lo))


def distribution_stats_from_counts(counts: Counter[int]) -> dict[str, float | int]:
    total = sum(counts.values())
    if total == 0:
        return distribution_stats([])
    weighted_sum = sum(value * count for value, count in counts.items())
    return {
        "count": total,
        "mean": weighted_sum / total,
        "p50": percentile_from_counts(counts, 0.50),
        "p90": percentile_from_counts(counts, 0.90),
        "p95": percentile_from_counts(counts, 0.95),
        "p99": percentile_from_counts(counts, 0.99),
        "max": max(counts),
    }


def mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {"mean": mean, "std": math.sqrt(variance)}


def load_tokenizer_config(tokenizer_dir: str) -> dict[str, Any]:
    path = Path(tokenizer_dir) / "tokenizer_config.json"
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tokenizer(tokenizer_dir: str):
    if tokenizer_dir:
        return RustBPETokenizer.from_directory(tokenizer_dir)
    return get_tokenizer()


def hf_download_file(repo_id: str, filename: str) -> str:
    try:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_files_only=True,
        )
    except Exception:
        return hf_hub_download(repo_id=repo_id, filename=filename)


class HuggingFaceTokenizerAdapter:
    """Small compatibility wrapper around Hugging Face `tokenizers` objects."""

    def __init__(
        self,
        backend: HFTokenizer,
        *,
        name: str,
        model_id: str,
        implementation: str,
        batch_size: int,
    ) -> None:
        self.backend = backend
        self.decode_strip = ""
        self.batch_size = batch_size
        self.tokenizer_config = {
            "name": name,
            "implementation": implementation,
            "source": "huggingface",
            "model_id": model_id,
            "vocab_size": backend.get_vocab_size(with_added_tokens=True),
        }

    @classmethod
    def from_identifier(
        cls,
        identifier: str,
        *,
        name: str,
        implementation: str,
        batch_size: int,
    ) -> "HuggingFaceTokenizerAdapter":
        path = Path(identifier)
        backend: HFTokenizer
        wants_wordpiece = "wordpiece" in implementation.lower()
        if path.is_file() and path.suffix == ".json":
            backend = HFTokenizer.from_file(str(path))
        elif path.is_dir() and (path / "tokenizer.json").is_file():
            backend = HFTokenizer.from_file(str(path / "tokenizer.json"))
        elif path.is_file() and path.name == "vocab.txt":
            backend = BertWordPieceTokenizer(str(path), lowercase=False, strip_accents=False)
        elif path.is_dir() and (path / "vocab.txt").is_file():
            backend = BertWordPieceTokenizer(
                str(path / "vocab.txt"),
                lowercase=False,
                strip_accents=False,
            )
        elif wants_wordpiece:
            vocab_path = hf_download_file(identifier, "vocab.txt")
            backend = BertWordPieceTokenizer(
                vocab_path,
                lowercase=False,
                strip_accents=False,
            )
        else:
            tokenizer_path = hf_download_file(identifier, "tokenizer.json")
            backend = HFTokenizer.from_file(tokenizer_path)
        resolved_name = name or identifier
        resolved_impl = implementation or type(backend.model).__name__.lower()
        return cls(
            backend,
            name=resolved_name,
            model_id=identifier,
            implementation=resolved_impl,
            batch_size=batch_size,
        )

    def encode(self, text_or_texts: str | list[str], num_threads: int = 0):
        del num_threads
        if isinstance(text_or_texts, str):
            return self.backend.encode(text_or_texts, add_special_tokens=False).ids
        return self.encode_with_offsets(text_or_texts)[0]

    def encode_with_offsets(
        self,
        texts: list[str],
    ) -> tuple[list[list[int]], list[list[tuple[int, int]]]]:
        all_ids: list[list[int]] = []
        all_offsets: list[list[tuple[int, int]]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start:start + self.batch_size]
            encodings = self.backend.encode_batch(chunk, add_special_tokens=False)
            all_ids.extend(encoding.ids for encoding in encodings)
            all_offsets.extend(encoding.offsets for encoding in encodings)
        return all_ids, all_offsets

    def decode(self, ids: list[int]) -> str:
        return self.backend.decode(ids, skip_special_tokens=False)

    def get_vocab_size(self) -> int:
        return self.backend.get_vocab_size(with_added_tokens=True)


def char_offsets_to_byte_offsets(text: str, char_offsets: tuple[int, ...]) -> set[int]:
    wanted = set(char_offsets)
    out: set[int] = set()
    byte_pos = 0
    for idx, ch in enumerate(text):
        if idx in wanted:
            out.add(byte_pos)
        byte_pos += len(ch.encode("utf-8"))
    if len(text) in wanted:
        out.add(byte_pos)
    return out


def token_byte_lengths(tokenizer, ids: list[int]) -> list[int]:
    return [len(tokenizer.enc.decode_single_token_bytes(token_id)) for token_id in ids]


def boundary_violation_stats(
    tokenizer,
    encoded: list[list[int]],
    visible_docs: list[str],
    boundary_offsets_by_doc: list[tuple[int, ...]],
    token_offsets_by_doc: list[list[tuple[int, int]]] | None = None,
) -> dict[str, float | int]:
    token_len_cache: dict[int, int] = {}
    boundary_count = 0
    crossed_boundaries = 0
    crossing_tokens = 0
    docs_with_crossing = 0
    eligible_docs = 0

    for doc_idx, (ids, text, char_offsets) in enumerate(zip(encoded, visible_docs, boundary_offsets_by_doc)):
        if not char_offsets:
            continue
        eligible_docs += 1
        boundary_count += len(char_offsets)
        doc_crossing = False
        if token_offsets_by_doc is not None:
            sorted_offsets = sorted(char_offsets)
            token_offsets = token_offsets_by_doc[doc_idx]
            for start, end in token_offsets:
                left = bisect.bisect_right(sorted_offsets, start)
                right = bisect.bisect_left(sorted_offsets, end)
                crossed_here = right - left
                if crossed_here:
                    crossing_tokens += 1
                    crossed_boundaries += crossed_here
                    doc_crossing = True
        else:
            byte_offsets = sorted(char_offsets_to_byte_offsets(text, char_offsets))
            cursor = 0
            for token_id in ids:
                token_len = token_len_cache.get(token_id)
                if token_len is None:
                    token_len = len(tokenizer.enc.decode_single_token_bytes(token_id))
                    token_len_cache[token_id] = token_len
                start = cursor
                end = cursor + token_len
                left = bisect.bisect_right(byte_offsets, start)
                right = bisect.bisect_left(byte_offsets, end)
                crossed_here = right - left
                if crossed_here:
                    crossing_tokens += 1
                    crossed_boundaries += crossed_here
                    doc_crossing = True
                cursor = end
        if doc_crossing:
            docs_with_crossing += 1

    return {
        "eligible_docs": eligible_docs,
        "boundary_count": boundary_count,
        "crossing_tokens": crossing_tokens,
        "crossed_boundaries": crossed_boundaries,
        "docs_with_crossing": docs_with_crossing,
        "crossed_boundary_rate": safe_div(crossed_boundaries, boundary_count),
        "docs_with_crossing_rate": safe_div(docs_with_crossing, eligible_docs),
    }


def word_fertility_stats(
    tokenizer,
    docs: list[str],
    *,
    max_words: int,
    num_threads: int,
) -> dict[str, Any]:
    words: list[str] = []
    for doc in docs:
        words.extend(WORD_RE.findall(doc))
        if max_words > 0 and len(words) >= max_words:
            words = words[:max_words]
            break
    if not words:
        return {
            "sample_words": 0,
            "unique_words": 0,
            "tokens": 0,
            "tokens_per_word": 0.0,
            "single_token_word_rate": 0.0,
            "multi_token_word_rate": 0.0,
            "token_count_distribution": distribution_stats([]),
            "long_word_tokens_per_word": 0.0,
            "long_word_count": 0,
        }

    unique_words = list(dict.fromkeys(words))
    encoded_unique = tokenizer.encode(unique_words, num_threads=num_threads)
    lengths_by_word = {
        word: len(ids)
        for word, ids in zip(unique_words, encoded_unique)
    }
    lengths = [lengths_by_word[word] for word in words]
    long_lengths = [
        length for word, length in zip(words, lengths)
        if len(word) >= 8
    ]
    single = sum(1 for length in lengths if length == 1)
    return {
        "sample_words": len(words),
        "unique_words": len(unique_words),
        "tokens": sum(lengths),
        "tokens_per_word": safe_div(sum(lengths), len(words)),
        "single_token_word_rate": safe_div(single, len(words)),
        "multi_token_word_rate": safe_div(len(words) - single, len(words)),
        "token_count_distribution": distribution_stats(lengths),
        "long_word_tokens_per_word": safe_div(sum(long_lengths), len(long_lengths)),
        "long_word_count": len(long_lengths),
    }


def word_fertility_stats_from_words(
    tokenizer,
    words: list[str],
    *,
    num_threads: int,
) -> dict[str, Any]:
    if not words:
        return {
            "sample_words": 0,
            "unique_words": 0,
            "tokens": 0,
            "tokens_per_word": 0.0,
            "single_token_word_rate": 0.0,
            "multi_token_word_rate": 0.0,
            "token_count_distribution": distribution_stats([]),
            "long_word_tokens_per_word": 0.0,
            "long_word_count": 0,
        }

    unique_words = list(dict.fromkeys(words))
    encoded_unique = tokenizer.encode(unique_words, num_threads=num_threads)
    lengths_by_word = {
        word: len(ids)
        for word, ids in zip(unique_words, encoded_unique)
    }
    lengths = [lengths_by_word[word] for word in words]
    long_lengths = [
        length for word, length in zip(words, lengths)
        if len(word) >= 8
    ]
    single = sum(1 for length in lengths if length == 1)
    return {
        "sample_words": len(words),
        "unique_words": len(unique_words),
        "tokens": sum(lengths),
        "tokens_per_word": safe_div(sum(lengths), len(words)),
        "single_token_word_rate": safe_div(single, len(words)),
        "multi_token_word_rate": safe_div(len(words) - single, len(words)),
        "token_count_distribution": distribution_stats(lengths),
        "long_word_tokens_per_word": safe_div(sum(long_lengths), len(long_lengths)),
        "long_word_count": len(long_lengths),
    }


def extract_segmented_words(
    text: str,
    *,
    boundary: str = MORPHEME_BOUNDARY,
) -> list[tuple[str, tuple[str, ...]]]:
    """Return `(surface, morphemes)` pairs from a boundary-marked document."""

    out: list[tuple[str, tuple[str, ...]]] = []
    for raw_word in SEGMENTED_WORD_RE.findall(text):
        morphemes = tuple(part for part in raw_word.split(boundary) if part)
        if not morphemes:
            continue
        surface = "".join(morphemes)
        if surface:
            out.append((surface, morphemes))
    return out


def collect_morph_metric_words(
    *,
    split: str,
    data_dir: str,
    text_column: str,
    morph_boundary: str,
    max_docs: int,
    max_words: int,
) -> list[tuple[str, tuple[str, ...]]]:
    """Collect segmented word references for MorphBPE paper metrics."""

    if max_words < 0:
        return []

    words: list[tuple[str, tuple[str, ...]]] = []
    docs_seen = 0
    for batch in parquets_iter_batched(
        split=split,
        data_dir=data_dir,
        text_column=text_column,
    ):
        if max_docs > 0:
            remaining_docs = max_docs - docs_seen
            if remaining_docs <= 0:
                break
            batch = batch[:remaining_docs]
        for doc in batch:
            words.extend(
                extract_segmented_words(
                    doc,
                    boundary=morph_boundary,
                )
            )
            if max_words > 0 and len(words) >= max_words:
                return words[:max_words]
        docs_seen += len(batch)
    return words


def token_piece_bytes(tokenizer, token_id: int) -> bytes:
    if hasattr(tokenizer, "enc"):
        return tokenizer.enc.decode_single_token_bytes(token_id)
    return tokenizer.decode([token_id]).encode("utf-8")


def token_piece_sequence_bytes(tokenizer, ids: list[int]) -> tuple[bytes, ...]:
    return tuple(token_piece_bytes(tokenizer, token_id) for token_id in ids)


def sequence_edit_distance(a: tuple[bytes, ...], b: tuple[bytes, ...]) -> int:
    """Levenshtein distance over morpheme/token byte-piece sequences."""

    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, start=1):
        current = [i]
        for j, right in enumerate(b, start=1):
            substitution = previous[j - 1] + (0 if left == right else 1)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def morphological_alignment_stats(
    tokenizer,
    segmented_words: list[tuple[str, tuple[str, ...]]],
    *,
    num_threads: int,
) -> dict[str, Any]:
    """Paper-style Morphological Edit Distance (`mu_e`) metrics."""

    if not segmented_words:
        return {
            "sample_word_occurrences": 0,
            "unique_surface_words": 0,
            "morphological_edit_distance": 0.0,
            "morphological_edit_distance_normalized": 0.0,
            "morphological_edit_distance_unique": 0.0,
            "morphological_edit_distance_normalized_unique": 0.0,
            "exact_morpheme_sequence_rate": 0.0,
            "mean_morphemes_per_word": 0.0,
            "mean_token_pieces_per_word": 0.0,
            "edit_distance_distribution": distribution_stats([]),
        }

    unique_surfaces = list(dict.fromkeys(surface for surface, _ in segmented_words))
    encoded_unique = tokenizer.encode(unique_surfaces, num_threads=num_threads)
    token_pieces_by_surface = {
        surface: token_piece_sequence_bytes(tokenizer, ids)
        for surface, ids in zip(unique_surfaces, encoded_unique)
    }

    occurrence_distances: list[int] = []
    occurrence_normalized: list[float] = []
    occurrence_morpheme_counts: list[int] = []
    occurrence_token_counts: list[int] = []
    exact_occurrences = 0

    first_by_surface: dict[str, tuple[str, ...]] = {}
    for surface, morphemes in segmented_words:
        first_by_surface.setdefault(surface, morphemes)
        gold = tuple(part.encode("utf-8") for part in morphemes)
        pred = token_pieces_by_surface[surface]
        distance = sequence_edit_distance(gold, pred)
        occurrence_distances.append(distance)
        occurrence_normalized.append(safe_div(distance, len(gold)))
        occurrence_morpheme_counts.append(len(gold))
        occurrence_token_counts.append(len(pred))
        if gold == pred:
            exact_occurrences += 1

    unique_distances: list[int] = []
    unique_normalized: list[float] = []
    for surface, morphemes in first_by_surface.items():
        gold = tuple(part.encode("utf-8") for part in morphemes)
        pred = token_pieces_by_surface[surface]
        distance = sequence_edit_distance(gold, pred)
        unique_distances.append(distance)
        unique_normalized.append(safe_div(distance, len(gold)))

    return {
        "sample_word_occurrences": len(segmented_words),
        "unique_surface_words": len(unique_surfaces),
        "morphological_edit_distance": safe_div(sum(occurrence_distances), len(occurrence_distances)),
        "morphological_edit_distance_normalized": safe_div(sum(occurrence_normalized), len(occurrence_normalized)),
        "morphological_edit_distance_unique": safe_div(sum(unique_distances), len(unique_distances)),
        "morphological_edit_distance_normalized_unique": safe_div(sum(unique_normalized), len(unique_normalized)),
        "exact_morpheme_sequence_rate": safe_div(exact_occurrences, len(segmented_words)),
        "mean_morphemes_per_word": safe_div(sum(occurrence_morpheme_counts), len(occurrence_morpheme_counts)),
        "mean_token_pieces_per_word": safe_div(sum(occurrence_token_counts), len(occurrence_token_counts)),
        "edit_distance_distribution": distribution_stats(occurrence_distances),
    }


def stable_hash_int(value: bytes | str) -> int:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


def cluster_morpheme_sets(
    surfaces: list[str],
    morpheme_sets: list[set[bytes]],
    *,
    n_clusters: int,
    seed: int,
) -> tuple[list[list[int]], str]:
    n_items = len(surfaces)
    if n_items == 0:
        return [], "empty"
    k = max(1, min(n_clusters, n_items))
    if k == n_items:
        return [[idx] for idx in range(n_items)], "singleton"

    try:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.feature_extraction import FeatureHasher

        features = [
            {piece.hex(): 1.0 for piece in sorted(morph_set)}
            for morph_set in morpheme_sets
        ]
        matrix = FeatureHasher(
            n_features=512,
            input_type="dict",
            alternate_sign=False,
        ).transform(features)
        labels = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            batch_size=min(4096, max(256, n_items)),
            n_init=3,
        ).fit_predict(matrix)
        clusters: list[list[int]] = [[] for _ in range(k)]
        for idx, label in enumerate(labels):
            clusters[int(label)].append(idx)
        return [cluster for cluster in clusters if cluster], "sklearn_minibatch_kmeans"
    except Exception:
        clusters = [[] for _ in range(k)]
        for idx, (surface, morph_set) in enumerate(zip(surfaces, morpheme_sets)):
            key = min(morph_set, default=surface.encode("utf-8"))
            clusters[stable_hash_int(key) % k].append(idx)
        return [cluster for cluster in clusters if cluster], "hash_fallback"


def sample_cluster_pairs(
    cluster: list[int],
    *,
    pairs_per_cluster: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    if len(cluster) < 2 or pairs_per_cluster <= 0:
        return []
    max_pairs = len(cluster) * (len(cluster) - 1) // 2
    if max_pairs <= pairs_per_cluster:
        return [
            (cluster[i], cluster[j])
            for i in range(len(cluster))
            for j in range(i + 1, len(cluster))
        ]

    pairs: set[tuple[int, int]] = set()
    while len(pairs) < pairs_per_cluster:
        left, right = rng.sample(cluster, 2)
        if left > right:
            left, right = right, left
        pairs.add((left, right))
    return list(pairs)


def f1_score(precision: float, recall: float) -> float:
    return safe_div(2.0 * precision * recall, precision + recall)


def morphological_consistency_stats(
    tokenizer,
    segmented_words: list[tuple[str, tuple[str, ...]]],
    *,
    num_threads: int,
    max_words: int,
    n_clusters: int,
    pairs_per_cluster: int,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Paper-style Morphological Consistency precision/recall/F1 (`mu_c`)."""

    if not segmented_words:
        return {
            "sample_unique_words": 0,
            "clusters_requested": n_clusters,
            "clusters_used": 0,
            "pairs_per_cluster": pairs_per_cluster,
            "resamples": resamples,
            "clustering": "empty",
            "precision_mean": 0.0,
            "precision_std": 0.0,
            "recall_mean": 0.0,
            "recall_std": 0.0,
            "f1_mean": 0.0,
            "f1_std": 0.0,
            "f1_from_mean_precision_recall": 0.0,
            "mean_pairs_per_resample": 0.0,
        }

    by_surface: dict[str, tuple[str, ...]] = {}
    for surface, morphemes in segmented_words:
        by_surface.setdefault(surface, morphemes)
        if max_words > 0 and len(by_surface) >= max_words:
            break

    surfaces = list(by_surface)
    encoded = tokenizer.encode(surfaces, num_threads=num_threads)
    morph_sets = [
        {part.encode("utf-8") for part in by_surface[surface]}
        for surface in surfaces
    ]
    token_sets = [
        set(token_piece_sequence_bytes(tokenizer, ids))
        for ids in encoded
    ]
    clusters, clustering_method = cluster_morpheme_sets(
        surfaces,
        morph_sets,
        n_clusters=n_clusters,
        seed=seed,
    )

    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    pair_counts: list[int] = []
    for sample_idx in range(max(1, resamples)):
        rng = random.Random(seed + sample_idx)
        true_positive = 0
        token_shared_pairs = 0
        morph_shared_pairs = 0
        total_pairs = 0
        for cluster in clusters:
            for left, right in sample_cluster_pairs(
                cluster,
                pairs_per_cluster=pairs_per_cluster,
                rng=rng,
            ):
                shares_morpheme = bool(morph_sets[left] & morph_sets[right])
                shares_token = bool(token_sets[left] & token_sets[right])
                if shares_token:
                    token_shared_pairs += 1
                if shares_morpheme:
                    morph_shared_pairs += 1
                if shares_token and shares_morpheme:
                    true_positive += 1
                total_pairs += 1
        precision = safe_div(true_positive, token_shared_pairs)
        recall = safe_div(true_positive, morph_shared_pairs)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1_score(precision, recall))
        pair_counts.append(total_pairs)

    precision_stats = mean_std(precisions)
    recall_stats = mean_std(recalls)
    f1_stats = mean_std(f1s)
    return {
        "sample_unique_words": len(surfaces),
        "clusters_requested": n_clusters,
        "clusters_used": len(clusters),
        "pairs_per_cluster": pairs_per_cluster,
        "resamples": max(1, resamples),
        "clustering": clustering_method,
        "precision_mean": precision_stats["mean"],
        "precision_std": precision_stats["std"],
        "recall_mean": recall_stats["mean"],
        "recall_std": recall_stats["std"],
        "f1_mean": f1_stats["mean"],
        "f1_std": f1_stats["std"],
        "f1_from_mean_precision_recall": f1_score(
            precision_stats["mean"],
            recall_stats["mean"],
        ),
        "mean_pairs_per_resample": safe_div(sum(pair_counts), len(pair_counts)),
    }


def vocabulary_stats(tokenizer) -> dict[str, Any]:
    if isinstance(tokenizer, HuggingFaceTokenizerAdapter):
        vocab = tokenizer.backend.get_vocab(with_added_tokens=True)
        decoded_lengths: list[int] = []
        utf8_decodable = 0
        for token_id in sorted(vocab.values()):
            decoded = tokenizer.decode([token_id])
            if not decoded:
                continue
            decoded_lengths.append(len(decoded.encode("utf-8")))
            utf8_decodable += 1
        return {
            "vocab_size": tokenizer.get_vocab_size(),
            "special_tokens": tokenizer.get_vocab_size() - len(tokenizer.backend.get_vocab(with_added_tokens=False)),
            "mergeable_tokens_seen": len(decoded_lengths),
            "utf8_decodable_token_rate": safe_div(utf8_decodable, len(decoded_lengths)),
            "token_byte_length_distribution": distribution_stats(decoded_lengths),
        }

    special_ids = {
        tokenizer.encode_special(token)
        for token in tokenizer.get_special_tokens()
    }
    byte_lengths: list[int] = []
    utf8_decodable = 0
    for token_id in range(tokenizer.get_vocab_size()):
        if token_id in special_ids:
            continue
        try:
            token_bytes = tokenizer.enc.decode_single_token_bytes(token_id)
        except KeyError:
            continue
        byte_lengths.append(len(token_bytes))
        try:
            token_bytes.decode("utf-8")
            utf8_decodable += 1
        except UnicodeDecodeError:
            pass
    return {
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": len(special_ids),
        "mergeable_tokens_seen": len(byte_lengths),
        "utf8_decodable_token_rate": safe_div(utf8_decodable, len(byte_lengths)),
        "token_byte_length_distribution": distribution_stats(byte_lengths),
    }


def load_requested_tokenizer(
    *,
    tokenizer_dir_arg: str,
    hf_tokenizer: str,
    tokenizer_name: str,
    tokenizer_implementation: str,
    hf_batch_size: int,
):
    if hf_tokenizer and tokenizer_dir_arg:
        raise ValueError("Use only one of --hf-tokenizer or --tokenizer-dir")

    if hf_tokenizer:
        tokenizer = HuggingFaceTokenizerAdapter.from_identifier(
            hf_tokenizer,
            name=tokenizer_name,
            implementation=tokenizer_implementation,
            batch_size=hf_batch_size,
        )
        return tokenizer, hf_tokenizer, tokenizer.tokenizer_config

    tokenizer_dir = tokenizer_dir_arg or get_tokenizer_dir()
    tokenizer_config = load_tokenizer_config(tokenizer_dir)
    tokenizer = load_tokenizer(tokenizer_dir_arg)
    return tokenizer, tokenizer_dir, tokenizer_config


def empty_metric_accumulator() -> dict[str, Any]:
    return {
        "docs_count": 0,
        "total_tokens": 0,
        "total_bytes": 0,
        "total_chars": 0,
        "total_words": 0,
        "unique_tokens": set(),
        "roundtrip_failures": 0,
        "docs_with_decode_strip": 0,
        "encode_seconds": 0.0,
        "token_length_counts": Counter(),
        "boundary_totals": {
            "eligible_docs": 0,
            "boundary_count": 0,
            "crossing_tokens": 0,
            "crossed_boundaries": 0,
            "docs_with_crossing": 0,
        },
        "row_group_tasks": 0,
        "metric_wall_seconds": 0.0,
        "effective_workers": 1,
    }


def merge_metric_accumulator(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key in (
        "docs_count",
        "total_tokens",
        "total_bytes",
        "total_chars",
        "total_words",
        "roundtrip_failures",
        "docs_with_decode_strip",
        "row_group_tasks",
    ):
        dst[key] += src[key]
    dst["encode_seconds"] += src["encode_seconds"]
    dst["unique_tokens"].update(src["unique_tokens"])
    dst["token_length_counts"].update(src["token_length_counts"])
    for key in dst["boundary_totals"]:
        dst["boundary_totals"][key] += int(src["boundary_totals"].get(key, 0) or 0)


def prepare_visible_docs(
    tokenizer,
    docs: list[str],
    *,
    input_has_morph_boundaries: bool,
    morph_boundary: str,
) -> tuple[list[str], list[tuple[int, ...]], int]:
    decode_strip = getattr(tokenizer, "decode_strip", "")
    visible_docs: list[str] = []
    boundary_offsets_by_doc: list[tuple[int, ...]] = []
    docs_with_decode_strip = 0
    for doc in docs:
        if decode_strip and decode_strip in doc:
            docs_with_decode_strip += 1
        if input_has_morph_boundaries:
            visible, offsets = strip_boundaries_with_offsets(
                doc,
                boundary=morph_boundary,
            )
            visible_docs.append(visible)
            boundary_offsets_by_doc.append(offsets)
        else:
            visible_docs.append(
                strip_morpheme_boundaries(doc, decode_strip)
                if decode_strip
                else doc
            )
            boundary_offsets_by_doc.append(())
    return visible_docs, boundary_offsets_by_doc, docs_with_decode_strip


def measure_docs_batch(
    tokenizer,
    docs: list[str],
    *,
    input_has_morph_boundaries: bool,
    morph_boundary: str,
    num_threads: int,
    collect_word_limit: int | None = None,
) -> tuple[dict[str, Any], list[str]]:
    visible_docs, boundary_offsets_by_doc, docs_with_decode_strip = prepare_visible_docs(
        tokenizer,
        docs,
        input_has_morph_boundaries=input_has_morph_boundaries,
        morph_boundary=morph_boundary,
    )

    t0 = time.time()
    if isinstance(tokenizer, HuggingFaceTokenizerAdapter):
        encoded, token_offsets_by_doc = tokenizer.encode_with_offsets(visible_docs)
    else:
        encoded = tokenizer.encode(visible_docs, num_threads=num_threads)
        token_offsets_by_doc = None
    encode_seconds = time.time() - t0

    batch_token_lengths = [len(ids) for ids in encoded]
    batch_words = [WORD_RE.findall(doc) for doc in visible_docs]
    boundary_totals = {
        "eligible_docs": 0,
        "boundary_count": 0,
        "crossing_tokens": 0,
        "crossed_boundaries": 0,
        "docs_with_crossing": 0,
    }
    batch_boundary_stats = boundary_violation_stats(
        tokenizer,
        encoded,
        visible_docs,
        boundary_offsets_by_doc,
        token_offsets_by_doc,
    )
    for key in boundary_totals:
        boundary_totals[key] += int(batch_boundary_stats.get(key, 0) or 0)

    word_metric_words: list[str] = []
    if collect_word_limit is not None:
        for words in batch_words:
            if collect_word_limit == 0:
                word_metric_words.extend(words)
            else:
                remaining = collect_word_limit - len(word_metric_words)
                if remaining <= 0:
                    break
                word_metric_words.extend(words[:remaining])

    metrics = {
        "docs_count": len(visible_docs),
        "total_tokens": sum(batch_token_lengths),
        "total_bytes": sum(len(doc.encode("utf-8")) for doc in visible_docs),
        "total_chars": sum(len(doc) for doc in visible_docs),
        "total_words": sum(len(words) for words in batch_words),
        "unique_tokens": set(token_id for ids in encoded for token_id in ids),
        "roundtrip_failures": sum(
            1 for doc, ids in zip(visible_docs, encoded)
            if tokenizer.decode(ids) != doc
        ),
        "docs_with_decode_strip": docs_with_decode_strip,
        "encode_seconds": encode_seconds,
        "token_length_counts": Counter(batch_token_lengths),
        "boundary_totals": boundary_totals,
        "row_group_tasks": 1,
    }
    return metrics, word_metric_words


def build_parquet_row_group_tasks(
    *,
    split: str,
    data_dir: str,
    text_column: str,
    max_docs: int,
) -> list[tuple[str, int, int]]:
    parquet_paths = list_parquet_files(data_dir)
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run `python -m nanochat.dataset -n 8`?"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    tasks: list[tuple[str, int, int]] = []
    docs_seen = 0
    stop = False
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        if text_column not in pf.schema_arrow.names:
            raise KeyError(f"Column {text_column!r} not found in {filepath}; columns: {pf.schema_arrow.names}")
        for rg_idx in range(pf.num_row_groups):
            row_count = pf.metadata.row_group(rg_idx).num_rows
            doc_limit = 0
            if max_docs > 0:
                remaining = max_docs - docs_seen
                if remaining <= 0:
                    stop = True
                    break
                if remaining < row_count:
                    doc_limit = remaining
                    docs_seen += remaining
                    stop = True
                else:
                    docs_seen += row_count
            tasks.append((filepath, rg_idx, doc_limit))
            if stop:
                break
        if stop:
            break
    return tasks


def init_metrics_worker(
    tokenizer_spec: dict[str, Any],
    text_column: str,
    input_has_morph_boundaries: bool,
    morph_boundary: str,
    num_threads: int,
) -> None:
    global _WORKER_TOKENIZER
    global _WORKER_TEXT_COLUMN
    global _WORKER_INPUT_HAS_MORPH_BOUNDARIES
    global _WORKER_MORPH_BOUNDARY
    global _WORKER_NUM_THREADS
    _WORKER_TOKENIZER, _, _ = load_requested_tokenizer(**tokenizer_spec)
    _WORKER_TEXT_COLUMN = text_column
    _WORKER_INPUT_HAS_MORPH_BOUNDARIES = input_has_morph_boundaries
    _WORKER_MORPH_BOUNDARY = morph_boundary
    _WORKER_NUM_THREADS = num_threads


def process_row_group_task(task: tuple[str, int, int]) -> dict[str, Any]:
    filepath, row_group_idx, doc_limit = task
    pf = pq.ParquetFile(filepath)
    row_group = pf.read_row_group(row_group_idx, columns=[_WORKER_TEXT_COLUMN])
    docs = row_group.column(_WORKER_TEXT_COLUMN).to_pylist()
    if doc_limit:
        docs = docs[:doc_limit]
    metrics, _ = measure_docs_batch(
        _WORKER_TOKENIZER,
        docs,
        input_has_morph_boundaries=_WORKER_INPUT_HAS_MORPH_BOUNDARIES,
        morph_boundary=_WORKER_MORPH_BOUNDARY,
        num_threads=_WORKER_NUM_THREADS,
    )
    return metrics


def collect_word_metric_words(
    *,
    split: str,
    data_dir: str,
    text_column: str,
    input_has_morph_boundaries: bool,
    morph_boundary: str,
    decode_strip: str,
    max_docs: int,
    max_words: int,
) -> list[str]:
    if max_words <= 0:
        return []
    words: list[str] = []
    docs_seen = 0
    for batch in parquets_iter_batched(
        split=split,
        data_dir=data_dir,
        text_column=text_column,
    ):
        if max_docs > 0:
            remaining_docs = max_docs - docs_seen
            if remaining_docs <= 0:
                break
            batch = batch[:remaining_docs]
        for doc in batch:
            if input_has_morph_boundaries:
                doc, _ = strip_boundaries_with_offsets(
                    doc,
                    boundary=morph_boundary,
                )
            elif decode_strip:
                doc = strip_morpheme_boundaries(doc, decode_strip)
            words.extend(WORD_RE.findall(doc))
            if len(words) >= max_words:
                return words[:max_words]
        docs_seen += len(batch)
    return words


def run_serial_metrics(args: argparse.Namespace, tokenizer) -> tuple[dict[str, Any], list[str]]:
    aggregate = empty_metric_accumulator()
    word_metric_words: list[str] = []
    wall_t0 = time.time()

    for batch in parquets_iter_batched(
        split=args.split,
        data_dir=args.data_dir or DATA_DIR,
        text_column=args.text_column,
    ):
        if args.max_docs > 0:
            remaining_docs = args.max_docs - aggregate["docs_count"]
            if remaining_docs <= 0:
                break
            batch = batch[:remaining_docs]
        if not batch:
            continue

        collect_word_limit: int | None = None
        if args.max_word_metrics == 0:
            collect_word_limit = 0
        elif len(word_metric_words) < args.max_word_metrics:
            collect_word_limit = args.max_word_metrics - len(word_metric_words)

        batch_metrics, batch_words = measure_docs_batch(
            tokenizer,
            batch,
            input_has_morph_boundaries=args.input_has_morph_boundaries,
            morph_boundary=args.morph_boundary,
            num_threads=args.num_threads,
            collect_word_limit=collect_word_limit,
        )
        merge_metric_accumulator(aggregate, batch_metrics)
        word_metric_words.extend(batch_words)

        if args.max_docs > 0 and aggregate["docs_count"] >= args.max_docs:
            break

    aggregate["metric_wall_seconds"] = time.time() - wall_t0
    return aggregate, word_metric_words


def run_parallel_metrics(
    args: argparse.Namespace,
    tokenizer_spec: dict[str, Any],
    tokenizer,
) -> tuple[dict[str, Any], list[str]]:
    if args.max_word_metrics == 0:
        raise ValueError("--workers > 1 requires --max-word-metrics to be positive")

    resolved_data_dir = args.data_dir or DATA_DIR
    tasks = build_parquet_row_group_tasks(
        split=args.split,
        data_dir=resolved_data_dir,
        text_column=args.text_column,
        max_docs=args.max_docs,
    )
    if not tasks:
        raise RuntimeError("No parquet row groups found for tokenizer metrics")

    workers = min(max(1, args.workers), len(tasks))
    word_metric_words = collect_word_metric_words(
        split=args.split,
        data_dir=resolved_data_dir,
        text_column=args.text_column,
        input_has_morph_boundaries=args.input_has_morph_boundaries,
        morph_boundary=args.morph_boundary,
        decode_strip=getattr(tokenizer, "decode_strip", ""),
        max_docs=args.max_docs,
        max_words=args.max_word_metrics,
    )

    aggregate = empty_metric_accumulator()
    aggregate["effective_workers"] = workers
    wall_t0 = time.time()
    print(
        (
            f"Running tokenizer metrics over {len(tasks)} row groups "
            f"with {workers} worker processes and {args.num_threads} tokenizer "
            "thread(s) per worker."
        ),
        flush=True,
    )
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_metrics_worker,
        initargs=(
            tokenizer_spec,
            args.text_column,
            args.input_has_morph_boundaries,
            args.morph_boundary,
            args.num_threads,
        ),
    ) as executor:
        futures = [executor.submit(process_row_group_task, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            merge_metric_accumulator(aggregate, future.result())
            if (
                args.progress_every > 0
                and (done == len(futures) or done % args.progress_every == 0)
            ):
                elapsed = time.time() - wall_t0
                print(
                    (
                        f"progress row_groups={done}/{len(futures)} "
                        f"docs={aggregate['docs_count']} "
                        f"tokens={aggregate['total_tokens']} "
                        f"wall_seconds={elapsed:.1f} "
                        f"docs_per_wall_sec={safe_div(aggregate['docs_count'], elapsed):.1f}"
                    ),
                    flush=True,
                )

    aggregate["metric_wall_seconds"] = time.time() - wall_t0
    return aggregate, word_metric_words


def finalize_metrics(
    *,
    args: argparse.Namespace,
    tokenizer,
    tokenizer_dir: str,
    tokenizer_config: dict[str, Any],
    aggregate: dict[str, Any],
    word_metric_words: list[str],
    morph_metric_words: list[tuple[str, tuple[str, ...]]],
) -> dict[str, Any]:
    docs_count = aggregate["docs_count"]
    total_tokens = aggregate["total_tokens"]
    total_bytes = aggregate["total_bytes"]
    total_chars = aggregate["total_chars"]
    total_words = aggregate["total_words"]
    if docs_count == 0:
        raise RuntimeError("No documents found for tokenizer metrics")

    boundary_stats = dict(aggregate["boundary_totals"])
    boundary_stats["crossed_boundary_rate"] = safe_div(
        boundary_stats["crossed_boundaries"],
        boundary_stats["boundary_count"],
    )
    boundary_stats["docs_with_crossing_rate"] = safe_div(
        boundary_stats["docs_with_crossing"],
        boundary_stats["eligible_docs"],
    )
    boundary_stats["crossing_tokens_per_1k_tokens"] = (
        1000.0 * safe_div(boundary_stats["crossing_tokens"], total_tokens)
    )

    word_metric_threads = args.num_threads
    if args.workers > 1:
        word_metric_threads = max(args.num_threads, min(args.workers, 32))

    morphology_metrics: dict[str, Any] = {
        "enabled": bool(args.input_has_morph_boundaries and not args.disable_morphology_metrics),
        "sample_word_occurrences": len(morph_metric_words),
    }
    if morphology_metrics["enabled"]:
        morphology_metrics["alignment"] = morphological_alignment_stats(
            tokenizer,
            morph_metric_words,
            num_threads=word_metric_threads,
        )
        morphology_metrics["consistency"] = morphological_consistency_stats(
            tokenizer,
            morph_metric_words,
            num_threads=word_metric_threads,
            max_words=args.morph_consistency_max_words,
            n_clusters=args.morph_consistency_clusters,
            pairs_per_cluster=args.morph_consistency_pairs_per_cluster,
            resamples=args.morph_consistency_resamples,
            seed=args.morph_consistency_seed,
        )

    return {
        "tokenizer_name": tokenizer_config.get("name") or get_tokenizer_name(),
        "tokenizer_dir": tokenizer_dir,
        "tokenizer_config": tokenizer_config,
        "split": args.split,
        "data_dir": args.data_dir or DATA_DIR,
        "text_column": args.text_column,
        "input_has_morph_boundaries": args.input_has_morph_boundaries,
        "morph_boundary_codepoint": (
            f"U+{ord(args.morph_boundary):04X}"
            if len(args.morph_boundary) == 1
            else ""
        ),
        "docs": docs_count,
        "bytes": total_bytes,
        "chars": total_chars,
        "words": total_words,
        "tokens": total_tokens,
        "unique_tokens_in_sample": len(aggregate["unique_tokens"]),
        "unique_token_rate_in_sample": safe_div(len(aggregate["unique_tokens"]), tokenizer.get_vocab_size()),
        "decode_strip": getattr(tokenizer, "decode_strip", ""),
        "docs_with_decode_strip": aggregate["docs_with_decode_strip"],
        "bytes_per_token": safe_div(total_bytes, total_tokens),
        "chars_per_token": safe_div(total_chars, total_tokens),
        "tokens_per_byte": safe_div(total_tokens, total_bytes),
        "tokens_per_char": safe_div(total_tokens, total_chars),
        "tokens_per_word": safe_div(total_tokens, total_words),
        "token_fertility": safe_div(total_tokens, total_words),
        "tokens_per_doc_distribution": distribution_stats_from_counts(aggregate["token_length_counts"]),
        "word_fertility_isolated": word_fertility_stats_from_words(
            tokenizer,
            word_metric_words,
            num_threads=word_metric_threads,
        ),
        "morphology": morphology_metrics,
        "vocabulary": vocabulary_stats(tokenizer),
        "morph_boundary": boundary_stats,
        "roundtrip_failures": aggregate["roundtrip_failures"],
        "roundtrip_failure_rate": safe_div(aggregate["roundtrip_failures"], docs_count),
        "encode_seconds": aggregate["encode_seconds"],
        "encode_docs_per_sec": safe_div(docs_count, aggregate["encode_seconds"]),
        "encode_tokens_per_sec": safe_div(total_tokens, aggregate["encode_seconds"]),
        "metric_wall_seconds": aggregate["metric_wall_seconds"],
        "metric_docs_per_wall_sec": safe_div(docs_count, aggregate["metric_wall_seconds"]),
        "metric_tokens_per_wall_sec": safe_div(total_tokens, aggregate["metric_wall_seconds"]),
        "metric_workers": aggregate.get("effective_workers", args.workers),
        "metric_tokenizer_threads_per_worker": args.num_threads,
        "metric_row_group_tasks": aggregate["row_group_tasks"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenizer ablation metrics")
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default="",
        help="Tokenizer directory. Default = active nanochat tokenizer.",
    )
    parser.add_argument(
        "--hf-tokenizer",
        type=str,
        default="",
        help=(
            "Hugging Face tokenizer repo id or local tokenizer.json/vocab.txt path. "
            "Uses tokenizer files only, never model weights."
        ),
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="",
        help="Optional display name for --hf-tokenizer metrics.",
    )
    parser.add_argument(
        "--tokenizer-implementation",
        type=str,
        default="",
        help="Optional implementation label for --hf-tokenizer metrics.",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="Optional parquet directory. Default = active nanochat dataset dir.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default=os.environ.get("NANOCHAT_TEXT_COLUMN", "text"),
        help="Parquet text column to evaluate.",
    )
    parser.add_argument(
        "--input-has-morph-boundaries",
        action="store_true",
        help=(
            "Strip morpheme boundary markers before encoding and compute "
            "token-crosses-morpheme-boundary metrics."
        ),
    )
    parser.add_argument("--morph-boundary", type=str, default=MORPHEME_BOUNDARY)
    parser.add_argument(
        "--max-docs",
        type=int,
        default=10000,
        help="Maximum documents to evaluate. Use 0 for the full dataset.",
    )
    parser.add_argument(
        "--max-word-metrics",
        type=int,
        default=200000,
        help=(
            "Maximum word tokens for isolated word fertility metrics. "
            "Use 0 for all words in sampled docs."
        ),
    )
    parser.add_argument(
        "--max-morph-metrics",
        type=int,
        default=200000,
        help=(
            "Maximum segmented word occurrences for MorphBPE paper metrics. "
            "Use 0 for all words in sampled docs, or -1 to skip collection."
        ),
    )
    parser.add_argument(
        "--disable-morphology-metrics",
        action="store_true",
        help="Skip MorphBPE paper metrics even when morpheme boundaries are present.",
    )
    parser.add_argument(
        "--morph-consistency-max-words",
        type=int,
        default=50000,
        help="Maximum unique surface words for Morph-Consistency sampling.",
    )
    parser.add_argument(
        "--morph-consistency-clusters",
        type=int,
        default=100,
        help="Morph-Consistency clusters; MorphBPE paper uses k=100.",
    )
    parser.add_argument(
        "--morph-consistency-pairs-per-cluster",
        type=int,
        default=50,
        help="Word pairs sampled inside each cluster; MorphBPE paper uses C=50.",
    )
    parser.add_argument(
        "--morph-consistency-resamples",
        type=int,
        default=10,
        help="Bootstrap resamples for Morph-Consistency; MorphBPE paper uses N=10.",
    )
    parser.add_argument(
        "--morph-consistency-seed",
        type=int,
        default=13,
        help="Deterministic seed for Morph-Consistency clustering/pair sampling.",
    )
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of process workers over parquet row groups. "
            "Use 1 for the historical single-process path."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print parallel progress every N completed row groups. Use 0 to disable.",
    )
    parser.add_argument("--hf-batch-size", type=int, default=1000)
    parser.add_argument("--output", type=str, default="", help="Optional JSON output path")
    parser.add_argument("--no-report", action="store_true", help="Do not write into nanochat.report.")
    args = parser.parse_args()

    tokenizer_spec = {
        "tokenizer_dir_arg": args.tokenizer_dir,
        "hf_tokenizer": args.hf_tokenizer,
        "tokenizer_name": args.tokenizer_name,
        "tokenizer_implementation": args.tokenizer_implementation,
        "hf_batch_size": args.hf_batch_size,
    }
    tokenizer, tokenizer_dir, tokenizer_config = load_requested_tokenizer(**tokenizer_spec)

    if args.workers > 1:
        aggregate, word_metric_words = run_parallel_metrics(args, tokenizer_spec, tokenizer)
    else:
        aggregate, word_metric_words = run_serial_metrics(args, tokenizer)

    morph_metric_words: list[tuple[str, tuple[str, ...]]] = []
    if args.input_has_morph_boundaries and not args.disable_morphology_metrics:
        morph_metric_words = collect_morph_metric_words(
            split=args.split,
            data_dir=args.data_dir or DATA_DIR,
            text_column=args.text_column,
            morph_boundary=args.morph_boundary,
            max_docs=args.max_docs,
            max_words=args.max_morph_metrics,
        )

    metrics = finalize_metrics(
        args=args,
        tokenizer=tokenizer,
        tokenizer_dir=tokenizer_dir,
        tokenizer_config=tokenizer_config,
        aggregate=aggregate,
        word_metric_words=word_metric_words,
        morph_metric_words=morph_metric_words,
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    if not args.no_report:
        from nanochat.report import get_report
        get_report().log(section="Tokenizer metrics", data=[metrics])


if __name__ == "__main__":
    main()
