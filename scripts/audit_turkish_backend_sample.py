"""Seal a review-only quality audit for a bounded Turkish backend sample.

This command consumes the immutable source plan, backend calibration, object
and bucket receipts, and the bounded cluster output. It verifies the complete
receipt/file chain before reporting content statistics. Successful execution
means only that the audit is internally consistent; it never approves a
mixture or authorizes a production run.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import re
import shutil
import tempfile
import urllib.parse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

import nanochat.turkish_backend as backend
from nanochat.experiment_manifest import (
    canonical_json,
    file_sha256,
    load_json_strict,
    seal_manifest,
    verify_manifest_hash,
    write_json_atomic,
)
from nanochat.turkish_corpus import (
    TurkishCorpusError,
    _qa_document_metrics,
    dominant_register,
    load_corpus_policy,
    select_mixture_bucket,
)


AUDIT_KIND = "turkish_bounded_backend_sample_quality_audit"
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


class SampleAuditError(ValueError):
    pass


class _NumericSketch:
    """Order-independent sample containing the smallest identity hashes."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.seen = 0
        self._heap: list[tuple[int, float]] = []

    def add(self, identity: str, raw_value: Any) -> None:
        if isinstance(raw_value, bool):
            value = float(int(raw_value))
        else:
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                return
        if not math.isfinite(value):
            return
        self.seen += 1
        rank = int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest()[:8], "big")
        item = (-rank, value)
        if len(self._heap) < self.capacity:
            heapq.heappush(self._heap, item)
        elif item > self._heap[0]:
            heapq.heapreplace(self._heap, item)

    def summary(self, quantiles: Sequence[float]) -> dict[str, Any]:
        values = sorted(value for _rank, value in self._heap)
        result: dict[str, Any] = {
            "observations": self.seen,
            "deterministic_sample_size": len(values),
        }
        if not values:
            return result
        result.update(
            {
                "minimum": values[0],
                "maximum": values[-1],
                "mean_of_sample": sum(values) / len(values),
            }
        )
        for quantile in quantiles:
            index = round(float(quantile) * (len(values) - 1))
            result[f"q{float(quantile):.4f}"] = values[index]
        return result


class _ExampleSampler:
    def __init__(self, capacity: int, max_characters: int) -> None:
        self.capacity = capacity
        self.max_characters = max_characters
        self._heaps: dict[
            tuple[str, str, str, str, str],
            list[tuple[int, str, str]],
        ] = defaultdict(list)

    def add(
        self,
        stratum: tuple[str, str, str, str],
        decision: str,
        row: Mapping[str, Any],
        *,
        rejection_reason: str | None,
        quality_flags: Sequence[str],
        metrics: Mapping[str, Any],
    ) -> None:
        document_id = str(row["document_id"])
        key = (*stratum, decision)
        identity = hashlib.sha256(
            "\0".join((*key, document_id)).encode("utf-8")
        ).hexdigest()
        text = str(row["text"])
        payload = {
            "sample_sha256": identity,
            "source_id": stratum[0],
            "mixture_id": stratum[1],
            "wds_bin": stratum[2],
            "register": stratum[3],
            "decision": decision,
            "rejection_reason": rejection_reason,
            "quality_filter_flags": list(quality_flags),
            "document_id": document_id,
            "dedup_cluster_id": str(row["dedup_cluster_id"]),
            "url": str(row.get("url") or ""),
            "metrics": dict(sorted(metrics.items())),
            "text": text[: self.max_characters],
            "text_truncated": len(text) > self.max_characters,
        }
        # Include the complete payload in the rank so a malformed upstream that
        # repeats a document ID cannot make heap ordering depend on input order.
        payload_json = canonical_json(payload).removesuffix("\n")
        rank_sha256 = hashlib.sha256(
            f"{identity}\0{payload_json}".encode("utf-8")
        ).hexdigest()
        item = (-int(rank_sha256[:16], 16), rank_sha256, payload_json)
        heap = self._heaps[key]
        if len(heap) < self.capacity:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)

    def rows(self, decision: str) -> list[dict[str, Any]]:
        rows = [
            json.loads(payload_json)
            for key, heap in self._heaps.items()
            if key[-1] == decision
            for _rank, _identity, payload_json in heap
        ]
        return sorted(
            rows,
            key=lambda row: (
                row["source_id"],
                row["mixture_id"],
                row["wds_bin"],
                row["register"],
                row["sample_sha256"],
            ),
        )


def _safe_run_path(run_root: Path, relative: Any, label: str) -> Path:
    raw = str(relative or "")
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme or not raw or Path(raw).is_absolute():
        raise SampleAuditError(f"{label} must be a relative run-directory path")
    root = run_root.resolve()
    unresolved = root / raw
    path = unresolved.resolve()
    current = unresolved
    symlinked = False
    while current != root:
        if current.is_symlink():
            symlinked = True
            break
        if root not in current.parents:
            break
        current = current.parent
    if root not in path.parents or symlinked or not path.is_file():
        raise SampleAuditError(f"{label} is unsafe or missing: {raw}")
    return path


def _strict_flags(raw: Any) -> list[str]:
    if not isinstance(raw, str):
        raise SampleAuditError("quality_filter_flags must be a JSON string")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SampleAuditError("quality_filter_flags contains malformed JSON") from exc
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise SampleAuditError("quality_filter_flags must encode non-empty strings")
    return value


def _merge_stage_counts(target: dict[str, Counter[str]], raw: Any) -> None:
    if not isinstance(raw, Mapping):
        raise SampleAuditError("object stage_counts must be a mapping")
    for stage, values in raw.items():
        if not isinstance(values, Mapping):
            raise SampleAuditError("object stage count must be a mapping")
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SampleAuditError("object stage counts must be non-negative integers")
            target[str(stage)][str(key)] += value


def _source_input_summary(
    objects: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    stages: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    records: list[dict[str, Any]] = []
    for receipt in objects:
        source_id = str(receipt["source_id"])
        rank = int(receipt["rank"])
        if rank < 0 or rank >= len(plan["objects"]):
            raise SampleAuditError("object receipt rank is outside source plan")
        counts = receipt.get("counts")
        raw_object = receipt.get("raw_object")
        candidate = receipt.get("candidate_file")
        if not all(isinstance(value, Mapping) for value in (counts, raw_object, candidate)):
            raise SampleAuditError("object receipt lacks count/raw/candidate mappings")
        numeric = {
            "sampled_objects": 1,
            "raw_input_bytes": int(raw_object["size_bytes"]),
            "documents_seen": int(counts["documents_seen"]),
            "utf8_bytes_seen": int(counts["utf8_bytes_seen"]),
            "candidate_documents": int(counts["candidates"]),
            "candidate_characters": int(counts["candidate_characters"]),
            "candidate_file_bytes": int(candidate["size_bytes"]),
        }
        if any(value < 0 for value in numeric.values()):
            raise SampleAuditError("object receipt contains a negative count")
        totals[source_id].update(numeric)
        _merge_stage_counts(stages[source_id], counts.get("stage_counts"))
        plan_object = plan["objects"][rank]
        records.append(
            {
                "rank": rank,
                "source_id": source_id,
                "wds_bin": plan_object.get("wds_bin"),
                "receipt_sha256": receipt["canonical_sha256"],
                "source_uri_sha256": hashlib.sha256(
                    str(receipt["source_uri"]).encode("utf-8")
                ).hexdigest(),
                "raw_input": {
                    "size_bytes": raw_object["size_bytes"],
                    "sha256": raw_object["sha256"],
                },
                "candidate_file": {
                    key: candidate[key]
                    for key in ("path", "size_bytes", "sha256", "rows")
                },
                "counts": {
                    key: counts[key]
                    for key in (
                        "documents_seen",
                        "candidates",
                        "characters_seen",
                        "utf8_bytes_seen",
                        "candidate_characters",
                    )
                },
                "stage_counts": {
                    name: dict(sorted(values.items()))
                    for name, values in sorted(
                        (
                            (str(name), Counter(raw_values))
                            for name, raw_values in counts["stage_counts"].items()
                        ),
                        key=lambda item: item[0],
                    )
                },
            }
        )
    summaries = [
        {
            "source_id": source_id,
            **dict(sorted(totals[source_id].items())),
            "stage_counts": {
                stage: dict(sorted(values.items()))
                for stage, values in sorted(stages[source_id].items())
            },
        }
        for source_id in sorted(totals)
    ]
    return summaries, sorted(records, key=lambda item: item["rank"])


def _write_examples(
    root: Path,
    decision: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    jsonl = root / f"{decision}_examples.jsonl"
    plaintext = root / f"{decision}_examples.txt"
    with jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            # canonical_json currently emits one LF. Normalize explicitly so
            # this remains valid JSONL if that helper's rendering ever changes.
            handle.write(canonical_json(row).rstrip("\n") + "\n")
    with plaintext.open("w", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(rows, 1):
            handle.write(
                f"[{index}] {row['decision']} / {row['source_id']} / "
                f"{row['mixture_id']} / WDS={row['wds_bin']} / "
                f"register={row['register']}\n"
            )
            handle.write(f"sample_sha256: {row['sample_sha256']}\n")
            handle.write(f"reason: {row['rejection_reason']}\n")
            handle.write(
                "quality_filter_flags: "
                + json.dumps(row["quality_filter_flags"], ensure_ascii=False)
                + "\n"
            )
            handle.write(f"url: {row['url']}\n")
            handle.write(str(row["text"]) + "\n\n")
    return {
        "rows": len(rows),
        "jsonl": {
            "path": jsonl.name,
            "size_bytes": jsonl.stat().st_size,
            "sha256": file_sha256(jsonl),
        },
        "plaintext": {
            "path": plaintext.name,
            "size_bytes": plaintext.stat().st_size,
            "sha256": file_sha256(plaintext),
        },
    }


def build_sample_quality_audit(
    policy_path: str | Path,
    source_plan_path: str | Path,
    calibration_path: str | Path,
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    examples_per_stratum: int | None = None,
    max_example_characters: int | None = None,
    quantile_sample_size: int | None = None,
) -> dict[str, Any]:
    policy_file = Path(policy_path)
    plan_file = Path(source_plan_path)
    calibration_file = Path(calibration_path)
    run_root = Path(run_dir).resolve()
    destination = Path(output_dir)
    if destination.exists():
        if any(destination.iterdir()):
            raise FileExistsError(f"refusing non-empty output directory: {destination}")
        destination.rmdir()
    destination.parent.mkdir(parents=True, exist_ok=True)

    policy = load_corpus_policy(policy_file)
    qa_policy = policy["quality_assurance"]
    minimum_examples = int(qa_policy["examples_per_stratum_and_decision"])
    minimum_characters = int(qa_policy["max_example_characters"])
    minimum_quantile_sample = int(qa_policy["quantile_sample_size"])
    examples_per_stratum = (
        minimum_examples if examples_per_stratum is None else examples_per_stratum
    )
    max_example_characters = (
        minimum_characters
        if max_example_characters is None
        else max_example_characters
    )
    quantile_sample_size = (
        minimum_quantile_sample
        if quantile_sample_size is None
        else quantile_sample_size
    )
    if examples_per_stratum < minimum_examples:
        raise SampleAuditError("examples_per_stratum weakens the frozen QA policy")
    if max_example_characters < minimum_characters:
        raise SampleAuditError("max_example_characters weakens the frozen QA policy")
    if quantile_sample_size < minimum_quantile_sample:
        raise SampleAuditError("quantile_sample_size weakens the frozen QA policy")
    policy_sha256 = hashlib.sha256(canonical_json(policy).encode("utf-8")).hexdigest()
    plan = load_json_strict(plan_file)
    calibration = load_json_strict(calibration_file)
    backend.validate_source_plan(plan, policy)
    backend.validate_backend_calibration(calibration, policy)
    plan_sha256 = verify_manifest_hash(plan)
    calibration_sha256 = verify_manifest_hash(calibration)

    cluster_path = run_root / "cluster_receipt.json"
    if cluster_path.is_symlink() or not cluster_path.is_file():
        raise SampleAuditError("bounded sample cluster receipt is missing or unsafe")
    cluster = load_json_strict(cluster_path)
    cluster_sha256 = verify_manifest_hash(cluster)
    if (
        cluster.get("schema_version") != "1.0"
        or cluster.get("kind") != backend.CLUSTER_RECEIPT_KIND
        or cluster.get("sample_mode") is not True
        or cluster.get("source_plan_sha256") != plan_sha256
        or cluster.get("calibration_sha256") != calibration_sha256
    ):
        raise SampleAuditError("cluster receipt is not the requested bounded sample")

    objects = backend._load_object_receipts(
        run_root, plan, calibration, sample_mode=True
    )
    buckets = backend._load_bucket_receipts(
        run_root, plan, calibration, sample_mode=True
    )
    object_hashes = [item["canonical_sha256"] for item in objects]
    bucket_hashes = [item["canonical_sha256"] for item in buckets]
    if cluster.get("object_receipt_sha256") != object_hashes:
        raise SampleAuditError("cluster/object receipt hash binding drift")
    if cluster.get("bucket_receipt_sha256") != bucket_hashes:
        raise SampleAuditError("cluster/bucket receipt hash binding drift")

    source_summaries, object_records = _source_input_summary(objects, plan)
    quantiles = tuple(float(value) for value in policy["quality_assurance"]["quantiles"])
    counts: dict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
    flags_by_stratum: dict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
    reasons_by_stratum: dict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
    metrics: dict[
        tuple[str, str, str, str],
        dict[str, dict[str, _NumericSketch]],
    ] = defaultdict(lambda: defaultdict(dict))
    source_cluster_counts: dict[str, Counter[str]] = defaultdict(Counter)
    examples = _ExampleSampler(examples_per_stratum, max_example_characters)
    output_records: list[dict[str, Any]] = []
    seen_output_paths: set[str] = set()
    total_rows = 0

    required_columns = set(backend.BACKEND_COLUMNS)
    output_files = cluster.get("output_files")
    if not isinstance(output_files, list) or not output_files:
        raise SampleAuditError("bounded cluster receipt has no output files")
    for index, raw_file in enumerate(output_files):
        if not isinstance(raw_file, Mapping):
            raise SampleAuditError("cluster output record must be a mapping")
        relative_path = str(raw_file.get("path") or "")
        if relative_path in seen_output_paths:
            raise SampleAuditError("cluster output file is listed more than once")
        seen_output_paths.add(relative_path)
        path = _safe_run_path(run_root, raw_file.get("path"), f"cluster output {index}")
        if (
            path.stat().st_size != raw_file.get("size_bytes")
            or file_sha256(path) != raw_file.get("sha256")
        ):
            raise SampleAuditError(f"cluster output drift: {path.name}")
        parquet = pq.ParquetFile(path)
        if not required_columns <= set(parquet.schema_arrow.names):
            missing = sorted(required_columns - set(parquet.schema_arrow.names))
            raise SampleAuditError(f"cluster output lacks backend columns: {missing}")
        if parquet.metadata.num_rows != raw_file.get("rows"):
            raise SampleAuditError("cluster output row-count drift")
        output_records.append(
            {
                "path": str(raw_file["path"]),
                "size_bytes": raw_file["size_bytes"],
                "sha256": raw_file["sha256"],
                "rows": raw_file["rows"],
            }
        )

        for batch in parquet.iter_batches(batch_size=2_048):
            for row in batch.to_pylist():
                total_rows += 1
                if not isinstance(row.get("dedup_keep"), bool):
                    raise SampleAuditError("dedup_keep must be boolean")
                if not isinstance(row.get("text"), str) or not isinstance(
                    row.get("document_id"), str
                ):
                    raise SampleAuditError("backend row text/document_id schema drift")
                quality_flags = _strict_flags(row.get("quality_filter_flags"))
                source_id = str(row.get("source_id") or "")
                if not source_id:
                    raise SampleAuditError("backend row lacks source_id")
                try:
                    routed = select_mixture_bucket(source_id, row, policy)
                    register = dominant_register(row)
                except (TurkishCorpusError, KeyError, TypeError, ValueError) as exc:
                    raise SampleAuditError(
                        f"backend row routing schema drift for source {source_id}"
                    ) from exc
                mixture_id = routed[0] if routed is not None else "unrouted"
                wds_bin = (
                    str(row["wds_bin"])
                    if row.get("wds_bin") is not None
                    else "not_applicable"
                )
                stratum = (source_id, mixture_id, wds_bin, register)
                text = row["text"]
                utf8_bytes = len(text.encode("utf-8"))
                row_counts = counts[stratum]
                row_counts["total_documents"] += 1
                row_counts["total_utf8_bytes"] += utf8_bytes
                source_cluster_counts[source_id]["total_documents"] += 1
                source_cluster_counts[source_id]["total_utf8_bytes"] += utf8_bytes
                if routed is None:
                    row_counts["selector_unrouted_documents"] += 1
                    source_cluster_counts[source_id]["selector_unrouted_documents"] += 1
                else:
                    row_counts["selector_routed_documents"] += 1
                    source_cluster_counts[source_id]["selector_routed_documents"] += 1
                if row["dedup_keep"]:
                    row_counts["dedup_survived_documents"] += 1
                    source_cluster_counts[source_id]["dedup_survived_documents"] += 1
                else:
                    row_counts["dedup_removed_documents"] += 1
                    source_cluster_counts[source_id]["dedup_removed_documents"] += 1
                if row["dedup_keep"] and quality_flags:
                    row_counts["quality_rejected_documents"] += 1
                    source_cluster_counts[source_id]["quality_rejected_documents"] += 1
                elif row["dedup_keep"]:
                    row_counts["quality_passed_documents"] += 1
                    source_cluster_counts[source_id]["quality_passed_documents"] += 1
                for flag in quality_flags:
                    flags_by_stratum[stratum][flag] += 1

                accepted = bool(row["dedup_keep"] and not quality_flags and routed is not None)
                if accepted:
                    decision = "accepted"
                    rejection_reason = None
                    row_counts["accepted_documents"] += 1
                    row_counts["accepted_utf8_bytes"] += utf8_bytes
                    source_cluster_counts[source_id]["accepted_documents"] += 1
                    source_cluster_counts[source_id]["accepted_utf8_bytes"] += utf8_bytes
                else:
                    decision = "rejected"
                    if row["dedup_keep"] is not True:
                        rejection_reason = "dedup_removed"
                    elif routed is None:
                        rejection_reason = "selector_unrouted"
                    else:
                        rejection_reason = "quality_filter"
                    row_counts["rejected_documents"] += 1
                    row_counts["rejected_utf8_bytes"] += utf8_bytes
                    source_cluster_counts[source_id]["rejected_documents"] += 1
                    source_cluster_counts[source_id]["rejected_utf8_bytes"] += utf8_bytes
                    reasons_by_stratum[stratum][rejection_reason] += 1

                _normalized_text, qa_metrics = _qa_document_metrics(row)
                qa_metrics = dict(qa_metrics)
                qa_metrics.update(
                    {
                        "source_lid_probability": row.get("source_lid_probability"),
                        "text_characters": len(text),
                        "text_utf8_bytes": utf8_bytes,
                        "text_words": len(_WORD_RE.findall(text)),
                    }
                )
                numeric_metrics = {
                    key: value
                    for key, value in qa_metrics.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
                for population in ("all", decision):
                    for metric_name, value in numeric_metrics.items():
                        sketch = metrics[stratum][population].setdefault(
                            metric_name, _NumericSketch(quantile_sample_size)
                        )
                        sketch.add(
                            f"{row['document_id']}\0{population}\0{metric_name}", value
                        )
                examples.add(
                    stratum,
                    decision,
                    row,
                    rejection_reason=rejection_reason,
                    quality_flags=quality_flags,
                    metrics=qa_metrics,
                )

    if total_rows != sum(int(item["rows"]) for item in output_files):
        raise SampleAuditError("cluster output total row-count drift")
    if total_rows <= 0:
        raise SampleAuditError("bounded cluster output is empty")
    cluster_counts = cluster.get("counts")
    if isinstance(cluster_counts, Mapping) and cluster_counts.get("output_rows") != total_rows:
        raise SampleAuditError("cluster receipt output_rows differs from audited rows")
    candidate_documents = sum(
        int(item["candidate_documents"]) for item in source_summaries
    )
    if candidate_documents != total_rows:
        raise SampleAuditError("object candidate rows differ from cluster output rows")
    candidates_by_source = {
        str(item["source_id"]): int(item["candidate_documents"])
        for item in source_summaries
    }
    for source_id in sorted(set(candidates_by_source) | set(source_cluster_counts)):
        if (
            candidates_by_source.get(source_id, 0)
            != source_cluster_counts[source_id]["total_documents"]
        ):
            raise SampleAuditError(
                f"object/cluster candidate count drift for source {source_id}"
            )

    computed_cluster_counts = {
        "output_rows": total_rows,
        "dedup_kept": sum(
            values["dedup_survived_documents"] for values in counts.values()
        ),
        "dedup_removed": sum(
            values["dedup_removed_documents"] for values in counts.values()
        ),
        "quality_kept": sum(
            values["quality_passed_documents"] for values in counts.values()
        ),
        "quality_removed": sum(
            values["quality_rejected_documents"] for values in counts.values()
        ),
    }
    if isinstance(cluster_counts, Mapping):
        for key, expected in computed_cluster_counts.items():
            if key in cluster_counts and cluster_counts[key] != expected:
                raise SampleAuditError(f"cluster receipt {key} differs from audited rows")

    strata = []
    for stratum in sorted(counts):
        populations = {
            population: {
                metric_name: sketch.summary(quantiles)
                for metric_name, sketch in sorted(metric_sketches.items())
            }
            for population, metric_sketches in sorted(metrics[stratum].items())
        }
        strata.append(
            {
                "source_id": stratum[0],
                "mixture_id": stratum[1],
                "wds_bin": stratum[2],
                "register": stratum[3],
                "counts": dict(sorted(counts[stratum].items())),
                "dedup_survival_rate": counts[stratum]["dedup_survived_documents"]
                / max(1, counts[stratum]["total_documents"]),
                "accepted_document_rate": counts[stratum]["accepted_documents"]
                / max(1, counts[stratum]["total_documents"]),
                "quality_filter_flag_rejections": dict(
                    sorted(flags_by_stratum[stratum].items())
                ),
                "rejection_reasons": dict(sorted(reasons_by_stratum[stratum].items())),
                "numeric_distributions": populations,
            }
        )

    accepted_examples = examples.rows("accepted")
    rejected_examples = examples.rows("rejected")
    build = Path(tempfile.mkdtemp(prefix=f".{destination.name}.build-", dir=destination.parent))
    try:
        example_records = {
            "accepted": _write_examples(build, "accepted", accepted_examples),
            "rejected": _write_examples(build, "rejected", rejected_examples),
        }
        expected_mixtures = sorted(str(bucket["id"]) for bucket in policy["mixture"])
        observed_accepted = {
            item["mixture_id"]
            for item in strata
            if item["mixture_id"] != "unrouted"
            and item["counts"].get("accepted_documents", 0) > 0
        }
        observed_rejected = {
            item["mixture_id"]
            for item in strata
            if item["mixture_id"] != "unrouted"
            and item["counts"].get("rejected_documents", 0) > 0
        }
        report = seal_manifest(
            {
                "schema_version": "1.0",
                "kind": AUDIT_KIND,
                "integrity_checks_passed": True,
                "policy_sha256": policy_sha256,
                "source_plan_sha256": plan_sha256,
                "calibration_sha256": calibration_sha256,
                "cluster_receipt_sha256": cluster_sha256,
                "input_artifacts": {
                    "policy": {
                        "path": policy_file.name,
                        "file_sha256": file_sha256(policy_file),
                    },
                    "source_plan": {
                        "path": plan_file.name,
                        "file_sha256": file_sha256(plan_file),
                    },
                    "calibration": {
                        "path": calibration_file.name,
                        "file_sha256": file_sha256(calibration_file),
                    },
                    "cluster_receipt": {
                        "path": cluster_path.name,
                        "file_sha256": file_sha256(cluster_path),
                    },
                    "object_receipt_sha256": object_hashes,
                    "bucket_receipt_sha256": bucket_hashes,
                    "cluster_output_files": output_records,
                },
                "sample_contract": {
                    "sample_mode": True,
                    "expected_object_ranks": backend.select_resource_sample_ranks(plan),
                    "quantiles": list(quantiles),
                    "quantile_sampling": "smallest_sha256_per_document_population_metric",
                    "quantile_sample_size": quantile_sample_size,
                },
                "source_input_and_candidates": source_summaries,
                "sampled_objects": object_records,
                "cluster_source_counts": [
                    {
                        "source_id": source_id,
                        **dict(sorted(source_cluster_counts[source_id].items())),
                    }
                    for source_id in sorted(source_cluster_counts)
                ],
                "cluster_totals": {
                    "documents": total_rows,
                    "accepted_documents": sum(
                        item["counts"].get("accepted_documents", 0) for item in strata
                    ),
                    "rejected_documents": sum(
                        item["counts"].get("rejected_documents", 0) for item in strata
                    ),
                    "accepted_utf8_bytes": sum(
                        item["counts"].get("accepted_utf8_bytes", 0) for item in strata
                    ),
                    "rejected_utf8_bytes": sum(
                        item["counts"].get("rejected_utf8_bytes", 0) for item in strata
                    ),
                },
                "cluster_receipt_processing_summary": {
                    "receipt_counts": dict(sorted(cluster_counts.items()))
                    if isinstance(cluster_counts, Mapping)
                    else None,
                    "audited_counts": computed_cluster_counts,
                    "filter_stage_counts": cluster.get("filter_stage_counts"),
                    "formatting_and_safety_incidence": cluster.get(
                        "formatting_and_safety_incidence"
                    ),
                },
                "strata": strata,
                "coverage": {
                    "expected_mixtures": expected_mixtures,
                    "mixtures_with_accepted_rows": sorted(observed_accepted),
                    "mixtures_with_rejected_rows": sorted(observed_rejected),
                    "mixtures_without_accepted_rows": sorted(
                        set(expected_mixtures) - observed_accepted
                    ),
                    "mixtures_without_rejected_rows": sorted(
                        set(expected_mixtures) - observed_rejected
                    ),
                },
                "example_sampling": {
                    "method": "smallest_sha256_per_source_mixture_wds_register_decision",
                    "examples_per_stratum_and_decision": examples_per_stratum,
                    "max_example_characters": max_example_characters,
                    "files": example_records,
                },
                "manual_review_required": True,
                "automatic_mixture_approval": False,
                "review_status": "pending",
                "canonical_sha256": None,
            }
        )
        write_json_atomic(build / "sample_quality_audit_report.json", report)
        os.replace(build, destination)
        return report
    except BaseException:
        shutil.rmtree(build, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples-per-stratum", type=int)
    parser.add_argument("--max-example-characters", type=int)
    parser.add_argument("--quantile-sample-size", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_sample_quality_audit(
            args.policy,
            args.source_plan,
            args.calibration,
            args.run_dir,
            args.output_dir,
            examples_per_stratum=args.examples_per_stratum,
            max_example_characters=args.max_example_characters,
            quantile_sample_size=args.quantile_sample_size,
        )
    except (OSError, ValueError, TurkishCorpusError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
