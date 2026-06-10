"""Generate a UHeM tokenizer-prep operations report.

The report combines Slurm accounting with per-shard segmentation sidecars. It
is intended for long tokenizer ablation runs where segmentation, tokenizer
training, model training, and publishing are separate dependent Slurm jobs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report UHeM tokenizer-prep job status")
    parser.add_argument("--host", default="uhem-altay")
    parser.add_argument("--segmentation-dir", required=True)
    parser.add_argument("--array-job", required=True)
    parser.add_argument("--finalize-job", required=True)
    parser.add_argument("--gpu-job", required=True)
    parser.add_argument("--hf-job", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--ssh-option", action="append", default=["-o", "BatchMode=yes"])
    return parser.parse_args()


def run_ssh(args: argparse.Namespace, remote_cmd: str) -> str:
    cmd = ["ssh", *args.ssh_option, args.host, remote_cmd]
    return subprocess.check_output(cmd, text=True)


def parse_duration_seconds(value: str) -> float:
    value = (value or "").strip()
    if not value or value == "Unknown":
        return 0.0
    days = 0
    if "-" in value:
        day_part, value = value.split("-", 1)
        days = int(day_part)
    parts = value.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    else:
        return 0.0
    return days * 86400 + int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def hours(seconds: float) -> float:
    return seconds / 3600.0


def collect_sacct(args: argparse.Namespace) -> list[dict[str, str]]:
    job_expr = ",".join([args.array_job, args.finalize_job, args.gpu_job, args.hf_job])
    command = (
        f"sacct -j {job_expr} "
        "--format=JobID,JobName%32,Partition,State,ExitCode,Elapsed,TotalCPU,"
        "AllocCPUS,ReqMem,NodeList,Start,End -P 2>/dev/null"
    )
    text = run_ssh(args, command)
    reader = csv.DictReader(text.splitlines(), delimiter="|")
    return [dict(row) for row in reader if row.get("JobID")]


def collect_squeue(args: argparse.Namespace) -> str:
    job_expr = ",".join([args.array_job, args.finalize_job, args.gpu_job, args.hf_job])
    return run_ssh(
        args,
        f"squeue -j {job_expr} -o '%.18i %.9P %.36j %.8T %.10M %.9l %.6D %R'",
    ).strip()


def collect_shards(args: argparse.Namespace) -> list[dict[str, Any]]:
    code = r"""
import json
from pathlib import Path
root = Path(__import__("sys").argv[1])
rows = []
for path in sorted(root.glob("*.done.json")):
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        rows.append({"sidecar": path.name, "error": str(exc)})
        continue
    payload["sidecar"] = path.name
    rows.append(payload)
print(json.dumps(rows, ensure_ascii=False))
"""
    escaped = json.dumps(code)
    escaped_dir = json.dumps(args.segmentation_dir)
    return json.loads(
        run_ssh(
            args,
            "python3 -c "
            + shlex.quote(code)
            + " "
            + shlex.quote(args.segmentation_dir),
        )
    )


def collect_file_counts(args: argparse.Namespace) -> dict[str, int]:
    code = r"""
from pathlib import Path
import json, sys
root = Path(sys.argv[1])
payload = {
    "done_json": len(list(root.glob("*.done.json"))),
    "segmented_parquet": len(list(root.glob("*.segmented.parquet"))),
    "tmp": len(list(root.glob("*.tmp.*"))),
    "manifest_json": int((root / "manifest.json").is_file()),
    "dataset_manifest_json": int((root / "fineweb2_manifest.json").is_file()),
}
print(json.dumps(payload))
"""
    return json.loads(
        run_ssh(
            args,
            "python3 -c " + shlex.quote(code) + " " + shlex.quote(args.segmentation_dir),
        )
    )


def is_top_level_job(row: dict[str, str], job_ids: set[str]) -> bool:
    job_id = row.get("JobID", "")
    if "." in job_id:
        return False
    if "_" in job_id:
        suffix = job_id.split("_", 1)[1]
        return suffix.isdigit()
    return job_id in job_ids


def array_placeholder_count(job_id: str, array_job: str) -> int:
    prefix = f"{array_job}_["
    if not job_id.startswith(prefix) or not job_id.endswith("]"):
        return 0
    body = job_id[len(prefix):-1].split("%", 1)[0]
    total = 0
    for part in body.split(","):
        if "-" in part:
            start, end = part.split("-", 1)
            total += int(end) - int(start) + 1
        elif part:
            total += 1
    return total


def summarize_sacct(rows: list[dict[str, str]], job_ids: set[str], array_job: str) -> dict[str, Any]:
    top = [row for row in rows if is_top_level_job(row, job_ids)]
    for row in top:
        elapsed_seconds = parse_duration_seconds(row.get("Elapsed", ""))
        total_cpu_seconds = parse_duration_seconds(row.get("TotalCPU", ""))
        alloc_cpus = int(row.get("AllocCPUS") or 0)
        row["elapsed_seconds"] = elapsed_seconds
        row["total_cpu_hours"] = hours(total_cpu_seconds)
        row["allocated_cpu_hours"] = hours(elapsed_seconds * alloc_cpus)

    seg_rows = [row for row in top if row.get("JobID", "").startswith(f"{array_job}_")]
    completed = [row for row in seg_rows if row.get("State") == "COMPLETED"]
    running = [row for row in seg_rows if row.get("State") == "RUNNING"]
    pending = [row for row in seg_rows if row.get("State") == "PENDING"]
    pending_placeholder_tasks = sum(
        array_placeholder_count(row.get("JobID", ""), array_job)
        for row in rows
        if row.get("State") == "PENDING"
    )
    failed = [
        row for row in seg_rows
        if row.get("State") not in {"COMPLETED", "RUNNING", "PENDING"}
    ]

    return {
        "top_level_rows": top,
        "segmentation": {
            "completed_tasks": len(completed),
            "running_tasks": len(running),
            "pending_tasks": len(pending) + pending_placeholder_tasks,
            "failed_or_other_tasks": len(failed),
            "completed_allocated_cpu_hours": sum(row["allocated_cpu_hours"] for row in completed),
            "completed_total_cpu_hours": sum(row["total_cpu_hours"] for row in completed),
            "running_allocated_cpu_hours_so_far": sum(row["allocated_cpu_hours"] for row in running),
            "running_total_cpu_hours_so_far": sum(row["total_cpu_hours"] for row in running),
        },
    }


def summarize_shards(shards: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed = [safe_float(row.get("elapsed_seconds")) for row in shards if safe_float(row.get("elapsed_seconds"))]
    docs = sum(int(row.get("docs") or 0) for row in shards)
    words = sum(int(row.get("word_count") or 0) for row in shards)
    split_words = sum(int(row.get("split_words") or 0) for row in shards)
    fallback_words = sum(int(row.get("fallback_words") or 0) for row in shards)
    return {
        "completed_shards": len(shards),
        "docs": docs,
        "words": words,
        "split_words": split_words,
        "fallback_words": fallback_words,
        "split_word_rate": split_words / words if words else 0.0,
        "fallback_word_rate": fallback_words / words if words else 0.0,
        "elapsed_hours_sum": hours(sum(elapsed)),
        "elapsed_hours_mean": hours(mean(elapsed)) if elapsed else 0.0,
        "elapsed_hours_median": hours(median(elapsed)) if elapsed else 0.0,
        "elapsed_hours_max": hours(max(elapsed)) if elapsed else 0.0,
    }


def fmt(value: float, digits: int = 2) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def render_markdown(payload: dict[str, Any]) -> str:
    shard = payload["shard_summary"]
    slurm = payload["slurm_summary"]["segmentation"]
    files = payload["file_counts"]
    rows = payload["slurm_summary"]["top_level_rows"]

    lines = [
        "# TRmorph MorphBPE 32k UHeM Operations Report",
        "",
        f"Generated at: `{payload['generated_at_utc']}`",
        "",
        "## Job Chain",
        "",
        f"- Segmentation array: `{payload['jobs']['array_job']}`",
        f"- Finalize/tokenizer job: `{payload['jobs']['finalize_job']}`",
        f"- Single-node A100 model job: `{payload['jobs']['gpu_job']}`",
        f"- HF tokenizer publish job: `{payload['jobs']['hf_job']}`",
        "",
        "Current Slurm queue snapshot:",
        "",
        "```text",
        payload["squeue"],
        "```",
        "",
        "## Segmentation Progress",
        "",
        f"- Completed shard sidecars: `{files['done_json']}`",
        f"- Completed segmented parquet files: `{files['segmented_parquet']}`",
        f"- Active temp shard files: `{files['tmp']}`",
        f"- Final segmentation manifest exists: `{bool(files['manifest_json'])}`",
        f"- Final dataset manifest exists: `{bool(files['dataset_manifest_json'])}`",
        "",
        "Completed-shard content totals:",
        "",
        f"- Documents: `{shard['docs']:,}`",
        f"- Word tokens seen by segmenter: `{shard['words']:,}`",
        f"- Split words: `{shard['split_words']:,}` ({fmt(100 * shard['split_word_rate'])}%)",
        f"- Fallback words: `{shard['fallback_words']:,}` ({fmt(100 * shard['fallback_word_rate'])}%)",
        "",
        "Shard wall-time stats from `.done.json` sidecars:",
        "",
        f"- Sum over completed shards: `{fmt(shard['elapsed_hours_sum'])}` shard-hours",
        f"- Mean per completed shard: `{fmt(shard['elapsed_hours_mean'])}` hours",
        f"- Median per completed shard: `{fmt(shard['elapsed_hours_median'])}` hours",
        f"- Max completed shard: `{fmt(shard['elapsed_hours_max'])}` hours",
        "",
        "## UHeM Resource Accounting",
        "",
        "Slurm accounting is reported two ways:",
        "",
        "- Allocated CPU-hours: `Elapsed * AllocCPUS`; this is the conservative cluster-allocation view.",
        "- TotalCPU hours: Slurm's measured CPU time; this is the process-consumption view.",
        "",
        f"- Completed segmentation tasks: `{slurm['completed_tasks']}`",
        f"- Running segmentation tasks: `{slurm['running_tasks']}`",
        f"- Pending segmentation tasks: `{slurm['pending_tasks']}`",
        f"- Failed/other segmentation tasks: `{slurm['failed_or_other_tasks']}`",
        f"- Completed allocated CPU-hours: `{fmt(slurm['completed_allocated_cpu_hours'])}`",
        f"- Completed TotalCPU hours: `{fmt(slurm['completed_total_cpu_hours'])}`",
        f"- Running allocated CPU-hours so far: `{fmt(slurm['running_allocated_cpu_hours_so_far'])}`",
        f"- Running TotalCPU hours so far: `{fmt(slurm['running_total_cpu_hours_so_far'])}`",
        "",
        "## Slurm Task Table",
        "",
        "| JobID | State | Elapsed | TotalCPU | AllocCPUS | Alloc CPU-hours | Node | Start | End |",
        "|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| `{JobID}` | `{State}` | `{Elapsed}` | `{TotalCPU}` | `{AllocCPUS}` | `{alloc}` | `{NodeList}` | `{Start}` | `{End}` |".format(
                JobID=row.get("JobID", ""),
                State=row.get("State", ""),
                Elapsed=row.get("Elapsed", ""),
                TotalCPU=row.get("TotalCPU", ""),
                AllocCPUS=row.get("AllocCPUS", ""),
                alloc=fmt(row.get("allocated_cpu_hours", 0.0)),
                NodeList=row.get("NodeList", ""),
                Start=row.get("Start", ""),
                End=row.get("End", ""),
            )
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- Tokenizer training has not started until the finalizer job leaves `PENDING`.",
        "- The model run and HF tokenizer upload are dependency-held until the finalizer succeeds.",
        "- Refresh this report by rerunning `scripts/report_uhem_tokenizer_jobs.py` with the same job IDs.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    sacct_rows = collect_sacct(args)
    squeue = collect_squeue(args)
    shards = collect_shards(args)
    file_counts = collect_file_counts(args)
    job_ids = {args.array_job, args.finalize_job, args.gpu_job, args.hf_job}
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": args.host,
        "segmentation_dir": args.segmentation_dir,
        "jobs": {
            "array_job": args.array_job,
            "finalize_job": args.finalize_job,
            "gpu_job": args.gpu_job,
            "hf_job": args.hf_job,
        },
        "squeue": squeue,
        "sacct_rows": sacct_rows,
        "slurm_summary": summarize_sacct(sacct_rows, job_ids, args.array_job),
        "file_counts": file_counts,
        "shard_summary": summarize_shards(shards),
        "completed_shards": shards,
    }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Wrote {output_md}")
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
