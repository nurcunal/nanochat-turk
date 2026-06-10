import json
import os
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nanochat.dataset import MANIFEST_FILE, list_parquet_files


def write_raw_shard(path, rows):
    table = pa.table({
        "id": [row[0] for row in rows],
        "text": [row[1] for row in rows],
    })
    pq.write_table(table, path, row_group_size=1)


def run_segmenter(data_dir, output_dir, *extra_args):
    cmd = [
        sys.executable,
        "-m",
        "scripts.morph_segment_corpus",
        "--backend",
        "identity",
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--row-group-batch-size",
        "1",
        "--segment-batch-size",
        "2",
        *extra_args,
    ]
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def test_morph_segment_corpus_resume_rewrites_incomplete_and_skips_done(tmp_path):
    data_dir = tmp_path / "raw"
    output_dir = tmp_path / "segmented"
    data_dir.mkdir()

    first = data_dir / "b.parquet"
    second = data_dir / "a.parquet"
    write_raw_shard(first, [("1", "evlerden geldik"), ("2", "okula")])
    write_raw_shard(second, [("3", "istanbul")])
    (data_dir / MANIFEST_FILE).write_text(
        json.dumps({"filenames": [first.name, second.name], "text_column": "text"}),
        encoding="utf-8",
    )

    output_dir.mkdir()
    incomplete = output_dir / "b.segmented.parquet"
    incomplete.write_text("not a parquet file", encoding="utf-8")

    first_run = run_segmenter(data_dir, output_dir, "--resume")
    assert "Removing incomplete shard before resume" in first_run.stdout
    assert (output_dir / "b.segmented.parquet.done.json").is_file()
    assert (output_dir / "a.segmented.parquet.done.json").is_file()
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "manifest.partial.json").is_file()
    assert (output_dir / f"{MANIFEST_FILE}.partial").is_file()

    final_manifest = json.loads((output_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert final_manifest["complete"] is True
    assert final_manifest["filenames"] == ["b.segmented.parquet", "a.segmented.parquet"]
    assert [os.path.basename(path) for path in list_parquet_files(str(output_dir))] == [
        "b.segmented.parquet",
        "a.segmented.parquet",
    ]

    second_run = run_segmenter(data_dir, output_dir, "--resume")
    assert "Skipping completed shard" in second_run.stdout
    assert second_run.stdout.count("Skipping completed shard") == 2


def test_morph_segment_corpus_shard_index_and_finalize_only(tmp_path):
    data_dir = tmp_path / "raw"
    output_dir = tmp_path / "segmented"
    data_dir.mkdir()

    first = data_dir / "first.parquet"
    second = data_dir / "second.parquet"
    write_raw_shard(first, [("1", "birinci shard")])
    write_raw_shard(second, [("2", "ikinci shard")])
    (data_dir / MANIFEST_FILE).write_text(
        json.dumps({"filenames": [first.name, second.name], "text_column": "text"}),
        encoding="utf-8",
    )

    run_segmenter(data_dir, output_dir, "--resume", "--shard-index", "0")
    assert (output_dir / "first.segmented.parquet.done.json").is_file()
    assert not (output_dir / MANIFEST_FILE).exists()

    run_segmenter(data_dir, output_dir, "--resume", "--shard-index", "1")
    assert (output_dir / "second.segmented.parquet.done.json").is_file()
    assert not (output_dir / MANIFEST_FILE).exists()

    run_segmenter(data_dir, output_dir, "--resume", "--finalize-only")
    final_manifest = json.loads((output_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert final_manifest["complete"] is True
    assert final_manifest["filenames"] == [
        "first.segmented.parquet",
        "second.segmented.parquet",
    ]


def test_morph_segment_corpus_parallel_workers_preserve_order(tmp_path):
    data_dir = tmp_path / "raw"
    output_dir = tmp_path / "segmented"
    data_dir.mkdir()

    shard = data_dir / "ordered.parquet"
    rows = [(str(i), f"belge {i} evlerden okula") for i in range(8)]
    write_raw_shard(shard, rows)
    (data_dir / MANIFEST_FILE).write_text(
        json.dumps({"filenames": [shard.name], "text_column": "text"}),
        encoding="utf-8",
    )

    run_segmenter(
        data_dir,
        output_dir,
        "--resume",
        "--num-workers",
        "2",
        "--worker-mode",
        "thread",
        "--max-in-flight",
        "4",
    )

    out = pq.read_table(output_dir / "ordered.segmented.parquet")
    assert out.column("source_row").to_pylist() == list(range(8))
    assert out.column("id").to_pylist() == [str(i) for i in range(8)]
    assert out.column("segmented_text").to_pylist() == [row[1] for row in rows]

    sidecar = json.loads(
        (output_dir / "ordered.segmented.parquet.done.json").read_text(encoding="utf-8")
    )
    assert sidecar["num_workers"] == 2
    assert sidecar["worker_mode"] == "thread"
    assert sidecar["max_in_flight"] == 4

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["num_workers"] == 2
    assert manifest["worker_mode"] == "thread"
    assert manifest["completed_shard_num_workers"] == [2]


def test_morph_segment_corpus_process_workers_when_available(tmp_path):
    data_dir = tmp_path / "raw"
    output_dir = tmp_path / "segmented"
    data_dir.mkdir()

    shard = data_dir / "process.parquet"
    rows = [(str(i), f"belge {i} evlerden okula") for i in range(6)]
    write_raw_shard(shard, rows)
    (data_dir / MANIFEST_FILE).write_text(
        json.dumps({"filenames": [shard.name], "text_column": "text"}),
        encoding="utf-8",
    )

    try:
        run_segmenter(
            data_dir,
            output_dir,
            "--resume",
            "--num-workers",
            "2",
            "--worker-mode",
            "process",
            "--max-in-flight",
            "4",
        )
    except subprocess.CalledProcessError as exc:
        if "PermissionError" in exc.stderr and "SC_SEM_NSEMS_MAX" in exc.stderr:
            pytest.skip("Local sandbox blocks Python multiprocessing semaphores")
        raise

    out = pq.read_table(output_dir / "process.segmented.parquet")
    assert out.column("source_row").to_pylist() == list(range(6))
    assert out.column("id").to_pylist() == [str(i) for i in range(6)]
    assert out.column("segmented_text").to_pylist() == [row[1] for row in rows]

    sidecar = json.loads(
        (output_dir / "process.segmented.parquet.done.json").read_text(encoding="utf-8")
    )
    assert sidecar["num_workers"] == 2
    assert sidecar["worker_mode"] == "process"
