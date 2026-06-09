import json
import os
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq

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
