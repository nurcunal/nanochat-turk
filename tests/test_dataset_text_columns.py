import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.dataset import MANIFEST_FILE, list_parquet_files, parquets_iter_batched
from nanochat.morphology import MORPHEME_BOUNDARY


def write_parquet(path, text, segmented_text):
    table = pa.table({
        "text": [text],
        "segmented_text": [segmented_text],
    })
    pq.write_table(table, path)


def test_parquets_iter_batched_uses_custom_text_column_and_manifest_order(tmp_path):
    first = tmp_path / "b.segmented.parquet"
    second = tmp_path / "a.segmented.parquet"
    write_parquet(first, "raw train", f"raw{MORPHEME_BOUNDARY} train")
    write_parquet(second, "raw val", f"raw{MORPHEME_BOUNDARY} val")
    (tmp_path / MANIFEST_FILE).write_text(
        json.dumps({
            "text_column": "segmented_text",
            "filenames": [os.path.basename(first), os.path.basename(second)],
        }),
        encoding="utf-8",
    )

    assert [os.path.basename(path) for path in list_parquet_files(str(tmp_path))] == [
        "b.segmented.parquet",
        "a.segmented.parquet",
    ]

    train_batches = list(parquets_iter_batched(
        "train",
        data_dir=str(tmp_path),
        text_column="segmented_text",
    ))
    val_batches = list(parquets_iter_batched(
        "val",
        data_dir=str(tmp_path),
        text_column="segmented_text",
    ))

    assert train_batches == [[f"raw{MORPHEME_BOUNDARY} train"]]
    assert val_batches == [[f"raw{MORPHEME_BOUNDARY} val"]]


def test_document_batches_uses_configured_text_column(tmp_path, monkeypatch):
    import nanochat.dataset as dataset
    import nanochat.dataloader as dataloader

    first = tmp_path / "train.segmented.parquet"
    second = tmp_path / "val.segmented.parquet"
    write_parquet(first, "raw train", f"raw{MORPHEME_BOUNDARY} train")
    write_parquet(second, "raw val", f"raw{MORPHEME_BOUNDARY} val")
    (tmp_path / MANIFEST_FILE).write_text(
        json.dumps({
            "text_column": "segmented_text",
            "filenames": [os.path.basename(first), os.path.basename(second)],
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(dataset, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(dataloader, "TEXT_COLUMN", "segmented_text")

    train_batch, train_state = next(dataloader._document_batches(
        "train",
        resume_state_dict=None,
        tokenizer_batch_size=8,
    ))
    val_batch, val_state = next(dataloader._document_batches(
        "val",
        resume_state_dict=None,
        tokenizer_batch_size=8,
    ))

    assert train_batch == [f"raw{MORPHEME_BOUNDARY} train"]
    assert train_state == (0, 0, 1)
    assert val_batch == [f"raw{MORPHEME_BOUNDARY} val"]
    assert val_state == (0, 0, 1)
