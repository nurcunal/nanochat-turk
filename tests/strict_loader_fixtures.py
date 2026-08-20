from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.experiment_manifest import file_sha256, seal_manifest, write_json_atomic
from nanochat.exposure import (
    SourceDocument,
    build_exposure_manifest,
    build_training_exposure_plan,
)


STUDY_HASH = "3" * 64
TOKENIZER_HASH = "4" * 64
MANIFEST_FILE = "fineweb2_manifest.json"


class CharacterFixtureTokenizer:
    def __init__(self, alphabet: str):
        self.character_to_id = {
            character: index + 1
            for index, character in enumerate(sorted(set(alphabet), key=ord))
        }
        self.id_to_character = {
            token_id: character
            for character, token_id in self.character_to_id.items()
        }

    def get_bos_token_id(self):
        return 0

    def encode(self, text, *, prepend=None, num_threads=None):
        del num_threads

        def one(value):
            tokens = [self.character_to_id[character] for character in value]
            return tokens if prepend is None else [prepend, *tokens]

        if isinstance(text, list):
            return [one(value) for value in text]
        return one(text)

    def decode(self, token_ids):
        return "".join(self.id_to_character[token_id] for token_id in token_ids)

    def token_bytes(self):
        values = [0] * (len(self.character_to_id) + 1)
        for character, token_id in self.character_to_id.items():
            values[token_id] = len(character.encode("utf-8"))
        return values


def strict_dataset(
    tmp_path: Path,
    train_texts=("ab", "çd"),
    *,
    validation_texts=("va",),
):
    train = tmp_path / "train.parquet"
    validation = tmp_path / "validation.parquet"
    pq.write_table(pa.table({"text": list(train_texts)}), train, row_group_size=2)
    pq.write_table(pa.table({"text": list(validation_texts)}), validation)
    manifest = seal_manifest(
        {
            "schema_version": "1.0",
            "manifest_type": "dataset",
            "profile": "strict",
            "dataset": {
                "repo_id": "example/strict-loader-fixture",
                "repo_type": "dataset",
                "path": "data/tur_Latn/train",
                "requested_revision": "1" * 40,
                "resolved_revision": "1" * 40,
            },
            "text_column": "text",
            "ordered_files": [
                {
                    "path": train.name,
                    "size_bytes": train.stat().st_size,
                    "sha256": file_sha256(train),
                },
                {
                    "path": validation.name,
                    "size_bytes": validation.stat().st_size,
                    "sha256": file_sha256(validation),
                },
            ],
            "validation_file": validation.name,
            "created_by": {"git_commit": "2" * 40, "tool": "test"},
        }
    )
    write_json_atomic(tmp_path / MANIFEST_FILE, manifest)
    documents = [
        SourceDocument("train.parquet", 0, index, text)
        for index, text in enumerate(train_texts)
    ]
    exposure = build_exposure_manifest(
        documents,
        mode="documents",
        target_value=len(documents),
        source_dataset_manifest_sha256=manifest["canonical_sha256"],
        study_sha256=STUDY_HASH,
    )
    tokenizer = CharacterFixtureTokenizer("".join(train_texts) + "".join(validation_texts))
    return manifest, exposure, tokenizer


def strict_training_plan(
    dataset_manifest: dict,
    *,
    world_size: int = 1,
    target_token_positions: int = 1_000_000,
    seed: int = 42,
) -> dict:
    return build_training_exposure_plan(
        estimand="equal_token",
        source_dataset_manifest_sha256=dataset_manifest["canonical_sha256"],
        study_sha256=STUDY_HASH,
        tokenizer_sha256=TOKENIZER_HASH,
        seed=seed,
        world_size=world_size,
        target_token_positions=target_token_positions,
    )


def strict_validation_manifest(
    dataset_manifest: dict, texts: tuple[str, ...] = ("va",)
) -> dict:
    return build_exposure_manifest(
        [
            SourceDocument("validation.parquet", 0, index, text)
            for index, text in enumerate(texts)
        ],
        mode="validation",
        target_value=len(texts),
        source_dataset_manifest_sha256=dataset_manifest["canonical_sha256"],
        study_sha256=STUDY_HASH,
    )
