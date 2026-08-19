import json

import pytest

from scripts.prepare_turkce_atlas_sft import prepare_atlas


SYSTEM = "Sen yalnızca Türkçe yanıt veren bir asistansın."


def _source_row(user, assistant, system=SYSTEM):
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def _write_source(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_prepare_atlas_preserves_dialogue_and_groups_duplicate_prompts(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "prepared"
    rows = [
        _source_row("aynı\nsoru", "Birinci yanıt — değiştirme"),
        _source_row("başka soru", "İkinci yanıt"),
        _source_row("aynı\nsoru", "Alternatif yanıt"),
        _source_row("son soru", "Son yanıt"),
    ]
    _write_source(source, rows)

    manifest = prepare_atlas(
        source,
        output,
        validation_size=1,
        split_seed=7,
        tokenizer_dir=None,
    )

    train = _read_jsonl(output / "train.jsonl")
    validation = _read_jsonl(output / "validation.jsonl")
    prepared = train + validation
    prepared_pairs = {
        (row[0]["content"], row[1]["content"])
        for row in prepared
    }
    source_pairs = {
        (row["messages"][1]["content"], row["messages"][2]["content"])
        for row in rows
    }

    assert prepared_pairs == source_pairs
    assert all([message["role"] for message in row] == ["user", "assistant"] for row in prepared)
    duplicate_locations = {
        split_name
        for split_name, split_rows in (("train", train), ("validation", validation))
        if any(row[0]["content"] == "aynı\nsoru" for row in split_rows)
    }
    assert len(duplicate_locations) == 1
    assert (output / "source_system_prompt.txt").read_text(encoding="utf-8") == SYSTEM
    assert manifest["normalization"]["source_system_prompt"]["retained_in_output"] is False
    assert manifest["source"]["rows"] == 4
    assert manifest["normalization"]["rows_removed"] == 0
    assert manifest["normalization"]["user_or_assistant_text_rewritten"] is False
    assert sum(item["rows"] for item in manifest["outputs"].values()) == 4


def test_prepare_atlas_refuses_to_drop_non_shared_system_messages(tmp_path):
    source = tmp_path / "source.jsonl"
    _write_source(
        source,
        [
            _source_row("bir", "yanıt", system="Sistem bir"),
            _source_row("iki", "yanıt", system="Sistem iki"),
        ],
    )

    with pytest.raises(ValueError, match="exactly one shared prompt"):
        prepare_atlas(source, tmp_path / "prepared", validation_size=1, tokenizer_dir=None)


def test_prepare_atlas_removes_known_incomplete_responses_without_rewriting(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "prepared"
    rows = [
        _source_row("tam soru bir", "Tam yanıt bir"),
        _source_row("bozuk özet", "S:"),
        _source_row("tam soru iki", "Tam yanıt iki"),
    ]
    _write_source(source, rows)

    manifest = prepare_atlas(
        source,
        output,
        validation_size=1,
        split_seed=7,
        tokenizer_dir=None,
    )

    prepared = _read_jsonl(output / "train.jsonl") + _read_jsonl(
        output / "validation.jsonl"
    )
    assert len(prepared) == 2
    assert all(row[1]["content"] != "S:" for row in prepared)
    assert manifest["source"]["rows"] == 3
    assert manifest["normalization"]["rows_removed"] == 1
    assert manifest["normalization"]["filters"]["removed_counts"] == {"S:": 1}
