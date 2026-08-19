import json

import pytest

from tasks.customjson import CustomJSON


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_customjson_accepts_standard_messages_object(tmp_path, lazy):
    path = tmp_path / "conversations.jsonl"
    rows = [
        {
            "messages": [
                {"role": "user", "content": "Merhaba"},
                {"role": "assistant", "content": "Merhaba!"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "Türkçe yanıt ver."},
                {"role": "user", "content": "Nasılsın?"},
                {"role": "assistant", "content": "İyiyim."},
            ]
        },
    ]
    _write_jsonl(path, rows)

    task = CustomJSON(str(path), lazy=lazy)

    assert len(task) == 2
    assert task[0] == rows[0]
    assert task[1] == rows[1]


def test_customjson_accepts_legacy_bare_message_array(tmp_path):
    path = tmp_path / "legacy.jsonl"
    messages = [
        {"role": "user", "content": "Soru"},
        {"role": "assistant", "content": "Yanıt"},
    ]
    _write_jsonl(path, [messages])

    assert CustomJSON(str(path))[0] == {"messages": messages}


def test_customjson_rejects_incomplete_dialogue(tmp_path):
    path = tmp_path / "invalid.jsonl"
    _write_jsonl(path, [{"messages": [{"role": "user", "content": "Eksik"}]}])

    with pytest.raises(AssertionError, match="at least 2 messages"):
        CustomJSON(str(path))
