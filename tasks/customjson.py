"""Load nanochat conversations from JSONL files.

Accepted line formats are either a bare message array or the standard
``{"messages": [...]}`` object. Conversations may begin with one system
message, followed by alternating user and assistant messages.
"""

import os
import json
from tasks.common import Task

class CustomJSON(Task):
    """
    Load conversations from a JSONL file.

    ``lazy=True`` stores byte offsets instead of materializing every message in
    memory. This is preferable for large SFT corpora and remains compatible
    with Task's random-access interface.
    """

    def __init__(self, filepath, lazy=False, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.lazy = lazy
        self.conversations = []
        self.offsets = []
        self._file = None

        # Load all conversations from the JSONL file
        if not os.path.exists(filepath):
            # Helpful error message due to recent change. Will be removed in the future.
            print("-" * 80)
            print(f"Warning: File {filepath} does not exist")
            print("HINT (Oct 21 2025)")
            print("If you recently did a git pull and suddenly see this, it might be due to the new addition of identity conversations")
            print("See this discussion for more details: https://github.com/karpathy/nanochat/discussions/139")
            print("Quick fix: simply run the following command to download the file and you're done:")
            print(f"curl -L -o {filepath} https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl")
            print("-" * 80)

        elif self.lazy:
            with open(filepath, "rb") as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    self._parse_line(line.decode("utf-8"), len(self.offsets) + 1)
                    self.offsets.append(offset)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_number, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:  # skip empty lines
                        continue
                    messages = self._parse_line(line, line_number)
                    self.conversations.append(messages)

        self.length = len(self.offsets) if self.lazy else len(self.conversations)

    @staticmethod
    def _validate_messages(messages, line_number):
        location = f"line {line_number}"
        assert isinstance(messages, list), f"{location}: expected a list of messages, got {type(messages)}"
        assert len(messages) >= 2, f"{location}: conversation must have at least 2 messages, got {len(messages)}"

        first_role = messages[0].get("role") if isinstance(messages[0], dict) else None
        first_dialogue_index = 1 if first_role == "system" else 0
        dialogue_messages = messages[first_dialogue_index:]
        assert len(dialogue_messages) >= 2, f"{location}: conversation must contain a user/assistant pair"
        assert len(dialogue_messages) % 2 == 0, (
            f"{location}: conversation must contain complete user/assistant pairs"
        )

        for i, message in enumerate(messages):
            assert isinstance(message, dict), f"{location}: message {i} must be an object"
            assert "role" in message, f"{location}: message {i} missing 'role' field"
            assert "content" in message, f"{location}: message {i} missing 'content' field"
            if i == 0 and first_dialogue_index == 1:
                expected_role = "system"
            else:
                dialogue_index = i - first_dialogue_index
                expected_role = "user" if dialogue_index % 2 == 0 else "assistant"
            assert message["role"] == expected_role, (
                f"{location}: message {i} has role {message['role']} but should be {expected_role}"
            )
            assert isinstance(message["content"], str), (
                f"{location}: message {i} content must be a string"
            )
        return messages

    @classmethod
    def _parse_line(cls, line, line_number):
        payload = json.loads(line)
        if isinstance(payload, dict):
            assert "messages" in payload, f"line {line_number}: object is missing 'messages'"
            messages = payload["messages"]
        else:
            messages = payload
        return cls._validate_messages(messages, line_number)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        if self.lazy:
            if self._file is None:
                self._file = open(self.filepath, "rb")
            self._file.seek(self.offsets[index])
            line = self._file.readline().decode("utf-8")
            messages = self._parse_line(line, index + 1)
        else:
            messages = self.conversations[index]
        conversation = {
            "messages": messages,
        }
        return conversation

    def __del__(self):
        if self._file is not None:
            self._file.close()
