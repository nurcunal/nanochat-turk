"""Regression coverage for the June CETVEL adapter's protocol mistakes."""

import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from nanochat.lm_eval_protocol import (
    ContinuationBoundaryError,
    EVALUATION_PROTOCOL_VERSION,
    encode_loglikelihood_pair,
    generation_controls,
    trim_at_first_stop,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        ({"until": ["\n", "END"], "max_gen_toks": 5}, (["\n", "END"], 5)),
        ({"until": "END", "max_gen_toks": 100}, (["END"], 8)),
        ({"until": None}, ([], 8)),
        (["", "END"], (["END"], 8)),
        (None, ([], 8)),
    ],
)
def test_generation_controls_reads_kwargs_and_legacy_lists(value, expected):
    assert generation_controls(value, cli_max_tokens=8) == expected


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "8"])
def test_generation_controls_rejects_invalid_task_budget(limit):
    with pytest.raises((ValueError, TypeError)):
        generation_controls({"max_gen_toks": limit}, cli_max_tokens=8)


def test_stop_trimming_uses_first_occurrence_not_list_order():
    assert trim_at_first_stop("Ankara\nEND trailing", ["END", "\n"]) == ("Ankara", True)


def test_pair_moves_trailing_whitespace_before_tokenizing():
    encodings = {"Q": [0, 1], "Q ": [0, 1, 2], "Q A": [0, 1, 3]}
    assert encode_loglikelihood_pair(encodings.__getitem__, "Q ", "A") == ([0, 1, 3], 2)


def test_pair_rejects_cross_boundary_merge_instead_of_perfect_empty_score():
    encodings = {"ev": [0, 1], "evler": [0, 2]}
    with pytest.raises(ContinuationBoundaryError, match="crossed"):
        encode_loglikelihood_pair(encodings.__getitem__, "ev", "ler")


@pytest.fixture
def adapter_class(monkeypatch):
    # Import the real adapter with only lm-eval's registration seam stubbed.
    # No optional harness install, checkpoint loading or GPU is needed.
    for name in ("lm_eval", "lm_eval.api", "lm_eval.api.model", "lm_eval.api.registry"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["lm_eval.api.model"].LM = object
    sys.modules["lm_eval.api.registry"].register_model = lambda _name: lambda cls: cls
    path = Path(__file__).parents[1] / "nanochat" / "lm_eval_nanochat.py"
    spec = importlib.util.spec_from_file_location("adapter_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NanochatLM


class FakeTokenizer:
    def __init__(self, encodings=None, pieces=None):
        self.encodings = encodings or {}
        self.pieces = pieces or {}

    def encode(self, text):
        return self.encodings[text]

    def decode(self, ids):
        return "".join(self.pieces[token] for token in ids)

    def encode_special(self, _text):
        return 99


class FakeEngine:
    def __init__(self, tokens):
        self.tokens = tokens
        self.calls = []
        self.yielded = []

    def generate(self, ids, **kwargs):
        self.calls.append((ids, kwargs))
        for token in self.tokens[:kwargs["max_tokens"]]:
            self.yielded.append(token)
            yield [token], None


def make_adapter(adapter_class, tokenizer):
    adapter = adapter_class.__new__(adapter_class)
    adapter.tokenizer = tokenizer
    adapter.bos_id = 0
    adapter.device = torch.device("cpu")
    adapter.model_max_len = 16
    adapter.max_gen_tokens = 8
    return adapter


def test_generation_honors_task_stop_and_trims_same_token_suffix(adapter_class):
    tokenizer = FakeTokenizer({"prompt": [1]}, {2: "An", 3: "kara\nextra", 4: "ignored"})
    adapter = make_adapter(adapter_class, tokenizer)
    adapter.engine = FakeEngine([2, 3, 4])
    result = adapter.generate_until([SimpleNamespace(args=("prompt", {"until": "\n", "max_gen_toks": 5}))])
    assert result == ["Ankara"]
    assert adapter.engine.yielded == [2, 3]
    assert adapter.engine.calls[0][1]["max_tokens"] == 5


def test_generation_reserves_context_budget_and_stops_at_assistant_end(adapter_class):
    adapter = make_adapter(adapter_class, FakeTokenizer({"long": list(range(1, 20))}, {2: "yes"}))
    adapter.engine = FakeEngine([2, 99, 3])
    assert adapter.generate_until([SimpleNamespace(args=("long", {"max_gen_toks": 5}))]) == ["yes"]
    assert adapter.engine.calls[0][0] == list(range(9, 20))
    assert adapter.engine.yielded == [2, 99]


def test_generation_matches_stop_spanning_token_pieces(adapter_class):
    adapter = make_adapter(adapter_class, FakeTokenizer({"p": [1]}, {2: "A\n", 3: "\nB"}))
    adapter.engine = FakeEngine([2, 3])
    assert adapter.generate_until([SimpleNamespace(args=("p", {"until": ["\n\n"]}))]) == ["A"]


def test_generation_rejects_impossible_context_budget(adapter_class):
    adapter = make_adapter(adapter_class, FakeTokenizer())
    adapter.model_max_len = 8
    with pytest.raises(ValueError, match="prompt token"):
        adapter.generate_until([SimpleNamespace(args=("p", {}))])


class UniformModel:
    def __init__(self):
        self.inputs = []

    def __call__(self, ids):
        self.inputs.append(ids.tolist())
        return torch.zeros((*ids.shape, 10))


def test_likelihood_scores_whitespace_merged_answer_token(adapter_class):
    adapter = make_adapter(adapter_class, FakeTokenizer({"Q": [1], "Q ": [1, 2], "Q A": [1, 3]}))
    adapter.model = UniformModel()
    score, greedy = adapter.loglikelihood([SimpleNamespace(args=("Q ", "A"))])[0]
    assert score == pytest.approx(-math.log(10))
    assert greedy is False
    assert adapter.model.inputs == [[[0, 1]]]


def test_likelihood_keeps_full_input_window_and_all_continuation_tokens(adapter_class):
    adapter = make_adapter(adapter_class, FakeTokenizer({"context": [1, 2, 3, 4], "contextAB": [1, 2, 3, 4, 5, 6]}))
    adapter.model_max_len = 4
    adapter.model = UniformModel()
    score, _ = adapter.loglikelihood([SimpleNamespace(args=("context", "AB"))])[0]
    assert score == pytest.approx(-2 * math.log(10))
    assert adapter.model.inputs == [[[2, 3, 4, 5]]]


def test_likelihood_rejects_overlong_continuation_instead_of_scoring_suffix(adapter_class):
    adapter = make_adapter(adapter_class, FakeTokenizer({"Q": [1], "Qlong": [1, 2, 3, 4]}))
    adapter.model_max_len = 2
    with pytest.raises(ValueError, match="exceeds"):
        adapter.loglikelihood([SimpleNamespace(args=("Q", "long"))])


def test_empty_continuation_is_zero_without_model_inference(adapter_class):
    adapter = make_adapter(adapter_class, FakeTokenizer({"Q ": [1, 2]}))
    assert adapter.loglikelihood([SimpleNamespace(args=("Q ", ""))]) == [(0.0, True)]


def test_cetvel_records_adapter_protocol_in_result_json():
    from scripts.cetvel_eval import _run_lm_eval

    evaluator = SimpleNamespace(simple_evaluate=lambda **_kwargs: {"results": {"example": {"acc": 0.5}}})
    args = SimpleNamespace(batch_size="1", device="cpu", limit=None)
    result = _run_lm_eval(evaluator, "", ["example"], args, None, None)
    assert result["nanochat_adapter_protocol"] == EVALUATION_PROTOCOL_VERSION
