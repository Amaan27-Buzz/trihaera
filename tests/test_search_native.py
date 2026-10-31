"""Search-native policy, LoRA masking and runtime regression tests.

No network access or downloaded model required. These tests do NOT prove
search-answer quality; they verify the executable contract and safety gates.
"""
import json
from types import SimpleNamespace

import pytest

from lra.search_native import parse_decision, render_state, target_json, POLICY_SYSTEM
from lra.search_examples import examples, split_examples, to_messages
from lra.search_runtime import SearchRuntime, LocalSearXNG
from scripts.train_search_adapter import encode_supervised


def payload(decision, answer="", queries=None, citations=None):
    return json.dumps({
        "decision": decision, "answer": answer,
        "queries": queries or [], "citations": citations or [],
    })


def test_valid_action_decisions_and_source_alignment():
    search = parse_decision(payload("search", queries=["new chip release"]))
    assert search.decision == "search"
    assert search.queries == ("new chip release",)
    answer = parse_decision(
        payload("answer", "The response is 4.", citations=["s1"]),
        valid_sources={"s1", "s2"},
    )
    assert answer.citations == ("s1",)
    abstain = parse_decision(payload("abstain", "I don't know."))
    assert abstain.decision == "abstain"


@pytest.mark.parametrize("invalid", [
    "not JSON",
    '{"decision":"search"}',
    payload("search"),
    payload("search", answer="hallucination", queries=["query"]),
    payload("answer", "x", queries=["rogue"]),
    payload("answer", ""),
    payload("abstain", "unknown", citations=["s1"]),
    payload("search", queries=["", "valid"]),
    payload("search", queries=["a", "b", "c", "d"]),
    payload("execute_shell", "bad"),
])
def test_reject_malformed_or_unpermitted_actions(invalid):
    with pytest.raises(ValueError):
        parse_decision(invalid)


def test_citations_must_be_actual_sources():
    with pytest.raises(ValueError):
        parse_decision(payload("answer", "made-up", citations=["s99"]),
                       valid_sources={"s1"})
    with pytest.raises(ValueError):
        render_state("Question?", [{"id":"s1", "title":"A", "text":"a"},
                                   {"id":"s1", "title":"B", "text":"b"}])


def test_offline_dataset_splits_disjoint_and_balanced():
    train, test = split_examples("train"), split_examples("test")
    assert len(train) >= 40
    assert len(test) >= 10
    ids_train = {x["id"] for x in train}
    ids_test = {x["id"] for x in test}
    assert not ids_train & ids_test
    assert len(ids_train | ids_test) == len(examples())
    assert {x["kind"] for x in train} == {
        "stable", "fresh", "evidence", "private", "conflict"
    }
    assert {x["kind"] for x in test} == {
        "stable", "fresh", "evidence", "private", "conflict"
    }
    for row in train + test:
        render = to_messages(row)
        assert render[0]["role"] == "system"
        assert render[-1]["role"] == "user"
        assert row["question"] in render[-1]["content"]
        parsed = parse_decision(
            target_json(row["target"]),
            valid_sources={s["id"] for s in row["evidence"]}
        )
        assert parsed.decision == row["target"]["decision"]


class FakeTokenizer:
    eos_token_id = 2
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        assert messages[-1]["role"] == "user"
        return [10, 11, 12]
    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        assert '"decision":' in text
        return [21, 22, 23, 24]


def test_lora_only_trains_assistant_action_tokens():
    row = split_examples("train")[0]
    encoded = encode_supervised(FakeTokenizer(), row, max_length=20)
    assert encoded["input_ids"] == [10,11,12,21,22,23,24,2]
    assert encoded["labels"] == [-100,-100,-100,21,22,23,24,2]
    assert encoded["kind"] == row["kind"]
    with pytest.raises(ValueError, match="Increase --max-length"):
        encode_supervised(FakeTokenizer(), row, max_length=5)


class FakePolicy:
    def __init__(self, *decisions):
        self.decisions = list(decisions)
    def generate(self, messages):
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        return SimpleNamespace(text=self.decisions.pop(0))


class FakeBackend:
    def __init__(self, results=None):
        self.queries = []
        self.results = results if results is not None else [
            {"title":"Source", "text":"A documented fact.", "url":"https://example.org/fact"}
        ]
    def search(self, query, *, limit):
        self.queries.append(query)
        return self.results[:limit]


def test_model_decides_search_and_then_cites_real_result():
    policy = FakePolicy(
        payload("search", queries=["factual query"]),
        payload("answer", "A documented fact.", citations=["s1"]),
    )
    backend = FakeBackend()
    report = SearchRuntime(policy, backend).ask("What is the fact?")
    assert backend.queries == ["factual query"]
    assert report["status"] == "answer"
    assert report["sources"][0]["url"] == "https://example.org/fact"
    assert report["answer"] == "A documented fact."
    assert report["search_rounds"] == 1


def test_search_result_without_citation_not_treated_as_verified():
    policy = FakePolicy(
        payload("search", queries=["fact"]),
        payload("answer", "I guessed the fact."),
    )
    result = SearchRuntime(policy, FakeBackend()).ask("What?")
    assert result["status"] == "unverified"
    assert result["answer"].startswith("I couldn't verify")


def test_no_results_and_search_budget_are_bounded():
    backend = FakeBackend(results=[])
    answer = SearchRuntime(FakePolicy(payload("search", queries=["test"])),
                           backend).ask("Question")
    assert answer["status"] == "no_results"
    assert len(backend.queries) == 1
    still = SearchRuntime(
        FakePolicy(payload("search", queries=["first"]),
                   payload("search", queries=["second"])),
        FakeBackend(), max_rounds=1,
    ).ask("Question")
    assert still["status"] == "budget_exhausted"


def test_bad_source_id_fails_closed():
    policy = FakePolicy(
        payload("search", queries=["fact"]),
        payload("answer", "unsupported", citations=["s99"]),
    )
    with pytest.raises(ValueError, match="Citations"):
        SearchRuntime(policy, FakeBackend()).ask("Question")


def test_no_search_when_model_answers_stable_fact():
    policy = FakePolicy(payload("answer", "Deep learning uses multilayer networks."))
    backend = FakeBackend()
    report = SearchRuntime(policy, backend).ask("What is deep learning?")
    assert report["status"] == "answer"
    assert report["sources"] == []
    assert backend.queries == []


def test_search_backend_is_loopback_only():
    LocalSearXNG("http://127.0.0.1:8080")
    for endpoint in (
        "https://evil.example", "http://10.0.0.1:8080",
        "http://127.0.0.1:8080/secret", "http://user:pass@localhost:8080",
    ):
        with pytest.raises(ValueError):
            LocalSearXNG(endpoint)
