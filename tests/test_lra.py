import tempfile
from pathlib import Path

import pytest

from lra.storage import History
from lra.memory import extract_explicit_facts
from lra.context import build_context
from lra.app import LRA
from lra.cli import main


class Embedder:
    model_id = "local-test-embedding"

    def encode(self, text):
        t = text.lower()
        return [float("carmine" in t), float("project" in t), float("preference" in t)]


def count_tokens(messages):
    return 12 + sum(len(m["content"]) // 4 + 8 for m in messages)


def test_immutable_canonical_history_survives_restart(tmp_path):
    path = tmp_path / "lifelong.sqlite3"
    with History(path) as h:
        first = h.append("user", "The archival code was carmine-orbit-47.")
        second = h.append("assistant", "I will remember that.")
        for i in range(120):
            h.append("user", f"Current project diary entry {i}")
        assert h.get(first)["content"] == "The archival code was carmine-orbit-47."
        assert len(h.messages(limit=3)) == 3
        assert h.count() == 122
    with History(path) as h:
        assert h.count() == 122
        assert h.get(second)["role"] == "assistant"
        found = h.search("archival code carmine", limit=3)
        assert first in [x["id"] for x in found]


def test_corrections_preserve_original_sources(tmp_path):
    with History(tmp_path / "db") as h:
        a = h.append("user", "My favorite editor is A.")
        note1 = h.remember("editor", "A", source_message_id=a)
        b = h.append("user", "Correction: my editor is B.")
        note2 = h.remember("editor", "B", source_message_id=b)
        assert h.memory("editor")["value"] == "B"
        history = h.memory_history("editor")
        assert len(history) == 2
        assert history[0]["is_active"] == 0
        assert history[1]["supersedes_id"] == note1
        assert history[1]["id"] == note2
        assert h.get(a)["content"] == "My favorite editor is A."
        assert h.get(b)["content"] == "Correction: my editor is B."
        assert h.stats()["memory_revisions"] == 2


def test_reject_non_user_memory_sources(tmp_path):
    with History(tmp_path / "db") as h:
        m = h.append("assistant", "I prefer things")
        with pytest.raises(ValueError):
            h.remember("preference", "X", source_message_id=m)
        with pytest.raises(ValueError):
            h.remember("preference", "X", source_message_id=999)


def test_source_grounded_long_range_retrieval_and_bounded_context(tmp_path):
    with History(tmp_path / "db") as h:
        very_old = h.append("user", "The blue antenna project access word is carmine.")
        for i in range(90):
            h.append("user", f"Brief exchange about garden and tea number {i}")
            h.append("assistant", "Noted.")
        current = h.append("user", "What is the blue antenna project access word?")
        plan = build_context(
            h, "blue antenna project access word",
            token_count=count_tokens, budget=450, recent_messages=5, recall=5,
        )
        assert very_old in plan.retrieved_ids
        assert any(f"message #{very_old}" in item["content"] for item in plan.messages)
        assert plan.messages[-1]["content"] == h.get(current)["content"]
        assert plan.input_tokens <= 450
        assert h.count() == 182
        assert h.get(very_old)["content"].endswith("carmine.")


def test_context_budget_raises_without_silently_truncating_current(tmp_path):
    with History(tmp_path / "db") as h:
        h.append("user", "X" * 3000)
        with pytest.raises(ValueError, match="Newest"):
            build_context(h, "anything", token_count=count_tokens, budget=128)


def test_semantic_index_with_fake_embeddings(tmp_path):
    e = Embedder()
    with History(tmp_path / "db") as h:
        a = h.append("user", "The carmine identifier is recorded", embedder=e)
        b = h.append("user", "Discussing a project")
        assert h.index_vectors(e) == 1
        assert h.index_vectors(e) == 0
        assert h.stats()["vectors"] == 2
        results = h.search("carmine", embedder=e, limit=2)
        assert results[0]["id"] == a
        assert {r["id"] for r in results} == {a,b}


def test_explicit_fact_capture_only():
    assert ("profile:name", "Morgan") in extract_explicit_facts("Actually my name is Morgan.")
    assert ("preference:style", "short answers") in extract_explicit_facts("I prefer short answers.")
    assert extract_explicit_facts("The weather seems to be rainy.") == []


class FakeResult:
    text = "Verified answer from local model."
    latency_seconds = 0.001
    output_tokens = 8
    max_cuda_allocated_bytes = None


class FakeModel:
    window = 1024
    max_new_tokens = 100

    def count_chat_tokens(self, messages):
        return count_tokens(messages)

    def generate(self, messages):
        assert any(m["role"] == "user" for m in messages)
        return FakeResult()


def test_end_to_end_one_conversation_with_fake_model(tmp_path):
    path = tmp_path / "db.sqlite"
    with History(path) as h:
        lra = LRA(h, FakeModel(), max_input_tokens=700)
        first = lra.ask("My name is Morgan.")
        assert first["answer"] == FakeResult.text
        assert h.memory("profile:name")["value"] == "Morgan"
        for i in range(30):
            h.append("user", f"A diary turn {i}")
            h.append("assistant", "Okay.")
        second = lra.ask("What was my name?")
        assert second["assistant_message_id"] is not None
        assert h.memory("profile:name")["source_message_id"] == first["user_message_id"]
        assert h.count() == 64
    with History(path) as h:
        assert h.count() == 64
        assert h.get(first["user_message_id"])["content"] == "My name is Morgan."


def test_cli_stats_and_unsupported_recall(tmp_path, capsys):
    db = str(tmp_path / "db.sqlite")
    main(["--db", db, "remember", "favorite editor=Code"])
    main(["--db", db, "correct", "favorite editor=Neovim"])
    main(["--db", db, "notes"])
    main(["--db", db, "stats"])
    out = capsys.readouterr().out
    assert "Neovim" in out
    assert "'messages': 2" in out
    with History(db) as h:
        assert len(h.memory_history("favorite editor")) == 2
