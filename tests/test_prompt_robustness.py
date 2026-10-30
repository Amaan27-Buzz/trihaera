"""LRA prompt-contamination regressions (no model downloads required)."""
from pathlib import Path
from types import SimpleNamespace

from lra.context import BASE_SYSTEM, build_context
from lra.storage import History
from scripts.evaluate_companion import run_benchmark


def tokens(messages):
    return 10 + sum(len(m["content"]) // 4 + 6 for m in messages)


def test_current_user_instruction_stays_last_and_specific(tmp_path):
    with History(tmp_path / "one.sqlite3") as db:
        for i in range(12):
            db.append("user", f"Entry {i} says I walked in the park.")
            db.append("assistant", "Noted. We can continue our ongoing conversation.")
        db.append("user", "Say the word under.")
        plan = build_context(
            db, "Say the word under.", token_count=tokens,
            budget=1800, recent_messages=10,
        )
        assert plan.messages[-1] == {
            "role": "user", "content": "Say the word under."
        }
        assert "Answer the latest user message directly" in BASE_SYSTEM
        # Prompt selection suppresses repeated identical assistant boilerplate
        assert sum(
            m["content"] == "Noted. We can continue our ongoing conversation."
            for m in plan.messages[1:]
        ) <= 1
        assert db.count() == 25  # Nothing deleted from canonical history.


def test_history_search_filters_duplicate_stock_responses(tmp_path):
    with History(tmp_path / "many.sqlite3") as db:
        for i in range(30):
            db.append("assistant", "Noted. We can continue our ongoing conversation.")
        db.append("user", "I noted the project code was lantern 987.")
        found = db.search("noted project code", limit=8)
        assert len([m for m in found if m["role"] == "assistant"]) <= 1
        assert any("lantern 987" in m["content"] for m in found)
        assert db.count() == 31


class Fake:
    model_id = "fake-diagnostic"
    window = 8192
    max_new_tokens = 80

    def count_chat_tokens(self, messages):
        return tokens(messages)

    def generate(self, messages):
        return SimpleNamespace(
            text="Intentionally not answering.", latency_seconds=.001,
            output_tokens=3, max_cuda_allocated_bytes=None,
        )


def test_revised_benchmark_distractors_not_duplicated(tmp_path):
    path = tmp_path / "isolated.sqlite3"
    r = run_benchmark(
        path, Fake(), filler_pairs=24, max_input_tokens=1100,
        quick=True,
    )
    assert r["summary"]["auto_passed"] == 0
    assert r["storage_restarted"]
    assert r["original_source_preserved"]
    with History(path) as db:
        filler = db.db.execute(
            "SELECT content FROM messages WHERE role='assistant' "
            "AND content LIKE 'Regarding entry %' ORDER BY id"
        ).fetchall()
        assert len(filler) == 24
        assert len({row[0] for row in filler}) == 24
