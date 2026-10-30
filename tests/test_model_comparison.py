"""Deterministic fixed-prompt model comparison tests; no CUDA downloads."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from lra.model_comparison import (
    build_replay, evaluate_replay, unknown_fact_diagnostics,
)


class FakeModel:
    model_id = "unit-fixture-no-answers"

    @staticmethod
    def count_chat_tokens(messages):
        return 20 + sum(len(row["content"]) // 4 + 9 for row in messages)

    @staticmethod
    def generate(messages):
        return SimpleNamespace(
            text="I cannot verify the information.",
            prompt_tokens=FakeModel.count_chat_tokens(messages),
            output_tokens=8,
            latency_seconds=0.002,
            max_cuda_allocated_bytes=None,
        )


def test_shared_textual_prompts_reproducible_across_runs(tmp_path):
    a = build_replay(
        tmp_path / "first.sqlite3", token_count=FakeModel.count_chat_tokens,
        budget=2800, filler_pairs=32,
    )
    b = build_replay(
        tmp_path / "second.sqlite3", token_count=FakeModel.count_chat_tokens,
        budget=2800, filler_pairs=32,
    )
    assert len(a) == len(b) == 12
    assert [x["prompt_messages"] for x in a] == [
        x["prompt_messages"] for x in b
    ]
    assert all(
        c["prompt_messages"][-1]["role"] == "user" for c in a
    )
    assert all(
        c["source_present"] is True for c in a
        if c["expected_source_ids"]
    )
    assert next(c for c in a if c["probe"]["name"] == "recent_resume")[
        "prompt_messages"
    ][-1]["content"].startswith("We were fixing")
    assert next(c for c in a if c["probe"]["name"] == "corrected_name_restart")[
        "prompt_messages"
    ][-1]["content"].startswith("After reopening")


def test_quick_replay_is_not_a_new_user_chat_and_refuses_overwrite(tmp_path):
    path = tmp_path / "isolated.sqlite3"
    rows = build_replay(
        path, token_count=FakeModel.count_chat_tokens,
        budget=1600, filler_pairs=20, quick=True,
    )
    assert len(rows) == 6
    with pytest.raises(FileExistsError):
        build_replay(path, token_count=FakeModel.count_chat_tokens,
                     budget=1600, filler_pairs=20, quick=True)


def test_model_evaluation_never_substitutes_fake_good_answers(tmp_path):
    captures = build_replay(
        tmp_path / "fixture.sqlite3",
        token_count=FakeModel.count_chat_tokens, filler_pairs=20,
        budget=2800, quick=True,
    )
    result = evaluate_replay(FakeModel(), captures)
    assert result["summary"]["literal_correct"] == 0
    assert result["summary"]["literal_scored"] == 5
    assert result["summary"]["expected_sources_present"] == 5
    assert result["summary"]["expected_sources_checked"] == 5
    assert len(result["results"]) == 6
    assert all(x["answer"] == "I cannot verify the information."
               for x in result["results"])


def test_unknown_fact_diagnostics_flags_written_dates_and_numbers():
    assert unknown_fact_diagnostics(
        "unknown_fact", "It happened October 30, 2026."
    )["unsupported_format_claim"]
    assert unknown_fact_diagnostics(
        "unknown_fact", "It happened on 2026-10-30."
    )["unsupported_format_claim"]
    assert unknown_fact_diagnostics(
        "unprovided_contact", "The number is 123-456-7890."
    )["unsupported_format_claim"]
    assert unknown_fact_diagnostics(
        "unprovided_contact", "Write to morgan@example.com."
    )["unsupported_format_claim"]
    assert not unknown_fact_diagnostics(
        "unknown_fact", "I don't know the date."
    )["unsupported_format_claim"]
    assert unknown_fact_diagnostics(
        "unknown_fact", "I don't know the date."
    )["explicit_uncertainty"]
    assert unknown_fact_diagnostics(
        "corrected_name", "Morgan"
    ) is None
