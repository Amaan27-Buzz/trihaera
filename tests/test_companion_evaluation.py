"""Benchmark tests do not download models or fake passing SmolLM answers."""
import pytest

from lra.evaluation import Probe, normalize, probes, score_answer, summarize
from lra.storage import History
from scripts.evaluate_companion import run_benchmark


class DeliberatelyWrongModel:
    """Simulates model failures; evaluator must report them, not hide them."""
    model_id = "fixture-wrong-answers"
    window = 8192
    max_new_tokens = 80

    def count_chat_tokens(self, messages):
        return 12 + sum(len(m["content"]) // 4 + 8 for m in messages)

    def generate(self, messages):
        class Result:
            text = "I do not know."
            latency_seconds = 0.001
            output_tokens = 5
            max_cuda_allocated_bytes = None
        return Result()


def test_literal_scoring_distinguishes_correct_incorrect_and_manual():
    probe = Probe("x", "recall", "What was it?",
                  (("cedar bridge 719",), ("recording", "microphone")))
    assert score_answer("It was cedar-bridge-719; the recording failed.", probe)["verdict"] == "pass"
    assert score_answer("The code was cedar, no issue", probe)["verdict"] == "fail"
    correction = Probe("y", "correction", "What name?", (("Morgan",),), forbidden=("Rowan",))
    assert score_answer("Morgan, formerly Rowan", correction)["verdict"] == "fail"
    unknown = Probe("z", "unknown", "Unknown?", requires_review=True)
    assert score_answer("Made up answer", unknown)["verdict"] == "review"


def test_case_suite_is_stable_and_covers_companion_behaviors():
    cases = probes()
    assert len(cases) == 12
    assert len({case.name for case in cases}) == len(cases)
    assert {"correction", "restart", "continuity", "long_range", "uncertainty"} <= {
        c.category for c in cases
    }


def test_benchmark_uses_one_archive_restarts_and_scores_actual_failures(tmp_path):
    db_path = tmp_path / "isolated_bench.sqlite3"
    report = run_benchmark(
        db_path, DeliberatelyWrongModel(), filler_pairs=24,
        max_input_tokens=850, quick=True
    )
    assert report["storage_restarted"] is True
    assert report["original_source_preserved"] is True
    assert report["summary"]["prompts"] == 6
    assert report["summary"]["auto_failed"] >= 4
    assert report["summary"]["auto_passed"] == 0
    assert report["summary"]["cases_needing_human_review"] == 1
    assert report["summary"]["cases_with_expected_sources"] >= 4
    assert report["final_message_count"] > 55
    assert len(report["source_message_ids"]) == 7
    assert report["memory_snapshots"][0]["active_name"]["value"] == "Morgan"
    with History(db_path) as history:
        assert history.count() == report["final_message_count"]
        assert len(history.memory_history("profile:name")) == 2
        assert history.memory("profile:name")["value"] == "Morgan"
        assert history.get(report["source_message_ids"]["intro"])["content"] == "My name is Rowan."


def test_existing_benchmark_database_is_never_replaced(tmp_path):
    db = tmp_path / "already.sqlite3"
    db.write_text("user-existing-content", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_benchmark(db, DeliberatelyWrongModel(), filler_pairs=20, quick=True)
    assert db.read_text(encoding="utf-8") == "user-existing-content"


def test_summary_keeps_source_recall_separate_from_answer_quality():
    records = [
        {"evaluation": {"verdict": "fail"}, "source_message_ids": [2],
         "evidence_source_ids": [2]},
        {"evaluation": {"verdict": "pass"}, "source_message_ids": [4],
         "evidence_source_ids": []},
        {"evaluation": {"verdict": "review"}, "source_message_ids": [],
         "evidence_source_ids": []},
    ]
    r = summarize(records)
    assert r["auto_scored"] == 2
    assert r["auto_passed"] == 1
    assert r["source_recall_rate"] == .5
    assert r["cases_needing_human_review"] == 1
