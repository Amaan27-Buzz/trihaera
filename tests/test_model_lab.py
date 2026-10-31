"""CPU-only tests for honest baseline design and heuristic evaluation."""
import json

import pytest

from model_lab.catalog import MODELS, resolve_model
from model_lab.cases import Case, fixed_cases, needle_case
from model_lab.scoring import assess, summarize


def test_catalog_uses_official_repositories_and_different_loaders():
    assert len(MODELS) == 2
    assert resolve_model("smol").loader == "causal"
    assert resolve_model("qwen").loader == "multimodal"
    assert resolve_model("qwen").advertised_context == 262144
    with pytest.raises(ValueError):
        resolve_model("other")


def test_case_list_is_not_a_fake_perfect_score():
    rows = fixed_cases()
    assert len(rows) == 14
    assert len({x.name for x in rows}) == len(rows)
    assert sum(x.check == "manual" for x in rows) >= 3
    assert {x.category for x in rows} >= {
        "instruction", "reasoning", "continuity", "uncertainty", "grounding",
    }
    for row in rows:
        assert row.chat()[-1]["role"] == "user"
        assert all(role in {"user", "assistant"} for role, _ in row.messages)


@pytest.mark.parametrize("case_id,right,wrong", [
    ("one_word", "Under.", "I'm an assistant."),
    ("two_words", "blue comet", "blue comet and stars"),
    ("arithmetic", "28", "28 or 30"),
    ("fact_evidence", "0.25 volts", "0.35 volts"),
    ("two_facts", "Mira Sen; cedar 471", "Mira Sen"),
    ("corrected_name", "Morgan.", "Rowan"),
    ("resume_issue", "keyboard focus jumps", "onboarding needs improvement"),
])
def test_automated_cases_record_actual_failures(case_id, right, wrong):
    case = next(row for row in fixed_cases() if row.name == case_id)
    assert assess(case, right)["verdict"] == "pass"
    assert assess(case, wrong)["verdict"] == "fail"


def test_unknown_personal_facts_flag_fabrications():
    phone = next(row for row in fixed_cases() if row.name == "private_phone")
    assert assess(phone, "You never gave me that phone number.")["verdict"] == "pass"
    assert assess(phone, "Your number is 123-456-7890.")["verdict"] == "fail"
    date = next(row for row in fixed_cases() if row.name == "private_date")
    assert assess(date, "I don't know the exact date.")["verdict"] == "pass"
    assert assess(date, "The date was October 30, 2026.")["verdict"] == "fail"


def test_structured_json_requires_complete_valid_object():
    obj = next(row for row in fixed_cases() if row.name == "json_schema")
    valid = json.dumps({"topic": "deep learning", "examples": ["vision", "audio"]})
    assert assess(obj, valid)["verdict"] == "pass"
    assert assess(obj, '{"topic":"deep learning","examples":[]}')["verdict"] == "fail"
    assert assess(obj, "something like JSON")["verdict"] == "fail"


def test_manual_cases_not_counted_as_correct():
    case = next(row for row in fixed_cases() if row.name == "deep_learning")
    evaluation = assess(case, "A plausible sounding but false answer")
    assert evaluation["verdict"] == "review"
    results = [{"evaluation": evaluation}, {"evaluation": {"verdict": "fail"}}]
    summary = summarize(results)
    assert summary["total"] == 2
    assert summary["manual_review"] == 1
    assert summary["passed"] == 0
    assert summary["pass_rate"] == 0


def test_context_probes_are_explicitly_native_only():
    with pytest.raises(ValueError):
        needle_case(90, "short")
    row = needle_case(512, "Routine history")
    assert row.category == "context"
    assert row.chat()[0]["role"] == "user"
    assert row.chat()[-1]["role"] == "user"
    assert "glacier lime 407" in row.chat()[0]["content"]
    assert "native-context" in row.note
