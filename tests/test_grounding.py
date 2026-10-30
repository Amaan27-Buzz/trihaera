"""Regression checks: source authority, topic bleed, invented personal facts.

No CUDA, Hugging Face download or trainable model needed.
"""
from types import SimpleNamespace

from lra.app import LRA
from lra.context import build_context
from lra.grounding import review_personal_claim
from lra.storage import History


def tokens(messages):
    return 10 + sum(len(m["content"]) // 4 + 6 for m in messages)


def test_personal_source_search_ignores_self_generated_answer(tmp_path):
    with History(tmp_path / "test.sqlite") as db:
        user_id = db.append("user", "My Copper Finch launch reference is cedar bridge 719.")
        db.append("assistant", "I am guessing your number is 123-456-7890.")
        db.append("assistant", "That phone number is 123-456-7890.")
        hits = db.search("phone number", role="user", limit=5)
        assert all(r["role"] == "user" for r in hits)
        assert not any("123-456-7890" in r["content"] for r in hits)
        assert user_id in [r["id"] for r in db.search(
            "Copper Finch launch", role="user", limit=5,
        )]
        assert db.count() == 3


def test_recall_mode_filters_unrelated_recent_assistant_messages(tmp_path):
    with History(tmp_path / "test.sqlite") as db:
        db.append("user", "We need to fix onboarding screen keyboard focus after login.")
        db.append("assistant", "Yes, we should check the focus handler.")
        db.append("user", "What is our other project's launch reference?")
        db.append("assistant", "The unrelated reference is cedar bridge 719.")
        db.append("user", "We were fixing the onboarding screen. What needs fixing?")
        plan = build_context(
            db, "We were fixing the onboarding screen. What needs fixing?",
            token_count=tokens, budget=1800, recent_messages=6,
        )
        assert plan.messages[-1]["role"] == "user"
        assert any("keyboard focus" in m["content"] for m in plan.messages)
        assert not any(
            "The unrelated reference" in m["content"] for m in plan.messages
        )
        assert db.count() == 5


def test_guard_blocks_unsupported_phone_email_date_without_hiding_raw():
    cases = [
        ("What's my private work phone number?",
         "Your phone number is 123-456-7890."),
        ("What is my email address?",
         "You told me morgan@example.com."),
        ("On what exact date did I give my first presentation?",
         "You presented on 2026-10-30."),
    ]
    for question, raw in cases:
        reviewed = review_personal_claim(question, raw, [])
        assert reviewed["grounding_override"]
        assert reviewed["raw_answer"] == raw
        assert "can't verify" in reviewed["answer"]
        assert raw != reviewed["answer"]


def test_grounding_does_not_block_supported_or_unrelated_values():
    phone = review_personal_claim(
        "What is my phone number?", "Your phone number is +1 202-555-0139.",
        ["My phone number is +1 202-555-0139."],
    )
    assert not phone["grounding_override"]
    date = review_personal_claim(
        "On what exact date did I present?", "You presented on 2026-10-30.",
        ["I presented on 2026-10-30."],
    )
    assert not date["grounding_override"]
    launch = review_personal_claim(
        "What was the launch reference?", "It was cedar bridge 719.", [],
    )
    assert not launch["grounding_override"]


class BadModel:
    window = 1024
    max_new_tokens = 50

    def count_chat_tokens(self, messages):
        return tokens(messages)

    def generate(self, messages):
        return SimpleNamespace(
            text="Your private work phone number is 123-456-7890.",
            latency_seconds=.001,
            output_tokens=12,
            max_cuda_allocated_bytes=None,
        )


def test_served_answer_grounded_and_archive_never_promotes_guess(tmp_path):
    with History(tmp_path / "one.sqlite") as history:
        app = LRA(history, BadModel(), max_input_tokens=700)
        response = app.ask("What's my private work phone number?")
        assert response["grounding_override"]
        assert response["raw_answer"].endswith("123-456-7890.")
        assert "123-456-7890" not in response["answer"]
        assert history.get(response["assistant_message_id"])["content"] == response["answer"]
        assert history.count() == 2
        assert all(
            "123-456-7890" not in message["content"]
            for message in history.messages(limit=2)
        )


def test_recall_preserves_explicit_current_correction(tmp_path):
    with History(tmp_path / "one.sqlite") as history:
        old = history.append("user", "My name is Rowan.")
        history.remember("profile:name", "Rowan", source_message_id=old)
        new = history.append("user", "Actually my name is Morgan.")
        history.remember("profile:name", "Morgan", source_message_id=new)
        for n in range(20):
            history.append("user", f"Diary entry {n} about the garden.")
            history.append("assistant", f"About the garden entry {n}.")
        history.append("user", "I changed my name earlier. What is the current one?")
        plan = build_context(
            history, "I changed my name earlier. What is the current one?",
            token_count=tokens, budget=1600,
        )
        assert "profile:name: Morgan" in plan.messages[0]["content"]
        assert new in plan.memory_source_ids
        assert history.memory("profile:name")["value"] == "Morgan"
        assert history.get(old)["content"] == "My name is Rowan."
