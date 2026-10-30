"""Capacity-isolation benchmark: replay IDENTICAL LRA prompt text to models.

Unlike the interactive companion benchmark, assistant responses here are
scripted, not generated. Thus model A cannot change the history seen by
model B. This is a CONTROL for model capability, not a replacement for
end-to-end companion evaluation.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re

from .context import build_context
from .evaluation import probes, score_answer
from .memory import extract_explicit_facts
from .storage import History


INTRO = (
    ("intro", "My name is Rowan."),
    ("style", "I prefer short, concise technical answers."),
    ("project", "I'm working on a project called Copper Finch. Its milestone is violet ridge 482."),
    ("launch", "The launch reference for Copper Finch is cedar bridge 719."),
    ("blocked", "Copper Finch was paused because the recording module crashed on startup."),
)
QUICK = {
    "name_early", "milestone_old", "corrected_name",
    "corrected_name_restart", "recent_resume", "unprovided_contact",
}
_MONTH_DATE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},?\s+\d{4}\b", re.I,
)
_DAY_MONTH = re.compile(
    r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|"
    r"August|September|October|November|December)\s+\d{4}\b", re.I,
)
_ISO_DATE = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b")
_PHONE = re.compile(r"\b(?:\+?\d[\d .()-]{7,}\d)\b")
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_ABSTAIN = re.compile(
    r"\b(?:don't know|do not know|can't verify|cannot verify|"
    r"not (?:provided|mentioned|specified|given)|"
    r"never (?:told|shared|provided)|no (?:record|information))\b", re.I,
)


def unknown_fact_diagnostics(name: str, answer: str) -> dict | None:
    """Known-absent specifics in synthetic fixtures, independent of word score."""
    if name == "unknown_fact":
        suspicious = any(p.search(answer) for p in (_MONTH_DATE, _DAY_MONTH, _ISO_DATE))
    elif name == "unprovided_contact":
        suspicious = bool(_PHONE.search(answer) or _EMAIL.search(answer))
    else:
        return None
    return {
        "unsupported_format_claim": suspicious,
        "explicit_uncertainty": bool(_ABSTAIN.search(answer)),
        "human_review_required": True,
        "warning": (
            "Format detector only. Does not prove factuality; "
            "some paraphrases/number formats are missed."
        ),
    }


def build_replay(db_path: Path, *, token_count, budget: int = 2800,
                 filler_pairs: int = 100, quick: bool = False) -> list[dict]:
    """One chronological synthetic timeline, same model-independent snapshots.

    The database path must NOT exist. User's actual LRA database is untouched.
    """
    if db_path.exists():
        raise FileExistsError("Replay database already exists; refusing overwrite")
    if filler_pairs < 20 or budget < 256:
        raise ValueError("Invalid replay history/context budget")
    selected = {p.name:p for p in probes() if not quick or p.name in QUICK}
    source_ids = {}
    captures = []

    def append_source(db, label, statement, reply):
        mid = db.append("user", statement)
        for key, value in extract_explicit_facts(statement):
            db.remember(key, value, source_message_id=mid)
        source_ids[label] = mid
        db.append("assistant", reply)

    def capture(db, name):
        if name not in selected:
            return
        p = selected[name]
        db.append("user", p.prompt)
        # Freeze only SYNTHETIC fixture timestamps so independent runs create
        # byte-for-byte comparable prompt snapshots. Message ordering remains
        # recorded by the canonical incremental IDs.
        db.db.execute(
            "UPDATE messages SET created_at = "
            "strftime('%Y-%m-%dT%H:%M:%SZ', '2025-01-01T00:00:00Z', "
            "'+' || id || ' seconds')"
        )
        db.db.commit()
        plan = build_context(
            db, p.prompt, token_count=token_count, budget=budget,
            recent_messages=8, recall=8,
        )
        ids = {int(x) for x in plan.retrieved_ids + plan.recent_ids + plan.memory_source_ids}
        expected_ids = [source_ids[x] for x in p.source_labels]
        captures.append({
            "probe": asdict(p),
            "prompt_messages": plan.messages,
            "context_tokens_reference": plan.input_tokens,
            "expected_source_ids": expected_ids,
            "source_present": (all(x in ids for x in expected_ids)
                               if expected_ids else None),
            "retrieved_user_message_ids": list(plan.retrieved_ids),
            "memory_source_ids": list(plan.memory_source_ids),
        })
        # This neutral, deterministic exchange belongs to both models'
        # timeline. It deliberately contains no answer to the probe.
        db.append("assistant", "We can discuss the next question.")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with History(db_path) as db:
        replies = (
            "Thanks for telling me.",
            "I'll keep that detail in mind.",
            "We can continue discussing it.",
            "That's useful context.",
            "I understand the situation.",
        )
        for i, (label, text) in enumerate(INTRO):
            append_source(db, label, text, replies[i])

        filler = (
            ("I read about plants.", "The article sounded interesting."),
            ("I cooked some rice.", "How did the meal turn out?"),
            ("I visited a local museum.", "Which exhibit stood out?"),
            ("I took a walk in the evening.", "How was the walk?"),
            ("I practiced a guitar chord.", "Practice can improve consistency."),
            ("I updated my reading list.", "It's useful to keep a record."),
            ("The bus was late today.", "Unexpected delays can be frustrating."),
            ("I saw a bird by the window.", "Did you recognize the species?"),
        )
        for i in range(filler_pairs):
            user, assistant = filler[i % len(filler)]
            db.append("user", f"Diary {i:03d}: {user}")
            db.append("assistant", assistant)

        for name in (
            "name_early", "style_old", "milestone_old",
            "launch_old", "blocker_old", "unknown_fact",
        ):
            capture(db, name)

        append_source(
            db, "name_correction", "Actually my name is Morgan.",
            "Thanks for the correction.",
        )
        capture(db, "corrected_name")

    # Genuine SQLite reopen; exact same historical text for every model.
    with History(db_path) as db:
        capture(db, "corrected_name_restart")
        capture(db, "multi_fact")
        append_source(
            db, "recent_issue",
            "We still need to fix the onboarding screen: keyboard focus "
            "jumps immediately after login.",
            "Let's revisit the keyboard focus behavior.",
        )
        capture(db, "recent_resume")
        capture(db, "old_vs_current")
        capture(db, "unprovided_contact")

    return captures


def evaluate_replay(model, captures: list[dict]) -> dict:
    """Use model.generate, never update canonical history with model outputs."""
    results = []
    for row in captures:
        p = row["probe"]
        generated = model.generate(row["prompt_messages"])
        verdict = score_answer(
            generated.text,
            next(case for case in probes() if case.name == p["name"]),
        )
        results.append({
            "name": p["name"],
            "category": p["category"],
            "answer": generated.text,
            "evaluation": verdict,
            "unknown_claim_check": unknown_fact_diagnostics(p["name"], generated.text),
            "source_present": row["source_present"],
            "context_tokens_reference": row["context_tokens_reference"],
            "input_tokens": generated.prompt_tokens,
            "output_tokens": generated.output_tokens,
            "generation_seconds": generated.latency_seconds,
            "peak_cuda_allocated_bytes": generated.max_cuda_allocated_bytes,
        })
    scored = [r for r in results if r["evaluation"]["verdict"] != "review"]
    unknown = [r["unknown_claim_check"] for r in results if r["unknown_claim_check"]]
    return {
        "model": model.model_id,
        "results": results,
        "summary": {
            "literal_correct": sum(r["evaluation"]["verdict"] == "pass" for r in scored),
            "literal_scored": len(scored),
            "expected_sources_present": sum(r["source_present"] is True for r in results),
            "expected_sources_checked": sum(r["source_present"] is not None for r in results),
            "unknown_fact_format_flags": sum(u["unsupported_format_claim"] for u in unknown),
            "unknown_fact_uncertainty_phrases": sum(u["explicit_uncertainty"] for u in unknown),
            "median_generation_seconds": sorted(r["generation_seconds"] for r in results)[len(results)//2] if results else None,
            "max_cuda_allocated_bytes": max(
                (r["peak_cuda_allocated_bytes"] or 0 for r in results),
                default=None,
            ),
        },
    }
