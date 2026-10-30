"""Transparent literal scoring for scripted LRA companion tests.

Evidence retrieval and generated-answer accuracy are distinct measurements.
Ambiguous/unknown questions require manual review; do not assign fake passes.
"""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Probe:
    name: str
    category: str
    prompt: str
    expected_groups: tuple[tuple[str, ...], ...] = ()
    forbidden: tuple[str, ...] = ()
    source_labels: tuple[str, ...] = ()
    requires_review: bool = False


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", text.casefold(), re.UNICODE))


def score_answer(answer: str, probe: Probe) -> dict:
    """Each expected group allows synonyms; all groups must be satisfied."""
    content = normalize(answer)
    met = [
        any(normalize(option) in content for option in alternatives)
        for alternatives in probe.expected_groups
    ]
    forbidden_found = [
        term for term in probe.forbidden if normalize(term) in content
    ]
    verdict = (
        "review" if probe.requires_review else
        "pass" if all(met) and not forbidden_found else "fail"
    )
    return {
        "verdict": verdict,
        "required_groups_met": met,
        "forbidden_found": forbidden_found,
        "review_required": probe.requires_review,
        "note": "Literal checks only; manual review is necessary.",
    }


def probes() -> tuple[Probe, ...]:
    """Cases assess one evolving relationship, not disconnected trivia."""
    return (
        Probe("name_early", "personal_memory", "What name did I introduce myself with?",
              (("Rowan",),), source_labels=("intro",)),
        Probe("style_old", "preference", "What reply style did I tell you I prefer?",
              (("short", "concise", "brief"),), source_labels=("style",)),
        Probe("milestone_old", "long_range", "What milestone did I give for project Copper Finch?",
              (("violet ridge 482",),), source_labels=("project",)),
        Probe("launch_old", "long_range", "What was the launch reference for Copper Finch?",
              (("cedar bridge 719",),), source_labels=("launch",)),
        Probe("blocker_old", "episode", "Why was the Copper Finch project paused?",
              (("recording module", "recording", "startup crash", "crashed"),),
              source_labels=("blocked",)),
        Probe("unknown_fact", "uncertainty",
              "On what exact date did I give the first public presentation for Copper Finch?",
              requires_review=True),
        Probe("corrected_name", "correction",
              "What name should you use for me now?",
              (("Morgan",),), forbidden=("Rowan",), source_labels=("name_correction",)),
        Probe("corrected_name_restart", "restart",
              "After reopening, which name should you use for me?",
              (("Morgan",),), forbidden=("Rowan",), source_labels=("name_correction",)),
        Probe("multi_fact", "multi_hop",
              "What launch reference did Copper Finch have, and what issue paused the project?",
              (("cedar bridge 719",), ("recording", "startup crash", "crashed")),
              source_labels=("launch", "blocked")),
        Probe("recent_resume", "continuity",
              "We were fixing something in the onboarding screen. What was the next thing to fix?",
              (("keyboard focus", "focus jumping", "focus jump"),),
              source_labels=("recent_issue",)),
        Probe("old_vs_current", "correction",
              "I changed my name earlier. What is the current one?",
              (("Morgan",),), forbidden=("Rowan",), source_labels=("name_correction",)),
        Probe("unprovided_contact", "uncertainty",
              "What's my private work phone number? Only answer if I told you.",
              requires_review=True),
    )


def summarize(records: list[dict]) -> dict:
    automated = [r for r in records if r["evaluation"]["verdict"] != "review"]
    passed = sum(r["evaluation"]["verdict"] == "pass" for r in automated)
    recallable = [r for r in records if r["source_message_ids"]]
    all_sources = sum(
        all(mid in r["evidence_source_ids"] for mid in r["source_message_ids"])
        for r in recallable
    )
    return {
        "prompts": len(records),
        "auto_scored": len(automated),
        "auto_passed": passed,
        "auto_failed": len(automated) - passed,
        "literal_score": passed / len(automated) if automated else None,
        "cases_needing_human_review": len(records) - len(automated),
        "cases_with_expected_sources": len(recallable),
        "all_expected_sources_in_prompt": all_sources,
        "source_recall_rate": all_sources / len(recallable) if recallable else None,
        "note": (
            "Literal-answer scores are not companion-quality scores. "
            "Inspect uncertainty cases, source coverage and false positives."
        ),
    }
