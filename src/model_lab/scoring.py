"""Transparent heuristic scoring. Persist full raw generations for review."""
import json
import re

from .cases import Case


def normal(s: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", s.casefold()))


def assess(case: Case, answer: str) -> dict:
    kind = case.check
    out = {"kind": kind, "verdict": "review", "note": case.note}
    normalized = normal(answer)
    if kind == "exact":
        out["verdict"] = "pass" if normalized == normal(case.expected) else "fail"
    elif kind == "contains":
        out["verdict"] = "pass" if normal(case.expected) in normalized else "fail"
    elif kind == "all":
        out["verdict"] = "pass" if all(normal(x) in normalized for x in case.expected.split("|")) else "fail"
    elif kind == "number":
        nums = re.findall(r"\d+(?:\.\d+)?", answer)
        out["verdict"] = "pass" if nums == [case.expected] else "fail"
    elif kind == "abstain":
        uncertain = bool(re.search(
            r"\b(?:don't|do not|cannot|can't|not provided|not shared|never shared|"
            r"no information|unsure|unknown|haven't been told)\b", answer, re.I
        ))
        invented = bool(re.search(
            r"\b\d{4}-\d\d-\d\d\b|\b\d{3}[- ]\d{3}[- ]\d{4}\b|"
            r"\b\w+@\w+\.\w+\b|\b(?:January|February|March|April|May|June|"
            r"July|August|September|October|November|December) \d{1,2},? \d{4}\b",
            answer, re.I
        ))
        out["verdict"] = "pass" if uncertain and not invented else "fail"
        out["note"] = "Heuristic only; review unknown fact cases manually."
    elif kind == "json":
        try:
            data = json.loads(answer)
            valid = (isinstance(data, dict) and set(data) == {"topic", "examples"}
                     and isinstance(data["topic"], str) and isinstance(data["examples"], list)
                     and len(data["examples"]) == 2
                     and all(isinstance(x, str) for x in data["examples"]))
            out["verdict"] = "pass" if valid else "fail"
        except (TypeError, ValueError):
            out["verdict"] = "fail"
    elif kind != "manual":
        raise ValueError("Invalid check type")
    return out


def summarize(rows: list[dict]) -> dict:
    done = [r for r in rows if r["evaluation"]["verdict"] != "review"]
    success = sum(r["evaluation"]["verdict"] == "pass" for r in done)
    return {
        "total": len(rows), "automated": len(done), "passed": success,
        "failed": len(done) - success,
        "manual_review": len(rows) - len(done),
        "pass_rate": success / len(done) if done else None,
        "caution": "Small scripted test only; not a measure of general ability or hallucination probability.",
    }
