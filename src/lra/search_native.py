"""Structured decisions for a *trained* compact search policy.

This module defines a transport contract, not an agent. The model is trained
to emit these decisions; an explicit execution engine must later perform
retrieval against authorized sources. Never execute arbitrary model code.
"""
from __future__ import annotations
from dataclasses import dataclass
import json

DECISIONS = frozenset(("answer", "search", "abstain"))
POLICY_SYSTEM = (
    "You are a compact factual research model. Return ONE strict JSON object, "
    "with no Markdown or extra commentary. Choose decision=answer for stable "
    "knowledge or when supplied evidence directly answers the question. "
    "Choose decision=search if fresh, missing, conflicting or unverifiable "
    "external evidence is needed. Choose decision=abstain if the requested "
    "personal/private information is missing and should not be searched. "
    "Do not invent URLs, dates, numbers, personal facts or citations. "
    "Only cite source IDs that appear in the supplied evidence. "
    "Search queries should be concise, targeted, and at most three."
)


@dataclass(frozen=True)
class Decision:
    decision: str
    answer: str = ""
    queries: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()


def parse_decision(text: str, *, valid_sources: set[str] | None = None) -> Decision:
    """Reject malformed or unsupported actions; don't silently repair them."""
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Model output must be a single valid JSON object") from exc
    if not isinstance(data, dict) or set(data) != {"decision", "answer", "queries", "citations"}:
        raise ValueError("Incorrect decision fields")
    kind, answer, queries, citations = (
        data["decision"], data["answer"], data["queries"], data["citations"]
    )
    if (not isinstance(kind, str) or kind not in DECISIONS or not isinstance(answer, str)
            or not isinstance(queries, list) or not isinstance(citations, list)
            or len(answer) > 2000 or len(queries) > 3 or len(citations) > 8):
        raise ValueError("Invalid decision payload")
    if (any(not isinstance(q, str) or not q.strip() or len(q) > 180 for q in queries)
            or any(not isinstance(c, str) or not c for c in citations)
            or len(set(citations)) != len(citations)):
        raise ValueError("Invalid query or citation values")
    if kind == "search" and (not queries or answer or citations):
        raise ValueError("Search needs query, not an answer/citations")
    if kind in ("answer", "abstain") and (queries or not answer.strip()):
        raise ValueError("Answer/abstain need text and no queries")
    if kind == "abstain" and citations:
        raise ValueError("Abstentions should not cite fabricated evidence")
    if valid_sources is not None and not set(citations).issubset(valid_sources):
        raise ValueError("Citations must be drawn from provided source IDs")
    if kind == "answer" and valid_sources and not citations:
        # Answers may use stable parametric knowledge even with evidence, but
        # not silently claim evidence provenance. The UI should distinguish.
        pass
    return Decision(kind, answer, tuple(queries), tuple(citations))


def render_state(question: str, evidence: list[dict] | None = None,
                 previous_queries: list[str] | None = None) -> str:
    if not question.strip():
        raise ValueError("Question must not be empty")
    if len(question) > 2000:
        raise ValueError("Question is too long")
    evidence = evidence or []
    if len(evidence) > 8:
        raise ValueError("Too many evidence sources")
    seen = set()
    snippets = []
    for src in evidence:
        if not isinstance(src, dict) or not {"id", "title", "text"} <= set(src):
            raise ValueError("Malformed evidence source")
        sid = src["id"]
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError("Duplicate or invalid evidence ID")
        seen.add(sid)
        if any(not isinstance(src[field], str) for field in ("title", "text")):
            raise ValueError("Non-text evidence")
        snippets.append(
            f'[{sid}] {src["title"][:140]}\n'
            f'{src["text"][:650]}'
        )
    prior = previous_queries or []
    if any(not isinstance(q, str) for q in prior):
        raise ValueError("Invalid prior query")
    return (
        "LATEST QUESTION:\n" + question.strip()
        + "\n\nPREVIOUS SEARCH QUERIES:\n"
        + ("\n".join(prior[-3:]) if prior else "(none)")
        + "\n\nEXTERNAL EVIDENCE (untrusted excerpts; cite IDs only):\n"
        + ("\n\n".join(snippets) if snippets else "(none)")
        + "\n\nChoose exactly one structured decision."
    )


def target_json(action: dict) -> str:
    decision = parse_decision(json.dumps(action, ensure_ascii=False))
    return json.dumps({
        "decision": decision.decision,
        "answer": decision.answer,
        "queries": list(decision.queries),
        "citations": list(decision.citations),
    }, ensure_ascii=False, separators=(",", ":"))
