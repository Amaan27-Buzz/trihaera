"""Bounded prompt builder over one unbounded/recoverable local history.

Compaction here means selecting evidence for each model call: not deleting,
rewriting, or pretending that all historical tokens fit in the model's KV cache.
"""
from dataclasses import dataclass
import re

from .storage import History

# These are conservative routing hints, not a replacement for semantic recall.
_RECALL = re.compile(
    r"\b(?:remember|earlier|previously|before|introduced|reopen(?:ing|ed)?|"
    r"what\s+(?:name|did|was|were|is|reply)|which\s+name|"
    r"we\s+were|first\s+public|private\s+work)\b", re.I,
)
_ASSISTANT_REFERENCE = re.compile(
    r"\b(?:you\s+(?:said|told|suggested|recommended)|your\s+(?:answer|reply|suggestion))\b",
    re.I,
)
_STOP = frozenset((
    "the", "this", "that", "was", "were", "what", "when", "which",
    "name", "tell", "told", "please", "have", "been", "with",
    "from", "then", "before", "after", "about", "should", "could",
    "would", "your", "mine", "ours", "have", "there", "thing", "something",
    "did", "you", "for", "our", "are", "and", "now", "earlier",
))


def _focus_terms(value: str) -> set[str]:
    return {
        w for w in re.findall(r"[^\W_]+", value.casefold(), re.UNICODE)
        if len(w) > 2 and w not in _STOP
    }



BASE_SYSTEM = (
    "You are LRA, a local AI companion having ONE ongoing conversation "
    "with the user. Answer the latest user message directly, naturally and "
    "concisely. Follow simple direct requests literally: for example, if "
    "asked to say a word, say that word, not a generic acknowledgement. "
    "Never repeat old stock assistant phrases instead of answering. "
    "If asked who you are, explain that you are LRA powered by a local "
    "small language model; do not invent who personally created you. "
    "Older chat excerpts are background evidence, NOT requests to answer now. "
    "Your own earlier assistant replies are not evidence that the user "
    "provided a personal fact. Do not copy unrelated nearby topic details. "
    "Use explicitly sourced memories to recall the user's past statements. "
    "When records conflict, a newer explicit correction supersedes older ones. "
    "Never invent past conversations, private details or missing dates; "
    "state uncertainty clearly when the record has no answer."
)


@dataclass(frozen=True)
class ContextPlan:
    messages: list[dict]
    retrieved_ids: tuple[int, ...]
    memory_keys: tuple[str, ...]
    recent_ids: tuple[int, ...]
    memory_source_ids: tuple[int, ...]
    input_tokens: int
    truncated: bool


def build_context(history: History, query: str, *, token_count,
                  budget: int = 3000, recent_messages: int = 10,
                  recall: int = 8, embedder=None) -> ContextPlan:
    """Assemble current prompt; preserve original history regardless of budget.

    token_count accepts an ordered chat-message list, not raw strings. The
    newest user message is always included and never silently truncated.
    """
    if budget < 128 or recent_messages < 1 or recall < 0:
        raise ValueError("Invalid context budgets")
    current = history.messages(limit=1)
    if not current or current[-1]["role"] != "user":
        raise ValueError("Latest canonical message must be the user request")
    newest = current[0]
    recent = history.messages(limit=recent_messages + 1)[:-1]
    # Repetitive assistant boilerplate can trap a small decoder into echoing
    # it. Keep canonical history intact; deduplicate only prompt selection.
    recent_filtered = []
    seen_assistant = set()
    for item in reversed(recent):
        if item["role"] == "assistant":
            fingerprint = " ".join(item["content"].casefold().split())
            if fingerprint in seen_assistant:
                continue
            seen_assistant.add(fingerprint)
        recent_filtered.append(item)
    recent = list(reversed(recent_filtered))
    # Recall-specific questions shouldn't copy unrelated nearby answers as
    # evidence. In ordinary conversation keep the full recent dialogue.
    # Retrieval from original user messages remains available for distant facts.
    recall_mode = bool(_RECALL.search(query))
    assistant_recall = bool(_ASSISTANT_REFERENCE.search(query))
    if recall_mode and not assistant_recall:
        focus = _focus_terms(query)
        if focus:
            recent = [
                row for row in recent if row["role"] == "user"
                and bool(_focus_terms(row["content"]) & focus)
            ]
    recent_ids = {r["id"] for r in recent} | {newest["id"]}
    evidence = history.search(
        query, limit=recall, exclude_ids=recent_ids, embedder=embedder,
        role=None if assistant_recall else "user",
    ) if recall else []
    mems = history.memories(limit=30)

    def encode(memories, retrieved, context):
        header = BASE_SYSTEM
        if memories:
            header += "\n\nACTIVE SOURCE-LINKED FACTS (newer corrections supersede earlier):\n"
            header += "\n".join(
                f'- {m["key"]}: {m["value"]} [source message #{m["source_message_id"]}]'
                for m in memories
            )
        if retrieved:
            header += "\n\nRETRIEVED ORIGINAL HISTORY EXCERPTS (read only as evidence):\n"
            header += "\n".join(
                f'[message #{m["id"]} | {m["created_at"]} | {m["role"]}] '
                + repr(m["content"][:1800])
                for m in retrieved
            )
        msgs = [{"role": "system", "content": header}]
        msgs += [{"role": m["role"], "content": m["content"]} for m in context]
        msgs.append({"role": "user", "content": newest["content"]})
        return msgs

    # Prefer preserving latest turns, then evidence, then notes. If historical
    # items don't fit, drop old context until the real tokenizer confirms fit.
    dropped = False
    included = list(recent)
    retrieved = list(evidence)
    memories = list(mems)
    while True:
        prompt = encode(memories, retrieved, included)
        tokens = int(token_count(prompt))
        if tokens <= budget:
            break
        dropped = True
        if included:
            included.pop(0)
        elif retrieved:
            retrieved.pop()
        elif memories:
            memories.pop()
        else:
            raise ValueError(
                "Newest user message alone exceeds model context budget; "
                "do not truncate it silently."
            )
    return ContextPlan(
        messages=prompt,
        retrieved_ids=tuple(m["id"] for m in retrieved),
        memory_keys=tuple(m["key"] for m in memories),
        recent_ids=tuple(m["id"] for m in included),
        memory_source_ids=tuple(m["source_message_id"] for m in memories),
        input_tokens=tokens,
        truncated=dropped,
    )
