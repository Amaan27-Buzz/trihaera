"""Conservative provenance gate for unsupported *personal* exact details.

This is NOT a general factuality checker. It only intercepts specific
private-number, email, and exact-date claims when the assistant introduces a
literal value absent from selected canonical USER statements. It preserves
the raw generation for audit while avoiding self-reinforcing false assistant
memories in the canonical conversation.
"""
import re

PHONE = re.compile(r"(?<![\w])\+?\d[\d\s().-]{6,}\d(?![\w])")
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
DATE = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b")
PRIVATE_CONTACT = re.compile(
    r"\b(?:phone|mobile|telephone|contact\s+number|e-?mail|email\s+address)\b",
    re.I,
)
EXACT_DATE = re.compile(
    r"\b(?:exact\s+date|on\s+what\s+date|which\s+date|what\s+date)\b", re.I
)
PERSONAL = re.compile(r"\b(?:my|mine|i|me|our|we)\b", re.I)


def review_personal_claim(query: str, generated: str, user_sources: list[str]) -> dict:
    """Return served answer plus evidence and explicit override metadata.

    Does not hallucinate a value or infer unprovided information. A user source
    must include the *exact* claimed value; sources from assistant messages
    are intentionally excluded. This simple v0.1 gate does not resolve natural
    language dates, phone number formatting, or cross-message provenance.
    """
    result = {
        "answer": generated, "raw_answer": generated,
        "grounding_override": False, "grounding_reason": None,
    }
    if not PERSONAL.search(query):
        return result
    claims = []
    if PRIVATE_CONTACT.search(query):
        claims.extend(x.group(0).strip() for x in PHONE.finditer(generated))
        claims.extend(x.group(0) for x in EMAIL.finditer(generated))
    if EXACT_DATE.search(query):
        claims.extend(x.group(0) for x in DATE.finditer(generated))
    if not claims:
        return result
    evidence = "\n".join(user_sources).casefold()
    unsupported = [x for x in claims if x.casefold() not in evidence]
    if unsupported:
        result.update({
            "answer": "I can't verify that exact personal detail from our conversation.",
            "grounding_override": True,
            "grounding_reason": "Literal personal identifier/date absent from selected original user messages",
        })
    return result
