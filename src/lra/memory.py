"""Small, deliberately conservative source-linked memory extraction.

No unsupported inference. Every memory refers to the original user message;
corrections supersede notes while leaving the underlying history intact.
"""
import re


_PATTERNS = [
    ("profile:name", re.compile(r"\b(?:actually\s+)?my name is\s+([^.!?\n]{2,120})", re.I)),
    ("profile:project", re.compile(r"\b(?:i am|i'm) working on\s+([^.!?\n]{3,160})", re.I)),
    ("preference:style", re.compile(r"\bi prefer\s+([^.!?\n]{3,160})", re.I)),
    ("preference:avoid", re.compile(r"\bi (?:do not|don't) like\s+([^.!?\n]{3,160})", re.I)),
    ("profile:tools", re.compile(r"\bi use\s+([^.!?\n]{2,160})", re.I)),
]


def extract_explicit_facts(text: str) -> list[tuple[str, str]]:
    """Capture directly expressed facts only, without LLM speculation."""
    facts = []
    for key, pattern in _PATTERNS:
        match = pattern.search(text)
        if match:
            value = match.group(1).strip(" \t\"'")
            if value:
                facts.append((key, value))
    return facts
