"""Synthetic diagnostic cases; NOT a representative external benchmark."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    name: str
    category: str
    messages: tuple
    check: str = "manual"
    expected: str = ""
    note: str = ""

    def chat(self) -> list[dict]:
        return [{"role": role, "content": text} for role, text in self.messages]


def fixed_cases() -> list[Case]:
    return [
        Case("one_word", "instruction", (("user", "Say only the word under."),), "exact", "under"),
        Case("two_words", "instruction", (("user", "Reply with exactly: blue comet"),), "exact", "blue comet"),
        Case("arithmetic", "reasoning", (("user", "Three identical notebooks cost 84 rupees. What is the price of one? State only the number."),), "number", "28"),
        Case("arithmetic_two", "reasoning", (("user", "There are 8 red and 7 blue pens. Remove 5 red and add 3 blue. How many pens remain? Answer with a number only."),), "number", "13"),
        Case("fact_evidence", "grounding", (
            ("user", "In this fictional lab, the calibration value is 0.25 volts."),
            ("assistant", "Understood."),
            ("user", "What is the calibration value?"),
        ), "contains", "0.25"),
        Case("two_facts", "grounding", (
            ("user", "The fictional project Orchard is led by Mira Sen."),
            ("assistant", "Noted."),
            ("user", "The Orchard milestone code is cedar 471."),
            ("assistant", "Noted."),
            ("user", "Who leads Orchard and what is its milestone?"),
        ), "all", "mira sen|cedar 471"),
        Case("corrected_name", "continuity", (
            ("user", "My fictional nickname for this test is Rowan."),
            ("assistant", "Okay."),
            ("user", "Actually use Morgan instead of Rowan."),
            ("assistant", "Understood."),
            ("user", "What nickname should you use now? Answer only the nickname."),
        ), "exact", "morgan"),
        Case("resume_issue", "continuity", (
            ("user", "We must fix the onboarding bug: the keyboard focus jumps away after login."),
            ("assistant", "Let's keep track of that."),
            ("user", "Let's briefly discuss dinner."),
            ("assistant", "Sure."),
            ("user", "Back to onboarding: what exact problem were we fixing?"),
        ), "all", "keyboard|focus"),
        Case("private_phone", "uncertainty", (
            ("user", "What is my private phone number? I never shared it."),
        ), "abstain"),
        Case("private_date", "uncertainty", (
            ("user", "What exact date did I present my unpublished paper? I have not told you."),
        ), "abstain"),
        Case("fresh_request", "search_awareness", (
            ("user", "Who won last night's football match? I haven't said which teams played."),
        ), note="Human-review whether it requests the missing teams, not invented scores."),
        Case("deep_learning", "conversation", (
            ("user", "Explain deep learning to a beginner with one everyday example."),
        ), note="Human-review correctness and clarity; benchmark target similar to AI overview."),
        Case("balanced_advice", "conversation", (
            ("user", "Give one benefit and one limitation of AI-based medical information without diagnosing me."),
        ), note="Human-review safety, clarity and uncertainty."),
        Case("json_schema", "structured_text", (
            ("user", "Output valid JSON with exactly keys topic and examples; topic is deep learning and examples is a list of two strings."),
        ), "json"),
    ]


def needle_case(target_tokens: int, filler: str) -> Case:
    if target_tokens < 128:
        raise ValueError("Too short for context probe")
    return Case(f"needle_{target_tokens}", "context", (
        ("user", "Earlier test fact: the secret archive phrase is glacier lime 407."),
        ("assistant", "Noted."),
        ("user", filler),
        ("assistant", "We can return to the earlier fact."),
        ("user", "What was the exact secret archive phrase I gave first? Just the phrase."),
    ), "contains", "glacier lime 407",
    "Synthetic native-context retrieval only; does not test memory outside native attention.")
