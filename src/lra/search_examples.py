"""Hand-authored, OFFLINE search-policy bootstrap examples.

These are diagnostic illustrations, NOT a research-quality training corpus.
Never evaluate training examples as held-out successes. A later milestone
requires thousands of deduplicated licensed trajectories with verified sources.
"""
from __future__ import annotations
from .search_native import POLICY_SYSTEM, parse_decision, render_state, target_json

STABLE = [
    ("What is deep learning?", "Deep learning is machine learning using multilayer neural networks."),
    ("Explain what a Python dictionary is.", "A Python dictionary maps keys to values."),
    ("What does RAM stand for?", "RAM stands for random-access memory."),
    ("What is photosynthesis?", "Photosynthesis converts light energy into chemical energy in organisms such as plants."),
    ("What is a CPU?", "A CPU is the central processing unit that executes computer instructions."),
    ("What is a variable in programming?", "A variable is a named reference to a value in a program."),
    ("How does binary search work?", "Binary search halves a sorted search range at each step."),
    ("What is evaporation?", "Evaporation is the change of a liquid into gas from its surface."),
    ("What is an algorithm?", "An algorithm is a finite sequence of instructions for solving a problem."),
    ("Define supervised learning.", "Supervised learning trains on examples with known target labels."),
    ("Explain the difference between RAM and SSD storage.", "RAM is volatile working memory; an SSD provides persistent storage."),
    ("What is HTTP?", "HTTP is the protocol used to exchange web requests and responses."),
    ("What is a neural network?", "A neural network is a model composed of connected computational units."),
    ("What is backpropagation?", "Backpropagation computes gradients of a loss through a neural network."),
]

NEEDS_SEARCH = [
    ("Who won yesterday's local election?", ["official local election results yesterday"]),
    ("What is the latest stable Python release?", ["Python official latest stable release"]),
    ("What is the current price of gold per gram in India?", ["India gold price per gram today"]),
    ("What was published this morning about the new chip?", ["new processor announcement this morning official"]),
    ("What is the current weather in Kochi?", ["Kochi current weather forecast official"]),
    ("Which movie is currently number one on the box office chart?", ["latest box office chart weekend"]),
    ("What is the newest documented version of Rust?", ["rust-lang latest stable release"]),
    ("Who won the Champions League match last night?", ["UEFA Champions League results last night"]),
    ("What is the latest guidance from the university admission office?", ["university admissions official updated guidance"]),
    ("Can you verify this claim from today's news?", ["today's news fact check original reporting"]),
    ("What are the most recent Mars mission findings?", ["NASA Mars mission latest science findings"]),
    ("Is that airport currently delayed?", ["airport official live flight delays"]),
    ("Which new GPUs launched this month?", ["GPU manufacturers new releases this month"]),
    ("What is the live USD to INR exchange rate?", ["USD INR current spot exchange rate"]),
    ("Where did the latest software security advisory originate?", ["vendor official software security advisory latest"]),
    ("Has the company's CEO changed this week?", ["company official CEO announcement this week"]),
    ("What were the final scores of yesterday's match?", ["official match scores yesterday"]),
    ("Is that product still in stock?", ["product availability retailer current"]),
]

EVIDENCE = [
    ("What is the code name of the fictional lab?", "The lab uses code name Blue Lantern.", "Blue Lantern"),
    ("Which engine powers prototype Helio?", "Prototype Helio uses the Orion engine.", "Orion"),
    ("What color was selected for the prototype UI?", "Design notes select moss green for the UI.", "moss green"),
    ("When is the fictional workshop scheduled?", "The lab schedule lists the workshop on 18 April.", "18 April"),
    ("Who is the fictional mission coordinator?", "The mission coordinator is Naomi Vale.", "Naomi Vale"),
    ("What is the fabrication material?", "The technical notes specify recycled aluminum.", "recycled aluminum"),
    ("Which fictional station receives the signal?", "Station Echo receives the next signal.", "Station Echo"),
    ("What is the document's version?", "The specification header says version 2.4.", "2.4"),
    ("Which device failed during the test?", "The log says the optical sensor failed.", "optical sensor"),
    ("Which process produced the sample?", "The sample was produced using cold forging.", "cold forging"),
    ("What is the target calibration value?", "The test target is 0.25 volts.", "0.25 volts"),
    ("Where is the fictional archive stored?", "The archive is stored in the North Vault.", "North Vault"),
    ("Which algorithm did the report recommend?", "The report recommends Dijkstra's algorithm.", "Dijkstra's algorithm"),
    ("Who signed the fictional proposal?", "The document was signed by Elena March.", "Elena March"),
]

PRIVATE = [
    "What's my private phone number? I have never told you.",
    "Tell me my personal bank account number.",
    "What is my secret recovery phrase?",
    "What is the exact date of my private appointment? I didn't share it.",
    "What is the password to my laptop?",
    "What is my colleague's private mobile number?",
    "What was my childhood address? It's not in our conversation.",
    "Which private medical record number belongs to me?",
    "What is the PIN of my bank card?",
    "Tell me the private email address I haven't provided.",
]

CONFLICTS = [
    ("Which of the two conflicting reports is correct?",
     "Report A gives the measurement as 4.5.", "Report B gives the measurement as 9.1.",
     ["original measurement instrument log", "official corrected report"]),
    ("What is the actual publication date?",
     "Blog summary says publication was in June.", "Publisher page says publication was in July.",
     ["publisher official publication date"]),
    ("Which organization owns the project now?",
     "A cached page lists Group East.", "An update lists Group West.",
     ["official ownership change project"]),
    ("What is the definitive component size?",
     "One catalog lists a 12 mm part.", "Another catalog lists 14 mm.",
     ["manufacturer official component datasheet"]),
    ("Was the event held on Friday or Saturday?",
     "A forum post claims Friday.", "A poster says Saturday.",
     ["organizer event official schedule"]),
    ("What is the confirmed quantity produced?",
     "A social post estimates 500 units.", "A news snippet says 800.",
     ["manufacturer production disclosure"]),
]


def examples() -> list[dict]:
    data = []
    def add(kind, idx, question, evidence, target):
        sources = {s["id"] for s in evidence}
        parse_decision(target_json(target), valid_sources=sources)
        data.append({
            "id": f"{kind}_{idx:02d}",
            "kind": kind,
            "question": question,
            "evidence": evidence,
            "previous_queries": [],
            "target": target,
        })

    for i, (question, answer) in enumerate(STABLE):
        add("stable", i, question, [], {
            "decision": "answer", "answer": answer,
            "queries": [], "citations": [],
        })
    for i, (question, queries) in enumerate(NEEDS_SEARCH):
        add("fresh", i, question, [], {
            "decision": "search", "answer": "",
            "queries": queries, "citations": [],
        })
    for i, (question, excerpt, answer) in enumerate(EVIDENCE):
        src = {"id": "s1", "title": "Fictional archived research record",
               "text": excerpt}
        add("evidence", i, question, [src], {
            "decision": "answer", "answer": answer,
            "queries": [], "citations": ["s1"],
        })
    for i, question in enumerate(PRIVATE):
        add("private", i, question, [], {
            "decision": "abstain",
            "answer": "I can't verify that private detail from our conversation.",
            "queries": [], "citations": [],
        })
    for i, (question, left, right, queries) in enumerate(CONFLICTS):
        evidence = [
            {"id": "s1", "title": "Unconfirmed report A", "text": left},
            {"id": "s2", "title": "Unconfirmed report B", "text": right},
        ]
        add("conflict", i, question, evidence, {
            "decision": "search", "answer": "",
            "queries": queries, "citations": [],
        })
    return data


def split_examples(name: str) -> list[dict]:
    """Hold out every fifth example IN EACH CATEGORY, not a random sentence."""
    if name not in ("train", "test"):
        raise ValueError("Split must be train or test")
    data = []
    for example in examples():
        case_idx = int(example["id"].rsplit("_", 1)[1])
        is_test = case_idx % 5 == 0
        if is_test == (name == "test"):
            data.append(example)
    return data


def to_messages(row: dict) -> list[dict]:
    return [
        {"role": "system", "content": POLICY_SYSTEM},
        {"role": "user", "content": render_state(
            row["question"], row["evidence"], row["previous_queries"]
        )},
    ]
