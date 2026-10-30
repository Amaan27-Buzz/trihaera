"""Diagnose SmolLM instruction following independently of LRA retrieval.

Runs 5 local generations on SYNTHETIC prompts; no personal chat is opened.
Distinguishes base checkpoint failure, system-prompt failure, historical
repetition and long-range recall. Results are NOT guaranteed successes.

python scripts/diagnose_generation.py --device cuda --offline
"""
import argparse
import json
from pathlib import Path
import tempfile
from lra.model import SmolLM, DEFAULT_MODEL
from lra.app import LRA
from lra.storage import History


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--output", type=Path,
                   default=Path("artifacts/lra_prompt_diagnostics.json"))
    a = p.parse_args()
    model = SmolLM(
        model_id=a.model, device=a.device, offline=a.offline,
        max_new_tokens=64,
    )
    results = []

    def direct(name, messages, expected):
        response = model.generate(messages)
        item = {
            "name": name, "actual": response.text, "expected": expected,
            "literal_match": response.text.strip().casefold().strip(" .!") == expected.casefold(),
            "prompt_tokens": response.prompt_tokens,
            "generation_seconds": response.latency_seconds,
        }
        print(f'{name}: expected={expected!r} generated={response.text!r}')
        results.append(item)

    direct("bare_instruction", [
        {"role": "user", "content": "Say only the word under."}
    ], "under")
    direct("brief_system_instruction", [
        {"role": "system", "content": "Reply directly to the latest request. Do not copy previous responses."},
        {"role": "user", "content": "Say only the word under."},
    ], "under")
    direct("literal_in_context", [
        {"role": "system", "content": "Answer from the supplied note if it contains the answer."},
        {"role": "user", "content": "The fictional project code is silver goose 514."},
        {"role": "assistant", "content": "I understand."},
        {"role": "user", "content": "What is the fictional project code? Give only the code."},
    ], "silver goose 514")
    direct("repetition_stress", [
        {"role": "system", "content": "Answer the most recent user request directly."},
        {"role": "user", "content": "Tell me you understand."},
        {"role": "assistant", "content": "Noted. We can continue our ongoing conversation."},
        {"role": "user", "content": "Tell me you understand."},
        {"role": "assistant", "content": "Noted. We can continue our ongoing conversation."},
        {"role": "user", "content": "Say only the word under."},
    ], "under")
    with tempfile.TemporaryDirectory(prefix="lra-diag-") as directory:
        with History(Path(directory) / "history.sqlite3") as db:
            db.append("user", "My name is Rowan.")
            db.append("assistant", "Nice to meet you.")
            for n in range(30):
                db.append("user", f"Earlier diary entry {n} was about tea and cooking.")
                db.append("assistant", f"We discussed recipes in diary entry {n}.")
            app = LRA(db, model, max_input_tokens=2000)
            res = app.ask("What was the name I told you earlier?")
            rec = {
                "name": "lra_old_fact_recall",
                "actual": res["answer"],
                "expected": "Rowan",
                "literal_match": "rowan" in res["answer"].casefold(),
                "retrieval_ids": res["retrieved_message_ids"],
                "active_memory_keys": res["active_memory_keys"],
                "prompt_tokens": res["input_context_tokens"],
                "generation_seconds": res["generation_seconds"],
            }
            results.append(rec)
            print(f'lra_old_fact_recall: generated={res["answer"]!r}')

    report = {
        "kind": "LRA generation/prompt isolation diagnostic",
        "model": a.model,
        "device": a.device,
        "cases": results,
        "total_pass": sum(item["literal_match"] for item in results),
        "total_cases": len(results),
        "warning": (
            "Use these results to separate base model weakness from prompt "
            "or LRA retrieval problems; this diagnostic does not train a model."
        ),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": report["total_pass"], "total": report["total_cases"],
        "saved": str(a.output),
    }, indent=2))


if __name__ == "__main__":
    main()
