"""Run multi-prompt, multi-session LRA companion evaluation on a local GPU.

Actual SmolLM answers are evaluated separately from evidence retrieval. Uses
ONLY a synthetic, isolated SQLite database; never modifies data/lra.sqlite3.
The model is loaded once and reused over the whole continuing conversation.

Examples:
    python scripts/evaluate_companion.py --device cuda --quick --offline
    python scripts/evaluate_companion.py --device cuda --offline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

from lra.app import LRA
from lra.evaluation import probes, score_answer, summarize
from lra.model import DEFAULT_MODEL, SmolLM
from lra.storage import History


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


def run_benchmark(db_path: Path, model, *, filler_pairs: int = 100,
                  max_input_tokens: int = 2800,
                  recent_messages: int = 8,
                  quick: bool = False,
                  on_progress=None) -> dict:
    if filler_pairs < 20:
        raise ValueError("Need at least 20 filler pairs to exercise long-range recall")
    if db_path.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing database. Use a new --db path."
        )
    emit = on_progress or (lambda _: None)
    selection = tuple(p for p in probes() if not quick or p.name in QUICK)
    report = {
        "kind": "LRA one_conversation_companion_benchmark_v1",
        "model": getattr(model, "model_id", "fake-test-model"),
        "model_window_tokens": model.window,
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": model.max_new_tokens,
        "synthetic_filler_pairs": filler_pairs,
        "filler_generation": "varied synthetic exchanges (not model generated)",
        "scripted_probe_count": len(selection),
        "records": [],
        "memory_snapshots": [],
        "storage_restarted": False,
        "original_source_preserved": None,
        "summary": None,
        "notes": [
            "All prompts, persona details and distractors are synthetic.",
            "The benchmark does not train/fine-tune SmolLM.",
            "Source hits, answer string checks and human review are distinct.",
            "It is a controlled smoke test, not evidence of lifelong companion quality.",
            "The benchmark database is separate from the user's actual one-chat archive.",
        ],
    }
    origins = {}
    started = time.perf_counter()
    probe_by_name = {p.name: p for p in selection}

    def new_app(history):
        return LRA(
            history, model, max_input_tokens=max_input_tokens,
            recent_messages=recent_messages,
        )

    def ask_seed(app, label, prompt):
        result = app.ask(prompt)
        origins[label] = result["user_message_id"]
        emit(f"[seed:{label}] {prompt}")

    def execute(app, history, name):
        case = probe_by_name.get(name)
        if case is None:
            return
        outcome = app.ask(case.prompt)
        original_ids = [origins[key] for key in case.source_labels]
        source_ids = sorted(set(
            outcome["retrieved_message_ids"]
            + outcome.get("context_recent_message_ids", [])
            + outcome.get("active_memory_source_ids", [])
        ))
        record = {
            "name": case.name,
            "category": case.category,
            "question": case.prompt,
            "answer": outcome["answer"],
            "evaluation": score_answer(outcome["answer"], case),
            "source_message_ids": original_ids,
            "evidence_source_ids": source_ids,
            "sources_available": all(mid in source_ids for mid in original_ids),
            "retrieved_message_ids": outcome["retrieved_message_ids"],
            "active_memory_keys": outcome["active_memory_keys"],
            "context_tokens": outcome["input_context_tokens"],
            "context_compacted": outcome["context_compacted"],
            "generation_seconds": outcome["generation_seconds"],
            "generated_tokens": outcome["output_tokens"],
            "peak_cuda_allocated_bytes": outcome["peak_cuda_allocated_bytes"],
        }
        report["records"].append(record)
        emit(
            f'[{case.name}] {record["evaluation"]["verdict"].upper()} | '
            f'evidence={record["sources_available"]} | '
            f'{record["generation_seconds"]:.2f}s | '
            f'answer={outcome["answer"][:140]!r}'
        )

    with History(db_path) as history:
        app = new_app(history)
        for label, prompt in INTRO:
            ask_seed(app, label, prompt)
        # Synthetic history must not teach the model to parrot one stock
        # assistant phrase. Vary both speakers and include actual useful
        # responses. These are clearly synthetic, NOT SmolLM generations.
        themes = (
            ("I walked past the park this afternoon.",
             "The trees along the park path sound pleasant."),
            ("My laptop fan was unusually loud today.",
             "Check temperatures and vents before assuming a hardware fault."),
            ("I tried making pasta for dinner.",
             "You can save the cooking water to loosen the sauce."),
            ("I read a chapter about astronomy.",
             "Was the chapter about stars, planets, or galaxies?"),
            ("I looked through some old photographs.",
             "Would you like to talk about what they reminded you of?"),
            ("The afternoon rain stopped our walk.",
             "You could try again when the weather clears."),
            ("I wrote a short shopping list.",
             "Checking it before leaving might save another trip."),
            ("I watched a documentary about oceans.",
             "Deep-sea ecosystems can be surprisingly varied."),
            ("My houseplants needed watering.",
             "Check the soil before adding more water."),
            ("I reorganized a small shelf.",
             "Keeping frequently used things in reach should help."),
            ("We discussed a bread recipe.",
             "Proofing time depends on dough temperature."),
            ("I repaired a loose drawer handle.",
             "Retightening the screw after some use might help."),
        )
        for i in range(filler_pairs):
            user, reply = themes[i % len(themes)]
            history.append("user", f"Diary entry {i:03d}: {user}")
            history.append("assistant", f"Regarding entry {i:03d}: {reply}")

        for name in (
            "name_early", "style_old", "milestone_old",
            "launch_old", "blocker_old", "unknown_fact",
        ):
            execute(app, history, name)

        ask_seed(app, "name_correction", "Actually my name is Morgan.")
        report["memory_snapshots"].append({
            "checkpoint": "after_correction",
            "active_name": history.memory("profile:name"),
            "name_revisions": history.memory_history("profile:name"),
        })
        execute(app, history, "corrected_name")
        before_restart = history.count()
        report["message_count_before_restart"] = before_restart

    # Construct a new History and LRA over the same canonical database.
    with History(db_path) as history:
        app = new_app(history)
        report["storage_restarted"] = history.count() == before_restart
        report["original_source_preserved"] = all(
            history.get(message_id) is not None for message_id in origins.values()
        )
        execute(app, history, "corrected_name_restart")
        execute(app, history, "multi_fact")
        ask_seed(
            app, "recent_issue",
            "We still need to fix the onboarding screen: keyboard focus "
            "jumps immediately after login.",
        )
        execute(app, history, "recent_resume")
        execute(app, history, "old_vs_current")
        execute(app, history, "unprovided_contact")
        report["final_message_count"] = history.count()
        report["final_memory_count"] = len(history.memories())
        report["source_message_ids"] = origins
        report["summary"] = summarize(report["records"])

    report["total_seconds"] = time.perf_counter() - started
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--quick", action="store_true",
                   help="6 scored prompts instead of 12, plus setup turns")
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--input-budget", type=int, default=2800)
    p.add_argument("--filler-pairs", type=int, default=100)
    p.add_argument("--output", type=Path,
                   default=Path("artifacts/lra_companion_benchmark.json"))
    p.add_argument("--db", type=Path, help="Keep a NEW isolated benchmark DB (optional)")
    args = p.parse_args(argv)
    if args.max_new_tokens < 1 or args.input_budget < 256 or args.filler_pairs < 20:
        p.error("Invalid generation, context or distractor budget")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = SmolLM(
        model_id=args.model, device=args.device,
        max_new_tokens=args.max_new_tokens, offline=args.offline,
    )

    def run(path):
        report = run_benchmark(
            path, model, filler_pairs=args.filler_pairs,
            max_input_tokens=args.input_budget, quick=args.quick,
            on_progress=print,
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n",
                               encoding="utf-8")
        print("\nSummary:", json.dumps(report["summary"], indent=2))
        print("Report:", args.output)
        print("Literal checks can misgrade paraphrases. Inspect all answers.")

    if args.db is not None:
        run(args.db)
    else:
        with tempfile.TemporaryDirectory(prefix="lra-companion-bench-") as temp:
            run(Path(temp) / "benchmark.sqlite3")


if __name__ == "__main__":
    main()
