"""Actual CUDA LRA smoke: old fact retrieval, one chat turn, restart integrity.

Runs SmolLM2-360M-Instruct on the user's GPU. No training occurs.
The model's answer is recorded separately from whether indexed retrieval
returned the original evidence. This avoids hiding model recall failures.
"""
import argparse
import json
from pathlib import Path
import tempfile
import time

from lra.app import LRA
from lra.storage import History
from lra.model import SmolLM, DEFAULT_MODEL


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--output", type=Path, default=Path("artifacts/lra_gpu_smoke.json"))
    a = p.parse_args()
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="lra-") as directory:
        path = Path(directory) / "single_chat.sqlite3"
        with History(path) as db:
            old = db.append(
                "user", "For the blue bird experiment, the archival secret "
                        "code is amethyst lantern 731."
            )
            for n in range(160):
                db.append("user", f"Routine historical note {n} about coffee and books.")
                db.append("assistant", "Historical note recorded.")
            count_before = db.count()
            model = SmolLM(
                model_id=a.model, device=a.device,
                max_new_tokens=a.max_new_tokens, offline=a.offline
            )
            app = LRA(db, model, max_input_tokens=2800, recent_messages=8)
            result = app.ask(
                "What was the archival secret code for the blue bird experiment?"
            )
            hit = old in result["retrieved_message_ids"]
            expected = "amethyst lantern 731"
            answer_hit = expected in result["answer"].casefold()
            if not hit:
                raise RuntimeError(
                    "Archive retrieval missed original source; inspect FTS/ranker"
                )
            assert db.get(old)["content"].endswith(expected + ".")
        with History(path) as reopened:
            persisted = reopened.count() == count_before + 2
            archived = reopened.get(old) is not None
            if not (persisted and archived):
                raise RuntimeError("Canonical history lost after SQLite restart")
    report = {
        "test": "LRA local GPU end-to-end, no training",
        "model": a.model, "device": a.device,
        "archived_messages_before_query": count_before,
        "retrieved_old_message_id": old,
        "retrieval_hit": hit,
        "exact_expected_answer_found_in_generation": answer_hit,
        "generated_text": result["answer"],
        "token_budget_used": result["input_context_tokens"],
        "model_output_tokens": result["output_tokens"],
        "generation_seconds": result["generation_seconds"],
        "peak_cuda_allocated_bytes": result["peak_cuda_allocated_bytes"],
        "restart_preserves_original_history": persisted and archived,
        "total_seconds_including_model_load": time.perf_counter() - start,
        "warning": (
            "This verifies retrieval+model inference on a seeded archive, "
            "not lifelong memory intelligence, knowledge learning, or training. "
            "Generation accuracy is reported separately from retrieval success."
        )
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
