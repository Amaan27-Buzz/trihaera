"""Compare SmolLM2-360M vs 1.7B on the SAME LRA prompt snapshots.

This is a controlled model-capacity test: fixture dialogue and context are
frozen BEFORE any generation, so neither model can change what the other
sees. Does not train models or modify the real personal chat database.

Windows:
    python scripts/compare_smollm.py --device cuda --quick
    python scripts/compare_smollm.py --device cuda
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import tempfile
from datetime import datetime, timezone

from lra.model import SmolLM
from lra.model_comparison import build_replay, evaluate_replay

SMALL = "HuggingFaceTB/SmolLM2-360M-Instruct"
LARGE = "HuggingFaceTB/SmolLM2-1.7B-Instruct"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--models", default=f"{SMALL},{LARGE}")
    parser.add_argument("--quick", action="store_true",
                        help="6 prompts instead of 12, with same 100 filler pairs")
    parser.add_argument("--offline", action="store_true",
                        help="Require BOTH checkpoints already in HF cache")
    parser.add_argument("--filler-pairs", type=int, default=100)
    parser.add_argument("--input-budget", type=int, default=2800)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/lra_smol_comparison.json"))
    args = parser.parse_args(argv)
    model_ids = [x.strip() for x in args.models.split(",") if x.strip()]
    if not model_ids or len(model_ids) > 2 or len(set(model_ids)) != len(model_ids):
        parser.error("Choose one or two distinct model IDs")
    if args.input_budget < 256 or args.filler_pairs < 20 or args.max_new_tokens < 1:
        parser.error("Invalid replay/generation size")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    import torch
    with tempfile.TemporaryDirectory(prefix="lra-capacity-test-") as folder:
        captures = None
        report = {
            "kind": "LRA fixed-prompt SmolLM capacity comparison",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "models": model_ids,
            "device": args.device,
            "filler_pairs": args.filler_pairs,
            "quick": args.quick,
            "reference_context_model": model_ids[0],
            "runs": {},
            "limitations": [
                "Identical TEXTUAL prompt snapshots across models, not an interactive run.",
                "Only one synthetic conversation fixture and one generation per question.",
                "Model tokenizers may count the same text differently.",
                "Literal matching can miss valid paraphrases and falsely credit copied text.",
                "Unknown personal fact flags are format detectors, not fact-checking.",
                "No fine-tuning, no comparison of training quality, no native long context.",
            ],
        }
        for index, model_id in enumerate(model_ids):
            print(f"Loading {model_id} (first-time Hub download if not cached)...", flush=True)
            model = SmolLM(
                model_id=model_id, device=args.device,
                max_new_tokens=args.max_new_tokens, offline=args.offline,
            )
            if captures is None:
                db = Path(folder) / "one_continuous_fixture.sqlite3"
                captures = build_replay(
                    db, token_count=model.count_chat_tokens,
                    budget=args.input_budget, filler_pairs=args.filler_pairs,
                    quick=args.quick,
                )
                serial = json.dumps(
                    [r["prompt_messages"] for r in captures],
                    ensure_ascii=False, sort_keys=True,
                )
                report["prompt_snapshots_sha256"] = hashlib.sha256(
                    serial.encode("utf-8")
                ).hexdigest()
                report["probe_count"] = len(captures)
                print(f"Fixed {len(captures)} prompt snapshots; "
                      f"SHA256={report['prompt_snapshots_sha256'][:16]}...", flush=True)
            # Explicitly fail rather than silently truncate on either model.
            for r in captures:
                tokens = model.count_chat_tokens(r["prompt_messages"])
                if tokens + model.max_new_tokens > model.window:
                    raise ValueError(
                        f"Prompt too long for {model_id} ({tokens} tokens)"
                    )
            outcome = evaluate_replay(model, captures)
            report["runs"][model_id] = outcome
            args.output.write_text(json.dumps(report, indent=2) + "\n",
                                   encoding="utf-8")
            for r in outcome["results"]:
                mark = r["evaluation"]["verdict"].upper()
                warning = r["unknown_claim_check"]
                flag = " UNSUPPORTED_FORMAT" if warning and warning["unsupported_format_claim"] else ""
                print(f'  {r["name"]}: {mark}{flag} | '
                      f'{r["generation_seconds"]:.2f}s | '
                      f'{r["answer"][:135]!r}', flush=True)
            print(f"Summary for {model_id}: "
                  f"{json.dumps(outcome['summary'], sort_keys=True)}", flush=True)
            del model
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()

        if len(model_ids) == 2:
            a = report["runs"][model_ids[0]]["summary"]
            b = report["runs"][model_ids[1]]["summary"]
            report["comparison"] = {
                "second_minus_first_literal_correct": (
                    b["literal_correct"] - a["literal_correct"]
                ),
                "first": model_ids[0],
                "second": model_ids[1],
                "warning": "Single deterministic fixture; no generality claim",
            }
            args.output.write_text(json.dumps(report, indent=2) + "\n",
                                   encoding="utf-8")
        print(f"Results written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
