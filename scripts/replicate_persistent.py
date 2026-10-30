"""Compare depth-persistent sidecars on identical held-out WikiText-2 text.

All tested sidecars read frozen SmolLM2 hidden states. Equal token exposures
and trainable parameter counts are recorded, but these comparisons are NOT
equal FLOP or wall-clock training budgets. Do not treat as architecture proof.

PYTHONPATH=src python scripts/replicate_persistent.py --device cuda
"""

import argparse
import json
import math
from pathlib import Path
import random

import torch

from trihaera.experiment_stats import paired_summary
from trihaera.persistent import (
    PersistentStateAdapterLM, persistent_parameters, persistent_parameter_report,
)
from trihaera.pilot_data import pack_text_windows
from pilot_pretrained import evaluate, train_adapters
from replicate_pretrained import git_sha, parse_csv

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
MODES = (
    "mixed", "no_mixing", "real_equivalent",
    "real_matched_orthogonal", "real_matched_orthogonal_no_mixing",
    "real_equivalent_no_mixing", "real_six", "real_six_no_mixing",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", default="17,23,42")
    parser.add_argument("--modes", default="mixed,no_mixing,real_equivalent,real_matched_orthogonal")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--train-windows", type=int, default=512)
    parser.add_argument("--eval-windows", type=int, default=64)
    parser.add_argument("--train-rows", type=int, default=9000)
    parser.add_argument("--eval-rows", type=int, default=3000)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--data-seed", type=int, default=139)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/persistent_replication_135m.json"))
    args = parser.parse_args()

    try:
        seed_keys = parse_csv(args.seeds)
        modes = parse_csv(args.modes, set(MODES))
        seeds = [int(seed) for seed in seed_keys]
    except ValueError as exc:
        parser.error(str(exc))
    if (any(seed < 0 for seed in seeds) or "no_mixing" not in modes or
        args.steps < 1 or args.batch_size < 1 or args.rank < 1 or
        args.train_windows < 2 or args.eval_windows < 2 or
        args.train_rows < 1 or args.eval_rows < 1 or
        not 8 <= args.sequence_length <= 4096 or
        not 0 < args.residual_scale <= 1 or
        not 0 < args.learning_rate <= 0.01 or
        not 0 < args.clip_norm <= 100 or
        not all(math.isfinite(v) for v in
                (args.learning_rate, args.clip_norm, args.residual_scale))):
        parser.error("Invalid replication parameters")
    if args.output.suffix.lower() != ".json":
        parser.error("Output path must end in .json")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    try:
        import datasets
        import transformers
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing transformers/datasets/safetensors in TRIHAERA's .venv"
        ) from exc

    print("Loading WikiText-2 official train and validation splits...", flush=True)
    train_data = load_dataset(DATASET_ID, DATASET_CONFIG, split="train")
    validation_data = load_dataset(DATASET_ID, DATASET_CONFIG, split="validation")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False, local_files_only=True)
    train_windows, train_info = pack_text_windows(
        train_data, tokenizer, sequence_length=args.sequence_length,
        max_windows=args.train_windows, max_rows=args.train_rows,
        seed=args.data_seed,
    )
    validation_windows, validation_info = pack_text_windows(
        validation_data, tokenizer, sequence_length=args.sequence_length,
        max_windows=args.eval_windows, max_rows=args.eval_rows,
        seed=args.data_seed + 1,
    )
    del train_data, validation_data

    def load_base():
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, use_safetensors=True, trust_remote_code=False,
            torch_dtype="auto",
        ).to(device).eval()
        base.config.use_cache = False
        return base

    base = load_base()
    revision = getattr(base.config, "_commit_hash", None)
    base_loss = evaluate(base, validation_windows, batch_size=args.batch_size,
                         device=device)
    del base
    if device.type == "cuda":
        torch.cuda.empty_cache()
    report = {
        "kind": "persistent_state_sidecar_replication_not_compute_matched",
        "git_commit": git_sha(),
        "base_model": MODEL_ID,
        "model_revision": revision,
        "dataset": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "train": train_info,
        "validation": validation_info,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "base_validation_loss": base_loss,
        "seeds": seeds,
        "modes": modes,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "rank": args.rank,
        "learning_rate": args.learning_rate,
        "residual_scale": args.residual_scale,
        "clip_norm": args.clip_norm,
        "runs": {},
        "summary": None,
        "limitations": [
            "Frozen SmolLM2, no adapter feedback into pretrained attention.",
            "Pretrained model may have previously seen WikiText-2 content.",
        "real_equivalent exactly re-encodes three complex states using six real channels.",
        "real_matched_orthogonal matches initialization but has more O(6) mixer degrees of freedom.",
            "Equal tokens and parameters do not guarantee equal compute.",
            "Real-six orthogonal mixer has more unconstrained effective mixing degrees of freedom.",
            "Three seeds and a short pilot do not establish statistical significance.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Baseline NLL={base_loss:.6f}, validation windows={len(validation_windows)}, "
        f"seeds={seeds}, modes={modes}", flush=True
    )
    for seed in seeds:
        report["runs"][str(seed)] = {}
        for mode in modes:
            torch.manual_seed(seed)
            random.seed(seed)
            model = PersistentStateAdapterLM(
                load_base(), rank=args.rank, mode=mode,
                residual_scale=args.residual_scale,
            ).to(device)
            parameters = list(persistent_parameters(model))
            if not parameters:
                raise RuntimeError(f"No trainable parameters in {mode}")
            params = persistent_parameter_report(model)
            initial = evaluate(
                model, validation_windows, batch_size=args.batch_size,
                device=device,
            )
            if abs(initial - base_loss) > 1e-5:
                raise RuntimeError(
                    f"Initial pretrained parity failed for {mode}, seed {seed}: "
                    f"{initial} vs {base_loss}"
                )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            print(
                f"Seed {seed} / {mode}: training {args.steps} steps, "
                f"{params['trainable_parameters']:,} trainable parameters...",
                flush=True,
            )
            outcome = train_adapters(
                model, parameters, train_windows, steps=args.steps,
                batch_size=args.batch_size, device=device,
                learning_rate=args.learning_rate,
                seed=seed, clip=args.clip_norm,
            )
            final_loss = evaluate(
                model, validation_windows, batch_size=args.batch_size,
                device=device,
            )
            if not math.isfinite(final_loss):
                raise RuntimeError(f"Nonfinite held-out loss for {mode}/{seed}")
            report["runs"][str(seed)][mode] = {
                **params, **outcome,
                "initial_validation_loss": initial,
                "final_validation_loss": final_loss,
                "validation_loss_delta": final_loss - base_loss,
                "peak_cuda_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated())
                    if device.type == "cuda" else None
                ),
            }
            # Preserve completed measurements; do not resume old results
            # automatically or mix runs from different source revisions.
            args.output.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"Seed {seed} / {mode}: NLL {final_loss:.6f} "
                f"(delta {final_loss-base_loss:+.6f}); "
                f"train_time={outcome['training_elapsed_seconds']:.2f}s",
                flush=True,
            )
            del model, parameters
            if device.type == "cuda":
                torch.cuda.empty_cache()

    report["summary"] = paired_summary(report["runs"], reference="no_mixing")
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved: {args.output}")
    print("Caution: not FLOP-matched; no claim of TRIHAERA superiority.")


if __name__ == "__main__":
    main()
