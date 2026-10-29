"""Reproducible multi-seed adapter comparison with paired loss differences.

SmolLM2-135M is a pretrained third-party model; this is a *derived*
architecture adapter experiment, NOT TRIHAERA pretraining.

Assumes PYTHONPATH=src and installed transformers, datasets, safetensors.
Uses cached Hugging Face model / WikiText-2 after first download.
"""

import argparse
import json
import math
from pathlib import Path
import random
import subprocess

import torch

from trihaera.experiment_stats import paired_summary
from trihaera.pilot_data import pack_text_windows
from trihaera.pretrained import attach_adapters, adapter_parameters
from pilot_pretrained import evaluate, train_adapters

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
DATASET_ID = "Salesforce/wikitext"
CONFIG_ID = "wikitext-2-raw-v1"
MODES = ("real_adapter", "no_mixing", "trihaera", "collapsed_mixing")


def parse_csv(text: str, allowed: set[str] | None = None) -> list:
    values = [value.strip() for value in text.split(",")]
    if any(not value for value in values) or len(values) != len(set(values)):
        raise ValueError("No empty or duplicated comma-separated values allowed")
    if allowed is not None and any(v not in allowed for v in values):
        raise ValueError(f"Unsupported value; choose from {sorted(allowed)}")
    return values


def git_sha() -> str | None:
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, check=True, timeout=5)
        return p.stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", default="17,23,42")
    parser.add_argument("--modes", default="real_adapter,no_mixing,trihaera")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--train-windows", type=int, default=512)
    parser.add_argument("--eval-windows", type=int, default=64)
    parser.add_argument("--train-rows", type=int, default=9000)
    parser.add_argument("--eval-rows", type=int, default=3000)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--data-seed", type=int, default=139)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/replication_135m.json"))
    args = parser.parse_args()
    try:
        seed_strings = parse_csv(args.seeds)
        seeds = [int(seed) for seed in seed_strings]
        modes = parse_csv(args.modes, set(MODES))
    except ValueError as exc:
        parser.error(str(exc))
    if (not seeds or any(seed < 0 for seed in seeds) or
        "no_mixing" not in modes or args.steps < 1 or args.batch_size < 1 or
        args.rank < 1 or args.train_windows < 2 or args.eval_windows < 2 or
        args.train_rows < 1 or args.eval_rows < 1 or
        not 8 <= args.sequence_length <= 4096 or
        not 0 < args.residual_scale <= 1 or
        not 0 < args.learning_rate <= 0.01 or
        not 0 < args.clip_norm <= 100 or
        not all(math.isfinite(x) for x in
                (args.residual_scale, args.learning_rate, args.clip_norm))):
        parser.error("Invalid experiment configuration")
    if not args.output.suffix.lower() == ".json":
        parser.error("Output must be a JSON path")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA not available")

    try:
        import datasets
        import transformers
        from datasets import load_dataset
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit(
            "Use TRIHAERA's independent environment; install datasets, "
            "transformers and safetensors if missing."
        ) from exc

    print("Loading WikiText-2 train/validation only...", flush=True)
    train_set = load_dataset(DATASET_ID, CONFIG_ID, split="train")
    val_set = load_dataset(DATASET_ID, CONFIG_ID, split="validation")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    train_windows, train_manifest = pack_text_windows(
        train_set, tokenizer, sequence_length=args.sequence_length,
        max_windows=args.train_windows, max_rows=args.train_rows,
        seed=args.data_seed,
    )
    val_windows, validation_manifest = pack_text_windows(
        val_set, tokenizer, sequence_length=args.sequence_length,
        max_windows=args.eval_windows, max_rows=args.eval_rows,
        seed=args.data_seed + 1,
    )
    def load_model():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, trust_remote_code=False,
            use_safetensors=True, torch_dtype="auto"
        ).to(device).eval()
        model.config.use_cache = False
        return model

    base = load_model()
    base_loss = evaluate(base, val_windows, batch_size=args.batch_size, device=device)
    revision = getattr(base.config, "_commit_hash", None)
    del base
    if device.type == "cuda":
        torch.cuda.empty_cache()
    report = {
        "experiment": "multi_seed_pretrained_adapter_replication_not_compute_matched",
        "repo_commit": git_sha(),
        "model": MODEL_ID,
        "model_commit": revision,
        "dataset": DATASET_ID,
        "dataset_config": CONFIG_ID,
        "dataset_train_fingerprint": train_set._fingerprint,
        "dataset_validation_fingerprint": val_set._fingerprint,
        "train": train_manifest,
        "validation": validation_manifest,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "frozen_base_validation_loss": base_loss,
        "seeds": seeds,
        "modes": modes,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "residual_scale": args.residual_scale,
        "rank": args.rank,
        "clip_norm": args.clip_norm,
        "runs": {},
        "summary": None,
        "limitations": [
            "Frozen original pretrained model plus layerwise residual adapters.",
            "State mixing is immediately read out: not persistent state superposition.",
            "Equal training windows and steps do not equalize FLOPs or time.",
            "WikiText-2 may have been included in original SmolLM2 pretraining.",
            "Reported seed variances are descriptive, not a significance test.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Base validation NLL: {base_loss:.6f}; "
          f"{len(val_windows)} windows; {len(seeds)} seeds", flush=True)
    for seed in seeds:
        report["runs"][str(seed)] = {}
        for mode in modes:
            torch.manual_seed(seed)
            random.seed(seed)
            model = load_model()
            parameter_stats = attach_adapters(
                model, rank=args.rank, mode=mode,
                residual_scale=args.residual_scale,
            )
            parameters = list(adapter_parameters(model))
            original_loss = evaluate(model, val_windows, batch_size=args.batch_size,
                                     device=device)
            if abs(original_loss - base_loss) > 1e-5:
                raise RuntimeError(
                    f"Seed {seed} mode {mode} changed pretrained validation "
                    f"NLL at initialization: {original_loss} vs {base_loss}"
                )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            print(f"Seed {seed} / {mode}: {args.steps} steps...", flush=True)
            training = train_adapters(
                model, parameters, train_windows, steps=args.steps,
                batch_size=args.batch_size, device=device,
                learning_rate=args.learning_rate, seed=seed,
                clip=args.clip_norm,
            )
            validation_loss = evaluate(
                model, val_windows, batch_size=args.batch_size,
                device=device,
            )
            if not math.isfinite(validation_loss):
                raise RuntimeError(f"Non-finite validation loss for {seed}/{mode}")
            measurement = {
                **parameter_stats,
                **training,
                "initial_validation_loss": original_loss,
                "final_validation_loss": validation_loss,
                "validation_loss_delta": validation_loss - base_loss,
                "max_cuda_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated())
                    if device.type == "cuda" else None
                ),
            }
            report["runs"][str(seed)][mode] = measurement
            # Persist completed runs on failure; rerunning starts from scratch,
            # so never silently mix partial runs with a different code/config.
            args.output.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            print(f"Seed {seed} / {mode}: validation NLL "
                  f"{validation_loss:.5f} (change {validation_loss-base_loss:+.5f}), "
                  f"train_s={training['training_elapsed_seconds']:.2f}", flush=True)
            del model, parameters
            if device.type == "cuda":
                torch.cuda.empty_cache()
    report["summary"] = paired_summary(report["runs"], reference="no_mixing")
    args.output.write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved complete report: {args.output}", flush=True)
    print("No equal-FLOP experiment or claim of general architecture advantage "
          "has yet been completed.")


if __name__ == "__main__":
    main()
