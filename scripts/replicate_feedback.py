"""Paired small-data experiment for inter-layer feedback on SmolLM2-135M.

Not FLOP- or time-matched. Reuses cached pretrained weights and WikiText-2.
"""
import argparse
import json
import math
from pathlib import Path
import random

import torch

from trihaera.feedback import FeedbackCausalLM, feedback_parameters, feedback_report
from trihaera.feedback_experiments import evaluate_nll, sample_schedule, train_feedback
from trihaera.experiment_stats import paired_summary
from trihaera.pilot_data import pack_text_windows
from replicate_pretrained import git_sha, parse_csv

MODEL = "HuggingFaceTB/SmolLM2-135M"
DATASET = "Salesforce/wikitext"
CONFIG = "wikitext-2-raw-v1"
MODES = ("mixed", "no_mixing", "real_matched_orthogonal")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--modes", default=",".join(MODES))
    p.add_argument("--seeds", default="17,23,42")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--sequence-length", type=int, default=128)
    p.add_argument("--train-windows", type=int, default=512)
    p.add_argument("--eval-windows", type=int, default=64)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--residual-scale", type=float, default=0.1)
    p.add_argument("--clip-norm", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--output", type=Path, default=Path("artifacts/feedback_replication_135m.json"))
    a = p.parse_args()
    try:
        modes = parse_csv(a.modes, set(MODES))
        seeds = [int(x) for x in parse_csv(a.seeds)]
    except ValueError as exc:
        p.error(str(exc))
    if ("no_mixing" not in modes or any(s < 0 for s in seeds) or
            a.steps < 1 or not 1 <= a.batch_size <= 2 or a.rank < 1 or
            a.train_windows < 2 or a.eval_windows < 2 or
            not 8 <= a.sequence_length <= 256 or a.warmup < 0 or
            not 0 < a.learning_rate <= 0.01 or
            not 0 < a.residual_scale <= 1 or not 0 < a.clip_norm <= 100 or
            not all(math.isfinite(x) for x in (a.learning_rate, a.residual_scale, a.clip_norm))):
        p.error("Invalid short-budget feedback configuration")
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable")
    from datasets import load_dataset
    import datasets
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading WikiText-2 official train/validation splits...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False, local_files_only=True)
    train_data = load_dataset(DATASET, CONFIG, split="train")
    validation_data = load_dataset(DATASET, CONFIG, split="validation")
    train, train_meta = pack_text_windows(
        train_data, tokenizer, sequence_length=a.sequence_length,
        max_windows=a.train_windows, max_rows=9000, seed=139)
    validation, validation_meta = pack_text_windows(
        validation_data, tokenizer, sequence_length=a.sequence_length,
        max_windows=a.eval_windows, max_rows=3000, seed=140)
    provenance = {
        "train_dataset_fingerprint": train_data._fingerprint,
        "validation_dataset_fingerprint": validation_data._fingerprint,
        "train": train_meta, "validation": validation_meta,
    }
    del train_data, validation_data

    def load_base():
        base = AutoModelForCausalLM.from_pretrained(
            MODEL, trust_remote_code=False, use_safetensors=True,
            local_files_only=True, torch_dtype="auto").to(device).eval()
        base.config.use_cache = False
        return base

    base = load_base()
    revision = getattr(base.config, "_commit_hash", None)
    base_loss = evaluate_nll(base, validation, batch_size=a.batch_size, device=device)
    del base
    if device.type == "cuda":
        torch.cuda.empty_cache()
    report = {
        "kind": "feedback_pilot_NOT_compute_matched",
        "git_commit": git_sha(), "model": MODEL, "revision": revision,
        "dataset": DATASET, "configuration": CONFIG, **provenance,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "datasets": datasets.__version__, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seeds": seeds, "modes": modes, "steps": a.steps,
        "sequence_length": a.sequence_length, "batch_size": a.batch_size,
        "learning_rate": a.learning_rate, "rank": a.rank,
        "clip_norm": a.clip_norm, "residual_scale": a.residual_scale,
        "frozen_base_validation_nll": base_loss, "runs": {}, "summary": None,
        "limitations": [
            "Only a frozen SmolLM2-derived feedback adapter, not full TRIHAERA.",
            "WikiText-2 validation is held out from adapter training, not necessarily from SmolLM2 pretraining.",
            "Same steps/tokens do not match FLOPs or training time.",
            "O(6) has 540 more trainable mixer parameters than U(3).",
            "Short training and three seeds cannot establish generalization."
        ]
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Frozen-base NLL={base_loss:.6f}; {len(validation)} validation windows", flush=True)
    for si, seed in enumerate(seeds):
        report["runs"][str(seed)] = {}
        schedule = sample_schedule(len(train), a.steps, a.batch_size, seed)
        order = modes[si % len(modes):] + modes[:si % len(modes)]
        for mode in order:
            random.seed(seed)
            torch.manual_seed(seed)
            model = FeedbackCausalLM(
                load_base(), rank=a.rank, mode=mode,
                residual_scale=a.residual_scale).to(device)
            params = list(feedback_parameters(model))
            initial = evaluate_nll(model, validation, batch_size=a.batch_size, device=device)
            if abs(initial - base_loss) > 1e-5:
                raise RuntimeError(f"Pretrained-logit parity failed: {seed}/{mode}")
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            print(f"Seed {seed}/{mode}: {a.steps} feedback steps...", flush=True)
            metrics = train_feedback(
                model, params, train, schedule, device=device,
                learning_rate=a.learning_rate, clip_norm=a.clip_norm, warmup=a.warmup)
            final = evaluate_nll(model, validation, batch_size=a.batch_size, device=device)
            if not math.isfinite(final):
                raise RuntimeError(f"Nonfinite validation NLL: {seed}/{mode}")
            report["runs"][str(seed)][mode] = {
                **feedback_report(model), **metrics,
                "initial_validation_loss": initial,
                "final_validation_loss": final,
                "validation_loss_delta": final - base_loss,
                "peak_cuda_allocated_bytes": (
                    torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None),
                "peak_cuda_reserved_bytes": (
                    torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None),
            }
            a.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"{mode}: NLL {final:.6f}, median_step_s={metrics['step_time_median_seconds']:.4f}", flush=True)
            del model, params
            if device.type == "cuda":
                torch.cuda.empty_cache()
    report["summary"] = paired_summary(report["runs"], reference="no_mixing")
    report["paired_O6_minus_U3"] = (
        [report["runs"][str(s)]["real_matched_orthogonal"]["final_validation_loss"] -
         report["runs"][str(s)]["mixed"]["final_validation_loss"] for s in seeds]
        if {"mixed", "real_matched_orthogonal"}.issubset(modes) else None)
    a.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": report["summary"],
                      "paired_O6_minus_U3": report["paired_O6_minus_U3"]}, indent=2))
    print("No architecture advantage or compute efficiency has been established.")


if __name__ == "__main__":
    main()
