"""Small controlled pretrained adapter pilot on WikiText-2.

Compares frozen base, real low-rank adapter, no-mixing three-state adapter,
and mixed three-state adapter using the SAME dataset, train window order,
training tokens, optimizer settings and frozen pretrained backbone.

NOT parameter/FLOP-matched and NOT evidence of full TRIHAERA capabilities.
"""

import argparse
import json
import math
from pathlib import Path
import random
import time

import torch

from trihaera.pilot_data import pack_text_windows
from trihaera.pretrained import attach_adapters, adapter_parameters

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
MODES = ("real_adapter", "no_mixing", "trihaera")


@torch.no_grad()
def evaluate(model, windows: torch.Tensor, *, batch_size: int,
             device: torch.device) -> float:
    model.eval()
    weighted_loss = 0.0
    targets = 0
    for start in range(0, len(windows), batch_size):
        batch = windows[start:start + batch_size].to(device)
        result = model(input_ids=batch, labels=batch, use_cache=False)
        count = batch.shape[0] * (batch.shape[1] - 1)
        weighted_loss += float(result.loss.item()) * count
        targets += count
    return weighted_loss / targets


def train_adapters(model, parameters, windows: torch.Tensor, *,
                   steps: int, batch_size: int, device: torch.device,
                   learning_rate: float, seed: int, clip: float) -> dict:
    # Deliberately keep backbone in eval mode to avoid dropout differences.
    # This does NOT disable autograd through the pretrained layers.
    model.eval()
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    rng = random.Random(seed)
    step_losses = []
    grad_norms = []
    start = time.perf_counter()
    for _ in range(steps):
        indices = [rng.randrange(len(windows)) for __ in range(batch_size)]
        batch = windows[indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=batch, labels=batch, use_cache=False).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss: stopping pilot")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=clip,
                                              error_if_nonfinite=True)
        optimizer.step()
        step_losses.append(float(loss.item()))
        grad_norms.append(float(norm.item()))
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"train_steps": steps,
            "training_tokens_approx": steps * batch_size * (windows.shape[1] - 1),
            "step_loss_first": step_losses[0],
            "step_loss_last": step_losses[-1],
            "mean_preclip_gradient_norm": sum(grad_norms) / len(grad_norms),
            "max_preclip_gradient_norm": max(grad_norms),
            "training_elapsed_seconds": time.perf_counter() - start}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--train-windows", type=int, default=256)
    parser.add_argument("--eval-windows", type=int, default=12)
    parser.add_argument("--train-rows", type=int, default=4000)
    parser.add_argument("--eval-rows", type=int, default=600)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--data-seed", type=int, default=139)
    parser.add_argument("--modes", default=",".join(MODES),
                        help="Comma-separated subset: real_adapter,no_mixing,trihaera")
    parser.add_argument("--output", type=Path, default=Path("artifacts/pilot_135m.json"))
    args = parser.parse_args()
    modes = [mode.strip() for mode in args.modes.split(",")]
    if not modes or len(set(modes)) != len(modes) or any(mode not in MODES for mode in modes):
        parser.error("Invalid or duplicate adapter modes")
    if (args.steps < 1 or args.batch_size < 1 or args.rank < 1 or
        args.train_windows < 2 or args.eval_windows < 2 or
        args.train_rows < 1 or args.eval_rows < 1 or
        not 8 <= args.sequence_length <= 8192 or
        not 0 < args.learning_rate <= 0.01 or
        not 0 < args.residual_scale <= 1 or args.clip_norm <= 0):
        parser.error("Invalid positive/finite pilot budget arguments")
    for attr in ("learning_rate", "residual_scale", "clip_norm"):
        if not math.isfinite(getattr(args, attr)):
            parser.error(f"{attr} must be finite")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import transformers
        from datasets import load_dataset
        import datasets
    except ImportError as exc:
        raise SystemExit(
            "Missing library: install 'datasets' (and transformers/safetensors if needed) "
            "in TRIHAERA's venv only. See docs/PRETRAINED_ADAPTATION.md"
        ) from exc

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable: specify --device cpu")
    print("Loading official WikiText-2 raw train/validation splits...", flush=True)
    # Keep strictly separate official splits; do not evaluate on train samples.
    train_set = load_dataset(DATASET_ID, DATASET_CONFIG, split="train")
    val_set = load_dataset(DATASET_ID, DATASET_CONFIG, split="validation")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    train_windows, train_meta = pack_text_windows(
        train_set, tokenizer, sequence_length=args.sequence_length,
        max_windows=args.train_windows, max_rows=args.train_rows, seed=args.data_seed
    )
    val_windows, val_meta = pack_text_windows(
        val_set, tokenizer, sequence_length=args.sequence_length,
        max_windows=args.eval_windows, max_rows=args.eval_rows, seed=args.data_seed + 1
    )
    print(f"Train: {train_meta['num_windows']} windows; "
          f"validation: {val_meta['num_windows']} windows.", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    def load_base():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, trust_remote_code=False, use_safetensors=True,
            torch_dtype="auto",
        ).to(device)
        model.config.use_cache = False
        return model.eval()

    base = load_base()
    base_validation_loss = evaluate(
        base, val_windows, batch_size=args.batch_size, device=device)
    model_revision = getattr(base.config, "_commit_hash", None)
    del base
    if device.type == "cuda":
        torch.cuda.empty_cache()
    report = {
        "kind": "small_data_adapter_pilot_NOT_full_qualification",
        "model": MODEL_ID,
        "model_commit": model_revision,
        "dataset": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_train_fingerprint": train_set._fingerprint,
        "dataset_validation_fingerprint": val_set._fingerprint,
        "dataset_license": "CC BY-SA 3.0 / GFDL (check upstream conditions)",
        "train_windows": train_meta,
        "validation_windows": val_meta,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "residual_scale": args.residual_scale,
        "rank": args.rank,
        "clip_norm": args.clip_norm,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "datasets_version": datasets.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "frozen_base_validation_loss": base_validation_loss,
        "baseline_perplexity": math.exp(min(base_validation_loss, 40)),
        "modes": {},
        "limitations": (
            "Adapter pilot, not compute matched, only one seed by default. "
            "Train and validation splits are separate. Small validation windows "
            "do not establish general capability. Base may have previously seen "
            "some WikiText content during pretraining."
        ),
    }
    for mode in modes:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model = load_base()
        params_report = attach_adapters(
            model, rank=args.rank, mode=mode,
            residual_scale=args.residual_scale)
        params = list(adapter_parameters(model))
        before = evaluate(model, val_windows, batch_size=args.batch_size,
                          device=device)
        if abs(before - base_validation_loss) > 1e-5:
            raise RuntimeError(
                f"Adapter {mode} changed validation loss at initialization: "
                f"{before} vs base {base_validation_loss}"
            )
        print(f"Training {mode}: {params_report['trainable_parameters']:,} "
              "parameters...", flush=True)
        training = train_adapters(
            model, params, train_windows, steps=args.steps,
            batch_size=args.batch_size, device=device,
            learning_rate=args.learning_rate, seed=args.seed,
            clip=args.clip_norm)
        after = evaluate(model, val_windows,
                         batch_size=args.batch_size, device=device)
        mode_result = {
            **params_report, **training,
            "initial_validation_loss": before,
            "final_validation_loss": after,
            "validation_loss_delta": after - before,
            "validation_perplexity": math.exp(min(after, 40)),
            "max_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated())
                if device.type == "cuda" else None
            ),
        }
        report["modes"][mode] = mode_result
        print(
            f"{mode}: validation NLL {before:.5f} -> {after:.5f} "
            f"(delta {after-before:+.5f}), "
            f"trainable={params_report['trainable_parameters']:,}, "
            f"train_s={training['training_elapsed_seconds']:.2f}",
            flush=True,
        )
        del model, params
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved report to {args.output}")
    print("Do not interpret this small, single-seed, non-compute-matched pilot "
          "as architecture proof.")


if __name__ == "__main__":
    main()
