"""Pretrained SmolLM2-135M adapter parity and gradient probe.

Downloads the official Apache-2.0 pretrained BASE checkpoint from HF once.
Does NOT benchmark a trained model or produce a foundation model.
PYTHONPATH=src python scripts/smoke_pretrained.py --device cuda
"""

import argparse
import json
from pathlib import Path
import time

import torch

from trihaera.pretrained import attach_adapters, adapter_parameters

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=("trihaera", "no_mixing", "real_adapter"),
                        default="trihaera")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    if args.rank < 1 or not 4 <= args.max_length <= 256:
        parser.error("rank >= 1, max-length within [4,256] required")
    if args.model != DEFAULT_MODEL:
        parser.error("Only the vetted SmolLM2-135M base is enabled in this first pilot")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import transformers
    except ImportError as exc:
        raise SystemExit(
            "Install Transformers in TRIHAERA's .venv only. "
            "See docs/PRETRAINED_ADAPTATION.md"
        ) from exc

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; choose --device cpu")
    print(f"Loading {args.model} ({args.device}). First run downloads ~270 MB.",
          flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=False, use_safetensors=True,
        torch_dtype="auto",
    ).to(device).eval()
    model.config.use_cache = False
    prompt = (
        "The scientific method requires a hypothesis, a controlled experiment, "
        "a reliable measurement, and a careful interpretation of the results."
    )
    ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=args.max_length)["input_ids"].to(device)
    if ids.shape[-1] < 4:
        raise RuntimeError("Tokenizer unexpectedly produced fewer than four tokens")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    with torch.inference_mode():
        reference = model(input_ids=ids, use_cache=False).logits.detach().clone()

    report = attach_adapters(model, rank=args.rank, mode=args.mode)
    with torch.no_grad():
        initial = model(input_ids=ids, use_cache=False).logits
    max_diff = (initial.float() - reference.float()).abs().max().item()
    if not torch.equal(initial, reference):
        raise RuntimeError(
            f"Initial adaptation unexpectedly changed logits: {max_diff}"
        )

    optimizer = torch.optim.AdamW(list(adapter_parameters(model)), lr=1e-3)
    model.train()
    before = None
    start = time.perf_counter()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        # Next-token loss uses the pretrained model's causal-LM shift.
        loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        if step == 0:
            before = float(loss.item())
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    model.eval()
    with torch.no_grad():
        changed = model(input_ids=ids, use_cache=False).logits
    output_change = float(
        (changed.float() - reference.float()).square().mean().sqrt().item()
    )
    if output_change <= 0:
        raise RuntimeError("Adapter was not able to modify model outputs")
    result = {
        "experiment": "pretrained_adapter_correctness_not_benchmark",
        "upstream_model": args.model,
        "mode": args.mode,
        "adapter_rank": args.rank,
        "device": str(device),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "token_count": int(ids.numel()),
        **report,
        "initial_max_logit_difference": max_diff,
        "initial_training_loss": before,
        "post_two_step_logit_rms_change": output_change,
        "two_step_elapsed_seconds": elapsed,
        "max_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        ),
        "warning": (
            "Two steps on one short sample are only a compatibility test, "
            "not evidence of quality"
        ),
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
