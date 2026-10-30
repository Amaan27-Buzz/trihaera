"""CUDA compatibility smoke for inter-layer state feedback on SmolLM2-135M.

Reuses the cached model (no new weight downloads). No KV cache, no language
benchmark. Autograd crosses frozen pretrained decoder operations.
"""

import argparse
import json
from pathlib import Path
import time

import torch

from trihaera.feedback import FeedbackCausalLM, feedback_parameters, feedback_report

MODEL = "HuggingFaceTB/SmolLM2-135M"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=("mixed", "no_mixing", "real_matched_orthogonal"),
                        default="mixed")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--output", type=Path, default=Path("artifacts/feedback_smoke.json"))
    args = parser.parse_args()
    if not 1 <= args.steps <= 10 or args.rank < 1 or not 0 < args.learning_rate <= 0.01:
        parser.error("Invalid smoke configuration")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    torch.manual_seed(23)
    print("Loading cached SmolLM2-135M Base...", flush=True)
    backbone = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=False, use_safetensors=True,
        local_files_only=True, torch_dtype="auto",
    ).to(device).eval()
    backbone.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, trust_remote_code=False, local_files_only=True
    )
    ids = tokenizer(
        "A scientific hypothesis needs reproducible experiments and controls.",
        return_tensors="pt",
    )["input_ids"].to(device)
    with torch.no_grad():
        original = backbone(input_ids=ids, use_cache=False).logits.detach().clone()
    model = FeedbackCausalLM(backbone, rank=args.rank, mode=args.mode).to(device)
    with torch.no_grad():
        initial = model(input_ids=ids).logits
    parity_error = float((initial.float() - original.float()).abs().max().item())
    if not torch.equal(initial, original):
        raise RuntimeError(f"Initial pretrained logit parity failed: {parity_error}")

    parameters = list(feedback_parameters(model))
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    losses = []
    grad_norms = []
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=ids).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0,
                                               error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.item()))
        grad_norms.append(float(norm.item()))
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    with torch.no_grad():
        after = model(input_ids=ids).logits
    change = float((after.float() - original.float()).square().mean().sqrt().item())
    if change <= 0:
        raise RuntimeError("Feedback pathway has not altered logits")
    if model.carrier.active or model.carrier.state is not None:
        raise RuntimeError("Forward state leaked between calls")
    result = {
        "kind": "feedback_integration_smoke_not_quality_benchmark",
        "model": MODEL, "device": str(device),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "token_count": int(ids.numel()), **feedback_report(model),
        "initial_max_logit_error": parity_error,
        "post_training_logit_rms_change": change,
        "loss_first": losses[0], "loss_last": losses[-1],
        "max_preclip_gradient_norm": max(grad_norms),
        "training_seconds": seconds,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
        ),
        "warning": "Correctness only; frozen weights still require backprop through later layers",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
