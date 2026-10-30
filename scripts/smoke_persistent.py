"""Check a depth-persistent three-state sidecar on SmolLM2-135M BASE.

Reuses cached SmolLM2 checkpoint. This is a compatibility test, not a
language-model quality evaluation or a standalone TRIHAERA foundation model.
Run with PYTHONPATH=src and TRIHAERA's independent Python environment.
"""

import argparse
import json
from pathlib import Path
import time

import torch

from trihaera.persistent import (
    PersistentStateAdapterLM, persistent_parameters, persistent_parameter_report,
)

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=("mixed", "no_mixing", "real_six", "real_six_no_mixing", "real_equivalent", "real_equivalent_no_mixing", "real_matched_orthogonal", "real_matched_orthogonal_no_mixing"), default="mixed")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--output", type=Path, default=Path("artifacts/persistent_smoke.json"))
    args = parser.parse_args()
    if args.steps < 1 or args.rank < 1:
        parser.error("Steps and rank must be positive")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers

    torch.manual_seed(23)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    print("Loading cached SmolLM2-135M checkpoint...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, use_safetensors=True, trust_remote_code=False,
        torch_dtype="auto",
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False, local_files_only=True)
    input_ids = tokenizer(
        "A scientific hypothesis needs controlled experiments and reproducible results.",
        return_tensors="pt",
    )["input_ids"].to(device)
    with torch.inference_mode():
        reference = base(
            input_ids=input_ids, output_hidden_states=True, use_cache=False,
        ).logits.clone()
    base.config.use_cache = False
    adapted = PersistentStateAdapterLM(
        base, rank=args.rank, mode=args.mode,
        residual_scale=args.residual_scale,
    ).to(device)
    with torch.no_grad():
        initial = adapted(input_ids=input_ids).logits
    max_difference = (initial.float() - reference.float()).abs().max().item()
    if not torch.equal(initial, reference):
        raise RuntimeError(
            f"Zero-initialized sidecar changed base logits: {max_difference}"
        )
    parameters = list(persistent_parameters(adapted))
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    gradient_norms = []
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = adapted(input_ids=input_ids, labels=input_ids, use_cache=False).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, max_norm=1.0, error_if_nonfinite=True,
        )
        gradient_norms.append(float(norm.item()))
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    with torch.no_grad():
        after = adapted(input_ids=input_ids).logits
    drift = float(
        (after.float() - reference.float()).square().mean().sqrt().item()
    )
    if drift <= 0:
        raise RuntimeError("No adapter output change after optimization")

    result = {
        "kind": "persistent_sidecar_correctness_not_architecture_benchmark",
        "base_model": MODEL_ID,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": str(device),
        "tokens": int(input_ids.numel()),
        **persistent_parameter_report(adapted),
        "initial_max_logit_difference": max_difference,
        "post_training_logit_rms_change": drift,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "last_training_loss": float(loss.item()),
        "max_gradient_norm_before_clipping": max(gradient_norms),
        "train_seconds": elapsed,
        "max_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated())
            if device.type == "cuda" else None
        ),
        "warning": (
            "Sidecar states persist across depth but do not modify backbone "
            "attention; the initial smoke is not a held-out quality test."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
