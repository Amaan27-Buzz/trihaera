"""LoRA fine-tuning of SmolLM2-360M for structured search decisions.

This changes ADAPTER weights (not the frozen base). A deliberately tiny
hand-curated diagnostic corpus establishes the training/inference mechanics.
It is NOT sufficient to claim Google AI Overview parity or generalization.

Requires existing CUDA PyTorch + Transformers, and `pip install peft`.
Run from repository root, with PYTHONPATH=src.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import random
import time

from lra.search_examples import split_examples, to_messages
from lra.search_native import target_json

MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


def encode_supervised(tokenizer, row: dict, max_length: int) -> dict:
    """Loss ONLY on canonical assistant action, never context or fake web text."""
    prompt_ids = tokenizer.apply_chat_template(
        to_messages(row), tokenize=True, add_generation_prompt=True
    )
    response_ids = tokenizer.encode(
        target_json(row["target"]), add_special_tokens=False
    )
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise RuntimeError("Model tokenizer must have an EOS token")
    full = prompt_ids + response_ids + [eos_id]
    if len(full) > max_length:
        raise ValueError(
            f'Sample {row["id"]}: {len(full)} tokens > {max_length}. '
            "Increase --max-length; do not silently truncate supervision."
        )
    if not response_ids:
        raise ValueError("Missing supervised assistant target")
    return {
        "input_ids": full,
        "attention_mask": [1] * len(full),
        "labels": [-100] * len(prompt_ids) + response_ids + [eos_id],
        "id": row["id"],
        "kind": row["kind"],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--accumulation", type=int, default=4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--output-dir", type=Path,
                   default=Path("artifacts/lra_search_360m_adapter"))
    a = p.parse_args(argv)
    if (not 1 <= a.steps <= 10000 or not 1 <= a.accumulation <= 64
            or not 1 <= a.rank <= 64 or not 128 <= a.max_length <= 8192
            or not 0 < a.lr <= .005 or not math.isfinite(a.lr)):
        p.error("Invalid training configuration")
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        p.error("Output directory not empty; choose a fresh adapter directory")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    if a.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA is unavailable")
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    dtype = (
        torch.bfloat16 if a.device == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16 if a.device == "cuda" else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(
        a.model, trust_remote_code=False, local_files_only=a.offline
    )
    base = AutoModelForCausalLM.from_pretrained(
        a.model, trust_remote_code=False, local_files_only=a.offline,
        use_safetensors=True, dtype=dtype,
    )
    base.config.use_cache = False
    config = LoraConfig(
        r=a.rank, lora_alpha=a.rank * 2, lora_dropout=0,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(base, config).to(device)
    model.train()
    train_rows = [
        encode_supervised(tokenizer, row, a.max_length)
        for row in split_examples("train")
    ]
    heldout = [
        encode_supervised(tokenizer, row, a.max_length)
        for row in split_examples("test")
    ]
    parameters = [x for x in model.parameters() if x.requires_grad]
    if not parameters:
        raise RuntimeError("No trainable adapter parameters")
    optimizer = torch.optim.AdamW(parameters, lr=a.lr, weight_decay=0.0)
    generator = random.Random(a.seed)
    if a.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    losses = []
    model.train()
    for step in range(a.steps):
        optimizer.zero_grad(set_to_none=True)
        values = []
        for _ in range(a.accumulation):
            sample = train_rows[generator.randrange(len(train_rows))]
            ids = torch.tensor([sample["input_ids"]], device=device)
            labels = torch.tensor([sample["labels"]], device=device)
            result = model(input_ids=ids, labels=labels, use_cache=False)
            loss = result.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")
            (loss / a.accumulation).backward()
            values.append(float(loss.detach().item()))
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        losses.append(sum(values) / len(values))
        if step == 0 or (step + 1) % 10 == 0:
            print(
                f"step {step+1}/{a.steps} loss={losses[-1]:.4f} "
                f"preclip_grad={float(norm):.3f}",
                flush=True,
            )
    if a.device == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    # Teacher-forced held-out loss, NOT a semantic evaluation.
    model.eval()
    val_losses = []
    with torch.inference_mode():
        for row in heldout:
            ids = torch.tensor([row["input_ids"]], device=device)
            labels = torch.tensor([row["labels"]], device=device)
            value = float(model(input_ids=ids, labels=labels).loss.item())
            if not math.isfinite(value):
                raise RuntimeError("Nonfinite held-out loss")
            val_losses.append(value)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(a.output_dir)
    record = {
        "kind": "search_native_bootstrap_lora_sft",
        "base_model": a.model, "rank": a.rank,
        "optimizer_steps": a.steps, "accumulation": a.accumulation,
        "trained_examples": len(train_rows), "heldout_examples": len(heldout),
        "seed": a.seed, "max_length": a.max_length, "lr": a.lr,
        "train_loss_first": losses[0], "train_loss_last": losses[-1],
        "heldout_teacher_forced_mean_loss": sum(val_losses) / len(val_losses),
        "training_seconds": elapsed,
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if a.device == "cuda" else None
        ),
        "limitations": [
            "Tiny diagnostic corpus; do not infer search skill generalization",
            "Only action targets updated, not model backbone or web index",
            "No live internet search occurred while training",
            "Evaluate generated decisions and grounded final answers separately",
        ],
    }
    (a.output_dir / "train_report.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
