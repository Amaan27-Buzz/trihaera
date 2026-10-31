"""Phase 1: raw generation/VRAM/latency baseline on local GPU, no training.

Windows example:
  $env:PYTHONPATH=(Join-Path (Get-Location) 'src')
  python scripts/benchmark_models.py --model smol --offline
  python scripts/benchmark_models.py --model qwen

Each model runs in its OWN process to avoid residual GPU allocations.
All answers are synthetic research prompts, not the user's private chats.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

from model_lab.catalog import resolve_model
from model_lab.cases import fixed_cases, needle_case
from model_lab.scoring import assess, summarize


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            timeout=3, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def load_reference(spec, device: str, offline: bool):
    import torch
    dtype = (torch.bfloat16 if torch.cuda.is_available() and
             device == "cuda" and torch.cuda.is_bf16_supported()
             else torch.float16 if device == "cuda" else torch.float32)
    if spec.loader == "causal":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            spec.model_id, local_files_only=offline, trust_remote_code=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            spec.model_id, local_files_only=offline,
            trust_remote_code=False, use_safetensors=True, dtype=dtype
        )
        formatter = tokenizer
    elif spec.loader == "multimodal":
        # Text-only input to the OFFICIAL full Qwen3.5 checkpoint.
        # AutoProcessor initializes a video processor requiring Torchvision,
        # even though no image/video is supplied. Loading the tokenizer by
        # itself avoids that unnecessary dependency and retains the exact
        # original multimodal model weights.
        try:
            from transformers import AutoModelForMultimodalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3.5 requires Transformers with "
                "AutoModelForMultimodalLM support."
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(
            spec.model_id, local_files_only=offline, trust_remote_code=False
        )
        model = AutoModelForMultimodalLM.from_pretrained(
            spec.model_id, local_files_only=offline, trust_remote_code=False,
            use_safetensors=True, dtype=dtype
        )
        formatter = tokenizer
    else:
        raise ValueError("Unknown model loader")
    model = model.to(device).eval()
    return model, formatter, tokenizer, str(dtype)


def encode(formatter, messages, device):
    payload = formatter.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    if not hasattr(payload, "to") or "input_ids" not in payload:
        raise TypeError("Tokenizer/processor did not return model-ready input IDs")
    return payload.to(device)


def count_chat_tokens(formatter, messages) -> int:
    """Count ACTUAL model-ready token IDs, not batch/dictionary outer length."""
    payload = formatter.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    ids = payload["input_ids"]
    if hasattr(ids, "shape"):
        return int(ids.shape[-1])
    if isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)):
        return len(ids[0])
    return len(ids)


def build_needle(formatter, target: int):
    """Construct the smallest repeated diary filling at least target tokens.

    Exponential search then binary search is fast and avoids tokenizing an
    ever-growing string hundreds of times. It works for different model
    tokenizers and doesn't presume tokens per English sentence.
    """
    if target < 128:
        raise ValueError("Context target must be >=128 tokens")
    segment = "Diary note: I walked in the park and cooked dinner afterward. "
    def candidate(repetitions: int):
        return needle_case(target, segment * repetitions)
    low, high = 0, 1
    while count_chat_tokens(formatter, candidate(high).chat()) < target:
        low, high = high, high * 2
        if high > 100000:
            raise RuntimeError("Tokenizer cannot reach target context length")
    # Find first length >= target; minimize accidental context-window overrun.
    while low + 1 < high:
        mid = (low + high) // 2
        if count_chat_tokens(formatter, candidate(mid).chat()) >= target:
            high = mid
        else:
            low = mid
    return candidate(high)

def sync(device: str):
    if device == "cuda":
        import torch
        torch.cuda.synchronize()


def generate_once(model, formatter, tokenizer, case, device, max_new_tokens,
                  decoding: str, seed: int, first_token_probe: bool):
    import torch
    messages = case.chat()
    inputs = encode(formatter, messages, device)
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    if prompt_tokens + max_new_tokens > int(
        getattr(model.config, "max_position_embeddings", 262144)
    ):
        raise ValueError("Prompt plus output would exceed config max_position_embeddings")

    def run(max_tokens: int):
        sync(device)
        start = time.perf_counter()
        with torch.inference_mode():
            kwargs = {"max_new_tokens": max_tokens, "use_cache": True,
                      "pad_token_id": tokenizer.eos_token_id}
            if decoding == "recommended":
                # Qwen3.5 model card's non-thinking text policy; fixed random
                # seed for repeatability, still not deterministic across devices.
                kwargs.update(do_sample=True, temperature=1.0,
                              top_p=1.0, top_k=20)
            else:
                kwargs.update(do_sample=False)
            output = model.generate(**inputs, **kwargs)
        sync(device)
        return output, time.perf_counter() - start

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    first_token_time = None
    if first_token_probe:
        torch.manual_seed(seed)
        _, first_token_time = run(1)
    torch.manual_seed(seed)
    output, total_time = run(max_new_tokens)
    new_ids = output[0, prompt_tokens:]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return {
        "id": case.name, "category": case.category,
        "chat_messages": messages,
        "expected_check": case.check, "expected_value": case.expected,
        "raw_answer": raw, "evaluation": assess(case, raw),
        "prompt_tokens": prompt_tokens, "generated_tokens": int(new_ids.numel()),
        "time_total_seconds": total_time,
        "first_token_probe_seconds": first_token_time,
        "generation_tokens_per_total_second": (
            int(new_ids.numel()) / total_time if total_time else None
        ),
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if device == "cuda" else None
        ),
        "peak_cuda_reserved_bytes": (
            int(torch.cuda.max_memory_reserved()) if device == "cuda" else None
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("smol", "qwen"), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--decoding", choices=("greedy", "recommended"), default="greedy")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--context-targets", default="512,2048",
                        help="Short native-context probes only; comma-separated token targets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-first-token-probe", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not 16 <= args.max_new_tokens <= 256 or args.seed < 0:
        parser.error("Invalid generation budget/seed")
    try:
        targets = [int(t.strip()) for t in args.context_targets.split(",") if t.strip()]
    except ValueError:
        parser.error("Expected context token targets like 512,2048")
    if len(targets) > 5 or any(t < 128 or t > 8192 for t in targets):
        parser.error("Context probes must be between 128 and 8192 tokens")
    import torch
    import transformers
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    spec = resolve_model(args.model)
    if any(t + args.max_new_tokens > spec.advertised_context for t in targets):
        parser.error("Context probe plus generation exceeds advertised model window")
    output = args.output or Path(f"artifacts/baseline_{spec.key}_{args.decoding}.json")
    if output.exists():
        parser.error(f"Refusing to overwrite existing report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    report = {
        "kind": "model_first_unmodified_reference_diagnostic",
        "model": spec.model_id, "loader": spec.loader,
        "advertised_context_tokens": spec.advertised_context,
        "decoding": args.decoding, "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(), "git_sha": git_revision(),
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "cuda_memory_total_bytes": (
            torch.cuda.get_device_properties(0).total_memory if device == "cuda" else None
        ),
        "results": [], "status": "running",
        "limitations": [
            "Unmodified reference models only; no training/optimization claim.",
            "Small synthetic, partly automatically scored diagnostic; all raw answers included.",
            "No internet search, multi-chat memory, or generative UI tested.",
            "First-token probe includes prefill; no streaming/browser overhead.",
            "Greedy Qwen3.5 is deterministic for profiling but not the vendor's recommended quality setting.",
            "Context probes are within native context and do not demonstrate long-term memory.",
        ],
    }
    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    model = None
    try:
        print(f"Loading {spec.model_id} on {device}...", flush=True)
        start = time.perf_counter()
        model, formatter, tokenizer, dtype = load_reference(spec, device, args.offline)
        sync(device)
        report["model_loading_seconds"] = time.perf_counter() - start
        report["model_dtype"] = dtype
        report["model_revision"] = getattr(model.config, "_commit_hash", None)
        report["trainable_params_reference"] = sum(p.numel() for p in model.parameters())
        # An untimed warmup also forces CUDA kernels and caches to initialize.
        warm = fixed_cases()[0]
        sync(device)
        with torch.inference_mode():
            data = encode(formatter, warm.chat(), device)
            model.generate(**data, max_new_tokens=2, do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)
        sync(device)
        cases = fixed_cases() + [build_needle(formatter, target) for target in targets]
        for idx, case in enumerate(cases):
            print(f"[{idx+1}/{len(cases)}] {case.name}", flush=True)
            result = generate_once(
                model, formatter, tokenizer, case, device, args.max_new_tokens,
                args.decoding, args.seed, not args.no_first_token_probe,
            )
            report["results"].append(result)
            report["summary"] = summarize(report["results"])
            save()
            print(f'   {result["evaluation"]["verdict"]} '
                  f'{result["time_total_seconds"]:.2f}s '
                  f'{result["generated_tokens"]} tokens '
                  f'peak {round(result["peak_cuda_allocated_bytes"]/2**20) if result["peak_cuda_allocated_bytes"] is not None else "N/A"} MiB',
                  flush=True)
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        save()
        raise
    finally:
        if model is not None:
            del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    save()
    print(json.dumps({"status": report["status"], "model": spec.model_id,
                      "summary": report.get("summary"), "report": str(output)}, indent=2))


if __name__ == "__main__":
    main()
