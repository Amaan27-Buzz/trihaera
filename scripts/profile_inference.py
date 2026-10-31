"""Profile native prefill vs fixed-length generation on an unchanged checkpoint.

This is not a kernel-level profiler. It reports synchronized wall time,
total new-token throughput, and peak CUDA allocation under current kernels.
Run the SMOL and QWEN checkpoints in separate processes. Do not install
or change any CUDA extensions to use this script.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import time

try:
    from .benchmark_models import load_reference, encode, build_needle, sync
except ImportError:
    # Direct execution: python scripts/profile_inference.py
    from benchmark_models import load_reference, encode, build_needle, sync
from model_lab.catalog import resolve_model


def _measure(model, inputs, tokenizer, *, device, decode_tokens, repeats):
    import torch
    def timed(do_generate: bool):
        sync(device)
        start = time.perf_counter()
        with torch.inference_mode():
            if do_generate:
                # Fixed-length generation ensures equivalent token workloads,
                # independent of early EOS. Text output is not evaluated here.
                result = model.generate(
                    **inputs, do_sample=False, use_cache=True,
                    min_new_tokens=decode_tokens, max_new_tokens=decode_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
                count = result.shape[-1] - inputs["input_ids"].shape[-1]
            else:
                model(**inputs, use_cache=False)
                count = 0
        sync(device)
        return time.perf_counter() - start, count

    # Explicit warmups for each distinct shape before measurements.
    timed(False)
    timed(True)
    prefill = []
    generate = []
    generated = []
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        a, _ = timed(False)
        b, n = timed(True)
        prefill.append(a)
        generate.append(b)
        generated.append(n)
    if len(set(generated)) != 1 or generated[0] != decode_tokens:
        raise RuntimeError("Expected fixed-length token generation failed")
    return {
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        "generated_tokens_per_run": decode_tokens,
        "prefill_median_seconds": statistics.median(prefill),
        "full_generation_median_seconds": statistics.median(generate),
        "decode_including_prefill_tokens_per_second": (
            decode_tokens / statistics.median(generate)
        ),
        "prefill_run_seconds": prefill,
        "generation_run_seconds": generate,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if device == "cuda" else None
        ),
        "peak_cuda_reserved_bytes": (
            int(torch.cuda.max_memory_reserved()) if device == "cuda" else None
        ),
    }


def run_profile(model, formatter, tokenizer, device, *,
                targets=(128, 512, 2048), decode_tokens=32, repeats=3):
    if repeats < 1 or decode_tokens < 1 or any(n < 128 for n in targets):
        raise ValueError("Invalid profiling controls")
    results = []
    for target in targets:
        case = build_needle(formatter, target)
        inputs = encode(formatter, case.chat(), device)
        print(f"Profiling {inputs['input_ids'].shape[-1]} input tokens...", flush=True)
        stats = _measure(
            model, inputs, tokenizer, device=device,
            decode_tokens=decode_tokens, repeats=repeats,
        )
        stats["requested_prompt_tokens"] = target
        results.append(stats)
        print(f'  prefill {stats["prefill_median_seconds"]:.4f}s | '
              f'fixed {decode_tokens}-token generate '
              f'{stats["full_generation_median_seconds"]:.4f}s', flush=True)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("smol", "qwen"), required=True)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--targets", default="128,512,2048")
    p.add_argument("--decode-tokens", type=int, default=32)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    try:
        targets = [int(x) for x in args.targets.split(",") if x.strip()]
    except ValueError:
        p.error("Targets must be comma-separated integers")
    if (not targets or len(targets) > 4 or
        any(x < 128 or x > 8192 for x in targets) or
        not 8 <= args.decode_tokens <= 128 or not 1 <= args.repeats <= 10):
        p.error("Invalid input sizes, repeats or decode count")
    import torch
    import transformers
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA not available")
    spec = resolve_model(args.model)
    output = args.output or Path(f"artifacts/profile_{args.model}.json")
    if output.exists():
        p.error(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "kind": "original_checkpoint_pytorch_kernel_path_profile",
        "model": spec.model_id, "device": args.device,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "requested_targets": targets, "decode_tokens": args.decode_tokens,
        "repeats": args.repeats, "results": [],
        "status": "running",
        "limitations": [
            "Not kernel-level attribution, power, or production serving speed.",
            "Forced fixed-length output is for computing cost, not answer quality.",
            "Each measurement includes Python dispatch and GPU synchronization.",
            "Original fallback kernels are not modified or optimized.",
            "Warmups and results are in one process; compare identical hardware only.",
        ],
    }
    def save():
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    try:
        model, formatter, tokenizer, dtype = load_reference(
            spec, args.device, args.offline
        )
        report["dtype"] = dtype
        for result in run_profile(
            model, formatter, tokenizer, args.device,
            targets=targets, decode_tokens=args.decode_tokens,
            repeats=args.repeats,
        ):
            report["results"].append(result)
            save()
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__,
                      error=str(exc))
        save()
        raise
    save()
    print(json.dumps({
        "status": report["status"], "results": report["results"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
