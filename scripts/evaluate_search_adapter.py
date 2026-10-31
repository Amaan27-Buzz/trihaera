"""Evaluate generated search-native decisions, with or without LoRA adapter.

Records every generated JSON/text and parse failure on held-out scenarios.
Decision accuracy does NOT equal final factuality or live search quality.
"""
import argparse
import json
from pathlib import Path
import time

from lra.search_examples import split_examples, to_messages
from lra.search_native import parse_decision

MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--output", type=Path,
                   default=Path("artifacts/lra_search_policy_eval.json"))
    a = p.parse_args(argv)
    if not 24 <= a.max_new_tokens <= 512:
        p.error("Use 24..512 max generation tokens")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if a.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable")
    dtype = (
        torch.bfloat16 if a.device == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16 if a.device == "cuda" else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(
        a.model, trust_remote_code=False, local_files_only=a.offline
    )
    model = AutoModelForCausalLM.from_pretrained(
        a.model, trust_remote_code=False, use_safetensors=True,
        local_files_only=a.offline, dtype=dtype,
    ).to(a.device)
    if a.adapter:
        from peft import PeftModel
        if not (a.adapter / "adapter_config.json").is_file():
            p.error("No LoRA adapter_config.json found at --adapter")
        model = PeftModel.from_pretrained(
            model, a.adapter, is_trainable=False
        )
    model.eval()
    rows = split_examples("test")
    results = []
    if a.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    for row in rows:
        text = tokenizer.apply_chat_template(
            to_messages(row), tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, add_special_tokens=False,
                           return_tensors="pt").to(a.device)
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs, do_sample=False,
                max_new_tokens=a.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        if a.device == "cuda":
            torch.cuda.synchronize()
        duration = time.perf_counter() - start
        generated = tokenizer.decode(
            output[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        try:
            action = parse_decision(
                generated, valid_sources={s["id"] for s in row["evidence"]}
            )
            parsed = True
            error = None
            decision = action.decision
            sources = list(action.citations)
            action_queries = list(action.queries)
            answer = action.answer
        except ValueError as exc:
            parsed = False
            error = str(exc)
            decision, sources, action_queries, answer = None, [], [], ""
        target = row["target"]
        results.append({
            "id": row["id"], "kind": row["kind"],
            "question": row["question"],
            "expected_decision": target["decision"],
            "actual_decision": decision,
            "decision_correct": decision == target["decision"],
            "valid_json_action": parsed,
            "parse_error": error,
            "actual_queries": action_queries,
            "actual_citations": sources,
            "citation_ids_valid": parsed,
            "expected_citations": target["citations"],
            "answer_text": answer,
            "raw_model_output": generated,
            "generation_seconds": duration,
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "output_tokens": int(output.shape[1] - inputs["input_ids"].shape[1]),
        })
        print(
            f'{row["id"]}: expected={target["decision"]} '
            f'actual={decision} valid={parsed} {duration:.2f}s',
            flush=True,
        )
    summary = {
        "total": len(results),
        "valid_action_count": sum(x["valid_json_action"] for x in results),
        "decision_correct_count": sum(x["decision_correct"] for x in results),
        "accuracy": sum(x["decision_correct"] for x in results) / len(results),
        "by_kind": {
            kind: {
                "total": sum(x["kind"] == kind for x in results),
                "correct": sum(x["kind"] == kind and x["decision_correct"]
                               for x in results),
            }
            for kind in sorted({x["kind"] for x in results})
        },
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated())
            if a.device == "cuda" else None
        ),
        "note": (
            "Small synthetic held-out diagnostic only. "
            "Search query quality, answer factuality and citation support "
            "need separate verification against real retrieved documents."
        ),
    }
    record = {
        "kind": "small_model_search_policy_diagnostic",
        "model": a.model,
        "adapter": str(a.adapter) if a.adapter else None,
        "summary": summary, "results": results,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(record, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
