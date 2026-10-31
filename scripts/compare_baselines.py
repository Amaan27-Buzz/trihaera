"""Compare baseline JSON reports on EXACT shared case IDs.

No checkpoint downloads or GPU required. Distinguishes diagnostic score
changes from real-world capability claims and flags incomparable settings.
"""
import argparse
import json
from pathlib import Path
import statistics


EXPECTED = {
    "smol": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "qwen": "Qwen/Qwen3.5-0.8B",
}


def _load(path: Path, expected: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "complete":
        raise ValueError(f"{path}: baseline did not complete")
    if data.get("kind") != "model_first_unmodified_reference_diagnostic":
        raise ValueError(f"{path}: wrong report format")
    if data.get("model") != expected:
        raise ValueError(f"{path}: expected {expected}, got {data.get('model')}")
    results = data.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError(f"{path}: missing results")
    ids = [row["id"] for row in results]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: repeated case IDs")
    return data


def _seconds(rows):
    return statistics.median(row["time_total_seconds"] for row in rows)


def _common_stats(rows):
    checked = [r for r in rows if r["evaluation"]["verdict"] != "review"]
    return {
        "total": len(rows), "scored": len(checked),
        "passed": sum(r["evaluation"]["verdict"] == "pass" for r in checked),
        "manual": len(rows) - len(checked),
        "median_total_generation_seconds": _seconds(rows),
        "peak_allocated_mib": max(
            r["peak_cuda_allocated_bytes"] or 0 for r in rows
        ) / (1024 * 1024),
        "aggregate_generated_tokens_per_elapsed_second": (
            sum(r["generated_tokens"] for r in rows) /
            sum(r["time_total_seconds"] for r in rows)
        ),
    }


def compare(smol: dict, qwen: dict) -> dict:
    ref = {row["id"]: row for row in smol["results"]}
    other = {row["id"]: row for row in qwen["results"]}
    shared = sorted(ref.keys() & other.keys())
    if not shared:
        raise ValueError("No common case IDs")
    warnings = []
    for field in ("decoding", "max_new_tokens", "seed"):
        if smol.get(field) != qwen.get(field):
            warnings.append(
                f"Incomparable {field}: {smol.get(field)!r} vs {qwen.get(field)!r}"
            )
    if smol.get("gpu") != qwen.get("gpu"):
        warnings.append("Hardware differs between benchmark runs")
    pairs = []
    for case_id in shared:
        a, b = ref[case_id], other[case_id]
        if a["category"] != b["category"]:
            raise ValueError(f"Mismatch category for {case_id}")
        if a.get("chat_messages") != b.get("chat_messages"):
            # Native-context synthetic probes may have different tokenizer
            # lengths/contents, so report separately, never treat as paired.
            warnings.append(f"Different actual prompt text for {case_id}; not paired")
            continue
        pairs.append({
            "id": case_id, "category": a["category"],
            "smol": {"verdict": a["evaluation"]["verdict"],
                     "raw_answer": a["raw_answer"],
                     "total_seconds": a["time_total_seconds"],
                     "output_tokens": a["generated_tokens"]},
            "qwen": {"verdict": b["evaluation"]["verdict"],
                     "raw_answer": b["raw_answer"],
                     "total_seconds": b["time_total_seconds"],
                     "output_tokens": b["generated_tokens"]},
        })
    included = {x["id"] for x in pairs}
    return {
        "kind": "paired_original_baseline_comparison",
        "shared_ids": shared,
        "truly_identical_prompt_ids": sorted(included),
        "unpaired_smol": sorted(set(ref) - included),
        "unpaired_qwen": sorted(set(other) - included),
        "smol": _common_stats([ref[x] for x in sorted(included)]),
        "qwen": _common_stats([other[x] for x in sorted(included)]),
        "comparability_warnings": warnings,
        "per_case": pairs,
        "interpretation": (
            "Tiny synthetic raw-output comparison. "
            "Pass rate excludes manual reviews and mismatched prompts. "
            "Do not infer model superiority from this sample alone. "
            "Output-token counts differ; latency medians are end-to-end "
            "generate wall times, not kernel benchmarks."
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smol", type=Path, required=True)
    p.add_argument("--qwen", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("artifacts/baseline_comparison.json"))
    a = p.parse_args()
    if a.output.exists():
        p.error(f"Refusing to overwrite {a.output}")
    report = compare(_load(a.smol, EXPECTED["smol"]),
                     _load(a.qwen, EXPECTED["qwen"]))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({
        "smol": report["smol"], "qwen": report["qwen"],
        "paired_cases": len(report["per_case"]),
        "warnings": report["comparability_warnings"],
        "output": str(a.output),
    }, indent=2))


if __name__ == "__main__":
    main()
