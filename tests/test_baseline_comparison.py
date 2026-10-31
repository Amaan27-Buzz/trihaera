"""Offline regression checks: fair model pairing and profiling controls."""
from copy import deepcopy
import pytest
from scripts.compare_baselines import compare
from scripts.profile_inference import run_profile


def item(i, verdict, *, category="test", text=None, latency=.5, output=10):
    return {
        "id": i, "category": category,
        "chat_messages": text if text is not None else [{"role":"user", "content":i}],
        "raw_answer": i, "evaluation": {"verdict": verdict},
        "time_total_seconds": latency, "generated_tokens": output,
        "peak_cuda_allocated_bytes": 1048576,
    }


def report(model, rows):
    return {
        "kind":"model_first_unmodified_reference_diagnostic",
        "status":"complete", "model":model, "decoding":"greedy",
        "max_new_tokens":96, "seed":42, "gpu":"same",
        "results":rows,
    }


def test_shared_prompts_are_compared_and_manual_excluded():
    a = report("smol", [
        item("a", "pass"), item("b", "fail"), item("c", "review"),
        item("needle_512", "pass", text=[{"role":"user","content":"Smol filler"}]),
    ])
    b = report("qwen", [
        item("a", "fail"), item("b", "pass"), item("c", "review"),
        item("needle_512", "pass", text=[{"role":"user","content":"Qwen filler"}]),
    ])
    result = compare(a, b)
    assert len(result["per_case"]) == 3
    assert result["smol"]["scored"] == 2
    assert result["qwen"]["scored"] == 2
    assert result["smol"]["passed"] == result["qwen"]["passed"] == 1
    assert "needle_512" in result["unpaired_smol"]
    assert any("needle_512" in warning for warning in result["comparability_warnings"])


def test_different_decoding_is_flagged():
    a = report("smol", [item("a","fail")])
    b = report("qwen", [item("a","pass")])
    b["decoding"] = "recommended"
    assert any("decoding" in x for x in compare(a,b)["comparability_warnings"])


def test_no_common_cases_fails():
    with pytest.raises(ValueError, match="No common"):
        compare(report("smol", [item("a", "pass")]),
                report("qwen", [item("b", "pass")]))


def test_profile_rejects_invalid_sizing_without_torch():
    with pytest.raises(ValueError):
        run_profile(None, None, None, "cpu", targets=(10,), repeats=3)
    with pytest.raises(ValueError):
        run_profile(None, None, None, "cpu", targets=(512,), repeats=0)


def test_benchmark_files_contain_original_raw_output_only():
    result = compare(
        report("smol", [item("a", "fail")]),
        report("qwen", [item("a", "pass")]),
    )
    assert result["per_case"][0]["smol"]["raw_answer"] == "a"
    assert result["per_case"][0]["qwen"]["raw_answer"] == "a"
    assert result["smol"]["passed"] == 0
