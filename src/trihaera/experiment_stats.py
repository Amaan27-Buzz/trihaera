"""Descriptive paired experimental statistics; does NOT perform significance testing."""

import math
import statistics


def paired_summary(runs: dict, *, reference: str = "no_mixing") -> dict:
    """Summarize final validation NLL and paired differences across shared seeds.

    Input: {"17": {"no_mixing": {"final_validation_loss": ...,
                                "training_elapsed_seconds": ...}, ...}, ...}
    Mean and sample standard deviation are only descriptive with few seeds.
    """
    if not runs:
        raise ValueError("No runs to summarize")
    modes = set.intersection(*(set(group) for group in runs.values()))
    if reference not in modes:
        raise ValueError("Missing reference mode for paired comparisons")
    result = {"reference": reference, "num_seeds": len(runs), "modes": {}}
    for mode in sorted(modes):
        values = []
        differences = []
        times = []
        for by_mode in runs.values():
            record = by_mode[mode]
            baseline = by_mode[reference]
            loss = float(record["final_validation_loss"])
            diff = loss - float(baseline["final_validation_loss"])
            seconds = float(record["training_elapsed_seconds"])
            if not all(math.isfinite(v) for v in (loss, diff, seconds)):
                raise ValueError("Nonfinite measurement")
            values.append(loss)
            differences.append(diff)
            times.append(seconds)
        result["modes"][mode] = {
            "mean_final_validation_loss": statistics.mean(values),
            "std_final_validation_loss": statistics.stdev(values) if len(values) > 1 else None,
            "mean_difference_from_reference": statistics.mean(differences),
            "std_paired_difference": statistics.stdev(differences) if len(values) > 1 else None,
            "paired_differences_by_seed": differences,
            "mean_train_seconds": statistics.mean(times),
        }
    result["warning"] = (
        "Descriptive paired statistics only. Few seeds, short training, "
        "unequal FLOPs and possible upstream pretraining corpus overlap. "
        "No confidence claim or architecture proof."
    )
    return result
