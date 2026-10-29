import pytest

from trihaera.experiment_stats import paired_summary


def test_paired_differences_across_identical_seeds():
    runs = {
        "17": {
            "no_mixing": {"final_validation_loss": 3.2, "training_elapsed_seconds": 3.0},
            "trihaera": {"final_validation_loss": 3.3, "training_elapsed_seconds": 5.0}
        },
        "23": {
            "no_mixing": {"final_validation_loss": 3.4, "training_elapsed_seconds": 3.2},
            "trihaera": {"final_validation_loss": 3.3, "training_elapsed_seconds": 5.2}
        },
        "42": {
            "no_mixing": {"final_validation_loss": 3.5, "training_elapsed_seconds": 3.0},
            "trihaera": {"final_validation_loss": 3.4, "training_elapsed_seconds": 5.1}
        }
    }
    summary = paired_summary(runs)
    assert summary["num_seeds"] == 3
    tri = summary["modes"]["trihaera"]
    assert tri["mean_final_validation_loss"] == pytest.approx(3.3333333333)
    assert tri["mean_difference_from_reference"] == pytest.approx(-0.0333333333)
    assert tri["std_paired_difference"] > 0
    assert tri["mean_train_seconds"] == pytest.approx(5.1)
    assert len(tri["paired_differences_by_seed"]) == 3


def test_no_reference_rejected():
    with pytest.raises(ValueError, match="reference"):
        paired_summary({"1": {"real_adapter": {
            "final_validation_loss": 3.5, "training_elapsed_seconds": 3.0
        }}})


def test_single_seed_has_no_deceptive_standard_deviation():
    x = paired_summary({"17": {"no_mixing": {
        "final_validation_loss": 3.2, "training_elapsed_seconds": 3.0
    }}})
    assert x["modes"]["no_mixing"]["std_final_validation_loss"] is None
    assert x["modes"]["no_mixing"]["mean_difference_from_reference"] == 0
