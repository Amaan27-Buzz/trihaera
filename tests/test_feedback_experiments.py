"""CPU correctness checks for common feedback training protocol."""
import copy
import pytest
import torch

from trihaera.feedback import FeedbackCausalLM, feedback_parameters
from trihaera.feedback_experiments import (
    sample_schedule, schedule_digest, evaluate_nll, train_feedback)
from test_feedback import TinyLM


def test_schedule_is_deterministic():
    a = sample_schedule(50, 12, 2, seed=42)
    assert a == sample_schedule(50, 12, 2, seed=42)
    assert a != sample_schedule(50, 12, 2, seed=43)
    assert schedule_digest(a) == schedule_digest(sample_schedule(50, 12, 2, 42))


@pytest.mark.parametrize("mode", ["mixed", "no_mixing", "real_matched_orthogonal"])
def test_train_and_eval_frozen_feedback(mode):
    torch.manual_seed(138)
    base = TinyLM()
    model = FeedbackCausalLM(copy.deepcopy(base), mode=mode, rank=5)
    windows = torch.randint(0, 31, (12, 8))
    torch.testing.assert_close(model(input_ids=windows[:3]).logits,
                               base(input_ids=windows[:3]).logits, atol=0, rtol=0)
    pre = evaluate_nll(model, windows, batch_size=3, device=torch.device("cpu"))
    weights = {n:p.detach().clone() for n,p in model.backbone.named_parameters()
               if ".original." in n or n.startswith(("embedding.", "lm_head."))}
    schedule = sample_schedule(12, 5, 2, seed=11)
    out = train_feedback(
        model, list(feedback_parameters(model)), windows, schedule,
        device=torch.device("cpu"), learning_rate=3e-3, warmup=1)
    assert out["train_steps"] == 5
    assert out["training_tokens_approx"] == 70
    assert out["sampling_sha256"] == schedule_digest(schedule)
    assert out["step_time_median_seconds"] > 0
    assert out["training_tokens_per_second"] > 0
    post = evaluate_nll(model, windows, batch_size=3, device=torch.device("cpu"))
    assert abs(pre - post) > 1e-5
    assert model.carrier.state is None and not model.carrier.active
    for name, original in weights.items():
        torch.testing.assert_close(model.backbone.get_parameter(name), original,
                                   atol=0, rtol=0)


def test_invalid_batches_and_evaluation():
    with pytest.raises(ValueError):
        sample_schedule(0, 5, 1, 1)
    with pytest.raises(ValueError):
        sample_schedule(5, 0, 1, 1)
    with pytest.raises(ValueError):
        evaluate_nll(TinyLM(), torch.empty(0, 1, dtype=torch.long),
                     batch_size=1, device=torch.device("cpu"))
    model = FeedbackCausalLM(TinyLM(), rank=4)
    with pytest.raises(ValueError):
        train_feedback(model, list(feedback_parameters(model)),
                       torch.randint(0,31,(5,8)), [[10]],
                       device=torch.device("cpu"), learning_rate=3e-4)
