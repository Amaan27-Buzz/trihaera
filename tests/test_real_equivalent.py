"""Exact realification controls for persistent complex classical states."""

import copy

import pytest
import torch

from trihaera.persistent import PersistentStep, PersistentStateAdapterLM
from trihaera.real_equivalent import (
    MatchedRealOrthogonalMixer, RealEquivalentMixer, RealEquivalentStep,
)
from test_persistent import TinyLM


@pytest.mark.parametrize(
    "complex_mode,real_mode",
    [
        ("mixed", "real_equivalent"),
        ("no_mixing", "real_equivalent_no_mixing"),
    ],
)
def test_realification_step_forward_and_gradients(complex_mode, real_mode):
    torch.manual_seed(80)
    complex_step = PersistentStep(13, 7, complex_mode)
    real_step = RealEquivalentStep(13, 7, real_mode)
    real_step.load_state_dict(complex_step.state_dict(), strict=True)
    complex_state = real_state = None
    for _ in range(4):
        hidden = torch.randn(2, 9, 13)
        complex_state = complex_step(hidden, complex_state)
        real_state = real_step(hidden, real_state)
        torch.testing.assert_close(
            real_state,
            torch.cat((complex_state.real, complex_state.imag), dim=2),
            atol=2e-6, rtol=2e-5,
        )
    complex_state.abs().square().sum().backward()
    real_state.square().sum().backward()
    for (left_name, left), (right_name, right) in zip(
        complex_step.named_parameters(), real_step.named_parameters()
    ):
        assert left_name == right_name
        if left.requires_grad:
            torch.testing.assert_close(
                left.grad, right.grad, atol=2e-5, rtol=1e-4,
            )


@pytest.mark.parametrize(
    "complex_mode,real_mode",
    [
        ("mixed", "real_equivalent"),
        ("no_mixing", "real_equivalent_no_mixing"),
    ],
)
def test_pretrained_outputs_and_gradients_match(complex_mode, real_mode):
    torch.manual_seed(81)
    original = TinyLM()
    complex_model = PersistentStateAdapterLM(
        copy.deepcopy(original), rank=5, mode=complex_mode
    )
    real_model = PersistentStateAdapterLM(
        copy.deepcopy(original), rank=5, mode=real_mode
    )
    real_model.load_state_dict(complex_model.state_dict(), strict=True)
    with torch.no_grad():
        complex_model.readout.weight.normal_(0, 0.12)
        real_model.readout.weight.copy_(complex_model.readout.weight)
    tokens = torch.randint(0, 31, (2, 7))
    complex_prediction = complex_model(input_ids=tokens, labels=tokens)
    real_prediction = real_model(input_ids=tokens, labels=tokens)
    torch.testing.assert_close(
        complex_prediction.logits, real_prediction.logits,
        atol=3e-6, rtol=1e-5,
    )
    complex_prediction.loss.backward()
    real_prediction.loss.backward()
    for (left_name, left), (right_name, right) in zip(
        complex_model.named_parameters(), real_model.named_parameters()
    ):
        assert left_name == right_name
        if left.requires_grad:
            torch.testing.assert_close(
                left.grad, right.grad, atol=2e-5, rtol=1e-4,
            )


def test_real_orthogonal_initialized_at_complex_mixer():
    real = RealEquivalentMixer()
    general = MatchedRealOrthogonalMixer()
    torch.testing.assert_close(
        real.matrix(), general.matrix(), atol=1e-6, rtol=1e-6,
    )
    matrix = general.matrix()
    torch.testing.assert_close(
        matrix.T @ matrix, torch.eye(6), atol=1e-6, rtol=1e-6,
    )


def test_real_matched_orthogonal_starts_equal_and_receives_gradient():
    torch.manual_seed(85)
    complex_model = PersistentStateAdapterLM(
        TinyLM(), rank=5, mode="real_equivalent"
    )
    generalized = PersistentStateAdapterLM(
        copy.deepcopy(complex_model.backbone),
        rank=5, mode="real_matched_orthogonal",
    )
    for left, right in zip(complex_model.steps, generalized.steps):
        right.drive.load_state_dict(left.drive.state_dict())
        right.transform.load_state_dict(left.transform.state_dict())
        right.phases.data.copy_(left.phases.data)
        right.gains.data.copy_(left.gains.data)
    with torch.no_grad():
        complex_model.readout.weight.normal_(0, 0.1)
        generalized.readout.weight.copy_(complex_model.readout.weight)
    tokens = torch.randint(0, 31, (2, 7))
    torch.testing.assert_close(
        complex_model(input_ids=tokens).logits,
        generalized(input_ids=tokens).logits,
        atol=3e-6, rtol=1e-5,
    )
    generalized(input_ids=tokens, labels=tokens).loss.backward()
    assert generalized.steps[0].mix.raw.grad is not None
    assert torch.isfinite(generalized.steps[0].mix.raw.grad).all()


def test_unmixed_real_controls_freeze_unused_mixer_parameters():
    equivalent = RealEquivalentStep(12, 5, "real_equivalent_no_mixing")
    generalized = RealEquivalentStep(
        12, 5, "real_matched_orthogonal_no_mixing"
    )
    assert not any(p.requires_grad for p in equivalent.mix.parameters())
    assert not any(p.requires_grad for p in generalized.mix.parameters())
