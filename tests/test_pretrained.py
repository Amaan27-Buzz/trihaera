import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from trihaera.pretrained import (StateResidual, DecoderLayerWithStateAdapter,
                                attach_adapters, adapter_parameters)


class FakeDecoderLayer(nn.Module):
    def __init__(self, size, as_tuple=False):
        super().__init__()
        self.linear = nn.Linear(size, size)
        self.as_tuple = as_tuple

    def forward(self, x, **kwargs):
        y = self.linear(x)
        return (y, "metadata") if self.as_tuple else y


class FakeLM(nn.Module):
    def __init__(self, as_tuple=False):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=12)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            FakeDecoderLayer(12, as_tuple=as_tuple) for _ in range(3)
        ])
        self.output = nn.Linear(12, 7)

    def forward(self, x):
        for layer in self.model.layers:
            out = layer(x)
            x = out[0] if isinstance(out, tuple) else out
        return self.output(x)


@pytest.mark.parametrize("mode", ["trihaera", "no_mixing", "real_adapter"])
@pytest.mark.parametrize("as_tuple", [False, True])
def test_initial_conversion_preserves_logits_and_freezes_backbone(mode, as_tuple):
    torch.manual_seed(17)
    original = FakeLM(as_tuple=as_tuple)
    converted = copy.deepcopy(original)
    x = torch.randn(2, 5, 12)
    initial_logits = original(x)
    report = attach_adapters(converted, mode=mode, rank=4)
    torch.testing.assert_close(converted(x), initial_logits, rtol=0, atol=0)
    assert report["trainable_parameters"] == sum(
        p.numel() for p in adapter_parameters(converted)
    )
    assert report["trainable_fraction"] < 0.8
    assert not any(p.requires_grad for p in converted.output.parameters())


@pytest.mark.parametrize("mode", ["trihaera", "no_mixing", "real_adapter"])
def test_adapter_can_train_without_backbone_grad(mode):
    torch.manual_seed(19)
    model = FakeLM()
    attach_adapters(model, rank=4, mode=mode)
    x = torch.randn(2, 6, 12)
    opt = torch.optim.AdamW(list(adapter_parameters(model)), lr=0.03)
    before = model(x).detach().clone()
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss = model(x).square().mean()
        loss.backward()
        opt.step()
    after = model(x)
    assert not torch.allclose(before, after)
    assert all(p.grad is None for p in model.output.parameters())
    assert all(torch.isfinite(p).all() for p in adapter_parameters(model))


def test_interference_operator_eventually_receives_gradient():
    torch.manual_seed(20)
    model = FakeLM()
    attach_adapters(model, rank=4, mode="trihaera")
    opt = torch.optim.AdamW(list(adapter_parameters(model)), lr=0.05)
    x = torch.randn(2, 5, 12)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        model(x).square().mean().backward()
        opt.step()
    opt.zero_grad(set_to_none=True)
    model(x).square().mean().backward()
    mix_grad = model.model.layers[0].adapter.mix.symmetric.grad
    assert mix_grad is not None and mix_grad.abs().sum() > 0


def test_second_attachment_rejected():
    model = FakeLM()
    attach_adapters(model)
    with pytest.raises(ValueError):
        attach_adapters(model)


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        StateResidual(8, 4, "something_else")


def test_unsupported_output_does_not_silently_continue():
    class BadLayer(nn.Module):
        def forward(self, x):
            return {"hidden_states": x}
    wrapper = DecoderLayerWithStateAdapter(BadLayer(), StateResidual(8))
    with pytest.raises(TypeError):
        wrapper(torch.zeros(1, 3, 8))


@pytest.mark.parametrize("mode", ["trihaera", "no_mixing", "real_adapter"])
def test_adapter_residual_scale_and_exact_initial_identity(mode):
    torch.manual_seed(33)
    original = FakeLM()
    model = copy.deepcopy(original)
    input_values = torch.randn(2, 4, 12)
    report = attach_adapters(model, rank=4, mode=mode, residual_scale=0.1)
    assert report["residual_scale"] == 0.1
    torch.testing.assert_close(model(input_values), original(input_values),
                               atol=0, rtol=0)
    for wrapper in model.model.layers:
        assert wrapper.adapter.residual_scale == 0.1


def test_inactive_mixer_is_not_trainable_or_counted():
    model = FakeLM()
    stats = attach_adapters(model, mode="no_mixing")
    for wrapper in model.model.layers:
        assert not any(p.requires_grad for p in wrapper.adapter.mix.parameters())
    assert sum(p.numel() for p in adapter_parameters(model)) == stats["trainable_parameters"]


def test_invalid_residual_scale_rejected():
    for value in [0.0, -0.01, 1.1]:
        with pytest.raises(ValueError):
            StateResidual(8, 4, residual_scale=value)


def test_residual_scale_controls_amplitude():
    torch.manual_seed(33)
    small = StateResidual(12, 4, mode="real_adapter", residual_scale=0.1)
    large = copy.deepcopy(small)
    large.residual_scale = 1.0
    with torch.no_grad():
        small.up.weight.fill_(0.02)
        large.up.weight.copy_(small.up.weight)
    values = torch.randn(2, 4, 12)
    torch.testing.assert_close(10 * small(values), large(values), atol=1e-6, rtol=1e-5)
