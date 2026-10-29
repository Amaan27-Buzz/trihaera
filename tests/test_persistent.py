import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from trihaera.persistent import (
    PersistentStateAdapterLM, persistent_parameters, persistent_parameter_report,
)


class TinyLM(nn.Module):
    """Minimal causal HF-like model interface for CPU correctness tests."""

    def __init__(self, width=12, depth=3, vocab=31):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=width)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            nn.Linear(width, width) for _ in range(depth)
        ])
        self.embedding = nn.Embedding(vocab, width)
        self.lm_head = nn.Linear(width, vocab, bias=False)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False,
                use_cache=False, return_dict=True):
        h = self.embedding(input_ids)
        saved = [h]
        for layer in self.model.layers:
            h = torch.tanh(layer(h))
            saved.append(h)
        return SimpleNamespace(
            logits=self.lm_head(h), hidden_states=tuple(saved),
        )


@pytest.mark.parametrize("mode", ["mixed", "no_mixing"])
def test_exact_initial_logits_and_frozen_backbone(mode):
    torch.manual_seed(41)
    base = TinyLM()
    x = torch.randint(0, 31, (2, 6))
    original = base(input_ids=x).logits.detach().clone()
    wrapped = PersistentStateAdapterLM(base, rank=5, mode=mode)
    out = wrapped(input_ids=x, labels=x)
    torch.testing.assert_close(out.logits, original, rtol=0, atol=0)
    assert torch.isfinite(out.loss)
    assert not any(p.requires_grad for p in wrapped.backbone.parameters())
    assert len(wrapped.steps) == len(base.model.layers)
    assert persistent_parameter_report(wrapped)["trainable_parameters"] > 0


def test_backbone_stays_frozen_after_training():
    torch.manual_seed(41)
    base = TinyLM()
    before = copy.deepcopy(base.state_dict())
    wrapped = PersistentStateAdapterLM(base, rank=5)
    opt = torch.optim.AdamW(list(persistent_parameters(wrapped)), lr=0.01)
    x = torch.randint(0, 31, (2, 7))
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        wrapped(input_ids=x, labels=x).loss.backward()
        opt.step()
    with torch.no_grad():
        assert (
            wrapped(input_ids=x).logits - base(input_ids=x).logits
        ).abs().max() > 0
    for name, original in before.items():
        torch.testing.assert_close(
            base.state_dict()[name], original, rtol=0, atol=0,
        )


def test_mixer_gradients_after_readout_warms_up():
    torch.manual_seed(43)
    wrapped = PersistentStateAdapterLM(TinyLM(), rank=5, mode="mixed")
    opt = torch.optim.AdamW(list(persistent_parameters(wrapped)), lr=0.01)
    x = torch.randint(0, 31, (2, 7))
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        wrapped(input_ids=x, labels=x).loss.backward()
        opt.step()
    opt.zero_grad(set_to_none=True)
    wrapped(input_ids=x, labels=x).loss.backward()
    grad = wrapped.steps[0].mix.symmetric.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_mixing_changes_output():
    torch.manual_seed(44)
    mixed = PersistentStateAdapterLM(TinyLM(depth=3), rank=5, mode="mixed")
    plain = PersistentStateAdapterLM(
        copy.deepcopy(mixed.backbone), rank=5, mode="no_mixing",
    )
    plain.load_state_dict(mixed.state_dict(), strict=True)
    with torch.no_grad():
        mixed.readout.weight.normal_(0, 0.1)
        plain.readout.weight.copy_(mixed.readout.weight)
        for m, p in zip(mixed.steps, plain.steps):
            m.mix.symmetric.mul_(3.0)
            p.mix.symmetric.copy_(m.mix.symmetric)
    x = torch.randint(0, 31, (2, 6))
    assert not torch.allclose(
        mixed(input_ids=x).logits, plain(input_ids=x).logits,
    )
    assert not any(
        p.requires_grad for step in plain.steps
        for p in step.mix.parameters()
    )


def test_loss_causal_shift_and_zero_residual_parity():
    torch.manual_seed(45)
    base = TinyLM(depth=2)
    wrapped = PersistentStateAdapterLM(base, rank=5)
    x = torch.randint(0, 31, (2, 8))
    out = wrapped(input_ids=x, labels=x)
    expected = nn.functional.cross_entropy(
        out.logits[:, :-1].reshape(-1, 31), x[:, 1:].reshape(-1),
    )
    torch.testing.assert_close(out.loss, expected)
    with pytest.raises(ValueError, match="KV cache"):
        wrapped(input_ids=x, use_cache=True)


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        PersistentStateAdapterLM(TinyLM(), rank=0)
    with pytest.raises(ValueError):
        PersistentStateAdapterLM(TinyLM(), mode="broken")
    with pytest.raises(ValueError):
        PersistentStateAdapterLM(TinyLM(), residual_scale=0)


def test_earlier_mixer_affects_final_readout_when_last_mixer_same():
    torch.manual_seed(103)
    wrapped = PersistentStateAdapterLM(TinyLM(depth=3), rank=5)
    different = copy.deepcopy(wrapped)
    with torch.no_grad():
        wrapped.readout.weight.normal_(0, 0.2)
        different.readout.weight.copy_(wrapped.readout.weight)
        different.steps[0].mix.enabled = False  # Keep final mixer unchanged.
    x = torch.randint(0, 31, (2, 7))
    assert not torch.allclose(
        wrapped(input_ids=x).logits, different(input_ids=x).logits,
        atol=1e-6,
    )


def test_causal_prefix_consistency():
    torch.manual_seed(105)
    wrapped = PersistentStateAdapterLM(TinyLM(depth=2), rank=4)
    with torch.no_grad():
        wrapped.readout.weight.normal_(0, 0.1)
    tokens = torch.randint(0, 31, (1, 7))
    changed = tokens.clone()
    changed[:, 5:] = (changed[:, 5:] + 2) % 31
    with torch.no_grad():
        a = wrapped(input_ids=tokens).logits[:, :5]
        b = wrapped(input_ids=changed).logits[:, :5]
    torch.testing.assert_close(a, b, atol=0, rtol=0)
