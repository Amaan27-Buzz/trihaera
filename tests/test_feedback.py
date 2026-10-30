"""CPU contracts for inter-layer feedback (no HF downloads)."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from trihaera.feedback import FeedbackCausalLM, FeedbackDecoderLayer, feedback_parameters


class CausalLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Linear(width, width)

    def forward(self, hidden, **kwargs):
        return torch.tanh(self.weight(hidden))


class TinyLM(nn.Module):
    def __init__(self, width=12, depth=3, vocab=31):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=width)
        self.embedding = nn.Embedding(vocab, width)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([CausalLayer(width) for _ in range(depth)])
        self.lm_head = nn.Linear(width, vocab, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True):
        h = self.embedding(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        return SimpleNamespace(logits=self.lm_head(h))


@pytest.mark.parametrize("mode", ["mixed", "no_mixing", "real_matched_orthogonal"])
def test_zero_feedback_preserves_backbone_exactly(mode):
    torch.manual_seed(110)
    base = TinyLM()
    original = copy.deepcopy(base)
    tokens = torch.randint(0, 31, (2, 7))
    model = FeedbackCausalLM(base, rank=5, mode=mode)
    actual = model(input_ids=tokens, labels=tokens)
    torch.testing.assert_close(actual.logits, original(tokens).logits, rtol=0, atol=0)
    assert torch.isfinite(actual.loss)
    assert all(not p.requires_grad for p in model.backbone.embedding.parameters())
    assert len(list(feedback_parameters(model))) > 0
    assert model.carrier.state is None and not model.carrier.active


@pytest.mark.parametrize("mode", ["mixed", "no_mixing", "real_matched_orthogonal"])
def test_training_changes_output_but_not_backbone(mode):
    torch.manual_seed(112)
    model = FeedbackCausalLM(TinyLM(), rank=5, mode=mode)
    originals = {
        n: p.detach().clone() for n, p in model.backbone.named_parameters()
        if ".original." in n or n.startswith(("embedding.", "lm_head."))
    }
    tokens = torch.randint(0, 31, (2, 7))
    before = model(input_ids=tokens).logits.detach().clone()
    optimizer = torch.optim.AdamW(list(feedback_parameters(model)), lr=0.01)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        model(input_ids=tokens, labels=tokens).loss.backward()
        optimizer.step()
    after = model(input_ids=tokens).logits
    assert not torch.allclose(before, after)
    for name, param in model.backbone.named_parameters():
        if name in originals:
            torch.testing.assert_close(param, originals[name], rtol=0, atol=0)
    assert model.carrier.state is None


def test_early_feedback_changes_next_pretrained_layer_input():
    torch.manual_seed(115)
    model = FeedbackCausalLM(TinyLM(), rank=5)
    tokens = torch.randint(0, 31, (1, 7))
    observed = []
    hook = model.backbone.model.layers[1].original.register_forward_pre_hook(
        lambda _, args: observed.append(args[0].detach().clone())
    )
    model(input_ids=tokens)
    with torch.no_grad():
        model.backbone.model.layers[0].to_hidden.weight.normal_(0, 0.1)
    model(input_ids=tokens)
    hook.remove()
    assert len(observed) == 2
    assert not torch.allclose(observed[0], observed[1])


def test_early_mixer_gradient_after_feedback_warmup():
    torch.manual_seed(118)
    model = FeedbackCausalLM(TinyLM(), rank=5, mode="mixed")
    optimizer = torch.optim.AdamW(list(feedback_parameters(model)), lr=0.02)
    tokens = torch.randint(0, 31, (2, 7))
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        model(input_ids=tokens, labels=tokens).loss.backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model(input_ids=tokens, labels=tokens).loss.backward()
    grad = model.backbone.model.layers[0].state_step.mix.symmetric.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_prefix_causality_and_no_state_leakage():
    torch.manual_seed(120)
    model = FeedbackCausalLM(TinyLM(), rank=5)
    with torch.no_grad():
        for layer in model.backbone.model.layers:
            layer.to_hidden.weight.normal_(0, 0.1)
    tokens = torch.randint(0, 31, (1, 8))
    changed = tokens.clone()
    changed[:, 5:] = (changed[:, 5:] + 2) % 31
    with torch.no_grad():
        a = model(input_ids=tokens).logits[:, :5]
        b = model(input_ids=changed).logits[:, :5]
        c = model(input_ids=tokens).logits[:, :5]
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(a, c, rtol=0, atol=0)
    assert model.carrier.state is None


def test_errors_reset_state_and_do_not_double_wrap():
    with pytest.raises(ValueError):
        FeedbackCausalLM(TinyLM(), mode="other")
    model = FeedbackCausalLM(TinyLM())
    tokens = torch.randint(0, 31, (1, 7))
    with pytest.raises(ValueError, match="KV cache"):
        model(input_ids=tokens, use_cache=True)
    with pytest.raises(TypeError, match="Unsupported"):
        model(input_ids=tokens, output_hidden_states=True)
    with pytest.raises(ValueError, match="Labels"):
        model(input_ids=tokens, labels=tokens[:, :3])
    assert not model.carrier.active and model.carrier.state is None
    model(input_ids=tokens)
    with pytest.raises(ValueError):
        FeedbackCausalLM(model.backbone)
