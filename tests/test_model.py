import pytest
import torch

from trihaera import DenseTransformerLM, ModelConfig, TrihaeraLM


@pytest.fixture
def config():
    return ModelConfig(vocab_size=51, d_model=32, n_layers=2, n_heads=4,
                       max_seq_len=16, ffn_multiplier=2)


@pytest.mark.parametrize("architecture", [TrihaeraLM, DenseTransformerLM])
def test_forward_backward(architecture, config):
    torch.manual_seed(123)
    model = architecture(config)
    tokens = torch.randint(0, config.vocab_size, (2, 8))
    out = model(tokens[:, :-1], tokens[:, 1:])
    assert out["logits"].shape == (2, 7, config.vocab_size)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)


def test_interference_gradient_flows(config):
    torch.manual_seed(111)
    model = TrihaeraLM(config)
    tokens = torch.randint(0, config.vocab_size, (2, 8))
    model(tokens[:, :-1], tokens[:, 1:])["loss"].backward()
    g = model.layers[0].mix.symmetric.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_causal_mask(config):
    torch.manual_seed(44)
    for model in [TrihaeraLM(config), DenseTransformerLM(config)]:
        model.eval()
        x = torch.randint(0, config.vocab_size, (1, 8))
        y = x.clone()
        y[0, 5:] = (y[0, 5:] + 3) % config.vocab_size
        with torch.no_grad():
            a = model(x)["logits"][0, :5]
            b = model(y)["logits"][0, :5]
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_batch_target_shape(config):
    model = TrihaeraLM(config)
    x = torch.randint(0, config.vocab_size, (1, 4))
    with pytest.raises(ValueError):
        model(x, x[:, :3])


def test_interference_changes_output(config):
    torch.manual_seed(72)
    with_mix = TrihaeraLM(config, interference=True).eval()
    torch.manual_seed(72)
    no_mix = TrihaeraLM(config, interference=False).eval()
    x = torch.randint(0, config.vocab_size, (1, 7))
    with torch.no_grad():
        a = with_mix(x)["logits"]
        b = no_mix(x)["logits"]
    assert not torch.allclose(a, b, atol=1e-6, rtol=1e-6)
