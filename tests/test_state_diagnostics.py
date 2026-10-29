import math

import torch

from trihaera import ModelConfig, TrihaeraLM
from trihaera.diagnostics import probe_states, state_statistics
from trihaera.layers import ComplexCausalSelfAttention, PhaseInterference


def test_initial_phase_copies_are_rank_one():
    torch.manual_seed(12)
    x = torch.randn(2, 6, 16)
    angles = torch.tensor([-0.6, 0.0, 0.6])
    z = x.unsqueeze(2).to(torch.complex64) * torch.polar(
        torch.ones(3), angles
    ).view(1, 1, 3, 1) / math.sqrt(3)
    stats = state_statistics(z)
    assert abs(stats["effective_rank"] - 1.0) < 1e-3
    assert stats["mean_pairwise_overlap"] > 0.999


def test_independent_state_vectors_are_rank_three():
    z = torch.zeros((1, 1, 3, 3), dtype=torch.complex64)
    z[0, 0] = torch.eye(3, dtype=torch.complex64)
    stats = state_statistics(z)
    assert abs(stats["effective_rank"] - 3.0) < 1e-5


def test_unitary_mixer_cannot_increase_linear_state_rank():
    torch.manual_seed(8)
    x = torch.randn(2, 5, 12)
    angles = torch.tensor([-0.6, 0.0, 0.6])
    z = x.unsqueeze(2).to(torch.complex64) * torch.polar(
        torch.ones(3), angles
    ).view(1, 1, 3, 1)
    before = state_statistics(z)
    after = state_statistics(PhaseInterference()(z))
    assert abs(before["effective_rank"] - after["effective_rank"]) < 1e-3


def test_attention_is_covariant_under_global_channel_phases():
    torch.manual_seed(6)
    attention = ComplexCausalSelfAttention(width=16, heads=4, dropout=0).eval()
    z = torch.complex(torch.randn(2, 5, 3, 16), torch.randn(2, 5, 3, 16))
    phases = torch.polar(torch.ones(3), torch.tensor([-0.7, 0.1, 1.2]))
    rotation = phases.view(1, 1, 3, 1)
    with torch.no_grad():
        lhs = attention(z * rotation)
        rhs = attention(z) * rotation
    torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)


def test_probe_records_intermediate_layers_without_mutating_model():
    cfg = ModelConfig(vocab_size=32, d_model=16, n_layers=2, n_heads=4,
                      max_seq_len=8, ffn_multiplier=2)
    model = TrihaeraLM(cfg).train()
    tokens = torch.randint(0, cfg.vocab_size, (2, 6))
    probe = probe_states(model, tokens)
    assert model.training is True
    assert [row["location"] for row in probe["statistics"]] == [
        "input", "block_1", "block_2"
    ]
    assert abs(probe["statistics"][0]["effective_rank"] - 1) < 1e-3
    assert probe["logits"].shape == (2, 6, cfg.vocab_size)


def test_state_statistics_rejects_non_complex_input():
    try:
        state_statistics(torch.zeros(1, 2, 3, 4))
    except ValueError:
        return
    assert False, "Expected ValueError"
