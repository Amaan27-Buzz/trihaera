import torch

from trihaera.layers import PhaseInterference


def test_state_mixing_unitary():
    torch.manual_seed(0)
    op = PhaseInterference()
    with torch.no_grad():
        op.symmetric.add_(0.1 * torch.randn(3, 3))
        op.skew.add_(0.1 * torch.randn(3, 3))
    u = op.matrix()
    torch.testing.assert_close(u.conj().T @ u, torch.eye(3, dtype=u.dtype),
                               atol=1e-5, rtol=1e-5)


def test_unitary_mixing_preserves_state_norm():
    op = PhaseInterference()
    x = torch.complex(torch.randn(2, 4, 3, 8), torch.randn(2, 4, 3, 8))
    y = op(x)
    torch.testing.assert_close(x.abs().square().sum(dim=2),
                               y.abs().square().sum(dim=2),
                               atol=1e-5, rtol=1e-5)


def test_identity_ablation():
    op = PhaseInterference(enabled=False)
    x = torch.complex(torch.randn(2, 4, 3, 8), torch.randn(2, 4, 3, 8))
    assert torch.equal(x, op(x))
