"""Explicit classical simulation of three complex amplitude channels.

Nonlinear Transformer operations and classical tensors mean this is not
quantum computing or an implementation of physical quantum superposition.
Attention and FFN projection weights are shared across state channels.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ComplexRMSNorm(nn.Module):
    """Shared scale based on mean |z|^2, independent of global phase."""

    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mean_square = (z.real.square() + z.imag.square()).mean(-1, keepdim=True)
        return z * torch.rsqrt(mean_square + self.eps) * self.weight


def shared_complex_linear(layer: nn.Linear, z: torch.Tensor) -> torch.Tensor:
    """Same real matrix acts on real/imag components of every state."""
    if layer.bias is not None:
        raise ValueError("Complex shared projections must be bias-free")
    return torch.complex(
        F.linear(z.real, layer.weight),
        F.linear(z.imag, layer.weight),
    )


class PhaseInterference(nn.Module):
    """A unitary 3x3 mixing operator, NOT a unitary Transformer block."""

    def __init__(self, states: int = 3, enabled: bool = True) -> None:
        super().__init__()
        if states != 3:
            raise ValueError("Only three-state mixing is implemented")
        self.enabled = enabled
        self.symmetric = nn.Parameter(torch.zeros(states, states))
        self.skew = nn.Parameter(torch.zeros(states, states))
        with torch.no_grad():
            self.symmetric[0, 1] = 0.12
            self.symmetric[1, 2] = 0.12
            self.symmetric[2, 0] = -0.08

    def matrix(self) -> torch.Tensor:
        a = (self.symmetric + self.symmetric.T) / 2
        b = (self.skew - self.skew.T) / 2
        generator = torch.complex(a, b)  # Hermitian matrix
        return torch.matrix_exp(1j * generator)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return z
        if z.shape[-2] != 3:
            raise ValueError("Expected a three-state penultimate axis")
        mix = self.matrix()
        return torch.einsum("oi,btid->btod", mix, z)


class ComplexCausalSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.dropout = dropout

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Complex tensor [batch, tokens, states, width]
        b, t, s, width = z.shape
        qkv = shared_complex_linear(self.qkv, z).permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        def packed(tensor: torch.Tensor) -> torch.Tensor:
            tensor = tensor.reshape(b * s, t, self.n_heads, self.head_width)
            tensor = tensor.transpose(1, 2)
            return torch.cat((tensor.real, tensor.imag), dim=-1)

        # Dot of concatenated real/imag parts == Re(q * conjugate(k)).
        # A single real attention distribution is applied to complex V.
        q2, k2, v2 = packed(q), packed(k), packed(v)
        y = F.scaled_dot_product_attention(
            q2, k2, v2, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
            scale=1.0 / math.sqrt(self.head_width),
        )
        yr, yi = y.chunk(2, dim=-1)
        y = torch.complex(yr, yi).transpose(1, 2).reshape(b, s, t, width)
        y = y.permute(0, 2, 1, 3)
        return shared_complex_linear(self.out, y)


class ComplexFeedForward(nn.Module):
    def __init__(self, width: int, multiplier: int) -> None:
        super().__init__()
        self.up = nn.Linear(width, width * multiplier, bias=False)
        self.down = nn.Linear(width * multiplier, width, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = shared_complex_linear(self.up, z)
        # Explicitly nonunitary, classical component-wise activation.
        h = torch.complex(F.silu(h.real), F.silu(h.imag))
        return shared_complex_linear(self.down, h)


class ThreeStateBlock(nn.Module):
    def __init__(self, width: int, heads: int, multiplier: int, dropout: float,
                 interference: bool = True) -> None:
        super().__init__()
        self.pre_attn = ComplexRMSNorm(width)
        self.attn = ComplexCausalSelfAttention(width, heads, dropout)
        self.pre_ffn = ComplexRMSNorm(width)
        self.ffn = ComplexFeedForward(width, multiplier)
        self.mix = PhaseInterference(enabled=interference)
        self.phase = nn.Parameter(torch.tensor([-0.2, 0.0, 0.2]))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        phase = torch.polar(torch.ones_like(self.phase), self.phase)
        z = z + phase.view(1, 1, 3, 1) * self.attn(self.pre_attn(z))
        z = z + self.ffn(self.pre_ffn(z))
        return self.mix(z)
