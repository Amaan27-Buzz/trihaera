"""Six-real-channel implementations of the complex persistent sidecar.

These modules expose an exact (within floating-point error) real-valued
representation of complex TRIHAERA operations. Not a quantum speedup.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class RealEquivalentMixer(nn.Module):
    """Realification of exp(iG) using a real 6x6 matrix exponential.

    G=A+iB, A symmetric, B skew symmetric. exp(iG) is represented by
    exp([[-B,-A],[A,-B]]) in [Re(z0:3),Im(z0:3)] ordering.
    """

    def __init__(self, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled
        self.symmetric = nn.Parameter(torch.zeros(3, 3))
        self.skew = nn.Parameter(torch.zeros(3, 3))
        with torch.no_grad():
            self.symmetric[0, 1] = 0.12
            self.symmetric[1, 2] = 0.12
            self.symmetric[2, 0] = -0.08
        if not enabled:
            self.symmetric.requires_grad_(False)
            self.skew.requires_grad_(False)

    def generator(self) -> torch.Tensor:
        a = (self.symmetric + self.symmetric.T) / 2
        b = (self.skew - self.skew.T) / 2
        return torch.cat((torch.cat((-b, -a), dim=-1),
                          torch.cat((a, -b), dim=-1)), dim=0)

    def matrix(self) -> torch.Tensor:
        return torch.matrix_exp(self.generator())

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.is_complex() or state.ndim != 4 or state.shape[-2] != 6:
            raise ValueError("Expected six real channels [B,T,6,R]")
        if not self.enabled:
            return state
        return torch.einsum("oi,btir->btor", self.matrix(), state)


class MatchedRealOrthogonalMixer(nn.Module):
    """Full real O(6) mixer initialized at realification of complex U(3)."""

    def __init__(self, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled
        initial = RealEquivalentMixer().generator().detach()
        self.raw = nn.Parameter(initial.clone())
        if not enabled:
            self.raw.requires_grad_(False)

    def matrix(self) -> torch.Tensor:
        skew = (self.raw - self.raw.T) / 2
        return torch.matrix_exp(skew)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.is_complex() or state.ndim != 4 or state.shape[-2] != 6:
            raise ValueError("Expected six real channels [B,T,6,R]")
        return (torch.einsum("oi,btir->btor", self.matrix(), state)
                if self.enabled else state)


class RealEquivalentStep(nn.Module):
    """Equivalent six-real computation or matched-start O(6) control.

    The equivalent mode has the same parameter names and shapes as the
    complex PersistentStep, permitting strict state_dict conversion.
    The matched orthogonal mode uses otherwise identical state injection
    and nonlinear processing while relaxing U(3) mixing to O(6).
    """

    def __init__(self, width: int, rank: int, mode: str) -> None:
        super().__init__()
        if mode not in (
            "real_equivalent", "real_equivalent_no_mixing",
            "real_matched_orthogonal", "real_matched_orthogonal_no_mixing",
        ):
            raise ValueError(f"Unsupported real-equivalent mode: {mode}")
        self.drive = nn.Linear(width, rank, bias=False)
        self.transform = nn.Linear(rank, rank, bias=False)
        self.phases = nn.Parameter(torch.tensor([-0.6, 0.0, 0.6]))
        self.gains = nn.Parameter(torch.tensor([0.9, 1.0, 1.1]))
        enabled = not mode.endswith("no_mixing")
        self.mix = (MatchedRealOrthogonalMixer(enabled=enabled)
                    if mode.startswith("real_matched_orthogonal")
                    else RealEquivalentMixer(enabled=enabled))
        nn.init.normal_(self.drive.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.transform.weight, mean=0.0, std=0.02)

    def forward(self, hidden: torch.Tensor,
                state: torch.Tensor | None) -> torch.Tensor:
        drive = self.drive(hidden.float())
        factors = self.gains.view(1, 1, 3, 1) / math.sqrt(3)
        real = drive.unsqueeze(2) * torch.cos(self.phases).view(1, 1, 3, 1) * factors
        imag = drive.unsqueeze(2) * torch.sin(self.phases).view(1, 1, 3, 1) * factors
        injected = torch.cat((real, imag), dim=2)
        if state is None:
            state = injected
        else:
            state = 0.8 * state + 0.2 * injected
        state = state + 0.2 * F.silu(self.transform(state))
        return self.mix(state)
