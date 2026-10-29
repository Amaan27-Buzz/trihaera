"""Low-risk pretrained-weight bridge for TRIHAERA architectural experiments.

This IS NOT the end-to-end complex-valued TRIHAERA decoder: pretrained Llama
blocks stay intact; three-state computations enter as learnable residual
adapters between blocks. Their near-parameter-matched real adapter serves as a
necessary control. No pretrained backbone weights are copied or overwritten.
"""

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .layers import PhaseInterference

AdapterMode = Literal["trihaera", "no_mixing", "real_adapter"]


class StateResidual(nn.Module):
    """Shared low-rank projections with optional three-state complex processing.

    State readout is initialized to zero to preserve original pretrained logits.
    There is no recurrent state across blocks; this is an adaptation pilot.
    """

    def __init__(self, hidden_size: int, rank: int = 16,
                 mode: AdapterMode = "trihaera", residual_scale: float = 0.1) -> None:
        super().__init__()
        if hidden_size < 1 or rank < 1:
            raise ValueError("hidden_size and rank must be positive")
        if mode not in ("trihaera", "no_mixing", "real_adapter"):
            raise ValueError(f"Unknown adapter mode: {mode}")
        if not (0 < residual_scale <= 1):
            raise ValueError("residual_scale must be in (0, 1]")
        self.mode = mode
        self.residual_scale = float(residual_scale)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)
        if mode != "real_adapter":
            self.phases = nn.Parameter(torch.tensor([-0.6, 0.0, 0.6]))
            self.phase_step = nn.Parameter(torch.tensor([-0.2, 0.0, 0.2]))
            self.mix = PhaseInterference(enabled=mode == "trihaera")
            if mode == "no_mixing":
                for parameter in self.mix.parameters():
                    parameter.requires_grad_(False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # Using float32 for the adapter avoids unsupported bf16 complex kernels;
        # the backbone can remain bf16 on an 8GB GPU.
        low = self.down(hidden.float())
        if self.mode == "real_adapter":
            transformed = F.silu(low)
        else:
            phase = torch.polar(torch.ones_like(self.phases), self.phases)
            z = (low.unsqueeze(-2).to(torch.complex64)
                 * phase.view(1, 1, 3, 1) / math.sqrt(3))
            # Phased residual and nonlinear operation break trivial rank-one
            # representation, but useful specialization remains unproven.
            step = torch.polar(torch.ones_like(self.phase_step), self.phase_step)
            z = z + step.view(1, 1, 3, 1) * torch.complex(
                F.silu(z.real), F.silu(z.imag)
            )
            z = self.mix(z)
            transformed = z.sum(dim=-2).real / math.sqrt(3)
        return (self.residual_scale * self.up(transformed)).to(dtype=hidden.dtype)


class DecoderLayerWithStateAdapter(nn.Module):
    """Wrap a pretrained decoder layer, retaining its input/output contract."""

    def __init__(self, original: nn.Module, adapter: StateResidual) -> None:
        super().__init__()
        self.original = original
        self.adapter = adapter

    def forward(self, *args, **kwargs):
        result = self.original(*args, **kwargs)
        if isinstance(result, torch.Tensor):
            return result + self.adapter(result)
        if isinstance(result, tuple) and result and isinstance(result[0], torch.Tensor):
            return (result[0] + self.adapter(result[0]), *result[1:])
        raise TypeError(
            f"Unsupported decoder layer output {type(result)!r}; "
            "verify installed Transformers version before adapting it"
        )


def attach_adapters(model: nn.Module, *, rank: int = 16,
                    mode: AdapterMode = "trihaera", freeze_backbone: bool = True,
                    residual_scale: float = 0.1) -> dict:
    """Attach to a HF Llama-style CausalLM without replacing original weights.

    All original layers must be under model.model.layers and expose hidden_size.
    This modifies model IN PLACE, returns an auditable parameter report.
    """
    decoder = getattr(model, "model", None)
    layers = getattr(decoder, "layers", None)
    cfg = getattr(model, "config", None)
    hidden_size = getattr(cfg, "hidden_size", None)
    if layers is None or not isinstance(layers, nn.ModuleList) or not hidden_size:
        raise TypeError("Requires a Llama-style model.model.layers ModuleList and hidden_size")
    if not layers or any(isinstance(layer, DecoderLayerWithStateAdapter) for layer in layers):
        raise ValueError("No decoder layers, or adapters already attached")
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    for i, old_layer in enumerate(layers):
        adapter = StateResidual(hidden_size, rank, mode, residual_scale=residual_scale)
        device = next(old_layer.parameters()).device
        layers[i] = DecoderLayerWithStateAdapter(old_layer, adapter.to(device=device))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"mode": mode, "rank": rank, "residual_scale": residual_scale,
            "layers": len(layers),
            "trainable_parameters": trainable, "total_parameters": total,
            "trainable_fraction": trainable / total}


def adapter_parameters(model: nn.Module):
    for module in model.modules():
        if isinstance(module, DecoderLayerWithStateAdapter):
            yield from (p for p in module.adapter.parameters() if p.requires_grad)
