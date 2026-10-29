"""Persistent three-state sidecar over frozen pretrained decoder hidden states.

Unlike layer-local StateResidual, complex [B,T,3,R] activations persist over
decoder depth. Nonlinear processing follows mixing in successive blocks, so
the entire depth trajectory cannot generally collapse to a final linear
three-coefficient readout.

This is still a classical sidecar: it does NOT feed back into pretrained
attention and is NOT a trained foundation model.
"""

import math
from types import SimpleNamespace
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .layers import PhaseInterference, shared_complex_linear

PersistentMode = Literal["mixed", "no_mixing"]


class PersistentStep(nn.Module):
    """Update a carried complex state with one pretrained layer's features."""

    def __init__(self, width: int, rank: int, mode: PersistentMode) -> None:
        super().__init__()
        self.drive = nn.Linear(width, rank, bias=False)
        self.transform = nn.Linear(rank, rank, bias=False)
        self.phases = nn.Parameter(torch.tensor([-0.6, 0.0, 0.6]))
        self.gains = nn.Parameter(torch.tensor([0.9, 1.0, 1.1]))
        self.mix = PhaseInterference(enabled=(mode == "mixed"))
        if mode == "no_mixing":
            for parameter in self.mix.parameters():
                parameter.requires_grad_(False)
        nn.init.normal_(self.drive.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.transform.weight, mean=0.0, std=0.02)

    def forward(self, hidden: torch.Tensor,
                state: torch.Tensor | None) -> torch.Tensor:
        drive = self.drive(hidden.float())
        phases = torch.polar(torch.ones_like(self.phases), self.phases)
        injected = (drive.unsqueeze(2).to(torch.complex64)
                    * phases.view(1, 1, 3, 1)
                    * self.gains.view(1, 1, 3, 1) / math.sqrt(3))
        if state is None:
            state = injected
        else:
            state = state * 0.8 + injected * 0.2
        transformed = shared_complex_linear(self.transform, state)
        # This activation is non-holomorphic and nonunitary by design.
        nonlinear = torch.complex(F.silu(transformed.real),
                                  F.silu(transformed.imag))
        state = state + 0.2 * nonlinear
        # The *next* step applies fresh nonlinear processing to the MIXED state.
        return self.mix(state)


class PersistentStateAdapterLM(nn.Module):
    """Frozen pretrained LM plus a separate depth-persistent sidecar.

    Only full-sequence teacher-forced loss/evaluation is implemented. No
    generation API, cache, training of original weights, or feedback to
    pretrained attention is implied.
    """

    def __init__(self, backbone: nn.Module, *, rank: int = 16,
                 mode: PersistentMode = "mixed", residual_scale: float = 0.1) -> None:
        super().__init__()
        if mode not in ("mixed", "no_mixing"):
            raise ValueError("mode must be 'mixed' or 'no_mixing'")
        if rank < 1 or not 0 < residual_scale <= 1 or not math.isfinite(residual_scale):
            raise ValueError("Invalid rank/residual scale")
        cfg = getattr(backbone, "config", None)
        model = getattr(backbone, "model", None)
        layers = getattr(model, "layers", None)
        width = getattr(cfg, "hidden_size", None)
        output = backbone.get_output_embeddings() if hasattr(backbone, "get_output_embeddings") else None
        if (not isinstance(layers, nn.ModuleList) or not layers or
            not isinstance(width, int) or width < 1 or
            not isinstance(output, nn.Linear) or output.weight.shape[1] != width):
            raise TypeError("Requires a Llama-style causal LM with linear output head")
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.config = cfg
        self.rank = rank
        self.mode = mode
        self.residual_scale = residual_scale
        device = output.weight.device
        self.steps = nn.ModuleList([
            PersistentStep(width, rank, mode).to(device=device)
            for _ in range(len(layers))
        ])
        self.readout = nn.Linear(rank, width, bias=False).to(device=device)
        nn.init.zeros_(self.readout.weight)

    def forward(self, *, input_ids: torch.Tensor,
                labels: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None,
                use_cache: bool = False):
        if use_cache:
            raise ValueError("KV cache/generation is unsupported by persistent sidecar")
        # Frozen backbone always runs without dropout or an autograd graph.
        self.backbone.eval()
        with torch.no_grad():
            base = self.backbone(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True, use_cache=False,
                return_dict=True,
            )
        hidden = base.hidden_states
        if hidden is None or len(hidden) != len(self.steps) + 1:
            raise RuntimeError("Expected n_layers+1 frozen hidden-state tensors")
        state = None
        for step, layer_hidden in zip(self.steps, hidden[1:]):
            state = step(layer_hidden, state)
        assert state is not None
        merged = state.sum(dim=2).real / math.sqrt(3)
        projected = self.readout(merged) * self.residual_scale
        head = self.backbone.get_output_embeddings()
        correction = F.linear(projected.to(head.weight.dtype), head.weight)
        logits = base.logits + correction.to(base.logits.dtype)
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels and input_ids must have matching shape")
            if labels.shape[-1] < 2:
                raise ValueError("Need at least two tokens for causal LM loss")
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().float().reshape(-1, logits.shape[-1]),
                labels[:, 1:].contiguous().reshape(-1), ignore_index=-100,
            )
        return SimpleNamespace(logits=logits, loss=loss)


def persistent_parameters(model: PersistentStateAdapterLM):
    return (parameter for parameter in model.parameters()
            if parameter.requires_grad)


def persistent_parameter_report(model: PersistentStateAdapterLM) -> dict:
    trainable = sum(p.numel() for p in persistent_parameters(model))
    total = sum(p.numel() for p in model.parameters())
    return {
        "mode": model.mode, "rank": model.rank, "depth": len(model.steps),
        "residual_scale": model.residual_scale,
        "trainable_parameters": trainable, "total_parameters": total,
        "trainable_fraction": trainable / total,
    }
