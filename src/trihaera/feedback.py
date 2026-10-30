"""Inter-layer TRIHAERA feedback over frozen pretrained Llama decoder blocks.

The complex state is persistent across depth and alters subsequent backbone
hidden states. Pretrained weights are frozen but autograd must traverse their
operations; this is not a scratch-trained model. No KV cache or generation.
"""

import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from .persistent import PersistentStep
from .real_equivalent import RealEquivalentStep

MODES = ("mixed", "no_mixing", "real_matched_orthogonal")


class _Carrier:
    """Forward-scoped state, never a process-global variable."""

    def __init__(self):
        self.active = False
        self.state = None
        self.completed = 0


class FeedbackDecoderLayer(nn.Module):
    """Wrap a pretrained block; route persistent states into the next block."""

    def __init__(self, original: nn.Module, width: int, rank: int,
                 mode: str, scale: float, carrier: _Carrier) -> None:
        super().__init__()
        self.original = original
        self.state_step = (
            RealEquivalentStep(width, rank, mode)
            if mode == "real_matched_orthogonal"
            else PersistentStep(width, rank, mode)
        )
        self.to_hidden = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.to_hidden.weight)
        self.scale = scale
        self.mode = mode
        self.carrier = carrier

    def forward(self, *args, **kwargs):
        if not self.carrier.active:
            raise RuntimeError("Feedback layers require an active scoped forward")
        output = self.original(*args, **kwargs)
        if isinstance(output, torch.Tensor):
            hidden, tail = output, None
        elif isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            hidden, tail = output[0], output[1:]
        else:
            raise TypeError("Unsupported pretrained decoder output type")
        state = self.state_step(hidden, self.carrier.state)
        self.carrier.state = state
        self.carrier.completed += 1
        if state.is_complex():
            fused = state.sum(dim=2).real / math.sqrt(3)
        else:
            fused = state[:, :, :3, :].sum(dim=2) / math.sqrt(3)
        residual = self.to_hidden(fused) * self.scale
        updated = hidden + residual.to(hidden.dtype)
        return updated if tail is None else (updated, *tail)


class FeedbackCausalLM(nn.Module):
    """Frozen pretrained causal decoder with trainable state feedback.

    State correction after layer j changes inputs to pretrained layer j+1.
    Requires full-sequence forward; no caching, reentrancy, or checkpointing.
    """

    def __init__(self, backbone: nn.Module, *, rank: int = 16,
                 mode: str = "mixed", residual_scale: float = 0.1) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unsupported feedback mode: {mode}")
        if rank < 1 or not (0 < residual_scale <= 1) or not math.isfinite(residual_scale):
            raise ValueError("rank and residual scale must be finite and positive")
        cfg = getattr(backbone, "config", None)
        decoder = getattr(backbone, "model", None)
        layers = getattr(decoder, "layers", None)
        width = getattr(cfg, "hidden_size", None)
        if not isinstance(layers, nn.ModuleList) or not layers or not isinstance(width, int):
            raise TypeError("Requires pretrained Llama model.model.layers and hidden_size")
        if any(isinstance(layer, FeedbackDecoderLayer) for layer in layers):
            raise ValueError("Feedback layers already installed")
        if getattr(decoder, "gradient_checkpointing", False):
            raise ValueError("Gradient checkpointing unsupported by state carrier")
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        backbone.eval()
        self.backbone = backbone
        self.config = cfg
        self.rank = rank
        self.mode = mode
        self.residual_scale = float(residual_scale)
        self.carrier = _Carrier()
        for index, layer in enumerate(layers):
            device = next(layer.parameters()).device
            layers[index] = FeedbackDecoderLayer(
                layer, width, rank, mode, self.residual_scale, self.carrier
            ).to(device=device)

    def forward(self, *, input_ids: torch.Tensor, labels: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None,
                use_cache: bool = False, **kwargs):
        if use_cache:
            raise ValueError("KV cache/generation unsupported in feedback pilot")
        if kwargs:
            raise TypeError(f"Unsupported feedback forward arguments: {sorted(kwargs)}")
        if self.carrier.active:
            raise RuntimeError("Concurrent/reentrant forward unsupported")
        if input_ids.ndim != 2 or input_ids.shape[1] < 2:
            raise ValueError("Expected token IDs [batch, sequence>=2]")
        self.backbone.eval()
        self.carrier.active = True
        self.carrier.state = None
        self.carrier.completed = 0
        try:
            result = self.backbone(
                input_ids=input_ids, attention_mask=attention_mask,
                use_cache=False, return_dict=True
            )
            if self.carrier.completed != len(self.backbone.model.layers):
                raise RuntimeError("Not every decoder block ran exactly once")
            logits = result.logits
            loss = None
            if labels is not None:
                if labels.shape != input_ids.shape:
                    raise ValueError("Labels must match input IDs")
                loss = F.cross_entropy(
                    logits[:, :-1].contiguous().float().reshape(-1, logits.size(-1)),
                    labels[:, 1:].contiguous().reshape(-1), ignore_index=-100
                )
            return SimpleNamespace(logits=logits, loss=loss)
        finally:
            self.carrier.state = None
            self.carrier.completed = 0
            self.carrier.active = False


def feedback_parameters(model: FeedbackCausalLM):
    return (p for p in model.parameters() if p.requires_grad)


def feedback_report(model: FeedbackCausalLM) -> dict:
    trainable = sum(p.numel() for p in feedback_parameters(model))
    total = sum(p.numel() for p in model.parameters())
    return {
        "mode": model.mode, "depth": len(model.backbone.model.layers),
        "rank": model.rank, "residual_scale": model.residual_scale,
        "trainable_parameters": trainable, "total_parameters": total,
        "trainable_fraction": trainable / total,
        "limitation": "Frozen pretrained weights, trainable inter-layer feedback; no KV cache",
    }
