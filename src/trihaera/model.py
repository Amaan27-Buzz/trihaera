"""Three-state language model and conventional decoder comparator."""

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .layers import ThreeStateBlock


class TrihaeraLM(nn.Module):
    """Experimental phase-interference decoder; no language ability implied."""

    def __init__(self, config: ModelConfig, interference: bool = True) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positions = nn.Embedding(config.max_seq_len, config.d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.positions.weight, mean=0.0, std=0.02)
        self.layers = nn.ModuleList([
            ThreeStateBlock(config.d_model, config.n_heads,
                            config.ffn_multiplier, config.dropout,
                            interference=interference)
            for _ in range(config.n_layers)
        ])
        self.output_norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.input_phases = nn.Parameter(torch.tensor([-0.6, 0.0, 0.6]))

    def forward(self, token_ids: torch.Tensor,
                targets: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        if token_ids.ndim != 2 or token_ids.dtype != torch.long:
            raise ValueError("token_ids must be LongTensor[batch, tokens]")
        batch, tokens = token_ids.shape
        if not 0 < tokens <= self.config.max_seq_len:
            raise ValueError("Sequence length exceeds configured maximum")
        positions = torch.arange(tokens, device=token_ids.device)
        embedded = self.embedding(token_ids) + self.positions(positions)[None, ...]
        amps = torch.polar(torch.ones_like(self.input_phases), self.input_phases)
        z = embedded.unsqueeze(2).to(torch.complex64) * amps.view(1, 1, 3, 1) / math.sqrt(3)
        for layer in self.layers:
            z = layer(z)
        # Constructive/destructive classical amplitude recombination.
        fused = z.sum(dim=2).real / math.sqrt(3)
        logits = self.lm_head(self.output_norm(fused))
        out = {"logits": logits}
        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError("targets must match token_ids shape")
            out["loss"] = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                          targets.reshape(-1))
        return out


class DenseBlock(nn.Module):
    def __init__(self, width: int, heads: int, multiplier: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = heads
        self.head_width = width // heads
        self.norm1 = nn.RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.norm2 = nn.RMSNorm(width)
        self.up = nn.Linear(width, multiplier * width, bias=False)
        self.down = nn.Linear(multiplier * width, width, bias=False)
        self.dropout = dropout

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b, t, d = h.shape
        q, k, v = self.qkv(self.norm1(h)).chunk(3, -1)

        def split(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(b, t, self.n_heads, self.head_width).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            split(q), split(k), split(v), is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        h = h + self.proj(y.transpose(1, 2).reshape(b, t, d))
        return h + self.down(F.silu(self.up(self.norm2(h))))


class DenseTransformerLM(nn.Module):
    """Conventional decoder for initial testing, not a proven matched baseline."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positions = nn.Embedding(config.max_seq_len, config.d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.positions.weight, mean=0.0, std=0.02)
        self.layers = nn.ModuleList([
            DenseBlock(config.d_model, config.n_heads,
                       config.ffn_multiplier, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.output_norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, token_ids: torch.Tensor,
                targets: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        if token_ids.ndim != 2 or token_ids.dtype != torch.long:
            raise ValueError("token_ids must be LongTensor[batch, tokens]")
        b, t = token_ids.shape
        if not 0 < t <= self.config.max_seq_len:
            raise ValueError("Sequence length exceeds configured maximum")
        pos = torch.arange(t, device=token_ids.device)
        h = self.embedding(token_ids) + self.positions(pos)[None, ...]
        for layer in self.layers:
            h = layer(h)
        logits = self.lm_head(self.output_norm(h))
        out = {"logits": logits}
        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError("targets must match token_ids shape")
            out["loss"] = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                          targets.reshape(-1))
        return out
