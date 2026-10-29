from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    d_model: int = 96
    n_layers: int = 3
    n_heads: int = 4
    max_seq_len: int = 128
    ffn_multiplier: int = 4
    dropout: float = 0.0
    n_states: int = 3

    def __post_init__(self) -> None:
        dims = (self.vocab_size, self.d_model, self.n_layers, self.n_heads,
                self.max_seq_len, self.ffn_multiplier)
        if min(dims) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_states != 3:
            raise ValueError("Reference architecture supports exactly three states")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
