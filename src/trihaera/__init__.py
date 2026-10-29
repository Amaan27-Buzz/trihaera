"""TRIHAERA: classical three-state Transformer research prototype."""

from .config import ModelConfig
from .model import DenseTransformerLM, TrihaeraLM

__all__ = ["ModelConfig", "DenseTransformerLM", "TrihaeraLM"]
