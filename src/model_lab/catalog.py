"""Phase 1: immutable, reference model specifications."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    loader: str
    advertised_context: int


MODELS = {
    "smol": ModelSpec("smol", "HuggingFaceTB/SmolLM2-360M-Instruct", "causal", 8192),
    "qwen": ModelSpec("qwen", "Qwen/Qwen3.5-0.8B", "multimodal", 262144),
}


def resolve_model(key: str) -> ModelSpec:
    if key not in MODELS:
        raise ValueError(f"Unknown model {key!r}; choose {list(MODELS)}")
    return MODELS[key]
