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


def local_model_directory(spec: ModelSpec, root="hf"):
    """Mirror Hugging Face repo IDs inside an ignored project-local folder."""
    from pathlib import Path
    return Path(root).expanduser().resolve().joinpath(*spec.model_id.split("/"))


def verify_local_model(spec: ModelSpec, root="hf"):
    """Check files required for text-only Transformers loading, never the network."""
    import json
    from pathlib import Path

    folder = local_model_directory(spec, root)
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Missing local model: {folder}. Run: "
            f"python scripts/download_models.py --models {spec.key}"
        )
    for filename in ("config.json", "tokenizer.json"):
        if not (folder / filename).is_file():
            raise FileNotFoundError(
                f"Missing {filename} in {folder}; rerun the downloader."
            )
    shards = sorted(folder.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"Missing safetensors model weights in {folder}")
    index_file = folder / "model.safetensors.index.json"
    if index_file.exists():
        index = json.loads(index_file.read_text(encoding="utf-8"))
        required = set(index.get("weight_map", {}).values())
        if not required:
            raise ValueError(f"Invalid safetensors index in {folder}")
        missing = [name for name in sorted(required)
                   if not (folder / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing checkpoint shards in {folder}: {missing}"
            )
    for shard in shards:
        if shard.stat().st_size == 0:
            raise ValueError(f"Empty checkpoint shard: {shard}")
    return folder
