"""Fetch official model snapshots into THIS repository's ignored hf/ folder.

Example from the project root, while its .venv is activated:
    python scripts/download_models.py --models smol qwen --root hf
    python scripts/download_models.py --verify-only --root hf

No Git LFS, model conversion, or Torch/CUDA installation is involved.
Hugging Face Hub metadata and Xet cache are also confined to hf/.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from model_lab.catalog import MODELS, local_model_directory, verify_local_model

# Only top-level model inference assets; DO NOT fetch SmolLM's huge ONNX,
# optimizer/checkpoint-training outputs, or unrelated repo subdirectories.
DOWNLOAD_PATTERNS = [
    "*.safetensors",
    "*.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer*",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
    "chat_template*",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "LICENSE",
    "README.md",
]
SKIP_PATTERNS = [
    "onnx/*",
    "runs/*",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.onnx",
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS),
                        default=list(MODELS),
                        help="Model keys to fetch (default: both)")
    parser.add_argument("--root", type=Path, default=Path("hf"),
                        help="Project-local download root (default: ./hf)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Check already-downloaded assets without network")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    # Project-local artifacts ONLY. These assignments affect this Python
    # process, not the user's permanent Windows environment.
    os.environ["HF_HOME"] = str(root / ".hub-cache")
    os.environ["HF_XET_CACHE"] = str(root / ".xet-cache")
    os.environ["HF_HUB_CACHE"] = str(root / ".hub-cache" / "hub")

    if not args.verify_only:
        from huggingface_hub import snapshot_download

    for key in dict.fromkeys(args.models):
        spec = MODELS[key]
        directory = local_model_directory(spec, root)
        if args.verify_only:
            located = verify_local_model(spec, root)
            print(f"VERIFIED {spec.model_id}: {located}", flush=True)
            continue
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {spec.model_id} -> {directory}", flush=True)
        snapshot_download(
            repo_id=spec.model_id,
            local_dir=str(directory),
            allow_patterns=DOWNLOAD_PATTERNS,
            ignore_patterns=SKIP_PATTERNS,
            max_workers=4,
        )
        verified = verify_local_model(spec, root)
        weight_bytes = sum(p.stat().st_size
                           for p in verified.glob("*.safetensors"))
        print(
            f"VERIFIED {spec.model_id}: {verified} "
            f"({weight_bytes / 1_000_000_000:.2f} GB of weight files)",
            flush=True,
        )
    print("Models are now stored under the project-local hf/ directory.", flush=True)


if __name__ == "__main__":
    main()
