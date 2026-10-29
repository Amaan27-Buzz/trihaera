"""Inspect three-state representation diversity, not language quality.

Uses the SAME untrained weights with mixing on/off to expose effects of the
state-interference operator. Run after setting PYTHONPATH=src.
"""

import argparse
import json
from pathlib import Path

import torch

from trihaera import ModelConfig, TrihaeraLM
from trihaera.diagnostics import probe_states


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch < 1 or not 1 <= args.tokens <= 128:
        parser.error("batch must be >= 1 and tokens must be in [1,128]")

    torch.manual_seed(args.seed)
    cfg = ModelConfig()
    model = TrihaeraLM(cfg).to(args.device).eval()
    tokens = torch.randint(0, cfg.vocab_size, (args.batch, args.tokens),
                           device=args.device)
    enabled = probe_states(model, tokens)
    for block in model.layers:
        block.mix.enabled = False
    disabled = probe_states(model, tokens)
    for block in model.layers:
        block.mix.enabled = True

    rms_difference = (enabled["logits"] - disabled["logits"]).square().mean().sqrt()
    report = {
        "description": "Random-initialization architectural diagnostic; not model benchmark",
        "device": args.device,
        "seed": args.seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "interference_on": enabled["statistics"],
        "interference_off": disabled["statistics"],
        "logit_rms_difference": float(rms_difference.item()),
        "interpretation": ("Rank 1 at input is expected. Rank >1 later shows linear "
                           "state diversification, NOT enhanced model intelligence."),
    }
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
