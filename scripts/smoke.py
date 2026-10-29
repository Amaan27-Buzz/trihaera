"""Forward/backward correctness smoke test. PYTHONPATH=src required."""

import argparse
import torch
from trihaera import DenseTransformerLM, ModelConfig, TrihaeraLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.manual_seed(0)
    cfg = ModelConfig()
    for name, cls in [("trihaera", TrihaeraLM), ("dense", DenseTransformerLM)]:
        model = cls(cfg).to(args.device)
        x = torch.randint(0, cfg.vocab_size, (2, 16), device=args.device)
        out = model(x[:, :-1], x[:, 1:])
        out["loss"].backward()
        params = sum(p.numel() for p in model.parameters())
        print(f"{name}: params={params:,}, loss={out['loss'].item():.4f}, device={args.device}")
    print("Smoke test passed. No general language capability has been established.")


if __name__ == "__main__":
    main()
