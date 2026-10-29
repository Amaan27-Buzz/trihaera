"""Single-sequence optimizer smoke test, not a training benchmark."""

import argparse
import torch
from trihaera import DenseTransformerLM, ModelConfig, TrihaeraLM


def run(name: str, model: torch.nn.Module, device: str, steps: int) -> None:
    model = model.to(device)
    cfg = model.config
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    torch.manual_seed(7)
    x = torch.randint(0, cfg.vocab_size, (2, 12), device=device)
    inputs, labels = x[:, :-1], x[:, 1:]
    initial = None
    last = None
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(inputs, labels)["loss"]
        if step == 0:
            initial = loss.item()
        loss.backward()
        optimizer.step()
        last = loss.item()
    print(f"{name}: initial_loss={initial:.4f}, last_loss={last:.4f}, "
          f"steps={steps}, params={sum(p.numel() for p in model.parameters()):,}")
    if not torch.isfinite(loss):
        raise RuntimeError("Training produced nonfinite loss")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=15)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be >= 1")
    cfg = ModelConfig(vocab_size=48, d_model=32, n_layers=2, n_heads=4,
                      max_seq_len=16, ffn_multiplier=2)
    for name, klass in [("trihaera", TrihaeraLM), ("dense", DenseTransformerLM)]:
        torch.manual_seed(17)
        run(name, klass(cfg), args.device, args.steps)
    print("Successful optimization confirms software integration, not model quality.")


if __name__ == "__main__":
    main()
