"""Non-invasive state-representation diagnostics for TRIHAERA.

Diagnostic statistics are NOT a measure of language-model capability.
"""

import math

import torch


def state_statistics(z: torch.Tensor) -> dict[str, float | list[float]]:
    """Summarize linear diversity of the state axis in complex [B,T,3,D].

    Effective rank is computed from a 3x3 Gram matrix after flattening B,T,D.
    It measures linear state diversity, not reasoning or representation quality.
    """
    if z.ndim != 4 or z.shape[2] != 3 or not z.is_complex():
        raise ValueError("Expected complex state activations [batch,tokens,3,width]")
    with torch.no_grad():
        matrix = z.detach().permute(2, 0, 1, 3).reshape(3, -1)
        gram = matrix @ matrix.conj().transpose(0, 1)
        eigenvalues = torch.linalg.eigvalsh(gram).real.clamp_min(0)
        total = eigenvalues.sum()
        if total.item() <= 0:
            return {"effective_rank": 0.0, "energy_fractions": [0.0] * 3,
                    "mean_pairwise_overlap": 0.0, "coherent_sum_fraction": 0.0}
        probabilities = eigenvalues / total
        effective_rank = torch.exp(
            -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
        )
        energies = matrix.abs().square().sum(-1)
        fractions = (energies / energies.sum()).tolist()
        denom = (energies[:, None] * energies[None, :]).clamp_min(1e-30).sqrt()
        overlap = (gram.abs() / denom).triu(1).sum() / 3
        coherent_fraction = (
            matrix.sum(dim=0).abs().square().sum() / (3 * energies.sum())
        )
        return {
            "effective_rank": float(effective_rank.item()),
            "energy_fractions": [float(value) for value in fractions],
            "mean_pairwise_overlap": float(overlap.item()),
            "coherent_sum_fraction": float(coherent_fraction.item()),
        }


def probe_states(model: torch.nn.Module, token_ids: torch.Tensor) -> dict:
    """Record input + per-block states using hooks; restore training state."""
    for name in ("embedding", "positions", "input_phases", "layers"):
        if not hasattr(model, name):
            raise ValueError(f"Model missing expected attribute: {name}")
    if model.input_phases.numel() != 3:
        raise ValueError("Expected exactly three input phases")
    entries = []
    hooks = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            positions = torch.arange(token_ids.shape[1], device=token_ids.device)
            embedded = model.embedding(token_ids) + model.positions(positions)[None]
            phases = torch.polar(torch.ones_like(model.input_phases), model.input_phases)
            initial = (
                embedded.unsqueeze(2).to(torch.complex64)
                * phases.view(1, 1, 3, 1) / math.sqrt(3)
            )
            entries.append({"location": "input", **state_statistics(initial)})

        def make_hook(index):
            def capture(_module, _args, output):
                entries.append({
                    "location": f"block_{index + 1}", **state_statistics(output)
                })
            return capture

        hooks = [
            layer.register_forward_hook(make_hook(index))
            for index, layer in enumerate(model.layers)
        ]
        with torch.no_grad():
            logits = model(token_ids)["logits"]
        return {"statistics": entries, "logits": logits.detach()}
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)
