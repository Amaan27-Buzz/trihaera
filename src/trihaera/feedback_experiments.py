"""Deterministic small-budget training and timing for feedback experiments.

No online model/data access here. Does not imply FLOP-matched controls.
"""

import hashlib
import json
import math
import random
import statistics
import time

import torch


def sample_schedule(n_windows: int, steps: int, batch_size: int,
                    seed: int) -> list[list[int]]:
    if min(n_windows, steps, batch_size) < 1:
        raise ValueError("Window, step and batch counts must be positive")
    if seed < 0:
        raise ValueError("Seed must be nonnegative")
    rng = random.Random(seed)
    return [
        [rng.randrange(n_windows) for _ in range(batch_size)]
        for _ in range(steps)
    ]


def schedule_digest(schedule: list[list[int]]) -> str:
    return hashlib.sha256(
        json.dumps(schedule, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@torch.no_grad()
def evaluate_nll(model, windows: torch.Tensor, *, batch_size: int,
                 device: torch.device) -> float:
    if windows.ndim != 2 or windows.shape[1] < 2 or batch_size < 1:
        raise ValueError("Expected windows [N,T>=2] and positive batch size")
    model.eval()
    numerator = 0.0
    targets = 0
    for start in range(0, windows.shape[0], batch_size):
        batch = windows[start:start + batch_size].to(device)
        loss = model(input_ids=batch, labels=batch, use_cache=False).loss
        count = batch.shape[0] * (batch.shape[1] - 1)
        numerator += float(loss.item()) * count
        targets += count
    if not targets:
        raise ValueError("No validation windows")
    return numerator / targets


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    low = int(index)
    high = min(len(ordered) - 1, low + 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def train_feedback(model, parameters: list[torch.nn.Parameter],
                   windows: torch.Tensor, schedule: list[list[int]],
                   *, device: torch.device, learning_rate: float,
                   clip_norm: float = 1.0, warmup: int = 2) -> dict:
    if (not parameters or not schedule or windows.ndim != 2 or
        windows.shape[1] < 2 or warmup < 0 or
        not 0 < learning_rate <= 0.01 or
        not math.isfinite(learning_rate) or not 0 < clip_norm <= 100 or
        not math.isfinite(clip_norm)):
        raise ValueError("Invalid feedback training configuration")
    if any(not indices or min(indices) < 0 or
           max(indices) >= len(windows) for indices in schedule):
        raise ValueError("Schedule references an out-of-range window")

    # Frozen backbone weights remain requires_grad=False, but backward flows
    # through their operations to the state feedback residuals.
    model.eval()
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    def step(indices: list[int], *, update: bool) -> tuple[float, float]:
        batch = windows[indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=batch, labels=batch, use_cache=False).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite feedback training loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters, clip_norm, error_if_nonfinite=True
        )
        if update:
            optimizer.step()
        return float(loss.item()), float(grad_norm.item())

    # Repeat the first scheduled example without updating parameters.
    # This warms the actual forward+backward kernels equally for every mode.
    # No training-token consumption or optimizer update occurs.
    for _ in range(warmup):
        step(schedule[0], update=False)
    optimizer.zero_grad(set_to_none=True)
    _sync(device)
    times = []
    losses = []
    gradients = []
    for indices in schedule:
        _sync(device)
        started = time.perf_counter()
        loss, grad = step(indices, update=True)
        _sync(device)
        times.append(time.perf_counter() - started)
        losses.append(loss)
        gradients.append(grad)
    summary = {
        "train_steps": len(schedule),
        "training_tokens_approx": sum(len(v) for v in schedule) *
                                  (windows.shape[1] - 1),
        "sampling_sha256": schedule_digest(schedule),
        "warmup_no_update_steps": warmup,
        "first_step_loss": losses[0],
        "last_step_loss": losses[-1],
        "mean_preclip_gradient_norm": statistics.mean(gradients),
        "max_preclip_gradient_norm": max(gradients),
        "training_elapsed_seconds": sum(times),
        "step_time_median_seconds": statistics.median(times),
        "step_time_p10_seconds": _percentile(times, .1),
        "step_time_p90_seconds": _percentile(times, .9),
        "step_time_max_seconds": max(times),
        "training_tokens_per_second": (
            sum(len(v) for v in schedule) * (windows.shape[1] - 1)
            / max(sum(times), 1e-12)
        ),
        "timing_note": (
            "Warm forward+backward, per-step device synchronization. "
            "Wall-clock includes Python/optimizer overhead and contention; "
            "not a FLOP estimate or controlled kernel benchmark."
        ),
    }
    return summary
