# TRIHAERA — authoritative development checkpoint

**Date:** 2026-10-30  
**Repository:** Amaan27-Buzz/trihaera  
**Branch policy:** WORK ON **main ONLY**. No side branches or PRs unless explicitly authorized.

## Stage status

- **Stage 1 — Architecture research/formalization: IN PROGRESS.** Found rank-one input representation, weak initial diversity, mathematically reducible local mixer. Now testing persistent-state sidecar.
- **Stage 2 — Reference implementation: STARTED.** Original 3-state decoder, local pretrained residual adapter, and depth-persistent sidecar reference implementations; latest sidecar awaiting owner's real-model CUDA validation.
- **Stages 3–8 — NOT STARTED.** No matched-compute proof, 500–700M from-scratch pretraining, or TRIHAERA-Instruct checkpoint.

## Verified owner machine and environment

- Windows; `D:\Programming\03_Projects\personal-projects\05_ML\trihaera`
- Independent TRIHAERA .venv copied from Levis, with copied Levis editable .pth removed. Levis unchanged.
- RTX 5060 Laptop GPU, 8GB dedicated VRAM; PyTorch 2.14.0+cu130, Transformers 5.18.0, datasets 5.1.0, safetensors 0.8.0.
- 44 prior tests passed locally and on CI at commit `a25f58a`.
- Untrained three-state rank: input 1.000 → blocks 1.128, 1.204, 1.222, versus last-block rank 1.227 without mixer; highly redundant initial channels.

## SmolLM2 135M pretrained-weight adaptation (frozen backbone)

- Initial compatibility: 30 layer-local adapters; ~553,680 trainable parameters; exact initial logits.
- Original high-rate smoke two-step logit RMS drift 9.25787; scaled/gated smoke drift 0.63579 after lr 3e-4 and residual scale .1. Neither is a quality benchmark.
- Single-seed 20-step pilot, 12 validation windows: frozen base 3.45328 NLL; real adapter 3.39929; no-mixing 3.34908; mixed 3.35118. Mixing slower and slightly worse.
- **Replicated pilot**: three seeds 17/23/42; 100 training steps; 64 validation windows; initial frozen SmolLM2 NLL **3.705553**.
  - Real adapter mean NLL **3.403005**, mean 14.6857s.
  - Three-state without mixing mean NLL **3.395234**, mean 18.4824s.
  - Three-state with mixing mean NLL **3.388140**, mean 30.5823s.
  - Mixed-minus-unmixed paired NLL: **−0.021348, −0.000446, +0.000513** (mean −0.007093; SD 0.012354).
  - Mixed training was **~65.5% slower** than no mixing. The NLL improvement is concentrated in one seed. NOT matched compute/parameters.
- **Finding:** layer-local adapter mixes state then sums immediately: full 3×3 matrix collapses to 3 complex readout coefficients. It is not meaningful persistent interaction.

## New implementation checkpoint

- Depth-persistent complex sidecar `src/trihaera/persistent.py`: separate frozen-base features, state carrier across all 30 decoder depths, per-depth shared transformations, nonlinear processing between mixing operations, zero-initialized additive logit projection.
- Controls: identical structure with `mixed` and `no_mixing` modes; frozen unused mixer weights excluded.
- Local CPU tests passed on a mock pretrained model for output parity, frozen backbone, gradient flow, causal prefix, earlier-mixer influence and error handling.
- **Not yet verified** on the owner's real SmolLM2 135M checkpoint or CUDA.
- **Limitation:** this is a sidecar and does not feed back into pretrained attention. Full TRIHAERA remains a separate later architectural goal.

## Immediate owner action

1. Pull main, run `python -m pytest -q`.
2. Run `scripts/smoke_persistent.py` on CUDA for `--mode mixed` and `--mode no_mixing` using cached SmolLM2-135M weights; save output under ignored `artifacts/`.
3. Share parameter counts, initial-logit parity, logit drift, training time and CUDA peak memory.
4. If compatible, compare persistent mixed vs no-mixing on the SAME WikiText-2 windows, multiple seeds, real-valued persistent/recurrent controls and matched FLOPs/time.
5. Only advance Stage 1/2 gates after proof of useful architecture; 500–700M pretraining and instruction tuning remain conditional on validation and compute/data planning.

Reference: [persistent states](PERSISTENT_STATE.md), [architecture](ARCHITECTURE.md), [pretrained adaptation](PRETRAINED_ADAPTATION.md), [experiment protocol](EXPERIMENT_PROTOCOL.md), [roadmap](ROADMAP.md).
