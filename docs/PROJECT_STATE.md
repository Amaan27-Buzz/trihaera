# TRIHAERA — authoritative project checkpoint

**Date:** 2026-10-30  
**Repository:** Amaan27-Buzz/trihaera  
**Branch rule:** **main only**, no side branches or pull requests unless explicitly authorized.

## Stages

- **Stage 1 (architecture research): IN PROGRESS.** Mathematical operator v0.2, basic rank tests, phase analysis and *new adapter mixer-collapse proof*. Novelty and practical benefit remain unproven.
- **Stage 2 (reference implementation): STARTED.** Stand-alone three-state decoder and a distinct pretrained-model adapter bridge pass GPU compatibility and unit tests.
- **Stages 3–8:** NOT STARTED, including compute-matched comparisons, research-derived training recipe, 500–700M from-scratch base model and instruction-tuned version.

## Owner's verified machine

- Windows under `D:\Programming\03_Projects\personal-projects\05_ML\trihaera`; independent .venv cloned from Levis, Levis editable .pth link removed.
- PyTorch 2.14.0+cu130, Transformers 5.18.0, datasets 5.1.0, safetensors 0.8.0, RTX 5060 Laptop GPU (8GB).
- **38 unit tests passed** locally before latest multi-seed changes.
- Standalone untrained three-state rank: input 1.000, blocks 1.128, 1.204, 1.222; no mixer 1.227 at last block; output RMS difference 0.001512. Initial representations largely redundant.

## SmolLM2-135M Base adaptation experiments (derived model, not pretrained TRIHAERA)

- 30 Llama-style pretrained layers, original weights frozen.
- Three-state residual rank=16, ~553,680 adapter parameters (~0.41% total). Exact initial pretrained output parity.
- Original 2-step (lr 1e-3) logit drift RMS **9.25787** on 23 tokens.
- Stabilized 2-step (lr 3e-4, residual_scale 0.1, grad_clip 1.0) logit drift RMS **0.63579**, base logits RMS **9.99494**, one input example. No quality claim.
- First actual held-out WikiText-2 pilot, **single seed, 20 steps, 12 validation windows**:
  - Frozen base validation NLL **3.45328**
  - Real adapter **3.39929**, 552,960 trainable parameters, 2.39s
  - Three-state without mixer **3.34908**, 553,140 trainable parameters, 3.07s
  - Three-state with mixer **3.35118**, 553,680 trainable parameters, 4.83s
- **Mixer did not improve held-out loss** in the pilot; no-mixing was marginally better, mixed model ~57% slower. Samples and steps are too few for broad claims.

## Stage 1 mathematical negative finding

The pretrained adapter constructs three complex channels and applies a 3×3 unitary U **immediately before summation**. The result reduces algebraically to a three-coefficient complex readout `w=1^T U`. It is not a persistent state mixer between nonlinear computations. Exact forward and backward equivalence are now encoded in tests and the optional `collapsed_mixing` mode. This reduction **does not automatically apply** to the standalone decoder's inter-layer state mixer.

## Immediate next action

1. Pull latest main; run `python -m pytest -q` (including new collapse tests).
2. Run three-seed 100-step replicated WikiText-2 adapter experiment via `scripts/replicate_pretrained.py`, 64 validation windows, on owner's CUDA GPU; model/dataset cached.
3. Inspect seed-paired NLL differences, train time, gradients and peak memory; do not infer from mean alone.
4. If learned mixing remains no better than no-mixing or real adapter, **reject this adapter mixer as the architectural innovation**, and prototype *persistent state interactions before subsequent nonlinear layers*.
5. Keep original 500–700M from-scratch base and derived chat checkpoint conditional on a decisive Stage 3 architecture validation and a researched costed Stage 4 training plan.

## Key documentation

[Architecture](ARCHITECTURE.md) · [Pretrained pilot](PRETRAINED_ADAPTATION.md) · [Experiment protocol](EXPERIMENT_PROTOCOL.md) · [Research decisions](DECISIONS.md) · [Roadmap](ROADMAP.md).

No benchmarks yet show a parameter-/compute-matched TRIHAERA architectural advantage, longer context, reduced hallucinations or 36-layer-equivalent reasoning.
