# TRIHAERA — authoritative current development checkpoint

**Updated:** 2026-10-30  
**Repository:** Amaan27-Buzz/trihaera  
**Owner's branch rule:** All work directly on **main**; never open a branch or PR unless requested.

## Stage status

- **Stage 1 (architecture research): IN PROGRESS.** Rank-one input limitation; unitary rank preservation; local mixer/readout reduction; persistent-state sidecar with new real six-channel control. No validated architecture improvement.
- **Stage 2 (reference implementation): STARTED.** Stand-alone three-state decoder and pretrained SmolLM2-derived residual/persistent sidecars. Persistent SmolLM2 GPU integration passed.
- **Stages 3–8: NOT STARTED.** No equal-compute architectural advantage, no researched full training methodology, no 500–700M scratch model, no instruction-tuned derivative.

## Machine / environment

Independent TRIHAERA virtual environment at `D:\Programming\03_Projects\personal-projects\05_ML\trihaera\.venv`; copied installed Levis packages, Levis unchanged. RTX 5060 Laptop GPU (8 GB), PyTorch 2.14.0+cu130, Transformers 5.18.0, datasets 5.1.0. Original SmolLM2-135M model and WikiText-2 dataset already downloaded/cached.

## Scientific findings so far

1. Initial three phase copies have rank one. Pure 3×3 unitary mixing preserves rank; observed untrained effective rank only 1.22 after three blocks (1.227 without mixing).
2. The original pretrained per-layer adapter's 3×3 unitary mixer is followed immediately by a sum, mathematically collapsing to a three-coefficient complex readout. This cannot provide persistent cross-state transformations.
3. **100-step, three-seed WikiText-2 replication of old local adapters**, 64 validation windows: frozen SmolLM2 NLL 3.705553; real adapter mean 3.403005 in 14.686s; three-state without mixing 3.395234 in 18.482s; with mixing 3.388140 in 30.582s. Paired mixer-minus-no-mixer NLL differences: −0.021348, −0.000446, +0.000513. Mean −0.007093, SD 0.012354. **No convincing compute-efficient mixer advantage; one seed drives most of the mean.**
4. New sidecar carries a complex three-state tensor *across all pretrained decoder depths*, with nonlinear transformations between mixings. This avoids the local sum-before-next-block issue but remains a frozen-backbone sidecar: no influence on pretrained self-attention.
5. **Owner's 2026-10-30 GPU smoke:** **53 tests passed.** Mixed persistent sidecar: 294,096 trainable params; 0.723428s two steps; logit RMS drift 0.106377; CUDA peak 350,269,440 bytes. Unmixed: 293,556 trainable params; 0.441396s; RMS drift 0.112375; CUDA peak 350,023,680 bytes. Both modes preserved original SmolLM2 logits **exactly** at initialization, and two steps produced finite gradients/losses. This measures compatibility, not held-out quality.
6. On that 2-step input, mixing adds ~64% runtime overhead, a caution requiring larger timed trials to assess. No trained language-ability conclusions.

## Current code changes (next GPU validation pending)

- Added `real_six` and `real_six_no_mixing` as **six real scalar channels** with shared low-rank projections and optional learned real orthogonal (6×6) mixing. The 3 complex channels also have 6 real scalar components at matched rank.
- Real orthogonal mixing offers a stronger classical comparator: its 15-dimensional Lie algebra is less constrained than a structured three-complex-channel unitary's 9-dimensional algebra; parameter counts are close but not identical.
- Added tests: exact initial parity, orthogonality/energy preservation, gradient flow, trainability, causal prefix.
- Added `scripts/replicate_persistent.py`: four modes, three seeds, 100 steps, matched training windows, 64 validation windows, parameter counts, elapsed training time, gradient norms and peak CUDA allocated bytes, source/model revision hashes.

## Immediate next owner action

```powershell
cd 'D:\Programming\03_Projects\personal-projects\05_ML\trihaera'
git pull --ff-only origin main
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest -q
python scripts/replicate_persistent.py --device cuda --seeds 17,23,42 --steps 100 --eval-windows 64 --output artifacts/persistent_replication_135m.json
```

**Interpretation gate:** this is a same-token comparison, **NOT FLOP-/time-matched**, not multiple independent datasets, and not a full TRIHAERA model. Compare seed-paired differences versus the 6-real-channel comparator and no mixing; do not spend money scaling the architecture on a small noisy gain.

## Supporting docs

[Architecture](ARCHITECTURE.md) · [Persistent sidecar](PERSISTENT_STATE.md) · [Experiment protocol](EXPERIMENT_PROTOCOL.md) · [Pretrained adapter](PRETRAINED_ADAPTATION.md) · [Research decisions](DECISIONS.md) · [Roadmap](ROADMAP.md)
