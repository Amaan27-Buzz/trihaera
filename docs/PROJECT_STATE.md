# TRIHAERA — authoritative project state (2026-10-30)

**Repository:** Amaan27-Buzz/trihaera  
**Branch rule:** Work directly on **main only**. No branches or PRs unless requested.

## Stage status

- Stage 1 mathematical analysis and novelty audit: **IN PROGRESS**; exact real-equivalence and matched O(6) controls established; no validated advantage.
- Stage 2 reference implementation: **IN PROGRESS**; standalone decoder, local adapter, detached persistent sidecar, and **new inter-layer feedback pilot**.
- Stages 3–8: not started. No compute-matched architectural win; no independently trained 500–700M foundation base or its instruction-tuned derivative.

## Verified environment

Windows RTX 5060 Laptop 8GB, PyTorch 2.14.0+cu130, Transformers 5.18.0, datasets 5.1.0. Independent TRIHAERA .venv; pretrained SmolLM2-135M and WikiText-2 cached. Owner's prior local 60 tests passed; GitHub CI on commit 3872f19 passed **67 tests**. Owner also passed the 77-test suite and real-GPU feedback smokes for all three variants; a held-out feedback experiment is still pending.

## Structural findings

- Three phase copies of one input embedding start at effective rank 1. A 3x3 unitary alone cannot increase their linear rank.
- Old layer-local adapter's unitary mixing immediately followed by a sum reduces exactly to a constrained three-complex-coefficient readout.
- Detached persistent sidecar adds depth-wise state processing but does not affect pretrained attention.
- Three complex states are exactly expressible as six real channels. No inherent quantum acceleration.
- U(3) is a constrained subset of real O(6). A matched-initialization O(6) baseline is necessary.

## Owner's latest 3-seed WikiText-2 experiment (commit 3872f19)

Frozen SmolLM2 NLL: 3.705553, 64 validation windows, 100 steps per mode/seed, seeds 17,23,42.

| Model | Mean NLL ↓ |
| --- | ---: |
| Complex persistent U(3) | 3.5383315 |
| Complex persistent, no mixing | 3.5395197 |
| Exact six-real U(3) re-encoding | 3.5383393 |
| Matched-start six-real O(6) | **3.5375221** |

Complex vs exact real mean difference ~−0.0000078, consistent with numerical parity. O(6) better than U(3) by ~0.000809 mean, across all three seeds, with 540 more trainable parameters. This tiny margin does NOT prove O(6) generally wins or complex mixing is harmful. U(3) beats no-mixing by only ~0.001188 NLL, without a compute-normalized benefit. Training wall times had extreme outliers (seed 23 no-mixing 43.87s, exact-real 48.92s); current time means are not reliable benchmarking evidence.

## Latest owner integrated-feedback CUDA smoke

**77 local tests passed**, frozen SmolLM2-135M, 9 input tokens, two optimizer steps, exact logits at initialization in every mode:
- U(3) mixed: **561,360** trainable parameters; **1.0255s** two steps; post-training logit RMS drift **0.24797**; peak allocated **355,622,400 bytes**.
- No mixing: **560,820** trainable; **0.7221s**; RMS drift **0.59690**; **355,377,152 bytes**.
- Real O(6) matched-start: **561,900** trainable; **0.5389s**; RMS drift **0.26104**; **355,315,712 bytes**.

No validated language-quality results for integrated feedback yet. Timings on nine tokens are not performance evidence. VRAM at sequence length 128 remains unmeasured.

## Research decision and new integration

Stop repeating detached sidecar sweeps. Introduce a depth-persistent feedback path into pretrained hidden states via `src/trihaera/feedback.py`. It wraps the frozen decoder layers, carries state between them, and injects **zero-initialized** trainable corrections so pretrained logits are unchanged at time zero. Later original attention operates on modified activations after learning begins. Unlike detached sidecar, backprop must pass through frozen decoder layers; 8GB peak usage is **unknown**. Supported full-sequence teacher-forced only: no KV cache, generation, concurrent model forwards, or gradient checkpointing.

**Previous local GPU gate (PASSED at 9 tokens):**
```powershell
git pull --ff-only origin main
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest -q
python scripts/smoke_feedback.py --device cuda --mode mixed --output artifacts/feedback_mixed_smoke.json
python scripts/smoke_feedback.py --device cuda --mode real_matched_orthogonal --output artifacts/feedback_real_matched_smoke.json
```

See [FEEDBACK_RESEARCH.md](FEEDBACK_RESEARCH.md) for data/analysis and [ARCHITECTURE.md](ARCHITECTURE.md) for equations. Never claim 12 layers = 36, enhanced reasoning, or superior efficiency based on these short trials.

## Next staged feedback evaluation

See [FEEDBACK_RESEARCH.md](FEEDBACK_RESEARCH.md) for the reproducible held-out experiment and cautions. New `scripts/replicate_feedback.py` first uses seed 17, 10 steps, 128 tokens, 32 validation windows to validate sequence-length and memory safety, then offers a 3-seed 40-step run with 64 validation windows on the same fixed data. Newly added CPU unit tests and GitHub CI must pass before execution. **Stage 1/2 remain IN PROGRESS.**
