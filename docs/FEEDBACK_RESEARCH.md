# TRIHAERA — October 30, 2026 matched-real findings and feedback direction

## Verified owner results

Hardware: RTX 5060 Laptop GPU (8GB), PyTorch 2.14.0+cu130, Transformers 5.18.0, cached SmolLM2-135M. Git main `3872f19` checked out correctly after a transient Windows DNS failure (no change to TRIHAERA needed). Official WikiText-2 raw train and validation splits; 64 validation windows. Same training token exposures, three seeds 17/23/42, 100 steps/variant. **NOT FLOP-, wall-time- or fully capacity-matched**, and not an independent pretraining result.

| Mode | Seed 17 NLL | Seed 23 NLL | Seed 42 NLL | Mean NLL ↓ |
| --- | ---: | ---: | ---: | ---: |
| Complex U(3) mixer | 3.531837 | 3.533814 | 3.549343 | 3.5383315 |
| Same model without mixing | 3.533848 | 3.534739 | 3.549972 | 3.5395197 |
| Exact 6-real U(3) re-encoding | 3.531814 | 3.533825 | 3.549378 | 3.5383393 |
| 6-real matched-initialization O(6) mixer | **3.530686** | **3.532996** | **3.548884** | **3.5375221** |

- Complex-minus-exact-real mean NLL difference: **−0.0000078**. This is negligible numerical drift, consistent with analytical equivalence. Complex arithmetic is not required to represent TRIHAERA's computation.
- O(6)-minus-U(3) mean NLL difference: **−0.0008094** (O(6) lower loss on each tested seed), with **540 additional trainable parameters**. The win is too small and the control has a larger reachable rotation space, so it cannot establish a general O(6) superiority.
- U(3)-minus-no-mixer mean NLL difference: **−0.0011882**, with increased compute. **No compelling evidence for the specific mixer as a quality-per-compute advantage.**
- Wall-clock timings are demonstrably unstable, e.g. seed 23 `no_mixing` ~43.87 s vs ~8.69/10.70 s for other seeds; seed 23 `real_equivalent` ~48.92 s vs ~14.26/18.11 s. GPU throttling/host contention or scheduling are possible explanations, **not established causes**. Do not rank kernel efficiency by averages from this run.

The exact realification is a deterministic **identity in mathematics**, not evidence of superiority. The three-seed data do not prove or disprove generalization beyond this small corpus, tokenizer and training regime. The earlier 6-real control had different state initialization; the current matched-control results supersede that comparison.

## Research decision

The current frozen-backbone persistent sidecar is **not a sufficient test of the proposed Transformer**: it observes original hidden states but cannot affect pretrained attention. More short sidecar training is lower priority than building a true feedback connection and confirming it works.

We retain **three-state constrained mixing as a hypothesis** (potential useful inductive bias), not a verified source of efficiency or an inherently quantum benefit. Prioritize direct feedback to downstream Transformer blocks, preserving pretrained logits at zero initialization; compare U(3), no-mixing and matched O(6) once correctness is established. Reject any costly scaling until a genuine compute-normalized comparison demonstrates repeatable gains.

## New integration design

`src/trihaera/feedback.py` wraps each existing pretrained decoder block, freezes all original parameters, persists complex (or matched-real) state across depth and adds a **zero-initialized learnable residual to the block's hidden output**. This modified hidden value flows into the next pretrained block and its attention. The backward graph goes through frozen pretrained computations because those hidden values depend on trainable feedback.

This is closer to an integrated Transformer than the prior detached sidecar, but it is **not** a from-scratch three-state Transformer, does not alter attention kernels directly, and has no decoding/cache support. Expected additional VRAM means the first checkpoint is a two-step GPU compatibility smoke, **not** another 12-run benchmark.

### Hardware gate

```powershell
cd 'D:\Programming\03_Projects\personal-projects\05_ML\trihaera'
git pull --ff-only origin main
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest -q
python scripts/smoke_feedback.py --device cuda --mode mixed --output artifacts/feedback_mixed_smoke.json
python scripts/smoke_feedback.py --device cuda --mode real_matched_orthogonal --output artifacts/feedback_real_matched_smoke.json
```

Verify (1) original logits match bit-for-bit before training, (2) early-layer corrections change later pretrained hidden inputs, (3) the backward graph and optimizer update work, (4) no state leaks between calls, (5) peak GPU memory fits. Compare *correctness* first, not tiny 2-step train losses. Do not rerun old sidecar benchmark while this gate is pending.

## Stop/go criteria pending

Stage 1 cannot be called complete without a method for matched compute including timing/activation RAM, stronger modern dense and real-recurrent baselines, and a predeclared validation threshold based on seed/data variance. A 500–700M scratch foundation model remains **conditional**, not scheduled.

## RTX 5060 integrated feedback compatibility — owner verified 2026-10-30

On GitHub main `121105e`, owner's local **77 tests passed**, and all three SmolLM2-135M full-decoder feedback smoke runs passed with exact original-logit parity.

| Variant | Trainable adapter parameters | Two-step seconds | Post-step logit RMS drift | Peak CUDA allocated |
| --- | ---: | ---: | ---: | ---: |
| Complex U(3) mixed | 561,360 | 1.0255 | 0.24797 | 355,622,400 bytes |
| Complex no mixing | 560,820 | 0.7221 | 0.59690 | 355,377,152 bytes |
| Matched real O(6) | 561,900 | 0.5389 | 0.26104 | 355,315,712 bytes |

All modes used **9 tokens and 2 optimizer steps**. The different final training losses and logit shifts are *not validation-quality measurements*. The ~339 MiB allocation is specific to this short sequence and **does not predict memory at 128 or 256 tokens**. Single short timings must not be used for speed ranking. All original backbone weights remain frozen, but backward gradients traverse subsequent Transformer layers.

### New measured feedback pilot

`scripts/replicate_feedback.py` runs **integrated** feedback on WikiText-2 official train/validation splits (never the former detached sidecar), with:
- Identical tokenized train/validation windows, deterministic per-seed training batch schedules, and fixed optimizer settings.
- Frozen-base NLL and exact zero-feedback initial validation parity checked for every configuration.
- Three modes `mixed`, `no_mixing`, `real_matched_orthogonal`; mode order rotates between seeds.
- **Warm forward+backward calls without updates** before measurement; explicit CUDA synchronization at each timed training step.
- Per-mode trainable parameters, step-time median/p10/p90/max, token throughput, gradient norms, VRAM allocated/reserved and paired validation NLL.
- JSON output saved after every completed mode, with model revision, git SHA, dataset fingerprints and token-window hashes.

**First GPU gate**: one seed, 10 steps, 128 tokens. Only if this works and GPU usage remains safe should the owner run the 3-seed 40-step diagnostic. Still **not FLOP/time-matched**, **not a full TRIHAERA foundation model** and **not evidence of advantage**. Exact model weights are cached; no new model is required.

```powershell
cd 'D:\Programming\03_Projects\personal-projects\05_ML\trihaera'
git pull --ff-only origin main
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest -q
python scripts/replicate_feedback.py --device cuda --seeds 17 --steps 10 --sequence-length 128 --eval-windows 32 --output artifacts/feedback_gate_128.json
```

If the gate passes:

```powershell
python scripts/replicate_feedback.py --device cuda --seeds 17,23,42 --steps 40 --sequence-length 128 --eval-windows 64 --output artifacts/feedback_replication_135m.json
```

Interpretation should prioritize paired NLL against no-mixing, matched O(6), and *measured* compute overhead, not small early-stage train loss changes or isolated FPS claims. Later, a stronger dense/shared-weight baseline and equal-FLOP/time budget are still required.
