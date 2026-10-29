# TRIHAERA — persistent-state pretrained sidecar v0.1

**Research stage:** Stage 1 architectural hypothesis under examination (2026-10-30).  
**Source:** `src/trihaera/persistent.py`  
**Pretrained base:** `HuggingFaceTB/SmolLM2-135M`, frozen and unchanged.  
**Objective:** Test depth-persistent, three-state classical complex processing rather than a mixer followed immediately by readout.

## Why replace the old adapter experiment?

The previously tested local residual adapter forms three channels, applies a 3×3 unitary mixer, then immediately sums them. That makes the whole mixer equivalent to three constrained complex readout coefficients. It cannot demonstrate persistent cross-state computation.

The owner's **three-seed, 100-step WikiText-2** replication (64 validation windows; RTX 5060) found:

| Variant | Mean validation NLL ↓ | Mean train seconds |
| --- | ---: | ---: |
| Frozen base | 3.705553 | — |
| Real residual | 3.403005 | 14.6857 |
| Three-state, no mixing | 3.395234 | 18.4824 |
| Three-state, mixed | **3.388140** | 30.5823 |

Paired mixer-minus-no-mixer NLL differences by seed 17,23,42 were **−0.021348, −0.000446, +0.000513**; mean **−0.007093**, sample SD **0.012354**. Mixed training time was **~1.65×** no-mixing. This is an inconclusive, non-compute-matched quality advantage; the mixer's extra cost is not justified yet.

## What the new sidecar does

Let pretrained hidden states `h_l ∈ R^[B,T,D]` be the outputs of the frozen model's decoder layers. TRIHAERA maintains **the same complex tensor `Z_l ∈ C^[B,T,3,r]` across all depths**:

```text
drive_l = W_down_l(h_l)                             [B,T,r], real
inject_l,s = phase_l,s * gain_l,s * drive_l / sqrt(3)

Z_pre_l = inject_l                       if l = 1
        = 0.8 * Z_(l-1) + 0.2 * inject_l otherwise

Z_mid_l = Z_pre_l + 0.2 * (
    SiLU(Re(W_shared_l * Z_pre_l))
    + i * SiLU(Im(W_shared_l * Z_pre_l))
)
Z_l = U_l * Z_mid_l                     U_l = exp(i G_l), 3x3 unitary
```

The trainable **real** matrices `W_down_l` and `W_shared_l` are shared between all three channels, per layer. `Z_l` is passed unchanged into the **next nonlinear block**, except for its explicit depth update. **This specific structure cannot generally be collapsed to a final three-coefficient readout**, unlike the local adapter.

After the last layer, `readout(Re(sum_s Z_L,s)/sqrt(3))` is projected to the original hidden width, scaled by 0.1, transformed through the frozen LM head weights, and **added to original pretrained logits**.

The final real projection is initialized to zero, giving exact pretrained-logit parity before adaptation. Mode `no_mixing` disables all mixers while retaining the same carried-state and nonlinear operations. Inactive mixer parameters are frozen and excluded from trainable-parameter accounting.

## Limitations

- This is a **sidecar** receiving frozen hidden states. It **does not feed back** into the pretrained decoder's attention or learned hidden states. It is not the complete TRIHAERA architecture, which would carry state *inside* backbone computation.
- There is no cross-token attention inside the sidecar: causal information comes from the pretrained backbone.
- The three classical complex state channels entail extra activations, computation and potentially substantial matrix-exponential overhead.
- This is not a quantum implementation, not an independently pretrained model, and not evidence for additional sequential depth or reduced hallucinations.
- Current implementation is teacher-forced full-sequence **evaluation/training only**: no decoding KV cache/generation, quantization or checkpoint exporting.
- A regular real-valued depth-recurrent sidecar with matched effective width, parameters and compute is a necessary future control.
- The residual pathway is zero-initialized; internal mixer parameters initially have no loss gradient. This is intentional for exact base-model preservation; ensure gradients reach them after readout updates.

## Run on Windows RTX 5060

Existing `.venv` and SmolLM2 weights are reused. There is no additional large download.

```powershell
cd 'D:\Programming\03_Projects\personal-projects\05_ML\trihaera'
git pull --ff-only origin main
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest -q
python scripts/smoke_persistent.py --device cuda --mode mixed --output artifacts/persistent_mixed_smoke.json
python scripts/smoke_persistent.py --device cuda --mode no_mixing --output artifacts/persistent_no_mixing_smoke.json
```

**Inspect:** both modes must preserve initial logits exactly; both must train without nonfinite gradients or excessive VRAM. Compare trainable parameters, elapsed time and output drift. Do **not** compare 2-step losses as a quality benchmark. After real-model CUDA integration passes, implement a proper three-seed, identical-token held-out comparison against a real-valued recurrent sidecar and no-mixing control before attempting larger models.

**Decision gate:** Stage 1 remains open; if depth-persistent interference fails to outperform comparators at acceptable measured compute, revise or reject the hypothesis rather than automatically scaling.

## Owner's RTX 5060 persistent-sidecar smoke: 2026-10-30

PyTorch 2.14.0+cu130, Transformers 5.18.0, 53 unit tests passed. On cached SmolLM2-135M weights, both persistent variants preserved base logits exactly and completed two optimizer steps:

| Mode | Trainable parameters | Last train loss | Logit RMS drift | Two-step seconds | Peak allocated bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mixed` | 294096 | 4.435823 | 0.106377 | 0.723428 | 350269440 |
| `no_mixing` | 293556 | 4.436238 | 0.112375 | 0.441396 | 350023680 |

This shows correct integration and ~64% two-step runtime overhead for mixing. It is **not** a validation loss comparison. Initial state readout was zero; early gradients take time to reach internal state-mixing parameters.

### Next: six-real-channel controls and held-out replication

The new `real_six` and `real_six_no_mixing` variants maintain 6 real channels at rank 16 across all 30 decoder depths. They match the raw real scalar channel width of `3 × complex64` states. All use shared low-rank projections and nonlinear processing between depth steps; the optional real 6×6 orthogonal mixer has more independent mixing directions than constrained complex 3×3 unitary mixing. This is a deliberately strong real-valued comparator, not an exact FLOP match.

```powershell
git pull --ff-only origin main
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest -q
python scripts/replicate_persistent.py --device cuda --seeds 17,23,42 --steps 100 --eval-windows 64 --output artifacts/persistent_replication_135m.json
```

All variants train on the same token order and use the same WikiText-2 train/validation split; adapter capacities and compute are reported but not exactly matched. This remains a 135M **pretrained-derived** sidecar study, not full TRIHAERA pretraining.
