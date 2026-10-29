# TRIHAERA pretrained-weight adaptation track

**Status:** Stage 1/2 experimental bridge; not a trained TRIHAERA model.  
**First checkpoint:** [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) (BASE, Apache-2.0).  
**Why 135M:** ~270MB pretrained checkpoint, 30 Llama-style decoder layers, 576 hidden width, grouped-query attention and RoPE; suitable for first 8GB-VRAM compatibility tests. Larger 360M and ~600M checkpoints come later if justified. Model's *weights* are permissively licensed; do not claim its entire training corpus is independently licensed/fully reproducible without auditing it.

## Research choice

The owner's October 30, 2026, *untrained* state probe reported:
- Rank: 1.00 at input; 1.128, 1.204 and 1.222 after three blocks.
- Pairwise overlap: 1.00 at input and ~0.929 by block 3.
- Interference-enabled vs disabled logits: RMS ~0.001512.
- The mixing-disabled control had **slightly higher** final rank (~1.227), so higher initial diversity cannot be attributed to the mixer.

This supports a concern that initial states are mostly redundant, but no conclusion about learned representations, perplexity or efficiency is possible. Training and matched baselines are required.

## First adaptation implementation

`src/trihaera/pretrained.py` retains the real pretrained Llama decoder in full, freezes its weights, wraps each block, and attaches **trainable low-rank residuals**. The adapter's real down/up projections are shared across three classical complex state channels. State-dependent phases and a 3x3 unitary mixer provide an experimental computation path. Its output projection starts at zero, so the pretrained logits must be **bit-for-bit identical before training**. Backprop then trains only the adapters.

A clear limit: these three states are **formed and recombined within each residual adapter**, not maintained as a persistent superposition through all pretrained layers. The resulting model is a **SmolLM2-derived architecture experiment**, NOT from-scratch TRIHAERA-650M Base. It cannot establish the original hypothesis on its own.

Controls provided:
- `trihaera`: three-state low-rank residual with learned mixing;
- `no_mixing`: same computational states, mixer disabled;
- `real_adapter`: ordinary low-rank SiLU residual control.

The real adapter has slightly fewer parameters (no phase/mixer). Exact count is logged; even near-equal parameter counts **do not** imply compute matching.

## RTX 5060 Windows smoke

From `D:\Programming\03_Projects\personal-projects\05_ML\trihaera`:

```powershell
git pull --ff-only origin main
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest -q
python -m pip show transformers safetensors
```

If **Transformers or safetensors is missing**, install **inside TRIHAERA only**:

```powershell
python -m pip install 'transformers>=4.45,<6' safetensors
```

No new PyTorch installation is required. Transformers may need to obtain a few small dependencies, but **the first SmolLM2 run will download ~270 MB of pretrained weights** from Hugging Face. Model weights are cached outside this repository. Repeated runs should reuse that cache.

```powershell
python scripts/smoke_pretrained.py --device cuda --mode trihaera --output artifacts/smollm2_135m_trihaera_smoke.json
python scripts/smoke_pretrained.py --device cuda --mode real_adapter --output artifacts/smollm2_135m_real_adapter_smoke.json
python scripts/smoke_pretrained.py --device cuda --mode no_mixing --output artifacts/smollm2_135m_no_mixing_smoke.json
```

First run requires network access. It checks baseline logit equality at initialization, adapter-only two-step gradient updates, trainable parameter count, elapsed time and peak allocated CUDA memory. It **does not train a usable language model** or measure real effectiveness.

Do not commit upstream weights, cache, sensitive raw data, or `artifacts/` to GitHub.

## Proposed next controlled development

1. Verify compatibility with the owner's actual Transformers version and 8GB RTX 5060.
2. Record exact pretrained model revision (commit SHA), tokenizer and license to make subsequent runs reproducible.
3. Freeze a held-out corpus and tokenization for tests; define contamination and provenance policy.
4. Compare base-only, plain real adapter, no-mixing three-state and full three-state using equal trainable parameter budgets, identical training tokens, multiple seeds and *measured* GPU time/memory.
5. Reconcile difference between residual adapter prototype and persistent-state TRIHAERA architecture. If adaptation does not provide evidence, do not misrepresent it as a new pretrained model.
6. Stage 1 exit remains gated on mathematical formalization and Stage 3 validation design; first 500–700M from-scratch base is conditional on evidence and compute.

Reference: [original SmolLM2 repository](https://github.com/huggingface/smollm/tree/main/text), [model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M).

## October 30, 2026 — first real CUDA SmolLM2 checkpoint

Owner confirmed:
- Windows PyTorch 2.14.0+cu130 / Transformers 5.18.0 / CUDA RTX 5060 8GB;
- All 29 unit tests passed;
- 30 wrapped decoder layers with 553,680 adapter parameters (0.41% of ~135M total);
- Exact initial logit parity (max difference **0**);
- Two steps on 23 tokens, initial loss **2.418364**, elapsed **1.466s**, allocated peak **375,484,416 bytes**;
- **Post-step RMS logit change 9.25787** at the OLD learning rate **1e-3** and no gradient clipping.

This initial compatibility check is not evidence of a quality gain. The large output drift is a practical **stability concern**, not a result to optimize for. We have not yet measured validation loss or prediction fidelity after updates.

## Stabilized adapter configuration

New `StateResidual` applies an explicit **residual_scale=0.1** to the trainable adapter output; the zero-initialized output projection still leaves pretrained logits exactly unchanged at initialization. New smoke uses **AdamW lr=3e-4**, **gradient norm clipping=1.0**, and deterministic eval mode to avoid dropout confounds. The no-mixing control now freezes inactive mixer weights and does not count them as trainable. Compare against the original high-drift checkpoint using **normalized logit RMS drift** and held-out language-model loss, not just the ability to change outputs.

Run the stabilized smoke:

```powershell
git pull --ff-only origin main
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest -q
python scripts/smoke_pretrained.py --device cuda --mode trihaera --output artifacts/smollm2_stable_smoke.json
```

Then run the first **small-data held-out** adapter pilot. [WikiText-2 raw](https://huggingface.co/datasets/Salesforce/wikitext) provides official independent train and validation splits. Its documented licenses include CC BY-SA 3.0 and GFDL; using it for an experiment is not a blanket license claim for redistributing derived training data.

```powershell
python -m pip show datasets
# Only if missing:
python -m pip install datasets
python scripts/pilot_pretrained.py --device cuda --steps 20 --batch-size 1 --sequence-length 128 --output artifacts/pilot_135m.json
```

The script downloads the WikiText-2 raw split on first run (small relative to the 270MB model). It uses deterministic text packing, a frozen unmodified SmolLM2 baseline, identical optimizer and training token order across three modes, and separately reports initial/validation loss, gradient norms, trainable parameter count, training time and peak allocated GPU memory.

**These are preliminary controls, NOT fair equal-FLOP/equal-parameter scientific validation.** Training-data overlap with the pretrained model is also possible; the held-out split protects only against contamination from this pilot's own training steps. No decision to scale should be based on the outcome of just 20 steps.

## Owner: first WikiText-2 validation results — 2026-10-30

All 38 unit tests passed locally on RTX 5060; stable 2-step smoke kept exact initial logits, reduced post-update logit drift to **0.63579 RMS** (relative to base logit RMS **9.99494**). This reflects changes to residual scale, optimizer and clipping together, not a controlled isolated causal attribution.

First pilot used one seed, **20 steps × batch 1 × sequence length 128**, **256 training windows and 12 validation windows**:
- Frozen pretrained base held-out NLL **3.45328**
- Real adapter: **3.39929**; 552,960 trainable params; **2.39 s**
- Three-state with no mixer: **3.34908**; 553,140 trainable params; **3.07 s**
- Three-state with learned mixer: **3.35118**; 553,680 trainable params; **4.83 s**

The no-mixing three-state model won on this tiny trial. Mixing produced a worse loss (+0.00210 NLL vs no mixing), ~1.57× the training time, and no demonstrated added capability. Training time is not FLOP-matching and can vary with GPU clocks, kernels and cache.

### Structural reduction and follow-up

The 3×3 unitary adapter mixer is followed immediately by a sum over state channels. Algebraically, its effect is **exactly reducible** to three complex coefficients (a constrained complex readout). This explains why we must not overinterpret the unitary matrix as cross-state reasoning. The `collapsed_mixing` adapter is mathematically equivalent, and forward/gradient tests are included.

For stronger evidence without downloading a new model or dataset:

```powershell
git pull --ff-only origin main
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest -q
python scripts/replicate_pretrained.py --device cuda --seeds 17,23,42 --steps 100 --eval-windows 64 --output artifacts/replication_135m.json
```

This separately trains real, no-mixing and TRIHAERA mixer adapters at equal token exposures, fixed hyperparameters, and common seeded data order, reports all seed-level outcomes and descriptive paired standard deviations. It logs throughput, peak CUDA memory and training time, **not matched FLOPs**. A positive single-seed result does not justify scaling; a null result means redirect the mechanism toward persistent states interacting before subsequent nonlinearities.
