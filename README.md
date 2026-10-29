# TRIHAERA

**Dawnlight Labs — experimental three-state, quantum-inspired Transformer**

**Current state: Stage 1 architecture research in progress; Stage 2 exploratory implementation in progress.** The owner's RTX 5060 passed initial correctness tests and a random-initialization representation probe. There is no demonstrated architectural advantage or trained TRIHAERA language model.

TRIHAERA investigates whether three simultaneously represented classical complex-amplitude channels, labelled **−1, 0, +1**, combined through state interference and shared Transformer weights, can achieve superior language-model quality per parameter **and** per unit of computation.

These are **classically simulated amplitudes, not actual quantum hardware superposition**. Three channels of a 12-layer Transformer are not automatically a 36-layer model. Longer context, better reasoning, lower hallucination and architectural novelty remain hypotheses.

## Model objective

- First serious foundation pretraining target: **500–700 million parameters**, with a provisional 650M configuration.
- Chat model: instruction-tune **that same base checkpoint** rather than pretrain an unrelated model.
- General-purpose, text-first language modelling: dialogue, knowledge, mathematics and coding.
- Future 1B+ scaling requires architecture validation and a costed training recipe.
- Independent of Levis/PhaseMemory and PLURIMENS.

## Eight stages

| # | Stage | Current status |
| --- | --- | --- |
| 1 | Mathematical architecture, novelty and hypotheses | **In progress** |
| 2 | Modular PyTorch reference implementation and correctness | Prototype started |
| 3 | Matched experimental validation and ablations | Not started |
| 4 | Evidence-led training recipe research | Not started |
| 5 | 500–700M foundation model pretraining | Not started |
| 6 | Instruction/chat post-training of the same model | Not started |
| 7 | Research documentation, evaluations and public release | Not started |
| 8 | Optimization, scaling and integrations | Not started |

Full plans: [Roadmap](docs/ROADMAP.md) · [Project state](docs/PROJECT_STATE.md) · [Architecture](docs/ARCHITECTURE.md) · [Pretrained adaptation](docs/PRETRAINED_ADAPTATION.md) · [Research decisions](docs/DECISIONS.md) · [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md) · [Prior art](docs/PRIOR_ART.md).

## Development

**Windows project root:** D:/Programming/03_Projects/personal-projects/05_ML/trihaera

Clone only the authorized repository:

```powershell
cd 'D:\Programming\03_Projects\personal-projects\05_ML'
git clone https://github.com/Amaan27-Buzz/trihaera.git
cd trihaera
```

Create an **independent TRIHAERA virtual environment** while copying the installed Python packages from Levis (including PyTorch), rather than sharing Levis's environment or redownloading its dependencies. Follow [Windows environment setup](docs/WINDOWS_SETUP.md).

From PowerShell in the cloned TRIHAERA folder, one command creates the separate environment and validates its PyTorch installation without downloading packages:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_windows.ps1
```

This creates or safely resumes `trihaera\.venv` using Levis's installed packages, without modifying Levis. If the original bootstrap failed with `NameError: purelib`, pull the latest `main` and rerun this command without deleting the partially created environment. See the Windows environment setup document.

Activate TRIHAERA's independent environment and run the checks:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest -q
python scripts/smoke.py --device cuda
python scripts/train_sanity.py --device cuda --steps 15
python scripts/probe_states.py --device cuda
```

The tests/smoke runs confirm software correctness only. The state probe reports per-block linear diversity (effective rank, energy and overlap) with interference enabled/disabled at the same random weights; it does not establish language ability.

**Copied Levis editable-install warning:** If you created .venv with the previous bootstrap, remove only the copied editable .pth in TRIHAERA before probing:

```powershell
$LevisEditable = '.\.venv\Lib\site-packages\__editable__.levis_mk1-0.1.0.pth'
if (Test-Path $LevisEditable) { Remove-Item -LiteralPath $LevisEditable }
```

Never remove anything inside Levis's original environment. New bootstrap runs sanitize copied Levis editable .pth references automatically.

## Pretrained adaptation research track

The first weight-bearing experiment will adapt **[HuggingFaceTB/SmolLM2-135M Base](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)** (Apache-2.0, approx. 270 MB). This is deliberately **a derived model experiment**, not the future independently pretrained 500–700M TRIHAERA foundation model.

A three-state low-rank residual can be inserted between the original Llama-style blocks while keeping pretrained weights frozen and **exactly preserving baseline logits at initialization**. Controls include an ordinary real adapter and three-state processing without mixing. This preliminary adapter **does not** carry state amplitudes coherently through all layers.

Pull the latest `main`, then:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest -q
python scripts/smoke_pretrained.py --device cuda --mode trihaera --output artifacts/smollm2_135m_trihaera_smoke.json
```

The first pretrained smoke downloads approximately 270 MB once and requires `transformers` and `safetensors` inside TRIHAERA's independent `.venv`. For installation checks, controls, licensing and limitations, read [Pretrained adaptation](docs/PRETRAINED_ADAPTATION.md).

## Next controlled local experiment

Your previous first CUDA integration passed (29 tests, exact initial parity), but produced a **9.26 RMS logit change after just two steps with lr=1e-3**. This is a stability warning, not improved capability.

The new bridge uses a scaled residual (0.1), 3e-4 learning rate, gradient clipping and fixes inactive control parameter counts. For first held-out experiments with **WikiText-2 raw** independent train/validation splits (license: CC BY-SA / GFDL), run:

```powershell
git pull --ff-only origin main
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest -q
python scripts/smoke_pretrained.py --device cuda --mode trihaera --output artifacts/smollm2_stable_smoke.json
# If needed in TRIHAERA's own .venv: python -m pip install datasets
python scripts/pilot_pretrained.py --device cuda --steps 20 --batch-size 1 --sequence-length 128 --output artifacts/pilot_135m.json
```

The pilot compares original frozen-model validation loss against three separately trained adapters on identical data/token budgets. It is **not FLOP-matched**, is not a 500–700M pretraining run, and is not proof of improved model quality. Details: [Pretrained adaptation](docs/PRETRAINED_ADAPTATION.md).

## Next: replicated architecture-control experiment (Stage 1)

The first small held-out WikiText-2 pilot (one seed, 20 steps, 12 validation windows) gave validation NLL of **3.39929** for the real adapter, **3.34908** for the three-state no-mixing adapter and **3.35118** for TRIHAERA's mixer, from a frozen-base **3.45328**. Mixer training took **4.83 s**, versus **3.07 s** without mixing and **2.39 s** for the real adapter. **This does not support an advantage from the current mixer.**

Moreover, the layerwise pretrained adapter sums the state channels *immediately after the unitary mixer*. Mathematically, that full mixer reduces exactly to a **3-coefficient complex readout** at that location. No persistent multi-state computation occurs across pretrained layers. See [architecture analysis](docs/ARCHITECTURE.md).

The new `collapsed_mixing` implementation and tests verify this reduction. The next study compares adapter variations across paired random seeds on more real held-out text:

```powershell
git pull --ff-only origin main
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest -q
python scripts/replicate_pretrained.py --device cuda --seeds 17,23,42 --steps 100 --eval-windows 64 --output artifacts/replication_135m.json
```

The 135M SmolLM2 checkpoint and WikiText-2 dataset are already cached from the first pilot. Results remain **not FLOP-matched**, and this is a pretrained adapter study, not TRIHAERA foundation pretraining.

## Persistent-state research track (new)

**Three-seed SmolLM2 results:** mixed layer-local adapters achieved mean WikiText-2 validation NLL **3.38814** vs **3.39523** without mixing, but took **30.58s vs 18.48s** training per 100-step run. The mean advantage of −0.00709 NLL is small, seed-dependent and not compute-matched. **Do not scale this old layer-local mixer.**

The new `src/trihaera/persistent.py` prototype retains a shared complex three-state carrier across all **30 frozen pretrained decoder depths**, performs a nonlinear state update before each subsequent mixing operation, and introduces output correction only through a zero-initialized projection. This avoids the previous immediate mixer/readout linear reduction. **It is still a sidecar, not a full three-state Transformer**, and has no influence on the pretrained backbone's attention.

The cached 135M weights are reused; no new large download. Test on your RTX 5060:

```powershell
git pull --ff-only origin main
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest -q
python scripts/smoke_persistent.py --device cuda --mode mixed --output artifacts/persistent_mixed_smoke.json
python scripts/smoke_persistent.py --device cuda --mode no_mixing --output artifacts/persistent_no_mixing_smoke.json
```

The smoke verifies exact original-logit parity, gradient flow, training time and GPU memory. It does **not** measure held-out language quality. Read [Persistent-state research](docs/PERSISTENT_STATE.md) and [current checkpoint](docs/PROJECT_STATE.md).

## Evaluation standard

Measure trainable parameters, FLOPs, training tokens, validation loss, inference latency, throughput and VRAM separately. Include dense, recurrent/shared-weight and interference-disabled baselines; predeclare meaningful thresholds and avoid cherry-picking. A negative research outcome is valid. **Do not claim quantum speedups or model-quality gains without evidence.**

## Repository policy

**Work directly on `main` only. Do not create feature branches or pull requests without explicit owner instruction.** See [AGENTS.md](AGENTS.md).
