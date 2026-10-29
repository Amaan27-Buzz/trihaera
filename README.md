# TRIHAERA

**Dawnlight Labs — experimental three-state, quantum-inspired Transformer**

**Current state: Stage 1 (architecture research and formalization) in progress.** A Stage 2 exploratory PyTorch prototype is included for correctness testing, not a trained language model or demonstrated architecture advantage.

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

Full plans: [Roadmap](docs/ROADMAP.md) · [Project state](docs/PROJECT_STATE.md) · [Architecture](docs/ARCHITECTURE.md) · [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md) · [Prior art](docs/PRIOR_ART.md).

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

This creates `trihaera\.venv` using Levis's installed packages, without modifying Levis. For troubleshooting and manual steps, see the Windows environment setup document.

With that local environment activated:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest -q
python scripts/smoke.py --device cuda
python scripts/train_sanity.py --device cuda --steps 15
```

The tests/smoke runs confirm software correctness only. They do not establish language ability or empirical architectural value.

## Evaluation standard

Measure trainable parameters, FLOPs, training tokens, validation loss, inference latency, throughput and VRAM separately. Include dense, recurrent/shared-weight and interference-disabled baselines; predeclare meaningful thresholds and avoid cherry-picking. A negative research outcome is valid. **Do not claim quantum speedups or model-quality gains without evidence.**

## Repository policy

**Work directly on `main` only. Do not create feature branches or pull requests without explicit owner instruction.** See [AGENTS.md](AGENTS.md).
