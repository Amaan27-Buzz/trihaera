# TRIHAERA — authoritative development checkpoint

**Updated:** 2026-10-30  
**Repository:** Amaan27-Buzz/trihaera  
**Authorized working branch:** **main ONLY**; no new branches/PRs without owner's explicit request.

## Stage status

| Stage | Current status |
|---|---|
| 1. Architecture research/formalization | **IN PROGRESS — specification v0.2 and representation diagnostics added** |
| 2. Reference implementation | **STARTED — CPU and local NVIDIA CUDA correctness pass** |
| 3. Compute-matched architectural validation | Not started |
| 4. Training recipe research | Not started |
| 5. 500–700M foundation pretraining | Not started |
| 6. Chat instruction tuning of same checkpoint | Not started |
| 7. Research/public release | Not started |
| 8. Optimization/scaling | Not started |

## Confirmed owner local results (2026-10-30)

- Windows PowerShell environment cloning resumed successfully. 45,625 files, ~3.763 GB copied from Levis's **site-packages** into independent TRIHAERA `.venv`, no PyTorch download or Levis modification.
- Local TRIHAERA Python: `D:\Programming\03_Projects\personal-projects\05_ML\trihaera\.venv\Scripts\python.exe`.
- PyTorch **2.14.0+cu130**, CUDA detected NVIDIA **RTX 5060 Laptop GPU (8 GB)**.
- Initial 10 correctness unit tests passed on CPU, 5.94 s.
- CUDA smoke: TRIHAERA **369,378** params, loss **5.5251**; dense **369,312** params, loss **5.5003** (untrained random initialization; NOT a capability comparison).
- CUDA 15-step fixed-sequence optimizer sanity: TRIHAERA loss **3.8787→3.2051**, dense loss **3.8786→3.2095** (toy sequence memorization, NOT language-model performance).
- One copied editable `.pth` file referenced Levis: `__editable__.levis_mk1-0.1.0.pth`. Remove that **copy inside TRIHAERA only**; do not alter Levis.

## Research findings

1. Input three-state representation is mathematically rank one (three phase copies of one embedding).
2. Unitary mixing cannot independently increase linear rank; diversity must be learned through state-conditional residuals and nonlinearities.
3. Classically simulated three-complex-state activations can be represented as six real channels, thus no automatic quantum speedup or memory compression.
4. Conventional causal attention remains phase-covariant under a uniform state-specific phase; that is a restriction on how interference can help.
5. Stage 1 prior-art review now includes phase-coherent Transformers and PRISM (2026), increasing novelty risk.
6. Added read-only diagnostic probe of effective rank, overlap, energy fractions and coherent recombination, with interference-on/off controls. **Owner's GPU probe output is not yet available**.

## Next actions

1. Owner: `git pull --ff-only origin main`, remove only TRIHAERA's stale copied Levis editable .pth, run `python -m pytest -q` and `python scripts/probe_states.py --device cuda`.
2. Inspect resulting per-block state effective ranks and interference on/off logit difference; record observations without overinterpreting random-init data.
3. Implement and measure strong controls (plain complex shared weights, real six-channel, frozen phase, state-count variants); report measured memory and compute.
4. Complete mathematical comparisons, decide how to make three state channels actually nonredundant, and set the Stage 3 gate.
5. Only after architecture validation and a researched, costed training method, attempt full 500–700M foundation pretraining and instruction-tune the same checkpoint.

See [architecture](ARCHITECTURE.md), [prior art](PRIOR_ART.md), [experiments](EXPERIMENT_PROTOCOL.md), [roadmap](ROADMAP.md).

**Do not advance the stage or claim superior intelligence based on 15 training steps or an effective-rank statistic.**
