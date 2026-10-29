# TRIHAERA — authoritative development state

**Updated:** 2026-10-30  
**Repository:** Amaan27-Buzz/trihaera  
**Authorized development branch:** main ONLY; no new branches or pull requests unless the owner asks.

## Current stage

- Stage 1 — Architecture research and formalization: **IN PROGRESS**.
- Stage 2 — First experimental reference implementation: **STARTED**, awaiting owner's CUDA results.
- Stages 3–8: **NOT STARTED**.
- No model has been pretrained, instruction-tuned or validated against competitive baselines.

## Agreed product/research goals

- Three concurrent, *classically simulated* complex state channels labelled −1, 0, +1.
- Shared Transformer parameters and learnable phase-based state mixing.
- Research goal: better quality/efficiency at fixed parameters and compute; the alleged equivalence of 12 to 36 layers is **not established**.
- A trained 500–700M general-purpose text foundation model, provisional name TRIHAERA-650M-Base, conditional on architectural proof.
- Chat model is a post-trained variant of that very same model, provisional TRIHAERA-650M-Instruct.
- Scale further only with sufficient empirical evidence and training resources.
- Independent Dawnlight Labs project, distinct from LEVIS and PLURIMENS.

## Current code

- Config and conventional decoder baseline.
- Three-state complex hidden representations, causal attention using shared real weights, phase-dependent residuals, and a 3×3 unitary state-mixing operator.
- Minimal forward/backward, causality and interference tests; smoke and single-sequence optimizer sanity commands.
- Tests on the author's initial scaffold were run on CPU; CUDA tests on owner's RTX 5060 8GB laptop **have not been run or confirmed**.

## Remaining critical work

1. Owner clones main and creates a **separate** local .venv, copying installed packages from Levis (no PyTorch redownload), then tests CUDA and shares output.
2. Audit whether state channels retain non-redundant information and whether the apparent interference has a conventional real-network equivalent.
3. Measure parameter counts, FLOPs, peak VRAM, step time and mixed-precision limitations.
4. Implement shared-weight/recurrent baseline and interference ablations; define experimental go/no-go before data runs.
5. Design a true multi-seed, compute-matched validation benchmark. Avoid promoting tiny synthetic optimization tests as language capability.
6. After evidence gate, research and formalize a credible training pipeline for the 500–700M foundation model.

**Do not mark Stage 1 complete solely because code exists.**

## Hardware / location

Windows project folder: D:\Programming\03_Projects\personal-projects\05_ML\trihaera  
Levis package source: D:\Programming\03_Projects\personal-projects\05_ML\levis-mk1\.venv  
Local GPU: RTX 5060 Laptop GPU, 8GB VRAM.

No automatic modifications to Levis; target TRIHAERA .venv must be independent. See docs/WINDOWS_SETUP.md.

## Local environment bootstrap defect — 2026-10-30

- First Windows bootstrap hit `NameError: name 'purelib' is not defined` because PowerShell 5.1 stripped nested quotes in a Python `-c` argument.
- Root cause corrected by moving introspection into a separate Python script and supporting verified resume of a partially created target `.venv`.
- Python helper and CPU prototype tests passed locally on the development host; **owner's Windows bootstrap/CUDA retest is pending**.
