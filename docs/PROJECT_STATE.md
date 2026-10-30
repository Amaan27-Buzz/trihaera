# TRIHAERA — authoritative development state

**Updated:** 2026-10-30  
**Repository:** Amaan27-Buzz/trihaera  
**Branch rule:** Work on **main only**, no branches or PRs unless explicitly authorized.

## Stages
- Stage 1 (formal mathematics, originality, research controls): **IN PROGRESS**; matched classical-real representation added.
- Stage 2 (modular reference implementation): **STARTED**; pretrained frozen-backbone adapter and persistent sidecar operational.
- Stages 3–8: **NOT STARTED** (no equal-compute proof, full training recipe, independently pretrained 500–700M base, instruction-tuned derivative, or scaling).

## Environment and verified owner tests

Windows `D:\Programming\03_Projects\personal-projects\05_ML\trihaera`; independent .venv cloned without reinstalling large PyTorch packages. RTX 5060 Laptop GPU 8GB, CUDA PyTorch 2.14.0+cu130, Transformers 5.18.0, datasets 5.1.0. Cached SmolLM2-135M Base and WikiText-2. **60 tests passed on owner's machine** before the latest real-equivalent controls. Previous full CI passed.

## Structural research findings

- Initial three-state input representation starts with rank 1; unitary mixing by itself cannot increase rank.
- Old per-layer adapters' unitary mixer plus immediate sum algebraically collapses into a 3-coefficient complex readout.
- Current sidecar carries its state across 30 pretrained depths and uses nonlinearity between mixers, so that immediate collapse does not apply.
- All complex computations admit a **six-real-channel equivalent**; no quantum acceleration is implied. The new `real_equivalent` path explicitly reproduces complex computations with matching gradients.
- New `real_matched_orthogonal` starts from the same injection/rotations as complex but can learn the wider O(6) group. This is a stronger, more fairly initialized comparator than old `real_six` (which used different injection coefficients).
- Sidecar still **does not feed back into pretrained attention**. It is a derived-model research adapter, not a new pretrained foundation model.

## Previous 100-step local-adapter experiment

WikiText-2, three seeds and 64 validation windows. Frozen SmolLM2 baseline NLL 3.705553. Local-adapter means: real 3.403005; complex no-mixing 3.395234; complex mixing 3.388140. Mixed training 30.582s vs no-mixing 18.482s. One seed drove most of the difference. Local mixer's immediate-sum reduction makes it unsuitable as the final architecture.

## Most recent 100-step persistent-state experiment (owner supplied)

Three seeds (17,23,42), four modes, 64 WikiText-2 validation windows, 60 tests passed.

| Variant | Mean NLL ↓ | NLL SD | Mean training time |
| --- | ---: | ---: | ---: |
| mixed (3 complex channels, U3) | **3.538331** | 0.009588 | 26.2015s |
| no_mixing (3 complex) | 3.539520 | 0.009063 | 12.0664s |
| real_six (6 real, O6) | 3.565990 | 0.006624 | 19.9907s |
| real_six_no_mixing | 3.598689 | 0.002931 | 7.5680s |

Paired mixed-minus-no-mixing NLL: −0.002011, −0.000925, −0.000629. **Mean advantage just −0.001188**, while mixed training is ~2.17× slower. The old six-real comparison is initialization-confounded. **No established useful advantage from mixing.** Intermittent Hugging Face DNS errors were handled by cache; new script prefers cached pretrained weights.

## New changes and next owner experiment

- `src/trihaera/real_equivalent.py`: six-real exact implementation of the constrained complex mixer and matched-start unrestricted real O(6) control.
- Expanded persistent-state mode names: `real_equivalent`, `real_equivalent_no_mixing`, `real_matched_orthogonal`, `real_matched_orthogonal_no_mixing`.
- Tests verify exact real-vs-complex forward/gradient equivalence, initial O(6) parity, trainability, and inactive mixer freeze. Local isolated tests passed; latest GitHub CI status must be checked.
- Next: run paired 100-step, 3-seed matched-start replication on cached data/model. Report mean/paired NLL, training time, parameter counts, and peak GPU allocations.

```powershell
git pull --ff-only origin main
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest -q
python scripts/replicate_persistent.py --device cuda --modes mixed,no_mixing,real_equivalent,real_matched_orthogonal --seeds 17,23,42 --steps 100 --eval-windows 64 --output artifacts/real_matched_replication_135m.json
```

**Do not advance** to expensive full-scale training based on this short, non-FLOP-matched experiment; Stage 1 gates remain open.

See [PERSISTENT_STATE.md](PERSISTENT_STATE.md), [ARCHITECTURE.md](ARCHITECTURE.md), [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md), [DECISIONS.md](DECISIONS.md).
