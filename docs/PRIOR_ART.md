# Prior art / novelty risk register (Stage 1)

**Updated 2026-10-30. Not a systematic literature or patent search.**

A broad claim that quantum-inspired phases, superposition, shared Transformer parameters or improved compact language models are new would be false. TRIHAERA's precise combination may still yield a testable contribution, but novelty has to be supported by more than three labels and unitary state mixing.

| Research | Publication | Key overlap | What TRIHAERA still needs to distinguish |
|---|---|---|---|
| Universal Transformers | https://arxiv.org/abs/1807.03819 | Repeated shared computation | Three-state mixing is not automatically additional recursive depth |
| ALBERT | https://arxiv.org/abs/1909.11942 | Cross-layer parameter sharing | Evaluate true FLOPs/quality tradeoff |
| Deep Complex Networks (ICLR 2018) | https://www.microsoft.com/en-us/research/publication/deep-complex-networks/ | Complex neural operators, normalization and activation | Show interference provides benefit beyond ordinary complex networks |
| Relaxed Recursive Transformers (ICLR 2025) | https://arxiv.org/abs/2410.20672 | Weight sharing and lightweight specialization | Compete with strong shared-weight/recursive baselines |
| Superposition in Transformers: A Novel Way of Building Mixture of Experts (2025) | https://arxiv.org/abs/2501.00530 | Superposed neural hidden representations | Paper targets autoencoder blending / domain adaptation, not our exact three-channel decoder |
| Quixer (2024) | https://arxiv.org/abs/2406.04305 | Quantum-inspired Transformer family | Avoid implying physical quantum parallelism |
| Complex-Valued Phase-Coherent Transformer (May 2026 preprint) | https://arxiv.org/abs/2605.10123 | Phase-coherent sequence processing and complex attention | Particularly important: classical complex attention is an established active research area |
| PRISM: Language as a Wave Phenomenon (ICML 2026) | https://proceedings.mlr.press/v306/yildirim26a.html | Phase-based sequence modelling, interference, constrained unit phase | Must control for phase-enabled real baselines; descriptive state mixing is insufficient novelty |

## Research position

- **Already known:** complex-valued networks, phase interference, state combination, recursive sharing and multiple computational branches.
- **Possibly distinctive:** a particular learned 3x3 unitary channel mixer coupled to state-conditioned Transformer computation and a validated quality-per-FLOP gain.
- **Not established:** novelty, IP freedom, quantum advantage, longer context, reduced hallucination or 12-vs-36 layer equivalence.
- **High risk:** our current attention is equivariant to global channel phases, and the input state matrix is rank one. Extra channels can be redundant.
- **Strong falsification:** a parameter-and-compute-matched real-valued or complex-valued baseline reproduces any gain.

## Next reading/questions

1. Compare PRISM and Phase-Coherent Transformer operator equations, data scale and ablation definitions.
2. Explore whether unitary mixing is representationally reducible to an ordinary learned 3x3 real/complex feature mixer under current nonlinearities.
3. Run state-rank probes during *actual held-out language modelling*, not just random initialization.
4. Reassess if fixing state count at three is justified.
5. Conduct broader prior-art and patent search before public originality claims.
