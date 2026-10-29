# Prior art — research reading list, NOT a novelty certification

This list is provisional; deeper literature/patent research is needed before publishing claims.

- Universal Transformers — recurrent computation and repeated attention: https://arxiv.org/abs/1807.03819
- ALBERT — parameter sharing across Transformer layers: https://arxiv.org/abs/1909.11942
- Relaxed Recursive Transformers — repeated blocks with small layer-specific adaptations: https://arxiv.org/abs/2410.20672
- Quixer — quantum Transformer research: https://arxiv.org/abs/2406.04305
- Superposition in Transformers — model representation superposition: https://arxiv.org/abs/2501.00530

Questions for originality and efficiency:
- Is TRIHAERA mathematically distinct from conventional complex-valued or wider real-valued networks?
- Why exactly 3 states? Would 2 or 4 be better under matched compute?
- Does phase mixing make a causal language model better, or only increase compute?
- Does it collapse to existing gating, mixing or shared-weight mechanisms?
- Can it be efficiently implemented on classical GPUs?

The words **quantum** and **superposition** describe inspiration/representations, not actual hardware quantum speedups.
