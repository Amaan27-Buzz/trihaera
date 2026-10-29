# TRIHAERA engineering instructions

1. **Main only:** make changes directly on the existing `main` branch. Do not create branches or PRs unless the owner explicitly asks.
2. Research integrity comes first: distinguish hypotheses, tests, and verified experimental results. Never claim a physical quantum computer, intrinsic 3× intelligence, longer memory, hallucination reduction, or novelty without evidence.
3. Track the authoritative stage state in docs/PROJECT_STATE.md and gates in docs/ROADMAP.md. Update them whenever a substantive milestone is actually completed.
4. Keep code modular in src/trihaera; require tests for attention causality, state mixing, gradients and optimization.
5. Use fair baselines. Separate equal-parameter, equal-token and equal-compute comparisons. Record measured memory, latency and throughput.
6. Treat the 500–700M foundation model as the post-validation target; its chat model is an instruction-tuned version of that same checkpoint.
7. Local experiments must run in TRIHAERA's own .venv, not modify Levis's dependencies. Reuse the package files from Levis where practical, with relocation checks.
8. Do not begin expensive pretraining without the Stage 3 evidence gate, a proper Stage 4 training protocol and owner-controlled compute budget.
9. TRIHAERA is independent of LEVIS/PhaseMemory and PLURIMENS. Criticize questionable assumptions; do not act as a yes-man.
