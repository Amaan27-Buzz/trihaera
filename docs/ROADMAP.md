# TRIHAERA — eight-stage development plan

These are research gates, not progress percentages. Stages may overlap where sensible; no unproven result may be presented as achieved.

## Stage 1: Formalize the architecture (IN PROGRESS)

Specify classical three-state amplitude representation, shared weights, complex causal attention, state mixing, output readout, numerical properties, cost model and proposed novelty. Review nearest work on qutrit representations, complex attention, ALBERT, Universal Transformers and recursive/shared-weight networks.

**Exit:** reviewed operator equations; testable claims and explicit null hypothesis; realistic originality position; ablation checklist.

## Stage 2: Implement verified reference model (STARTED)

Build modular PyTorch architecture and dense decoder. Check shape, causal leakage, gradients, unitary mixer, numerical robustness, device precision and initialization. Add reproducible reference training and checkpointing, then optimize bottlenecks only when profiled.

**Exit:** robust CPU/CUDA tests, measured memory/throughput, reference reproductions, well-structured package.

## Stage 3: Prove or reject useful advantage (NOT STARTED)

Compare dense, weight-shared/recurrent and three-state decoders; disable interference and vary state count. Run separate equal-parameter/equal-token and equal-compute comparisons, three or more seeds where affordable, held-out language-model loss and grounded downstream tasks.

**Exit:** predeclared reproducible efficiency/capability improvement, or documented stop/pivot. One extra-compute win is not proof.

## Stage 4: Research training methodology (NOT STARTED)

Analyze published Qwen, DeepSeek, Llama, Gemma, SmolLM and other model/lab methods critically. Audit data permissions and provenance, dedup/filter, mixtures, tokenizer, optimization, LR schedules, token budgets, continual evaluation, multi-GPU strategy and post-training; adapt to phase stability and state collapse.

**Exit:** documented reproducible and costed TRIHAERA training protocol, with sources and experiment evidence.

## Stage 5: Pretrain 500–700M foundation model (NOT STARTED)

Choose final parameter configuration (650M provisional), source and process legal high-quality data, allocate substantial compute, train from scratch, checkpoint with recovery, measure cross-domain text quality and efficiency.

**Exit:** validated base model with reproducible training artifact and evaluated general language capability.

## Stage 6: Instruction-tune the same model (NOT STARTED)

Supervised instruction data, optional preference optimization where justified, chat formatting, code/math/factuality tests, safety behavior and inference packaging.

**Exit:** TRIHAERA-Instruct checkpoint derived from the Stage 5 base—not a separate pretrained model.

## Stage 7: Research/public release (NOT STARTED)

Paper-style technical report, ablations, negative findings, reproducible charts, transparent limitations, data/weight licenses, model cards, optional Hugging Face weights/demo and Dawnlight research page.

**Exit:** truthful, reproducible public-facing research release; decide IP timing before disclosing unpublished claims.

## Stage 8: Scale and optimize (NOT STARTED)

Only if evidence warrants: 1B/3B/7B+, better inference kernels, KV cache and longer context, training cost improvements, selective Dawnlight integration, compute/funding partnerships.

**Exit:** justified scaling plan rather than parameter-count vanity metrics.

## Rules

- No staged declaration of success without exit-gate evidence.
- Tiny correctness tests are permitted; tiny synthetic loss alone cannot validate reasoning or broad language capabilities.
- Parameter matching and FLOP matching must be reported separately.
- Work on main only. No branches/PRs without an explicit instruction.
