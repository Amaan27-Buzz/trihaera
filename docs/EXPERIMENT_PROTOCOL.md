# Controlled experiment protocol — preregistration draft

**Hypothesis:** Three-state interference improves held-out language-model quality and/or task results at competitive parameter storage and training/inference compute.

**Null:** apparent gains can be explained by additional operations, activation width, ordinary complex computation, data differences or random variance.

## Comparators

1. Ordinary causal Transformer, equal-parameter.
2. Shared-weight / looped Transformer with controlled effective computation.
3. TRIHAERA three-state reference implementation.
4. TRIHAERA with state mixing disabled.
5. Fixed phase / frozen mixer, equal-width real and complex alternatives, and one/two/four-state alternatives where implemented.

## Evaluation regimes

**A. Equal parameters and tokens:** report additional FLOPs explicitly, never confuse this with equal-compute efficiency.  
**B. Equal FLOPs and training data:** adjust architecture, steps and batch schedule as warranted; show tradeoff curves.  
**C. Inference cost:** compare prefill latency, decode throughput, parameter bytes, activation/peak VRAM at specified contexts.

Use genuinely held-out data, common tokenizer/preprocessing, multiple seeds, and a *predeclared* meaningful improvement threshold after measuring baseline variance.

## Experiment ledger

For every run capture git SHA, dataset manifest+hash, tokenizer, seeds, Torch/CUDA versions, model config, parameter counts, optimizer, token count, FLOP estimates, step timing, learning curves, eval NLL/perplexity, inference latency and GPU peak VRAM. Archive negative results and failed runs.

Do not extrapolate hallucination reduction, long-context advantage or general reasoning capability from synthetic memorization exercises.
