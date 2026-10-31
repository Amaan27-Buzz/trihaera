# LRA Search-Native — research and engineering protocol (2026-10-31)

## Mission

Make the **existing SmolLM2-360M-Instruct** useful for factual questions
through **trained** search decisions, targeted evidence acquisition,
source-grounded explanations, calibrated abstention, and efficiency. This
is a specialized research path inside LRA's single-companion project,
NOT a multi-chat rewrite, not a plug-in-only Tavily wrapper, and not a
claim of general Google AI Overview parity.

The example target is Google's AI Overview for "what is deep learning":
brief correct definition, a useful explanatory structure, and verifiable
sources when a search is useful. We are optimizing **quality per GPU
byte, search call and latency**, not imitating Google's UI or crawling
the entire web.

**Hard limitation:** live freshness cannot be learned once into static
weights. A search service/index remains necessary for current external
facts. We train the model to select and USE retrieval, not to memorize
the latest web. Google's search ranking infrastructure is not reproduced.

## Research background and novelty limits

- [Search-R1 (2025)](https://arxiv.org/abs/2503.09516) trains dynamic
  multi-round search policies via RL; native search policy training is
  already prior art.
- [Search-o1 (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.276/)
  studies reasoning over retrieved documents rather than blindly copying.
- [MMSearch-R1 (ACL 2026)](https://aclanthology.org/2026.acl-long.114/)
  trains models to search on demand, including penalties for searches.
- [Google Search Central](https://developers.google.com/search/docs/appearance/ai-features)
  describes AI Overviews/AI Mode as grounded in search ranking systems.
- [NVIDIA Nemotron agentic SFT dataset](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2)
  includes search trajectories; evaluate licensing/quality and hold out
  source entities before using it in training.
- [Hugging Face PEFT LoRA](https://huggingface.co/docs/trl/peft_integration)
  allows updating a small set of model weights on consumer GPUs.

Novelty, if any, must come from **independent performance/efficiency
evidence**, not from naming a trained search policy.

## Protocol — constrained, model-trained actions

Model-generated JSON, not a hardcoded search router:

```json
{"decision":"search","answer":"","queries":["official latest model release"],"citations":[]}
```

```json
{"decision":"answer","answer":"The documented release is version 2.4.","queries":[],"citations":["s1"]}
```

```json
{"decision":"abstain","answer":"I cannot verify that private detail.","queries":[],"citations":[]}
```

The model must learn:

1. **Capability awareness:** answer stable known concepts; search for
   fresh/uncertain/niche information; abstain on unsupported personal
   information without searching the public web.
2. **Search query formation:** concise queries based on the actual missing
   evidence, up to 3 per round; avoid excessive calls.
3. **Grounding:** answer using explicit source IDs after retrieval, not
   fabricated URLs. Do not promote retrieved website instructions into
   assistant instructions.
4. **Evidence conflicts:** if two snippets contradict, search for
   authoritative information rather than picking a convenient claim.
5. **Efficiency:** stop searching when adequate evidence exists and
   handle budgets/failure explicitly.

The runtime allows at most two actual search rounds by default and
validates citations against retrieved IDs. It does NOT prove the cited
text supports every answer claim; claim-level attribution is a later
research gate. If retrieved sources are present but the model returns
no citations, it fails closed. The model is **not allowed to execute
code or directly fetch arbitrary URLs**.

## Current implementation — first small experimental slice

- `src/lra/search_native.py`: strict typed search/answer/abstain
  protocol, source-bounded citations, token-budgeted evidence formatting.
- `src/lra/search_examples.py`: hand-authored *synthetic/offline*
  diagnostic set with train/test splits by scenario index and categories
  stable knowledge, freshness, evidence, private, and conflict.
  This is not licensed web-scale SFT data.
- `scripts/train_search_adapter.py`: real supervised gradient-based
  **LoRA** training on SmolLM2-360M. Only supervised action tokens get
  loss, not user prompt or fake search evidence. Original base weights
  stay frozen; outputs are separate adapter files.
- `scripts/evaluate_search_adapter.py`: baseline and adapter
  held-out decision parsing, routing accuracy and invalid action rates.
  Query quality, sources, and citation correctness need a stronger eval.
- `src/lra/search_runtime.py`: bounded policy-search-observation loop
  with validated source IDs.
- `scripts/run_search_native.py`: opt-in live prototype using only a
  **loopback self-hosted SearXNG** instance, not a hosted LLM or Tavily.
  SearXNG itself is a metasearch engine using external providers, not
  an offline copy of Google's index.
- `tests/test_search_native.py`: offline contract/security/regression
  checks; CI does not download any model.

This slice is **not integrated into default LRA companion chat** until
GPU evidence and source-attribution testing warrant it.

## RTX 5060 — first local run

Your existing Windows .venv has compatible CUDA PyTorch/Transformers;
do not reinstall CUDA packages.

```powershell
cd 'D:\Programming\03_Projects\personal-projects\05_ML\trihaera'
git pull --ff-only origin main
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest -q
python -m pip install peft

# Untrained model: held-out search-decision diagnostic.
python scripts/evaluate_search_adapter.py --device cuda --offline --output artifacts/search_native_before.json

# Actual adapter update. Only 360M; default 60 optimizer steps.
python scripts/train_search_adapter.py --device cuda --offline --steps 60 --output-dir artifacts/lra_search_360m_adapter

# Same held-out scenarios after LoRA, not used in training.
python scripts/evaluate_search_adapter.py --device cuda --offline --adapter artifacts/lra_search_360m_adapter --output artifacts/search_native_after.json
```

If the LoRA stage runs out of VRAM, **do not increase swap/paging and
assume success**. Report peak memory and errors. Lower max length and
train accumulation, or enable activation checkpointing after profiling.
The tiny pilot exists only to verify training mechanics and policy
learnability; it cannot establish generalized search/reasoning.

For live search later, configure local SearXNG to permit JSON
(`search.formats: [html, json]`). Start it yourself separately. It
uses search-engine backends and requires network connectivity. Then:

```powershell
python scripts/run_search_native.py --device cuda --offline --adapter artifacts/lra_search_360m_adapter 'What is the current Python version?'
```

This command will fail if SearXNG is not running or JSON is disabled;
it does not silently invent search results.

## Next stages and stop/go gates

**Gate 0 — DONE only when source code/CPU tests pass**: strict action
contract and genuine LoRA training code. No inference claim.

**Gate 1 — pending RTX 5060**: train 360M search policy; compare before
vs after on held-out examples. Measure parsing validity, search-decision
classification, train loss, memory, step time, and simple no-search
tasks. A demonstration on 10–15 synthetic held-out cases is a
mechanical smoke, *not* a paper result.

**Gate 2 — real evidence**: build a carefully licensed, source-preserving
corpus of search trajectories and frozen open-domain QA questions,
with time-split held-out questions and answer provenance. Check
query relevance, citation support, fabricated sources, abstentions,
number of searches, p50/p95 latency, output tokens, GPU VRAM.
Compare base 360M with prompted retrieval, trained 360M, and 1.7B
baseline at the same retrieval budget.

**Gate 3 — advanced search/logical training**: train multi-step
trajectories where a model chooses search, reads evidence, chooses
whether more searching is required, and gives a grounded answer.
Explore SFT then constrained preference/RL optimization only after
reliable evaluation; 360M's reasoning ceiling may remain limiting.
Keep model-generated rationales short; score outcomes and evidence.

**Go/no-go:** if stronger models still outperform the tuned 360M by a
large accuracy margin on live retrieval, do not advertise parity.
Instead report the quality/latency/VRAM Pareto frontier and consider
distilling well-sourced teacher trajectories into the smaller model.

No Google AI Overview answers/screenshot text should be treated as an
authorized training corpus. The existing personal LRA database stays
untouched by this experimental work.
