# Conversational AI — Dawnlight Labs

> **Research reset · October 2026 · Working name: LRA (subject to change)**

A local-first, open-source conversational AI platform built around **small, retrained and highly optimized language models**. It should deliver useful, fast, grounded everyday conversations on consumer hardware while supporting multiple independent chats, shared long-term memory, effectively long conversations, and adaptive interactive answers.

**This project starts with the models.** The serving harness and user-facing applications come only after we can demonstrate worthwhile capability, reliability, and efficiency improvements.

**Status:** Vision and research specification only. No active implementation, trained adapter, runnable demo, evaluation result, or finished app exists in the current branch.

## 1. Product vision

Build the kind of everyday AI chat application people expect from ChatGPT, but local-first, lightweight, open, and adapted to small-device inference.

The final product should provide:

- **Multiple conversations:** a traditional chat sidebar, independent conversational threads, search and organization; not one permanent chat.
- **Shared memory:** user-approved facts, preferences and ongoing goals available across chats, plus private context that belongs only to a specific thread. Corrections and deletions must be respected.
- **Selectable locally optimized models:** at least two distinct research tracks, starting with a SmolLM-class model and Qwen3.5-0.8B. These are retrained/optimized versions we evaluate, not merely untouched downloaded checkpoints.
- **Grounded search:** the model learns *when it needs evidence, what to search for, how to read sources, when to search again, how to cite, and when to say it cannot verify*. Search infrastructure remains a separate necessity for live information.
- **Long effective conversations:** retain and retrieve useful information from much longer histories than the prompt currently held in GPU memory; reduce context loss and instruction drift without pretending the native attention window is infinite.
- **Context-health indicator ("drunk meter"):** an understandable, empirically calibrated warning when a thread's available context is likely to be degraded. Recommend continuing in a new chat with a user-reviewed handoff when warranted.
- **Generative UI in answers:** structured text, comparisons, charts, timelines, explanations, calculators, diagrams and safe interactive controls where they materially help. Plain prose remains the default when it is better.
- **Real local ownership:** transparent storage, export, removal, optional encryption, explicit network behavior, and no mandatory hosted LLM subscription.
- **Conversational focus:** friendly, capable discussion, learning, everyday research and explanations. **Not** a coding agent, autonomous computer operator, or heavyweight orchestration product.

**Product form:** web application first; then local desktop apps for Windows and macOS; then mobile. Cross-platform plans must not compromise the model-first experimental sequence.

## 2. Model candidates

| Candidate | Official checkpoint | Published native context | Role |
| --- | --- | --- | --- |
| SmolLM | [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | 8,192 tokens | Smallest practical starting baseline for optimization and training tests |
| Qwen | [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 262,144 tokens | Stronger compact hybrid-model candidate, including non-thinking and thinking modes |

The Qwen3.5-0.8B model's official context length is **262K**, not 128K. This is a model specification, **not** evidence that 262K will fit at acceptable latency or memory usage on an 8GB GPU. Native model context, maximum technically usable context, and *reliable effective context* are three different things.

**Initial development machine:** Windows laptop with NVIDIA RTX 5060 Laptop GPU (**8GB VRAM**). Optimize for that first; do not assume a datacenter GPU, rented server, or large-scale scratch pretraining. Qwen3.5's vision components may be assessed separately; the initial target is **text conversation**.

We will test the two model families independently rather than assume that every training recipe or inference kernel supports both. SmolLM is a conventional Transformer; Qwen3.5-0.8B uses a different hybrid architecture. Training and acceleration must respect those differences.

## 3. Research objectives — MODEL FIRST

### A. Capability-aware small models

Train the model to make better *decisions* in addition to writing better text:

1. **Answer** stable, familiar questions directly when it can do so reliably.
2. **Search** when a question is fresh, niche, contentious, underspecified, or requires verification.
3. **Investigate** by forming focused queries, interpreting results, identifying gaps, reconciling conflicts, and stopping when sufficient evidence exists.
4. **Cite** actual retrieved sources with support for individual factual claims, not just plausible-looking links.
5. **Abstain or qualify** when information is absent or unreliable. Do not invent personal data or authoritative sources.
6. **Maintain conversation state** across follow-ups, corrections and changing topics without repeating synthetic benchmark phrases.
7. **Choose presentation mode:** straightforward text unless structured/interactive UI would clearly improve comprehension.

Target **training/model behavior**, not just a rigid prompt, hardcoded keyword router, or a Tavily-style agent wrapper. Inference still requires safe execution around model-produced search actions and access to some search backend/index. Training cannot permanently encode live web freshness into weights.

Candidate techniques, subject to measurable evidence: curated supervised fine-tuning, LoRA/QLoRA where supported, preference optimization, knowledge distillation from verified teacher trajectories, compact planning/action traces, retrieval-grounded QA, quantization, batching, efficient attention/state handling, and inference profiling.

A two-dozen-example scripted SFT demonstration is **not** enough to claim search competence. Training corpora must be licensed, deduplicated and split to prevent source/answer leakage.

### B. Efficient inference on local hardware

Optimize **quality at a fixed cost** rather than simply shrinking weights:

- Cold start, first-token latency, sustained output tokens/second, energy where measurable, memory allocated/reserved, and responsiveness while browsing or using a desktop.
- Multiple quantization and precision settings; validate quality after every speed optimization.
- Context-dependent activation and cache memory rather than treating weight size as total VRAM cost.
- Fast and stable structured outputs and multi-turn generation.
- Baseline vs our retrained/optimized checkpoint, plus a stronger reference model at matched search and latency/token budgets.

Avoid claims of 10x speed or Google-level intelligence until independently measured.

### C. Beyond the perceived context window

The aim is **effectively longer and more reliable conversation**, not an impossible unlimited native attention window.

Investigate:

- A complete, durable timeline of messages inside each chat.
- Hierarchical summaries or episodic checkpoints with provenance, plus source-linked retrieval of exact old turns.
- A separate shared-memory layer spanning the user's chats.
- Adaptive selection of recent turns, long-term facts, open threads and retrieved evidence.
- Targeted context rehydration, reversible compaction and source-linked handoffs between chats.
- Model training on long conversations, conflicts, delayed callbacks and corrected memories.
- Where justified, native context optimization/extension **validated by long-context tests**, not merely increasing a configuration number.

The user should feel that more of a very long conversation remains usable, while the system remains honest about what it can and cannot recover.

**Important boundaries:** SmolLM2-360M has an 8K native window; Qwen3.5-0.8B advertises 262K. We must benchmark *real usable window length* on the target GPU. Storage and retrieval can exceed these windows in total historical volume, but they are not equivalent to the model attending to all original tokens simultaneously.

### D. Calibrated conversation-health / “drunk” meter

This is a product feature to research, **not a magic hallucination detector**.

Candidate measurable signals:
- Active context utilization and tokens displaced by compaction.
- Number and age of unresolved or unrecoverable referenced facts.
- Recall failure on inserted long-range probes and temporal contradiction tests.
- Instruction-following drift, contradictory responses and unsupported citation claims.
- Retrieval failures, stale evidence and uncertainty-calibration indicators.
- Differences between fresh-chat and continued-chat performance.

Expose an understandable status (e.g. **Healthy / Getting crowded / Consider a fresh chat**) with explanatory evidence, **not a fabricated hallucination percentage**. The indicator must be calibrated on held-out data and tested for false alarms. It should never replace real source verification.

If a restart is advisable, the application may offer **Start fresh with context**: preserve shared memory and transfer a concise, editable, source-linked summary of the current subject to a new chat. The user decides; there is no forced reset.

### E. Model-native structured / generative UI

Reference: [OpenAI's October 2026 Intelligent UI announcement](https://openai.com/index/gpt-6-for-everyone/). We aim to learn the useful design principle, not to copy a proprietary implementation.

- Models learn when to use **plain text** vs **structured presentation** vs **small interactive components**.
- Outputs use a **versioned, validated, typed schema** and a bounded component library. The harness and clients render it into trusted native components.
- A future web renderer can support streaming partial layout, interactive charts, tables, diagrams, calculators, comparisons, and educational controls.
- No arbitrary model-generated executable JavaScript, HTML, filesystem/network authority or unrestricted browser actions.
- Visual quality, factual accuracy, accessibility, latency, token cost and interactivity must all be evaluated.
- Do not build the UI framework before proving the small models can reliably emit and use its schema.

## 4. Model evaluation requirements

Model quality is judged by a repeatable benchmark suite and **raw generated answers**, not by polished demos.

| Evaluation | What must be tested |
| --- | --- |
| Conversational behavior | Natural dialogue, direct instruction following, follow-ups, corrections, persona consistency |
| Grounded factual answers | Attribution to authoritative sources, source entailment, disagreement resolution, freshness |
| Search decisions | Search precision/recall, unnecessary calls, query relevance, number of search rounds, latency |
| Honesty | Fabricated citations, private facts, dates, unsupported claims, abstention precision |
| Long-session continuity | Recall at increasing distances, topic switching, summary errors, correction retention |
| Context health | Reliability prediction, calibration, false alarms, safe handoff to a new chat |
| Generative UI decisions | Correct schema, renderability, usefulness, accessibility, proportionate complexity |
| Efficiency | Tokens/sec, first-token latency, total time, VRAM, RAM, quality degradation after quantization |

Required experimental discipline:
- Reproducible environment, data provenance, fixed splits, seeds, budgets and saved raw outputs.
- **Separate training, validation and truly unseen evaluation questions**; prevent templated paraphrase leakage.
- Compare with the original base models and stronger references under equivalent retrieved evidence and search budgets.
- Use authentic source-backed evaluations alongside synthetic diagnostics.
- Document failures instead of rewriting benchmarks until they pass.
- Apply strict go/no-go gates before building downstream layers.

Historical TRIHAERA/LRA trials are preserved in Git history for lessons, **not** carried over as evidence for this new model program.

## 5. Planned architecture — AFTER model validation

### Phase 1 — Trained and optimized models (START HERE)

Deliverables:
- Verified baselines for SmolLM2-360M and Qwen3.5-0.8B on the 8GB GPU.
- Genuine training pipelines, evaluation dataset/protocol and saved artifacts.
- Improved, reproducible results for conversations, factual questions, grounded search decisions and structured UI selection.
- Benchmarked inference optimization and a declared reliability/efficiency trade-off.
- Explicit recommendation of which trained model becomes the initial default and which remains an alternate.

**Exit gate:** evidence that at least one optimized model provides valuable practical improvement at acceptable local cost. No UI implementation is required to pass this gate.

### Phase 2 — General-purpose conversational harness

Only after Phase 1:
- Independent chat threads and a sidebar-friendly conversation API.
- Common **user-global memory** plus chat-local working context, scoped permissions and source provenance.
- Pluggable model selection and local inference serving.
- Search orchestration, evidence ingestion, safe action validation, citation verification.
- Hierarchical history storage, context rehydration/compaction, recovery and handoff.
- Calibrated drunk/context-health meter; export, delete, correction and backup.
- Structured generative-UI message protocol and safe rendering contracts.
- Streaming events, tests and observable latency/failure telemetry that stays local by default.

The harness is a chat platform, **not** a general coding/OS execution agent.

### Phase 3 — Web application

- ChatGPT-style multiple-chat sidebar and search.
- Model picker for **our optimized SmolLM / Qwen variants**.
- Streaming text and useful native interactive elements inside messages.
- Shared-memory settings, source citations, context-health meter and transparent new-chat handoffs.
- Clean, fast, responsive UI built against the validated harness. A strong design system should be established then.

### Phase 4 — Local desktop (Windows + macOS), then mobile

Package the validated backend and user experience for desktop, with platform-specific model acceleration and hardware detection. Mobile follows only after runtime, storage, thermal and memory feasibility are measured. Local inference is the preference; don't assume identical model sizes will run well everywhere.

## 6. Out of scope and explicit constraints

- No single lifelong-chat-only architecture. **Multiple chats + common memory is fundamental.**
- No automatic coding agent, IDE integration, shell execution, screen control or heavy agent workflows as the core product.
- No claim of infinite native context, perfect hallucination detection, or Google's full proprietary Search/AI Overview performance.
- No requirement to depend on Tavily, a hosted LLM API or a proprietary cloud provider.
- No training from scratch of a large foundational model as the first milestone.
- No premature web UI or dashboard work before the model evidence exists.
- No automatic uploading of private conversations or training on user data without explicit consent.
- No silent manipulation of memories: corrections, provenance and erasure must be understandable.

## 7. Reset and repository handling

**October 31, 2026:** The earlier TRIHAERA experiment, single-chat LRA prototype, and exploratory search-native implementation have been retired from the **active** `main` working tree. Their code, documents and benchmark history remain accessible through Git commits; this is a new research direction, not an endorsement of prior results.

The current repo name `Amaan27-Buzz/trihaera` is a **legacy URL**. Naming the new project, renaming the repository and selecting the final brand can happen separately. Avoid assuming any old acronym defines the new product.

**This README is the only tracked file at the reset checkpoint.** There is intentionally no runnable code, no CI workflow, no dependencies and no project scaffolding. Future work should first establish the training/evaluation baseline, using a feature plan derived from this README.

Locally ignored environments, model caches, databases and experiment outputs are **not** automatically deleted by a Git pull. They may contain useful data and should only be removed through a separate, explicit cleanup decision.

## 8. References

- [SmolLM2-360M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
- [Qwen3.5-0.8B official model card](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [OpenAI — GPT-6 and Intelligent UI for everyone (October 2026)](https://openai.com/index/gpt-6-for-everyone/)
- [Search-R1: search-intensive reasoning with reinforcement learning](https://arxiv.org/abs/2503.09516)
- [Search-o1: agentic search-enhanced reasoning](https://aclanthology.org/2025.emnlp-main.276/)

---

**Build order: Model → Harness → Web → Desktop → Mobile.**

**Success criterion:** a genuinely useful, efficient, retrained conversational model and a trustworthy product built around it—not a benchmark score, a search wrapper, or a polished UI alone.
