# LRA — Companion product specification

## North star

One lifelong conversation with one personal AI companion. No multiple
chat rooms, tabs or automatic session resets. The application remembers
what was said, how preferences evolved, and what unfinished topics should
be resumed, while respecting user correction and control.

## Desired experience

Open the application tomorrow and type "what were we saying?".
The companion should use recent continuity and the actual earlier discussion,
not invent an answer. Mention an old plan months later; it should retrieve the
relevant original discussion. Correct a fact and the active companion view
should prefer the new fact, while allowing the user to inspect its history.

Ordinary dialogue is the main interface. Search, citations and memory controls
are available when helpful, but the user should not need to operate a
knowledge-base product to have a conversation.

## Architecture layers

1. **Canonical lifetime timeline**: one append-only chronological stream in
   SQLite. No Chat objects or per-conversation IDs. Original messages remain
   directly inspectable and searchable.
2. **Continuity state**: most recent messages, open threads and stable
   companion instructions; v0.1 has recency only, open-thread extraction
   is a future milestone.
3. **Long-term explicit memories**: source-linked stable facts and corrections,
   with confidence/provenance planned. v0.1 uses conservative simple patterns;
   no inference should be passed off as a fact.
4. **Episodic recall**: hybrid lexical and optionally semantic search over
   exact past messages. Re-ranking, conversation episode extraction and
   temporal reasoning are future work.
5. **Budget manager**: fits user turn, recent dialogue, selected memories
   and evidence under a measured native model-token limit. Storage is
   unbounded by context but finite in disk space.
6. **Local LLM**: SmolLM2-360M-Instruct in BF16 on RTX 5060 for first
   validation, no hosted inference. Stronger models may follow if necessary.
7. **Companion interaction**: v0.1 terminal, later a single-thread desktop
   application showing one scrollable history with natural resume/recall.

## Explicit non-goals for first milestone

- "Unlimited context" inside the model (not physically possible here).
- Full model-weight training or continuous LoRA tuning.
- Artificial consciousness, fake sentience, or manufactured attachment.
- Background activity, unsolicited monitoring or surveillance.
- Multi-chat organization, workspaces, bots or agents as the main surface.
- Assertions that facts were retained if they weren't actually retrieved.

## Quality gates

- Restart continuity: same database, old turns intact.
- Time-distance recall: old user fact recoverable after 100+ intervening turns.
- Explicit corrections: most recent fact wins, old original is auditable.
- Search correctness separated from generation correctness.
- Measured context selection below tokenizer context; never silently truncate
  current user prompt.
- GPU inference within the 8GB VRAM ceiling and recorded latency.
- Then: conversational flow, preference retention, conflict management,
  temporal recall and longitudinal tests with real human-reviewed cases.

## Current constraints

The initial version does not robustly understand all personal references;
a 360M model is intentionally a lightweight baseline. The next milestone
must add an actual continuity model beyond a few regexes: episode summaries,
important moments, temporal facts, open threads, explicit forgotten-fact
handling and adversarial evaluations. User correction and deletion controls
must be expanded before sensitive personal deployment.

## Project handling

The old TRIHAERA code is removed from the active branch tree and retained
in Git history for reproducibility. All active LRA code is on `main`.
Use existing independent Windows CUDA environment without modifying Levis.

## Evaluation milestone — multi-turn, single-companion tests

The first companion quality check is the reproducible
`scripts/evaluate_companion.py` harness. It generates actual SmolLM answers
across a continuing SQLite history, includes more than 200 synthetic
distractor turns and verifies the same history survives a restart.

**Critically, evaluation separates three quantities:**
1. Whether the relevant original messages or corrected active memories were
   accessible in the *actual* bounded model prompt.
2. Whether the model's output contains each predeclared expected fact; checks
   are literal and may miss valid paraphrases.
3. Whether the output is trustworthy, appropriately uncertain, natural and
   companion-like; subjective and unknown-fact cases must be reviewed by a
   human, not counted as automatically correct.

A scripted history with known answers is a debugging baseline, not a proof
of autonomous long-term memory. Future milestones require natural dialogue,
temporal and emotional context, conflicting beliefs, open-thread continuity,
robust corrections, opt-in memory deletion and non-synthetic human trials.

The six-question quick path is `--quick`; use the default 12 scored probes
for the complete diagnostic. Both generate additional setup replies through
the real SmolLM model. Benchmark data is **strictly separate from the one
personal LRA conversation**, and stored report content is synthetic.
