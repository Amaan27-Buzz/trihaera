# LRA — One continuous AI companion

**One conversation. One companion. No separate chat threads.**

LRA is a local-first experiment in a personal AI companion that can continue
the **same chronological conversation** over months or years. The whole history
lives on your machine. A small local language model receives a bounded,
contextually selected view of the ongoing relationship.

The previous TRIHAERA quantum-inspired architecture has been cancelled. This
repository was **cleaned and repurposed** on `main` for LRA; previous source
and experimental results remain recoverable in Git history, but are no longer
part of the active working tree.

## Product principle

**The user should just open LRA and continue talking.** They should not have
to choose a chat, start a conversation, manually search an archive, or explain
who they are again at every launch.

The single conversation is the product. Retrieval, memory extraction and
context packing are invisible internal mechanisms, not separate chat features.

### Present first prototype

- One local SQLite conversation journal; no conversation/thread IDs.
- Full original messages with IDs and timestamps, surviving restart.
- Lexical full-text recall of older messages plus recent-turn context.
- Automatically extracted **explicit** basic facts and preferences with
  source-message links; updated notes supersede earlier versions.
- Optional local semantic recall via MiniLM embeddings.
- Token-budgeted prompt assembly for an instruction-tuned SmolLM.
- A terminal companion chat. `/search`, `/history`, `/memories` and
  `/correct` are debugging/inspection controls, **not normal chat workflow**.
- A GPU smoke that verifies original-message retrieval separately from
  answer accuracy, and CPU regression tests for continuity and corrections.

**Prototype limitations:** This is not autonomous lifelong understanding.
There is no natural-language conflict resolution beyond simple explicit
patterns; SmolLM may fail to use retrieved evidence and may hallucinate.
No GUI, voice, multimodal memory, model fine-tuning, automatic conversation
summaries, ANN vector index, or proactive interaction yet. The native
model context is finite even though the archive can grow on disk.

## First RTX 5060 GPU validation

Start with
[SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
(approximately 360 million parameters, native 8,192-token context, Apache-2.0).
This small instruct checkpoint is inexpensive for validating the architecture,
not a claim of state-of-the-art conversational quality.

Your existing Windows `.venv` already contains CUDA PyTorch and Transformers;
do **not** reinstall the large PyTorch package.

```powershell
cd 'D:\Programming\03_Projects\personal-projects\05_ML\trihaera'
git pull --ff-only origin main
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest -q
python -m pip show torch transformers safetensors
python scripts/gpu_smoke.py --device cuda --output artifacts/lra_gpu_smoke.json
```

The initial GPU test downloads the 360M model/tokenizer once from Hugging
Face. It simulates a long single conversation, queries an old personal detail,
and reports both retrieval and generated-answer success. If connectivity
intermittently fails, retry the download; use `--offline` on subsequent runs.

## Run the companion

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m lra --device cuda --db data/lra.sqlite3 chat
```

Close and restart it with the **same database path**: it continues the one
conversation rather than creating a new thread.

For a one-off terminal interaction that still joins the SAME conversation:

```powershell
python -m lra --device cuda --db data/lra.sqlite3 ask 'What did we discuss yesterday?'
```

Optional inspection commands in the terminal chat:

```text
/history 20
/search your previous project
/memories
/remember preferred_reply_length=short
/correct preferred_reply_length=detailed
/stats
/quit
```

### Optional semantic memory

Plain SQLite FTS5 works without any extra dependency. For local semantic
embeddings, in the LRA environment only:

```powershell
python -m pip install 'sentence-transformers>=3,<6'
python -m lra --db data/lra.sqlite3 --semantic index
python -m lra --db data/lra.sqlite3 --device cuda --semantic chat
```

Semantic vectors are a rebuildable index over canonical messages, not a
replacement for them. This prototype scans them in Python, so it has not
been stress-tested for millions of messages.

### Privacy and limitations

- Local conversation data: `data/lra.sqlite3`; ignored by Git.
- LRA does not send conversation text to an API. Model downloads access
  Hugging Face; after download, `--offline` disables model Hub lookups.
- A local database is not encrypted by this prototype. Disk-level encryption
  and backups remain your responsibility. Anyone with filesystem access can
  read it.
- Manually correcting a memory does not rewrite original messages.
  A future version needs granular deletion/export and transparent retention
  settings; do not use v0.1 for sensitive data you cannot safeguard.
- LRA does not run in the background and will not contact you proactively.

## Multi-prompt companion evaluation

The single-prompt GPU smoke is only a startup check. To evaluate whether
LRA actually behaves like **one continuing companion**, use the scripted,
isolated multi-prompt evaluation.

```powershell
cd 'D:\Programming\03_Projects\personal-projects\05_ML\trihaera'
git pull --ff-only origin main
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest -q

# First: 6 scored questions plus setup turns; model already downloaded.
python scripts/evaluate_companion.py --device cuda --offline --quick --output artifacts/lra_eval_quick.json

# Then: all 12 scored questions, 100 pairs of synthetic filler turns,
# one memory correction and a database restart.
python scripts/evaluate_companion.py --device cuda --offline --output artifacts/lra_eval_full.json
```

The benchmark **never edits your actual `data/lra.sqlite3` conversation**:
by default it constructs and destroys an independent temporary SQLite
journal. Pass `--db artifacts/lra_eval_inspect.sqlite3` if you want to
inspect its full synthetic conversation afterward; the script refuses to
overwrite a preexisting database.

It runs multiple real SmolLM generations across one growing chat, including:
- remembering an introduced name, project milestone and stated style;
- an older specific launch reference and reason a project was paused;
- a direct correction of the user's name, including after database restart;
- a multi-fact answer, an unfinished onboarding task, and questions for
  which no factual answer was given;
- 100 pairs of deliberately synthetic intervening user/assistant turns to
  exceed the recent-message window without generating hundreds of tokens.

`artifacts/lra_eval_full.json` preserves each actual prompt, generated
answer, expected source IDs, source availability in the prompt, literal
answer verdict, latency and context usage, plus aggregate numbers.
Unknown-fact probes require **manual inspection**; literal matching can
mis-score paraphrases. The test does not train or fine-tune SmolLM and
must not be reported as comprehensive companion intelligence.

A failure should be classified as: **source not retrieved**, **source in
context but model ignored it**, **stale/corrected fact**, or **unclear
answer**. Review actual model outputs before changing the architecture.

## 2026-10-30: SmolLM response collapse — diagnostic checkpoint

**Observed on user's RTX 5060:**
- Normal chat handled greetings but refused the simple instruction `say under` three times, giving an irrelevant generic assistant statement.
- Multi-prompt quick evaluation: **0/5** literal answer checks passed.
- Full scripted evaluation: **0/10** literal answer checks passed.
- Every expected source was available in the context builder (reported 100% source coverage); yet each response repeated `Noted. We can continue our ongoing conversation.`.
- Cause is **not proven**. However, the old benchmark inserted that exact fake assistant reply 100 times in synthetic history, a strong generation-contamination confound. The prompt also described the companion as a local archive.

**Changes:**
1. Replace archive-centric prompt with concise companion-first instructions emphasizing the latest actual user turn.
2. Deduplicate exact repeated assistant messages **only in the prompt view**, never in the stored canonical history. Likewise deduplicate exact repeated retrieved evidence.
3. Replace the 100 identical benchmark acknowledgements with distinct synthetic conversation exchanges covering multiple mundane topics.
4. Add a five-case diagnostic that distinguishes **bare-model instruction following**, simple system prompt, supplied-context recall, repeated-response stress, and LRA old-fact recall. Do not assume the changes succeeded until the actual GPU result is measured.
5. Regression tests verify the user request remains the final turn, canonical messages are intact, and boilerplate does not occupy every evidence slot.

```powershell
git pull --ff-only origin main
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest -q

# First isolate model capability versus prompt/retrieval issues (5 generations).
python scripts/diagnose_generation.py --device cuda --offline --output artifacts/lra_prompt_diagnostics.json

# Then rerun the same six-case pilot with revised synthetic dialogue.
python scripts/evaluate_companion.py --device cuda --offline --quick --output artifacts/lra_eval_quick_v2.json

# Only after reviewing the quick answers, run the full 12-prompt suite.
python scripts/evaluate_companion.py --device cuda --offline --output artifacts/lra_eval_full_v2.json
```

If the **bare** `say under` test fails, prompting and memory alone cannot
explain the problem; inspect the model checkpoint, tokenizer/generation route,
and compare a stronger local instruct model (e.g. SmolLM2-1.7B-Instruct)
before implementing more memory complexity. If bare works but LRA fails,
inspect the rendered context and retrieval placement.

The revised benchmark is still synthetic; it has not been GPU-validated by
the assistant. The user must supply actual outputs for model-quality claims.

## Latest owner GPU results and grounding work (2026-10-30)

The revised scripted companion benchmark improved from 0/10 to **8/10
literal recall checks** with 10/10 source presence; this did **not** mean
the companion was accurate. Two follow-ups used the wrong project reference
and the model hallucinated a private phone number and an exact presentation
date that were never supplied. It also copied the synthetic prefix
`Regarding entry 099` into every answer.

A new diagnostic checkpoint isolates source authority, topical continuity
and exact unsupported personal details. The model's **raw** output remains
visible for evaluation; a bounded personal-fact safeguard may replace an
unsupported literal identifier with an uncertainty response in the served
conversation. This is a narrow safeguard, not broad factuality verification.

```powershell
git pull --ff-only origin main
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest -q
python scripts/evaluate_companion.py --device cuda --offline --quick --output artifacts/lra_eval_quick_v3.json
python scripts/evaluate_companion.py --device cuda --offline --output artifacts/lra_eval_full_v3.json
```

See [the measured results and limitations](docs/EVALUATION_FINDINGS.md).

## Capacity checkpoint: fixed-prompt SmolLM2 comparison

**2026-10-30 owner v3 GPU results:** full synthetic benchmark measured
**7/10** literal answers correct (down from v2's 8/10), with **10/10**
expected sources present. Incorrect answers misidentified the user's
corrected name as LRA and reduced an unresolved keyboard-focus bug to
generic onboarding advice. The small model also invented a presentation
date in words, which bypassed the prototype's numeric-only date guard.
The copied `Regarding entry 099` prefix disappeared. These findings
do **not** establish that adding more retrieval heuristics will help.

We now freeze the LRA context builder and compare **two SmolLM2 Instruct
models** using identical textual snapshots from one scripted continuing
conversation (including 100 filler pairs, correction, and restart):

```powershell
git pull --ff-only origin main
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest -q

# Initial comparison: 6 probes on both 360M and 1.7B.
# Do NOT add --offline on first use: the 1.7B weights must download.
python scripts/compare_smollm.py --device cuda --quick --output artifacts/lra_smol_quick_comparison.json

# Only after resource compatibility: 12 probes, identical prompt content.
python scripts/compare_smollm.py --device cuda --output artifacts/lra_smol_full_comparison.json
```

The larger checkpoint is `HuggingFaceTB/SmolLM2-1.7B-Instruct`. Its
BF16 parameter weights are about 3.4 GB and its configured context length
is 8,192 tokens. On an 8GB RTX 5060 this is a reasonable inference trial,
but memory usage must be measured. Model downloads only; **no training**.
You can add `--offline` once both checkpoints are cached.

The comparison runner constructs the same chronological **synthetic**
user/assistant history for both models. It captures each prompt **before
generating** with either model, so unlike a normal conversation, one
model's replies cannot change the other's evidence. Each result records
literal correctness, source presence, latency, VRAM, and *separate*
flags for fabricated-looking dates or contact details. The latter are
format checks requiring human review, not a comprehensive fact checker.

This is a **model-capacity control**, not the final interactive companion
test. If 1.7B does better, rerun the actual `evaluate_companion.py`
end-to-end benchmark with `--model HuggingFaceTB/SmolLM2-1.7B-Instruct`
to verify it carries over when past assistant replies are model-generated.

## Engineering and research

See [COMPANION_SPEC.md](docs/COMPANION_SPEC.md) for the product definition,
boundaries and next milestones. The first question is whether a 360M model
with retrieval and persistent personal context behaves like one continuous
conversation—not whether it can store unlimited text.

**Repository path still says `trihaera` for compatibility with the existing
checkout.** The active Python package is `lra` and this is a new project.
