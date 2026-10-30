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

## Engineering and research

See [COMPANION_SPEC.md](docs/COMPANION_SPEC.md) for the product definition,
boundaries and next milestones. The first question is whether a 360M model
with retrieval and persistent personal context behaves like one continuous
conversation—not whether it can store unlimited text.

**Repository path still says `trihaera` for compatibility with the existing
checkout.** The active Python package is `lra` and this is a new project.
