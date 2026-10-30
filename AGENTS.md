# LRA development instructions

- LRA is **one continuous AI companion**, not multi-chat software, not merely
  an archival search utility. Prioritize natural conversation continuity.
- Single chronological persistent conversation with clear canonical history.
- Local-first, private-by-default; no remote inference and no telemetry.
- Distinguish retrieved evidence from generated claims. No false memory.
- Avoid silent context truncation and avoid modifying archived messages.
- Preserve corrections and source provenance; implement deletion controls
  before sensitive production use.
- Keep the first baseline GPU-friendly: SmolLM2-360M-Instruct on 8GB VRAM.
- CPU tests must pass without downloading model weights or installing Torch.
- Add tests and measurable recall / quality checks with each change.
- **Commit directly to main only**. Do not open branches or PRs.
- Former TRIHAERA code remains accessible through Git history; do not
  reintroduce the cancelled research direction without an explicit request.
