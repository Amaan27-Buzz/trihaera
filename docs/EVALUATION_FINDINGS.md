# LRA — 2026-10-30 companion GPU evaluation review

## Measured owner results — SmolLM2-360M-Instruct, RTX 5060 8GB

Owner confirmed commit `2d8cf6b`, 17 CPU tests passed, no user data shared with the synthetic benchmark.

| Experimental stage | Literal correct | Expected source presence | Caveat |
| --- | ---: | ---: | --- |
| Original quick benchmark | 0/5 | 5/5 | 100 identical synthetic assistant filler responses |
| Original full benchmark | 0/10 | 10/10 | all answers repeated identical filler phrase |
| Revised diverse quick benchmark | **5/5** | 5/5 | raw private contact fabricated |
| Revised diverse full benchmark | **8/10** | 10/10 | two wrong-topic answers, two unsupported private claims |
| Generation isolation diagnostic | **2/5** | N/A | bare `under` and short in-context code passed; system-augmented instruction and contaminated dialogue failed |

The two manual-review unknown-fact probes were **not successes**:
- For a first public presentation date absent from the conversation, the model invented `2026-10-30`.
- For an undisclosed work phone, the model invented `123-456-7890`.
- The quick test likewise invented `morgan@example.com` as a phone-number answer.

The full benchmark's final two scored failures were:
- Asked what needed fixing in onboarding: model answered unrelated launch reference `cedar bridge 719`.
- Asked user's current name after correction: model again answered the launch reference instead of the updated name.

All revised outputs still began with `Regarding entry 099` (a synthetic assistant filler label), showing that context contamination persisted even though exact-answer performance improved.

**Interpretation:** `8/10` is a literal recall metric, **not 80% companion quality**, not calibrated hallucination rate, not generalized to unconstrained user histories. `10/10` expected source presence means the relevant source was in the prompt or active memory, not that the model attended to it or that the answer was grounded.

## Next code changes under evaluation (GPU validation pending)

1. **Source authority:** for ordinary personal recall, retrieve original **user** messages rather than using older model-generated assistant replies as factual evidence. Preserve canonical storage and optionally support assistant-history recall for questions explicitly about what LRA said.
2. **Topic focus:** for identifiable past-fact questions, restrict recent-turn injections to topically relevant user text. Ordinary non-recall conversation keeps the regular recent dialogue. These are heuristics, not a full episode or temporal-reasoning mechanism.
3. **Unsupported personal-detail guard:** when SmolLM generates a literal phone/email/exact date absent from selected original user messages in a directly personal query, serve an uncertainty response instead. Record the raw generation and override separately. **This does not verify all facts, paraphrased dates or arbitrary identity claims, and does not fix model reasoning.**
4. **Assessment integrity:** each benchmark record reports raw model answer, served answer, grounding override and source coverage. Do not count the guard's abstention as proof the base model learned to avoid hallucinating.

## Follow-up test protocol

```powershell
git pull --ff-only origin main
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest -q
python scripts/evaluate_companion.py --device cuda --offline --quick --output artifacts/lra_eval_quick_v3.json
python scripts/evaluate_companion.py --device cuda --offline --output artifacts/lra_eval_full_v3.json
```

Compare individual answers, exact source IDs and raw/served outputs to the prior v2 reports. Passing CPU tests is not a GPU language-quality result.

If the same two wrong-topic failures persist, compare SmolLM2-1.7B-Instruct on the same local GPU (model option supported by scripts), then consider more substantive memory extraction/re-ranking. Do not invent training or fine-tuning benefit in advance. LRA remains **one continuous companion chat**, not multiple sessions.
