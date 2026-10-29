# TRIHAERA — Stage 1 research decision record

**2026-10-30. Owner rule: direct updates on main only.**

| ID | Decision | Rationale | Revision trigger |
|---|---|---|---|
| T-001 | Treat −1,0,+1 as classically simulated complex channels, never physical qutrits | Running on RTX 5060 / CUDA with nonunitary FFN and softmax | Actual quantum hardware and appropriate quantum circuit |
| T-002 | Keep the same shared attention/FFN projection weights across 3 channels | Isolates effect of phase and mixing; comparable *stored parameters* | Evidence that state-specific modules are required and justify overhead |
| T-003 | Do not equate three channels with 36 serial Transformer layers | No additional serial depth; state mixing isn't independent blocks | Never, unless precisely redefined with valid empirical computation metric |
| T-004 | Initial phase copies are rank one; U alone cannot diversify | Linear algebra and diagnostic tests | Changed state initialization or state-aware operators, with proofs |
| T-005 | Report rank/energy/interference only as structural diagnostics | Random nonlinearities also increase rank | Use trained model + task-specific causal ablations for usefulness |
| T-006 | Include real-equivalent, complex and recurrent baseline controls | Must distinguish architectural gain from FLOPs/width | Stronger baseline research |
| T-007 | Require equal-FLOPs as well as equal-parameter comparisons | Extra complex state computation may dominate | Explicitly different objective such as parameter-only storage |
| T-008 | 500–700M base and instruction-tune same checkpoint remain conditional | Architecture and training cost not established | After controlled Stage 3 gate and researched training plan |
| T-009 | Keep Stage 1 IN PROGRESS | Missing GPU probe, rigorous novelty audit, cost estimates, go/no-go threshold | Exit checklist completed and reviewed |

Conclusions must be revisable based on results, including negative ones. No fabricated benchmark claims.

| T-010 | Adapt SmolLM2-135M Base first (Apache-2.0 pretrained weights) | Small, architecturally compatible, less compute and ~270MB download | Fails local compatibility/VRAM or better fully reproducible target |
| T-011 | Start with a zero-initialized low-rank residual rather than destructive weight remapping | Preserve pretrained logits exactly; isolate intervention | Stable state-aware full-weight conversion demonstrates superior transfer |
| T-012 | Treat three-state residual adapters as an intermediate bridge, not full TRIHAERA | States are formed/recombined inside each layer, not persistent end to end | Implement a verified persistent-state decoder, with compatible checkpoints |
| T-013 | Always retain base/no-mixer/real-adapter controls | Pretrained knowledge or additional adapter FLOPs alone can explain gains | Stronger controls supersede these |
