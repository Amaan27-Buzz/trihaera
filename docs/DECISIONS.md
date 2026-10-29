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

| T-014 | Stabilize adapters using residual scale=0.1, AdamW lr=3e-4, clip=1.0 | First 23-token CUDA run changed logits RMS=9.25787 after only two steps | Reassess after held-out drift/loss and learning curves |
| T-015 | Use WikiText-2 raw train/validation for first small-data pilot | Lightweight real language corpus with explicit splits; not a training-recipe claim | Better curated, license-compatible corpus and multi-seed tests |
| T-016 | Record unused mixer parameters as frozen in no-mixing control | Inactive weights are not trainable capacity | Alternative control implementation |

| T-017 | Do not credit the current 3×3 pretrained adapter mixer as novel persistent-state computation | Mixer followed immediately by channel sum is exactly equivalent to a constrained 3-complex-coefficient readout | Move mixing before further cross-state/nonlinear processing and test against real-equivalent controls |
| T-018 | Run three paired seeds and 64 validation windows before further architectural investment | First one-seed 20-step pilot had no-mixing ahead by 0.00210 NLL; time 3.07s vs 4.83s mixed | Replace with full compute-/data-matched Stage 3 comparison after stability |
| T-019 | Keep 135M derived model experiment separate from future 500–700M from-scratch foundation | Pretraining weights from SmolLM2 provide prior knowledge; adapter evidence cannot prove TRIHAERA general architecture | Independent later pretraining after Stage 3 go/no-go |

| T-020 | Do not scale local three-state adapter on the three-seed NLL result | Mixer improvement mean −0.007093 NLL, dominated by seed 17; ~65.5% more training time | Compute-matched replicated advantage |
| T-021 | Experiment with depth-persistent state over frozen pretrained layer features | Avoids immediate 3×3 mixer/readout reduction; enables nonlinearity after mixing | Fails GPU compatibility or rigorous usefulness controls |
| T-022 | Separate sidecar architecture from full TRIHAERA | Sidecar reads pretrained hidden states but does not modify attention or propagate them into backbone layers | Explicit fully integrated state-carrying decoder built and validated |

| T-023 | Accept CUDA correctness of 30-depth persistent sidecar but not claim quality | Mixed and unmixed were exactly baseline-preserving; ~64% slower two-step mix with no held-out comparison | Rigorous held-out / throughput results |
| T-024 | Add six-real-channel depth-recurrent controls, including an unconstrained real orthogonal mixer | Controls for 3 complex channels having 6 real scalar components and stronger mixing; classical simulation provides no free quantum advantage | Measured compute/parameter matching and better real baselines |
| T-025 | Replicate persistent sidecar and real controls using identical token exposures and 3 seeds | Prior local mixer advantage noisy and training time 65% greater; new mechanism avoids immediate readout collapse | Statistically powered and FLOP-matched follow-up |
