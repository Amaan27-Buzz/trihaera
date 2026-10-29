# Architecture hypothesis v0.1 (UNVALIDATED)

## Representation

Three state labels are −1, 0, +1. For a real token embedding X in R^(B×T×D), construct a classical complex tensor Z in C^(B×T×3×D):

Z_s = X · exp(i φ_s) / sqrt(3)

At initialization the state-axis amplitude magnitudes have squared sum 1 per scalar coordinate. That does NOT mean the subsequent nonlinear Transformer represents a normalized quantum wavefunction.

## Proposed block

- Apply shared trainable real-valued QKV and output maps to complex tensors by applying the same matrix to both real and imaginary components.
- For each state compute Q_s K_s^H, retain its real part, scale and apply causal softmax. Attention acts on complex V_s with real attention weights.
- Apply shared complex RMS normalization based on |Z|² and a component-wise SiLU FFN. All of these are classical and generally nonunitary.
- Use learnable per-state phase factors on the attention residual.
- Mix the three states after each block with U=exp(iG), for Hermitian 3×3 generator G. This mixer is unitary but the full Transformer is NOT a quantum circuit.
- Combine state amplitudes before the token-logit readout. Classical destructive/constructive phase effects can occur.

## Intended test

Does the phase-interference construction offer meaningful improvements versus a conventional real decoder, a matched shared-weight/recurrent decoder and a non-interfering three-channel model under equal token, parameter and compute budgets?

The three-state representation can always be encoded into a larger real-valued state tensor; therefore quantum novelty/acceleration cannot be inferred. Its extra activation storage, higher FLOPs and potential loss of optimized attention kernels are genuine liabilities.

## Known limitations

No RoPE, KV caching, streaming generation, native optimized kernels, variable-state control, long-context mechanisms, factuality guarantees, or demonstrated 36-layer equivalence yet.

The 500–700M target is a **future foundation pretraining size** and not the size of this initial correctness prototype.

Do not treat operator novelty, performance superiority or lower hallucinations as proven.
