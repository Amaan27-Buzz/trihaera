# TRIHAERA architecture research: mathematical specification v0.2

**Status:** Stage 1 working hypothesis; reference implementation only. **Date:** 2026-10-30.  
**Claim:** none of the proposed capabilities, efficiency gains or novelty are established.

## 1. What the model actually is

TRIHAERA v0 is a **classically simulated complex-valued causal decoder** with exactly three simultaneous state channels indexed by `s ∈ {−1, 0, +1}`. It uses normal PyTorch GPU operations. The phase coordinates are *not* physical qutrits, quantum entanglement or an actual quantum circuit. The model does not implement a quantum measurement.

Let `B, T, D, H, K` denote batch, time, hidden width, heads and number of layers. Input token IDs have shape `[B,T]`. Learned token and absolute positional embeddings:

```text
e[b,t,:] = token_embedding[x[b,t]] + position_embedding[t]
z[0,b,t,s,:] = e[b,t,:] * exp(i * input_phase[s]) / sqrt(3)
shape(z) = [B,T,3,D] complex64
```

This is a 3-channel **rank-one** state construction at input, not three independent neural representations.

## 2. The implemented block, precisely

For each block `l = 0, …, K−1`:

1. Complex RMS normalization, with a shared learned real gain vector `g`:

```text
rms(z)[...,d] = g[d] * z[...,d] /
                sqrt((sum_j |z[...,j]|^2 / D) + epsilon)
```

2. A real-valued, bias-free weight matrix acts identically on the real and imaginary components of every state. Hence `W(a+ib)=W(a)+i W(b)`, with a shared W for all three states.
3. Split `q_s,k_s,v_s` into heads; per-state causal attention uses a **real** score:

```text
score_s[b,h,t,u] = Re(sum_j q_s[b,h,t,j] * conj(k_s[b,h,u,j]))
                   / sqrt(D/H)
A_s[b,h,t,:] = softmax(score_s[b,h,t,:] masked to u <= t)
att_s = W_out(sum_{u<=t} A_s[b,h,t,u] * v_s[b,h,u,:])
```

No cross-state attention occurs here; the channels interact only via the mixer. The doubled-real concatenation in code computes the stated real inner product exactly (subject to floating-point error).

4. Apply a **different learned phase** to each state's attention residual:

```text
z'[:, :, s, :] = z[:, :, s, :] + exp(i * theta[l,s]) * att_s(rms(z_s))
```

5. Feed-forward network with common real matrices; separate real/imag SiLU activations:

```text
ff(z') = W_down(SiLU(Re(W_up*rms(z'))) + i*SiLU(Im(W_up*rms(z'))))
z'' = z' + ff(z')
```

This FFN is **not complex-linear or globally phase-equivariant**. The overall block is nonlinear and nonunitary.

6. State mixing uses a learned Hermitian generator:

```text
G = (A + A^T)/2 + i*(B - B^T)/2   [3x3 Hermitian]
U = matrix_exp(i*G)                  [3x3 unitary]
z_next[b,t,s,d] = sum_r U[s,r]*z''[b,t,r,d]
```

The 3x3 `U` is unitary, but the rest of the model is **not**. State-vector energy conservation through `U` alone does not imply wavefunction normalization elsewhere.

7. After the final block:

```text
h[b,t,d] = Re(sum_s z[K,b,t,s,d]) / sqrt(3)
logits = tied_embedding_head(RMSNorm(h))
```

We discard the imaginary component at readout. Output token probabilities arise from ordinary softmax over logits, not a quantum measurement.

## 3. Proved structural limitations of this version

**A. Three states are not 36 sequential layers.** There are K consecutive blocks, each processing three channels; no additional sequential depth is created by the state dimension. Thus 12 physical blocks do not equal 36 independently learned/serial blocks.

**B. Rank-one initial states.** Let `c_s=exp(i*phi_s)/sqrt(3)`, flatten all token and feature coordinates into a vector `e`, then `Z0 = c e^T`. Therefore `rank(Z0)<=1`. If a unitary `U` multiplies the state axis alone, `rank(U Z0)=rank(Z0)` because U is invertible. Mixing alone cannot create state diversity.

Nonlinear per-state processing and residual phases **can** diversify channels; whether they do so in a useful, stable way is untested. We now measure effective rank, pairwise overlap, per-state energy, and coherent-sum fraction after every block.

**C. Classical real representation exists.** Three complex D-dimensional channels can be rewritten as six real D-dimensional channels with constrained/shared weights and standard real operations. Consequently, classical simulation offers **no inherent quantum parallel-computation speedup**.

**D. Phase symmetry in this attention.** Multiplying every token/feature of one channel by a unit-magnitude phase `exp(i*alpha_s)` leaves the real QK similarity matrix unchanged and rotates the attention output by the same phase. Therefore the existing attention primitive cannot directly exploit a globally uniform phase shift *within* a channel. Phase-dependent residuals, non-equivariant FFN, mixer, and readout break this symmetry elsewhere. A test encodes the covariance property.

**E. Unit mixing changes orientation, not rank.** The mixer preserves the per-token squared norm over channels and the spectrum of a channel Gram matrix *if no nonlinear computation intervenes*. Meaningful differentiation must be evaluated **after** processing.

## 4. Metrics (diagnostic, not language quality)

For state matrix `Z` shaped `[3, B*T*D]`:

- Gram `C=Z Z^H`. Eigenvalues `lambda_i>=0`. `p_i=lambda_i/sum lambda`.
- State effective rank `r_eff=exp(-sum_i p_i log(p_i))`, from 1 (collinear) to 3 (balanced orthogonal).
- Mean pairwise overlap: mean `|C_ij|/sqrt(C_ii*C_jj)` over i<j (1 indicates collinear).
- Energy fractions: `C_ss/trace(C)`. One-channel dominance is a collapse warning.
- Coherent sum fraction: `||sum_s Z_s||²/(3*sum_s ||Z_s||²)` in [0,1], a descriptive interference statistic. This is not the final real-valued logit readout.

These indicators **cannot** establish useful reasoning, knowledge retention, factuality or competitive training efficiency.

## 5. Compute/parameter accounting and technical debt

- Attention and FFN projections share learned matrices across states; parameter count is near the dense comparator, not triple.
- Each state is complex64 (two float32 components). Activation storage for 3 states can be approximately **6x** a same-width float32 baseline for corresponding hidden tensors, before temporary buffers and kernels.
- Shared-weight operations still execute separately on all three complex channels; forward/backward FLOPs and elapsed time are **not** parameter-count matched. Measure them before claiming efficiency.
- State mixing costs `O(B*T*3²*D)` plus the small 3x3 matrix exponential and its gradient.
- Reference absolute positional embeddings, ordinary softmax, and full-sequence attention have no specialized long-context architecture. RoPE, KV cache, faster kernels, mixed-precision validation and generation remain future work.
- Complex autograd and matrix exponential may have backend-specific performance/stability costs.
- A 500–700M model is **not** this initial reference implementation.

## 6. Alternative hypotheses to test before finalizing architecture

- **H0 (null):** Any improvement is explained by effective width, extra computation, more state activations, or ordinary mixing; no compute-matched benefit exists.
- **H1 (representational):** Different state channels learn complementary *task-relevant* features and remain noncollapsed.
- **H2 (interference):** Learned phase mixing adds benefit over identical architecture with no mixing, frozen mixing, or ordinary real-valued mixing at matched cost.
- **H3 (efficiency):** TRIHAERA reduces validation loss or improves downstream tasks at comparable trainable parameters **and** overall compute (training/inference measured separately).
- **H4 (scaling):** Any observed benefit persists beyond small correctness tests; no unearned promise of 36-layer equivalence.

State count 3 is an **arbitrary design hypothesis**, not theoretically optimal; investigate 1/2/4 controls and equal-activation real-valued baselines in Stage 3.

## 7. Stage 1 exit checklist

- [x] Versioned exact equations for implemented operators
- [x] Mathematical rank and unitary-mixing limitations stated
- [x] Full-classical-simulation / real-equivalence caveat stated
- [x] State-rank and phase-covariance diagnostic tests authored
- [x] Prior-art shortlisting updated through October 2026
- [ ] Complete comparative mathematical analysis against modern complex-valued attention baselines
- [ ] Record NVIDIA GPU output of representation probes and real-world memory/time costs
- [ ] Predeclare sufficiently strong ablation controls and a Stage 3 go/no-go threshold informed by baseline variance

**Conclusion:** Stage 1 remains IN PROGRESS. Current reference code demonstrates a testable mechanism, not an original breakthrough.
