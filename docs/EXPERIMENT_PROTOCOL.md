# TRIHAERA controlled experiment protocol (pre-registration draft)

## Null hypothesis

There is no reliable quality-per-FLOP or quality-per-latency benefit from learned three-state phase interference beyond ordinary real-/complex-valued models when both parameters and compute are properly controlled.

## Alternatives

- State channels become nonredundant *and task-relevant*.
- Learned 3x3 unitary mixing adds quality beyond no-mixing, frozen-mixing and real-valued mixing controls.
- A generalization improvement persists under equal training FLOPs and similar inference cost.

## Diagnostic plan (not a capability benchmark)

Run `python scripts/probe_states.py --device cuda` after local setup; use same randomly initialized checkpoint for interference on/off. The script reports per-block:
- effective rank [1,3] (input is mathematically 1);
- pairwise overlap [0,1]; 1 means highly redundant;
- state energy fractions (detect one-channel dominance);
- coherent sum fraction [0,1] (not quantum measurement);
- on/off logit RMS difference (effect existence, not improvement).

A high effective rank by itself **does not** imply useful learned specialization. Randomly initialized nonlinearities can create diverse noise.

## Genuine architectural validation: Stage 3

Controls:
1. Modern dense causal Transformer of equal parameter count.
2. Shared-weight/recurrent Transformer with matched depth/computation.
3. Ordinary complex-valued attention architecture without the three-state unitary mixer.
4. TRIHAERA without learned mixing, with fixed mixing, and with random/frozen phases.
5. At least one comparable-width real architecture (six real channels or structured real equivalent) to exclude a complex-valued bookkeeping advantage.
6. State count controls 1/2/4 if supported.

Two separate regimes:
- Equal trainable params and tokens: honestly disclose additional train/inference FLOPs.
- Equal training FLOPs: compare quality and generalization under the same data budget.
Also separately report peak GPU memory, prefill/decode wall time, throughput, and loss versus elapsed compute.

Use held-out language data and multiple seeds. Predeclare a meaningful effect threshold based on baseline variance **before** seeing candidate rankings. Do not confuse tiny repeated-sequence sanity training with meaningful LLM pretraining.

## Run record

Log commit SHA, configs, data source/license/hash, tokenizer, GPU and torch version, precision, seeds, params, peak VRAM, training tokens, estimated/measured FLOPs, losses, downstream evals and negative results.

**Go/no-go:** if any advantage is explained by FLOPs, width or ordinary mixing, revise or stop the architecture; do not upscale because three states sound novel.

## Stage 1/2 bridge: small WikiText-2 adapter pilot

As a preliminary stability and transfer experiment, `scripts/pilot_pretrained.py` compares a completely frozen SmolLM2-135M baseline, a small real-valued residual adapter, a no-mixing three-state adapter, and the mixed three-state adapter. All use the same official WikiText-2 raw train and validation splits, deterministic token packing and training-window order, identical training steps/learning rate/clip norm, and frozen pretrained backbone.

**Not scientifically decisive:** This pilot has only 20 steps by default, and the modes differ in extra compute and slightly in parameters. Dataset content might overlap SmolLM2's *original pretraining data*. Positive results require replication with data provenance controls, multiple seeds, wider baselines and **separate FLOP/latency matching**. Keep failure data and null results.

## 2026-10-30 first pilot measurements (owner-supplied)

Single-seed held-out WikiText-2, 20 steps and 12 validation windows: frozen base 3.45328 NLL, real adapter 3.39929, no-mixing 3.34908, learned mixer 3.35118. Wall times 2.39/3.07/4.83 seconds respectively. **The mixed variant did not win.** These time measurements are not FLOP counts and do not imply rigorous efficiency rankings.

### Replication protocol (new)

`scripts/replicate_pretrained.py` uses the SAME model checkpoint and token packing for all variants. It draws from official WikiText-2 train/validation splits, records source fingerprints/window hashes, trains across default seeds 17, 23, 42, uses 100 steps, and expands validation to 64 windows. It records seed-paired NLL differences against no-mixing, sample standard deviations, train wall times, parameter counts and CUDA peak allocation. It saves output after each run for recovery review.

The variants are still not FLOP matched and have a small parameter-count difference. These runs are **diagnostic replication**, not confirmatory experimental evidence. The collapsed-mixing implementation can optionally be added with `--modes real_adapter,no_mixing,trihaera,collapsed_mixing` to verify the math on pretrained data. Do not interpret a 3-seed descriptive mean as a statistical significance test.

## 2026-10-30 persistent-sidecar comparison protocol (Stage 1)

Model: frozen SmolLM2-135M Base on owner's 8GB CUDA RTX 5060. Training: WikiText-2 official train split, held-out pilot validation from official validation split, identical tokenizer, packed windows, seeds, steps, learning rate and gradient clipping. Controls: complex depth-persistent `mixed`, complex depth-persistent `no_mixing`, 6-real-channel recurrent `real_six` and `real_six_no_mixing`.

For each seed log initial zero-readout parity, trainable params, final validation NLL, step-loss endpoints, gradient norms, measured training time and peak CUDA memory. Summarize paired differences against complex no-mixing; inspect **real-six** comparisons explicitly. This is an *experimental pilot*, NOT equal-FLOP or confidence-calibrated evidence. The O(6) real mixing group is a stronger comparator than U(3) in terms of possible 6-real-dimensional rotations. Follow with timing/FLOP-matched and additional datasets **only if** the persistent design earns further investigation.

## October 30 next falsification: exact-real and matched O(6)

**Existing owner results:** persistent mixed NLL 3.538331 vs no-mixing 3.539520 (3 seeds; 100 steps; 64 validation windows), requiring 26.20 vs 12.07 seconds of training. Prior independent 6-real baselines used differently initialized input states, so they do not isolate complex arithmetic.

New controls:
- `real_equivalent`: exact 6-real representation of mixed complex U(3), same state injection, same weights. At matched initial weights should reproduce **forward logits and gradients** within floating-point tolerance; meaningful quality difference is a bug or numerical-training divergence.
- `real_matched_orthogonal`: same 6-real state initialization and all other operators as complex, but O(6) learns rotations beyond U(3). It starts from the equivalent U(3) transform to isolate learned mixing capacity rather than initialization.
- Include `mixed` and `no_mixing` for continuity with prior runs.

Use identical train and validation packed tokens, 100 steps, seeds 17/23/42 and document per-mode trainable parameters and time. The experiment remains **not FLOP-matched**, even though raw real state width is now comparable. A claim of increased capability requires multi-dataset testing, stronger transformer/recurrent baselines, and matched training compute.
