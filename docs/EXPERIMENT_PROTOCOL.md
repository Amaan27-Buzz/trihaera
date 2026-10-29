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
