# Session 10: Falsifier-MoE v1

Date: 2026-08-30

Status: architecture and prompt-held-out training plumbing implemented. CPU
and CUDA BF16/Dion3 smoke pass, and a standalone Raptor Lake AVX-VNNI runtime
ceiling is below 5 ms per four-row round. Real training remains deliberately
blocked on the underfilled on-policy corpus; optimizer ablations, native FP8
training, exact native parity, and engine shadow integration remain open. See
`audits/s10-falsifier-vnni.md` for the runtime and INT8 numerical results.

## Why this exists

The scalar falsifiers found useful speed points, but their quality model is
too weak for recurrent generation. The sharpest counterexample is MathArena
ArXivLean problem 40 at the prefill regret cap 0.0005:

- forced 64-token PPL changed only +3.15%, inside the user's +3.5% budget;
- cosine was 0.965076 and MSE was 0.4668;
- prefill reached 15.61 tokens/s, about 15% above exact layer-major prefill;
- the 320-token free output began coherently, then collapsed into repetitive
  factorization/theorem fragments.

Same-prefix PPL is therefore necessary but not sufficient. The controller
needs the previous target/DFlash logits, current routing and cache state,
causal hidden sketches, and explicit multi-horizon autoregressive failure
heads. `audits/s10-learned-falsifier.md` remains the trace and solver design;
this document freezes the trainable v1.

## Implemented geometry

`tools/falsifier_moe.py` implements a task-specific controller, not a small
language model:

| component | v1 geometry |
|---|---:|
| model width | 192 |
| semantic mHC streams | 4 |
| weight-tied controller passes | 3 |
| MLA heads / head width | 4 / 32 |
| cached MLA latent | 64 |
| routed experts / active | 256 / 2 |
| routed latent / expert hidden | 96 / 128 |
| full-width shared expert hidden | 384 |
| routed-expert inactivity | 99.21875% |
| total parameters | 10,089,763 |
| routed-expert parameters | 9,437,184 |
| active resident parameter estimate | 726,307 (7.20%) |

The 99% claim is deliberately scoped: 254 of 256 routed experts are inactive
per event. It is not a false claim that 99% of the complete network is skipped.
The full-width shared branch, projections, heads, and encoders remain active.
The recurrent cell is invoked three times, so those active weights are visited
three times even though they occupy memory once.

### Four evidence streams

The binary trace already has the correct modalities, so each receives its own
projection and residual stream:

1. previous-target/current-DFlash scalars and three 64-wide vocabulary
   CountSketches;
2. top-32 target-router IDs, logits, corrected choice scores, executed action,
   weights, and row-union multiplicity;
3. causal 64-wide hidden CountSketch plus layer/verify-row position;
4. host-ready, in-flight, device, pinned, overlap, and union-residency state.

Candidate expert IDs use a 16-wide embedding. Each candidate is encoded with
its score, corrected score, four residency bits, and rank, then pooled. All
runtime inputs precede expert execution.

### mHC, MLA, and Attention Residuals

The four streams use a dynamic 4x4 residual mixer. Six Sinkhorn row/column
normalizations project the matrix toward the Birkhoff polytope; it is
identity-biased at initialization. Tests independently stress row/column sums.
This follows the stability motivation of
[mHC](https://arxiv.org/abs/2512.24880), while remaining small enough to fuse.

Causal history uses a 64-wide compressed KV latent. The implementation expands
K/V for training, but its cache contract contains only the latent and the
projection is absorbable for inference. The output has Kimi K3's
input-dependent channel gate. At controller-depth boundaries, block depth
attention selects among the input and prior completed controller passes. This
is the small-controller analogue of
[Attention Residuals](https://arxiv.org/abs/2603.15031); the cell parameters are
shared, so depth does not multiply the model's resident size.

### Stable LatentMoE

The routed branch implements the ordering in
[Kimi K3](https://arxiv.org/html/2607.24653v2):

```text
z = W_down(x)
raw = sigmoid(W_router(x))
route = top2(raw + frozen_bias)
weight = selected_raw / sum(selected_raw)
u = sum(weight * expert(z))
routed = W_up(RMSNorm(u))
output = routed + full_width_shared_expert(x)
```

Experts operate only at width 96. Both routed and shared experts use SiTU-GLU:

```text
[4 tanh(g/4) sigmoid(g)] * [25 tanh(u/25)]
```

so each activation coordinate is smoothly bounded by 100. Quantile Balancing
uses biased Top-(k+1), the raw-score margin against the biased cutoff, the
`1-k/E` quantile, mean-centering, and one-optimizer-step-late commit. The
routing bias is a buffer, not a parameter. There is no contradictory
load-balance auxiliary loss; only a tiny router-logit numerical penalty.

## Heads and labels

Every `(round, target sparse layer, verify row)` event emits:

- immediate log-MSE, cosine damage, log-KL, log-JS, and top-1-flip risk;
- same-prefix failure probability and peak composite damage at 8/16/32 output
  records;
- separate **free-trajectory** failure probabilities at 8/16/32 tokens;
- repetition-collapse and entropy-collapse probabilities;
- a PSD 8x8 contribution Gram estimate (rank four plus nonnegative diagonal);
- risk and cost outputs for prefix actions k=3..8;
- accepted-prefix logits.

The separation between same-prefix and free-trajectory heads is important.
The present dataset can derive same-prefix future targets from forced logits,
but it has no causally aligned free-run trace labels. The free and collapse
heads are therefore present and exactly masked to zero loss. They cannot
silently learn from forced PPL and acquire a misleading name. A v3 corpus must
join free-run feature events to first divergence, repetition, and entropy
collapse before those heads train.

Exact-teacher traces train only the contribution-Gram head. On-policy feature
traces train downstream row and forced-horizon heads. This prevents the
previously measured trajectory mismatch—where 811/1,302 routes differed
between exact and approximate p10 states—from leaking back into training.

## Corpus reality check

The current local corpus was read successfully by the new loader:

| item | available |
|---|---:|
| prompt split units | 7 |
| on-policy policy trajectories | 9 |
| on-policy output rows | 279 |
| on-policy target-layer events | 11,718 |
| exact Gram events | 3,906 |
| top-1 failures | 5 |
| aligned free-trajectory labels | 0 |
| aligned collapse labels | 0 |

With three shared-cell passes and top-2 routing there are only 274.6 routing
opportunities per expert on average, or 91.6 per expert per pass, before load
skew. That is enough to test code and nowhere near enough to train 256 experts.
The default command therefore refuses a normal run below 10,000 on-policy
rows. `--smoke` is the only bypass and is explicitly recorded in the result.

The first honest collection target remains 10k--20k output tokens/rows over
ArXivLean-like formal reasoning, hard code, multilingual prose, long context,
ordinary arithmetic, and adversarial low-margin continuations. At 10k rows,
42 sparse target layers, three controller passes, and top-2 routing, the mean
opportunity count is about 9,844 assignments per expert before imbalance.
Single interventions and on-policy/DAgger waves are still required; merely
adding near-duplicate forced rows is not enough.

## Optimizer experiment, precisely stated

The four required arms are:

1. AdamW reference;
2. official Muon plus post-step per-head Q/K clipping (MuonClip);
3. official Dion3 without Q/K clipping;
4. **novel** Dion3 matrix updates plus the same post-step Q/K clip.

Arm four is called `dion3-qkclip`, not “official MuonClip.” Dion3 is the
current `NorDion2` alias in the
[Microsoft implementation](https://github.com/microsoft/dion): Dion2
submatrix selection/error feedback plus NorMuon per-neuron normalization.
Embeddings, norms, biases, routing bias, mHC scalars, and output heads use
AdamW. Q and expanded K projections are separate per-head parameter groups.
The two flat routed-expert matrices use 256 independent Dion blocks rather
than one cross-expert orthogonalization.

The E:-drive environment installed the exact current Microsoft package at git
commit `e64832041d8e01989abf609c9550f6307efbff2a`. AdamW CPU smoke passes.
Dion3 is intentionally reserved for CUDA: its official package uses
`torch.compile` for orthogonalization, while the recovered Windows install has
no `cl.exe`. On glm-box, raising TorchDynamo's recompile limit to 128 and
accumulating 1,024 samples per optimizer step made the real BF16 CUDA smoke
succeed: 28.092 seconds cold and 5.584 seconds warm. This validates plumbing,
not optimizer quality.

## FP8 contract

The current scaffold runs FP32 on CPU and BF16 autocast on CUDA. It does **not**
yet claim native FP8. The next implementation step is NVIDIA Transformer
Engine 2.18 on the Ada box:

- BF16 master parameters and FP32 optimizer/error-feedback state remain;
- E4M3 forward / E5M2 backward HYBRID scaling covers FP8-safe dense linears;
- all dimensions exposed to FP8 GEMMs are multiples of 16;
- routed experts require a live Ada capability/throughput probe of
  `GroupedLinear`; documentation establishes grouped MoE semantics, but its
  best fused paths are architecture-sensitive;
- BF16 remains the numerical reference, and FP8 is accepted only after
  loss/gradient and held-out calibration parity.

Transformer Engine's current `autocast` API and precision constraints are
documented by
[NVIDIA](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html).
Calling ordinary BF16 expert einsums from inside an FP8 context would not count
as native FP8 routed training.

## Live checks completed

An isolated CPython 3.12.13 environment was created at
`E:\tools\insignia-falsifier-venv`; nothing was installed on C:. It contains
PyTorch 2.13.0, NumPy 2.5.2, and the pinned Dion checkout above.

```text
tools/test_falsifier_moe.py          PASS
tools/test_train_falsifier_moe.py    PASS
tools/test_falsifier_dataset.py      PASS
tools/test_falsifier_baseline.py     PASS
```

The real-data AdamW smoke trained one default 10.09M-parameter step and ran a
prompt-held-out validation pass in 0.31 seconds on the local CPU. Both the
exact-Gram-only and on-policy-row loss paths were exercised separately. A
normal run correctly stopped with:

```text
refusing underfilled training corpus: 279 on-policy rows < --minimum-rows 10000
```

The standalone eager CUDA incremental runtime was rejected at 507.509 ms per
four-row round. The purpose-built i7-14700KF AVX-VNNI ceiling instead reached a
3.1618 ms seven-run median with real exported matrix weights. Complete
full-controller INT8 fake quantization over 13,104 events retained
0.999707--0.999975 cosine across output heads and 97.00% mean Top-2 expert
membership. Those are arithmetic/runtime results, not learned predictor quality
or integrated engine speed; the checkpoint contains only one smoke update.

## Promotion gates

1. Collect at least 10k on-policy rows and enough exact/intervention labels to
   populate every expert; publish prompt-family counts and router load tails.
2. Beat logistic, tree, dense-MLP, and no-MLA/no-mHC/no-MoE ablations on
   prompt-held-out q99 risk, not only mean loss.
3. Populate free-trajectory and collapse labels; the ArXivLean looping case
   must rank as unsafe before any deployment claim.
4. Compare AdamW, MuonClip, Dion3, and Dion3-QKClip at matched update/token
   budgets. Do not select an optimizer from training throughput alone.
5. Establish BF16 reference, then FP8 loss/gradient parity and speed on Ada.
6. Shadow runtime overhead target: below 5 ms/verify round initially, below
   1 ms after CUDA graph/fusion. The controller must preserve more saved I/O
   than it adds compute latency.
7. Replay forced logits with MSE/cosine/KL/JS/PPL and calibrated false-negative
   intervals, then run 30/40/100/240-token free trajectories and hard decoded
   outputs. The user's +3.5% PPL ceiling applies only when those hard outputs
   remain useful.

## Commands

Local plumbing:

```powershell
E:\tools\insignia-falsifier-venv\Scripts\python.exe tools\test_falsifier_moe.py
E:\tools\insignia-falsifier-venv\Scripts\python.exe tools\test_train_falsifier_moe.py
E:\tools\insignia-falsifier-venv\Scripts\python.exe tools\train_falsifier_moe.py `
  --smoke --data "scratch/falsifier-data-20260830/*-onpolicy.npz"
```

Real training remains blocked until the corpus gate passes:

```bash
python tools/train_falsifier_moe.py \
  --data '/var/lib/insignia/falsifier-corpus-v3/*.npz' \
  --optimizer dion3-qkclip --precision bf16 \
  --output /var/lib/insignia/falsifier-runs/v1-dion3-qkclip.json
```
