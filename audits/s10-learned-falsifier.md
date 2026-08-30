# Session 10: learned causal falsifier and expert-subset controller

Date: 2026-08-30

Status: design frozen enough to capture data. Training and runtime integration
remain open.

## Decision

Replace the scalar DFlash logit guard with a small causal model plus an exact
joint decision solver. The model is not asked merely whether a row looks hard.
It predicts:

1. the local error geometry of retaining any subset of the eight routed
   experts;
2. the probability and magnitude of a final-logit failure;
3. the probability that each verify row will be committed;
4. the latency and I/O cost of each choice under the current host/VRAM cache
   state.

The solver then chooses the cheapest expert subsets satisfying a configured
risk budget. This turns excess CPU/GPU arithmetic into avoided NVMe, host-RAM,
PCIe, and NVFP4 decode traffic.

The first production target remains the safer prefix action space `k=3..8`.
The data format also supports arbitrary subsets of the top eight, because a
prefix is not guaranteed to be the best subset once expert cancellation and
cross-row union reuse are considered.

## Why the hand-written guard is insufficient

The measured `mass80 + draft-margin<0.75 => exact` baseline is useful precisely
because it failed in an informative way:

| prompt | exact bracket | adaptive + guard | exactified rows | result |
|---|---:|---:|---:|---|
| GSM8K p02, 32 tokens | 481.5--569.5 ms/token | 410.7 ms/token | 4/32 | first divergence still token 30; acceptance fell 2.13 -> 1.88 |

The same guard improved GSM full-logit MSE/cosine but slightly worsened MATH
MSE/cosine. In both prompts it failed to repair the only top-1 mismatch. A
row-local final-logit symptom is too late and too lossy: earlier approximate
expert outputs have already perturbed mHC streams, KDA recurrent state, later
routers, and later verification rows.

The falsifier therefore must be causal and stateful over `(verify round,
layer, row)`, and it must distinguish a likely committed prefix from a likely
rejected tail.

## Runtime observation boundary

The policy may only consume information available before the selected expert
records are executed. Using exact expert-output errors as an input would leak
the answer.

Available before a verify round:

- previous target top logits and a deterministic vocabulary sketch;
- all current DFlash2 draft logits, including row-0 target/draft calibration;
- prior accept/reject lengths, empty-round streak, and acceptance EMA;
- previous policy choices and their observed consequences;
- target and draft token IDs, ranks, entropy, margins, and disagreement;
- current host-tier and VRAM-tier occupancy summaries.

Available at sparse layer `l`, before its expert union is staged:

- baseline top-eight and widened top-32 expert IDs for every verify row;
- raw router logits, sigmoid scores, bias-adjusted choice scores, normalized
  routed weights, gaps, entropy, and cumulative mass;
- overlap with earlier rows, layers, rounds, predicted routes, and the current
  layer's row union;
- per-selected-expert host-ready, host-in-flight, and VRAM-resident bits;
- a small deterministic sketch of the normalized input hidden state;
- layer type (KDA or MLA), layer index, verify row, position, and draft length.

Only observations from layers `<= l` may affect the action at layer `l`.
Future real router choices are forbidden during training and inference. A
separate causal pre-attention route predictor may supply explicitly marked
predictions for future layers.

## Exact contribution-geometry teacher

The exact verification path already materializes every routed expert output
`e_i` and its router weight `w_i`. Define the weighted contribution

```
v_i = w_i e_i,             i in 0..7
y   = sum_i v_i
G_ij = dot(v_i, v_j) / hidden
```

`G` is an 8x8 positive-semidefinite Gram matrix. Its 36 upper-triangle values
determine the local MSE, output norm, and cosine for **all 256 expert subsets**
without another model execution:

```
error(S)^2 / hidden = sum_(i,j not in S) G_ij
norm(S)^2  / hidden = sum_(i,j in S)     G_ij
dot(y, y_S)/hidden  = sum_(i, all j in S) G_ij
```

This is strictly more informative than recording seven prefix errors. It also
exposes cancellation: a low-score expert can be important because it cancels
another contribution, while two rows can keep the same expert for the I/O
price of one union member.

The exact Gram matrix, local prefix frontier, and downstream full-logit errors
are labels. They are never runtime features.

## Cache-aware near-tie substitution

Expert pruning and expert substitution are separate action dimensions. Pruning
reduces compute and union size by executing fewer than eight baseline experts.
Substitution retains eight-way compute but replaces a low-ranked baseline
expert with a top-16/32 near tie that is already resident. The latter can avoid
a 12.8 MiB record even when quality requires k=8.

The trace schema therefore records top-32 IDs, raw logits, corrected choice
scores, and four 32-bit residency masks in addition to the baseline top eight.
The original top-eight partial sort still executes unchanged; widened sorting
is diagnostics-only and cannot perturb the effective model.

The first substitution controller holds the top six or seven baseline experts
fixed and lets the joint solver choose the tail under both a learned damage
bound and actual union-transfer cost. Router-score regret is a feature and a
weak prior, not a quality label. Alternative-expert teacher labels require
targeted on-policy interventions because the baseline contribution Gram cannot
describe an expert that was not executed.

## Model: Falsifier-MoE v0

The deliberately small first model is sized to stay resident and to be fused
into one invocation per sparse layer:

- `d_model=160`;
- four mHC residual streams;
- two causal blocks;
- four query heads with a 48-wide compressed MLA KV latent;
- short incremental context over the current and previous verify rounds;
- eight FFN experts, top two, intermediate width 192;
- approximately 1.8 million parameters;
- BF16 master weights for training, FP8 or exact INT8/VNNI inference after
  calibration.

The four residual streams represent different evidence channels rather than
four copies of one feature vector:

1. target/draft logit and acceptance history;
2. router score and expert-identity history;
3. hidden/KDA/MLA state sketches and layer position;
4. cache residency, union reuse, and measured cost.

mHC constrains the learned four-by-four residual mixing matrices with Sinkhorn
normalization. This gives the controller stable multi-stream propagation while
still letting evidence move between channels. The mHC paper motivates the
doubly-stochastic constraint as a way to recover identity-like residual
stability in hyper-connections: <https://arxiv.org/abs/2512.24880>.

MLA is useful here for a reason beyond fashion. The controller needs a causal
history spanning roughly `42 sparse layers x verify rows` but its complete KV
history should remain tiny. DeepSeek-V2 introduced MLA specifically to compress
the inference KV representation: <https://arxiv.org/abs/2405.04434>.

### Heads

For each `(layer,row)` event the model emits:

- a rank-4 factor plus diagonal for a normalized 8x8 contribution Gram
  prediction;
- calibrated `P(top1 flip | action)` for each prefix `k=3..8`;
- predicted final-logit MSE, cosine loss, and KL quantiles (median, q95, q99);
- acceptance hazard for this row and the predicted committed-prefix
  distribution;
- expected incremental NVMe bytes, H2D bytes, and latency for each action;
- an uncertainty score used by active teacher-data collection.

The Gram head supplies structure and sample efficiency. The direct risk heads
learn nonlinear propagation through later attention, mHC, KDA recurrence, and
MoE routing. Neither is trusted alone.

## Joint action solver

At one sparse layer the controller chooses all verify rows together. For the
initial prefix action space:

```
A_r = {3, 4, 5, 6, 7, 8}
```

For four rows there are only `6^4 = 1,296` assignments; even seven rows require
`6^7 = 279,936`. Exact enumeration or branch-and-bound is cheap on the
i7-14700KF compared with reading one 12.8 MiB expert record.

For assignment `a`, minimize

```
sum_r commit_probability[r] * predicted_damage[r, a_r]
+ lambda_io   * missing_union_bytes(a, residency)
+ lambda_h2d  * missing_device_bytes(a, residency)
+ lambda_time * predicted_layer_latency(a)
+ lambda_tail * rejected_tail_waste(a)
```

subject to configurable q99 MSE/cosine/KL and top-1-flip budgets. Union cost is
computed exactly from expert IDs and residency bits, not approximated as a sum
of independent row costs.

Phase two may admit arbitrary top-eight subsets. The Gram head and exhaustive
256-subset teacher make that possible, but it must beat the prefix controller
on held-out prompts before production integration.

## On-policy and counterfactual data

Random row splits are invalid because adjacent rows from the same prompt leak
state and vocabulary. Split by prompt family and complete conversation.

Collection proceeds in waves:

1. **Exact teacher.** Run exact top eight and capture causal features, expert
   contribution Gram matrices, logits, acceptance, and real I/O timing.
2. **Broad policy arms.** Run fixed k, retained-mass, prefix-shaped, and a small
   number of random subset masks on the same forced token streams.
3. **Single interventions.** At selected `(round,layer,row)` events, change one
   action while holding the earlier prefix exact. Measure immediate and future
   target-logit effects. This identifies propagation rather than correlation.
4. **On-policy shadow.** Let the current model choose actions, execute an exact
   shadow on uncertain/high-value examples, and add disagreements.
5. **DAgger loop.** Retrain on states induced by the policy itself, with extra
   sampling around q99 violations, long accepted prefixes, KDA-heavy spans,
   and cache-state transitions.

The first useful corpus target is 10k--20k tokens across coding, prose,
GSM8K-like arithmetic, MATH-like symbolic work, long context, multilingual,
and adversarial low-margin continuations. A 1.8M-parameter controller should
not be accepted from the current few dozen rows. Scale toward 100k tokens only
after learning curves show that data, rather than missing features or label
noise, is the constraint.

## Vocabulary representation

Writing every 154,880-wide logit vector into the training tensor is wasteful.
Retain the raw dumps as ground truth, then build causal features from:

- top-32 IDs and centered values;
- entropy, top-1 probability, top-1/top-2 margin, and selected ranks;
- exact/draft top-set overlap and Jensen-Shannon disagreement;
- a deterministic 64-wide signed CountSketch of centered, standardized logits;
- short deltas against the prior block and an incremental MLA history.

The target/DFlash output projection is shared, so token-ID embeddings are
meaningful across both distributions. Rare IDs are hashed into additional
buckets rather than expanded into a 154k controller embedding table.

## Loss and calibration

Use a multi-task loss with explicit false-negative pressure:

- asymmetric focal BCE for top-1 flips and acceptance failures;
- Huber loss on `log1p(MSE)`, cosine loss, KL, bytes, and latency;
- pinball loss for q95 and q99 damage;
- normalized Frobenius loss on the Gram matrix plus subset-consistency loss on
  sampled masks;
- Brier/calibration loss for every probability;
- regret loss between the chosen assignment and the teacher's cheapest safe
  assignment;
- standard MoE load-balance auxiliary loss and router z-loss.

Calibrate held-out q99 estimates with conformal residuals per prompt family and
context bucket. Runtime risk thresholds operate on calibrated upper bounds,
not raw neural confidence.

## Optimizer

Train hidden two-dimensional matrices with Muon, using separate parameter
groups for Q, K, and V. Train embeddings, gains, biases, normalization
parameters, output heads, and small router/mHC parameters with AdamW. Keep FP32
optimizer state and BF16 forward weights; quantize only after the BF16 model is
frozen.

The Muon paper reports that careful weight decay and update scaling matter and
uses the optimizer in MoE training: <https://arxiv.org/abs/2502.16982>. The
official implementation likewise recommends Muon for hidden 2-D matrices and
AdamW for embeddings, heads, gains, and biases, and notes that separate Q/K/V
updates work better: <https://github.com/KellerJordan/muon>.

## Required baselines

Complexity earns its keep only if it beats all of these on prompt-held-out
data:

1. retained-mass threshold;
2. current draft-margin guard;
3. calibrated logistic regression;
4. a small gradient-boosted tree over the identical causal features;
5. a dense two-layer MLP with the same parameter and inference budget;
6. ablations removing MLA, mHC, MoE, hidden sketch, cache state, and acceptance
   history one at a time.

The learned MoE proceeds to runtime only if its q99 safety/cost frontier is
strictly better, not merely its average validation loss.

## Deployment gates

1. **Trace validation:** binary records have a versioned 64-byte header,
   cache-line-aligned fixed records, stable hashes, and no future information.
2. **Offline calibration:** prompt-held-out top-1 false-negative upper bound,
   MSE, cosine, KL, PPL, and expected regret are reported with bootstrap
   intervals.
3. **Shadow mode:** inference runs but actions are ignored; overhead must be
   below 1 ms/verify round initially and below 0.5 ms after fusion.
4. **Replay mode:** policy actions run on fixed teacher tokens. Report full
   logits and recurrent drift, not only text resemblance.
5. **Free generation:** targeted 30/40/100/240-token campaigns with exact
   bracket timings and acceptance histograms.
6. **Production:** a hard env knob selects the maximum calibrated risk. Exact
   top eight remains the immediate fallback.

## Immediate implementation order

1. Emit one fixed-size event record containing top-32 router context and
   residency, hidden CountSketch, and the 36-value baseline contribution Gram
   label while the exact MoE diagnostic path already has all eight outputs.
2. Assemble event records with target/DFlash logits into prompt-sharded NPZ
   datasets and reject any epoch/row alignment mismatch.
3. Fit logistic/tree/dense baselines before installing a training stack.
4. Train Falsifier-MoE v0 with Muon on the E: drive.
5. Integrate shadow inference and measure overhead.
6. Add the joint union-aware solver, first for prefix k and then for arbitrary
   subsets if the held-out frontier justifies it.

## 2026-08-30 on-policy trace correction

The first three prompt datasets combined an exact-teacher event trace with
logits from the approximate arm. That is useful for a one-step counterfactual
but is not an on-policy corpus: once an approximate action changes the hidden
trajectory, later exact-trace sketches/router/cache observations describe the
wrong state.

`INSIGNIA_GLM53_DF_FALSIFIER_FEATURE_TRACE=<path>` now emits the same 896-byte
record on the actual approximate/cache-aware trajectory, immediately before
expert staging and execution. It records:

- the actual selected expert IDs/weights and executed `k`;
- the untouched top-32 router candidate ranking, from which baseline top eight
  is recoverable;
- pre-staging host/in-flight/device/pinned residency masks;
- router summary and the current hidden-state CountSketch/input norm.

It deliberately performs no expert-output download. Header Gram-present bit 0
is clear and event flag bit 2 is set. The dataset builder replaces unavailable
Gram/tail labels with NaN and emits `event_label_mask=false`; row-level MSE,
cosine, KL, JS, top-1, top-10, and max-error labels remain available from the
paired exact/approximate full-logit dumps. Exact traces retain their original
bit-identical v2 layout and Gram validation.

The quality harness now captures a separate feature trace for every
approximate policy arm. This closes the trajectory mismatch before fitting any
learned baseline or Falsifier-MoE.

### Live validation

Targeted p02/p10/p12 cache-route captures each produced 1,302 events (42 sparse
layers x 31 verification rows) and joined without an epoch/layer/row mismatch.
Comparing p10's old exact-teacher features with its new on-policy features
showed why the correction is mandatory:

- only 31/1,302 hidden CountSketches remained bit-identical;
- hidden-sketch cosine mean/p01/min was 0.98243/0.86004/0.77137;
- relative hidden-sketch L2 median/p95/max was 13.33%/37.37%/68.50%;
- the baseline top-eight route differed at 811/1,302 events (1,259 expert-ID
  substitutions), and top-32 candidate overlap fell as low as 43.75%.

## Previous-logit calibration guard

The first compute-rich falsifier arm is
`INSIGNIA_GLM53_DF_CALIBRATION_GUARD_JS=<threshold>`. The engine retains the
previous accepted target's full logits, compares them with DFlash row zero via
exact full-vocabulary Jensen-Shannon divergence, and suppresses approximate
expert routing for the block when risk exceeds the threshold. This spends
14700KF math to avoid an unsafe bandwidth optimization; it does not fetch any
additional expert outputs. Accepted target logits add one 620 KiB D2H copy per
round.

At JS=0.6000 on p12, runtime reproduced the offline first-block value exactly
(0.656733). It exactified only epoch 1; epochs 2--8 stayed cache-aware. The
guard retained 955/1,086 (87.9%) of cache substitutions while changing quality:

| metric | cache route | + JS guard | change |
|---|---:|---:|---:|
| top-1 agreement | 31/32 | 32/32 | flip removed |
| mean cosine | 0.991227 | 0.992543 | +0.001316 |
| mean MSE | 0.1391 | 0.1230 | -11.6% |
| mean KL | 0.008453 | 0.004430 | -47.6% |
| mean JS | 0.001960 | 0.001028 | -47.6% |
| forced-token PPL | 1.0790 | 1.0573 | -2.0% |

The p12 free-generation bracket was storage-noisy (exact controls 624.6 and
474.2 ms/token). Cache route measured 557.0 ms/token and guarded cache route
523.7 ms/token, so no decode win is claimed. Draft time increased from 17.9 to
21.5 ms/speculative round: about 3.6 ms/round or 1.2 ms/token at 2.91 accepted
tokens/round. The risky first p12 round rejected before target verification, so
the free-generation guard did not alter that sequence; this is useful evidence
that the guard pays little but must be validated on prompts where the risky
block actually verifies.

### Prompt-held-out CPU baseline

`tools/train_falsifier_baseline.py` now builds two causal feature views and
runs nested prompt-held-out ridge evaluation without sklearn/PyTorch:

- `logit_features`: previous-target/current-DFlash scalars plus three 64-wide
  vocabulary CountSketches;
- `full_runtime_features`: the logit view plus on-policy hidden/router/cache
  aggregates, executed-vs-baseline action features, coarse layer bins, and
  deterministic route sketches.

The risk target combines MSE, cosine damage, KL, JS, and a high penalty for a
top-1 flip. With only p02/p10/p12 (93 rows total), the honest result is mixed:

| features | held out | NRMSE / train-constant | Spearman | top-25% risk captured | flip rank |
|---|---|---:|---:|---:|---:|
| logits | p02 | 1.001 | 0.208 | 28.8% | - |
| logits | p10 | 0.988 | 0.116 | 39.2% | - |
| logits | p12 | 0.997 | 0.256 | 44.0% | 7/31 |
| full runtime | p02 | 1.019 | 0.093 | 20.6% | - |
| full runtime | p10 | 0.951 | 0.303 | 39.4% | - |
| full runtime | p12 | 0.963 | 0.441 | 37.8% | 19/31 |

Previous logits alone rank the one held-out flip inside the top quarter, while
the much wider runtime view improves continuous ranking on p10/p12 but
overfits the flip. This is evidence for the previous-logit hypothesis and
against deploying the 1.8M-parameter Falsifier-MoE from three prompts. The
pipeline is ready; prompt diversity and on-policy interventions are now the
binding constraint.

### Seven-prompt wave and flip head

The targeted corpus wave added p01/p03 GSM8K and p09/p11 MATH without an ABCD
sweep: one exact pass and one cache-policy pass, 32 forced rows each. All four
traces joined at 1,302 events. It added four top-1 failures (p01 row 10, p03
rows 4/30, p09 row 11); p11 stayed 32/32. The corpus now has 217 rows over
seven prompt split units and five positive top-1 failures.

The class-balanced L2 logistic head confirms that five failures are still not
enough for deployment. Logit-only held-out flip ranks were p01 26/31, p03
3/14 of 31, p09 27/31, and p12 5/31; the wider runtime view was worse on the
rare-class target. It generalizes across p03/p12 but reverses risk on p01/p09.
This rules out threshold/model tuning on the current corpus. The next data wave
needs explicit single-row interventions and more prompt families, especially
low-margin/calibration-shift cases.

## Joint layer-union cache routing

Independent cache routing minimizes each row's immediate transfers, but the
engine stages the union of all verify rows. An offline exact solver now keeps
the six strongest feasible actions per row and exhaustively evaluates at most
`6^4 = 1,296` assignments per sparse layer under the same per-row regret cap.
Across the original p02/p10/p12 traces at K=32, retain=7, regret=0.0025:

| prompt | independent union saved | joint union saved | independent H2D saved | joint H2D saved |
|---|---:|---:|---:|---:|
| p02 | 2.86% | 9.71% | 8.61% | 11.24% |
| p10 | 2.53% | 9.02% | 8.81% | 11.00% |
| p12 | 3.22% | 9.43% | 9.07% | 11.57% |

The C++ arm is opt-in with
`INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS=6`, specialized to retain-7 and k<=4;
larger chunks retain the locally optimal action. On the live p02 on-policy
trace, independently recomputed observed savings exactly matched the joint
solver: 15.90% disk, 11.74% H2D, and 9.97% union, versus an independent
ceiling of 15.85%/8.84%/3.25% on the same state.

Quality also moved in the favorable direction on p02: 32/32 top-1 remained,
cosine improved 0.996381->0.997045, MSE fell 0.07376->0.05930 (-19.6%), KL
fell 0.005793->0.003719 (-35.8%), JS fell 0.001392->0.000891 (-36.0%), and
forced-token PPL moved 1.1262->1.1136. This is one prompt, so it is evidence
for the arm rather than a global quality claim; cold decode timing follows.

The first p02 cold bracket measured independent cache routing at 426.5
ms/token (2.34 tok/s) and joint routing at 406.2 ms/token (2.46 tok/s): -4.8%
latency / +5.0% throughput. Prompt prefill was unchanged at 20.371 versus
20.383 seconds for 74 tokens (~3.63 tok/s). Joint routing reduced verify time
from 1,134.7 to 1,102.6 ms/verified round and changed the speculative path from
15 rounds/2.13 accepted to 14 rounds/2.29 accepted. Exact brackets still swung
569.1->466.4 ms/token, so a reverse-order repeat is required before promoting
the result beyond opt-in.
