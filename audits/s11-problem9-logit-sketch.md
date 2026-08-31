# Problem 9 logit sketches: collision accepted; exact GPU guard integrated

Date: 2026-08-30

## Decision

The report found a real and severe defect in the dataset-v3 Falsifier logit
representation.  Preserve its collision as a permanent regression fixture and
do not train another production checkpoint that assumes the three public,
independently-standardized CountSketch rows are sufficient.

Do **not** apply the delivered v4/TRF-JS patch as a production schema.  Its
mathematical feature map is useful, but the proposed exact-head correction is
not connected to the current model input, key rotation is checkpoint
incompatible, and its NumPy encoder is slower than v3.  The first production
experiment therefore spends GPU compute on an exact fused divergence reduction
while retaining one previous logit vector on device.  That path is now wired
behind explicit DFlash guard knobs; it is both more accurate and simpler for
this engine's fixed hardware.

## Provenance

- Source: `C:\Users\Pufos\Downloads\problem9_solution_bundle.zip`
- Archive SHA-256:
  `75481904c0dbaa42cfeddb39c32a9d473606831ad00c091989499b12116e0604`
- Review extraction:
  `scratch/problem9-solution-review-20260830/`
- The supplied patch was treated as evidence and was not applied.
- The collision is now locked independently by
  `tools/test_falsifier_v3_collision.py`.

The random-feature identity is consistent with Abdullah et al., *Sketching,
Embedding and Dimensionality Reduction for Information Theoretic Spaces*,
AISTATS 2016: <https://proceedings.mlr.press/v51/abdullah16.html>.  The report's
Gap-Hamming premise is also consistent with the established linear randomized
communication lower bound.  Those results support the mathematical direction;
they do not validate this repository's proposed runtime wiring.

## Reproduced v3 collision

Command on the local Ryzen box, using the E:-resident Python environment:

```text
rtk E:\coding\python-envs\insignia-win\Scripts\python.exe tools\test_falsifier_v3_collision.py
```

Result:

```text
falsifier v3 collision: PASS JS=0.691768078 centered_cos=-0.998642374 dangerous_mass=2.539e-13->0.998010376
```

The construction reserves a shared strict Top-32 head, then swaps logits 9
and -20 inside every public `(bucket, sign)` cell.  Safe and risky worlds have:

- bit-identical 16 scalar + 3x64 sketch inputs;
- identical Top-32 IDs and centered values;
- identical logit multisets, entropy, top-1 probability, and margin;
- natural Jensen--Shannon divergence 0.691768078, near `ln(2)`;
- centered cosine -0.998642374;
- selected tail-set probability changing from `2.539e-13` to 0.998010376.

This is not a random hash collision search.  It follows deterministically from
the public rank-64 linear map and survives the per-vector standardization that
v3 applies.  The test exercises the repository's actual `CountSketch` and
`OnlineLogitState`, not a reimplementation from the archive.

## What in the report is useful

1. V3 discards independent additive shifts, so raw cosine is unidentifiable.
2. Standardizing `current-prior` discards change magnitude, so centered MSE is
   not recoverable from the third row.
3. A fixed public 64-row map has an enormous deterministic nullspace.  Private
   random-sketch average-case bounds are not adversarial guarantees for it.
4. KL is unbounded and should be an exact fused scalar whenever both vectors
   coexist, not inferred from a fixed tiny generic sketch.
5. The token-random-frequency JS feature map is an unbiased way to represent
   JS geometry.  Its 124-counter estimate is most useful when residual JS is
   diffuse, or when an exact head removes most absolute heavy-tail error.
6. Restoring prior/current mean and centered RMS is necessary if a learned
   controller is expected to reason about affine logit geometry.

The delivered 192-float accounting is internally consistent as a message:
64 linear counters + 124 JS counters + four per-vector statistics = 192
float32 values = 768 bytes.  Its suggested event accounting is 20 scalar +
188 relation coordinates = the existing 208-float model width.  Equal byte
counts, however, do not imply runtime or checkpoint compatibility.

## Independent local benchmark

A bounded reproduction used two pairs per non-adversarial class, three private
seeds, and five Windows timing iterations on the Ryzen 5 5600X:

```text
rtk E:\coding\python-envs\insignia-win\Scripts\python.exe \
  scratch\problem9-solution-review-20260830\benchmark_logit_sketches.py \
  --output scratch\problem9-solution-review-20260830\root-check-win \
  --pairs 2 --seeds 3 --timing-iterations 5
```

| NumPy operation | median |
|---|---:|
| v3 full triplet | 5.0014 ms |
| v3 with prior cached | 3.3672 ms |
| proposed total encode | 11.7593 ms |
| proposed after reusable softmax stats, exact sin/cos | 7.7604 ms |
| proposed after reusable softmax stats, LUT4096 | 8.8416 ms |

On this machine the random LUT is slower than vectorized sin/cos.  The proposed
post-softmax encoder is 2.30x the cached-v3 NumPy time, not a performance win.
The earlier WSL reproduction measured 8.528 ms versus 1.276 ms, showing that
this prototype's ratio is also backend-sensitive.  Neither is a native-kernel
measurement.

The bounded accuracy run reproduced the exact collision and showed the TRF-JS
estimate clearly separates it from zero, but with substantial seed variance:
0.59336, 0.75041, and 0.88219 for a true value of 0.69177.  Non-adversarial
results were synthetic and mixed: the proposed centered-cosine estimate was
better on flat and adversarial inputs, but worse than v3 on this run's
heavy-tail and multimodal inputs.  This is not enough to select a training
schema.

An independent 128-seed rerun on the exact collision measured mean 0.6970,
sample standard deviation 0.08225, minimum 0.4800, 5th percentile 0.5683, and
maximum 0.8799.  At the current 0.6000 calibration-JS threshold, 8.59% of these
dangerous estimates fall below the guard; 47.66% exceed the true JS ceiling
`ln(2)`.  The estimator therefore needs clamping plus a calibrated confidence
policy or repetitions before it can guard a round.

The delivered allocation sweep also does not select its stated 64/124 split.
Under its own small synthetic score, 48 linear / 140 JS coordinates scores
0.081926 and 96/92 scores 0.082666, while the chosen 64/124 scores 0.099759.
That sweep uses only four pairs per class and three seeds, so none of these is
a trustworthy production choice; the discrepancy is another reason to defer
the schema.

## Runtime integration defects in the proposal

### Top-32 control variate is absent from the model

`TraceEncoder.forward` consumes only `row_scalars` and flattened
`row_logit_sketch`.  The dataset writes `prior_top_ids/values` and
`draft_top_ids/values`, but no current model path reads them.  Online code uses
the IDs only to emit Top-8 and Top-32 overlap scalars and discards the values.

Consequently the report's decoder-side operation--subtract the Top-32
intersection from both hashed feature vectors, compute the residual squared
norm, and add exact head JS--does not happen in Insignia.  The proposed
20+64+124 layout also reserves no scalar coordinate for exact head JS.  Its
benchmark evaluates a nonlinear external decoder, while the deployed encoder
first presents raw coordinates to `Linear(208,d)+RMSNorm`.  A later nonlinear
network may learn a rough norm, but it is not the claimed estimator or exact
control variate.

Even in the standalone decoder, reconstructing Top-32 probabilities from
float32 centered values and top-1 probability is numerically near-exact rather
than literal: the audit observed maximum probability error `1.93e-8` and head
JS error `4.0e-11`.  This is harmless beside the missing plumbing, but the
runtime contract should use tolerance language rather than bit-exact language.

### A rotated private key silently corrupts a checkpoint

The first learned layer can attach semantics to individual hash/frequency
coordinates.  Changing the keyed token map changes those 188 inputs; it is not
an invariant operation.  In a direct same-pair check, encodings made with keys
101 and 202 had cosine -0.158.  A fixed key per checkpoint is possible, but request
or epoch rotation requires multi-key training or pre-reducing invariant scalar
metrics before the model.

Current dataset metadata, PyTorch checkpoint config, VNNI export, and native
loader contain no end-to-end feature-schema/key fingerprint.  The VNNI path
checks width 208 and payload checksums, so it could silently run v3 weights on
v4 inputs or run a checkpoint with the wrong key.  Any v4 work must hard-reject
that mismatch at every artifact boundary.

## Integrated exact GPU path (opt-in)

The calibration-JS guard now retains one accepted target-logit vector on device
and compares it with DFlash draft row zero using the exact fused GPU reducer. It
downloads only the 192-byte metric record. The draft-uncertainty guards run the
seven-row GPU max/log-sum-exp/Top-1 reducer and download 224 bytes. The legacy
`INSIGNIA_GLM53_DF_RETRY_TOP1_DROP` post-verify retry remains a distinct
CPU/full-logit path and is not the calibration-JS implementation. All GPU
allocations and launches are conditional on their guard knobs; no guard is
enabled by default.

The implemented compute-for-bandwidth path is:

1. Allocate one persistent 154880-float device prior buffer (605 KiB).
2. On acceptance, copy the chosen `verify_logits_` row device-to-device into
   that buffer before the next draft overwrites the workspace.
3. Fuse max/log-sum-exp and exact JS reduction for prior versus draft row zero
   on the GPU.  In the same passes, optionally reduce mean, RMS, MSE, raw and
   centered dot products, KL, entropy, and top-1 probability.
4. Transfer only the resulting scalar record to the CPU.

This removes the extra 619,520-byte accepted-row D2H and the three scalar CPU
passes.  The VRAM cost is less than one twentieth of one NVFP4 expert record
and negligible beside the existing DFlash workspace.  It uses the user's idle
Ada compute to preserve full-vocabulary information instead of approximating
it into 1 KiB.  FP8-compressed prior storage is a later A/B arm only after the
exact GPU baseline establishes MSE/cos/KL/JS error and speed.

TRF-JS remains a worthwhile ablation if a future architecture truly cannot
retain the vector.  Before that arm can train:

- add explicit decoded JS/head-JS/cosine scalars or a tested invariant norm
  reducer;
- define a dataset-v4 schema and immutable feature-key ID;
- propagate both through dataset, checkpoint, export, fixture, and native
  loader with hard mismatch rejection;
- rebuild the corpus and retrain--never reinterpret a v3 checkpoint;
- either fix one key per checkpoint or train across keys with an invariant
  downstream representation.

## Exact GPU reducer and generator integration

The exact reducer is implemented in:

- `include/insignia_glm53_logit_metrics.cuh`
- `src/glm53_logit_metrics.cu`
- `tests/test_glm53_logit_metrics.cu`

The pair reducer keeps double accumulators and returns max/log-sum-exp, means,
centered RMS, MSE, centered MSE, raw and centered cosine, both KL directions,
JS, entropy, Top-1 probability, and argmax in one 192-byte result.  A separate
batched reducer returns max/log-sum-exp/Top-1 for all seven DFlash rows.  Inputs
must be finite; zero-norm behavior is fail-closed for the asymmetric case
(cosine zero) and returns one only when both norms vanish.  The test also
contains the exact CountSketch collision witness.

The focused sm_89 build passed with:

```text
nvcc -std=c++20 -O3 -lineinfo -arch=sm_89 \
  -Xcompiler=-Wall,-Wextra,-Werror ...
```

Against the float64 CPU oracle, maximum absolute field error was `1.06e-13`
on random logits, `5.96e-12` on heavy tails, `9.78e-11` on a large constant
shift, and `8.36e-11` on the collision.  The GPU reproduced collision JS
`0.691768078356`.

On the local RTX 4070 SUPER, seven independent runs of 101 warm samples gave:

| Path | Median |
|---|---:|
| exact pair metrics only | 0.232160 ms |
| 619,520-byte prior D2D + pair metrics + 192-byte pinned D2H | 0.272477 ms |
| existing 619,520-byte D2H + CPU JS | 5.612517 ms |
| seven-row GPU stats + 224-byte D2H | 0.261488 ms |
| seven CPU row scans | 5.214313 ms |

That is about `20.6x` for the exposed prior-retention guard path and `19.9x`
for seven-row statistics in this isolated test.  The generator now retains the
accepted target row in a persistent 605 KiB device buffer and invokes the exact
pair reducer for `INSIGNIA_GLM53_DF_CALIBRATION_GUARD_JS`; its uncertainty guard
uses the seven-row GPU reducer.  Only the compact metric records return to the
CPU, and both paths remain opt-in.  The standard build now compiles and runs the
focused metric test.  These are still kernel-path measurements, not end-to-end
decode claims; an in-engine timing A/B and 4070 Ti SUPER repeat remain required
before either guard can become a default policy.

## Acceptance gate

1. CUDA scalar reductions agree with the existing float64 reference on random,
   heavy-tail, flat, multimodal, shifted/scaled, and exact-collision vectors.
2. Report MSE, raw/centered cosine, KL, JS, top-1, Top-10, max error, and PPL;
   no quality claim may rely on qualitative output inspection.
3. Measure seven-run warm medians for guard latency, accepted-row D2H bytes,
   DFlash acceptance, decode ms/token, and committed tokens/second.
4. Require no new full-logit D2H and a net guard-time reduction.  A metric that
   is exact but slower stays diagnostic-only.
5. Retraining requires at least the existing 10,000 on-policy-row floor,
   prompt-family holdouts, free-run collapse labels, and hard MathArena output
   review.  The user permits up to +3.5% PPL only if hard answers remain good.
6. Any approximate v4 arm must fail safely on this collision witness and hard
   reject schema/key mismatches before native inference.
