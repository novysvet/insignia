# Session 10: native AVX-VNNI Falsifier runtime

Date: 2026-08-30

Status: the standalone i7-14700KF runtime ceiling passes the initial 5 ms per
four-row verify-round gate. The complete v2 export now includes every learned
auxiliary tensor, and causal online feature state matches the stored corpus.
It is not yet wired into `glm53-generate`, native/PyTorch fixture execution on
glm-box remains pending, and the available checkpoint is a one-step smoke
checkpoint rather than a trained policy.

## Verdict

The eager CUDA implementation is dead for inference. On the 4070 Ti SUPER it
took 507.509 ms per four-row round, including 12.084 ms per target-layer group.
That is larger than the latency it is meant to save and competes with the main
model for scarce VRAM and launch bandwidth.

The correct deployment target is the otherwise underused i7-14700KF. A
Raptor-Lake-only INT8 runtime now executes the complete synthetic controller
pipeline with real exported matrix weights in a seven-run median of **3.1618
ms per four-row verify round**, or 0.07528 ms per synchronized target-layer
group. Loading and applying all 18 exact auxiliary tensors moves that median
only to **3.1849 ms** (+0.73%). The matrix-only ceiling is **1.2106 ms**, 378.6
physical GMAC/s. The gap
is normalization, activation, routing, attention, quantization, and barriers;
it is not missing matrix work.

These numbers are a runtime ceiling, not a claim that a trained controller is
already improving GLM output or engine wall time.

## What was measured

The benchmark is `tools/benchmark_falsifier_vnni_pipeline.cpp`. Each round has
four verify rows and 42 sparse target layers. Each event executes the exact v1
matrix ledger:

| quantity | value |
|---|---:|
| logical matrix MAC/event | 2,721,856 |
| physical padded MAC/event | 2,728,000 |
| physical matrix MAC/round | 458,304,000 |
| matrix-only seven-run median | 1.2106 ms |
| matrix-weights-only pipeline median | 3.1618 ms |
| complete-state pipeline median | 3.1849 ms |
| complete-state range | 3.1562--3.3057 ms |
| complete-state layer-group median | 0.07583 ms |

The full pipeline includes dynamic activation INT8 quantization/dequantization,
SiLU and SiTU-GLU, Top-2/Top-3 routing machinery, six Sinkhorn iterations,
four-stream mHC mixing, block-depth attention, absorbed causal MLA over a
growing latent history, final heads, persistent pinned workers, and one barrier
per target layer. Every run first checks `VPDPBUSD` against a scalar signed
INT8 dot product. All seven measured runs produced checksum
`3657744675213494808`.

The benchmark does **not** yet prove native/PyTorch output parity. Its feature
inputs are deterministic synthetic values. Export format v2 and the C++ loader
now consume the 24 large linears plus all 18 exact FP32 auxiliary tensors:
positional matrices, embeddings, RMS weights, mHC base logits/scales, stream
scales, and routing bias. A deterministic 42-event fixture generator and C++
head/route comparator are implemented and structurally tested, but the real
i7 fixture run could not be completed after glm-box went offline.

## Optimization progression

| change | four-row full-pipeline time | result |
|---|---:|---|
| initial scalar-output VNNI pipeline | 5.2379 ms | baseline |
| four-output row tile | 4.6933 ms | kept |
| streaming absorbed MLA + predequantized/transposed K/V | 3.2677 ms | kept |
| checksum-validated real matrix weights | 3.2224 ms | kept |
| tied-register VEX `VPDPBUSD` | 3.1618 ms | kept/default |
| all 18 exact FP32 auxiliary tensors | 3.1849 ms | kept; +0.73% |
| eight-output row tile | ~3.276 ms | rejected; register spill/copy carousel |

In a strict alternating 500-iteration A/B, the intrinsic and tied-assembly
medians were 3.26218 and 3.18245 ms: a **2.44%** reduction, with assembly faster
in all five pairs. A clean seven-run build subsequently measured 3.1618 ms.
The assembly uses a tied `+x` accumulator and explicit VEX encoding:

```asm
vpdpbusd memory_weight, ymm_input, ymm_accumulator
```

GCC 16 otherwise treated the intrinsic result as a fresh SSA destination and
copied four accumulators after each K tile. Disassembly confirmed `C4` VEX
encoding and removal of that register-copy chain. This exact host has no
AVX-512; forcing VEX is required because an unconstrained assembler spelling
selected an illegal EVEX form.

`build/falsifier-vnni.sh` freezes the required `-ffast-math` build contract.
Without it, the same binary measured about 5.57 ms because scalar normalization
and activation functions dominated. Fast math changes the floating-point
checksum, so native-vs-PyTorch tolerances still need to be established; it does
not affect the exact INT8 dot-product check.

## Weight export

`tools/export_falsifier_vnni.py` exports a 64-byte-aligned, per-row symmetric
INT8 format with padded K dimensions, signed-to-unsigned correction terms,
scales, biases, a per-payload FNV-1a checksum, and a manifest checksum.

The complete-state artifact is:

```text
/var/lib/insignia/falsifier-runs/v1-dion3-smoke-v2.ifvnni
size                 11,286,720 bytes (10.764 MiB)
SHA-256              e7c624d9602197505778074e323a3b1e596dec38dc231693d0065e5452019104
manifest checksum    7542981408582747553
entries              24 INT8 matrices + 18 FP32 tensors
worst weight cosine  0.9999755345
worst weight MSE     1.463347e-7
worst max abs error  6.947368e-4
```

The loader verifies every payload and the aggregate manifest before execution;
it remains backward-compatible with the original 24-entry v1 artifact.

## Online feature state

`tools/falsifier_online_features.py` now defines the causal state machine for
the nine per-event residency/reuse features. Replaying every local on-policy
shard reproduces `event_derived` and `expert_multiplicity` bit-for-bit,
including previous-layer, previous-row, previous-round, layer-union, and
uniqueness signals.

The 208-wide logit encoder also had three unused padded inputs. Dataset v3 uses
them for previous-round target-logit JS divergence, centered cosine, and Top-1
disagreement. The old noncausal `block_fraction` (which depended on knowing the
campaign's eventual length) is replaced by causal log-round position. The
online and offline builders produce bit-identical 16-scalar plus 3x64-sketch
features in the synthetic alignment test. Matrix geometry and runtime MACs do
not change. Dataset v2 remains loadable for old artifacts; new collection
should use v3 before real training.

## Full-controller INT8 numerical gate

`tools/evaluate_falsifier_int8.py` applies per-row weight fake quantization and
dynamic activation fake quantization to the complete PyTorch controller. The
absorbed MLA latent remains floating point, matching the intended CPU runtime.
All 13,104 remote on-policy events were evaluated in 58 batches:

| output | cosine | MSE | argmax agreement |
|---|---:|---:|---:|
| hidden | 0.999851 | 2.974e-4 | 97.89% |
| immediate risk | 0.999843 | 1.163e-4 | 99.18% |
| forced hazard | 0.999782 | 1.111e-4 | 99.14% |
| forced peak | 0.999970 | 2.366e-5 | 99.31% |
| free hazard | 0.999791 | 1.313e-4 | 99.17% |
| collapse | 0.999707 | 1.115e-4 | 99.32% |
| contribution Gram | 0.999868 | 3.799e-4 | 98.36% |
| action risk | 0.999795 | 1.208e-4 | 98.89% |
| action cost | 0.999975 | 3.987e-5 | 99.29% |
| acceptance | 0.999785 | 1.150e-4 | 98.27% |
| router logits | 0.999851 | 1.070e-4 | 97.78% |

Exact ordered expert-slot agreement is 94.42%. Exact unordered Top-2 set
agreement is 94.07%, and mean overlap is 1.9400/2: **97.00% expert-membership
retention** over 39,312 route events. This establishes INT8 arithmetic
viability. It does not establish predictor quality because the source
checkpoint contains only one Dion3 smoke update.

The complete report is stored at
`/var/lib/insignia/falsifier-runs/v1-dion3-smoke-int8-quality.json`.

## Dion3 CUDA smoke

The official Dion package originally hit TorchDynamo's recompile limit. Raising
the limit to 128 and accumulating 1,024 samples per optimizer step fixed the
failure. A real BF16 CUDA Dion3 step, validation, and checkpoint now succeeds:

| run | wall time |
|---|---:|
| cold compile | 28.092 s |
| warm | 5.584 s |

This proves the optimizer path runs. It does not select Dion3 over AdamW,
MuonClip, or Dion3-QKClip; that requires the gated on-policy corpus and matched
training ablations.

## Commands

```bash
bash build/falsifier-vnni.sh

/var/tmp/insignia-build-raptor/benchmark-falsifier-vnni 4 500

/var/tmp/insignia-build-raptor/benchmark-falsifier-vnni-pipeline \
  4 500 /var/lib/insignia/falsifier-runs/v1-dion3-smoke-v2.ifvnni

python tools/evaluate_falsifier_int8.py \
  --data '/var/lib/insignia/falsifier-corpus-v3/*.npz' \
  --checkpoint /var/lib/insignia/falsifier-runs/v1-dion3-smoke-warm.pt \
  --output /var/lib/insignia/falsifier-runs/v1-dion3-smoke-int8-quality.json
```

## Next gate

1. Run the implemented fixed native fixture on glm-box and compare hidden,
   concatenated heads, and all three Top-2 routes against PyTorch.
2. Extract the monolithic benchmark into a reusable runtime with explicit
   feature/state structs.
3. Build online previous-target/current-draft CountSketches without a new
   full-vocabulary D2H transfer; populate the remaining inputs from
   `DfFalsifierEventV2`.
4. Add an environment-gated shadow worker that never changes routing and
   measure how much of its ~3.16 ms is hidden under target GPU execution.
5. Collect the >=10k on-policy rows plus aligned free-run labels, train the four
   optimizer arms, and only then permit controller decisions behind the hard
   MSE/cosine/KL/JS/PPL and decoded-output gates.
