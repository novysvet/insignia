# Task 2: exact compressed FP8 residency with fused decode

## Goal

Determine whether the engine's E4M3 weight bytes can be stored in a lossless,
fixed-block compressed representation and reconstructed inside the Ada tensor-
core consumer cheaply enough to reduce VRAM/disk traffic. Spend integer/bitwise
compute and shared-memory work to avoid global-memory bytes.

The primary target is the 8.13 GiB dense FP8 group-64 cache; the 1.07 GiB fixed
DFlash2 FP8 cache is a secondary target after the dense path works. Exact mode
must reproduce every original E4M3 byte and every FP16 scale byte. This task is
not the already assigned expert-scale codec and is not cross-expert dictionary
compression.

## Start from a clean repository

```bash
git clone --branch glm53-dflash2-4070ti-super \
  https://github.com/novysvet/insignia.git insignia-s10-fp8-codec
cd insignia-s10-fp8-codec
git fetch origin
git rev-parse --verify e48f633^{commit}
git switch --create codex/s10-exact-fp8-residency e48f633
git status --short
```

Expected HEAD is `e48f633`. Commit `78e1a1c` already completed H8 cross-head
FP8 MLA decode, and `e48f633` completed fused H4×Q8 FP8 MLA prefill. Do not
independently rebuild either completed path.

Read these tracked files first:

- `AGENTS.md`
- `progress.md`
- `audits/s9-reclaim-session.md`
- `audits/s8-gpu-expand-session.md`
- `src/glm53_q8_index.cpp`
- `src/glm53_q8.cu`
- `src/glm53_generate.cu`
- `src/glm53_dflash2.cu`
- `tools/quantize_glm53_q8.py`
- `tools/quantize_dflash2.py`
- `tools/test_e4m3fn.py`

Useful discovery commands:

```bash
git grep -n -E "IGLMF8A1|Q8Stager|q8_budget|e4m3|group.?64|DFLASH2_FP8" -- \
  src include tools
git show --stat e48f633
git log --oneline --decorate -20 e48f633
```

Reports are untrusted evidence. Confirm layout from the index reader and
quantizer before parsing bytes.

## Current architecture and measured constants

- The dense cache contains 699 E4M3 group-64 matrices and is 8.13 GiB. Its
  measured model-wide cosine is 0.9994. The current file/index magic is
  `IGLMF8A1`; verify the complete on-disk structure in source.
- Each group contains 64 one-byte E4M3 values and an FP16 scale. Scale overhead
  is about 3.125% before index/alignment. The prior Problem 4 already owns scale
  coding; do not spend this task redesigning scale streams.
- The FP8 tensor-core dense GEMV measured 24.8 microseconds versus 91.9
  microseconds for BF16 and about 698 GB/s effective bandwidth. A decoder that
  materializes a second full FP8 matrix in global memory will almost certainly
  lose; reconstruction must feed the MMA tile directly through registers or
  shared memory.
- The Ada target is `sm_89`, approximately 800 GB/s observed memory bandwidth,
  with FP8 tensor cores but no Blackwell FP4 MMA. Integer/bit operations and
  tensor compute are underused during a weight-bandwidth-bound GEMV.
- The GPU has 16,376 MiB. Any bytes truly removed from resident/staging budgets
  can be reassigned to dense residency or roughly 13.5 MiB expanded expert
  slots. A nominal 10% of 8.13 GiB is about 0.8 GiB, roughly 60 slots, but the
  final report must prove that the residency allocator can actually use it.
- Real-prompt end-to-end medians around session 9 were approximately 500.4
  ms/token scalar and 539.6 ms/token DFlash2 on its A/B sample. WSL variance is
  large; use paired repeated medians.

## Required non-Git byte sample

The entropy decision requires real bytes. Request or create
`fp8-residency-sample-v1.tar.zst` containing only non-Git model data:

```text
manifest.tsv
dense/<tensor-id>.weights.e4m3.bin
dense/<tensor-id>.scales.f16le.bin
dflash/<tensor-id>.weights.e4m3.bin        # optional first pass
dflash/<tensor-id>.scales.f16le.bin        # optional first pass
SHA256SUMS
```

`manifest.tsv` must contain:

```text
source_kind tensor_name rows cols group_size source_index_sha256
sample_row_begin sample_rows weight_file weight_bytes weight_sha256
scale_file scale_bytes scale_sha256 sampling_reason
```

Sampling requirements:

- use complete, row-aligned slices, never arbitrary byte windows;
- cover every major matrix family and early/middle/late layers;
- include complete small matrices and stratified slices of large matrices;
- include at least 256 MiB of dense weight bytes, preferably 512 MiB;
- preserve the source tensor name and shape but include no checkpoint content
  other than the declared sample;
- independently hash every file and the archive. Split an archive larger than
  512 MiB into independently hashed 512 MiB parts.

Verify `SHA256SUMS`, file sizes, row geometry, and group count before analysis.
If the sample does not exist, the first deliverable is a dependency-light
`tools/sample_fp8_cache.py` and a one-command owner-box collection recipe. Do
not copy any tracked source into the archive and do not infer compression ratios
from synthetic Gaussian bytes.

## Mathematical formulation

For each tensor family and fixed tile size `T` in `{128, 256, 512, 1024}`
bytes, measure:

```text
H0 = -sum_v p(v) log2 p(v)                 # byte-symbol entropy
D_T = number of distinct byte values/tile
Z_T = zero and signed-zero frequency/tile
R = (payload + headers + exceptions + padding) / raw_bytes
```

`H0` is only a lower bound. The decision metric is the **actual padded random-
access ratio** `R`, including tile headers, exception tables, alignment, and
the index needed to address any matrix row/tensor-core tile without decoding a
long prefix.

Evaluate a small set of GPU-decodable exact formats, for example:

- per-tile palette plus 4/5/6-bit indices and literal exceptions;
- E4M3 sign/exponent/mantissa bitplanes with fixed-width exception lanes;
- base/XOR residuals or zero/sign-specialized modes;
- a raw-tile escape when compression is not profitable.

Avoid serial Huffman/ANS unless a concrete massively parallel random-access
decoder beats the fixed-block alternatives. Per-tile mode selection is allowed
if the mode word and branch/warp behavior are included in the model.

For a fused consumer, use the first-order bound

```text
T_raw  ~= B_raw / BW_mem
T_new  ~= max(B_comp / BW_mem, Ops_decode / Throughput_decode)
         + T_metadata + T_sync + T_occupancy_loss.
```

Use measured kernel bandwidth and measured decoder throughput in the final
model. A byte ratio alone is not a speed result.

## Ordered experiment plan

1. **Validate and inventory the sample.** Reconstruct tensor slices using the
   repository's exact E4M3/group-64 interpretation. Produce frequency,
   distinct-count, zero/special-value, bitplane, and adjacent-byte correlation
   tables by matrix family. Completion evidence: hashes plus a machine-readable
   CSV/JSON inventory.
2. **Establish the coding ceiling.** Compute entropy lower bounds and actual
   padded sizes for the candidate fixed-block formats. Use a train/held-out
   split by tensor, not random bytes from the same tensor. Report weighted
   whole-cache estimates with bootstrap intervals. Kill weak formats here.
3. **Build an exact CPU reference codec.** It must support independently
   addressable tiles, raw escapes, malformed-input rejection, and deterministic
   encoding. Fuzz random and adversarial E4M3 byte patterns. Decode must match
   the original SHA-256 byte for byte, including signed zero, NaNs, and every
   otherwise unusual bit pattern even if the sample lacks it.
4. **Write the GPU roofline/design note.** Map a compressed tile to the current
   FP8 MMA tile. Account for loads, bit extraction, shuffles, shared memory,
   registers, barriers, occupancy, and alignment. Explain exactly where decoded
   bytes live and prove that no full global-memory expansion occurs.
5. **Microbenchmark before integration.** Add a standalone CUDA test/benchmark
   that compares raw FP8 MMA consumption with fused exact decode plus the same
   MMA on representative shapes from each matrix family. Validate output bits,
   disassemble the hot kernel, report registers/spills/occupancy, and sweep hot
   and cold cache conditions.
6. **Integrate behind an opt-in format/version gate.** Start with the matrix
   family having the strongest held-out ratio. Preserve the current `IGLMF8A1`
   reader as the default. Ensure allocation/residency accounting uses compressed
   bytes and explicitly measures the expert slots or Q8 residency gained.
7. **Run paired end-to-end tests.** Alternate exact baseline/candidate runs at
   least five times per arm. Report matrix time, total dense time, decode
   ms/token, actual disk/H2D/VRAM bytes, DFlash acceptance, slot counts, and WSL
   dispersion.

Only after the exact path is decided may the agent explore a separate, clearly
labeled lossy mode that reconstructs a restricted FP8 palette or FP6-like code.
That mode needs its own sensitivity sweep and quality gates; it must never be
described as exact.

## Deliverables

- A sample collector/validator and documented manifest schema if no valid
  sample was supplied.
- Machine-readable entropy, padded-ratio, and held-out generalization results.
- A deterministic CPU reference codec with fuzz/property tests.
- A GPU roofline and tile-mapping note with disassembly/resource accounting.
- A standalone CUDA microbenchmark and raw result files.
- If gates pass: an opt-in versioned cache encoder/reader and fused MMA consumer,
  plus exact parity tests and paired end-to-end measurements.
- A final report stating bytes saved by matrix family, bytes actually reclaimed
  by the allocator, expert slots/residency gained, speed, quality, failures, all
  commands, commit IDs, and data SHA-256 values.

## Exactness, quality, and performance gates

For the lossless path:

- every decoded weight and scale byte must match its source SHA-256;
- malformed or truncated tiles must fail safely in the offline validator;
- digit-identical top-10 logits, greedy IDs, and 30/40/100/240-token sequences;
- no change to accumulation or expert order;
- weighted held-out padded ratio `R <= 0.90`, with at least one high-byte-volume
  family at `R <= 0.85`;
- fused matrix consumption at least 1.08x faster on the winning families, or a
  proven allocator benefit of at least 384 MiB/28 expanded expert slots without
  more than a 2% kernel regression;
- at least 3% paired whole-engine decode improvement before enabling it by
  default is considered.

For any optional lossy path, report at least next-token perplexity, logit KL and
cosine, greedy agreement, output entropy/repetition measures, and held-out
GSM8K/MATH-500 score. Require no NaNs or repetition collapse, perplexity no
worse than 1.03x for a conservative point, and no more than 2 absolute points
of task-score loss. An aggressive point may relax those bounds only if it is
explicitly labeled and reaches at least a 15% end-to-end speed improvement.

## Kill criteria

Stop before CUDA implementation if any is true:

- held-out byte entropy exceeds 7.6 bits/byte and the best realistic padded
  fixed-block ratio is worse than 0.94;
- the ratio depends on a global dictionary or long serial prefix that prevents
  independent MMA-tile access;
- the apparent ratio comes mainly from scale bytes, which belong to prior
  Problem 4;
- the allocator cannot use the saved representation to increase useful
  residency or reduce transfers.

Kill or revert integration if exact decode slows the winning matrices by more
than 2%, causes occupancy/spill collapse, materializes a full decoded matrix in
global memory, fails byte/parity tests, or cannot repeat in paired medians.

## Forbidden duplication and scope

This task owns full E4M3 dense/drafter weight-byte compression and fused exact
consumption only. The complete exclusion list is:

- full-prompt layer-major prefill;
- compact exact MLA absorb and exact MLA-prefix reconstruction;
- completed H8 cross-head FP8 decode (`78e1a1c`) and completed fused H4×Q8
  cross-head FP8 prefill (`e48f633`);
- low-risk VRAM reclaim and sequential speculative-snapshot elision;
- rejected KDA persistent-state Task 5, exact-INT8 Task 7, causal expert
  predictor Task 8, and staged-verification Task 9;
- the prior Problems 1--7: DFlash controller, Belady cache oracle, campaign
  statistics, expert-scale codec, cross-expert dictionary/body compression,
  CPU bypass roofline, and reader scheduling;
- any edit to ignored reference clones.

In particular, do not claim the already assigned scale stream or NVFP4 expert
body as part of this task. The target is complete E4M3 matrix-weight bytes used
by the dense and, secondarily, DFlash2 tensor-core paths.
