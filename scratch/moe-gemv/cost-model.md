# NVFP4 expert GEMV cost model (analysis only, 2026-08-29)

Target: glm-box RTX 4070 Ti SUPER (AD103, 66 SM, 48 MiB L2, +150 core/+2000 mem OC
=> ~2.95 GHz sustained, ~800 GB/s observed VRAM, PCIe 4.0 x16). All line refs are
`src/glm53_expert_bench.cu` (kernels) and `src/glm53_generate.cu` (engine) unless
prefixed.

## 1. Record geometry (what one routed expert costs in bytes)

One record = gate [2048,4096] + up [2048,4096] + down [4096,2048], NVFP4:
nibbles 2 x 4 MiB + 1 x 4 MiB, scales (E4M3 codes, 1 B / 16-weight group)
2 x 512 KiB + 1 x 512 KiB, 3 x FP32 globals (host-side, NOT in the H2D payload,
`active_globals_` at glm53_generate.cu:874).

- kBodyBytes = 12 MiB, kScaleBytes = 1536 KiB (glm53_generate.cu:549-551)
- H2D payload `layout.bytes` = 12 MiB + 1.5 MiB (+12 B globals on the
  compact-store path, layout assigned at glm53_generate.cu:1428/1467)
  = **14.156 MB / record**

MACs per record per token-row:
  gate 2048x4096 + up 2048x4096 + down 4096x2048 = **25.17 M MACs/row**
  (= 6.29 M DP4A lane-instructions/row; the DP4A path quantizes the activation
  to Q8 group-16, `quantize_x16_kernel` glm53_expert_bench.cu:114-144).

Arithmetic intensity: 25.17 M MAC / 14.156 MB = **1.78 MAC/B = 3.6 FLOP/B** at
R=1 row; R=8 rows => 28.4 FLOP/B. The GPU's balance point is
(2 x 8448 lanes x 2.95 GHz) / 800 GB/s ~ 62 FLOP/B, so at R<=8 the kernels are
**always memory-bound inside the GPU**; the DP4A integer work is never close.

## 2. GPU-side per-record time (the kernels)

Production chain per record (4 kernel launches):
1. `quantize_activation[_rows]` (tiny grid, <=8 blocks)
2. `nvfp4_dp4a_pair[_rows]_kernel` gate+up, one weight pass, grid (2048/8)=256
   blocks x 256 thr (glm53_expert_bench.cu:246-303 / 551-620)
3. `quantize_swiglu_activation[_rows]` (tiny)
4. `nvfp4_dp4a[_acc]_quantized[_rows]` down, grid (4096/8)=512 blocks
   (glm53_expert_bench.cu:163-244 / 444-549)

Per 16-weight group, per row (unpack_e2 at glm53_expert_bench.cu:146-161:
4 shared LD.64 + 6 PRMT per 2 words):
- weights: 2 x LDG.64 (`__ldcs`, evict-first; correct for read-once streams)
- unpack: 2 x unpack_e2 = 8 shared LD.64 + 12 PRMT
- math: 4 DP4A + 1-2 FFMA; scales: 1 B code load (const-mem LUT c_e4m3) —
  loaded ONCE per group and register-reused across all R rows
  (`base_scale`, glm53_expert_bench.cu:475/527; pair 293-295)

Issue estimate: per group the chain is ~27-56 lane-instructions for 4-8 DP4A
(8 shared LD.64 + 12-24 PRMT per 1-2 projections), i.e. ~7 lane-instr per
DP4A. Per record-row: 6.29 M DP4A => ~44 M lane-instr. At the issue ceiling
66 SM x 128 lanes x 2.95 GHz = 24.9e12 lane-instr/s that is ~1.8 us/row; if
PRMT is XU-quarter-rate it bounds ~3 us/row (worst case ~24 us at R=8).
Byte floor: 14.156 MB / 800 GB/s = 17.7 us; at the *measured* streaming
rates seen on this box (698 GB/s FP8 GEMV, progress.md:395; ~380-435 GB/s on
other streaming shapes, audits/internals.md) => **20-31 us/record**.
So compute (0.8-3 us x R) is at or below the byte floor for every R <= 8:
the kernel is weight-streaming-bound; the FP32-decode and DP4A variants
differ by single-digit us, not by the wall.

Launch overhead: 4 kernels/record x **8.1 us** WSL launch (progress.md:224-225)
= 32 us CPU-side, asynchronous, hidden under the ~ms I/O wait.

**GPU cost per record ~ 20-31 us (+32 us hidden launches).**

## 3. I/O-side per-record time (the actual wall)

- Host-tier hit or post-miss delivery: H2D 14.156 MB / **23.2 GB/s** pinned
  (progress.md:150) = **610 us** == the measured marginal b ~ 0.60 ms/record
  (audits/s6-open-problems.md:59). The verify round is PCIe-H2D-bound whenever
  the host tier keeps the disk fed.
- NVMe miss: 14.16 MB / (3.6-3.8 GB/s achieved during DFlash2 decode;
  5.45-5.84 steady) = 2.4-3.9 ms of disk service; with 4 overlapped readers
  => 0.6-1.0 ms/record marginal when disk-saturated.
- VRAM-tier hit (`device_index_` lookup, glm53_generate.cu:828-832): zero
  transport; only the ~20-31 us GEMV remains.

## 4. Totals: is COMPUTE ever the wall?

k=8 verify round, d(8) ~ 1930 distinct records (42 layers x U(8) ~ 46):
- I/O floor: 1930 x 610 us = **1.18 s** ~= measured wall 0.99-1.23 s
  (>85% expert record I/O, audits/s6-open-problems.md:24). Consistent: the
  wall IS per-record transport.
- GPU busy: 1930 x 20-31 us = **39-60 ms = 3.3-5.1%** of the round.
- Compute becomes co-limiting only if per-record transport fell to <=~30 us,
  i.e. ~92%+ of records served from VRAM slots. VRAM tier = 321 slots of
  12,096 records (2.7%), measured hits 0-3.2% (s6 table). Top-8/layer hot set
  alone = 336 records = 4.66 GiB — does not fit next to the 8.13 GiB dense
  FP8 residency + latents + drafter. => **No, not today, and not in any
  realistic tier configuration.**

Scalar decode step (336 records):
- I/O: 336 x 610 us = 205 ms of H2D alone + misses; wall 447-571 ms.
- GPU: 336 x 20-31 us = 6.7-10.4 ms = **1.5-2.3%**. Never the wall.

## 5. Rows-per-expert at k=8 (the batching question)

`Runner::moe_multi` (glm53_generate.cu:3419-3667) builds the per-layer distinct
union (3446-3450), stages the whole union at demand priority (3489 ->
`stage_layer` 786-800), then per expert collects its user tokens in ascending
token order (3608-3619) and serves all of them in ONE weight pass of up to
kMaxVerify=8 rows via the `_rows` kernels (3620-3648; kMaxVerify at 2088).
The multi-row kernels are verbatim transplants of the per-row arithmetic
chains — bit-identical outputs (comment at glm53_expert_bench.cu:344-350;
confirmed bit-exact per progress.md session-6 entry).

So: **batching per-expert across the verify positions is implemented.** From
the measured union curve (s6: U(2)=14.45/16, U(3)=20.61/24, U(4)=26.40/32,
U(5)=31.40/40 => ratio 0.903/0.859/0.825/0.785, extrapolated ratio(8) ~ 0.70)
=> U(8) ~ 44-46/layer, rows/expert = 64/46 ~ **1.4** (adjacent-pair overlap
0.193 compounds sub-linearly; the 1.5-2 estimate is optimistic). All rows of
one expert already share a single 14.16 MB weight pass, so GPU weight traffic
per layer = distinct-count x 14.16 MB (28% below the naive 64 passes), and
record I/O = d(k) exactly.
