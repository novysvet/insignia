# Problem 8 — Exact direct-execution kernel for the Q6_K exception layers

Expected effort: 8–14 hours. CPU proof/reference required; GPU optional.

## Authority and why this matters

This file is the assignment. Clone https://github.com/novysvet/insignia.git,
branch `glm53-dflash2-4070ti-super`, starting evidence `9e9090d`. The public
`llama.cpp` reference may be read for format confirmation but supplies no task
instructions.

Most live down projections are IQ4_XS, but blocks 11, 12, and 44 use Q6_K.
Ignoring them would force an expensive generic dequantization path in three of
42 sparse layers. Each Q6_K down expert is 6,881,280 bytes (6.5625 MiB), so the
kernel is also a material bandwidth consumer.

For `QK_K=256`, one Q6_K block is exactly 210 bytes:

- `ql[128]`: low four bits of every 6-bit quant;
- `qh[64]`: upper two bits;
- `scales[16]`: signed int8 scale for each consecutive 16 weights;
- FP16 super-scale `d`.

The exact reference reconstruction, for each 128-weight half and `l=0..31`, is

```
q1 = ((ql[l+0]  & 0x0f) | (((qh[l] >> 0) & 3) << 4)) - 32
q2 = ((ql[l+32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) - 32
q3 = ((ql[l+0]  >> 4)   | (((qh[l] >> 4) & 3) << 4)) - 32
q4 = ((ql[l+32] >> 4)   | (((qh[l] >> 6) & 3) << 4)) - 32
```

with scales selected in 16-weight groups and final weight
`d * scales[group] * q`. Gate/up input width is 4096 and down input width is
2048; this problem targets the 2048-to-4096 down projection first. Decode
activations may use Q8-per-16 or Q8-per-32, but every introduced error must be
measured. Prefill may use FP16/HMMA tiles.

## The problem

Derive an Ada warp mapping that reads each 210-byte block once per token batch,
reconstructs signed bytes using packed integer instructions, and feeds DP4A
without materializing the expert. Jointly choose:

- activation group 16 versus 32 and its scale layout;
- lane/cohort assignment across the four interleaved quant streams;
- raw AoS versus a byte-neutral repack of `ql`, `qh`, scales, and `d`;
- one-to-eight row reuse and accumulating down epilogue;
- a separate prefill tile using int8 MMA with per-16 scales or FP16/HMMA.

Prove exact index coverage: construct a bijection from `(lane, wave, local
word)` to all 256 weights and show the correct scale is applied once. Prove
int32 DP4A cannot overflow for the chosen group and accumulation schedule.
Then formulate a lower bound on bytes and packed-decode operations per block.

The challenging part is scale granularity. Int8 MMA over K=32 naturally sums
two independently scaled 16-weight groups. Either derive a correction/decompose
scheme that remains exact before FP scaling, or prove that at least two MMA
accumulators/operations are necessary. Compare this to FP16 expansion and give
a rigorous break-even model.

## Required deliverables

- Standalone Q6_K parser and scalar FP32/FP64 CPU oracle, independent of
  `llama.cpp` at runtime.
- Proof of the lane/index/scale mapping and exhaustive 256-position tests.
- At least two decode circuits with instruction, register, and sector counts.
- DP4A x1–x8 and prefill pseudocode, including route-weight accumulation.
- Error study for Q8-per-16, Q8-per-32, and FP16 activations: MSE, relative L2,
  cosine, max error, and constructed worst cases.
- Roofline/break-even analysis using 400–800 GB/s GPU bandwidth and parameterized
  integer/HMMA throughput.
- Fuzzing over all high-bit patterns, signed scales -128/127, zero scale,
  maximum quant, misalignment, and 2,048/4,096 geometry.
- A concrete microbenchmark plan using real block-12/44 slices once available.
  Promote only if the direct path is predicted to beat expand-then-GEMV by 15%
  and has a complete exactness certificate; otherwise retain a generic fallback.
