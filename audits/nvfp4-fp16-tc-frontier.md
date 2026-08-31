# NVFP4 small-M Tensor Core frontier (local Ada, 2026-08-31)

## Verdict

The expanded-scale FP16 Tensor Core construction is numerically sound but is a
large performance regression on the local RTX 4070 SUPER. It is **rejected for
production dispatch**. The implementation remains only in the standalone
`glm53-expert-bench` build; `INSIGNIA_GLM53_NO_MAIN` removes the kernels and
their wrappers completely, and no public header/API surface remains.

The follow-up raw signed INT8 IMMA experiment was also implemented. It
reconstructs the existing 32-lane FP32 reduction tree byte-for-byte and removes
FP16 expansion plus the WMMA fragment round trip. It passes every exactness
gate, but the first row-major B gather is still 1.5-3.3x slower than DP4A at
T=8/16. It also remains benchmark-only.

## Construction tested

For each activation group, the existing signed-Q8 quantizer's integer is
exactly representable in FP16. The signed E2M1 weight code is also exactly
representable in FP16. Consequently, an `m16n16k16` FP16 MMA produces the same
16-term integer dot as four DP4As: the largest possible dot is far below the
FP32 exact-integer limit. Only the later FP32 group accumulation is
reassociated.

The tested CTA computes 16 token rows by 8 output rows:

- eight warps retain coalesced row-major weight reads;
- one 32-group slab is unpacked into FP16 shared memory at a time;
- the upper eight columns of each `m16n16k16` B tile are zero padding;
- each warp processes four of the 32 groups;
- the opaque WMMA accumulator is stored to warp-private shared memory, then
  multiplied by the per-token activation scale and per-output weight scale;
- the scale expression matches the fused gate/up path exactly:
  `((0.5f * xscale) * wscale) * global_scale`.

The benchmark wrapper hard-rejects every column count except the two GLM
expert widths, 2048 and 4096. Counts 1..16 are supported.

## Compiler and machine evidence

Own local build command used CUDA 13.3, `-arch=sm_89 -O3 --use_fast_math`.

- TC GEMM: 48 registers/thread, 46,080 bytes static shared memory, zero spill
  loads/stores.
- FP16 quantizer: 30 registers/thread, zero spills.
- `cuobjdump --dump-sass` shows repeated `HMMA.16816.F32` instructions in the
  experimental kernel.
- A separate `-DINSIGNIA_GLM53_NO_MAIN` object compiled successfully; `nm`
  found zero `nvfp4_tc` symbols.

## Focused performance

Fixture: `/var/lib/insignia/l3e0-gate.ig53`, shape 2048x4096, resident on the
C:-backed WSL VHD. Times are CUDA-event medians of three complete runs after
the final scale-association fix, 500 timed iterations per small-M arm. WSL
timings still varied materially, but every comparison rejects the TC arm.

| Shape/path | Existing DP4A | FP16 TC | Result |
|---|---:|---:|---:|
| T=8 total | 37.491 us | 176.509 us | TC 4.71x slower |
| T=8 matrix kernel only | 34.871 us | 173.158 us | TC 4.97x slower |
| T=16 total (DP4A is two T=8 passes) | 89.070 us | 187.316 us | TC 2.10x slower |
| T=16 matrix kernel only | 91.672 us | 184.064 us | TC 2.01x slower |

The fixture's approximately 4.5 MiB NVFP4 body+scale payload fits in Ada L2, so
this is a hot-resident mechanism test, not a cold-DRAM bandwidth result. That
does not rescue the arm: its reported effective payload rate is only about
25-27 GB/s, showing that staging, synchronization, fragment materialization,
and scaled epilogue work dominate.

## Focused numerical gates

Six noncontiguous-id shapes were exercised: R=1, 2, 4, 8, 9, and 16. Inputs
were deterministic cyclic perturbations of the captured activation rather than
sixteen identical copies. Across the shape sweep, TC versus DP4A had:

- maximum MSE: `3.584e-14`;
- cosine: `1.000000000` at printed precision;
- maximum absolute error: `9.53674e-7`;
- maximum KL: `2.623e-14`;
- maximum JS: `6.369e-15`;
- synthetic softmax-PPL delta: `+0.000000%` at printed precision;
- top-1 mismatch: none.

For R=16, TC versus the direct FP32 NVFP4 oracle had MSE `3.361e-5`, relative
L2 `0.004514`, cosine `0.999989811`, maximum error `0.0301968`, KL
`1.683e-5`, JS `4.207e-6`, synthetic softmax-PPL delta `+0.001683%`, and the
same top-1. DP4A has the same values at printed precision. The softmax-PPL
number is only an expert-output diagnostic; it is explicitly not language-model
PPL. No full-model quality campaign is justified for a kernel that is 2-5x
slower.

## Why extra compute lost

Tensor throughput was not the bottleneck. Each useful group requires packed
E2M1 decode, a global-to-shared activation transpose, a global-to-shared weight
expansion, two CTA barriers per 32-group slab, an MMA that wastes half its N
columns, a full fragment store/reload, and 128 separately scaled FP32 group
contributions. The existing DP4A kernel instead keeps unpack, dot, scale, and
accumulation in registers while streaming the packed row once. The Tensor Core
instruction cannot repay the staging tax at M<=16.

## Strict INT8-TC result

Ada's
`mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` variant was implemented next.
The local read-only ggml reference (`ggml/src/ggml-cuda/mma.cuh`, around its
signed-int8 MMA overload) supplied the operand ABI: four u32 A registers, two
u32 B registers, and four s32 accumulator registers per lane.

The strict kernel does all of the following:

- places each K=16 Q8/E2M1 group in the low half of K=32 and makes the high
  half literal zero;
- loads the row-major A fragment with `ldmatrix.sync.aligned.m8n8.x4.b16`;
- converts each exact int32 group dot to FP32 before applying its original
  group scale;
- templates the two real scale associations separately: fused gate/up uses
  `((0.5*xscale)*wscale)*global`, while single/down uses
  `((0.5*wscale)*global)*xscale`;
- accumulates `l,l+32,...` in increasing order for each logical DP4A lane;
- consumes leaves in depth-first XOR order
  `0,16,8,24,4,20,12,28,...` through five named four-float stack levels, which
  reproduces the legacy `16,8,4,2,1` reduction tree.

### Exactness

On the captured 2048x4096 fixture, noncontiguous row-ID cases R=1,2,4,8,9,16
were all byte-for-byte identical to the single-path DP4A oracle. MSE, maximum
error, KL, and JS were exactly zero; cosine was one; top-1 was unchanged. The
separate fused-pair association template was also byte-for-byte identical to
the DP4A pair oracle (using the same captured matrix in both slots to isolate
the expression order).

SASS contains `IMMA.16832.S8.S8`. The single-association kernel uses 64
registers/thread and the pair-association specialization uses 62; both use
4,096 bytes shared memory and have zero spill loads/stores. The quantizer uses
28 registers and has zero spills.

### Performance verdict

CUDA-event medians of three final runs, 500 iterations each:

| Shape/path | Existing DP4A | strict INT8 IMMA | Result |
|---|---:|---:|---:|
| T=8 total | 36.921 us | 114.743 us | IMMA 3.11x slower |
| T=8 matrix kernel only | 34.980 us | 116.062 us | IMMA 3.32x slower |
| T=16 total (DP4A is two T=8 passes) | 76.885 us | 116.820 us | IMMA 1.52x slower |
| T=16 matrix kernel only | 70.294 us | 113.973 us | IMMA 1.62x slower |

One earlier run had IMMA total at 126.837 us versus an anomalous 138.750 us
DP4A total, but the required repeated medians reject that apparent win.

The remaining bottleneck is the checkpoint layout. One IMMA warp needs eight
output rows at one group, while NVFP4 is row-major. Its B fragment therefore
gathers eight 8-byte segments separated by a full row stride. The current
prototype reaches only about 41 GB/s of useful payload while DP4A, whose lanes
read consecutive groups of one row, reaches about 133 GB/s in the same focused
test. Extra integer Tensor Core throughput cannot compensate for the scattered
B transactions.

The only credible continuation is a coalesced multi-group B staging scheme or
a staging-time layout transform. That must preserve the exact group order and
will need an independent 4096x2048 down fixture as well as the 2048x4096
gate/up fixture. It is deliberately deferred; neither strict IMMA specialization
is connected to generator dispatch.

## SSD validity

An unrelated Git-LFS download was actively writing E: during these tests, so
any expert-streaming or striped-store benchmark would have been invalid. This
fixture was fully loaded from `/var/lib/insignia` on the C:-backed VHD before
CUDA-event timing, and the timed regions perform only GPU work. Therefore the
focused kernel A/B is valid; no conclusion about SSD bandwidth or end-to-end
decode/prefill is drawn.
