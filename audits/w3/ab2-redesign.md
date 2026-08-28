# W3 / AB2 — in_proj_a+b pair kernels: verdict + designs (9B INSIG4 vs 27B)

Task: decide whether `mxfp4_gemv_ab2_q8_i4_kernel` (src/mxfp4_i4.cu:157, the LIVE
spec-decode pair path via `linear2`/decode.cu:66-70) needs a 48-head/5120-col
redesign for Qwen3.8-27B, and deliver the kernel(s) the 27B actually needs.
Read-only audit; nothing in `src/` was touched. Evidence file:
`audits/w2/loader-27b-spec.md` (INSIDX02 tensor census, headers parsed directly).

## 0. Verdict (short)

**The INSIG4 ab2 redesign is not needed. No model will ever execute it at 48/5120.**

- The 9B is and stays 32 v-heads / hidden 4096 (audits/w2/shape-constants.md — 48
  appears only in the 27B column). The existing kernel is exactly a 32/4096
  specialization and is correct for the only checkpoint that has INSIG4 a/b.
- The 27B's `in_proj_a/b` are **BF16 [48,5120], 491,520 B (480 KiB) each**, listed
  in `modules_to_not_convert` (loader-27b-spec.md §1.3) and present as BF16 in
  every one of the 48 linear-layer shards (§2.3 template, §2.8 per-shard census).
  The F8 census is 407 tensors (§0) — `in_proj_a/b` are not among them. There is
  no quantized a/b to feed an INSIG4 (or FP8) ab2 kernel at any head count.
- Even a hypothetical future INSIG4 requant of the 27B keeps a/b tiny-bf16 (the
  vendor convert list does; so does the 9B quantizer for `mtp.fc` — the engine
  already runs that one through `bf16_gemv`, decode.cu:151). And if someone
  quantized them anyway, the generic per-token path (decode.cu:72-73 →
  `mxfp4_gemv_v2_i4`, rows-generic, cols%1024==0 ✓ 5120) already computes them
  correctly — 4 launches instead of 1, wrong answers never.

So W2 risk-cluster-2 "three coupled redesigns" reduces to: (a) add
**launcher guards** to the three 9B ab2 kernels so a bad call throws instead of
silently half-staging (synthesis bug #5 pattern), and (b) add a **bf16 A+B
pair-2-row GEMV** for the 27B spec path. Both delivered below.

## 1. Evidence table (from loader-27b-spec.md, verified this session)

| fact | value | source |
|---|---|---|
| `linear_attn.in_proj_a.weight` | BF16 **[48, 5120]** 491,520 B | §2.3 template; §2.8 layers-0 off 94,952 |
| `linear_attn.in_proj_b.weight` | BF16 **[48, 5120]** 491,520 B | §2.3; layers-0 off 586,472 |
| modules_to_not_convert | includes `in_proj_a/b` (alongside norms, A_log, dt_bias, conv1d, mtp.fc, embed, lm_head) | §1.3 |
| F8 census | 407 tensors, all big projections; scale shapes [80,40],[48,40],[40,48],[40,136],[136,40],[8,40],[96,40] — none is 48x40-style for a [48,5120] a/b | §0, §2.7 |
| 27B hidden / v-heads | 5120 / 48 | config §1.2 |
| 9B (INSIG4 ckpt) | a/b INSIG4 [32,4096] ≈ 68 KB each; heads 32 forever | shape-constants §mxfp4_i4.cu |
| pair path caller | decode.cu:66-70, `pair = (T==2)`, x = `pf_n` [2,cols], out `pf_a`/`pf_b` [T][heads] | read this session |

Note the bf16 a/b are also **512-aligned only every 2nd tensor** (off mod 512 = 232
for a, 232 for b in layers-0) — irrelevant here: rows are 10,240 B strides, so every
row start stays 8B-aligned (uint2 loads legal); the loader tier handles the base.

## 2. Workload reality check — why this kernel is a 2 µs problem

Per DeltaNet layer per spec step (27B): 4 GEMVs (a@x0, a@x1, b@x0, b@x1), each
48x5120. Weight bytes: 2 x 491,520 = **983,040 B ≈ 0.94 MB** — ~2% of the 4070S's
~50 MB L2, and < 1/50 of a layer's 373 MB F8 stream. FLOPs: 983,040 FMA.
DRAM floor if fully cold: 0.94 MB / 504 GB/s ≈ **1.9 µs**; warm-L2 traffic floor
~0.4 µs. That is the same order as one graph-node overhead (~0.5-1.5 µs) — i.e.
**the kernel is latency/launch-bound, and the only decisions that matter are
(1) one launch instead of four, and (2) enough blocks to spread the tail.**
Grid: **96 blocks = 48 A-rows + 48 B-rows, 256 threads, one block reduces one row
over 5120 cols (20 cols/thread)** → 75% SM coverage on 128 SMs, no cross-block
reduction, deterministic. Rejected: 12 blocks x 8 warps (9% SM fill); 192-block
split-K + atomicAdd (nondeterministic sum order breaks reproducible parity
reruns — AGENTS.md wants benchmark+parity evidence, and float atomics would
make two runs of the same test differ).

## 3. Design B — `bf16_gemv_ab2_pair` (the kernel the 27B needs)

Deliberate properties, in project style:

- **One launch per DeltaNet layer replaces 4 GEMVs** (the same win the 9B i4 ab2
  kernel already banks: 4 -> 1 nodes in `capture_spec`).
- **No dynamic smem, one `__syncthreads()`** at the cross-warp reduction, with the
  shared arrays written strictly before and read strictly after — immune to the
  read-then-overwrite race class found in ops.cu (synthesis bug #2). x needs no
  staging at all: within a block each x element is consumed by exactly one thread,
  once, in registers.
- **bf16 -> f32 by bit surgery, not conversion instructions**: a bf16 is the high
  half of an f32, so lane k of a u32 is `__uint_as_float(u & 0xffff0000u)` and
  lane k+1 is `__uint_as_float(u << 16)` — two of four weights per u32 are pure
  AND-masks. Exact (no rounding), per AGENTS.md "bit manipulate floats".
- **fp32 accumulation** over 5120 terms — same convention as every GEMV here, and
  what the NumPy parity harness expects (bf16 accumulation over 5120 terms would
  be visible in cosine).
- Vector loads only: per 4 cols, one `uint2` (8 B) of weights + one `float4` of
  each activation row. Coalesced; weights stream through L2 (`__ldcs`), x is L1-
  hot (`__ldg`) since `pf_n` was just written by rmsnorm.
- Launcher **throws** on any dims it cannot honor (synthesis bug #5: the existing
  ab2 launchers throw on nothing). Constraint is cols%4==0 (8B/16B alignment);
  5120 ✓, 4096 ✓ — deliberately shape-generic so it also serves any future
  bf16 small tensor pair, while the 27B call site passes 48/5120.

Fits in `src/fp8.cu` next to `fp8_gemv2` (same file, same include set, the 27B
kernel home; `bf16_gemv_rows` is already *declared* in insignia_fp8.cuh:27 and
unimplemented — this is its specialized pair sibling, not a duplicate).

```cpp
// Fused in_proj_a + in_proj_b pair GEMV, bf16 weights (Qwen3.8-27B small tensors):
// wa/wb [heads,cols] bf16, x [2,cols] f32  ->  ya/yb [2,heads] f32 (row-major,
// ya[row] = wa[row]*x0, ya[heads+row] = wa[row]*x1). One launch per DeltaNet
// layer instead of four GEMVs on the spec pair path. 2*heads blocks (A rows then
// B rows), 256 threads; one block reduces one row, fp32 accum, deterministic
// tree order. bf16->f32 is exact bit surgery: a bf16 IS the high half of an f32.
__global__ __launch_bounds__(256) void bf16_gemv_ab2_pair_kernel(const uint16_t *__restrict__ wa, const uint16_t *__restrict__ wb, const float *__restrict__ x, float *__restrict__ ya, float *__restrict__ yb, int heads, int cols) {
    const bool is_a = blockIdx.x < heads;
    const int row = is_a ? int(blockIdx.x) : int(blockIdx.x) - heads;
    const uint16_t *__restrict__ row_w = (is_a ? wa : wb) + static_cast<size_t>(row) * cols;
    const float *__restrict__ x0 = x, *__restrict__ x1 = x + cols;
    float acc0 = 0.f, acc1 = 0.f;
    #pragma unroll 2
    for (int c = threadIdx.x * 4; c < cols; c += 1024) {   // 5120 cols -> exactly 5 rounds
        const uint2 p = __ldcs(reinterpret_cast<const uint2 *>(row_w + c));   // 4 bf16
        const float4 v0 = __ldg(reinterpret_cast<const float4 *>(x0 + c));
        const float4 v1 = __ldg(reinterpret_cast<const float4 *>(x1 + c));
        const float2 w01 = make_float2(__uint_as_float(p.x << 16), __uint_as_float(p.x & 0xffff0000u));
        const float2 w23 = make_float2(__uint_as_float(p.y << 16), __uint_as_float(p.y & 0xffff0000u));
        acc0 = fmaf(w01.x, v0.x, acc0); acc0 = fmaf(w01.y, v0.y, acc0);
        acc0 = fmaf(w23.x, v0.z, acc0); acc0 = fmaf(w23.y, v0.w, acc0);
        acc1 = fmaf(w01.x, v1.x, acc1); acc1 = fmaf(w01.y, v1.y, acc1);
        acc1 = fmaf(w23.x, v1.z, acc1); acc1 = fmaf(w23.y, v1.w, acc1);
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    #pragma unroll
    for (int m = 16; m; m >>= 1) { acc0 += __shfl_xor_sync(0xffffffff, acc0, m); acc1 += __shfl_xor_sync(0xffffffff, acc1, m); }
    __shared__ float red[2][8];   // written by all warps, read only after the fence
    if (!lane) { red[0][warp] = acc0; red[1][warp] = acc1; }
    __syncthreads();
    if (!warp) {
        float s0 = lane < 8 ? red[0][lane] : 0.f, s1 = lane < 8 ? red[1][lane] : 0.f;
        #pragma unroll
        for (int m = 4; m; m >>= 1) { s0 += __shfl_xor_sync(0xffffffff, s0, m); s1 += __shfl_xor_sync(0xffffffff, s1, m); }
        if (!lane) {
            float *__restrict__ y = is_a ? ya : yb;
            y[row] = s0;
            y[heads + row] = s1;
        }
    }
}
void bf16_gemv_ab2_pair(const uint16_t *wa, const uint16_t *wb, const float *x, float *ya, float *yb, int heads, int cols, cudaStream_t stream) {
    if (heads <= 0 || cols <= 0 || (cols & 3)) throw std::runtime_error("insignia: bad ab2 dims heads=" + std::to_string(heads) + " cols=" + std::to_string(cols));
    bf16_gemv_ab2_pair_kernel<<<2 * heads, 256, 0, stream>>>(wa, wb, x, ya, yb, heads, cols);
}
```

Header line (include/insignia_fp8.cuh, next to `bf16_gemv_rows`):

```cpp
void bf16_gemv_ab2_pair(const uint16_t *wa, const uint16_t *wb, const float *x /*[2,cols]*/, float *ya /*[2,heads]*/, float *yb /*[2,heads]*/, int heads, int cols, cudaStream_t stream = nullptr);
```

Call site (decode.cu:66-70, 27B branch — after the loader exposes a bf16 kind on
`QuantMatrix`, which is INSIDX02 loader work, not kernel work):

```cpp
if (pair) { /* fp8_gemv2 in_proj_qkv / in_proj_z as elsewhere */
    auto ma = w_.matrix(a + ".in_proj_a"), mb = w_.matrix(a + ".in_proj_b");
    bf16_gemv_ab2_pair((const uint16_t *)ma.weight.data, (const uint16_t *)mb.weight.data,
                       x_.pf_n, x_.pf_a, x_.pf_b, ma.rows /*48*/, ma.cols /*5120*/, x_.stream);
    w_.release(a + ".in_proj_a"); w_.release(a + ".in_proj_b");
}
```

Output layout matches the consumer exactly: `deltanet_params_batch` reads
`pf_a`/`pf_b` as [T][heads] (decode.cu:78), i.e. `ya[h]`, `ya[heads+h]` — the same
convention the i4 ab2 kernel uses with 32. `pf_a`/`pf_b` sizing 64x32 -> 64x48 is
already on the W2 decode.cu alloc list (cluster 3). Because the launch config
changes, `capture_spec`/`capture_step` must be re-captured (standing W2 rule).

Expected cost: ~1.5-2.5 µs warm (96 blocks, 0.94 MB L2-resident, latency-bound);
cold-DRAM floor 1.9 µs. Compulsory traffic is read exactly once — there is no
bandwidth left to optimize, only launches, and this design has the minimum: one.

## 4. Fusing a+b into the in_proj_qkv GEMV pass — quantified, then rejected

Data flow allows it: qkv, z, a, b all consume the same `pf_n`. The 27B qkv pair
GEMV is `fp8_gemv2` on e4m3 [10240,5120] + bf16 128x128 block scales; a fused
kernel would extend the grid by 12 blocks (96 bf16 rows / 8 warps-per-row blocks,
block-uniform branch, `if (blockIdx.x >= qkv_blocks)`). What it saves vs the
standalone pair kernel above is **one graph node per DeltaNet layer**:

| scenario | saved per layer | per spec step (48 delta layers) | fraction of step |
|---|---|---|---|
| graph replay (the live path, decode.cu:232) | node ~0.5-1.5 µs + tail ~0.5 µs | ~50-100 µs | 0.003-0.013% of 0.78-1.5 s (synthesis feasibility) |
| no graph (spec_step eager) | gap ~2-3 µs + body ~1.5 µs | ~170-215 µs | ~0.02% |
| hypothetical all-VRAM 27B (~54 ms/step) | same ~1.5-2 µs | ~75-100 µs | ~0.15-0.2% |

The 27B on this rig is an I/O problem (synthesis: "GPU compute never binds
decode"); even the best-case number is under the run-to-run noise of a step. The
price is real: a second weight dtype, scale-less path and second output layout
inside the single hottest, correctness-critical kernel of the 27B decode path —
the kernel every one of the 48 layers x 2 rows flows through, and the one the
NumPy parity work will lean on. Under AGENTS.md conventions an optimization needs
a measurement to justify it; this one has a ceiling below measurability.
**Verdict: standalone `bf16_gemv_ab2_pair` (design B) — yes (it banks the only
launch win that exists: 4 GEMVs -> 1 node). qkv fusion — no.** Revisit only if the
27B ever runs all-VRAM (never, at 25.65 GB on 12 GB) or if the engine moves to
deep-verify T=3-4 where node counts per layer grow.

For calibration, the 9B's existing fused ab2 bought the same class of win:
24 delta layers x 3 launches saved x ~2.5 µs ≈ 180 µs per pair step out of tens of
ms — already marginal, already paid for. That is the ceiling of this whole idea
family; do not spend 27B correctness surface chasing the remainder.

## 5. Design C — CPU tier twin (a/b resident on host)

2 x 480 KiB bf16 in host RAM; 983,040 FMA per layer per spec step. Scalar that is
~0.17 ms/layer (x48 layers ≈ 8 ms/step — 1st-order cost next to the 9.6 ms/layer
DRAM budget); AVX2 (Zen 3: 2x FMA ports, 8-wide) makes it ~20 µs/layer ≈ 1 ms/step
spread over the whole model, i.e. free. Keep weights **bf16 in RAM** (2 x 480 KiB,
don't pre-expand: bf16->f32 on Zen 3 is one `vpmovzxwd` + `vpslld 16` — the same
"bf16 is the high half of f32" trick as the GPU kernel; expanding at load would
double resident bytes for zero speed). Compile the CPU-tier TU with `/arch:AVX2`
(5600X is Zen 3; MSVC does not define `__AVX2__` without it — the guard keeps the
fallback honest). Rows are embarrassingly parallel — slice the 96 row-dots across
the worker pool the CPU tier already has (colibri-style SPMC), no atomics needed,
deterministic per-row order preserved for parity.

```cpp
// CPU tier: wa/wb [heads,cols] bf16 (raw checkpoint bytes), x [2,cols] f32,
// ya/yb [2,heads] f32 out. bf16 -> f32 is a 16-bit shift left into the high half.
#include <immintrin.h>
void bf16_gemv_ab2_pair_cpu(const uint16_t *wa, const uint16_t *wb, const float *x, float *ya, float *yb, int heads, int cols) {
    if (heads <= 0 || cols <= 0 || (cols & 7)) throw std::runtime_error("bad ab2 cpu dims");
    for (int m = 0; m < 2; m++) {
        const uint16_t *w = m ? wb : wa;
        float *y = m ? yb : ya;
        for (int r = 0; r < heads; r++) {
            const uint16_t *row = w + size_t(r) * cols;
#if defined(__AVX2__)
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            for (int c = 0; c < cols; c += 8) {
                const __m256i u = _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(row + c))), 16);
                const __m256 wv = _mm256_castsi256_ps(u);            // exact bf16 -> f32
                a0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + c), a0);
                a1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + cols + c), a1);
            }
            const __m128 h0 = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0, 1));
            const __m128 h1 = _mm_add_ps(_mm256_castps256_ps128(a1), _mm256_extractf128_ps(a1, 1));
            __m128 s0 = _mm_add_ps(h0, _mm_movehl_ps(h0, h0)); s0 = _mm_add_ss(s0, _mm_shuffle_ps(s0, s0, 1));
            __m128 s1 = _mm_add_ps(h1, _mm_movehl_ps(h1, h1)); s1 = _mm_add_ss(s1, _mm_shuffle_ps(s1, s1, 1));
            y[r] = _mm_cvtss_f32(s0); y[heads + r] = _mm_cvtss_f32(s1);
#else
            float s0 = 0.f, s1 = 0.f;
            for (int c = 0; c < cols; c++) {
                const float wv;
                const uint32_t b = uint32_t(row[c]) << 16; memcpy(&wv, &b, 4);
                s0 = fmaf(wv, x[c], s0); s1 = fmaf(wv, x[cols + c], s1);
            }
            y[r] = s0; y[heads + r] = s1;
#endif
        }
    }
}
```

(Same function answers the "expand at load?" question: no expansion, shift on the
fly — the u16<<16 op is cheaper than the extra 480 KiB of RAM traffic per pass.)

## 6. Correctness test sketch (src/test_bf16_ab2.cu, house style of test_fp8.cu)

```
wmain:
  rng mt19937(777); normal nd(0, 0.05);
  heads=48, cols=5120  (also run heads=32/cols=4096 to prove shape-genericity)
  wa/wb: u16 via f32_to_bf16_bits (copy helper from test_fp8.cu:33),
         wref doubles via bf16_host(bits)  -> reference is EXACT for bf16 inputs
  x[2*cols] f32 from nd
  dev alloc/copy; bf16_gemv_ab2_pair(dw, dwb, dx, dya, dyb, heads, cols);
  sync + error string check; copy back
  per tensor ya/yb and per activation row k: cosine vs double dot
     expect cos > 1 - 1e-6 (fp32 accumulation of 5120 terms vs double;
      the e4m3 kernels print ~0.999999x, bf16 inputs must be tighter since
      weights are exact — only accumulation rounds)
  ALSO negative tests (the point of the launcher):
     cols=5122 -> must throw; heads=0 -> must throw
  ALSO CPU twin vs the same double reference: bitwise-identical across runs
     (deterministic), max-abs-diff reported
  ALSO row-vs-generic cross-check: ya row r == bf16_gemv(wa, x0) row r by
     construction (calls existing bf16_gemv, qwen_kernels.cu:68)
```

Pass criterion per AGENTS.md: cosine (not max-abs) for the float compare, printed
`printf("bf16_gemv_ab2_pair cos=%.8f\n", ...)` per house convention; a hard
`return 1` on any throw-failure to throw.

## 7. The only change the 9B kernels need — launcher guards

The three 9B ab2 launchers validate nothing today (synthesis bug #5 family);
their kernels hard-require groups==128 (staging `threadIdx.x>>7 / &127`) and 64
concatenated rows (32+32). Since the 27B never routes here, the correct fix is to
make wrong routing loud, not to generalize (specialization is the project's
religion; silent garbage is not). One line per launcher:

```cpp
// mxfp4_i4.cu  mxfp4_gemv_ab2_q8_i4 (and the mxfp4.cu ab2_q8 / ab2_q8g twins):
if (cols != 4096) throw std::runtime_error("insignia: ab2 pair kernel is 9B-specialized "
    "(in_proj_a/b 32x4096 INSIG4), refusing cols=" + std::to_string(cols));
```

(`cols==4096` pins groups==128; the 32/32 row split is a checkpoint property with
no runtime parameter — the caller passes only cols today. If a 48-head INSIG4
model ever materializes, the correct move is design B's shape-generic pattern at
0.5 B/weight, not resurrecting the 64-row concat scheme at 96 rows.)

## 8. Integration checklist (for the implementer, in dependency order)

1. Add `bf16_gemv_ab2_pair` to src/fp8.cu + declaration to include/insignia_fp8.cuh
   (fulfills the neighborhood of the still-unimplemented `bf16_gemv_rows` decl).
2. Guards on the three 9B ab2 launchers (section 7).
3. test_bf16_ab2.cu per section 6; run under the existing build/*.bat path
   (vcvars64 + nvcc -arch=sm_89).
4. 27B decode wiring happens with INSIDX02 loader work: QuantMatrix bf16 kind ->
   decode.cu:66-70 branch; pf_a/pf_b 64x48 (W2 cluster 3); re-capture
   capture_spec/capture_step.
5. CPU twin lands with the CPU tier (needs the AVX2 flag on that TU).

## 9. TL;DR (10 lines)

1. Verdict: INSIG4 ab2 48-head redesign is DEAD — no such tensor exists at 27B.
2. Evidence: loader-27b-spec §1.3/§2.3 — in_proj_a/b are BF16 [48,5120] 480 KiB each,
   in modules_to_not_convert; the 407-tensor F8 census contains none of them.
3. The 9B keeps 32 heads / 4096 cols forever; its kernel is a correct specialization.
4. Even a fantasy INSIG4-27B a/b would fall back to the generic per-token
   mxfp4_gemv_v2_i4 path (rows-generic, 5120%1024==0) — correct, just 4 launches.
5. What the 27B needs: `bf16_gemv_ab2_pair` — one launch computing a@x0, a@x1,
   b@x0, b@1 — full source above, project style, launcher throws on bad dims.
6. Shape: 96 blocks x 256 threads, one block per row, 20 cols/thread, ~1.5-2.5 µs;
   0.94 MB of weights is L2-resident; latency-bound, so launches are the only lever.
7. Kernel hygiene: no dynamic smem, single fence, exact bf16->f32 bit surgery
   (u<<16 / u&0xffff0000), fp32 accum, deterministic tree reduce (parity-safe).
8. qkv fusion quantified and REJECTED: saves ~50-215 µs per 1.5 s spec step
   (<=0.2% even all-VRAM) for a second dtype inside the hottest 27B kernel.
9. CPU twin: AVX2 shift+FMA, ~20 µs/layer, keep weights bf16 in RAM (no expand).
10. Only 9B-side change: cols!=4096 throws in the three ab2 launchers (bug #5 class).
