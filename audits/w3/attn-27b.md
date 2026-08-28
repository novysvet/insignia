# W3: full-attention kernels at 27B dims — paste-ready code + verdicts

Scope: every full-attention kernel on the 9B→27B path, at 27B dims
(24 q-heads non-pow2, 4 kv-heads, head_dim 256, partial rope 64, theta 1e7,
q_proj [12288] q+gate interleaved per head, GQA group 6). Read-only audit;
code below is paste-ready against the current tree (files read in full:
attention.cu, prefill.cu, qwen_kernels.cu, ops.cu, decode.cu + all 4 headers).

---

## 0. CRITICAL VERIFICATION — `kvh = head>>2` is WRONG at 27B. shape-constants.md is in error.

**Verdict: shape-constants.md lines 169 and 365 claim `head>>2` "stays" / "keeps GQA
`head>>2` valid" at 24 q-heads — this is FALSE, and it contradicts synthesis.md line 34,
which already states the truth: "GQA group = 24/4 = 6 (kernel must map kvh = head/6,
NOT head>>2)."**

The math (verified exhaustively, see table below):

- 9B: 16 q-heads / 4 kv → group 4 → `kvh = head>>2` ∈ {0,1,2,3} ✓
- 27B: 24 q-heads / 4 kv → group 6 → `kvh = head/6` ∈ {0,1,2,3}
  - `head>>2` yields groups of 4, not 6 → **16 of 24 heads get the WRONG kv head**
    (heads 4-5, 8-11, 12-17 all disagree with `h/6`).
  - heads 16..19 → `kvh=4`, heads 20..23 → `kvh=5` — **OUT OF RANGE** for a 4-head
    cache (`kvh ∈ [0,4)`).

What the OOB actually reads (this is the nasty part — no crash, plausible-looking data):
the KV row index is `(t*4 + kvh)*256`. With `kvh=4`: `t*4+4 = (t+1)*4 + 0`, i.e. **the
next token's kv-head-0 row**; `kvh=5` → next token's kv-head-1 row. So heads 16..23
attend to the FUTURE token's keys/values. At the last filled slot (`t = tokens-1 =
max_context-1`), row `ctx*4` and `ctx*4+1` are genuinely past the layer's cache slice —
reads the next attention layer's cache, or one/two rows past the whole allocation for
the last layer. Silent numerical corruption + true OOB at full context.

Confirmed in source today:
- `E:\coding\Insignia\src\attention.cu` line 7: `const int head=blockIdx.x,tid=threadIdx.x,kvh=head>>2;`
- `E:\coding\Insignia\src\prefill.cu` line 103: `const int head = blockIdx.x, t = blockIdx.y, tid = threadIdx.x, kvh = head >> 2;`

Full mapping table (h : h>>2 : h/6):

```
h   : >>2 : /6      h   : >>2 : /6      h   : >>2 : /6
0-3 : 0-3 : 0       8-11: 2   : 1 WRONG 16-19: 4 OOB : 2
4-5 : 1   : 0 WRONG 12-15: 3  : 2 WRONG 20-23: 5 OOB : 3
6-7 : 1   : 1                          (>>2 groups of 4; /6 groups of 6)
```

**Fix: `kvh = head/6`** (see kernels below). Magic-shift alternative verified by
exhaustive check: `(h*171)>>10 == h/6` for all h in [0,24] (171/1024 ≈ 1/6; e.g.
h=5: 855>>10=0 ✓, h=6: 1026>>10=1 ✓, h=17: 2907>>10=2 ✓, h=23: 3933>>10=3 ✓).
**Recommendation: write plain `head/6`.** `blockIdx.x` is `unsigned`; nvcc lowers a
compile-time-constant divisor to exactly this mul-hi+shift sequence (~3 int ops, once
per block, against a 256-wide FMA loop) — hand-rolling the magic buys zero instructions
and loses the self-documenting property. Same class of reasoning as synthesis backlog:
measured/survivable tricks only, no superstition.

Also to correct in the same file while you are in there: shape-constants line 365
("24 q heads keep GQA head>>2 valid") is the same error in the closing summary.

---

## 1. Parameterization decision: `template <int QH>` (compile-time), NOT runtime `int`

Picked template. Justification:

1. **Runtime `int QH` emits a real hardware divide** (`head/QH` on a variable = ~20+
   instructions or a called subroutine). Still negligible in absolute terms, but there
   is no benefit to pay it — the model shape is fixed per engine build, and the whole
   codebase bakes dims as literals (that is the project's stated constitution).
2. **Compile-time QH lets the compiler fold the group division**: QH=16 → `head/(16/4)`
   = `>>2` (today's exact codegen, zero 9B regression), QH=24 → `head/6` → mul-magic.
3. **Both instantiations coexist** so 9B parity tests and the NumPy reference keep
   running during migration (`<16>` for 9B, `<24>` for 27B) with one launcher flip.
4. Kernel public signatures in the 4 headers are UNCHANGED (all head counts are already
   implicit in the launchers), so decode.cu call sites need no signature edits — only
   buffer sizes and `sigmoid_mul` counts (§8).

KV heads stay 4 in both models, so the group is `QH/4` and the KV row layout
`(t*4+kvh)*256` is untouched. Grid arithmetic per kernel:

| kernel | 9B grid | 27B grid | changes |
|---|---|---|---|
| qk_norm_rope (ops.cu) | 20 | **28** (=24q+4k) | `isq=head<24`, `k+(head-24)*256` |
| qk_norm_rope_batch (prefill.cu) | dim3(20,T) | **dim3(28,T)** | same + `t*24` q stride |
| gqa_decode (attention.cu) | 16 | **24** | `kvh=head/6` |
| gqa_prefill (prefill.cu) | dim3(16,T) | **dim3(24,T)** | `kvh`, `t*24` q/out strides |
| split_q_gate (qwen_kernels.cu) | 16 | **24** | `i<6144` |
| expand_gate_heads | 16 | **24** | `i<6144` |
| split_q_gate_batch (prefill.cu) | T*16 | **T*24** | `/24`,`%24`,`t*12288`,`t*6144` |
| store_kv / store_kv_batch | 4 / dim3(4,T) | **unchanged** | see §5 |

### 1a. `src/ops.cu` — qk_norm_rope (decode, single token) — FULL REPLACEMENT

Verified present in the current code (synthesis bug #2 fix): the norm scale lives in a
dedicated `__shared__ float nsc` slot, NOT `mem[0]`, so the pos>0 rope staging that
clobbers `mem[0..63]` cannot race warps that have not yet read the scale. Kept verbatim
below. Also verified unchanged at 27B: theta `10000000.0f`, partial rope `tid<64`,
`mem[64]`, exponent `-2*half/64`, head_dim `ss/256`, `1e-6f` eps (matches 27B config:
partial_rotary_factor 0.25 → 64/256, rope_theta 1e7).

```cuda
// template <int QH> replaces the plain kernel; QH = q heads (16 = 9B, 24 = 27B).
// Grid: QH+4 blocks (24 q + 4 k at 27B). `gate` param stays for signature stability (unused).
template <int QH>
__global__ void qk_norm_rope(float *q, float *k, const uint16_t *qw, const uint16_t *kw, float *gate, const int *pos_dev, int off) {
    const int head = blockIdx.x, tid = threadIdx.x;
    const bool isq = head < QH;                       // 27B: 24 q blocks, then 4 k blocks
    float *p = isq ? q + head * 256 : k + (head - QH) * 256;
    const uint16_t *w = isq ? qw : kw;
    float v = p[tid], ss = v * v;
    for (int m = 16; m; m >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, m);
    __shared__ float mem[64];
    __shared__ float nsc;  // norm scale lives apart: mem[0..63] is later clobbered by the roped staging (race fix)
    int lane = tid & 31, warp = tid >> 5;
    if (lane == 0) mem[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        ss = lane < 8 ? mem[lane] : 0;
        for (int m = 16; m; m >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, m);
        if (lane == 0) nsc = rsqrtf(ss / 256 + 1e-6f);
    }
    __syncthreads();
    v *= nsc * __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(w + tid));
    const int pos = __ldg(pos_dev) + off;
    if (pos != 0 && tid < 64) mem[tid] = v;
    __syncthreads();
    if (pos != 0 && tid < 64) {
        const int half = tid & 31;
        const float inv = __powf(10000000.0f, -float(2 * half) / 64.0f), a = float(pos) * inv, c = __cosf(a), s = __sinf(a);
        const float other = mem[tid < 32 ? tid + 32 : tid - 32];
        v = v * c + (tid < 32 ? -other : other) * s;
    }
    p[tid] = v;
}
void qwen35_qk_norm_rope_gate(float *q, float *k, const uint16_t *qw, const uint16_t *kw, float *g, const int *pos_dev, int off, cudaStream_t s) {
    qk_norm_rope<24><<<28, 256, 0, s>>>(q, k, qw, kw, g, pos_dev, off);  // 27B: 24 q + 4 k blocks (9B: <16><<<20,...>>>)
}
```

Buffer notes (decode path): q = `x_.qkv` (24*256 = 6144 floats used; alloc 10240 covers),
k = `x_.key` (1024 = 4*256, unchanged).

### 1b. `src/prefill.cu` — qk_norm_rope_batch — FULL REPLACEMENT

Race fix (`__shared__ float nsc` at old line 62) verified present and preserved.

```cuda
// QK norm + partial RoPE for T tokens; token t uses position pos_dev[0] + t.
// Grid dim3(QH+4, T): x in [0,QH) = q heads, x in [QH,QH+4) = the 4 kv heads.
template <int QH>
__global__ void qk_norm_rope_batch_kernel(float *__restrict__ q, float *__restrict__ k, const uint16_t *__restrict__ qw, const uint16_t *__restrict__ kw, const int *__restrict__ pos_dev) {
    const int t = blockIdx.y, head = blockIdx.x, tid = threadIdx.x;
    const bool isq = head < QH;                        // 27B: head < 24
    float *p = isq ? q + (static_cast<size_t>(t) * QH + head) * 256 : k + (static_cast<size_t>(t) * 4 + head - QH) * 256;
    const uint16_t *w = isq ? qw : kw;
    float v = p[tid], ss = v * v;
    for (int m = 16; m; m >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, m);
    __shared__ float mem[64];
    __shared__ float nsc;  // norm scale lives apart: mem[0..63] is later clobbered by the roped staging (race fix)
    int lane = tid & 31, warp = tid >> 5;
    if (lane == 0) mem[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        ss = lane < 8 ? mem[lane] : 0;
        for (int m = 16; m; m >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, m);
        if (lane == 0) nsc = rsqrtf(ss / 256 + 1e-6f);
    }
    __syncthreads();
    v *= nsc * __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(w + tid));
    const int pos = __ldg(pos_dev) + t;
    if (pos != 0 && tid < 64) mem[tid] = v;
    __syncthreads();
    if (pos != 0 && tid < 64) {
        const int half = tid & 31;
        const float inv = __powf(10000000.0f, -float(2 * half) / 64.0f), a = float(pos) * inv, c = __cosf(a), s = __sinf(a);
        const float other = mem[tid < 32 ? tid + 32 : tid - 32];
        v = v * c + (tid < 32 ? -other : other) * s;
    }
    p[tid] = v;
}
void qk_norm_rope_batch(float *q, float *k, const uint16_t *qw, const uint16_t *kw, const int *pos_dev, int T, cudaStream_t stream) {
    qk_norm_rope_batch_kernel<24><<<dim3(28, T), 256, 0, stream>>>(q, k, qw, kw, pos_dev);  // 27B (9B: <16>, dim3(20,T))
}
```

Buffer fit check: `pf_q` = 64*6144 (must grow from 64*4096 — §8); k side indexes
`(t*4 + head-24)*256 + tid` max = `(63*4+3)*256+255` = 65535 = last float of
`pf_k` (64*1024) — exact fit, no slack, unchanged k_proj row width.

---

## 2. `src/attention.cu` — gqa_decode — FULL REPLACEMENT (minimal 27B port)

Only two changes from today's kernel: grid 16→24 and `kvh = head>>2` → `head/6`
(the §0 bug fix). Body intentionally untouched so the 9B parity baseline carries over;
the access-pattern upgrade is the separate split-K kernel in §7 (v2).

```cuda
#include "insignia_attention.cuh"
#include <cuda_runtime.h>
namespace insignia {
// GQA decode for one query token against the contiguous KV cache. Token count is read from
// device memory (pos_dev[0] + base + 1) so the whole step is CUDA-graph replayable; the
// score buffer is a fixed 4096-slot array instead of dynamic shared memory for the same reason.
// 27B: 24 q heads over 4 kv heads -> GQA group 6 -> kvh = head/6. head>>2 is the 9B mapping
// and is WRONG at 24 heads (16/24 heads mis-grouped; heads 16..23 read the NEXT token's kv
// rows 0/1 — silent corruption, OOB past the cache slice at the last filled slot).
template <int QH>
__global__ __launch_bounds__(256, 2) void gqa_decode_kernel(const float *__restrict__ q, const float *__restrict__ kc, const float *__restrict__ vc, float *__restrict__ out, const int *__restrict__ pos_dev, int base) {
    const int head = blockIdx.x, tid = threadIdx.x, kvh = head / (QH / 4);  // QH=24 -> /6, compiler emits mul-magic
    const int tokens = __ldg(pos_dev) + base + 1;
    __shared__ float score[4096];
    const float scale = .0625f;
    float mx = -3.402823466e+38F;
    for (int t = tid; t < tokens; t += 256) {
        float z = 0;
        for (int d = 0; d < 256; d++) z = fmaf(q[head * 256 + d], kc[(size_t(t) * 4 + kvh) * 256 + d], z);
        z *= scale;
        score[t] = z;
        mx = fmaxf(mx, z);
    }
    for (int m = 16; m; m >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, m));
    __shared__ float red[8];
    int lane = tid & 31, warp = tid >> 5;
    if (lane == 0) red[warp] = mx;
    __syncthreads();
    if (warp == 0) {
        mx = lane < 8 ? red[lane] : -3.402823466e+38F;
        for (int m = 16; m; m >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, m));
        if (lane == 0) red[0] = mx;
    }
    __syncthreads();
    mx = red[0];
    float den = 0;
    for (int t = tid; t < tokens; t += 256) {
        float e = __expf(score[t] - mx);
        score[t] = e;
        den += e;
    }
    for (int m = 16; m; m >>= 1) den += __shfl_xor_sync(0xffffffff, den, m);
    if (lane == 0) red[warp] = den;
    __syncthreads();
    if (warp == 0) {
        den = lane < 8 ? red[lane] : 0;
        for (int m = 16; m; m >>= 1) den += __shfl_xor_sync(0xffffffff, den, m);
        if (lane == 0) red[0] = 1.f / den;
    }
    __syncthreads();
    float z = 0;
    for (int t = 0; t < tokens; t++) z = fmaf(score[t] * red[0], vc[(size_t(t) * 4 + kvh) * 256 + tid], z);
    out[head * 256 + tid] = z;
}
void gqa_decode(const float *q, const float *k, const float *v, float *out, const int *pos_dev, int base, int max_context, cudaStream_t s) {
    (void)max_context;
    gqa_decode_kernel<24><<<24, 256, 0, s>>>(q, k, v, out, pos_dev, base);  // 27B (9B: <16><<<16,...>>>)
}
}
```

`score[4096]` stays: it is the context cap (max_context policy 1..4096, decode.hpp),
numerically coincidental with 9B hidden. smem = 16KB + 32B < 48KB static cap;
`__launch_bounds__(256,2)` still legal (2*16.1KB < 99KB/SM).

Buffer fits: q = `x_.qkv` first 6144 floats (head<24 → max idx 6143; alloc 10240 ✓);
out = `x_.core` (must grow to 6144, §8).

---

## 3. `src/prefill.cu` — gqa_prefill — FULL REPLACEMENT

Changes: `dim3(16,T)` → `dim3(24,T)`, `kvh = head>>2` → `head/6`, q/out strides
`(t*16+head)*256` → `(t*24+head)*256`. Everything else (warp-per-key-row coalesced
scoring, `score[4096]`, `part[8][256]` cross-warp V reduction) is dimension-correct at
27B already: KV row width stays `4*256`, warps stride 8 over tokens, smem
1KB+16KB+32B+8KB = 25.1KB < 48KB static cap, `__launch_bounds__(256,2)` → 50.2KB/SM ≤ 99KB.

```cuda
// One (query head, token) per block; causal over pos_dev[0]+t+1 cache keys.
// Warps own whole key rows (8 dims per lane) so K/V reads stay coalesced.
// 27B: 24 q heads / 4 kv heads -> kvh = head/6 (GQA group 6; head>>2 is the 9B bug).
template <int QH>
__global__ __launch_bounds__(256, 2) void gqa_prefill_kernel(const float *__restrict__ q, const float *__restrict__ kc, const float *__restrict__ vc, float *__restrict__ out, const int *__restrict__ pos_dev) {
    const int head = blockIdx.x, t = blockIdx.y, tid = threadIdx.x, kvh = head / (QH / 4);
    const int tokens = __ldg(pos_dev) + t + 1;
    __shared__ float qs[256];
    __shared__ float score[4096];
    __shared__ float red[8];
    __shared__ float part[8][256];
    const float *qrow = q + (static_cast<size_t>(t) * QH + head) * 256;   // 27B: t*24+head
    qs[tid] = qrow[tid];
    __syncthreads();
    const int lane = tid & 31, warp = tid >> 5;
    float mx = -3.402823466e+38F;
    for (int j = warp; j < tokens; j += 8) {
        const float *krow = kc + (size_t(j) * 4 + kvh) * 256 + lane * 8;
        float z = 0;
        #pragma unroll
        for (int d = 0; d < 8; d++) z = fmaf(qs[lane * 8 + d], __ldg(krow + d), z);
        for (int m = 16; m; m >>= 1) z += __shfl_xor_sync(0xffffffff, z, m);
        if (lane == 0) { z *= .0625f; score[j] = z; mx = fmaxf(mx, z); }
    }
    for (int m = 16; m; m >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, m));
    if (lane == 0) red[warp] = mx;
    __syncthreads();
    if (warp == 0) {
        mx = lane < 8 ? red[lane] : -3.402823466e+38F;
        for (int m = 16; m; m >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, m));
        if (lane == 0) red[0] = mx;
    }
    __syncthreads();
    mx = red[0];
    float den = 0;
    for (int j = warp; j < tokens; j += 8) {
        float e = __expf(score[j] - mx);
        if (lane == 0) { score[j] = e; den += e; }
    }
    for (int m = 16; m; m >>= 1) den += __shfl_xor_sync(0xffffffff, den, m);
    if (lane == 0) red[warp] = den;
    __syncthreads();
    if (warp == 0) {
        den = lane < 8 ? red[lane] : 0;
        for (int m = 16; m; m >>= 1) den += __shfl_xor_sync(0xffffffff, den, m);
        if (lane == 0) red[0] = 1.f / den;
    }
    __syncthreads();
    const float inv_den = red[0];
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int j = warp; j < tokens; j += 8) {
        const float p = score[j] * inv_den;
        const float *vrow = vc + (size_t(j) * 4 + kvh) * 256 + lane * 8;
        #pragma unroll
        for (int d = 0; d < 8; d++) acc[d] = fmaf(p, __ldg(vrow + d), acc[d]);
    }
    #pragma unroll
    for (int d = 0; d < 8; d++) part[warp][lane * 8 + d] = acc[d];
    __syncthreads();
    float o = 0;
    #pragma unroll
    for (int w2 = 0; w2 < 8; w2++) o += part[w2][tid];
    out[(static_cast<size_t>(t) * QH + head) * 256 + tid] = o;           // 27B: t*24+head
}
void gqa_prefill(const float *q, const float *kc, const float *vc, float *out, const int *pos_dev, int T, int max_context, cudaStream_t stream) {
    (void)max_context;
    gqa_prefill_kernel<24><<<dim3(24, T), 256, 0, stream>>>(q, kc, vc, out, pos_dev);  // 27B (9B: <16>, dim3(16,T))
}
```

Grid at T=64: 1536 blocks — parallelism is not a concern on the prefill side.

---

## 4. q_proj q/gate split + gate expand

q_proj at 27B is [12288, 5120] = 24 heads x 512 rows, per-head interleave
(256 q | 256 gate) — the **512-per-head stride is unchanged**; only the head count
moves. `h = i>>8` stays valid for any head count (6144>>8 = 23 ✓).

### 4a. `src/qwen_kernels.cu` — split_q_gate + expand_gate_heads — FULL REPLACEMENTS

```cuda
namespace insignia {
// src: q_proj output, 24 heads x 512 interleaved (q | gate); q/gate out: 24 x 256 each = 6144.
template <int QH>
__global__ void split_q_gate_kernel(const float *src, float *q, float *g) {
    int i = blockIdx.x * 256 + threadIdx.x;               // grid QH blocks -> i < QH*256 always; guard kept for generality
    if (i < QH * 256) { int h = i >> 8, d = i & 255; q[i] = src[h * 512 + d]; g[i] = src[h * 512 + 256 + d]; }
}
void split_q_gate(const float *s, float *q, float *g, cudaStream_t stream) {
    split_q_gate_kernel<24><<<24, 256, 0, stream>>>(s, q, g);  // 27B (9B: <16><<<16,...>>>, guard i<4096)
}
}

namespace insignia {
template <int QH>
__global__ void expand_gate_kernel(const float *g, float *out) { int i = blockIdx.x * 256 + threadIdx.x; if (i < QH * 256) out[i] = g[i]; }
void expand_gate_heads(const float *g, float *out, cudaStream_t s) {
    expand_gate_kernel<24><<<24, 256, 0, s>>>(g, out);         // 27B (9B: <16><<<16,...>>>)
}
}
```

Decode-path buffer fits: src = `x_.gate` (17408 covers q_proj's 12288 ✓); q out =
`x_.qkv` (uses first 6144 of 10240 ✓); gate out = `x_.attn_gate` (**must grow 4096→6144**, §8).

### 4b. `src/prefill.cu` — split_q_gate_batch — FULL REPLACEMENT

`>>4`/`&15` are illegal at 24 heads (blockIdx.x=16 decodes to t=1,h=0). Mission-specified
minimal diff: `/24` + `%24` — nvcc lowers each to a mul-hi+shift (~3 int ops), and it is
**once per thread**, against a 3-load/2-store body: noise. (If wanted for free later:
switch the grid to `dim3(24, T)` and read `h=blockIdx.x, t=blockIdx.y` — kills the
div/mod entirely, matching the sibling kernels' style; both are correct.)

```cuda
__global__ void split_q_gate_batch_kernel(const float *__restrict__ src, float *__restrict__ q, float *__restrict__ gate) {
    const int t = blockIdx.x / 24, h = blockIdx.x % 24, d = threadIdx.x;   // 9B was >>4 / &15
    const size_t base = static_cast<size_t>(t) * 12288 + h * 512;          // q_proj rows 12288 = 24 x (256q+256gate)
    q[static_cast<size_t>(t) * 6144 + h * 256 + d] = src[base + d];
    gate[static_cast<size_t>(t) * 6144 + h * 256 + d] = src[base + 256 + d];
}
void split_q_gate_batch(const float *src, float *q, float *gate, int T, cudaStream_t stream) {
    split_q_gate_batch_kernel<<<T * 24, 256, 0, stream>>>(src, q, gate);   // 9B was T*16
}
```

Buffer fits: src = `pf_scratch` (**must grow 64*8192 → 64*12288**, §8 — q_proj's 12288
rows, not just conv's 10240); q/gate = `pf_q`/`pf_g` (**64*6144** each).

---

## 5. store_kv — VERIFIED UNCHANGED at 27B

Both variants write a 1024-float row (4 kv heads x 256 dims) from the k_proj/v_proj
outputs into `kc/vc[pos*1024 + i]`. At 27B, k_proj and v_proj are [1024, 5120] — row
count 1024 is unchanged, head_dim 256 unchanged, kv-head count 4 unchanged. Verified:

- `src/qwen_kernels.cu:15-16` `store_kv_kernel`: `i<1024`, `pos*1024`, launch `<<<4,256>>>`
  (4*256 = 1024 threads, guard redundant but harmless). **No edit.**
- `src/prefill.cu:88-98` `store_kv_batch_kernel`: `i>=1024` guard, `pos*1024`,
  `k[t*1024+i]`, launch `dim3(4,T)`. **No edit.**

The 9B→27B change that DOES touch the KV cache is allocation only: 8→16 full-attn
layers (`kv_keys`/`kv_values` = `size_t(16)*ctx*1024`, §8) and the layer index
`ai = l/4` now ranges 0..15 (formula unchanged; the `l<64` loop bound is the edit).

---

## 6. KV dtype — verdict: **f32 for v1**, bf16 behind a compile flag for v2

Rationale: the engine has not yet reached coherent-token parity on full-attention
layers (AGENTS.md); bf16 KV changes numerics vs the NumPy reference (rounding K/V on
every store) and would confound the open parity hunt. The VRAM saving alone is marginal;
the real v2 payoff is halving KV read bytes (halves the §7 bandwidth floor).

VRAM math (K+V caches for the 16 full-attn layers; 4 kv heads x 256 x 4B/2B per token
per cache; caches are `cudaMalloc`ed → VRAM regardless of the layer's weight tier):

| ctx | f32 (K+V x16) | bf16 | saved |
|---|---|---|---|
| 1024 | 134 MB | 67 MB | 67 MB |
| 2048 | 268 MB | 134 MB | 134 MB |
| 4096 | 537 MB | 268 MB | 268 MB (2.2% of 12 GB) |

Bandwidth payoff at ctx 2048: unique KV read 16.8 MB → 8.4 MB → decode-attention BW
floor 33 µs → 16.5 µs at 504 GB/s.

Design (flag `INSIG_KV_BF16`, typedef `kvt`):

```cuda
// ---- store: f32 -> bf16 round-to-nearest-even, 3 int ops, no cvt round-trip ----
__device__ __forceinline__ uint16_t f32_to_bf16_rne(float x) {
    const uint32_t b = __float_as_uint(x);
    return uint16_t((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);   // RNE via bottom-bit carry
}
// ---- widen: bf16 -> f32, one shift, EXACT (bf16 is the top half of f32) ----
__device__ __forceinline__ float bf16_to_f32(uint16_t h) { return __uint_as_float(uint32_t(h) << 16); }

// store_kv body under the flag:
//   kc16[pos * 1024 + i] = f32_to_bf16_rne(k[i]);   (kc/vc become uint16_t*/__nv_bfloat16*)
// gqa_decode dot under the flag (per-element widen, 1 SHL per load):
//   z = fmaf(q[head * 256 + d], bf16_to_f32(kc16[(size_t(t) * 4 + kvh) * 256 + d]), z);
// vector variant (sm_89 CVT, 1 instr per 2 elements): load __nv_bfloat162, __bfloat1622float2.
```

Parity note if/when enabled: K/V magnitudes after qk-norm are O(1) — well inside bf16's
~3 decimal digits; expect cosine-to-reference degradation of maybe 1e-7 → 1e-3-ish per
layer. Verify with `tools/reference_*.py` before trusting.

---

## 7. Occupancy sanity + split-K decode design

**First, correct the premise: the 4070 SUPER is AD104 with 56 SMs, not 128** (128 SMs
is 4090-class). So 24 blocks = **43%** of SMs (single wave; `__launch_bounds__(256,2)`
would allow 2 blocks/SM), not 19%.

- **Small/medium ctx (≤ ~1024): acceptable for v1.** Each thread handles ≤4 tokens;
  wall time is launch + latency dominated, ~5-15 µs. The "10-20 µs latency-bound" frame
  holds — fine at 16 attn layers per token (~0.1-0.3 ms/token total attention).
- **Long ctx (≥ ~2048): bandwidth-bound on KV, and 24 blocks is thin.** Unique KV bytes
  at ctx C = 4 kvh x C x 2 (K,V) x 1 KB = 8.2 KB/token → 16.8 MB at 2048 → 33 µs floor
  at 504 GB/s. Each of the 4 kv rows is read by its 6 group-sibling blocks → L2 dedup
  (unique set 16.8 MB < ~36 MB L2), so DRAM sees ~16.8 MB, not 100 MB. The risk is
  *request-side*: today's decode body is thread-per-token, so a warp's 32 lanes touch 32
  different 1 KB rows → 32 sectors per warp-load (12.5% lane efficiency per transaction;
  L1 sector reuse keeps DRAM volume near-unique but burns LSU wavefronts), and 24 SMs
  must generate enough outstanding sectors to fill ~504 GB/s. Whether that reaches, say,
  50-90% of peak is not analytically settled — **measure (ctx sweep 512/1024/2048/4096)
  before deciding**, per the project's measurement rule. This is synthesis backlog #5
  ("coalesced Q·K warp-per-key-row, split-K over tokens, bf16 KV") — not a 27B
  regression; 27B just raises the stake from 16 to 24 heads (same 4 kvh).

**Split-K variant (v2, ship behind the threshold):** grid `dim3(24, S)` with S=4 → 96
blocks (1.7 waves on 56 SMs), warp-per-key-row coalesced scoring (the proven
`gqa_prefill` pattern), partial (m, l, acc[256]) per split in a small global scratch,
then a 24-block combine kernel (deterministic reduction order — parity-friendly, no
atomics). Graph-safe: S and the scratch pointer are frozen at capture; `tokens` stays
dynamic from `pos_dev`. Enable when `max_context > 1024` (host picks at capture time —
below that, extra blocks just add the combine launch for nothing). Constraint:
`(ctx + S - 1)/S ≤ 1024` — satisfied by S=4 for ctx ≤ 4096 (score[] shrinks to 1024).

Scratch: `float scratch[S][24][258]` (m, l, acc[256]) = 4*24*258*4 B ≈ 97 KB — add one
`cudaMalloc` to DecodeWorkspace (stable pointer → graph-safe).

```cuda
// ---- split-K GQA decode (v2, enable when max_context > 1024) ----
// grid (QH, S); block (head, s) scores tokens [s*chunk, min(tokens,(s+1)*chunk)) with
// warp-per-key-row coalesced loads; partials land at scratch[s][head] = {m, l, acc[256]}.
// Requires (max_context + S - 1) / S <= 1024 (score[] capacity): S=4 covers ctx<=4096.
template <int QH>
__global__ __launch_bounds__(256, 2) void gqa_decode_split_kernel(const float *__restrict__ q, const float *__restrict__ kc, const float *__restrict__ vc, float *__restrict__ scratch, const int *__restrict__ pos_dev, int base) {
    const int head = blockIdx.x, tid = threadIdx.x, kvh = head / (QH / 4);
    const int S = gridDim.y, s = blockIdx.y;
    const int tokens = __ldg(pos_dev) + base + 1;
    const int chunk = (tokens + S - 1) / S;
    const int t0 = s * chunk, n = min(tokens - t0, chunk);   // this split's token count (may be 0)
    __shared__ float qs[256];
    __shared__ float score[1024];
    __shared__ float red[8];
    __shared__ float part[8][256];
    const float *qrow = q + head * 256;
    qs[tid] = qrow[tid];
    __syncthreads();
    const int lane = tid & 31, warp = tid >> 5;
    float mx = -3.402823466e+38F;
    for (int j = warp; j < n; j += 8) {                      // warp owns whole key rows: coalesced
        const float *krow = kc + (size_t(t0 + j) * 4 + kvh) * 256 + lane * 8;
        float z = 0;
        #pragma unroll
        for (int d = 0; d < 8; d++) z = fmaf(qs[lane * 8 + d], __ldg(krow + d), z);
        for (int m = 16; m; m >>= 1) z += __shfl_xor_sync(0xffffffff, z, m);
        if (lane == 0) { z *= .0625f; score[j] = z; mx = fmaxf(mx, z); }
    }
    for (int m = 16; m; m >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, m));
    if (lane == 0) red[warp] = mx;
    __syncthreads();
    if (warp == 0) {
        mx = lane < 8 ? red[lane] : -3.402823466e+38F;
        for (int m = 16; m; m >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, m));
        if (lane == 0) red[0] = mx;
    }
    __syncthreads();
    mx = red[0];
    float den = 0;
    for (int j = warp; j < n; j += 8) {
        float e = __expf(score[j] - mx);
        if (lane == 0) { score[j] = e; den += e; }
    }
    for (int m = 16; m; m >>= 1) den += __shfl_xor_sync(0xffffffff, den, m);
    if (lane == 0) red[warp] = den;
    __syncthreads();
    if (warp == 0) {
        den = lane < 8 ? red[lane] : 0;
        for (int m = 16; m; m >>= 1) den += __shfl_xor_sync(0xffffffff, den, m);
        if (lane == 0) red[0] = den > 0 ? den : 1.f;   // empty split: keep l=0 but never 1/0
    }
    __syncthreads();
    const float l_s = red[0];
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int j = warp; j < n; j += 8) {
        const float p = score[j] / l_s;
        const float *vrow = vc + (size_t(t0 + j) * 4 + kvh) * 256 + lane * 8;
        #pragma unroll
        for (int d = 0; d < 8; d++) acc[d] = fmaf(p, __ldg(vrow + d), acc[d]);
    }
    #pragma unroll
    for (int d = 0; d < 8; d++) part[warp][lane * 8 + d] = acc[d];
    __syncthreads();
    if (tid < 256) {
        float o = 0;
        #pragma unroll
        for (int w2 = 0; w2 < 8; w2++) o += part[w2][tid];
        float *pk = scratch + (static_cast<size_t>(s) * QH + head) * 258;
        pk[0] = n > 0 ? mx : -3.402823466e+38F;   // empty split -> -inf max, weight 0 in combine
        pk[1] = n > 0 ? l_s : 0.f;
        pk[2 + tid] = o;
    }
}

// Combine: flash-style rescale. out = sum_s acc_s * e^{m_s - M} / sum_s l_s * e^{m_s - M}.
// den is recomputed per thread (S<=4 fmaf, cheaper than a block reduce).
template <int QH>
__global__ void gqa_decode_combine_kernel(const float *__restrict__ scratch, float *__restrict__ out, int S) {
    const int head = blockIdx.x, tid = threadIdx.x;
    const int stride = 258 * QH;
    float M = -3.402823466e+38F;
    for (int s = 0; s < S; s++) M = fmaxf(M, scratch[s * stride + head * 258]);
    float num = 0, den = 0;
    for (int s = 0; s < S; s++) {
        const float *p = scratch + s * stride + head * 258;
        const float w = __expf(p[0] - M);        // e^{-inf - M} = 0 for empty splits
        num = fmaf(w, p[2 + tid], num);
        den = fmaf(w, p[1], den);
    }
    out[head * 256 + tid] = num / den;           // den >= e^{0}=1 from the first (non-empty) split
}

// Launcher (signature-compatible; header gains: `float *split_scratch = nullptr`):
// void gqa_decode(const float *q, const float *k, const float *v, float *out,
//                 const int *pos_dev, int base, int max_context, cudaStream_t s,
//                 float *split_scratch);
void gqa_decode(const float *q, const float *k, const float *v, float *out, const int *pos_dev, int base, int max_context, cudaStream_t s, float *split_scratch) {
    (void)max_context;
    if (split_scratch) {                    // host chose the split path at capture: max_context > 1024
        const int S = 4;                    // (max_context + S - 1) / S <= 1024 for ctx <= 4096
        gqa_decode_split_kernel<24><<<dim3(24, S), 256, 0, s>>>(q, k, v, split_scratch, pos_dev, base);
        gqa_decode_combine_kernel<24><<<24, 256, 0, s>>>(split_scratch, out, S);
    } else {
        gqa_decode_kernel<24><<<24, 256, 0, s>>>(q, k, v, out, pos_dev, base);
    }
}
```

smem for the split kernel: 1KB + 4KB + 32B + 8KB = 13.1 KB → 2 blocks/SM comfortably.
Empty-split edge is handled: mx=-inf, l_s forced ≥1 (never 1/0), acc=0 → combine weight
`e^{-inf}=0`. `num/den` safe because split 0 is never empty (`t0=0 < tokens`, tokens≥1).

Expected effect at ctx 2048 (f32 KV): from ≤~24-SM-limited, transaction-throttled
(33 µs floor, realistically above) to 96 coalesced blocks that can actually pull
~504 GB/s → ~33 µs + ~3-5 µs combine. With §6 bf16 KV: ~17 µs. Numbers to be confirmed
by the ctx-sweep bench, not assumed.

---

## 8. Call-site diffs (`src/decode.cu`) — attention path only

Kernel launch call sites need **no signature changes** (all edits live inside the
launchers). The required decode.cu edits (subset of the w2 W-list that the attention
path depends on — misses here are OOB writes into the next buffer):

| line | now | 27B | why |
|---|---|---|---|
| 14 | `alloc(&attn_gate,4096)` | `6144` | split_q_gate gate out = 24x256 |
| 14 | `alloc(&core,4096)` | `6144` | gqa_decode out = 24x256 |
| 14 | `alloc(&kv_keys,size_t(8)*ctx*1024)` | `size_t(16)*ctx*1024` | 16 full-attn layers |
| 14 | `alloc(&kv_values,size_t(8)*ctx*1024)` | `size_t(16)*ctx*1024` | 16 full-attn layers |
| 22 | `alloc(&pf_scratch,64*8192)` | `64*12288` | q_proj rows 12288 (NOT 10240) |
| 23 | `alloc(&pf_q,64*4096)`, `pf_g` | `64*6144` | 24 q heads x 256 |
| 23 | `alloc(&pf_core,64*4096)` | `64*6144` | gqa_prefill out |
| 47 | `for(int l=0;l<32;l++)` | `l<64` | prefill loop; `ai=l/4` then spans 0..15 |
| 62 | `sigmoid_mul(...,size_t(T)*4096,...)` | `size_t(T)*6144` | attn gate apply (prefill path) |
| 127 | `sigmoid_mul(x_.core,x_.qkv,4096,...)` | `6144` | attn gate apply (decode path) |
| 172 | `sigmoid_mul(x_.core,x_.qkv,4096,...)` | `6144` | attn gate apply (MTP layer) |

Unchanged-but-load-bearing (verify only, verified in this audit): `ai=l/4` + stride
`ai*max_context*1024` (formula identical, range widens with l<64); `store_kv`,
`store_kv_batch`, `split_q_gate`, `split_q_gate_batch`, `qk_norm_rope_batch`,
`gqa_decode`, `gqa_prefill`, `qwen35_qk_norm_rope_gate`, `expand_gate_heads` call sites
(signatures stable); mtp_keys/mtp_values `ctx*1024` (1 MTP layer, 4 kvh x 256);
`x_.qkv` as q buffer (6144 of 10240 used). Graph capture (`capture_step`/`capture_spec`,
lines 232-259) must be deleted and re-captured after any of this lands — grids and
buffer pointers are frozen per AGENTS/w2 note.

Split-K (§7) additionally: one `alloc(&gqa_scratch, 4*24*258)` (~97 KB) + pass
`x_.gqa_scratch` to `gqa_decode` when `max_context > 1024` (decide once before
`capture_step`; the branch is capture-stable).

---

## 9. Verification checklist

1. **kvh mapping unit check**: extend `test_attention.cu` reference to `h/6` (w2
   instrumentation table already flags `H=16→24` + reference `h/4→/6`; the file also
   needs the current gqa_decode signature fix). Assert block 0..23 → kvh {0,0,0,0,0,0,1,...,3}
   against a KV cache poisoned per (token, kvh) — this catches any `>>2` relapse.
2. **Race-fix regression**: qk_norm_rope/batch pos>0 deterministic repeat (the
   synthesis bug #2 symptom) — both kernels verified to carry the `nsc` slot; keep it
   that way in review.
3. **Parity gates**: layer-3 (first full-attn layer) cosine vs NumPy reference before
   and after the 27B port; then the 16 attn layers at l%4==3.
4. **Bench**: gqa_decode ctx sweep {512, 1024, 2048, 4096} x {simple, split-K, split-K
   + bf16 KV} — decide §7 threshold and §6 flag on numbers, not vibes.

## TL;DR

1. shape-constants.md is WRONG twice (lines 169, 365): `kvh=head>>2` does NOT survive
   24 q-heads. 16/24 heads mis-group; heads 16..23 go OOB (kvh 4/5) and silently read
   the NEXT token's kv rows; true OOB past the cache slice at the last filled slot.
   synthesis.md line 34 already had it right. Fix: `kvh = head/6` in attention.cu:7 and
   prefill.cu:103 (compiler emits the mul-magic; `(h*171)>>10` verified equivalent for
   h<24 — write the division, not the magic).
2. All kernels delivered as `template <int QH>` (compile-time): runtime divisor would
   emit a real divide for zero benefit; 9B (`<16>`) and 27B (`<24>`) coexist for parity;
   public signatures unchanged → decode.cu call sites only change buffer sizes.
3. qk_norm_rope grids 20→28, `isq=head<24`, `k+(head-24)*256`; theta 1e7 / rope-64 /
   dim-256 literals all verified to stay; the `nsc` smem race-fix is present in BOTH
   ops.cu and prefill.cu today and is preserved.
4. gqa_decode grid 16→24 + kvh/6 (body untouched for parity); gqa_prefill dim3(24,T) +
   kvh/6 + `(t*24+head)*256` strides; both `score[4096]` (context cap, not hidden).
5. split_q_gate/expand: `<<<24,256>>>`, `i<6144`; `h=i>>8` valid (6144>>8=24); the 512
   per-head q|gate interleave is unchanged — only the head count moves.
6. split_q_gate_batch: `/24`, `%24` (compiler mul-shift, once per thread — fine),
   `t*12288`, out `t*6144`; pf_scratch must be 64*12288 (q_proj rows, not conv's 10240).
7. store_kv/store_kv_batch verified UNCHANGED (4 kvh x 256 = 1024-row stays).
8. KV dtype: f32 for v1 (open parity hunt must not change numerics); bf16 flag design
   included (RNE bit-trick store, `<<16` exact widen) — VRAM saving is marginal
   (67-268 MB = 0.5-2.2% of 12 GB), the real win is halving the KV read (33→16.5 µs
   floor at ctx 2048); enable only with reference-checked parity.
9. Occupancy premise corrected: 4070 SUPER has 56 SMs, not 128 — 24 blocks = 43%,
   single wave; ≤1024 ctx is latency-bound and acceptable; ≥2048 is KV-BW-bound
   (16.8 MB unique at 2048, L2 dedups the 6-block groups) where thread-per-token's
   32-sector warp loads + 24 SMs likely underdeliver — measure with a ctx sweep.
10. Split-K v2 designed + coded: dim3(24,4) warp-per-row coalesced partials + 24-block
    deterministic combine (no atomics, graph-safe, ~97 KB scratch); enable at
    max_context>1024, expect ~33 µs + 3-5 µs at ctx 2048 vs a likely-much-worse simple
    path; keep the simple kernel as default until the bench says otherwise.
