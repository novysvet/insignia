# CPU FP8 compute tier (w3) — `insignia_cpu.hpp` design for RAM-resident Qwen3.8-27B-FP8 layers

Date: 2026-08-25. Scope: CPU-side kernels for the 27B layers that stay out of VRAM, on the
verified rig (Ryzen 5 5600X: Zen 3, 6C/12T, AVX2+FMA+F16C, **no AVX-512 / VNNI / fp16
arithmetic**; DDR4 dual-channel ≈ 37 GB/s real stream). Inputs read firsthand this session:
`AGENTS.md`, `audits/synthesis.md`, `audits/w2/ggml-cuda-cpu.md`, `audits/w2/shape-constants.md`,
`audits/w2/loader-27b-spec.md`, `audits/w3/fp8-kernels.md`, `audits/w3/loader-gaps.md`,
`src/fp8.cu`, `include/insignia_fp8.cuh`, `src/deltanet.cu`, `src/ops.cu`, `src/attention.cu`,
`src/qwen_kernels.cu`, `src/decode.cu` (via shape audit), `tools/index_safetensors.py`,
`Qwen3.8-27B-FP8/config.json`. All numeric claims (e4m3 decode exactness, scale fold,
exp polynomial) re-verified by exhaustive/200k-sample Python proofs this session (no numpy).
Read-only audit: nothing built, nothing committed; the only file written is this report.

---

## 0. TL;DR

1. **The CPU FP8 GEMV is DRAM-bound with ~11× compute headroom.** 6 cores × 2 FMA/cyc ×
   4.2 GHz × 8 lanes = 403 G weight-mult/s vs 37 G weights/s from DRAM. The brief's "CPU
   cannot FMA every weight byte" panic came from two errors (1 FMA per *byte* instead of per
   8-byte vector; single-core FMA rate compared against socket-wide bandwidth) — corrected
   rigorously in §2.
2. **e4m3→fp32 via the fp16-bit trick + F16C is the right decode**: exact for all 254 finite
   codes *including subnormals* (re-verified exhaustively), 18 vector ops + 4 FMA per 32
   weights ≈ 8 cycles of port time vs the 21.8-cycle DRAM budget per core — free.
3. The ×256 rides **inside the block scale**: `scale256 = bf16_to_f32(s) × 256` precomputed
   at layer setup. Exact bit form `bits += 0x04000000` (exponent += 8), guarded for ±0 scales
   (verified 0/200k mismatches; **not** 0x40000000 — that adds 2^128).
4. **LUT verdict**: scalar LUT (1 load/weight) cannot sustain 37 GB/s (96 loads / 32 weights
   = 48 cycles of load ports > 21.8-cycle budget). Gather LUT fits but at 12–20 cycles —
   3–6× the trick's consumption. Keep the trick; LUT kept in-code only as a parity/debug A-B.
5. Full implementable **`insignia_cpu.hpp`** in §4: fp8 GEMV (+pair/T=2 for MTP verify),
   bf16 GEMV, 6-worker persistent pool (packed-gen atomic tickets, no ABA, caller-must-finish
   progress guarantee), rmsnorm/gated-rmsnorm/silu_mul/conv1d/deltanet-step/GQA-decode
   (token-split online-softmax, f32 or bf16 KV)/qk-norm-rope, f64 parity reference.
6. Numbers: linear layer 382.73 MB F8 → **10.34 ms**; full-attn 372.24 MB → **10.06 ms**
   (+0.45 ms f32 KV @ctx2048); deltanet state traffic 9.4 MB → 0.26 ms. 23 CPU layers
   ≈ **245 ms/token clean, ~283 ms while NVMe staging steals DRAM** (§6).
7. Threading: 6 workers (1 per physical core) saturate DRAM; SMT siblings host main + IOCP.
   12 GEMV threads would add streams, not bandwidth.
8. Parity: raw fp32 accumulation per 128-col block, scale promoted once per block (same
   pattern as `fp8.cu`/TRT-LLM). Expected vs f64 ref: max rel ~5e-5, cos > 0.999999;
   CPU-vs-GPU threshold 1e-4 max-rel (§7).
9. Everything fits the 8-thread core budget: 6 GEMV + 1 IOCP + 1 main on 12 LPs.
10. Builds with plain `cl /arch:AVX2 /O2 /fp:precise` (MSVC never contracts mul+add, so the
    numerics are as written); per AGENTS.md, adoption still requires bench + parity vs the
    NumPy reference + a disassembly check that the inner loop stays spill-free.

---

## 1. Layer inventory (what the CPU tier actually has to compute)

From `audits/w2/loader-27b-spec.md` (verified shard census) and `config.json`:
hidden 5120, inter 17408, 64 layers = 48 linear-attention (gated DeltaNet) + 16 full-attention
(`(i&3)==3`), all F8_E4M3 weights + BF16 `weight_scale_inv` [ceil(r/128)][ceil(c/128)]
(0.012% overhead). a/b [48,5120], conv1d [10240,1,4], norms, A_log/dt_bias are bf16.
MTP stays on GPU. Decode is token-serial: per weight matrix, each byte is read **once**
per token (all mats > 24 MB; k/v_proj and a/b are L3-resident — see §6 note).

Linear layer: in_proj_qkv [10240,5120], in_proj_z [6144,5120], out_proj [5120,6144],
mlp gate/up [17408,5120]×2, down [5120,17408] (F8); in_proj_a/b [48,5120] bf16; conv1d;
48-head DeltaNet state 48×128×128 f32 = 3.15 MB.
Full-attn layer: q_proj [12288,5120] (q+gate interleaved per head), k/v_proj [1024,5120],
o_proj [5120,6144], same MLP (F8); KV cache f32 2×ctx×1024×4 = 16.78 MB @ctx2048.
(The brief's "231 MB" lower bound is a tier-split layer — some mats in VRAM; §6's per-mat
table makes any split computable.)

---

## 2. The budget, done rigorously

### 2.1 Machine constants (Zen 3 / Vermeer, per core)

| resource | rate | note |
|---|---|---|
| dispatch | 6 μops/cycle | 6-wide rename/dispatch |
| vector FMA | 2 × 256-bit/cycle | 2 FMA pipes, 4-cyc latency, 8 fp32 lanes each |
| vector ALU | 2 × 256-bit/cycle more | the 2 non-FMA FP pipes run `vpand/vpsllw/vpmovzx`-class ops — **do not contend with FMA pipes** |
| loads | 2 × 32 B/cycle | L1; L2→L1 32 B/cycle sustained |
| F16C | `vcvtph2ps ymm` ≈ 1/cycle (worst case 0.5) | convert-only (no fp16 FMA on Zen 3) |
| DRAM (socket) | ≈ 37 GB/s read | DDR4 dual-channel, real stream; ~32 GB/s when sharing with ~6.8 GB/s of NVMe→RAM staging writes |

### 2.2 Aggregate supply vs demand — the correct arithmetic

Per token, the weight stream is `W[rows,cols]·x[cols]` GEMVs, vectorized **over the
dot-product dimension**: one `vfmaddps ymm` consumes **8 e4m3 weights** (dequantized to
8 fp32 lanes) and 8 x values, accumulating 8 lane-partials.

- FMA demand: 1 vec-FMA per 8 weight-bytes → 37e9/8 = **4.6 G vec-FMA/s**.
- FMA supply: 6 cores × 2/cyc × 4.2 GHz = **50.4 G vec-FMA/s** → utilization **9.2%**, headroom **10.9×**.
- Loads: 5 loads per 32 weights (1 weight stream + 4 x vectors) → 0.156 loads/cyc/core vs 2 → **7.8%**.
- Total μops (dequant path, §3): ~28 per 32 weights → 1.28 μops/cyc/core vs 6-wide → **21%**.

**Every dimension sits below ~25%. The kernel is purely DRAM-bound.** Any dequant costing
under ~22 cycles per 32 weights is invisible; the chosen one costs ≤ 8.

### 2.3 The brief's panic, resolved (two fallacies)

The mission text talked itself into "CPU CANNOT FMA every weight byte at DRAM rate"
via: *"37G FMA/s needed vs 6.9G (3.45GHz×2)"* and later *"55.2G weight-mults/s vs 37G — 1.5×
headroom"*. Both are wrong in instructive ways:

1. **1 FMA per byte vs per 8 bytes.** A GEMV dot `Σ wᵢxᵢ` vectorized over *i* puts 8 weights
   in the 8 lanes of one FMA. Each e4m3 byte therefore costs 1/8 of a vector-FMA, not one.
   37 GB/s of e4m3 bytes needs only 4.6 G vec-FMA/s — not 37 G.
2. **Wrong denominator.** "55.2 G" is one core's lane-rate (6.9 G vec-FMA/s × 8) compared
   against the *whole socket's* DRAM. The per-core DRAM share with 6 workers is 6.17 GB/s
   (1.47 B/cycle), i.e. **21.8 cycles of budget per 32 weight bytes per core**. One core can
   FMA 16 B/cycle (2 FMA × 8 lanes) — 10.9× its DRAM share. Six threads exist for memory
   parallelism (each core sustains only ~12–15 GB/s of streams by itself), not for FMA throughput.

Recomputed cleanly: **fp32-FMA dequant-then-FMA is viable with 10.9× headroom**; integer
dp4a tricks (ggml's answer) are *not needed on the CPU* — unlike the GPU there is no
tensor-core rate to chase, only the DRAM stream, and fp32 lanes clear it by an order of
magnitude. (dp4a is useless for e4m3 anyway: non-affine in the byte, ggml-cuda-cpu.md §4.9.)

### 2.4 Activation (x) residency

x is fp32 [cols]: 20 KB (5120), 24.6 KB (6144), 40 KB (qkv rows view n/a — x is always
[cols]), 68 KB (17408). L1D is 32 KB: cols ≤ 6144 fits beside the weight stream only
partially (the stream evicts it); L2 is 512 KB per core and holds x in all cases.
x re-read per row = 4·cols bytes:
- cols=5120: 20 KB from L1/L2 per row; row DRAM time = 5120/1.47 = 3483 cyc; L1/L2 x time
  = 20480/32 = 640 cyc → 5.4× margin.
- cols=17408: 68 KB from L2 per row; 11842 cyc DRAM vs 2176 cyc L2 → 5.4× margin.
No x tiling needed at these ratios (a 2-rows-per-pass variant would halve it; not required —
kept out to keep the loop spill-free, §5.1).

### 2.5 Why 6 threads and not 12

Decode GEMV is a pure DRAM stream. Per-core stream rate tops out ~12–15 GB/s; 4–6 streams
already saturate the memory controller; beyond that, SMT siblings add outstanding misses,
not bandwidth, and steal cycles from the IOCP threads doing NVMe→RAM staging. The plan:
**6 persistent GEMV workers pinned 1-per-physical-core (LP 0–5), main thread + IOCP worker
threads on the SMT siblings (LP 6–11)** — 8 execution contexts on 12 LPs, matching the
brief's core budget. (colibri does the same split: main = coordinator+CPU compute, I/O
never on the compute team.)

---

## 3. e4m3→fp32 dequant: the bit trick vs the LUT

### 3.1 The chosen trick (exact, subnormals included)

Build fp16 bits per byte: `h = ((b&0x7f)<<7) | ((b&0x80)<<8)`, convert with F16C
(`vcvtph2ps`), and fold the ×256 into the block scale (§3.2). Proof of exactness (and
re-verified exhaustively this session, **0/254 finite codes mismatch**, including both signs
and all subnormals; spot values 0x00→0, 0x01→2⁻⁹, 0x07→7·2⁻⁹, 0x08→2⁻⁶, 0x38→1.0,
0x7E→448, 0xFE→−448):

- Normal (E∈1..15): constructed fp16 has exponent field E (bits 10–13) and mantissa m at
  bits 7–9 → value (1+m/8)·2^(E−15); ×256 → (1+m/8)·2^(E−7) — the exact OCP E4M3 value,
  3 mantissa bits drop in without rounding.
- Subnormal (E=0): fp16 subnormal m·2⁻¹⁷ ×256 = m·2⁻⁹ — exact, since fp16 subnormal steps
  (2⁻²⁴) divide e4m3 subnormal steps (2⁻¹⁷).
- Max intermediate 1.875 pre-scale (480 after) — nowhere near fp16/fp32 limits.
- e4m3's only NaN codes 0x7F/0xFF decode to ±480 instead of NaN (documented; weight
  checkpoints contain none).

On AVX2, process 32 weights (one `__m256i`) at a time:

| op | count/32 weights | notes |
|---|---|---|
| `vmovdqu ymm` (weights) | 1 | unaligned; see §5.2 |
| `vpmovzxbw` | 2 | 16 bytes → 16 u16 lanes |
| `vpand`/`vpsllw`/`vpor` | 10 | 5 per 16 lanes: `(x&0x7f)<<7 \| (x<<8)&0x8000` |
| `vextracti128` | 2 | high halves |
| `vcvtph2ps` (F16C) | 4 | exact |
| `vmovups` (x, fp32) | 4 | 128 B of activation per 32 weights |
| `vfmaddps` | 4 | 2/cyc, FMA pipes |
| `_mm_prefetch` | 1 | optional 256 B ahead |
| **total** | **28 μops / 32 weights** | **≤ 8 cycles of port time** (cvtph-chain worst case 8c if 0.5/cyc; dispatch 28/6 = 4.7c) vs **21.8-cycle DRAM budget** → 21% issue utilization, free. |

### 3.2 The ×256 fold — into the scale, not the weights

The GPU kernel multiplies each converted value by 256 (`fp8.cu` does it in the two FMULs of
`e4m3x2`). On the CPU, dequantized weights are accumulated **raw** (unscaled) per 128-col
block and the block scale is applied once per block — so ×256 multiplies the *scale*, once
per block, at setup:

```cpp
// exact: bf16→f32 has a zero low mantissa, so adding 8 to the exponent field never rounds.
inline float bf16_scale_x256(uint16_t u) {
    const uint32_t b = uint32_t(u) << 16;
    if (!(b << 1)) return 0.f;        // ±0 guard: bit-add would yield 2^8
    const uint32_t r = b + 0x04000000u;   // exponent += 8  ==  ×256
    float f; memcpy(&f, &r, sizeof f); return f;
}
```

Verified this session: 200 000 random bf16-rounded scales in [1e-9, 1e2] (both signs),
**0 mismatches** vs `float(scale)*256.0f`. Two notes: the magic is `0x0400_0000` (8·2²³) —
a sibling report's `0x400000` (w3 fp8-kernels.md perf item 4) is wrong by 64×; and scales
near bf16 max (2^119+) would overflow — real block scales live around 1e-3…1, irrelevant.

### 3.3 The LUT alternative, honestly priced

**256-entry f32 LUT (1 KB, L1-resident).**

- *Scalar LUT (1 load per weight)*: per weight ≈ 3 loads (byte, LUT, x) + 1 FMA + extract
  overhead ≈ 5 μops → per 32 weights ≈ 96 loads (48 cycles of load ports at 2/cyc) +
  ~160 μops (27 cycles of dispatch) **> 21.8-cycle budget → cannot sustain 37 GB/s even at
  6 threads. REJECTED.** (This is the brief's "8 loads+8 FMA per 8 weights ≈ 8 cycles,
  2.4× over budget" — confirmed.)
- *Gather LUT (`vpgatherdps`)*: per 8 weights 1 `vpmovzxbd` + 1 gather + 1 FMA + x load.
  Zen 3 gathers are microcoded (~8 load μops + merge), ≈3–5 cycles throughput each →
  **12–20 cycles per 32 weights** — fits the 21.8-cycle budget but consumes 3–6× what the
  trick does, crowds the load ports, and its ~12-cycle latency forces deep unrolling to hide.
- *pshufb LUT*: only works for 4-bit formats (16 entries); e4m3 has 256 codes — not applicable
  (ggml uses it for MXFP4, not e4m3).

**Verdict: bit trick.** It is simultaneously the cheapest, the only one exact on subnormals,
and dependency-free pure dataflow. The LUT (gather) is kept in the header behind
`INSIG_CPU_FP8_LUT` purely as a parity/A-B debugging switch.

---

## 4. `insignia_cpu.hpp` — complete source

Design constraints honored: single self-contained header, MSVC 19.51 x64, `/arch:AVX2 /O2
/fp:precise` (MSVC does not contract — every FMA in the source is explicit), no CUDA
dependencies, engine conventions (`namespace insignia`, bf16 checkpoint dtypes, weight-scale
layout `[r/128][c/128]` col-major-block row-major, GQA `kvh = head/6`, deltanet `kh = head/3`).

```cpp
// ============================================================================
//  insignia_cpu.hpp — CPU compute tier for RAM-resident Qwen3.8-27B-FP8 layers
//  Target: AMD Ryzen 5 5600X (Zen 3, 6C/12T, AVX2+FMA+F16C; NO AVX-512 / VNNI /
//  fp16 arithmetic). MSVC 19.51 x64, /arch:AVX2 /O2 /fp:precise (no contraction).
//
//  DRAM budget model: ~37 GB/s socket read; per core (6 workers @4.2 GHz) that is
//  1.47 B/cycle = 21.8 cycles per 32 weight bytes. The GEMV inner block is ~28 uops
//  (<= 8 cycles of port time) — memory-bound with ~11x FMA headroom. Analysis:
//  audits/w3/cpu-fp8.md. Per AGENTS.md: adopt after bench + NumPy parity + disasm.
// ============================================================================
#pragma once
#if !defined(__AVX2__) || !defined(__FMA__) || !defined(__F16C__)
#error "insignia_cpu.hpp needs /arch:AVX2 (implies FMA+F16C on MSVC x64)"
#endif

#include <immintrin.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#ifndef INSIG_CPU_FP8_LUT
#define INSIG_CPU_FP8_LUT 0        // 1 = debug A/B via 256-entry f32 LUT (see audit §3.3)
#endif
#ifndef INSIG_PREFETCH_DIST
#define INSIG_PREFETCH_DIST 256    // bytes ahead for weight prefetch; 0 disables
#endif

namespace insignia {

// ─────────────────────────── scalar/bit helpers ───────────────────────────

inline float bf16_to_f32(uint16_t u) {
    const uint32_t b = uint32_t(u) << 16;
    float f; memcpy(&f, &b, sizeof f);
    return f;
}

// Block scale with the e4m3->f32 x256 FOLDED IN (audit §3.2; exact; +-0 guarded).
inline float bf16_scale_x256(uint16_t u) {
    const uint32_t b = uint32_t(u) << 16;
    if (!(b << 1)) return 0.f;                 // +-0: bit-add would yield 2^8
    const uint32_t r = b + 0x04000000u;        // exponent += 8  ==  x256, no rounding
    float f; memcpy(&f, &r, sizeof f);
    return f;
}

// Checkpoint scales (bf16 [r/128][c/128], i.e. "weight_scale_inv") -> f32 x256.
// Run once per layer load; output stays hot (largest mat: 80x40 floats = 12.8 KB).
inline void fp8_prepare_scales(const uint16_t *__restrict s_bf16, float *__restrict out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = bf16_scale_x256(s_bf16[i]);
}

// bf16 -> f32 widen, 8 lanes: (u16 << 16) bitcast. 1 load + 2 ops per 8 values.
inline __m256 bf16_widen8(const uint16_t *__restrict p) {
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)p)), 16));
}

inline float hsum256_ps(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    __m128 d = _mm_movehdup_ps(s);
    s = _mm_add_ps(s, d);
    d = _mm_movehl_ps(d, s);                   // lane0 = s[2]
    s = _mm_add_ss(s, d);
    return _mm_cvtss_f32(s);
}

// ───────────────────── e4m3 -> fp32 dequant (audit §3.1) ─────────────────────
//
// 32 e4m3 bytes -> 4 __m256 fp32, EXACT for all finite codes incl. subnormals.
// fp16 pattern per byte: mag = (b&0x7f)<<7 (bits 7..13), sign -> bit 15; F16C converts;
// the x256 is folded into the block scale (bf16_scale_x256), NOT applied here.
// 18 vector ops per 32 weights: 2 vpmovzxbw + 10 and/sll/or + 2 vextracti128 + 4 vcvtph2ps.
inline void e4m3x32_f32(const __m256i w, __m256 *__restrict y) {
    const __m256i m7 = _mm256_set1_epi16(0x007f);
    const __m256i s8 = _mm256_set1_epi16((short)0x8000);
    const __m256i lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(w));
    const __m256i hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(w, 1));
    const __m256i l16 = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(lo, m7), 7),
                                        _mm256_and_si256(_mm256_slli_epi16(lo, 8), s8));
    const __m256i h16 = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(hi, m7), 7),
                                        _mm256_and_si256(_mm256_slli_epi16(hi, 8), s8));
    y[0] = _mm256_cvtph_ps(_mm256_castsi256_si128(l16));
    y[1] = _mm256_cvtph_ps(_mm256_extracti128_si256(l16, 1));
    y[2] = _mm256_cvtph_ps(_mm256_castsi256_si128(h16));
    y[3] = _mm256_cvtph_ps(_mm256_extracti128_si256(h16, 1));
}

// Debug alternative: 256-entry f32 LUT (1 KB, L1-resident) via gather. ~3-6x the cycles
// of the trick on Zen 3 (microcoded gathers); kept ONLY for parity A/B. Audit §3.3.
struct alignas(64) Fp8Lut {
    float v[256];
    Fp8Lut() {
        for (int b = 0; b < 256; ++b) {
            const unsigned short h = (unsigned short)(((b & 0x7f) << 7) | ((b & 0x80) << 8));
            v[b] = _cvtsh_ss(h) * 256.f;       // same values, table form
        }
    }
};
inline const Fp8Lut &fp8_lut() { static const Fp8Lut t; return t; }

inline void e4m3x32_f32_lut(const __m256i w, const float *__restrict lut, __m256 *__restrict y) {
    const __m128i lo = _mm256_castsi256_si128(w), hi = _mm256_extracti128_si256(w, 1);
    const __m128i b[4] = {lo, _mm_srli_si128(lo, 8), hi, _mm_srli_si128(hi, 8)};
    for (int q = 0; q < 4; ++q)
        y[q] = _mm256_i32gather_ps(lut, _mm256_cvtepu8_epi32(b[q]), 4);
}

// ───────────────────────── vector exp / silu / sigmoid ─────────────────────────
//
// exp(x) = 2^(x·log2e); n=rint, f∈[-½,½]; 2^f ≈ Remez deg-4 (max rel err 2.6e-6,
// fitted this session; CUDA __expf-class accuracy, far below fp8 noise 2^-4.3 rms).
// x clamped to ±87.3 so n ∈ [-126,126] (2^n never overflows/underflows fp32).
inline __m256 vexp256_ps(__m256 x) {
    const __m256 c0 = _mm256_set1_ps(0.9999992617f), c1 = _mm256_set1_ps(0.6931214847f);
    const __m256 c2 = _mm256_set1_ps(0.2402472204f), c3 = _mm256_set1_ps(0.05591962983f);
    const __m256 c4 = _mm256_set1_ps(0.009571308824f);
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.3f)), _mm256_set1_ps(87.3f));
    const __m256 t = _mm256_mul_ps(x, _mm256_set1_ps(1.4426950408889634f));
    const __m256i n = _mm256_cvtps_epi32(t);                    // round-to-nearest-even
    const __m256 f = _mm256_sub_ps(t, _mm256_cvtepi32_ps(n));
    __m256 p = _mm256_fmadd_ps(f, c4, c3);
    p = _mm256_fmadd_ps(f, p, c2);
    p = _mm256_fmadd_ps(f, p, c1);
    p = _mm256_fmadd_ps(f, p, c0);
    const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(n, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

inline __m256 vsigmoid256_ps(__m256 x) {
    const __m256 e = vexp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), x));
    return _mm256_div_ps(_mm256_set1_ps(1.f), _mm256_add_ps(_mm256_set1_ps(1.f), e));
}

// ───────────────────────────── thread pool (§5.3) ─────────────────────────────
//
// 6 persistent workers (1 per physical core; LP 0..5 assumed primary threads),
// main + IOCP threads stay on the SMT siblings. Jobs are serial (decode is serial):
// exactly one launch() at a time; the caller participates and is the progress
// guarantee — even if every worker sleeps, the caller finishes all tickets alone.
//
// Ticket claiming is a single packed atomic claim_ = (gen<<32)|next_ticket. The
// generation shares the word with the counter, so a straggler's CAS against a
// finished generation can never succeed (no ABA, no stale-fn execution: fn/ctx of
// gen g are immutable until every ticket of g has executed, because left_ hits 0
// only after the last fn returns, and the dispatcher publishes g+1 only after that).
class CpuPool {
public:
    using Job = void (*)(void *ctx, int ticket);

    static CpuPool &get() { static CpuPool p; return p; }

    int threads() const { return nthreads_; }

    // Blocking fan-out over [0,tickets). NOT reentrant; one dispatcher thread only.
    void launch(Job fn, void *ctx, int tickets, bool caller_helps = true) {
        if (tickets <= 0) return;
        std::lock_guard<std::mutex> disp(launch_mut_);
        const uint64_t g = ++gen_;
        Slot &s = slots_[g & 1];
        s.fn = fn; s.ctx = ctx; s.n = tickets;
        s.left.store(tickets, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(m_);
            claim_.store(g << 32, std::memory_order_release);   // publish: new gen, ticket 0
            cv_.notify_all();
        }
        if (caller_helps) drive(g);
        uint64_t spin = 0;
        while (s.left.load(std::memory_order_acquire) > 0) {
            if (++spin < (1u << 15)) { _mm_pause(); drive(g); }
            else {
                std::unique_lock<std::mutex> lk(m_);
                cvd_.wait_for(lk, std::chrono::microseconds(500),
                              [&] { return s.left.load(std::memory_order_acquire) == 0; });
                drive(g);
            }
        }
    }

private:
    struct alignas(64) Slot {
        std::atomic<int> left{0};
        int n{0};
        Job fn{nullptr};
        void *ctx{nullptr};
    };

    void drive(uint64_t g) {                     // claim+execute tickets of generation g
        Slot &s = slots_[g & 1];
        for (;;) {
            uint64_t c = claim_.load(std::memory_order_acquire);
            if ((c >> 32) != g) return;          // gen moved on; this job is over
            const uint64_t t = c & 0xffffffffu;
            if (t >= uint64_t(s.n)) return;      // all tickets claimed
            if (!claim_.compare_exchange_weak(c, (g << 32) | (t + 1),
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire)) continue;
            s.fn(s.ctx, int(t));
            if (s.left.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lk(m_);
                cvd_.notify_all();
            }
        }
    }

    void worker_main() {
        uint64_t seen = 0;
        for (;;) {
            // brief spin first (small parallel ops, e.g. the 27us a/b GEMV, must not
            // pay a full wake/park cycle), then park on generation change.
            for (uint64_t sp = 0; sp < 4096; ++sp) {
                _mm_pause();
                const uint64_t g = claim_.load(std::memory_order_acquire) >> 32;
                if (g != seen && g != 0) { seen = g; drive(g); sp = 0; }
                if (stop_.load(std::memory_order_acquire)) return;
            }
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait(lk, [&] {
                if (stop_.load(std::memory_order_acquire)) return true;
                const uint64_t g = claim_.load(std::memory_order_acquire) >> 32;
                return g != seen && g != 0;
            });
            if (stop_.load(std::memory_order_acquire)) return;
            const uint64_t g = claim_.load(std::memory_order_acquire) >> 32;
            if (g != seen && g != 0) { seen = g; lk.unlock(); drive(g); }
        }
    }

    CpuPool() {
        unsigned hc = std::thread::hardware_concurrency();     // 12 on 5600X
        nthreads_ = std::max(1u, std::min(6u, hc / 2 ? hc / 2 : 1));
        for (int i = 0; i < nthreads_; ++i) {
            th_.emplace_back([this] { worker_main(); });
#if defined(_WIN32)
            // Zen CCX enumeration: LP 0..5 = physical cores 0..5, LP 6..11 = siblings.
            // Pin workers one per physical core; main + IOCP inherit the rest.
            SetThreadAffinityMask(th_.back().native_handle(), DWORD_PTR(1) << i);
#endif
        }
    }
    ~CpuPool() {
        { std::lock_guard<std::mutex> lk(m_); stop_.store(true, std::memory_order_release); cv_.notify_all(); }
        for (auto &t : th_) if (t.joinable()) t.join();
    }

    std::mutex m_, launch_mut_;
    std::condition_variable cv_, cvd_;
    std::atomic<uint64_t> claim_{0};             // (gen<<32) | next_ticket
    std::atomic<bool> stop_{false};
    Slot slots_[2];
    uint64_t gen_{0};                            // dispatcher-only (under launch_mut_)
    std::vector<std::thread> th_;
    int nthreads_{0};
};

// ─────────────────────── FP8 block-scaled GEMV (§5.1) ───────────────────────
//
// y[r] = (Σ_c W[r,c]·x[c]) with per-128x128-block scales (x256-folded, f32).
// Inner: per 128-col scale block, 4 chunks of 32 weights; raw lane accumulators a0..a3,
// one horizontal sum + one scale FMA per block (TRT-LLM promote pattern = fp8.cu shape).
// ~28 uops + prefetch per 32 weights vs 21.8-cycle DRAM budget — memory-bound.
struct Fp8GemvJob {
    const uint8_t *__restrict w;      // e4m3 [rows, cols] row-major (any alignment)
    const float *__restrict s256;     // f32 scales x256 [rows>>7][cols>>7] (fp8_prepare_scales)
    const float *__restrict x0;       // fp32 [cols] (L1/L2-resident across rows)
    const float *__restrict x1;       // pair mode: second row, or nullptr
    float *__restrict y0;             // [rows]
    float *__restrict y1;             // pair mode: [rows], or nullptr
    int rows, cols, rpt;              // rows per ticket (10..32; see wrappers)
    bool pair;
};

#if INSIG_CPU_FP8_LUT
#define INSIG_F8_DEQ(WR, WV) e4m3x32_f32_lut(_mm256_loadu_si256((const __m256i *)(WR)), fp8_lut().v, (WV))
#else
#define INSIG_F8_DEQ(WR, WV) e4m3x32_f32(_mm256_loadu_si256((const __m256i *)(WR)), (WV))
#endif

static void fp8_gemv_rowrange(const Fp8GemvJob &j, int r0, int r1) {
    const int kb = j.cols >> 7;
    for (int r = r0; r < r1; ++r) {
        const uint8_t *__restrict wr = j.w + size_t(r) * j.cols;
        const float *__restrict sr = j.s256 + size_t(r >> 7) * kb;
        float acc = 0.f;
        for (int c = 0; c < j.cols; c += 128) {
            const float *__restrict xp = j.x0 + c;
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(),
                   a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            _mm_prefetch((const char *)wr + c + INSIG_PREFETCH_DIST, _MM_HINT_T0);
            { __m256 wv[4]; INSIG_F8_DEQ(wr + c, wv);
              a0 = _mm256_fmadd_ps(wv[0], _mm256_loadu_ps(xp + 0), a0);
              a1 = _mm256_fmadd_ps(wv[1], _mm256_loadu_ps(xp + 8), a1);
              a2 = _mm256_fmadd_ps(wv[2], _mm256_loadu_ps(xp + 16), a2);
              a3 = _mm256_fmadd_ps(wv[3], _mm256_loadu_ps(xp + 24), a3); }
            { __m256 wv[4]; INSIG_F8_DEQ(wr + c + 32, wv);
              a0 = _mm256_fmadd_ps(wv[0], _mm256_loadu_ps(xp + 32), a0);
              a1 = _mm256_fmadd_ps(wv[1], _mm256_loadu_ps(xp + 40), a1);
              a2 = _mm256_fmadd_ps(wv[2], _mm256_loadu_ps(xp + 48), a2);
              a3 = _mm256_fmadd_ps(wv[3], _mm256_loadu_ps(xp + 56), a3); }
            { __m256 wv[4]; INSIG_F8_DEQ(wr + c + 64, wv);
              a0 = _mm256_fmadd_ps(wv[0], _mm256_loadu_ps(xp + 64), a0);
              a1 = _mm256_fmadd_ps(wv[1], _mm256_loadu_ps(xp + 72), a1);
              a2 = _mm256_fmadd_ps(wv[2], _mm256_loadu_ps(xp + 80), a2);
              a3 = _mm256_fmadd_ps(wv[3], _mm256_loadu_ps(xp + 88), a3); }
            { __m256 wv[4]; INSIG_F8_DEQ(wr + c + 96, wv);
              a0 = _mm256_fmadd_ps(wv[0], _mm256_loadu_ps(xp + 96), a0);
              a1 = _mm256_fmadd_ps(wv[1], _mm256_loadu_ps(xp + 104), a1);
              a2 = _mm256_fmadd_ps(wv[2], _mm256_loadu_ps(xp + 112), a2);
              a3 = _mm256_fmadd_ps(wv[3], _mm256_loadu_ps(xp + 120), a3); }
            const __m256 p01 = _mm256_add_ps(a0, a1), p23 = _mm256_add_ps(a2, a3);
            acc += (hsum256_ps(p01) + hsum256_ps(p23)) * sr[c >> 7];   // scale promote/block
        }
        j.y0[r] = acc;
    }
}

// Pair (T=2, MTP verify): one weight pass, two outputs — +8 FMA +8 loads per 32 weights,
// still nothing vs the 21.8-cycle budget.
static void fp8_gemv2_rowrange(const Fp8GemvJob &j, int r0, int r1) {
    const int kb = j.cols >> 7;
    for (int r = r0; r < r1; ++r) {
        const uint8_t *__restrict wr = j.w + size_t(r) * j.cols;
        const float *__restrict sr = j.s256 + size_t(r >> 7) * kb;
        float acc0 = 0.f, acc1 = 0.f;
        for (int c = 0; c < j.cols; c += 128) {
            const float *__restrict xa = j.x0 + c, *__restrict xb = j.x1 + c;
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(),
                   a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps(),
                   b2 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
            _mm_prefetch((const char *)wr + c + INSIG_PREFETCH_DIST, _MM_HINT_T0);
#define INSIG_PAIR_CHUNK(K)                                                        \
    { __m256 wv[4]; INSIG_F8_DEQ(wr + c + (K) * 32, wv);                            \
      a0 = _mm256_fmadd_ps(wv[0], _mm256_loadu_ps(xa + (K) * 32 + 0), a0);          \
      a1 = _mm256_fmadd_ps(wv[1], _mm256_loadu_ps(xa + (K) * 32 + 8), a1);          \
      a2 = _mm256_fmadd_ps(wv[2], _mm256_loadu_ps(xa + (K) * 32 + 16), a2);         \
      a3 = _mm256_fmadd_ps(wv[3], _mm256_loadu_ps(xa + (K) * 32 + 24), a3);         \
      b0 = _mm256_fmadd_ps(wv[0], _mm256_loadu_ps(xb + (K) * 32 + 0), b0);          \
      b1 = _mm256_fmadd_ps(wv[1], _mm256_loadu_ps(xb + (K) * 32 + 8), b1);          \
      b2 = _mm256_fmadd_ps(wv[2], _mm256_loadu_ps(xb + (K) * 32 + 16), b2);         \
      b3 = _mm256_fmadd_ps(wv[3], _mm256_loadu_ps(xb + (K) * 32 + 24), b3); }
            INSIG_PAIR_CHUNK(0)
            INSIG_PAIR_CHUNK(1)
            INSIG_PAIR_CHUNK(2)
            INSIG_PAIR_CHUNK(3)
#undef INSIG_PAIR_CHUNK
            const __m256 p01 = _mm256_add_ps(a0, a1), p23 = _mm256_add_ps(a2, a3);
            const __m256 q01 = _mm256_add_ps(b0, b1), q23 = _mm256_add_ps(b2, b3);
            const float sc = sr[c >> 7];
            acc0 += (hsum256_ps(p01) + hsum256_ps(p23)) * sc;
            acc1 += (hsum256_ps(q01) + hsum256_ps(q23)) * sc;
        }
        j.y0[r] = acc0;
        j.y1[r] = acc1;
    }
}

static void fp8_gemv_ticket(void *p, int t) {
    const Fp8GemvJob &j = *(const Fp8GemvJob *)p;
    const int r0 = t * j.rpt;
    if (r0 >= j.rows) return;
    const int r1 = std::min(j.rows, r0 + j.rpt);
    if (j.pair) fp8_gemv2_rowrange(j, r0, r1);
    else fp8_gemv_rowrange(j, r0, r1);
}

// Single-token GEMV. rows arbitrary; cols % 128 == 0 (all 27B mats qualify).
inline void fp8_gemv_mt(const uint8_t *w, const float *s256, const float *x, float *y,
                        int rows, int cols) {
    if (rows <= 0 || cols <= 0 || (cols & 127))
        throw std::runtime_error("insignia cpu: bad fp8 gemv dims");
    Fp8GemvJob j{w, s256, x, nullptr, y, nullptr, rows, cols,
                 std::max(1, std::min(32, rows / 96)), false};
    CpuPool::get().launch(fp8_gemv_ticket, &j, (rows + j.rpt - 1) / j.rpt);
}

// Pair GEMV (spec verify T=2): x = [2, cols] contiguous, y = [2, rows].
inline void fp8_gemv2_mt(const uint8_t *w, const float *s256, const float *x, float *y,
                         int rows, int cols) {
    if (rows <= 0 || cols <= 0 || (cols & 127))
        throw std::runtime_error("insignia cpu: bad fp8 gemv dims");
    Fp8GemvJob j{w, s256, x, x + cols, y, y + rows, rows, cols,
                 std::max(1, std::min(32, rows / 96)), true};
    CpuPool::get().launch(fp8_gemv_ticket, &j, (rows + j.rpt - 1) / j.rpt);
}

// ───────────────────────────── bf16 GEMV (§5.5) ─────────────────────────────
// For in_proj_a/b [48,5120] and any bf16 mat (mtp.fc shape too, if ever needed).
// Per 16 weights: 1 load + 2 widen chains (cvt+slli) + 1 extract + 2 FMA = ~10 uops.
struct Bf16GemvJob {
    const uint16_t *__restrict w;
    const float *__restrict x;
    float *__restrict y;
    int rows, cols, rpt;
};
static void bf16_gemv_rowrange(const Bf16GemvJob &j, int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
        const uint16_t *__restrict wr = j.w + size_t(r) * j.cols;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        float acc = 0.f;
        for (int c = 0; c < j.cols; c += 16) {
            const __m256i p = _mm256_loadu_si256((const __m256i *)(wr + c));
            const __m256 w0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(p)), 16));
            const __m256 w1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(p, 1)), 16));
            a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(j.x + c), a0);
            a1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(j.x + c + 8), a1);
            acc += hsum256_ps(_mm256_add_ps(a0, a1));            // per-16 promote (cheap, tiny mats)
            a0 = _mm256_setzero_ps(); a1 = _mm256_setzero_ps();
        }
        j.y[r] = acc;
    }
}
static void bf16_gemv_ticket(void *p, int t) {
    const Bf16GemvJob &j = *(const Bf16GemvJob *)p;
    const int r0 = t * j.rpt;
    if (r0 >= j.rows) return;
    bf16_gemv_rowrange(j, r0, std::min(j.rows, r0 + j.rpt));
}
inline void bf16_gemv_mt(const uint16_t *w, const float *x, float *y, int rows, int cols) {
    if (rows <= 0 || cols <= 0 || (cols & 15)) throw std::runtime_error("insignia cpu: bad bf16 gemv dims");
    Bf16GemvJob j{w, x, y, rows, cols, std::max(1, std::min(32, rows / 96))};
    CpuPool::get().launch(bf16_gemv_ticket, &j, (rows + j.rpt - 1) / j.rpt);
}

// ───────────────────────────── small ops (§5.4) ─────────────────────────────

// RMSNorm over one row; weight bf16 (checkpoint norm weights). zero_centered mirrors
// rms_bf (1+w vs w). Scalar rsqrt for parity with rsqrtf-class accuracy.
inline void rmsnorm_cpu(const float *__restrict x, const uint16_t *__restrict w,
                        float *__restrict y, int cols, bool zero_centered, float eps = 1e-6f) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps(),
           s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
    for (int c = 0; c < cols; c += 32) {
        const __m256 v0 = _mm256_loadu_ps(x + c), v1 = _mm256_loadu_ps(x + c + 8);
        const __m256 v2 = _mm256_loadu_ps(x + c + 16), v3 = _mm256_loadu_ps(x + c + 24);
        s0 = _mm256_fmadd_ps(v0, v0, s0); s1 = _mm256_fmadd_ps(v1, v1, s1);
        s2 = _mm256_fmadd_ps(v2, v2, s2); s3 = _mm256_fmadd_ps(v3, v3, s3);
    }
    const float ss = hsum256_ps(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
    const float inv = 1.f / std::sqrt(ss / float(cols) + eps);
    for (int c = 0; c < cols; c += 16) {
        const __m256 w0 = bf16_widen8(w + c), w1 = bf16_widen8(w + c + 8);
        __m256 z0 = _mm256_mul_ps(_mm256_loadu_ps(x + c), _mm256_set1_ps(inv));
        __m256 z1 = _mm256_mul_ps(_mm256_loadu_ps(x + c + 8), _mm256_set1_ps(inv));
        if (zero_centered) {
            z0 = _mm256_mul_ps(z0, _mm256_add_ps(_mm256_set1_ps(1.f), w0));
            z1 = _mm256_mul_ps(z1, _mm256_add_ps(_mm256_set1_ps(1.f), w1));
        } else {
            z0 = _mm256_mul_ps(z0, w0);
            z1 = _mm256_mul_ps(z1, w1);
        }
        _mm256_storeu_ps(y + c, z0);
        _mm256_storeu_ps(y + c + 8, z1);
    }
}

// Gated per-head RMSNorm+silu-gate (linear layers): x,y,gate are [heads][128].
// Mirrors rmsnorm_gated_silu: y = x·rsqrt(mean(x²)+ε)·w · silu(gate).
inline void gated_rmsnorm_per_head_cpu(const float *__restrict x, const uint16_t *__restrict w,
                                       const float *__restrict gate, float *__restrict y,
                                       int heads, int hd = 128, float eps = 1e-6f) {
    const int nv = hd >> 3;                        // 16 vecs per head at hd=128
    for (int h = 0; h < heads; ++h, x += hd, gate += hd, y += hd, w += hd) {
        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
        for (int v = 0; v < nv; v += 2) {
            const __m256 a = _mm256_loadu_ps(x + v * 8), b = _mm256_loadu_ps(x + v * 8 + 8);
            s0 = _mm256_fmadd_ps(a, a, s0);
            s1 = _mm256_fmadd_ps(b, b, s1);
        }
        const float ss = hsum256_ps(_mm256_add_ps(s0, s1));
        const float inv = 1.f / std::sqrt(ss / float(hd) + eps);
        for (int v = 0; v < nv; v++) {
            const __m256 z = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + v * 8), _mm256_set1_ps(inv)),
                                           bf16_widen8(w + v * 8));
            const __m256 g = _mm256_loadu_ps(gate + v * 8);
            const __m256 sg = _mm256_div_ps(g, _mm256_add_ps(_mm256_set1_ps(1.f),
                                   vexp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), g))));
            _mm256_storeu_ps(y + v * 8, _mm256_mul_ps(z, sg));
        }
    }
}

// y = silu(g)·u  (n % 8 == 0; 5120/6144/17408 all qualify)
inline void silu_mul_cpu(const float *__restrict g, const float *__restrict u,
                         float *__restrict y, int n) {
    for (int i = 0; i < n; i += 8) {
        const __m256 gv = _mm256_loadu_ps(g + i);
        const __m256 sg = _mm256_div_ps(gv, _mm256_add_ps(_mm256_set1_ps(1.f),
                               vexp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), gv))));
        _mm256_storeu_ps(y + i, _mm256_mul_ps(sg, _mm256_loadu_ps(u + i)));
    }
}

// x *= sigmoid(g) (attention output gate)
inline void sigmoid_mul_cpu(float *__restrict x, const float *__restrict g, int n) {
    for (int i = 0; i < n; i += 8)
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vsigmoid256_ps(_mm256_loadu_ps(g + i))));
}

inline void residual_add_cpu(float *__restrict x, const float *__restrict d, int n) {
    for (int i = 0; i < n; i += 8)
        _mm256_storeu_ps(x + i, _mm256_add_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(d + i)));
}

// Causal conv1d (width 4) + silu + state shift. State layout [ch][3] matches the engine
// (conv_state[c*3+i]); weights pre-expanded at layer load to f32 wt[tap][ch] (transpose of
// the checkpoint's [ch][4] bf16 — do it in expand_conv_weights below). Scalar body: the
// [ch][3] state makes vectorization a gather; 10240 ch ≈ 30 us serial = 0.3% of a layer.
inline void causal_conv4_silu_cpu(float *__restrict x, float *__restrict state,
                                  const float *__restrict wt /*[4][ch]*/, int ch) {
    for (int c = 0; c < ch; ++c) {
        float *__restrict st = state + c * 3;
        const float z = st[0] * wt[c] + st[1] * wt[ch + c] + st[2] * wt[2 * ch + c] + x[c] * wt[3 * ch + c];
        st[0] = st[1]; st[1] = st[2]; st[2] = x[c];
        x[c] = z / (1.f + expf(-z));
    }
}
// checkpoint conv1d [ch][4] bf16 -> f32 [4][ch] (setup)
inline void expand_conv_weights(const uint16_t *__restrict src, float *__restrict dst, int ch) {
    for (int tap = 0; tap < 4; ++tap)
        for (int c = 0; c < ch; ++c)
            dst[tap * ch + c] = bf16_to_f32(src[c * 4 + tap]);
}

// a/b gating params (48 heads): b = sigmoid(b); a = -exp(A_log)·softplus(a + dt_bias).
// Mirrors src/qwen_kernels.cu params kernel exactly (including the z>20 overflow guard).
inline void deltanet_parameters_cpu(float *__restrict a, float *__restrict b,
                                    const float *__restrict A_log, const uint16_t *__restrict dt_bias,
                                    int heads) {
    for (int h = 0; h < heads; ++h) {
        b[h] = 1.f / (1.f + expf(-b[h]));
        const float z = a[h] + bf16_to_f32(dt_bias[h]);
        const float soft = z > 20.f ? z : log1pf(expf(z));
        a[h] = -expf(A_log[h]) * soft;
    }
}

// ─────────────────────── DeltaNet recurrent step (§5.4) ───────────────────────
// 48 v-heads / 16 k-heads (kh = head/3 for 27B — NOT >>1), state per head 128×128 f32.
// Mirrors src/deltanet.cu including the asymmetric q-norm constant 1/sqrt(128)·rsqrt
// folded on q only. Two passes over S (S·(k̂·decay) then fused update+S·q̂):
// 3 vec-FMA per state element, traffic 3×64 KB per head = 9.4 MB/layer → 0.26 ms DRAM,
// 0.295 M vec-FMA/layer → ~6 us compute. Parallel across heads via the pool.
struct DeltaJob {
    float *__restrict state;              // [48][128][128]
    const float *__restrict q, *__restrict k, *__restrict v;  // q,k: [16][128]; v: [48][128]
    const float *__restrict g, *__restrict b;                  // a,b after parameters
    float *__restrict out;                // [48][128]
    int heads;
};
static void deltanet_head_step(float *__restrict S, const float *__restrict qh_,
                               const float *__restrict kh_, const float *__restrict v,
                               float g, float beta, float *__restrict out) {
    __m256 sq0 = _mm256_setzero_ps(), sk0 = _mm256_setzero_ps();
    for (int i = 0; i < 128; i += 16) {
        for (int j = 0; j < 2; ++j) {
            const __m256 q = _mm256_loadu_ps(qh_ + i + j * 8), k = _mm256_loadu_ps(kh_ + i + j * 8);
            sq0 = _mm256_fmadd_ps(q, q, sq0);
            sk0 = _mm256_fmadd_ps(k, k, sk0);
        }
    }
    const float qs = 1.f / std::sqrt(hsum256_ps(sq0) + 1e-6f) * 0.08838834764831845f;  // 1/sqrt(128) fold
    const float ks = 1.f / std::sqrt(hsum256_ps(sk0) + 1e-6f);
    alignas(32) float qh[128], kh[128], delta[128];
    for (int i = 0; i < 128; ++i) { qh[i] = qh_[i] * qs; kh[i] = kh_[i] * ks; }
    const float decay = expf(g);
    const __m256 betav = _mm256_set1_ps(beta);
    // pass A: dacc[j] += S[i][j] · (k̂[i]·decay)   (row-major S, vector over j)
    __m256 d[16];
    for (int j = 0; j < 16; ++j) d[j] = _mm256_setzero_ps();
    for (int i = 0; i < 128; ++i) {
        const __m256 kb = _mm256_set1_ps(kh[i] * decay);
        const float *__restrict sr = S + i * 128;
        for (int j = 0; j < 16; ++j) d[j] = _mm256_fmadd_ps(_mm256_loadu_ps(sr + j * 8), kb, d[j]);
    }
    for (int j = 0; j < 16; ++j) {
        const __m256 dv = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(v + j * 8), d[j]), betav);
        _mm256_store_ps(delta + j * 8, dv);
    }
    // pass B: S[i][:] = S[i][:]·decay + k̂[i]·delta ; out[:] += S_new[i][:]·q̂[i]
    __m256 o[16];
    for (int j = 0; j < 16; ++j) o[j] = _mm256_setzero_ps();
    const __m256 decv = _mm256_set1_ps(decay);
    for (int i = 0; i < 128; ++i) {
        const __m256 kv = _mm256_set1_ps(kh[i]), qv = _mm256_set1_ps(qh[i]);
        float *__restrict sr = S + i * 128;
        for (int j = 0; j < 16; ++j) {
            const __m256 cell = _mm256_fmadd_ps(_mm256_loadu_ps(sr + j * 8), decv,
                                                 _mm256_mul_ps(kv, _mm256_load_ps(delta + j * 8)));
            _mm256_storeu_ps(sr + j * 8, cell);
            o[j] = _mm256_fmadd_ps(cell, qv, o[j]);
        }
    }
    for (int j = 0; j < 16; ++j) _mm256_storeu_ps(out + j * 8, o[j]);
}
struct DeltaTicket { DeltaJob j; int heads_per_ticket; };
static void deltanet_ticket(void *p, int t) {
    const DeltaTicket &dt = *(const DeltaTicket *)p;
    const DeltaJob &j = dt.j;
    const int h0 = t * dt.heads_per_ticket;
    const int h1 = std::min(j.heads, h0 + dt.heads_per_ticket);
    for (int h = h0; h < h1; ++h)
        deltanet_head_step(j.state + size_t(h) * 128 * 128, j.q + size_t(h / 3) * 128,
                           j.k + size_t(h / 3) * 128, j.v + size_t(h) * 128,
                           j.g[h], j.b[h], j.out + size_t(h) * 128);
}
inline void deltanet_step_cpu(float *state, const float *q, const float *k, const float *v,
                              const float *g, const float *b, float *out, int heads = 48) {
    DeltaTicket dt{{state, q, k, v, g, b, out, heads}, std::max(1, heads / 6)};
    CpuPool::get().launch(deltanet_ticket, &dt, (heads + dt.heads_per_ticket - 1) / dt.heads_per_ticket);
}

// ─────────────────── full-attention: q/k norm + partial RoPE ───────────────────
// Mirrors src/ops.cu qk_norm_rope: per-head RMSNorm over 256 (eps 1e-6), bf16 weight,
// partial rope on the first 64 dims with theta 1e7, pairs (i, i+32), i<32.
// cos/sin table computed once per call (same angles for q and k at this pos).
inline void qk_norm_rope_cpu(float *__restrict q /*[24][256]*/, float *__restrict k /*[4][256]*/,
                             const uint16_t *__restrict qw, const uint16_t *__restrict kw, int pos,
                             int qheads = 24, int kvheads = 4) {
    float cs[32], sn[32];
    for (int i = 0; i < 32; ++i) {
        const float a = float(pos) * std::pow(1e7f, -float(2 * i) / 64.f);
        cs[i] = std::cos(a); sn[i] = std::sin(a);
    }
    auto head = [&](float *__restrict p, const uint16_t *__restrict w) {
        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
        for (int d = 0; d < 256; d += 16) {
            const __m256 a = _mm256_loadu_ps(p + d), b = _mm256_loadu_ps(p + d + 8);
            s0 = _mm256_fmadd_ps(a, a, s0); s1 = _mm256_fmadd_ps(b, b, s1);
        }
        const float nsc = 1.f / std::sqrt(hsum256_ps(_mm256_add_ps(s0, s1)) / 256.f + 1e-6f);
        for (int d = 0; d < 256; d += 8) {
            const __m256 z = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(p + d), _mm256_set1_ps(nsc)),
                                           bf16_widen8(w + d));
            _mm256_storeu_ps(p + d, z);
        }
        if (pos)
            for (int i = 0; i < 32; ++i) {
                const float a = p[i], b2 = p[i + 32];
                p[i]      = a * cs[i] - b2 * sn[i];
                p[i + 32] = a * sn[i] + b2 * cs[i];
            }
    };
    for (int h = 0; h < qheads; ++h) head(q + h * 256, qw + h * 256);
    for (int h = 0; h < kvheads; ++h) head(k + h * 256, kw + h * 256);
}

// q_proj is [q|gate] interleaved per head (src[h*512+d] = q, src[h*512+256+d] = gate).
inline void split_q_gate_cpu(const float *__restrict src, float *__restrict q,
                             float *__restrict gate, int qheads = 24) {
    for (int h = 0; h < qheads; ++h)
        for (int d = 0; d < 256; d += 8) {
            _mm256_storeu_ps(q + h * 256 + d, _mm256_loadu_ps(src + h * 512 + d));
            _mm256_storeu_ps(gate + h * 256 + d, _mm256_loadu_ps(src + h * 512 + 256 + d));
        }
}

inline void store_kv_cpu(const float *__restrict k, const float *__restrict v,
                         float *__restrict kc, float *__restrict vc, int pos, int kvrow = 1024) {
    memcpy(kc + size_t(pos) * kvrow, k, kvrow * 4);
    memcpy(vc + size_t(pos) * kvrow, v, kvrow * 4);
}

// ─────────────────────── GQA decode on CPU (§5.4) ───────────────────────
// 24 q-heads / 4 kv-heads (kvh = head/6), head_dim 256, scale 1/16.
// Parallelized over TOKEN RANGES (not heads): every thread streams a disjoint slice of
// the KV cache exactly once (16.8 MB @ctx2048 read from DRAM once, not 6x), computing a
// flash-style partial (running max m, sum l, partial out o) per head; the caller merges.
// KV may be f32 (engine today) or bf16 (halves traffic; widen via <<16).
struct GqaScratch {                       // [nsplit][24][2 + 256]: m, l, o[]
    std::vector<float> buf;
    int nsplit = 0, heads = 24;
    float *at(int s, int h) { return buf.data() + (size_t(s) * heads + h) * 258; }
};

// One (thread, token-range) partial for ALL 24 heads: score+softmax+V-mix fused per
// token (K row dot, online-softmax rescale, V row axpy). kvh = head/6 (27B GQA 6:1 —
// NOT head>>2). KV rows are 1024 elems/token (4 kv-heads × 256), f32 or bf16.
struct GqaJob {
    const float *__restrict q;             // [24][256]
    const uint8_t *__restrict kc, *__restrict vc;   // f32 or bf16 rows of 1024 elems/token
    int tokens, nsplit;
    bool bf16;
    GqaScratch *__restrict sc;
};
template <bool KVBF16>
static void gqa_head_range(const GqaJob &J, int t0, int t1, float *part) {
    for (int h = 0; h < 24; ++h) {
        const float *__restrict qh = J.q + h * 256;
        const int kvh = h / 6;
        float m = -3.402823466e+38F, l = 0.f;
        __m256 o[32];
        for (int j = 0; j < 32; ++j) o[j] = _mm256_setzero_ps();
        for (int t = t0; t < t1; ++t) {
            const size_t base = size_t(t) * 1024 + size_t(kvh) * 256;
            __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps(),
                   s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
            const uint8_t *kp;
            if constexpr (KVBF16) kp = J.kc + base * 2; else kp = J.kc + base * 4;
            for (int d = 0; d < 256; d += 32) {
                const __m256 k0 = KVBF16 ? bf16_widen8((const uint16_t *)kp + d)     : _mm256_loadu_ps((const float *)kp + d);
                const __m256 k1 = KVBF16 ? bf16_widen8((const uint16_t *)kp + d + 8) : _mm256_loadu_ps((const float *)kp + d + 8);
                const __m256 k2 = KVBF16 ? bf16_widen8((const uint16_t *)kp + d + 16): _mm256_loadu_ps((const float *)kp + d + 16);
                const __m256 k3 = KVBF16 ? bf16_widen8((const uint16_t *)kp + d + 24): _mm256_loadu_ps((const float *)kp + d + 24);
                s0 = _mm256_fmadd_ps(_mm256_loadu_ps(qh + d), k0, s0);
                s1 = _mm256_fmadd_ps(_mm256_loadu_ps(qh + d + 8), k1, s1);
                s2 = _mm256_fmadd_ps(_mm256_loadu_ps(qh + d + 16), k2, s2);
                s3 = _mm256_fmadd_ps(_mm256_loadu_ps(qh + d + 24), k3, s3);
            }
            const float s = (hsum256_ps(_mm256_add_ps(s0, s1)) + hsum256_ps(_mm256_add_ps(s2, s3))) * 0.0625f;
            const float mn = s > m ? s : m;
            const float r = (m == -3.402823466e+38F) ? 0.f : expf(m - mn);
            const float p = expf(s - mn);
            l = l * r + p;
            const __m256 pv = _mm256_set1_ps(p), rv = _mm256_set1_ps(r);
            const uint8_t *vp;
            if constexpr (KVBF16) vp = J.vc + base * 2; else vp = J.vc + base * 4;
            for (int j = 0; j < 32; ++j) {
                const __m256 v = KVBF16 ? bf16_widen8((const uint16_t *)vp + j * 8) : _mm256_loadu_ps((const float *)vp + j * 8);
                o[j] = _mm256_fmadd_ps(pv, v, _mm256_mul_ps(o[j], rv));
            }
            m = mn;
        }
        float *__restrict pa = part + size_t(h) * 258;
        pa[0] = m; pa[1] = l;
        for (int j = 0; j < 32; ++j) _mm256_storeu_ps(pa + 2 + j * 8, o[j]);
    }
}
static void gqa_ticket(void *p, int t) {
    const GqaJob &J = *(const GqaJob *)p;
    const int per = (J.tokens + J.nsplit - 1) / J.nsplit;
    const int t0 = t * per, t1 = std::min(J.tokens, t0 + per);
    float *part = J.sc->at(t, 0);
    if (t0 >= t1) {   // empty range: neutral partials
        for (size_t i = 0; i < size_t(24) * 258; ++i) part[i] = 0.f;
        for (int h = 0; h < 24; ++h) part[size_t(h) * 258] = -3.402823466e+38F;
        return;
    }
    if (J.bf16) gqa_head_range<true>(J, t0, t1, part);
    else gqa_head_range<false>(J, t0, t1, part);
}
inline void gqa_decode_cpu(const float *q, const void *kc, const void *vc, int tokens,
                           float *out /*[24][256]*/, bool kv_bf16 = false, int nsplit = 6) {
    if (tokens <= 0) throw std::runtime_error("insignia cpu: gqa needs tokens>=1");
    nsplit = std::min(nsplit, tokens);
    thread_local GqaScratch sc;
    sc.nsplit = nsplit; sc.heads = 24;
    sc.buf.assign(size_t(nsplit) * 24 * 258, 0.f);
    GqaJob J{q, (const uint8_t *)kc, (const uint8_t *)vc, tokens, nsplit, kv_bf16, &sc};
    CpuPool::get().launch(gqa_ticket, &J, nsplit);
    for (int h = 0; h < 24; ++h) {         // merge partials (online-softmax combine)
        float M = -3.402823466e+38F;
        for (int s = 0; s < nsplit; ++s) M = std::max(M, sc.at(s, h)[0]);
        float L = 0.f;
        __m256 O[32];
        for (int j = 0; j < 32; ++j) O[j] = _mm256_setzero_ps();
        for (int s = 0; s < nsplit; ++s) {
            const float *__restrict pa = sc.at(s, h);
            const float w = (pa[0] == -3.402823466e+38F) ? 0.f : expf(pa[0] - M);
            L += pa[1] * w;
            const __m256 wv = _mm256_set1_ps(w);
            for (int j = 0; j < 32; ++j)
                O[j] = _mm256_fmadd_ps(wv, _mm256_loadu_ps(pa + 2 + j * 8), O[j]);
        }
        const float inv = 1.f / L;
        for (int j = 0; j < 32; ++j)
            _mm256_storeu_ps(out + size_t(h) * 256 + j * 8, _mm256_mul_ps(O[j], _mm256_set1_ps(inv)));
    }
}

// ─────────────────────── f64 parity reference (§7) ───────────────────────
inline float e4m3_scalar(uint8_t b) {
    const unsigned short h = (unsigned short)(((b & 0x7f) << 7) | ((b & 0x80) << 8));
    return _cvtsh_ss(h) * 256.f;
}
// Double-accumulated reference with raw bf16 scales (no x256 fold — uses true scale).
inline void fp8_gemv_f64_ref(const uint8_t *__restrict w, const uint16_t *__restrict sb,
                             const float *__restrict x, double *__restrict y, int rows, int cols) {
    const int kb = (cols + 127) >> 7;
    for (int r = 0; r < rows; ++r) {
        double acc = 0.0;
        for (int cb = 0; cb < kb; ++cb) {
            double p = 0.0;
            const int c0 = cb * 128;
            for (int c = c0; c < c0 + 128 && c < cols; ++c)
                p += double(e4m3_scalar(w[size_t(r) * cols + c])) * double(x[c]);
            acc += p * double(bf16_to_f32(sb[size_t(r >> 7) * kb + cb]));
        }
        y[r] = acc;
    }
}
struct Parity { double cos, max_rel; };
inline Parity compare_f64(const float *y, const double *ref, int n) {
    double dot = 0, na = 0, nb = 0, mx = 0, refmax = 0;
    for (int i = 0; i < n; ++i) refmax = std::max(refmax, std::fabs(ref[i]));
    for (int i = 0; i < n; ++i) {
        dot += double(y[i]) * ref[i];
        na += double(y[i]) * double(y[i]);
        nb += ref[i] * ref[i];
        const double d = std::fabs(double(y[i]) - ref[i]);
        if (std::fabs(ref[i]) > refmax * 1e-6) mx = std::max(mx, d / std::fabs(ref[i]));
    }
    return {dot / (std::sqrt(na) * std::sqrt(nb) + 1e-300), mx};
}

}  // namespace insignia
```

Wiring sketch (one linear CPU layer, decode T=1), matching `src/decode.cu`'s GPU flow:

```
rmsnorm_cpu(x, norm_w, n, 5120, /*zero_centered=*/true);
fp8_gemv_mt(qkv_w, qkv_s256, n, qkv, 10240, 5120);
fp8_gemv_mt(z_w,   z_s256,   n, z,    6144, 5120);
bf16_gemv_mt(a_w, n, a, 48, 5120);  bf16_gemv_mt(b_w, n, b, 48, 5120);
deltanet_parameters_cpu(a, b, A_log, dt_bias, 48);
causal_conv4_silu_cpu(qkv, conv_state, conv_w_f32, 10240);
deltanet_step_cpu(delta_state, qkv, qkv+2048, qkv+4096, a, b, core, 48);
gated_rmsnorm_per_head_cpu(core, norm2_w, z, core, 48, 128);
fp8_gemv_mt(out_w, out_s256, core, core2, 5120, 6144);  residual_add_cpu(x, core2, 5120);
rmsnorm_cpu(x, norm3_w, n, 5120, true);
fp8_gemv_mt(gate_w, s, n, gate, 17408, 5120);  fp8_gemv_mt(up_w, s, n, up, 17408, 5120);
silu_mul_cpu(gate, up, gate, 17408);
fp8_gemv_mt(down_w, s, gate, down, 5120, 17408);  residual_add_cpu(x, down, 5120);
```

Full-attn layer adds `split_q_gate_cpu` → `qk_norm_rope_cpu` → `store_kv_cpu` →
`gqa_decode_cpu` → `sigmoid_mul_cpu` → `o_proj` (and uses fp8_gemv2_mt for the pair/verify
path, exactly like `fp8_gemv2` on GPU).

---

## 5. Design notes

### 5.1 GEMV

- **Scale promote per 128-col block** (raw accumulation, one hsum + one scalar FMA per
  block): identical shape to `fp8.cu`'s `acc = fmaf(part, sc, acc)` and TRT-LLM's
  blockwise-FP8 promote; costs ~16 ops per 512 weights. 32-weight chunks never straddle a
  128-col scale block (128 % 32 == 0).
- **Accumulator structure**: 4 independent `__m256` lane-accumulators per row (FMA latency
  4 cycles, one FMA per accumulator per 32-weight chunk ≈ every 21.8 cycles — no latency
  chain). Register budget ≈ 12–14 ymm live in the inner block — spill-free by construction;
  still verify in the disassembly per AGENTS.md conventions.
- **Prefetch**: 256 B ahead (tunable `INSIG_PREFETCH_DIST`). Zen 3's L2 stream prefetcher
  does the real work (6 sequential streams, one per worker, ~43 cycles per line — trivially
  inside its window); the hint buys latency tolerance at the stream head. Weight loads are
  plain `loadu` (see 5.2); x loads hit L1/L2 (§2.4).
- **Tickets**: rows partitioned into contiguous blocks of `clamp(rows/96, 1, 32)` rows
  (10–32 as briefed: 10240→32, 1024→10, 48→1). Contiguous rows per ticket keep each
  worker's DRAM stream sequential; the atomic ticket rebalances stragglers (an IOCP wake
  stealing a core for 200 µs just costs a ticket handoff, not a stripe).
- **NT stores avoided**: outputs are ≤ 70 KB per mat and immediately consumed by the next
  op — normal stores keep them in cache.
- **Pair (T=2)**: weights dequantized once, two FMA chains (x0/x1). +8 FMA +8 loads per
  32 weights — still ≤ 6 cycles of issue per 21.8-cycle budget. This is the MTP-verify
  shape; at DRAM-bound the second token's GEMV is free (matches the GPU bandwidth argument
  in synthesis.md).

### 5.2 Alignment

Weights may be ring-staged (4096-aligned slots) or mmap-direct (safetensors offsets =
arbitrary mod 64). All loads are `loadu`: on Zen 3 an unaligned 32B load costs nothing
unless it crosses a 64B line, which the hardware splits transparently. Every 27B row
stride (5120/6144/10240/17408/12288) is a multiple of 64, so the crossing pattern repeats
identically per row — no layout change needed. `x`/`y` are engine buffers; `loadu`/`storeu`
everywhere; the only aligned ops are on stack arrays (`alignas(32)`).

### 5.3 Thread pool

Generation-packed single-word claim (`(gen<<32)|ticket`) — the same idea as colibri's
`(gen<<8|idx)` CAS claim, widened: no ABA (gen monotonic), no stale-fn execution (fn/ctx of
gen g are immutable until every ticket of g has returned, because the dispatcher publishes
g+1 only after `left_==0`), and the **caller participates**, which is the liveness
guarantee — if all workers are parked the caller alone finishes the job (no deadlock even
with a missed CV wakeup; the completion wait also self-heals on a 500 µs timeout).
Workers spin ~4096 pauses before parking so sub-100µs ops (the a/b GEMV is ~27 µs) don't
pay the full wake latency. Dispatch overhead per GEMV: one mutex + one CAS per ticket
(~320–1088 tickets) ≈ 20–60 µs — 0.3–0.6% of a 10 ms mat; acceptable at DRAM-bound.

Why not 12 workers: §2.5. Why not Windows thread pool / OpenMP: colibri's audit
(colibri-sched-deep.md §1.5) documents the yield storms and spin-burn of parked OpenMP
teams; a 6-thread dedicated pool with spin-then-park is the controllable option.

### 5.4 Small ops & attention & deltanet

Costs are in §6; design points:

- **rmsnorm**: 4-chain vector sumsq, *scalar* `1/sqrtf` for parity with `rsqrtf`-class GPU
  accuracy (Zen `_mm256_rsqrt_ps` is 3×10⁻⁴ — too coarse for the 1e-5 parity bar).
- **silu/sigmoid**: Remez deg-4 `exp` (max rel err 2.6×10⁻⁶, verified) with range-reduction
  by RNE `cvtps_epi32` and 2^n by integer shift — the standard vector exp shape, ~11 μops
  per 8 lanes. Serial small ops (norms, conv, params) stay off the pool: their compute is
  ≤ 40 µs each vs ~20 µs of pool wake cost — not worth the complexity except silu_mul at
  17408 (~5 µs compute) which is fine serial too.
- **conv1d**: scalar due to the engine's `[ch][3]` state layout (vectorizing = strided
  gather; measured-class cost 30 µs serial = 0.3% of the layer — not worth transposing the
  snapshot layout; revisit only if the engine ever moves conv state to `[3][ch]`).
- **deltanet step**: exactly mirrors `src/deltanet.cu` semantics (including the asymmetric
  `1/√128` folded into the q-norm scale and `kh = head/3`), vectorized over the v-dimension
  with row-major S so both passes are pure sequential streams. 2 passes = minimum (delta
  needs the full S·k̂ row set before the update); traffic 3×3.15 MB.
- **GQA decode**: token-range split (not head split!) so the KV cache is read **once** from
  DRAM across all threads (a naive head-split re-streams K/V up to 6× = 2.7 ms instead of
  0.45 ms). Online-softmax partials + flash-style merge. 24 heads per thread over its token
  range; q re-reads are L1/L2-trivial. Scalar `expf` in the softmax path (2 per token per
  head → 98 K/layer ≈ 25 µs across 6 cores) — vectorizing gains ~20 µs, skipped. bf16 KV
  halves the 16.8 MB to 0.23 ms via the `bf16_widen8` path (template).

### 5.5 bf16 GEMV

Widen `(u16<<16)` bitcast — the ggml `vec.cpp:174` pattern — 1 load + 2 ops per 8 weights
(no LUT, no rounding, bit-exact). Used for in_proj_a/b (0.98 MB, L3-resident after first
touch) and available for mtp.fc [5120,10240] if MTP ever moves CPU-side (it stays GPU per
the mission).

---

## 6. Numbers

Assumptions: 37 GB/s clean read (32 GB/s while NVMe staging writes ~6.8 GB/s share the
controller), 4.2 GHz sustained all-core, 6 workers, ctx 2048.

### 6.1 Per-mat GEMV (T=1)

| matrix | shape [r,c] | F8 bytes | ms @37 GB/s | vec-FMA | FMA ms (50.4 G/s) | note |
|---|---|---|---|---|---|---|
| in_proj_qkv | 10240×5120 | 52.43 MB | 1.417 | 6.55 M | 0.130 | |
| in_proj_z | 6144×5120 | 31.46 MB | 0.850 | 3.93 M | 0.078 | |
| out_proj | 5120×6144 | 31.46 MB | 0.850 | 3.93 M | 0.078 | |
| mlp.gate / mlp.up | 17408×5120 | 89.13 MB | 2.409 | 11.14 M | 0.221 | ×2 |
| mlp.down | 5120×17408 | 89.13 MB | 2.409 | 11.14 M | 0.221 | x = 68 KB → L2 |
| q_proj | 12288×5120 | 62.91 MB | 1.701 | 7.86 M | 0.156 | |
| k_proj / v_proj | 1024×5120 | 5.24 MB | (0.142) | 0.66 M | 0.013 | **L3-resident after 1st token → ~25 µs** |
| o_proj | 5120×6144 | 31.46 MB | 0.850 | 3.93 M | 0.078 | borderline L3 |
| a/b (bf16) | 48×5120 | 0.98 MB | (0.027) | 0.12 M | 0.002 | L3-resident |

Every mat: FMA time ≤ 9.2% of DRAM time → **all GEMVs are DRAM-bound with ~11× compute
headroom**; the dequant's ~21% issue utilization is invisible.

### 6.2 Non-GEMV per layer

| op | traffic | ms @37 GB/s | vec-FMA | compute |
|---|---|---|---|---|
| deltanet step (48 heads) | 9.44 MB (3× state r/w) | **0.255** | 0.295 M | 5.9 µs |
| conv1d + state | 0.37 MB | 0.010 | — | 30 µs serial |
| norms ×2 + gated rmsnorm + silu_mul + sigmoid_mul + residuals | ~1.5 MB r/w | 0.04 | ~0.1 M | ~25 µs |
| **linear layer small-op total** | | **≈ 0.31** | | |
| GQA decode f32 (ctx 2048) | 16.78 MB | **0.454** | 3.15 M | 62 µs |
| GQA decode bf16 KV | 8.39 MB | 0.227 | 3.15 M | ~70 µs |
| qk-norm-rope + split + store_kv | 0.03 MB | ~0 | ~0.3 M | ~15 µs |
| **full-attn small-op total (f32 KV)** | | **≈ 0.50** | | |

### 6.3 Per-layer and per-token tier totals

| layer type | F8 GEMV | small ops | total/layer |
|---|---|---|---|
| linear-attention (48) | 382.73 MB → 10.344 ms | 0.31 ms | **10.66 ms** |
| full-attention (16) | 372.24 MB → 10.061 ms | 0.51 ms | **10.57 ms** |

23 CPU layers at the census ratio 48:16 = 3:1 → 17 linear + 6 full:

- **Clean: 17×10.66 + 6×10.57 = 244.6 ms/token** (bf16 KV: −1.4 ms).
- **With NVMe staging concurrently stealing ~5 GB/s of DRAM (effective 32 GB/s): ≈ 283 ms.**
- FMA cross-check: 23 layers × ~48 M vec-FMA ≈ 1.10 G vec-FMA… × /50.4 G/s = **21.8 ms of
  FMA per token** vs 245 ms of DRAM — 8.9% aggregate, consistent with §2.

### 6.4 Token budget & overlap (placement L21/M23/N21 from synthesis.md)

| tier | per token | threads/cores |
|---|---|---|
| VRAM layers (21) | 21 × 0.76 ms = 16 ms | GPU + 1 submit thread |
| **CPU layers (23)** | **245–283 ms** | 6 pinned workers (LP 0–5) |
| NVMe layers (21) | 21 × 56.5 ms = 1187 ms | IOCP + staging on LP 6–11 |
| serial total | ≈ 1.45–1.49 s | 0.67–0.69 tok/s; MTP ×1.6 → **~1.05–1.1 tok/s** |

CPU tier = 17–19% of the token time; the six GEMV workers are idle during VRAM/NVMe layer
execution and can double as staging/copy muscle (colibri early-issue pattern). The decode
loop is token-serial, so NVMe read-ahead hides latency, not bandwidth — the numbers above
assume full serialization (conservative; synthesis.md reached the same conclusion).

---

## 7. Numeric parity

- **CPU path**: fp32 lane accumulation (4 chains) raw per 128-col block → hsum → ×scale256
  (one rounding per block) → row sum in fp32. Block order: left-to-right, deterministic.
- **GPU path** (`fp8.cu`): fp32 per-lane 16-col partials → ×scale per 512-col round.
  Different summation order — both are exact-value block-scaled dots up to fp32 rounding.
- **Expected error**: K=5120 dot, ε=2⁻²⁴ per FMA → RMS rel vs f64 ≈ √K·ε ≈ 7.6×10⁻⁷,
  worst-case accumulation ≈ K·ε ≈ 3×10⁻⁴ before cancellation; empirically (ggml experience
  + GPU kernel's cosine history) **max rel ~5×10⁻⁵, cosine > 0.999999**.
- **Thresholds for the parity tests** (use `fp8_gemv_f64_ref` + `compare_f64`):
  - CPU vs f64 ref: **cos ≥ 0.999999, max rel ≤ 2×10⁻⁴** (floor |ref| > max|ref|·10⁻⁶).
  - CPU vs GPU (`fp8_gemv`): **max rel ≤ 1×10⁻³, cos ≥ 0.999999** (two fp32 orderings).
  - Subnormal correctness is *free* on CPU (unlike the fp32-magic path) — the f64 ref
    decodes them exactly, so any decode regression shows up immediately.
  - End-to-end layer parity vs `tools/reference_all_layers.py` (f64): same thresholds as
    the GPU engine bar; wait for coherent token parity before claiming correctness
    (AGENTS.md rule).
- exp/silu: vector poly is 2.6×10⁻⁶ max rel — two orders below e4m3 quantization noise
  (2⁻⁴·³ rms) and at CUDA `__expf` class; invisible in the cosine.
- Build with `/fp:precise` (MSVC default; never contracts) so the written FMAs are the
  executed FMAs — the ggml audit's `-ffp-contract` warning does not apply on MSVC.

---

## 8. Risks & open items

1. **`vcvtph2ps` throughput assumption** (1/cycle; 0.5 worst case): at 0.5 the dequant is
   8 cycles/32 weights — still 2.7× inside budget. Bench once; the LUT fallback exists.
2. **MSVC codegen of the 4-chunk block**: watch for spills in the disassembly (16-ymm
   pressure is the ceiling; the pair variant ~14 live). If MSVC spills, split the chunk
   macro into two 2-chunk steps — budget has room for worse scheduling.
3. **Pool wake latency on tiny ops**: workers spin 4096 pauses (~10–20 µs) before parking;
   if a/b GEMV latency matters, raise the spin or run it serial (`caller_helps`-only).
4. **DRAM contention model**: 32 GB/s under NVMe staging is an estimate; measure with the
   real IOCP depth (QD 8–16) before trusting the 283 ms number.
5. **k/v_proj L3 residency**: 5.24 MB mats go L3-resident after the first token — the
   table's 0.142 ms is an upper bound (~25 µs steady-state). L3 is 32 MB shared with the
   GEMV streams; re-measure with neighbors resident.
6. **`weight_scale_inv` semantics**: assumed dequant-multiplied (`W = fp8 × scale`, same as
   `fp8.cu`'s `fmaf(part, sc, acc)`). Confirmed against the loader census (407/407 linked
   scales); if the checkpoint's naming ever proves inverted, only `fp8_prepare_scales`
   changes.
7. **Engine integration points** (not in this header, tracked by w2 shape audit): host-side
   buffers for s256/conv-weights per CPU layer, `DType::f8e4m3` in the model index, decode
   graph split CPU-layer/GPU-layer, and the CPU layer's activation handoff to the next GPU
   layer over PCIe (one 20 KB float per boundary — 8 µs at 25 GB/s, negligible).

## TL;DR

The 5600X can feed 37 GB/s of e4m3 weights through an exact fp16-bit-trick dequant +
fp32-FMA GEMV at 9% FMA utilization and 21% issue utilization — the CPU tier is purely a
DRAM-stream problem at ~10.4 ms per linear layer / ~10.6 ms per full-attn layer, 245 ms per
token across 23 layers (283 ms contended), on 6 pinned workers with main+IOCP on the SMT
siblings. Full implementable header in §4; parity harness with f64 reference and thresholds
in §7; per-mat/per-layer budget tables in §6.
