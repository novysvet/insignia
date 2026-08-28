// ============================================================================
//  insignia_cpu.hpp — CPU compute tier for RAM-resident Qwen3.8-27B-FP8 layers
//  Target: AMD Ryzen 5 5600X (Zen 3, 6C/12T, AVX2+FMA+F16C; NO AVX-512 / VNNI /
//  fp16 arithmetic). MSVC 19.51 x64, /arch:AVX2 /O2 /fp:precise (no contraction).
//
//  DRAM budget model: ~37 GB/s socket read; per core (6 workers @4.2 GHz) that is
//  1.47 B/cycle = 21.8 cycles per 32 weight bytes. The GEMV inner block is ~28 uops
//  (<= 8 cycles of port time) — memory-bound with ~11x FMA headroom. Analysis:
//  audits/w3/cpu-fp8.md; implementation deltas documented in audits/w3/cpu-impl.md.
//  Per AGENTS.md: adopt after bench + parity + disasm.
//
//  Semantics mirrored from the GPU engine:
//   - fp8 block-scaled GEMV: src/fp8.cu (W = e4m3 x bf16 scale [r/128][c/128]).
//   - deltanet step: src/deltanet.cu (S = state + head*128*128, S[k*128+v],
//     k=key index, v=value index; q-norm folds 1/sqrt(128); 27B shares k-heads
//     across head/3 v-heads).
//   - rmsnorm / gated rmsnorm / silu / sigmoid_mul / conv1d / a-b params:
//     src/qwen_kernels.cu. Norm weights are bf16 and SHARED across heads
//     (norm [128], q/k norm [256]) — the kernel advances x/gate, never w.
//   - qk norm + partial RoPE (64 of 256 dims, theta 1e7, pairs (i,i+32)):
//     src/ops.cu qk_norm_rope.
//   - GQA decode: src/attention.cu (24 q / 4 kv heads, kvh = head/6, head_dim 256,
//     scale 1/16, KV rows [4][256] per token), online-softmax variant.
// ============================================================================
#pragma once
#if defined(_MSC_VER)
#if !defined(__AVX2__)
#error "insignia_cpu.hpp needs /arch:AVX2 on MSVC x64 (AVX2 implies FMA+F16C there)"
#endif
#elif !defined(__AVX2__) || !defined(__FMA__) || !defined(__F16C__)
#error "insignia_cpu.hpp needs -mavx2 -mfma -mf16c"
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
#define INSIG_CPU_FP8_LUT 0        // 1 = debug A/B via 256-entry f32 LUT (audit §3.3)
#endif
#ifndef INSIG_PREFETCH_DIST
#define INSIG_PREFETCH_DIST 256    // bytes ahead for weight prefetch; 0 disables
#endif

namespace insignia::cpu {

// ─────────────────────────── scalar/bit helpers ───────────────────────────

inline float bf16_to_f32(uint16_t u) {
    const uint32_t b = uint32_t(u) << 16;
    float f; memcpy(&f, &b, sizeof f);
    return f;
}

// fp16 -> fp32 scalar via F16C (MSVC has no _cvtsh_ss; _mm_cvtph_ps is available
// under /arch:AVX2).
inline float f16_to_f32(uint16_t h) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(int32_t(uint32_t(h)))));
}

// f32 -> bf16 round-to-nearest-even (same as src/test_fp8.cu / engine casts).
inline uint16_t f32_to_bf16_bits(float v) {
    uint32_t bits;
    memcpy(&bits, &v, 4);
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return uint16_t(bits >> 16);
}

// Block scale with the e4m3->f32 x256 FOLDED IN (audit §3.2; exact).
// Bit-add only for NORMAL bf16 (e>=1): subnormals (e==0) would have their
// mantissa reinterpreted as a normal mantissa, and e>=0xF7 would overflow /
// carry into the sign — both take the multiply path (itself exact: bf16->f32 is
// exact and x256 only shifts the exponent). Real block scales are ~1e-3..1; the
// guards make all 65536 bf16 codes exact regardless (exhaustively verified).
inline float bf16_scale_x256(uint16_t u) {
    const uint32_t b = uint32_t(u) << 16;
    const uint32_t e = (b >> 23) & 0xffu;
    if (e == 0 || e >= 0xf7u)                     // +-0, subnormal, |s|>=2^120, inf, nan
        return bf16_to_f32(u) * 256.f;
    const uint32_t r = b + 0x04000000u;           // exponent += 8  ==  x256, no rounding
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
    __m128 d = _mm_movehdup_ps(s);                // [s1,s1,s3,s3]
    s = _mm_add_ps(s, d);                         // lane0=s0+s1, lane2=s2+s3
    d = _mm_movehl_ps(d, s);                      // lane0 = s[2]
    s = _mm_add_ss(s, d);
    return _mm_cvtss_f32(s);
}

// ───────────────────── e4m3 -> fp32 dequant (audit §3.1) ─────────────────────
//
// 32 e4m3 bytes -> 4 __m256 fp32, EXACT for all 256 codes incl. subnormals
// (0x7F/0xFF decode to +/-480 per this engine's convention — e4m3 has no inf,
// and these checkpoints contain no NaN).
// fp16 pattern per byte: mag = (b&0x7f)<<7 (bits 7..13), sign -> bit 15; F16C
// converts; the x256 is folded into the block scale (bf16_scale_x256), NOT here.
// 18 vector ops per 32 weights: 2 vpmovzxbw + 10 and/sll/or + 2 vextracti128 + 4 vcvtph2ps.
inline void e4m3x32_f32(const __m256i w, __m256 *__restrict y) {
    const __m256i m7 = _mm256_set1_epi16(0x007f);
    const __m256i s8 = _mm256_set1_epi16(-32768);          // 0x8000
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
// Register-output form (named locals, no wv[4] array): MSVC otherwise spills the
// array to stack before the FMAs (verified in disasm — audit risk §8.2).
inline void e4m3x32_rr(const __m256i w, __m256 &y0, __m256 &y1, __m256 &y2, __m256 &y3) {
    const __m256i m7 = _mm256_set1_epi16(0x007f);
    const __m256i s8 = _mm256_set1_epi16(-32768);          // 0x8000
    const __m256i lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(w));
    const __m256i hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(w, 1));
    const __m256i l16 = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(lo, m7), 7),
                                        _mm256_and_si256(_mm256_slli_epi16(lo, 8), s8));
    const __m256i h16 = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(hi, m7), 7),
                                        _mm256_and_si256(_mm256_slli_epi16(hi, 8), s8));
    y0 = _mm256_cvtph_ps(_mm256_castsi256_si128(l16));
    y1 = _mm256_cvtph_ps(_mm256_extracti128_si256(l16, 1));
    y2 = _mm256_cvtph_ps(_mm256_castsi256_si128(h16));
    y3 = _mm256_cvtph_ps(_mm256_extracti128_si256(h16, 1));
}

// Debug alternative: 256-entry f32 LUT (1 KB, L1-resident) via gather. ~3-6x the
// cycles of the trick on Zen 3 (microcoded gathers); kept ONLY for parity A/B.
struct alignas(64) Fp8Lut {
    float v[256];
    Fp8Lut() {
        for (int b = 0; b < 256; ++b) {
            const unsigned short h = (unsigned short)(((b & 0x7f) << 7) | ((b & 0x80) << 8));
            v[b] = f16_to_f32(h) * 256.f;          // same values, table form
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

// ───────────────────────── vector exp / sigmoid ─────────────────────────
//
// exp(x) = 2^(x*log2e); n=rint, f in [-1/2,1/2]; 2^f ~= Remez deg-4 (max rel err
// 2.6e-6, fitted in the w3 audit; CUDA __expf-class accuracy, two orders below
// e4m3 quantization noise). x clamped to +-87.3 so n in [-126,126].
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
// Persistent workers (default 6, one per physical core; LP 0..5 assumed primary
// threads on this Zen 3 box), main + IOCP threads stay on the SMT siblings.
// Jobs are serial (decode is serial): exactly one launch() at a time; the caller
// participates and is the progress guarantee — even if every worker sleeps, the
// caller finishes all tickets alone. NOT reentrant: never launch() from a job.
//
// Ticket claiming is a single packed atomic claim_ = (gen<<32)|next_ticket. The
// generation shares the word with the counter, so a straggler's CAS against a
// finished generation can never succeed (no ABA, no stale-fn execution: fn/ctx of
// gen g are immutable from publish until completion because the dispatcher
// publishes g+1 only after every ticket of g has returned).
class CpuPool {
public:
    using Job = void (*)(void *ctx, int ticket);

    static CpuPool &get() {
        static CpuPool p;
        return p;
    }

    int threads() const { return nthreads_; }

    // Blocking fan-out over [0,tickets). One dispatcher thread at a time.
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
                lk.unlock();                       // drive() takes m_ itself on the last ticket
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
            // brief spin first (small parallel ops, e.g. the ~27us a/b GEMV, must
            // not pay a full wake/park cycle), then park on generation change.
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
        unsigned want = hc / 2 ? hc / 2 : 1;                    // 6: one per physical core
#if defined(_WIN32)
        {   // optional override for scaling experiments (workers + participating main)
            char buf[16];
            size_t len = 0;
            if (getenv_s(&len, buf, sizeof buf, "INSIG_CPU_THREADS") == 0 && len > 0) {
                const int v = atoi(buf);
                if (v > 0) want = unsigned(v);
            }
        }
#endif
        nthreads_ = int(std::max(1u, std::min(32u, want)));
        for (int i = 0; i < nthreads_; ++i) {
            th_.emplace_back([this] { worker_main(); });
#if defined(_WIN32) && !defined(INSIG_CPU_NO_AFFINITY)
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
// y[r] = (sum_c W[r,c]*x[c]) with per-128x128-block scales (x256-folded, f32).
// Inner: per 128-col scale block, 4 chunks of 32 weights; raw lane accumulators
// a0..a3, one horizontal sum + one scale FMA per block (TRT-LLM promote pattern
// = fp8.cu shape). ~28 uops + prefetch per 32 weights vs 21.8-cycle DRAM budget.
struct Fp8GemvJob {
    const uint8_t *__restrict w;      // e4m3 [rows, cols] row-major (any alignment)
    const float *__restrict s256;     // f32 scales x256 [ceil(rows/128)][cols/128]
    const float *__restrict x0;       // fp32 [cols] (L1/L2-resident across rows)
    const float *__restrict x1;       // pair mode: second row, or nullptr
    float *__restrict y0;             // [rows]
    float *__restrict y1;             // pair mode: [rows], or nullptr
    int rows, cols, rpt;              // rows per ticket (10..32; see wrappers)
    bool pair;
};

#if INSIG_CPU_FP8_LUT
#define INSIG_F8_DEQ(WR, Y0, Y1, Y2, Y3) \
    { __m256 wv_[4]; e4m3x32_f32_lut(_mm256_loadu_si256((const __m256i *)(WR)), fp8_lut().v, wv_); \
      Y0 = wv_[0]; Y1 = wv_[1]; Y2 = wv_[2]; Y3 = wv_[3]; }
#else
#define INSIG_F8_DEQ(WR, Y0, Y1, Y2, Y3) e4m3x32_rr(_mm256_loadu_si256((const __m256i *)(WR)), Y0, Y1, Y2, Y3)
#endif

inline void fp8_gemv_rowrange(const Fp8GemvJob &j, int r0, int r1) {
    const int kb = j.cols >> 7;
    for (int r = r0; r < r1; ++r) {
        const uint8_t *__restrict wr = j.w + size_t(r) * j.cols;
        const float *__restrict sr = j.s256 + size_t(r >> 7) * kb;
        float acc = 0.f;
        for (int c = 0; c < j.cols; c += 128) {
            const float *__restrict xp = j.x0 + c;
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(),
                   a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
#if INSIG_PREFETCH_DIST
            _mm_prefetch((const char *)wr + c + INSIG_PREFETCH_DIST, _MM_HINT_T0);
#endif
            { __m256 w0, w1, w2, w3; INSIG_F8_DEQ(wr + c, w0, w1, w2, w3);
              a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xp + 0), a0);
              a1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xp + 8), a1);
              a2 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xp + 16), a2);
              a3 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xp + 24), a3); }
            { __m256 w0, w1, w2, w3; INSIG_F8_DEQ(wr + c + 32, w0, w1, w2, w3);
              a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xp + 32), a0);
              a1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xp + 40), a1);
              a2 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xp + 48), a2);
              a3 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xp + 56), a3); }
            { __m256 w0, w1, w2, w3; INSIG_F8_DEQ(wr + c + 64, w0, w1, w2, w3);
              a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xp + 64), a0);
              a1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xp + 72), a1);
              a2 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xp + 80), a2);
              a3 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xp + 88), a3); }
            { __m256 w0, w1, w2, w3; INSIG_F8_DEQ(wr + c + 96, w0, w1, w2, w3);
              a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xp + 96), a0);
              a1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xp + 104), a1);
              a2 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xp + 112), a2);
              a3 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xp + 120), a3); }
            const __m256 p01 = _mm256_add_ps(a0, a1), p23 = _mm256_add_ps(a2, a3);
            acc += (hsum256_ps(p01) + hsum256_ps(p23)) * sr[c >> 7];   // scale promote/block
        }
        j.y0[r] = acc;
    }
}

// Pair (T=2, MTP verify): one weight pass, two outputs — +8 FMA +8 loads per 32
// weights, still nothing vs the 21.8-cycle budget.
inline void fp8_gemv2_rowrange(const Fp8GemvJob &j, int r0, int r1) {
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
#if INSIG_PREFETCH_DIST
            _mm_prefetch((const char *)wr + c + INSIG_PREFETCH_DIST, _MM_HINT_T0);
#endif
#define INSIG_PAIR_CHUNK(K)                                                        \
    { __m256 w0, w1, w2, w3; INSIG_F8_DEQ(wr + c + (K) * 32, w0, w1, w2, w3);       \
      a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xa + (K) * 32 + 0), a0);             \
      a1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xa + (K) * 32 + 8), a1);             \
      a2 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xa + (K) * 32 + 16), a2);            \
      a3 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xa + (K) * 32 + 24), a3);            \
      b0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xb + (K) * 32 + 0), b0);             \
      b1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xb + (K) * 32 + 8), b1);             \
      b2 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xb + (K) * 32 + 16), b2);            \
      b3 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xb + (K) * 32 + 24), b3); }
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

inline void fp8_gemv_ticket(void *p, int t) {
    const Fp8GemvJob &j = *(const Fp8GemvJob *)p;
    const int r0 = t * j.rpt;
    if (r0 >= j.rows) return;
    const int r1 = std::min(j.rows, r0 + j.rpt);
    if (j.pair) fp8_gemv2_rowrange(j, r0, r1);
    else fp8_gemv_rowrange(j, r0, r1);
}

// Single-token GEMV via the pool. rows arbitrary; cols % 128 == 0 (27B mats all
// qualify; 5120/6144/17408).
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

// Serial fallback / 1-thread benchmark entry (also the pool-less path).
inline void fp8_gemv_st(const uint8_t *w, const float *s256, const float *x, float *y,
                        int rows, int cols) {
    if (rows <= 0 || cols <= 0 || (cols & 127))
        throw std::runtime_error("insignia cpu: bad fp8 gemv dims");
    Fp8GemvJob j{w, s256, x, nullptr, y, nullptr, rows, cols, rows, false};
    fp8_gemv_rowrange(j, 0, rows);
}

// ───────────────────────────── bf16 GEMV (§5.5) ─────────────────────────────
// For in_proj_a/b [48,5120] and any bf16 mat. Widen (u16<<16) bitcast is exact:
// 1 load + 2 ops per 8 weights (no LUT, no rounding).
struct Bf16GemvJob {
    const uint16_t *__restrict w;
    const float *__restrict x;
    float *__restrict y;
    int rows, cols, rpt;
};
inline void bf16_gemv_rowrange(const Bf16GemvJob &j, int r0, int r1) {
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
            acc += hsum256_ps(_mm256_add_ps(a0, a1));            // per-16 promote (tiny mats)
            a0 = _mm256_setzero_ps(); a1 = _mm256_setzero_ps();
        }
        j.y[r] = acc;
    }
}
inline void bf16_gemv_ticket(void *p, int t) {
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

// RMSNorm over one row; weight bf16 [cols] (checkpoint norm weights).
// zero_centered mirrors rms_bf (1+w vs w). Scalar 1/sqrt for rsqrtf-class parity.
inline void rmsnorm_cpu(const float *__restrict x, const uint16_t *__restrict w,
                        float *__restrict y, int cols, bool zero_centered, float eps = 1e-6f) {
    if (cols & 31) throw std::runtime_error("insignia cpu: rmsnorm cols%32");
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

// Gated per-head RMSNorm + silu-gate (linear layers), mirroring gated_rmsnorm_
// bf16 = rms_bf<false,true>: y = x*rsqrt(mean(x^2)+eps)*w * silu(gate).
// x, gate, y are [heads][hd]; the norm weight w is [hd] and SHARED across heads
// (engine layout: norm [128] — fixed vs the w3 design draft, which advanced w
// per head). Vectorized silu uses the Remez exp (2.6e-6 rel).
inline void gated_rmsnorm_per_head_cpu(const float *__restrict x, const uint16_t *__restrict w,
                                       const float *__restrict gate, float *__restrict y,
                                       int heads, int hd = 128, float eps = 1e-6f) {
    if (hd & 15) throw std::runtime_error("insignia cpu: gated rmsnorm hd%16");
    const int nv = hd >> 3;                        // 16 vecs per head at hd=128
    for (int h = 0; h < heads; ++h, x += hd, gate += hd, y += hd) {
        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
        for (int v = 0; v < nv; v += 2) {
            const __m256 a = _mm256_loadu_ps(x + v * 8), b = _mm256_loadu_ps(x + v * 8 + 8);
            s0 = _mm256_fmadd_ps(a, a, s0);
            s1 = _mm256_fmadd_ps(b, b, s1);
        }
        const float ss = hsum256_ps(_mm256_add_ps(s0, s1));
        const float inv = 1.f / std::sqrt(ss / float(hd) + eps);
        for (int v = 0; v < nv; ++v) {
            const __m256 z = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + v * 8), _mm256_set1_ps(inv)),
                                           bf16_widen8(w + v * 8));
            const __m256 g = _mm256_loadu_ps(gate + v * 8);
            const __m256 sg = _mm256_div_ps(g, _mm256_add_ps(_mm256_set1_ps(1.f),
                                   vexp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), g))));
            _mm256_storeu_ps(y + v * 8, _mm256_mul_ps(z, sg));
        }
    }
}

// y = silu(g)*u  (n % 8 == 0; 5120/6144/17408 all qualify)
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

// Causal conv1d (width 4) + silu + state shift. State layout [ch][3] matches the
// engine (conv_state[c*3+i]); weights pre-expanded at layer load to f32
// wt[tap][ch] (transpose of the checkpoint's [ch][4] bf16 — expand_conv_weights).
// Scalar body: the [ch][3] state makes vectorization a gather; 10240 ch is
// ~30 us serial = 0.3% of a layer.
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

// a/b gating params (per v-head): b = sigmoid(b); a = -exp(A_log)*softplus(a+dt_bias).
// Mirrors src/qwen_kernels.cu params kernel exactly (incl. the z>20 guard).
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
// Mirrors src/deltanet.cu deltanet_decode_kernel exactly:
//   S = state + head*128*128, S[i*128 + v]: i = key index, v = value index.
//   qhat = q * rsqrt(sum(q^2)+1e-6) * (1/sqrt(128))    [fold on q only]
//   khat = k * rsqrt(sum(k^2)+1e-6)
//   dot[v]  = sum_k S[k*128+v] * decay * khat[k]
//   delta[v]= (v[v] - dot[v]) * beta
//   S[k][v] = S[k][v]*decay + khat[k]*delta[v]
//   out[v]  = sum_k S_new[k][v] * qhat[k]
// q,k are [kheads][128] (khead = head/kshare), v/out are [heads][128].
// 27B: 48 v-heads, kshare=3 (kh = head/3). 9B GPU kernel uses kshare=2 (>>1).
// Two passes over S: 3 vec-FMA per state element, 3x64 KB traffic per head.
struct DeltaJob {
    float *__restrict state;                        // [heads][128][128]
    const float *__restrict q, *__restrict k;       // [kheads][128]
    const float *__restrict v;                      // [heads][128]
    const float *__restrict g, *__restrict b;       // a,b after parameters
    float *__restrict out;                          // [heads][128]
    int heads, kshare;
};
inline void deltanet_head_step(float *__restrict S, const float *__restrict qh_,
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
    // pass A: dacc[v] += S[k*128+v] * (khat[k]*decay)   (sequential over k)
    __m256 d[16];
    for (int j = 0; j < 16; ++j) d[j] = _mm256_setzero_ps();
    for (int k = 0; k < 128; ++k) {
        const __m256 kb = _mm256_set1_ps(kh[k] * decay);
        const float *__restrict sr = S + k * 128;
        for (int j = 0; j < 16; ++j) d[j] = _mm256_fmadd_ps(_mm256_loadu_ps(sr + j * 8), kb, d[j]);
    }
    for (int j = 0; j < 16; ++j) {
        const __m256 dv = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(v + j * 8), d[j]), betav);
        _mm256_store_ps(delta + j * 8, dv);
    }
    // pass B: S[k][:] = S[k][:]*decay + khat[k]*delta[:] ; out[:] += S_new[k][:]*qhat[k]
    __m256 o[16];
    for (int j = 0; j < 16; ++j) o[j] = _mm256_setzero_ps();
    const __m256 decv = _mm256_set1_ps(decay);
    for (int k = 0; k < 128; ++k) {
        const __m256 kv = _mm256_set1_ps(kh[k]), qv = _mm256_set1_ps(qh[k]);
        float *__restrict sr = S + k * 128;
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
inline void deltanet_ticket(void *p, int t) {
    const DeltaTicket &dt = *(const DeltaTicket *)p;
    const DeltaJob &j = dt.j;
    const int h0 = t * dt.heads_per_ticket;
    const int h1 = std::min(j.heads, h0 + dt.heads_per_ticket);
    for (int h = h0; h < h1; ++h)
        deltanet_head_step(j.state + size_t(h) * 128 * 128, j.q + size_t(h / j.kshare) * 128,
                           j.k + size_t(h / j.kshare) * 128, j.v + size_t(h) * 128,
                           j.g[h], j.b[h], j.out + size_t(h) * 128);
}
inline void deltanet_step_cpu(float *state, const float *q, const float *k, const float *v,
                              const float *g, const float *b, float *out,
                              int heads = 48, int kshare = 3) {
    if (heads <= 0 || heads % kshare) throw std::runtime_error("insignia cpu: bad deltanet heads");
    DeltaTicket dt{{state, q, k, v, g, b, out, heads, kshare}, std::max(1, heads / 6)};
    CpuPool::get().launch(deltanet_ticket, &dt, (heads + dt.heads_per_ticket - 1) / dt.heads_per_ticket);
}

// ─────────────────── full-attention: q/k norm + partial RoPE ───────────────────
// Mirrors src/ops.cu qk_norm_rope: per-head RMSNorm over 256 (eps 1e-6), bf16
// weight SHARED across heads ([256] — fixed vs the w3 design draft), partial
// rope on the first 64 dims, theta 1e7, pairs (i, i+32), i < 32.
inline void qk_norm_rope_cpu(float *__restrict q /*[24][256]*/, float *__restrict k /*[4][256]*/,
                             const uint16_t *__restrict qw /*[256]*/, const uint16_t *__restrict kw /*[256]*/,
                             int pos, int qheads = 24, int kvheads = 4) {
    float cs[32], sn[32];
    for (int i = 0; i < 32; ++i) {
        // angles in f64, single cast: pos*theta reaches ~1.2e3 rad; a float-side
        // pow/cos would inject ~1e-4 absolute error into the rotation.
        const double a = double(pos) * std::pow(1e7, -double(2 * i) / 64.0);
        cs[i] = float(std::cos(a)); sn[i] = float(std::sin(a));
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
    for (int h = 0; h < qheads; ++h) head(q + h * 256, qw);
    for (int h = 0; h < kvheads; ++h) head(k + h * 256, kw);
}

// q_proj is [q|gate] interleaved per head: src[h*512+d] = q, src[h*512+256+d] = gate.
inline void split_q_gate_cpu(const float *__restrict src, float *__restrict q,
                             float *__restrict gate, int qheads = 24) {
    for (int h = 0; h < qheads; ++h)
        for (int d = 0; d < 256; d += 8) {
            _mm256_storeu_ps(q + h * 256 + d, _mm256_loadu_ps(src + h * 512 + d));
            _mm256_storeu_ps(gate + h * 256 + d, _mm256_loadu_ps(src + h * 512 + 256 + d));
        }
}

// KV cache store, one token. kvrow = kvheads*256 = 1024. f32 or bf16 rows.
inline void store_kv_cpu(const float *__restrict k, const float *__restrict v,
                         float *__restrict kc, float *__restrict vc, int pos, int kvrow = 1024) {
    memcpy(kc + size_t(pos) * kvrow, k, kvrow * 4);
    memcpy(vc + size_t(pos) * kvrow, v, kvrow * 4);
}
inline void store_kv_bf16_cpu(const float *__restrict k, const float *__restrict v,
                              uint16_t *__restrict kc, uint16_t *__restrict vc, int pos, int kvrow = 1024) {
    for (int i = 0; i < kvrow; ++i) kc[size_t(pos) * kvrow + i] = f32_to_bf16_bits(k[i]);
    for (int i = 0; i < kvrow; ++i) vc[size_t(pos) * kvrow + i] = f32_to_bf16_bits(v[i]);
}

// ─────────────────────── GQA decode on CPU (§5.4) ───────────────────────
// 24 q-heads / 4 kv-heads (kvh = head/6), head_dim 256, scale 1/16.
// Parallelized over TOKEN RANGES (not heads): every thread streams a disjoint
// slice of the KV cache exactly once (16.8 MB @ctx2048 read from DRAM once, not
// 6x), computing a flash-style partial (running max m, sum l, partial out o) per
// head; the caller merges. KV may be f32 (engine today) or bf16 (halves traffic).
struct GqaScratch {                       // [nsplit][24][2 + 256]: m, l, o[]
    std::vector<float> buf;
    int nsplit = 0, heads = 24;
    float *at(int s, int h) { return buf.data() + (size_t(s) * heads + h) * 258; }
};

struct GqaJob {
    const float *__restrict q;             // [24][256]
    const uint8_t *__restrict kc, *__restrict vc;   // rows of 1024 elems/token, f32 or bf16
    int tokens, nsplit;
    bool bf16;
    GqaScratch *__restrict sc;
};
// kv-group-major, block-batched online softmax. The 6 q-heads of a kv group
// (27B GQA 6:1, kvh = hg — NOT head>>2) share each K/V row walk: pass 1 loads a
// K row once and dots it against all 6 heads (L1 re-reads), pass 2 accumulates
// per head in 4-register chunks. A per-head loop instead re-walks K/V 6x with a
// 4KB stride (TLB/L3-hostile; measured 2-3x slower and unstable), and a naive
// per-token o[32] spills to stack every token.
template <bool KVBF16>
inline void gqa_head_range(const GqaJob &J, int t0, int t1, float *part) {
    constexpr float NINF = -3.402823466e+38F;
    alignas(32) float sbuf[6][72];         // scores per group head; tail padded
    alignas(32) float pbuf[6][64];         // exp(s - mn)
    alignas(32) float obuf[6][256];        // running output per group head
    float mst[6], lst[6], rst[6];
    for (int hg = 0; hg < 4; ++hg) {
        const int h0 = hg * 6;
        for (int hh = 0; hh < 6; ++hh) {
            mst[hh] = NINF; lst[hh] = 0.f;
            memset(obuf[hh], 0, sizeof obuf[hh]);
        }
        for (int tb = t0; tb < t1; tb += 64) {
            const int te = std::min(t1, tb + 64);
            const int n = te - tb;
            // pass 1: one K row walk, 6 score dots per row
            for (int i = 0; i < n; ++i) {
                const size_t kbase = size_t(tb + i) * 1024 + size_t(hg) * 256;
                for (int hh = 0; hh < 6; ++hh) {
                    const float *__restrict qh = J.q + (h0 + hh) * 256;
                    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps(),
                           s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
                    if constexpr (KVBF16) {
                        const uint16_t *__restrict kp = (const uint16_t *)J.kc + kbase;
                        for (int d = 0; d < 256; d += 32) {
                            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(qh + d),      bf16_widen8(kp + d), s0);
                            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(qh + d + 8),  bf16_widen8(kp + d + 8), s1);
                            s2 = _mm256_fmadd_ps(_mm256_loadu_ps(qh + d + 16), bf16_widen8(kp + d + 16), s2);
                            s3 = _mm256_fmadd_ps(_mm256_loadu_ps(qh + d + 24), bf16_widen8(kp + d + 24), s3);
                        }
                    } else {
                        const float *__restrict kp = (const float *)J.kc + kbase;
                        for (int d = 0; d < 256; d += 32) {
                            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(qh + d),      _mm256_loadu_ps(kp + d), s0);
                            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(qh + d + 8),  _mm256_loadu_ps(kp + d + 8), s1);
                            s2 = _mm256_fmadd_ps(_mm256_loadu_ps(qh + d + 16), _mm256_loadu_ps(kp + d + 16), s2);
                            s3 = _mm256_fmadd_ps(_mm256_loadu_ps(qh + d + 24), _mm256_loadu_ps(kp + d + 24), s3);
                        }
                    }
                    sbuf[hh][i] = (hsum256_ps(_mm256_add_ps(s0, s1)) + hsum256_ps(_mm256_add_ps(s2, s3))) * 0.0625f;
                }
            }
            // per-head softmax update (vectorized exp, one pass per 8 scores)
            for (int hh = 0; hh < 6; ++hh) {
                for (int i = n; i < ((n + 7) & ~7); ++i) sbuf[hh][i] = -100.f;   // -> p = 0
                float mblk = NINF;
                for (int i = 0; i < n; ++i) mblk = sbuf[hh][i] > mblk ? sbuf[hh][i] : mblk;
                const float mn = mblk > mst[hh] ? mblk : mst[hh];
                const float r = (mst[hh] == NINF) ? 0.f : expf(mst[hh] - mn);
                float psum = 0.f;
                for (int i = 0; i < n; i += 8) {
                    const __m256 pv = vexp256_ps(_mm256_sub_ps(_mm256_load_ps(sbuf[hh] + i), _mm256_set1_ps(mn)));
                    _mm256_store_ps(pbuf[hh] + i, pv);
                    psum += hsum256_ps(pv);
                }
                lst[hh] = lst[hh] * r + psum;
                mst[hh] = mn; rst[hh] = r;
            }
            // pass 2: V accumulation per head, 8 chunks x 4 live registers
            for (int hh = 0; hh < 6; ++hh) {
                const __m256 rv = _mm256_set1_ps(rst[hh]);
                const float *__restrict pb = pbuf[hh];
                for (int ch = 0; ch < 8; ++ch) {
                    const int d0 = ch * 32;
                    __m256 c0 = _mm256_mul_ps(_mm256_load_ps(obuf[hh] + d0), rv),
                           c1 = _mm256_mul_ps(_mm256_load_ps(obuf[hh] + d0 + 8), rv),
                           c2 = _mm256_mul_ps(_mm256_load_ps(obuf[hh] + d0 + 16), rv),
                           c3 = _mm256_mul_ps(_mm256_load_ps(obuf[hh] + d0 + 24), rv);
                    for (int i = 0; i < n; ++i) {
                        const size_t vbase = size_t(tb + i) * 1024 + size_t(hg) * 256 + d0;
                        const __m256 p = _mm256_set1_ps(pb[i]);
                        if constexpr (KVBF16) {
                            const uint16_t *__restrict vp = (const uint16_t *)J.vc + vbase;
                            c0 = _mm256_fmadd_ps(p, bf16_widen8(vp + 0), c0);
                            c1 = _mm256_fmadd_ps(p, bf16_widen8(vp + 8), c1);
                            c2 = _mm256_fmadd_ps(p, bf16_widen8(vp + 16), c2);
                            c3 = _mm256_fmadd_ps(p, bf16_widen8(vp + 24), c3);
                        } else {
                            const float *__restrict vp = (const float *)J.vc + vbase;
                            c0 = _mm256_fmadd_ps(p, _mm256_loadu_ps(vp + 0), c0);
                            c1 = _mm256_fmadd_ps(p, _mm256_loadu_ps(vp + 8), c1);
                            c2 = _mm256_fmadd_ps(p, _mm256_loadu_ps(vp + 16), c2);
                            c3 = _mm256_fmadd_ps(p, _mm256_loadu_ps(vp + 24), c3);
                        }
                    }
                    _mm256_store_ps(obuf[hh] + d0, c0); _mm256_store_ps(obuf[hh] + d0 + 8, c1);
                    _mm256_store_ps(obuf[hh] + d0 + 16, c2); _mm256_store_ps(obuf[hh] + d0 + 24, c3);
                }
            }
        }
        for (int hh = 0; hh < 6; ++hh) {
            float *__restrict pa = part + size_t(h0 + hh) * 258;
            pa[0] = mst[hh]; pa[1] = lst[hh];
            for (int j = 0; j < 32; ++j) _mm256_storeu_ps(pa + 2 + j * 8, _mm256_load_ps(obuf[hh] + j * 8));
        }
    }
}
inline void gqa_ticket(void *p, int t) {
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
// out must be [24][256]. kc/vc: [tokens][1024] f32 or bf16 (kv_bf16).
inline void gqa_decode_cpu(const float *q, const void *kc, const void *vc, int tokens,
                           float *out, bool kv_bf16 = false, int nsplit = 6) {
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
    return f16_to_f32(h) * 256.f;
}
// Double-accumulated reference with raw bf16 scales (no x256 fold — true scale).
inline void fp8_gemv_f64_ref(const uint8_t *__restrict w, const uint16_t *__restrict sb,
                             const float *__restrict x, double *__restrict y, int rows, int cols) {
    const int kb = cols >> 7;
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
struct Parity { double cos, max_rel, max_abs_rel; };
// max_rel: per-element |y-ref|/|ref|, floored at floor_rel*refmax (elements that
// cancelled against the output scale have no meaningful relative error in fp32).
// max_abs_rel: max |y-ref| / refmax — the scale-normalized error, the honest
// gate for fp32-accumulated kernels against an f64 reference.
inline Parity compare_f64(const float *y, const double *ref, int n, double floor_rel = 1e-6) {
    double dot = 0, na = 0, nb = 0, mx = 0, mxa = 0, refmax = 0;
    for (int i = 0; i < n; ++i) refmax = std::max(refmax, std::fabs(ref[i]));
    for (int i = 0; i < n; ++i) {
        dot += double(y[i]) * ref[i];
        na += double(y[i]) * double(y[i]);
        nb += ref[i] * ref[i];
        const double d = std::fabs(double(y[i]) - ref[i]);
        mxa = std::max(mxa, d);
        if (std::fabs(ref[i]) > refmax * floor_rel) mx = std::max(mx, d / std::fabs(ref[i]));
    }
    return {dot / (std::sqrt(na) * std::sqrt(nb) + 1e-300), mx, mxa / (refmax + 1e-300)};
}

}  // namespace insignia::cpu
