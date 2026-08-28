// test_cpu.cpp — correctness harness + benchmarks for include/insignia_cpu.hpp
// (CPU tier for RAM-resident Qwen3.8-27B-FP8 layers; see audits/w3/cpu-fp8.md §7).
//
// Build:
//   cl /nologo /arch:AVX2 /O2 /std:c++20 /fp:precise /EHsc /Iinclude src\test_cpu.cpp /Fe:build\test_cpu.exe
//
// Parity policy (mission): GEMV/bf16 GEMV rel<=1e-4 & cos>0.999999 vs a double
// reference; norms/deltanet/gqa "exact-ish" rel<=1e-5. Weight quantizers are the
// same non-RNE rounding as src/test_fp8.cu (self-consistent).
#include "insignia_cpu.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <string>
#include <vector>

using namespace insignia;
using namespace insignia::cpu;

// ─────────────────────── quantizers (copied from src/test_fp8.cu) ───────────────────────
static float e4m3_host(unsigned code) {
    const unsigned mag = code & 0x7f, e = mag >> 3, m = mag & 7;
    float v;
    if (e == 0) v = float(m) * (1.f / 512.f);            // m * 2^-9 (subnormal)
    else v = ldexpf(1.f + m / 8.f, int(e) - 7);          // (1+m/8)*2^(e-7), bias 7
    return (code & 0x80) ? -v : v;
}
static uint8_t f32_to_e4m3(float v) {
    if (v < 0) return f32_to_e4m3(-v) | 0x80;
    if (!(v < 448.f)) return 0x7e;  // clamp to max normal
    int e;
    const float fr = frexpf(v, &e);  // v = fr * 2^e, 0.5 <= fr < 1
    int exp = e + 6;                 // e4m3 exponent field, bias 7
    if (exp <= 0) {                  // subnormal: value = m * 2^-9
        const int m = int(v * 512.f + 0.5f);
        return uint8_t(m < 8 ? m : 7);
    }
    int mant = int((fr * 2.f - 1.f) * 8.f + 0.5f);
    if (mant >= 8) { mant = 0; ++exp; }
    if (exp > 15) { return 0x7e; }
    return uint8_t((exp << 3) | mant);
}

// ─────────────────────── checking ───────────────────────
// Gate: cosine + floored max-relative + scale-normalized max-absolute error
// (max_rel floors at 1e-3*refmax for cancellation-prone kernels — an element that
// cancelled 1000:1 against the output scale has no measurable fp32 relative
// error; max_abs_rel is the honest bound). All three are printed either way.
static int failures = 0, checks = 0;
static void report(const char *name, const Parity &m, double rel_tol, double cos_tol, double abs_tol) {
    ++checks;
    const bool ok = m.max_rel <= rel_tol && m.cos >= cos_tol && m.max_abs_rel <= abs_tol;
    if (!ok) ++failures;
    printf("%-30s %s  cos=%.10f rel=%.2e abs=%.2e (tol rel %.0e abs %.0e)\n",
           name, ok ? "PASS" : "FAIL", m.cos, m.max_rel, m.max_abs_rel, rel_tol, abs_tol);
}
static void check_bool(const char *name, bool ok, const char *detail = "") {
    ++checks;
    if (!ok) ++failures;
    printf("%-30s %s  %s\n", name, ok ? "PASS" : "FAIL", detail);
}

// ─────────────────────── unit tests ───────────────────────
static void test_dequant_exhaustive() {
    alignas(32) uint8_t bytes[256];
    for (int b = 0; b < 256; ++b) bytes[b] = uint8_t(b);
    int bad = 0;
    for (int base = 0; base < 256; base += 32) {
        __m256 out[4];
        e4m3x32_f32(_mm256_loadu_si256((const __m256i *)(bytes + base)), out);
        alignas(32) float got[32];
        for (int q = 0; q < 4; ++q) _mm256_store_ps(got + q * 8, out[q]);
        for (int i = 0; i < 32; ++i) {
            // e4m3x32_f32 yields the raw fp16-widened value: e4m3_value / 256, exactly.
            const float want = e4m3_host(base + i) * (1.f / 256.f);
            if (got[i] != want) { if (bad < 4) printf("  code 0x%02x: got %.9g want %.9g\n", base + i, got[i], want); ++bad; }
        }
    }
    char buf[64];
    snprintf(buf, sizeof buf, "bad=%d/256", bad);
    check_bool("e4m3x32_f32 exhaustive", bad == 0, buf);
}

static void test_scale_fold_exhaustive() {
    int bad = 0, nan_ok = 0;
    for (uint32_t u = 0; u < 65536; ++u) {
        const float a = bf16_scale_x256(uint16_t(u));
        const float b = bf16_to_f32(uint16_t(u)) * 256.f;
        if (std::isnan(a) && std::isnan(b)) { ++nan_ok; continue; }
        uint32_t ba, bb;
        memcpy(&ba, &a, 4); memcpy(&bb, &b, 4);
        if (ba != bb) { if (bad < 4) printf("  bf16 0x%04x: %g(0x%08x) vs %g(0x%08x)\n", u, a, ba, b, bb); ++bad; }
    }
    char buf[64];
    snprintf(buf, sizeof buf, "bad=%d/65536 nan=%d", bad, nan_ok);
    check_bool("bf16_scale_x256 exhaustive", bad == 0, buf);
}

// fp8 GEMV parity: per-128x128-block absmax weights, same recipe as test_fp8.cu.
struct Fp8Mat {
    std::vector<uint8_t> w;
    std::vector<uint16_t> s;   // bf16 scales [rows/128][cols/128]
    std::vector<float> s256;   // folded, for the kernel
    int rows, cols;
};
static Fp8Mat make_fp8_mat(int rows, int cols, std::mt19937 &rng) {
    std::normal_distribution<float> nd(0.f, 0.05f);
    Fp8Mat m;
    m.rows = rows; m.cols = cols;
    m.w.resize(size_t(rows) * cols);
    const int kr = rows >> 7, kc = cols >> 7;
    m.s.resize(size_t(kr) * kc);
    m.s256.resize(size_t(kr) * kc);
    for (int br = 0; br < kr; ++br)
        for (int bc = 0; bc < kc; ++bc) {
            float amax = 0;
            for (int i = 0; i < 128 * 128; ++i) amax = std::fmax(amax, std::fabs(nd(rng)));
            const float sc = amax / 448.f;
            m.s[br * kc + bc] = f32_to_bf16_bits(sc);
            const float scf = bf16_to_f32(m.s[br * kc + bc]);
            for (int r = 0; r < 128; ++r)
                for (int c = 0; c < 128; ++c)
                    m.w[size_t(br * 128 + r) * cols + bc * 128 + c] = f32_to_e4m3(nd(rng) / scf);
        }
    fp8_prepare_scales(m.s.data(), m.s256.data(), m.s.size());
    return m;
}

static void test_fp8_gemv(std::mt19937 &rng) {
    std::normal_distribution<float> xd(0.f, 1.f);
    struct Shape { int rows, cols; };
    for (Shape sh : {Shape{2560, 5120}, Shape{256, 384}, Shape{1024, 5120}}) {
        Fp8Mat m = make_fp8_mat(sh.rows, sh.cols, rng);
        std::vector<float> x(sh.cols), y(sh.rows), y2(2 * sh.rows), xp(2 * sh.cols);
        for (auto &v : x) v = xd(rng);
        for (int i = 0; i < 2 * sh.cols; ++i) xp[i] = xd(rng);
        std::vector<double> ref(sh.rows), refp(2 * sh.rows);
        fp8_gemv_f64_ref(m.w.data(), m.s.data(), x.data(), ref.data(), sh.rows, sh.cols);
        char name[80];
        snprintf(name, sizeof name, "fp8_gemv %dx%d", sh.rows, sh.cols);
        fp8_gemv_mt(m.w.data(), m.s256.data(), x.data(), y.data(), sh.rows, sh.cols);
        report(name, compare_f64(y.data(), ref.data(), sh.rows, 1e-3), 1e-4, 0.999999, 1e-5);
        snprintf(name, sizeof name, "fp8_gemv2 %dx%d", sh.rows, sh.cols);
        fp8_gemv2_mt(m.w.data(), m.s256.data(), xp.data(), y2.data(), sh.rows, sh.cols);
        fp8_gemv_f64_ref(m.w.data(), m.s.data(), xp.data(), refp.data(), sh.rows, sh.cols);
        fp8_gemv_f64_ref(m.w.data(), m.s.data(), xp.data() + sh.cols, refp.data() + sh.rows, sh.rows, sh.cols);
        report(name, compare_f64(y2.data(), refp.data(), 2 * sh.rows, 1e-3), 1e-4, 0.999999, 1e-5);
        // serial path must match the pool path bit-exactly
        std::vector<float> ys(sh.rows);
        fp8_gemv_st(m.w.data(), m.s256.data(), x.data(), ys.data(), sh.rows, sh.cols);
        snprintf(name, sizeof name, "fp8_gemv st==mt %dx%d", sh.rows, sh.cols);
        check_bool(name, memcmp(ys.data(), y.data(), sh.rows * 4) == 0);
    }
}

static void test_bf16_gemv(std::mt19937 &rng) {
    std::normal_distribution<float> nd(0.f, 0.05f), xd(0.f, 1.f);
    struct Shape { int rows, cols; };
    for (Shape sh : {Shape{48, 5120}, Shape{512, 640}}) {
        std::vector<uint16_t> w(size_t(sh.rows) * sh.cols);
        for (auto &v : w) v = f32_to_bf16_bits(nd(rng));
        std::vector<float> x(sh.cols), y(sh.rows);
        for (auto &v : x) v = xd(rng);
        std::vector<double> ref(sh.rows);
        for (int r = 0; r < sh.rows; ++r) {
            double acc = 0;
            for (int c = 0; c < sh.cols; ++c) acc += double(bf16_to_f32(w[size_t(r) * sh.cols + c])) * double(x[c]);
            ref[r] = acc;
        }
        bf16_gemv_mt(w.data(), x.data(), y.data(), sh.rows, sh.cols);
        char name[80];
        snprintf(name, sizeof name, "bf16_gemv %dx%d", sh.rows, sh.cols);
        report(name, compare_f64(y.data(), ref.data(), sh.rows, 1e-3), 1e-4, 0.999999, 1e-5);
    }
}

static void test_rmsnorm(std::mt19937 &rng) {
    std::normal_distribution<float> nd(0.f, 1.f), wd(0.f, 0.2f);
    const int cols = 5120;
    std::vector<float> x(cols), y(cols);
    std::vector<uint16_t> w(cols);
    for (auto &v : x) v = nd(rng);
    for (auto &v : w) v = f32_to_bf16_bits(wd(rng) + 1.f);
    for (int zc = 0; zc <= 1; ++zc) {
        std::vector<double> ref(cols);
        double ss = 0;
        for (int c = 0; c < cols; ++c) ss += double(x[c]) * x[c];
        const double inv = 1.0 / std::sqrt(ss / cols + 1e-6);
        for (int c = 0; c < cols; ++c) {
            const double wv = bf16_to_f32(w[c]);
            ref[c] = double(x[c]) * inv * (zc ? 1.0 + wv : wv);
        }
        rmsnorm_cpu(x.data(), w.data(), y.data(), cols, zc != 0);
        report(zc ? "rmsnorm zero-centered" : "rmsnorm", compare_f64(y.data(), ref.data(), cols), 1e-5, 0.9999999, 1e-5);
    }
}

static void test_gated_rmsnorm(std::mt19937 &rng) {
    std::normal_distribution<float> nd(0.f, 1.f), wd(0.f, 0.2f);
    const int heads = 48, hd = 128;
    std::vector<float> x(size_t(heads) * hd), g(size_t(heads) * hd), y(size_t(heads) * hd);
    std::vector<uint16_t> w(hd);                       // [128] SHARED across heads (engine layout)
    for (auto &v : x) v = nd(rng);
    for (auto &v : g) v = nd(rng);
    for (auto &v : w) v = f32_to_bf16_bits(wd(rng) + 1.f);
    std::vector<double> ref(size_t(heads) * hd);
    for (int h = 0; h < heads; ++h) {
        double ss = 0;
        for (int i = 0; i < hd; ++i) ss += double(x[h * hd + i]) * x[h * hd + i];
        const double inv = 1.0 / std::sqrt(ss / hd + 1e-6);
        for (int i = 0; i < hd; ++i) {
            const double gv = g[h * hd + i];
            ref[h * hd + i] = double(x[h * hd + i]) * inv * bf16_to_f32(w[i]) * (gv / (1.0 + std::exp(-gv)));
        }
    }
    gated_rmsnorm_per_head_cpu(x.data(), w.data(), g.data(), y.data(), heads, hd);
    report("gated_rmsnorm_per_head", compare_f64(y.data(), ref.data(), heads * hd), 1e-5, 0.9999999, 1e-5);
}

static void test_silu_mul(std::mt19937 &rng) {
    std::normal_distribution<float> nd(0.f, 2.f);
    const int n = 17408;
    std::vector<float> g(n), u(n), y(n);
    for (auto &v : g) v = nd(rng);
    for (auto &v : u) v = nd(rng);
    std::vector<double> ref(n);
    for (int i = 0; i < n; ++i) ref[i] = (double(g[i]) / (1.0 + std::exp(-double(g[i])))) * u[i];
    silu_mul_cpu(g.data(), u.data(), y.data(), n);
    report("silu_mul 17408", compare_f64(y.data(), ref.data(), n), 1e-5, 0.9999999, 1e-5);

    std::vector<float> xx(n), gg(n);
    for (auto &v : xx) v = nd(rng);
    for (auto &v : gg) v = nd(rng);
    std::vector<double> ref2(n);
    for (int i = 0; i < n; ++i) ref2[i] = double(xx[i]) / (1.0 + std::exp(-double(gg[i])));
    sigmoid_mul_cpu(xx.data(), gg.data(), n);
    report("sigmoid_mul 17408", compare_f64(xx.data(), ref2.data(), n), 1e-5, 0.9999999, 1e-5);
}

static void test_conv(std::mt19937 &rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    const int ch = 10240;
    std::vector<uint16_t> w16(size_t(ch) * 4);
    for (auto &v : w16) v = f32_to_bf16_bits(nd(rng) * 0.2f);
    std::vector<float> wt(size_t(ch) * 4);
    expand_conv_weights(w16.data(), wt.data(), ch);
    std::vector<float> state(size_t(ch) * 3), x(ch), ref_state(size_t(ch) * 3);
    for (auto &v : state) v = nd(rng) * 0.5f;
    ref_state = state;
    for (int step = 0; step < 2; ++step) {
        for (auto &v : x) v = nd(rng);
        std::vector<double> ref(ch);
        for (int c = 0; c < ch; ++c) {
            const double z = double(ref_state[c * 3 + 0]) * bf16_to_f32(w16[c * 4 + 0]) +
                             double(ref_state[c * 3 + 1]) * bf16_to_f32(w16[c * 4 + 1]) +
                             double(ref_state[c * 3 + 2]) * bf16_to_f32(w16[c * 4 + 2]) +
                             double(x[c]) * bf16_to_f32(w16[c * 4 + 3]);
            ref_state[c * 3 + 0] = ref_state[c * 3 + 1];
            ref_state[c * 3 + 1] = ref_state[c * 3 + 2];
            ref_state[c * 3 + 2] = x[c];
            ref[c] = z / (1.0 + std::exp(-z));
        }
        causal_conv4_silu_cpu(x.data(), state.data(), wt.data(), ch);
        char name[64];
        snprintf(name, sizeof name, "conv1d+silu step%d", step);
        report(name, compare_f64(x.data(), ref.data(), ch, 1e-3), 1e-5, 0.9999999, 1e-5);
        bool st_ok = true;
        for (int c = 0; c < ch; ++c)
            for (int i = 0; i < 3; ++i)
                if (state[c * 3 + i] != ref_state[c * 3 + i]) st_ok = false;
        snprintf(name, sizeof name, "conv1d state shift step%d", step);
        check_bool(name, st_ok);
    }
}

static void test_deltanet_params(std::mt19937 &rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    const int heads = 48;
    std::vector<float> a(heads), b(heads), A(heads);
    std::vector<uint16_t> dt(heads);
    for (auto &v : a) v = nd(rng);
    for (auto &v : b) v = nd(rng);
    for (auto &v : A) v = nd(rng) * 0.5f;
    for (auto &v : dt) v = f32_to_bf16_bits(nd(rng));
    std::vector<double> ra(heads), rb(heads);
    for (int h = 0; h < heads; ++h) {
        rb[h] = 1.0 / (1.0 + std::exp(-double(b[h])));
        const double z = double(a[h]) + bf16_to_f32(dt[h]);
        const double soft = z > 20.0 ? z : std::log1p(std::exp(z));
        ra[h] = -std::exp(double(A[h])) * soft;
    }
    deltanet_parameters_cpu(a.data(), b.data(), A.data(), dt.data(), heads);
    report("deltanet_parameters a", compare_f64(a.data(), ra.data(), heads), 1e-5, 0.9999999, 1e-5);
    report("deltanet_parameters b", compare_f64(b.data(), rb.data(), heads), 1e-5, 0.9999999, 1e-5);
}

// Double reference mirroring src/deltanet.cu semantics (S[k*128+v], raw q/k
// vectors scaled per head, q-norm folds 1/sqrt(128), eps inside rsqrt).
static void deltanet_step_ref(std::vector<double> &S, const double *q, const double *k,
                              const double *v, const double *g, const double *b,
                              double *out, int heads, int kshare) {
    for (int h = 0; h < heads; ++h) {
        const double *qh = q + size_t(h / kshare) * 128;
        const double *kh = k + size_t(h / kshare) * 128;
        const double *vh = v + size_t(h) * 128;
        double *Sh = S.data() + size_t(h) * 128 * 128;
        double sq = 0, sk = 0;
        for (int i = 0; i < 128; ++i) { sq += qh[i] * qh[i]; sk += kh[i] * kh[i]; }
        const double qs = 1.0 / std::sqrt(sq + 1e-6) * 0.08838834764831845;
        const double ks = 1.0 / std::sqrt(sk + 1e-6);
        const double decay = std::exp(g[h]);
        alignas(64) double qh_[128], kh_[128], delta[128];
        for (int i = 0; i < 128; ++i) { qh_[i] = qh[i] * qs; kh_[i] = kh[i] * ks; }
        for (int j = 0; j < 128; ++j) {
            double dot = 0;
            for (int i = 0; i < 128; ++i) dot += Sh[size_t(i) * 128 + j] * decay * kh_[i];
            delta[j] = (vh[j] - dot) * b[h];
        }
        for (int j = 0; j < 128; ++j) {
            double acc = 0;
            for (int i = 0; i < 128; ++i) {
                const double cell = Sh[size_t(i) * 128 + j] * decay + kh_[i] * delta[j];
                Sh[size_t(i) * 128 + j] = cell;
                acc += cell * qh_[i];
            }
            out[size_t(h) * 128 + j] = acc;
        }
    }
}

static void test_deltanet_step(std::mt19937 &rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    struct Cfg { int heads, kshare, steps; };
    for (Cfg cfg : {Cfg{48, 3, 2}, Cfg{32, 2, 1}}) {
        const int heads = cfg.heads, kheads = heads / cfg.kshare;
        std::vector<float> S(size_t(heads) * 128 * 128), q(size_t(kheads) * 128),
            k(size_t(kheads) * 128), v(size_t(heads) * 128), g(heads), b(heads),
            out(size_t(heads) * 128);
        for (auto &s : S) s = nd(rng) * 0.05f;
        for (auto &s : q) s = nd(rng);
        for (auto &s : k) s = nd(rng);
        for (auto &s : v) s = nd(rng);
        for (auto &s : g) s = nd(rng) * 0.2f;
        for (auto &s : b) s = 0.5f + 0.4f * nd(rng);
        std::vector<double> Sref(S.begin(), S.end());
        for (int step = 0; step < cfg.steps; ++step) {
            if (step) {  // evolve inputs so state chaining is exercised
                for (auto &s : q) s = nd(rng);
                for (auto &s : v) s = nd(rng);
            }
            std::vector<double> qr(q.begin(), q.end()), kr(k.begin(), k.end()),
                vr(v.begin(), v.end()), gr(g.begin(), g.end()), br(b.begin(), b.end()),
                oref(size_t(heads) * 128);
            deltanet_step_ref(Sref, qr.data(), kr.data(), vr.data(), gr.data(), br.data(),
                              oref.data(), heads, cfg.kshare);
            deltanet_step_cpu(S.data(), q.data(), k.data(), v.data(), g.data(), b.data(),
                              out.data(), heads, cfg.kshare);
            char name[64];
            snprintf(name, sizeof name, "deltanet out h%d/k%d s%d", heads, cfg.kshare, step);
            report(name, compare_f64(out.data(), oref.data(), heads * 128, 1e-3), 1e-3, 0.9999999, 1e-5);
            snprintf(name, sizeof name, "deltanet state h%d/k%d s%d", heads, cfg.kshare, step);
            report(name, compare_f64(S.data(), Sref.data(), heads * 128 * 128, 1e-3), 1e-3, 0.9999999, 1e-5);
        }
    }
}

static void test_qk_norm_rope(std::mt19937 &rng) {
    std::normal_distribution<float> nd(0.f, 1.f), wd(0.f, 0.2f);
    const int qh = 24, kvh = 4;
    std::vector<float> q(size_t(qh) * 256), k(size_t(kvh) * 256);
    std::vector<uint16_t> qw(256), kw(256);            // [256] SHARED across heads (engine layout)
    for (auto &v : q) v = nd(rng);
    for (auto &v : k) v = nd(rng);
    for (auto &v : qw) v = f32_to_bf16_bits(wd(rng) + 1.f);
    for (auto &v : kw) v = f32_to_bf16_bits(wd(rng) + 1.f);
    for (int pos : {0, 1234}) {
        std::vector<float> qc(q), kc(k);
        std::vector<double> ref(size_t(qh + kvh) * 256);
        float cs[32], sn[32];
        for (int i = 0; i < 32; ++i) {
            const double a = double(pos) * std::pow(1e7, -double(2 * i) / 64.0);
            cs[i] = float(std::cos(a)); sn[i] = float(std::sin(a));
        }
        for (int h = 0; h < qh + kvh; ++h) {
            const bool isq = h < qh;
            std::vector<double> p(isq ? qc.begin() + h * 256 : kc.begin() + (h - qh) * 256,
                                  isq ? qc.begin() + (h + 1) * 256 : kc.begin() + (h - qh + 1) * 256);
            const uint16_t *w = isq ? qw.data() : kw.data();
            double ss = 0;
            for (int d = 0; d < 256; ++d) ss += p[d] * p[d];
            const double nsc = 1.0 / std::sqrt(ss / 256.0 + 1e-6);
            for (int d = 0; d < 256; ++d) p[d] = p[d] * nsc * bf16_to_f32(w[d]);
            if (pos)
                for (int i = 0; i < 32; ++i) {
                    const double a = p[i], b2 = p[i + 32];
                    p[i] = double(a * cs[i] - b2 * sn[i]);
                    p[i + 32] = double(a * sn[i] + b2 * cs[i]);
                }
            for (int d = 0; d < 256; ++d) ref[size_t(h) * 256 + d] = p[d];
        }
        qk_norm_rope_cpu(qc.data(), kc.data(), qw.data(), kw.data(), pos, qh, kvh);
        char name[64];
        snprintf(name, sizeof name, "qk_norm_rope pos=%d", pos);
        report(name, compare_f64(qc.data(), ref.data(), qh * 256, 1e-3), 1e-5, 0.9999999, 1e-5);
        snprintf(name, sizeof name, "k_norm_rope pos=%d", pos);
        report(name, compare_f64(kc.data(), ref.data() + size_t(qh) * 256, kvh * 256, 1e-3), 1e-5, 0.9999999, 1e-5);
    }
}

static void test_split_and_store(std::mt19937 &rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> src(size_t(24) * 512), q(24 * 256), g(24 * 256);
    for (auto &v : src) v = nd(rng);
    split_q_gate_cpu(src.data(), q.data(), g.data(), 24);
    bool ok = true;
    for (int h = 0; h < 24; ++h)
        for (int d = 0; d < 256; ++d)
            if (q[h * 256 + d] != src[h * 512 + d] || g[h * 256 + d] != src[h * 512 + 256 + d]) ok = false;
    check_bool("split_q_gate", ok);

    std::vector<float> k(1024), v(1024), kc(8 * 1024, -1.f), vc(8 * 1024, -1.f);
    std::vector<uint16_t> kc16(8 * 1024, 0), vc16(8 * 1024, 0);
    for (auto &x : k) x = nd(rng);
    for (auto &x : v) x = nd(rng);
    store_kv_cpu(k.data(), v.data(), kc.data(), vc.data(), 5);
    ok = memcmp(kc.data() + 5 * 1024, k.data(), 4096) == 0 && memcmp(vc.data() + 5 * 1024, v.data(), 4096) == 0;
    for (int t = 0; t < 8 && ok; ++t)     // every other row untouched
        if (t != 5) ok = kc[t * 1024] == -1.f && vc[t * 1024] == -1.f;
    check_bool("store_kv_cpu f32", ok);
    store_kv_bf16_cpu(k.data(), v.data(), kc16.data(), vc16.data(), 5);
    ok = true;
    for (int i = 0; i < 1024 && ok; ++i)
        ok = kc16[5 * 1024 + i] == f32_to_bf16_bits(k[i]) && vc16[5 * 1024 + i] == f32_to_bf16_bits(v[i]);
    for (int t = 0; t < 8 && ok; ++t)
        if (t != 5) ok = kc16[t * 1024] == 0 && vc16[t * 1024] == 0;
    check_bool("store_kv_bf16_cpu", ok);
}

// GQA double reference: plain softmax, mirroring src/attention.cu (kvh=head/6,
// scale 1/16).
static void gqa_ref(const float *q, const std::vector<float> &k, const std::vector<float> &v,
                    int tokens, std::vector<double> &out) {
    out.assign(24 * 256, 0.0);
    for (int h = 0; h < 24; ++h) {
        std::vector<double> s(tokens);
        double mx = -1e300;
        for (int t = 0; t < tokens; ++t) {
            double z = 0;
            const float *kt = k.data() + size_t(t) * 1024 + size_t(h / 6) * 256;
            for (int d = 0; d < 256; ++d) z += double(q[h * 256 + d]) * kt[d];
            s[t] = z / 16.0;
            mx = std::max(mx, s[t]);
        }
        double den = 0;
        for (int t = 0; t < tokens; ++t) { s[t] = std::exp(s[t] - mx); den += s[t]; }
        for (int t = 0; t < tokens; ++t) {
            const float *vt = v.data() + size_t(t) * 1024 + size_t(h / 6) * 256;
            for (int d = 0; d < 256; ++d) out[size_t(h) * 256 + d] += s[t] * vt[d] / den;
        }
    }
}

static void test_gqa(std::mt19937 &rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int tokens : {2048, 77}) {
        std::vector<float> q(24 * 256), out(24 * 256);
        for (auto &x : q) x = nd(rng);
        std::vector<float> k(size_t(tokens) * 1024), v(size_t(tokens) * 1024);
        for (auto &x : k) x = nd(rng);
        for (auto &x : v) x = nd(rng);
        std::vector<double> ref;
        gqa_ref(q.data(), k, v, tokens, ref);
        gqa_decode_cpu(q.data(), k.data(), v.data(), tokens, out.data(), false);
        char name[64];
        snprintf(name, sizeof name, "gqa_decode f32 t=%d", tokens);
        report(name, compare_f64(out.data(), ref.data(), 24 * 256, 1e-3), 1e-3, 0.9999999, 1e-5);
        if (tokens == 2048) {
            std::vector<uint16_t> k16(size_t(tokens) * 1024), v16(size_t(tokens) * 1024);
            for (size_t i = 0; i < k16.size(); ++i) { k16[i] = f32_to_bf16_bits(k[i]); v16[i] = f32_to_bf16_bits(v[i]); }
            for (size_t i = 0; i < k16.size(); ++i) { k[i] = bf16_to_f32(k16[i]); v[i] = bf16_to_f32(v16[i]); }
            gqa_ref(q.data(), k, v, tokens, ref);
            gqa_decode_cpu(q.data(), k16.data(), v16.data(), tokens, out.data(), true);
            report("gqa_decode bf16 t=2048", compare_f64(out.data(), ref.data(), 24 * 256, 1e-3), 1e-3, 0.9999999, 1e-5);
        }
    }
}

// ─────────────────────── benchmarks ───────────────────────
static std::normal_distribution<float> nd2(0.f, 1.f);
static double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}
template <typename F>
static double bench_min_ms(F &&f, int iters, int warmup = 2) {
    for (int i = 0; i < warmup; ++i) f();
    double best = 1e30;
    for (int i = 0; i < iters; ++i) {
        const double t0 = now_ms();
        f();
        best = std::min(best, now_ms() - t0);
    }
    return best;
}

struct BenchMat {
    std::vector<uint8_t> w;
    std::vector<float> s256, x, y;
    int rows, cols;
};
static BenchMat make_bench_mat(int rows, int cols, std::mt19937 &rng) {
    BenchMat b;
    b.rows = rows; b.cols = cols;
    b.w.resize(size_t(rows) * cols);
    for (size_t i = 0; i < b.w.size(); ++i) b.w[i] = uint8_t(rng() & 0xff);
    const int kr = (rows + 127) >> 7, kc = (cols + 127) >> 7;
    std::vector<uint16_t> s(size_t(kr) * kc);
    std::uniform_real_distribution<float> sd(0.001f, 0.01f);
    for (auto &v : s) v = f32_to_bf16_bits(sd(rng));
    b.s256.resize(size_t(kr) * kc);
    fp8_prepare_scales(s.data(), b.s256.data(), s.size());
    b.x.resize(cols);
    for (auto &v : b.x) v = nd2(rng);
    b.y.resize(rows);
    return b;
}

static double bench_fp8_mt(BenchMat &b, int iters) {
    return bench_min_ms([&] { fp8_gemv_mt(b.w.data(), b.s256.data(), b.x.data(), b.y.data(), b.rows, b.cols); }, iters);
}

static void run_bench(std::mt19937 &rng) {
    const int pool_workers = CpuPool::get().threads();
    printf("\n-- benchmarks (workers=%d + participating main thread) --\n", pool_workers);

    // pool scaling: 1 thread (serial) vs pool on the in_proj_qkv shape
    BenchMat qkv = make_bench_mat(10240, 5120, rng);
    const double bytes = double(qkv.rows) * qkv.cols;
    const double t1 = bench_min_ms([&] { fp8_gemv_st(qkv.w.data(), qkv.s256.data(), qkv.x.data(), qkv.y.data(), qkv.rows, qkv.cols); }, 10);
    const double t6 = bench_fp8_mt(qkv, 60);
    printf("pool scaling 10240x5120: 1T %.3f ms (%.1f GB/s) | %dT+main %.3f ms (%.1f GB/s) | speedup %.2fx\n",
           t1, bytes / t1 / 1e6, pool_workers, t6, bytes / t6 / 1e6, t1 / t6);

    // pair GEMV (MTP verify): one weight pass, two tokens
    {
        std::vector<float> x2(2 * qkv.cols), y2(2 * qkv.rows);
        const double tp = bench_min_ms([&] { fp8_gemv2_mt(qkv.w.data(), qkv.s256.data(), x2.data(), y2.data(), qkv.rows, qkv.cols); }, 30);
        printf("fp8_gemv2 10240x5120 T=2:  %.3f ms (stream %.1f GB/s, effective %.1f GB/s)\n",
               tp, bytes / tp / 1e6, 2 * bytes / tp / 1e6);
    }

    // 27B layer shapes
    struct Shape { const char *name; int rows, cols; };
    const Shape shapes[] = {
        {"in_proj_qkv 10240x5120", 10240, 5120},
        {"in_proj_z    6144x5120",   6144, 5120},
        {"out_proj     5120x6144",   5120, 6144},
        {"mlp gate/up  17408x5120", 17408, 5120},
        {"mlp down     5120x17408",  5120, 17408},
    };
    double t[5] = {0, 0, 0, 0, 0};
    printf("%-24s %10s %10s %10s\n", "shape", "MB", "ms", "GB/s");
    for (int i = 0; i < 5; ++i) {
        BenchMat b = make_bench_mat(shapes[i].rows, shapes[i].cols, rng);
        t[i] = bench_fp8_mt(b, 24);
        printf("%-24s %10.2f %10.3f %10.1f\n", shapes[i].name,
               double(b.rows) * b.cols / 1e6, t[i], double(b.rows) * b.cols / t[i] / 1e6);
    }
    // linear-attention layer = qkv + z + out + 2x(gate/up) + down (all F8)
    const double linear_ms = t[0] + t[1] + t[2] + 2 * t[3] + t[4];
    const double linear_mb = 10240.0 * 5120 + 6144.0 * 5120 + 5120.0 * 6144 +
                             2.0 * 17408.0 * 5120 + 5120.0 * 17408;
    printf("27B linear layer (F8 GEMV only): %.2f ms (%.1f GB/s avg, %.1f MB)\n",
           linear_ms, linear_mb / linear_ms / 1e6, linear_mb / 1e6);

    // deltanet step timing
    {
        const int heads = 48;
        std::vector<float> S(size_t(heads) * 128 * 128), q(16 * 128), k(16 * 128),
            v(size_t(heads) * 128), g(heads), b(heads), out(size_t(heads) * 128);
        for (auto &s : S) s = nd2(rng) * 0.05f;
        for (auto &s : q) s = nd2(rng);
        for (auto &s : k) s = nd2(rng);
        for (auto &s : v) s = nd2(rng);
        for (auto &s : g) s = nd2(rng) * 0.2f;
        for (auto &s : b) s = 0.7f;
        const double td = bench_min_ms([&] { deltanet_step_cpu(S.data(), q.data(), k.data(), v.data(), g.data(), b.data(), out.data(), heads, 3); }, 60);
        printf("deltanet_step 48 heads:    %.0f us (%.1f GB/s on 3x3.15MB state traffic)\n",
               td * 1000.0, 9.437184 / td);
    }

    // GQA decode timing (ctx 2048)
    {
        const int tokens = 2048;
        std::vector<float> q(24 * 256), out(24 * 256);
        std::vector<float> k(size_t(tokens) * 1024), v(size_t(tokens) * 1024);
        std::vector<uint16_t> k16(size_t(tokens) * 1024), v16(size_t(tokens) * 1024);
        for (auto &x : q) x = nd2(rng);
        for (size_t i = 0; i < k.size(); ++i) {
            k[i] = nd2(rng); v[i] = nd2(rng);
            k16[i] = f32_to_bf16_bits(k[i]); v16[i] = f32_to_bf16_bits(v[i]);
        }
        const double mb = double(tokens) * 2 * 1024 * 4 / 1e6;
        const double tf = bench_min_ms([&] { gqa_decode_cpu(q.data(), k.data(), v.data(), tokens, out.data(), false); }, 20);
        const double tb = bench_min_ms([&] { gqa_decode_cpu(q.data(), k16.data(), v16.data(), tokens, out.data(), true); }, 20);
        printf("gqa_decode ctx2048:        f32 %.3f ms (%.1f GB/s) | bf16 %.3f ms (%.1f GB/s)\n",
               tf, mb / tf, tb, mb / 2.0 / tb);
    }
}

int main(int argc, char **argv) {
    const std::string mode = argc > 1 ? argv[1] : "all";
    std::mt19937 rng(777);
    printf("insignia_cpu test/bench — mode=%s, hardware_concurrency=%u\n",
           mode.c_str(), std::thread::hardware_concurrency());
    if (mode == "all" || mode == "test") {
        test_dequant_exhaustive();
        test_scale_fold_exhaustive();
        test_fp8_gemv(rng);
        test_bf16_gemv(rng);
        test_rmsnorm(rng);
        test_gated_rmsnorm(rng);
        test_silu_mul(rng);
        test_conv(rng);
        test_deltanet_params(rng);
        test_deltanet_step(rng);
        test_qk_norm_rope(rng);
        test_split_and_store(rng);
        test_gqa(rng);
        printf("tests: %d checks, %d failures\n", checks, failures);
    }
    if (mode == "all" || mode == "bench")
        run_bench(rng);
    return failures ? 1 : 0;
}
