// FP8 e4m3 + bf16 block-scale kernel correctness vs host double reference.
#include "insignia_fp8.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
using namespace insignia;

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
static uint16_t f32_to_bf16_bits(float v) {
    uint32_t bits;
    memcpy(&bits, &v, 4);
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return uint16_t(bits >> 16);
}
static float bf16_host(uint16_t u) {
    uint32_t b = uint32_t(u) << 16;
    float v;
    memcpy(&v, &b, 4);
    return v;
}

int wmain(int argc, wchar_t **argv) {
    if (argc < 0) return 2;
    std::mt19937 rng(777);
    std::normal_distribution<float> nd(0.f, 0.05f);
    const int rows = 10240, cols = 5120;
    const int kb_r = rows >> 7, kb_c = cols >> 7;
    std::vector<uint8_t> w(size_t(rows) * cols);
    std::vector<uint16_t> s(size_t(kb_r) * kb_c);
    std::vector<double> wref(size_t(rows) * cols);
    for (int br = 0; br < kb_r; br++)
        for (int bc = 0; bc < kb_c; bc++) {
            float amax = 0;
            for (int r = 0; r < 128; r++)
                for (int c = 0; c < 128; c++) amax = fmaxf(amax, fabsf(nd(rng)));
            const float sc = amax / 448.f;
            s[br * kb_c + bc] = f32_to_bf16_bits(sc);
            const float scf = bf16_host(s[br * kb_c + bc]);
            for (int r = 0; r < 128; r++)
                for (int c = 0; c < 128; c++) {
                    const size_t i = size_t(br * 128 + r) * cols + bc * 128 + c;
                    const uint8_t code = f32_to_e4m3(nd(rng) / scf);
                    w[i] = code;
                    wref[i] = double(e4m3_host(code)) * scf;
                }
        }
    uint8_t *dw; uint16_t *ds; float *dx, *dy;
    cudaMalloc(&dw, w.size()); cudaMalloc(&ds, s.size() * 2);
    cudaMalloc(&dx, 64 * cols * 4); cudaMalloc(&dy, 2 * rows * 4);
    cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, s.data(), s.size() * 2, cudaMemcpyHostToDevice);
    std::vector<float> x(64 * cols);   // 64 rows: GEMV uses rows 0..1, GEMM zero-pads the rest
    for (auto &v : x) v = nd(rng);
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    std::vector<float> y(2 * rows);
    double xy = 0, xx = 0, yy = 0;
    auto ref_of = [&](int k, int r) {
        double acc = 0;
        for (int c = 0; c < cols; c++) acc += double(x[k * cols + c]) * wref[size_t(r) * cols + c];
        return acc;
    };
    // GEMV row 0
    fp8_gemv(dw, ds, dx, dy, rows, cols, nullptr);
    { const cudaError_t e = cudaDeviceSynchronize(); if (e != cudaSuccess) printf("gemv err: %s\n", cudaGetErrorString(e)); }
    cudaMemcpy(y.data(), dy, rows * 4, cudaMemcpyDeviceToHost);
    for (int r = 0; r < rows; r++) { const double ref = ref_of(0, r); xy += ref * y[r]; xx += ref * ref; yy += double(y[r]) * y[r]; }
    printf("fp8_gemv cos=%.8f\n", xy / (sqrt(xx) * sqrt(yy) + 1e-30));
    // pair GEMV
    fp8_gemv2(dw, ds, dx, dy, rows, cols, nullptr);
    { const cudaError_t e = cudaDeviceSynchronize(); if (e != cudaSuccess) printf("gemv2 err: %s\n", cudaGetErrorString(e)); }
    cudaMemcpy(y.data(), dy, 2 * rows * 4, cudaMemcpyDeviceToHost);
    xy = xx = yy = 0;
    for (int k = 0; k < 2; k++)
        for (int r = 0; r < rows; r++) { const double ref = ref_of(k, r); xy += ref * y[k * rows + r]; xx += ref * ref; yy += double(y[k * rows + r]) * y[k * rows + r]; }
    printf("fp8_gemv2 cos=%.8f\n", xy / (sqrt(xx) * sqrt(yy) + 1e-30));
    // GEMM T=3, x bf16
    std::vector<__nv_bfloat16> x16(64 * cols);
    for (int i = 0; i < 64 * cols; i++) x16[i] = __float2bfloat16(i < 3 * cols ? x[i] : 0.f);
    __nv_bfloat16 *dx16;
    cudaMalloc(&dx16, 64 * cols * 2);
    cudaMemcpy(dx16, x16.data(), 64 * cols * 2, cudaMemcpyHostToDevice);
    float *dyT;
    cudaMalloc(&dyT, size_t(64) * rows * 4);
    fp8_gemm(dw, ds, dx16, dyT, rows, cols, 3, nullptr);
    { const cudaError_t e = cudaDeviceSynchronize(); if (e != cudaSuccess) printf("gemm err: %s\n", cudaGetErrorString(e)); }
    std::vector<float> yT(64 * rows);
    cudaMemcpy(yT.data(), dyT, yT.size() * 4, cudaMemcpyDeviceToHost);
    {
        int nan_cnt = 0, nonzero = 0;
        for (int t = 0; t < 3; t++)
            for (int r = 0; r < rows; r++) {
                const float v = yT[t * rows + r];
                if (v != v) nan_cnt++;
                else if (v != 0.f) nonzero++;
            }
        printf("gemm diag: nan=%d nonzero=%d yT[0]=%f yT[1]=%f yT[rows]=%f\n", nan_cnt, nonzero, yT[0], yT[1], yT[rows]);
    }
    xy = xx = yy = 0;
    for (int t = 0; t < 3; t++)
        for (int r = 0; r < rows; r++) {
            double acc = 0;
            for (int c = 0; c < cols; c++) acc += double(__bfloat162float(x16[t * cols + c])) * wref[size_t(r) * cols + c];
            xy += acc * yT[t * rows + r]; xx += acc * acc; yy += double(yT[t * rows + r]) * yT[t * rows + r];
        }
    printf("fp8_gemm cos=%.8f\n", xy / (sqrt(xx) * sqrt(yy) + 1e-30));
    // T=33 crosses the wm tile boundary (rows 32..63 store-guarded); row-sliced reference
    for (int i = 0; i < 64 * cols; i++) x16[i] = __float2bfloat16(i < 33 * cols ? x[i] : 0.f);
    cudaMemcpy(dx16, x16.data(), 64 * cols * 2, cudaMemcpyHostToDevice);
    fp8_gemm(dw, ds, dx16, dyT, rows, cols, 33, nullptr);
    { const cudaError_t e = cudaDeviceSynchronize(); if (e != cudaSuccess) printf("gemm33 err: %s\n", cudaGetErrorString(e)); }
    cudaMemcpy(yT.data(), dyT, yT.size() * 4, cudaMemcpyDeviceToHost);
    xy = xx = yy = 0;
    for (int t = 32; t < 33; t++)
        for (int r = 0; r < 256; r++) {
            double acc = 0;
            for (int c = 0; c < cols; c++) acc += double(__bfloat162float(x16[t * cols + c])) * wref[size_t(r) * cols + c];
            xy += acc * yT[t * rows + r]; xx += acc * acc; yy += double(yT[t * rows + r]) * yT[t * rows + r];
        }
    printf("fp8_gemm T=33 (tile-boundary rows) cos=%.8f\n", xy / (sqrt(xx) * sqrt(yy) + 1e-30));
    bool threw = false;
    try { fp8_gemm(dw, ds, dx16, dyT, rows, cols, 65, nullptr); } catch (const std::exception &) { threw = true; }
    printf("fp8_gemm T=65 throws: %s\n", threw ? "yes" : "NO (BUG)");
    return 0;
}
