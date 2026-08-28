// fp8_gemm vs host reference, minimal: T=1, rows=10240/17408, signed random weights.
// Prints engine-vs-ref for sampled rows to expose the structure of any disagreement.
#include "insignia_fp8.cuh"
#include <cuda_bf16.h>
#include <cstdio>
#include <vector>

static float e4m3_host(uint8_t b) {
    const int sign = b & 0x80 ? -1 : 1;
    const int e = (b >> 3) & 0xf, m = b & 7;
    return float(sign) * (e == 0 ? float(m) / 8.f * (1.f / 64.f) : ldexpf(1.f + float(m) / 8.f, e - 7));
}

int main() {
    cudaStream_t s; cudaStreamCreate(&s);
    for (int rows : {10240, 17408}) {
        const int cols = 5120;
        std::vector<uint8_t> hw(size_t(rows) * cols);
        unsigned st = 12345;
        for (size_t i = 0; i < hw.size(); i++) { st = st * 1664525u + 1013904223u; hw[i] = uint8_t(st >> 23); }
        const int sr = (rows + 127) / 128, sc = cols / 128;
        std::vector<uint16_t> hs(size_t(sr) * sc);
        for (size_t i = 0; i < hs.size(); i++) { st = st * 1664525u + 1013904223u; hs[i] = uint16_t(0x3d00 + (st >> 24) % 256); }  // ~0.03..0.06
        uint8_t *dw; cudaMalloc(&dw, hw.size()); cudaMemcpy(dw, hw.data(), hw.size(), cudaMemcpyHostToDevice);
        uint16_t *ds; cudaMalloc(&ds, hs.size() * 2); cudaMemcpy(ds, hs.data(), hs.size() * 2, cudaMemcpyHostToDevice);
        const int T = 1;
        std::vector<float> hx(size_t(cols));
        for (int i = 0; i < cols; i++) { st = st * 1664525u + 1013904223u; hx[i] = float(int(st >> 24) - 128) / 64.f; }
        __nv_bfloat16 *dx; cudaMalloc(&dx, size_t(64) * cols * 2); cudaMemset(dx, 0, size_t(64) * cols * 2);
        static __nv_bfloat16 h16_arr[5120];
        __nv_bfloat16 *h16 = h16_arr;
        for (int i = 0; i < cols; i++) h16[i] = __float2bfloat16(hx[i]);
        cudaMemcpy(dx, h16, cols * 2, cudaMemcpyHostToDevice);
        float *dy; cudaMalloc(&dy, size_t(64) * rows * 4); cudaMemset(dy, 0, size_t(64) * rows * 4);
        insignia::fp8_gemm(dw, ds, dx, dy, rows, cols, T, s);
        cudaStreamSynchronize(s);
        std::vector<float> hy(rows);
        cudaMemcpy(hy.data(), dy, rows * 4, cudaMemcpyDeviceToHost);
        auto ref = [&](int n) {
            double acc = 0;
            for (int k = 0; k < cols; k++) {
                const float scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(&hs[(n >> 7) * sc + (k >> 7)]));
                acc += double(e4m3_host(hw[size_t(n) * cols + k])) * double(scale) * double(__bfloat162float(h16[k]));
            }
            return acc;
        };
        int bad = 0; double worst = 0;
        for (int n = 0; n < rows; n++) {
            const double r = ref(n);
            const double rel = std::abs(r - hy[n]) / std::max(1e-9, std::abs(r));
            worst = std::max(worst, rel);
            if (rel > 1e-2) bad++;
        }
        printf("rows=%d T=1: bad=%d/%d worst_rel=%.3e\n", rows, bad, rows, worst);
        for (int n : {0, 1, 100, 4095, 4096, 8192, 12287, 13312, 17407}) {
            if (n < rows) printf("  n=%5d eng=%12.5f ref=%12.5f\n", n, hy[n], ref(n));
        }
        cudaFree(dw); cudaFree(ds); cudaFree(dx); cudaFree(dy);
    }
    return 0;
}
