#include "insignia_layout.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#define CUDA_OK(call) do{cudaError_t e_=(call);if(e_!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e_));return 2;}}while(0)
int main() {
    const int shapes[][2] = {{8192,4096},{12288,4096},{4096,12288},{248320,4096}};
    int rc = 0;
    for (auto &sh : shapes) {
        const int rows = sh[0], cols = sh[1], groups = cols / 32, T = 64;
        std::vector<uint32_t> w(size_t(rows) * groups * 4);
        std::vector<uint8_t> s(size_t(rows) * groups);
        std::vector<float> x(size_t(T) * cols), y(size_t(T) * rows);
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> fx(-1.5f, 1.5f);
        for (auto &v : x) v = fx(rng);
        for (int r = 0; r < rows; r++)
            for (int g = 0; g < groups; g++) {
                s[size_t(r) * groups + g] = uint8_t(120 + (r + g) % 9);
                for (int lane = 0; lane < 32; lane++) {
                    uint8_t q = uint8_t((r * 5 + g * 3 + lane * 7) & 15);
                    w[(size_t(r) * groups + g) * 4 + (lane >> 3)] |= uint32_t(q) << ((lane & 7) * 4);
                }
            }
        uint32_t *dw; uint8_t *ds; float *dx, *dy;
        CUDA_OK(cudaMalloc(&dw, w.size() * 4)); CUDA_OK(cudaMalloc(&ds, s.size()));
        CUDA_OK(cudaMalloc(&dx, x.size() * 4)); CUDA_OK(cudaMalloc(&dy, y.size() * 4));
        CUDA_OK(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(ds, s.data(), s.size(), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        insignia::mxfp4_gemm_mlx(dw, ds, dx, dy, rows, cols, T);
        CUDA_OK(cudaDeviceSynchronize());
        CUDA_OK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
        // Host check on 8 sample rows across several t, normalized by the
        // cancellation-free magnitude (near-zero dots make plain relative error meaningless).
        float err = 0;
        for (int si = 0; si < 8; si++) {
            const int r = (si * 2654435761u) % rows;
            for (int t = 0; t < T; t += 7) {
                double z = 0, mag = 0;
                for (int g = 0; g < groups; g++) {
                    const float scale = insignia::e8m0(s[size_t(r) * groups + g]);
                    for (int lane = 0; lane < 32; lane++) {
                        const uint8_t q = uint8_t(w[(size_t(r) * groups + g) * 4 + (lane >> 3)] >> ((lane & 7) * 4));
                        const double term = double(insignia::fp4_e2m1(q) * scale) * x[size_t(t) * cols + g * 32 + lane];
                        z += term;
                        mag += fabs(term);
                    }
                }
                err = fmaxf(err, float(fabs(z - y[size_t(t) * rows + r]) / (mag + 1e-6)));
            }
        }
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        const int iters = 30;
        cudaEventRecord(a);
        for (int i = iters; i; i--) insignia::mxfp4_gemm_mlx(dw, ds, dx, dy, rows, cols, T);
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b); ms /= iters;
        const double wbytes = double(w.size() * 4 + s.size());
        const double flops = double(T) * rows * cols * 2;
        std::printf("%dx%d T=%d max_rel=%g %.3f ms | %.0f GiB/s weights | %.1f TFLOPS\n",
                    rows, cols, T, err, ms, wbytes / 1073741824.0 / (ms / 1000.0), flops / 1e12 / (ms / 1000.0));
        if (err > 2e-2) { std::printf("FAIL\n"); rc = 1; }
        cudaFree(dw); cudaFree(ds); cudaFree(dx); cudaFree(dy);
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    return rc;
}
