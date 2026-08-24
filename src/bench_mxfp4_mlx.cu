#include "insignia_layout.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>
#define CUDA_OK(call) do{cudaError_t e_=(call);if(e_!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e_));return 2;}}while(0)
struct Shape { int rows, cols; };
int main() {
    const Shape shapes[] = {{8192,4096},{4096,4096},{12288,4096},{4096,12288},{1024,4096},{248320,4096}};
    int rc = 0;
    for (const Shape &sh : shapes) {
        const int rows = sh.rows, cols = sh.cols, groups = cols / 32;
        std::vector<uint32_t> w(size_t(rows) * groups * 4);
        std::vector<uint8_t> s(size_t(rows) * groups);
        std::vector<float> x(cols), ref(rows), y(rows);
        for (int i = 0; i < cols; i++) x[i] = float((i * 17) % 31 - 15) / 16.f;
        for (int r = 0; r < rows; r++) {
            double z = 0;
            for (int g = 0; g < groups; g++) {
                s[size_t(r) * groups + g] = uint8_t(124 + (r + g) % 7);
                float scale = insignia::e8m0(s[size_t(r) * groups + g]);
                for (int lane = 0; lane < 32; lane++) {
                    uint8_t q = uint8_t((r * 5 + g + lane * 7) & 15);
                    w[(size_t(r) * groups + g) * 4 + (lane >> 3)] |= uint32_t(q) << ((lane & 7) * 4);
                    z += double(insignia::fp4_e2m1(q) * scale) * x[g * 32 + lane];
                }
            }
            ref[r] = float(z);
        }
        uint32_t *dw; uint8_t *ds; float *dx, *dy;
        CUDA_OK(cudaMalloc(&dw, w.size() * 4)); CUDA_OK(cudaMalloc(&ds, s.size()));
        CUDA_OK(cudaMalloc(&dx, x.size() * 4)); CUDA_OK(cudaMalloc(&dy, y.size() * 4));
        CUDA_OK(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(ds, s.data(), s.size(), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        double bytes = double(w.size() * 4 + s.size() + x.size() * 4);
        auto bench_one = [&](const char *name, int warps, bool v2) {
            auto run = [&] { v2 ? insignia::mxfp4_gemv_v2(dw, ds, dx, dy, rows, cols)
                                 : insignia::mxfp4_gemv_mlx(dw, ds, dx, dy, rows, cols, warps); };
            run(); CUDA_OK(cudaDeviceSynchronize());
            CUDA_OK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
            float err = 0;
            for (int r = 0; r < rows; r++) err = fmaxf(err, fabsf(y[r] - ref[r]) / (fabsf(ref[r]) + 1e-5f));
            constexpr int iters = 100;
            cudaEventRecord(a);
            for (int i = iters; i; i--) run();
            cudaEventRecord(b); cudaEventSynchronize(b);
            float ms; cudaEventElapsedTime(&ms, a, b); ms /= iters;
            std::printf("%dx%d %-10s w=%d max_rel=%g %.3f ms %.1f GiB/s\n", rows, cols, name, warps, err, ms, (bytes / 1073741824.) / (ms / 1000.));
            if (err > 3e-4) rc = 1;
        };
        bench_one("mlx", 1, false);
        bench_one("mlx", 2, false);
        bench_one("v2", 0, true);
        cudaFree(dw); cudaFree(ds); cudaFree(dx); cudaFree(dy);
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    return rc;
}
