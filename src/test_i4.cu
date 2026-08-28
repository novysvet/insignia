// Standalone correctness check for the INSIG4 kernels against host double reference.
// dllshim entry: test_i4 <seed>
#include "insignia_layout.cuh"
#include "insignia_prefill.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
using namespace insignia;

static float e2m1_host(unsigned c) {
    static const float t[16] = {0,.5f,1,1.5f,2,3,4,6,-0.f,-.5f,-1,-1.5f,-2,-3,-4,-6};
    return t[c & 15];
}

int wmain(int argc, wchar_t **argv) {
    if (argc < 1) return 2;
    std::mt19937 rng(12345);
    std::normal_distribution<float> nd(0.f, 0.03f);
    const int rows = 8192, cols = 4096;
    const int groups = cols >> 5;
    // host INSIG4 quantization (amax/6 per 64-group; codes snapped like the python tool)
    std::vector<uint32_t> w(rows * groups * 4);
    std::vector<uint16_t> s16(rows * (cols >> 6));
    std::vector<double> wref(rows * cols);
    for (int r = 0; r < rows; r++) {
        for (int sg = 0; sg < cols >> 6; sg++) {
            float blk[64];
            float amax = 0;
            for (int i = 0; i < 64; i++) { blk[i] = nd(rng); amax = fmaxf(amax, fabsf(blk[i])); }
            float sc = amax / 6.f;
            __half sh = __float2half(sc);
            s16[r * (cols >> 6) + sg] = *reinterpret_cast<uint16_t *>(&sh);
            float scf = __half2float(sh);
            for (int i = 0; i < 64; i++) {
                float q = blk[i] / scf;
                float m = fabsf(q), best = 0, bd = 1e30f;
                for (int c = 0; c < 8; c++) { float d = fabsf(m - e2m1_host(c)); if (d < bd) { bd = d; best = e2m1_host(c); } }
                float dv = (q < 0 ? -best : best) * scf;
                wref[size_t(r) * cols + sg * 64 + i] = dv;
                unsigned code = 0;
                float mm = 1e30f; for (int c = 0; c < 8; c++) { float d = fabsf(m - e2m1_host(c)); if (d < mm) { mm = d; code = unsigned(c); } }
                code |= unsigned(q < 0 ? 8 : 0);
                const int g = sg * 2 + (i >> 5), j = i & 31;
                w[size_t(r) * groups * 4 + size_t(g) * 4 + (j >> 3)] |= code << (4 * (j & 7));
            }
        }
    }
    std::vector<float> x(cols), y(rows, 0.f);
    for (auto &v : x) v = nd(rng);
    uint32_t *dw; uint16_t *ds; float *dx, *dy, *dout;
    cudaMalloc(&dw, w.size() * 4); cudaMalloc(&ds, s16.size() * 2);
    cudaMalloc(&dx, cols * 4); cudaMalloc(&dy, rows * 4); cudaMalloc(&dout, cols * 4);
    cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, s16.data(), s16.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), cols * 4, cudaMemcpyHostToDevice);
    // 1) GEMV
    mxfp4_gemv_v2_i4(dw, ds, dx, dy, rows, cols, nullptr);
    cudaDeviceSynchronize();
    cudaMemcpy(y.data(), dy, rows * 4, cudaMemcpyDeviceToHost);
    double mx = 0, dot_abs = 0, dot_xy = 0, dot_xx = 0, dot_yy = 0;
    int bad = 0, badr = -1;
    for (int r = 0; r < rows; r++) {
        double ref = 0;
        for (int c = 0; c < cols; c++) ref += double(x[c]) * wref[size_t(r) * cols + c];
        const double rel = fabs(ref - double(y[r])) / (fabs(ref) + 1e-3);
        if (rel > 2e-2) { bad++; if (badr < 0) badr = r; }
        mx = fmax(mx, rel);
        dot_abs += fabs(ref);
        dot_xy += ref * y[r]; dot_xx += ref * ref; dot_yy += double(y[r]) * y[r];
    }
    printf("gemv_v2_i4 max_rel=%.3e cos=%.8f (mean|dot|=%.3f) bad=%d first_bad_row=%d\n", mx, dot_xy / (sqrt(dot_xx) * sqrt(dot_yy) + 1e-30), dot_abs / rows, bad, badr);
    if (badr >= 0) {
        double ref = 0;
        for (int c = 0; c < cols; c++) ref += double(x[c]) * wref[size_t(badr) * cols + c];
        printf("  row %d: ref=%f got=%f ratio=%f\n", badr, ref, y[badr], ref / (y[badr] + 1e-30));
    }
    // 2) GEMM (T=3)
    const int T = 3;
    std::vector<float> xT(T * cols);
    for (auto &v : xT) v = nd(rng);
    std::vector<float> yT(64 * rows, 0.f);
    float *dxT, *dyT;
    cudaMalloc(&dxT, xT.size() * 4); cudaMalloc(&dyT, yT.size() * 4);
    cudaMemcpy(dxT, xT.data(), xT.size() * 4, cudaMemcpyHostToDevice);
    mxfp4_gemm_mlx_i4(dw, ds, dxT, dyT, rows, cols, T, nullptr);
    cudaDeviceSynchronize();
    cudaMemcpy(yT.data(), dyT, yT.size() * 4, cudaMemcpyDeviceToHost);
    mx = 0; dot_xy = dot_xx = dot_yy = 0;
    for (int t = 0; t < T; t++)
        for (int r = 0; r < rows; r++) {
            double ref = 0;
            for (int c = 0; c < cols; c++) ref += double(xT[t * cols + c]) * wref[size_t(r) * cols + c];
            mx = fmax(mx, fabs(ref - double(yT[t * rows + r])) / (fabs(ref) + 1e-3));
            dot_xy += ref * yT[t * rows + r]; dot_xx += ref * ref; dot_yy += double(yT[t * rows + r]) * yT[t * rows + r];
        }
    printf("gemm_mlx_i4 max_rel=%.3e cos=%.8f\n", mx, dot_xy / (sqrt(dot_xx) * sqrt(dot_yy) + 1e-30));
    // 3) embed gather
    int row_dev_h = 17, *row_dev;
    cudaMalloc(&row_dev, 4); cudaMemcpy(row_dev, &row_dev_h, 4, cudaMemcpyHostToDevice);
    mxfp4_get_row_i4(dw, ds, dout, row_dev, cols, nullptr);
    cudaDeviceSynchronize();
    std::vector<float> o(cols);
    cudaMemcpy(o.data(), dout, cols * 4, cudaMemcpyDeviceToHost);
    mx = 0;
    for (int c = 0; c < cols; c++) mx = fmax(mx, fabs(wref[17ull * cols + c] - double(o[c])));
    {
        int firstbad = -1;
        for (int c = 0; c < cols; c++) if (fabs(wref[17ull * cols + c] - double(o[c])) > 1e-4) { firstbad = c; break; }
        printf("get_row_i4 max_abs=%.3e first_bad=%d\n", mx, firstbad);
        if (firstbad >= 0)
            for (int c = firstbad; c < firstbad + 8; c++)
                printf("  c=%d ref=%.6f got=%.6f\n", c, wref[17ull * cols + c], o[c]);
    }
    // 4) pair dp4a
    std::vector<float> x2(2 * cols);
    for (auto &v : x2) v = nd(rng);
    std::vector<float> y2(2 * rows, 0.f);
    cudaMemcpy(dxT, x2.data(), x2.size() * 4, cudaMemcpyHostToDevice);
    mxfp4_gemv2_q8_i4(dw, ds, dxT, dyT, rows, cols, nullptr);
    cudaDeviceSynchronize();
    cudaMemcpy(y2.data(), dyT, y2.size() * 4, cudaMemcpyDeviceToHost);
    mx = 0; dot_xy = dot_xx = dot_yy = 0;
    for (int r = 0; r < rows; r++)
        for (int k = 0; k < 2; k++) {
            double ref = 0;
            for (int c = 0; c < cols; c++) ref += double(x2[k * cols + c]) * wref[size_t(r) * cols + c];
            mx = fmax(mx, fabs(ref - double(y2[k * rows + r])) / (fabs(ref) + 1e-3));
            dot_xy += ref * y2[k * rows + r]; dot_xx += ref * ref; dot_yy += double(y2[k * rows + r]) * y2[k * rows + r];
        }
    printf("gemv2_q8_i4 max_rel=%.3e cos=%.8f\n", mx, dot_xy / (sqrt(dot_xx) * sqrt(dot_yy) + 1e-30));
    // 5) v21_i4 pipelined GEMM vs f64 ref AND vs mlx_i4 (cross-kernel agreement), T=33
    {
        const int T5 = 33;
        std::vector<float> x5(64 * cols, 0.f);
        for (int i = 0; i < T5 * cols; i++) x5[i] = nd(rng);
        std::vector<__nv_bfloat16> x16(64 * cols);
        for (int i = 0; i < 64 * cols; i++) x16[i] = __float2bfloat16(x5[i]);
        float *dx16, *dy5a, *dy5b;
        cudaMalloc(&dx16, 64 * cols * 2); cudaMalloc(&dy5a, 64 * rows * 4); cudaMalloc(&dy5b, 64 * rows * 4);
        cudaMemcpy(dx16, x16.data(), 64 * cols * 2, cudaMemcpyHostToDevice);
        mxfp4_gemm_v21_i4(dw, ds, dx16, dy5a, rows, cols, T5, nullptr);
        cudaDeviceSynchronize();
        std::vector<float> y5a(64 * rows), y5b(64 * rows);
        cudaMemcpy(y5a.data(), dy5a, 64 * rows * 4, cudaMemcpyDeviceToHost);
        // f64 ref (bf16-rounded A) + mlx_i4 (f32 A) cross-check
        mx = 0; dot_xy = dot_xx = dot_yy = 0;
        for (int t = 0; t < T5; t += 8)
            for (int r = 0; r < rows; r += 257) {
                double ref = 0;
                for (int c = 0; c < cols; c++) ref += double(__bfloat162float(x16[t * cols + c])) * wref[size_t(r) * cols + c];
                mx = fmax(mx, fabs(ref - double(y5a[t * rows + r])) / (fabs(ref) + 1e-3));
                dot_xy += ref * y5a[t * rows + r]; dot_xx += ref * ref; dot_yy += double(y5a[t * rows + r]) * y5a[t * rows + r];
            }
        printf("gemm_v21_i4 T=33 sampled max_rel=%.3e cos=%.8f\n", mx, dot_xy / (sqrt(dot_xx) * sqrt(dot_yy) + 1e-30));
        std::vector<float> x5f(T5 * cols);
        for (int i = 0; i < T5 * cols; i++) x5f[i] = x5[i];
        cudaMemcpy(dxT, x5f.data(), x5f.size() * 4, cudaMemcpyHostToDevice);
        mxfp4_gemm_mlx_i4(dw, ds, dxT, dy5b, rows, cols, T5, nullptr);
        cudaDeviceSynchronize();
        cudaMemcpy(y5b.data(), dy5b, 64 * rows * 4, cudaMemcpyDeviceToHost);
        double worst = 1;
        for (int t = 0; t < T5; t++)
            for (int r = 0; r < rows; r += 61) worst = fmin(worst, fabs(double(y5a[t * rows + r]) - double(y5b[t * rows + r])) / (fabs(double(y5b[t * rows + r])) + 1e-2));
        printf("gemm_v21_i4 vs mlx_i4 worst_rel_agreement=%.3e (1.0=perfect)\n", worst);
    }
    // 6) ab_i4 fused a+b GEMM vs two v21_i4 GEMMs on rows 0..31 / 32..63 of the same weights
    {
        const int T6 = 5;
        std::vector<float> x6(64 * cols, 0.f);
        for (int i = 0; i < T6 * cols; i++) x6[i] = nd(rng);
        std::vector<__nv_bfloat16> x16(64 * cols);
        for (int i = 0; i < 64 * cols; i++) x16[i] = __float2bfloat16(x6[i]);
        float *dx16, *dya, *dyb, *dref;
        cudaMalloc(&dx16, 64 * cols * 2); cudaMalloc(&dya, 64 * 32 * 4); cudaMalloc(&dyb, 64 * 32 * 4); cudaMalloc(&dref, 64 * 64 * 4);
        cudaMemcpy(dx16, x16.data(), 64 * cols * 2, cudaMemcpyHostToDevice);
        mxfp4_gemm_ab_i4(dw, ds, dw, ds, dx16, dya, dyb, T6, cols, nullptr);  // a=b=first 32 rows
        mxfp4_gemm_v21_i4(dw, ds, dx16, dref, 32, cols, T6, nullptr);         // reference over the same 32 rows
        cudaDeviceSynchronize();
        std::vector<float> ya(64 * 32), yb(64 * 32), ref(64 * 64);
        cudaMemcpy(ya.data(), dya, 64 * 32 * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(yb.data(), dyb, 64 * 32 * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(ref.data(), dref, 64 * 64 * 4, cudaMemcpyDeviceToHost);
        double wa = 0, wb = 0;
        for (int t = 0; t < T6; t++)
            for (int r = 0; r < 32; r++) {
                wa = fmax(wa, fabs(double(ya[t * 32 + r]) - double(ref[t * 32 + r])) / (fabs(double(ref[t * 32 + r])) + 1e-2));
                wb = fmax(wb, fabs(double(yb[t * 32 + r]) - double(ref[t * 32 + r])) / (fabs(double(ref[t * 32 + r])) + 1e-2));
            }
        printf("gemm_ab_i4 vs v21_i4(32 rows) max_rel: a=%.3e b=%.3e\n", wa, wb);
    }
    return 0;
}
