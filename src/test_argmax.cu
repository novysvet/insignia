#include "insignia_qwen_kernels.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
using namespace insignia;
// Compare argmax_fast (two-stage atomic) against argmax_logits (single block) on the
// real lm_head logits distribution shape: a few big values, many near-ties.
int wmain(int argc, wchar_t **argv) {
    if (argc != 1) return 2;
    const int n = 248320;
    float *h = (float *) malloc(n * 4);
    srand(1234);
    for (int i = 0; i < n; i++) h[i] = -8.f + 6.f * (rand() / 32768.f);
    for (int k = 0; k < 5; k++) h[rand() % n] = 12.f + k * 0.001f;      // top candidates
    h[100000] = h[123456] = 3.25f;                                       // exact tie
    float *d; cudaMalloc(&d, n * 4);
    cudaMemcpy(d, h, n * 4, cudaMemcpyHostToDevice);
    int *o1, *o2; unsigned long long *scr;
    cudaMalloc(&o1, 4); cudaMalloc(&o2, 4); cudaMalloc(&scr, 8);
    int bad = 0;
    for (int it = 0; it < 200; it++) {
        for (int k = 0; k < 5; k++) h[rand() % n] = 12.f + (rand() % 1000) * 0.001f;
        h[rand() % n] = -0.f; // negative zeros and negatives everywhere else
        cudaMemcpy(d, h, n * 4, cudaMemcpyHostToDevice);
        argmax_logits(d, n, o1, nullptr);
        argmax_fast(d, n, o2, scr, nullptr);
        cudaDeviceSynchronize();
        int a, b; cudaMemcpy(&a, o1, 4, cudaMemcpyDeviceToHost); cudaMemcpy(&b, o2, 4, cudaMemcpyDeviceToHost);
        float va = h[a], vb = h[b];
        if (va != vb) { printf("iter %d MISMATCH old idx %d val %.6f | fast idx %d val %.6f\n", it, a, va, b, vb); if (++bad > 5) break; }
    }
    printf(bad ? "FAIL\n" : "OK: 200 random argmax rounds agree\n");
    return bad ? 1 : 0;
}
