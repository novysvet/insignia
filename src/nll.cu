#include "insignia_decode.hpp"
#include "insignia_layout.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
// argv: index tokens(comma-separated) [chunk=64]
// Teacher-forced NLL / perplexity over the token stream: prefill chunks of T through
// the batched path, one all-positions lm_head GEMM per chunk, fused logsumexp reduce.
// Prints total NLL, token count, and mean perplexity per token.
__global__ void row_logp_kernel(const float *__restrict__ logits, const int *__restrict__ targets, float *__restrict__ logp, int vocab) {
    const int row = blockIdx.x;               // one block per position
    const float *l = logits + size_t(row) * vocab;
    __shared__ float red[32];
    float best = -3.402823466e+38F;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) best = fmaxf(best, __ldg(l + i));
    for (int m = 16; m; m >>= 1) best = fmaxf(best, __shfl_xor_sync(0xffffffff, best, m));
    if (!(threadIdx.x & 31)) red[threadIdx.x >> 5] = best;
    __syncthreads();
    if (threadIdx.x < 32) {
        best = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : -3.402823466e+38F;
        for (int m = 16; m; m >>= 1) best = fmaxf(best, __shfl_xor_sync(0xffffffff, best, m));
        if (!threadIdx.x) red[8] = best;
    }
    __syncthreads();
    const float mx = red[8];
    float s = 0;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) s += __expf(__ldg(l + i) - mx);
    for (int m = 16; m; m >>= 1) s += __shfl_xor_sync(0xffffffff, s, m);
    if (!(threadIdx.x & 31)) red[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        s = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
        for (int m = 16; m; m >>= 1) s += __shfl_xor_sync(0xffffffff, s, m);
        if (!threadIdx.x) red[9] = s;
    }
    __syncthreads();
    if (!threadIdx.x) {
        const float tgt = __ldg(l + __ldg(targets + row));
        logp[row] = tgt - mx - __logf(red[9]);
    }
}
int wmain(int argc, wchar_t **argv) {
    if (argc < 3) return 2;
    try {
        insignia::ModelFile m(argv[1]);
        std::vector<int> tokens;
        {
            char buf[1 << 20];
            size_t n = wcstombs(buf, argv[2], sizeof(buf) - 1);
            buf[n] = 0;
            for (char *s = strtok(buf, ","); s; s = strtok(nullptr, ",")) tokens.push_back(atoi(s));
        }
        if (tokens.size() < 2) return 2;
        const int chunk = argc > 3 ? _wtoi(argv[3]) : 64;
        insignia::DecodeWorkspace x(4096);
        insignia::Qwen35Weights w(m, 6ull << 30, x.stream);
        insignia::Qwen35Decode d(w, x);
        const int vocab = insignia::Qwen35Shape::vocab;
        float *logitsT;
        if (cudaMalloc(&logitsT, size_t(chunk) * vocab * 4)) throw std::runtime_error("logits alloc");
        int *targets;
        cudaMalloc(&targets, chunk * 4);
        float *logp;
        cudaMalloc(&logp, chunk * 4);
        double total = 0;
        size_t count = 0;
        // targets: for chunk rows [t0, t0+T), the predicted-next token of row i is tokens[t0+i+1]
        size_t done = 0;
        while (done + 1 < tokens.size()) {
            const size_t remain = tokens.size() - done - 1;   // positions that have a target
            const int T = int(remain) >= chunk ? chunk : int(remain);
            d.prefill_chunk(tokens.data() + done, T);         // advances state by T; pf_n holds the chunk's final norm
            std::vector<int> tgt(tokens.begin() + done + 1, tokens.begin() + done + 1 + T);
            cudaMemcpyAsync(targets, tgt.data(), T * 4, cudaMemcpyHostToDevice, x.stream);
            {   // all-positions lm_head GEMM straight from the chunk's normalized activations
            auto lh = w.matrix("language_model.lm_head");
            if (lh.insig4) insignia::mxfp4_gemm_mlx_i4((const uint32_t *)lh.weight.data, (const uint16_t *)lh.scales.data, x.pf_n, logitsT, lh.rows, lh.cols, T, x.stream);
            else insignia::mxfp4_gemm_mlx((const uint32_t *)lh.weight.data, (const uint8_t *)lh.scales.data, x.pf_n, logitsT, lh.rows, lh.cols, T, x.stream);
            w.release("language_model.lm_head");
            }
            row_logp_kernel<<<T, 256, 0, x.stream>>>(logitsT, targets, logp, vocab);
            std::vector<float> lp(T);
            cudaMemcpyAsync(lp.data(), logp, T * 4, cudaMemcpyDeviceToHost, x.stream);
            cudaStreamSynchronize(x.stream);
            for (float v : lp) total += -double(v);
            count += T;
            done += T;
        }
        printf("tokens=%zu nll=%.4f ppl=%.5f\n", count, total, exp(total / count));
        return 0;
    } catch (const std::exception &e) { fprintf(stderr, "%s\n", e.what()); return 3; }
}
