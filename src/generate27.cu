// generate27 — Qwen3.8-27B-FP8 tiered driver.
//   generate27 <index> <manifest> <tokens-comma> <max_new>       greedy decode
//   generate27 dump <index> <manifest> <tokens> <out.f32>        per-layer seams of one prefill pass [64][T][5120]
//   generate27 nll  <index> <manifest> <tokens>                   teacher-forced NLL in 64-token chunks
#include "insignia_27b.hpp"
#include "insignia_bf16.cuh"
#include "insignia_layout.cuh"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

__global__ void row_logp27_kernel(const float *__restrict__ logits, const int *__restrict__ targets, float *__restrict__ logp, int vocab) {
    const int row = blockIdx.x;
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
    float sum = 0;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) sum += __expf(__ldg(l + i) - mx);
    for (int m = 16; m; m >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, m);
    if (!(threadIdx.x & 31)) red[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
        for (int m = 16; m; m >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, m);
        if (!threadIdx.x) red[9] = sum;
    }
    __syncthreads();
    if (!threadIdx.x) {
        const float tgt = __ldg(l + __ldg(targets + row));
        logp[row] = tgt - mx - __logf(red[9]);
    }
}

static std::vector<int> parse_tokens(const wchar_t *arg) {
    std::vector<int> tokens;
    char buf[1 << 20];
    size_t n = wcstombs(buf, arg, sizeof(buf) - 1);
    buf[n] = 0;
    for (char *s = strtok(buf, ","); s; s = strtok(nullptr, ",")) tokens.push_back(atoi(s));
    return tokens;
}

int wmain(int argc, wchar_t **argv) {
    if (argc < 5) { fprintf(stderr, "usage: generate27 [dump|nll] <index> <manifest> <tokens> [max_new|out.f32]\n"); return 2; }
    try {
        const bool dump_mode = !wcscmp(argv[1], L"dump");
        const bool nll_mode = !wcscmp(argv[1], L"nll");
        const int base = (dump_mode || nll_mode) ? 2 : 1;
        const int ctx = 2048;
        insignia::Workspace27 x(ctx);
        insignia::TieredStorage27 st(argv[base], argv[base + 1], x.stream);
        std::vector<int> tokens = parse_tokens(argv[base + 2]);
        if (tokens.empty()) return 2;
        insignia::Qwen38Decode d(st, x);
        d.set_position(0);

        if (dump_mode) {
            if (_wgetenv(L"INSIG_STAGE27")) insignia::g_dump_stage27 = true;
            FILE *f = _wfopen(argv[base + 3], L"wb");
            if (!f) return 2;
            int T = int(tokens.size()); if (T > 64) T = 64;
            struct Ctx { FILE *f; int T; std::vector<float> buf; } c{f, T, std::vector<float>(size_t(T) * 5120)};
            auto seam = [](int layer, const float *pf_x, int T, void *user) {
                Ctx *c = static_cast<Ctx *>(user);
                cudaMemcpy(c->buf.data(), pf_x, size_t(T) * 5120 * 4, cudaMemcpyDeviceToHost);
                fwrite(c->buf.data(), 4, size_t(T) * 5120, c->f);
                printf("seam layer %d written\n", layer);
            };
            d.prefill_chunk_seam(tokens.data(), T, seam, &c);
            fclose(f);
            printf("dump: %d layers x %d tokens x 5120 -> %ls\n", 64, T, argv[base + 3]);
            return 0;
        }

        if (nll_mode) {
            float *logitsT;
            if (cudaMalloc(&logitsT, size_t(64) * 248320 * 4)) throw std::runtime_error("logits alloc");
            int *targets; cudaMalloc(&targets, 64 * 4);
            float *logp; cudaMalloc(&logp, 64 * 4);
            void *n16; cudaMalloc(&n16, size_t(64) * 5120 * 2);
            double total = 0; size_t count = 0, done = 0;
            while (done + 1 < tokens.size()) {
                const size_t remain = tokens.size() - done - 1;
                const int T = int(remain) >= 64 ? 64 : int(remain);
                d.prefill_chunk(tokens.data() + done, T);
                std::vector<int> tgt(tokens.begin() + done + 1, tokens.begin() + done + 1 + T);
                cudaMemcpyAsync(targets, tgt.data(), T * 4, cudaMemcpyHostToDevice, x.stream);
                if (T < 64) cudaMemsetAsync((char *)n16 + size_t(T) * 5120 * 2, 0, size_t(64 - T) * 5120 * 2, x.stream);
                insignia::f32_to_bf16(x.pf_n, n16, size_t(T) * 5120, x.stream);
                insignia::bf16_gemm(st.lm_head(), n16, logitsT, 248320, 5120, T, x.stream);
                row_logp27_kernel<<<T, 256, 0, x.stream>>>(logitsT, targets, logp, 248320);
                std::vector<float> lp(T);
                cudaMemcpyAsync(lp.data(), logp, T * 4, cudaMemcpyDeviceToHost, x.stream);
                cudaStreamSynchronize(x.stream);
                for (float v : lp) total += -double(v);
                count += T; done += T;
            }
            const double nll = total / double(count ? count : 1);
            printf("tokens=%zu nll=%.4f ppl=%.5f\n", count, nll, exp(nll));
            return 0;
        }

        const int max_new = _wtoi(argv[base + 3]);
        if (max_new <= 0) return 2;
        int next = 0;
        size_t done = 0;
        while (done < tokens.size()) {
            const int T = int(tokens.size() - done) >= 64 ? 64 : int(tokens.size() - done);
            next = d.prefill_chunk(tokens.data() + done, T);
            done += T;
        }
        printf("[prefill %zu tokens done, first next=%d]\n", tokens.size(), next);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < max_new; i++) {
            next = d.decode_token(next);
            printf("%d ", next);
            fflush(stdout);
        }
        auto t1 = std::chrono::steady_clock::now();
        const double s = std::chrono::duration<double>(t1 - t0).count();
        printf("\n[decode %d tokens in %.2fs: %.3f tok/s]\n", max_new, s, max_new / s);
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        return 3;
    }
}
