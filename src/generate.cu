#include "insignia_decode.hpp"
#include "insignia_layout.cuh"
#include "insignia_prefill.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <vector>
// argv: index tokens(comma-separated prompt ids) max_new [benchmark]
// Prefill runs batched chunks, then the whole speculative step (MTP draft + pair verify +
// device-side commit/rollback) is captured as one CUDA graph and replayed back-to-back.
// The host only checks the committed id stream every few steps for EOS/max_new.

// NLL mode: "nll" <index> <tokens> — teacher-forced perplexity via batched chunks.
__global__ void row_logp_kernel(const float *__restrict__ logits, const int *__restrict__ targets, float *__restrict__ logp, int vocab) {
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
static int run_nll(int argc, wchar_t **argv) {
    insignia::ModelFile m(argv[2]);
    std::vector<int> tokens;
    {
        char buf[1 << 20];
        size_t n = wcstombs(buf, argv[3], sizeof(buf) - 1);
        buf[n] = 0;
        for (char *s2 = strtok(buf, ","); s2; s2 = strtok(nullptr, ",")) tokens.push_back(atoi(s2));
    }
    if (tokens.size() < 2) return 2;
    insignia::DecodeWorkspace x(4096);
    insignia::Qwen35Weights w(m, 6ull << 30, x.stream);
    insignia::Qwen35Decode d(w, x);
    const int vocab = insignia::Qwen35Shape::vocab;
    float *logitsT;
    if (cudaMalloc(&logitsT, size_t(64) * vocab * 4)) throw std::runtime_error("logits alloc");
    int *targets;
    cudaMalloc(&targets, 64 * 4);
    float *logp;
    cudaMalloc(&logp, 64 * 4);
    double total = 0;
    size_t count = 0, done = 0;
    while (done + 1 < tokens.size()) {
        const size_t remain = tokens.size() - done - 1;
        const int T = int(remain) >= 64 ? 64 : int(remain);
        d.prefill_chunk(tokens.data() + done, T);
        std::vector<int> tgt(tokens.begin() + done + 1, tokens.begin() + done + 1 + T);
        cudaMemcpyAsync(targets, tgt.data(), T * 4, cudaMemcpyHostToDevice, x.stream);
        {
            auto lh = w.matrix("language_model.lm_head");
            const bool i4 = lh.insig4;
            if (i4) insignia::mxfp4_gemm_mlx_i4((const uint32_t *)lh.weight.data, (const uint16_t *)lh.scales.data, x.pf_n, logitsT, lh.rows, lh.cols, T, x.stream);
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
}

int wmain(int argc, wchar_t** argv) {
    if (argc > 3 && wcscmp(argv[1], L"nll") == 0) {
        try { return run_nll(argc, argv); }
        catch (const std::exception& e) { fprintf(stderr, "%s\n", e.what()); return 3; }
    }
    if (argc < 4) return 2;
    try {
        insignia::ModelFile m(argv[1]);
        std::vector<int> tokens;
        {
            char buf[1 << 16];
            size_t n = wcstombs(buf, argv[2], sizeof(buf) - 1);
            buf[n] = 0;
            for (char* s = strtok(buf, ","); s; s = strtok(nullptr, ","))
                tokens.push_back(atoi(s));
        }
        int max_new = _wtoi(argv[3]);
        if (tokens.empty() || max_new <= 0) return 2;
        int ctx = int(tokens.size()) + max_new + 16;
        // graph replay bypasses the eager KV-full guard, so over-budget runs must be
        // rejected here rather than clamped (clamped ctx + unclamped want_total = OOB KV)
        if (ctx > 4090) throw std::runtime_error("context overflow: prompt+max_new+16 exceeds the 4090 cache cap; reduce max_new");
        insignia::DecodeWorkspace x(ctx);
        insignia::Qwen35Weights w(m, 6ull << 30, x.stream);
        insignia::Qwen35Decode d(w, x);
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        cudaEventRecord(a, x.stream);
        size_t done = 0;
        int first = -1;
        while (done < tokens.size()) {
            int T = int(tokens.size() - done) >= 64 ? 64 : int(tokens.size() - done);
            first = d.prefill_chunk(tokens.data() + done, T);
            done += T;
        }
        // committed stream starts with the prompt ids; the warmup spec step commits
        // [first, second] itself, so `first` is not written here
        d.append_committed_host(tokens.data(), int(tokens.size()));
        d.prime_spec(first);
        if (argc > 4 && wcscmp(argv[4], L"probe") == 0) {  // MTP probe: dump-style drive right after prefill
            d.set_mtp_position(int(tokens.size()) - 1);
            d.mtp_layer();
            int draft = -1;
            cudaMemcpyAsync(&draft, x.next_dev, sizeof(int), cudaMemcpyDeviceToHost, x.stream);
            cudaStreamSynchronize(x.stream);
            printf("probe: tok=%d mtp_pos=%d draft=%d (TF next after tok should be plausible)\n",
                   first, int(tokens.size()) - 1, draft);
            cudaMemsetAsync(x.mtp_keys, 0, size_t(x.max_context) * 1024 * 4, x.stream);
            cudaMemsetAsync(x.mtp_values, 0, size_t(x.max_context) * 1024 * 4, x.stream);
        }
        int warm = d.spec_step(first);  // eager warmup: primes every kernel + static init
        if (argc > 4 && (wcscmp(argv[4], L"probe") == 0 || wcscmp(argv[4], L"eager") == 0)) {
            int trip[7];
            cudaMemcpyAsync(trip, x.pos_dev, 7 * sizeof(int), cudaMemcpyDeviceToHost, x.stream);
            cudaStreamSynchronize(x.stream);
            printf("warm: pos=%d t0=%d draft=%d t2=%d after=%d acc=%d\n", trip[0], first, trip[4], trip[3], trip[2], trip[6] ? 1 : 0);
        }
        const int want_total = int(tokens.size()) + max_new;
        const bool eager = argc > 4 && wcscmp(argv[4], L"eager") == 0;
        if (eager) {  // per-step synced fallback for differential testing against the graph path
            while (d.committed_count() < want_total) {
                int prev_pending = d.next_token();
                d.spec_step(prev_pending);
                {   // dump the step decision triple for alignment with the reference
                    int trip[8];
                    cudaMemcpyAsync(trip, x.pos_dev, 8 * sizeof(int), cudaMemcpyDeviceToHost, x.stream);
                    cudaStreamSynchronize(x.stream);
                    printf("spec pos=%d t0=%d draft=%d t2=%d after=%d acc=%d\n", trip[0], prev_pending, trip[4], trip[3], trip[2], trip[6] ? 1 : 0);
                }
                d.read_committed(x.host_committed, d.committed_count());
                bool eos = false;
                for (int i = int(tokens.size()); i < d.committed_count(); i++)
                    if (x.host_committed[i] == 151643 || x.host_committed[i] == 151645) { eos = true; break; }
                if (eos) break;
            }
        } else {
            d.capture_spec();
        }
        (void)warm;
        cudaEventRecord(b, x.stream);
        cudaEventSynchronize(b);
        float prefill_ms;
        cudaEventElapsedTime(&prefill_ms, a, b);
        cudaEventRecord(a, x.stream);
        if (!eager) {
            for (;;) {
                for (int i = 0; i < 4; i++) d.spec_graph_step();  // bounded by count checks: <= want_total+7 ids
                int count = d.committed_count();  // syncs the stream
                if (count >= want_total) break;
                d.read_committed(x.host_committed, count);
                bool eos = false;
                for (int i = int(tokens.size()); i < count; i++)
                    if (x.host_committed[i] == 151643 || x.host_committed[i] == 151645) { eos = true; break; }
                if (eos) break;
            }
        }
        cudaEventRecord(b, x.stream);
        cudaEventSynchronize(b);
        int count = d.committed_count();
        d.read_committed(x.host_committed, count);
        // trim: cap at max_new generated, cut at EOS (keep everything before it)
        int end = count;
        if (end > want_total) end = want_total;
        for (int i = int(tokens.size()); i < end; i++)
            if (x.host_committed[i] == 151643 || x.host_committed[i] == 151645) { end = i; break; }
        int gen = end - int(tokens.size());
        float total_ms;
        cudaEventElapsedTime(&total_ms, a, b);
        printf("prefill %d tok in %.2f ms (%.1f us/tok); spec decode %d tok total %.3f ms = %.2f ms/tok %.1f tok/s\n",
               (int)tokens.size(), prefill_ms, prefill_ms * 1000.0f / tokens.size(),
               gen, total_ms, gen ? total_ms / gen : 0.f, gen ? gen * 1000.0f / total_ms : 0.f);
        printf("ids:");
        for (int i = int(tokens.size()); i < end; i++) printf(" %d", x.host_committed[i]);
        printf("\n");
        return 0;
    } catch (const std::exception& e) { fprintf(stderr, "%s\n", e.what()); return 3; }
}
