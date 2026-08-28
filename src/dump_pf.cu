// Chunked-prefill parity dump: argv = index tokens(t0,t1,...) T out.f32
// Runs prefill_chunk once over the first T tokens and writes 32 layer seams
// (each T x 4096 f32, row-major per token) plus the final post-norm seam.
#include "insignia_decode.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
static FILE *g_f;
static int g_T;
static void seam(int l, const float *pf_x, int T, void *) {
    std::vector<float> h(size_t(T) * 4096);
    cudaMemcpy(h.data(), pf_x, h.size() * 4, cudaMemcpyDeviceToHost);
    fwrite(h.data(), 4, h.size(), g_f);
}
int wmain(int argc, wchar_t **argv) {
    if (argc != 5) return 2;
    try {
        insignia::ModelFile m(argv[1]);
        std::vector<int> tokens;
        {
            static char buf[1 << 20];
            size_t n = wcstombs(buf, argv[2], sizeof(buf) - 1);
            buf[n] = 0;
            for (char *s = strtok(buf, ","); s; s = strtok(nullptr, ",")) tokens.push_back(atoi(s));
        }
        g_T = _wtoi(argv[3]);
        if (g_T < 1 || g_T > 64 || (int)tokens.size() < g_T) return 2;
        insignia::DecodeWorkspace x(4096);
        insignia::Qwen35Weights w(m, 6ull << 30, x.stream);
        insignia::Qwen35Decode d(w, x);
        g_f = _wfopen(argv[4], L"wb");
        int tok_dev[64];
        for (int i = 0; i < g_T; i++) tok_dev[i] = tokens[i];
        d.prefill_chunk_seam(tok_dev, g_T, seam, nullptr);
        // final normalized seam was already computed inside; append it via pf_n
        std::vector<float> hn(size_t(g_T) * 4096);
        cudaMemcpyAsync(hn.data(), x.pf_n, hn.size() * 4, cudaMemcpyDeviceToHost, x.stream);
        cudaStreamSynchronize(x.stream);
        fwrite(hn.data(), 4, hn.size(), g_f);
        fclose(g_f);
        printf("dumped 32 prefill seams T=%d\n", g_T);
        return 0;
    } catch (const std::exception &e) { fprintf(stderr, "%s\n", e.what()); return 3; }
}
