#include "insignia_decode.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
// argv: index n_tokens - warm/cold prefill timing; second run is warm (weights resident).
int wmain(int argc, wchar_t** argv) {
    if (argc != 3) return 2;
    try {
        insignia::ModelFile m(argv[1]);
        int n = _wtoi(argv[2]);
        insignia::DecodeWorkspace x(4096);
        insignia::Qwen35Weights w(m, 6ull << 30, x.stream);
        insignia::Qwen35Decode d(w, x);
        std::vector<int> tokens;
        for (int i = 0; i < n; i++) tokens.push_back(1000 + (i * 7) % 20000);
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        float cold = 0, warm = 0;
        for (int run = 0; run < 2; run++) {
            cudaEventRecord(a, x.stream);
            size_t done = 0;
            while (done < (size_t)n) {
                int T = n - (int)done >= 64 ? 64 : n - (int)done;
                d.prefill_chunk(tokens.data() + done, T);
                done += T;
            }
            cudaEventRecord(b, x.stream);
            cudaEventSynchronize(b);
            float ms;
            cudaEventElapsedTime(&ms, a, b);
            if (run == 0) cold = ms; else warm = ms;
        }
        printf("prefill %d tok: cold %.1f ms (%.2f ms/tok), warm %.1f ms (%.2f ms/tok)\n",
               n, cold, cold / n, warm, warm / n);
        return 0;
    } catch (const std::exception& e) { fprintf(stderr, "%s\n", e.what()); return 3; }
}
