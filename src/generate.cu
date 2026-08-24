#include "insignia_decode.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
// argv: index tokens(comma-separated prompt ids) max_new [benchmark]
// Prefill runs per-token forwards, then the decode step is captured as a CUDA graph and
// replayed self-feedingly (device-side token/position). Prints generated ids and timing.
int wmain(int argc, wchar_t** argv) {
    if (argc < 4) return 2;
    try {
        bool bench = argc > 4 && wcscmp(argv[4], L"benchmark") == 0;
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
        int ctx = int(tokens.size()) + max_new + 8;
        if (ctx > 4096) ctx = 4096;
        insignia::DecodeWorkspace x(ctx);
        insignia::Qwen35Weights w(m, 6ull << 30, x.stream);
        insignia::Qwen35Decode d(w, x);
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        // Prefill: sequential target forwards (uploads all weights; batched prefill is future work).
        cudaEventRecord(a, x.stream);
        for (int tok : tokens) d.forward_token(tok);
        cudaEventRecord(b, x.stream);
        cudaEventSynchronize(b);
        float prefill_ms;
        cudaEventElapsedTime(&prefill_ms, a, b);
        int first = d.logits_argmax();  // next_dev now holds the first generated token
        d.capture_step();
        std::vector<int> out;
        out.push_back(first);
        float decode_ms = 0;
        int steps = 0;
        int next = first;
        cudaEventRecord(a, x.stream);
        while ((int)out.size() < max_new && x.position < x.max_context) {
            d.step(next);
            next = d.next_token();
            out.push_back(next);
            steps++;
            if (next == 151643 || next == 151645) break;
        }
        cudaEventRecord(b, x.stream);
        cudaEventSynchronize(b);
        float total_ms;
        cudaEventElapsedTime(&total_ms, a, b);
        printf("prefill %d tok in %.2f ms (%.1f us/tok); graph decode %d tok total %.3f ms = %.2f ms/tok %.1f tok/s\n",
               (int)tokens.size(), prefill_ms, prefill_ms * 1000.0f / tokens.size(),
               steps, total_ms, steps ? total_ms / steps : 0.f, steps ? steps * 1000.0f / total_ms : 0.f);
        printf("ids:");
        for (int t : out) printf(" %d", t);
        printf("\n");
        (void)bench; (void)decode_ms;
        return 0;
    } catch (const std::exception& e) { fprintf(stderr, "%s\n", e.what()); return 3; }
}
