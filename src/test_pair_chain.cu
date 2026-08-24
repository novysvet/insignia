#include "insignia_decode.hpp"
#include <cuda_runtime.h>
#include <cstdio>
// argv: index
// Chains true-greedy pairs through the spec pair path: every step must accept
// (draft == t2 == known greedy next). The first step whose row0 argmax disagrees
// with the greedy chain localizes state corruption in the pair forward.
int wmain(int argc, wchar_t **argv) {
    if (argc != 2) return 2;
    try {
        insignia::ModelFile m(argv[1]);
        insignia::DecodeWorkspace x(128);
        insignia::Qwen35Weights w(m, 6ull << 30, x.stream);
        insignia::Qwen35Decode d(w, x);
        const int prompt[] = {760, 6511, 314, 9338, 369, 11751, 13};
        const int greedy[] = {198, 760, 6511, 314, 9338, 369, 11751, 13, 198, 760, 6511, 314, 9338, 369, 11751, 13, 198};
        const int P = sizeof(prompt) / 4, G = sizeof(greedy) / 4;
        d.prefill_chunk(prompt, P);
        // sequential single-token sanity: greedy chain via decode_token
        {
            insignia::DecodeWorkspace y(128);
            insignia::Qwen35Weights w2(m, 6ull << 30, y.stream);
            insignia::Qwen35Decode d2(w2, y);
            d2.prefill_chunk(prompt, P);
            int bad = 0;
            for (int i = 0; i < G; i++) {
                int prev = i ? greedy[i - 1] : 13;
                (void)prev;
                int got = i == 0 ? -1 : d2.decode_token(greedy[i - 1]);
                if (i > 0 && got != greedy[i]) { printf("single-token path WRONG at %d: got %d want %d\n", i, got, greedy[i]); bad++; break; }
            }
            printf("single-token chain: %s\n", bad ? "FAIL" : "matches greedy");
        }
        // pair chain: pair i = [greedy[i], greedy[i+1]] (must accept), t2 must equal greedy[i+1]
        int tok0 = greedy[0];
        for (int i = 0; i + 1 < G; i += 2) {
            int pair[2] = {greedy[i], greedy[i + 1]};
            int after = d.prefill_chunk(pair, 2);
            int t2 = -1;
            cudaMemcpyAsync(&t2, x.next2_dev, sizeof(int), cudaMemcpyDeviceToHost, x.stream);
            cudaStreamSynchronize(x.stream);
            int want_t2 = pair[1];  // row0 = greedy[i]; true next is greedy[i+1] = pair[1]
            printf("pair %d [%d,%d]: t2=%d want=%d after=%d\n", i, pair[0], pair[1], t2, want_t2, after);
            if (t2 != want_t2 || (i + 2 < G && after != greedy[i + 2])) { printf(">>> DIVERGED at pair %d (row0 ctx ends ...%d)\n", i, pair[0]); break; }
            tok0 = after;
        }
        (void)tok0;
        return 0;
    } catch (const std::exception &e) { fprintf(stderr, "%s\n", e.what()); return 3; }
}
