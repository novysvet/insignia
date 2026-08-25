#include "insignia_decode.hpp"
#include <cuda_runtime.h>
#include <cstdio>
int wmain(int argc, wchar_t** argv) {
    if (argc != 2) return 2;
    try {
        insignia::ModelFile m(argv[1]);
        insignia::DecodeWorkspace x(64);
        insignia::Qwen35Weights w(m, 6ull << 30, x.stream);
        insignia::Qwen35Decode d(w, x);
        // Reference: after "The"(760) next is 2614; after "The capital"(6511) next is 314.
        int pair[2] = {760, 6511};
        int last = d.prefill_chunk(pair, 2);
        int t2 = -1;
        cudaMemcpyAsync(&t2, x.next2_dev, sizeof(int), cudaMemcpyDeviceToHost, x.stream);
        cudaStreamSynchronize(x.stream);
        printf("pair [760,6511]: argmax(row0)=%d (want 2614), argmax(row1)=%d (want 314)\n", t2, last);
        // Same tokens sequentially for comparison.
        insignia::DecodeWorkspace y(64);
        insignia::Qwen35Weights w2(m, 6ull << 30, y.stream);
        insignia::Qwen35Decode d2(w2, y);
        int p1[2] = {760};
        int s1 = d2.prefill_chunk(p1, 1);
        int s2 = d2.decode_token(6511);
        printf("sequential: after 760 -> %d (want 2614), after 6511 -> %d (want 314)\n", s1, s2);
        return 0;
    } catch (const std::exception& e) { fprintf(stderr, "%s\n", e.what()); return 3; }
}
