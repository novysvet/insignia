#include "insignia_decode.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
// argv: index tokens(comma-separated prompt ids) max_new [benchmark]
// Prefill runs batched chunks, then the whole speculative step (MTP draft + pair verify +
// device-side commit/rollback) is captured as one CUDA graph and replayed back-to-back.
// The host only checks the committed id stream every few steps for EOS/max_new.
int wmain(int argc, wchar_t** argv) {
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
        if (ctx > 4090) ctx = 4090;
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
