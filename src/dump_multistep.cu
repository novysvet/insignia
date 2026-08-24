#include "insignia_decode.hpp"
#include "insignia_layout.cuh"
#include "insignia_ops.cuh"
#include "insignia_qwen_kernels.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
// argv: index tokens(comma-separated, e.g. 760,6511,314,9338,369) outfile
// Dumps per step: 32 layer outputs + final model.norm output (33 rows of 4096 f32).
// Prints greedy argmax token per step (teacher-forced steps + one generated step).
int wmain(int argc, wchar_t** argv) {
    if (argc != 4) return 2;
    try {
        insignia::ModelFile m(argv[1]);
        insignia::Qwen35Weights w(m, 6ull << 30);
        std::vector<int> tokens;
        {
            char buf[1 << 16];
            size_t n = wcstombs(buf, argv[2], sizeof(buf) - 1);
            buf[n] = 0;
            for (char* s = strtok(buf, ","); s; s = strtok(nullptr, ","))
                tokens.push_back(atoi(s));
        }
        if (tokens.empty()) return 2;
        insignia::DecodeWorkspace x(64);
        insignia::Qwen35Decode d(w, x);
        FILE* f = _wfopen(argv[3], L"wb");
        std::vector<float> h(4096);
        auto mat = [&](const char* base, const float* in, float* out) {
            auto z = w.matrix(base);
            insignia::mxfp4_gemv_mlx((const uint32_t*)z.weight.data, (const uint8_t*)z.scales.data, in, out, z.rows, z.cols, 2);
            w.release(base);
        };
        int next = -1;
        for (size_t step = 0; step <= tokens.size(); step++) {
            int tok = step < tokens.size() ? tokens[step] : next;
            w.embed(tok, x.hidden);
            for (int l = 0; l < 32; l++) {
                d.layer(l);
                cudaMemcpy(h.data(), x.hidden, 16384, cudaMemcpyDeviceToHost);
                fwrite(h.data(), 4, h.size(), f);
            }
            auto nw = w.storage().acquire("language_model.model.norm.weight");
            insignia::rmsnorm_bf16(x.hidden, (const uint16_t*)nw.data, x.norm, 1, 4096, false, x.stream);
            w.storage().release("language_model.model.norm.weight");
            cudaMemcpy(h.data(), x.norm, 16384, cudaMemcpyDeviceToHost);
            fwrite(h.data(), 4, h.size(), f);
            mat("language_model.lm_head", x.norm, x.logits);
            int* dt = nullptr;
            cudaMalloc(&dt, sizeof(int));
            insignia::argmax_logits(x.logits, 248320, dt, x.stream);
            cudaMemcpy(&next, dt, sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dt);
            x.position++;
            int draft = d.mtp_draft(tok);
            fflush(f);
            printf("step %zu token %d -> next %d draft %d\n", step, tok, next, draft);
        }
        fclose(f);
        return 0;
    } catch (const std::exception& e) { fprintf(stderr, "%s\n", e.what()); return 3; }
}
