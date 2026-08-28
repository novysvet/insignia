// Surgical parity probe: run fp8_gemv on REAL layer-0 in_proj_qkv (+ out_proj, gate)
// with a deterministic input; dump y for numpy comparison against reference27 dequant.
#include "insignia_27b.hpp"
#include "insignia_fp8.cuh"

#include <cstdio>

int main() {
    insignia::Workspace27 x(64);
    insignia::TieredStorage27 st(L"build/qwen38-27b-fp8.insignia-index", L"build/manifest-v1.txt", x.stream);
    const insignia::LayerView27 &v0 = st.layer(0);
    // x = deterministic pattern in [-1,1]
    std::vector<float> hx(5120);
    for (int i = 0; i < 5120; i++) hx[i] = float((i * 2654435761u) >> 8 & 0xffff) / 65535.f * 2.f - 1.f;
    cudaMemcpy(x.norm, hx.data(), 5120 * 4, cudaMemcpyHostToDevice);
    auto run = [&](const insignia::Fp8View &f, const char *out_name) {
        insignia::fp8_gemv(f.w, f.s, x.norm, x.gate, f.rows, f.cols, x.stream);   // gate is [17408], largest
        cudaStreamSynchronize(x.stream);
        std::vector<float> y(f.rows);
        cudaMemcpy(y.data(), x.gate, size_t(f.rows) * 4, cudaMemcpyDeviceToHost);
        FILE *fp = fopen(out_name, "wb");
        fwrite(y.data(), 4, y.size(), fp);
        fclose(fp);
        printf("%s rows=%d cols=%d -> %s\n", out_name, f.rows, f.cols, out_name);
    };
    {   // dump the gate SCALES the engine sees (bf16 [136,40])
        std::vector<uint16_t> hs(136 * 40);
        cudaMemcpy(hs.data(), v0.gate.s, 136 * 40 * 2, cudaMemcpyDeviceToHost);
        FILE *fp = fopen("build/p27-gate-scales.u16", "wb");
        fwrite(hs.data(), 2, hs.size(), fp);
        fclose(fp);
        fflush(stdout);
    }
    run(v0.qkv, "build/p27-qkv.f32");
    run(v0.out, "build/p27-out.f32");
    run(v0.gate, "build/p27-gate.f32");
    return 0;
}
