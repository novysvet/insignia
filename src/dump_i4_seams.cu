// INSIG4 layer-0 seam dump: embed/nrm/qkv/z/a/b/conv/deltanet/gated/o/mlp into one file.
#include "insignia_decode.hpp"
#include "insignia_layout.cuh"
#include "insignia_deltanet.cuh"
#include "insignia_qwen_kernels.cuh"
#include "insignia_ops.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
using namespace insignia;
int wmain(int argc, wchar_t **argv) {
    if (argc != 3) return 2;
    try {
        insignia::ModelFile m(argv[1]);
        insignia::DecodeWorkspace x(16);
        insignia::Qwen35Weights w(m, 6ull << 30, x.stream);
        insignia::Qwen35Decode d(w, x);
        w.embed(42, x.hidden);
        FILE *f = _wfopen(argv[2], L"wb");
        auto dump = [&](const float *p, int n) { cudaStreamSynchronize(x.stream); std::vector<float> h(n); cudaMemcpy(h.data(), p, n * 4, cudaMemcpyDeviceToHost); fwrite(h.data(), 4, n, f); };
        dump(x.hidden, 4096);  // seam 0: embed
        const std::string p = "language_model.model.layers.0", a = p + ".linear_attn";
        auto inw = w.storage().acquire(p + ".input_layernorm.weight");
        rmsnorm_bf16(x.hidden, (const uint16_t *)inw.data, x.norm, 1, 4096, false, x.stream);
        w.storage().release(p + ".input_layernorm.weight");
        dump(x.norm, 4096);  // seam 1: nrm
        {   // seams 1a-d: raw dequantized qkv rows 0,1,4096,8191 via get_row_i4
            auto m = w.matrix(a + ".in_proj_qkv");
            static int rows_probe[4] = {0, 1, 4096, 8191};
            for (int pr = 0; pr < 4; pr++) {
                int *rd = (int *)x.pf_tokens;
                cudaMemcpyAsync(rd, &rows_probe[pr], 4, cudaMemcpyHostToDevice, x.stream);
                mxfp4_get_row_i4((const uint32_t *)m.weight.data, (const uint16_t *)m.scales.data, x.down, rd, 4096, x.stream);
                cudaDeviceSynchronize();
                dump(x.down, 4096);
            }
            w.release(a + ".in_proj_qkv");
        }
        { auto m = w.matrix(a + ".in_proj_qkv"); mxfp4_gemv_v2_i4((const uint32_t *)m.weight.data, (const uint16_t *)m.scales.data, x.norm, x.qkv, m.rows, m.cols, x.stream); w.release(a + ".in_proj_qkv"); } dump(x.qkv, 8192);  // seam 2
        { auto m = w.matrix(a + ".in_proj_z"); mxfp4_gemv_v2_i4((const uint32_t *)m.weight.data, (const uint16_t *)m.scales.data, x.norm, x.z, m.rows, m.cols, x.stream); w.release(a + ".in_proj_z"); } dump(x.z, 4096);  // seam 3
        { auto m = w.matrix(a + ".in_proj_a"); mxfp4_gemv_v2_i4((const uint32_t *)m.weight.data, (const uint16_t *)m.scales.data, x.norm, x.a, m.rows, m.cols, x.stream); w.release(a + ".in_proj_a"); } dump(x.a, 32);  // seam 4
        { auto m = w.matrix(a + ".in_proj_b"); mxfp4_gemv_v2_i4((const uint32_t *)m.weight.data, (const uint16_t *)m.scales.data, x.norm, x.b, m.rows, m.cols, x.stream); w.release(a + ".in_proj_b"); } dump(x.b, 32);  // seam 5
        auto cw = w.storage().acquire(a + ".conv1d.weight");
        causal_conv4_silu(x.qkv, x.conv_state, (const uint16_t *)cw.data, 8192, x.stream);
        w.storage().release(a + ".conv1d.weight");
        dump(x.qkv, 8192);  // seam 6: post conv+silu
        auto A = w.storage().acquire(a + ".A_log"), dt = w.storage().acquire(a + ".dt_bias");
        deltanet_parameters(x.a, x.b, (const float *)A.data, (const uint16_t *)dt.data, 32, x.stream);
        w.storage().release(a + ".A_log"); w.storage().release(a + ".dt_bias");
        dump(x.a, 32); dump(x.b, 32);  // seams 7,8: decay/beta
        cudaMemsetAsync(x.delta_state, 0, 24 * 32 * 128 * 128 * 4, x.stream);
        deltanet_decode(x.delta_state, x.qkv, x.qkv + 2048, x.qkv + 4096, x.a, x.b, x.core, x.stream);
        dump(x.core, 4096);  // seam 9
        auto nw = w.storage().acquire(a + ".norm.weight");
        gated_rmsnorm_bf16(x.core, (const uint16_t *)nw.data, x.z, x.core, 32, 128, x.stream);
        w.storage().release(a + ".norm.weight");
        dump(x.core, 4096);  // seam 10
        { auto m = w.matrix(a + ".out_proj"); mxfp4_gemv_v2_i4((const uint32_t *)m.weight.data, (const uint16_t *)m.scales.data, x.core, x.down, m.rows, m.cols, x.stream); w.release(a + ".out_proj"); }
        residual_add(x.hidden, x.down, 4096, x.stream);
        dump(x.hidden, 4096);  // seam 11
        auto post = w.storage().acquire(p + ".post_attention_layernorm.weight");
        rmsnorm_bf16(x.hidden, (const uint16_t *)post.data, x.norm, 1, 4096, false, x.stream);
        w.storage().release(p + ".post_attention_layernorm.weight");
        { auto m = w.matrix(p + ".mlp.gate_proj"); mxfp4_gemv_v2_i4((const uint32_t *)m.weight.data, (const uint16_t *)m.scales.data, x.norm, x.gate, m.rows, m.cols, x.stream); w.release(p + ".mlp.gate_proj"); }
        { auto m = w.matrix(p + ".mlp.up_proj"); mxfp4_gemv_v2_i4((const uint32_t *)m.weight.data, (const uint16_t *)m.scales.data, x.norm, x.up, m.rows, m.cols, x.stream); w.release(p + ".mlp.up_proj"); }
        silu_mul(x.gate, x.up, x.gate, 12288, x.stream);
        dump(x.gate, 12288);  // seam 12
        { auto m = w.matrix(p + ".mlp.down_proj"); mxfp4_gemv_v2_i4((const uint32_t *)m.weight.data, (const uint16_t *)m.scales.data, x.gate, x.down, m.rows, m.cols, x.stream); w.release(p + ".mlp.down_proj"); }
        residual_add(x.hidden, x.down, 4096, x.stream);
        dump(x.hidden, 4096);  // seam 13
        fclose(f);
        printf("dumped 14 seams\n");
        return 0;
    } catch (const std::exception &e) { fprintf(stderr, "%s\n", e.what()); return 3; }
}
