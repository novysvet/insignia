// INSIG4 T=2 chunk-path seams for layer 0 (pair/dp4a + batch kernels).
#include "insignia_decode.hpp"
#include "insignia_layout.cuh"
#include "insignia_deltanet.cuh"
#include "insignia_qwen_kernels.cuh"
#include "insignia_ops.cuh"
#include "insignia_prefill.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
using namespace insignia;
int wmain(int argc, wchar_t **argv) {
    if (argc != 3) return 2;
    try {
        ModelFile m(argv[1]);
        DecodeWorkspace x(64);
        Qwen35Weights w(m, 6ull << 30, x.stream);
        Qwen35Decode d(w, x);
        const int T = 2, toks0[2] = {42, 99};
        const int toks[2] = {760, 6511};
        d.prefill_chunk(toks0, 2);  // chunk 1: real history, states now carried
        {   // dump carried states for layer 0 (di=0): conv 8192*3 + delta 32*128*128
            cudaStreamSynchronize(x.stream);
            std::vector<float> cv(8192 * 3), ds(32 * 128 * 128);
            cudaMemcpy(cv.data(), x.conv_state, cv.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(ds.data(), x.delta_state, ds.size() * 4, cudaMemcpyDeviceToHost);
            FILE *g = _wfopen(argv[2], L"wb");
            fwrite(cv.data(), 4, cv.size(), g);
            fwrite(ds.data(), 4, ds.size(), g);
            fclose(g);
        }
        cudaMemcpyAsync(x.pf_tokens, toks, 8, cudaMemcpyHostToDevice, x.stream);
        FILE *f = _wfopen(argv[2], L"wb");
        auto dump = [&](const float *p, int n) { cudaStreamSynchronize(x.stream); std::vector<float> h(n); cudaMemcpy(h.data(), p, n * 4, cudaMemcpyDeviceToHost); fwrite(h.data(), 4, n, f); };
        {   // embed via the CHUNK kernel (embed_gather_i4)
            auto mm = w.matrix("language_model.model.embed_tokens");
            embed_gather_i4((const uint32_t *)mm.weight.data, (const uint16_t *)mm.scales.data, x.pf_tokens, x.pf_x, T, x.stream);
            w.release("language_model.model.embed_tokens");
        }
        dump(x.pf_x, 2 * 4096);  // seam 0
        const std::string p = "language_model.model.layers.0", a = p + ".linear_attn";
        auto inw = w.storage().acquire(p + ".input_layernorm.weight");
        rmsnorm_bf16(x.pf_x, (const uint16_t *)inw.data, x.pf_n, T, 4096, false, x.stream);
        w.storage().release(p + ".input_layernorm.weight");
        dump(x.pf_n, 2 * 4096);  // seam 1
        auto gemv2 = [&](const std::string &base, const float *in, float *out) {
            auto mm = w.matrix(base);
            mxfp4_gemv2_q8_i4((const uint32_t *)mm.weight.data, (const uint16_t *)mm.scales.data, in, out, mm.rows, mm.cols, x.stream);
            w.release(base);
        };
        gemv2(a + ".in_proj_qkv", x.pf_n, x.pf_qkv); dump(x.pf_qkv, 2 * 8192);  // seam 2
        gemv2(a + ".in_proj_z", x.pf_n, x.pf_z); dump(x.pf_z, 2 * 4096);  // seam 3
        {   // fused ab2
            auto ma = w.matrix(a + ".in_proj_a"), mb = w.matrix(a + ".in_proj_b");
            mxfp4_gemv_ab2_q8_i4((const uint32_t *)ma.weight.data, (const uint16_t *)ma.scales.data, (const uint32_t *)mb.weight.data, (const uint16_t *)mb.scales.data, x.pf_n, x.pf_a, x.pf_b, ma.cols, x.stream);
            w.release(a + ".in_proj_a"); w.release(a + ".in_proj_b");
        }
        dump(x.pf_a, 2 * 32); dump(x.pf_b, 2 * 32);  // seams 4,5
        {   // batch conv
            auto cw = w.storage().acquire(a + ".conv1d.weight");
            conv_prefill_silu(x.pf_qkv, x.pf_scratch, x.conv_state, (const uint16_t *)cw.data, T, x.stream, x.snap_conv);
            w.storage().release(a + ".conv1d.weight");
        }
        dump(x.pf_scratch, 2 * 8192);  // seam 6
        {
            auto A = w.storage().acquire(a + ".A_log"), dt = w.storage().acquire(a + ".dt_bias");
            deltanet_params_batch(x.pf_a, x.pf_b, (const float *)A.data, (const uint16_t *)dt.data, T, x.stream);
            w.storage().release(a + ".A_log"); w.storage().release(a + ".dt_bias");
        }
        dump(x.pf_a, 2 * 32); dump(x.pf_b, 2 * 32);  // seams 7,8
        deltanet_prefill(x.delta_state, x.pf_scratch, x.pf_a, x.pf_b, x.pf_core, T, x.stream, x.snap_delta);
        dump(x.pf_core, 2 * 4096);  // seam 9
        {
            auto nw = w.storage().acquire(a + ".norm.weight");
            gated_rmsnorm_bf16(x.pf_core, (const uint16_t *)nw.data, x.pf_z, x.pf_core, size_t(T) * 32, 128, x.stream);
            w.storage().release(a + ".norm.weight");
        }
        dump(x.pf_core, 2 * 4096);  // seam 10
        gemv2(a + ".out_proj", x.pf_core, x.pf_down);
        residual_add(x.pf_x, x.pf_down, 2 * 4096, x.stream);
        dump(x.pf_x, 2 * 4096);  // seam 11
        {
            auto post = w.storage().acquire(p + ".post_attention_layernorm.weight");
            rmsnorm_bf16(x.pf_x, (const uint16_t *)post.data, x.pf_n, T, 4096, false, x.stream);
            w.storage().release(p + ".post_attention_layernorm.weight");
        }
        gemv2(p + ".mlp.gate_proj", x.pf_n, x.pf_gate);
        gemv2(p + ".mlp.up_proj", x.pf_n, x.pf_up);
        silu_mul(x.pf_gate, x.pf_up, x.pf_gate, 2 * 12288, x.stream);
        dump(x.pf_gate, 2 * 12288);  // seam 12
        gemv2(p + ".mlp.down_proj", x.pf_gate, x.pf_down);
        residual_add(x.pf_x, x.pf_down, 2 * 4096, x.stream);
        dump(x.pf_x, 2 * 4096);  // seam 13
        // layers 1,2 (delta) + 3 (full attn) chunk seams
        for (int l = 1; l < 32; l++) {
            const std::string pp = "language_model.model.layers." + std::to_string(l);
            auto in2 = w.storage().acquire(pp + ".input_layernorm.weight");
            rmsnorm_bf16(x.pf_x, (const uint16_t *)in2.data, x.pf_n, T, 4096, false, x.stream);
            w.storage().release(pp + ".input_layernorm.weight");
            if ((l & 3) != 3) {
                const std::string aa2 = pp + ".linear_attn";
                gemv2(aa2 + ".in_proj_qkv", x.pf_n, x.pf_qkv);
                gemv2(aa2 + ".in_proj_z", x.pf_n, x.pf_z);
                {auto ma = w.matrix(aa2 + ".in_proj_a"), mb = w.matrix(aa2 + ".in_proj_b");
                 mxfp4_gemv_ab2_q8_i4((const uint32_t *)ma.weight.data, (const uint16_t *)ma.scales.data, (const uint32_t *)mb.weight.data, (const uint16_t *)mb.scales.data, x.pf_n, x.pf_a, x.pf_b, ma.cols, x.stream);
                 w.release(aa2 + ".in_proj_a"); w.release(aa2 + ".in_proj_b");}
                auto cw2 = w.storage().acquire(aa2 + ".conv1d.weight");
                const int di = l - l / 4;
                conv_prefill_silu(x.pf_qkv, x.pf_scratch, x.conv_state + size_t(di) * 8192 * 3, (const uint16_t *)cw2.data, T, x.stream, x.snap_conv + size_t(di) * 8192 * 3);
                w.storage().release(aa2 + ".conv1d.weight");
                auto A2 = w.storage().acquire(aa2 + ".A_log"), dt2 = w.storage().acquire(aa2 + ".dt_bias");
                deltanet_params_batch(x.pf_a, x.pf_b, (const float *)A2.data, (const uint16_t *)dt2.data, T, x.stream);
                w.storage().release(aa2 + ".A_log"); w.storage().release(aa2 + ".dt_bias");
                deltanet_prefill(x.delta_state + size_t(di) * 32 * 128 * 128, x.pf_scratch, x.pf_a, x.pf_b, x.pf_core, T, x.stream, x.snap_delta + size_t(di) * 32 * 128 * 128);
                auto nw2 = w.storage().acquire(aa2 + ".norm.weight");
                gated_rmsnorm_bf16(x.pf_core, (const uint16_t *)nw2.data, x.pf_z, x.pf_core, size_t(T) * 32, 128, x.stream);
                w.storage().release(aa2 + ".norm.weight");
                gemv2(aa2 + ".out_proj", x.pf_core, x.pf_down);
            } else {
                const std::string sa = pp + ".self_attn";
                gemv2(sa + ".q_proj", x.pf_n, x.pf_scratch);
                split_q_gate_batch(x.pf_scratch, x.pf_q, x.pf_g, T, x.stream);
                gemv2(sa + ".k_proj", x.pf_n, x.pf_k);
                gemv2(sa + ".v_proj", x.pf_n, x.pf_v);
                auto qw = w.storage().acquire(sa + ".q_norm.weight"), kw = w.storage().acquire(sa + ".k_norm.weight");
                qk_norm_rope_batch(x.pf_q, x.pf_k, (const uint16_t *)qw.data, (const uint16_t *)kw.data, x.pos_dev, T, x.stream);
                w.storage().release(sa + ".q_norm.weight"); w.storage().release(sa + ".k_norm.weight");
                const int ai = l / 4;
                float *kc = x.kv_keys + size_t(ai) * x.max_context * 1024, *vc = x.kv_values + size_t(ai) * x.max_context * 1024;
                store_kv_batch(x.pf_k, x.pf_v, kc, vc, x.pos_dev, T, x.max_context, x.stream);
                gqa_prefill(x.pf_q, kc, vc, x.pf_core, x.pos_dev, T, x.max_context, x.stream);
                sigmoid_mul(x.pf_core, x.pf_g, size_t(T) * 4096, x.stream);
                gemv2(sa + ".o_proj", x.pf_core, x.pf_down);
            }
            residual_add(x.pf_x, x.pf_down, 2 * 4096, x.stream);
            auto post2 = w.storage().acquire(pp + ".post_attention_layernorm.weight");
            rmsnorm_bf16(x.pf_x, (const uint16_t *)post2.data, x.pf_n, T, 4096, false, x.stream);
            w.storage().release(pp + ".post_attention_layernorm.weight");
            gemv2(pp + ".mlp.gate_proj", x.pf_n, x.pf_gate);
            gemv2(pp + ".mlp.up_proj", x.pf_n, x.pf_up);
            silu_mul(x.pf_gate, x.pf_up, x.pf_gate, 2 * 12288, x.stream);
            gemv2(pp + ".mlp.down_proj", x.pf_gate, x.pf_down);
            residual_add(x.pf_x, x.pf_down, 2 * 4096, x.stream);
            dump(x.pf_x, 2 * 4096);  // seams 14,15,16 = layers 1,2,3
        }
        {
            auto fnw = w.storage().acquire("language_model.model.norm.weight");
            rmsnorm_bf16(x.pf_x, (const uint16_t *)fnw.data, x.pf_n, T, 4096, false, x.stream);
            w.storage().release("language_model.model.norm.weight");
        }
        {
            auto lh = w.matrix("language_model.lm_head");
            float *logits;
            cudaMalloc(&logits, size_t(64) * 248320 * 4);
            mxfp4_gemm_mlx_i4((const uint32_t *)lh.weight.data, (const uint16_t *)lh.scales.data, x.pf_n, logits, lh.rows, lh.cols, T, x.stream);
            cudaStreamSynchronize(x.stream);
            std::vector<float> lg(2 * 248320);
            cudaMemcpy(lg.data(), logits, 2 * 248320 * 4, cudaMemcpyDeviceToHost);
            fwrite(lg.data(), 4, lg.size(), f);
            cudaFree(logits);
        }
        fclose(f);
        printf("dumped 14 chunk seams\n");
        return 0;
    } catch (const std::exception &e) { fprintf(stderr, "%s\n", e.what()); return 3; }
}
