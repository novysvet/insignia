#pragma once
#include <cuda_runtime.h>
#include <cstdint>
namespace insignia {
// Batched prefill kernels (T tokens at once, base position read from pos_dev[0]).
void embed_gather(const uint32_t *embed_w, const uint8_t *embed_s, const int *tokens_dev, float *out, int T, cudaStream_t stream = nullptr);
void embed_gather_i4(const uint32_t *embed_w, const uint16_t *embed_s, const int *tokens_dev, float *out, int T, cudaStream_t stream = nullptr);
void split_q_gate_batch(const float *src, float *q, float *gate, int T, cudaStream_t stream = nullptr);
void qk_norm_rope_batch(float *q, float *k, const uint16_t *qw, const uint16_t *kw, const int *pos_dev, int T, cudaStream_t stream = nullptr);
void store_kv_batch(const float *k, const float *v, float *kc, float *vc, const int *pos_dev, int T, int max_context, cudaStream_t stream = nullptr);
void gqa_prefill(const float *q, const float *kc, const float *vc, float *out, const int *pos_dev, int T, int max_context, cudaStream_t stream = nullptr);
void conv_prefill_silu(float *x, float *scratch, float *state, const uint16_t *w, int T, cudaStream_t stream = nullptr, float *row0_snap = nullptr);
void deltanet_params_batch(float *a, float *b, const float *A_log, const uint16_t *dt_bias, int T, cudaStream_t stream = nullptr);
void deltanet_params_batch_h(float *a, float *b, const float *A_log, const uint16_t *dt_bias, int T, int heads, cudaStream_t stream = nullptr);
void deltanet_prefill(float *state, const float *qkv, const float *a, const float *b, float *out, int T, cudaStream_t stream = nullptr, float *row0_snap = nullptr);
void addi_kernel_launch(int *p, int v, cudaStream_t stream = nullptr);
// Device-state speculative step plumbing (graph replayable).
void spec_prologue(int *pos, cudaStream_t stream = nullptr);
void spec_setup(int *pos, int *pf_tokens, cudaStream_t stream = nullptr);
void spec_commit(int *pos, int *committed, cudaStream_t stream = nullptr);
void spec_rollback(const float *snap_delta, const float *snap_conv, float *delta_state, float *conv_state, const float *pf_x, float *hidden, const int *pos, cudaStream_t stream = nullptr);
}
