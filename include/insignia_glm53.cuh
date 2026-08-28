#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace insignia::glm53 {

// Workspace contains Q8 activations plus one FP32 scale per NVFP4 block.
size_t nvfp4_workspace_bytes(int cols);

// Uploads the E2M1 integer and E4M3 scale tables for the current CUDA device.
cudaError_t initialize_nvfp4();

// Exact checkpoint decode: packed E2M1 x E4M3-per-16 x global FP32.
cudaError_t nvfp4_gemv_f32(
    const uint8_t *weights,
    const uint8_t *scales,
    float global_scale,
    const float *x,
    float *y,
    int rows,
    int cols,
    cudaStream_t stream = nullptr);

// Ada path: Q8 activation quantization followed by signed INT8 DP4A.  The
// checkpoint bytes stay untouched; E2M1 magnitudes are represented as
// {0,1,2,3,4,6,8,12} and the missing factor 1/2 is folded into the scale.
cudaError_t nvfp4_gemv_dp4a(
    const uint8_t *weights,
    const uint8_t *scales,
    float global_scale,
    const float *x,
    float *y,
    int rows,
    int cols,
    void *workspace,
    cudaStream_t stream = nullptr);

cudaError_t nvfp4_quantize_activation(
    const float *x,
    int cols,
    void *workspace,
    cudaStream_t stream = nullptr);

cudaError_t nvfp4_gemv_dp4a_quantized(
    const uint8_t *weights,
    const uint8_t *scales,
    float global_scale,
    const void *workspace,
    float *y,
    int rows,
    int cols,
    cudaStream_t stream = nullptr);

// Accumulating variant: y[row] = fmaf(dot, combine_weight, y[row]) with the
// same rounding the separate scale_add pass had, minus the round trip.
cudaError_t nvfp4_gemv_dp4a_acc_quantized(
    const uint8_t *weights,
    const uint8_t *scales,
    float global_scale,
    const void *workspace,
    float *y,
    float combine_weight,
    int rows,
    int cols,
    cudaStream_t stream = nullptr);

// Two projections share one Q8 activation and one launch (expert gate + up).
cudaError_t nvfp4_gemv2_dp4a_quantized(
    const uint8_t *weights_a,
    const uint8_t *scales_a,
    float global_scale_a,
    const uint8_t *weights_b,
    const uint8_t *scales_b,
    float global_scale_b,
    const void *workspace,
    float *y_a,
    float *y_b,
    int rows,
    int cols,
    cudaStream_t stream = nullptr);

// GLM's clamped SwiGLU immediately requantized for the expert down projection.
cudaError_t quantize_swiglu_activation(
    const float *gate,
    const float *up,
    int cols,
    void *workspace,
    cudaStream_t stream = nullptr);

// GLM-5.3's dimensions are part of the ABI: 4 residual streams are baked into
// the mHC kernels and never parameterized.  Widths default to the Flash
// geometry (hidden 4096, 64 KDA heads of 128, MLA heads 64 of 256) so the
// production call sites stay literal; the 84M toy checkpoint passes its own
// dims explicitly.  head counts/dims must be powers of two >= 32.
constexpr int kHidden = 4096;
constexpr int kHyperStreams = 4;
constexpr int kKdaHeads = 64;
constexpr int kKdaHeadDim = 128;
constexpr int kMlaHeadDim = 256;
constexpr int kMlaMaxContext = 256;

// Scratch for the mHC dot-product and RMS partials.
size_t mhc_workspace_bytes();

// Derives mHC pre/post gates and the 4x4 doubly-stochastic residual mixer,
// then collapses four residual streams for the sublayer input.  fn is BF16
// [24,16384], while base[24] and scale[3] are FP32 checkpoint tensors.
// When rms_weight (BF16 [width/kHyperStreams]) is non-null the collapse is
// RMS-normalized in-kernel (bit-identical to the separate rms launch).
cudaError_t mhc_analyze(
    const uint16_t *fn,
    const float *base,
    const float *scale,
    const float *streams,
    const uint16_t *rms_weight,
    float *post,
    float *comb,
    float *collapsed,
    void *workspace,
    int width = kHyperStreams * kHidden,
    cudaStream_t stream = nullptr);

// Applies a sublayer result to all four streams.  Input and output stream
// buffers must not alias.
cudaError_t mhc_mix(
    const float *streams,
    const float *sublayer,
    const float *post,
    const float *comb,
    float *out_streams,
    int hidden = kHidden,
    cudaStream_t stream = nullptr);

// One-token KDA recurrence. q/k/v/g are [heads,head_dim], beta is [heads],
// state is [heads,head_dim,head_dim]. g contains log-decay values. head_dim
// is the thread-block width; the query scale is 1/sqrt(head_dim).
cudaError_t kda_decode(
    float *state,
    const float *q,
    const float *k,
    const float *v,
    const float *g,
    const float *beta,
    float *output,
    int heads = kKdaHeads,
    int head_dim = kKdaHeadDim,
    cudaStream_t stream = nullptr);

// Applies GLM's causal depthwise width-4 convolution and SiLU to one
// heads*head_dim-wide KDA projection. history is a three-slot ring of raw
// projection values.  Flash stores the taps BF16, the tiny oracle FP32.
cudaError_t kda_conv_silu(
    float *projection,
    const uint16_t *conv,
    float *history,
    int position,
    int count = kKdaHeads * kKdaHeadDim,
    bool weights_fp32 = false,
    cudaStream_t stream = nullptr);

// The three KDA projections in one launch: history segments stay at
// {q,k,v} x 3 x count exactly as the separate calls laid them out.
cudaError_t kda_conv_silu3(
    float *q,
    float *k,
    float *v,
    const void *conv_q,
    const void *conv_k,
    const void *conv_v,
    float *history,
    int position,
    int count = kKdaHeads * kKdaHeadDim,
    bool weights_fp32 = false,
    cudaStream_t stream = nullptr);

// Exact dense attention for the first kMlaMaxContext positions. GLM's sparse
// index top-k is 2048, so no candidate is omitted in this range.
// query/output are [heads,head_dim], kv is [heads,2*head_dim], and each cache
// is [kMlaMaxContext,heads,head_dim]. key and value dims must be equal.
cudaError_t mla_decode(
    const float *query,
    const float *kv,
    float *key_cache,
    float *value_cache,
    float *output,
    int position,
    int heads = kKdaHeads,
    int head_dim = kMlaHeadDim,
    cudaStream_t stream = nullptr);

// Forward-only causal FlashAttention-2 specialization for prompt chunks.
// query/output are token-major [tokens,heads,head_dim], while kv is
// token-major [tokens,heads,2*head_dim].  Existing cache entries before
// position_base are attended normally; this call appends the whole chunk.
cudaError_t mla_flash2_prefill(
    const float *query,
    const float *kv,
    float *key_cache,
    float *value_cache,
    float *output,
    int tokens,
    int position_base,
    int heads = kKdaHeads,
    int head_dim = kMlaHeadDim,
    cudaStream_t stream = nullptr);

}  // namespace insignia::glm53
