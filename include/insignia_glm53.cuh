#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace insignia::glm53 {

// Workspace contains Q8 activations plus one FP32 scale per NVFP4 block.
size_t nvfp4_workspace_bytes(int cols);

// Uploads the E2M1 integer and E4M3 scale tables for the current CUDA device.
cudaError_t initialize_nvfp4();

// Expands one projection's packed 4-bit scale-code stream on-device. The
// host supplies one exclusive escape-count prefix per 256 packed bytes;
// output is byte-identical to the AVX2 sidecar decoder.
cudaError_t expand_nvfp4_scale_nibbles(
    const uint8_t *packed,
    const uint8_t *escapes,
    const uint8_t *codebook,
    const uint32_t *block_prefix,
    uint8_t *output,
    size_t output_bytes,
    cudaStream_t stream = nullptr);

// Warp uint32 batch variant: one thread per packed word (8 codes, one
// 64-bit store). Same bytes, requires output_bytes % 4096 == 0 and natural
// alignment (packed % 4, block_prefix % 4, output % 8 — all guaranteed by
// the expert staging arena).
cudaError_t expand_nvfp4_scale_nibbles_v2(
    const uint8_t *packed,
    const uint8_t *escapes,
    const uint8_t *codebook,
    const uint32_t *block_prefix,
    uint8_t *output,
    size_t output_bytes,
    cudaStream_t stream = nullptr);

// Fused three-projection variants for the merged H2D transport: one launch
// expands all three scale planes from the contiguous device scratch blob
// into the interleaved destination slot (bodies at output_pitch strides,
// scale plane at scale_offset inside each). Offsets are absolute bytes into
// `scratch`; *_v2 is the warp uint32 worker. Output is byte-identical to
// three single-projection calls.
cudaError_t expand_nvfp4_scale_nibbles3(
    const uint8_t *scratch,
    const size_t *packed_offsets,
    const size_t *escape_offsets,
    const size_t *codebook_offsets,
    const size_t *prefix_offsets,
    uint8_t *output_base,
    size_t output_pitch,
    size_t scale_offset,
    size_t projection_bytes,
    cudaStream_t stream = nullptr);

cudaError_t expand_nvfp4_scale_nibbles3_v2(
    const uint8_t *scratch,
    const size_t *packed_offsets,
    const size_t *escape_offsets,
    const size_t *codebook_offsets,
    const size_t *prefix_offsets,
    uint8_t *output_base,
    size_t output_pitch,
    size_t scale_offset,
    size_t projection_bytes,
    cudaStream_t stream = nullptr);

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

// Multi-row (R <= 8) variants for the verify path: one weight pass serves all
// token rows that share an expert. Row r of the workspace holds the same
// layout nvfp4_workspace_bytes(cols) defines for a single row. Per-row output
// accumulation order is bit-identical to looping the 1-row kernels.
size_t nvfp4_workspace_rows_bytes(int cols, int rows);
cudaError_t nvfp4_quantize_activation_rows(
    const float *x,
    int cols,
    const int *row_ids,
    int count,
    void *workspace,
    cudaStream_t stream = nullptr);
cudaError_t quantize_swiglu_activation_rows(
    const float *gate,
    const float *up,
    int cols,
    const int *row_ids,
    int count,
    void *workspace,
    cudaStream_t stream = nullptr);
cudaError_t nvfp4_gemv_dp4a_quantized_rows(
    const uint8_t *weights,
    const uint8_t *scales,
    float global_scale,
    const void *workspace,
    int count,
    float *y,
    const int *y_ids,
    int rows,
    int cols,
    cudaStream_t stream = nullptr);
cudaError_t nvfp4_gemv_dp4a_acc_quantized_rows(
    const uint8_t *weights,
    const uint8_t *scales,
    float global_scale,
    const void *workspace,
    int count,
    float *y,
    const int *y_ids,
    const float *combine,
    int rows,
    int cols,
    cudaStream_t stream = nullptr);
cudaError_t nvfp4_gemv2_dp4a_quantized_rows(
    const uint8_t *weights_a,
    const uint8_t *scales_a,
    float global_scale_a,
    const uint8_t *weights_b,
    const uint8_t *scales_b,
    float global_scale_b,
    const void *workspace,
    int count,
    float *y_a,
    float *y_b,
    const int *y_ids,
    int rows,
    int cols,
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
constexpr int kMlaMaxContext = 262144;
constexpr int kMlaLatentDim = 512;
constexpr int kMlaLatentGroupSize = 64;
constexpr int kMlaLatentGroups = kMlaLatentDim / kMlaLatentGroupSize;

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

// Exact-prefix attention over a transient, fully reconstructed kv_b output.
// expanded_kv is token-major [context,heads,2*head_dim].  Unlike the legacy
// entry points these calls do not append to a persistent expanded K/V cache;
// the caller recreates rows 0..position from exact FP32 latents immediately
// before attention.  This trades tensor-core work for ~315 MiB of VRAM.
cudaError_t mla_decode_reconstructed(
    const float *query,
    const float *expanded_kv,
    float *output,
    int position,
    int heads = kKdaHeads,
    int head_dim = kMlaHeadDim,
    cudaStream_t stream = nullptr);

cudaError_t mla_flash2_prefill_reconstructed(
    const float *query,
    const float *expanded_kv,
    float *output,
    int tokens,
    int position_base,
    int heads = kKdaHeads,
    int head_dim = kMlaHeadDim,
    cudaStream_t stream = nullptr);

// Latent-cache MLA: the cache holds the compressed post-kv_a-layernorm
// latent [kMlaLatentDim] per (layer, position) instead of the expanded
// per-head K/V, so a full 8192-token cache costs ~50 MiB in FP8 instead of
// ~12 GiB in FP32.  Attention runs in absorbed form: scores are
// (q_head @ W_uk_head) . latent and the output is
// W_uv_head @ (softmax-weighted latent sum), which is algebraically the
// same attention but touches the 512-wide latent instead of the 16384-wide
// expanded K/V.  w_uk and w_uv are FP32-dequantized
// [heads,head_dim,latent_dim]
// row-major (j-major) per-head blocks of the kv_b_proj K and V halves.
// When cache_f32 is non-null the cache is FP32 and scales are ignored
// (INSIGNIA_GLM53_KV_FP8=0 A/B path); otherwise cache is e4m3 with one
// FP32 absmax/448 scale per 64-wide group, matching the dense FP8 path.

// Appends `tokens` latents [tokens,latent_dim] at positions
// position_base..; writes e4m3 bytes into cache and [tokens,8] scales
// (or FP32 into cache_f32 when it is non-null).
cudaError_t mla_store_latent(
    const float *latent,
    uint8_t *cache,
    float *scales,
    float *cache_f32,
    int tokens,
    int position_base,
    int latent_dim = kMlaLatentDim,
    cudaStream_t stream = nullptr);

// Single-token decode against the latent cache.  query is
// [heads,head_dim]; latent is the current token's [latent_dim] pre-norm
// latent, appended to the cache at `position` by this call.  When
// expanded_kv and exact_value_cache are both non-null (positions < 256), the
// expanded FP32 V half is cached and consumed directly while scores remain
// absorbed; otherwise partial scratch holds [heads,tiles,latent_dim+2] FP32
// merge data for the fully latent path.
cudaError_t mla_decode_latent(
    const float *query,
    const float *latent,
    const float *expanded_kv,
    uint8_t *cache,
    float *scales,
    float *cache_f32,
    float *exact_value_cache,
    const float *w_uk,
    const float *w_uv,
    float *partial,
    float *output,
    int position,
    int heads = kKdaHeads,
    int head_dim = kMlaHeadDim,
    int latent_dim = kMlaLatentDim,
    cudaStream_t stream = nullptr);

// Exact on-consumption variant: reconstructs the absorbed W_uk/W_uv
// coefficients directly from the named resident E4M3FN kv_b_proj tensor and
// FP16 group-64 scales. This removes the 64 MiB FP32 duplicate per MLA layer
// without changing either FMA chain.
cudaError_t mla_decode_latent_fp8_absorb(
    const float *query,
    const float *latent,
    uint8_t *cache,
    float *scales,
    float *cache_f32,
    const uint8_t *kv_b_fp8,
    const uint16_t *kv_b_scales,
    float *partial,
    float *output,
    int position,
    int heads = kKdaHeads,
    int head_dim = kMlaHeadDim,
    int latent_dim = kMlaLatentDim,
    cudaStream_t stream = nullptr);

// Approximate Ada decode path for long contexts. q_eff is quantized once per
// head, then one CTA shares each latent tile across eight heads and evaluates
// scores with m16n8k32 E4M3 tensor-core MMA. qeff_* are caller-owned scratch
// [heads,latent_dim] and [heads,latent_dim/64]. The partial ABI is unchanged.
cudaError_t mla_decode_latent_cross_head_fp8_absorb(
    const float *query,
    const float *latent,
    uint8_t *cache,
    float *scales,
    const uint8_t *kv_b_fp8,
    const uint16_t *kv_b_scales,
    uint8_t *qeff_fp8,
    float *qeff_scales,
    float *partial,
    float *output,
    int position,
    int heads = kKdaHeads,
    int head_dim = kMlaHeadDim,
    int latent_dim = kMlaLatentDim,
    cudaStream_t stream = nullptr);

// Prefill `tokens` queries [tokens,heads,head_dim] against the latent
// cache, appending the `tokens` latents [tokens,latent_dim] first (causal
// mask inside the chunk mirrors the expanded flash2 path).  Passing both
// expanded_kv and exact_value_cache selects the prefix exact-value path.
cudaError_t mla_prefill_latent(
    const float *query,
    const float *latents,
    const float *expanded_kv,
    uint8_t *cache,
    float *scales,
    float *cache_f32,
    float *exact_value_cache,
    const float *w_uk,
    const float *w_uv,
    float *output,
    int tokens,
    int position_base,
    int heads = kKdaHeads,
    int head_dim = kMlaHeadDim,
    int latent_dim = kMlaLatentDim,
    cudaStream_t stream = nullptr);

cudaError_t mla_prefill_latent_fp8_absorb(
    const float *query,
    const float *latents,
    uint8_t *cache,
    float *scales,
    float *cache_f32,
    const uint8_t *kv_b_fp8,
    const uint16_t *kv_b_scales,
    float *output,
    int tokens,
    int position_base,
    int heads = kKdaHeads,
    int head_dim = kMlaHeadDim,
    int latent_dim = kMlaLatentDim,
    cudaStream_t stream = nullptr);

// Approximate H4 x Q8 long-context prefill counterpart of the cross-head
// decode path. qeff scratch is token-major. One persistent CTA scans the full
// causal prefix for each (eight queries, four heads) tile and projects W_uv.
cudaError_t mla_prefill_latent_cross_head_fp8_absorb(
    const float *query,
    const float *latents,
    uint8_t *cache,
    float *scales,
    const uint8_t *kv_b_fp8,
    const uint16_t *kv_b_scales,
    uint8_t *qeff_fp8,
    float *qeff_scales,
    float *output,
    int tokens,
    int position_base,
    int heads = kKdaHeads,
    int head_dim = kMlaHeadDim,
    int latent_dim = kMlaLatentDim,
    cudaStream_t stream = nullptr);

}  // namespace insignia::glm53
