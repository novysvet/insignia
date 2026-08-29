// FP8-TC MLA score path — kernel sketch (design artifact, NOT compiled).
// Gate: INSIGNIA_GLM53_MLA_FP8_SCORE=1 (off by default; parity-gated A/B knob).
// Numerics contract: K-side bytes/scales are the EXISTING group-64 E4M3 latent
// cache, untouched. q_eff is computed by the EXACT FP32 fmaf absorb chain used
// today (glm53_ops.cu:1043-1049), then re-quantized E4M3 group-64 (absmax/448).
// Softmax/merge stay FP32 with the same tile-512 online-softmax tree and the
// same sequential tile-merge order as mla_decode_latent_partial/merge.
// NOT bit-exact (score values change) -> measured by logit-cos / flip-rate.

namespace insignia::glm53 {
namespace {

constexpr int kScorePanelHeads = 8;   // n8 of the MMA = 8 heads (latent K shared)
constexpr int kScoreKeyTile = 128;    // 128 keys x 512 B = 64 KiB smem tile
constexpr int kScoreKeysPerWarp = 16; // one m16 per warp

// Step 1: absorb + quantize q for one 8-head panel.
// q_eff[h][512] = q[h][256] @ W_uk[h]  (verbatim fmaf chain), then per-64
// group absmax -> scale (absmax/448), E4M3 bytes. Output layout [8][512] B
// plus [8][8] FP32 scales. Runs once per token per layer: 8x(256x512) fmaf.
__global__ __launch_bounds__(256) void mla_quantize_q_panel_kernel(
    const float *__restrict__ query,        // [64 heads, 256]
    const float *__restrict__ w_uk,         // [64,256,512] per layer
    uint8_t *__restrict__ q_fp8,            // [8 panels, 8 heads, 512]
    float *__restrict__ q_scales,           // [8 panels, 8 heads, 8 groups]
    int panel) {
    // 256 threads; thread owns dims (element, element+256) of head
    // (panel*8 + warp) — identical ownership as the current absorb so the
    // fmaf sequence matches glm53_ops.cu:1043-1049 expression-for-expression.
    ...
}

// Step 2: score + online softmax per (512-key tile, 8-head panel).
// A = K latent [keys x 512] E4M3 (cache rows reinterpreted per 16-key x
// 32-dim fragments, exactly how fp8_tc_gemv reads weight rows). B = q panel
// [512 x 8] col-major. 8 groups x 2 mma (k32) per m16n8 tile; group scales
// applied between groups: score += d * (q_s[row,g] * k_s[col,g]).
__global__ __launch_bounds__(256) void mla_score_fp8_partial_kernel(
    const uint8_t *__restrict__ cache,      // [pos,512] E4M3 (per layer)
    const float *__restrict__ scales,       // [pos,8] FP32 (per layer)
    const uint8_t *__restrict__ q_fp8,      // from step 1
    const float *__restrict__ q_scales,
    float *__restrict__ partial,            // [64 heads, tiles, 514] (same as today)
    int position,
    int tiles) {
    // blockIdx.x = 512-key tile t (kMlaDecodeTile kept for the merge tree),
    // blockIdx.y = head panel p (0..7). 256 threads = 8 warps; warp w owns
    // keys [t*512 + sub*128 + w*16, +16) of the 128-key smem sub-tile.
    __shared__ __align__(16) uint8_t ktile[kScoreKeyTile][512];
    __shared__ float ktile_scale[kScoreKeyTile][kMlaLatentGroups];
    // Online state per head of the panel (8 x {max, den, acc[512]}): acc in
    // registers per thread for dims (element, element+256) of each head —
    // same per-thread latent-dim ownership as the current kernel.
    ...
    for (int sub = 0; sub < 4; ++sub) {          // 4 x 128-key sub-tiles
        load_tile(ktile, ktile_scale, cache, scales);   // 64 KiB + 4 KiB
        __syncthreads();
        // per warp: m16(16 keys) x n8(8 heads) x k512
        float d0, d1, d2, d3, score[16][8 / 2 /*per lane 2 cols*/];
        for (int g = 0; g < 8; ++g) {
            float dg0 = 0, dg1 = 0, dg2 = 0, dg3 = 0;
            mma_e4m3(dg0, dg1, dg2, dg3, a0, a1, a2, a3, b0, b1);  // k = g*64 .. +32
            mma_e4m3(dg0, dg1, dg2, dg3, a0, a1, a2, a3, b0, b1);  // k = +32 .. +64
            // group epilogue: 4 accumulators x fmaf(d, q_s[row,g]*k_s[col,g], s)
            // q_s[row,g], k_s[col,g] from smem (2 loads + 4 mul + 4 fma per lane)
            ...
        }
        // stage the 16x8 raw scores to smem [16][8], then per head run the
        // EXACT per-key update sequence in ascending key order (the decode
        // tree, glm53_ops.cu:1080-1086):
        //   maximum_new = fmaxf(maximum, score);
        //   correction  = expf(maximum - maximum_new);
        //   weight      = expf(score - maximum_new);
        //   denominator = denominator * correction + weight;
        //   acc         = fmaf(weight, k_dim, acc * correction);  // v2 PV-TC variant
        ...
        __syncthreads();
    }
    // Write [max, den, acc0..acc1] per head exactly like
    // mla_decode_latent_partial_kernel:1089-1095; mla_decode_latent_merge_kernel
    // runs UNCHANGED on top (same 514-float partial layout).
}

// Step 3 (v2, the real speedup): PV on tensor cores.
//   out[8 heads x 512] += w'[8 x keys] @ V[keys x 512]  (mma m16n8k32)
// where w'[h, c] = softmax weight x k_scale[c, n_group], re-quantized E4M3
// per 64-key group (per n-group; V IS the latent cache, already E4M3 g64).
// NOTE: this reassociates the value-side accumulation tree (P5(c)) — the
// 6.67e-6-RMS / 78%-error-energy term measured in audits/mla-latent-session.md.
// Gate separately (INSIGNIA_GLM53_MLA_FP8_PV) and never mix into the v1 A/B.

// Launcher mirrors mla_decode_latent (glm53_ops.cu:1264-1309):
//   grid dim3(tiles, 8 panels) per layer; partial layout unchanged so the
//   merge kernel and its sequential tile fold are reused verbatim.

}  // namespace
}  // namespace insignia::glm53
