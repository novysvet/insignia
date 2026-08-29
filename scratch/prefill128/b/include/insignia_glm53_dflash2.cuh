#pragma once

// DFlash2 block-diffusion drafter for GLM-5.3-Flash (z-lab/dflash, ICML'26).
// One 8-position block forward per decode round: [anchor, mask x7] through a
// 5-layer Qwen3-style backbone whose K/V context is a projection of the
// target's layer-{5,14,24,33,42} outputs (mean over the 4 mHC streams, fc,
// RMSNorm, per-layer k/v projections, k_norm + neox RoPE theta 1e4). The
// target's embedding table and lm_head are shared. A greedy host-side
// selector walks top-16 candidates with bilinear codebooks (rank 256).
// Draft quality only affects speed: verification in glm53_generate.cu is
// exact, so committed tokens always match plain greedy decode.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "insignia_glm53_q8_index.hpp"

namespace insignia::glm53 {

class DFlash2Drafter {
public:
    static constexpr int kLayers = 5;
    static constexpr int kHidden = 4096;
    static constexpr int kBlock = 8;    // anchor + 7 mask positions
    static constexpr int kDrafts = 7;
    static constexpr int kQHeads = 32;
    static constexpr int kKVHeads = 8;
    static constexpr int kHeadDim = 128;
    static constexpr int kIntermediate = 12288;
    static constexpr int kMaxTokens = 128;     // capture rows per commit batch (>= chunk cap)
    static constexpr int kMaxCtx = 264;        // 256 target positions + block
    static constexpr int kMaskToken = 154856;
    static constexpr int kTopK = 16;
    static constexpr int kRank = 256;

    // index_path/model_root address the BF16 drafter safetensors (small
    // tensors: norms, conv base kernels, selector codebooks); fp8_prefix is
    // the quantized g64 cache of the big matrices.
    DFlash2Drafter(const std::string &index_path, const std::string &model_root,
                   const std::string &fp8_prefix, int vocab);
    ~DFlash2Drafter();
    DFlash2Drafter(const DFlash2Drafter &) = delete;
    DFlash2Drafter &operator=(const DFlash2Drafter &) = delete;

    // Copy one staged BF16 target-embedding row (4096 elems, device) into the
    // block buffer; call for t=0 (anchor) then t=1..7 (mask rows) before
    // forward().
    void set_block_row(int t, const uint16_t *device_row);

    // Block forward for `anchor` (token id, dumped for the NumPy oracle)
    // whose committed K/V context covers positions 0..anchor_position.
    // Leaves the 7 draft hiddens (mask positions) in device memory; read
    // them with draft_hidden().
    void forward(int anchor, int anchor_position);

    // Append `count` committed tokens' captured target features (the buffer
    // behind capture_row, filled by the Runner at layers 5/14/24/33/42) to
    // the drafter K/V cache at absolute positions pos0..pos0+count-1.
    void commit(int count, int pos0);

    // Selector over downloaded lm_head logits [7 x vocab] (fp32, host) and
    // the drafter hidden projection (device -> host by the caller through
    // hidden_projection_host()). Walks one coherent greedy path.
    std::vector<int> select(const float *logits_host, const float *hp_host, int anchor) const;

    // Device pointers shared with the Runner.
    float *capture_row(int capture_idx, int token);   // [4096] write target
    const float *draft_hidden() const { return hidden_; }       // [7, 4096]
    const float *hidden_projection() const { return hp_dev_; }  // [7, 256]
    size_t logits_span() const { return size_t(kDrafts) * vocab_; }

private:
    struct Fp8Mat {
        uint8_t *w = nullptr;
        uint16_t *s = nullptr;
        int rows = 0, cols = 0;
    };
    void load_fp8(Fp8Mat &into, const Q8Index &index, const std::string &name);

    int vocab_ = 0;
    // FP8 g64 matrices (device).
    Fp8Mat q_[kLayers], k_[kLayers], v_[kLayers], o_[kLayers];
    Fp8Mat gate_[kLayers], up_[kLayers], down_[kLayers];
    Fp8Mat akp_[kLayers], mkp_[kLayers];
    Fp8Mat fc_a_, fc_b_, hp_;
    // FP32 smalls (device).
    float *input_ln_[kLayers] = {}, *post_ln_[kLayers] = {};
    float *q_norm_[kLayers] = {}, *k_norm_[kLayers] = {};
    float *conv_base_[kLayers * 2] = {};   // [side][tap][channel] flattened
    float *hidden_norm_ = nullptr, *final_norm_ = nullptr;
    // Host selector codebooks (BF16 bits, [vocab, 256]).
    std::vector<uint16_t> pred_bits_, succ_bits_;
    // K/V cache: [layer][kMaxCtx][kv heads][head dim], fp32.
    float *kcache_ = nullptr, *vcache_ = nullptr;
    // Workspaces.
    float *x_block_ = nullptr;      // [kBlock, kHidden] residual stream
    float *xn_ = nullptr;           // [kBlock, kHidden] post-LN
    float *branch_ = nullptr;       // [kBlock, kHidden] conv output / sublayer out
    float *sub_ = nullptr;          // [kBlock, kHidden] sublayer result pre-conv-finish
    float *qbuf_ = nullptr;         // [kBlock, q heads * head dim]
    float *kblk_ = nullptr;         // [kBlock, kv heads * head dim]
    float *vblk_ = nullptr;
    float *dyn_ = nullptr;          // [kBlock, 1024] kernel projection
    float *dyn1_ = nullptr;         // [kBlock, 512] saved finish-side dynamics
    float *gbuf_ = nullptr;         // [kBlock, kIntermediate]
    float *ubuf_ = nullptr;
    float *hidden_ = nullptr;       // [kDrafts, kHidden]
    float *fc_in_a_ = nullptr;      // [kMaxTokens, 10240]
    float *fc_in_b_ = nullptr;      // [kMaxTokens, 10240]
    float *fc_out_ = nullptr;       // [kMaxTokens, kHidden]
    float *fc_tmp_ = nullptr;       // [kMaxTokens, kHidden]
    float *ctx_x_ = nullptr;        // [kMaxTokens, kHidden]
    float *ck_ = nullptr;           // [kMaxTokens, kv heads * head dim]
    float *cv_ = nullptr;
    float *capture_ = nullptr;      // [kLayers][kMaxTokens][kHidden]
    float *hp_dev_ = nullptr;       // [kDrafts, kRank]
    void *workspace_ = nullptr;     // fp8 activation quantization scratch
};

}  // namespace insignia::glm53
