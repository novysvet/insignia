#pragma once
// Qwen3.8-27B-FP8 tiered engine: shapes, storage views, decode API.
// Weights stay F8_E4M3 + bf16 128x128 block scales (never re-quantized). Tiers:
//   V = VRAM slab (device), Z = pinned host (UVA zero-copy reads), N = NVMe ring slot
//       (pinned slot, UVA zero-copy reads), plus ALWAYS-VRAM arenas for the tiny
//       tensors (all .scales, norms widened to f32 with +1 baked per HF zero-center
//       convention, A_log widened bf16->f32, conv/dt/a/b raw bf16) and lm_head.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <string>
#include <vector>

#include "insignia_model.hpp"
#include "insignia_streaming.hpp"

namespace insignia {

extern bool g_dump_stage27;   // test hook: layer-0 intermediate dumps

struct Q38Shape {
    static constexpr int hidden = 5120, inter = 17408, layers = 64, vocab = 248320;
    static constexpr int q_heads = 24, kv_heads = 4, head_dim = 256;
    static constexpr int dv = 48, dk = 16;          // DeltaNet v / k heads
    static constexpr int full_attn_layers = 16;     // (i & 3) == 3
    static constexpr bool full_attention(int i) { return (i & 3) == 3; }
    static constexpr int lm_rows = 248320;
};

enum class Tier : uint8_t { V = 0, Z = 1, N = 2, C = 3 };

struct Fp8View {                    // one quantized linear: body + VRAM-resident scales
    const uint8_t *w{};             // tier memory (device / pinned / ring slot)
    const uint16_t *s{};            // ALWAYS device VRAM
    int rows{}, cols{};
};

struct LayerView27 {
    Tier tier{Tier::N};
    Fp8View qkv, z, out;            // linear-attention layers
    Fp8View q, k, v, o;             // full-attention layers
    Fp8View gate, up, down;         // both
    const float *in_norm{}, *post_norm{}, *q_norm{}, *k_norm{};              // f32 (+1 baked), device
    const uint16_t *la_norm_bf16{}, *conv{}, *dt{};                          // bf16 arenas
    const void *ab[2]{};                                               // bf16 a/b [48,hidden] weight pairs (cast to u32 view)
    const float *a_log{};           // f32 widened
};

// ---------------------------------------------------------------------------
class TieredStorage27 final {
public:
    TieredStorage27(const wchar_t *index_path, const wchar_t *manifest_path, cudaStream_t stream);
    ~TieredStorage27();
    TieredStorage27(const TieredStorage27 &) = delete;
    TieredStorage27 &operator=(const TieredStorage27 &) = delete;

    const LayerView27 &layer(int l) const noexcept { return views_[l]; }
    const uint32_t *lm_head() const noexcept { return lm_head_; }        // bf16 [248320,5120] as u32, device
    // embed rows are pread from NVMe on demand into a pinned host staging area (UVA-read by kernels)
    const uint16_t *embed_row(int token);                                 // single row
    const uint16_t *embed_rows(const int *tokens, int T);                 // T rows (prefill)
    const float *final_norm() const noexcept { return final_norm_; }     // f32 (+1 baked), device

    // N-tier streaming discipline: call begin_epoch() at each token step, then
    // acquire(l)/release(l) strictly in ascending N-layer order.
    void begin_epoch();
    const void *acquire(int l);          // blocks until the layer's slot is READY; fills N-tier pointers (UVA zero-copy: plain-load kernels only)
    const void *acquire_staged(int l);   // same, then copies the F8 bodies into a VRAM staging slab and repoints the
                                         // views there (fp8_gemm's cp.async cannot cross PCIe; prefill path)
    void release(int l);                 // after the GPU work reading the slot has completed (caller enforces via events)
    bool healthy() const noexcept { return feeder_.healthy(); }
    Tier tier_of(int l) const noexcept { return tier_[l]; }

private:
    void load_smalls_and_scales();
    void load_v_layers();
    void build_n_plans();
    bool   misaligned_shard(int l) const;
    int    count_tier(Tier t) const;
    double free_vram_mb() const;
    void  *dev_tensor_chunked(const TensorView &t);
    ModelFile model_;
    cudaStream_t stream_;
    LayerFeeder feeder_{4ull * 368 * 1024 * 1024, 4, 2, LayerFeeder::ConsumeMode::zero_copy};
    Tier tier_[Q38Shape::layers];
    std::vector<ReadPlan> n_plans_;                    // [n_count] one plan per N layer (7 F8 requests each)
    std::vector<int> n_layer_ids_;
    std::vector<std::wstring> n_paths_;                // keeps ReadRequest::path alive (reserved upfront)
    // arenas (device)
    uint32_t *lm_head_{};                 // [vocab*hidden/2] u32
    float   *norm_arena_{};               // f32, +1 baked: in/post per layer (2*hidden) + q/k norms + final
    uint16_t *bf16_arena_{};              // la_norm, conv, dt, a/b per layer
    float   *alog_arena_{};               // [64*48] f32
    uint16_t *scales_arena_{};            // all .scales concatenated (device)
    uint16_t *embed_{};                   // host pinned staging rows [64][hidden] (bf16) — pread target
    const float *final_norm_{};
    void    *embed_file_{};               // HANDLE to outside.safetensors (buffered)
    uint64_t embed_off_{};
    // V/Z layer slabs
    std::vector<void *> slabs_;           // device allocs (V) — Z tier unused in v1 manifests
    LayerView27 views_[Q38Shape::layers];
    // host staging for startup loads
    void *bounce_{};                      // pinned 64 MiB
    size_t bounce_bytes_{};
    uint8_t *n_stage_{};                  // VRAM staging slab for N-tier prefill (fp8_gemm needs device weights)
    size_t n_stage_bytes_{};
    // per-N-layer tensor offsets inside the slot (filled at acquire)
    struct NTensor { uint16_t shard; uint64_t off; uint64_t bytes; int view_slot; };
    std::vector<std::vector<NTensor>> n_tensors_;      // [n_count][tensor]
};

// ---------------------------------------------------------------------------
struct Workspace27 {
    float *hidden{}, *norm{}, *qkv{}, *z{}, *a{}, *b{}, *core{}, *gate{}, *up{}, *down{}, *logits{};
    float *key{}, *value{};                            // full-attn k/v [1024]
    float *delta_state{}, *conv_state{}, *kv_keys{}, *kv_values{};   // f32 KV (v1 parity config)
    // prefill (T<=64)
    float *pf_x{}, *pf_n{}, *pf_qkv{}, *pf_z{}, *pf_a{}, *pf_b{}, *pf_scratch{}, *pf_gate{}, *pf_up{}, *pf_down{}, *pf_core{}, *pf_q{}, *pf_g{}, *pf_k{}, *pf_v{};
    __nv_bfloat16 *pf_bf16{};           // [64][inter] staging (largest cols user)
    int *pf_tokens{}, *pos_dev{}, *token_dev{}, *next_dev{};
    int *next_host{}, *pos_host{};
    unsigned long long *am_scratch{};
    int max_context{}, position{};
    cudaStream_t stream{};
    explicit Workspace27(int max_context);
    ~Workspace27();
    Workspace27(const Workspace27 &) = delete;
    Workspace27 &operator=(const Workspace27 &) = delete;
};

// ---------------------------------------------------------------------------
class Qwen38Decode final {
public:
    Qwen38Decode(TieredStorage27 &st, Workspace27 &x) : st_(st), x_(x) {}
    void forward_token(int token);      // processes token at x.position, then bumps
    int decode_token(int token);        // forward_token + argmax
    int logits_argmax();
    int prefill_chunk(const int *tokens, int T);                   // T<=64, returns argmax after last
    void prefill_chunk_seam(const int *tokens, int T, void (*seam)(int layer, const float *pf_x, int T, void *user), void *user);
    void set_position(int pos);
private:
    void linear(const Fp8View &m, const float *in, float *out);            // T=1 decode GEMV
    void linear_batch(const Fp8View &m, const float *in, float *out, int T); // T<=64 via fp8_gemm
    void layer_gpu(int l);                                                  // V/Z/N unified (N pointers pre-filled at acquire)
    void delta_layer(int l);
    void attention_layer(int l);
    void forward_body();
    TieredStorage27 &st_;
    Workspace27 &x_;
};

}  // namespace insignia
