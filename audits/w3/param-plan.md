# W3 plan: shape parameterization for 9B + 27B (204-site census resolution)

Input: AGENTS.md, audits/synthesis.md, audits/w2/shape-constants.md (204 sites: 148 engine,
56 instrumentation). All source claims below re-verified against src/ on 2026-08-25.
Verified 27B dims against `Qwen3.8-27B-FP8/config.json` (text_config): hidden 5120,
inter 17408, 64 layers, interval 4, head_dim 256, 24 q / 4 kv heads, linear 48 v / 16 k
heads, 128/128 dims, conv 4, rope_theta 1e7, partial 64/256, mtp 1 layer, vocab 248320.

---

## 0. Decision: option (c) — runtime `Shape` POD + runtime grids. No templates. Verified.

**The per-element census first** (this is the whole argument):

| Class | Count | Sites | Runtime-dim cost |
|---|---|---|---|
| workspace-alloc | 43 | decode.cu ctor allocs/memsets | zero (host code) |
| per-element | **3** | prefill.cu:172 `idx/8192, idx%8192` (conv — a real div+mod per thread, and 10240 is **not** a power of 2); prefill.cu:12-13/29-30 embed `row*512`, `row*128`, `row*64` (multiply-only, identical cost as runtime) | conv: fixed **by grid restructure, not templates** — `dim3(ceil(c/256), T)` makes `t=blockIdx.y, c=...` and deletes the div/mod for BOTH models (also faster at 9B). embed: free. |
| per-block | ~12 | `kh=head>>1` (deltanet.cu:5, prefill.cu:221), `kvh=head>>2` (attention.cu:7, prefill.cu:103), `isq=head<16` (ops.cu:9, prefill.cu:57), split decodes `>>4,&15` / `i>>8` (prefill.cu:44, qwen_kernels.cu:73,78), guards `h>=32`,`i<4096`,`i<1024`,`c>=8192` | one integer div (or one compare) per CTA, amortized over 128-256 threads × inner loops. Free. |
| per-launch | ~90 | all grids (`<<<16,256>>>`, `<<<32,128>>>`, `dim3(20,T)`…), host strides, memset/copy sizes, loop bounds `l<32`, GEMM/FP8 divisibility gates, graph capture markers | zero (host code / launch config) |

**Register-allocation audit (the "does any kernel need compile-time heads" question): NO.**
No kernel declares any register or smem array dimensioned by q_heads / v_heads / layers /
hidden. The only shape-sized shared arrays are `score[4096]` (context cap, identical in both
models), `qs[256]`/`part[8][256]` (head_dim 256, identical), `delta[128]`, `mem[64]`
(rope_dim 64, identical), and DeltaNet's 64 KB dynamic state (128×128, identical). Head
counts only ever appear as `blockIdx.x` extent or a per-block index divide. No shuffle or
warp-semantic depends on head counts being powers of two (the only pow-2 tricks are the
`>>4/&15` and `>>2/>>1` index decodes listed above — all per-block).

The one near-miss is the fused `ab2` pair kernel family (mxfp4_i4.cu:157-242,
mxfp4.cu:519-679): staging bakes hidden=4096 (128 groups, `threadIdx.x>>7/&127`) and the
compute loop's `rr<32` / 8-warp×8-row=64-row capacity bakes 32+32 a/b rows. But its
row-scheduled accumulators are unrolled-loop scalars, not register arrays, and the binding
constraint is **capacity** (96 rows needed at 27B), i.e. a redesign — not a
parameterization detail. Crucially `mxfp4_gemv_ab2_q8_i4` already throws on `cols!=4096`
(mxfp4_i4.cu:239) and **27B is FP8**: it uses fp8.cu (`fp8_gemv/gemv2/gemm` — already
fully runtime-rows/cols, `test_fp8.cu` already exercises 10240×5120). So ab2 stays a
9B-INSIG4 specialist (AGENTS spirit: specialization is fine); if a future INSIG4-5120
requant wants it back, THAT is the single place where `template<int VH>` (32 vs 48, for
unroll depth) would be worth considering — listed as contingency, not plan.

Therefore:
- **(a) constexpr Shape template param**: doubles nvcc time across ~10 engine TUs and
  doubles SASS for zero measured benefit — the census shows no per-element consumer.
  Rejected (revisit only with a profile showing per-block div cost, and then only for the
  one kernel).
- **(b) loose runtime ints everywhere**: 43-buffer ctor + 10 dims per call site = a bug
  farm (the audit's own trap list: `attn_gate/z/core/pf_q/pf_g 4096->6144 not 5120`).
  Rejected as the primary mechanism.
- **(c) one runtime `Shape` POD as single source of truth + kernels take plain int params
  (rows/cols already do) with grids computed in launchers**: chosen. Least invasive:
  signature churn confined to the workspace ctor + launcher wrappers; `Qwen35Decode`
  stays non-template; CUDA graphs freeze per-shape at capture time exactly as today.

---

## 1. The `Shape` struct (new header `include/insignia_shape.hpp`)

```cpp
#pragma once
#include <cstdint>
#include <stdexcept>
namespace insignia {

struct Shape {
    // ---- primary dims (all inferred at load; see §5) ----
    int hidden{}, vocab{}, layers{}, intermediate{};
    int q_heads{}, kv_heads{}, head_dim{}, rope_dim{};      // full-attention GQA
    int interval{4};                                        // full_attention cadence
    int delta_vheads{}, delta_kheads{}, delta_kdim{}, delta_vdim{};  // gated DeltaNet
    int conv_width{4};                                      // causal conv taps (state = w-1 = 3)
    int mtp_layers{};                                       // 1 in both known models

    // ---- derived: every known model satisfies these identities; validate() enforces ----
    int full_layers()  const { return layers / interval; }                    // 8 | 16
    int delta_layers() const { return layers - layers / interval; }          // 24 | 48
    int qkv_rows()     const { return delta_kdim*(2*delta_kheads+delta_vheads); } // 8192 | 10240
    int z_rows()       const { return delta_vheads * delta_vdim; }           // 4096 | 6144
    int q_proj_rows()  const { return q_heads * 2 * head_dim; }              // 8192 | 12288 (q+gate interleave)
    int attn_out()     const { return q_heads * head_dim; }                  // 4096 | 6144
    int core_rows()    const { return z_rows() > attn_out() ? z_rows() : attn_out(); } // 4096 | 6144
    int kv_row()       const { return kv_heads * head_dim; }                 // 1024 | 1024
    int gqa_group()    const { return q_heads / kv_heads; }                  // 4 | 6
    int k_share()      const { return delta_vheads / delta_kheads; }         // 2 | 3
    int k_off()        const { return delta_kheads * delta_kdim; }           // 2048 | 2048 (q|k|v offsets)
    int v_off()        const { return 2*delta_kheads * delta_kdim; }         // 4096 | 4096
    int mtp_fc_cols()  const { return 2 * hidden; }                          // 8192 | 10240 ([hidden, 2*hidden])
    size_t delta_state_per_layer() const { return size_t(delta_vheads)*delta_kdim*delta_vdim; } // 512Ki | 768Ki floats

    // ---- layer-index helpers (formulas identical in both models; only ranges differ) ----
    static constexpr bool full_attention(int i) { return (i & 3) == 3; }  // requires interval==4, asserted
    static constexpr int  delta_index(int i)    { return i - i / 4; }     // valid ONLY when !full_attention(i)
    static constexpr int  attn_index(int i)     { return i / 4; }
    int kvh_of(int qh) const { return qh / gqa_group(); }   // qh>>2 (9B) | qh/6 (27B)
    int kh_of(int vh)  const { return vh / k_share(); }     // vh>>1 (9B) | vh/3 (27B)

    static Shape Q9B();                                     // presets for asserts/tests
    static Shape Q27B();
    static Shape from_model(const class ModelFile &m);      // §5
    void validate() const;                                  // throws with a named dimension on any violation
};

}
```

### Field values

| Field | 9B (Qwen3.5-9B / INSIG4) | 27B (Qwen3.8-27B-FP8) |
|---|---|---|
| hidden | 4096 | 5120 |
| vocab | 248320 | 248320 |
| layers | 32 | 64 |
| intermediate | 12288 | 17408 |
| q_heads | 16 | 24 (non-pow2) |
| kv_heads | 4 | 4 |
| head_dim / rope_dim | 256 / 64 | 256 / 64 |
| interval / conv_width | 4 / 4 | 4 / 4 |
| delta_vheads | 32 | 48 |
| delta_kheads | 16 | 16 (q=k assumption, see §5) |
| delta_kdim / delta_vdim | 128 / 128 | 128 / 128 |
| mtp_layers | 1 | 1 |
| derived: full/delta layers | 8 / 24 | 16 / 48 |
| qkv_rows / z_rows | 8192 / 4096 | 10240 / 6144 |
| q_proj_rows / attn_out | 8192 / 4096 | 12288 / 6144 |
| core_rows / kv_row | 4096 / 1024 | 6144 / 1024 |
| gqa_group / k_share | 4 / 2 | 6 / 3 |
| k_off / v_off | 2048 / 4096 | 2048 / 4096 (stays!) |
| mtp_fc_cols | 8192 | 10240 |

### `delta_index` derivation and verification

Pattern: full attention at `(i&3)==3`, i.e. layers 0,1,2 delta; 3 full; 4,5,6 delta; 7
full; … A delta layer i has exactly `i/4` full-attention layers strictly below it, so
**`di = i - i/4`** — which is what the engine already uses (decode.cu:74 `di=l-l/4`,
decode.cu:124 same). Verification, both models (identical layout rule, only the range
differs):

| i | type | di = i−i/4 | note |
|---|---|---|---|
| 0 | delta | **0** | first conv/delta state slot |
| 1 | delta | 1 | |
| 2 | delta | 2 | |
| 4 | delta | 4−1 = **3** | layers 0,1,2 took slots 0..2 |
| 5 | delta | 4 | |
| 6 | delta | 5 | |
| 3 | FULL | n/a — `attn_index(3)=0` | formula never applied to full layers |
| 7 | FULL | n/a — `attn_index(7)=1` | |
| last delta | 9B i=30→**23**; 27B i=62→**47** | 24 / 48 slots total ✓ |

The candidate `di = i - i/4 - 1` suggested in the task brief is **wrong**: it returns −1
at i=0 (layer 0's conv state would wrap to the last layer's slot) and shifts every state
slot by one. Keep the engine's existing expression; do not "fix" it to the −1 variant.

`kvh_of` expression pick: **`qh / (q_heads/kv_heads)`** with the group precomputed
(`gqa_group()`), passed to kernels as one int. The literal form `qh/q_heads*kv_heads` is
a trap — C++ evaluates `(qh/q_heads)*kv_heads == 0` for every qh < q_heads. `kh_of`:
**`vh / (delta_vheads/delta_kheads)`** with `k_share()` precomputed (2 / 3), one int
kernel param; replaces `head>>1`.

### `validate()` (fail loud, not silent — synthesis bug #5)

Throw with the dimension named if any of: `interval==4` (the helpers assume it);
`q_heads%kv_heads==0`; `delta_vheads%delta_kheads==0`; `qkv_rows()%128==0`;
`intermediate >= q_proj_rows()` (gate/up also stage q_proj output, 9B 8192≤12288,
27B 12288≤17408); `intermediate%1024==0 && hidden%1024==0` (i4/MLX fast-GEMV gates,
mxfp4.cu:148 / mxfp4_i4.cu:70); `intermediate%128==0` (fp8 scale blocks); `vocab==embed
rows`; `mtp_layers==1`; `mtp_fc_cols()==2*hidden`; `ctx<=4096` (caller-supplied, gqa
`score[4096]` smem cap). Additionally: replace the silent early-returns in
mxfp4_i4.cu:66,147 and gemm.cu:351 with throws during this refactor (same bug class).

---

## 2. Where the Shape instance lives

**One runtime `Shape` member in `DecodeWorkspace` (the alloc owner), copied from a
`Shape` member in `Qwen35Weights` (the load owner).** `Qwen35Decode` reads it through
`x_.shape` (it already holds `DecodeWorkspace& x_`); no template on `Qwen35Decode<S>`.

- `Qwen35Weights` gains `Shape shape_` set in its ctor from `Shape::from_model(m)` and
  exposed as `const Shape& shape() const`. This fixes the today-dead
  `Qwen35Shape::layers/intermediate` drift noted in the audit (qwen35.hpp:7 —
  `layers`/`intermediate` are currently declared but never used; `hidden/vocab` are used
  by qwen35.cu only). `Qwen35Shape` shrinks to nothing; qwen35.cu switches to the
  instance.
- `DecodeWorkspace` gains `Shape shape;` and its ctor takes it (§3). Entry points
  construct in dependency order — shape inference needs only the `ModelFile`, which they
  already open first:

```cpp
insignia::ModelFile m(argv[1]);
insignia::Shape sh = insignia::Shape::from_model(m);   // throws on unknown geometry
insignia::DecodeWorkspace x(sh, 4096);                 // ctx cap, §4
insignia::Qwen35Weights w(m, 6ull<<30, x.stream);      // re-derives, asserts sh == w.shape()
insignia::Qwen35Decode d(w, x);
```

- Kernel launchers gain int params (`q_heads`, `group`, `v_heads`, `k_share`, `groups`,
  `channels`, …) exactly like the existing `rows/cols` convention; the per-launch sites
  in decode.cu/prefill.cu pass `x_.shape.*`. Kernels keep taking loose ints (no struct
  param needed — none uses more than 3 dims; passing the whole POD would only cost
  param-register pressure in `__launch_bounds__`-tight kernels like gqa_decode).
- CUDA graphs: `capture_step`/`capture_spec` (decode.cu:234-260) already freeze launches
  at capture; per-shape capture happens once per model load. No change beyond re-capture
  if a second model were hot-swapped in one process (not a goal).

Template `<int QH>` specialization: **verified unnecessary** (§0). Contingency only for a
future INSIG4 ab2 revival.

---

## 3. `DecodeWorkspace` ctor — the 43-site alloc block rewritten

Replaces decode.cu:11-28. (`sh` = the member `shape`, `ctx` = max_context.)

```cpp
DecodeWorkspace::DecodeWorkspace(const Shape &sh, int ctx, cudaStream_t s)
    : shape(sh), max_context(ctx), stream(s) {
    sh.validate();
    if (ctx <= 0 || ctx > 4096) throw std::runtime_error("max_context must be in 1..4096 (gqa score buffer)");
    const int H = sh.hidden, INT = sh.intermediate;
    const int QKV = sh.qkv_rows(), Z = sh.z_rows(), AO = sh.attn_out(), CR = sh.core_rows();
    const int VH = sh.delta_vheads, KVR = sh.kv_row();
    const size_t DSP = sh.delta_state_per_layer(), CCP = size_t(sh.qkv_rows()) * (sh.conv_width - 1);
    if (!stream) { auto e = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking); if (e) throw std::runtime_error(cudaGetErrorString(e)); }
    alloc(&hidden, H); alloc(&norm, H); alloc(&qkv, QKV); alloc(&attn_gate, AO); alloc(&key, KVR); alloc(&value, KVR);
    alloc(&z, Z); alloc(&a, VH); alloc(&b, VH); alloc(&core, CR); alloc(&gate, INT); alloc(&up, INT); alloc(&down, H);
    alloc(&logits, size_t(sh.vocab) * 2);
    alloc(&delta_state, sh.delta_layers() * DSP);        // 24*512Ki | 48*768Ki floats
    alloc(&conv_state, sh.delta_layers() * CCP);         // 24*8192*3 | 48*10240*3
    alloc(&kv_keys,   size_t(sh.full_layers()) * ctx * KVR);   // 8 | 16 layers x ctx x 1024
    alloc(&kv_values, size_t(sh.full_layers()) * ctx * KVR);
    alloc(&mtp_keys,   size_t(ctx) * KVR);               // 1 mtp layer (validated)
    alloc(&mtp_values, size_t(ctx) * KVR);
    auto e = cudaMalloc(&pos_dev, 16 * sizeof(int)); if (e) throw std::runtime_error(cudaGetErrorString(e));
    token_dev = pos_dev + 1; next_dev = pos_dev + 2; next2_dev = pos_dev + 3; draft_dev = pos_dev + 4;
    count_dev = pos_dev + 5; accflag_dev = pos_dev + 6; mtp_pos_dev = pos_dev + 7;
    if (cudaHostAlloc(&next_host, sizeof(int), cudaHostAllocDefault)) throw std::runtime_error("pinned alloc failed");
    if (cudaHostAlloc(&pos_host, sizeof(int), cudaHostAllocDefault)) throw std::runtime_error("pinned alloc failed");
    if (cudaHostAlloc(&host_committed, 16384 * sizeof(int), cudaHostAllocDefault)) throw std::runtime_error("pinned alloc failed");
    e = cudaMalloc(&am_scratch, 8); if (e) throw std::runtime_error(cudaGetErrorString(e));
    e = cudaMalloc(&committed, 16384 * sizeof(int)); if (e) throw std::runtime_error(cudaGetErrorString(e));
    e = cudaMalloc(&pf_tokens, 64 * sizeof(int)); if (e) throw std::runtime_error(cudaGetErrorString(e));
    // prefill chunk buffers (T = 64 rows)
    const int PFSCR = QKV > sh.q_proj_rows() ? QKV : sh.q_proj_rows();   // 8192 | 12288 (!)
    alloc(&pf_x, 64*H); alloc(&pf_n, 64*H); alloc(&pf_qkv, 64*QKV); alloc(&pf_scratch, 64*PFSCR); alloc(&pf_z, 64*Z);
    alloc(&pf_q, 64*AO); alloc(&pf_g, 64*AO); alloc(&pf_k, 64*KVR); alloc(&pf_v, 64*KVR); alloc(&pf_core, 64*CR);
    alloc(&pf_down, 64*H); alloc(&pf_gate, 64*INT); alloc(&pf_up, 64*INT); alloc(&pf_a, 64*VH); alloc(&pf_b, 64*VH);
    alloc(&snap_delta, sh.delta_layers() * DSP);         // == delta_state size
    alloc(&snap_conv,  sh.delta_layers() * CCP);         // == conv_state size
    {   // pair staging: 2 rows x (intermediate/32) dp4a groups x 8 u32 + scales; bf16 GEMM A-scratch
        const int G = INT >> 5;                          // 384 | 544
        auto e2 = cudaMalloc(&pf_xq8, size_t(2) * G * 8 * sizeof(unsigned)); if (e2) throw std::runtime_error(cudaGetErrorString(e2));
        e2 = cudaMalloc(&pf_xs8, size_t(2) * G * sizeof(float)); if (e2) throw std::runtime_error(cudaGetErrorString(e2));
        e2 = cudaMalloc(&pf_bf16, size_t(64) * INT * 2); if (e2) throw std::runtime_error(cudaGetErrorString(e2));
    }
    cudaMemsetAsync(pos_dev, 0, 16 * sizeof(int), stream);
    cudaMemsetAsync(am_scratch, 0, 8, stream);
    cudaMemsetAsync(delta_state, 0, sh.delta_layers() * DSP * 4, stream);
    cudaMemsetAsync(conv_state, 0, sh.delta_layers() * CCP * 4, stream);
}
```

Trap sheet honored (audit cluster 3): `attn_gate/z/core/pf_q/pf_g = attn_out()/z_rows()`
(6144 at 27B, **not** 5120); `a/b/pf_a/pf_b = delta_vheads` (48); `delta_state/conv_state/
snap = delta_layers × per-layer` (48×768Ki / 48×10240×3); `kv = full_layers (16) × ctx ×
1024`; `pf_scratch = max(qkv_rows, q_proj_rows)` = 12288 (not 10240); `pf_bf16/pf_gate/
pf_up = intermediate` (17408). At 9B every expression evaluates to today's literal —
phase 0 is byte-for-byte identical allocation. Total at 27B/ctx 4096 ≈ 1.73 GiB (vs 9B
≈ 0.95 GiB), see §4.

---

## 4. Context cap for 27B

`score[4096]` static smem in attention.cu:7 / prefill.cu:107 is a **context** cap
(numeric coincidence with 9B hidden), sized for graph replay. Recommendation: **keep
default ctx = 4096 for both models**; the ctor keeps the `1..4096` guard and
generate.cu's 4090 ceiling stays.

Memory math at 27B, ctx 4096: delta_state 576 MiB + snap_delta 576 MiB + KV 2×16×4096×
1024×4 = 512 MiB + conv/snap ≈ 12 MiB + mtp KV 32 MiB + misc (logits 2 MiB, pf_* ≈ 25
MiB) ≈ **1.73 GiB** workspace. With the 6 GiB weight budget + ~0.6 GiB CUDA context on a
12 GiB board there is headroom. If the 27B weight budget later grows (lm_head 2.5 GiB
must stay VRAM-resident), ctx 2048 halves KV to 256 MiB — one ctor arg, zero kernel
changes (the smem array is an upper bound, not ctx-sized). Do NOT convert `score[]` to
dynamic smem sized by ctx: static 16 KB keeps `__launch_bounds__(256,2)` occupancy and
simplifies graph capture for zero benefit.

(Future, out of scope: snap_delta/snap_conv cost another 588 MiB at 27B — snapshotting
only the pending step's layer or fusing accept/rollback would reclaim most of it.)

---

## 5. How the engine picks the Shape at load

**Recommendation: infer from tensor shapes at load (`Shape::from_model(const ModelFile&)`),
not index metadata.** Rationale:

- The index (INSIDX01, tools/index_safetensors.py, src/model_file.cpp) has **no metadata
  field**; adding one means a version bump to 02, changes in the Python writer, the C++
  parser, and re-indexing both models — and creates a second source of truth that can
  disagree with the weights it describes. Tensor shapes cannot lie or go stale.
- Every dim is uniquely derivable from names+shapes already in the index, for both
  quant formats, with **one** documented assumption (delta q=k heads, true for both
  Qwen3.5-9B and Qwen3.8-27B and cross-checkable):

```cpp
Shape Shape::from_model(const ModelFile &m) {
    Shape s;
    auto need = [&](const char *n) { const TensorView *t = m.find(n); if (!t) throw std::runtime_error(std::string("shape inference: missing ") + n); return t; };
    const TensorView *emb = need("language_model.model.embed_tokens.weight");
    s.vocab = int(emb->shape[0]); s.hidden = int(emb->shape[1]);           // [248320, 4096|5120], any dtype
    for (s.layers = 0; ; s.layers++) {                                     // count layer blocks by name
        char n[96]; snprintf(n, sizeof n, "language_model.model.layers.%d.input_layernorm.weight", s.layers);
        if (!m.find(n)) break;
    }
    s.intermediate = int(need("language_model.model.layers.0.mlp.gate_proj.weight")->shape[0]);
    s.head_dim = 256; s.rope_dim = 64;                                     // anchors, asserted below
    s.q_heads  = int(need("language_model.model.layers.3.self_attn.q_proj.weight")->shape[0]) / (2*s.head_dim); // 512 per head (q+gate)
    s.kv_heads = int(need("language_model.model.layers.3.self_attn.k_proj.weight")->shape[0]) / s.head_dim;     // 1024/256 = 4
    s.delta_vheads = int(need("language_model.model.layers.0.linear_attn.in_proj_a.weight")->shape[0]);         // 32 | 48
    const int qkv_rows = int(need("language_model.model.layers.0.linear_attn.in_proj_qkv.weight")->shape[0]);
    s.delta_kdim = s.delta_vdim = 128;                                     // anchors
    // in_proj_qkv = 128*(dq + dk + dv); both known models have dq == dk (16/16):
    const int dqk = qkv_rows / 128 - s.delta_vheads;
    if (dqk <= 0 || dqk & 1) throw std::runtime_error("shape inference: in_proj_qkv split not q==k");
    s.delta_kheads = dqk / 2;                                              // 16 | 16
    s.conv_width = int(need("language_model.model.layers.0.linear_attn.conv1d.weight")->shape[2]);              // [channels,1,4]
    s.mtp_layers = m.find("language_model.mtp.fc.weight") ? 1 : 0;
    // pattern check: names give the true layer types; must equal the (i&3)==3 helper
    for (int i = 0; i < s.layers; i++) {
        char q[96]; snprintf(q, sizeof q, "language_model.model.layers.%d.self_attn.q_proj.weight", i);
        if (bool(m.find(q)) != full_attention(i)) throw std::runtime_error("layer pattern != (i&3)==3; Shape helpers assume interval 4");
    }
    s.validate();
    return s;
}
```

`validate()` additionally cross-checks the anchors that can be read from shapes
(`conv1d.shape[0]==qkv_rows()`, `mtp.fc == [hidden, 2*hidden]`, `head_dim==256` via
`k_proj rows % 256 == 0`, `q_proj rows % 512 == 0`) and the divisibility gates of §1.
Unknown-but-self-consistent future geometries load; anything that violates a kernel
assumption throws at load with the dimension named instead of corrupting silently.

For the record, the rejected alternative — index metadata — would extend INSIDX as:
`MAGIC "INSIDX02"`, then `uint32 meta_len` + JSON blob
`{"hidden_size":…,"intermediate_size":…,"num_hidden_layers":…,"num_attention_heads":…,
"num_key_value_heads":…,"head_dim":256,"linear_num_value_heads":…,"linear_num_key_heads":…,
"full_attention_interval":4,"vocab_size":…,"mtp_num_hidden_layers":1}` copied from
config.json, then tensor entries as today. Only worth doing later as a cross-check;
inference is the recommended single source.

Note for the 27B load path: `Qwen35Weights::matrix()` requires MXFP4 u32 weights; the
27B's bf16 embed/lm_head/mtp.fc go through separate bf16/fp8 paths (bf16_gemv already
runtime-dims; fp8 kernels runtime-dims). That is existing 27B bring-up work, orthogonal
to this parameterization — the shared kernels this plan fixes are the same ones both
engine closures (ENGINE / ENGINE27 in tools/mk.py) compile.

---

## 6. Migration order — small phases, 9B green throughout

Baseline first (record both NLL numbers before touching anything):

```
python tools/mk.py nll build\qwen35.insignia-index 760,6511,314,9338,369
python tools/mk.py nll build\qwen35-insig4.insignia-index 760,6511,314,9338,369
```

Per-phase gates use exactly: `smoke`, `test-deltanet`, `test-attention`, `nll`,
`dump-multistep`, `dump-pf` (+ the reference comparers). Phase order goes
bottom-up: alloc → unit-tested kernels → batch kernels → prefill cluster → decode body →
instrumentation, so every phase has a test that would catch its specific failure class.

### Phase 0 — Shape type + inference + workspace ctor (cluster: W, 43 sites)
Add `insignia_shape.hpp`, `Shape::from_model`, `validate()`; rewrite the DecodeWorkspace
ctor per §3; re-plumb the ~10 workspace call sites (nll.cu:57, generate.cu:57/116,
dump_multistep/dump_pf/dump_i4_*/test_full_model/test_layer/test_generate/test_mtp) to
`DecodeWorkspace(Shape::from_model(m), ctx)`. All 9B numbers evaluate identical.
Turn the silent launcher early-returns (mxfp4_i4.cu:66,147; gemm.cu:351) into throws.
**Catch:** any mis-derived buffer = OOB/garbage →
`python tools/mk.py nll build\qwen35.insig4.insignia-index 760,6511,314,9338,369`
(NLL must match baseline exactly) and `python tools/mk.py smoke`.

### Phase 1 — DeltaNet + GQA decode kernels (cluster: K-head, 8 sites + 2 stale tests)
deltanet.cu: kernel takes `k_share` (`kh=head/k_share`), grid `<<<v_heads,128>>>`;
attention.cu: takes `group` (`kvh=head/group`), grid `<<<q_heads,256>>>`; qwen_kernels
`deltanet_parameters` grid `<<<1,v_heads>>>`; decode.cu call sites pass `x_.shape` values.
Fix the stale test_attention.cu (5-arg `gqa_decode` no longer compiles — new harness with
`pos_dev`); parameterize both unit tests to loop the two geometries: DeltaNet
{H=32, kh=H/2} and {H=48, kh=H/3}; GQA {16 heads, group 4} and {24 heads, group 6}.
**Catch:** `python tools/mk.py test-deltanet`, `python tools/mk.py test-attention`
(27B head geometry proven at 9B cost), then `python tools/mk.py nll …` (9B e2e).

### Phase 2 — head-count elementwise + RoPE (cluster: K-q-heads, ~12 sites)
ops.cu qk_norm_rope: `isq = head < q_heads` param, grid `<<<q_heads+kv_heads,256>>>`;
qwen_kernels split_q_gate/expand_gate: element count param (attn_out), grid `ceil(n/256)`;
prefill.cu split_q_gate_batch: `t=blockIdx.x/q_heads, h=blockIdx.x%q_heads`, stride
`q_proj_rows()`; qk_norm_rope_batch: `isq` param, `(t*q_heads+head)`, `k+(t*kv_heads+…)*256`,
grid `dim3(q_heads+kv_heads,T)`; store_kv_batch: kv_row param (1024 both, pass anyway).
**Catch:** `python tools/mk.py dump-pf build\qwen35-insig4.insignia-index 760,6511,314,9338,369 64 build\pf.f32`
then `python tools/reference_pf_i4.py` (batch RoPE/split/gqa seams mis-index loudly),
plus `nll`.

### Phase 3 — prefill DeltaNet cluster (clusters: K-delta + embed + conv, ~25 sites)
embed_gather/embed_gather_i4: `groups` runtime (loop `for g=tid; g<groups; g+=blockDim`
with 160-thread blocks or keep 128 + 2 groups/thread), out stride `hidden`;
conv_prefill: **grid restructure** `dim3(ceil(ch/256),T)` killing the `/8192 %8192`;
conv_roll_state: `c < channels` param; deltanet_params_batch: `<<<T,v_heads>>>`,
`t*v_heads` strides; deltanet_prefill: grid `<<<v_heads,128>>>`, `kh=head/k_share`,
`a[t*v_heads+head]`, qkv offsets `k_off()/v_off()`, out `(t*v_heads+head)*128`;
spec_rollback: sizes from params (`delta_layers()*delta_state_per_layer()` etc.).
**Catch:** `python tools/mk.py dump-pf … 64 build\pf.f32` + reference_pf_i4.py (delta
seams), `python tools/mk.py dump-multistep build\qwen35-insig4.insignia-index 760,6511,314,9338,369 build\ms.f32`
+ tools/reference_multistep_i4.py.

### Phase 4 — decode.cu engine body (cluster: K engine body, 30 sites)
All literals in prefill_chunk_device/delta_layer/attention_layer/forward_body/mtp_layer →
`x_.shape.*` and the helpers (`delta_index(l)`, `attn_index(l)`, `qkv_rows()`,
`attn_out()` for sigmoid_mul, `intermediate` for silu_mul, `mtp_fc_rows()/cols()`,
KV strides `ai*max_context*kv_row()`); layer guards `l < layers()`. Graphs re-capture
per shape at capture time — no code change, but run the graph paths once after.
**Catch:** `python tools/mk.py dump-multistep …` + reference_multistep_i4.py
(teacher-forced per-step, layer-count/stride errors visible at step 1), `nll` both
indexes (mlx + insig4 = both quant paths), plus one graph-mode generate run
(`python tools/mk.py generate build\qwen35-insig4.insignia-index 760,6511,314,9338,369 8`)
as an extra since only generate exercises capture_spec.

### Phase 5 — instrumentation sweep + ctx policy (cluster: T/IDX/CTX, 56 sites)
Dumps/tests/benches read `Shape::from_model` instead of literals (dump_multistep 33→65
rows, dump_i4_* strides, test_qwen35 assert `12288×5120` at 27B index, test_model tensor
count from the index, bench tables gain `{10240,5120},{17408,5120},{5120,17408},{248320,5120}`
rows); generate.cu ctx policy per §4. **Catch:** all six targets green:
`smoke`, `test-deltanet`, `test-attention`, `nll` ×2 indexes, `dump-multistep`,
`dump-pf` (+ references). After this phase the shared TUs compile into both ENGINE and
ENGINE27 closures unchanged — 27B bring-up (bf16 embed gather / fp8 GEMM wiring,
nll27/dump-layers27 future targets) proceeds on top with no further shape edits.

Cluster → phase → catching test summary:

| Cluster (audit numbering) | Sites | Phase | Test that catches it |
|---|---|---|---|
| 3 DecodeWorkspace alloc | 43 | 0 | nll (both indexes) |
| 1 DeltaNet v:k ratio + head grids | ~8 | 1 | test-deltanet (now loops 2:1 and 3:1) |
| 4 Q-heads 16→24 (non-pow2) | ~14 | 1-2 | test-attention (loops g4/g6), dump-pf + reference_pf_i4.py |
| 5 Embedding gather strides | ~8 | 3 | dump-pf seam 0 / nll |
| 2 ab2 fused pair kernels | 6 | none — stays 9B-specialized (throws on cols!=4096); 27B is FP8 | test-pair (9B spec path) |
| KV strides / conv channels / state strides | ~15 | 3-4 | dump-multistep + reference_multistep_i4.py |
| decode body literals / mtp / graphs | ~30 | 4 | dump-multistep, nll, generate (graph) |
| CTX policy | 4 | 0+5 | smoke (ctor guard), generate |
| Instrumentation | 56 | 5 | all six |

---

## 7. Out of scope / contingencies

- ab2 (mxfp4.cu:519-679, mxfp4_i4.cu:157-240): remains 9B-only; `template<int VH>` is the
  fallback if an INSIG4 requant of 27B ever needs the fused pair path (capacity 64→96
  rows + 160-group staging is a redesign either way).
- 27B-specific loads (bf16 embed/lm_head rows, fp8 wiring, multi-shard index for the
  layers-N.safetensors layout) are bring-up work on top of phase 5, not shape sites.
- The full-attention parity bug (RoPE smem race, synthesis #2) and the F16-scale dtype
  mismatch (synthesis #1) are prerequisites for trusting 9B parity numbers but are
  independent fixes; this plan's phases remain valid with them fixed in any order.
