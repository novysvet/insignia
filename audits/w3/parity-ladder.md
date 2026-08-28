# W3 — verification ladder for the Qwen3.8-27B-FP8 port

Date: 2026-08-25. Read-only investigation: live header reads of
`E:\coding\Insignia\Qwen3.8-27B-FP8\*.safetensors`, engine source reads
(`src/qwen35.cu`, `src/decode.cu`, `src/fp8.cu`, `src/test_fp8.cu`,
`include/insignia_fp8.cuh`, `include/insignia_qwen35.hpp`,
`include/insignia_qwen_kernels.cuh`), reference-script reads
(`tools/reference_all_layers{,_i4}.py`, `tools/reference_pf_i4.py`,
`tools/reference_multistep_i4.py`, `tools/nll_compare.py`, `tools/tok.py`,
`tools/chat.py`), config/crc/tokenizer inspection, one empirical dequant
experiment on real 27B weights, two web checks. No builds, no git changes.

## 0. Environment facts that shape the ladder

- `python -m pip list`: **numpy 2.5.2, tokenizers 0.22.2, safetensors 0.8.0,
  transformers 5.15.1, jinja2 3.1.6 — NO torch**. transformers without torch
  cannot run the model, so **the NumPy reference IS the ground truth** for the
  27B. There is no HF/torch cross-check available locally; every "expected"
  value in this ladder is either read from the checkpoint or computed by the
  independent NumPy reference.
- The existing reference pattern (kept for the f8 scripts): parse safetensors
  header (`u64 len + json`), `np.frombuffer` at `8+hdr+data_offsets`, bf16 via
  `(u16<<16).view(f32)`, dequant in numpy, run layer math in f32, compare
  `.f32` dumps from the engine by cosine/max/mean.
- **The 9B scripts read a single `model.safetensors`; the 27B is 66 shards**
  (layers-0..63, mtp, outside). The f8 reference scripts must take a model
  *directory* and open `layers-{l}.safetensors` per layer plus
  `outside.safetensors` for embed/norm/lm_head. Everything else in the pattern
  carries over.

## 1. Ground-truth tensor census (live header reads)

`__metadata__` is `{'format': 'pt'}` in all shards. All data regions are
contiguous (pad = 0). Verified filesize == `8 + header_len + Σbytes`.

### 1.1 CRITICAL: shard names do NOT match engine names

The engine acquires tensors as `base + ".weight"` / `base + ".scales"` with
bases like `language_model.model.layers.0.linear_attn.in_proj_qkv`
(`src/qwen35.cu:7-10`, `src/decode.cu:48/123/141-155`). The 27B-FP8 shards use
**HF names with different prefixes and a different scale suffix**:

| engine name (qwen35.cu/decode.cu) | 27B shard name (verbatim) |
|---|---|
| `language_model.model.layers.N.<sub>.weight` | `model.language_model.layers.N.<sub>.weight` |
| `language_model.model.layers.N.<m>.scales` | `model.language_model.layers.N.<m>.weight_scale_inv` |
| `language_model.model.embed_tokens.weight/.scales` | `model.language_model.embed_tokens.weight` (BF16, no scales) |
| `language_model.model.norm.weight` | `model.language_model.norm.weight` |
| `language_model.lm_head.weight/.scales` | `lm_head.weight` (BF16, no scales) |
| `language_model.mtp.<sub>.weight` | `mtp.<sub>.weight` |
| `language_model.mtp.layers.0.<sub>.scales` | `mtp.layers.0.<sub>.weight_scale_inv` |

The 9B MLX checkpoint (`~\.cache\huggingface\hub\models--sleepyeldrazi--Qwen3.5-9B-MXFP4-MTP`)
natively used the engine names. `tools/index_safetensors.py` copies names
**verbatim** and its `DTYPES` map has **no F8_E4M3 entry** (line 8; it raises
`unsupported dtype F8_E4M3`). The port therefore needs an index-side rename +
dtype mapping (recommended: extend the index tool with a 27B name map and
`"F8_E4M3": 7`), or engine-side name changes. Ladder rung R0/R3 test exactly
this.

### 1.2 layers-0.safetensors (linear/DeltaNet layer template) — 20 tensors, 383,865,448 B

| name (after `model.language_model.`) | dtype | shape |
|---|---|---|
| `layers.0.input_layernorm.weight` | BF16 | [5120] |
| `layers.0.linear_attn.A_log` | **BF16** | [48] |
| `layers.0.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] |
| `layers.0.linear_attn.dt_bias` | BF16 | [48] |
| `layers.0.linear_attn.in_proj_a.weight` | **BF16** | [48, 5120] |
| `layers.0.linear_attn.in_proj_b.weight` | **BF16** | [48, 5120] |
| `layers.0.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] |
| `layers.0.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] |
| `layers.0.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] |
| `layers.0.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] |
| `layers.0.linear_attn.norm.weight` | BF16 | [128] |
| `layers.0.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] |
| `layers.0.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] |
| `layers.0.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] |
| `layers.0.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] |
| `layers.0.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] |
| `layers.0.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] |
| `layers.0.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] |
| `layers.0.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] |
| `layers.0.post_attention_layernorm.weight` | BF16 | [5120] |

**Every scale shape is `[ceil(rows/128), ceil(cols/128)]` of the weight shape —
none is transposed** (checked exhaustively across all four shards, e.g.
`out_proj [5120,6144] → [40,48]`, `down_proj [5120,17408] → [40,136]`,
`q_proj [12288,5120] → [96,40]`).

### 1.3 layers-3.safetensors (full-attention template) — 18 tensors, 372,313,744 B

| name (after `model.language_model.`) | dtype | shape |
|---|---|---|
| `layers.3.input_layernorm.weight` | BF16 | [5120] |
| `layers.3.self_attn.q_proj.weight` | F8_E4M3 | [12288, 5120] |
| `layers.3.self_attn.q_proj.weight_scale_inv` | BF16 | [96, 40] |
| `layers.3.self_attn.k_proj.weight` | F8_E4M3 | [1024, 5120] |
| `layers.3.self_attn.k_proj.weight_scale_inv` | BF16 | [8, 40] |
| `layers.3.self_attn.v_proj.weight` | F8_E4M3 | [1024, 5120] |
| `layers.3.self_attn.v_proj.weight_scale_inv` | BF16 | [8, 40] |
| `layers.3.self_attn.o_proj.weight` | F8_E4M3 | [5120, 6144] |
| `layers.3.self_attn.o_proj.weight_scale_inv` | BF16 | [40, 48] |
| `layers.3.self_attn.q_norm.weight` | BF16 | [256] |
| `layers.3.self_attn.k_norm.weight` | BF16 | [256] |
| `layers.3.mlp.{gate,up,down}_proj.weight` | F8_E4M3 | [17408,5120]×2, [5120,17408] |
| `layers.3.mlp.{gate,up,down}_proj.weight_scale_inv` | BF16 | [136,40]×2, [40,136] |
| `layers.3.post_attention_layernorm.weight` | BF16 | [5120] |

No conv1d/A_log/dt_bias (attention layer). `q_proj` 12288 = 24 heads × 512
(256 q + 256 gate, interleaved per head — engine `split_q_gate`). 4 KV heads
× 256. GQA group = 24/4 = **6** → kvh = h/6.

### 1.4 mtp.safetensors — 22 tensors, 477,202,224 B

| name | dtype | shape |
|---|---|---|
| `mtp.fc.weight` | BF16 | [5120, 10240] |
| `mtp.pre_fc_norm_embedding.weight` | BF16 | [5120] |
| `mtp.pre_fc_norm_hidden.weight` | BF16 | [5120] |
| `mtp.norm.weight` | BF16 | [5120] |
| `mtp.layers.0.input_layernorm.weight` | BF16 | [5120] |
| `mtp.layers.0.self_attn.q_proj.weight` | F8_E4M3 | [12288, 5120] (+ scale [96,40]) |
| `mtp.layers.0.self_attn.k_proj.weight` | F8_E4M3 | [1024, 5120] (+ scale [8,40]) |
| `mtp.layers.0.self_attn.v_proj.weight` | F8_E4M3 | [1024, 5120] (+ scale [8,40]) |
| `mtp.layers.0.self_attn.o_proj.weight` | F8_E4M3 | [5120, 6144] (+ scale [40,48]) |
| `mtp.layers.0.self_attn.q_norm.weight` | BF16 | [256] |
| `mtp.layers.0.self_attn.k_norm.weight` | BF16 | [256] |
| `mtp.layers.0.mlp.{gate,up,down}_proj.weight` | F8_E4M3 | as layer 3 (+ scales) |
| `mtp.layers.0.post_attention_layernorm.weight` | BF16 | [5120] |

**MTP concat order verified against the engine** — `src/decode.cu:141-151`:
`rmsnorm(embed(token), pre_fc_norm_embedding) → x_.up` FIRST,
`rmsnorm(hidden, pre_fc_norm_hidden) → x_.norm` SECOND, then
`concat(x_.up, x_.norm, x_.qkv, 4096)` (`include/insignia_qwen_kernels.cuh:14`,
a then b), then `bf16_gemv(fc, …, rows=4096, cols=8192)` for 9B. 27B stored
`mtp.fc` is `[5120, 10240]` = `[rows=out, cols=in]` — same convention, so the
27B call is `bf16_gemv(fc, x, y, 5120, 10240)`. **Reference math:
`fc @ concat(rmsnorm(embed_row, pre_fc_norm_embedding),
rmsnorm(hidden, pre_fc_norm_hidden))` — embed-normed half first.**

### 1.5 outside.safetensors — 336 tensors, 6,007,102,112 B

Text-relevant (rest is `model.visual.*` — 27 vision blocks + merger + patch
embed + pos embed, skipped by the engine):

| name | dtype | shape |
|---|---|---|
| `lm_head.weight` | BF16 | [248320, 5120] |
| `model.language_model.embed_tokens.weight` | BF16 | [248320, 5120] |
| `model.language_model.norm.weight` | BF16 | [5120] |

### 1.6 Dtype/layout deltas vs the 9B checkpoint (from live headers, both sides)

| tensor | 9B MXFP4 ckpt | 27B FP8 ckpt | engine impact |
|---|---|---|---|
| `A_log` | **F32** [32] | **BF16** [48] | `decode.cu:124/77` casts `(const float*)A.data` — misreads bf16 as f32. Must reinterpret as `uint16_t` bf16 (like dt_bias). |
| `in_proj_a/b` | U32 MXFP4 + U8 scales | **BF16** [48,5120] | `matrix()` requires u32 → a/b path must switch to `bf16_gemv`. |
| `embed_tokens`, `lm_head` | U32 MXFP4 + U8 scales | **BF16** [248320,5120] | embed row gather → `bf16_get_row` (exists in `fp8.cu:192`); lm_head GEMV → bf16 path (`bf16_gemv_rows`, `fp8.cuh:27`). |
| `conv1d.weight` | BF16 [8192, 4, 1] (MLX layout) | BF16 [10240, 1, 4] (torch layout) | both linearize to `[out][tap]`; reference `reshape(C,4)` and engine `causal_conv4_silu` keep working; newest tap stays index 3. |
| scales | `.scales` U8 (e8m0) | `.weight_scale_inv` BF16 | name + semantics change (see §3). |
| quant weights | U32 nibbles | F8_E4M3 bytes | all matmuls switch to fp8 kernels (§6 R1). |
| names | `language_model.*` | `model.language_model.*` / `mtp.*` | see §1.1. |

Config (`config.json` text_config) confirms: hidden 5120, inter 17408, 64
layers (`layer_types` = linear×3, full×1 repeating; `(i&3)==3` full → 48
linear + 16 full), heads 24 / kv 4 / head_dim 256, vocab 248320,
`full_attention_interval` 4, `partial_rotary_factor` 0.25 (→ 64 rope dims),
`rope_parameters.rope_theta` 1e7, `linear_num_key_heads` 16,
`linear_num_value_heads` 48, `linear_key/value_head_dim` 128,
`linear_conv_kernel_dim` 4, `mamba_ssm_dtype` float32, `mtp_num_hidden_layers`
1, rms eps 1e-6, `quantization_config`: fp8 e4m3, dynamic activation,
`weight_block_size [128,128]`; embed/lm_head/norms/A_log/dt_bias/conv1d/
in_proj_a/b/mtp.fc are in `modules_to_not_convert` (matches census).

## 2. numpy e4m3 decoder (bit-exact, self-tested live)

**Variant:** safetensors `F8_E4M3` is the **OCP e4m3 variant = torch
`float8_e4m3fn`** (verified via safetensors `src/tensor.rs` — "F8_E4M3, float8
e4m3 (OCP variant)"): bias 7, subnormals m·2⁻⁹, **NaN only 0x7F/0xFF, no
infinity, max ±448 (0x7E/0xFE)**. It is NOT `fnuz` (AMD: max 240, 0x80 = NaN).
torch is not installed locally, so the torch cross-check must be skipped or run
elsewhere; the table below is verified against the OCP spec analytically.

Decoder + self-test (run live today; PASS after correcting two hand-arithmetic
errors in my own expectation table — the decoder itself never changed):

```python
import numpy as np
def e4m3_fn(code):            # OCP e4m3 'fn': bias 7, NaN only 0x7F/0xFF, max +-448
    mag = int(code) & 0x7F; e = mag >> 3; m = mag & 7
    if mag == 0x7F: return float('nan')
    if e == 0: val = m * (2.0 ** -9)                      # subnormal m * 2^(1-7-3)
    else:      val = (2.0 ** (e - 7)) * (1.0 + m / 8.0)   # normal, bias 7
    return -val if code & 0x80 else val
LUT8 = np.array([e4m3_fn(c) for c in range(256)], np.float32)

anchors = {0x00:0.0, 0x80:-0.0, 0x7F:float('nan'), 0xFF:float('nan'),
           0x7E:448.0, 0xFE:-448.0,                      # max finite, no inf
           0x01:2**-9, 0x07:7*2**-9, 0x08:2**-6,         # subnormal -> smallest normals
           0x0F:0.029296875,                             # e1,m7 = 2^-6 * 1.875
           0x38:1.0, 0x3F:1.875, 0x40:2.0, 0x47:3.75, 0x50:8.0, 0x77:240.0}
# assert each; assert strict monotone over codes 0x01..0x7E; assert 254 finite + 2 NaN.
```

Result: 254/256 finite, 2 NaN codes, strict
monotonicity over positive codes, ±0 distinct. **PASS.**

Device-side check (`include/insignia_fp8.cuh:12-16`, `e4m3x2`): fp16 bit-trick
`(mag<<7)|(sign<<8)` then ×256 is **exact for all finite codes including
subnormals** (derived: normal 2^(e-15)·1.m·2^8 = 2^(e-7)·1.m; subnormal
m·2^7/2^10·2^-14·2^8 = m·2^-9). Only deviation: 0x7F/0xFF (NaN) decode to
±480. **Census of real weights: 0 NaN-code bytes in in_proj_qkv
(52,428,800 B) and gate_proj (89,128,960 B)** — the deviation never fires on
this checkpoint. Subnormal-magnitude code rate measured at 0.024% — the
"treat-as-zero" fast paths are safe.

**Bug found in `src/test_fp8.cu:11-17`:** host reference `e4m3_host` uses
`(1 << (e - 1))` — bias 1 instead of **bias 7** — so every normal code's
reference value is 64× too large. The existing test still "passes" because
cosine is invariant to the uniform 64× scaling (subnormals break uniformity
only at the 1e-4 level). Fix before trusting any max/rel-err metric out of
test_fp8: `v = float(1 << 6 >> (7 - e))`-style, i.e. `ldexpf(1.f + m / 8.f, e - 7)`.
(Encoder `f32_to_e4m3` is correct — `exp = e + 6` with frexp is bias-7 right.)

## 3. Dequant semantics: W = F8 × weight_scale_inv (multiply)

- **Web-verified against DeepSeek-V3's official `README_WEIGHTS.md`** (the
  convention this checkpoint copies, per `quantization_config`):
  "(128x128 weight block) * weight_scale_inv"; scale shape
  `(ceil(H/128), ceil(W/128))` for weight H×W. Sources:
  [DeepSeek-V3 README_WEIGHTS.md (HF)](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/9672b384bf8a07c8968cf874cde35020f146fc64/README_WEIGHTS.md),
  [GitHub mirror](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/README_WEIGHTS.md).
- **Empirically verified on layers-0 `in_proj_qkv` today**: scales (bf16) span
  1.10e-4..9.73e-4, median 2.44e-4 ≈ amax/448; **16/16 sampled 128×128 blocks
  contain a ±448 (0x7E/0xFE) code** (per-block amax saturates the format, as
  amax/448 scaling predicts); dequant with × gives sane weight stats
  (region amax 0.106, std 0.015) while ÷ gives absurd 1e8 magnitudes.
- Reference dequant (f8 analogue of `dq()`):

```python
def dq_f8(shard_get, base):   # base without '.weight' suffix
    w = shard_get(base + '.weight')                    # u8 [R, C]
    s = shard_get(base + '.weight_scale_inv')          # bf16 [R/128, C/128] -> f32
    R, C = w.shape
    return (LUT8[w.astype(np.int32)]
            * np.repeat(np.repeat(s, 128, 0), 128, 1)).astype(np.float32)  # exact: C multiple of 128
```

All 27B weight dims are multiples of 128 (cols 5120/6144/17408; rows
1024/5120/6144/10240/12288/17408), so `np.repeat` is exact — no ceil padding
needed. Every matrix keeps `[rows=out, cols=in]` orientation; the matvec is
`W @ x` with no transpose, same as the 9B scripts.

## 4. Reference layer math for 27B (deltas from reference_all_layers_i4.py)

All deltas verified against census + config + engine kernels:

| item | 9B | 27B |
|---|---|---|
| hidden | 4096 | **5120** |
| layers | 32 (24Δ+8A) | **64 (48Δ+16A)**, still `(l&3)==3` full |
| deltanet index `di` | `l - l//4` (0..23) | same formula (0..47) |
| attn index `ai` | `l//4` (0..7) | same formula (0..15) |
| q_proj out | 8192 → `reshape(16,512)` | 12288 → **`reshape(24,512)`**; per head `q=raw[:,:256]`, `gate=raw[:,256:]` (interleave per head, engine `split_q_gate`) |
| k/v heads | 4×256 | 4×256 (unchanged); **GQA kvh = h/6** (NOT h>>2) |
| head_dim / rope | 256, 64 dims, theta 1e7, pairs (i, i+32) | unchanged (0.25×256=64, theta 1e7) |
| softmax scale | 1/16 (=1/√256) | unchanged |
| in_proj_qkv | [8192,4096], split [2048,4096] | **[10240,5120], split [2048,6144]** (q 2048 | k 2048 | v 6144) |
| q/k/v heads (Δ) | q,k 16×128; v 32×128; repeat ×2 | q,k **16×128**; v **48×128**; **k share = v-head /3** (9B was /2) |
| z / norm | z 4096 → reshape(32,128); norm [128] | **z 6144 → reshape(48,128)**; norm [128] |
| conv1d | [8192,4] taps, silu | **[10240,4]** taps (torch [10240,1,4] reshapes identically), silu |
| A_log/dt_bias/a/b | F32[32]/BF16[32]/mxfp4 | **BF16[48]/BF16[48]/BF16[48,5120]** — read as bf16, dequant = identity |
| state | 24 × (32,128,128) f32 | **48 × (48,128,128) f32** (151 MB total) |
| MLP | gate/up [12288,4096], down [4096,12288] | **[17408,5120] / [5120,17408]** |
| embed/lm_head | mxfp4 u32+u8 | **bf16**; embed = row slice of `model.language_model.embed_tokens.weight`; lm_head = bf16 [248320,5120] |
| MTP | fc [4096,8192] bf16 | **fc [5120,10240] bf16**; input = concat(embed-normed[5120], hidden-normed[5120]) — **embed first** (§1.4) |
| final norm | `language_model.model.norm.weight` | `model.language_model.norm.weight` [5120] |

Math itself (rms eps 1e-6, silu, sigmoid gates, ΔNet update
`state *= exp(-exp(A_log)·softplus(a+dt_bias))`, `mem=Σk`, `state+=β(v-mem)⊗k`,
out=Σ over k then gated rmsnorm with z-silu) is unchanged from
`tools/reference_all_layers_i4.py:9-15`. Open risk carried over from 9B: the
RoPE pairing `(i, i+32)` (halves) is the 9B reference's convention and was
never fully parity-validated on an attention layer (the smem-race bug,
synthesis §2, masked it); config's `mrope_interleaved` refers to mrope
sections and collapses to plain partial rope for text-only. R5 exists to kill
this risk.

## 5. Memory-feasible reference (16 GB host)

Per-matrix f32 dequant peaks at the largest matrix: 17408×5120×4 = **357 MB**
(one-matrix-at-a-time policy). Full linear-layer live set ≈ 1.53 GB f32 — fine
for the single-layer scripts (R4/R5) which hold one shard's matrices; for the
multi-layer scripts bound the dq cache at ≤ 4 matrices (reference_pf_i4.py's
cache of 14 would be ~4 GB worst case — lower it).

- States: 48 × 48·128·128 × 4 B = 151 MB; conv states 48 × 10240×3 × 4 B =
  5.9 MB; KV at T ≤ 64: 16 × 64×(4+4)×256×4 B = 16.8 MB. Total < 200 MB.
- **Embed: never materialize [248320,5120]** — mmap `outside.safetensors`,
  slice the token's row (5120×2 B at the header-known offset), bf16→f32.
- **lm_head argmax: never dequant to f32 wholesale** (2.5 GB bf16 → 5 GB f32).
  Stream in 8192-row bf16 chunks (168 MB f32 scratch per chunk), compute
  logits, running argmax — the `lm_head_argmax` pattern from
  `tools/reference_multistep_i4.py:94-101`, reading bf16 instead of nibbles.
- Windows page cache will hold ~half the 25.65 GB weight set after a pass;
  the R6/R9 scripts are I/O bound (~7 GB/s NVMe cold, ~40 GB/s warm), compute
  is numpy-BLAS trivial by comparison. Estimate: full 64-layer seam pass over
  T=14 ≈ minutes; greedy 8 tokens ≈ 10–25 min. Feasible; keep rungs
  one-model-dir-in, one-dump-out.

## 6. The ladder

Each rung: artifact → command → metric → threshold → gate. Gates are strict:
a rung below threshold blocks everything above it. Engine-side tools are built
via the `build\*.bat` path only (AGENTS.md); new tools below get their own bat
and the `reference_*_f8.py` scripts are new (pattern-clones of the i4 ones
with the §1.1 naming, §2 LUT, §3 dq_f8, §4 shapes, §5 memory policy).
No cross-compare between 9B and 27B outputs — different models.

- **R0 — loader/index bring-up (prerequisite, new).**
  Artifact: extended `tools/index_safetensors.py` (F8_E4M3 dtype id, 27B name
  map or engine rename) + per-shard indexes. Command: `python
  tools/index_safetensors.py Qwen3.8-27B-FP8\layers-0.safetensors build\27b\layers-0.insignia-index` (×66).
  Metric: every engine-expected name resolves; dtype/shape table diff vs §1
  census = empty. Threshold: 100% of §1 tensors mapped, zero orphans.
  Gate: R3+. (R1/R2 don't need the index.)
- **R1 — fp8 GEMV/GEMM units vs f64.**
  Artifact: `src/test_fp8.cu` extended (after fixing the `e4m3_host` bias bug,
  §2): shapes = all 27B matrix geometries {(10240,5120),(6144,5120),
  (5120,6144),(17408,5120),(5120,17408),(12288,5120),(1024,5120)}; T ∈ {3, 33, 64}
  for `fp8_gemm` (zero-padded 64-row staging), plus `fp8_gemv`/`fp8_gemv2`.
  Command: `build\test-fp8.bat` → `build\test-fp8.exe`.
  Metric: cosine of GPU vs f64 host reference per shape/T.
  Threshold: cos > 0.999999 each; additionally max |Δ|/‖y‖ < 1e-4 once the
  host reference is fixed (cosine alone hid the bias bug).
  Gate: R4+ (kernel math correct before any layer runs).
  Note: `fp8_gemm` computes all 64 staged rows regardless of T; T > 64 must
  chunk at the caller (prefill_chunk already chunks at 64). The pf_bf16
  staging buffer sized 64×12288 in decode.cu:26 must grow to 64×17408×2 B for
  the down_proj input shape — assert in R0.
- **R2 — CPU kernels (future).** Artifact: host fp8 GEMV (Zen3 F16C path per
  synthesis). Metric: rel err vs f64 on R1 shapes. Threshold: < 1e-6.
  Gate: CPU-tier decode enablement only; does not block R3–R10.
- **R3 — IO integrity.**
  Artifact: none beyond the reader + `crc32.txt` (77 lines, covers all 66
  safetensors; format `crc32hex␣␣filename`).
  Command: stream `layers-5.safetensors` through the TieredStorage reader;
  compare byte-for-byte against a direct `np.memmap` read; compute
  `zlib.crc32` per shard.
  Metric: bytes equal (count + content), crc32 equality vs the `crc32.txt`
  line (e.g. layers-5 `b52e883f`, mtp `837f3352`, outside `8dc22e46`).
  Threshold: 100% byte equality, all 66 shard crcs match.
  Gate: R4+ (all parity dumps assume honest bytes).
- **R4 — layer 0 (DeltaNet) prefill seam.**
  Artifact: `dump_pf_27b` (clone of `src/dump_pf.cu` at 27B shapes: 65 seams —
  64 layer outputs + final norm — × T × 5120 f32) via
  `Qwen35Decode::prefill_chunk_seam` (`src/decode.cu:113`).
  Command: `build\dump-pf-27b.bat` → dump; `python tools/reference_pf_f8.py
  Qwen3.8-27B-FP8 760,3712,…(GOLDEN-14) build\27b\pf-seams.f32`.
  Metric: per-seam cosine ref vs engine, over T=14 tokens.
  Threshold: every layer-0 seam cos > 0.99999 (9B layer 0 achieved 0.9999998).
  Gate: R5.
- **R5 — full-attn layer 3 seam (kills the RoPE/naming risk).**
  Artifact: same dump as R4 (seams 0..3 include layer 3).
  Command: same reference script, report row l=3.
  Metric: layer-3 seam cosine per token, positions 0..13 (rope active > 0,
  GQA h/6 path, q/k norm, output gate).
  Threshold: cos > 0.99999 for all T. If this flickers run-to-run, the smem
  race fix (synthesis §2) is incomplete — hard stop.
  Gate: R6.
- **R6 — all-64-layer seams.**
  Artifact: `tools/reference_all_layers_f8.py` (multi-shard §4; layer-at-a-time
  f32, dq cache ≤ 4) + the R4 dump.
  Command: `python tools/reference_all_layers_f8.py Qwen3.8-27B-FP8 760,3712,…
  build\27b\pf-seams.f32`.
  Metric: min and median cosine over 65 seams × 14 tokens (910 vectors).
  Threshold: min ≥ 0.9999, median ≥ 0.99999, zero NaN seams.
  Gate: R7 (whole-model prefill is coherent).
- **R7 — multistep decode states (4 steps).**
  Artifact: `dump_multistep_27b` (clone of `src/dump_multistep.cu`:
  (steps+1) × 65 × 5120 f32, greedy argmax printed per step) +
  `tools/reference_multistep_f8.py` (conv state [10240,3]×48, Δ-state
  (48,128,128)×48, KV 16×[64,4,256], bf16 embed row slice, chunked lm_head
  argmax).
  Command: `build\dump-multistep-27b.bat 760,3712,314,23470 build\27b\multi.f32`;
  `python tools/reference_multistep_f8.py Qwen3.8-27B-FP8 760,3712,314,23470
  build\27b\multi.f32`.
  Metric: per-step worst-layer cos, final-norm cos, ref argmax sequence.
  Threshold: worst-layer cos ≥ 0.9999 every step, final-norm cos ≥ 0.99999,
  ref argmax equals engine argmax for all 4 steps.
  Gate: R8.
- **R8 — NLL on fixed 128-token text.**
  Artifact: 27B index + `nll.dll` 27B build (`build\nll-27b.bat`, same link
  set at 27B shapes) + `tools/nll_compare.py` pointed at the 27B dir/tokenizer
  (single-index run; no 9B cross-compare).
  Command: `python tools/rundll.py build\nll.dll build\27b\<index> <GOLDEN-128
  ids>` vs `python tools/reference_nll_f8.py` (same ids, reference computes
  NLL in f32 from the same fp8 weights).
  Metric: |NLL_engine − NLL_ref|, ppl ratio.
  Threshold: |ΔNLL| < 0.02 nat and ppl ratio ∈ (0.98, 1.02) on first clean
  run; tighten to 0.005 once R1–R7 are green. (Both consume identical fp8
  weights; residual gap is bf16-compute ordering only.)
  Gate: R9.
- **R9 — greedy 8-token continuation vs NumPy ground truth.**
  No torch locally (§0) → the reference generates its own greedy tokens.
  Artifact: `tools/reference_greedy_f8.py` (teacher-forced GOLDEN-14 prefill,
  then 8 greedy steps; token-major, one 64-layer pass per step, per-layer
  state persisted across steps in the script; ~26 GB reads/step, est. 1–3
  min/step) + engine `generate.dll` 27B run on the same prefix.
  Command: engine: `python tools/chat.py build\27b\<index>
  Qwen3.8-27B-FP8 "<prompt>"` (or generate_ids); ref: `python
  tools/reference_greedy_f8.py Qwen3.8-27B-FP8 <GOLDEN-14 ids> 8`.
  Metric: exact greedy id sequence (8 ids) + per-step final-norm cos where
  dumps exist.
  Threshold: 8/8 ids identical. Any divergence → bisect with R6/R7 dumps at
  the first diverging step.
  Gate: R10 + "engine is correct on 27B" claim (per AGENTS.md, coherent token
  parity before correctness claims).
- **R10 — endurance/stability.**
  Artifact: 1000-token greedy generation through the full heterogeneous stack
  (spec-decode path included), logits/seams sampled every 100 tokens.
  Command: `python tools/chat.py …` extended with max_new=1000 and a stats
  hook (or `generate_ids` + wrapper).
  Metric: finiteness (no NaN/Inf in sampled seams/logits), NLL of its own
  output bounded (< 5 nat/token), committed-token monotonicity, VRAM/RAM
  budget drift, tok/s vs the placement model.
  Threshold: zero non-finite values, no state drift growth (‖seam‖ within
  3× of R6 norms), budgets flat, tok/s ≥ 80% of placement prediction.
  Gate: release of the 27B port.

## 7. Tokenizer

The 27B dir **ships its own tokenizer files**: `tokenizer.json` (19,989,325 B
at 9B — the 27B's is its own), `tokenizer_config.json`, `vocab.json`,
`merges.txt`, `chat_template.jinja`. `tools/tok.py` loads
`<model_dir>/tokenizer.json` via `tokenizers.Tokenizer` and optionally renders
the dir's own chat template with jinja2; `tools/chat.py` likewise takes the
model dir. So tests use `Qwen3.8-27B-FP8` directly — no borrowing needed.
Byte-compare 9B vs 27B `tokenizer.json`: **not identical** (different sha256;
`get_vocab_size()` 248070 vs 248077 — tokenizer-internal counts, both models
pad vocab to 248320), but probe encodings are **id-identical** (same ids for
every sentence tried, e.g. 14/14 match). Rule: always tokenize with the 27B's
own files; the 9B tokenizer is not a fallback. tok.py decode round-trips
special tokens with `skip_special_tokens=False`.

## 8. Golden test tokens (hardcode in tests)

Prompt A (unit/seam tests, 14 ids):
`"The history of computing machinery is in part the history of automatic arithmetic."`
→ `[760, 3712, 314, 23470, 25044, 369, 303, 919, 279, 3712, 314, 16465, 33633, 13]`

Prompt B (smoke, 2 ids): `"Hello!"` → `[9419, 0]`

Prompt C (R7 multistep, 4 ids): first 4 of A → `[760, 3712, 314, 23470]`

GOLDEN-128 (R8 NLL): first 128 ids of the `tools/nll_compare.py` probe text
(full text = 173 ids), encoded with the 27B tokenizer:
```
760,3712,314,23470,25044,369,303,919,279,3712,314,16465,33633,13,14496,417,54595,47434,264,6463,4560,421,1000,5474,6134,45735,5568,1472,279,1654,314,33093,11461,11,17068,3611,1412,494,279,5492,314,86844,20569,12269,13,357,9018,2843,11,13934,60875,12215,51373,11,321,279,9476,1957,1801,279,5484,19565,13,4236,5924,279,491,16698,25,7233,998,5158,11,1301,11,17736,888,11,321,30687,1040,22133,13,17722,13771,2878,8754,11391,314,10895,303,279,854,264,21461,30805,7481,2957,11,3482,279,7326,5154,314,24332,11,55404,11,321,12293,6922,49666,391,3611,13,561,1786,23438,4089,264,18826,279,1560,264,6700,30697,264,1647
```

## 9. Open risks / notes for implementers

1. Name map + F8_E4M3 in the index tool (§1.1) blocks every engine rung.
2. `A_log` now BF16 — the `(const float*)` casts in `src/decode.cu:77-78,124`
   silently read garbage if unported (no dtype assert).
3. `test_fp8.cu` host-reference bias bug (§2) — fix before R1 numbers mean
   anything; add a non-scale-invariant metric.
4. `pf_bf16` staging (64×12288 bf16) too small for 27B down_proj inputs
   (64×17408); `DecodeWorkspace` hardcodes 9B sizes throughout
   (`decode.cu:14,22-26`).
5. RoPE pairing convention (halves vs interleaved) still unvalidated on any
   attention layer — R5 is the designed kill-shot; run it multiple times to
   also catch the nondeterministic smem-race class.
6. fp8_gemm writes all 64 staged rows; zero-padding of the T-tail is
   load-bearing (decode.cu already memsets; keep it in the 27B path).
7. Reference scripts must open shards per layer (66 files) — keep the
   filehandle per shard open once, seek per tensor (the 9B single-file `get()`
   pattern otherwise reopens 26 GB of files per token).

Sources:
[safetensors repo (F8_E4M3 = OCP e4m3)](https://github.com/huggingface/safetensors),
[safetensors tensor.rs](https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs),
[huggingface.js safetensors metadata parser](https://github.com/huggingface/huggingface.js/blob/main/packages/hub/src/lib/parse-safetensors-metadata.ts),
[DeepSeek-V3 README_WEIGHTS.md (Hugging Face)](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/9672b384bf8a07c8968cf874cde35020f146fc64/README_WEIGHTS.md),
[DeepSeek-V3 README_WEIGHTS.md (GitHub)](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/README_WEIGHTS.md),
[DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437v1),
[llama.cpp issue #14781 (weight_scale_inv mapping)](https://github.com/ggml-org/llama.cpp/issues/14781)
