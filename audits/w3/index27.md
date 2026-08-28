# W3 — INSIDX02 index for Qwen3.8-27B-FP8: format, name map, build validation

Date: 2026-08-25. Deliverables: `tools/index27.py` (builder), 
`build/qwen38-27b-fp8.insignia-index` (119,017 B, 66 shards, 1,273 tensors), 
this report. Nothing else written; model directory untouched.

Build command (the only known-good invocation):

```
python tools/index27.py Qwen3.8-27B-FP8 build/qwen38-27b-fp8.insignia-index
```

Full build incl. the crc32 pass over all 28.75 GiB: **31.5 s** wall. 
`--no-crc` skips verification (debug only; default FAILS the build on any 
crc32 mismatch — verified against `Qwen3.8-27B-FP8\crc32.txt`, 66/66 matched).

---

## 1. INSIDX02 binary format (implemented exactly as below)

Little-endian throughout, packed, **no alignment games** (the reader stages 
shard-major per loader-gaps §7.3, so the index keeps raw absolute offsets 
and computes its own windows). All offsets are from file start.

| off | size | field |
|---|---|---|
| 0 | 8 | magic `"INSIDX02"` |
| 8 | 4 | u32 `version = 2` |
| 12 | 4×9 | **shape header**, 9 × u32, in this order: `hidden, layers, vocab, q_heads, kv_heads, delta_v, delta_k, inter, full_attention_interval` |
| 48 | 4 | u32 `shard_count` (= 66) |
| 52 | — | **shard table**, `shard_count` × |

Shard entry (variable length, back to back):

| size | field |
|---|---|
| 4 | u32 `path_len` |
| `path_len` | utf-8 `path`, **relative to the index file's directory**, native separators (e.g. `..\Qwen3.8-27B-FP8\layers-0.safetensors` — deliberately NOT the 9B index's absolute HF-cache path) |
| 8 | u64 `file_bytes` (exact shard file size) |
| 4 | u32 `crc32` (zlib/standard CRC-32 of the whole shard file, verified at build) |

Then:

| size | field |
|---|---|
| 4 | u32 `tensor_count` (= 1273) |

Tensor entry (variable length, back to back), **sorted by engine name** 
(byte-wise; all names ASCII so Python and `std::string` agree — the index 
builder hard-fails on non-ASCII):

| size | field |
|---|---|
| 2 | u16 `name_len` |
| `name_len` | utf-8 **engine name** (see §2) |
| 1 | u8 `dtype`: `1=F32, 2=BF16, 7=F8_E4M3` — matches the existing `DType` enum + the loader-gaps §1 extension `f8_e4m3=7` |
| 1 | u8 `rank` (1..3 in this checkpoint; builder rejects >255) |
| 8×rank | u64 `dims[rank]` |
| 2 | u16 `shard_idx` (index into the shard table) |
| 8 | u64 `offset_in_file` — **absolute byte offset in the shard file** (= safetensors `data_start + data_offsets[0]`, header included) |
| 8 | u64 `bytes` (tensor payload length) |

Deltas vs INSIDX01 (why a v2 reader is needed, §4): no single payload path; 
tensors are no longer a flat `{off,bytes}` into one mapping — they are 
`(shard_idx, offset_in_file, bytes)` triples; `rank`/`dims`/`dtype` semantics 
unchanged; name sort + binary-search `find()` unchanged.

Concrete first entries (byte-verified after build):

```
tensor[0]: language_model.lm_head.weight          dtype=2 rank=2 (248320,5120) shard=65 off=38080        bytes=2542796800
tensor[1]: language_model.model.embed_tokens.weight dtype=2 rank=2 (248320,5120) shard=65 off=2542834880  bytes=2542796800
tensor[2]: language_model.model.layers.0.input_layernorm.weight dtype=2 rank=1 (5120,) shard=0 off=2600  bytes=10240
```

## 2. Name map (checkpoint → engine), verified live against all 66 headers

The engine (`src/qwen35.cu:7` `matrix()`, `src/decode.cu:48,122-186`) looks 
tensors up as `base+".weight"` / `base+".scales"`. INSIDX02 stores **engine 
names**, with each F8 tensor's `weight_scale_inv` companion renamed to 
`base+".scales"`, so `matrix()` needs only the loader-gaps §4.3 patch 
(dispatch on dtype, not on scale presence).

| checkpoint name (verbatim, live-verified) | engine name in index | dtype |
|---|---|---|
| `model.language_model.layers.N.<sub>.weight` | `language_model.model.layers.N.<sub>.weight` | BF16 / F8_E4M3 |
| `model.language_model.layers.N.<m>.weight_scale_inv` | `language_model.model.layers.N.<m>.scales` | BF16 |
| `model.language_model.embed_tokens.weight` | `language_model.model.embed_tokens.weight` | BF16 [248320,5120] |
| `model.language_model.norm.weight` | `language_model.model.norm.weight` | BF16 [5120] |
| `lm_head.weight` | `language_model.lm_head.weight` | BF16 [248320,5120] |
| `mtp.fc.weight` | `language_model.mtp.fc.weight` | BF16 [5120,10240] |
| `mtp.pre_fc_norm_embedding.weight` | `language_model.mtp.pre_fc_norm_embedding.weight` | BF16 [5120] |
| `mtp.pre_fc_norm_hidden.weight` | `language_model.mtp.pre_fc_norm_hidden.weight` | BF16 [5120] |
| `mtp.norm.weight` | `language_model.mtp.norm.weight` | BF16 [5120] |
| `mtp.layers.0.<sub>` (+ `.weight_scale_inv`) | `language_model.mtp.layers.0.<sub>` (+ `.scales`) | as layers |
| `model.visual.*` (333 tensors, 921,460,192 B) | — **SKIPPED** (vision tower) | BF16 |

Rename mechanics: `X.weight_scale_inv` → strip the suffix to the stem 
`X` → apply the prefix map → append `.scales`. So 
`model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale_inv` → 
`language_model.model.layers.0.linear_attn.in_proj_qkv.scales`, and 
`matrix("language_model.model.layers.0.linear_attn.in_proj_qkv")` finds 
both `.weight` (F8) and `.scales` (BF16) exactly as the 9B convention 
demands. NOTE the suffix arithmetic: the scale is 
`stem + ".weight_scale_inv"` where the F8 weight is `stem + ".weight"` — 
i.e. `f8_name + "_scale_inv"`, NOT `f8_name + ".weight_scale_inv"`.

Engine lookup families covered (grep of qwen35.cu/decode.cu, all resolve):
`embed_tokens`, `norm.weight`, `lm_head`, `layers.N.{input_layernorm,
post_attention_layernorm}.weight`, `layers.N.linear_attn.{in_proj_qkv,
in_proj_z,in_proj_a,in_proj_b,out_proj}` (matrix), `layers.N.linear_attn.
{conv1d.weight,A_log,dt_bias,norm.weight}`, `layers.N.self_attn.{q_proj,
k_proj,v_proj,o_proj}` (matrix), `self_attn.{q_norm,k_norm}.weight`,
`mlp.{gate_proj,up_proj,down_proj}` (matrix), `mtp.{fc,pre_fc_norm_embedding,
pre_fc_norm_hidden,norm}.weight`, `mtp.layers.0.*` (full-attn template).

**FLAG — where the 9B convention can't express a 27B tensor:** 99 bf16 2-D 
matrices have NO `.scales` companion (verified by probe): 
`language_model.lm_head` (1), `language_model.model.embed_tokens` (1), 
`language_model.mtp.fc` (1), `...linear_attn.in_proj_a` (48), 
`...linear_attn.in_proj_b` (48). `matrix()` (qwen35.cu:7) acquires 
`base+".scales"` unconditionally and would throw on these — the v2 
`matrix()` MUST tolerate a missing `.scales` for `DType::bf16` kind (this 
also fixes the latent 9B `mtp.fc` throw noted in loader-gaps §4.3.2).

## 3. Build validation results (all from the actual build run)

- **crc32**: 66/66 shards match `crc32.txt` (format `crc32hex␣␣filename`, 
  77 lines; only the 66 safetensors lines consumed). Build fails hard on 
  mismatch. 31.5 s for the full pass (28.75 GiB, ~0.94 GiB/s incl. indexing).
- **shape header vs config.json** (hard-fail check): hidden 5120, layers 64, 
  vocab 248320, q_heads 24, kv_heads 4, delta_v 48, delta_k 16, inter 17408, 
  full_attention_interval 4; head_dim 256, linear dims 128, 16 full-attn 
  layers in `layer_types` — all agree.
- **counts**: checkpoint 1606 tensors = 1199 BF16 + 407 F8_E4M3; **333 
  vision tensors skipped** (`model.visual.*`, all in outside.safetensors); 
  **1273 indexed** = 407 F8 + 407 `.scales` + 459 other BF16. Engine-name 
  uniqueness after rename: 1273/1273, zero collisions.
- **scale links**: 407/407 F8 weights have their `weight_scale_inv` in the 
  same shard; every scale is BF16 with shape exactly 
  `[ceil(r/128), ceil(c/128)]` (0 violations, 0 orphans). All dims are 
  multiples of 128 so ceil never rounds.
- **byte accounting**: for every shard `8 + header_len + Σ(tensor bytes) == 
  file_bytes`, gapless and flush, with skipped-vision bytes included — 66/66 
  ✓. Totals: text F8 23.003 GiB + text BF16 4.886 GiB = 27.889 GiB indexed 
  payload; vision 0.858 GiB skipped; Σ shard file sizes = 30,866,866,928 B 
  (28.75 GiB).
- **dtype/shape/bytes consistency**: per-tensor `bytes == Π(shape) × 
  bpe(dtype)` checked for all 1273 (1 B f8, 2 B bf16).
- **self-read**: the index was parsed back with an independent reader and 
  10 random tensors (seed 27) verified `(shard, offset_in_file, bytes, 
  dtype, shape)` against freshly-parsed safetensors headers — 10/10 match 
  (mix of F8 weights, scales, bf16 smalls, conv-adjacent tensors).
- **matrix() probe**: all 407 F8 bases have `.scales` companions; bf16 
  families flagged in §2.

Per-shard summary (tensors/F8/BF16 indexed; skip = vision bytes):

| shard class | count | tensors | F8 | BF16 | F8 MiB | BF16 MiB | skip MiB | hdr B |
|---|---|---|---|---|---|---|---|---|
| linear layers (N%4!=3) | 48 | 20 | 6 | 14 | 365.00 | 1.08 | 0 | 2600/2624 |
| full-attn layers (N%4==3) | 16 | 18 | 7 | 11 | 355.00 | 0.06 | 0 | 2320/2336 |
| mtp.safetensors | 1 | 22 | 7 | 15 | 355.00 | 100.09 | 0 | 2480 |
| outside.safetensors | 1 | 3 | 0 | 3 | 0.00 | 4850.01 | 878.77 | 38080 |
| **total** | **66** | **1273** | **407** | **866** | **23.003 GiB** | **4.886 GiB** | **0.858 GiB** | — |

Full 66-row table is printed by the builder on every run.

## 4. model_file.cpp v2 reader requirements (to consume INSIDX02)

Current reader is INSIDX01-only (single path, single mapping, flat offsets). 
Required, in implementation order:

1. **Header parse**: accept magic `INSIDX02` + `version==2`; read the 9 × u32 
   shape header at offset 12 (optionally feed `Qwen35Shape` 27B constants: 
   hidden 5120, layers 64, vocab 248320, q 24, kv 4, dv 48, dk 16, inter 
   17408, interval 4 — the engine should still assert these against its 
   compiled expectations rather than trust the file). Keep INSIDX01 branch 
   for the 9B indexes.
2. **Shard table**: array of `{path (relative — resolve against the index 
   file's own directory, converting utf-8 → wchar via MultiByteToWideChar as 
   today), file_bytes, crc32}`. Per loader-gaps §2.3: open all 66 eagerly 
   (O_DIRECT+OVERLAPPED handle set on one IOCP for the staging reader; 
   optional second mapped set for warmup/parity). No LRU of handles.
3. **Tensor table**: entries `{name, dtype(u8), rank, dims, shard_idx(u16), 
   offset_in_file(u64), bytes(u64)}`; table is name-sorted — `find()` stays 
   `std::lower_bound`, signature unchanged. Names arrive pre-remapped to 
   engine convention, including `.scales` — the C++ never needs the 
   checkpoint-side `weight_scale_inv` strings.
4. **TensorView v2**: replace `const std::byte* data` (no single mapping to 
   point into) with `{uint16_t shard; uint64_t off; uint64_t bytes;}`; the 
   per-shard bounds check `off+bytes <= shard.file_bytes` replaces the 
   single-mapping escape check (model_file.cpp:36). Zero-copy `.data` 
   pointers only exist per-shard in the optional mapped set.
5. **DType enum**: add `f8_e4m3=7` (`insignia_model.hpp`), exactly matching 
   the index's dtype byte.
6. **crc32**: store-and-compare only at build time; do NOT re-hash 28.75 GiB 
   at load (loader-gaps §7.1). The value is provenance.
7. **Downstream (qwen35.cu / decode.cu)**: `matrix()` gains the 
   dtype-dispatch patch (loader-gaps §4.3): F8 → acquire `base+".scales"`, 
   assert shape `[ceil(r/128),ceil(c/128)]`, NO `cols*8`; **bf16 → tolerate 
   missing `.scales`** (the 99 flagged tensors); u32 MXFP4 path unchanged. 
   `release()` must likewise skip `.scales` when absent. `test_model.cpp`'s 
   `size()==699` / `.scales` assertions must be updated (1273 / dtype-probe).
8. **Staging note**: offsets are absolute; shard-major slot math is 
   `in_slot_off = offset_in_file - (data_start & ~4095)` per loader-gaps 
   §7.3 — the index deliberately does NOT precompute it (no alignment games 
   in the format, per this task's spec).

## 5. Deviations from loader-gaps §2.2 sketch (deliberate, per task spec)

- Simpler per-tensor record: no `scale_idx` / `in_slot_off` / `align_base` / 
  `data_start` / `payload_len` / `flags` fields — the `.scales` rename makes 
  the scale link implicit by name (matrix()'s existing string convention), 
  and staging math belongs to the reader. Byte cost: none (name lookup 
  replaces index link).
- Vision tensors are **skipped** (not indexed-with-flag) per task spec; the 
  count/bytes are reported here and re-derivable from `file_bytes` minus 
  indexed bytes.
- Shape header added (9 × u32) per task spec — loader-gaps had none.

## 6. Reproduce

```
python tools/index27.py Qwen3.8-27B-FP8 build/qwen38-27b-fp8.insignia-index
# exits nonzero on: crc mismatch, byte-accounting failure, missing/orphan/
# bad-shape scales, non-multiples in bytes-vs-shape, config/shape drift,
# engine-name collisions, self-read probe mismatch.
```
