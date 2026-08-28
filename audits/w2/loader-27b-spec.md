# INSIDX02 loader audit + spec — Qwen3.8-27B-FP8 (week 2)

Audit date: 2026-08-25. Scope: load `E:\coding\Insignia\Qwen3.8-27B-FP8\` **without requantizing**.
Method: every safetensors header (8-byte LE length + JSON) parsed directly; tensor payloads never read.
Engine files reviewed: `src/model_file.cpp`, `src/storage.cu`, `include/insignia_model.hpp`, `include/insignia_storage.hpp`, `tools/index_safetensors.py`, `src/qwen35.cu`, `include/insignia_qwen35.hpp`, `src/test_model.cpp`.
No engine source was modified.

## 0. Executive numbers

| metric | value |
|---|---|
| safetensors shards | **66** (64 `layers-N` + `mtp` + `outside`) |
| tensors total | **1606** |
| F8_E4M3 | **407 tensors, 24,699,207,680 B = 23.00 GiB** (80.0% of payload) |
| BF16 | **1199 tensors, 6,167,455,584 B = 5.74 GiB** |
| weight payload | 30,866,663,264 B (28.75 GiB); files 30,866,866,928 B; headers only 203,664 B |
| per linear layer (48x) | 382,730,240 B F8 + 1,132,608 B BF16 |
| per full-attn layer (16x) | 372,244,480 B F8 + 66,944 B BF16 |
| mtp shard | 372,244,480 B F8 + 104,955,264 B BF16 (fc alone 104,857,600 B BF16) |
| outside shard | 0 B F8 + 6,007,064,032 B BF16 (embed 2,542,796,800 + lm_head 2,542,796,800 + final norm 10,240 + vision 921,460,192) |
| CRC32 | all 66 safetensors **match** `crc32.txt`; 9 text files mismatch (git CRLF) |
| alignment | only 45/1606 tensor starts 4096-aligned, 144/1606 512-aligned; **all** F8 sizes are 4096-multiples; **zero** pad gaps in any shard |

## 1. config.json — every field, engine implication

### 1.1 Top level

| field | value | engine implication |
|---|---|---|
| architectures | `Qwen3_5ForConditionalGeneration` | VLM wrapper; engine loads `text_config` only |
| model_type | `qwen3_5` | same family as the 9B already implemented |
| language_model_only | false | vision weights exist in checkpoint (`model.visual.*`, 921,460,192 B BF16) but engine skips them (AGENTS.md); INSIDX02 should carry a skip flag |
| image_token_id / video_token_id | 248056 / 248057 | tokenizer-level only; must not crash embed lookup (both < vocab 248320) |
| vision_start/end_token_id | 248053 / 248054 | ditto |
| tie_word_embeddings | false (also in text_config) | **lm_head is a separate 2,542,796,800 B BF16 tensor** — cannot alias embed_tokens; untied output head costs 2.37 GiB |
| transformers_version | 5.8.0.dev0 | provenance only |

### 1.2 text_config

| field | value | engine implication |
|---|---|---|
| hidden_size | **5120** | engine hardcodes 4096 (`Qwen35Shape::hidden` in include/insignia_qwen35.hpp) — must become 5120; 5120 = 128*40, clean for 128-wide FP8 blocks |
| intermediate_size | **17408** | vs 12288 hardcoded; 17408 = 128*136 (gate/up [17408,5120], down [5120,17408]) |
| num_hidden_layers | **64** | vs 32 hardcoded; per-layer state arrays double |
| layer_types / full_attention_interval | 64-entry list; interval **4** | full attention iff `(i & 3) == 3` — **identical pattern to the 9B** (`Qwen35Shape::full_attention` already correct); 48 linear (DeltaNet) + 16 full (3,7,...,63) |
| num_attention_heads | **24** | full-attn Q = 24 heads; with head_dim 256 -> 6144 attn dim |
| num_key_value_heads | **4** | GQA 6:1; k_proj/v_proj [1024,5120] |
| head_dim | **256** | kernels must handle 256-dim heads (9B-era assumptions die here) |
| attn_output_gate | **true** | q_proj is [**12288**,5120] = 2x6144: [q, gate] halves; gate = sigmoid (swish-type), product before attention scores; o_proj stays [5120,6144] |
| output_gate_type | `swish` | gate nonlinearity in the above |
| attention_bias | false | no bias tensors exist (confirmed in inventory) |
| attention_dropout | 0.0 | no-op |
| q/k norm | (implied by tensors) | q_norm/k_norm [256] per-head-dim RMSNorm, full-attn layers only (3,7,...,63); eps below |
| rms_norm_eps | 1e-06 | all RMSNorms incl. gated norms |
| hidden_act | `silu` | SwiGLU MLP (gate/up/down); plain dense MLP — no MoE and no shared-expert tensors exist in this checkpoint despite the config not-convert list naming them |
| partial_rotary_factor | **0.25** | rotary dim = 0.25 x 256 = **64 of 256**; dims beyond 64 per head get no rotation |
| rope_parameters.rope_theta | **1e7** | 10,000,000 |
| rope_parameters.rope_type | default | no YaRN baked in |
| rope_parameters.mrope_interleaved | **true** | interleaved multimodal RoPE frequency layout; text-only decoding must still match HF frequency ordering |
| rope_parameters.mrope_section | [11,11,10] | sums to 32 = rotary_dim/2 -> sections t/h/w over the 32 freq pairs |
| linear_num_key_heads / linear_key_head_dim | **16** / **128** | DeltaNet k width 2048 |
| linear_num_value_heads / linear_value_head_dim | **48** / **128** | DeltaNet v width 6144; 3 v-heads per k-head |
| linear_conv_kernel_dim | 4 | conv1d [10240,1,4] over qkv |
| mamba_ssm_dtype | **float32** | DeltaNet recurrent state in FP32 (matches existing engine choice) |
| vocab_size | **248320** | unchanged vs 9B |
| max_position_embeddings | **262144** | 256k ctx; KV cache only for the 16 full-attn layers (+ MTP layer) |
| bos/eos_token_id | 248044 / 248044 | generation_config adds second eos 248046 |
| pad_token_id | null | engine n/a |
| mtp_num_hidden_layers | **1** | one MTP layer in mtp.safetensors (full-attention type) |
| mtp_use_dedicated_embeddings | **false** | MTP shares embed_tokens + lm_head; no extra embedding tensors in mtp shard (confirmed) |
| use_cache | true | engine already caches |
| dtype | bfloat16 | reference dtype of unquantized tensors |
| model_type | qwen3_5_text | - |
| initializer_range | 0.02 | training only |

### 1.3 quantization_config

| field | value | engine implication |
|---|---|---|
| quant_method / fmt | fp8 / **e4m3** | weight dtype F8_E4M3 (1 B/elem); `tools/index_safetensors.py` currently **raises** on it (DTYPES lacks it) |
| activation_scheme | dynamic | reference stack quantizes activations at runtime; engine may instead dequant-on-use with FP32 accumulation — engine decision, not the loader's |
| weight_block_size | **[128,128]** | scales are per-128x128-block: `X.weight_scale_inv` BF16, shape [ceil(r/128), ceil(c/128)]; scale multiplies the dequantized block (inverse of the quant divisor) |
| modules_to_not_convert | ~600 entries | everything small stays BF16: all norms, q/k norms, A_log, dt_bias, conv1d, in_proj_a/b, all vision, embed, lm_head, **mtp.fc**; verified: 407/407 F8 tensors have a linked BF16 scale, 0 orphans |

### 1.4 generation_config.json (complete)

`bos_token_id: 248044, do_sample: true, eos_token_id: [248046, 248044], pad_token_id: 248044, temperature: 1.0, top_k: 20, top_p: 0.95`

Engine (greedy today) must treat **both 248046 and 248044** as stop ids; sampling values are the vendor defaults for parity harnesses.

## 2. Tensor inventory

### 2.1 Shard layout

66 shards, `30,866,866,928` B total. Layer shards take only 4 distinct sizes (header JSON grows with layer-index digit count):

| shard class | count | file size | header_len | data_start | tensors | F8 bytes | BF16 bytes |
|---|---|---|---|---|---|---|---|
| linear layer, 1-digit idx | 8 | 383,865,448 | 2592 | 2600 | 20 | 382,730,240 | 1,132,608 |
| linear layer, 2-digit idx | 40 | 383,865,472 | 2616 | 2624 | 20 | 382,730,240 | 1,132,608 |
| full-attn layer, 1-digit idx | 2 | 372,313,744 | 2312 | 2320 | 18 | 372,244,480 | 66,944 |
| full-attn layer, 2-digit idx | 14 | 372,313,760 | 2328 | 2336 | 18 | 372,244,480 | 66,944 |
| mtp | 1 | 477,202,224 | 2472 | 2480 | 22 | 372,244,480 | 104,955,264 |
| outside | 1 | 6,007,102,112 | 38072 | 38080 | 336 | 0 | 6,007,064,032 |

`__metadata__` = `{"format": "pt"}` on all layer+mtp shards; outside.safetensors has **no** metadata block.

### 2.2 Naming prefix — checkpoint vs engine

| checkpoint (this repo) | engine today | note |
|---|---|---|
| `model.language_model.layers.N.*` | `language_model.model.layers.N.*` | components swapped |
| `model.language_model.embed_tokens.weight` | `language_model.model.embed_tokens` | - |
| `model.language_model.norm.weight` | `language_model.model.norm.weight` | - |
| `lm_head.weight` | `language_model.lm_head` | - |
| `mtp.*` | `language_model.mtp.*` | e.g. engine `matrix("language_model.mtp.fc")` |
| `model.visual.*` | — | skipped, text-only |
| `X.weight` F8 + `X.weight_scale_inv` BF16 | `X.weight` u32 + `X.scales` u8/f16 (MXFP4) | **name and semantics both differ**; `Qwen35Weights::matrix` (src/qwen35.cu:7) hard-asserts MXFP4 dtypes and throws on FP8 |

All 1606 names in the shards exactly equal the 1606 entries of `model.safetensors.index.json` (set-diff empty both ways).

### 2.3 Linear-attention layer template (48 shards: i%4 != 3) — from layers-0

| tensor (prefix `model.language_model.layers.N.`) | dtype | shape | bytes | role |
|---|---|---|---|---|
| `input_layernorm.weight` | BF16 | [5120] | 10,240 | gated RMSNorm in |
| `linear_attn.A_log` | BF16 | [48] | 96 | DeltaNet A log (softplus) |
| `linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | short conv over qkv |
| `linear_attn.dt_bias` | BF16 | [48] | 96 | dt bias |
| `linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | per-v-head a gate [48,5120] |
| `linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | per-v-head b gate [48,5120] |
| `linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | linear_attn.in_proj_qkv scale (BF16 128x128-block scale) |
| `linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | linear_attn.in_proj_z scale (BF16 128x128-block scale) |
| `linear_attn.norm.weight` | BF16 | [128] | 256 | k-side RMSNorm [128] |
| `linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | linear_attn.out_proj scale (BF16 128x128-block scale) |
| `mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | mlp.down_proj scale (BF16 128x128-block scale) |
| `mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | mlp.gate_proj scale (BF16 128x128-block scale) |
| `mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | mlp.up_proj scale (BF16 128x128-block scale) |
| `post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | gated RMSNorm out |
| `linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | q2048+k2048+v6144 = 10240 |
| `linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | z gate 6144 |
| `linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | out 6144->5120 |
| `mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 17408->5120 |
| `mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | SwiGLU gate |
| `mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | SwiGLU up |

### 2.4 Full-attention layer template (16 shards: i%4==3) — from layers-3

| tensor (prefix `model.language_model.layers.N.`) | dtype | shape | bytes | role |
|---|---|---|---|---|
| `input_layernorm.weight` | BF16 | [5120] | 10,240 | gated RMSNorm |
| `mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | mlp.down_proj scale (BF16 128x128-block scale) |
| `mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | mlp.gate_proj scale (BF16 128x128-block scale) |
| `mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | mlp.up_proj scale (BF16 128x128-block scale) |
| `post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | gated RMSNorm |
| `self_attn.k_norm.weight` | BF16 | [256] | 512 | per-head-dim RMSNorm |
| `self_attn.k_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | self_attn.k_proj scale (BF16 128x128-block scale) |
| `self_attn.o_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | self_attn.o_proj scale (BF16 128x128-block scale) |
| `self_attn.q_norm.weight` | BF16 | [256] | 512 | per-head-dim RMSNorm |
| `self_attn.q_proj.weight_scale_inv` | BF16 | [96, 40] | 7,680 | self_attn.q_proj scale (BF16 128x128-block scale) |
| `self_attn.v_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | self_attn.v_proj scale (BF16 128x128-block scale) |
| `mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | SwiGLU out |
| `mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | SwiGLU |
| `mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | SwiGLU |
| `self_attn.k_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 4 KV heads x256 |
| `self_attn.o_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 6144->5120 |
| `self_attn.q_proj.weight` | F8_E4M3 | [12288, 5120] | 62,914,560 | 24 heads x256 Q, **x2 for output gate** = 12288 |
| `self_attn.v_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 4 KV heads x256 |

No A_log/conv1d/dt_bias/in_proj/norm in full-attn layers; no q/k_norm in linear layers — the two templates are disjoint beyond layernorms+MLP.

### 2.5 mtp.safetensors (complete, 22 tensors)

One **full-attention** layer (`mtp.layers.0.*`, tensor set identical to a main full-attn layer) plus MTP plumbing:

| tensor | dtype | shape | bytes | role |
|---|---|---|---|---|
| `mtp.fc.weight` | BF16 | [5120, 10240] | 104,857,600 | BF16! concat(embed 5120, hidden 5120) -> 10240 -> hidden 5120 |
| `mtp.layers.0.input_layernorm.weight` | BF16 | [5120] | 10,240 |  |
| `mtp.layers.0.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 |  |
| `mtp.layers.0.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 |  |
| `mtp.layers.0.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 |  |
| `mtp.layers.0.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 |  |
| `mtp.layers.0.self_attn.k_norm.weight` | BF16 | [256] | 512 |  |
| `mtp.layers.0.self_attn.k_proj.weight_scale_inv` | BF16 | [8, 40] | 640 |  |
| `mtp.layers.0.self_attn.o_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 |  |
| `mtp.layers.0.self_attn.q_norm.weight` | BF16 | [256] | 512 |  |
| `mtp.layers.0.self_attn.q_proj.weight_scale_inv` | BF16 | [96, 40] | 7,680 |  |
| `mtp.layers.0.self_attn.v_proj.weight_scale_inv` | BF16 | [8, 40] | 640 |  |
| `mtp.norm.weight` | BF16 | [5120] | 10,240 | MTP-final RMSNorm (own weights, before shared lm_head) |
| `mtp.pre_fc_norm_embedding.weight` | BF16 | [5120] | 10,240 | RMSNorm over embed half |
| `mtp.pre_fc_norm_hidden.weight` | BF16 | [5120] | 10,240 | RMSNorm over hidden half |
| `mtp.layers.0.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 |  |
| `mtp.layers.0.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 |  |
| `mtp.layers.0.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 |  |
| `mtp.layers.0.self_attn.k_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 |  |
| `mtp.layers.0.self_attn.o_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 |  |
| `mtp.layers.0.self_attn.q_proj.weight` | F8_E4M3 | [12288, 5120] | 62,914,560 |  |
| `mtp.layers.0.self_attn.v_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 |  |

`mtp_use_dedicated_embeddings: false` confirmed: no embed/lm_head in shard — shares `model.language_model.embed_tokens` / `lm_head`.

### 2.6 outside.safetensors (336 tensors)

| group | n | bytes |
|---|---|---|
| `lm_head.weight` BF16 [248320,5120] | 1 | 2,542,796,800 |
| `model.language_model.embed_tokens.weight` BF16 [248320,5120] | 1 | 2,542,796,800 |
| `model.language_model.norm.weight` BF16 [5120] | 1 | 10,240 |
| `model.visual.blocks.0..26.*` (12 tensors/block: qkv, proj, fc1, fc2 w+b; norm1/2 w+b) | 324 | 822,933,216 |
| `model.visual.merger.*` | 6 | 89,677,312 |
| `model.visual.patch_embed.proj.*` (weight [1152,3,2,16,16] + bias) | 2 | 3,541,248 |
| `model.visual.pos_embed.weight` [2304,1152] | 1 | 5,308,416 |

Vision = **100% BF16, 921,460,192 B (0.86 GiB)**, fully separable (leading `model.visual.`); deepstack indexes empty -> no `deepstack_merger_list` tensors exist despite config listing them.

### 2.7 weight_scale_inv shapes (complete census, 407 scales = one per F8 tensor)

| scale shape | count | applies to |
|---|---|---|
| [80,40] | 48 | linear in_proj_qkv [10240,5120] |
| [48,40] | 48 | linear in_proj_z [6144,5120] |
| [40,48] | 65 | out/o_proj [5120,6144] (48 linear + 16 full + 1 mtp) |
| [40,136] | 65 | down_proj [5120,17408] |
| [136,40] | 130 | gate+up [17408,5120] |
| [8,40] | 34 | k/v_proj [1024,5120] (17 layers x 2) |
| [96,40] | 17 | q_proj [12288,5120] |

All verified programmatically: scale shape == [ceil(rows/128), ceil(cols/128)] for all 407 pairs; every F8 tensor and its scale live in the same shard.

### 2.8 Complete per-tensor inventory

1606 rows. `off` = absolute byte offset of tensor start inside its shard file. `r4K`/`r512` = off mod 4096 / mod 512 (0 = aligned). Order: file order by absolute offset.

#### layers-0.safetensors (size 383,865,448, data_start 2,600)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.0.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,600 | 2600 | 40 |
| `model.language_model.layers.0.linear_attn.A_log` | BF16 | [48] | 96 | 12,840 | 552 | 40 |
| `model.language_model.layers.0.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,936 | 648 | 136 |
| `model.language_model.layers.0.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,856 | 648 | 136 |
| `model.language_model.layers.0.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,952 | 744 | 232 |
| `model.language_model.layers.0.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,472 | 744 | 232 |
| `model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,077,992 | 744 | 232 |
| `model.language_model.layers.0.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,392 | 3048 | 488 |
| `model.language_model.layers.0.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,232 | 2792 | 232 |
| `model.language_model.layers.0.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,488 | 3048 | 488 |
| `model.language_model.layers.0.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,328 | 2792 | 232 |
| `model.language_model.layers.0.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,208 | 1384 | 360 |
| `model.language_model.layers.0.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,088 | 4072 | 488 |
| `model.language_model.layers.0.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,968 | 2664 | 104 |
| `model.language_model.layers.0.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,208 | 616 | 104 |
| `model.language_model.layers.0.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,008 | 616 | 104 |
| `model.language_model.layers.0.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,288 | 616 | 104 |
| `model.language_model.layers.0.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,568 | 616 | 104 |
| `model.language_model.layers.0.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,528 | 616 | 104 |
| `model.language_model.layers.0.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,488 | 616 | 104 |

#### layers-1.safetensors (size 383,865,448, data_start 2,600)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.1.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,600 | 2600 | 40 |
| `model.language_model.layers.1.linear_attn.A_log` | BF16 | [48] | 96 | 12,840 | 552 | 40 |
| `model.language_model.layers.1.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,936 | 648 | 136 |
| `model.language_model.layers.1.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,856 | 648 | 136 |
| `model.language_model.layers.1.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,952 | 744 | 232 |
| `model.language_model.layers.1.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,472 | 744 | 232 |
| `model.language_model.layers.1.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,077,992 | 744 | 232 |
| `model.language_model.layers.1.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,392 | 3048 | 488 |
| `model.language_model.layers.1.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,232 | 2792 | 232 |
| `model.language_model.layers.1.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,488 | 3048 | 488 |
| `model.language_model.layers.1.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,328 | 2792 | 232 |
| `model.language_model.layers.1.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,208 | 1384 | 360 |
| `model.language_model.layers.1.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,088 | 4072 | 488 |
| `model.language_model.layers.1.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,968 | 2664 | 104 |
| `model.language_model.layers.1.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,208 | 616 | 104 |
| `model.language_model.layers.1.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,008 | 616 | 104 |
| `model.language_model.layers.1.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,288 | 616 | 104 |
| `model.language_model.layers.1.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,568 | 616 | 104 |
| `model.language_model.layers.1.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,528 | 616 | 104 |
| `model.language_model.layers.1.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,488 | 616 | 104 |

#### layers-2.safetensors (size 383,865,448, data_start 2,600)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.2.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,600 | 2600 | 40 |
| `model.language_model.layers.2.linear_attn.A_log` | BF16 | [48] | 96 | 12,840 | 552 | 40 |
| `model.language_model.layers.2.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,936 | 648 | 136 |
| `model.language_model.layers.2.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,856 | 648 | 136 |
| `model.language_model.layers.2.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,952 | 744 | 232 |
| `model.language_model.layers.2.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,472 | 744 | 232 |
| `model.language_model.layers.2.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,077,992 | 744 | 232 |
| `model.language_model.layers.2.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,392 | 3048 | 488 |
| `model.language_model.layers.2.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,232 | 2792 | 232 |
| `model.language_model.layers.2.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,488 | 3048 | 488 |
| `model.language_model.layers.2.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,328 | 2792 | 232 |
| `model.language_model.layers.2.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,208 | 1384 | 360 |
| `model.language_model.layers.2.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,088 | 4072 | 488 |
| `model.language_model.layers.2.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,968 | 2664 | 104 |
| `model.language_model.layers.2.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,208 | 616 | 104 |
| `model.language_model.layers.2.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,008 | 616 | 104 |
| `model.language_model.layers.2.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,288 | 616 | 104 |
| `model.language_model.layers.2.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,568 | 616 | 104 |
| `model.language_model.layers.2.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,528 | 616 | 104 |
| `model.language_model.layers.2.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,488 | 616 | 104 |

#### layers-3.safetensors (size 372,313,744, data_start 2,320)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.3.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,320 | 2320 | 272 |
| `model.language_model.layers.3.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 12,560 | 272 | 272 |
| `model.language_model.layers.3.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 23,440 | 2960 | 400 |
| `model.language_model.layers.3.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 34,320 | 1552 | 16 |
| `model.language_model.layers.3.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 45,200 | 144 | 144 |
| `model.language_model.layers.3.self_attn.k_norm.weight` | BF16 | [256] | 512 | 55,440 | 2192 | 144 |
| `model.language_model.layers.3.self_attn.k_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 55,952 | 2704 | 144 |
| `model.language_model.layers.3.self_attn.o_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 56,592 | 3344 | 272 |
| `model.language_model.layers.3.self_attn.q_norm.weight` | BF16 | [256] | 512 | 60,432 | 3088 | 16 |
| `model.language_model.layers.3.self_attn.q_proj.weight_scale_inv` | BF16 | [96, 40] | 7,680 | 60,944 | 3600 | 16 |
| `model.language_model.layers.3.self_attn.v_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 68,624 | 3088 | 16 |
| `model.language_model.layers.3.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 69,264 | 3728 | 144 |
| `model.language_model.layers.3.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 89,198,224 | 3728 | 144 |
| `model.language_model.layers.3.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 178,327,184 | 3728 | 144 |
| `model.language_model.layers.3.self_attn.k_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 267,456,144 | 3728 | 144 |
| `model.language_model.layers.3.self_attn.o_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 272,699,024 | 3728 | 144 |
| `model.language_model.layers.3.self_attn.q_proj.weight` | F8_E4M3 | [12288, 5120] | 62,914,560 | 304,156,304 | 3728 | 144 |
| `model.language_model.layers.3.self_attn.v_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 367,070,864 | 3728 | 144 |

#### layers-4.safetensors (size 383,865,448, data_start 2,600)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.4.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,600 | 2600 | 40 |
| `model.language_model.layers.4.linear_attn.A_log` | BF16 | [48] | 96 | 12,840 | 552 | 40 |
| `model.language_model.layers.4.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,936 | 648 | 136 |
| `model.language_model.layers.4.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,856 | 648 | 136 |
| `model.language_model.layers.4.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,952 | 744 | 232 |
| `model.language_model.layers.4.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,472 | 744 | 232 |
| `model.language_model.layers.4.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,077,992 | 744 | 232 |
| `model.language_model.layers.4.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,392 | 3048 | 488 |
| `model.language_model.layers.4.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,232 | 2792 | 232 |
| `model.language_model.layers.4.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,488 | 3048 | 488 |
| `model.language_model.layers.4.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,328 | 2792 | 232 |
| `model.language_model.layers.4.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,208 | 1384 | 360 |
| `model.language_model.layers.4.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,088 | 4072 | 488 |
| `model.language_model.layers.4.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,968 | 2664 | 104 |
| `model.language_model.layers.4.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,208 | 616 | 104 |
| `model.language_model.layers.4.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,008 | 616 | 104 |
| `model.language_model.layers.4.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,288 | 616 | 104 |
| `model.language_model.layers.4.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,568 | 616 | 104 |
| `model.language_model.layers.4.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,528 | 616 | 104 |
| `model.language_model.layers.4.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,488 | 616 | 104 |

#### layers-5.safetensors (size 383,865,448, data_start 2,600)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.5.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,600 | 2600 | 40 |
| `model.language_model.layers.5.linear_attn.A_log` | BF16 | [48] | 96 | 12,840 | 552 | 40 |
| `model.language_model.layers.5.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,936 | 648 | 136 |
| `model.language_model.layers.5.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,856 | 648 | 136 |
| `model.language_model.layers.5.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,952 | 744 | 232 |
| `model.language_model.layers.5.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,472 | 744 | 232 |
| `model.language_model.layers.5.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,077,992 | 744 | 232 |
| `model.language_model.layers.5.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,392 | 3048 | 488 |
| `model.language_model.layers.5.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,232 | 2792 | 232 |
| `model.language_model.layers.5.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,488 | 3048 | 488 |
| `model.language_model.layers.5.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,328 | 2792 | 232 |
| `model.language_model.layers.5.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,208 | 1384 | 360 |
| `model.language_model.layers.5.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,088 | 4072 | 488 |
| `model.language_model.layers.5.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,968 | 2664 | 104 |
| `model.language_model.layers.5.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,208 | 616 | 104 |
| `model.language_model.layers.5.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,008 | 616 | 104 |
| `model.language_model.layers.5.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,288 | 616 | 104 |
| `model.language_model.layers.5.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,568 | 616 | 104 |
| `model.language_model.layers.5.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,528 | 616 | 104 |
| `model.language_model.layers.5.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,488 | 616 | 104 |

#### layers-6.safetensors (size 383,865,448, data_start 2,600)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.6.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,600 | 2600 | 40 |
| `model.language_model.layers.6.linear_attn.A_log` | BF16 | [48] | 96 | 12,840 | 552 | 40 |
| `model.language_model.layers.6.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,936 | 648 | 136 |
| `model.language_model.layers.6.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,856 | 648 | 136 |
| `model.language_model.layers.6.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,952 | 744 | 232 |
| `model.language_model.layers.6.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,472 | 744 | 232 |
| `model.language_model.layers.6.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,077,992 | 744 | 232 |
| `model.language_model.layers.6.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,392 | 3048 | 488 |
| `model.language_model.layers.6.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,232 | 2792 | 232 |
| `model.language_model.layers.6.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,488 | 3048 | 488 |
| `model.language_model.layers.6.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,328 | 2792 | 232 |
| `model.language_model.layers.6.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,208 | 1384 | 360 |
| `model.language_model.layers.6.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,088 | 4072 | 488 |
| `model.language_model.layers.6.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,968 | 2664 | 104 |
| `model.language_model.layers.6.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,208 | 616 | 104 |
| `model.language_model.layers.6.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,008 | 616 | 104 |
| `model.language_model.layers.6.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,288 | 616 | 104 |
| `model.language_model.layers.6.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,568 | 616 | 104 |
| `model.language_model.layers.6.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,528 | 616 | 104 |
| `model.language_model.layers.6.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,488 | 616 | 104 |

#### layers-7.safetensors (size 372,313,744, data_start 2,320)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.7.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,320 | 2320 | 272 |
| `model.language_model.layers.7.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 12,560 | 272 | 272 |
| `model.language_model.layers.7.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 23,440 | 2960 | 400 |
| `model.language_model.layers.7.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 34,320 | 1552 | 16 |
| `model.language_model.layers.7.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 45,200 | 144 | 144 |
| `model.language_model.layers.7.self_attn.k_norm.weight` | BF16 | [256] | 512 | 55,440 | 2192 | 144 |
| `model.language_model.layers.7.self_attn.k_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 55,952 | 2704 | 144 |
| `model.language_model.layers.7.self_attn.o_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 56,592 | 3344 | 272 |
| `model.language_model.layers.7.self_attn.q_norm.weight` | BF16 | [256] | 512 | 60,432 | 3088 | 16 |
| `model.language_model.layers.7.self_attn.q_proj.weight_scale_inv` | BF16 | [96, 40] | 7,680 | 60,944 | 3600 | 16 |
| `model.language_model.layers.7.self_attn.v_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 68,624 | 3088 | 16 |
| `model.language_model.layers.7.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 69,264 | 3728 | 144 |
| `model.language_model.layers.7.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 89,198,224 | 3728 | 144 |
| `model.language_model.layers.7.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 178,327,184 | 3728 | 144 |
| `model.language_model.layers.7.self_attn.k_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 267,456,144 | 3728 | 144 |
| `model.language_model.layers.7.self_attn.o_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 272,699,024 | 3728 | 144 |
| `model.language_model.layers.7.self_attn.q_proj.weight` | F8_E4M3 | [12288, 5120] | 62,914,560 | 304,156,304 | 3728 | 144 |
| `model.language_model.layers.7.self_attn.v_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 367,070,864 | 3728 | 144 |

#### layers-8.safetensors (size 383,865,448, data_start 2,600)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.8.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,600 | 2600 | 40 |
| `model.language_model.layers.8.linear_attn.A_log` | BF16 | [48] | 96 | 12,840 | 552 | 40 |
| `model.language_model.layers.8.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,936 | 648 | 136 |
| `model.language_model.layers.8.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,856 | 648 | 136 |
| `model.language_model.layers.8.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,952 | 744 | 232 |
| `model.language_model.layers.8.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,472 | 744 | 232 |
| `model.language_model.layers.8.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,077,992 | 744 | 232 |
| `model.language_model.layers.8.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,392 | 3048 | 488 |
| `model.language_model.layers.8.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,232 | 2792 | 232 |
| `model.language_model.layers.8.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,488 | 3048 | 488 |
| `model.language_model.layers.8.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,328 | 2792 | 232 |
| `model.language_model.layers.8.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,208 | 1384 | 360 |
| `model.language_model.layers.8.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,088 | 4072 | 488 |
| `model.language_model.layers.8.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,968 | 2664 | 104 |
| `model.language_model.layers.8.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,208 | 616 | 104 |
| `model.language_model.layers.8.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,008 | 616 | 104 |
| `model.language_model.layers.8.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,288 | 616 | 104 |
| `model.language_model.layers.8.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,568 | 616 | 104 |
| `model.language_model.layers.8.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,528 | 616 | 104 |
| `model.language_model.layers.8.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,488 | 616 | 104 |

#### layers-9.safetensors (size 383,865,448, data_start 2,600)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.9.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,600 | 2600 | 40 |
| `model.language_model.layers.9.linear_attn.A_log` | BF16 | [48] | 96 | 12,840 | 552 | 40 |
| `model.language_model.layers.9.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,936 | 648 | 136 |
| `model.language_model.layers.9.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,856 | 648 | 136 |
| `model.language_model.layers.9.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,952 | 744 | 232 |
| `model.language_model.layers.9.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,472 | 744 | 232 |
| `model.language_model.layers.9.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,077,992 | 744 | 232 |
| `model.language_model.layers.9.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,392 | 3048 | 488 |
| `model.language_model.layers.9.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,232 | 2792 | 232 |
| `model.language_model.layers.9.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,488 | 3048 | 488 |
| `model.language_model.layers.9.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,328 | 2792 | 232 |
| `model.language_model.layers.9.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,208 | 1384 | 360 |
| `model.language_model.layers.9.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,088 | 4072 | 488 |
| `model.language_model.layers.9.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,968 | 2664 | 104 |
| `model.language_model.layers.9.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,208 | 616 | 104 |
| `model.language_model.layers.9.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,008 | 616 | 104 |
| `model.language_model.layers.9.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,288 | 616 | 104 |
| `model.language_model.layers.9.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,568 | 616 | 104 |
| `model.language_model.layers.9.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,528 | 616 | 104 |
| `model.language_model.layers.9.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,488 | 616 | 104 |

#### layers-10.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.10.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.10.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.10.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.10.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.10.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.10.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.10.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.10.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.10.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.10.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.10.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.10.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.10.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.10.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.10.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.10.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.10.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.10.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.10.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.10.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-11.safetensors (size 372,313,760, data_start 2,336)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.11.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,336 | 2336 | 288 |
| `model.language_model.layers.11.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 12,576 | 288 | 288 |
| `model.language_model.layers.11.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 23,456 | 2976 | 416 |
| `model.language_model.layers.11.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 34,336 | 1568 | 32 |
| `model.language_model.layers.11.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 45,216 | 160 | 160 |
| `model.language_model.layers.11.self_attn.k_norm.weight` | BF16 | [256] | 512 | 55,456 | 2208 | 160 |
| `model.language_model.layers.11.self_attn.k_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 55,968 | 2720 | 160 |
| `model.language_model.layers.11.self_attn.o_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 56,608 | 3360 | 288 |
| `model.language_model.layers.11.self_attn.q_norm.weight` | BF16 | [256] | 512 | 60,448 | 3104 | 32 |
| `model.language_model.layers.11.self_attn.q_proj.weight_scale_inv` | BF16 | [96, 40] | 7,680 | 60,960 | 3616 | 32 |
| `model.language_model.layers.11.self_attn.v_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 68,640 | 3104 | 32 |
| `model.language_model.layers.11.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 69,280 | 3744 | 160 |
| `model.language_model.layers.11.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 89,198,240 | 3744 | 160 |
| `model.language_model.layers.11.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 178,327,200 | 3744 | 160 |
| `model.language_model.layers.11.self_attn.k_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 267,456,160 | 3744 | 160 |
| `model.language_model.layers.11.self_attn.o_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 272,699,040 | 3744 | 160 |
| `model.language_model.layers.11.self_attn.q_proj.weight` | F8_E4M3 | [12288, 5120] | 62,914,560 | 304,156,320 | 3744 | 160 |
| `model.language_model.layers.11.self_attn.v_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 367,070,880 | 3744 | 160 |

#### layers-12.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.12.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.12.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.12.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.12.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.12.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.12.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.12.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.12.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.12.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.12.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.12.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.12.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.12.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.12.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.12.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.12.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.12.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.12.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.12.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.12.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-13.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.13.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.13.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.13.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.13.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.13.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.13.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.13.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.13.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.13.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.13.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.13.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.13.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.13.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.13.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.13.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.13.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.13.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.13.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.13.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.13.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-14.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.14.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.14.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.14.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.14.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.14.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.14.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.14.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.14.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.14.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.14.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.14.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.14.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.14.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.14.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.14.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.14.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.14.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.14.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.14.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.14.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-15.safetensors (size 372,313,760, data_start 2,336)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.15.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,336 | 2336 | 288 |
| `model.language_model.layers.15.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 12,576 | 288 | 288 |
| `model.language_model.layers.15.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 23,456 | 2976 | 416 |
| `model.language_model.layers.15.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 34,336 | 1568 | 32 |
| `model.language_model.layers.15.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 45,216 | 160 | 160 |
| `model.language_model.layers.15.self_attn.k_norm.weight` | BF16 | [256] | 512 | 55,456 | 2208 | 160 |
| `model.language_model.layers.15.self_attn.k_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 55,968 | 2720 | 160 |
| `model.language_model.layers.15.self_attn.o_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 56,608 | 3360 | 288 |
| `model.language_model.layers.15.self_attn.q_norm.weight` | BF16 | [256] | 512 | 60,448 | 3104 | 32 |
| `model.language_model.layers.15.self_attn.q_proj.weight_scale_inv` | BF16 | [96, 40] | 7,680 | 60,960 | 3616 | 32 |
| `model.language_model.layers.15.self_attn.v_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 68,640 | 3104 | 32 |
| `model.language_model.layers.15.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 69,280 | 3744 | 160 |
| `model.language_model.layers.15.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 89,198,240 | 3744 | 160 |
| `model.language_model.layers.15.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 178,327,200 | 3744 | 160 |
| `model.language_model.layers.15.self_attn.k_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 267,456,160 | 3744 | 160 |
| `model.language_model.layers.15.self_attn.o_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 272,699,040 | 3744 | 160 |
| `model.language_model.layers.15.self_attn.q_proj.weight` | F8_E4M3 | [12288, 5120] | 62,914,560 | 304,156,320 | 3744 | 160 |
| `model.language_model.layers.15.self_attn.v_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 367,070,880 | 3744 | 160 |

#### layers-16.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.16.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.16.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.16.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.16.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.16.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.16.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.16.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.16.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.16.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.16.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.16.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.16.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.16.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.16.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.16.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.16.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.16.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.16.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.16.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.16.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-17.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.17.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.17.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.17.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.17.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.17.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.17.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.17.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.17.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.17.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.17.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.17.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.17.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.17.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.17.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.17.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.17.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.17.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.17.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.17.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.17.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-18.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.18.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.18.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.18.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.18.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.18.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.18.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.18.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.18.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.18.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.18.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.18.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.18.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.18.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.18.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.18.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.18.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.18.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.18.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.18.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.18.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-19.safetensors (size 372,313,760, data_start 2,336)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.19.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,336 | 2336 | 288 |
| `model.language_model.layers.19.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 12,576 | 288 | 288 |
| `model.language_model.layers.19.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 23,456 | 2976 | 416 |
| `model.language_model.layers.19.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 34,336 | 1568 | 32 |
| `model.language_model.layers.19.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 45,216 | 160 | 160 |
| `model.language_model.layers.19.self_attn.k_norm.weight` | BF16 | [256] | 512 | 55,456 | 2208 | 160 |
| `model.language_model.layers.19.self_attn.k_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 55,968 | 2720 | 160 |
| `model.language_model.layers.19.self_attn.o_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 56,608 | 3360 | 288 |
| `model.language_model.layers.19.self_attn.q_norm.weight` | BF16 | [256] | 512 | 60,448 | 3104 | 32 |
| `model.language_model.layers.19.self_attn.q_proj.weight_scale_inv` | BF16 | [96, 40] | 7,680 | 60,960 | 3616 | 32 |
| `model.language_model.layers.19.self_attn.v_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 68,640 | 3104 | 32 |
| `model.language_model.layers.19.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 69,280 | 3744 | 160 |
| `model.language_model.layers.19.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 89,198,240 | 3744 | 160 |
| `model.language_model.layers.19.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 178,327,200 | 3744 | 160 |
| `model.language_model.layers.19.self_attn.k_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 267,456,160 | 3744 | 160 |
| `model.language_model.layers.19.self_attn.o_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 272,699,040 | 3744 | 160 |
| `model.language_model.layers.19.self_attn.q_proj.weight` | F8_E4M3 | [12288, 5120] | 62,914,560 | 304,156,320 | 3744 | 160 |
| `model.language_model.layers.19.self_attn.v_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 367,070,880 | 3744 | 160 |

#### layers-20.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.20.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.20.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.20.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.20.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.20.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.20.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.20.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.20.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.20.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.20.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.20.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.20.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.20.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.20.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.20.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.20.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.20.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.20.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.20.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.20.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-21.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.21.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.21.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.21.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.21.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.21.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.21.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.21.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.21.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.21.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.21.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.21.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.21.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.21.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.21.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.21.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.21.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.21.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.21.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.21.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.21.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-22.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.22.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.22.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.22.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.22.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.22.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.22.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.22.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.22.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.22.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.22.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.22.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.22.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.22.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.22.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.22.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.22.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.22.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.22.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.22.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.22.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-23.safetensors (size 372,313,760, data_start 2,336)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.23.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,336 | 2336 | 288 |
| `model.language_model.layers.23.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 12,576 | 288 | 288 |
| `model.language_model.layers.23.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 23,456 | 2976 | 416 |
| `model.language_model.layers.23.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 34,336 | 1568 | 32 |
| `model.language_model.layers.23.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 45,216 | 160 | 160 |
| `model.language_model.layers.23.self_attn.k_norm.weight` | BF16 | [256] | 512 | 55,456 | 2208 | 160 |
| `model.language_model.layers.23.self_attn.k_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 55,968 | 2720 | 160 |
| `model.language_model.layers.23.self_attn.o_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 56,608 | 3360 | 288 |
| `model.language_model.layers.23.self_attn.q_norm.weight` | BF16 | [256] | 512 | 60,448 | 3104 | 32 |
| `model.language_model.layers.23.self_attn.q_proj.weight_scale_inv` | BF16 | [96, 40] | 7,680 | 60,960 | 3616 | 32 |
| `model.language_model.layers.23.self_attn.v_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 68,640 | 3104 | 32 |
| `model.language_model.layers.23.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 69,280 | 3744 | 160 |
| `model.language_model.layers.23.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 89,198,240 | 3744 | 160 |
| `model.language_model.layers.23.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 178,327,200 | 3744 | 160 |
| `model.language_model.layers.23.self_attn.k_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 267,456,160 | 3744 | 160 |
| `model.language_model.layers.23.self_attn.o_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 272,699,040 | 3744 | 160 |
| `model.language_model.layers.23.self_attn.q_proj.weight` | F8_E4M3 | [12288, 5120] | 62,914,560 | 304,156,320 | 3744 | 160 |
| `model.language_model.layers.23.self_attn.v_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 367,070,880 | 3744 | 160 |

#### layers-24.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.24.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.24.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.24.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.24.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.24.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.24.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.24.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.24.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.24.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.24.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.24.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.24.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.24.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.24.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.24.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.24.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.24.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.24.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.24.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.24.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-25.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.25.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.25.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.25.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.25.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.25.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.25.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.25.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.25.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.25.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.25.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.25.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.25.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.25.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.25.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.25.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.25.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.25.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.25.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.25.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.25.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-26.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.26.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.26.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.26.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.26.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.26.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.26.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.26.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.26.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.26.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.26.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.26.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.26.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.26.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.26.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.26.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.26.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.26.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.26.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.26.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.26.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-27.safetensors (size 372,313,760, data_start 2,336)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.27.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,336 | 2336 | 288 |
| `model.language_model.layers.27.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 12,576 | 288 | 288 |
| `model.language_model.layers.27.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 23,456 | 2976 | 416 |
| `model.language_model.layers.27.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 34,336 | 1568 | 32 |
| `model.language_model.layers.27.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 45,216 | 160 | 160 |
| `model.language_model.layers.27.self_attn.k_norm.weight` | BF16 | [256] | 512 | 55,456 | 2208 | 160 |
| `model.language_model.layers.27.self_attn.k_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 55,968 | 2720 | 160 |
| `model.language_model.layers.27.self_attn.o_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 56,608 | 3360 | 288 |
| `model.language_model.layers.27.self_attn.q_norm.weight` | BF16 | [256] | 512 | 60,448 | 3104 | 32 |
| `model.language_model.layers.27.self_attn.q_proj.weight_scale_inv` | BF16 | [96, 40] | 7,680 | 60,960 | 3616 | 32 |
| `model.language_model.layers.27.self_attn.v_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 68,640 | 3104 | 32 |
| `model.language_model.layers.27.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 69,280 | 3744 | 160 |
| `model.language_model.layers.27.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 89,198,240 | 3744 | 160 |
| `model.language_model.layers.27.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 178,327,200 | 3744 | 160 |
| `model.language_model.layers.27.self_attn.k_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 267,456,160 | 3744 | 160 |
| `model.language_model.layers.27.self_attn.o_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 272,699,040 | 3744 | 160 |
| `model.language_model.layers.27.self_attn.q_proj.weight` | F8_E4M3 | [12288, 5120] | 62,914,560 | 304,156,320 | 3744 | 160 |
| `model.language_model.layers.27.self_attn.v_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 367,070,880 | 3744 | 160 |

#### layers-28.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.28.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.28.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.28.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.28.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.28.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.28.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.28.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.28.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.28.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.28.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.28.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.28.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.28.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.28.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.28.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.28.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.28.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.28.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.28.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.28.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-29.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.29.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.29.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.29.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.29.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.29.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.29.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.29.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.29.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.29.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.29.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.29.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.29.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.29.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.29.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.29.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.29.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.29.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.29.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.29.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.29.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-30.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.30.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.30.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.30.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.30.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.30.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.30.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.30.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.30.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.30.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.30.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.30.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.30.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.30.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.30.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.30.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.30.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.30.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.30.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.30.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.30.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-31.safetensors (size 372,313,760, data_start 2,336)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.31.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,336 | 2336 | 288 |
| `model.language_model.layers.31.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 12,576 | 288 | 288 |
| `model.language_model.layers.31.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 23,456 | 2976 | 416 |
| `model.language_model.layers.31.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 34,336 | 1568 | 32 |
| `model.language_model.layers.31.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 45,216 | 160 | 160 |
| `model.language_model.layers.31.self_attn.k_norm.weight` | BF16 | [256] | 512 | 55,456 | 2208 | 160 |
| `model.language_model.layers.31.self_attn.k_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 55,968 | 2720 | 160 |
| `model.language_model.layers.31.self_attn.o_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 56,608 | 3360 | 288 |
| `model.language_model.layers.31.self_attn.q_norm.weight` | BF16 | [256] | 512 | 60,448 | 3104 | 32 |
| `model.language_model.layers.31.self_attn.q_proj.weight_scale_inv` | BF16 | [96, 40] | 7,680 | 60,960 | 3616 | 32 |
| `model.language_model.layers.31.self_attn.v_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 68,640 | 3104 | 32 |
| `model.language_model.layers.31.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 69,280 | 3744 | 160 |
| `model.language_model.layers.31.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 89,198,240 | 3744 | 160 |
| `model.language_model.layers.31.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 178,327,200 | 3744 | 160 |
| `model.language_model.layers.31.self_attn.k_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 267,456,160 | 3744 | 160 |
| `model.language_model.layers.31.self_attn.o_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 272,699,040 | 3744 | 160 |
| `model.language_model.layers.31.self_attn.q_proj.weight` | F8_E4M3 | [12288, 5120] | 62,914,560 | 304,156,320 | 3744 | 160 |
| `model.language_model.layers.31.self_attn.v_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 367,070,880 | 3744 | 160 |

#### layers-32.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.32.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.32.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.32.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.32.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.32.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.32.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.32.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.32.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.32.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.32.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.32.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.32.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.32.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.32.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.32.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.32.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.32.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.32.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.32.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.32.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-33.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.33.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.33.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.33.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.33.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.33.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.33.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.33.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.33.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.33.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.33.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.33.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.33.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.33.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.33.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.33.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.33.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.33.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.33.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.33.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.33.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-34.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.34.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.34.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.34.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.34.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.34.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.34.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.34.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.34.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.34.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.34.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.34.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.34.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.34.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.34.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.34.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.34.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.34.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.34.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.34.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.34.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-35.safetensors (size 372,313,760, data_start 2,336)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.35.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,336 | 2336 | 288 |
| `model.language_model.layers.35.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 12,576 | 288 | 288 |
| `model.language_model.layers.35.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 23,456 | 2976 | 416 |
| `model.language_model.layers.35.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 34,336 | 1568 | 32 |
| `model.language_model.layers.35.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 45,216 | 160 | 160 |
| `model.language_model.layers.35.self_attn.k_norm.weight` | BF16 | [256] | 512 | 55,456 | 2208 | 160 |
| `model.language_model.layers.35.self_attn.k_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 55,968 | 2720 | 160 |
| `model.language_model.layers.35.self_attn.o_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 56,608 | 3360 | 288 |
| `model.language_model.layers.35.self_attn.q_norm.weight` | BF16 | [256] | 512 | 60,448 | 3104 | 32 |
| `model.language_model.layers.35.self_attn.q_proj.weight_scale_inv` | BF16 | [96, 40] | 7,680 | 60,960 | 3616 | 32 |
| `model.language_model.layers.35.self_attn.v_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 68,640 | 3104 | 32 |
| `model.language_model.layers.35.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 69,280 | 3744 | 160 |
| `model.language_model.layers.35.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 89,198,240 | 3744 | 160 |
| `model.language_model.layers.35.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 178,327,200 | 3744 | 160 |
| `model.language_model.layers.35.self_attn.k_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 267,456,160 | 3744 | 160 |
| `model.language_model.layers.35.self_attn.o_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 272,699,040 | 3744 | 160 |
| `model.language_model.layers.35.self_attn.q_proj.weight` | F8_E4M3 | [12288, 5120] | 62,914,560 | 304,156,320 | 3744 | 160 |
| `model.language_model.layers.35.self_attn.v_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 367,070,880 | 3744 | 160 |

#### layers-36.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.36.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.36.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.36.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.36.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.36.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.36.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.36.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.36.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.36.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.36.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.36.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.36.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.36.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.36.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.36.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.36.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.36.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.36.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.36.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.36.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-37.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.37.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.37.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.37.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.37.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.37.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.37.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.37.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.37.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.37.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.37.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.37.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.37.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.37.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.37.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.37.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.37.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.37.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.37.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.37.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.37.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-38.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.38.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.38.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.38.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.38.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.38.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.38.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.38.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.38.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.38.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.38.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.38.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.38.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.38.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.38.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.38.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.38.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.38.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.38.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.38.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.38.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-39.safetensors (size 372,313,760, data_start 2,336)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.39.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,336 | 2336 | 288 |
| `model.language_model.layers.39.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 12,576 | 288 | 288 |
| `model.language_model.layers.39.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 23,456 | 2976 | 416 |
| `model.language_model.layers.39.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 34,336 | 1568 | 32 |
| `model.language_model.layers.39.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 45,216 | 160 | 160 |
| `model.language_model.layers.39.self_attn.k_norm.weight` | BF16 | [256] | 512 | 55,456 | 2208 | 160 |
| `model.language_model.layers.39.self_attn.k_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 55,968 | 2720 | 160 |
| `model.language_model.layers.39.self_attn.o_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 56,608 | 3360 | 288 |
| `model.language_model.layers.39.self_attn.q_norm.weight` | BF16 | [256] | 512 | 60,448 | 3104 | 32 |
| `model.language_model.layers.39.self_attn.q_proj.weight_scale_inv` | BF16 | [96, 40] | 7,680 | 60,960 | 3616 | 32 |
| `model.language_model.layers.39.self_attn.v_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 68,640 | 3104 | 32 |
| `model.language_model.layers.39.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 69,280 | 3744 | 160 |
| `model.language_model.layers.39.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 89,198,240 | 3744 | 160 |
| `model.language_model.layers.39.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 178,327,200 | 3744 | 160 |
| `model.language_model.layers.39.self_attn.k_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 267,456,160 | 3744 | 160 |
| `model.language_model.layers.39.self_attn.o_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 272,699,040 | 3744 | 160 |
| `model.language_model.layers.39.self_attn.q_proj.weight` | F8_E4M3 | [12288, 5120] | 62,914,560 | 304,156,320 | 3744 | 160 |
| `model.language_model.layers.39.self_attn.v_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 367,070,880 | 3744 | 160 |

#### layers-40.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.40.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.40.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.40.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.40.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.40.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.40.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.40.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.40.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.40.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.40.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.40.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.40.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.40.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.40.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.40.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.40.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.40.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.40.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.40.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.40.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-41.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.41.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.41.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.41.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.41.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.41.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.41.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.41.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.41.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.41.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.41.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.41.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.41.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.41.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.41.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.41.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.41.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.41.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.41.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.41.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.41.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-42.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.42.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.42.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.42.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.42.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.42.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.42.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.42.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.42.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.42.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.42.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.42.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.42.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.42.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.42.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.42.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.42.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.42.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.42.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.42.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.42.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-43.safetensors (size 372,313,760, data_start 2,336)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.43.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,336 | 2336 | 288 |
| `model.language_model.layers.43.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 12,576 | 288 | 288 |
| `model.language_model.layers.43.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 23,456 | 2976 | 416 |
| `model.language_model.layers.43.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 34,336 | 1568 | 32 |
| `model.language_model.layers.43.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 45,216 | 160 | 160 |
| `model.language_model.layers.43.self_attn.k_norm.weight` | BF16 | [256] | 512 | 55,456 | 2208 | 160 |
| `model.language_model.layers.43.self_attn.k_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 55,968 | 2720 | 160 |
| `model.language_model.layers.43.self_attn.o_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 56,608 | 3360 | 288 |
| `model.language_model.layers.43.self_attn.q_norm.weight` | BF16 | [256] | 512 | 60,448 | 3104 | 32 |
| `model.language_model.layers.43.self_attn.q_proj.weight_scale_inv` | BF16 | [96, 40] | 7,680 | 60,960 | 3616 | 32 |
| `model.language_model.layers.43.self_attn.v_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 68,640 | 3104 | 32 |
| `model.language_model.layers.43.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 69,280 | 3744 | 160 |
| `model.language_model.layers.43.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 89,198,240 | 3744 | 160 |
| `model.language_model.layers.43.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 178,327,200 | 3744 | 160 |
| `model.language_model.layers.43.self_attn.k_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 267,456,160 | 3744 | 160 |
| `model.language_model.layers.43.self_attn.o_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 272,699,040 | 3744 | 160 |
| `model.language_model.layers.43.self_attn.q_proj.weight` | F8_E4M3 | [12288, 5120] | 62,914,560 | 304,156,320 | 3744 | 160 |
| `model.language_model.layers.43.self_attn.v_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 367,070,880 | 3744 | 160 |

#### layers-44.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.44.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.44.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.44.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.44.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.44.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.44.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.44.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.44.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.44.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.44.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.44.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.44.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.44.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.44.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.44.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.44.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.44.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.44.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.44.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.44.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-45.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.45.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.45.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.45.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.45.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.45.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.45.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.45.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.45.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.45.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.45.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.45.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.45.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.45.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.45.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.45.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.45.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.45.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.45.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.45.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.45.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-46.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.46.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.46.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.46.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.46.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.46.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.46.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.46.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.46.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.46.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.46.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.46.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.46.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.46.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.46.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.46.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.46.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.46.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.46.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.46.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.46.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-47.safetensors (size 372,313,760, data_start 2,336)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.47.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,336 | 2336 | 288 |
| `model.language_model.layers.47.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 12,576 | 288 | 288 |
| `model.language_model.layers.47.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 23,456 | 2976 | 416 |
| `model.language_model.layers.47.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 34,336 | 1568 | 32 |
| `model.language_model.layers.47.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 45,216 | 160 | 160 |
| `model.language_model.layers.47.self_attn.k_norm.weight` | BF16 | [256] | 512 | 55,456 | 2208 | 160 |
| `model.language_model.layers.47.self_attn.k_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 55,968 | 2720 | 160 |
| `model.language_model.layers.47.self_attn.o_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 56,608 | 3360 | 288 |
| `model.language_model.layers.47.self_attn.q_norm.weight` | BF16 | [256] | 512 | 60,448 | 3104 | 32 |
| `model.language_model.layers.47.self_attn.q_proj.weight_scale_inv` | BF16 | [96, 40] | 7,680 | 60,960 | 3616 | 32 |
| `model.language_model.layers.47.self_attn.v_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 68,640 | 3104 | 32 |
| `model.language_model.layers.47.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 69,280 | 3744 | 160 |
| `model.language_model.layers.47.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 89,198,240 | 3744 | 160 |
| `model.language_model.layers.47.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 178,327,200 | 3744 | 160 |
| `model.language_model.layers.47.self_attn.k_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 267,456,160 | 3744 | 160 |
| `model.language_model.layers.47.self_attn.o_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 272,699,040 | 3744 | 160 |
| `model.language_model.layers.47.self_attn.q_proj.weight` | F8_E4M3 | [12288, 5120] | 62,914,560 | 304,156,320 | 3744 | 160 |
| `model.language_model.layers.47.self_attn.v_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 367,070,880 | 3744 | 160 |

#### layers-48.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.48.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.48.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.48.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.48.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.48.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.48.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.48.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.48.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.48.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.48.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.48.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.48.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.48.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.48.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.48.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.48.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.48.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.48.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.48.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.48.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-49.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.49.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.49.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.49.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.49.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.49.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.49.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.49.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.49.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.49.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.49.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.49.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.49.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.49.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.49.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.49.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.49.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.49.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.49.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.49.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.49.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-50.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.50.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.50.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.50.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.50.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.50.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.50.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.50.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.50.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.50.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.50.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.50.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.50.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.50.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.50.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.50.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.50.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.50.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.50.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.50.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.50.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-51.safetensors (size 372,313,760, data_start 2,336)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.51.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,336 | 2336 | 288 |
| `model.language_model.layers.51.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 12,576 | 288 | 288 |
| `model.language_model.layers.51.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 23,456 | 2976 | 416 |
| `model.language_model.layers.51.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 34,336 | 1568 | 32 |
| `model.language_model.layers.51.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 45,216 | 160 | 160 |
| `model.language_model.layers.51.self_attn.k_norm.weight` | BF16 | [256] | 512 | 55,456 | 2208 | 160 |
| `model.language_model.layers.51.self_attn.k_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 55,968 | 2720 | 160 |
| `model.language_model.layers.51.self_attn.o_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 56,608 | 3360 | 288 |
| `model.language_model.layers.51.self_attn.q_norm.weight` | BF16 | [256] | 512 | 60,448 | 3104 | 32 |
| `model.language_model.layers.51.self_attn.q_proj.weight_scale_inv` | BF16 | [96, 40] | 7,680 | 60,960 | 3616 | 32 |
| `model.language_model.layers.51.self_attn.v_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 68,640 | 3104 | 32 |
| `model.language_model.layers.51.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 69,280 | 3744 | 160 |
| `model.language_model.layers.51.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 89,198,240 | 3744 | 160 |
| `model.language_model.layers.51.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 178,327,200 | 3744 | 160 |
| `model.language_model.layers.51.self_attn.k_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 267,456,160 | 3744 | 160 |
| `model.language_model.layers.51.self_attn.o_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 272,699,040 | 3744 | 160 |
| `model.language_model.layers.51.self_attn.q_proj.weight` | F8_E4M3 | [12288, 5120] | 62,914,560 | 304,156,320 | 3744 | 160 |
| `model.language_model.layers.51.self_attn.v_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 367,070,880 | 3744 | 160 |

#### layers-52.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.52.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.52.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.52.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.52.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.52.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.52.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.52.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.52.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.52.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.52.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.52.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.52.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.52.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.52.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.52.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.52.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.52.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.52.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.52.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.52.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-53.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.53.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.53.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.53.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.53.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.53.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.53.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.53.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.53.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.53.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.53.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.53.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.53.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.53.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.53.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.53.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.53.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.53.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.53.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.53.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.53.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-54.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.54.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.54.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.54.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.54.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.54.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.54.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.54.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.54.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.54.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.54.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.54.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.54.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.54.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.54.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.54.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.54.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.54.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.54.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.54.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.54.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-55.safetensors (size 372,313,760, data_start 2,336)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.55.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,336 | 2336 | 288 |
| `model.language_model.layers.55.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 12,576 | 288 | 288 |
| `model.language_model.layers.55.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 23,456 | 2976 | 416 |
| `model.language_model.layers.55.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 34,336 | 1568 | 32 |
| `model.language_model.layers.55.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 45,216 | 160 | 160 |
| `model.language_model.layers.55.self_attn.k_norm.weight` | BF16 | [256] | 512 | 55,456 | 2208 | 160 |
| `model.language_model.layers.55.self_attn.k_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 55,968 | 2720 | 160 |
| `model.language_model.layers.55.self_attn.o_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 56,608 | 3360 | 288 |
| `model.language_model.layers.55.self_attn.q_norm.weight` | BF16 | [256] | 512 | 60,448 | 3104 | 32 |
| `model.language_model.layers.55.self_attn.q_proj.weight_scale_inv` | BF16 | [96, 40] | 7,680 | 60,960 | 3616 | 32 |
| `model.language_model.layers.55.self_attn.v_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 68,640 | 3104 | 32 |
| `model.language_model.layers.55.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 69,280 | 3744 | 160 |
| `model.language_model.layers.55.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 89,198,240 | 3744 | 160 |
| `model.language_model.layers.55.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 178,327,200 | 3744 | 160 |
| `model.language_model.layers.55.self_attn.k_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 267,456,160 | 3744 | 160 |
| `model.language_model.layers.55.self_attn.o_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 272,699,040 | 3744 | 160 |
| `model.language_model.layers.55.self_attn.q_proj.weight` | F8_E4M3 | [12288, 5120] | 62,914,560 | 304,156,320 | 3744 | 160 |
| `model.language_model.layers.55.self_attn.v_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 367,070,880 | 3744 | 160 |

#### layers-56.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.56.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.56.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.56.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.56.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.56.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.56.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.56.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.56.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.56.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.56.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.56.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.56.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.56.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.56.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.56.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.56.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.56.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.56.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.56.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.56.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-57.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.57.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.57.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.57.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.57.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.57.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.57.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.57.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.57.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.57.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.57.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.57.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.57.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.57.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.57.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.57.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.57.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.57.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.57.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.57.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.57.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-58.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.58.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.58.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.58.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.58.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.58.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.58.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.58.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.58.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.58.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.58.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.58.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.58.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.58.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.58.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.58.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.58.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.58.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.58.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.58.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.58.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-59.safetensors (size 372,313,760, data_start 2,336)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.59.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,336 | 2336 | 288 |
| `model.language_model.layers.59.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 12,576 | 288 | 288 |
| `model.language_model.layers.59.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 23,456 | 2976 | 416 |
| `model.language_model.layers.59.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 34,336 | 1568 | 32 |
| `model.language_model.layers.59.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 45,216 | 160 | 160 |
| `model.language_model.layers.59.self_attn.k_norm.weight` | BF16 | [256] | 512 | 55,456 | 2208 | 160 |
| `model.language_model.layers.59.self_attn.k_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 55,968 | 2720 | 160 |
| `model.language_model.layers.59.self_attn.o_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 56,608 | 3360 | 288 |
| `model.language_model.layers.59.self_attn.q_norm.weight` | BF16 | [256] | 512 | 60,448 | 3104 | 32 |
| `model.language_model.layers.59.self_attn.q_proj.weight_scale_inv` | BF16 | [96, 40] | 7,680 | 60,960 | 3616 | 32 |
| `model.language_model.layers.59.self_attn.v_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 68,640 | 3104 | 32 |
| `model.language_model.layers.59.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 69,280 | 3744 | 160 |
| `model.language_model.layers.59.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 89,198,240 | 3744 | 160 |
| `model.language_model.layers.59.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 178,327,200 | 3744 | 160 |
| `model.language_model.layers.59.self_attn.k_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 267,456,160 | 3744 | 160 |
| `model.language_model.layers.59.self_attn.o_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 272,699,040 | 3744 | 160 |
| `model.language_model.layers.59.self_attn.q_proj.weight` | F8_E4M3 | [12288, 5120] | 62,914,560 | 304,156,320 | 3744 | 160 |
| `model.language_model.layers.59.self_attn.v_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 367,070,880 | 3744 | 160 |

#### layers-60.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.60.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.60.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.60.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.60.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.60.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.60.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.60.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.60.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.60.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.60.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.60.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.60.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.60.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.60.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.60.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.60.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.60.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.60.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.60.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.60.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-61.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.61.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.61.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.61.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.61.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.61.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.61.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.61.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.61.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.61.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.61.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.61.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.61.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.61.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.61.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.61.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.61.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.61.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.61.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.61.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.61.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-62.safetensors (size 383,865,472, data_start 2,624)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.62.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,624 | 2624 | 64 |
| `model.language_model.layers.62.linear_attn.A_log` | BF16 | [48] | 96 | 12,864 | 576 | 64 |
| `model.language_model.layers.62.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 | 12,960 | 672 | 160 |
| `model.language_model.layers.62.linear_attn.dt_bias` | BF16 | [48] | 96 | 94,880 | 672 | 160 |
| `model.language_model.layers.62.linear_attn.in_proj_a.weight` | BF16 | [48, 5120] | 491,520 | 94,976 | 768 | 256 |
| `model.language_model.layers.62.linear_attn.in_proj_b.weight` | BF16 | [48, 5120] | 491,520 | 586,496 | 768 | 256 |
| `model.language_model.layers.62.linear_attn.in_proj_qkv.weight_scale_inv` | BF16 | [80, 40] | 6,400 | 1,078,016 | 768 | 256 |
| `model.language_model.layers.62.linear_attn.in_proj_z.weight_scale_inv` | BF16 | [48, 40] | 3,840 | 1,084,416 | 3072 | 0 |
| `model.language_model.layers.62.linear_attn.norm.weight` | BF16 | [128] | 256 | 1,088,256 | 2816 | 256 |
| `model.language_model.layers.62.linear_attn.out_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 1,088,512 | 3072 | 0 |
| `model.language_model.layers.62.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 1,092,352 | 2816 | 256 |
| `model.language_model.layers.62.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,103,232 | 1408 | 384 |
| `model.language_model.layers.62.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 1,114,112 | 0 | 0 |
| `model.language_model.layers.62.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 1,124,992 | 2688 | 128 |
| `model.language_model.layers.62.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] | 52,428,800 | 1,135,232 | 640 | 128 |
| `model.language_model.layers.62.linear_attn.in_proj_z.weight` | F8_E4M3 | [6144, 5120] | 31,457,280 | 53,564,032 | 640 | 128 |
| `model.language_model.layers.62.linear_attn.out_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 85,021,312 | 640 | 128 |
| `model.language_model.layers.62.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 116,478,592 | 640 | 128 |
| `model.language_model.layers.62.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 205,607,552 | 640 | 128 |
| `model.language_model.layers.62.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 294,736,512 | 640 | 128 |

#### layers-63.safetensors (size 372,313,760, data_start 2,336)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `model.language_model.layers.63.input_layernorm.weight` | BF16 | [5120] | 10,240 | 2,336 | 2336 | 288 |
| `model.language_model.layers.63.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 12,576 | 288 | 288 |
| `model.language_model.layers.63.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 23,456 | 2976 | 416 |
| `model.language_model.layers.63.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 34,336 | 1568 | 32 |
| `model.language_model.layers.63.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 45,216 | 160 | 160 |
| `model.language_model.layers.63.self_attn.k_norm.weight` | BF16 | [256] | 512 | 55,456 | 2208 | 160 |
| `model.language_model.layers.63.self_attn.k_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 55,968 | 2720 | 160 |
| `model.language_model.layers.63.self_attn.o_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 56,608 | 3360 | 288 |
| `model.language_model.layers.63.self_attn.q_norm.weight` | BF16 | [256] | 512 | 60,448 | 3104 | 32 |
| `model.language_model.layers.63.self_attn.q_proj.weight_scale_inv` | BF16 | [96, 40] | 7,680 | 60,960 | 3616 | 32 |
| `model.language_model.layers.63.self_attn.v_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 68,640 | 3104 | 32 |
| `model.language_model.layers.63.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 69,280 | 3744 | 160 |
| `model.language_model.layers.63.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 89,198,240 | 3744 | 160 |
| `model.language_model.layers.63.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 178,327,200 | 3744 | 160 |
| `model.language_model.layers.63.self_attn.k_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 267,456,160 | 3744 | 160 |
| `model.language_model.layers.63.self_attn.o_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 272,699,040 | 3744 | 160 |
| `model.language_model.layers.63.self_attn.q_proj.weight` | F8_E4M3 | [12288, 5120] | 62,914,560 | 304,156,320 | 3744 | 160 |
| `model.language_model.layers.63.self_attn.v_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 367,070,880 | 3744 | 160 |

#### mtp.safetensors (size 477,202,224, data_start 2,480)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `mtp.fc.weight` | BF16 | [5120, 10240] | 104,857,600 | 2,480 | 2480 | 432 |
| `mtp.layers.0.input_layernorm.weight` | BF16 | [5120] | 10,240 | 104,860,080 | 2480 | 432 |
| `mtp.layers.0.mlp.down_proj.weight_scale_inv` | BF16 | [40, 136] | 10,880 | 104,870,320 | 432 | 432 |
| `mtp.layers.0.mlp.gate_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 104,881,200 | 3120 | 48 |
| `mtp.layers.0.mlp.up_proj.weight_scale_inv` | BF16 | [136, 40] | 10,880 | 104,892,080 | 1712 | 176 |
| `mtp.layers.0.post_attention_layernorm.weight` | BF16 | [5120] | 10,240 | 104,902,960 | 304 | 304 |
| `mtp.layers.0.self_attn.k_norm.weight` | BF16 | [256] | 512 | 104,913,200 | 2352 | 304 |
| `mtp.layers.0.self_attn.k_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 104,913,712 | 2864 | 304 |
| `mtp.layers.0.self_attn.o_proj.weight_scale_inv` | BF16 | [40, 48] | 3,840 | 104,914,352 | 3504 | 432 |
| `mtp.layers.0.self_attn.q_norm.weight` | BF16 | [256] | 512 | 104,918,192 | 3248 | 176 |
| `mtp.layers.0.self_attn.q_proj.weight_scale_inv` | BF16 | [96, 40] | 7,680 | 104,918,704 | 3760 | 176 |
| `mtp.layers.0.self_attn.v_proj.weight_scale_inv` | BF16 | [8, 40] | 640 | 104,926,384 | 3248 | 176 |
| `mtp.norm.weight` | BF16 | [5120] | 10,240 | 104,927,024 | 3888 | 304 |
| `mtp.pre_fc_norm_embedding.weight` | BF16 | [5120] | 10,240 | 104,937,264 | 1840 | 304 |
| `mtp.pre_fc_norm_hidden.weight` | BF16 | [5120] | 10,240 | 104,947,504 | 3888 | 304 |
| `mtp.layers.0.mlp.down_proj.weight` | F8_E4M3 | [5120, 17408] | 89,128,960 | 104,957,744 | 1840 | 304 |
| `mtp.layers.0.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 194,086,704 | 1840 | 304 |
| `mtp.layers.0.mlp.up_proj.weight` | F8_E4M3 | [17408, 5120] | 89,128,960 | 283,215,664 | 1840 | 304 |
| `mtp.layers.0.self_attn.k_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 372,344,624 | 1840 | 304 |
| `mtp.layers.0.self_attn.o_proj.weight` | F8_E4M3 | [5120, 6144] | 31,457,280 | 377,587,504 | 1840 | 304 |
| `mtp.layers.0.self_attn.q_proj.weight` | F8_E4M3 | [12288, 5120] | 62,914,560 | 409,044,784 | 1840 | 304 |
| `mtp.layers.0.self_attn.v_proj.weight` | F8_E4M3 | [1024, 5120] | 5,242,880 | 471,959,344 | 1840 | 304 |

#### outside.safetensors (size 6,007,102,112, data_start 38,080)

| tensor | dtype | shape | bytes | off | r4K | r512 |
|---|---|---|---|---|---|---|
| `lm_head.weight` | BF16 | [248320, 5120] | 2,542,796,800 | 38,080 | 1216 | 192 |
| `model.language_model.embed_tokens.weight` | BF16 | [248320, 5120] | 2,542,796,800 | 2,542,834,880 | 1216 | 192 |
| `model.language_model.norm.weight` | BF16 | [5120] | 10,240 | 5,085,631,680 | 1216 | 192 |
| `model.visual.blocks.0.attn.proj.bias` | BF16 | [1152] | 2,304 | 5,085,641,920 | 3264 | 192 |
| `model.visual.blocks.0.attn.proj.weight` | BF16 | [1152, 1152] | 2,654,208 | 5,085,644,224 | 1472 | 448 |
| `model.visual.blocks.0.attn.qkv.bias` | BF16 | [3456] | 6,912 | 5,088,298,432 | 1472 | 448 |
| `model.visual.blocks.0.attn.qkv.weight` | BF16 | [3456, 1152] | 7,962,624 | 5,088,305,344 | 192 | 192 |
| `model.visual.blocks.0.mlp.linear_fc1.bias` | BF16 | [4304] | 8,608 | 5,096,267,968 | 192 | 192 |
| `model.visual.blocks.0.mlp.linear_fc1.weight` | BF16 | [4304, 1152] | 9,916,416 | 5,096,276,576 | 608 | 96 |
| `model.visual.blocks.0.mlp.linear_fc2.bias` | BF16 | [1152] | 2,304 | 5,106,192,992 | 608 | 96 |
| `model.visual.blocks.0.mlp.linear_fc2.weight` | BF16 | [1152, 4304] | 9,916,416 | 5,106,195,296 | 2912 | 352 |
| `model.visual.blocks.0.norm1.bias` | BF16 | [1152] | 2,304 | 5,116,111,712 | 2912 | 352 |
| `model.visual.blocks.0.norm1.weight` | BF16 | [1152] | 2,304 | 5,116,114,016 | 1120 | 96 |
| `model.visual.blocks.0.norm2.bias` | BF16 | [1152] | 2,304 | 5,116,116,320 | 3424 | 352 |
| `model.visual.blocks.0.norm2.weight` | BF16 | [1152] | 2,304 | 5,116,118,624 | 1632 | 96 |
| `model.visual.blocks.1.attn.proj.bias` | BF16 | [1152] | 2,304 | 5,116,120,928 | 3936 | 352 |
| `model.visual.blocks.1.attn.proj.weight` | BF16 | [1152, 1152] | 2,654,208 | 5,116,123,232 | 2144 | 96 |
| `model.visual.blocks.1.attn.qkv.bias` | BF16 | [3456] | 6,912 | 5,118,777,440 | 2144 | 96 |
| `model.visual.blocks.1.attn.qkv.weight` | BF16 | [3456, 1152] | 7,962,624 | 5,118,784,352 | 864 | 352 |
| `model.visual.blocks.1.mlp.linear_fc1.bias` | BF16 | [4304] | 8,608 | 5,126,746,976 | 864 | 352 |
| `model.visual.blocks.1.mlp.linear_fc1.weight` | BF16 | [4304, 1152] | 9,916,416 | 5,126,755,584 | 1280 | 256 |
| `model.visual.blocks.1.mlp.linear_fc2.bias` | BF16 | [1152] | 2,304 | 5,136,672,000 | 1280 | 256 |
| `model.visual.blocks.1.mlp.linear_fc2.weight` | BF16 | [1152, 4304] | 9,916,416 | 5,136,674,304 | 3584 | 0 |
| `model.visual.blocks.1.norm1.bias` | BF16 | [1152] | 2,304 | 5,146,590,720 | 3584 | 0 |
| `model.visual.blocks.1.norm1.weight` | BF16 | [1152] | 2,304 | 5,146,593,024 | 1792 | 256 |
| `model.visual.blocks.1.norm2.bias` | BF16 | [1152] | 2,304 | 5,146,595,328 | 0 | 0 |
| `model.visual.blocks.1.norm2.weight` | BF16 | [1152] | 2,304 | 5,146,597,632 | 2304 | 256 |
| `model.visual.blocks.10.attn.proj.bias` | BF16 | [1152] | 2,304 | 5,146,599,936 | 512 | 0 |
| `model.visual.blocks.10.attn.proj.weight` | BF16 | [1152, 1152] | 2,654,208 | 5,146,602,240 | 2816 | 256 |
| `model.visual.blocks.10.attn.qkv.bias` | BF16 | [3456] | 6,912 | 5,149,256,448 | 2816 | 256 |
| `model.visual.blocks.10.attn.qkv.weight` | BF16 | [3456, 1152] | 7,962,624 | 5,149,263,360 | 1536 | 0 |
| `model.visual.blocks.10.mlp.linear_fc1.bias` | BF16 | [4304] | 8,608 | 5,157,225,984 | 1536 | 0 |
| `model.visual.blocks.10.mlp.linear_fc1.weight` | BF16 | [4304, 1152] | 9,916,416 | 5,157,234,592 | 1952 | 416 |
| `model.visual.blocks.10.mlp.linear_fc2.bias` | BF16 | [1152] | 2,304 | 5,167,151,008 | 1952 | 416 |
| `model.visual.blocks.10.mlp.linear_fc2.weight` | BF16 | [1152, 4304] | 9,916,416 | 5,167,153,312 | 160 | 160 |
| `model.visual.blocks.10.norm1.bias` | BF16 | [1152] | 2,304 | 5,177,069,728 | 160 | 160 |
| `model.visual.blocks.10.norm1.weight` | BF16 | [1152] | 2,304 | 5,177,072,032 | 2464 | 416 |
| `model.visual.blocks.10.norm2.bias` | BF16 | [1152] | 2,304 | 5,177,074,336 | 672 | 160 |
| `model.visual.blocks.10.norm2.weight` | BF16 | [1152] | 2,304 | 5,177,076,640 | 2976 | 416 |
| `model.visual.blocks.11.attn.proj.bias` | BF16 | [1152] | 2,304 | 5,177,078,944 | 1184 | 160 |
| `model.visual.blocks.11.attn.proj.weight` | BF16 | [1152, 1152] | 2,654,208 | 5,177,081,248 | 3488 | 416 |
| `model.visual.blocks.11.attn.qkv.bias` | BF16 | [3456] | 6,912 | 5,179,735,456 | 3488 | 416 |
| `model.visual.blocks.11.attn.qkv.weight` | BF16 | [3456, 1152] | 7,962,624 | 5,179,742,368 | 2208 | 160 |
| `model.visual.blocks.11.mlp.linear_fc1.bias` | BF16 | [4304] | 8,608 | 5,187,704,992 | 2208 | 160 |
| `model.visual.blocks.11.mlp.linear_fc1.weight` | BF16 | [4304, 1152] | 9,916,416 | 5,187,713,600 | 2624 | 64 |
| `model.visual.blocks.11.mlp.linear_fc2.bias` | BF16 | [1152] | 2,304 | 5,197,630,016 | 2624 | 64 |
| `model.visual.blocks.11.mlp.linear_fc2.weight` | BF16 | [1152, 4304] | 9,916,416 | 5,197,632,320 | 832 | 320 |
| `model.visual.blocks.11.norm1.bias` | BF16 | [1152] | 2,304 | 5,207,548,736 | 832 | 320 |
| `model.visual.blocks.11.norm1.weight` | BF16 | [1152] | 2,304 | 5,207,551,040 | 3136 | 64 |
| `model.visual.blocks.11.norm2.bias` | BF16 | [1152] | 2,304 | 5,207,553,344 | 1344 | 320 |
| `model.visual.blocks.11.norm2.weight` | BF16 | [1152] | 2,304 | 5,207,555,648 | 3648 | 64 |
| `model.visual.blocks.12.attn.proj.bias` | BF16 | [1152] | 2,304 | 5,207,557,952 | 1856 | 320 |
| `model.visual.blocks.12.attn.proj.weight` | BF16 | [1152, 1152] | 2,654,208 | 5,207,560,256 | 64 | 64 |
| `model.visual.blocks.12.attn.qkv.bias` | BF16 | [3456] | 6,912 | 5,210,214,464 | 64 | 64 |
| `model.visual.blocks.12.attn.qkv.weight` | BF16 | [3456, 1152] | 7,962,624 | 5,210,221,376 | 2880 | 320 |
| `model.visual.blocks.12.mlp.linear_fc1.bias` | BF16 | [4304] | 8,608 | 5,218,184,000 | 2880 | 320 |
| `model.visual.blocks.12.mlp.linear_fc1.weight` | BF16 | [4304, 1152] | 9,916,416 | 5,218,192,608 | 3296 | 224 |
| `model.visual.blocks.12.mlp.linear_fc2.bias` | BF16 | [1152] | 2,304 | 5,228,109,024 | 3296 | 224 |
| `model.visual.blocks.12.mlp.linear_fc2.weight` | BF16 | [1152, 4304] | 9,916,416 | 5,228,111,328 | 1504 | 480 |
| `model.visual.blocks.12.norm1.bias` | BF16 | [1152] | 2,304 | 5,238,027,744 | 1504 | 480 |
| `model.visual.blocks.12.norm1.weight` | BF16 | [1152] | 2,304 | 5,238,030,048 | 3808 | 224 |
| `model.visual.blocks.12.norm2.bias` | BF16 | [1152] | 2,304 | 5,238,032,352 | 2016 | 480 |
| `model.visual.blocks.12.norm2.weight` | BF16 | [1152] | 2,304 | 5,238,034,656 | 224 | 224 |
| `model.visual.blocks.13.attn.proj.bias` | BF16 | [1152] | 2,304 | 5,238,036,960 | 2528 | 480 |
| `model.visual.blocks.13.attn.proj.weight` | BF16 | [1152, 1152] | 2,654,208 | 5,238,039,264 | 736 | 224 |
| `model.visual.blocks.13.attn.qkv.bias` | BF16 | [3456] | 6,912 | 5,240,693,472 | 736 | 224 |
| `model.visual.blocks.13.attn.qkv.weight` | BF16 | [3456, 1152] | 7,962,624 | 5,240,700,384 | 3552 | 480 |
| `model.visual.blocks.13.mlp.linear_fc1.bias` | BF16 | [4304] | 8,608 | 5,248,663,008 | 3552 | 480 |
| `model.visual.blocks.13.mlp.linear_fc1.weight` | BF16 | [4304, 1152] | 9,916,416 | 5,248,671,616 | 3968 | 384 |
| `model.visual.blocks.13.mlp.linear_fc2.bias` | BF16 | [1152] | 2,304 | 5,258,588,032 | 3968 | 384 |
| `model.visual.blocks.13.mlp.linear_fc2.weight` | BF16 | [1152, 4304] | 9,916,416 | 5,258,590,336 | 2176 | 128 |
| `model.visual.blocks.13.norm1.bias` | BF16 | [1152] | 2,304 | 5,268,506,752 | 2176 | 128 |
| `model.visual.blocks.13.norm1.weight` | BF16 | [1152] | 2,304 | 5,268,509,056 | 384 | 384 |
| `model.visual.blocks.13.norm2.bias` | BF16 | [1152] | 2,304 | 5,268,511,360 | 2688 | 128 |
| `model.visual.blocks.13.norm2.weight` | BF16 | [1152] | 2,304 | 5,268,513,664 | 896 | 384 |
| `model.visual.blocks.14.attn.proj.bias` | BF16 | [1152] | 2,304 | 5,268,515,968 | 3200 | 128 |
| `model.visual.blocks.14.attn.proj.weight` | BF16 | [1152, 1152] | 2,654,208 | 5,268,518,272 | 1408 | 384 |
| `model.visual.blocks.14.attn.qkv.bias` | BF16 | [3456] | 6,912 | 5,271,172,480 | 1408 | 384 |
| `model.visual.blocks.14.attn.qkv.weight` | BF16 | [3456, 1152] | 7,962,624 | 5,271,179,392 | 128 | 128 |
| `model.visual.blocks.14.mlp.linear_fc1.bias` | BF16 | [4304] | 8,608 | 5,279,142,016 | 128 | 128 |
| `model.visual.blocks.14.mlp.linear_fc1.weight` | BF16 | [4304, 1152] | 9,916,416 | 5,279,150,624 | 544 | 32 |
| `model.visual.blocks.14.mlp.linear_fc2.bias` | BF16 | [1152] | 2,304 | 5,289,067,040 | 544 | 32 |
| `model.visual.blocks.14.mlp.linear_fc2.weight` | BF16 | [1152, 4304] | 9,916,416 | 5,289,069,344 | 2848 | 288 |
| `model.visual.blocks.14.norm1.bias` | BF16 | [1152] | 2,304 | 5,298,985,760 | 2848 | 288 |
| `model.visual.blocks.14.norm1.weight` | BF16 | [1152] | 2,304 | 5,298,988,064 | 1056 | 32 |
| `model.visual.blocks.14.norm2.bias` | BF16 | [1152] | 2,304 | 5,298,990,368 | 3360 | 288 |
| `model.visual.blocks.14.norm2.weight` | BF16 | [1152] | 2,304 | 5,298,992,672 | 1568 | 32 |
| `model.visual.blocks.15.attn.proj.bias` | BF16 | [1152] | 2,304 | 5,298,994,976 | 3872 | 288 |
| `model.visual.blocks.15.attn.proj.weight` | BF16 | [1152, 1152] | 2,654,208 | 5,298,997,280 | 2080 | 32 |
| `model.visual.blocks.15.attn.qkv.bias` | BF16 | [3456] | 6,912 | 5,301,651,488 | 2080 | 32 |
| `model.visual.blocks.15.attn.qkv.weight` | BF16 | [3456, 1152] | 7,962,624 | 5,301,658,400 | 800 | 288 |
| `model.visual.blocks.15.mlp.linear_fc1.bias` | BF16 | [4304] | 8,608 | 5,309,621,024 | 800 | 288 |
| `model.visual.blocks.15.mlp.linear_fc1.weight` | BF16 | [4304, 1152] | 9,916,416 | 5,309,629,632 | 1216 | 192 |
| `model.visual.blocks.15.mlp.linear_fc2.bias` | BF16 | [1152] | 2,304 | 5,319,546,048 | 1216 | 192 |
| `model.visual.blocks.15.mlp.linear_fc2.weight` | BF16 | [1152, 4304] | 9,916,416 | 5,319,548,352 | 3520 | 448 |
| `model.visual.blocks.15.norm1.bias` | BF16 | [1152] | 2,304 | 5,329,464,768 | 3520 | 448 |
| `model.visual.blocks.15.norm1.weight` | BF16 | [1152] | 2,304 | 5,329,467,072 | 1728 | 192 |
| `model.visual.blocks.15.norm2.bias` | BF16 | [1152] | 2,304 | 5,329,469,376 | 4032 | 448 |
| `model.visual.blocks.15.norm2.weight` | BF16 | [1152] | 2,304 | 5,329,471,680 | 2240 | 192 |
| `model.visual.blocks.16.attn.proj.bias` | BF16 | [1152] | 2,304 | 5,329,473,984 | 448 | 448 |
| `model.visual.blocks.16.attn.proj.weight` | BF16 | [1152, 1152] | 2,654,208 | 5,329,476,288 | 2752 | 192 |
| `model.visual.blocks.16.attn.qkv.bias` | BF16 | [3456] | 6,912 | 5,332,130,496 | 2752 | 192 |
| `model.visual.blocks.16.attn.qkv.weight` | BF16 | [3456, 1152] | 7,962,624 | 5,332,137,408 | 1472 | 448 |
| `model.visual.blocks.16.mlp.linear_fc1.bias` | BF16 | [4304] | 8,608 | 5,340,100,032 | 1472 | 448 |
| `model.visual.blocks.16.mlp.linear_fc1.weight` | BF16 | [4304, 1152] | 9,916,416 | 5,340,108,640 | 1888 | 352 |
| `model.visual.blocks.16.mlp.linear_fc2.bias` | BF16 | [1152] | 2,304 | 5,350,025,056 | 1888 | 352 |
| `model.visual.blocks.16.mlp.linear_fc2.weight` | BF16 | [1152, 4304] | 9,916,416 | 5,350,027,360 | 96 | 96 |
| `model.visual.blocks.16.norm1.bias` | BF16 | [1152] | 2,304 | 5,359,943,776 | 96 | 96 |
| `model.visual.blocks.16.norm1.weight` | BF16 | [1152] | 2,304 | 5,359,946,080 | 2400 | 352 |
| `model.visual.blocks.16.norm2.bias` | BF16 | [1152] | 2,304 | 5,359,948,384 | 608 | 96 |
| `model.visual.blocks.16.norm2.weight` | BF16 | [1152] | 2,304 | 5,359,950,688 | 2912 | 352 |
| `model.visual.blocks.17.attn.proj.bias` | BF16 | [1152] | 2,304 | 5,359,952,992 | 1120 | 96 |
| `model.visual.blocks.17.attn.proj.weight` | BF16 | [1152, 1152] | 2,654,208 | 5,359,955,296 | 3424 | 352 |
| `model.visual.blocks.17.attn.qkv.bias` | BF16 | [3456] | 6,912 | 5,362,609,504 | 3424 | 352 |
| `model.visual.blocks.17.attn.qkv.weight` | BF16 | [3456, 1152] | 7,962,624 | 5,362,616,416 | 2144 | 96 |
| `model.visual.blocks.17.mlp.linear_fc1.bias` | BF16 | [4304] | 8,608 | 5,370,579,040 | 2144 | 96 |
| `model.visual.blocks.17.mlp.linear_fc1.weight` | BF16 | [4304, 1152] | 9,916,416 | 5,370,587,648 | 2560 | 0 |
| `model.visual.blocks.17.mlp.linear_fc2.bias` | BF16 | [1152] | 2,304 | 5,380,504,064 | 2560 | 0 |
| `model.visual.blocks.17.mlp.linear_fc2.weight` | BF16 | [1152, 4304] | 9,916,416 | 5,380,506,368 | 768 | 256 |
| `model.visual.blocks.17.norm1.bias` | BF16 | [1152] | 2,304 | 5,390,422,784 | 768 | 256 |
| `model.visual.blocks.17.norm1.weight` | BF16 | [1152] | 2,304 | 5,390,425,088 | 3072 | 0 |
| `model.visual.blocks.17.norm2.bias` | BF16 | [1152] | 2,304 | 5,390,427,392 | 1280 | 256 |
| `model.visual.blocks.17.norm2.weight` | BF16 | [1152] | 2,304 | 5,390,429,696 | 3584 | 0 |
| `model.visual.blocks.18.attn.proj.bias` | BF16 | [1152] | 2,304 | 5,390,432,000 | 1792 | 256 |
| `model.visual.blocks.18.attn.proj.weight` | BF16 | [1152, 1152] | 2,654,208 | 5,390,434,304 | 0 | 0 |
| `model.visual.blocks.18.attn.qkv.bias` | BF16 | [3456] | 6,912 | 5,393,088,512 | 0 | 0 |
| `model.visual.blocks.18.attn.qkv.weight` | BF16 | [3456, 1152] | 7,962,624 | 5,393,095,424 | 2816 | 256 |
| `model.visual.blocks.18.mlp.linear_fc1.bias` | BF16 | [4304] | 8,608 | 5,401,058,048 | 2816 | 256 |
| `model.visual.blocks.18.mlp.linear_fc1.weight` | BF16 | [4304, 1152] | 9,916,416 | 5,401,066,656 | 3232 | 160 |
| `model.visual.blocks.18.mlp.linear_fc2.bias` | BF16 | [1152] | 2,304 | 5,410,983,072 | 3232 | 160 |
| `model.visual.blocks.18.mlp.linear_fc2.weight` | BF16 | [1152, 4304] | 9,916,416 | 5,410,985,376 | 1440 | 416 |
| `model.visual.blocks.18.norm1.bias` | BF16 | [1152] | 2,304 | 5,420,901,792 | 1440 | 416 |
| `model.visual.blocks.18.norm1.weight` | BF16 | [1152] | 2,304 | 5,420,904,096 | 3744 | 160 |
| `model.visual.blocks.18.norm2.bias` | BF16 | [1152] | 2,304 | 5,420,906,400 | 1952 | 416 |
| `model.visual.blocks.18.norm2.weight` | BF16 | [1152] | 2,304 | 5,420,908,704 | 160 | 160 |
| `model.visual.blocks.19.attn.proj.bias` | BF16 | [1152] | 2,304 | 5,420,911,008 | 2464 | 416 |
| `model.visual.blocks.19.attn.proj.weight` | BF16 | [1152, 1152] | 2,654,208 | 5,420,913,312 | 672 | 160 |
| `model.visual.blocks.19.attn.qkv.bias` | BF16 | [3456] | 6,912 | 5,423,567,520 | 672 | 160 |
| `model.visual.blocks.19.attn.qkv.weight` | BF16 | [3456, 1152] | 7,962,624 | 5,423,574,432 | 3488 | 416 |
| `model.visual.blocks.19.mlp.linear_fc1.bias` | BF16 | [4304] | 8,608 | 5,431,537,056 | 3488 | 416 |
| `model.visual.blocks.19.mlp.linear_fc1.weight` | BF16 | [4304, 1152] | 9,916,416 | 5,431,545,664 | 3904 | 320 |
| `model.visual.blocks.19.mlp.linear_fc2.bias` | BF16 | [1152] | 2,304 | 5,441,462,080 | 3904 | 320 |
| `model.visual.blocks.19.mlp.linear_fc2.weight` | BF16 | [1152, 4304] | 9,916,416 | 5,441,464,384 | 2112 | 64 |
| `model.visual.blocks.19.norm1.bias` | BF16 | [1152] | 2,304 | 5,451,380,800 | 2112 | 64 |
| `model.visual.blocks.19.norm1.weight` | BF16 | [1152] | 2,304 | 5,451,383,104 | 320 | 320 |
| `model.visual.blocks.19.norm2.bias` | BF16 | [1152] | 2,304 | 5,451,385,408 | 2624 | 64 |
| `model.visual.blocks.19.norm2.weight` | BF16 | [1152] | 2,304 | 5,451,387,712 | 832 | 320 |
| `model.visual.blocks.2.attn.proj.bias` | BF16 | [1152] | 2,304 | 5,451,390,016 | 3136 | 64 |
| `model.visual.blocks.2.attn.proj.weight` | BF16 | [1152, 1152] | 2,654,208 | 5,451,392,320 | 1344 | 320 |
| `model.visual.blocks.2.attn.qkv.bias` | BF16 | [3456] | 6,912 | 5,454,046,528 | 1344 | 320 |
| `model.visual.blocks.2.attn.qkv.weight` | BF16 | [3456, 1152] | 7,962,624 | 5,454,053,440 | 64 | 64 |
| `model.visual.blocks.2.mlp.linear_fc1.bias` | BF16 | [4304] | 8,608 | 5,462,016,064 | 64 | 64 |
| `model.visual.blocks.2.mlp.linear_fc1.weight` | BF16 | [4304, 1152] | 9,916,416 | 5,462,024,672 | 480 | 480 |
| `model.visual.blocks.2.mlp.linear_fc2.bias` | BF16 | [1152] | 2,304 | 5,471,941,088 | 480 | 480 |
| `model.visual.blocks.2.mlp.linear_fc2.weight` | BF16 | [1152, 4304] | 9,916,416 | 5,471,943,392 | 2784 | 224 |
| `model.visual.blocks.2.norm1.bias` | BF16 | [1152] | 2,304 | 5,481,859,808 | 2784 | 224 |
| `model.visual.blocks.2.norm1.weight` | BF16 | [1152] | 2,304 | 5,481,862,112 | 992 | 480 |
| `model.visual.blocks.2.norm2.bias` | BF16 | [1152] | 2,304 | 5,481,864,416 | 3296 | 224 |
| `model.visual.blocks.2.norm2.weight` | BF16 | [1152] | 2,304 | 5,481,866,720 | 1504 | 480 |
| `model.visual.blocks.20.attn.proj.bias` | BF16 | [1152] | 2,304 | 5,481,869,024 | 3808 | 224 |
| `model.visual.blocks.20.attn.proj.weight` | BF16 | [1152, 1152] | 2,654,208 | 5,481,871,328 | 2016 | 480 |
| `model.visual.blocks.20.attn.qkv.bias` | BF16 | [3456] | 6,912 | 5,484,525,536 | 2016 | 480 |
| `model.visual.blocks.20.attn.qkv.weight` | BF16 | [3456, 1152] | 7,962,624 | 5,484,532,448 | 736 | 224 |
| `model.visual.blocks.20.mlp.linear_fc1.bias` | BF16 | [4304] | 8,608 | 5,492,495,072 | 736 | 224 |
| `model.visual.blocks.20.mlp.linear_fc1.weight` | BF16 | [4304, 1152] | 9,916,416 | 5,492,503,680 | 1152 | 128 |
| `model.visual.blocks.20.mlp.linear_fc2.bias` | BF16 | [1152] | 2,304 | 5,502,420,096 | 1152 | 128 |
| `model.visual.blocks.20.mlp.linear_fc2.weight` | BF16 | [1152, 4304] | 9,916,416 | 5,502,422,400 | 3456 | 384 |
| `model.visual.blocks.20.norm1.bias` | BF16 | [1152] | 2,304 | 5,512,338,816 | 3456 | 384 |
| `model.visual.blocks.20.norm1.weight` | BF16 | [1152] | 2,304 | 5,512,341,120 | 1664 | 128 |
| `model.visual.blocks.20.norm2.bias` | BF16 | [1152] | 2,304 | 5,512,343,424 | 3968 | 384 |
| `model.visual.blocks.20.norm2.weight` | BF16 | [1152] | 2,304 | 5,512,345,728 | 2176 | 128 |
| `model.visual.blocks.21.attn.proj.bias` | BF16 | [1152] | 2,304 | 5,512,348,032 | 384 | 384 |
| `model.visual.blocks.21.attn.proj.weight` | BF16 | [1152, 1152] | 2,654,208 | 5,512,350,336 | 2688 | 128 |
| `model.visual.blocks.21.attn.qkv.bias` | BF16 | [3456] | 6,912 | 5,515,004,544 | 2688 | 128 |
| `model.visual.blocks.21.attn.qkv.weight` | BF16 | [3456, 1152] | 7,962,624 | 5,515,011,456 | 1408 | 384 |
| `model.visual.blocks.21.mlp.linear_fc1.bias` | BF16 | [4304] | 8,608 | 5,522,974,080 | 1408 | 384 |
| `model.visual.blocks.21.mlp.linear_fc1.weight` | BF16 | [4304, 1152] | 9,916,416 | 5,522,982,688 | 1824 | 288 |
| `model.visual.blocks.21.mlp.linear_fc2.bias` | BF16 | [1152] | 2,304 | 5,532,899,104 | 1824 | 288 |
| `model.visual.blocks.21.mlp.linear_fc2.weight` | BF16 | [1152, 4304] | 9,916,416 | 5,532,901,408 | 32 | 32 |
| `model.visual.blocks.21.norm1.bias` | BF16 | [1152] | 2,304 | 5,542,817,824 | 32 | 32 |
| `model.visual.blocks.21.norm1.weight` | BF16 | [1152] | 2,304 | 5,542,820,128 | 2336 | 288 |
| `model.visual.blocks.21.norm2.bias` | BF16 | [1152] | 2,304 | 5,542,822,432 | 544 | 32 |
| `model.visual.blocks.21.norm2.weight` | BF16 | [1152] | 2,304 | 5,542,824,736 | 2848 | 288 |
| `model.visual.blocks.22.attn.proj.bias` | BF16 | [1152] | 2,304 | 5,542,827,040 | 1056 | 32 |
| `model.visual.blocks.22.attn.proj.weight` | BF16 | [1152, 1152] | 2,654,208 | 5,542,829,344 | 3360 | 288 |
| `model.visual.blocks.22.attn.qkv.bias` | BF16 | [3456] | 6,912 | 5,545,483,552 | 3360 | 288 |
| `model.visual.blocks.22.attn.qkv.weight` | BF16 | [3456, 1152] | 7,962,624 | 5,545,490,464 | 2080 | 32 |
| `model.visual.blocks.22.mlp.linear_fc1.bias` | BF16 | [4304] | 8,608 | 5,553,453,088 | 2080 | 32 |
| `model.visual.blocks.22.mlp.linear_fc1.weight` | BF16 | [4304, 1152] | 9,916,416 | 5,553,461,696 | 2496 | 448 |
| `model.visual.blocks.22.mlp.linear_fc2.bias` | BF16 | [1152] | 2,304 | 5,563,378,112 | 2496 | 448 |
| `model.visual.blocks.22.mlp.linear_fc2.weight` | BF16 | [1152, 4304] | 9,916,416 | 5,563,380,416 | 704 | 192 |
| `model.visual.blocks.22.norm1.bias` | BF16 | [1152] | 2,304 | 5,573,296,832 | 704 | 192 |
| `model.visual.blocks.22.norm1.weight` | BF16 | [1152] | 2,304 | 5,573,299,136 | 3008 | 448 |
| `model.visual.blocks.22.norm2.bias` | BF16 | [1152] | 2,304 | 5,573,301,440 | 1216 | 192 |
| `model.visual.blocks.22.norm2.weight` | BF16 | [1152] | 2,304 | 5,573,303,744 | 3520 | 448 |
| `model.visual.blocks.23.attn.proj.bias` | BF16 | [1152] | 2,304 | 5,573,306,048 | 1728 | 192 |
| `model.visual.blocks.23.attn.proj.weight` | BF16 | [1152, 1152] | 2,654,208 | 5,573,308,352 | 4032 | 448 |
| `model.visual.blocks.23.attn.qkv.bias` | BF16 | [3456] | 6,912 | 5,575,962,560 | 4032 | 448 |
| `model.visual.blocks.23.attn.qkv.weight` | BF16 | [3456, 1152] | 7,962,624 | 5,575,969,472 | 2752 | 192 |
| `model.visual.blocks.23.mlp.linear_fc1.bias` | BF16 | [4304] | 8,608 | 5,583,932,096 | 2752 | 192 |
| `model.visual.blocks.23.mlp.linear_fc1.weight` | BF16 | [4304, 1152] | 9,916,416 | 5,583,940,704 | 3168 | 96 |
| `model.visual.blocks.23.mlp.linear_fc2.bias` | BF16 | [1152] | 2,304 | 5,593,857,120 | 3168 | 96 |
| `model.visual.blocks.23.mlp.linear_fc2.weight` | BF16 | [1152, 4304] | 9,916,416 | 5,593,859,424 | 1376 | 352 |
| `model.visual.blocks.23.norm1.bias` | BF16 | [1152] | 2,304 | 5,603,775,840 | 1376 | 352 |
| `model.visual.blocks.23.norm1.weight` | BF16 | [1152] | 2,304 | 5,603,778,144 | 3680 | 96 |
| `model.visual.blocks.23.norm2.bias` | BF16 | [1152] | 2,304 | 5,603,780,448 | 1888 | 352 |
| `model.visual.blocks.23.norm2.weight` | BF16 | [1152] | 2,304 | 5,603,782,752 | 96 | 96 |
| `model.visual.blocks.24.attn.proj.bias` | BF16 | [1152] | 2,304 | 5,603,785,056 | 2400 | 352 |
| `model.visual.blocks.24.attn.proj.weight` | BF16 | [1152, 1152] | 2,654,208 | 5,603,787,360 | 608 | 96 |
| `model.visual.blocks.24.attn.qkv.bias` | BF16 | [3456] | 6,912 | 5,606,441,568 | 608 | 96 |
| `model.visual.blocks.24.attn.qkv.weight` | BF16 | [3456, 1152] | 7,962,624 | 5,606,448,480 | 3424 | 352 |
| `model.visual.blocks.24.mlp.linear_fc1.bias` | BF16 | [4304] | 8,608 | 5,614,411,104 | 3424 | 352 |
| `model.visual.blocks.24.mlp.linear_fc1.weight` | BF16 | [4304, 1152] | 9,916,416 | 5,614,419,712 | 3840 | 256 |
| `model.visual.blocks.24.mlp.linear_fc2.bias` | BF16 | [1152] | 2,304 | 5,624,336,128 | 3840 | 256 |
| `model.visual.blocks.24.mlp.linear_fc2.weight` | BF16 | [1152, 4304] | 9,916,416 | 5,624,338,432 | 2048 | 0 |
| `model.visual.blocks.24.norm1.bias` | BF16 | [1152] | 2,304 | 5,634,254,848 | 2048 | 0 |
| `model.visual.blocks.24.norm1.weight` | BF16 | [1152] | 2,304 | 5,634,257,152 | 256 | 256 |
| `model.visual.blocks.24.norm2.bias` | BF16 | [1152] | 2,304 | 5,634,259,456 | 2560 | 0 |
| `model.visual.blocks.24.norm2.weight` | BF16 | [1152] | 2,304 | 5,634,261,760 | 768 | 256 |
| `model.visual.blocks.25.attn.proj.bias` | BF16 | [1152] | 2,304 | 5,634,264,064 | 3072 | 0 |
| `model.visual.blocks.25.attn.proj.weight` | BF16 | [1152, 1152] | 2,654,208 | 5,634,266,368 | 1280 | 256 |
| `model.visual.blocks.25.attn.qkv.bias` | BF16 | [3456] | 6,912 | 5,636,920,576 | 1280 | 256 |
| `model.visual.blocks.25.attn.qkv.weight` | BF16 | [3456, 1152] | 7,962,624 | 5,636,927,488 | 0 | 0 |
| `model.visual.blocks.25.mlp.linear_fc1.bias` | BF16 | [4304] | 8,608 | 5,644,890,112 | 0 | 0 |
| `model.visual.blocks.25.mlp.linear_fc1.weight` | BF16 | [4304, 1152] | 9,916,416 | 5,644,898,720 | 416 | 416 |
| `model.visual.blocks.25.mlp.linear_fc2.bias` | BF16 | [1152] | 2,304 | 5,654,815,136 | 416 | 416 |
| `model.visual.blocks.25.mlp.linear_fc2.weight` | BF16 | [1152, 4304] | 9,916,416 | 5,654,817,440 | 2720 | 160 |
| `model.visual.blocks.25.norm1.bias` | BF16 | [1152] | 2,304 | 5,664,733,856 | 2720 | 160 |
| `model.visual.blocks.25.norm1.weight` | BF16 | [1152] | 2,304 | 5,664,736,160 | 928 | 416 |
| `model.visual.blocks.25.norm2.bias` | BF16 | [1152] | 2,304 | 5,664,738,464 | 3232 | 160 |
| `model.visual.blocks.25.norm2.weight` | BF16 | [1152] | 2,304 | 5,664,740,768 | 1440 | 416 |
| `model.visual.blocks.26.attn.proj.bias` | BF16 | [1152] | 2,304 | 5,664,743,072 | 3744 | 160 |
| `model.visual.blocks.26.attn.proj.weight` | BF16 | [1152, 1152] | 2,654,208 | 5,664,745,376 | 1952 | 416 |
| `model.visual.blocks.26.attn.qkv.bias` | BF16 | [3456] | 6,912 | 5,667,399,584 | 1952 | 416 |
| `model.visual.blocks.26.attn.qkv.weight` | BF16 | [3456, 1152] | 7,962,624 | 5,667,406,496 | 672 | 160 |
| `model.visual.blocks.26.mlp.linear_fc1.bias` | BF16 | [4304] | 8,608 | 5,675,369,120 | 672 | 160 |
| `model.visual.blocks.26.mlp.linear_fc1.weight` | BF16 | [4304, 1152] | 9,916,416 | 5,675,377,728 | 1088 | 64 |
| `model.visual.blocks.26.mlp.linear_fc2.bias` | BF16 | [1152] | 2,304 | 5,685,294,144 | 1088 | 64 |
| `model.visual.blocks.26.mlp.linear_fc2.weight` | BF16 | [1152, 4304] | 9,916,416 | 5,685,296,448 | 3392 | 320 |
| `model.visual.blocks.26.norm1.bias` | BF16 | [1152] | 2,304 | 5,695,212,864 | 3392 | 320 |
| `model.visual.blocks.26.norm1.weight` | BF16 | [1152] | 2,304 | 5,695,215,168 | 1600 | 64 |
| `model.visual.blocks.26.norm2.bias` | BF16 | [1152] | 2,304 | 5,695,217,472 | 3904 | 320 |
| `model.visual.blocks.26.norm2.weight` | BF16 | [1152] | 2,304 | 5,695,219,776 | 2112 | 64 |
| `model.visual.blocks.3.attn.proj.bias` | BF16 | [1152] | 2,304 | 5,695,222,080 | 320 | 320 |
| `model.visual.blocks.3.attn.proj.weight` | BF16 | [1152, 1152] | 2,654,208 | 5,695,224,384 | 2624 | 64 |
| `model.visual.blocks.3.attn.qkv.bias` | BF16 | [3456] | 6,912 | 5,697,878,592 | 2624 | 64 |
| `model.visual.blocks.3.attn.qkv.weight` | BF16 | [3456, 1152] | 7,962,624 | 5,697,885,504 | 1344 | 320 |
| `model.visual.blocks.3.mlp.linear_fc1.bias` | BF16 | [4304] | 8,608 | 5,705,848,128 | 1344 | 320 |
| `model.visual.blocks.3.mlp.linear_fc1.weight` | BF16 | [4304, 1152] | 9,916,416 | 5,705,856,736 | 1760 | 224 |
| `model.visual.blocks.3.mlp.linear_fc2.bias` | BF16 | [1152] | 2,304 | 5,715,773,152 | 1760 | 224 |
| `model.visual.blocks.3.mlp.linear_fc2.weight` | BF16 | [1152, 4304] | 9,916,416 | 5,715,775,456 | 4064 | 480 |
| `model.visual.blocks.3.norm1.bias` | BF16 | [1152] | 2,304 | 5,725,691,872 | 4064 | 480 |
| `model.visual.blocks.3.norm1.weight` | BF16 | [1152] | 2,304 | 5,725,694,176 | 2272 | 224 |
| `model.visual.blocks.3.norm2.bias` | BF16 | [1152] | 2,304 | 5,725,696,480 | 480 | 480 |
| `model.visual.blocks.3.norm2.weight` | BF16 | [1152] | 2,304 | 5,725,698,784 | 2784 | 224 |
| `model.visual.blocks.4.attn.proj.bias` | BF16 | [1152] | 2,304 | 5,725,701,088 | 992 | 480 |
| `model.visual.blocks.4.attn.proj.weight` | BF16 | [1152, 1152] | 2,654,208 | 5,725,703,392 | 3296 | 224 |
| `model.visual.blocks.4.attn.qkv.bias` | BF16 | [3456] | 6,912 | 5,728,357,600 | 3296 | 224 |
| `model.visual.blocks.4.attn.qkv.weight` | BF16 | [3456, 1152] | 7,962,624 | 5,728,364,512 | 2016 | 480 |
| `model.visual.blocks.4.mlp.linear_fc1.bias` | BF16 | [4304] | 8,608 | 5,736,327,136 | 2016 | 480 |
| `model.visual.blocks.4.mlp.linear_fc1.weight` | BF16 | [4304, 1152] | 9,916,416 | 5,736,335,744 | 2432 | 384 |
| `model.visual.blocks.4.mlp.linear_fc2.bias` | BF16 | [1152] | 2,304 | 5,746,252,160 | 2432 | 384 |
| `model.visual.blocks.4.mlp.linear_fc2.weight` | BF16 | [1152, 4304] | 9,916,416 | 5,746,254,464 | 640 | 128 |
| `model.visual.blocks.4.norm1.bias` | BF16 | [1152] | 2,304 | 5,756,170,880 | 640 | 128 |
| `model.visual.blocks.4.norm1.weight` | BF16 | [1152] | 2,304 | 5,756,173,184 | 2944 | 384 |
| `model.visual.blocks.4.norm2.bias` | BF16 | [1152] | 2,304 | 5,756,175,488 | 1152 | 128 |
| `model.visual.blocks.4.norm2.weight` | BF16 | [1152] | 2,304 | 5,756,177,792 | 3456 | 384 |
| `model.visual.blocks.5.attn.proj.bias` | BF16 | [1152] | 2,304 | 5,756,180,096 | 1664 | 128 |
| `model.visual.blocks.5.attn.proj.weight` | BF16 | [1152, 1152] | 2,654,208 | 5,756,182,400 | 3968 | 384 |
| `model.visual.blocks.5.attn.qkv.bias` | BF16 | [3456] | 6,912 | 5,758,836,608 | 3968 | 384 |
| `model.visual.blocks.5.attn.qkv.weight` | BF16 | [3456, 1152] | 7,962,624 | 5,758,843,520 | 2688 | 128 |
| `model.visual.blocks.5.mlp.linear_fc1.bias` | BF16 | [4304] | 8,608 | 5,766,806,144 | 2688 | 128 |
| `model.visual.blocks.5.mlp.linear_fc1.weight` | BF16 | [4304, 1152] | 9,916,416 | 5,766,814,752 | 3104 | 32 |
| `model.visual.blocks.5.mlp.linear_fc2.bias` | BF16 | [1152] | 2,304 | 5,776,731,168 | 3104 | 32 |
| `model.visual.blocks.5.mlp.linear_fc2.weight` | BF16 | [1152, 4304] | 9,916,416 | 5,776,733,472 | 1312 | 288 |
| `model.visual.blocks.5.norm1.bias` | BF16 | [1152] | 2,304 | 5,786,649,888 | 1312 | 288 |
| `model.visual.blocks.5.norm1.weight` | BF16 | [1152] | 2,304 | 5,786,652,192 | 3616 | 32 |
| `model.visual.blocks.5.norm2.bias` | BF16 | [1152] | 2,304 | 5,786,654,496 | 1824 | 288 |
| `model.visual.blocks.5.norm2.weight` | BF16 | [1152] | 2,304 | 5,786,656,800 | 32 | 32 |
| `model.visual.blocks.6.attn.proj.bias` | BF16 | [1152] | 2,304 | 5,786,659,104 | 2336 | 288 |
| `model.visual.blocks.6.attn.proj.weight` | BF16 | [1152, 1152] | 2,654,208 | 5,786,661,408 | 544 | 32 |
| `model.visual.blocks.6.attn.qkv.bias` | BF16 | [3456] | 6,912 | 5,789,315,616 | 544 | 32 |
| `model.visual.blocks.6.attn.qkv.weight` | BF16 | [3456, 1152] | 7,962,624 | 5,789,322,528 | 3360 | 288 |
| `model.visual.blocks.6.mlp.linear_fc1.bias` | BF16 | [4304] | 8,608 | 5,797,285,152 | 3360 | 288 |
| `model.visual.blocks.6.mlp.linear_fc1.weight` | BF16 | [4304, 1152] | 9,916,416 | 5,797,293,760 | 3776 | 192 |
| `model.visual.blocks.6.mlp.linear_fc2.bias` | BF16 | [1152] | 2,304 | 5,807,210,176 | 3776 | 192 |
| `model.visual.blocks.6.mlp.linear_fc2.weight` | BF16 | [1152, 4304] | 9,916,416 | 5,807,212,480 | 1984 | 448 |
| `model.visual.blocks.6.norm1.bias` | BF16 | [1152] | 2,304 | 5,817,128,896 | 1984 | 448 |
| `model.visual.blocks.6.norm1.weight` | BF16 | [1152] | 2,304 | 5,817,131,200 | 192 | 192 |
| `model.visual.blocks.6.norm2.bias` | BF16 | [1152] | 2,304 | 5,817,133,504 | 2496 | 448 |
| `model.visual.blocks.6.norm2.weight` | BF16 | [1152] | 2,304 | 5,817,135,808 | 704 | 192 |
| `model.visual.blocks.7.attn.proj.bias` | BF16 | [1152] | 2,304 | 5,817,138,112 | 3008 | 448 |
| `model.visual.blocks.7.attn.proj.weight` | BF16 | [1152, 1152] | 2,654,208 | 5,817,140,416 | 1216 | 192 |
| `model.visual.blocks.7.attn.qkv.bias` | BF16 | [3456] | 6,912 | 5,819,794,624 | 1216 | 192 |
| `model.visual.blocks.7.attn.qkv.weight` | BF16 | [3456, 1152] | 7,962,624 | 5,819,801,536 | 4032 | 448 |
| `model.visual.blocks.7.mlp.linear_fc1.bias` | BF16 | [4304] | 8,608 | 5,827,764,160 | 4032 | 448 |
| `model.visual.blocks.7.mlp.linear_fc1.weight` | BF16 | [4304, 1152] | 9,916,416 | 5,827,772,768 | 352 | 352 |
| `model.visual.blocks.7.mlp.linear_fc2.bias` | BF16 | [1152] | 2,304 | 5,837,689,184 | 352 | 352 |
| `model.visual.blocks.7.mlp.linear_fc2.weight` | BF16 | [1152, 4304] | 9,916,416 | 5,837,691,488 | 2656 | 96 |
| `model.visual.blocks.7.norm1.bias` | BF16 | [1152] | 2,304 | 5,847,607,904 | 2656 | 96 |
| `model.visual.blocks.7.norm1.weight` | BF16 | [1152] | 2,304 | 5,847,610,208 | 864 | 352 |
| `model.visual.blocks.7.norm2.bias` | BF16 | [1152] | 2,304 | 5,847,612,512 | 3168 | 96 |
| `model.visual.blocks.7.norm2.weight` | BF16 | [1152] | 2,304 | 5,847,614,816 | 1376 | 352 |
| `model.visual.blocks.8.attn.proj.bias` | BF16 | [1152] | 2,304 | 5,847,617,120 | 3680 | 96 |
| `model.visual.blocks.8.attn.proj.weight` | BF16 | [1152, 1152] | 2,654,208 | 5,847,619,424 | 1888 | 352 |
| `model.visual.blocks.8.attn.qkv.bias` | BF16 | [3456] | 6,912 | 5,850,273,632 | 1888 | 352 |
| `model.visual.blocks.8.attn.qkv.weight` | BF16 | [3456, 1152] | 7,962,624 | 5,850,280,544 | 608 | 96 |
| `model.visual.blocks.8.mlp.linear_fc1.bias` | BF16 | [4304] | 8,608 | 5,858,243,168 | 608 | 96 |
| `model.visual.blocks.8.mlp.linear_fc1.weight` | BF16 | [4304, 1152] | 9,916,416 | 5,858,251,776 | 1024 | 0 |
| `model.visual.blocks.8.mlp.linear_fc2.bias` | BF16 | [1152] | 2,304 | 5,868,168,192 | 1024 | 0 |
| `model.visual.blocks.8.mlp.linear_fc2.weight` | BF16 | [1152, 4304] | 9,916,416 | 5,868,170,496 | 3328 | 256 |
| `model.visual.blocks.8.norm1.bias` | BF16 | [1152] | 2,304 | 5,878,086,912 | 3328 | 256 |
| `model.visual.blocks.8.norm1.weight` | BF16 | [1152] | 2,304 | 5,878,089,216 | 1536 | 0 |
| `model.visual.blocks.8.norm2.bias` | BF16 | [1152] | 2,304 | 5,878,091,520 | 3840 | 256 |
| `model.visual.blocks.8.norm2.weight` | BF16 | [1152] | 2,304 | 5,878,093,824 | 2048 | 0 |
| `model.visual.blocks.9.attn.proj.bias` | BF16 | [1152] | 2,304 | 5,878,096,128 | 256 | 256 |
| `model.visual.blocks.9.attn.proj.weight` | BF16 | [1152, 1152] | 2,654,208 | 5,878,098,432 | 2560 | 0 |
| `model.visual.blocks.9.attn.qkv.bias` | BF16 | [3456] | 6,912 | 5,880,752,640 | 2560 | 0 |
| `model.visual.blocks.9.attn.qkv.weight` | BF16 | [3456, 1152] | 7,962,624 | 5,880,759,552 | 1280 | 256 |
| `model.visual.blocks.9.mlp.linear_fc1.bias` | BF16 | [4304] | 8,608 | 5,888,722,176 | 1280 | 256 |
| `model.visual.blocks.9.mlp.linear_fc1.weight` | BF16 | [4304, 1152] | 9,916,416 | 5,888,730,784 | 1696 | 160 |
| `model.visual.blocks.9.mlp.linear_fc2.bias` | BF16 | [1152] | 2,304 | 5,898,647,200 | 1696 | 160 |
| `model.visual.blocks.9.mlp.linear_fc2.weight` | BF16 | [1152, 4304] | 9,916,416 | 5,898,649,504 | 4000 | 416 |
| `model.visual.blocks.9.norm1.bias` | BF16 | [1152] | 2,304 | 5,908,565,920 | 4000 | 416 |
| `model.visual.blocks.9.norm1.weight` | BF16 | [1152] | 2,304 | 5,908,568,224 | 2208 | 160 |
| `model.visual.blocks.9.norm2.bias` | BF16 | [1152] | 2,304 | 5,908,570,528 | 416 | 416 |
| `model.visual.blocks.9.norm2.weight` | BF16 | [1152] | 2,304 | 5,908,572,832 | 2720 | 160 |
| `model.visual.merger.linear_fc1.bias` | BF16 | [4608] | 9,216 | 5,908,575,136 | 928 | 416 |
| `model.visual.merger.linear_fc1.weight` | BF16 | [4608, 4608] | 42,467,328 | 5,908,584,352 | 1952 | 416 |
| `model.visual.merger.linear_fc2.bias` | BF16 | [5120] | 10,240 | 5,951,051,680 | 1952 | 416 |
| `model.visual.merger.linear_fc2.weight` | BF16 | [5120, 4608] | 47,185,920 | 5,951,061,920 | 4000 | 416 |
| `model.visual.merger.norm.bias` | BF16 | [1152] | 2,304 | 5,998,247,840 | 4000 | 416 |
| `model.visual.merger.norm.weight` | BF16 | [1152] | 2,304 | 5,998,250,144 | 2208 | 160 |
| `model.visual.patch_embed.proj.bias` | BF16 | [1152] | 2,304 | 5,998,252,448 | 416 | 416 |
| `model.visual.patch_embed.proj.weight` | BF16 | [1152, 3, 2, 16, 16] | 3,538,944 | 5,998,254,752 | 2720 | 160 |
| `model.visual.pos_embed.weight` | BF16 | [2304, 1152] | 5,308,416 | 6,001,793,696 | 2720 | 160 |

## 3. Integrity (crc32.txt, README.md, md5)

- `crc32.txt` lists 77 files with standard CRC-32 (zlib/IEEE) hex values; **spot-checking works**: verified `layers-0.safetensors` -> e3a49ca5 MATCH and `mtp.safetensors` -> 837f3352 MATCH, then swept **all 77 entries**.
- Result: **all 66 safetensors shards match** — per-shard weight integrity is fully provable (each shard 0.35-6 GB, streams through zlib.crc32 quickly).
- 9 mismatches, all text files: chat_template.jinja, config.json, generation_config.json, merges.txt, model.safetensors.index.json, preprocessor_config.json, tokenizer_config.json, video_preprocessor_config.json, vocab.json. Cause: `.gitattributes` keeps `*.safetensors` + `tokenizer.json` as LFS/binary while the text files were checked out with CRLF normalization, so their bytes differ from the uploader's LF copies. Harmless for weights.
- `safetensors-md5sum.txt` is **0 bytes** (placeholder; no md5s exist).
- README.md documents "fine-grained fp8 quantization with block size of 128" and 262,144 native context; it contains **no checksum instructions** — crc32.txt is the only integrity source.
- One-liner spot check (Git Bash): `python -c "import zlib,sys;f=open(sys.argv[1],'rb');z=0
while 1:
 b=f.read(1<<24)
 if not b:break
 z=zlib.crc32(b,z)
print('%08x'%z)" layers-42.safetensors` then compare to crc32.txt (layers-42 = 5b2dcfe4).

## 4. Alignment audit (O_DIRECT readiness)

1. **Starts are essentially never aligned**: 45/1606 tensors start 4096-aligned, 144/1606 512-aligned. Residues mod 512 spread across 30 distinct values (top: 128 x298, 160 x245, 256 x224).
2. **Sizes are perfectly aligned**: all 407 F8 tensor sizes are multiples of 4096; every tensor >= 1 MiB (407 F8 + 115 BF16) has size % 4096 == 0.
3. **Zero padding anywhere**: across all 66 shards there is not a single inter-tensor gap (every tensor begins exactly at the previous tensor's end — 0 discontinuities) and every shard ends exactly at its last tensor's end (0 tail bytes). Layout per layer shard: BF16 smalls first, then F8 blocks, both alphabetically.
4. **Consequence — per-shard constant phase**: since F8 sizes are 4096-multiples and packing is gapless, *all F8 tensors in a given shard share one residue mod 4096*. The whole checkpoint has only 5 phase classes:

| shard class | first-F8 abs | residue (prefix overhang) |
|---|---|---|
| linear, 1-digit layer idx (0,1,2,4,5,6,8,9) | 1,135,208 | 616 |
| linear, 2-digit idx (40 shards) | 1,135,232 | 640 |
| full-attn, 1-digit idx (3,7) | 69,264 | 3728 |
| full-attn, 2-digit idx (14 shards) | 69,280 | 3744 |
| mtp | 104,957,744 | 1840 |

5. **O_DIRECT verdict**: raw tensor addresses cannot be handed to `ReadFile(FILE_FLAG_NO_BUFFERING)` (needs sector-aligned offset, length, and destination). But per (2)+(4) each big tensor is one aligned *window*: `win_off = off & ~4095`, `win_len = bytes` when `off % 4096` is the shard phase and bytes is a 4096 multiple — the only overhang is the <=4095 B prefix (small-BF16 tail). outside.safetensors' vision region breaks the chain (2,304 B biases interleaved), so windows there are computed per tensor with ceil-rounded length. Read the window into a 4096-aligned pinned slot and expose the tensor at `slot + (off & 4095)`.
6. If repacking were ever allowed, one <=4095 B pad after each shard header (<=265 KB total across 66 files) would 4096-align *every* tensor. This audit's constraint is no checkpoint modification, so INSIDX02 stores precomputed aligned windows instead.

## 5. INSIDX02 — multi-shard index + storage spec

### 5.1 Why INSIDX01 cannot work here

- `tools/index_safetensors.py` takes **one** file and embeds one payload path; this checkpoint is 66 files.
- Its `DTYPES` map `{F32:1,BF16:2,F16:3,U8:4,U32:5,I8:6}` raises `ValueError` on `F8_E4M3` (line 29-31).
- `src/model_file.cpp` opens exactly one payload mapping; all offsets are relative to one `payload_offset_`.
- Engine lookup names (`language_model.model.*`, `.scales` u8 MXFP4) do not exist in this checkpoint; `Qwen35Weights::matrix` (src/qwen35.cu:7) throws on any non-u32 weight.
- `src/test_model.cpp` asserts 699 tensors (9B); this model indexes 1606.

### 5.2 Binary format (all little-endian, packed)

```
Header      : char magic[8] = "INSIDX02"; u32 version = 2;
              u32 shard_count; u32 tensor_count;
Shard entry : u16 path_len; u64 file_size; u64 data_start;   // 8 + safetensors header len
              u64 f8_align_base;                              // (first F8 abs & ~4095); 0 if none
              u32 crc32;                                      // from crc32.txt, verified at build
              u8  flags;                                      // bit0 has_f8, bit1 skip_vision
              u8  path[path_len];                             // UTF-8, relative to index file dir
Tensor entry: u16 name_len; u8 dtype; u8 rank; u16 shard;     // index into shard table
              u32 scale_idx;                                  // tensor-table idx of linked
              //                                            weight_scale_inv; 0xFFFFFFFF = none
              u64 off;                                        // = safetensors begin (rel. data_start)
              u64 bytes; u64 dims[rank];
              u8 name[name_len];
```

Invariants the builder must enforce (all verified true for this checkpoint):

- tensor table sorted by name (binary-search `find()` keeps its signature and O(log n));
- names stored **verbatim from the checkpoint** (`model.language_model.*`, `lm_head`, `mtp.*`) — engine name construction changes, not the data;
- for every F8 tensor, `scale_idx` resolves to the BF16 `weight_scale_inv` in the **same shard** with shape [ceil(r/128), ceil(c/128)] (407/407 hold); scale entries point back to their weight;
- per shard: `off + bytes <= file_size - data_start` (replaces the single-mapping escape check at model_file.cpp:36);
- alignment class derivable from shard `f8_align_base` + `off` (windows computed on the fly);
- dtype registry extends DType: `f8_e4m3 = 7` (reserve `f8_e5m2 = 8`).

Builder changes to `tools/index_safetensors.py`: accept a checkpoint directory (or read `model.safetensors.index.json`'s weight_map), loop shards, add `"F8_E4M3": 7`, emit shard table + scale links, verify each shard CRC32 against crc32.txt at build time, optionally flag `model.visual.*` tensors so the loader can exclude 0.86 GiB of vision from the residency budget while still indexing them.

### 5.3 ModelFile changes (src/model_file.cpp, include/insignia_model.hpp)

- `TensorView` gains `uint16_t shard`, `uint64_t off`, `uint32_t scale_idx`; `data` resolves lazily through a per-shard base (`const std::byte* shard_base(uint16_t)`).
- Constructor maps all 66 shards eagerly (`CreateFileMappingW` + `MapViewOfFile` per shard); Windows faults pages on demand, so zero-copy mmap semantics survive and ~29 GiB of VA is nothing. Keep `FILE_FLAG_RANDOM_ACCESS`; the NVMe tier is simply "pages not resident / discarded under pressure".
- `find(name)` unchanged externally. New: `const TensorView* find_linked(const TensorView&)` returning the scale or nullptr, replacing the `base+".scales"` string convention and the runtime scale-shape assertion in `matrix()`.
- The bounds check moves into the per-shard validation loop.

### 5.4 Qwen35Weights / QuantMatrix changes

- Lookup bases become `model.language_model.layers.N.*`, `model.language_model.embed_tokens`, `model.language_model.norm.weight`, `lm_head`, `mtp.*`.
- `matrix()` gains an FP8 branch: weight `f8_e4m3` + BF16 scales [ceil(r/128), ceil(c/128)]; the MXFP4 branch stays so the 9B INSIDX01 index keeps loading.
- `Qwen35Shape` constants: hidden 4096 -> 5120, intermediate 12288 -> 17408, layers 32 -> 64, vocab unchanged 248320; `full_attention(i) = (i&3)==3` already correct for 64 layers.

### 5.5 TieredStorage: host-RAM tier with pinned slots (src/storage.cu)

Today: `entries_: name -> {device, bytes, pins, tick}` with one device budget; `make_room` LRU-evicts unpinned device copies; source is always the mmap (`cudaMemcpyAsync` H2D from OS page cache). The `MemoryTier { nvme_mapped, host_pinned, device }` enum already declares the third tier — implement it:

```c++
struct HostSlot { void *pinned;        // cudaHostAlloc'd, 4096-aligned, holds the aligned window
                 u64 win_off, win_len; // [off & ~4095, ceil) inside the shard file
                 u32 shard; u64 bytes; u32 pins; u64 tick; };
struct Entry    { void *device;        // VRAM copy or null (host-resident only)
                 HostSlot *host; u64 bytes; u32 pins; u64 tick;
                 const TensorView *view; };
```

Acquire path, in order:

1. **device hit** -> `pins++`, refresh `tick`, return DeviceView (unchanged fast path);
2. **host-slot hit** -> `make_room` (device budget) -> `cudaMalloc` + `cudaMemcpyAsync` from the *pinned* window (real DMA, no page-cache bounce) -> device entry -> return;
3. **miss** -> `make_host_room(window_len)` LRU-evicts an unpinned HostSlot -> `ReadFile` with `FILE_FLAG_NO_BUFFERING` straight into `pinned` using `win_off/win_len` (offset, length and destination all 4096-multiple by construction — section 4) -> step 2.

Policy:

- Two budgets: existing `device_budget_bytes` (VRAM) + new `host_budget_bytes` (pinned RAM). On this box (12282 MiB VRAM, 15.9 GiB RAM): device ~10.5 GiB for weights+KV+activations, host ~10 GiB pinned, the rest of the 28.75 GiB checkpoint stays NVMe-windowed — roughly a third in VRAM, a further third in RAM, a third streaming.
- Eviction order: free device copies first (cheap to re-upload from pinned); pinned slots last (re-fill costs an NVMe round trip). One LRU `tick` counter drives both; a pinned, unpinned-slot-backed entry is the ideal device-eviction source.
- At layer init, permanently pin the BF16 smalls: 1,132,608 B per linear layer, 66,944 B per full layer — all 64 layers of smalls total ~65 MiB, eliminating small-window churn per step; scales ride along with their weight via `scale_idx` (acquire both under one pin pair, mirroring today's `matrix()`/`release()` discipline).
- embed/lm_head (2.37 GiB BF16 each, untied): keep lm_head host-pinned, GEMV over PCIe, or device-resident if budget remains after 64 layers; embed can stay mapped (lookup touches one row per token).
- The plain mmap path remains valid for whole-layer warmup/prefill; the O_DIRECT host tier is for bandwidth-deterministic streaming under RAM pressure.

### 5.6 Layer budget arithmetic (FP8 path)

| item | bytes |
|---|---|
| per linear layer F8 (qkv 52,428,800 + z 31,457,280 + out 31,457,280 + mlp 3 x 89,128,960) | 382,730,240 |
| per full-attn layer F8 (q 62,914,560 + k 5,242,880 + v 5,242,880 + o 31,457,280 + mlp 3 x 89,128,960) | 372,244,480 |
| all 64 layers F8 (48 linear + 16 full) | 24,326,963,200 |
| MTP F8 + mtp.fc BF16 + norms | 372,244,480 + 104,857,600 + ~53k |
| embed + lm_head BF16 (untied) | 5,085,593,600 |

A full 64-layer FP8 residency alone is ~22.7 GiB — it does not fit the 4070 SUPER, so budget/pin/LRU is mandatory. That is exactly the hierarchy Insignia exists for.

*Document generated programmatically from shard headers; engine sources untouched.*