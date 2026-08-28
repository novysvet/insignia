# Qwen3.5 architecture formula sheet — Insignia audit (2026-08-25)

Scope: verify every Qwen3.5(-family) formula Insignia implements against >=2 independent
implementations, ahead of the Qwen3.8-27B-FP8 port. Sources used:

- **HF-Q35** `E:\coding\Insignia\_ref_modeling_qwen3_5.py` (local copy of transformers
  `src/transformers/models/qwen3_5/modeling_qwen3_5.py`, verified identical semantics to
  upstream `modeling_qwen3_next.py` on GitHub main, fetched 2026-08-25).
- **HF-Q3N** `E:\coding\Insignia\_ref_modeling_qwen3_next.py` (transformers qwen3_next).
- **vLLM-MTP** `E:\coding\Insignia\_ref_qwen3_5_mtp.py` (+`_ref_qwen3_next_mtp.py`) —
  the *only* MTP reference; HF transformers has no MTP modeling (ignores `^mtp.*` keys,
  HF-Q35:805; confirmed https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_moe
  has no `modeling_qwen3_moe_mtp.py`).
- **TRT** `E:\coding\Insignia\TensorRT-LLM\tensorrt_llm\_torch\auto_deploy\custom_ops\fla\{gdn_gating,fla_gated_delta}.py`
  and `...\models\custom\modeling_qwen3_next.py`.
- **FLA** flash-linear-attention `fla/ops/gated_delta_rule/fused_recurrent.py` (GitHub main).
- **lc** llama.cpp clone @ c060ca974 (2026-08-23): `src/models/qwen35.cpp`,
  `src/models/delta-net-base.cpp`, `conversion/qwen.py`.
- **CKPT** `E:\coding\Insignia\Qwen3.8-27B-FP8\*.safetensors` headers + norm-weight values,
  read directly (python, this session).

---

## 0. Model facts (all verified firsthand from config.json + shard headers)

| fact | value | evidence |
|---|---|---|
| arch | `Qwen3_5ForConditionalGeneration`, model_type `qwen3_5` / text `qwen3_5_text` | config.json:2-8 |
| layers | 64; `layer_types` = 3x linear_attention + 1 full_attention repeating, full at `l%4==3` (48 lin / 16 full) | config.json:21-86 |
| hidden / inter | 5120 / 17408 (dense SwiGLU MLP — **no MoE** in the 27B dense; `mlp.gate`/`shared_expert_gate` entries in `modules_to_not_convert` are quantizer-template noise) | config.json:18-20; mlp tensors in layers-N headers |
| full attn | 24 Q heads, 4 KV heads (group 6), head_dim 256, q_norm/k_norm [256], q_proj [12288,5120], k/v [1024,5120], o [5120,6144], attn_output_gate true, partial rope 0.25 (64 dims), theta 1e7, rms_eps 1e-6 | config.json:11,97-99,102,141; layers-3 header |
| linear attn | in_proj_qkv [10240,5120] = q(2048)|k(2048)|v(6144) plain concat, in_proj_z [6144,5120], in_proj_a/b [48,5120] bf16, conv1d [10240,1,4] bf16, A_log/dt_bias [48], norm [128], out_proj [5120,6144]; 16 k-heads x128, 48 v-heads x128 (k-sharing /3), fp32 state 48x128x128/layer | layers-0 header; HF-Q35:427-430 |
| MTP | 1 layer, full-attention type, fc bf16 [5120,10240], pre_fc_norm_{embedding,hidden}, mtp.norm; **no mtp.embed_tokens / mtp.lm_head tensors → shared with main model** (`mtp_use_dedicated_embeddings: false`) | mtp.safetensors header; config.json:94-95 |
| quant | F8_E4M3 weights + BF16 `weight_scale_inv [ceil(r/128), ceil(c/128)]`, dynamic activation | config.json:140-141,1028-1031 |
| outside | embed_tokens + lm_head bf16 [248320,5120] (2.54 GB each), final norm, + all vision (`model.visual.*`, ~0.92 GB) | outside.safetensors header |
| 9B engine shape today | `Qwen35Shape{hidden=4096, inter=12288, layers=32, vocab=248320}` — vocab already identical → tokenizer + logits buffers reusable | include/insignia_qwen35.hpp:7 |

---

## 1. Gated DeltaNet (linear attention) — ALL FORMULAS MATCH

References: HF-Q35:330-380 (torch_recurrent), 248-327 (chunk), 387-544 (module);
FLA fused_recurrent (web); TRT fla_gated_delta.py:30-31,49-56,92-94 + gdn_gating.py:96;
lc delta-net-base.cpp:289-370.

### 1.1 Projections & conv

```
QKV = in_proj_qkv(x)                  # [q(2048) | k(2048) | v(6144)] PLAIN CONCAT
z   = in_proj_z(x);  b = in_proj_b(x); a = in_proj_a(x)     # separate projections
y   = depthwise_conv1d(QKV, w[10240,1,4], causal, pad-left 3)
QKV = silu(y)                         # SiLU on the WHOLE qkv (all channels), per-channel
```

- SiLU placement: HF applies `ACT2FN[hidden_act]` to the entire conv output — q AND k AND v
  all pass through SiLU (HF-Q35:471-477 with 230-239). TRT/FLA identical.
- Ours: `conv4` computes `z = s0·w0+s1·w1+s2·w2+x·w3` then `x = z·σ(z)` on every channel;
  conv state stores the RAW pre-SiLU inputs (HF does the same — activation is applied to
  the output only, HF-Q35:210-216). VERIFIED: src/qwen_kernels.cu:7-8 (decode),
  src/prefill.cu:170-181 (prefill, output to scratch so raw x survives).
- Qwen3.5 uses PLAIN q|k|v concat (unlike Qwen3-Next's grouped per-k-head
  `[q,k,v,z]x16` layout, HF-Q3N:558-585 — llama.cpp regroups it in
  conversion/qwen.py:393-420; Qwen3.5 checkpoints need no regroup).

### 1.2 Head geometry & norms

```
q ∈ R^{16x128} (k-heads), k ∈ R^{16x128}, v ∈ R^{48x128};  v-head j uses k-head j//3
q̂ = q / sqrt(||q||² + 1e-6) * (1/sqrt(128));   k̂ = k / sqrt(||k||² + 1e-6)
```

- l2norm eps=1e-6 INSIDE the rsqrt (HF-Q35:242-245 `l2norm`, matches FLA `_l2norm`).
- Query scale = `k_head_dim**-0.5` = 1/√128 = **0.08838834764831845** (HF-Q35:352-353;
  FLA `scale = K**-0.5`; TRT fla_gated_delta.py:93). **The constant in our kernel is
  EXACTLY 1/√128 — there is no extra 1/√2 factor** (1/√128/√2 would be 0.0625; the
  prompt's decomposition was arithmetically off). Key gets NO scale (unit norm) — no
  `d**0.25` anywhere in any implementation.
- Ours: `sq[0]=rsqrtf(a+1e-6f)*0.08838834764831845f; sk[0]=rsqrtf(b+1e-6f)` — EXACT match
  (src/deltanet.cu:8, src/prefill.cu:239-240).
- k-sharing: HF `repeat_interleave(num_v/num_k=3, dim=2)` → v j uses k j//3
  (HF-Q35:501-503). Ours `kh = head>>1` (9B, /2). **27B needs head/3** (port item).

### 1.3 Gating scalars

```
β = sigmoid(b);  g = -exp(A_log) · softplus(a + dt_bias);  α = exp(g)
softplus(x) = x if x>20 else log1p(exp(x))     # PyTorch default threshold 20
```

- Ours `params` kernel: `b=1/(1+expf(-b)); z=a+dt; soft = z>20? z : log1pf(expf(z));
  a = -__expf(A)*soft` — EXACT (src/qwen_kernels.cu:9, src/prefill.cu:203-211).
  TRT gdn_gating.py:96 fuses the identical expression (beta=1, threshold=20).

### 1.4 Recurrence (state S ∈ R^{128(k) x 128(v)} per v-head, fp32)

```
for each token t:
    S ← α · S                          # decay FIRST
    mem[v] = Σ_k S[k][v] · k̂[k]       # memory read from the DECAYED state
    δ[v]   = β · (v[v] − mem[v])       # delta NOT multiplied by α
    S[k][v] ← S[k][v] + k̂[k] · δ[v]   # rank-1 update
    out[v] = Σ_k S[k][v] · q̂[k]       # output from the POST-update state
```

- HF: `S*=g_t; kv_mem=(S*k).sum(-2); delta=(v-kv_mem)*β; S+=k⊗δ; out=(S*q).sum(-2)`
  (HF-Q35:371-375; HF-Q3N:497-501). FLA fused_recurrent identical (`b_h *= exp(b_g)`
  before `b_v = b_beta*(b_v − Σ b_h·b_k)`; `b_h += b_k[:,None]*b_v`; delta carries no
  extra decay). TRT docstring: `S = S*exp(g) + k*(v − S*k)*beta` (fla_gated_delta.py:31).
  lc: `s*=exp(g); sk=sum_rows(s·k); d=(v−sk)·b; s+=k·d; o=sum_rows(s·q)`
  (delta-net-base.cpp:338-366). **All four agree; the alpha placement question is
  settled: decay-on-state-first, delta un-decayed.**
- Ours: `dot = Σ_i S[i*128+tid]*decay*k̂[i]` (mem), `delta=(v−dot)*β`,
  `cell = cell*decay + k̂[i]*delta` (= (S·α)+k̂δ since cell was not yet decayed),
  `acc = Σ_i S_new[i*128+tid]*q̂[i]` — state orientation S[k][v] (i=k-dim, tid=v-dim),
  EXACT match (src/deltanet.cu:10-12, src/prefill.cu:244-255).

### 1.5 Output norm + gate + proj

```
o_h = RMSNorm_128(core_h) · w_norm[128] · silu(z_h)     # per 128-dim v-head, w RAW
out = o · W_out_proj
```

- Order: fp32 RMS → ×weight (RAW, one-centered parameter — RMSNormGated init ones,
  NOT (1+w)) → ×silu(gate in fp32) (HF-Q35:167-184; HF-Q3N:57-74). Per-head grouping =
  one 128-wide norm per v-head (group_num=1) (HF-Q35:537-540 reshapes [-1, head_v_dim]).
- Ours: `gated_rmsnorm_bf16(core, norm.weight, z, core, rows=48?, cols=128)` with
  template Z=false (raw w) + `z *= v/(1+exp(-v))` after weight — EXACT
  (src/qwen_kernels.cu:5-6; decode call src/decode.cu:124 with 32 rows for 9B).
- Checkpoint: `linear_attn.norm.weight` values ∈ [0.785, 0.930] (one-centered) —
  raw use CORRECT for this tensor (measured this session).

## 2. Full attention

References: HF-Q35:630-702; HF-Q3N:236-310; TRT modeling_qwen3_next.py:470-560;
lc qwen35.cpp:260-330.

```
qg  = q_proj(x)                                # [12288] = per head: [256 q | 256 gate]
q   = RMSNorm_256(q) · (1+w_q); k = RMSNorm_256(k) · (1+w_k)      # per-head, eps 1e-6
q,k = rope_64(q,k; pos; θ=1e7);                # first 64 dims only, rest pass-through
                                      # inv_freq_i = θ^(−2i/64), i=0..31
                                      # rotate_half: pairs (i, i+32), i<32:
                                      # out[i]=x[i]c − x[i+32]s; out[i+32]=x[i+32]c + x[i]s
attn = softmax(q·kᵀ/√256)·v                    # causal, GQA: head h → kv-head h//6
out  = attn · sigmoid(gate)                    # gate AFTER value mix, per head
y    = o_proj(out)
```

- **q/gate interleave**: HF `q_proj(h).view(...,-1,head_dim*2)` then `chunk(2,dim=-1)`
  → head h occupies [h·512, h·512+256) for q and [h·512+256, h·512+512) for gate
  (HF-Q35:668-671). TRT identical (470-480). lc: view_3d with per-head stride
  2·head_dim, gate at offset head_dim (qwen35.cpp:272-295). Ours `split_q_gate`:
  `q[i]=src[h*512+d]; g[i]=src[h*512+256+d]` — EXACT (src/qwen_kernels.cu:73;
  batch twin src/prefill.cu:43-48). (9B: 4096=16·256; 27B needs 6144=24·256.)
- **q/k norm**: per-head RMS over 256, eps 1e-6 (HF `Qwen3_5RMSNorm` with
  `(1.0 + weight)`, HF-Q35:721-735). Ours: `nsc=rsqrtf(ss/256+1e-6f);
  v *= nsc * bf(w+tid)` — eps & scaling ✓, but **weight applied RAW, missing +1** for
  HF-format checkpoints (src/ops.cu:9, src/prefill.cu:69-72). See §7.1 — the one real
  formula mismatch.
- **RoPE**: partial 64 (=256·0.25), θ=1e7, `inv_freq=1/θ^(arange(0,64,2)/64)`
  (HF-Q35:117-125), rotate_half on the 64 roped dims → **pairs (i, i+32)** ✓
  (HF-Q35:547-551 + 578-590). Ours: `inv=powf(1e7,−2·half/64)`, `other=mem[tid<32?tid+32:tid−32]`,
  `v = v·c + (tid<32?−other:other)·s`, dims 64..255 untouched, pos==0 skip — EXACT
  convention match (src/ops.cu:9, src/prefill.cu:76-81). Note `__powf/__cosf/__sinf`
  are fast-math (2-3 ulp) — fine, but worth knowing for parity chasing.
  **mrope_interleaved (config) is a NO-OP for text**: HF expands position_ids to
  (3,...) and interleaves H/W freqs into T (HF-Q35:129-164), but with 1-D text
  positions all three channels are equal, so the permutation changes nothing.
- **Scale** 1/√256 = 0.0625 ✓ ours (src/attention.cu:7 `scale=.0625f`;
  src/prefill.cu:120). HF `head_dim**-0.5` (HF-Q35:639).
- **GQA mapping**: HF `num_key_value_groups = 24/4 = 6`; `repeat_kv` is
  repeat_interleave → **head h uses kv-head h//6** (HF-Q35:593-602,638).
  Ours: `kvh=head>>2` (9B group 4) — correct for 9B, **WRONG for 27B (must be head/6)**
  (src/attention.cu:7, src/prefill.cu:103).
- **Gate**: `attn_output = attn_output.reshape(...) * sigmoid(gate)` AFTER SDPA,
  BEFORE o_proj (HF-Q35:698-701; TRT 554; lc qwen35.cpp attn_pregate·sigmoid).
  Ours: `gqa_decode → expand_gate_heads → sigmoid_mul → o_proj` ✓ EXACT order
  (src/decode.cu:127, mtp twin :170-173).
- Causal mask over `pos+1` tokens ✓ (src/attention.cu:7 `tokens=__ldg(pos_dev)+base+1`).

## 3. MTP head (draft layer)

Reference: vLLM-MTP only (HF has none; lc has NextN tensors
`nextn.{eh_proj,enorm,hnorm,...}` in qwen35.cpp:113-118 = fc / pre_fc_norms).

```
e = embed_tokens(pending_token)                  # shared embedding table
h = target model final hidden (vLLM: POST model.norm output)
e' = pre_fc_norm_embedding(e);  h' = pre_fc_norm_hidden(h)     # RMSNorm (1+w)
x  = fc(concat([e', h']))                        # cat order: EMBED FIRST, then hidden
x  = MTP_layer(x)                                # ONE standard FULL-ATTN layer, own KV
y  = mtp_norm(x)                                 # RMSNorm (1+w)
logits = lm_head_shared(y)                       # shared lm_head (ckpt has none of its own)
```

- **Concat order**: `torch.cat([inputs_embeds, hidden_states], dim=-1)` — embed normed
  FIRST (vLLM-MTP:161; qwen3_next twin :133). Ours: `concat(up /*embed*/, norm /*hidden*/)`
  → out[i]=a[i], out[n+i]=b[i] ✓ EXACT (src/qwen_kernels.cu:69, call src/decode.cu:147).
- fc Linear(2h→h), bf16 in ckpt [5120,10240] ✓ ours takes the `bf16_gemv` branch for
  non-insig4 (src/decode.cu:148-152).
- MTP layer = full-attention block with own KV cache ✓ ours (src/decode.cu:155-186,
  mtp_keys/mtp_values; vLLM `layer_type="full_attention"` vLLM-MTP:118-125).
- Shared embed + shared lm_head ✓ (ckpt mtp.safetensors has neither tensor;
  `mtp_use_dedicated_embeddings: false`; ours uses `language_model.model.embed_tokens`
  and `language_model.lm_head`).
- **Deviation A (semantic, minor)**: vLLM feeds the main model's POST-final-norm hidden
  into MTP, then applies pre_fc_norm_hidden on top (vLLM's Qwen3_5Model applies
  self.norm before returning; vLLM-MTP:159-160 norms again). Ours feeds the PRE-norm
  hidden (`forward_body` leaves x_.hidden pre-norm, src/decode.cu:129 vs :144).
  Draft-quality/acceptance issue only — never affects verified output.
- **Deviation B (semantic, minor)**: our MTP layer attends at `mtp_pos = pos−1`
  (slot pos−1, RoPE pos−1; src/prefill.cu:277, src/decode.cu:154-170) — i.e. the MTP KV
  slots lag the main cache by one token and token 0's embedding is never in the MTP KV.
  The reference gives the draft token its true absolute position. Acceptance-rate
  impact only.
- Both deviations are consistent with the MTP being a pure draft generator (all outputs
  re-verified by the main layers; rejects cost throughput, not correctness).

## 4. Tokenizer / chat template / stop tokens

All files present in the model dir (tokenizer.json IS there, plus vocab.json +
merges.txt + chat_template.jinja):

- vocab 248,320 = 248,044 BPE merges + 276 added tokens (read from tokenizer.json).
- `<|endoftext|>` = **248044** (bos/pad; config bos/eos_token_id; tokenizer_config pad).
- `<|im_start|>` = 248045, **`<|im_end|>` = 248046 = eos_token** (tokenizer_config
  `eos_token: "<|im_end|>"`); generation_config.json eos = `[248046, 248044]`,
  temp 1.0 / top_k 20 / top_p 0.95 (greedy decode in Insignia is a choice, not a bug).
- **Chat stop token = 248046** (`<|im_end|>`); also accept 248044.
- Template (ChatML): `<|im_start|>{role}\n{content}<|im_end|>\n`; generation prompt =
  `<|im_start|>assistant\n` + `<think>\n` (thinking) or `<think>\n\n</think>\n\n`
  (enable_thinking=false). `<think>`=248068, `</think>`=248069 (non-special).
  Vision placeholders `<|vision_start|>` etc. never occur on the text-only path.
- tools/tok.py loads `tokenizer.json` + `chat_template.jinja` straight from the model
  dir and renders with `enable_thinking=False` → **reusable unchanged for the 27B**
  (9B vocab is the same 248,320, hard-coded in include/insignia_qwen35.hpp:7).

## 5. Vision tower — safely ignorable, confirmed

- All vision tensors live under `model.visual.*` in outside.safetensors: 27 blocks
  (LayerNorm + qkv/proj attn + GELU-MLP, hidden 1152, heads 16), patch_embed conv3d
  [1152,3,2,16,16], learned pos_embed [2304,1152], merger → out_hidden 5120
  (HF-Q35:831-1090). ~0.92 GB total. `deepstack_visual_indexes: []`.
- HF `Qwen3_5ForConditionalGeneration` runs vision only when pixel inputs are present;
  text-only forward never touches it (HF-Q35:1711-1785). Our engine reads only
  embed/lm_head/norm + layer shards from outside.safetensors → the 0.92 GB is dead
  weight on disk (could even be truncated/sparsified to save NVMe bandwidth budget).

---

## 6. Verdict table

| # | formula | ours (file:line) | reference (file:line) | verdict |
|---|---|---|---|---|
| 1 | conv1d w4 causal + SiLU on whole qkv | qwen_kernels.cu:7-8, prefill.cu:170-181 | HF-Q35:471-477,230-239 | ✓ exact |
| 2 | q̂ = q/‖q‖·(1/√128), k̂ = k/‖k‖, eps 1e-6 | deltanet.cu:8, prefill.cu:239-240 | HF-Q35:242-245,352-353; FLA; TRT fla:93 | ✓ exact (const = 1/√128, NOT 1/√256) |
| 3 | β=σ(b), g=−exp(A_log)·softplus(a+dt_bias), thr 20 | qwen_kernels.cu:9, prefill.cu:203-211 | HF-Q35:498-500; TRT gdn_gating.py:96 | ✓ exact |
| 4 | recurrence S=αS+k̂β(v−αSk̂)ᵀ…, out=q̂ᵀS (α on S only, δ un-decayed, S[k][v]) | deltanet.cu:9-12, prefill.cu:244-255 | HF-Q35:371-375; FLA fused_recurrent; TRT fla:30-31; lc delta-net-base.cpp:338-366 | ✓ exact (4 witnesses) |
| 5 | out norm: RMS₁₂₈·w_raw·silu(z), per-head | qwen_kernels.cu:5-6 + decode.cu:124 | HF-Q35:167-184,537-540 | ✓ exact |
| 6 | k-sharing v j → k j//3 (27B) | deltanet.cu:5 kh=head>>1 (9B /2) | HF-Q35:501-503 repeat_interleave | ✓ 9B; ✗ 27B port needs /3 |
| 7 | q_proj per-head [256q‖256gate] interleave | qwen_kernels.cu:73, prefill.cu:43-48 | HF-Q35:668-671; TRT:478-480; lc qwen35.cpp:272-295 | ✓ exact |
| 8 | q/k per-head RMS₂₅₆ eps 1e-6 **×(1+w)** | ops.cu:9, prefill.cu:69-72 (**raw w**) | HF-Q35:721-735 (1.0+w); TRT:50-55 + load hook; lc conversion/qwen.py:389-390 (+1 at convert) | **✗ for HF-format ckpt (see §7.1)** |
| 9 | partial RoPE 64 dims, θ=1e7, pairs (i,i+32) | ops.cu:9, prefill.cu:76-81 | HF-Q35:117-125,547-590 | ✓ exact |
| 10 | softmax scale 1/√256 = 0.0625 | attention.cu:7, prefill.cu:120 | HF-Q35:639 | ✓ exact |
| 11 | GQA head→kv = h//6 (27B) | attention.cu:7 kvh=head>>2 (9B /4) | HF-Q35:593-602,638 | ✓ 9B; ✗ 27B port needs /6 |
| 12 | sigmoid(gate) after SDPA, before o_proj | decode.cu:127,172 | HF-Q35:698-701; TRT:554; lc qwen35.cpp:330 | ✓ exact |
| 13 | MTP concat [embed',hidden'], fc 2h→h | qwen_kernels.cu:69, decode.cu:143-152 | vLLM-MTP:159-162 | ✓ exact |
| 14 | MTP layer = full-attn, own KV, mtp.norm, shared embed+lm_head | decode.cu:155-186 | vLLM-MTP:118-141,243-253; ckpt headers | ✓ (2 semantic deviations §3A/B) |
| 15 | stop tokens: im_end 248046, endoftext 248044 | (host-side policy) | tokenizer_config + generation_config | ✓ facts recorded |
| 16 | vision ignorable for text-only | engine never maps visual.* | HF-Q35:1711-1785; ckpt | ✓ confirmed |

## 7. Discrepancies & port items

### 7.1 CRITICAL — RMSNorm weight centering for HF-format checkpoints

The FP8 27B checkpoint stores HF-convention weights:

- `Qwen3_5RMSNorm` tensors (input_layernorm, post_attention_layernorm, q_norm, k_norm,
  model.norm, mtp.norm, pre_fc_norm_*): **zero-centered — consumer must apply (1+w)**.
  Measured: input_layernorm mean −0.03 (range −0.13..+0.20), q_norm mean +0.23 with
  negative entries, pre_fc_norm_embedding all-negative — raw use is impossible.
- `linear_attn.norm` (RMSNormGated): **one-centered, raw multiply** (measured
  [0.79, 0.93]) — matches our Z=false gated path.

Our engine multiplies RAW for everything (correct for the 9B **MLX-format** checkpoint,
where the conversion pre-shifted +1 — same trick llama.cpp uses at
conversion/qwen.py:389-390: `data_torch + 1` for every `norm.weight` except
`linear_attn.norm.weight`; TRT does it via a load-time pre-hook, TRT:50-55 comment).
**For the 27B FP8 port: add +1 to all Qwen3_5RMSNorm weights at load/index time (or
flip call sites to the existing Z=true path and keep RMSNormGated raw).** The
`rms_bf<true>` (1+w) path already exists at src/qwen_kernels.cu:5; today every caller
passes z=false (decode.cu:49,85,92,122,127,143-144,158,176,184) and qk_norm_rope
hard-codes raw w.

### 7.2 9B→27B hard-coded dims (all in the 9B engine today)

hidden 4096→5120, inter 12288→17408, layers 32→64 (include/insignia_qwen35.hpp:7;
decode.cu loops `l<32` at :47/:122/:129), conv/qkv 8192→10240 (v offset 4096→6144),
a/b 32→48 (params kernels: qwen_kernels.cu:10 `<<<1,32>>>` n=32; prefill.cu:205 `h>=32`),
deltanet grid 32→48 & kh>>1→/3 (deltanet.cu:5,14; prefill.cu:221), state
24×32→48×48 heads (decode.cu:14,25; prefill rollback :307-309), q heads 16→24
(split_q_gate 4096→6144 qwen_kernels.cu:73; qk rope grid 20→28 ops.cu:10,
prefill.cu:57,85; gqa grid 16→24 attention.cu:8, prefill.cu:164), kvh >>2→/6
(attention.cu:7, prefill.cu:103), q_proj 8192→12288, mtp fc 4096/8192→5120/10240
(decode.cu:150-151), MLP buffers 12288→17408 (decode.cu:14). Vocab/lm_head size
unchanged (248320). Plus: new FP8 e4m3 + block-scale dequant path (weights are
F8_E4M3 with [128,128] BF16 scale_inv blocks, not MXFP4).

### 7.3 Already-fixed items (vs audits/synthesis.md)

- RoPE smem race (synthesis #2): current ops.cu:9 and prefill.cu:54-83 use a dedicated
  `nsc` shared slot for the norm scale (comment prefill.cu:62) — the mem[0] clobber is
  gone in the tree as of today.
- MTP embed i4 branch (synthesis #3): decode.cu:137-138 now has the
  `if (m.insig4) embed_gather_i4` branch.

### 7.4 Semantics notes (no action required)

- mrope_interleaved is a no-op for text-only (§2); ignore mrope_section [11,11,10].
- Fast-math `__powf/__cosf/__sinf/__expf` in rope/gating: precision-only consideration
  for the pending full-attn parity chase.
- Our MTP feeds pre-final-norm hidden and lags KV positions by one (§3 A/B) —
  acceptable for a draft head; revisit only if acceptance rate is poor after the port.

## Sources (web)

- transformers qwen3_next modeling (upstream, fetched 2026-08-25):
  https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py
- transformers qwen3_5 model doc / config:
  https://huggingface.co/docs/transformers/en/model_doc/qwen3_5 ,
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/configuration_qwen3_5.py
- Qwen3.5 family cards: https://huggingface.co/Qwen/Qwen3.5-4B , .../Qwen3.5-9B ,
  .../Qwen3.5-0.8B ; https://modelscope.cn/models/qwen/Qwen3.5-2B ; Qwen3-Next blog
  (head_dim 128→256, partial rope 25%): https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd
- FLA fused recurrent gated delta rule:
  https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/fused_recurrent.py
- llama.cpp (clone @ c060ca974, 2026-08-23): src/models/qwen35.cpp,
  src/models/delta-net-base.cpp, conversion/qwen.py — qwen35/qwen35moe/qwen3next archs
  exist upstream (src/llama-arch.cpp:36-42) incl. NextN/MTP support.
- HF qwen3_moe has no MTP modeling file (checked):
  https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_moe
