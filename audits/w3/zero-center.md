# RMSNorm zero-centering — per-model dispatch audit (w3)

Audit date: 2026-08-25. Scope: every norm application site in the engine vs what the
Qwen3.8-27B-FP8 (HF format) and Qwen3.5-9B-MXFP4-MTP (MLX format) checkpoints actually
store; the dispatch design; the per-site patch list. Read-only audit except this file.
All checkpoint numbers below were measured firsthand this session (numpy, header-parse +
bf16→f32 bit shift, full tensors — norm weights are KB-scale).

Headline: **qwen35-arch.md finding #4 is confirmed exactly.** The 27B stores
zero-centered weights for every `Qwen3_5RMSNorm` tensor (engine must apply `(1+w)`);
`linear_attn.norm` alone stays one-centered raw. The 9B MLX checkpoint is pre-shifted
(all positive) everywhere, so the engine's current raw-everywhere behavior is **correct
for 9B and wrong for a 27B port** → per-model dispatch is required, not a global flip.

---

## 1. Checkpoint evidence (measured this session)

### 1.1 27B HF-format (`E:\coding\Insignia\Qwen3.8-27B-FP8\`)

| tensor (checkpoint name) | n | min | mean | max | neg count | verdict |
|---|---|---|---|---|---|---|
| `model.language_model.layers.0.input_layernorm.weight` (layers-0) | 5120 | −0.1318 | −0.0334 | +0.1982 | 4094 | **zero-centered** |
| `model.language_model.layers.0.post_attention_layernorm.weight` | 5120 | −0.9961 | −0.2173 | +0.0029 | 5119 | **zero-centered** |
| `model.language_model.layers.0.linear_attn.norm.weight` | 128 | +0.7852 | +0.8686 | +0.9297 | 0 | **one-centered, RAW** |
| `model.language_model.layers.3.self_attn.q_norm.weight` (layers-3) | 256 | −0.1758 | +0.2304 | +0.4805 | 5 | **zero-centered** |
| `model.language_model.layers.3.self_attn.k_norm.weight` | 256 | −0.5742 | +0.2203 | +0.7539 | 18 | **zero-centered** |
| `model.language_model.norm.weight` (outside.safetensors — final norm) | 5120 | −0.2852 | +0.9441 | +1.7109 | 4 | **zero-centered** |
| `mtp.pre_fc_norm_embedding.weight` (mtp.safetensors) | 5120 | −0.7500 | −0.4606 | −0.1855 | 5120 | **zero-centered** |
| `mtp.pre_fc_norm_hidden.weight` | 5120 | −0.3750 | −0.1572 | +0.4551 | 4922 | **zero-centered** |
| `mtp.layers.0.input_layernorm.weight` | 5120 | −0.2256 | +0.0361 | +0.9531 | 2563 | **zero-centered** |
| `mtp.layers.0.post_attention_layernorm.weight` | 5120 | −0.1621 | +0.2063 | +0.5156 | 8 | **zero-centered** |
| `mtp.layers.0.self_attn.q_norm.weight` | 256 | −0.5547 | +0.7906 | +1.9688 | 2 | **zero-centered** |
| `mtp.layers.0.self_attn.k_norm.weight` | 256 | −0.1143 | +0.7795 | +1.7891 | 1 | **zero-centered** |
| `mtp.norm.weight` | 5120 | −0.2246 | +1.2520 | +1.9297 | 1 | **zero-centered** |

(Control: `layers-3.self_attn.o_proj.weight_scale_inv` — all positive, ~2e-4 mean. The
min<0 probe is only meaningful on norm tensors selected **by name**; scale tensors are
naturally positive.)

Final-norm name in `outside.safetensors` verified: `model.language_model.norm.weight`
(BF16 [5120]) — not `model.language_model.model.norm.weight`.

Notes for probe design (see §3): detection density varies wildly across the family —
`input_layernorm` layer-0 has 80% negatives, but `mtp.layers.0.self_attn.k_norm` has
exactly **1** negative in 256 and `mtp.norm` **1** in 5120. A probe must scan the
**full tensor minimum** (KB-scale, trivial), never a sparse sample, and should prefer
the high-density `input_layernorm` family as its canonical target.

### 1.2 9B MLX-format (original HF-cache file, path from `build/qwen35.insignia-index`)

Payload path parsed from the index header:
`C:\Users\Pufos\.cache\huggingface\hub\models--sleepyeldrazi--Qwen3.5-9B-MXFP4-MTP\snapshots\18fa861767c634a60de5460c52eddc405cc6cded\model.safetensors`
(INSIDX01, 699 tensors, payload@90413 — matches loader-gaps §2.1.)

| tensor (checkpoint name) | n | min | mean | max | neg | verdict |
|---|---|---|---|---|---|---|
| `language_model.model.layers.0.input_layernorm.weight` | 4096 | 0.9375 | 1.0331 | 1.3047 | 0 | one-centered (pre-shifted) |
| `language_model.model.layers.0.post_attention_layernorm.weight` | 4096 | 0.0039 | 0.8875 | 1.0781 | 0 | one-centered (pre-shifted) |
| `language_model.model.layers.0.linear_attn.norm.weight` | 128 | 0.5312 | 0.8811 | 0.9609 | 0 | one-centered (raw param) |
| `language_model.model.layers.3.self_attn.q_norm.weight` | 256 | 0.7461 | 1.3409 | 1.9922 | 0 | one-centered (pre-shifted) |
| `language_model.model.layers.3.self_attn.k_norm.weight` | 256 | 0.1953 | 1.3254 | 2.1719 | 0 | one-centered (pre-shifted) |
| `language_model.model.layers.7.input_layernorm.weight` | 4096 | 0.3750 | 1.0212 | 1.7812 | 0 | one-centered (pre-shifted) |
| `language_model.model.norm.weight` | 4096 | 0.7773 | 2.1398 | 3.0000 | 0 | one-centered (pre-shifted) |
| `language_model.mtp.pre_fc_norm_embedding.weight` | 4096 | 0.2773 | 0.5203 | 0.7344 | 0 | one-centered (pre-shifted) |
| `language_model.mtp.pre_fc_norm_hidden.weight` | 4096 | 0.5469 | 0.7725 | 1.3750 | 0 | one-centered (pre-shifted) |
| `language_model.mtp.layers.0.input_layernorm.weight` | 4096 | 0.8203 | 1.0990 | 1.9844 | 0 | one-centered (pre-shifted) |
| `language_model.mtp.norm.weight` | 4096 | 0.8867 | 2.4258 | 3.2656 | 0 | one-centered (pre-shifted) |

**Every 9B norm tensor is all-positive.** The MLX converter pre-shifted +1 on the
Qwen3_5RMSNorm family — exactly the llama.cpp conversion trick (`conversion/qwen.py:389-390`,
`data_torch + 1` for every norm weight except `linear_attn.norm.weight`). Telling detail:
9B layer-0 `post_attention_layernorm` min = 0.0039 = 2⁻⁸ = exactly `1 + (−0.99609375)`,
the bf16 grid neighbor of the 27B's −0.9961 min for the same family — the pre-shift
signature (different models, 4096 vs 5120 hidden, so not identical weights).

**Conclusion: the two checkpoint formats disagree** → the engine needs a per-model flag.
Current raw-everywhere behavior is CORRECT for the 9B (independently confirmed: the
NumPy reference `rms(x,w)` multiplies raw — `tools/reference_layer0.py:8` — and layer-0
DeltaNet parity is cosine 0.9999998) and WRONG for the 27B.

---

## 2. Complete norm-site census and verdict table

Kernels that exist today (both already implement BOTH paths where noted):

- `src/ops.cu:5` `rms_kernel<ZERO_CENTERED,GATED>` — f32, has both paths.
  Launchers (`ops.cu:6`): `rmsnorm_zero_centered` (Z=true, **unit-tested** in
  `src/test_ops.cu:7` vs a CPU `(1+w)` reference), `rmsnorm_gated_silu` (Z=false, G=true).
  **No engine caller today** (f32 path; test-only).
- `src/qwen_kernels.cu:5` `rms_bf<Z,G>` — bf16 weights, has both paths.
  Launchers (`qwen_kernels.cu:6`):
  - `rmsnorm_bf16(x,w,y,rows,cols,bool zero_centered,stream)` — **the `false` literal in
    every decode.cu call IS the zero-center flag** (signature
    `include/insignia_qwen_kernels.cuh:5`); dispatches `rms_bf<true,false>` / `<false,false>`.
  - `gated_rmsnorm_bf16(...)` — hardwired `rms_bf<false,true>` (raw). **Correct as-is**
    (linear_attn.norm is one-centered in BOTH checkpoints — §1.1/§1.2).
- `src/ops.cu:9` `qk_norm_rope` (decode, per-token) — **hardcodes raw**
  `v *= nsc * bf(w+tid)`. Launcher `qwen35_qk_norm_rope_gate` (`ops.cu:10`,
  `include/insignia_ops.cuh:8`).
- `src/prefill.cu:54` `qk_norm_rope_batch_kernel` (prefill, T tokens) — **hardcodes raw**
  at `prefill.cu:72`. Launcher `qk_norm_rope_batch` (`prefill.cu:84`,
  `include/insignia_prefill.cuh:9`).

Per-site verdicts (weight tensor read → applied today → what 27B requires):

| # | site (file:line) | weight tensor | today | 27B needs | action |
|---|---|---|---|---|---|
| 1 | decode.cu:49 prefill `input_layernorm` | `layers.N.input_layernorm.weight` | raw (`false`) | (1+w) | pass flag |
| 2 | decode.cu:85 prefill `post_attention_layernorm` | `layers.N.post_attention_layernorm.weight` | raw | (1+w) | pass flag |
| 3 | decode.cu:92 prefill final norm | `language_model.model.norm.weight` (27B: `model.language_model.norm.weight`) | raw | (1+w) | pass flag |
| 4 | decode.cu:57 prefill q/k norm | `self_attn.q_norm/k_norm.weight` | **raw, hardcoded** | (1+w) | kernel change |
| 5 | decode.cu:81 prefill gated | `linear_attn.norm.weight` | raw (hardwired) | **raw** | none ✓ |
| 6 | decode.cu:122 delta_layer `input_layernorm` | same as #1 | raw | (1+w) | pass flag |
| 7 | decode.cu:124 delta_layer gated | `linear_attn.norm.weight` | raw (hardwired) | **raw** | none ✓ |
| 8 | decode.cu:125 delta_layer `post_attention_layernorm` | same as #2 | raw | (1+w) | pass flag |
| 9 | decode.cu:127 attn_layer `input_layernorm` | same as #1 | raw | (1+w) | pass flag |
| 10 | decode.cu:127 attn_layer q/k norm+rope | `q_norm/k_norm.weight` | **raw, hardcoded** | (1+w) | kernel change |
| 11 | decode.cu:127 attn_layer `post_attention_layernorm` | same as #2 | raw | (1+w) | pass flag |
| 12 | decode.cu:129 forward_body final norm | same as #3 | raw | (1+w) | pass flag |
| 13 | decode.cu:143 MTP `pre_fc_norm_embedding` | `mtp.pre_fc_norm_embedding.weight` | raw | (1+w) | pass flag |
| 14 | decode.cu:144 MTP `pre_fc_norm_hidden` | `mtp.pre_fc_norm_hidden.weight` | raw | (1+w) | pass flag |
| 15 | decode.cu:158 MTP layer `input_layernorm` | `mtp.layers.0.input_layernorm.weight` | raw | (1+w) | pass flag |
| 16 | decode.cu:166 MTP q/k norm+rope | `mtp.layers.0.self_attn.q_norm/k_norm.weight` | **raw, hardcoded** | (1+w) | kernel change |
| 17 | decode.cu:176 MTP `post_attention_layernorm` | `mtp.layers.0.post_attention_layernorm.weight` | raw | (1+w) | pass flag |
| 18 | decode.cu:184 MTP `mtp.norm` | `mtp.norm.weight` | raw | (1+w) | pass flag |
| 19 | dump_multistep.cu:49 final norm | same as #3 | raw | (1+w) | pass flag (parity tool) |
| 20 | dump_i4_seams.cu:24 / :62 in+post | #1/#2 family | raw | (1+w) | pass flag |
| 21 | dump_i4_seams.cu:55 gated | `linear_attn.norm.weight` | raw | **raw** | none ✓ |
| 22 | dump_i4_chunk.cu:43,84,98,137,148 in/post/final ×2 | #1/#2/#3 family | raw | (1+w) | pass flag |
| 23 | dump_i4_chunk.cu:75,116 gated ×2 | `linear_attn.norm.weight` | raw | **raw** | none ✓ |
| 24 | dump_i4_chunk.cu:126 qk batch | `q_norm/k_norm.weight` | **raw, hardcoded** | (1+w) | kernel change |
| 25 | dump_attention.cu:10 in+post rms ×2 | #1/#2 family | raw | (1+w) | pass flag |
| 26 | dump_attention.cu:10 qk norm+rope | `q_norm/k_norm.weight` | **raw, hardcoded** | (1+w) | kernel change |
| 27 | ops.cu:6 `rmsnorm_zero_centered` (f32) | (no engine caller) | (1+w) ✓ | (1+w) | none; tested |
| 28 | ops.cu:6 `rmsnorm_gated_silu` (f32) | (no engine caller) | raw ✓ | **raw** | none |
| 29 | qwen_kernels.cu:6 `rmsnorm_bf16` launcher | — | both paths exist | — | callers pass flag |
| 30 | embed path (qwen35.cu:13, prefill.cu:9-40 `embed_gather*`) | — | no norm applied | no norm (HF applies none to embeddings) | none ✓ |

Totals: **24 `rmsnorm_bf16(..., false, ...)` call sites** to flip to the flag;
**5 gated sites** that must NOT change; **3 qk kernel call sites** (2 kernels) that
hardcode raw and need the template change. The only embed-side norms are the MTP
`pre_fc_norm_*` pair (#13/#14) — the embedding gather itself is norm-free, correctly.

---

## 3. Dispatch design

### Option (b) — flag bit in the INSIDX02 index header — RECOMMENDED

- `tools/index_safetensors.py` (27B builder; `index27.py` already exists as a stub
  family member) probes at build time: for every tensor whose name belongs to the
  Qwen3_5RMSNorm family — suffixes `input_layernorm.weight`, `post_attention_layernorm.weight`,
  `q_norm.weight`, `k_norm.weight`, `pre_fc_norm_embedding.weight`,
  `pre_fc_norm_hidden.weight`, `mtp.norm.weight`, final `model.norm.weight` — EXCLUDING
  `linear_attn.norm.weight` and any `weight_scale_inv` — compute the full-tensor min
  (KB-scale reads during an index pass that already streams headers/payload).
  All-family min<0 → `norms_zero_centered = 1`. Mixed verdicts within one checkpoint
  (some tensors negative, some not) → **hard error at build** (corrupt/unknown export).
  Cost: one-time at index build; result is deterministic and frozen in the file.
- INSIDX02 header (loader-gaps §2.2 layout) gains one field, e.g. after `tensor_count`:
  `u8 model_flags; // bit0: norms_zero_centered` — read with the existing
  `take<T>`/pack(1) style in `model_file.cpp`. Zero per-token cost forever.
- `ModelFile` exposes `bool norms_zero_centered() const noexcept`. **INSIDX01
  (version 1) has no such field → constructor hardcodes false → the 9B path is
  byte-for-byte unchanged.** This is what makes per-model dispatch automatic:
  9B index ⇒ false, 27B index ⇒ true.
- `Qwen35Weights` ctor (`src/qwen35.cu:5`) copies the bit into a member
  (`bool zc_norms_`); `Qwen35Decode` reads `w_`'s member at every launch. If the
  workspace should own it instead (dump tools build `Qwen35Weights` too —
  dump_attention.cu:10 does), the weights object is the right owner since it wraps
  the ModelFile.

### Option (a) — runtime probe at first acquire — keep as ASSERT, not primary

At `Qwen35Weights` construction, scan the mmap for the canonical probe tensor
(`layers.0.input_layernorm.weight` — 80% negative density at 27B; fall back to any
family tensor): min<0 ⇒ checkpoint is zero-centered. Compare against the index flag;
disagreement ⇒ throw. Cost: ~1.4 MB of page-fault reads once at load (all norm
tensors: 64×2×10 KB + 16×1 KB q/k + mtp ≈ 1.4 MB), hidden in warmup. This protects
hand-edited indexes and catches builder regressions. Not recommended as primary:
it runs every load, depends on mmap residency, and can in principle be fooled by a
hypothetical zero-centered checkpoint whose weights are all > −1 (no 27B family tensor
is, but the guarantee is empirical, not structural).

### Option (c) — per-tensor encoding in the index — rejected (overkill)

The property is checkpoint-global by construction (the HF exporter writes every
`Qwen3_5RMSNorm` the same way; the MLX converter pre-shifts every one the same way).
130+ per-tensor bits saying one thing buys nothing, invites inconsistent states, and
still needs a call-site decision. Per AGENTS.md, one baked global constant IS the
specialized design.

### Rejected alternative: shift +1 into the data at load

Materializing `(1+w)` copies breaks zero-copy (TensorView.data points into the mmap),
adds ownership/staging complexity for ~1.4 MB, and the `Z=true` kernel paths already
exist and cost nothing per token. The llama.cpp/TRT strategy (shift at conversion time)
is equally fine **but requires re-writing the checkpoint**, which the loader-gaps audit
explicitly avoids ("no checkpoint modification").

---

## 4. Per-site patch list (27B port, ordered)

1. **Index header flag** — `tools/index_safetensors.py`/`index27.py`: probe (§3b), emit
   `model_flags bit0`. `include/insignia_model.hpp` + `src/model_file.cpp`: parse field,
   `norms_zero_centered()` accessor; INSIDX01 ⇒ false.
2. **Weights plumbing** — `include/insignia_qwen35.hpp`: `bool zc_norms_{}` +
   accessor on `Qwen35Weights`; init in ctor (`src/qwen35.cu:5`); optional §3a assert.
3. **`rmsnorm_bf16` call sites** — replace the literal `false` with
   `w_.zc_norms()` at decode.cu:49, 85, 92, 122, 125, 127 (×2), 129, 143, 144, 158,
   176, 184 and in the parity tools dump_multistep.cu:49, dump_i4_seams.cu:24, 62,
   dump_i4_chunk.cu:43, 84, 98, 137, 148, dump_attention.cu:10 (×2).
   **No kernel change needed** — `rms_bf<true,false>` already exists
   (qwen_kernels.cu:5-6). Note: `rms_bf<true>` is currently instantiated but never
   executed by any caller; `rmsnorm_zero_centered`'s f32 sibling is unit-tested
   (test_ops.cu:7), consider one bf16 parity check in test_qwen35 for free insurance.
4. **`qk_norm_rope` (decode)** — `src/ops.cu:9`: make the kernel
   `template<bool Z>`; the one-line change:
   ```cuda
   // today:  v *= nsc * __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(w+tid));
   v *= nsc * (Z ? 1.f + __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(w+tid))
                 :      __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(w+tid)));
   ```
   Launcher `qwen35_qk_norm_rope_gate` (`ops.cu:10`, header `insignia_ops.cuh:8`) gains
   `bool zero_centered` and dispatches both instantiations (mirror of `rmsnorm_bf16`).
   Template (not a runtime kernel param) matches the `rms_bf` precedent; zero per-thread
   cost. Callers: decode.cu:127, decode.cu:166, dump_attention.cu:10 → pass `w_.zc_norms()`.
   (The `gate` kernel param at ops.cu:9 is dead — unused in the body; not load-bearing.)
5. **`qk_norm_rope_batch` (prefill)** — `src/prefill.cu:54-83` kernel → `template<bool Z>`,
   same edit at line 72 (`v *= nsc * __bfloat162float(...)` → `nsc * (Z ? 1.f + ... : ...)`).
   Launcher `qk_norm_rope_batch` (`prefill.cu:84`, header `insignia_prefill.cuh:9`) gains
   the bool. Callers: decode.cu:57, dump_i4_chunk.cu:126.
6. **Gated path — deliberately untouched** — `gated_rmsnorm_bf16` (qwen_kernels.cu:6)
   stays hardwired `Z=false`: `linear_attn.norm.weight` is one-centered raw in BOTH
   checkpoints (27B [0.7852, 0.9297], 9B [0.5312, 0.9609]) — RMSNormGated parameters are
   never zero-centered in either export. Same for `rmsnorm_gated_silu` (f32 twin).
   Add the one comment this deserves at the launcher so nobody "fixes" it later.
7. **Reference scripts** — `tools/reference_*.py` hardcode raw
   (`def rms(x,w): return x/np.sqrt(np.mean(x*x)+1e-6)*w`, reference_layer0.py:8 — same
   pattern in the other reference_*.py). For 27B parity gates they must take a
   `zero_centered` switch too (`* (1+w)` when set), else the parity chase will "prove"
   the wrong kernel correct.

---

## 5. Sanity: (1+w) does not disturb the rope/norm math ordering — CONFIRMED

Both qk kernels compute, in order (ops.cu:9; prefill.cu:59-82):

1. `ss` accumulated from the **raw** q/k values (`float v = p[tid], ss = v*v`) —
   BEFORE any weight touches v;
2. `nsc = rsqrtf(ss/256 + 1e-6f)` — the norm scale is a function of x only,
   independent of w; eps sits inside the rsqrt per HF convention (verified §2 of
   qwen35-arch), also independent of centering;
3. `v *= nsc * bf(w+tid)` — the ONLY place w enters; this is the zero-center site;
4. RoPE afterwards: `inv/cos/sin` depend only on position and dim index, then
   `v = v*c ± other*s` — a rotation applied to the already-weight-scaled v.

Since step 4 is linear in v, changing step 3 from `nsc·w` to `nsc·(1+w)` is the
complete and only required change — exactly `v *= nsc * (Z ? 1.f + bf(w+tid) : bf(w+tid))`.
No reordering, no interaction with the partial-rope pass-through dims (64..255 are
rotated by identity, weight applies to all 256 equally), and the attention scale
1/√256 downstream is weight-independent. The KV-stored keys get the same (1+w_k)
treatment at store time (store_kv copies post-kernel k), so Q and K stay consistent —
both sides of the dot product shift together.

---

## 6. TL;DR

27B: every `Qwen3_5RMSNorm` tensor measured zero-centered (14/14 family tensors have
negatives; `mtp.norm` as thin as 1 negative in 5120 — probes must use full-tensor min);
`linear_attn.norm` alone is one-centered raw. 9B MLX: all-positive pre-shifted
(11/11 sampled) — current raw behavior correct, confirmed by the raw-multiply NumPy
reference and the layer-0 cosine-0.9999998 parity. Dispatch: add `norms_zero_centered`
bit to the INSIDX02 header (builder probes; INSIDX01 hardcodes false), runtime probe
as load-time assert. Patches: flip the `bool` at 24 `rmsnorm_bf16` call sites (the
`false` literals ARE the flag), templated `Z` for the two qk_norm_rope kernels (3 call
sites hardcode raw today), gated path untouched, reference scripts need the same switch.
