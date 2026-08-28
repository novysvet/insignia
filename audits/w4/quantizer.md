# INSIG4 quantizer audit (w4)

Date 2026-08-25. Scope: `tools/quantize_insig4.py` (uncommitted changes), norm zero-center
consistency across every consumer, `tools/fix_insig4_bf16.py`, A_log dtype chain, measured
quantization error INSIG4 vs source. Read-only except this file. All numbers measured
firsthand this session (numpy, mmap-style header reads) against:

- BF16 source: `C:\Users\Pufos\.cache\huggingface\hub\models--Qwen--Qwen3.5-9B\snapshots\c202236235762e1c871ad0ccb60c8ee5ba337b9a` (775 tensors incl. visual; the quantizer's actual input — verified by bit-exact re-encode below).
- MLX MXFP4 snapshot: `C:\Users\Pufos\.cache\huggingface\hub\models--sleepyeldrazi--Qwen3.5-9B-MXFP4-MTP\snapshots\18fa...\model.safetensors` (700 tensors).
- INSIG4 outputs: `build/qwen35-insig4{,-v2,-text,-good}.safetensors` (+ `.insignia-index` files).

## 0. Headline findings

1. **The uncommitted quantizer has NEVER been run.** `tools/quantize_insig4.py` mtime is
   23:00; every INSIG4 file predates it (newest `-good` = 18:06). Nothing on disk contains
   the baked `w+1` norm shift.
2. **The engine's active INSIG4 file is doubly broken.** `build/qwen35-insig4.insignia-index`
   (13:46) points to `qwen35-insig4-text.safetensors` (07:52), whose quantized weights are a
   **bad encode (10.8 dB SQNR vs BF16 source — 7.4 dB WORSE than the MXFP4 snapshot)**, and
   whose norms are stored **raw zero-centered while every engine call site multiplies raw**
   (`Z=false`) → every RMSNorm applies `w` (mean ≈ 0.03) instead of `1+w`. The 9B INSIG4
   path cannot produce sane tokens in this state. Fix = run the current quantizer (which
   bakes +1 and matches `-good`'s encode bit-for-bit) and re-index.
3. `qwen35-insig4-good.safetensors` (18:06) is the good encode (20.2 dB vs source, +2.0 dB
   over MXFP4) — bit-exact to the current `quantize_tensor` on a re-encoded slab — but its
   non-quantized tensors went through fp16→bf16 double rounding (fix-script signature, 47
   tensors affected) and its norms are also un-shifted.
4. Quantizer math itself is sound: fp16 super-group scales are strictly normal-range with
   no underflow risk (measured min 4.5e-4 vs fp16 normal floor 6.1e-5), nibble rounding is
   nearest-magnitude on the true E2M1 grid (ties→smaller, 0.018% of elements), and the
   engine/reference dequant **exactly inverts** the encode (codes are computed against the
   fp16-rounded scale; re-encode reproduces stored bytes bit-for-bit).
5. 27B zero-center and A_log-BF16 hazards (master-plan Phase A items 6/7) are both still
   live in `src/decode.cu` — see §3/§4.

## 1. Quantizer math verification

### 1.1 Scale policy (`optimal_scale`, quantize_insig4.py:26-38)

Per 64-elt super-group: golden-section ternary search (10 iters) on
`[0.6, 1.6] × amax/6` minimizing MSE against the true E2M1 grid, then RNE-cast to fp16
(`astype(np.float16)` = RNE). Codes are then computed against the **fp16-rounded** scale
(`.astype(np.float32)` of the f16 value, lines 49/53) — this is what makes dequant an
exact inverse.

Measured on a 512-row slab of `layers.0.mlp.down_proj` (98,304 super-groups):

| check | result |
|---|---|
| optimum/(amax/6) observed range | [0.789, 1.592] — upper bracket bound is touched but a wide-bracket [0.05,4]× search is **worse** (MSE 1.1459e-6 vs 1.1358e-6; nearest-grid MSE is non-unimodal and the narrow bracket is a good prior) → leave as is |
| fp16 scale rounding loss | MSE 1.135803e-6 vs 1.135201e-6 with f32 scales (Δ 6.0e-10 ≈ 0.0005 dB) — negligible |
| exact halfway ties in code assignment | 1123 / 6,291,456 (0.018%); `argmin` picks the smaller magnitude — MSE-harmless |

### 1.2 fp16 scale range adequacy (measured on `-good`, all classes)

| tensor class | n scales | min | max | median | zeros | subnormals |
|---|---|---|---|---|---|---|
| in_proj_qkv L0 | 524288 | 7.15e-4 | 6.08e-2 | 7.11e-3 | 0 | 0 |
| out_proj L0 | 262144 | 1.38e-3 | 1.40e-1 | 6.33e-3 | 0 | 0 |
| q_proj L3 | 524288 | 1.57e-3 | 7.19e-2 | 7.69e-3 | 0 | 0 |
| gate_proj L0 | 786432 | 1.65e-3 | 4.76e-2 | 4.86e-3 | 0 | 0 |
| down_proj L0 | 786432 | 2.17e-3 | 1.05e-1 | 4.76e-3 | 0 | 0 |
| lm_head rows 0-8191 | 524288 | 1.23e-3 | 3.89e-2 | 6.72e-3 | 0 | 0 |
| embed rows 0-8191 | 524288 | 4.51e-4 | 7.79e-2 | 6.39e-3 | 0 | 0 |

Every scale is a **normal** fp16 (≥7.4× above the subnormal floor 6.104e-5), ≥11-bit
relative precision; the E2M1 code error dominates by ~3 orders of magnitude. The `1e-30`
amax floor only guards div-by-zero; real quantized tensors (all big projections; tiny
conv1d/norms stay BF16) never approach it. **fp16 is adequate — no saturation, no
underflow.**

### 1.3 Nibble rounding + packing

`quant_codes`/final loop: `idx = argmin |w/s − E2M1|` on {0,.5,1,1.5,2,3,4,6}, sign in
bit 3 (`idx | (sign<<3)`). Packing: per 32-elt group = 4 consecutive u32; element
`8*word + j` of the group lives in word `word`, bits `4j..4j+3` (LSB-first, MLX style).
Scale per 2 groups (64 elts), row stride `cols/64`.

### 1.4 Does the engine dequant EXACTLY invert the encode? YES

- `src/mxfp4_i4.cu:11` `i4_scale(s,g) = half2float(s[g>>1])` — fp16→f32 is exact; scale
  index `g>>1` = per-64. `mxfp4_gemv_v2_i4_kernel` (line 35-36): `row_w = weights +
  row*groups*4`, `row_s = scales + row*(groups>>1)` — 4 words per 32-group, matches §1.3.
  `decode4`/`fp4_e2m1` (`include/insignia_layout.cuh:20,38`) = the same 16-entry LUT
  {0,.5,1,1.5,2,3,4,6,±}. `V2I` unpacks `(w_ >> (j*4)) & 15` — same bit order. Scale
  applied after the 32-group partial sum — algebraically identical to elementwise
  `lut[q]*scale` (reference `dq`, tools/reference_all_layers_i4.py:6), differing only in
  fma summation order.
- `mxfp4_gemm_v21_i4` (src/gemm.cu:366-406): "covers exactly one super-group → scale
  index = kb", `__half2float` — same semantics. `mxfp4_gemm_mlx_i4` (gemm.cu:330) and
  `mxfp4_gemm_ab_i4` (gemm.cu:498) likewise. `embed_gather_i4` (src/prefill.cu:26-38):
  `scale = half(s + row*64 + (g>>1))`, 4 words/32-group — matches (cols hardwired 4096).
- dp4a pair kernels (`mxfp4_gemv2_q8_i4`, `mxfp4_gemv_ab2_q8_i4`): weight side uses table
  {0,1,2,3,4,6,8,12} = 2×E2M1 with compensating `*0.5f` — weight decode stays exact; the
  int8 activation quantization is a separate lossy engine choice, not a quantizer issue.
- **Empirical proof**: re-running the current `quantize_tensor` on a 256-row slab of
  `down_proj L0` from the BF16 source reproduces `-good`'s stored packed u32 AND f16
  scales **bit-for-bit**.

## 2. Zero-center +1 consistency — the matrix

Kernel semantics: `rms_bf<Z>` (src/qwen_kernels.cu:5): `z = x*nsc*(Z ? 1+bf(w) : bf(w))`;
launcher `rmsnorm_bf16(..., bool z, ...)` (line 6). **Every decode.cu call site passes
`false`** (prefill: 49, 89, 96; delta: 126, 129; attn: 131; forward_body: 133; MTP: 147,
148, 162, 180, 188). `gated_rmsnorm_bf16` (linear_attn.norm) is hardwired `Z=false`
(lines 85, 128 — correct, see below). `qk_norm_rope`/`qk_norm_rope_batch`
(src/ops.cu:9, src/prefill.cu:54-72) hardcode raw `nsc*bf(w)` at 3 call sites
(decode.cu:57, 131, 170). Master plan Phase A item 7 (audits/w3/MASTER-PLAN.md:280-283)
says: flip 9 decode.cu sites to `Z=true` + template the qk kernels for the **27B** port;
`linear_attn.norm` stays raw. Prior art: audits/w3/zero-center.md (full census).

Consistency matrix (9B INSIG4 path = BF16 src → INSIG4 → engine):

| tensor | checkpoint stores | quantizer (current, uncommitted) does | engine kernel does | references do (all_layers_i4 / multistep_i4 / pf_i4) | verdict |
|---|---|---|---|---|---|
| input_layernorm ×32 (+mtp.layers.0) | zero-centered raw (negatives; L0: −0.064/+0.033/+0.303, 646/4096 neg) | **+1 baked**, RNE bf16 | `rms_bf<false>` raw multiply → 1+w total | `rms(x,w)=x*rsqrt*w` raw multiply → 1+w | ✔ consistent **after regen**; ✘ today (text/good store raw → engine gets w ≈ 0.03×) |
| post_attention_layernorm ×32 (+mtp) | zero-centered raw (L0: −0.996/+0.113/+0.078) | +1 baked | raw (decode.cu:89,129,131,180) | raw | ✔ after regen; ✘ today |
| self_attn q_norm/k_norm ×8 (+mtp) | zero-centered raw (L3: −0.254/+0.341, −0.805/+0.326) | +1 baked ('norm' in name, not excluded) | qk_norm_rope kernels raw (ops.cu:9, prefill.cu:72) | `q*rsqrt*get(q_norm)` raw | ✔ after regen; ✘ today |
| final model.norm | zero-centered raw (−0.223/+1.140/+1.992) | +1 baked | raw (decode.cu:96,133) | raw | ✔ after regen; ✘ today |
| mtp.norm, mtp.pre_fc_norm_embedding/hidden | zero-centered raw (−0.723/+0.480/−0.264 etc.) | +1 baked | raw (decode.cu:147,148,188) | raw (multistep only uses pre_fc implicitly via engine) | ✔ after regen; ✘ today |
| **linear_attn.norm** ×24 | ONE-centered raw (+0.533/+0.881/+0.963, F32) | excluded → raw RNE bf16 | `gated_rmsnorm_bf16` hardwired raw | raw `nw` | ✔ consistent today and after regen |
| A_log ×24 | F32 (both source repos) | F32 passthrough (line 132) | `const float*` read | F32 read | ✔ (see §4 for 27B) |
| dt_bias, conv1d, mtp.fc, embeddings/proj | bf16 / 2D quantized | bf16 RNE / INSIG4 | bf16 read / i4 kernels | same | ✔ |

27B path (`Qwen3.8-27B-FP8`, raw HF convention; zero-center.md §1.1 measured):

| consumer | applies | verdict |
|---|---|---|
| reference27.py:158,176-181 | `1.0 + bf16(w)` for every Qwen3_5RMSNorm; `normw` raw (line 193); A_log bf16→f32 (line 191) | correct |
| engine decode.cu (all sites above) | raw everywhere (`false` literals + hardcoded qk kernels) | **WRONG for 27B — parity WILL fail** until Phase A item 7 (per-checkpoint `norms_zero_centered` flag; INSIDX01/INSIG4-baked ⇒ false, 27B ⇒ true) |
| dump tools (dump_attention/dump_i4_seams/dump_i4_chunk/dump_multistep) | raw (`false`) | must follow the same flag for 27B parity |

Post-regen sanity, 9B: the INSIG4 index keeps `norms_zero_centered=false` since the shift
is baked — matches the plan's INSIDX01⇒false default, so no engine change is needed for
the 9B INSIG4 path. Full-sweep proof of the shift logic: BF16 src has exactly the
`Qwen3_5RMSNorm` family containing 'norm' in its names (visual.* skipped); the only
one-centered member, `linear_attn.norm.weight`, is the only excluded suffix — nothing is
double-shifted (source is raw HF, shift applied exactly once).

## 3. File forensics: what each INSIG4 artifact is

| file (mtime) | quantized path | non-quantized path | index |
|---|---|---|---|
| `qwen35-insig4.safetensors` (02:50) | v1, broken naming (`lm_head.weight.weight`), header has non-utf8 tail | fp16-bytes-labeled-BF16 (v1 bug) | — |
| `qwen35-insig4-v2.safetensors` (05:56) | names fixed; **A_log emitted BF16** (engine would misread) | fp16-bytes-labeled-BF16 (asFP16 reads sane, asBF16 garbage) | — |
| `qwen35-insig4-text.safetensors` (07:52) | **BAD encode**: 10.8 dB vs source; scales median 0.55× optimum (0.32-1.07 spread), 100% of packed bytes differ from a correct encode | **correct direct RNE bf16** (160 tensors) + F32 A_log (24, bitwise == src) | **`qwen35-insig4.insignia-index` (13:46) POINTS HERE** — also hardcoded in tools/nll_compare.py:48 |
| `qwen35-insig4-good.safetensors` (18:06) | **GOOD encode**: bit-exact to current `quantize_tensor`; 20.2 dB vs source | 113 direct-RNE + **47 tensors with fp16→bf16 double-rounding** (fix-script signature; e.g. conv1d L0: 38/32768 elements off — all in the fp16-subnormal range |w|<6e-5, ≤2 ulp bf16) + F32 A_log (bitwise == src) | `qwen35-insig4-good.insignia-index` (18:06) |

- **What fix_insig4_bf16.py fixes**: the v1/v2 bug of writing `w.astype(np.float16)`
  bytes under a `BF16` label (engine reads garbage). It streams the file once, converts
  fp16→f32→bf16-RNE, and forces A_log to F32.
- **Still needed?** No for new outputs — the current quantizer writes real RNE bf16
  (`f32_to_bf16_bytes`, quantize_insig4.py:67-71) and F32 A_log directly. Worse: running
  it on a **correctly**-generated file would corrupt it (it re-reads real bf16 as fp16).
  Its only legitimate remaining use is repairing v1/v2-era files; `-good`'s 47
  double-rounded tensors are its fingerprint. Recommend retiring it (or an in-file
  sanity assert) once the final regenerate lands.
- **Was the INSIG4 file regenerated after the RNE fix?** Partially: `-text` (07:52) has
  the direct-RNE non-quantized path but the broken quantized path; `-good` (18:06) has
  the fixed quantized path but fp16-double-rounded small tensors. **No file matches the
  current working-tree quantizer**, and none has the +1 norm bake. The active index still
  points at the worst file (`-text`).

## 4. A_log dtype chain

- 9B sources: A_log is **F32** in both the BF16 repo and the MXFP4 snapshot; quantizer
  emits F32 (`emit_raw(..., 'F32')`, line 132; comment "deltanet kernel reads A_log as
  float32"); `-text`/`-good` store it bitwise-identical to source. Engine:
  `deltanet_parameters` (src/qwen_kernels.cu:9-10, kernel `params(float*,float*,const
  float*A,const uint16_t*dt,...)`) and `deltanet_params_batch`
  (src/prefill.cu:212, `const float *A_log`) with call-site casts at
  **src/decode.cu:82 and src/decode.cu:128** — all consistent (f32) for 9B. ✔
  (`-v2`'s BF16 A_log was the broken intermediate; the engine would have read 2 bf16
  values per f32 word.)
- 27B: checkpoint stores A_log **BF16** (reference27.py:191 reads `sh.bf16(...A_log)`).
  The unchanged `const float*` casts would misread it (master-plan risk R4 "A_log
  BF16-as-F32, α≈1 signature"; Phase A item 6 = `const void* + bool a_log_f32` dispatch —
  NOT yet implemented anywhere). ✘ pending for the 27B port.

## 5. Quantization error, INSIG4 vs source MXFP4 (cos / SQNR, f64 accum)

INSIG4 = `-good` (the current-code encode); MXFP4 = sleepyeldrazi snapshot; src = BF16 repo.

| tensor | INSIG4-vs-MXFP4 | INSIG4-vs-src | MXFP4-vs-src |
|---|---|---|---|
| in_proj_qkv L0 [8192×4096] | 0.988540 / 16.42 dB | 0.995256 / 20.20 dB | 0.992817 / 18.20 dB |
| out_proj L0 [4096×4096] | 0.988138 / 16.27 dB | 0.995174 / 20.12 dB | 0.992494 / 17.98 dB |
| gate_proj L0 (first 4096 rows) | 0.988238 / 16.30 dB | 0.995246 / 20.19 dB | 0.992750 / 18.12 dB |
| up_proj L0 (first 4096 rows) | 0.988555 / 16.42 dB | 0.995266 / 20.21 dB | 0.992955 / 18.28 dB |
| down_proj L0 [4096×12288] | 0.988324 / 16.33 dB | 0.995231 / 20.17 dB | 0.992791 / 18.16 dB |
| lm_head rows 0-4095 | 0.988582 / 16.43 dB | 0.995286 / 20.22 dB | 0.992863 / 18.23 dB |
| embed rows 0-4095 | 0.987781 / 16.13 dB | 0.995060 / 20.02 dB | 0.992213 / 17.79 dB |

Uniform **+2.0..+2.2 dB over the MLX MXFP4 encode** across every class. The docstring's
"+5.9 dB over E8M0 MXFP4" is against a per-64 E8M0 baseline (tools/quant_study.py line
11); against the snapshot's per-32 E8M0 the honest number is ~+2 dB. Consider fixing the
comment. (`-text`, for contrast: 10.6-10.8 dB vs src on every class — obsolete junk.)

## 6. Ranked fixes

1. **Regenerate `qwen35-insig4-text.safetensors` with the current
   `tools/quantize_insig4.py`** (bakes w+1 norms, direct RNE bf16, F32 A_log, correct
   E2M1 encode) and rebuild `build/qwen35-insig4.insignia-index` against it. Until then
   the active engine file has both a 10.8 dB encode and ~30×-too-small RMSNorm gains
   (raw w, mean 0.03, applied instead of 1+w). This is the only blocking correctness item
   for the 9B INSIG4 path.
2. **Retire or guard `tools/fix_insig4_bf16.py`** — obsolete after the in-quantizer RNE
   fix, and actively harmful if run on a correctly-generated file (re-reads real bf16 as
   fp16). If `-good` is kept as a fallback, note its 47 double-rounded tensors.
3. **27B port gates (master-plan Phase A items 6+7, still unimplemented)**: per-checkpoint
   `norms_zero_centered` flag (INSIDX02 bit) + flip the 9 decode.cu `false` literals +
   template `qk_norm_rope`/`_batch` (ops.cu:9, prefill.cu:72; call sites decode.cu:57,
   131, 170); A_log `const void* + bool a_log_f32` in `params`/`params_batch`
   (qwen_kernels.cu:9, prefill.cu:212, decode.cu:82/128). reference27.py is already
   correct on both counts — engine parity R4 will fail without these.
4. **Reference scripts**: consistent as-is for a baked INSIG4 file (raw multiply). Do NOT
   add a +1 there; the shift now lives in the file (mirrors the MLX snapshot convention
   the references were already written against).
5. Cosmetic: quantizer docstring "+5.9 dB" → "+2 dB vs the shipped per-32 E8M0 MLX encode
   (+5.9 dB vs per-64 E8M0)". No code change; bracket [0.6,1.6]×amax/6 verified fine
   (§1.2).
