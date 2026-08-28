# W4 — line-by-line math audit of `tools/reference27.py` vs Qwen3.5 architecture ground truth

Date 2026-08-25. Target: `E:\coding\Insignia\tools\reference27.py` (570 lines). Ground truth:
live checkpoint headers/values (`Qwen3.8-27B-FP8\*.safetensors`, read via python), `config.json`,
HF source copy `_ref_modeling_qwen3_5.py`, llama.cpp clone `src/models/qwen35.cpp` + ggml
`ggml-cpu/ops.cpp`, engine `src/deltanet.cu`, known-good 9B reference `tools/reference_all_layers.py`,
`audits/w3/{MASTER-PLAN,parity-ladder,qwen35-arch}.md`, `audits/w3/diff-verify.md` H1.
Read-only + python analysis (selftest, one 4-token layer-0..3 smoke run). No builds, no edits.

## 0. Executive verdict

**reference27.py is mathematically correct for everything it implements** — 26/26 checked aspects
PASS against config.json, live checkpoint tensors, HF source, llama.cpp, ggml, and the engine
kernel. Two first-hand resolutions of previously-open questions:

1. **RoPE pairing convention — RESOLVED (was "THE critical unvalidated convention")**.
   Read `_ref_modeling_qwen3_5.py` rope code directly: `Qwen3_5TextRotaryEmbedding` +
   `apply_rotary_pos_emb` (lines 547-590) implement **rotate_half pairs (i, i+32) over the first
   64 dims of each 256-dim head**, `inv_freq_i = theta^(-i/32)`, theta 1e7, dims 64..255
   pass-through. reference27 `rope64` (lines 268-276) is an exact match. Second witness:
   llama.cpp `qwen35.cpp` uses `ggml_rope_multi`, whose `rotate_pairs` (ggml-cpu/ops.cpp:5931-5946)
   pairs `(j, j+n_offset)` — the same halves convention. **`mrope_interleaved: true` in config is
   verified a NO-OP for text-only**: `apply_interleaved_mrope` (HF lines 149-164) overwrites
   freqs positions from the H/W channels, but all 3 position channels are identical for 1-D text
   positions, so the written values equal the T-channel values already there.
2. **fp8 dequant orientation — RESOLVED empirically**. All 407 F8 tensors have scale shape
   exactly `[ceil(r/128), ceil(c/128)]` (census re-run live, 0 mismatches; non-square shapes make
   this unambiguous). Sampled blocks of `layers-0 in_proj_qkv`: per-block dequant amax 0.067–0.106
   (smooth), multiply gives region amax 0.1055/std 0.017, divide gives 2.5e6 (absurd). The exact
   `dq_f8` row-block loop was verified bit-equal to a naive 128x128 block map on synthetic data.

**One real gap: the MTP draft head is entirely absent** (no `mtp.fc`, no `pre_fc_norm_*`, no
`mtp.norm`, no mtp layer anywhere in the file — grep returns nothing). R7 multistep draft-path
parity and Phase F acceptance testing have no NumPy ground truth from this script.

## 1. Checklist table

| # | aspect | reference27.py line(s) | verdict | risk |
|---|---|---|---|---|
| 1 | Constants: H 5120, INTER 17408, LAYERS 64, VOCAB 248320 | 40-43 | PASS — config.json:18-20,98,117 | none |
| 2 | Layer typing `(l & 3) == 3` full-attn (48 lin + 16 full) | 42, 309 | PASS — layer_types list config:21-86 | none |
| 3 | Full attn QH/KVH/HD = 24/4/256; GQA = 6 | 44, 46 | PASS — config:97-99 | none |
| 4 | LK/LV/LD = 16/48/128; KSHARE = 3; CONV_C = 10240 | 47-49 | PASS — config:87-92 | none |
| 5 | rms eps 1e-6 everywhere (incl. gated norm, q/k norm) | 50, 94, 248-250, 260, 285-286 | PASS — config rms_norm_eps 1e-6 | none |
| 6 | e4m3 OCP decoder (bias 7, subnormal m*2^-9, NaN 0x7F/0xFF, max ±448) | 64-76, 405-438 | PASS — selftest run green, bitwise vs hand ldexp table | none |
| 7 | bf16→f32 `(u16<<16).view(f32)` | 79-80 | PASS — standard | none |
| 8 | Zero-center RMSNorm: input/post layernorm (1+w) | 176-177 | PASS — measured zero-centered (input ln mean −0.03, 80% neg; post ln 100% neg); pre-shift `1.0 +` | none |
| 9 | Zero-center RMSNorm: q_norm/k_norm (1+w) | 180-181 | PASS — measured negative entries (q_norm min −0.18, k_norm min −0.57) | none |
| 10 | Zero-center RMSNorm: final `model.norm` (1+w) | 158, 472 | PASS — measured mean 0.94 with negatives | none |
| 11 | linear_attn.norm RAW (one-centered, RMSNormGated) | 193, 260 | PASS — measured [0.785, 0.930], 0% negative; used raw | none |
| 12 | fp8 dequant: W = LUT8[F8] × bf16 scale, 128×128 blocks | 130-140 | PASS — multiply verified empirically (divide = 2.5e6 absurd); `np.repeat` broadcast == block map, bit-exact; 407/407 scale shapes `[ceil(r/128),ceil(c/128)]` | none |
| 13 | fp8 dequant: row-block index `s[i//128]`, col repeat 128 | 138-139 | PASS — dims all multiples of 128 → exact | none |
| 14 | q_proj split: reshape(24, 512), q=[0:256], gate=[256:512] per head | 281-282 | PASS — HF `view(...,-1,head_dim*2).chunk(2)`; lc view_3d stride 2·head_dim; engine split_q_gate | none |
| 15 | GQA kvh = h // 6 | 46, 294 | PASS — HF num_key_value_groups 6, repeat_interleave | none |
| 16 | Partial RoPE 64 dims, pairs (i,i+32), rotate_half, dims 64..255 pass | 53, 265-276 | PASS — **first-hand HF verification** (rotate_half lines 547-551,584-585); ggml rotate_pairs agrees | none (was critical-open; R5 still the runtime kill-shot) |
| 17 | RoPE theta 1e7, inv_freq = theta^(-i/32), f64 angles → f32 | 54, 265, 271-273 | PASS — HF `1/(base^(arange(0,64,2)/64))` identical; HF forces float32 cos/sin too | none |
| 18 | mrope_interleaved=true handled (no-op for text) | (absent — plain rope) | PASS by omission — HF apply_interleaved_mrope writes identical values when 3 channels equal (verified in source) | none |
| 19 | q/k norm (per-head RMS 256, (1+w)) BEFORE rope | 285-288 | PASS — HF norm→rope order (lc comment "Q norm, KV projection, K norm, RoPE") | none |
| 20 | Softmax scale 1/√256 = 0.0625 | 51, 295 | PASS — HF head_dim^-0.5 | none |
| 21 | Output gate sigmoid AFTER value mix, before o_proj; gate un-normed/un-roped | 282, 299 | PASS — HF `attn_output * sigmoid(gate)`; lc gate_sigmoid mul | none |
| 22 | DeltaNet in_proj_qkv split q@0/k@2048/v@6144 of [10240,5120] | 245-247 | PASS — census shapes; HF plain concat | none |
| 23 | conv1d [10240,1,4] causal, taps (oldest..newest)=(w0..w3), raw pre-SiLU state, SiLU on all 10240 ch | 190, 240-244 | PASS — HF pad-left-3 + depthwise conv; newest × w[3]; state stores raw | none |
| 24 | **In-chunk conv history for prefill t>0** | 239-243, 312-316 | PASS — conv state rolled every token inside run_layers; persists across greedy steps (shared `st`); diff-verify H1 gap class NOT present | none |
| 25 | q l2norm ×1/√128 (eps inside), k unit norm, no extra scales | 248-249 | PASS — HF l2norm eps 1e-6, scale k_head_dim^-0.5 = 1/√128 = 0.0883883 | none |
| 26 | k-sharing kh = j // 3 via np.repeat(k,3) | 48, 250-251 | PASS — HF repeat_interleave(3); `k48[j]=k[j//3]` identical to engine `kh=head/3` | none |
| 27 | β = sigmoid(b·x); α = exp(−exp(A_log)·softplus(a·x + dt_bias)) | 252-253 | PASS — sign matches engine `g=-__expf(A)*soft`, `decay=expf(g)` (deltanet.cu:9); measured exp(A_log) ∈ [0.0038, 0.338] → α ∈ (0,1) healthy | none |
| 28 | Softplus: np.logaddexp(0,x) vs torch threshold-20 | 253 | PASS (≈) — diff ≤ exp(−20) ≈ 2e-9 absolute for x>20; below any parity threshold | negligible |
| 29 | Recurrence: decay state FIRST, mem from decayed S, δ un-decayed, update, out from post-update S | 255-259 | PASS — HF/FLA/TRT/lc 4-witness consensus; literal match to deltanet.cu:10-12 | none |
| 30 | State orientation S[h][k][v] (`hkv`) matching GPU `state[i*128+tid]` | 205-211, 256-259 | PASS — matches deltanet.cu literally (i=key, tid=value); note 9B i4 scripts used the transpose — this one is engine-layout | none |
| 31 | Gated RMSNorm on v-path: per-head RMS 128 → ×w RAW → ×silu(z) | 260-261 | PASS — HF RMSNormGated order (norm → weight → activation(gate)) | none |
| 32 | z = in_proj_z(x) reshaped [48,128], silu elementwise | 187, 261 | PASS | none |
| 33 | out_proj [5120,6144] @ out.reshape(-1) head-major | 194, 262 | PASS | none |
| 34 | MLP SwiGLU with post_attention_layernorm (1+w) | 176, 229-233 | PASS | none |
| 35 | mtp.fc [5120,10240] orientation + concat order (embed-normed first) | **ABSENT** | **GAP** — no MTP code at all (grep "mtp" = 0 hits) | **HIGH (plan coverage)** |
| 36 | mtp.norm / pre_fc_norm_* zero-center conventions | **ABSENT** | **GAP** — same as above (would need 1+w; measured pre_fc_norm_embedding 100% negative) | **HIGH (plan coverage)** |
| 37 | lm_head [248320,5120] bf16, W @ h orientation, chunked 8192 rows, never materialized | 156-157, 163-164, 331-339 | PASS — header verified [248320,5120] BF16; rows→logits, no transpose | none |
| 38 | embed row slice (never materialize 2.5 GB) | 160-161 | PASS | none |
| 39 | NLL: f64 online logsumexp, teacher-forced targets | 342-360 | PASS | none |
| 40 | greedy: absolute positions pos_base=len(ids)+step, state persistence, first-index tie-break | 502-519 | PASS | none |

(Aspects 1-40; items are numbered aspects from the mission brief broken into checkable atoms.
35/36 are the only non-PASS rows.)

## 2. Verification evidence (what was run)

1. **Header census, all 66 shards**: 407 F8 tensors, every one has a BF16
   `weight_scale_inv` with shape exactly `[ceil(rows/128), ceil(cols/128)]` — 0 mismatches.
   Key shapes re-confirmed live: in_proj_qkv F8 [10240,5120], conv1d BF16 [10240,1,4], A_log/dt_bias
   BF16 [48], in_proj_a/b BF16 [48,5120], q_proj F8 [12288,5120], k_proj F8 [1024,5120],
   mtp.fc BF16 [5120,10240], lm_head/embed BF16 [248320,5120], final norm BF16 [5120].
2. **Norm-centering measured** (layers-0/3, mtp, outside): input_layernorm mean −0.03 (80% neg),
   post_attention_layernorm mean −0.22 (100% neg), q_norm min −0.18, k_norm min −0.57,
   pre_fc_norm_embedding min −0.75 (100% neg), final norm mean +0.94 with negatives → all
   zero-centered, (1+w) mandatory. linear_attn.norm ∈ [0.785, 0.930] 0% neg → one-centered, raw.
   reference27's split (lines 176-181, 158 pre-shift vs 193 raw) matches exactly.
3. **A_log semantics**: exp(A_log) ∈ [0.0038, 0.338] across [48] heads → reference27's
   `exp(−exp(A_log)·softplus)` gives α ∈ (0,1) (decay). Sign matches engine
   (`g = −__expf(A)·soft`, `decay = expf(g)` in src/deltanet.cu:9). A flipped sign would explode
   the state; this is right.
4. **fp8 block dequant**: multiply vs divide on a 256×512 region of in_proj_qkv (×: amax 0.1055,
   std 0.017; ÷: amax 2.5e6); per-block dequant amax 0.067–0.106 smooth across blocks; 3/8 sampled
   blocks contain a ±448 code; 0 NaN codes in region. `dq_f8`'s row-block loop verified bit-equal
   to a naive per-block multiply on synthetic data.
5. **RoPE**: HF `_ref_modeling_qwen3_5.py` read line-by-line (RotaryEmbedding init 105-125,
   forward 128-147, apply_interleaved_mrope 149-164, rotate_half 547-551, apply_rotary_pos_emb
   555-590). inv_freq formula identical to ROPE_INV (max rel diff < 1e-15). mrope interleave
   proven a no-op for text. ggml `rotate_pairs` halves-pairing is the second implementation
   witness; llama.cpp qwen35.cpp applies rope after Q/K norm with gate at per-head offset 256 —
   same structure reference27 implements.
6. **Selftest**: `python tools/reference27.py selftest` → PASS (254 finite + 2 NaN codes,
   monotone, sign-symmetric, anchors).
7. **End-to-end smoke**: `python tools/reference27.py layer 3 760,3712,314,23470 --no-save`
   → layers 0-2 (DeltaNet) + layer 3 (full attention, RoPE active at pos 0-3) all finite, norms
   11.8-52.9, monotonically growing — the whole dequant→conv→recurrence→attention→MLP path runs.

## 3. Ranked risks

1. **MTP draft head absent (HIGH — coverage, not correctness).** No `mtp.fc`, `pre_fc_norm_*`,
   `mtp.norm`, or the mtp attention layer anywhere in reference27.py. Consequences: R7's
   draft-path comparison and Phase F acceptance (p ≥ 0.55) have no NumPy ground truth; the
   mtp-side zero-center conventions (pre_fc_norm_embedding measured 100% negative — raw use is
   impossible) are unexercised. MASTER-PLAN risk #8 (mtp.fc dims/orientation + concat order
   "hides from token parity") explicitly depends on a reference for the draft path. Fix: add an
   `mtp` subcommand mirroring `_ref_qwen3_5_mtp.py` (fc @ concat(rms(embed,pre_fc_emb),
   rms(hidden,pre_fc_hid)), embed-first, then the mtp.layers.0 full-attn block, mtp.norm (1+w),
   shared lm_head).
2. **RoPE halves convention — now validated statically; keep R5 (LOW residual).** The code-level
   verification (HF + ggml + lc, plus reference27) is conclusive on paper; the only remaining
   exposure is a transcription slip in the engine port, which R5×5 runs is already designed to
   catch. Downgraded from "critical unvalidated" to "statically resolved, runtime-verify at R5".
3. **Numerics below parity thresholds (NEGLIGIBLE).** logaddexp vs threshold-20 softplus (≤2e-9
   abs); f64 rope angles cast f32 (HF forces f32 cos/sin too); `np.float32(np.sqrt(128))`
   double-rounded 1/√128 (identical at f32). None can move a cosine in the 5th decimal.
4. **Cosmetic warts (NONE math).** `QSCALE` (line 52) is dead — the scale is computed via
   `np.sqrt(LD)` at line 248 (same value). The `--attn` flag (line 533) claims to force
   attention math but `run_layers` ignores it (only a warning at 562-564 fires). Per-layer shard
   reload makes greedy ~26 GB reads/step (documented in parity-ladder §5; slow, correct).
5. **Structural notes for the engine compare (INFO).** reference27's seam output is
   snapshot-copied per layer (`traj[l] = X` copies), so no aliasing bug; state layout matches the
   GPU literally (S[h][k][v]); conv/KV/Δ states persist correctly across greedy steps with
   absolute positions — the multistep semantics R7 needs are already right.

## 4. Bottom line

reference27.py is a trustworthy ground truth for R4 (layer-0 DeltaNet seam), R5/R6 (attention +
all-layer seams), R8 (NLL), and R9 (greedy ids) — every checked formula matches the checkpoint,
HF, llama.cpp/ggml, and the engine kernel, including the two previously-unvalidated conventions
(RoPE pairing, fp8 scale orientation) which are now verified first-hand. The single actionable
finding: **the MTP draft path has no reference implementation** and must be added before Phase F
acceptance work.
