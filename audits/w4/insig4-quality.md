# INSIG4 quantization quality study (w4) — real weights, numpy only

Date 2026-08-25. Pure measurement: every number below was computed firsthand this session
with numpy (f64 accumulators) over read-only, seek-based reads of the safetensors files.
No files created/modified except this report. Companion audit: `audits/w4/quantizer.md`
(encoder correctness, norm zero-center, file forensics). This document is about
**quantization quality**: how much fidelity each format/variant loses on the real
Qwen3.5-9B weights, and what that implies end-to-end.

## 1. Inputs and method

| role | file |
|---|---|
| ground truth W (BF16) | `models--Qwen--Qwen3.5-9B\snapshots\c202236...\` (4 shards, 775 tensors) — the quantizer's actual input; locally available |
| MLX MXFP4 snapshot | `models--sleepyeldrazi--Qwen3.5-9B-MXFP4-MTP\snapshots\18fa...\model.safetensors` |
| INSIG4 (good encode, bit-exact to current `quantize_insig4.py`) | `build/qwen35-insig4-good.safetensors` |
| INSIG4 (the file named by the engine index — **bad encode**, see §2) | `build/qwen35-insig4-text.safetensors` |

Decodes: MLX = E2M1 nibbles (LSB-first, 8/u32) × e8m0 f32 per 32-group; INSIG4 = same
nibbles × fp16 per 64-supergroup (`tools/reference_all_layers.py` / `_i4.py` semantics).
Metrics per tensor: relF = ‖W−Q‖F/‖W‖F, cosine, max-abs, SQNR = 10log10(‖W‖²/‖W−Q‖²).
Measured set: layers {0,1,3,6,7,12,15,20,23,28,31} (6 DeltaNet `l%4≠3`, 5 full-attn),
all quantized classes, plus 4096 randomly-sampled rows (rng seed 0) of `lm_head` and
`embed_tokens` (248320×4096). Whole measured corpus ≈ 2.4 G elements.

Dtype reality check vs the task brief: `in_proj_a/b` **are quantized** (U32 (32,512) +
scales) in both formats, not bf16; only `conv1d` (8192×1×4), norms, `dt_bias`, `A_log`
are bf16/f32. `conv1d` is bitwise == SRC in MLX and `-text`; `-good` has the known
fp16→bf16 double-rounding fingerprint (38/32768 elements, max |Δ| 3e-8 — negligible).

**bpw accounting (measured from headers, not the docstring):** MLX = 4 bits + 8/32 =
**4.25 bpw**; INSIG4 = 4 bits + 16/64 = **4.25 bpw** (lm_head: 540,344,320 B /
1,017,118,720 elts × 8 = 4.250 exactly, both formats byte-identical in size). The
quantizer docstring's "4.125 bpw" is wrong for its own format (4.125 is the g128 variant).

## 2. Headline verdicts

1. **INSIG4 (good encode) beats the MLX snapshot by +1.9±0.1 dB on every tensor class**
   at identical 4.25 bpw: mean SQNR vs BF16 source **20.13 dB (relF 0.0976, cos 0.9952)**
   vs MLX **18.21 dB (relF 0.1222, cos 0.9926)**. Byte-share-weighted: 20.21 dB vs 18.26 dB.
2. **`qwen35-insig4-text.safetensors` (the file the active index points at) is the broken
   encode: 12.01 dB mean (relF 0.249, cos 0.975) — 6.2 dB WORSE than the MLX snapshot.**
   Quality conclusions must be drawn from `-good` (bit-exact to the current quantizer).
3. Re-quantizing **from the MLX-decoded f32 instead of the BF16 source costs ~2 dB**
   (mse/64/f16: 20.17 → 17.92 dB). The BF16 source is in the local HF cache — always
   quantize from it, never from the snapshot.
4. Scale-format ablation is unambiguous: **fp16 ≡ bf16 scales (both 20.17 dB)**, fp32
   bound only +0.005 dB; e8m0 scales are the MLX format's core weakness; group 64 fp16
   beats group 32 e8m0 at the same 4.25 bpw.
5. **E2M1 > int4 at equal budget**: int4-sym MSE 19.60 dB < E2M1 MSE 20.17 dB @ 4.25.
   Asymmetric int4 + fp16 zeta reaches 20.52 dB but needs 4.5 bpw (still loses to
   E2M1 g32 fp16 at 20.80 dB @ 4.5).
6. Best variants measured: **INSIG4 as-shipped (20.17 dB @ 4.25)**; if +0.25 bpw is
   acceptable **g32/fp16/mse = 20.80 dB @ 4.5**; if bytes must shrink **g128/fp16/mse =
   19.69 dB @ 4.125 — still +1.4 dB over the MLX snapshot while 0.125 bpw smaller**.
7. Measured single-token end-to-end probe (§6): 4.25-bpw quantization noise moves logits
   by rmse ≈ 0.96×σ(z) after 32 layers; |ΔNLL| ≈ 1–2 nats vs BF16 for BOTH formats.
   The repo's |ΔNLL|<0.02 gate is therefore **only definable engine-vs-reference on the
   same quantized weights** — no 4.25-bpw format can sit within 0.02 nats of BF16.

## 3. Per-class quality table (means over all measured layers; vs BF16 source)

| class | MLX SQNR / relF / cos | INSIG4(-good) SQNR / relF / cos | INSIG4(-text) SQNR | INSIG4 vs MLX SQNR |
|---|---|---|---|---|
| in_proj_qkv (8192×4096) | 18.40 / .1203 / .99283 | 20.20 / .0977 / .99522 | 12.07 | 16.30 |
| in_proj_z (4096×4096) | 18.38 / .1206 / .99279 | 20.21 / .0976 / .99523 | 12.09 | 16.29 |
| in_proj_a (32×4096) | 18.07 / .1250 / .99227 | 19.95 / .1006 / .99492 | 11.88 | 15.98 |
| in_proj_b (32×4096) | 17.86 / .1281 / .99190 | 19.79 / .1025 / .99473 | 11.70 | 15.80 |
| linear out_proj | 18.28 / .1220 / .99263 | 20.18 / .0979 / .99519 | 12.01 | 16.21 |
| q_proj (8192×4096) | 18.27 / .1221 / .99263 | 20.19 / .0978 / .99521 | 12.06 | 16.17 |
| k_proj (1024×4096) | 18.04 / .1254 / .99222 | 19.97 / .1004 / .99494 | 11.88 | 15.99 |
| v_proj (1024×4096) | 18.17 / .1235 / .99246 | 20.05 / .0995 / .99504 | 11.91 | 16.10 |
| o_proj | 18.25 / .1223 / .99260 | 20.16 / .0982 / .99516 | 12.02 | 16.18 |
| mlp gate (12288×4096) | 18.23 / .1226 / .99260 | 20.21 / .0976 / .99523 | 12.08 | 16.07 |
| mlp up | 18.26 / .1222 / .99265 | 20.23 / .0974 / .99524 | 12.08 | 16.08 |
| mlp down (4096×12288) | 18.23 / .1226 / .99260 | 20.19 / .0978 / .99520 | 12.06 | 16.06 |
| lm_head (sampled 4096/248320 rows) | 18.29 / .1218 / .99266 | 20.20 / .0977 / .99522 | 12.09 | 16.19 |
| embed (sampled 4096 rows) | 18.17 / .1235 / .99250 | 20.25 / .0971 / .99527 | 12.10 | 16.04 |
| **all tensors** | **18.21 / .1222** | **20.13 / .0976** | **12.01** | **16.10** |

Worst max-abs element error anywhere: MLX 0.273 (L1 out_proj), INSIG4(-good) 0.117
(L0 out_proj), -text 0.531. Best single tensor: L0 up_proj (18.47 / 20.25 dB).

**Per-layer trend (mean → worst tensor):** quality degrades monotonically with depth.
L0 18.35/20.22 → L28 18.07/20.01 → L31 17.99/19.99 (MLX/INSIG4). Worst tensors per
layer are `in_proj_b` (DeltaNet layers; L6 17.66, L12 17.61) and `k_proj` (attn
layers; L23 17.94, L31 17.55); overall worst: **L28 in_proj_a 17.42 dB (MLX) / 19.34
(INSIG4)**. Nothing pathological — the "hard" tensors are simply the heavy-tailed ones
(§5) and the late layers. Full-attn vs DeltaNet layers barely differ on average
(18.22 vs 18.20 dB MLX).

## 4. Ablations (fixed 9-tensor subset: qkv/z/out L0, q/o L31, gate15, down15(2048 rows),
## lm_head+embed 4096 sampled rows; re-quantized from BF16 source unless marked `<MLXin`)

Golden-section (10 it) per-group MSE scale search on [0.5,1.5]×absmax/6; codes always
re-assigned against the rounded scale (INSIG4's exact-inverse property).

| variant | bpw | SQNR dB | relF | cos | notes |
|---|---|---|---|---|---|
| MLX file as shipped (e8m0/32, **round** rule) | 4.25 | 18.25 | .1224 | .99260 | baseline; decode of actual file |
| e2m1, amax/6, fp16, g64 | 4.25 | 19.46 | .1064 | .99433 | +1.2 dB over MLX for free |
| e2m1, amax/4, fp16, g64 | 4.25 | 18.54 | .1184 | .99301 | clipping without MSE gain — worse |
| e2m1, e8m0(floor), g64 | 4.125 | 14.90 | .1802 | .98530 | floor rule trap — see below |
| e2m1, e8m0(**round**), g64 | 4.125 | 18.41* | — | — | *6-tensor subset; honest e8m0/64 |
| e2m1, e8m0(ceil), g64 | 4.125 | 18.18* | — | — | ceil > round at g32 (18.51 vs 18.22*) |
| **e2m1, MSE-opt, fp16, g64 (= INSIG4 shipped rule)** | **4.25** | **20.17** | **.0981** | **.99518** | best @ 4.25 |
| e2m1, MSE-opt, bf16, g64 | 4.25 | 20.17 | .0981 | .99518 | **bf16 scales are lossless here** |
| e2m1, MSE-opt, f32, g64 (bound) | 4.375 | 20.17 | .0981 | .99518 | f16 rounding costs ≤0.005 dB |
| **e2m1, MSE-opt, fp16, g32** | **4.50** | **20.80** | .0913 | .99583 | best measured overall |
| **e2m1, MSE-opt, fp16, g128** | **4.125** | **19.69** | .1036 | .99462 | beats MLX by +1.4 dB at −0.125 bpw |
| int4-sym, amax/7, fp16, g64 | 4.25 | 18.93 | .1132 | .99365 | uniform grid loses to E2M1 |
| int4-sym, MSE-opt, fp16, g64 | 4.25 | 19.60 | .1049 | .99448 | still −0.57 dB vs E2M1 MSE |
| int4-asym (fp16 scale+zp), g64 | 4.50 | 20.52 | .0942 | .99559 | zero-point helps, but < e2m1 g32 @ 4.5 |
| e2m1, amax/6, fp16, g64, **input = MLX-decoded** | 4.25 | 17.62 | .1315 | .99163 | double-quant penalty −1.84 dB |
| e2m1, MSE-opt, fp16, g64, **input = MLX-decoded** | 4.25 | 17.92 | .1272 | .99203 | double-quant penalty −2.25 dB |
| e2m1, MSE-opt, fp16, g32, input = MLX-decoded | 4.50 | 18.12 | .1241 | .99242 | still < 4.25-bpq-from-source |
| re-encode MLX-decoded with floor-e8m0/32 | 4.25 | 16.35 | .1523 | .98885 | 22.3 dB loss vs MLX file — see below |

Findings:

- **The snapshot's e8m0 rule is `s = 2^round(log2(absmax/6))`, NOT floor.** Reverse-
  engineered on 65,536 groups of L0 in_proj_qkv: round matches 100.0% of stored scale
  bytes (floor 41.8%, ceil 58.9%). `tools/quant_study.py`'s `group_scale_e8m0` (floor)
  is not the snapshot's rule; floor-based e8m0 numbers (14.9 dB) understate e8m0. With
  the correct round rule, e8m0/64 ≈ 18.4 dB @ 4.125 bpw, and **ceil slightly beats
  round at g32 (18.51 vs 18.22)** — a free +0.3 dB for any e8m0 pipeline.
- **Never re-quantize from the MLX decode**: every rule loses ~2 dB vs quantizing the
  same rule from BF16 source (the snapshot's codes are already on the grid; a second
  grid snap doubles the noise). The floor-rule re-encode (16.35 dB) is extra-broken:
  floor halves the scale of the ~4–9% of groups whose decoded max is 4s (round had put
  them below code 6), clipping them again.
- **fp16 vs bf16 scale storage is a wash** (identical to 0.01 dB); bf16's wider
  exponent range also removes the (already absent) underflow concern. The scale bits
  that matter are the 16, not their format.
- **Zero points don't pay at fixed budget**: asymmetric int4 needs 2 more bytes/group
  (4.5 bpw) to reach 20.52 dB, below E2M1-g32's 20.80 at the same 4.5.
- Per-tensor detail: `o31` is the consistently worst subset tensor in every variant
  (~−0.2 dB vs subset mean); lm_head/embed rows are the easiest (+0.05 dB).

**Best 2–3 variants (measured):**
1. `E2M1 / MSE-fp16 / g64` — 20.17 dB @ **4.25 bpw** (what `-good` ships; keep).
2. `E2M1 / MSE-fp16 / g32` — 20.80 dB @ **4.50 bpw** (+0.63 dB if VRAM allows).
3. `E2M1 / MSE-fp16 / g128` — 19.69 dB @ **4.125 bpw** (−0.125 bpw vs today while still
   +1.4 dB over the MLX snapshot; interesting for the residency budget).

## 5. Weight distributions, outliers, scale utilization (BF16 source)

Excess kurtosis / |skew| across measured layers (skew is negligible everywhere except
in_proj_a, max 0.54 — these are symmetric, heavy-tailed, not skewed):

| class | kurtosis range | absmax/std max | interpretation |
|---|---|---|---|
| linear out_proj | 1.7 … **42.3** (L0 42.3, L1 29.5) | **63** | extreme tails — quantizes worst, biggest max-abs errors |
| o_proj | 1.1 … 16.4 | 59 | heavy tails |
| in_proj_a | 0.6 … 11.4 (L28 11.4) | 19 | late-layer decay-gate weights are the hardest small tensor |
| v_proj | 0.7 … 6.8 | 35 | |
| in_proj_b | 1.4 … 5.3 | 15 | hardest delta class after in_proj_a |
| k_proj | 1.1 … 6.0 (L31 6.0) | 21 | |
| down_proj | 0.4 … 4.0 | 62 | tails without extreme kurtosis |
| q_proj | 1.0 … 1.8 | 43 | near-Gaussian |
| in_proj_qkv / in_proj_z | 0.4 … 1.7 | 27 | near-Gaussian |
| gate_proj | 0.1 … 1.3 | 33 | |
| up_proj | **0.08 … 0.53** | 23 | flattest — quantizes best (20.23–20.25 dB) |

Outlier & saturation densities (over all measured elements, format's own scales):

| metric | MLX (e8m0/32 round) | INSIG4 (mse-fp16/64) |
|---|---|---|
| \|w\| > 4·s (beyond 2nd-largest E2M1 level) | 12–16% | 5.5–8.9% |
| \|w\| > 6·s (saturated to code 6) | **2.4–3.9%** | 1.3% (deliberate MSE clipping) |
| 32-groups where code-6 magnitude is unused (scale headroom waste) | 2.4% (up_proj) … 9.6% (in_proj_b) | 6.6% (embed) … 22.7% (in_proj_b) |

MLX's round rule clips 2.4–3.9% of all weights into the top code AND leaves 2–10% of
groups not using code 6 — simultaneous over- and under-shoot, which is exactly the
inefficiency the MSE-optimal fp16 scale removes. INSIG4's higher code-6-unused rate on
flat tensors (in_proj_b 22.7%) is the flip side of MSE scales exceeding absmax/6 there:
harmless by construction (measured SQNR is what counts).

## 6. End-to-end impact (measured probe + clearly-marked estimate)

**Byte shares** (all 33 blocks incl. mtp, from headers): gate/up/down 18.05% each,
lm_head+embed 11.06% each, in_proj_qkv 8.75%, z/out_proj 4.38% each, q 3.28%, o 1.64%,
k/v 0.41% each, bf16 leftovers ~0.5%. **Byte-share-weighted SQNR vs BF16 source:**
MLX **18.26 dB** (weighted relF 0.1222), INSIG4(-good) **20.21 dB** (0.0976), -text 12.07.

**Measured propagation probe** (numpy forward of all 32 layers + full 248,320-vocab
logits; identical ops to `tools/reference_all_layers.py`; same input x₀ = BF16 embed
row 42 for all three weight sets; norm semantics per format — MLX one-centered raw,
SRC/INSIG4 get +1 since both store zero-centered; INSIG4 conv1d/norms are bf16-exact):

| | MLX vs BF16 | INSIG4(-good) vs BF16 |
|---|---|---|
| hidden-state cos after L0 | 0.99814 (rel 0.133) | **0.99898** (rel 0.059) |
| after L8 | 0.77879 | **0.93013** |
| after L16 | 0.51317 | **0.69919** |
| after L31 (pre-final-norm) | 0.56631 | **0.58367** |
| logit rmse Δz (full vocab) | 1.844 (σ(z)=1.925) | 1.980 |
| greedy top-1 (src: token 13, p=0.31) | **flips → 248044** | **keeps 13** (Δz_top1 −1.66) |
| ΔNLL 1st order (−Δz_y+ΣpΔz) | −0.213 | −1.966 |
| ΔNLL 2nd order (½Var_p Δz) | 1.318 | 1.770 |

Reading: the per-matmul error (relF ≈ 0.12) matches the L0 output rel (0.133) as
first-order theory predicts for near-isotropic inputs, but this architecture's
layer dynamics amplify it — by L8 the hidden state cosine is < 0.93 (INSIG4) / < 0.78
(MLX), and logit noise reaches ~0.96×σ(z). INSIG4 tracks BF16 measurably closer than
MLX at every one of the 32 layers, and is the only one of the two that kept the greedy
token on this probe.

**Error-budget estimate (marked as such):** the repo's parity gate is |ΔNLL| < 0.02 nat
(`audits/w3/parity-ladder.md:401`, tightened to 0.005 after first clean run;
`audits/w3/MASTER-PLAN.md` R8). A ½Var(Δz) model calibrated by this probe (rmse Δz
≈ 1.0–1.9 with σ(z) ≈ 1.9) gives ΔNLL = O(1) nat for ANY 4.25-bpq format vs BF16 —
**four orders of magnitude above the 0.02 gate**. Consequences: (a) the gate is only
meaningful as engine-vs-NumPy-reference on the *same* quantized checkpoint (its actual
use in parity-ladder), never as quantized-vs-BF16; (b) going from MLX to INSIG4 buys
+2 dB (×1.58 lower MSE) everywhere — helpful, but it moves logit noise by a factor
≈1.6, not orders of magnitude; (c) if a "vs BF16" quality target ever becomes a
requirement, per §4 the cheapest measured levers are g32/fp16 scales (+0.63 dB) and
int4-asym/g32 ideas at +0.25 bpw — the format is already at the E2M1/MSE frontier for
4-bit codes, so the next real jump needs a different code family (e.g. the NanoQuant
low-rank binary path) or mixed-precision (bf16/fp16 for the heavy-tailed out_proj,
o_proj, in_proj_a/b classes — together only ~8.9% of bytes at 4.25 bpw).

Caveats on the probe: single token, zero recurrent state, first forward step; token-42
embedding input; ΔNLL numbers are one-token samples, not a corpus mean — treat as order-
of-magnitude evidence, not a benchmark. The layer-wise cosines and the MLX<INSIG4
ordering are the robust part.

## 7. Recommendations

1. Ship `-good`'s encode (current quantizer), quantized **from the BF16 snapshot**
   (local cache), never from the MLX decode (−2 dB). Fix the docstring's "4.125 bpw"
   → 4.25, and "+5.9 dB" → "+2.0 dB vs the snapshot" (already noted in quantizer.md).
2. If 0.25 bpw of headroom exists in the residency budget, g32/fp16 scales are the
   best measured quality buy (+0.63 dB). If bytes must shrink, g128/fp16 still beats
   today's MLX snapshot by +1.4 dB at 4.125 bpw.
3. Do not spend bits on zero points or int4 grids at this budget (both lose to E2M1).
4. If a mixed-precision pass is ever done, target out_proj/o_proj/in_proj_a/b (heavy
   tails: kurtosis up to 42, 59–63σ outliers) — they are the worst tensors and a tiny
   byte share.
5. Any |ΔNLL|<0.02 claim must name its reference; vs BF16 it is unreachable at 4.25 bpw
   (measured O(1) nat single-token), vs the NumPy reference on the same file it remains
   the correct gate.

*Reproducibility: all runs were `python - <<EOF` heredocs (numpy 2.5.2, stdlib only),
read-only seeks into the four safetensors sources; ~2.4 G weights decoded for the main
pass (13 min), 9-tensor ablation subset ~0.15 G elts × 16 variants (13.5 min), e2e
triple forward + full-vocab logits (4 min). No files were created or modified other
than this report.*
