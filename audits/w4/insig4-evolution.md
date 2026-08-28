# INSIG4 format evolution — measured menu + top-2 specs

Agent: w4 quant-format audit, 2026-08-25.
Everything below is measured on this machine unless marked [lit]. Ground truth weights =
BF16 `Qwen--Qwen3.5-9B` snapshot `c2022362…`; activations = real engine seams dumped by
`build/dump-multistep.dll` on `build/qwen35-insig4-good.insignia-index` (48 teacher-forced
tokens of `build/prompt512.txt`; dump = (49,33,4096) f32). Study quantizer replica verified
**bit-identical** to the shipped scales (`tools/quantize_insig4.py` golden-section, 100%
of qkv_L0 scales match exactly), so all deltas are vs the real shipped format.

## 0. Measurement infrastructure status

- `python tools/rundll.py build/generate.dll nll <index> <tokens>` runs (400 evals).
  INSIG4-good: **nll=4304.5**; MXFP4-native (`qwen35.insignia-index`): **nll=39.28**.
  Both are orders of magnitude above a sane ~2–4 nats → the known full-attention parity
  bug dominates; downstream NLL cannot isolate quantization quality yet. All quality
  numbers below are therefore activation-weighted output SQNR computed in numpy
  (diag-H and, where noted, full-H validated), which is the metric of record this week.
- Real per-projection input second moments `h = E[x_j^2]` were collected:
  exact from engine seams (rmsnorm inputs: in_proj_qkv/z/a/b, q/k/v, lm_head input),
  and via the reference block math keyed to real seams for MLP (post-attn norm input,
  down_proj input) at layers 0/3/16. Reference block tracks the engine at next-seam
  cos 0.989 (L0) / 0.995 (L16); L3 = 0.82 (the parity bug itself) — h *spread statistics*
  at L3 match L0/L16, caveat noted.
- The key physical fact: **within-64-column-group h spread is p99/p50 = 24–558x**
  (max/mean 450–4100x); H off-diagonal mass is only 2–25% of diagonal → diagonal
  weighting captures almost everything (full-H check below confirms).
- On-disk scale range: 1.4e-3 … 3.5e-2 — well inside fp16 normals (min normal 6.1e-5).

## 1. The measured menu (weight-SQNR / output-SQNR, dB; higher better)

2048 sampled rows/tensor, BF16 originals, real h per tensor. "out" = Σh·(W−Q)² weighted
(diag-H); full-H rows validated separately.

| variant | bpw | Δbytes | qkv_L0 | q_L3 | o_L3 | gate_L16 | down_L0 | lm_head |
|---|---|---|---|---|---|---|---|---|
| e8m0/32 (MLX MXFP4) | 4.25 | 0 | 13.9/12.3 | 14.2/14.2 | 13.9/13.8 | 14.1/14.4 | 14.4/14.4 | 13.9/13.8 |
| **f16/64 (current INSIG4)** | 4.25 | 0 | 20.2/21.4 | 20.2/20.8 | 20.2/20.2 | 20.2/20.2 | 20.2/20.2 | 20.2/20.3 |
| bf16/64 | 4.25 | 0 | identical to f16/64 (±0.01 dB) | | | | | |
| e8m0/64 (packed) | 4.125 | −144 MB | 18.9/18.1 | 18.8/18.3 | 18.9/18.8 | 18.7/18.8 | 18.7/18.7 | 18.8/18.8 |
| f16/128 | 4.125 | −144 MB | 19.8/20.5 | 19.8/20.2 | 19.7/19.7 | 19.7/19.6 | 19.8/19.8 | 19.7/19.8 |
| f16/32 | 4.5 | +287 MB | 20.9/22.1 | 20.9/21.3 | 20.8/20.8 | 20.8/20.8 | 20.9/20.8 | 20.9/20.9 |
| **wfit/64 (Hessian-weighted scales)** | **4.25** | **0** | 19.7/**31.6** | 19.9/**30.1** | 20.0/20.7 | 19.7/**26.0** | 19.8/21.5 | 19.9/21.0 |
| asym/64 (wfit + zero-point) | 4.5 | +287 MB | 19.8/**36.8** | 20.1/**33.4** | 20.1/21.2 | 19.8/**27.6** | 19.9/22.4 | 20.1/21.6 |
| low-rank r=16/32/64 on top (NanoQuant-lite) | +0.08–0.38 | +2–9% | +1.1/+1.3/+1.8 out-dB (qkv); +0.2–0.3 (down); +1.3–1.7 (gate) | | | | | |
| outliers top-0.5%/1%/2% fp16 (on top of wfit) | +0.16/0.32/0.64 | +33/66/132 MB (18% of mass) | **36.2**/37.0/37.9 | — | — | — | 22.3/22.7/23.4 | — |

Full-H (true output MSE incl. off-diagonals), qkv_L0: current 22.12 dB → wfit **32.20 dB**
(+10.1 dB = 10x lower output error). lm_head: 27.31 → 27.99. Diag slightly *under*states
the wfit gain — the numbers above are conservative.

**Held-out validation** (fit h on tokens 0–23, evaluate H from tokens 24–47), qkv_L0:
current 22.02 dB → wfit 31.76 (1 iter) / 32.18 (3 iters). The gain is real, not
overfitting to the estimation set. Lloyd converges in 1–2 iterations.

Model mass (9,198.3M quantized params): MLP 52.5%, embed+lm_head 22.2%, linear_attn
qkv/z/out 17.6%, attn q/k/v/o 4.8%. Current file = 4.887 GB @ 4.25 bpw.

## 2. Analysis of each option

### (1) Hessian/greedy scale fitting — THE headline result
Current quantizer minimizes unweighted per-group MSE. Replacing the objective with
Σ_j h_j (w_j − s·g_j)² (per-column activation second moments, GPTQ/AwQ-style diagonal
Fisher, EXL3-style spirit) costs **zero bytes** — same layout, same F16 scales tensor,
kernels untouched — and delivers:

| tensor | out-SQNR gain | | tensor | gain |
|---|---|---|---|---|
| la.out_proj L1 | **+14.3 dB** | | mlp.gate L0/L3 | +4.7 / +5.4 dB |
| la.out/z/qkv L0, la.qkv L1 | +9.4…+9.5 dB | | mlp.gate/up L16 | +5.8 / +3.3 dB |
| attn.q L3 / L7 | +9.2 / +5.9 dB | | mlp.down L0/L16 | +1.3 / +2.3 dB |
| la.z L1 | +7.9 dB | | o_proj / lm_head | +0.5 / +0.7 dB |

Closed-form update (no search needed): with assignments fixed (nearest E2M1 point is
still optimal per-element under diagonal weighting), the optimal scale is
**s\* = Σ_j h_j w_j ĝ_j / Σ_j h_j ĝ_j²**; iterate assign→update 2–3x (Lloyd). Median
scale shift is ~0 but a small fraction of groups (the h-skewed ones, exactly where the
output error lives) move a lot — that's the whole gain. Bonus: 3 closed-form iterations
replace the 10-iteration golden search → **quantizer gets ~3x faster** at scale-fitting.

### (2) Scale dtype: fp16 vs bf16 vs e8m0
- **bf16 == fp16 to 0.01 dB** (scales sit at 1e-3..3e-2; mantissa granularity of either
  is invisible next to E2M1 error). No reason to change.
- **e8m0/64 saves 144 MB but loses 1.8–2.4 dB out-SQNR vs f16/64**, and at the *same*
  4.125 bpw it is **dominated by f16/128** (which loses only 0.5–0.9 dB). Math check:
  e8m0 only saves bytes when it replaces an f16-per-64 (16→8 bits/64); a pow2 lattice
  of scales throws away exactly the freedom the MSE-optimal fit exploits. Do not pursue.

### (3) Group size 64 vs 32 vs 128
f16/32 buys +0.5–0.65 dB out-SQNR for +287 MB (+5.9%); f16/128 saves 144 MB for
−0.5–0.9 dB. Both are **~20–50x worse dB-per-byte than wfit (free)**. Verdict: keep 64
globally; group-32 only as a per-tensor patch for the few worst tensors (see (6)).

### (4) Asymmetric (zero-point) E2M1
Measured with joint weighted (s,z) Lloyd (both closed-form): **+2–5 dB out-SQNR over
wfit** on la.*/attn.q tensors (qkv 31.6→36.8), ≤ +1 dB elsewhere. It works because
outlier-heavy groups are strongly skewed — the signed symmetric grid wastes half its
range. Costs +0.25 bpw (fp16 z per 64-group) **and** a kernel correction term
(y += z·Σ_group x), which breaks the dp4a pair kernels' integer trick. Good phase-2
candidate after outliers (below) — they reach the same quality cheaper.

### (5) Low-rank bf16 residual (NanoQuant-lite, arXiv 2602.06694 spirit)
r=16 on [12288,4096] = +0.083 bpw (+1 MB/tensor); on qkv +1.1 dB, on gate +1.3 dB,
on down +0.2 dB. GEMV cost is genuinely tiny (Uᵀx then V·, ~0.4% FLOPs at r=64), but
**dB-per-bpw is 10–30x worse than wfit and ~5x worse than outliers**. Low-rank pays at
1–2 bpw (its NanoQuant home turf), not at 4.25 bpw where E2M1+good scales already sit.

### (6) Mixed per-tensor granularity — sensitivity ranking
Most sensitive tensors under the CURRENT format (out-SQNR with real h, current fit):
**mlp.gate/up L0 & L3 ≈ 10.6–11.2 dB** (worst in model), attn.v L3 15.4, attn.k L3 16.1,
mlp.down L0 20.2. After wfit the floor rises to ~15–16 dB (gate/up L0/L3), 16.6 (attn.v L3).
Cheap targeted patch: gate/up L0+L3 to f16/32 (+6.3 MB total) or +outliers (+16 MB at k=1%).
Least sensitive: o_proj, lm_head, down (mid layers) — candidates for f16/128 if bytes
ever matter (−144 MB at −0.5 dB class).

### (7) Salient-outlier escape (SpQR-style)
Top-k by |w|·√h, removed before scale refit, stored exact fp16, k=0.5%: **qkv 31.1→36.2 dB
for +0.16 bpw (+33 MB over the 18% of mass that is la.*/attn.q)**; k=1% → 36.95. down_L0
21.6→22.3. Crucial kernel property: outlier positions quantize to **code 0 naturally**
(zero out → nearest code is 0) → **all existing main kernels stay byte- and semantics-
identical**; the correction is one extra tiny CSR pass per matvec.

## 3. Ranking (quality gain / implementation risk)

| # | option | quality | bytes | kernel risk | verdict |
|---|---|---|---|---|---|
| 1 | **wfit scales** | +6…+14 dB out-SQNR on 18% of mass; +0.5–5.8 on the rest | 0 | **none** (same layout) | **DO NOW** |
| 2 | **outlier sidecar on la.\*/attn.q (+L0/L3 gate/up)** | +5.1 dB (qkv), k=0.5% | +33 MB | one small CSR kernel; main kernels untouched | **DO NOW/NEXT** |
| 3 | asym z-point | +2–5 dB over wfit | +287 MB if global, +52 MB selective | correction in every i4 kernel + dp4a rework | phase 2 |
| 4 | mixed granularity (g32 for gate/up L0/L3; g128 for o/lm_head) | ±0.5–1 dB | −140…+6 MB | GEMV smem layout per group-size (v2_i4 hardcodes groups>>1 scale stride; g32 = stride groups, +0.5 KB smem/row) | optional, low value |
| 5 | low-rank residual | +0.2–1.8 dB | +2–9% | extra r-dot in every matvec | skip at 4.25 bpw |
| 6 | e8m0 anything | negative vs f16/128 | −144 MB | none | **rejected (measured)** |
| 7 | bf16 scales | 0.00 dB | 0 | none | rejected (no-op) |

## 4. Top-2 specs for this week

### 4.1 RECOMMENDATION 1 — Hessian-weighted scale fitting (INSIG4.1, zero-cost)
**Format: unchanged.** Same U32 nibbles, same F16 [rows, cols/64] scales tensor, same
index, same kernels. Only the scale *values* change. Engine does not need a rebuild to
evaluate (the existing good-index path already reads whatever quantizer emits).

`tools/collect_h.py` (new, ~60 lines, reuses this audit's method):
1. `dump-multistep.dll` seams for 48–256 tokens (already built; use ANY working index —
   h profiles are format-insensitive; for A-layer fidelity wait on the parity fix or use
   the reference block path like this audit did).
2. Accumulate `h += rmsnorm(seam_l, w_in_l)²` per layer → `hdiag` for in_proj_qkv/z/a/b,
   q/k/v_proj. lm_head: final-norm row (seam col 32).
3. For gate/up/down: reference block math keyed to real seams (code exists in
   `tools/reference_multistep_i4.py`; layer 0/3/16-style subset is enough — h profiles
   are stable across layers within a family; measured spreads agree within 2x).
4. Save `build/hdiag/<layer>.<proj>.npy` (float32, 4096/12288).

`tools/quantize_insig4.py` diff (quantize_tensor only):
```python
def quantize_tensor(w, h=None):          # h: per-column E[x^2], len == cols (or None)
    ...
    s = optimal_scale(g64)               # keep as init (or 4-iter search)
    if h is not None:
        hg = np.tile(h.reshape(-1, 64).astype(np.float32), (rows, 1))  # aligned (G,64)
        for _ in range(3):
            q = g64 / s[:, None]
            g_hat = np.sign(q) * E2M1[np.abs(q[..., None] - E2M1).argmin(-1)]  # nearest
            den = np.sum(hg * g_hat * g_hat, 1); num = np.sum(hg * g64 * g_hat, 1)
            s = np.where(den > 1e-30, num / np.maximum(den, 1e-30), s)
        s = s.astype(np.float16).astype(np.float32)
    # codes as today (chosen against the stored scale)
```
(nearest-grid via the existing broadcast; 3 iterations; measured convergence 1–2.)
Acceptance: per-tensor out-SQNR table from this audit (§1) as parity target; qkv_L0
wSQNR ≥ 29 dB; MLP ≥ +1 dB; lm_head ≥ +0.4 dB. Then rebuild index with
`tools/index_safetensors.py` and rerun the NLL harness — expect the INSIG4-vs-MXFP4
NLL gap to shrink once the attention parity bug lands its fix.

### 4.2 RECOMMENDATION 2 — top-0.5% fp16 outlier sidecar (INSIG4.2, +33 MB)
Apply to: `linear_attn.in_proj_qkv/in_proj_z/out_proj` (all 24 D-layers), `self_attn.q_proj`
(8 A-layers), optionally `mlp.gate/up` L0+L3. (~1.7B params, 18.5% of mass.)

Quantizer (after wfit scales are final):
```python
imp = np.abs(W) * np.sqrt(h)[None, :]            # h from collect_h
thr = np.quantile(imp.ravel(), 1 - k)            # k = 0.005
mask = imp >= thr
Wm  = np.where(mask, 0.0, W)
Q   = quantize_tensor(Wm, h=h)                   # refit scales WITHOUT outliers (wfit)
# outlier codes are 0 automatically (Wm==0 -> nearest code 0): main kernels untouched.
# emit sidecar: per row, u16 count, then (u16 col, f16 val) pairs  -> 4 B/outlier
```
Index/format: two new optional tensors `X.olist` (U8 blob) + `X.ocounts` (U16 [rows]);
absence = old behavior (engine checks the index, one branch, constexpr per-matrix).

Kernel (`src/mxfp4_i4.cu`, new ~30-line kernel, no edits to existing ones):
```cuda
// one warp per row; y[row] += sum(val * x[col]) over the row's CSR segment
__global__ void outlier_gemv_kernel(const uint32_t *__restrict__ csr,   // (col|val16<<16) pairs
                                    const uint16_t *__restrict__ cnts,
                                    const float *__restrict__ x, float *__restrict__ y, int rows)
```
- ~20 pairs/row at k=0.5% (4096 cols) → 20 scattered x loads from L1/L2 (x is 16 KB,
  fully resident) + 20 FFMA per row vs the 4096-elt main dot → <1% of GEMV time;
  zero extra traffic on the weight stream.
- Launch after each affected `mxfp4_gemv_v2_i4`/pair call in `src/decode.cu` (or fuse as
  a block tail later — keep it separate first for bisectability).
- Prefill (`src/gemm.cu`): same kernel per token, or a CSR-GEMM over the ≤64-token tile;
  outlier mass is 0.5% → any naive loop is fine.
- dp4a pair kernels unaffected (code-0 outliers contribute 0 integers).
Acceptance: qkv_L0 out-SQNR ≥ 36 dB (measured 36.2); NLL harness delta once parity fixed;
kernel timing: decode step-time within noise (±0.3 ms of 4 ms MTP / full-step budget).

## 5. What NOT to do (measured)
- e8m0 scales in any packing (dominated by f16/128 at equal bytes, §2.2).
- bf16 scales (zero delta).
- Low-rank residual at 4.25 bpw (dB/bpw 10–30x worse than free wfit).
- Global g32 (+287 MB for +0.6 dB; buy outliers instead: +5 dB for +33 MB).

## 6. Loose ends
- NLL A/B numbers (4304 vs 39.3) are engine-bug-dominated; re-run both indexes after the
  full-attention parity fix — the harness itself works and is the downstream gate for
  INSIG4.1/4.2.
- h for attention-layer MLPs relies on the reference block (cos 0.82 vs engine at L3);
  once parity lands, re-dump seams and refresh `build/hdiag/` (cheap: 1.5 s/layer).
- This audit's scratch data lives in %TEMP%\audit_*.npy + insig4_audit_seams.f32
  (rebuildable from the commands above; nothing in the repo was modified).
