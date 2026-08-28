# reference27.py — NumPy ground truth for Qwen3.8-27B-FP8

Date: 2026-08-25. Deliverable of the W3 ladder: `tools/reference27.py` (one
script, subcommands) implementing the independent NumPy reference for the 27B
FP8 checkpoint. No torch exists locally (parity-ladder §0), so this script IS
the ground truth for R4–R9. Model dir is only ever read (np.memmap views);
nothing writes to `Qwen3.8-27B-FP8`.

## 1. Loader

- Header parsed by hand per the i4-script pattern: `u64 len + json`, then
  `np.memmap(path, u1, mode='r')` sliced at `8+hdr+data_offsets` and viewed as
  `u1` (F8_E4M3) / `<u2` (BF16) — no `safetensors` lib dependency.
- Shard names per §1.1 census: `layers-{N}.safetensors` → prefix
  `model.language_model.layers.{N}.`; `outside.safetensors` →
  `model.language_model.embed_tokens.weight` (bf16 [248320,5120], row-sliced
  per token, never materialized), `lm_head.weight` (same, chunked),
  `model.language_model.norm.weight`.
- Dequant: `W = LUT8[f8] × bf16(weight_scale_inv)`, scales `[rows/128, cols/128]`
  broadcast per block, built 128 rows at a time (all 27B dims are multiples of
  128, so repeat is exact). Verified live: dequant amax 0.436 = 9.73e-4 (max
  scale) × 448 — exactly the audited scale span.
- e4m3 decoder = OCP `fn` variant (bias 7, subnormals m·2⁻⁹, NaN only
  0x7F/0xFF, max ±448); `selftest` compares all 256 codes bitwise against an
  independently hand-built `math.ldexp` table plus anchors/monotonicity/sign
  symmetry. PASS.

## 2. Formula sheet as implemented

### 2.1 Norm centering (the §7.1 split)

| tensor | centering | as implemented |
|---|---|---|
| input_layernorm, post_attention_layernorm, q_norm, k_norm, model.norm | ZERO | `1.0 + w` |
| linear_attn.norm (RMSNormGated) | ONE | raw `w` |

### 2.2 Gated DeltaNet layer (48 layers, `(l&3)!=3`)

```
nrm  = x/sqrt(mean(x²)+1e-6) · (1+w_ln)
qkvp = W_qkv[10240,5120] @ nrm;  z = W_z @ nrm
a = W_a(bf16)[48,5120] @ nrm;    b = W_b(bf16)[48,5120] @ nrm
A_log, dt_bias: bf16 → f32 [48]
conv (causal, pad-left 3, taps w[:,0..3], newest = 3):
  y = c0·s0 + c1·s1 + c2·s2 + c3·qkvp ; state keeps RAW pre-SiLU inputs
  y = silu(y)                        # whole 10240 channels (q, k AND v)
q = y[:2048]→[16,128]; k = y[2048:4096]→[16,128]; v = y[4096:]→[48,128]
q̂ = q/sqrt(‖q‖²+1e-6) · 1/√128 ; k̂ = k/sqrt(‖k‖²+1e-6)   (k: no scale)
k-sharing: v-head j → k-head j//3 (repeat ×3)
β = sigmoid(b);  α = exp(−exp(A_log)·softplus(a+dt_bias))   [logaddexp form]
```

**S orientation decision (documented per mission):** state per v-head is
stored `[128 k, 128 v]` — the literal `src/deltanet.cu` layout
`state[i*128+tid]` with `i` = key index, `tid` = value index. The GPU computes
`mem` as a dot **over the key axis** for each value, so in NumPy:

```
S *= α[:,None,None]                       # decay FIRST, δ un-decayed
mem = einsum('hkv,hk->hv', S, k̂48)       # mem[v] = Σ_k S[k][v]·k̂[k]
δ   = (v − mem)·β
S  += k̂48[:,:,None] · δ[:,None,:]        # S[k][v] += k̂[k]·δ[v]  (outer k⊗δ)
out = einsum('hkv,hk->hv', S, q̂48)       # out[v] = Σ_k S[k][v]·q̂[k]
```

The 9B i4 scripts stored the transpose (`'hvk'`, S[v][k]) — same math, different
layout. This reference matches the GPU layout literally, so an engine dump of
delta states maps over with **no transpose**.

```
out = out/sqrt(mean(out²,1)+1e-6) · w_norm(raw) · silu(z)   # per 128-wide v-head
x'  = x + W_out[5120,6144] @ out.reshape(-1)
MLP: x'' = x' + W_down @ (silu(W_gate @ nrm2) · (W_up @ nrm2)),  nrm2 with (1+w_post)
```

### 2.3 Full-attention layer (16 layers, `(l&3)==3`)

```
raw = (W_q[12288,5120] @ nrm).reshape(24, 512)   # per head [256 q | 256 gate]
q, gate = raw[:,:256], raw[:,256:] ; k,v = [4,256] each
q = RMS256(q)·(1+w_qn);  k = RMS256(k)·(1+w_kn)  # per-head, eps 1e-6
rope: 64 dims, θ=1e7, inv_i = θ^(−i/32), f64 angles → f32, rotate_half pairs
      (i, i+32): out[i] = x[i]c − x[i+32]s; out[i+32] = x[i+32]c + x[i]s
causal softmax over t ≤ pos, scores × 0.0625 (1/√256), GQA kvh = h//6
out = attn · sigmoid(gate)                        # gate AFTER value mix
x' = x + W_o[5120,6144] @ out.reshape(-1) ; MLP as above
```

### 2.4 Heads / lm_head / tokenizer

- embed: bf16 row slice of `embed_tokens` at the header-known offset.
- final norm: `(1+w)` then lm_head.
- lm_head bf16 [248320,5120]: read in 8192-row chunks (168 MB f32 scratch);
  NLL uses an online f64 logsumexp across chunks for all positions in one
  sweep; greedy keeps a running argmax (strict `>`, so lowest index wins ties).
- `enc`/`dec` load the model dir's own `tokenizer.json` via `tokenizers`
  (9B tokenizer is not a fallback). `enc` of the prompt-A sentence reproduces
  GOLDEN-14 exactly.

## 3. Memory policy (16 GB host, < 6 GB target)

One layer shard dequantized at a time, freed before the next: measured live
weight footprint **1.43 GiB** (linear layer) / **1.39 GiB** (attention layer);
dequant transients ≤ ~10 MB (128-row blocks). Persistent decode state:
48 × 48×128×128 f32 = 151 MB delta + 6 MB conv + KV (T≤64: ~17 MB). embed and
lm_head are never materialized. Observed process footprint stays ≈ 2 GB.

## 4. Commands (ladder rungs)

```
# R4 gate numbers (layer 0 = first DeltaNet seam):
python tools/reference27.py layer 0 760,3712,314,23470,25044
# R5 gate numbers (layer 3 = first full-attn seam; runs 0..3 from embed):
python tools/reference27.py layer 3 760,3712,314,23470,25044
# R6: 65 seams × T × 5120 f32 npy (seam-major, same layout as the engine dump):
python tools/reference27.py seams 760,3712,314,23470,25044,369,303,919,279,3712,314,16465,33633,13 build\27b\pf-seams.npy
# R8: NLL over GOLDEN-128 (f64 logsumexp):
python tools/reference27.py nll <GOLDEN-128 csv>
# R9: greedy continuation (teacher-forced prefill + n chunked argmax steps):
python tools/reference27.py greedy 760,3712,314,23470,25044,369,303,919,279,3712,314,16465,33633,13 8
# unit checks / tokenizer glue:
python tools/reference27.py selftest
python tools/reference27.py enc "The history of computing machinery is in part the history of automatic arithmetic."
python tools/reference27.py dec 760,3712,314
```

`layer N` runs layers 0..N over the ids starting from embed (the true seam-N
input) and saves `[N+1, T, 5120]` f32 (default `build\27b\layer{N}-traj.npy`,
`--no-save` to skip, `--out` to redirect). Engine-dump comparison snippet:

```python
import numpy as np
ref = np.load(r'build\27b\pf-seams.npy')          # [65, T, 5120]
eng = np.fromfile(r'build\27b\pf-seams.f32', np.float32).reshape(65, T, 5120)
cos = (ref*eng).sum(-1)/np.linalg.norm(ref,axis=-1)/np.linalg.norm(eng,axis=-1)
print(cos.min(), np.median(cos))
```

## 5. Measured validation (this session)

- `selftest`: **PASS** (256 codes: 254 finite + 2 NaN at 0x7F/0xFF, max ±448,
  subnormals m·2⁻⁹, bitwise-identical to the hand-built ldexp table, strict
  monotone 0x01..0x7E, sign-symmetric, anchors 1.0/1.875/240/448).
- **Layer 0 (R4 gate)**, tokens [760,3712,314,23470,25044], runtime 2.4 s:
  - output norms: 20.625986, 18.218061, 11.821807, 15.866237, 17.408922
  - tok 760 first5: [-0.025824, 0.034177, -0.015855, -0.059582, -0.061536]
- **Layer 3 (R5 gate)**, same tokens, runtime 8.8 s (includes layers 0–2):
  - L1 D norms: 31.9206, 22.2994, 15.3814, 18.4050, 19.4348
  - L2 D norms: 45.5452, 22.3036, 17.0509, 21.0979, 17.6378
  - L3 A norms: 52.917900, 33.867325, 26.609972, 30.375782, 29.116676
  - tok 760 first5: [0.007869, 0.013311, -0.203784, -0.064044, -0.104842]
- Causality probe: changing the last token leaves layers 0–3 outputs of all
  prefix tokens **bitwise identical** (conv shift + delta recurrence + KV
  append bookkeeping correct); changed token propagates (maxdiff 6.97).
- Full model: `nll 760,3712,314,23470,25044` → per-position
  [6.456270, 0.039546, 6.300881, 7.196468] nat, mean 4.998291, ppl 148.16
  (2 m 31 s; the 0.04-nat "history of" shows real LM behavior — a broken
  pipeline would sit near ln(248320) ≈ 12.4 everywhere).
- `greedy 760,3712,314,23470 1` → argmax 369 (" is") in 4 m 45 s (64-layer
  prefill + 1 decode step ≈ 2.5 min each; dequant-bound).
- `seams 9419,0` → [65, 2, 5120] f32, all finite; final-norm seam norms
  [139.524277, 138.446106].

## 6. Remaining TODOs

1. **MTP reference** (R9 draft parity): mtp.safetensors math is specified
   (qwen35-arch §3: `fc @ concat(rmsnorm(embed_row, pre_fc_norm_embedding),
   rmsnorm(hidden, pre_fc_norm_hidden))`, embed-normed half FIRST, then one
   full-attn layer with own KV, `mtp.norm`, shared lm_head) but not wired into
   LayerSet (it only opens `layers-N.safetensors`).
2. **R7 multistep dump compare**: the reference persists conv/delta/KV state
   across greedy steps in-process and prints per-step argmax, but a
   dump-vs-ref cosine script (clone of `reference_multistep_i4.py` reading the
   engine's `multi.f32`) still needs writing once `dump_multistep_27b` exists.
3. **Batch seams for weight-stationary dumps**: if the engine's
   weight-stationary prefill dumps full chunks per layer, a batched matmul
   path (T×5120 @ W.T) in `run_layers` would cut per-token GEMV overhead;
   today it is per-token (dequant dominates anyway at these T).
4. Speed: greedy ≈ 2.5 min/token (dequant of 25.65 GB per step). If this ever
   matters, a sliced matvec over the u8 view would skip materializing f32
   weight copies (~4× less write traffic) at the cost of code clarity.
5. `layer N --attn` on a linear slot will KeyError on the shard read (warning
   printed); it is only meaningful for attention-shaped shards (N%4==3).
