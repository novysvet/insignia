# W4: 27B decode attention + DeltaNet kernels — shape audit, perf model, split-K spec

Scope: `src/attention.cu`, `src/deltanet.cu` read in full; supporting kernels in
`src/ops.cu`, `src/qwen_kernels.cu`, `src/prefill.cu`, call sites in `src/decode.cu`;
reference `tools/reference27.py`; master plans `audits/w3/attn-27b.md` §6-7 and
`audits/w3/deltanet-27b.md`. Ran the already-built binaries (no rebuild):

- `build/test-attention.exe` → `GQA decode T=257 max_abs=2.56114e-09` PASS (9B shapes)
- `build/test-deltanet.exe` → `max_abs=9.31323e-10 max_rel=3.28628e-05  0.010 ms/layer`
  (9B shapes, 1000-iter timing loop — the one hard measurement this audit anchors on)

All 27B perf numbers below are analytic estimates calibrated to that measurement; the
only-bench rule applies before any of them is trusted as a decision.

---

## 1. Kernel shape-assumption classification (what is hardcoded vs runtime)

| kernel (file:line) | hardcoded today (9B) | runtime | 27B requires | state |
|---|---|---|---|---|
| `gqa_decode_kernel` (attention.cu:7) | 16 heads via `<<<16,256>>>`; `kvh=head>>2` (4 kvh, group 4); head_dim 256 (`d<256`); KV row `(t*4+kvh)*256`; `score[4096]` (ctx cap); scale `.0625f` | `tokens=__ldg(pos_dev)+base+1`; `max_context` ignored | grid 24; `kvh=head/6` | **BUG** `>>2` (16/24 heads wrong, heads 16-23 OOB — attn-27b §0) |
| `gqa_decode` launcher (attention.cu:8) | `<<<16,256,0,s>>>` | — | `<<<24,256,0,s>>>` | edit |
| `qk_norm_rope` (ops.cu:9) | `<<<20,256>>>`; `isq=head<16`; `k+(head-16)*256`; dim 256; rope 64; θ=1e7; eps 1e-6 | `pos=__ldg(pos_dev)+off` | `<<<28,256>>>`; `isq=head<24`; `head-24` | edit (θ/64/256 literals all stay) |
| `qk_norm_rope_batch` (prefill.cu:54-86) | `dim3(20,T)`; `head<16`; `t*16` strides | `pos_dev[0]+t` | `dim3(28,T)`; `head<24`; `t*24` | edit |
| `store_kv` (qwen_kernels.cu:15-16) | `i<1024` row (4 kvh × 256); `pos*1024` | pos from device | **unchanged** | verified |
| `split_q_gate` (qwen_kernels.cu:73-74) | `<<<16,256>>>`; `i<4096`; `h=i>>8`; 512 interleave | — | `<<<24,256>>>`; `i<6144` | edit (`h>>8` stays valid) |
| `expand_gate_heads` (qwen_kernels.cu:78-79) | `<<<16,256>>>`; `i<4096` | — | `<<<24,256>>>`; `i<6144` | edit |
| `conv4` (qwen_kernels.cu:7-8) | **none — n-generic** grid-stride | `n` from caller; decode.cu:128 passes `8192` | pass `10240` + state stride `10240*3` | call-site only |
| `params` (qwen_kernels.cu:9-10) | `<<<1,32>>>` grid literal; guard `i<n` | `n` (decode.cu:128 passes 32) | `<<<1,48>>>`, n=48 | **TRAP**: with grid 32 and n=48, heads 32..47 are silently never computed (stale a/b) |
| `params_batch` (prefill.cu:203-214) | `if(h>=32) return`; `t*32` strides | T | `h<48`; `t*48` | edit |
| `deltanet_decode_kernel` (deltanet.cu:4-13) | 32 heads via `<<<32,128>>>`; `kh=head>>1` (2:1); K=V=128; `0.08838834764831845` (=1/√128); state `S[k*128+tid]` in **global** | none | grid 48; `kh=head/3` | edit (dims/layout stay) |
| `deltanet_prefill_kernel` (prefill.cu:219-269) | `<<<32,128,66,048>>>`; `kh=head>>1`; qkv stride `t*8192`; a/b `t*32`; out `(t*32+head)*128`; smem 64KB+512 | T, `snap` opt | grid 48; `head/3`; `t*10240`; `t*48`; `(t*48+head)*128` | edit (smem fits unchanged: 66.6KB < 99KB cap) |
| `conv_prefill`/`roll` (prefill.cu:170-200) | 8192 everywhere | T | 10240 | edit |
| `A_log` read (decode.cu:128, prefill.cu:82) | cast to `const float*` | — | 27B ckpt is **BF16** → `const void*`+dispatch (deltanet-27b §0; BF16-as-F32 gives α≈1, no forgetting, silent) | **TRAP** |

Pattern: every kernel bakes 9B shapes as literals (per project constitution); the only
runtime dimensions anywhere in the set are `tokens` (device-side, graph-safe), conv's
`n`, and prefill `T`. `max_context` is accepted and deliberately unused by both gqa
launchers — capacity is enforced by `score[4096]` and the DecodeWorkspace ctx check.

---

## 2. gqa_decode at 27B shapes (24 q / 4 kv / 256 dim)

### 2.1 What the kernel actually does per block

One block per q-head, 256 threads, 8 warps. Two phases:

- **Phase 1 (Q·K)**: thread-per-token. `for t=tid; t<tokens; t+=256` then a serial
  `d=0..255` FMA loop. A warp instruction at a given `d` has its 32 lanes on 32
  *consecutive tokens* → key rows `(t*4+kvh)*256`, i.e. 32 addresses at 4 KB stride →
  **32 distinct 128 B lines / 32 sectors per warp load, 4 B useful per sector = 12.5%
  request efficiency**. L1 absorbs the reuse (each line is re-hit by the next 31 `d`
  steps), so DRAM bytes stay ~unique, but the LSU pays the wavefront bill every
  instruction regardless of hit/miss.
- **Phase 2 (V)**: `for t=0..tokens-1: z=fmaf(score[t]*sden, vc[(t*4+kvh)*256+tid], z)` —
  all 256 threads walk the *same* row, `tid` = value dim → fully coalesced 1 KB/row,
  but a serial `tokens`-long FMA chain per thread (latency hidden by 8 warps; ~2048
  iterations ≈ 10 µs at ctx 2048 — not the problem).

### 2.2 Traffic and occupancy

- Unique KV bytes/layer at ctx C: 4 kvh × C × 2 (K+V) × 1 KB = **8 KB/token** →
  4.2 / 16.8 / 33.6 MB at ctx 512 / 2048 / 4096 (f32).
- **L2 reuse**: each kv row is needed by its 6 group-sibling blocks (24/4=6); all run in
  the same 24-block wave; unique set ≤ 33.6 MB vs 48 MB L2 → DRAM sees ~unique bytes.
  L2 dedup works; the problem is elsewhere.
- **The two real ceilings of the monolithic 24-block grid**:
  1. **LSU wavefronts** (phase 1): sector-requests per block = 8 warp-instructions per
     token × 32 sectors = **256·C**. At 4 sectors/cycle/SM → 64·C cycles ≈ **77 µs at
     C=2048** (1.7 GHz). If the L1 instead charges ~1 line/cycle the same count is 4×
     worse: up to ~300 µs. Honest range: **77-308 µs/layer at ctx 2048**.
  2. **DRAM BW cap**: 24 blocks on 56 SMs cannot saturate the machine; ~24/56 × 504
     ≈ 216 GB/s → even perfectly coalesced, 16.8 MB takes ≥ 78 µs. Phase 1 sits at
     *both* ceilings simultaneously.

### 2.3 Per-layer estimates vs the CPU tier

CPU tier measured **0.414-0.657 ms/full-attn layer** at ctx 2048, 6 splits
(audits/w4/cpu-tier.md §1.4/R6). Estimates per layer (f32 KV):

| ctx | monolithic (this design) | BW floor @504 | split-K est (§4) | CPU tier |
|---|---|---|---|---|
| 512 | ~20-80 µs | 8.4 µs | not worth it (launch+combine ≈ savings) | — |
| 2048 | **~87-320 µs** | 33.4 µs (≥78 µs @24 blocks) | **~37-42 µs** | 414-657 µs |
| 4096 | ~170-640 µs | 66.7 µs (≥156 µs @24 blocks) | ~70-75 µs | — |

So the ported monolithic kernel is "only" ~2-8× faster than the CPU tier per layer —
same order of magnitude, not the 10-20× the bytes imply. The 16-layer per-token cost
at ctx 2048 is **1.4-5.1 ms** (mid ~2.4 ms) — it would be the single largest line item
in the whole decode step. This is the strongest quantitative argument for split-K being
schedule-critical rather than optional polish.

### 2.4 qk_norm_rope: grid of 28, and the RoPE convention

- **Grid 28 = 24 q-head blocks + 4 k-head blocks**: one block per head because 256
  threads map 1:1 onto the 256 head dims (RMS over 256 + weight + rope on first 64).
  `isq=head<24`, k blocks use `k+(head-24)*256`. Today: `<<<20,256>>>`, `head<16`
  (ops.cu:10). 28 blocks < 56 SMs, single wave, ~3-5 µs launch-bound (insig4-perf
  launch census: tiny kernels are 3-5 µs latency-bound). ×16 layers ≈ 50-80 µs/token.
- **Pairing convention — exact indexing** (ops.cu:9):
  - participation guard: `if(pos!=0 && tid<64)` — dims 64..255 untouched, pos 0 skipped
    (identity anyway: c=1, s=0).
  - pair fetch: `other = mem[tid<32 ? tid+32 : tid-32]` → pairs **(i, i+32)** for
    i∈[0,32) — *halves* convention on the 64-dim roped subspace (not GPT-J interleaved).
  - sign: `v = v*c + (tid<32 ? -other : other)*s` → first half `a·c − b·s`, second half
    `b·c + a·s`.
  - frequency: `inv = __powf(1e7, -2·half/64)`, `half=tid&31` → θ^(−i/32), i∈[0,32).
  - **vs tools/reference27.py:265-276** (`rope64`): `ROPE_INV = 1e7 ** (-arange(32)/32)`,
    `a = hh[:, :32]`, `b = hh[:, 32:64]`, `hh[:,:32] = a*c − b*s`,
    `hh[:,32:64] = b*c + a*s`, "pairs (i, i+32), rotate_half convention; dims 64..255
    untouched". **Exact match** — structure, signs, frequencies, participation set.
  - residual precision deltas (documented, not a bug): reference computes angles in f64
    then cos/sin→f32; kernel uses f32 `__powf` angle + `__cosf/__sinf` fast intrinsics.
    At pos ≤ a few hundred this is ~1e-7; at pos ~4096 the fast-intrinsic argument
    reduction degrades (angle up to 4096 rad) — worth adding to the open full-attn
    parity suspect list if the hunt ever reaches long contexts.
- The `nsc` smem race fix (norm scale in its own slot, not `mem[0]`) is present in both
  ops.cu:9 and prefill.cu (line 62 comment) — preserved.

---

## 3. DeltaNet decode at 27B (48 heads, state 48×128×128 f32)

### 3.1 What the measured 0.010 ms/layer says

9B measurement: 32 heads, state traffic = 32×64 KB read + 32×64 KB write = 4.19 MB in
10.0 µs → **~420 GB/s effective = 83% of the 504 GB/s peak**. The kernel is already
bandwidth-bound, not latency- or sync-bound.

27B extrapolation: 48 heads → 3.146 MB read + 3.146 MB write = 6.29 MB/layer →
floor 12.5 µs, measured-rate extrapolation **~15 µs/layer** → **0.60-0.72 ms per token
over 48 layers**. The mission's 0.6 ms figure is the floor; 0.72 ms is what the measured
efficiency predicts. Answer: yes, this design achieves (83-100% of) it — no redesign
needed for traffic, only the port (`<<<48,128>>>`, `kh=head/3`).

Why it works despite looking thin (48 blocks = 1/SM, 4 warps each):
- **No cross-token serialization in decode** — one token per launch; the "128-step
  recurrence" is two serial 128-iteration loops *per thread within one token*
  (dot pass then fused update+output pass). FMA-chain latency ≈ 256 iter × ~4-8 cyc
  ≈ 1-2K cycles ≈ ~1 µs — hidden under the 12.5 µs memory time.
- **`__syncthreads` count = 3** per launch (deltanet.cu lines 7, 8, 11: warp-partials,
  scale publish, delta publish). Negligible.
- **No smem state staging in decode** (544 B static: `sq/sk/delta`). State is accessed
  directly in global; `S[i*128+tid]` is coalesced (128 threads read 512 B per i).
  The dot pass re-reads what the update pass reads — L2 (48 MB ≫ 3.15 MB/layer state)
  absorbs the second read, so DRAM sees exactly read+write = 6.3 MB.
- Prefill scan is a different animal (state staged in 66 KB dynamic smem, 4 syncs × T
  sequential steps — unavoidable recurrence; ~57-70 µs/layer per 64-token chunk from
  smem traffic ≈ 1 µs/token — irrelevant to decode).

### 3.2 Could state be bf16?

Traffic: 6.29 → 3.15 MB/layer → decode 15 → ~7.5 µs/layer → **0.36 ms/token over 48
layers (−0.36 ms)**, plus snapshot/rollback copies halve too (151 → 75 MB), plus 151 MB
VRAM saved (live) + 151 MB (snap).

Precision argument — why this is riskier than bf16 KV: the KV cache is write-once
(rounding error enters once per entry, ~2^-9 relative, then decays through softmax).
The DeltaNet **state is rewritten every token** (`S ← α·S + k̂δ^T`), so bf16 rounding
re-injects ~2^-9 relative error every step into the *memory itself*, and the error
propagates through the recurrence. Bounding factor: decay α < 1 gives exponential
forgetting with horizon ~1/(1−α) tokens; for the typical Qwen-Next A/dt regime
(α ≈ 0.9-0.99) the accumulated drift is a bounded sum ≈ horizon·2^-9 — plausibly fine
(cosine ≥ 0.999), but for α → 0.999 (long-memory heads) it approaches a random walk
√T·2^-9 over a 4 K context — plausibly not fine. This must be measured, not argued.

**Parity experiment proposal** (cheap, self-contained, no engine change):
1. Extend `test_deltanet.cu` harness: T = 4096 sequential steps, two kernel variants
   (state stores rounded to bf16 RNE + widened on load, vs today's f32), CPU f32 shadow
   reference as today. Sweep `decay = exp(g) ∈ {0.9, 0.99, 0.999}`.
2. Metrics: per-token output cosine (min over run), final-state max-rel, and greedy
   argmax agreement on a 200-token real continuation via the `dump_multistep` harness.
3. Gate: min per-token cosine ≥ 0.999 AND ≥ 99% token agreement → adopt; else keep f32
   (0.72 ms is already 83% of peak; bf16 state is a −0.36 ms play, not a must-have).
4. If borderline: bf16 state only for the *read* path is incoherent (round-trip must
   round); the honest fallback ladder is f32 state + everything else in §5 first.

### 3.3 conv4 at 10240 and the a/b GEMVs

- **conv4** (qwen_kernels.cu:7-8): n-generic grid-stride; at n=10240 it is 40×256
  threads moving ~0.36 MB/layer (x 40 KB, state 120 KB r+w, weights 80 KB bf16) —
  ~1.4 µs of bytes but **3-5 µs launch-bound**, ×48 layers ≈ 0.15-0.24 ms/token of
  almost pure launch overhead. The history loads `state[i*3+k]` are 384 B/warp
  contiguous — fine. Only the decode.cu:128 literals (`8192`, `8192*3`) change.
- **a/b [48,5120] GEMVs** (decode.cu:127: two separate `linear()` calls): GAP — the
  27B native checkpoint stores a/b in **BF16** [48,5120] (983 KB both), but the decode
  path only has (a) `mxfp4_gemv_v2` via `linear()` for MXFP4/INSIG4 tensors and (b)
  single-tensor `bf16_gemv` (qwen_kernels.cu:67) used only for mtp.fc. There is **no
  bf16 pair GEMV** for decode (the prefill T==2 path has `mxfp4_gemv_ab2_q8`; prefill
  T<64 has `mxfp4_gemm_ab_i4` — both MXFP4-only). Needed: `bf16_gemv_ab_pair` reading
  x once and streaming both 48-row weights (rows are tiny — one 48-block launch, ~2 µs,
  saves one launch + one x pass per layer ≈ 2-4 µs × 48 ≈ 0.1-0.2 ms).
- **params TRAP restated** (§1): `params<<<1,32>>>` + decode.cu passing n=32 — at 27B
  n must be 48 *and* the grid must grow to `<<<1,48>>>`, otherwise heads 32..47 keep
  stale a/b garbage (silent, same failure class as the A_log dtype trap).

---

## 4. Split-K GQA decode — spec at 24 heads × ctx 4096 × 256 dim

w3 attn-27b §7 already carries paste-ready code; this audit adds the quantified *why*
and corrects the crossover claim.

**Design (flash-decode, two passes, deterministic):**
- Pass 1: grid `dim3(24, S)`, S=4 (ctx ≤ 4096 keeps chunk ≤ 1024 → `score[1024]`
  4 KB smem). Block (head, s) owns tokens `[s·chunk, (s+1)·chunk)`. Scoring uses the
  **proven gqa_prefill warp-per-key-row pattern**: 8 warps, each warp owns whole key
  rows, lane covers 8 consecutive dims → each lane reads exactly one aligned 32 B
  sector; a warp load moves 1 KB in 8 lines with **100% sector efficiency** (vs 12.5%
  monolithic = 8× fewer wavefronts per byte). Partials per split: `(m_s, l_s, acc[256])`
  → `scratch[s][head][258]`.
- Pass 2: 24 blocks × 256 threads flash-merge S=4 partials (fixed s-order reduction —
  no atomics, bitwise deterministic, parity-friendly; empty split → m=−inf, weight 0,
  l forced ≥ 1 so no 1/0).
- smem pass 1: 1 KB (qs) + 4 KB (score) + 32 B (red) + 8 KB (part) ≈ 13.1 KB →
  2 blocks/SM. 96 blocks = 1.7 waves on 56 SMs — enough concurrent sectors to actually
  pull ~504 GB/s (the monolithic grid's 24 blocks cap at ~216).
- **VRAM scratch**: S×24×258×4 B = **98.7 KB** for S=4 (197 KB at S=8 if ctx ever
  exceeds 4096). One `cudaMalloc` in DecodeWorkspace, stable pointer → graph-safe;
  S chosen at capture time (branch frozen in the graph).
- bf16-KV composes independently: halves the 8 KB/token unique bytes; does NOT fix the
  scattered access pattern (still 32 sectors/warp at 2 B lanes) — split-K first, bf16
  second.

**When it beats monolithic** (this audit's model): monolithic time ≈ max(64·C cycles
LSU, 8C KB / 216 GB/s BW-cap) ≈ 37-38 ns/token either way; split-K ≈ 8C KB / 504 GB/s
+ 5-8 µs (combine launch + work) ≈ 16 ns/token + fixed. Crossover: C ≈ 500-800 —
i.e. the plan's `max_context > 1024` threshold is conservative-safe; anything ≥ 1024
is a guaranteed win (≥ 2×), ctx 2048 is ~2.5-8×, ctx 4096 ~2.5-9×. The MTP draft
layer's `gqa_decode` (mtp_keys) inherits the same path for free.

**Expected effect at ctx 2048 (f32 KV): from ~87-320 µs/layer to ~37-42 µs/layer;
with bf16 KV to ~21 µs/layer.** Numbers to be confirmed by the §6 bench, not assumed.

---

## 5. Total per-token estimate, attention + DeltaNet, 27B ctx 2048

Components per token (16 full-attn + 48 delta layers):

| line item | ported-as-is | split-K | +bf16 KV | +bf16 state | +aux fusion |
|---|---|---|---|---|---|
| gqa_decode ×16 | 1.4-5.1 ms (mid 2.4) | 0.61-0.67 ms | 0.34 ms | — | — |
| attn aux (qk_norm_rope, store_kv, split/expand/sigmoid) ×16 | 0.18 ms | 0.18 ms | 0.18 ms | — | ~0.08 ms |
| deltanet_decode ×48 | 0.60-0.72 ms | — | — | 0.30-0.36 ms | — |
| delta aux (conv, params, gated norm, a/b ×2) ×48 | ~0.48 ms | — | — | — | ~0.20 ms |
| **TOTAL** | **2.7-6.5 ms (mid ~3.8)** | ~2.0 ms | ~1.75 ms | ~1.4 ms | **~1.2 ms** |

Context: the whole 27B step is weight-traffic bound (~4.99 GB/step ≈ 11-13 ms at 75%
peak, insig4-perf); attention+delta bytes (0.30 GB state r/w + 17 MB KV) ride on top.
Even the "ported-as-is" column only adds ~0.5 GB to a 5 GB stream — the ms matter
because they are *added serial latency*, not because they break the byte budget.

**Ranking by ms saved (from mid ~3.8 ms):**

1. **Split-K GQA (ctx ≥ 1024)** — **−1.9 ms** (range −1.0 to −4.7). Biggest, lowest
   risk: pattern is proven in gqa_prefill, deterministic combine, no numerics change
   vs two-pass softmax modulo fp association (parity-checkable against reference).
2. **bf16 DeltaNet state** — **−0.36 ms** (+302 MB VRAM). Gated on the §3.2 experiment.
3. **Aux-launch fusion** — **−0.35-0.40 ms**: fuse conv4+params+gated-norm into one
   launch per delta layer (all tiny, sequential dependencies are per-channel); delete
   `expand_gate_heads` (a pure 24 KB copy — make `sigmoid_mul` read `attn_gate`
   directly); a/b bf16 pair GEMV. Pure launch-count engineering, zero numerics risk.
4. **bf16 KV** — **−0.26 ms** (Phase G per master plan, after parity lock; also
   −268 MB VRAM at ctx 4096).
5. **qk_norm_rope folding** into the q/k GEMV epilogues — −0.05-0.10 ms, only after
   the above (small win, touches parity-sensitive code first).

---

## 6. Verification / bench checklist (no code written here)

1. Port gates: `kvh=head/6` poison-test (KV cache keyed per (t,kvh), assert 24 blocks
   → kvh {0×6,1×6,2×6,3×6}); params `<<<1,48>>>` + n=48; A_log BF16 dispatch.
2. gqa ctx sweep {512,1024,2048,4096} × {monolithic, split-K, split-K+bf16 KV} —
   decides §4 crossover and the bf16-KV flip on numbers; the 77-308 µs monolithic
   range above is a model, not a measurement.
3. DeltaNet 27B-shape timing (48 heads): expect ~15 µs/layer; if > 20 µs, check L2
   thrash between dot and update passes (state slices are per-block disjoint — should
   not happen).
4. bf16-state drift experiment (§3.2) before adoption; bf16-KV parity gate per
   attn-27b §6 before Phase G.
5. Re-capture both CUDA graphs after any grid/buffer edit (frozen launch configs).

## TL;DR

- All audited kernels hardcode 9B shapes; the 27B port is launcher-literal edits plus
  three real traps still in the tree: `kvh=head>>2` (attention.cu:7), `params<<<1,32>>>`
  silently skipping heads 32-47 at n=48, and A_log BF16 read as f32 (decode.cu:128).
- RoPE convention verified bit-for-bit equivalent to reference27.py rope64: pairs
  (i,i+32), halves/rotate-half, θ^(-i/32), dims 64..255 untouched, pos-0 identity skip.
- Measured anchor: deltanet decode 0.010 ms/layer @9B = 83% of DRAM peak → 27B
  extrapolates to 0.60-0.72 ms/token over 48 layers; the design is BW-bound and healthy
  (3 syncs, no cross-token serialization, no smem staging needed — L2 covers the
  state re-read).
- gqa_decode at 27B/ctx2048 is the outlier: 12.5%-efficient scattered K loads + a
  24-block BW cap put it at ~87-320 µs/layer (1.4-5.1 ms/token) — only 2-8× the CPU
  tier. Split-K (96 blocks, warp-per-row, 98.7 KB scratch, deterministic combine)
  model-predicts ~37-42 µs/layer (0.6 ms/token), crossover ≈ ctx 700, plan threshold
  1024 conservative-safe.
- Total attention+delta at ctx 2048: ~3.8 ms as-is → ~1.2 ms fully optimized.
  Ranked: split-K (−1.9) > bf16 state (−0.36, gated on drift experiment) ≈ launch
  fusion incl. the missing bf16 a/b pair GEMV (−0.35) > bf16 KV (−0.26, Phase G).
