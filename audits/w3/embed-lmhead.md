# w3: embed + lm_head — the two BF16 giants of Qwen3.8-27B-FP8 — placement & kernels (2026-08-25)

Scope: un-requantized placement + kernels for `lm_head.weight` and
`language_model.model.embed_tokens.weight`, both BF16 **[248320, 5120] = 2,542,796,800 B
(2.543 GB / 2.368 GiB) each** (w2/loader-27b-spec.md:207-208, `tie_word_embeddings=false` —
no aliasing). Read-only audit; no builds. Existing kernels audited firsthand:
`bf16_gemv` (src/qwen_kernels.cu:67-68), `argmax_fast` (:25-63), `embed_gather*`
(src/prefill.cu:9-40), GEMM family (src/gemm.cu), call sites (src/decode.cu:46,93-101,137-151,
186-187; src/nll.cu:78-82; src/generate.cu:76-82).

Hardware (verified, TechPowerUp / audits): 4070 SUPER = AD104, **56 SMs, 7168 cores,
boost 2.475 GHz, FP32 = 35.48 TFLOPS, GDDR6X 192-bit = 504.2 GB/s, L2 48 MB**,
PCIe 4.0 x16 (~21-23 GB/s real), 12,282 MiB VRAM, 15.9 GiB host RAM.

---

## 0. Verdict table

| question | verdict |
|---|---|
| lm_head placement | **VRAM resident** (2.368 GiB = 6.62 layer-slots). Per-sweep VRAM 5.4-5.9 ms vs PCIe 110-121 ms (20-22x) vs CPU 64-69 ms. Strongest counterfactual (RAM-pinned + slot refilled from NVMe) loses on the 15.9 GiB RAM budget — §2. |
| T=1 GEMV math | **The "12x/6x FMA shortage" is a units error.** 1 fp32 FMA per weight needs 252 G FMA/s = **1.4% of the 17.7 T FMA/s fp32 pipe**; total warp-instruction issue ~3%. DRAM-bound, no tensor cores needed — §3. |
| bf16 GEMV kernel | warp-per-row, lane owns 32-elt groups, 4x uint4 `__ldcs`, `__bfloat1622float2` + **fp32 FMA (exact products, fp32 accum — fp16/bf16 accum rejected, §3.3)**. Honest ceiling **430-480 GB/s → 5.3-5.9 ms/sweep** — §4. |
| T=2/4 spec verify lm_head | **always the wmma bf16 GEMM** (one 2.543 GB weight pass regardless of T; compute 0.25 ms at T=4 ≪ IO) — §5. |
| MTP draft lm_head @27B | full sweep, 5.4-5.9 ms of ~1.5 s step = 0.4% — negligible, keep exact. Slice-argmax is the **9B** lever (~10% of its 13.1 ms step), aligns with w3/spec-deepen.md — §6. |
| embed placement | **pinned host RAM + UVA zero-copy gather** (10.24 KB/row, ~3-5 µs/step). Never VRAM (6.62 slots for 30 KB/step). VRAM row cache rejected. Fallback: per-step host-staged rows (graph break is free at 1.5 s/step) — §7. |
| argmax | current 2-stage atomicMax kernel is ~4-6 µs and correct; keep. Merge the two T=2 calls into one launch; epilogue fusion not worth it — §8. |
| NLL lm_head (T=64) | bf16 wmma GEMM = `mxfp4_gemm_v21` minus the entire dequant/LUT/Braw path (B is cp.async'd raw). IO-bound 5.4-5.9 ms per 64-token chunk — §9. |

---

## 1. Shapes and traffic (verified)

- lm_head bf16 [248320, 5120]: one GEMV/GEMM sweep streams **2.543 GB of weights**.
  Activations in: T×10-20 KB; logits out: T×993.3 KB (fp32) — noise.
- Per spec step (MTP draft T=1 + pair verify T=2): **2 sweeps = 5.086 GB** through the same
  matrix, ~5.1 GB apart in the access stream → zero L2 reuse between them (48 MB L2).
- embed bf16 [248320, 5120]: decode reads **1 row = 10,240 B per token**; a spec step reads
  3 rows (MTP pending + verify pair [pending, draft]) = 30.7 KB; prefill reads T rows/chunk.
- 27B layer slot = 383.87 MB (synthesis) → each giant = **6.624 layer-equivalents**.

---

## 2. lm_head placement — the honest arithmetic

### 2.1 Per-sweep table (one 2.543 GB pass)

| tier | rate | time/sweep |
|---|---|---|
| VRAM GEMV | 430-504 GB/s (peak 504.2) | **5.0-5.9 ms** (design point ~5.5) |
| RAM pinned, UVA/PCIe read | 21-23 GB/s | 110-121 ms (mission's 116 ms ✓) |
| CPU GEMV (5600X, ~37-40 GB/s DRAM) | 37-40 GB/s | 64-69 ms (but adds a host round-trip + sync per sweep) |
| NVMe (6.8 GB/s, no UVA anyway) | 6.8 GB/s | 374 ms |

VRAM wins per sweep by **20-22x over PCIe** (mission's 20x ✓). CPU GEMV beats PCIe but
forces a per-sweep device→host→device dance for x/logits (10 KB + 2×993 KB over the same
PCIe) and serializes the GPU; dead on arrival.

### 2.2 The residency LP (what the 2.368 GiB VRAM slot *costs*)

The simple mission argument — "6.6 layers pushed to NVMe costs 6.6×56.5-59.5 ≈ 374-393 ms
≫ 5.4 ms" — compares against the *weakest* alternative (lm_head streamed from NVMe).
The strongest alternative must also be priced: **lm_head pinned in RAM, streamed over PCIe,
and its VRAM slot refilled with 6.62 layers pulled NVMe→VRAM** (tier costs from synthesis:
V 0.76 / R 15.4 / N 56.5 ms/layer; 2 sweeps/step):

| scenario | VRAM | RAM (pinned) | NVMe | step ms (weights) |
|---|---|---|---|---|
| **A: lm_head VRAM (recommended)** | lm_head + 21 L + mtp | 23 L + embed 2.54 GB = **11.37 GB** | 20 L | 21×0.76+354+1130+2×5.5+~3 ≈ **1514** |
| B: lm_head RAM + slot refilled | 27.62 L + mtp | 23 L + embed + lm_head = **13.92 GB** | 13.38 L | 21+354+756+2×116+~3 ≈ **1366** (−10%) |
| B′: B with 3 RAM layers demoted (RAM fit) | 27.62 L + mtp | 20 L + 5.09 GB = 12.79 GB | 16.38 L | 21+308+925+232+3 ≈ **1490** (−1.6%) |

So on *pure bandwidth arithmetic* B is ~150 ms/token better — **but it needs 13.92 GB of
pinned host RAM on a 15.9 GiB machine** (17.1 GB physical − Windows + apps 2.7-3.5 GB −
CUDA host context − pinned IO ring ≈ 13.2-14 GB usable): it does not fit; B′ fits and is
inside noise, while stacking three liabilities: (1) 13.9 GB of PCIe traffic per step sharing
one x16 link with the 23-layer RAM stream, (2) zero headroom for page-cache/working-set
spikes (hard-thrash cliff), (3) the draft sweep at 116 ms makes any future draft-slice /
T=4 deepening brutally expensive (B pays per verify row-count ×0, per *sweep* ×1 — sweeps
are what PCIe costs). **Verdict: lm_head VRAM-resident (A)**, matching synthesis. If the
host ever grows to 32-64 GB, re-run this table — B-flips become real.

VRAM fit check (A): lm_head 2.37 + 21 layers 7.51 + mtp 0.44 + KV (16 full-attn layers,
bf16 KV, 4 ctx-K) 0.28 + delta state fp32 ~0.16 + workspaces/logits ~0.3 (NLL adds 63.6 MB)
≈ **11.1 GiB of 12.28** ✓ (synthesis L=21 stands; ~1 GiB slack).

---

## 3. The FMA-throughput "crux" — resolved: it was never a problem

The mission's panic chain ("252 G FMA/s needed vs 41 G fp32 FMA/s → 12x/6x over →
IMPOSSIBLE") contains two accounting errors. Both fixed:

1. **Units.** 4070 SUPER FP32 = 35.48 TFLOPS (7168 lanes × 2 FLOP × 2.475 GHz) =
   **17.74e12 FMA/s = 17,740 G FMA/s** — "TFLOPS" is tera, not giga; the "41.3 G FMA/s"
   figure (and the 82.6 TF it came from) belongs to a 4090-class part and dropped three
   orders of magnitude in unit conversion. Needed: 504 GB/s ÷ 2 B = **252 G weights/s =
   252 G FMA/s = 1.4% of the fp32 pipe** (and 0.25% of the 71 TF bf16 tensor pipe).
2. **Per-thread vs per-warp instruction accounting.** "8 FMA per 8 weights" is *per lane*;
   one issued warp-instruction performs 32 lanes' work. Per 256 chip-weights (one warp
   executing the uint4 body: 1×LDG.128 + 4×cvt-pair + 8×FFMA + ~1 loop/addr ≈ 14
   warp-instructions): at 252 G w/s that is 252/256×14 ≈ **13.8 G warp-instr/s vs 554 G
   issue capacity (4 schedulers × 56 SM × 2.475 GHz) = 2.5%**. There is no instruction
   wall at any pipe: FMA 1.4%, cvt ~3% (even at quarter-rate), LSU ~2%.

**Why llama.cpp-style kernels do hit ~DRAM rate:** they were never compute-limited —
measured Ada evidence: 7B-F16 token generation on 4090 ≈ 70-85 t/s ⇒ ~0.94-1.0 TB/s
effective on 13.4 GB of F16 weights (~85-93% of 1008 GB/s); this repo's own MXFP4 lm_head
GEMV (a *heavier* kernel: LUT-LDS + scale per group, src/mxfp4.cu:90-144) already measures
**379-435 GiB/s on the 248320-row shape on this exact GPU** (audits/internals.md:18,28).
A bf16 GEMV does strictly less math per byte. Tensor cores for T=1 are unnecessary (they
are the right tool for T≥2 — §5).

### 3.1 Precision: fp32 FMA is mandatory and free

- bf16×bf16 products are **exact** in fp32 (8-bit × 8-bit mantissas fit); only the
  accumulation rounds. fp32 accumulation over 5120 terms: rel err ~2^-24×√5120 ≈ 3.6e-6 —
  logit-noise floor.
- fp16 accumulation (11-bit mantissa): RMS ≈ √5120 × 2^-12 × cancellation(3-10x) ≈
  **1-5%** of the dot — top-5 logit gaps are routinely <0.05, so argmax among near-ties
  flips. The mission's "~1e-2 rel err" is the *optimistic* end; bf16 accumulation (2^-8)
  is 8x worse. Both rejected. **fp32 accumulate, period** — and since compute is 1.4% of
  the pipe (§3), exactness costs zero throughput.
- `__bfloat1622float2` (2 SASS cvt per pair) can be swapped for the AGENTS.md-flavored
  bit trick — bf16→fp32 is exactly `bits << 16`: low half = `pair<<16` (SHF), high half =
  `pair & 0xffff0000` (LOP3). Same instruction count, ALU pipe instead of cvt pipe; both
  are 3%-of-capacity non-issues. Pick whichever the compiler schedules better (check SASS
  once; don't think about it again).

---

## 4. bf16 GEMV kernel design — `bf16_gemv` v2

### 4.1 Audit of the existing kernel (src/qwen_kernels.cu:67-68)

One 256-thread block per row (248320 blocks), scalar `bf()` cvt per element, scalar
`uint16` loads (64 B per warp transaction), block-strided loop, x re-read from L2 by every
block (248320 × 20 KB = 4.8 GB of L2 x-traffic). ~3+ SASS per *lane*-weight with 2-byte
global loads → est. 120-180 GB/s. Today it only serves 9B mtp.fc (67 MB) so it never
mattered; it is nowhere near good enough for a 2.543 GB lm_head sweep (~14-21 ms).

### 4.2 Recommended design (proven `mxfp4_gemv_v2` skeleton, decode path swapped)

Structure identical to src/mxfp4.cu:90-144 — 8 warps/block, **warp-per-row**, lane owns
whole 32-weight groups (4× uint4 = 64 B per lane per group, warp footprint 2 KB streamed
over 5 iterations for cols=5120), x staged **transposed** into smem once per block so
lane-per-group x reads are bank-conflict-free, `__ldcs` (evict-first) on weights, fp32
warp-shuffle reduce, one store per row. Only the per-group math changes:

```cuda
// per 32-weight group (4 uint4), replacing V2_WORD + scale apply:
const uint4 p0 = __ldcs(wr + g*4 + 0), p1 = __ldcs(wr + g*4 + 1),
             p2 = __ldcs(wr + g*4 + 2), p3 = __ldcs(wr + g*4 + 3);
const float *xg = sx + g;                      // xg[k*groups] == x[g*32+k], conflict-free
float s = 0.f;
#define BF16X2(u) { const float2 f = __bfloat1622float2( \
        *reinterpret_cast<const __nv_bfloat162*>(&(u))); \
        s = fmaf(f.x, xg[0*groups], s); s = fmaf(f.y, xg[1*groups], s); }
// interleave xg[k*groups] for k=0..15 per uint4 — 4 cvt-pairs + 8 FMA per 8 weights
```

Per warp-instruction body (256 chip-weights): 4 LDG.128 + 64 LDS.32 (x, conflict-free) +
16 cvt-pairs + 64... all warp-level: ≈ 0.22 warp-instr/weight → **~5% issue at DRAM rate**.
No dequant, no scales, no LUT — strictly cheaper than the MXFP4 kernel that already hits
379-435 GiB/s. Persistent grid-stride variant (per w3/insig4-perf.md §2.2: stage x once,
grid = 56×occupancy, rows via grid-stride) removes the 4.8 GB of x re-reads and the
31040-block launch tail — adopt it from day one here (x smem 20 KB for cols=5120 → 4-6
blocks/SM).

**Honest achievable bandwidth: 430-480 GB/s (85-95% of 504.2)** — anchored by (a) this
repo's 379-435 GiB/s on the same shape with a heavier kernel, (b) Ada F16-GEMV production
kernels at 85-93% (4090 7B-F16 tg numbers), (c) compute/issue ≤5% leaving only DRAM
efficiency as the limiter. **Sweep time 2.543 GB: 5.3-5.9 ms (design point 5.5 ms).**
The mission's 470-504 GB/s (5.1-5.4 ms) is the optimistic edge — 504 is theoretical peak;
never promise it. Bench protocol: cold-L2 (l2_sweep between reps) per insig4-perf §2.4 —
2.543 GB ≫ 48 MB L2 so reps are honest by construction, but the *bench shapes* must stay
≥2×L2.

### 4.3 The sneaky alternative: T=1 through the wmma GEMM (§5)

The bf16 wmma GEMM reads the same 2.543 GB regardless of T and its per-byte instruction
cost is ~0 (tensor cores idle anyway at T=1; A-tile zero-padding = 630 KB memset, 1.3 µs).
It may match or beat the GEMV (~460-490 GB/s) with one code path for all T. **Plan: build
the GEMM first, measure T=1 through it; add the specialized GEMV only if the GEMM T=1
loses.** The GEMV design above stands ready (it is also the right shape for mtp.fc
[5120,10240] = 105 MB and any small-row bf16 matrix where a 64-row-tile GEMM wastes grid).

---

## 5. Spec-verify lm_head (T=2/4): always the GEMM

Weights are read once per sweep whether T=1 or T=64; tensor-core compute is dead time on a
memory-bound pass: T=4 → 2×4×248320×5120 = 20.3 GFLOP ≈ 0.25-0.29 ms at 71-82 TF, inside a
5.5 ms IO stream. **Policy: T≥2 (pair verify, deep-verify T=3-5, NLL T=64) → one wmma GEMM
launch; T=1 (MTP draft, greedy single) → GEMV or GEMM, measured (§4.3).** This mirrors the
9B engine, whose T=2 verify already amortizes both rows in one weight pass via the pair
dp4a kernel `mxfp4_gemv2_q8` (src/decode.cu:94-98, one pass confirmed by w3/spec-deepen.md
F2/F3) — the 27B bf16 equivalent is strictly simpler (no int8 quantize-x staging).

Kernel: see §9 (same kernel, T is a parameter; A rows ≥T zero — existing memset-tail +
`f32_to_bf16` staging at decode.cu:36-39 reused verbatim).

---

## 6. MTP draft lm_head (argmax per spec step)

Sequence today (src/decode.cu:186-187): draft lm_head = `linear()` GEMV sweep + full-vocab
argmax; then pair verify does its own sweep. Per spec step: **2 sweeps = 10.9-11.8 ms at
27B**.

- **27B: keep the full sweep.** 2 sweeps ≈ 0.7-0.8% of the ~1.5 s step. A top-64K slice
  copy (655 MB) costs either 0.64 GB VRAM (0.27 GB... 655 MB = 1.7 layer-slots — no) or
  30 ms over PCIe (worse than the 5.5 ms full sweep it replaces) — slice is strictly
  losing at 27B scale. (Mission's "10.8 ms of 1.5 s ≈ negligible" ✓.)
- **9B: the sweep is the single biggest draft cost** — calibrated model (w3/spec-deepen.md
  §1.1): draft lm_head 1.3 ms of a 13.1 ms step (**10%**; total lm_head draft+verify =
  2.6/13.1 = **20%** — the mission's "42%" charges 2×1.9 ms against the 8.3 ms
  *per-token* number; per-step is 13.1 ms and each sweep is 1.2-1.4 ms at the measured
  379-435 GiB/s). **Recommend: draft-argmax-over-slice** — an offline-compacted top-64K
  lm_head copy (143 MB INSIG4 at 9B, rows reordered by corpus token frequency, built by
  the quantizer/indexer side, plus a 256 KB u32 global-id table). Draft GEMV+argmax over
  the slice ≈ 0.3-0.4 ms (−0.9 ms/step ≈ +7% tok/s at D=1; spec-deepen's D=2-3 tables:
  145-156 tok/s vs 135-138 full).
  Acceptance math: slicing only changes *proposals*; verify stays full-vocab and exact.
  Acceptance loss is bounded by P(target argmax ∉ top-64K) + edge effects ≈ 0.2-0.5%
  absolute on natural text (248K vocab is mostly multilingual/visual tail;
  `image_token_id`/`video_token_id` never appear in text — w2/loader-27b-spec.md:33).
  Gate: A/B accept-rate on the NLL corpus ≥ −1% relative before adopting; fall back to
  full sweep if violated.

---

## 7. Embed: pinned-RAM zero-copy gather

### 7.1 Placement

- **Never VRAM**: 2.368 GiB = 6.62 layer-slots to serve 30.7 KB/step of reads. Insane
  even by this project's standards.
- **Recommended: whole embed pinned in host RAM, UVA (`cudaHostAllocMapped`) zero-copy
  gather.** RAM cost 2.543 GB on top of the A-scenario budget (11.37 GB → fits, §2.2;
  one-time load from NVMe ≈ 0.4 s at load). Per-step cost: 3 rows × 10.24 KB over PCIe ≈
  1.4 µs of transfer + ~2-4 µs UVA latency, inside the captured graph (fixed pinned
  address ⇒ graph-safe, unlike any on-miss host IO).
- **Fallback if RAM gets tight** (e.g. B′-style experiments eat the slack): embed stays on
  the NVMe mmap; once per step the host stages the 3 needed rows (10 KB each) into a
  64 KB pinned ring. Requires breaking the graph per step — at 1.5 s/step a host sync is
  free (the every-4-steps batching in generate.cu exists for the 13 ms 9B step, not for
  this). Row for [pending] is known at commit time; row for [draft] only after the draft
  argmax — stage it in the same post-step window, before the next replay.
- **VRAM LRU row cache: rejected** — decode rows almost never repeat (an LRU over
  distinct recent tokens has near-zero hit rate on natural text), the only guaranteed
  re-read is `row(pending)` twice per step (§7.2), and a cache adds graph-hostile
  residency plumbing for ≤10 KB of savings.

### 7.2 Kernel (bf16 → fp32 activations, engine layout)

```cuda
// T rows gathered from pinned host embed (UVA); out fp32 [T,5120]. Graph-safe.
__global__ void embed_gather_bf16_kernel(const uint16_t *__restrict__ w,   // host pinned
                                         const int *__restrict__ tokens,
                                         float *__restrict__ out) {
    const int t = blockIdx.x;                       // 1 block per token (T ≤ 64)
    const size_t row = size_t(__ldg(tokens + t));   // device token id
    const uint4 *src = reinterpret_cast<const uint4 *>(w + row * 5120);
    float4 *dst = reinterpret_cast<float4 *>(out + size_t(t) * 5120);
    for (int i = threadIdx.x; i < 640; i += 256) {  // 5120 bf16 = 640 x 16B
        const uint4 p = src[i];                     // warp = 512B contiguous over PCIe
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&p.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&p.y));
        const float2 c = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&p.z));
        const float2 d = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&p.w));
        dst[2*i]     = make_float4(a.x, a.y, b.x, b.y);
        dst[2*i + 1] = make_float4(c.x, c.y, d.x, d.y);
    }
}
// launch: <<<T, 256>>> — T=1..3 decode (1-2 blocks is all it takes), 64 prefill.
```

Coalescing: each warp reads 512 B contiguous host memory → PCIe TLPs pack at link rate;
the row itself is 10.24 KB contiguous. Zero dequant (this tensor is honest bf16). Plugs
into decode.cu:46 (prefill pair) and :137-138 (MTP) behind a `dtype==BF16` branch.

**MTP reuse note:** `mtp_layer` embeds `token_dev[0]` (pending) into `x_.down`
(decode.cu:137) and the very next `prefill_chunk_device(T=2)` re-gathers the same row as
pf_x row 0 (decode.cu:46). That double-read is 10 KB — ignore. If you want it gone for
aesthetic reasons, device-copy `x_.down → pf_x[0..5120)` (20 KB DtoD, ~2 µs); not worth
a code path today.

---

## 8. Argmax over [T, 248320] — audit + verdict

Current `argmax_fast` (src/qwen_kernels.cu:25-63): stage 1 grid 64×512 (32768 threads,
~7.6 elements each, scalar `__ldg`), warp+block reduce, one monotonic-u64
`atomicMax(best, (orderbits<<32)|idx)` per block (64 atomics), stage 2 unpacks; preceded by
`cudaMemsetAsync(scratch,0,8)`. Traffic 993 KB/call; logits were just written by lm_head →
L2-resident → ~1-2 µs read + 3 launches ≈ **4-6 µs per call**; 3 calls per spec step
(draft + 2 verify rows) ≈ 15 µs = 0.11% of the 13.1 ms 9B step, 0.001% of the 27B step.

Verdict: **keep it**. Cheap upgrades if touched anyway:
1. **Merge the two T=2 calls** (decode.cu:97-98) into one launch: grid 128, row =
   blockIdx.x>>6, two scratch slots / two atomics — saves a launch + memset.
2. `float4` loads + `__ldcs` in stage 1 (logits are dead after argmax).
3. **Epilogue fusion into the lm_head GEMM/GEMV**: each output block already holds a
   64-row (GEMM) or 8-row (GEMV) slice of logits in registers; block-reduce then one
   atomic per (block,row) → 3880 atomics — feasible, saves ~3-5 µs and the logits
   round-trip, but that is 0.03-0.04% of the 9B step: **not worth the coupling now**;
   fold in only if the launch-count crusade (insig4-perf §4) reaches it.
4. Micro-nit found while auditing: `argmax_fast` breaks exact ties toward the **higher**
   index (u64 key packs index in low bits; atomicMax prefers larger), while the reference
   `argmax_logits` (:20) keeps the **lowest** index (`v>best` strictly). Both are valid
   argmaxes and fp32 logits from different kernels rarely tie exactly, but it is a
   determinism asymmetry between the two paths worth one comment in the code.

---

## 9. NLL-mode lm_head GEMM (T=64) — `bf16_gemm` spec

`Y[T,248320] = X[T,5120] · W^T`, Y fp32. **This is `mxfp4_gemm_v21` (src/gemm.cu:210-295)
with the entire dequant machinery deleted** — no `Braw`, no `lut`, no `dequant` stage, no
scales: B tiles are raw bf16 cp.async'd straight into `Bs`. Requirements check:
rows=248320 (%32=0, %64=0 ✓ → grid 3880 at NT=64), cols=5120 (%64=0 → 80 K-steps ✓),
T≤64 ✓ (A rows ≥T zero via existing memset-tail + `f32_to_bf16`, decode.cu:36-39).

```cuda
__global__ __launch_bounds__(256) void bf16_gemm_kernel(
        const __nv_bfloat16 *__restrict__ x16,        // [64,5120] zero-padded A (bf16)
        const __nv_bfloat16 *__restrict__ w,          // [248320,5120] row-major
        float *__restrict__ y, int rows, int cols, int T) {
    constexpr int KT = 64, NT = 64, APAD = 8, BPAD = 8;
    __shared__ __nv_bfloat16 As[2][64][KT+APAD];      // 18.4 KB x2... (see budget)
    __shared__ __nv_bfloat16 Bs[2][NT][KT+BPAD];
    const int n0 = blockIdx.x * NT, tid = threadIdx.x;
    auto prefetch = [&](int kb, int buf) {           // A: 64 rows x 128B; B: 64 rows x 128B
        const int k = kb * KT;
        for (int i = tid; i < 64*(KT/8); i += 256) {  // cp_async16 into As[buf]
            const int m = i/(KT/8), c8 = i%(KT/8);
            cp_async16(&As[buf][m][c8*8], reinterpret_cast<const char*>(x16) + (size_t(m)*cols+k)*2 + c8*16);
        }
        for (int i = tid; i < NT*(KT/8); i += 256) {  // cp_async16 into Bs[buf] — raw, no dequant
            const int n = i/(KT/8), c8 = i%(KT/8);
            cp_async16(&Bs[buf][n][c8*8], reinterpret_cast<const char*>(w) + (size_t(n0+n)*cols+k)*2 + c8*16);
        }
    };
    // 2-stage cp.async pipeline; 8 warps as 4x2 grid of 16x16 wmma tiles (v21 layout:
    // wm=warp>>1, wn=warp&1; 4 K-halves per step) — verbatim v21 minus dequant/scales;
    // fp32 accumulators, store with ldm=rows (row-major [T,rows], same as mxfp4 GEMMs
    // so row_logp_kernel reads logits + row*vocab unchanged).
}
```

smem: 2×(64×72 + 64×72)×2 B = 36.9 KB → 2 blocks/SM (16 warps) — ample for latency hiding
at 3880 blocks / 112 concurrent = 34.6 waves. Numbers: IO 2.543 GB @ 460-480 GB/s ≈
**5.3-5.5 ms per 64-token chunk**; compute 2×64×248320×5120 = 162.8 GFLOP ≈ 2.0-2.3 ms at
71-82 TF — fully hidden under the stream (IO-bound, 39-43% tensor utilization). Logits
buffer: 64×248320×4 = **63.6 MB** VRAM (nll.cu already allocates `chunk*vocab`; both
nll.cu:78-80 and generate.cu run_nll:76-79 need the `dtype==BF16` branch that routes
here instead of the MXFP4 GEMMs).

Incidental: `row_logp_kernel` (nll.cu:27-31) carries the same unsynchronized `red[0]`
read-overwrite window as the fixed RoPE bug — flagged formally racy in w3/diff-verify.md;
fix opportunistically (dedicated max slot) since NLL is this path's acceptance gate.

---

## 10. Integration checklist (for whoever implements)

1. New `bf16_gemm` (§9, T=1..64) + `bf16_gemv_v2` (§4.2, persistent) + `embed_gather_bf16`
   (§7.2). One dispatch: `matrix().dtype == DType::bf16`.
2. Call sites to branch: decode.cu:46 (pair embed), :93-101 (verify lm_head: T=2 → GEMM),
   :137-138 (MTP embed), :151 (mtp.fc [5120,10240]: GEMV or GEMM T=1), :186 (draft lm_head:
   GEMV/GEMM T=1 + argmax), nll.cu:78-80 + generate.cu:76-79 (NLL GEMM T≤64).
   9B `bf16_gemv` callers (mtp.fc) migrate to the new kernel; delete the scalar one.
3. lm_head pinned VRAM at load (2.368 GiB — include in the residency budget before
   allocating layer L=21..; guard: refuse to start if free VRAM < budget+slack).
4. embed: cudaHostAlloc 2,542,796,800 B (or VirtualAlloc+cudaHostRegister), one-time
   NVMe read ≈ 0.4 s; register the UVA pointer; graph-safe.
5. Benchmarks (cold-L2 protocol, insig4-perf §2.4): bf16 GEMV/GEMM on [248320,5120] vs
   the 5.3-5.9 ms prediction; embed gather latency; argmax unchanged.
   Parity: `tools/reference_*.py` — embed row equality is exact (pure copy+cvt);
   lm_head dot vs NumPy reference at fp32 tolerance (§3.1 ⇒ ~1e-6 rel).

Sources: [TechPowerUp — RTX 4070 SUPER specs (35.48 TFLOPS, 504.2 GB/s, 7168 cores)](https://www.techpowerup.com/gpu-specs/geforce-rtx-4070-super.c4186),
[Reddit/PugetSystems/GigaGPU — 4090 token-generation throughput discussions](https://www.reddit.com/r/LocalLLaMA/comments/13j5cxf/how_many_tokens_per_second_do_you_guys_get_with/),
[llama.cpp mul_mat_vec bandwidth-utilization history (21%→66% fix note)](https://pypi.org/project/llama-cpp-pydist/),
[llama.cpp issue #19081 — MUL_MAT_VEC per-kernel timings](https://github.com/ggml-org/llama.cpp/issues/19081),
plus repo-internal: audits/internals.md:17-28 (379-435 GiB/s lm_head streaming),
audits/w3/insig4-perf.md (bench protocol, persistent GEMV), audits/w3/spec-deepen.md
(draft cost model, slice tables), audits/w2/loader-27b-spec.md (tensor facts),
audits/synthesis.md (tier costs, placement).
