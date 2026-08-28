# w4 audit — weight-stationary (layer-major) 27B prefill: validation against live code + measurements

Date 2026-08-25. Read-only audit; the only file written is this report. Everything verified against
the live tree (`src/decode.cu`, `src/prefill.cu`, `src/fp8.cu`, `src/gemm.cu`, `src/deltanet.cu`,
`src/dump_pf.cu`, `src/storage.cu`, `src/streaming.cu`, `include/*.hpp`, `tools/reference_pf_i4.py`)
and against `audits/w3/MASTER-PLAN.md` Phase F.1 + `audits/w3/prefill-27b.md`. Three already-built
binaries were run for empirical grounding (`build/dump-pf.exe`, `build/generate.exe`,
`build/bench-gemm.exe` — no builds performed). Companion w4 reports cross-checked: `fp8-kernels.md`,
`streaming.md`, `tier-dispatch.md`, `cpu-tier.md`.

---

## 0. Verdict in one paragraph

The w3 weight-stationary design is **structurally correct against the live code** — the loop-reorder
invariant holds, all three seam-carry mechanisms (KV causality, conv roll, deltanet state) are real
and verified at file:line below — but it has **one latent correctness bug** (conv `row0_snap=nullptr`
fallback corrupts state for tail tiles T<3), **one wrong performance premise** (the w3 "0.42–0.50 s
of GEMM per 512-token sweep at 50–60 TF/s" is contradicted by measurement: the GEMM chain ingests
weights at only 65–90 GiB/s / 12–23 TF/s today, so 27B sweep compute is 0.4–1.6 s — the swing factor
that decides whether prefill is NVMe-bound or co-binding), and the **C-tier prefill question resolves
decisively for GPU staging**: `insignia_cpu.hpp` has no T>1 GEMM and CPU compute is arithmetically
hopeless (391.7 GF/layer vs ~150 GF/s effective CPU → ~2.6 s/layer). Master plan §2.4's
prefill(512) ≈ 3.0–3.5 s survives within error bars (realistic band 2.8–4.0 s), NVMe-bound at the
center estimate. Effort ≈ 500 engine LOC, matching Phase F.1's ~450.

### Measurements performed this session (already-built binaries only)

| measurement | result |
|---|---|
| `dump-pf.exe` T=14 seam dump (9B) | **runs clean** — the 66,048 B-smem `deltanet_prefill_kernel` launches and executes on the 4070 SUPER (opt-in already coded, prefill.cu:266) |
| `generate.exe` prefill 512 tok (9B, incl. cold TieredStorage upload) | 2852 ms |
| `generate.exe` prefill 1024 tok (same cold cost + 8 more warm chunks) | 3312 ms → **warm chunk = 57.5 ms / 64 tokens = 1.8 ms/layer/chunk**; cold upload ≈ 2.39 s for 4.77 GB ≈ **2.0 GB/s** (per-tensor `cudaMemcpyAsync`+sync, storage.cu:9 — the anti-pattern bulk C-tier copies must avoid) |
| `bench-gemm.exe` (mxfp4_gemm_v2, T=64, VRAM-resident) | 8192×4096: 0.256 ms, 65 GiB/s, **16.8 TF/s**; 12288×4096: 0.332 ms, 75 GiB/s, 19.4 TF/s; 4096×12288: 0.551 ms, 45 GiB/s, 11.7 TF/s; 248320×4096: 5.67 ms, 89 GiB/s, **23.0 TF/s** |

The 9B warm chunk decomposition closes: 24 delta layers × ~1.76 ms GEMM (summed from the bench
table) + 8 attn layers × ~2.0 ms ≈ 58 ms ≈ the measured 57.5 ms — the chunk pipeline is
**GEMM-chain-bound**, deltanet scans/gqa/elementwise hide inside it.

---

## 1. The current code (ground truth, all line refs live)

- **Chunk-major prefill**: `Qwen35Decode::prefill_chunk_device` — `src/decode.cu:42-109`. Embed
  gather (:46), then `for l in 0..31` (:47): rmsnorm (:49), full-attn branch (:51-63: q_proj /
  split_q_gate_batch / k,v / qk_norm_rope_batch / store_kv_batch / gqa_prefill / sigmoid_mul /
  o_proj) or DeltaNet branch (:64-86: in_proj_qkv/z/a+b via one `mxfp4_gemm_ab_i4` launch :75,
  conv_prefill_silu :79, deltanet_params_batch :82, deltanet_prefill :84, gated norm :85, out_proj),
  residual (:88), MLP (:90-92), residual (:93), seam fire (:94). Final norm (:96), lm_head + argmax
  (:97-105), hidden copy-out (:106), **`addi(pos_dev, T)` + `x_.position += T` per chunk**
  (:107-108). Weights are re-acquired **per layer per chunk** through `TieredStorage` (`tensor()`
  decode.cu:30 → storage.cu:9 acquire with LRU + per-miss `cudaMemcpyAsync`+`cudaStreamSynchronize`).
- **Drivers loop 64-token chunks**: `generate.cu:125-129`, `nll.cu:69-89` (NLL adds per-chunk
  lm_head GEMM + `row_logp_kernel`), `test_prefill.cu:24-29`.
- **Spec path is chunk-shaped and must stay byte-identical**: `spec_step` decode.cu:219-237 and
  `capture_spec` decode.cu:238-249 both call `prefill_chunk_device(x_.pf_tokens, 2)` (:224, :243)
  **inside stream capture** — the stationary prefill must be additive and never captured.
- **Seam plumbing exists** (not new): `prefill_chunk_seam` decode.cu:117-121 → seam callback
  `(int layer, const float *pf_x, int T, void *user)` fired after each layer's residual with a
  per-layer `cudaStreamSynchronize` (decode.cu:94). Consumer `dump_pf.cu`: seam fwriter :11-15,
  T≤64 cap :28, layout **[33, T, 4096] f32** (32 layer seams + final-norm seam appended from
  `pf_n` :37-40). Reference `tools/reference_pf_i4.py` is **token-major** (:79-90 loops tokens,
  layers inside, carrying `states`/`kvk`/`kvv`), reshape (33,T,4096) :74 — the exact computational
  inverse of layer-major, so seam comparison semantics survive the reorder unchanged.
- **Workspace** (`DecodeWorkspace` ctor decode.cu:11-27, fields insignia_decode.hpp:8-14): all
  `pf_*` buffers are 64-row (pf_x 64×4096 :22, pf_qkv 64×8192, pf_gate/up 64×12288 :24,
  `pf_bf16` 64×12288×2B :26; `pf_tokens` is **64 ints** :21); `snap_delta/snap_conv` :25;
  ctx guard `1..4096` (:12, `score[4096]` smem ceiling prefill.cu:106). KV `8*ctx*1024`,
  mtp KV `ctx*1024` (:14) — **mtp_keys/mtp_values are never memset at init** (:27 memsets only
  pos/am_scratch/delta_state/conv_state); generate.cu's probe path manually memsets them
  (:142-143) — the F.2/F7 garbage-KV gap is live in code today.
- **Shapes are 9B-hardcoded everywhere**: `Qwen35Shape` insignia_qwen35.hpp:7 (hidden 4096,
  intermediate 12288, layers 32, `full_attention(i)=(i&3)==3`); MTP fc dims hardcoded
  4096→8192 at decode.cu:151,154-155 (27B risk #8, unchanged).
- **Streaming layer exists and is smoke-tested**: `NvmeReader`/`PinnedRing`/`LayerFeeder`
  (include/insignia_streaming.hpp, src/streaming.cu; `streaming-smoke.exe` Aug 25 21:56).
  `LayerFeeder` ring defaults **4 slots × 368 MiB** (hpp:206-208, covers the 383.87 MB shard),
  `ConsumeMode{zero_copy, copy_out}` (hpp:203) where `copy_out` is the *reserved, unimplemented*
  slot→VRAM `cudaMemcpyAsync` hook (hpp:197-199 comment; `acquire_layer` streaming.cu:430-441
  returns the ring pointer only). Sequential release contract streaming.cu:443-450.
- **fp8 27B GEMM/GEMV pair exists**: `fp8_gemm` fp8.cu:186-190 — throws T>64 (:188), requires
  rows%32==0/cols%128==0 (:187), zero-padded 64-row bf16 A (comment :108), guarded store
  `if (wm*16 < T)` (:184, y must be 64-row — pf_* buffers are); `fp8_gemv2` pair with 99 KB
  opt-in (:100) throws at cols>25344 → 27B `down_proj` pair (cols 17408×2×4B=139 KB smem)
  **must** route through `fp8_gemm` T=2 (as fp8-kernels.md also concluded).
- **The Phase-0 v21 race fix is already in the tree**: gemm.cu:275-276
  `if (kb+2<ksteps) wait_prev else wait_all` (same pattern fp8.cu:167-168) — master plan's
  blocking-fix item is landed; do not double-count it.

What does **not** exist anywhere: an `h_A/h_B [S,5120]` activation ping-pong (today's residual
stream is the 64-row `pf_x`), a `prefill_layer_stationary` entry point, any T>1 CPU GEMM, the
`copy_out` feeder mode, and any T-row bf16 embed gather (only `bf16_get_row` single-row
fp8.cu:193-200).

---

## 2. Mission item 1 — design validation, piece by piece

### 2.1 Loop-reorder invariant: HOLDS (verified)

The only cross-layer channel is the residual stream h. Per-layer state is addressed by absolute
position (KV: `store_kv_batch` writes at `pos_dev[0]+t` prefill.cu:91, `gqa_prefill_kernel` scores
`j < pos_dev[0]+t+1` absolute cache rows prefill.cu:104,114-120,148-153 — causal across tiles and
turns) or is position-free recurrence (conv roll prefill.cu:184-195, deltanet state load/store
prefill.cu:224/263 — chained per tile by stream-ordered launches). Processing all tiles of layer l
before layer l+1 therefore produces byte-identical KV, conv, delta state and h. DeltaNet tile
chaining detail: host wrapper takes one T≤64 launch per call (prefill.cu:265-269), state round-trips
global→smem→global per tile — chaining N tiles is exactly what the 9B does across chunks today.

### 2.2 Activation ping-pong h_A/h_B [S,5120] f32: CORRECT, VRAM cost trivial, h_final is free

- S=512: 2×10.49 MB = **21.0 MB**; S=2048: **83.9 MB**; S=4096: **167.8 MB** (hard cap = ctx 4096
  from `score[4096]`, decode.cu:12). All fit trivially in the prefill window (master plan drops
  L=19→L=18 to free 384 MB for the staging slot; 292 MB spare exists even at L=19).
- **h_final needs no extra allocation**: 64 layers is even, so after the sweep h lives in the
  ping-pong buffer at known parity; keep the pointer. The MTP KV pass (§4) runs immediately after
  the sweep and consumes it raw (pre-final-norm residual — `mtp_layer` norms its hidden input
  itself, decode.cu:147-148). NLL mode doesn't need it either: run model.norm + lm_head GEMM on each
  tile's 64 rows right after layer 63 of that tile (generate.cu:69-89 pattern). This improves on
  w3 §2.1 which budgeted a separate 10 MB h_final copy.
- Per-tile in/out pointers: `in = h_prev + tile*64*5120`, `out = h_next + tile*64*5120`. The tile
  body needs `out` initialized from `in` — either a 1.3 MB device copy (~5 µs × 512 tile-visits ≈
  2.6 ms/sweep) or preferably a 3-arg `residual_add_out(in, d, out, n)` (~10 LOC) for zero copies.
  Everything else in the body already works on 64-row `pf_*` scratch — unchanged.
- Embed gather writes h_A directly: `embed_gather*` grid is `<<<T,128>>>` per row (prefill.cu:22,
  :39) — T=S rows in one launch works today; 27B needs the bf16 T-row variant (§6 item 4).

### 2.3 Position bumped ONCE per turn: CONFIRMED the classic bug, mechanism to fix it

Current: `addi(pos_dev,T)` + `x_.position+=T` **per chunk** (decode.cu:107-108). Stationary must
bump once (`addi_kernel_launch(pos_dev, S_total)` + host mirror). But per **(attn-layer, tile)** the
live position must read `turn_base + 64*tile`: all three consumers add only their in-tile `t`
(qk_norm_rope_batch prefill.cu:73, store_kv_batch :91, gqa_prefill :104). Cheapest correct mechanism:
a 4-byte device write per (attn layer, tile) — 16 attn layers × 8 tiles = 128 tiny launches ≈
0.2–0.3 ms/sweep (the `set_position` H2D memcpy, decode.cu:124, or a 1-thread `seti` kernel to stay
capture-free). DeltaNet/conv layers ignore position — no updates needed for the other 48 layers.
Optimization (device array of tile bases passed as kernel arg) is unnecessary at 0.3 ms.

### 2.4 row0_snap=nullptr during turn prefill: LEGAL for deltanet, **BUGGY for conv at T<3 — new finding**

- Deltanet: `if (t==0 && snap)` prefill.cu:258 — nullptr cleanly skipped ✓ (saves 48×8×3.146 MB ≈
  1.2 GB of pointless writes per 512-token sweep at 27B, ~2.5 ms — minor but free).
- Conv: `conv_prefill_silu(..., row0_snap)` passes `row0_snap ? row0_snap : state`
  (prefill.cu:199). With snap==state the kernel's snap writes (prefill.cu:187-189) land on
  `state[c*3+0..2]`, and the roll loop (prefill.cu:191-194) then reads `state[c*3+3+j]` for
  j=T−3+i<0 — **for T=1 it reads snap-clobbered state[1] (s2) where s1 is required, and for T=2
  reads clobbered state[2] (x0) where s2 is required.** For T≥3 all roll reads come from x and the
  alias writes are dead — correct. Every current call site passes a distinct snap
  (decode.cu:79 `x_.snap_conv+di*8192*3`), so the bug is **dormant today** and is triggered exactly
  by the stationary prefill's tail tiles (turn lengths ≡ 1,2 mod 64 → last tile T∈{1,2}).
  **This corrects w3 prefill-27b.md §0 ("Passing nullptr is legal and cleanly skipped") — it is
  not, for conv, at T<3.** Fix (6 LOC, must land before any nullptr usage): read the three window
  values into locals before the snap writes, or guard the snap block on `snap != state`.
- The spec pair path (`prefill_chunk_device(T=2)`) keeps its snaps — `spec_rollback` consumes them
  (decode.cu:226, prefill.cu:305-314). Non-negotiable.

### 2.5 Tail-tile zero-padding of pf_bf16: mechanism verified, contract to preserve

`linear_batch::stage_a` memsets rows T..64 of `pf_bf16` then converts rows 0..T (decode.cu:36-37);
same pattern inline for a/b (decode.cu:73-74). `fp8_gemm` prefetches all 64 A rows (fp8.cu:123-127)
— zero-padding is load-bearing; guarded store keeps y rows ≥T unwritten-or-zero within the 64-row
buffers (fp8.cu:182-184, gemm.cu:369-370,447). The stationary tile body inherits this verbatim since
it reuses `linear_batch`/stage_a with per-tile T. 27B sizing: `pf_bf16` 64×17408×2B (master plan B).
All 27B cols (5120/6144/10240/17408) satisfy fp8_gemm's %128 (fp8.cu:187) and all rows %32.

### 2.6 Seam variant extension: minimal-delta design

Keep the exact callback signature `(int layer, const float *h, int n, void*)` (decode.cu:117).
`prefill_layer_stationary_seam(tokens, S, seam, user)`: fire per layer **after its last tile** with
the full `h_next [S,5120]` (one sync + D2H + fwrite per layer, 64/sweep, parity builds only —
`dump_pf.cu:11-15` already streams per seam, no RAM ring). `dump_pf.cu` edits: drop the T≤64 cap
(dump_pf.cu:28), parameterize 4096→shape hidden, layout becomes **[65, S, 5120] f32**. The 27B
reference (`tools/reference_pf_f8.py`, w3 §9 spec) reshapes accordingly and stays token-major —
comparison semantics unchanged from `reference_pf_i4.py:74-90`.

---

## 3. Mission item 2 — riding the tiers: the cadence arithmetic (empirically corrected)

### 3.1 The compute anchor (this audit's main correction to w3)

w3 §7.1 assumed ~50–60 TF/s → 0.42–0.50 s of GEMM per 512-token sweep. Measured today
(bench-gemm.exe, table §0): the GEMM chain delivers **12–23 TF/s and 45–89 GiB/s weight ingest** at
T=64 on 9B shapes. The 9B warm chunk is GEMM-bound at 57.5 ms. Scaled to 27B (FLOPs ×1.71/layer by
rows×cols, ×2 layers → 24.9 TF/sweep):

- At measured fp4-class rates (16–23 TF/s): **sweep GEMM ≈ 1.1–1.6 s**.
- If fp8_gemm lands at the sibling fp8-kernels.md optimistic class (35–70 TF/s MMA-bound):
  **0.36–0.7 s**.
- **Nobody has benched fp8_gemm** (fp8-kernels.md: no measurement exists; zero call sites in tree).

⇒ Compute is the **swing variable**. Per layer per tile the ~8–10 weight-matrix GEMMs
(decode.cu:66-92: qkv, z, a+b fused, out, gate, up, down) move ~384 MB fp8; at 45–89 GiB/s ingest
that is **4.3–8.5 ms/layer/tile**, i.e. **34–68 ms/layer/sweep** — not w3's 6.8 ms. VRAM-resident
V-layer compute for 19 layers ≈ **0.65–1.3 s** per sweep, not 129 ms. **First action of Phase F:
bench `fp8_gemm` on [17408,5120] and [10240,5120] at T=64** (bench_gemm.cu pattern, no new
infrastructure) — it decides whether prefill is NVMe-bound or co-binding.

### 3.2 Per-tier cadence at 64-token tiles (S=512, 8 tiles/layer/sweep), v2 = L19/Z21/C9/N15

| tier | stream/fill (ms/layer) | compute after staging (ms/layer/sweep) | hides? |
|---|---|---|---|
| V (19 layers) | 0 | 34–68 (§3.1) | overlaps N fills |
| Z pinned (21) | 15.9 H2D pinned → VRAM slot (24 GB/s) | 34–68 | H2D ≪ 115.4 ✓ |
| N E: NVMe (15) | **115.4** (381 MB @ 3.3 GB/s) into ring; ring→VRAM 16 (v1/v1.5 pinned ring) or ~84 (v2 pageable ring) | 34–68 | copy+compute ≈ 50–152 ≪ 115.4 ✓ NVMe binds |
| C locked-pageable (9) | ~84 bulk pageable H2D (master plan; but see caveat) | 34–68 | host-blocking copy; overlaps N fills if interleaved |

Key structural point (answers the "ring double-buffer cadence" question): the stationary loop is
**host-orchestrated and needs no sequencer thread** (unlike decode's treadmill, tier-dispatch.md).
The host enqueues all 8 tiles of layer i−1 (≤68 ms of GPU work) and then **blocks on
`acquire_layer(i)`** (streaming.cu:430-441) for up to 115.4 ms — the CUDA stream queue absorbs the
gap; the IOCP reader self-arms slots ahead (hpp:191-199). Cadence per N layer =
max(115.4 NVMe, ~50–152 copy+compute) = **115.4 ms** as long as fp8 GEMM ingest ≥ ~10 GiB/s — i.e.
yes, **compute hides under the stream for all tiers at the measured rates**, but with far less
slack than w3 claimed (1.5–3×, not 17×).

Total per 512-token turn (additive where serialized, overlapped where pipelined):
`T ≈ max(15×115.4 NVMe, Σcompute 0.8–1.7 s) + 21×15.9 Z-H2D (overlappable) + 9×84 C-pageable
(host-serialized ≈ 0.76 s, partially overlappable with N fills) + tails/launches ~45 ms
(≈10k launches × ~3 µs + pos writes + syncs)` ⇒ **≈ 2.8–4.0 s, center ≈ 3.2 s**. Master plan's
3.0–3.5 s stands; the "V-compute 129" term in its formula should be restated as 0.65–1.3 s and
re-checked after the fp8 bench. NVMe traffic is 15×381 MB = **5.7 GB per turn** (same bytes as one
decode step) — prefill does NOT multiply NVMe traffic by tokens; it multiplies it by *turns*, and
chunk-major would multiply it by chunks: **current-code counterfactual at v2 placement: 8 chunks ×
(15×115.4 + 9×84 + 21×15.9) ≈ 8 × 2.03 s ≈ 16.2 s/turn (plus 8× re-uploads) vs ~3.2 s stationary —
the 5–8× win, IO-bound as designed.**

Two caveats on the C tier:
1. The 84 ms assumes a **bulk** pageable H2D. The only pageable-copy path measured in this tree is
   TieredStorage's per-tensor sync pattern at ~2.0 GB/s (storage.cu:9, §0) — a per-tensor or
   per-tensor-with-sync copy of a C layer would cost ~190 ms. The C-tier copy must be **one
   cudaMemcpy per layer slab** (or a few), issued from the stationary loop.
2. Optional knob (~40 LOC, G-knob class): a 2×32 MB **pinned bounce buffer** (cudaHostAlloc'd —
   64 MB is nothing against the 8,531 MB cap) with 2 CpuPool workers memcpy'ing pageable→pinned
   (~21 ms/layer) while `cudaMemcpyAsync` pinned→VRAM streams at 24 GB/s (15.9 ms) — drops C cost
   to ~35–40 ms/layer and makes the host non-blocking. Not required for 3.0–3.5 s; bench first.
   (cudaHostRegister of the C pages themselves is forbidden by the pinned cap: Z=21 already spends
   8,000 of 8,531 MB.)

### 3.3 CPU-tier prefill: no T>1 path exists, and CPU GEMM is arithmetically dead — recommend GPU staging

`include/insignia_cpu.hpp` contains exactly: `fp8_gemv_mt` (T=1, hpp:460-467), `fp8_gemv2_mt`
(T=2 pair, hpp:470-477), `fp8_gemv_st` (hpp:480-486), `bf16_gemv_mt` (hpp:520-524), and
single-token layer ops (deltanet_step_cpu hpp:721-727, gqa_decode_cpu hpp:925-952, norms, conv,
params). **There is no GEMM of any shape.** The two C-prefill options:

- **T-looped GEMV ×64** (what exists): CPU layer GEMV is 10.8 ms/token (master plan §1.3) →
  64 tokens × 10.8 = **691 ms/layer/tile**, × 8 tiles = 5.5 s/layer — matches the mission's
  "688 ms/layer unacceptable" and is 60× off budget. Dead.
- **New CPU blocked GEMM (AVX2)**: the weights-once property would fix bandwidth (384 MB @ 37 GB/s
  ≈ 10 ms/layer/sweep) but prefill at T=64..512 is **compute-bound on CPU**: 391.7 GF/layer
  (recomputed in w3 §7.1, checked) against a 5600X realistic ~150 GF/s for dequant+FMA
  (e4m3x32_rr costs ~18 vector ops per 32 weights, hpp:127-156 — a ~4.5× overhead on the FMA
  lanes; peak fp32 FMA is ~0.8 TF/s) → **~2.6 s/layer** even with perfect blocking, ×24 C+N layers
  = ~60 s. A register-blocked kernel lifts effective GF/s maybe 2×; still >20× off. **Do not build
  it.** (Contrast: decode T=1 is bandwidth-bound, where the CPU tier is excellent — 10.8 ms/layer
  is a *decode* number.)

**Recommendation (matches master plan §2.4): C layers stream pageable→VRAM slot for the prefill
window and compute on GPU like everything else; CPU stays idle during prefill** (or runs the
bounce-buffer memcpys, §3.2 caveat 2). The weight-stationary property makes this cheap: each C
layer's 384 MB crosses PCIe once per **turn** regardless of S.

### 3.4 VRAM staging slot & feeder integration

One 384 MB slot (master plan; drop L→18 during the window). `LayerFeeder::ConsumeMode::copy_out`
is the designed-but-unimplemented hook (hpp:197-199, streaming.cu:430-441 returns the ring pointer
only). Implement `copy_out` as: `acquire_layer` blocks until READY, then issues the caller's
slot→VRAM `cudaMemcpyAsync` chain (per-request offsets via `map()`, streaming.cu:452-458) and
returns; `release_layer` after the layer's last tile (the feeder's sequential contract
streaming.cu:443-450 matches layer-order consumption; V/Z/C layers simply don't touch the feeder
between N acquires — read-ahead depth slots−1 keeps the reader saturated). Z layers: direct
`cudaMemcpyAsync` pinned→slot (15.9 ms). C layers: bulk pageable→slot (~84 ms). 16-byte alignment:
the ring slot base is 4096-aligned and in-slot offsets come from the plan — the F8 rebasing pad
(master plan D.2) must be in the plan builder or fp8 kernels' `uint4` loads fault; assert
`(f8_base & 15) == 0` at acquire.

---

## 4. Mission item 3 — KV fill + MTP KV fill (master plan F.2)

**Main layers, in-loop**: per (attn layer, tile): `store_kv_batch` **before** `gqa_prefill`
(decode.cu:60-61 order) at absolute positions — tile t's gqa sees tiles 0..t's rows because
layer-major finishes all tiles of layer l before l+1 (and within a tile, store precedes attend).
Cross-turn: gqa reads the full per-layer cache (prefill.cu:104) — turn k+1 attends turn k's rows
with zero extra machinery. KV write traffic/sweep: 16 layers × 512 × 1024 × 4 B × 2 ≈ 67 MB —
noise. 27B capacity: `kv_keys/values 16*ctx*1024` f32 = 268 MB @ctx2048 (bf16 flag halves;
master plan G). `store_kv_batch` grid `dim3(4,T)` (4 kvh × 256 = 1024 — 27B GQA is also 4 kv heads,
24/6) — **unchanged kernel**; only `gqa_prefill` grid/kvh change (§5).

**MTP KV (the F.2 pass)** — design validated against live semantics:
- Today's gap: `mtp_keys/mtp_values` are allocated (decode.cu:14) and **never initialized**;
  `mtp_layer` writes one slot per draft (decode.cu:173 via `mtp_pos_dev`) and `gqa_decode` reads
  slots 0..pos (decode.cu:174) — first-draft attention over garbage. `generate.cu:142-143`'s manual
  probe memset confirms the need.
- Semantics (verified): MTP slot s is written from `(embed(t_{s+1}), h^main_s)` — decode path:
  `mtp_layer` embeds the *pending* token (decode.cu:141) and norms `x_.hidden` (the raw main
  residual, :147-148), concatenates embed-first (concat :151 — **embed-normed half first**, the
  risk-#8 order), fc GEMM (:153-157), then the layer's own q/k norm+rope at `mtp_pos` and
  `store_kv` (:170-173). `spec_prologue` sets `pos[7]=pos[0]-1` (prefill.cu:277-278) so the first
  draft's own slot lands at P−1 for a P-token prompt. Hence the post-prefill pass must fill slots
  **0..P−2** with exactly that pairing — the slots are independent (K/V of slot p need only
  (embed(t_{p+1}), h_p); no MTP residual chaining) ⇒ tile-parallel batch pass:
  per 64-row tile: bf16 embed gather of `tokens[1..]` (new T-row gather; single-row
  `bf16_get_row` exists fp8.cu:198), `rms_bf16` ×2 (pre_fc_norm_embedding/hidden), concat
  (embed-first), fc **bf16 GEMM** [10240,10240] (needs Phase C.4 `bf16_gemm`; 9B hardcodes
  4096→8192 at decode.cu:151,154-155), q/k/v projections (fp8_gemm T=64; v needed — store_kv
  consumes it), `qk_norm_rope_batch` (27B grid 28, isq<24) at pos = turn_base+tile, `store_kv_batch`
  into mtp caches. **Skip gate/o_proj/MLP entirely.** Weights touched ≈ fc 210 MB bf16 + q/k/v
  ~42 MB fp8 (all in the VRAM-pinned MTP shard) → ~0.5–1 ms/tile ⇒ **~4–8 ms per 512-token turn**,
  consistent with master plan's ~4 ms. Plus 1 LOC: memset both mtp caches in the ctor
  (decode.cu:27). Multi-turn: rerun for the turn's new positions; the turn's last slot is covered
  by the next draft (same off-by-one as w3 §4 note — the pairing (embed(t_{p+1}), h_p) must use
  this turn's tokens only).
- Rejected alternative (b) stays rejected: filling lazily inside the first spec step saves nothing,
  and the spec path is graph-captured (decode.cu:238-249) — a one-shot branch would poison the
  capture.

---

## 5. Mission item 4 — DeltaNet prefill scan at 27B: launcher verified, runs today

- **Location correction**: the prefill scan is in **src/prefill.cu:219-269**, not deltanet.cu
  (that file holds only the decode kernel, 15 lines, `deltanet_decode_kernel<<<32,128>>>` :14).
- **smem + opt-in**: dynamic smem `64*1024+512 = 66,048 B` — the launcher's static one-shot
  `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, 64*1024+512)` **is
  present** (prefill.cu:266) and the launch is `<<<32,128,66048,stream>>>` (:268). sm_89 allows
  99 KB opt-in/block; 66,048 + 544 B static (`sq/sk/delta`, :233) fits. **Empirically verified this
  session**: `dump-pf.exe` (which drives `prefill_chunk_seam` → `deltanet_prefill`) ran clean —
  the kernel compiles, launches, and produces the parity seams on this 4070 SUPER.
- **At 48 heads**: smem is per-block head state [128×128] + slack — **head-count independent**;
  only `blockIdx.x` count changes: `<<<48,128,66048>>>`. 48 ≤ 56 SMs and 66,048 B > 50 KB ⇒ 1
  block/SM ⇒ **exactly one wave on 48 of 56 SMs** (8 idle — the scan is serial-latency-bound per
  block; splitting heads×T-chunks with state handoff is possible but unnecessary — scans hide
  inside the GEMM chain per §0's decomposition). `kh = head>>1` → **`head/3`** (prefill.cu:221);
  qkv row stride `t*8192` → `t*10240` with offsets q@0/k@2048/v@4096 **staying** (:227-229, v
  length 4096→6144); a/b index `t*32+head` → `t*48+head` (:244-245); out row `t*32+head` →
  `t*48+head` (:256); state pointer stride per layer `di*32*128*128` → `di*48*128*128` at the
  call site (decode.cu:84 pattern). Decode twin: `deltanet_decode_kernel` same edits (:5,:14) +
  params kernel `h>=32` guard → 48 (prefill.cu:205-206).
- **Conv prefill at 10240**: `conv_prefill_kernel` hardcodes 8192 ×3 (prefill.cu:172,176,179),
  `conv_roll_state_kernel` ×2 (:185,:193), wrapper `n=T*8192`, grid `(8192+255)/256` (:197-199) —
  mechanical 10240 retune (shape-only, Phase B). State sizes 27B: conv 48×10240×3 f32 = 5.9 MB,
  delta 48×48×128×128 = 151 MB (decode.cu:14-15,25); snap_conv/snap_delta likewise (spec path).
  Conv seam carry across tiles verified (:176 `state[c*3+i]` when `t<3-i`), with the §2.4 nullptr
  caveat.

---

## 6. Mission item 5 — implementation order with LOC (dependencies in brackets)

| # | change | files | LOC | notes / gate |
|---|---|---|---|---|
| 0 | **Bench `fp8_gemm` [17408,5120] & [10240,5120] T=64** (bench_gemm.cu pattern) | new bench body in existing tool style | ~40 | decides §3.1's swing; gate for accepting the 3–3.5 s target |
| 1 | conv_roll nullptr fix (read-before-snap or `snap!=state` guard) | prefill.cu:184-199 | ~6 | **before any nullptr usage**; unit: T=1,2 roll vs reference |
| 2 | Workspace: `h_A/h_B [S_cap,5120]` f32 alloc/free + `S_cap` ctor param; memset mtp KV | decode.cu:11-29, insignia_decode.hpp:8-14 | ~25 | h_final = parity pointer, no alloc (§2.2) |
| 3 | `residual_add_out(in,d,out,n)` | ops.cu/cuh | ~10 | kills 512 tile copies |
| 4 | Tile-body extraction: parameterize chunk body (in/out/T/tile) keeping the T=2 pair path bit-identical | decode.cu:42-109 | ~75 moved | **gate: 9B dump-pf byte-identical before/after** (prefill_chunk_device stays the spec/graph path) |
| 5 | `prefill_layer_stationary(tokens,S)` + `_seam`: layer loop × 64-tiles, pos per (attn,tile) + one bump, embed gather→h_A, ctx guard once, greedy tail (decode.cu:103 pattern) / NLL tiles (generate.cu:69-89 pattern) | decode.cu + hpp | ~120 | never capture; run OUTSIDE graphs |
| 6 | Seam tooling: dump_pf S-cap + [65,S,5120] | dump_pf.cu:28,11-15,37-40 | ~15 | pairs with tools/reference_pf_f8.py (Phase E, ~250 LOC) |
| 7 | `LayerFeeder::copy_out` (slot→VRAM async chain via `map()`) + alignment assert | streaming.cu/hpp | ~80 | used by N (and Z direct memcpy) |
| 8 | Tier glue in the stationary loop: V direct / Z pinned→slot / C bulk pageable→slot / N feeder wait; single 384 MB slot; L→18 window | decode.cu (or a thin driver) | ~120 | R3 extension: byte-equal seams through the streamed path |
| 9 | MTP KV batched fill + bf16 T-row embed gather + fc bf16 GEMM reuse [Phase C.4] | decode.cu/new kernel chain | ~90 | acceptance-probe gate (p ≥ 0.55); slots 0..P−2 |
| 10 | 27B retunes of the prefill kernels (kvh→/6 grid 24, kh→/3 grid 48, strides 10240/12288/17408, conv 10240, `pf_*`/snap sizing) [Phase B] | prefill.cu, decode.cu:14-27 | ~60 sites | R4–R6 ride this |
| 11 | Drivers switch: generate/nll/test_prefill prompt loops → one stationary call (super-chunk at S_cap) | generate.cu:125-129, nll.cu:69-89, test_prefill.cu | ~20 | keep chunk loop behind a flag for the 9B |
| — | total engine | | **≈ 500** | master plan F said ~450 ✓ (+250 reference script, Phase E) |

Order matters: 1 → (2,3) → 4 → 5 → 6 gives a **VRAM-resident 27B stationary prefill parity path**
(R4-equivalent) with zero streaming; 7 → 8 adds tiers; 9 is independent after Phase C.4; 10 is
Phase B's checklist applied to prefill.cu. The 9B regression gate after every step (chunk path
byte-identical; `dump-pf` diff empty).

---

## 7. Corrections & risks register (deltas to master plan / w3)

1. **[NEW BUG, P1] conv `row0_snap=nullptr` corrupts state at T∈{1,2}** (prefill.cu:187-199) —
   dormant today, triggered by stationary tail tiles. Fix in step 1. (w3 prefill-27b §0's legality
   claim is half-wrong.)
2. **[PERF PREMISE, P1] sweep GEMM compute is 0.4–1.6 s, not 0.5 s** (measured 12–23 TF/s / 45–89
   GiB/s ingest, §3.1). Prefill(512) v2 = 2.8–4.0 s; NVMe-bound only if fp8_gemm ≥ ~20 TF/s.
   Bench first (step 0). Master plan §2.4's "V-compute 129" should read 0.65–1.3 s.
3. **[CONFIRMED] CPU-tier prefill must be GPU-staged** — no T>1 CPU path exists
   (insignia_cpu.hpp has gemv/gemv2 only); CPU GEMM ≈ 2.6 s/layer is arithmetically dead (§3.3).
   C-tier copy must be one bulk slab memcpy per layer (TieredStorage's per-tensor sync pattern
   measured ~2.0 GB/s would cost ~190 ms/layer, storage.cu:9).
4. **[GAP, live] mtp_keys/mtp_values uninitialized at ctor** (decode.cu:27) — the F.2 memset is
   1 LOC and also fixes the 9B probe path's manual memset (generate.cu:142-143).
5. **[HAZARD] stationary prefill must never run inside stream capture** — `capture_spec` wraps
   `prefill_chunk_device(T=2)` (decode.cu:243); the new entry point does host-blocking feeder
   waits (streaming.cu:430) and per-tile pos writes — graph-incompatible by construction. Keep it
   eager-only (matches master plan "no graphs at 27B").
6. **[ALIGNMENT] ring F8 16-byte rebasing** must be in the plan builder before copy_out consumes
   fp8 tensors through `uint4` loads (fp8.cu:32,76,126) — streaming.md flags it as unimplemented
   and inexpressible in ReadPlan today; the VRAM-slot path inherits it (assert at acquire).
7. **[MINOR] `pf_tokens` is 64 ints** (decode.cu:21) — stage tokens per tile (256 B memcpy) or
   size to S_cap; h_A gather wants the full token array device-side anyway (one S×4B H2D).
8. **[MINOR] TieredStorage per-tensor acquire is not the 27B prefill mechanism** (per-miss sync +
   one-tensor granularity, storage.cu:9) — the stationary path acquires a whole layer shard as one
   feeder plan/slot; TieredStorage remains the 9B/VRAM-tier mechanism.
9. **[CHECKED, no action] gemm.cu:275 last-K-step wait is already the fixed pattern** (same in
   fp8.cu:167-168) — master plan Phase 0's item is landed; don't re-fix.
10. **[SCOPE GUARD] ctx ≤ 4096** (score smem, decode.cu:12): S ≤ 4096 hard; ping-pong at S=4096 =
    168 MB is fine, but the turn-admission guard fires once per turn (decode.cu:45 → stationary).

## 8. What was run (all pre-built binaries; nothing compiled, nothing modified)

- `build/dump-pf.exe build/qwen35.insignia-index <14 ids> 14 out.f32` → "dumped 32 prefill seams
  T=14" (deltanet_prefill 66,048 B launch verified; temp output deleted).
- `build/generate.exe build/qwen35.insignia-index <512 ids> 4` and `<1024 ids> 4` → prefill
  timings 2852 / 3312 ms (warm-chunk isolation, §0).
- `build/bench-gemm.exe` → 4-shape T=64 GEMM table (§0).

All other claims are file:line code citations. Reference clones untouched; no builds; no source
edits; this report is the only file written.
