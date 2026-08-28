# w3 design: weight-stationary (layer-major) prefill for Qwen3.8-27B-FP8 — 2026-08-25

Scope: replace the chunk-major prefill (`prefill_chunk_device`, `src/decode.cu:42-105`) with a
layer-major pass for the tiered rig (VRAM ~10.8 GB app / RAM ~13.5 GB pinnable / NVMe 6.5 GB/s;
25.65 GB text weights, 64 layers, per-layer shards 383.88 MB linear / 372.33 MB full-attn).
Read-only design audit; nothing built, nothing committed. Every code claim below was re-read at
the cited line this session. Companion reports: `audits/synthesis.md` (model facts + tier costs),
`audits/w3/loader-gaps.md` (TieredStorage2, shard-major slots), `audits/w3/graph-hazards.md`
(KV-full / graph hazards), `audits/w3/spec-deepen.md` (MTP slot semantics, F7).

---

## 0. The three seam questions, answered from the 9B code (verified)

**Q1 — does the GQA prefill kernel read the layer's FULL KV cache (causal across tiles/chunks)?
YES.** `gqa_prefill_kernel` (`src/prefill.cu:102-161`): each (head, token) block computes
`tokens = __ldg(pos_dev) + t + 1` (`prefill.cu:104`) and loops `j = 0 .. tokens-1` over the
layer's cache `kc/vc` (`prefill.cu:114-120` scores, `:148-153` AV) — absolute positions from 0,
not just the current chunk. This is exactly why the 9B's 64-token chunked prefill works at all:
chunk c+1 at layer l attends chunks 0..c. `store_kv_batch` writes K/V at absolute
`pos_dev[0]+t` (`prefill.cu:88-98`) and runs *before* `gqa_prefill` in the same tile
(`decode.cu:60-61`). Layer-major just reorders the loops; per-tile store-then-attend inside a
layer produces byte-identical KV and attention results. The shared `score[4096]` buffer
(`prefill.cu:106`) bounds context to 4096 (enforced at `decode.cu:12`) — unchanged for 27B.
Required 27B deltas: `kvh = head >> 2` → `head / 6` (GQA group 24/4=6, `prefill.cu:103`), grid
16→24 heads, `split_q_gate_batch` stride 8192→12288 (24 heads × interleaved 256q+256gate — the
per-head interleave is what the kernel reads, `prefill.cu:43-51`; loader-gaps §4.2's "gate@6144
block" note is inconsistent with the kernel, keep the kernel's interleave reading of
`q_proj [12288,5120]`).

**Q2 — does conv1d state carry across tile seams? YES.** `conv_prefill_kernel`
(`prefill.cu:170-181`) takes the 3 pre-chunk inputs from `state[c*3+i]` when `t < 3-i`, and
`conv_roll_state_kernel` (`prefill.cu:184-195`) rolls the last 3 raw inputs into `state` after
each call. Chaining 8 T=64 calls (or any mixed sizes incl. a tail tile < 64) is the same
mechanism the 9B uses across chunks. 27B: channel count 8192→10240 (shape-only).

**Q3 — does the deltanet chunk-sequential scan carry state across calls? YES.**
`deltanet_prefill_kernel` (`prefill.cu:219-269`) loads the per-layer state from global into
shared (`:224`), runs T ≤ 64 tokens serially, writes state back (`:263`). Host wrapper limits T
to one launch (`:265-269`). Chain 8 calls per layer per 512-token sweep. 27B: 48 blocks (v-heads)
instead of 32, `kh = head>>1` → `head/3`, qkv row stride 8192→10240 (q@0, k@2048, v@4096 — v
offset is unchanged, only its length 4096→6144), a/b per token 32→48, state 48×128×128 f32 =
3.146 MB/layer.

Bonus verified fact: the row-0 snapshots (`snap` args, `prefill.cu:199,258-262`) are spec-rollback
artifacts. Passing `nullptr` is legal and cleanly skipped; the current chunk path always passes
them (`decode.cu:75,80`), wasting 48 layers × 8 tiles × 3.15 MB = 1.2 GB of pointless VRAM
writes per 512-token sweep (~2.5 ms). Turn-prefill should pass nullptr.

---

## 1. Algorithm — `prefill_layer_stationary(tokens, T_total)`

```
// one TURN of T_total ≤ ctx prompt tokens; weights for layer l come from l's tier
// (VRAM-resident read directly; RAM/NVMe tiers stream into a double-buffered VRAM
// staging slot of 384 MB — loader-gaps §3.3/§7.3 shard-major slots).

set_position(turn_base = x_.position)               // absolute base for this turn
gather embed rows for tokens → h[0]  [T_total, 5120] f32   // ping-pong buffer A
                                                    // (bf16 batch gather kernel, §2)
for l in 0..63:
    stage layer l's shard (prefetch began during layer l-1's compute):
        VRAM tier : weights already resident, pin for the layer
        RAM  tier : pinned slot → cudaMemcpyAsync → VRAM staging slot A/B  (17.4 ms)
        NVMe tier : IOCP fill pinned slot (59.1 ms, overlapped 1 layer ahead)
                    → pinned → VRAM staging slot A/B                    (17.4 ms, pipelined)
    for tile t in 0..ceil(T_total/64)-1:            // 8 tiles for 512 tokens
        T = min(64, T_total - 64*t);  set pos_dev = turn_base + 64*t   // §6 hazard 3
        in  = h_prev + 64*t*5120        // contiguous rows, no copy
        out = h_next + 64*t*5120
        rmsnorm(in → pf_n)                          // bf16 weights, f32 out
        if full_attention(l):                       // (l&3)==3, 16 layers
            fp8_gemm q/k/v (+q_gate) → split_q_gate; qk_norm_rope(pos=tile base)
            store_kv_batch(kc[l/4], vc[l/4])        // absolute pos, before gqa
            gqa_prefill over FULL layer cache      // causal across tiles (Q1)
            sigmoid_mul(core, gate); fp8_gemm o_proj → pf_down
        else:                                       // 48 DeltaNet layers
            fp8_gemm in_proj_qkv / in_proj_z; bf16 GEMM in_proj_a/b [48,5120]
            conv_prefill_silu(state=conv_state[di], snap=nullptr)     // seam carry (Q2)
            deltanet_params_batch (A_log now bf16)
            deltanet_prefill(state=delta_state[di], snap=nullptr)     // chain (Q3)
            gated_rmsnorm(z); fp8_gemm out_proj → pf_down
        residual_add(out, pf_down)
        rmsnorm(out → pf_n); fp8_gemm gate/up; silu_mul; fp8_gemm down; residual_add(out)
    release layer l's pins; staging slot returns to the pool
    [parity mode: fire seam(l, h_next, T_total) — streams to disk, §5]

final norm on h_next; keep h_final = h_next (MTP KV pass + NLL need it)
last-token path: lm_head GEMV row (T_total-1) → argmax → next_dev           // greedy
NLL mode:       lm_head fp8/bf16 GEMM in 64-row tiles over all T_total rows
copy h_final[last row] → x_.hidden; addi(pos_dev, T_total) ONCE; x_.position += T_total
mtp_prefill_kv(h_final, tokens[1..T_total-1])       // §4 — fills MTP KV slots 0..P-2
```

Why layer-major is safe (the invariant): nothing in layer l's math depends on another layer's
KV/delta/conv state — the only cross-layer channel is the residual stream h. Swapping the loop
order of (layer, chunk) therefore commutes exactly, including KV contents, recurrent state
contents, and rope positions (all addressed by absolute position).

Weight traffic: each layer's bytes cross the NVMe→pinned→VRAM path **once per sweep**, then serve
all 8 tiles from VRAM at ~500 GB/s. That is the entire win over chunk-major, which re-acquires
all 64 layers per 64-token chunk (8× per 512-token prompt).

---

## 2. Memory schedule

### 2.1 VRAM (10.8 GB app budget) — concrete placement (V=9 / M=31 / N=24)

| resident | bytes | note |
|---|---|---|
| lm_head bf16 [248320,5120] | 2543 MB | mandatory VRAM (synthesis; 5.1 ms/token vs 102 ms over PCIe) |
| embed bf16 [248320,5120] | 2543 MB | VRAM per loader-gaps manifest; pin permanently |
| mtp shard (fc bf16 105 MB + full-attn layer 372 MB) | 477 MB | VRAM-pinned — baked into the spec graph |
| 9 layer shards resident | 3428 MB | V tier |
| KV caches f32 16×ctx4096×1024×2 | 537 MB | bf16 KV halves → 268 MB (knob) |
| mtp KV | 34 MB | |
| delta state 48×3.146 MB (+spec snapshots 151 MB) | 302 MB | snapshots are decode-spec artifacts |
| conv state + snap | 12 MB | |
| staging slots ×2 (double buffer) | 768 MB | transient, LRU'd between layers |
| activation ping-pong h A/B, f32, S=512 | 21 MB | **the whole activation footprint** |
| h_final copy | 10 MB | input to MTP KV pass / NLL |
| pf_* tile scratch (64-row, 27B dims) + logits + misc | ~30 MB | §6 sizing |
| **total** | **≈ 7.28 GB + V shards** | V = floor((10.8−7.28)/0.381) = 9 |

RAM (13.5 GB): 31 layer shards pinned = 11.8 GB + small bf16 params ~65 MB (permanently pinned,
loader-gaps §3.3) + IOCP in-flight blocks (~0.5 GB) ≈ 12.4 GB warm. Generation mode needs **no**
activation ring in RAM (ping-pong lives in VRAM). NVMe: remaining 24 shards = 9.2 GB, read once
per sweep through 2 pinned slots.

### 2.2 The "activation ring" is a ping-pong, not a 64-deep ring

The mission framing ("keep h_l in a pinned RAM ring, 10.5 MB/layer × 64 = 671 MB") assumes the
full per-layer history must exist simultaneously. It must not:

- Generation needs only h_{l-1} → h_l: **two buffers of [S,5120] f32 = 21 MB at S=512, 168 MB at
  S=4096**, and they live in VRAM because every layer computes on the GPU (weights stream TO the
  GPU; activations never leave).
- The MTP KV pass and NLL need only the **final** layer output h_final ([S,5120], 10 MB).
- Even the parity dump streams: `dump_pf.cu` fwrites each seam incrementally
  (`src/dump_pf.cu:11-15`) — the 671 MB (512 tok) / 5.4 GB (4096 tok) lives **on disk**, never in
  RAM. The full-history pinned ring is needed only if activations are hosted in RAM (a CPU-tier
  compute variant) or if someone demands all-seams-in-RAM tooling. Budget if so: 64·S·20 KB —
  671 MB at S=512, 5.4 GB at S=4096 (the mission's violation case, avoided by not building it).

### 2.3 Heterogeneous-boundary PCIe math (mission item 3, stated clearly)

IF a layer boundary crosses devices (h ring in pinned RAM, GPU reads/writes via UVA): per
boundary, 10.5 MB × 2 (r+w) at ~22 GB/s = **0.95 ms**; × 64 boundaries = **61 ms per 512-token
prefill** — 3 % of a 2 s sweep, fine. For **decode** (the per-token regime): 20 KB × 2 × 64 = 2.6
MB/token over 22 GB/s = **~119 µs/token** — under 0.01 % of the ~1.5 s/token tiered decode, also
fine. Conclusion: crossing PCIe at layer boundaries is affordable in both regimes, so a future
CPU-compute tier does not need a new activation architecture; but since prefill/decode compute is
all-GPU here, keep h in VRAM and pay zero. bf16 activations would halve both numbers; prefer f32
for parity simplicity (§3) while VRAM allows (168 MB worst case at S=4096).

### 2.4 Staging-slot pipeline (why the 384 MB VRAM slot matters)

Alternative for RAM-tier layers — run the GEMM reading weights zero-copy over PCIe — re-reads the
380 MB layer **per tile**: 8 × 17.4 ms = 139 ms/layer. Staging into the VRAM slot once costs
17.4 ms total, then all 8 tiles read from VRAM at ~500 GB/s (~0.85 ms/tile). The slot converts
per-tile PCIe reads into a one-time H2D. Same logic for NVMe layers (59.1 ms fill dominates).
Double-buffered slots + the loader-gaps IOCP reader prefetching one layer ahead give steady-state
per-layer cost = max(NVMe 59.1, PCIe 17.4, compute ~6.8) per tier.

---

## 3. Numerics

- GEMM path = `fp8_gemm` (`src/fp8.cu:101-184`): A f32→bf16, B e4m3 × bf16 128×128 block scale
  (`weight_scale_inv`, multiply — loader-gaps §5/§6 verified) dequant to bf16, wmma bf16 with f32
  accumulate. Requires rows%32==0, cols%128==0, **T ≤ 64 with x16 zero-padded to 64 rows**
  (`fp8.cu:182`, padding done at `decode.cu:37` — keep per tail-tile). All 27B shapes satisfy the
  mod constraints (5120, 6144, 10240, 12288, 17408, 1024).
- Decode path (`fp8_gemv`, `fp8.cu:14-55`) is f32-exact activations against f32 dequant partials.
  Prefill therefore matches decode to bf16 rounding of A and scaled-B — the same relationship the
  9B has between `mxfp4_gemv_v2` and `mxfp4_gemm_mlx`, where layer parity (cos ≥ 0.9999) already
  carries. Store h in f32 (recommended) so layer boundaries add nothing; a bf16 h adds one
  2⁻⁹-relative rounding per boundary (standard practice, but unnecessary while VRAM allows).
- Keep the tail-tile zero-padding memset (rows ≥ T must read as 0 in `pf_bf16`) and all
  elementwise/convenience kernels driven by the true tail T (they already handle T < 64 — that is
  how 9B prompt tails work).
- 27B-specific dtype traps that prefill inherits: A_log is **bf16** (needs the bf16 params
  variant, loader-gaps §4.2), in_proj_a/b are bf16 [48,5120] (bf16 GEMM or per-token gemv),
  mtp.fc bf16 GEMM [5120,10240].

---

## 4. MTP KV prefill (mission item 5)

Problem: the first spec step's draft d1 attends MTP slots 0..P-1, of which only slot P-1 is
written by d1 itself — slots 0..P-2 are uninitialized cudaMalloc garbage today (graph-hazards /
spec-deepen F7; the reference fills them densely teacher-forced). Fixes:

**Recommended — explicit post-prefill MTP KV pass (option a).** After the main sweep, with
h_final [P,5120] retained (raw residual, pre-final-norm — that is what `mtp_layer` norms):
for each prompt position p in 0..P-2, MTP slot p's K/V are computed from
`fc(concat(norm_e(embed(t_{p+1})), norm_h(h^main_p)))` → q/k norm + rope at pos p → store_kv into
`mtp_keys/mtp_values`. Verified semantics: engine writes slot s from (embed(t_{s+1}), h^main_s)
(`decode.cu:133-170` consumes `token_dev`=pending + `x_.hidden`; `spec_prologue` sets
`pos[7]=pos[0]-1`, `prefill.cu:277`); the reference reaches the same pairing
(`reference_multistep.py`: `mtp_draft(am, x, step)` — "shifted pairing", which overwrites the
same slot the earlier same-step call wrote). Crucially the slots are **independent** — the MTP
residual chain is only needed to *propose* tokens, never to fill other slots' KV — so this is a
batched, tile-parallel pass over 64-row tiles: bf16-embed gather of tokens t_1..t_{P-1} (all
known prompt tokens — **no lm_head/argmax dependency**), two rmsnorms, concat, fc GEMM (bf16
[5120,10240], new trivial GEMM), q/k/v projections + norms + rope at pos = tile base,
`store_kv_batch` into the mtp caches. **Skip gate/o_proj/MLP entirely** — only K/V are consumed
later. Weights touched ≈ fc 105 MB + qkv 143 MB fp8, all VRAM-pinned (mtp shard): ~8 tiles ×
0.4 ms ≈ **3-5 ms** per turn. Multi-turn: rerun for the new positions each turn (h_final of that
turn, embeds of that turn's tokens; the turn's last slot is again covered by the next draft).

Option (b) — lazy batch-fill inside the first spec step — saves nothing (the fill must complete
before draft d1's attention anyway), complicates the captured spec graph with a one-shot branch,
and mixes prefill-rate work into a decode step. Rejected.

Residual defect (flag, out of scope): F7's *second* half — accepted spec steps skip MTP slots,
leaving holes during generation — is a decode-side bookkeeping fix (spec-deepen), not a prefill
issue; fixing the prompt fill should already lift acceptance p substantially.

---

## 5. Parity / seam instrumentation contract (mission item 6)

- `prefill_chunk_seam` (`decode.cu:113-117`) fires per layer per chunk with pf_x [T≤64, 4096].
  New `prefill_layer_stationary_seam` fires **per layer with the full h [S,5120]** after the
  layer's last tile (one sync per layer, 64 per sweep — parity mode only). Same information as
  the chunked dumps (seam content = h after layer l for every token), full-T instead of 64 rows.
- `dump_pf.cu`: drop the `T ≤ 64` argv cap (`dump_pf.cu:28`) → S up to the super-chunk cap;
  layout becomes **[65, S, 5120] f32** (64 layer seams + final model.norm seam, seam-major).
  File size 345 MB at S=512 (vs 34.6 MB today) — streamed per seam, no RAM ring (§2.2).
- The reference scripts are already **token-major** (`reference_pf_i4.py:79-90`: loops tokens,
  layers inside, carrying per-layer delta state and per-attn-layer KV lists) — the exact
  computational inverse of the engine's new loop order; semantics unchanged. The 27B adaptation
  is a new script, spec'd in §9.
- `prefill_chunk_device(T≤64)` itself must remain byte-identical: it is the spec verify path
  captured inside CUDA graphs (`decode.cu:218,237`). Layer-stationary prefill is additive.

---

## 6. Refactor TODO list (ordered)

1. **`prefill_layer_stationary(const int *tokens, int T_total)`** in `decode.cu` (+ `_seam`
   variant). Outer loop layers 0..63, inner tiles of 64. Check `turn_base + T_total ≤
   max_context` once (replaces the per-chunk guard at `decode.cu:45`). Parameterize the tile body
   with in/out residual pointers (`h_prev + tile*5120*64`, `h_next + …`); all `pf_*` intermediates
   stay 64-row workspace. lm_head epilogue: last-token argmax (existing `decode.cu:99-100`
   pattern) or NLL tiling (`generate.cu:69-89` pattern, 64-row logits buffer per tile).
2. **Position discipline** — THE classic reorder bug: today `addi(pos_dev, T)` runs per chunk
   (`decode.cu:103`); layer-major must advance position **once per turn**. Per (layer, tile), set
   `pos_dev = turn_base + 64*tile` before `qk_norm_rope_batch`/`store_kv_batch`/`gqa_prefill`
   (all read `pos_dev[0]`, e.g. `prefill.cu:73,91,104`). Cheapest correct: 4-byte
   `cudaMemcpyAsync` per (layer,tile) ≈ 512 × ~1.5 µs ≈ 0.8 ms/sweep; optimize later with a
   precomputed device array of tile bases + kernel arg.
3. **Activation buffers**: allocate `h_A/h_B [S_cap, 5120] f32` + `h_final` in the workspace
   (VRAM). S_cap = ctx or a policy cap (§7). Embed gather writes h_A directly.
4. **Kernels — new**: bf16 batch embed gather for T rows (27B embed is bf16; 9B MXFP4 gathers
   exist at `prefill.cu:9-40`); bf16 GEMM (fc, in_proj_a/b, lm_head prefill — "trivial wmma" per
   loader-gaps); `mtp_prefill_kv` pass (§4).
5. **Kernels — 27B shape retunes** (all in `src/prefill.cu`): gqa grid 16→24 heads, kvh→head/6,
   score layout unchanged; `qk_norm_rope_batch` grid (20,T)→(28,T), isq head<24, k row stride 4
   kv heads; `split_q_gate_batch` 8192→12288, 16→24 heads; `deltanet_prefill` 32→48 blocks,
   kh→head/3, qkv stride 10240, a/b 48; `conv_prefill_silu` 8192→10240; `gated_rmsnorm` 32→48;
   `params_batch` A_log bf16 variant; workspace allocs (`decode.cu:14-27`) to 27B dims
   (pf_qkv 64×10240, pf_gate/up 64×17408, pf_bf16 64×17408×2 B, delta_state 48×48×128×128,
   conv 48×10240×3, kv 16 slots, … ≈ 27 MB total tile scratch).
6. **Pass `row0_snap = nullptr`** for conv/deltanet during turn prefill (saves 1.2 GB/sweep of
   pointless snapshot writes; legality verified `prefill.cu:199,258`).
7. **Storage**: TieredStorage2 per loader-gaps §3.3 — shard-major pinned slots, IOCP reader,
   double-buffered 384 MB VRAM staging slots, prefetch one layer ahead during the previous
   layer's tiles, permanent pin for lm_head/embed/mtp + bf16 smalls; prefill acquires each layer's
   slot once per sweep and releases after its last tile. Placement manifest as data (V/M/N).
8. **MTP KV fill** (§4) wired into the turn driver after the main sweep, before `capture_spec`/
   first `spec_step`; per-turn for multi-turn.
9. **Tools**: `dump_pf.cu` S-cap + new layout (§5); `tools/reference_pf_f8.py` (§9);
   `generate.cu`/`nll.cu` drivers switch their prompt loops from 64-chunk iteration to one
   `prefill_layer_stationary` call (super-chunk loop if S_cap < turn length, §7).
10. **Graph hygiene**: layer-stationary prefill is host-orchestrated (acquire/release, syncs) and
    must never run inside a captured graph; capture_spec still wraps only the T=2 pair path.

---

## 7. Performance arithmetic (mission items 4 & 8)

### 7.1 Compute (recomputed — mission's per-layer FLOPs undercounted)

2·T·R·C per GEMM at T=512: qkv 53.7 GF, z 32.2, out_proj 32.2, gate/up/down 3× 91.2 →
**391.7 GF per linear layer**; attn layer: q 64.4 + k 5.4 + v 5.4 + o 32.2 + MLP 273.6 =
**381 GF**. Total ≈ **24.9 TF** for a 512-token sweep (mission said ~12.8 TF; it dropped z/out
and two of three MLP GEMMs). At the achievable ~50-60 TF/s (bf16 wmma + dequant, 4070S) ≈
**0.42-0.50 s of GEMM per sweep**. Per 64-token tile per layer the GEMM is bandwidth-bound:
380 MB fp8 from VRAM at ~450 GB/s ≈ 0.85 ms → ~6.8 ms/layer per 512-token sweep. Deltanet scan
(serial, state in smem): ~0.5-0.9 ms/layer/sweep → 25-45 ms total. Causal attention ≈ 1 ms/sweep
(grows S², 66 ms at S=4096 — still minor). **Total compute ≈ 0.5 s per 512 tokens, almost fully
hideable under IO.**

### 7.2 Per-tier sweep cost (380 MB avg layer; NVMe 6.5 GB/s, PCIe 22 GB/s)

| tier path | ms/layer | exposed (pipelined) |
|---|---|---|
| NVMe → pinned → VRAM slot | 59.1 + 17.4 | **59.1** (PCIe & compute overlap the next fill) |
| RAM pinned → VRAM slot | 17.4 | **17.4** (compute 6.8 hides under it) |
| VRAM resident | 0 | **~6.8** compute (overlaps NVMe stream of other layers) |

Sweep time T_sweep(S) ≈ 59.1·N + 17.4·M + 6.8·V·(S/512) → for the §2.1 placement (V=9, M=31,
N=24): 24×59.1 + 31×17.4 + 9×6.8 ≈ **2.02 s per 512-token sweep** (cold first sweep, when the
M-tier pinned copies aren't filled yet, degenerates to all-NVMe ≈ 3.8 s). Mission's checkpoints
confirmed: 59 ms/layer ✓; "3.8 s" is the all-NVMe 64-layer case (64×59.1), not 20 layers
(20×59.1 = 1.18 s); "L×0.76" is the decode-per-token figure — prefill VRAM-layer compute is
6.8 ms/layer/sweep, and the total ≈ one weight sweep + ~0.5 s compute ≈ **2.0-2.5 s for 512
tokens** (2.0-4.0 s across placements), versus chunk-major 8 sweeps = **16.2 s** (same
placement) / 30.2 s (all-NVMe; the mission's 31 s) — the 8× on the IO term. Sensitivities:
NVMe→RAM move saves 41.7 ms/sweep per layer; RAM→VRAM saves only 10.6 ms — prefill time is
N-dominated, so V/M boundary placement barely matters.

### 7.3 Super-chunk math (mission item 8)

Time for L tokens in super-chunks of S: **T(L,S) = ceil(L/S) · T_sweep(S)**, with T_sweep(S) =
C_io + 0.0133·V·S ms (C_io = 59.1N + 17.4M = 1958 ms here) — plus S²-attention, negligible.
Since C_io » compute for all S ≤ 4096, T is monotone decreasing in S: **make S as large as the
activation budget allows.** With VRAM ping-pong (this design), the budget is 2·S·20 KB — S=4096
costs 168 MB f32 — so **S = whole turn (single sweep) up to ctx**; super-chunking is unnecessary
for generation. The mission's ring premise (671 MB @ 512, 5.4 GB @ 4096 busts RAM) is real only
for a full-history RAM ring, which §2.2 eliminates. Corrected curve for L=4096 (placement above):

| S | sweeps | ring (if built) | total |
|---|---|---|---|
| 4096 | 1 | 5.37 GB (RAM — violates) / 168 MB (VRAM ping-pong ✓) | **2.03 s** |
| 2048 | 2 | 2.7 GB / 84 MB | 4.06 s |
| 1024 | 4 | 1.34 GB / 42 MB | 8.12 s |
| 512 | 8 | 671 MB / 21 MB | 16.2 s |

(The mission's "25.65 GB per 1024 tokens = 24.8 tok/s" is a slip: 25.65 GB @ 6.5 GB/s = 3.95 s
per sweep → 259 tok/s at S=1024 unchunked-tier; its "131 tok/s" matches S=512.)
**Policy: generation S = min(turn_len, ctx, ping-pong cap ~4096); parity mode S = 512**
(matches existing dump tooling, 671 MB streamed to disk). If a host-ring variant is ever built,
R = 2 GB → S_max = R/(64·20 KB) ≈ 1600 tokens, and the loss vs S=4096 is 3.0 s vs 2.0 s for a
4096-token prompt — the curve is flat above S ≈ 1000 because C_io dominates.

### 7.4 Misc costs

Embed gather 512 rows (5.2 MB bf16): ~15 µs from VRAM, ~240 µs pinned zero-copy — trivial either
way. lm_head last token 5.1 ms. MTP KV pass ~3-5 ms (§4). Tail syncs/launches: ~500 kernel
launches per tile × 8 tiles… launch overhead ≈ 64 layers × ~20 launches × 8 tiles × ~1.5 µs ≈
15 ms/sweep — absorbable, later fusable per insig4-perf backlog.

---

## 8. Multi-turn growth (mission item 7)

Turn k prefills only its new tokens through the same layer-stationary sweep: KV appends at
absolute positions (`store_kv_batch`, `prefill.cu:88-98`), gqa reads the full per-layer cache
across all turns (Q1), delta/conv state simply continues (they are position-indexed-free
recurrences), the activation ping-pong holds only the current turn's rows, h_final is that turn's
last layer output, and `mtp_prefill_kv` fills the new MTP slots. Position = cumulative
`x_.position` (set per turn, bumped once). No per-turn re-read of old tokens' weights — each
sweep touches weights exactly once regardless of history.

---

## 9. Spec: `tools/reference_pf_f8.py` (layer-major 27B parity)

Clone of `reference_pf_i4.py` with: (1) multi-shard INSIDX02 loading via the index, names
`model.language_model.layers.N.*` / bare `lm_head` / `mtp.*`; (2) dequant
`W = e4m3(weight) × weight_scale_inv` with bf16 [ceil(r/128), ceil(c/128)] scales
(np `repeat_interleave(128)` both axes — multiply, not divide); (3) 27B dims: hidden 5120, 64
layers (attn iff N%4==3), qkv split [2048,2048,6144] of 10240, a/b/heads 48, conv 10240×4,
A_log/dt_bias bf16, kvh = head//6 for attn and head//3 k-sharing for deltanet, partial rope 64 /
theta 1e7, 1/16 score scale; (4) native dump layout **[65, S, 5120] f32** seam-major (§5),
compared token-major exactly as today (per-token layer loop, per-layer delta state + KV lists,
worst-cosine report); (5) embed/lm_head/fc/a/b bf16 paths. Acceptance: worst seam cos ≥ 0.9999
vs the f64/f32 reference before any perf work (AGENTS.md rule).

## 10. Failure modes & policies

1. **Context overflow**: `turn_base + T_total > max_context` throws once per turn; policy cap
   prompt + max_new ≤ ctx (2048/4096; smem `score[4096]` is the hard ceiling). Decode-side
   KV-full-via-graph-replay (graph-hazards #2) is separate — enforce admission at turn start.
2. **Position drift**: any per-layer/per-tile `addi` is a correctness bug (would advance pos 8×64
   times per turn). One bump per turn (§6.2). Host mirror `x_.position` updated once.
3. **Tail tiles**: zero-pad `pf_bf16` rows ≥ T before every tail GEMM (`decode.cu:37` pattern);
   all reductions/grids use true T.
4. **VRAM pressure**: staging slots (768 MB) transient; if tight, single slot (+59 ms/sweep
   serialization) or bf16 KV (−268 MB). Ring never required (§2.2).
5. **RAM backpressure**: manifest-parse-time check that pinned M-tier + slots + in-flight ≤ 13.5
   GB (loader-gaps §3.3 rule) — never fails mid-sweep.
6. **Graph/stroage hazards**: prefill outside graphs; lm_head/embed/mtp permanently pinned so
   transient staging LRU can't free pointers the spec graph baked; spec pair path untouched.
7. **Cold first sweep** ≈ 3.8 s (M-tier pinned fill from NVMe); warm ≈ 2.0 s. Optional warmup
   sweep at load.
8. **MTP off-by-one**: fill slots 0..P-2 with (embed(t_{p+1}), h_p) — t_1..t_{P-1} are prompt
   tokens; slot P-1 belongs to the first draft. Wrong shift = silently depressed acceptance.
9. **Seam-mode cost**: 64 host syncs per sweep — parity builds only.

---

## 11. TL;DR

1. Layer-major weight-stationary prefill is a pure loop reorder of the verified 9B chunk path:
   cross-tile KV (Q1), conv carry (Q2), deltanet carry (Q3) all confirmed in `src/prefill.cu`.
2. Each layer's weights stream NVMe→pinned→VRAM staging slot once per sweep; 8 T-tiles then
   compute from VRAM (staging beats zero-copy per-tile PCIe reads 8:1 for RAM layers).
3. Activations are a VRAM ping-pong (2×S×20 KB — 21 MB @ 512), not a RAM ring; even parity dumps
   stream to disk. The 5.4 GB @ 4096-token RAM violation evaporates.
4. 512-token prefill ≈ one weight sweep + ~0.5 s compute ≈ 2.0-2.5 s on (V9/M31/N24) vs 16-31 s
   chunk-major — the 8× IO win; ~25 GF of GEMM (recomputed) is noise under 25 GB of IO.
5. Super-chunk curve T = ceil(L/S)·(59.1N + 17.4M + 0.0133·V·S ms): monotone in S — generation
   uses S = whole turn (≤ ctx); parity keeps S = 512 tool compatibility.
6. MTP KV: batched post-prefill pass over slots 0..P-2 using (embed(t_{p+1}), h^main_p) —
   slots independent, no residual chaining, ~4 ms, weights already VRAM; done before spec starts.
7. Refactor is additive: `prefill_layer_stationary` + per-tile in/out pointers + one pos bump per
   turn; `prefill_chunk_device(T=2)` stays byte-identical for the captured spec graph.
8. Numerics identical to decode's bf16-GEMM relationship; f32 ping-pong keeps boundaries free.
9. `reference_pf_f8.py` = token-major reference with fp8 dequant + 27B dims + [65,S,5120] dumps.
10. Failure modes bounded: ctx admission per turn, position discipline, tail padding, pinned
    budgets checked at manifest parse, permanent pins for graph-baked tensors.
