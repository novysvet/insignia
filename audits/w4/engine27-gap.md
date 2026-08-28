# engine27-gap — the file-by-file work order for closing the 27B tiered engine

Date 2026-08-25. Audit agent: w4/engine27-gap. Read-only except this file.
Ground truth: line-numbered reads of the live tree + `audits/w3/MASTER-PLAN.md`
(phases A–G) + the checkpoint headers themselves (every shape below was read out
of `Qwen3.8-27B-FP8/*.safetensors` this session, not copied from an audit).

---

## 0. What actually exists today (verified, with the mission-state corrections)

Exists and works:
- `tools/mk.py` — build driver, mtime cache, `ENGINE27` closure (mk.py:27-29) and
  FUTURE targets `generate27/nll27/dump-layers27` (mk.py:76-80).
- `tools/index27.py` — **INSIDX02 built**: `build/qwen38-27b-fp8.insignia-index`
  (119,017 B, 66 shards, 407 F8+scales links, crc32-verified, checkpoint names
  already remapped to the ENGINE convention `language_model.model.layers.N.*`,
  `language_model.lm_head`, `language_model.mtp.*` — index27.py:70-88). So
  master-plan Phase A item 5 (name remap) is **already done at index time**.
- `src/fp8.cu` — `fp8_gemv` / `fp8_gemv2` / `fp8_gemm` / `bf16_get_row`, tested by
  `src/test_fp8.cu` (bias-7 host decoder already fixed, test_fp8.cu:11-16).
- `include/insignia_cpu.hpp` (995 lines) + `src/test_cpu.cpp` — CPU tier complete
  and unit-tested vs f64 references: `fp8_gemv_mt/fp8_gemv2_mt/bf16_gemv_mt`,
  `rmsnorm_cpu` (zero-center flag), `gated_rmsnorm_per_head_cpu`,
  `causal_conv4_silu_cpu`, `deltanet_parameters_cpu`, `deltanet_step_cpu`
  (heads=48, kshare=3 defaults!), `qk_norm_rope_cpu` (24/4 defaults),
  `split_q_gate_cpu`, `store_kv_cpu`, `gqa_decode_cpu` (24 q / 4 kv, kvh=h/6,
  online-softmax, f32+bf16 KV), `CpuPool` (6 workers pinned LP 0-5).
- `src/streaming.cu` + `include/insignia_streaming.hpp` — NvmeReader (IOCP,
  NO_BUFFERING+OVERLAPPED, 2 MiB blocks, QD16, twin tails), PinnedRing
  (VirtualAlloc + cudaHostRegister, atomic FREE/FILLING/READY/IN_USE),
  LayerFeeder (`begin_epoch/acquire_layer/release_layer/map`, read-ahead =
  slots−1). Mission said "untested": **smoke was run and passed** (byte-exact vs
  buffered read, epoch re-arm, teardown-with-inflight; streaming-impl.md §1).
- `src/io_bench.cu` + `audits/w3/io-bench-results.md` — **run**: E: 3.22–3.25 GB/s
  at engine config (bar ≥ 3.0 passed), C: 6.5, dual E:+C: 9.7 additive.
- `tools/reference27.py` — NumPy ground truth for all 64 layers + NLL + greedy
  (layer/seams/nll/greedy/enc/dec). Zero-center (1+w) norms, `A_log` bf16,
  GQA h/6, k-head j/3, conv 10240, q-fold 1/√128 — all encoded. **Gap: no MTP
  subcommand** (the 9B `reference_multistep.py` pattern is not ported).
- Phase-0 fixes landed: v21 cp.async tail wait (gemm.cu:274-276), qk-norm `nsc`
  race fix (ops.cu:9, prefill.cu:62), gqa `red[8]/smx/sden` slots
  (attention.cu:7, prefill.cu:107), test-fp8 bias-7. Still open: Phase 0 C5
  (`DecodeWorkspace::~DecodeWorkspace` at decode.cu:29 lacks a first-line
  `cudaStreamSynchronize` — copy it into the 27B workspace class).

Missing (the actual gap this report orders):
1. `src/model_file.cpp:23` accepts **only INSIDX01** (`magic INSIDX01, version 1`,
   single payload file). The built 27B index is unreadable by the engine. This
   blocks every 27B code path that touches a weight.
2. No 27B decode loop, no tiered storage v2, no placement manifest, no
   generate27/nll27/dump_layers27, no bf16 GEMM, no 27B batch-kernel geometries.
3. `Qwen35Weights::matrix()` (qwen35.cu:7-30) fp8/bf16 branches exist **but**
   acquire through the old `TieredStorage` (storage.cu:9) which always
   cudaMalloc+cudaMemcpy H2D from the mmap — wrong for UVA/pinned/CPU tiers.
   `Qwen35Weights::embed_dev` (qwen35.cu:33) is mxfp4-only (27B embed is bf16).

---

## 1. Shape strategy decision — RECOMMEND: (c) new `src/decode27.cu` + `include/insignia_decode27.hpp`, with shared kernels templated in place

The three options, priced against the real code:

**(a) Template the decode class on shape (`Qwen35Shape` as template param).**
decode.cu has 84 shape-literal occurrences (counted this session; per-file:
decode.cu 84, prefill.cu 29, ops/attention/deltanet/qwen_kernels 9, generate/nll
4, dump tools 90 — 222 total, the rest of the master plan's ~450 estimate is
kernel-internal geometry: grids, `head>>2`, `kh=head>>1`, `h*512` strides,
`t*8192` strides, `i<1024` guards). A template would need to propagate into the
kernels anyway (they own the geometry), so (a) is really "(c) plus a second
instantiation of the 9B loop". Cost: doubles compile of the loop, and — decisive
— the 27B loop is **structurally different**, not just differently-constant:
tier dispatch per layer (V/Z/N/C), no CUDA graphs (master plan §2.4/D.4), embed
row pread + copy stream, CPU-tier hidden handoffs, weight-stationary layer-major
prefill (Phase F) vs the 9B token-major `prefill_chunk_device`. Those differences
would live in the template as `if constexpr` branches the 9B side never executes —
pure complexity in the file that must stay frozen for the 9B parity hunt.

**(b) Runtime `Qwen38Shape` struct threaded through.** Kills constexpr folding of
grids/strides (launch configs become runtime), forces `score[4096]`-class static
smem arrays (attention.cu:7, prefill.cu:106,107) into dynamic allocation, and
adds a struct parameter to ~30 kernels that today take none. attn-27b §1 already
rejected runtime QH for the same reason ("emits a real hardware divide...
compile-time QH lets the compiler fold the group division"). Also touches all 84
decode.cu sites — same edit count as (c) but inside the frozen 9B file.

**(c) New `src/decode27.cu` duplicating the loop, 27B-specialized. — CHOOSE THIS.**
Justification:
- **AGENTS.md constitution**: "keep the engine as specialized as possible",
  "sacrifice extensibility". Two single-shape engines beat one parameterized one.
- **The 9B file is a regression asset, not a liability**: master plan §3 requires
  "every phase ends with the 9B regression suite still green", and the 9B engine
  still has an unresolved full-attn parity issue (AGENTS.md:102-105). In-place
  edits (a/b) churn the exact file under investigation.
- **The duplication is cheap and mostly divergent anyway**: decode.cu is 268
  lines; the 27B twin is a copy of `delta_layer/attention_layer/mtp_layer` bodies
  with 84 constant edits + fp8/bf16 dispatch in `linear*` + acquire/release
  plumbing — and it *drops* `capture_step/capture_spec/step/spec_graph_step`
  (decode.cu:238-266, no graphs at 27B) and *replaces* `prefill_chunk_device`
  with the stationary variant. Net shared-with-9B code after edits: ~30%.
- **Honest accounting vs "~450-site edit" (Phase B)**: that estimate prices
  editing in place. Under (c) the ~150 kernel-side geometry sites still must be
  edited — but in the **shared kernel files** (ops/attention/deltanet/
  qwen_kernels/prefill), where attn-27b §1 already designed them as
  `template<int QH>` with `<16>` and `<24>` coexisting (9B tests keep running
  unchanged). The loop-side 84 sites land once, in the new file, with zero 9B
  risk. Same total edit count, half the risk surface.
- Keep ONE `DecodeWorkspace27` (new class; do not resize the 9B one — the 9B
  graphs bake buffer pointers, decode.cu:239-266).

**Kernel-side policy that makes (c) cheap** (from attn-27b §1, verified against
current sources): template on `QH` where the delta is a head count
(`qk_norm_rope`, `gqa_decode`, `split_q_gate*`, `expand_gate_heads`,
`gqa_prefill`); plain constant edits where the delta is a stride/channels
(conv 8192→10240, `t*32`→`t*48`); `head>>2`→`head/6` and `kh=head>>1`→`head/3`
are **semantic** changes (mis-grouping reads the next token's KV rows — master
plan risk #2/#3), guarded by new unit tests, not just constant swaps.

### 1.1 `DecodeWorkspace27` sizing (master plan B.2 trap list, priced)

9B ctor is decode.cu:11-27 (one alloc line at :14, pf line :22-24). 27B table —
every number cross-checked against shard shapes (§0) and the B.2 list:

| buffer | 9B | 27B | note / trap |
|---|---|---|---|
| hidden, norm | 4096 | **5120** | |
| qkv | 8192 | **10240** | conv channels; q\|k\|v = 2048\|2048\|6144 f32 |
| attn_gate, core, z | 4096 | **6144** | z = in_proj_z rows 48·128; gate/core = 24·256 |
| key, value | 1024 | 1024 | 4 kvh × 256 — unchanged |
| a, b | 32 | **48** | |
| gate, up | 12288 | **17408** | |
| down | 4096 | **5120** | |
| logits | 2·248320 | 2·248320 | vocab identical |
| delta_state | 24·32·128·128 | **48·48·128·128 = 144 MiB** | snap_delta same |
| conv_state | 24·8192·3 | **48·10240·3 = 5.6 MiB** | snap_conv same |
| kv_keys/values | 8·ctx·1024 | **16·ctx·1024** (f32) | 256 MiB @ctx2048; 16 full layers |
| mtp_keys/values | ctx·1024 | ctx·1024 | MTP layer is 24q/4kv like any full layer |
| committed / host_committed | 16384 | 16384 | ≥ ctx; fine |
| pf_x, pf_n, pf_down | 64·4096 | **64·5120** | |
| pf_qkv | 64·8192 | **64·10240** | |
| pf_scratch | 64·8192 | **64·12288** | **q_proj rows**, NOT conv's 10240 (B.2 trap) |
| pf_z, pf_q, pf_g, pf_core | 64·4096 | **64·6144** | |
| pf_k, pf_v | 64·1024 | 64·1024 | |
| pf_gate, pf_up | 64·12288 | **64·17408** | |
| pf_a, pf_b | 64·32 | **64·48** | |
| pf_bf16 | 64·12288·2 B | **64·17408·2 B** | fp8_gemm A-staging (largest cols = down_proj 17408) |
| pf_xq8, pf_xs8 | 6144 u32 / 768 f | **drop** | mxfp4 q8-pair staging only; 27B is fp8/f32-in |
| pos_dev block | 16 ints | 16 ints | layout comment prefill.cu:276 |

Workspace VRAM total ≈ 598 MiB (states 150 + snaps 150 + pf 24 + KV 256+16 +
logits 2 + vecs 0.3) — matches the master-plan ledger (§2.2 fixed block "decode
ws ~40 + snap 151+5.9 + pf ~27 + KV 268"). v1 keeps snaps in VRAM; D=4 later
moves the T−1 snapshots host-pinned (628 MB, master plan F.3).
Constructor must keep the C5 fix: `cudaStreamSynchronize(stream)` first line of
the dtor. `max_context` clamp stays ≤4096 (score[4096], attention.cu:7).

---

## 2. Tier dispatch + storage: `Qwen38Weights27` / `TieredStorage27` design

### 2.1 Why not extend the existing classes

`TieredStorage::acquire` (storage.cu:9) is a VRAM LRU cache: cudaMalloc +
cudaMemcpy from the mmapped host tensor, sync after every upload. At 27B:
(a) it would copy every streamed layer over PCIe twice (NVMe→page→VRAM) — the
master plan v1 consumes ring slots **in place via UVA** (§2.4 v1: "GPU consumes
via UVA zero-copy reads"); (b) it has no notion of a shard slot; (c) LRU
eviction + CUDA graphs are explicitly dead at 27B. Keep `TieredStorage` for the
9B binary; write a new `TieredStorage27` (per loader-gaps §3.3 contract) that
**owns** the LayerFeeder, the pinned/locked arenas, and the VRAM arena.

### 2.2 The layer handle — one struct, four lives

The INSIDX02 tensor table gives per-tensor `{shard, abs_off, bytes, dtype,
shape}`. For a whole-shard read from byte 0 (verified gapless; pcie-pipeline
§7.2 "shard contiguity confirmed"), **in_slot_off == abs_off** (streaming-impl
§4). So one resolved layer is:

```cpp
struct LayerHandle {           // one per (layer, tier) — resolved ONCE at startup
    const u8* base;            // V: VRAM arena ptr | Z: pinned ptr (UVA)
                               // N: slot base + abs_off (rebuilt per epoch)
                               // C: VirtualLock'd pageable ptr
    Tier tier;                 // vram | pinned | cpu | ring
    int  epoch_index;          // ring layers only: index into the epoch plan list
};
// per-tensor pointer inside a layer: base + t.abs_off  (zero remapping)
```

`matrix27(base_name)` then becomes: look up the layer handle by tensor shard
(INSIDX02 carries `shard` per tensor), return `QuantMatrix{weight=base+off,
scales=base+off, rows, cols, kind}` — **no per-tensor acquire/release at all**
for V/Z/C (weights static for the process lifetime; AGENTS "modify constant data
directly"). Only ring layers pay a per-layer acquire/release, and that is the
LayerFeeder's existing `acquire_layer/release_layer` (streaming.cu:430-450),
called once per layer per step by the decode loop — the "pin" granularity is the
shard, which loader-gaps §3.3 specified ("granularity = ONE SHARD (§7)").

### 2.3 Who copies what, when (the acquire contract)

Startup (once, ordered — failure = refuse to start):
1. Parse manifest (below) → tier_of[64], ring config, ctx/kv dtype.
2. Probe ladder: `cudaMemGetInfo` (VRAM budget for L), pinned-cap ladder
   (cudaHostAlloc 1 GiB steps until fail → Z budget; master plan D.3).
3. `lm_head` (2.47 GiB bf16, mandatory VRAM): read_once via NvmeReader's
   buffered twin / plain ReadFile into a pinned staging buffer → cudaMemcpy H2D
   → VRAM arena. Refuse if free VRAM < lm_head+MTP+workspace+L-layer bytes.
4. BF16 smalls arena (~65 MB all 64 layers: norms, q/k_norm [256], A_log/dt_bias
   [48], conv1d [10240,1,4], a/b [48,5120]): one `cudaHostAlloc` block,
   read_once filled, **permanently pinned** (loader-gaps §3.3 normative rule) —
   these are read by GPU kernels via UVA and by the CPU tier directly, and must
   survive independent of ring-slot turnover.
5. V layers 0..L−1: read_once whole shard → pinned staging (reuse one ring slot
   as the staging buffer) → cudaMemcpyAsync H2D into a contiguous VRAM arena;
   layered in layer order; MTP shard (477 MB) same path.
6. Z layers (v1.5/v2): read_once into `cudaHostAlloc`'d per-layer regions (or
   one big region + registration; plan ≤8,048 MB pending probe).
7. C layers (v2): read_once into `VirtualAlloc+VirtualLock` pageable regions
   (does NOT count against the WDDM cap — master plan §2.3 rule 2).
8. N layers: build `std::vector<ReadPlan>` — one plan per N shard, requests =
   per-tensor extents in shard order **with the 8-byte pad inserted at the
   BF16→F8 boundary** (see 2.5). Plans owned forever (paths are borrowed,
   streaming.hpp:55). Per step: `feeder.begin_epoch(n_plans)` — the prior epoch
   is fully released by construction (decode is layer-serial), so re-begin is
   legal each step (streaming.cu:374-392).

Per-token-step hot path (who copies what):
- **V layer**: nobody copies anything. Kernels launch on static VRAM pointers.
- **Z layer**: nobody copies. Same kernels launch on pinned host pointers; the
  GPU reads rows over PCIe via UVA zero-copy (18 GB/s plan rate).
- **N layer**: `acquire_layer(e)` returns the pinned slot; kernels read it via
  UVA exactly like Z; `release_layer(e)` after the layer's kernels are done.
  Release timing: v1 enqueues an event after the layer's last kernel and treats
  `cudaEventQuery` (checked at the next layer's acquire) as the release gate —
  a full sync also fits v1's slack (GPU-serial 981 ms vs NVMe 5,193 ms) if the
  event bookkeeping isn't ready; event-release is the first G-phase knob.
- **C layer**: `cudaMemcpyAsync` hidden[5120] f32 D2H into a 20 KB pinned
  staging cell + event; `CpuPool::launch(cpu_layer_step, ...)` runs the whole
  layer (insignia_cpu.hpp kernels) against the locked pageable weights, with
  the layer's delta/conv state and KV rows living in host RAM (107.9 MB states
  + 100.7 MB host KV total, master plan §2.3); result H2D back into `hidden`.
  ~0.3 ms handoff ×2 per C layer (cpu-impl.md).
- **embed row** (bf16, NVMe): buffered-twin `pread` of the 10,240 B row for the
  *pending* token — issued one step ahead at commit time (target id known),
  double-buffered pinned cells; consumed by `bf16_get_row` (fp8.cu:193-200)
  reading the pinned row via UVA and writing f32 `hidden` on device.
- **logits/argmax**: computed on VRAM lm_head; `next_host` mirror D2H on a
  dedicated copy stream (decode.cu pattern, kept).

**Colibri-style early issue**: the feeder's window math already issues the next
slot while the current one is consumed (read-ahead = slots−1, arm at release,
streaming.cu:399-408). Two additions complete it: (1) begin_epoch submits the
N-plans **in layer order with the Bresenham Z/N interleave of the manifest**, so
disk order == consumption order (colibri-sched-deep §8 cyclic schedule); (2) the
embed pread above is the second early-issue lane (one step ahead, separate
buffered handle, never contends with the direct-handle stream).

### 2.4 Placement manifest = the v1/v1.5/v2 switch, as data

One text file, parsed at startup, three shipped copies — the engine binary is
identical across the ladder (master plan §2.4 "three manifest files, zero engine
changes"):

```
# build/manifest-v1-allstream.txt
vram   0 18          # L=19: layers 0-18 in VRAM (15 lin + 4 full)
nvme   19 63         # everything else streams
ring_slots 3         # ×368 MiB (streaming.hpp:207) = 1,152 MB pinned (v1)
lm_head vram ; mtp vram ; embed nvme_pread
ctx 2048 ; kv f32

# build/manifest-v1.5-pinz.txt : vram 0 18 | pinned 19 40 | nvme 41 63 | ring_slots 3
# build/manifest-v2-cputier.txt: vram 0 18 | pinned 19 39 | cpu 40 48 | nvme 49 63 | ring_slots 2 + kv-host-for-C
```

Plug-in point: `TieredStorage27::configure(const char* manifest_path)` — first
call, before any allocation, builds tier_of[64], the arenas, the plan list, and
the reader/ring. Validation at parse time (not mid-decode): pinned total + ring
≤ probe cap; VRAM ledger vs `cudaMemGetInfo`; N-count × 119 ms vs placement
expectation (io-bench measured 3.22 GB/s ⇒ 119 ms/layer, io-bench-results §0).

### 2.5 The 16-byte rebasing pad (master plan D.2 CRITICAL, made concrete)

`data_start ≡ 8 (mod 16)` in every shard (e.g. layers-0 header = 8+2592 = 2600 ≡
8 mod 16, verified live by the streaming smoke). Whole-shard plans therefore
land F8 tensor bases ≡ 8 mod 16 → `uint4` loads in fp8_gemv/gemv2/gemm
(fp8.cu:32, 76, 126-133) fault or scalarize. Fix in the plan builder: walk the
shard's tensors in file order (INSIDX02 gives abs_off; they are sorted by
offset), group contiguous bf16 smalls into request 1, then emit one dummy
8-byte request (re-read of bytes [0,8) into the pad position) and put the F8
region (contiguous to shard end) into request 2. All F8 bases shift +8 in-slot
→ 16B-aligned; bf16 smalls are scalar u16 loads (alignment-free). Assert
`((base + f8_off) & 15) == 0` in `acquire_layer` consumers (R3 test target).

---

## 3. `src/generate27.cu` — main loop sketch

No CUDA graphs anywhere (master plan §2.4/D.4). Eager, one host thread (T0)
driving `x.stream` + a copy stream. MTP D=1/T=2 via the pair machinery
(spec_prologue/setup/commit/rollback, prefill.cu:277-313, parameterized —
see §4). Startup as §2.3; then:

```
prompt_ids = tokenizer ids (host)
# ---- prefill: weight-stationary, layer-major (Phase F) ----
# (v0 bring-up may first ship token-chunk prefill over the ring: 64× slower
#  per token but code-identical to decode; stationary lands in F.)
prefill_stationary(prompt_ids):
  feeder_epoch_for_prefill()            # same plans, holds slot across T rows
  embed_gather_bf16 rows -> h_A[5120-token tiles ping-pong h_A/h_B]
  for l in 0..63:                       # position bumped ONCE per turn
     acquire tier(l); layer kernels tile-by-tile on h_A/h_B; release
  final norm (smalls arena) -> h_last
next = argmax(bf16_gemv(lm_head_vram, h_last))         # 5.4 ms sweep
append_committed(prompt_ids); prime pending = next
set_position(P); set_mtp_position(P-1)
issue embed pread for `next` NOW (one step ahead)

loop while generated < max_new and not EOS:
  feeder.begin_epoch(n_plans)            # arms K-1 slots (reader starts NOW)
  # ---- draft (MTP, all-VRAM, ~6.7 ms; hides under the NVMe stream) ----
  wait embed-row pread (pinned cell); bf16_get_row -> x.hidden_of_pending
  mtp_layer27(pending, hidden_main)      # decode.cu:137-192 body, 27B shapes:
      # embed pending -> down; norm both halves (1+w); concat(embed, hidden, 5120)
      # bf16_gemv(fc[5120,10240]) -> hidden_draft
      # full-attn layer 0 on mtp_keys/mtp_values at mtp_pos
      # final mtp.norm (1+w); bf16_gemv(lm_head) -> argmax -> draft id
  # ---- verify: T=2 pair forward of [pending, draft] through 64 layers ----
  for l in 0..63:
    switch tier_of[l]:
      V: pair kernels on VRAM ptrs          (fp8_gemm T=2 via pf_bf16 staging;
                                             fp8_gemv2 where cols<=12288)
      Z: same kernels on pinned UVA ptrs
      N: base=feeder.acquire_layer(e++);
         pair kernels on slot ptrs (UVA); event; release_layer(e-1) on event
      C: D2H 2×5120 hidden; cpu pair layer (fp8_gemv2_mt etc.); H2D
  final norm; bf16 GEMM lm_head T=2 (both rows one pass); argmax each row
  # row0 argmax must equal `pending` (self-check, cheap); row1 = after
  acc = (draft == row0_argmax)...           # spec_commit semantics, eager:
  commit: accepted -> [pending, draft], new pending = after
          rejected -> [pending],            new pending = row1 (forwarded)
  spec_rollback27 (48·48·128·128 + 48·10240·3 + hidden 5120) if rejected
  append_committed_host(ids, n); position += (1 + acc)
  next embed pread issued for the new pending (id known at commit)
  logits D2H / EOS / budget checks on the copy stream (host_committed mirror)
```

Notes that are load-bearing:
- `spec_rollback_kernel` today hardcodes 9B sizes (prefill.cu:305-313:
  `24*32*128*128`, `24*8192*3`, `4096`) — parameterize (grid-stride, dims in).
- `row0 argmax == pending` self-check: nearly free (one 248320 compare) and
  catches tier-dispatch/parity bugs the moment they appear, before they poison
  state.
- Commit/rollback kernels write device state; the host never touches KV.
- v0 bring-up mode: `--no-mtp` plain greedy T=1 through the same dispatch
  (smallest thing that can first emit a correct token end-to-end).

---

## 4. Kernel gaps at 27B shapes — existing vs missing, per kernel

Shapes (from the shards, §0): linear F8 mats — qkv [10240,5120], z [6144,5120],
out [5120,6144], gate/up [17408,5120], down [5120,17408]; a/b bf16 [48,5120];
A_log/dt_bias bf16 [48]; norm [128]; conv1d [10240,1,4]. Full layer: q [12288,5120],
k/v [1024,5120], o [5120,6144]; q/k_norm [256]. lm_head/embed bf16 [248320,5120].
mtp.fc bf16 [5120,10240]. All fp8 cols (5120/6144/17408) are %128==0 ✓.

**Already parameterized — call-site constants only (no kernel edit):**
`rmsnorm_bf16` (rows,cols; qwen_kernels.cu:5-6 — flip the zero_centered arg to
true at the 9 call sites, master plan A.7), `gated_rmsnorm_bf16` (:6),
`causal_conv4_silu` (:7-8, n-generic; 8192→10240 at call),
`deltanet_parameters` (:9-10, n param; 32→48 + A_log dtype dispatch),
`silu_mul`/`residual_add`/`sigmoid_mul` (ops.cu:7-8, qwen_kernels.cu:11),
`bf16_gemv` (:67-68, rows/cols), `concat` (:69, n=5120),
`argmax_fast` (:59-63, n), `store_kv`/`store_kv_batch` (:15-16,
prefill.cu:88-98 — **unchanged at 27B**, 4 kvh × 256),
`addi/spec_prologue/spec_setup/spec_commit` (prefill.cu:271-302),
`bf16_get_row` (fp8.cu:193-200).

**Hardcoded geometry — edit in place (template or constants), 9B kept green:**

| kernel | file:line | 9B | 27B | edit |
|---|---|---|---|---|
| `qk_norm_rope` | ops.cu:9-10 | grid 20, `isq=head<16` | grid **28**, `isq=head<24`, `k+(head-24)*256` | `template<int QH>`; full replacement in attn-27b §1a (race fix preserved) |
| `qk_norm_rope_batch` | prefill.cu:54-86 | `dim3(20,T)`, `isq<16`, `t*16` | `dim3(28,T)`, `t*24` | same template; attn-27b §1b |
| `gqa_decode` | attention.cu:7-8 | grid 16, `kvh=head>>2` | grid **24**, `kvh=head/6` | template or plain edit; body untouched (attn-27b §1) |
| `gqa_prefill` | prefill.cu:102-165 | `dim3(16,T)`, `>>2`, `t*16` | `dim3(24,T)`, `/6`, `t*24` | same |
| `split_q_gate` | qwen_kernels.cu:73-74 | `i<4096`, grid 16, `h=i>>8` | `i<6144`, grid 24 (h=i>>8 **stays**: per-head src stride 512 unchanged) | constant edit |
| `split_q_gate_batch` | prefill.cu:43-51 | `>>4`,`&15`, `t*8192`,`t*4096` | `/24`,`%24`, `t*12288`,`t*6144` | constant edit (attn-27b §1 table) |
| `expand_gate_heads` | qwen_kernels.cu:78-79 | `i<4096`, grid 16 | `i<6144`, grid 24 | constant edit |
| `deltanet_decode` | deltanet.cu:5,14 | `<<<32,128>>>`, `kh=head>>1` | `<<<48,128>>>`, `kh=head/3` | grid + one operator; K=V=128 unchanged |
| `deltanet_prefill` | prefill.cu:219-269 | `<<<32,128,66048>>>`, `kh>>1`, stride 8192, `t*32` | `<<<48,128,66048>>>`, `kh=head/3`, stride **10240**, `t*48` | smem **66048 B is head-count-independent** (128·128·4 + scratch); the 99 KB opt-in `cudaFuncSetAttribute` already exists (prefill.cu:266) — no new opt-in needed, grid only |
| `conv_prefill` / `conv_roll_state` | prefill.cu:170-199 | `8192` ×6 literals | `10240` (pass `ch` param) | parameterize channel count |
| `params_batch` | prefill.cu:203-213 | `h<32` guard, `t*32`, A_log as f32 | `h<48`, `t*48`, A_log bf16 dispatch (`const void* + bool a32`) | deltanet-27b §5c; **A_log bf16-as-f32 is silent-risk #4** — the 27B call passes `A.dtype==DType::f32` |
| `spec_rollback` | prefill.cu:305-313 | 9B sizes ×3 | 27B sizes; grid-stride | also fixes the latent 9B partial-hidden bug (B.4) |

**Missing kernels (new code):**
1. `bf16_gemm` — raw bf16 [rows,cols] × bf16 A [T,cols] → f32 [T,rows]. Use the
   `mxfp4_gemm_v21_i4` skeleton (gemm.cu:371-454) minus the dequant (B tile is
   already bf16 → prefetch straight into `Bs`); rows%32, cols 5120/10240/17408
   all %64 ✓. Needed by: lm_head T≥2 verify, NLL chunks, mtp fc T-path. ~90 LOC.
   Build FIRST and bench T=1 through it vs `bf16_gemv` (master plan C.4).
2. `embed_gather_bf16` — T-row gather from bf16 [248320,5120] (prefill /
   parity); trivial sibling of `embed_gather` (prefill.cu:9-23) minus scales.
   Decode uses `bf16_get_row` on the pread row instead (§2.3).
3. Optional (defer to G unless bench says otherwise): persistent `bf16_gemv`
   v2 for lm_head T=1; merged multi-row argmax (`argmax_rows`) for T≥2 — v1
   can call `argmax_fast` per row (5.4 ms sweep dominates; two launches ≈ 10 µs).
4. NOT needed at 27B: `bf16_gemv_ab2_pair` (master plan C.4 design B) — a/b are
   bf16 [48,5120]; two `bf16_gemv` calls on 48 rows ≈ µs. The 9B i4 ab2
   launchers must instead get `cols!=4096 → throw` guards so nothing routes
   27B traffic into them (C.4 guard item, keep).
5. fp8 dispatch in `linear27/linear2_27/linear_batch27` (decode27.cu):
   `kind==fp8 → fp8_gemv / fp8_gemv2 / fp8_gemm`; `kind==bf16 → bf16_gemv /
   bf16_gemm`; mxfp4 branch never taken at 27B.

**fp8 kernel correctness/cap gaps found by reading fp8.cu (verify vs Phase C.3):**
- `fp8_gemv` (fp8.cu:52-55): dynamic smem = cols·4 B. cols=17408 → 69,632 B >
  the 48 KB default → **launch fails today with no error check**. Fix:
  `cudaFuncSetAttribute(...,99*1024)` opt-in (the gemv2 pattern, fp8.cu:100) +
  post-launch error check. cols=5120/6144/12288 are fine.
- `fp8_gemv2` (fp8.cu:96-103): smem = 2·cols·4; 17408 → 139,264 B > 101,376 B
  hardware max — the existing guard (fp8.cu:99) correctly throws; the pair path
  for cols=17408 must route to `fp8_gemm` T=2 (bf16 staging) as the error
  message already says. Dispatch: `pair → cols>12288 ? gemm : gemv2`.
- `fp8_gemm` F1: the `if (wm*16 < T)` store guard (fp8.cu:184) skips whole
  tiles only; a straddling tile still writes all 16 rows — callers must keep
  passing 64-row-padded y (all pf_* buffers are). Either keep the documented
  64-row contract in decode27 (preferred; zero cost) or do the smem-staged
  epilogue from master plan C.3 if any T-row y sneaks in (nll27 logitsT is
  chunk-row — fine).

---

## 5. Parity tools: `nll27` / `dump_layers27` (+ the reference gap)

- `src/dump_layers27.cu` (clone of dump_layers.cu, 27B-ized): ModelFile27 →
  TieredStorage27 with a **parity manifest** (all 64 layers walk one at a time;
  simplest: N-tier-only ring, consume via UVA — one 14-token sweep ≈ 29.9 GB ≈
  9-30 s, master plan R6). Per layer: copy `hidden` [5120] D2H, fwrite; 65 rows
  + final norm; feed `reference27.py seams` output (65×T×5120) to the existing
  compare scripts. Add a `--layer N` single-layer mode for R4/R5 (layer 0
  DeltaNet, layer 3 full-attn ×5 runs). ~90 LOC.
- `src/dump_multistep27.cu` (clone of dump_multistep.cu:1-77): 4-step
  teacher-forced decode with per-step 65-seam dump + per-step MTP draft — R7.
  Needs the bf16 lm_head branch (dump_multistep.cu:31-36 lambda is mxfp4-only)
  and the A_log dtype dispatch. ~140 LOC.
- `src/nll27.cu` (clone of nll.cu:44-94): swap the lm_head GEMM (nll.cu:78-80,
  mxfp4-only) for `bf16_gemm` on the VRAM-pinned lm_head (T≤64 tiles, logitsT
  [chunk,248320]); `row_logp_kernel` is vocab-generic — reuse verbatim.
  ~120 LOC. Cross-check: `tools/reference27.py nll <ids>` (|ΔNLL| < 0.02 →
  0.005, R8) and `greedy` for R9 8/8 ids.
- Reference-side gap: **`reference27.py` has no MTP path** — port
  `tools/reference_multistep.py` conv-roll/MTP/state logic to 27B shapes
  (`reference_mtp27.py`, ~150 LOC python) or extend reference27 with an `mtp`
  subcommand; needed by R7's draft-path comparison and the Phase F acceptance
  probe (draft parity is NOT covered by R9 — master plan risk #8).

---

## 6. Ordered task list (dependencies, LOC, parallelism)

Tracks are parallelizable at the marked points. "Gate" = the master-plan rung
the task unblocks. Every task ends with the 9B suite green (kernels are
templated so `<16>` and `<24>` coexist).

| # | task | files | LOC | depends on | gate |
|---|---|---|---|---|---|
| T1 | **ModelFile v2**: INSIDX02 parse (shard table, per-shard eager handles + mapped views for parity tools, `TensorView{shard,off,bytes}`, bounds check, keep INSIDX01 path) | model_file.cpp, insignia_model.hpp | ~180 | — | unblocks ALL 27B weight access |
| T2 | **bf16 kernels**: `bf16_gemm` (+bench T=1/T=2/64 vs bf16_gemv), `embed_gather_bf16`; ab2 guard throws | gemm.cu, prefill.cu | ~130 | — (parallel with T1) | lm_head/mtp.fc paths |
| T3 | **fp8 fixes**: gemv 99 KB opt-in + error checks; pair-cols dispatch; F1 decision | fp8.cu | ~30 | — | R1 re-run |
| T4 | **Kernel geometry pack** (shared, templated): qk_norm_rope±batch `<QH>`, gqa_decode/prefill `/6`, split_q_gate±batch, expand, deltanet_decode/prefill 48/`/3`/10240, conv ch param, params A_log dispatch, spec_rollback dims; unit tests (test_attention 24q poisoned-KV, test_deltanet 48/3) | ops.cu, attention.cu, deltanet.cu, qwen_kernels.cu, prefill.cu, test_*.cu | ~340 | — (parallel; only touches shared kernels) | R1, R5 |
| T5 | **`decode27.cu` + `DecodeWorkspace27`**: workspace (§1.1 table), linear dispatch on kind, delta/attention/mtp layer bodies with zero-center flips, single-token forward, `--no-mtp` greedy; names already engine-side via index | src/decode27.cu, include/insignia_decode27.hpp | ~550 | T1, T2, T4 | R4/R5 (single layer, parity manifest) |
| T6 | **`TieredStorage27` + manifest**: LayerHandle, arenas (lm_head/MTP/smalls/V/Z/C), probe ladder, plan builder with 8-byte F8 pad + `(base&15)==0` assert, feeder wiring, `matrix27/embed_row27` | new include/insignia_storage27.hpp + src/storage27.cu; qwen35.cu gains embed bf16 branch | ~500 | T1 | R3 (stream byte-equality + alignment), v1 end-to-end |
| T7 | **`generate27.cu`**: §3 loop, tier dispatch, eager MTP D=1 verify, embed pread lane, event-release, copy stream | src/generate27.cu | ~420 | T5, T6 | R6/R7 first full pass |
| T8 | **Parity tools**: dump_layers27, dump_multistep27, nll27, reference_mtp27.py | src/dump_layers27.cu, dump_multistep27.cu, nll27.cu, tools/reference_mtp27.py | ~500 | T5 (dump*), T7 (multistep) | R4→R9 ladder |
| T9 | **mk.py closure**: add decode27.cu/storage27.cu/streaming.cu/prefill.cu (batch kernels!)/gemm.cu to ENGINE27; wire generate27/nll27/dump-layers27/dump-multistep27 targets; flip future flags | tools/mk.py | ~25 | lands with T5-T8 | build |
| T10 | **v1.5 flip**: pinned-Z manifest + probe ladder + Z residency loader path | manifest + storage27 | ~60 | T7, R6 green | 3.0 s/step tier |
| T11 | **v2 CPU tier**: cpu_layer_step (insignia_cpu.hpp already complete), D2H/H2D handoff, host states/KV for C layers, GEMV team affinities | decode27.cu, storage27 | ~250 | T7, R6 green | 1.75 s/step tier |
| T12 | **Phase G**: event-release, stationary prefill (F), MTP KV fill (F), solver, knobs | per master plan G | ~800 | T10/T11 | R10 |

Critical path: **T1 → T5 → T7 → T8 (R6→R9) → T10/T11**. Day-1 parallel set:
T1, T2, T3, T4 (four independent agents possible — T4 is the big mechanical
one), plus reference_mtp27.py (python-only).

Budget: T1-T9 ≈ **2,575 LOC** new+edited (5-7 sessions) to a correct v1
all-stream engine generating tokens with parity ladder running — consistent
with the master plan's ~4,500-5,000 total once F/G (T10-T12) land.

Top silent-wrong-token reminders while implementing (master plan §4, all
designed-in above): zero-center flips at exactly 9 rms call sites per layer path
(linear_attn.norm stays RAW — measured [0.79,0.93]); `kvh=head/6` and
`kh=head/3` are semantic, unit-tested; A_log bf16 dispatch + α∈(0,1)≠1 dump
check; mtp.fc [5120,10240] with embed-normed half first (decode.cu:151 order
preserved); ring pad assert; `pf_scratch` is q_proj-sized 12288, not conv 10240.
