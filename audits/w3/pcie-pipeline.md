# W3 — PCIe pipeline: RAM→VRAM streaming vs zero-copy vs CPU for Qwen3.8-27B-FP8 decode

Date 2026-08-25. Rig: RTX 4070 SUPER (12,282 MiB VRAM, PCIe 4.0 x16, 22–25 GB/s
practical H2D, WDDM), Ryzen 5600X (6C/12T), 15.9 GiB RAM, NVMe ~6.5 GB/s O_DIRECT.
Model facts from `audits/w2/loader-27b-spec.md` (exec numbers), scheduling context
from `audits/w3/colibri-sched-deep.md` §8, per-layer tier costs from
`audits/synthesis.md`. Read-only audit; no engine files touched.

## 0. TL;DR

1. **VRAM holds 20 resident F8 layers** (15 linear + 5 full-attn) after lm_head
   (2.368 GiB, mandatory), MTP shard (0.444 GiB), context/graphs, KV, states —
   total 10.52 GiB of the 10.8 GiB app budget, 0.28 GiB slack for KV/WDDM drift.
2. **The killer discovery: Windows WDDM caps CUDA pinned memory at 50% of RAM**
   (≈7.95 GiB here) — cudaHostAlloc/cudaHostRegister cannot exceed it. The
   "34 pinned zero-copy layers" dream dies; 22 Z-layers max (7.81 GiB pinned).
3. Zero-copy UVA kernel reads are real but slower than memcpy: **15–20 GB/s
   practical** (vs 24–26 GB/s pinned memcpy) — fine for weights (17–25 ms/layer),
   reads far better than writes, data uncached (stream-once = exactly our pattern).
4. **CPU-compute layers don't need pinned memory at all** — VirtualLock'd pageable
   RAM works (no nonpaged-pool cost), so RAM splits into a pinned budget (GPU-fed
   Z layers) and a locked-pageable budget (CPU layers + NVMe ring slots).
5. **Staged-copy (S) tier is eliminated**: with NVMe the binding engine, PCIe
   efficiency (15.4 vs 17.4 ms/layer) buys nothing, and the 2×384 MB VRAM staging
   ring costs 2 resident layers. No device staging slots. No copy stream traffic
   beyond embed row / state tails / logits.
6. **The LP floor is set by physics, not placement**: RAM holds 35 resident
   layers of the 44 non-VRAM ones → **9 layers must be NVMe-fed every step** →
   9×381 MB @ 6.5 GB/s = **527 ms/step is the hard floor** on this rig.
7. Champion config: **L=20 VRAM / Z=22 pinned-UVA / C=13 CPU / N=9 NVMe→CPU**,
   step ≈ 545–590 ms → **~1.8 tok/s single, ~2.8–3.0 tok/s with MTP×1.6**.
   PCIe (381 ms) and CPU (220 ms) both hide under NVMe (527 ms) with margin.
8. embed stays **out of VRAM and out of pinned RAM**: mmap'd (the zero-copy mmap
   reader already exists) + CPU row gather of 10 KB/token, shipped H2D with the
   token. Pinned embed would burn 2.37 GiB = 6.7 layer slots ≈ 0.3 tok/s.
9. DRAM runs at ~85% of practical bandwidth in the champion config (36 of
   ~42 GB/s) — the real second bottleneck; N layers re-read every step also burn
   the SSD at ~6 GB/s (≈500 TB/day at 24/7 — endurance is the invoice).
10. LayerFeeder state machine (§7): NVME_FILLING → RAM_READY → {ZERO_READ |
    STREAMING→VRAM_READY} → CONSUMED, NVMe readers + CPU GEMV team + GPU compute
    stream, semaphore/CUDA-event choreography; STREAMING kept as a compiled
    fallback, unused in v1.

---

## 1. VRAM budget — byte-exact

Total 12,282 MiB (11.99 GiB). WDDM + display compositor + driver overheads eat
0.5–1 GiB → **app budget 10.8 GiB** (2.7 GiB margin below physical; the 0.5 GiB
beyond that absorbs compositor spikes and context growth).

Per-tensor bytes (from loader spec §2.3/2.4, byte-exact):

| object | bytes | GiB | note |
|---|---|---|---|
| linear layer (F8+BF16) | 383,862,848 | 0.35749 | 48× |
| full-attn layer (F8+BF16) | 372,311,424 | 0.34683 | 16× (i&3==3) |
| lm_head bf16 | 2,542,796,800 | 2.36831 | [248320,5120] bf16 |
| embed bf16 | 2,542,796,800 | 2.36831 | **not VRAM** (§5) |
| MTP shard | 477,199,744 | 0.44448 | F8 0.3466 + bf16 0.0978 (fc) |
| delta state f32 /lin layer | 3,145,728 | 0.00300 | 48×128×128×4; 151.0 MB total |
| conv state f32 /lin layer | 122,880 | 0.000117 | 10240×3×4; 5.9 MB total |
| KV /full layer @ctx2048 | 8,388,608 | 0.00800 | 2048×4h×256×2×2B |
| avg layer (3:1 mix) | 380,974,848 | 0.35472 | 48 lin + 16 full / 64 |

Fixed + resident table (champion placement, KV follows its layer; host KV for
non-VRAM full-attn layers lives in RAM):

| VRAM line | bytes | GiB |
|---|---|---|
| CUDA context + graphs | 419,430,400 | 0.400 |
| DecodeWorkspace 27B (pf buffers ~30 MB scaled from decode.cu:22-26, logits 2×248320×4, scalars) | 41,943,040 | 0.040 |
| lm_head bf16 | 2,542,796,800 | 2.368 |
| MTP shard | 477,199,744 | 0.444 |
| KV: 5 VRAM full-attn + MTP layer @2048 | 50,331,648 | 0.047 |
| delta states: 15 VRAM linear layers ×3 MiB | 47,185,920 | 0.044 |
| conv states: all 48 | 5,898,240 | 0.005 |
| **fixed subtotal** | **3,584,586,752** | **3.338** |
| 15 linear layers resident | 5,757,942,720 | 5.362 |
| 5 full-attn layers resident | 1,861,557,120 | 1.734 |
| **total** | **11,204,086,592** | **10.434** |
| **slack vs 10.8 GiB** | 377,487,360 | **0.366** |

Slack 0.366 GiB = KV growth headroom to ctx ≈ 7.5 k (69.6 KB/token for 17 KV
layers) + WDDM drift. **L = 20 is the ceiling** — 21 layers needs 10.79+ GiB with
zero slack, i.e. one compositor hiccup from eviction storms. Keep 20.

Drop `snap_delta` (decode.cu:29, the MTP-reject DeltaNet snapshot) or snapshot
only the MTP layer: saves 151 MB for 27B — worth 0.4 of a layer.

**Staging ring slots: none.** The streamed tier is eliminated (§4). For the
record, had streaming survived: 2 slots × 384 MB = 0.715 GiB = exactly 2
resident layers — that exchange is what kills it.

## 2. RAM budget — and the 50% pinned cap

Physical 15.9 GiB. Two distinct pools on Windows:

- **Pinned pool** (cudaHostAlloc / cudaHostRegister): WDDM driver limit
  **50% of RAM ≈ 7.95 GiB**, regardless of chunking (NVIDIA forum: "On Windows
  10/11 one can allocate only 50% of RAM using cudaHostAlloc"). Nonpaged-pool
  pressure and fragmentation can lower it further (reports of failures at
  700 MB on fragmented systems; 128 GB theoretical nonpaged ceiling is not the
  binding constraint here). **Probe at startup with an allocate-until-fail
  ladder**; plan around 7.5 GiB usable.
- **Locked pageable pool** (VirtualAlloc + VirtualLock / working-set growth):
  what the CPU-compute tiers and the NVMe ring actually need. No nonpaged-pool
  cost, no CUDA involvement; the colibri `compat_mlock` shim
  (w2/colibri-io.md, compat.h:195-212) is the ported pattern. Capacity =
  physical − OS − pinned − activations.

Champion RAM table:

| RAM line | GiB | note |
|---|---|---|
| pinned Z layers: 17 lin + 5 full | 7.81 | ≤7.95 cap; probe ladder, Z=21 fallback |
| locked C layers: 10 lin + 3 full | 4.62 | VirtualLock'd pageable |
| NVMe ring K=2 slots (2×384 MB, 4096-aligned) | 0.73 | locked pageable (CPU reads; no pin needed) |
| embed | ~0 (mmap) | page cache grows ≤2.37, evictable |
| host KV (11 non-VRAM full layers) + activations + logits staging | 0.15 | |
| OS + background (trimmed Win11) | 2.4 | measure on the box |
| **total** | **15.71** | of 15.9 — knife's edge; ladder trims C first |

Every 1 GiB of RAM freed = 2.8 more resident layers = 2.8 fewer NVMe layer-ms =
~163 ms/step. Strip the OS (services, Superfetch, browser) before touching any
kernel.

## 3. Tier cost model (per layer per step) and the engine view

Bandwidths (measured/verified): VRAM DRAM 504 GB/s (kernels currently 367–378,
ceiling ~454 — w3/insig4-perf.md); PCIe 4.0 x16 pinned memcpy **24–25 GB/s**
practical; **UVA zero-copy kernel read 15–20 GB/s** (§6); CPU DRAM ~38–42 GB/s;
NVMe O_DIRECT 6.5 GB/s (5.5–6.8 spread).

| tier | where weights live | engine consumed | ms/layer (avg 381 MB) |
|---|---|---|---|
| L resident | VRAM | VRAM DRAM | 0.76 (weights+state @504; 0.5 is the optimistic no-state number) |
| S staged-copy | pinned RAM | PCIe (DMA memcpy) | 15.4 transfer (PCIe-bound) + 0.5 hidden |
| Z zero-copy | pinned RAM | PCIe (SM reads over UVA) | 17.4 @22 / 19–25 @15–20 GB/s |
| C CPU-compute | locked RAM | CPU DRAM | ~10 (9.6 pure + contention) |
| N NVMe-fed | ring slot (locked RAM) | NVMe 6.5 GB/s | 57.3–59.1 (fill) + compute hidden |

The reframe the mission asks for, cleanly: **each layer's weights must cross
exactly one slow fabric once per step** — VRAM layers cross none; S crosses PCIe
via copy engine; Z crosses PCIe via SM loads; C crosses nothing (DRAM-local);
N crosses NVMe→RAM, then optionally PCIe (N_z) or not (N_c). Engines are
pipelines that run concurrently across layers (round-robin interleave,
colibri-sched-deep §8.1 correction 1), so:

```
T_step ≈ max( PCIe_ms , NVMe_ms , CPU_ms , GPU_ms ) + ~15 ms overhead
  PCIe_ms = 15.4·S + (17.4..25.4)·Z + 17.4·N_z        [shared, serialized bus]
  NVMe_ms = 57.9·N                                     [6.5 GB/s, the pace car]
  CPU_ms  = 10·(C + N_c)                               [12-thread GEMV team]
  GPU_ms  = 0.76·L + ~17 (2× lm_head 5.1 + MTP draft ~6)   [never binds]
```

NVMe→RAM→GPU-read is pipelined, not serial: reader fills slot i+1 (59 ms) while
GPU reads slot i over PCIe (17.4 ms) → per-layer cost max(59, 17.4) = 59, NVMe
dominates ✓ (mission's analysis confirmed). But the 17.4 ms still **adds to the
shared PCIe budget** — with Z already near saturation, N_z is strictly worse
than N_c unless the CPU team is saturated. CPU team: 22 CPU-layers × 10 ms =
220 ms of 12-thread time per 545 ms step — 40% duty, room to spare → **N_c**.

DRAM audit per step (champion): NVMe writes 9×381 MB + CPU reads 22×381 MB +
GPU DMA reads 22×381 MB + state tails ≈ 19 GB / 0.545 s ≈ **35 GB/s = ~85% of
practical**. This is the second-order constraint: if it throttles, T stretches
5–10%; the C↔Z split absorbs it (C reads DRAM at 38 GB/s vs Z's PCIe round-trip
— both land in the same DRAM controller). Flag, don't fear.

## 4. Stream vs zero-copy vs CPU — the trade, solved

Mission's trade matrix, with the numbers filled in:

- **S (staged copy)**: 15.4 ms PCIe + 0.5 hidden; costs 2×384 MB VRAM ring
  (double-buffered) + pinned RAM slot. PCIe-efficient (copy engine, 24 GB/s).
- **Z (zero-copy)**: 17.4–25 ms PCIe; costs **0 VRAM**, pinned RAM slot.
- **C (CPU)**: 10 ms; costs **0 VRAM, 0 pinned** (locked pageable), CPU cycles.

The LP over (L, S, Z, C, N_c, N_z), L+S+Z+C+N_c+N_z = 64:

```
minimize  max(15.4·S + z·Z + 17.4·N_z,  57.9·N,  10·(C+N_c))     z ∈ [17.4, 25.4]
s.t.      VRAM: 3.338 + 0.359·L + 0.715·[S>0]            ≤ 10.8
          pinned: 0.355·(S + Z + N_z) + K_z·0.355        ≤ 7.95   (WDDM 50%)
          physical RAM: 0.355·(S+Z+C+N_c+N_z+K) + 2.55   ≤ 15.9
          L ≤ 20 (from VRAM),  48 lin / 16 full composition
```

Solution walk (this is the whole story of this audit):

1. **RAM is the binding resource, not PCIe.** Physical RAM caps resident
   non-VRAM layers at ~35 (of 44). So **N ≥ 9 always**, NVMe_ms ≥ 527 ms.
2. **Therefore NVMe is the binding engine**, and PCIe/CPU/GPU just have to stay
   under 527 ms. PCIe budget: even all-pessimistic Z (25.4 ms) × 22 = 559 ms —
   marginal; balanced split or C-shift fixes it. CPU: 44 layers × 10 = 440 ms
   worst case ✓.
3. **S dies**: S buys PCIe efficiency (15.4 vs 17.4) on a bus that isn't
   binding, and pays 0.715 GiB VRAM = 2 L-layers = 2×57.9 ms more NVMe
   (L 20→18 forces N 9→11 via RAM arithmetic). Strictly dominated.
   **No staging ring, no device slots, no copy-engine traffic.**
4. **Z vs C split** is a free parameter balanced under the NVMe line: Z=22
   (pinned cap) + C=13 puts PCIe at 381 ms (22 GB/s) or 559 ms (15 GB/s) —
   both acceptable; C-shift is the pressure valve if UVA measures badly.
5. Sanity vs synthesis.md's earlier 0.66–1.2 tok/s: the gains come from (a)
   mmap embed instead of pinned (frees 6.7 RAM slots), (b) no VRAM ring
   (+2 L), (c) locked-pageable C layers dodging the 50% pinned cap, (d) N=9
   not 21.

### Top-3 configs

| # | L VRAM | Z pinned | C CPU | N NVMe→CPU | K | PCIe ms | NVMe ms | CPU ms | T_step | tok/s single | tok/s MTP×1.6 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 champion | 20 (15L+5A) | 22 | 13 | 9 | 2 | 381 (559 @15GB/s) | 527 | 220 | ~545 | **1.84** | **2.9** (2.8–3.1) |
| 2 bring-up (no UVA, no pinned) | 20 | 0 | 33 | 11 | 2 | ~0 | 637 | 440 | ~655 | 1.53 | **2.45** |
| 3 safety (fat OS / probe fails low / ctx headroom) | 19 | 14 | 13 | 18 | 3 | 243 | 1042 | 310 | ~1060 | 0.94 | **1.51** |

Config 2 is the v1 implementation target: zero UVA risk, zero pinned-CUDA
dependency, still ~2.5 tok/s; config 1 is v2 (+18%) once the UVA GEMV kernel and
the pinned probe exist. Config 3 is the degradation ladder floor.

Step-cost model (verified against mission): step = 1 full 64-layer pass (verify
T=2 reads weights once — bandwidth-bound, ≈ T=1 cost) + lm_head (5.1) + MTP
draft (mtp.fc + layer + lm_head ≈ 6.3) + embed/logits/overhead ≈ 15 ms.
tok/s = 1.6 / T_step at p=0.6 (colibri: depth-1 acceptance 85% → up to 1.85
tokens/step → config 1 could touch 3.3 tok/s on easy text).

Tier interleave: N layers every ~7th layer, L every ~3rd — Bresenham-style
spread over 0..63 so NVMe demand is one 384 MB layer per ~60 ms of wall clock
(colibri-sched-deep §8.1 correction 1). Never a contiguous N block.

## 5. The embed path (and why it is not pinned-UVA)

Decode touches exactly one 10 KB row per token. Three candidates:

1. **Pinned RAM + UVA row gather from kernel**: 10 KB over PCIe ≈ 0.4 µs
   transfer + ~20 µs latency (kernel launch + TLP round trips) — technically
   perfect, **but 2.37 GiB pinned = 6.7 layer slots = ~0.3 tok/s of opportunity
   cost** while RAM is the binding resource. Rejected.
2. **VRAM resident**: 2.37 GiB VRAM = 6.6 L-layers ≈ 380 ms/step of cost.
   Rejected harder.
3. **mmap'd (existing zero-copy mmap reader) + CPU row gather + H2D with the
   token** (winner): gather = memcpy 10 KB from page cache (~1 µs warm; cold =
   2–3 page faults ≈ 0.1–0.3 ms, amortized once per distinct token), ship from
   a tiny pinned 2-row staging buffer with the 20 KB hidden H2D. Prefill
   gathers T rows once — still trivial. Costs 0 resident budget. **Chosen.**
   Watch page-cache eviction under 15.7 GiB commit: faulted embed pages
   re-read from NVMe — the same disk the layer reader is saturating; keep a
   VirtualLock'd 256–512 MB hot-row LRU if faults ever show up in profiles.

MTP shares embed_tokens (`mtp_use_dedicated_embeddings: false`, loader spec
§2.5) — same path feeds the draft's embed half of `mtp.fc` input.

lm_head: VRAM, non-negotiable (5.1 ms resident vs 102 ms if streamed over PCIe).

## 6. Zero-copy UVA read performance — verification (mission §4)

What the outside world reports (sources at bottom):

- **Zero-copy kernel access ≤ PCIe bandwidth, and in practice well below
  memcpy**: NVIDIA forums state zero-copy's best possible access bandwidth is
  the PCIe link rate; measured kernel-read rates on PCIe 4.0 x16 land at
  **~15–20 GB/s** for well-coalesced streaming reads, vs **24–26 GB/s** for
  pinned cudaMemcpyAsync (Microway's transfer guide; NVIDIA forum threads).
  Planning number for Insignia: **18 ± 4 GB/s** → 21–27 ms per 381 MB layer.
  The mission's 22 GB/s (17.4 ms) is the optimistic edge of the band.
- **Reads are far better than writes** over UVA (non-coherent write path;
  reads pipeline well with many concurrent threads). We only read. Good.
- **Zero-copy accesses are uncached** (Lei Mao's analysis): GPU caches are
  bypassed for mapped host memory. Irrelevant-to-good for us: weights are
  read exactly once per step; no reuse to lose. (One caveat: the x-vector
  must NOT live in mapped memory — keep activations in VRAM.)
- **Scattered vs sequential matters**: coalesced 128 B-aligned streaming reads
  approach the top of the band; random 32 B reads collapse to a fraction.
  Our GEMV streams weight rows sequentially per block — near-best case. One
  wrinkle: F8 tensor bases in the shard are ≡88 (mod 128) (loader spec §2.8
  alignment census) → every row misaligns 128 B windows by 88 B; either eat a
  ~1–3% penalty or re-base tensors to 128 B when building the pinned copy at
  startup (one-time 6–9 memmoves per layer, we own the layout).
- **Kernel must keep enough loads in flight**: a grid-stride GEMV with the
  whole row-block resident has thousands of outstanding loads — no issue; the
  existing MXFP4 matvec shape (~150 GiB/s from VRAM) ports to a `__ldg`-style
  UVA-source variant with e4m3→bf16 decode (`cvt.rn.f16x2.e4m3x2`,
  synthesis.md) + bf16 block scales, FP32 accum.
- WDDM specifics: `cudaHostAllocMapped` works under WDDM (UVA is always mapped
  on 64-bit); the 50% cap of §2 is the real Windows constraint; WDDM's
  pageable-command batching adds occasional 1–2 ms submission spikes —
  `cudaSetDeviceFlags(cudaDeviceScheduleSpin)` and event-free spin handoffs
  (colibri #159: spin beats wake, 5 µs vs 10 ms layer) absorb them.

Verdict: **zero-copy reads are viable and worth 18% end-to-end (config 2→1)**,
but they are the *last* optimization, not the foundation — the C tier already
covers their failure.

## 7. Implementation plan

### 7.1 Architecture (champion, config 1)

```
            NVMe readers (3 threads, ReadFile+OVERLAPPED, O_DIRECT, QD8-16, 8-16 MB chunks)
                │  fills cyclic ring slots (K=2..4 × 384 MB, 4096-aligned, VirtualLock'd)
                ▼
 ┌────────────  pinned/locked RAM  ────────────┐   ┌──────── VRAM 10.43 GiB ─────────┐
 │ Z×22 layer weights (cudaHostAlloc, UVA)     │   │ L×20 F8 layers + their states   │
 │ C×13 layer weights (VirtualLock'd pageable) │   │ lm_head 2.37 + MTP 0.44 + KV    │
 │ ring K×2 slots (transient N layers)         │   │ workspace 0.04 + ctx 0.40       │
 │ embed (mmap, page cache) + host KV          │   └────────────▲────────────────────┘
 └──────┬───────────────┬──────────────────────┘                │
        │ N_c, C: CPU   │ Z: GPU reads rows directly over UVA   │ L, lm_head, MTP:
        ▼ GEMV team     │ (PCIe 18±4 GB/s, SM-driven)           ▼ compute stream
   12-thread OMP pool   └─────────────── PCIe 4.0 x16 ──────────┤ copy stream (tiny):
   (Zen3 F16C e4m3                                        logits D2H, embed row H2D,
    dequant-GEMV, 10 ms/layer)                            delta-state tails (3 MiB/CPU layer)
```

Three engines, one consumer cursor (T0): NVMe fill (pace car, 100% duty),
CPU GEMV (40% duty), PCIe (Z reads, ~70% duty). GPU DRAM nearly idle. The
copy stream exists but carries only KB–MB traffic; all bulk weight movement to
the GPU is SM-driven UVA reads.

Affinities (colibri-sched-deep §8.3): T0+GPU driver cores 0–1, readers 2–4
(they sleep in ReadFile), GEMV team 5–11, `OMP_PROC_BIND=close`,
KMP_BLOCKTIME=200, one parallel region per layer.

### 7.2 LayerFeeder

One state machine per *slot* (K slots) for NVMe-fed layers; resident layers
(L/Z/C) are static pointers outside the machine. STREAMING/VRAM_READY states
are implemented (mission requirement) but dormant in v1 — flip
`kFeederMode = STREAM` and add the 2×384 MB device ring only if a future
config wants S-layers.

```cpp
// layer_feeder.hpp — 3-engine choreography: NVMe readers, CPU GEMV team, GPU stream.
// AGENTS spirit: states are u8, transitions are release/acquire stores, zero locks on
// the hot path, one cursor writer (T0). TRUE = ring epoch (never 0 == free).
#include <atomic>
#include <cstdint>
#include <windows.h>

enum : uint8_t { ST_FREE = 0, ST_NVME_FILLING, ST_RAM_READY,
                 ST_STREAMING, ST_VRAM_READY, ST_ZERO_READ, ST_CPU_READ, ST_CONSUMED };

constexpr int   kSlots      = 2;                    // K: 1 filling + 1 ready (§4: 58 ms fill < ~60 ms spacing)
constexpr int   kChunks     = 24;                   // 16 MB chunks per 384 MB slot, QD = readers
constexpr uint64_t kSlotB   = 384ull << 20;         // covers 383.86 MB linear shards

struct alignas(64) Slot {                           // one cache line per slot (rule: align)
  uint8_t* base;                                    // 4096-aligned, VirtualLock'd (kFeederMode==CPU_READ)
                                                    // or cudaHostAlloc'd  (kFeederMode==ZERO_READ/STREAM)
  std::atomic<uint8_t> st;                          // ST_* above
  uint32_t epoch;                                   // slot generation; consumer matches wanted epoch
  uint32_t chunks_outstanding;                      // readers count down; last one publishes READY
  int      layer;                                   // model layer this fill is for
};

class LayerFeeder final {
public:
  enum Mode : uint8_t { CPU_FEED, ZERO_FEED, STREAM_FEED };  // where slot contents are consumed
  void init(int slot_count, Mode m, const uint32_t* nvme_order, int nvme_n);
  //  --- engine 1: NVMe (3 reader threads; cyclic schedule, flat out, never idle) ---
  void reader_loop(int id);                         // claims chunks via fetch_add; ReadFile+OVERLAPPED,
                                                    // O_DIRECT twin handles; publish ST_RAM_READY (release)
  //  --- engine 2: PCIe (only in STREAM_FEED mode; dormant in v1) ---
  void issue_stream(int layer, Slot& s);            // 12×32 MB cudaMemcpyAsync chain on copy_stream,
                                                    // cudaEventRecord(slot->vr[epoch&1]) -> ST_VRAM_READY
  //  --- engine 3: consumers ---
  const uint8_t* acquire_cpu(int layer);            // spin (_mm_pause) on ST_RAM_READY+epoch; -> ST_CPU_READ
  const uint8_t* acquire_zero(int layer);           // same, -> ST_ZERO_READ; GPU kernels read base via UVA
  const uint8_t* acquire_vram(int layer);           // cudaEventSynchronize(slot->vr[epoch&1]); -> ST_VRAM_READY
  void release(int layer);                          // after consumer truly done: -> ST_FREE (release),
                                                    // ring_tail++ (T0 sole writer), readers may refill
private:
  alignas(64) Slot   slots_[kSlots];
  uint32_t           ring_tail_{0};                 // consumer cursor (T0 only)
  std::atomic<uint32_t> ring_head_{0};              // reader cursor
  uint32_t           epoch_of_[64];                 // layer -> next epoch it will arrive in
  HANDLE             slot_ready_;                   // SRWLOCK+condvar: readers block when ring full
  cudaEvent_t        vr_[2];                        // STREAM_FEED double-buffer events (dormant v1)
  cudaStream_t       copy_stream_;                  // dedicated, non-blocking; chunked 32 MB async chain
  Mode               mode_{CPU_FEED};
};

// decode hot path (T0), round-robin tier order baked into tier_of[64] at startup:
//   for l in 0..63:
//     switch (tier_of[l]) {
//       VRAM:  launch_layer(l, resident_dev[l]);                       // static pointer
//       Z:     launch_layer_uva(l, resident_pin[l]);                   // kernel reads host rows
//       C:     cpu_gemv_layer(l, resident_lock[l]);                    // OMP team, ~10 ms
//       N:     p = feeder_.acquire_cpu(l); cpu_gemv_layer(l, p); feeder_.release(l);
//     }
//   lm_head (VRAM) -> logits D2H on copy_stream -> sample -> MTP draft (VRAM) -> next step
```

Choreography rules:

- **Publish/consume discipline**: NVMe readers publish `ST_RAM_READY` with
  `memory_order_release` after the last chunk's ReadFile completion; consumers
  spin-acquire on epoch match (`_mm_pause` — 5 µs wake vs 10 ms layer, spin
  wins); T0 is the sole writer of `ring_tail` (PipePool gen discipline).
- **STREAM mode events** (dormant): 32 MB chunks (12/slot) as an async chain on
  the dedicated copy stream — never one 384 MB memcpy (WDDM copy throttle would
  starve the latency-sensitive logits D2H and state tails behind a 15.4 ms
  monster); event per chain, not per chunk.
- **Shard contiguity confirmed** (loader spec §2.8): zero pad gaps in any
  shard; all tensors back-to-back from `data_start`. One slot = one contiguous
  382.9 MB blob + leading 1.1 MB small-tensor region → a single memcpy-able
  region if STREAM ever wakes; readers round O_DIRECT offsets down/up to 4 K
  and rebase tensor pointers by −`data_start` at index time (and to 128 B for
  Z-mode row alignment, §6).
- **CUDA graphs**: capture only the contiguous VRAM-resident layer runs +
  lm_head; N/C/Z layers break the graph (their pointers/sync are dynamic).
  This also dodges the graph-hazards class from synthesis.md bug 6.

### 7.3 WDDM notes (mission §3 checklist)

| concern | verdict |
|---|---|
| pinned >2 GB single alloc | chunk at 1 GiB (22 chunks); the 50% cap is on total, not per-call; fragmentation can still bite — probe ladder, fall back Z→C |
| cudaHostAlloc 4–8 GB on WDDM | 7.95 GiB hard ceiling (50% of 15.9); champion uses 7.81; if the probe returns less, `Z=ceil(probe/0.355)`, remainder to C — placement solver takes it as an input |
| memcpy on compute stream vs copy stream | dedicated non-blocking copy stream + events; in v1 it only carries embed row (20 KB), logits (1 MB), delta tails (3 MiB/CPU layer, 0.13 ms each — the colibri `kv_dev_sync` watermark pattern) |
| 384 MB one memcpy vs chunked | chunk 16–64 MB (rec. 32 MB) — overlap granularity + WDDM DMA fairness; moot in v1 (no bulk copies) |
| WDDM submission latency | ScheduleSpin + spin handoffs; ring K absorbs 1–2 ms spikes |
| VRAM overcommit | never: budget table sums to 10.43 of 10.8; no LRU eviction in the path (colibri LFRU not needed — static placement, dense access) |
| TDR | all kernels < 30 ms ✓ |

### 7.4 Startup sequence

1. Probe pinned cap (1 GiB ladder until fail) and free VRAM → solve the LP of
   §4 with measured numbers (it's ~20 lines: pick L from VRAM, split pinned
   Z / locked C by probe + balance check `PCIe ≤ NVMe·1.05`, N = 64−L−Z−C).
2. Allocate: VRAM residents + lm_head + MTP first (fail fast), pinned Z pool
   (1 GiB chunks), locked C pool, ring K=2, mmap embed.
3. Warm start: fill both ring slots + touch all C pages (VirtualLock growth)
   before token 1 (colibri cold-start lesson: 139 ms/token penalty).
4. Interleave map: Bresenham spread of L (stride 3.2) and N (stride 7.1) over
   0..63; bake `tier_of[64]` (no runtime checks — modify constant data, not
   engine logic).

## 8. Risks / open items

- **NVMe endurance**: 9 layers × 381 MB re-read every step = 3.4 GB/step ≈
  6 GB/s continuous at full speed = ~500 TB/day at 24/7. A 600 TBW drive lasts
  days of flat-out decode. This is the project's masochism tax; a 1 TB page
  cache of the hottest N layers (RAM permitting) or accepting config-3 speeds
  are the levers. (Windows file cache does NOT help — O_DIRECT bypasses it.)
- **DRAM at ~85%**: if CPU GEMV + NVMe writes + GPU DMA reads contend, expect
  5–10% stretch; measured, not assumed (AGENTS rule).
- **UVA rate variance** (15–22 GB/s): C-shift pressure valve; re-balance at
  startup from a 200 ms microbench, not hardcoded.
- **50% pinned probe** may return less than 7.95 GiB on the actual box
  (fragmentation reports as low as 700 MB exist); the ladder + config-3 floor
  keeps the engine alive at 1.5 tok/s regardless.
- **Prefill** rides the same machinery layer-major (weights-once sweep,
  FlexGen-style, colibri-sched-deep §8.6) — out of scope here, but the feeder
  API above is prefill-shaped already.
- Parity gate (AGENTS): all of this is irrelevant until full-attn layers match
  the NumPy reference — the RoPE smem race fix (synthesis bug 2) comes first.

## Sources (§6 verification)

- [NVIDIA forums — Zero Copy performance problem](https://forums.developer.nvidia.com/t/zero-copy-performance-problem/35132)
- [NVIDIA forums — 50% pinned memory limit on Windows 10/11](https://forums.developer.nvidia.com/t/change-limit-of-50-for-cudahostalloc-pinned-memory-on-windows-10-11/228235)
- [NVIDIA forums — unexpected cudaHostAlloc limits](https://forums.developer.nvidia.com/t/unexpected-limit-in-cudahostalloc-failing-to-allocate-large-amounts-of-pinned-page-locked-memory/19951)
- [Microway — Optimize CUDA GPU memory transfers](https://www.microway.com/hpc-tech-tips/optimize-cuda-host-to-device-transfers/)
- [Lei Mao — CUDA Zero-Copy Mapped Memory](https://leimao.github.io/blog/CUDA-Zero-Copy-Mapped-Memory/)
- [Microsoft — NonPaged pool size limits](https://learn.microsoft.com/en-nz/answers/questions/1372121/about-limitations-of-nonpaged-pool-size-in-window)
- [Stack Overflow — cudaHostAlloc limitations (WDDM vs TCC)](https://stackoverflow.com/questions/17090587/cudahostalloc-limitations)
- [EMOGI paper (zero-copy effective bandwidth example)](https://arxiv.org/pdf/2006.06890)
