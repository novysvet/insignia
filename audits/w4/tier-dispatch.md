# w4 — Tier dispatch ("the treadmill"): heterogeneous decode step for Qwen3.8-27B-FP8

Date 2026-08-25. Mission: the detailed per-step schedule for v2 (`L=19 / Z=21 / C=9 / N=15`,
≈1.75 s/step) with mixed engines — V VRAM/GPU, Z pinned-RAM/GPU-UVA, C locked-pageable/CPU,
N NVMe→ring/CPU (v2) or GPU-UVA (v1). Everything below is grounded in code that exists in
this tree: `src/streaming.cu` + `include/insignia_streaming.hpp` (NvmeReader / PinnedRing /
LayerFeeder, smoke-verified byte-exact on the real checkpoint — audits/w3/streaming-impl.md),
`include/insignia_cpu.hpp` (the full CPU tier + CpuPool), `src/fp8.cu`, `src/decode.cu`,
`src/generate.cu`, `src/qwen35.cu`. Reference clones present and read: `colibri/`,
`llama.cpp/`, `exllamav3/`, `ggml/`, `_mlx/` (via audits/w3/colibri-sched-deep.md, which
re-read them at file:line). Read-only audit; the only file written is this report.

Ground-truth inputs (MASTER-PLAN §1–2, all recomputed there from primary facts): per-layer
bytes lin 383.86 MB / full 372.31 MB / avg 380.97 MB; E: = 980 1TB Gen3, 3.3 GB/s effective
→ **115.4 ms/layer NVMe fill** (the binding resource); Z GPU-UVA @18 GB/s plan → **21.2
ms/layer**; C CPU @37 GB/s → **10.8 ms/layer**; V VRAM → 0.78/0.757 ms; lm_head VRAM sweep
5.4 ms; MTP draft (all-VRAM) 6.7 ms; pinned cap 8,531 MB hard / 8,048 planned; VRAM budget
11,300 MB committed (L=19 ⇒ 11,008 + 292 spare).

---

## 0. The one-paragraph design

One **sequencer thread T0** walks layers 0→63 in order, exactly like the current
`Qwen35Decode::forward_body()` (src/decode.cu:133) but dispatching each layer through a
baked-at-startup table instead of a single GPU path. V/Z layers are enqueued as CUDA kernels
on the compute stream (Z kernels take **host pinned pointers** — UVA zero-copy; V take
static device pointers). C layers and (in v2) N layers are computed by the existing
`insignia::cpu` kernels on the existing 6-worker `CpuPool` (include/insignia_cpu.hpp:217),
which T0 drives and **participates in** (`caller_helps=true`) — so T0 doubles as a GEMV
worker during CPU layers. N-tier weights arrive through the existing `LayerFeeder`
(`begin_epoch` / `acquire_layer` / `map` / `release_layer`, src/streaming.cu:373/430/452/443)
whose IOCP reader threads run flat-out, 2–3 slots ahead. The decode chain is **serial across
engines** (layer i+1 needs layer i's output — colibri-sched-deep §8.4: "layers alternate
engines but never split a layer"), so the *only* true concurrency is: reader ‖ everything,
copy-stream tails ‖ compute, MTP draft ‖ next-epoch reader fill. That is enough: the reader
(1,727 ms of NVMe floor) binds by 2.4× over the serial compute chain (719 ms + tails).

**Corrected v2 composition** (byte-exact, computed this session):

```
tiers    : V = layers 0–18   (4 full-attn at 3,7,11,15 — contiguous per MASTER-PLAN §2.5)
           Z = 21 layers, 8,003 MB pinned  (≤ 8,048 plan / 8,531 hard cap)
           C = 9 layers,  3,432 MB VirtualLock'd pageable
           N = 15 layers, 5.70 GB streamed/step from E:
19..63   : N C Z N Z C N Z Z N Z C N Z Z N C Z N Z C N Z Z N Z C N Z Z N C Z N Z C N Z Z N Z C Z Z N
           (N every ~3rd slot; first N at 19 establishes cadence; LAST N AT 63 — see §1.4)
engines   : eng_of[64] ∈ {GPU, CPU}:  V→GPU, Z→GPU-UVA, C→CPU, N→CPU (v2) / GPU-UVA (v1, ring pinned)
```

Simulated steady-state treadmill (reader paced, 15 ms head start carried from the previous
step, per §1.4): **T_step ≈ 1,736 ms** (paced N-waits 912 ms; NVMe floor 1,727 ms; GPU-serial
V 15.8 + Z 445.0 = 460.8 ms; CPU-serial 254.0 ms; PCIe 444.6 ms) → **0.58 tok/s single /
0.92 tok/s MTP-D1 (p=0.6) / ~1.44 MTP-D4** — reproduces MASTER-PLAN §2.4 v2 within rounding.

---

## 1. Per-step schedule — who issues layer i's work, when

### 1.1 Roles (4 thread classes, 9–10 threads total)

| role | threads | issues | blocking primitives |
|---|---|---|---|
| **T0 sequencer** | 1 | everything: per-layer kernel launches (compute stream), H2D/D2H activation handoffs (copy stream), `CpuPool::launch` for C/N layers, feeder acquire/release, lm_head/argmax/MTP | `cudaEventSynchronize` (spin), `CpuPool::launch` (blocks, caller participates), `LayerFeeder::acquire_layer` (blocks) |
| **NvmeReader workers** | 2 (default; 3 stretch) | `ReadFile` 2 MiB blocks, QD16, self-arming `top_up` (src/streaming.cu:210-220, 250-284) | `GetQueuedCompletionStatusEx` park |
| **CpuPool workers** | 6 | tickets of the current `launch()` generation (GEMV rows / GQA token-ranges / deltanet heads) | gen-tagged `claim_` CAS + cv (insignia_cpu.hpp:262-301) |
| **(GPU DMA engines)** | — | UVA SM reads for Z layers; copy-stream memcpys | stream order |

### 1.2 The layer walk (steady state, v2)

For layer `l` in 0..63, T0 executes exactly one of four cases (details in §7 pseudocode):

- **V** — `launch_layer_gpu(l, dev_ptr[l])`: kernels enqueued async on the compute stream,
  static pointers baked at startup. T0 does not wait; it proceeds to issue layer l+1
  immediately (stream order serializes correctly). Cost 0.78 ms; T0's issue loop runs ~15
  kernel launches in ~50 µs of host time — well under the 0.78 ms of GPU time per layer.
- **Z** — same as V but every `fp8_gemv/fp8_gemv2` takes the **host pinned pointer** of the
  Z copy (weights via UVA; scales from the VRAM smalls arena — §3). 21.2 ms/layer; GPU
  bandwidth-bound, T0 idles (spin) unless the next layer is CPU.
- **C** — boundary handoff GPU→CPU (§2), then `cpu_layer(l, host_locked_ptr[l])`: one
  `CpuPool::launch` per sub-op (`fp8_gemv_mt` per matrix ≈ 320–1088 tickets each,
  insignia_cpu.hpp:460-477); T0 participates. 10.8 ms/layer. Boundary handoff CPU→GPU after.
- **N** — `const void* slot = feeder.acquire_layer(idx_of(l))` (**blocks** until the slot is
  READY — this is the pacing wait, ~60 ms average, 912 ms/step total), then the same
  `cpu_layer(l, slot_ptr)` as C (v2; in v1: `launch_layer_gpu` with the slot pointer via
  UVA instead), then `feeder.release_layer(idx_of(l))` — which self-arms the next submit
  (src/streaming.cu:443-450 → arm_locked :399-408).

The reader is never told to "prefetch layer i+2" — its schedule **is** the epoch plan (the
N-tier shards in layer order); the ring window (`next_submit_ < released_ + slots`,
streaming.cu:400) keeps it exactly `slots−1` units ahead of the consumer, and
`on_done`→`arm_locked` (streaming.cu:416-428) re-arms on every unit completion. This is the
colibri §8.1 verdict implemented: dense + token-serial ⇒ the reader is a fixed cyclic list
run flat out; the consumer is throttled by blocking on `acquire_layer` (backpressure by
construction; colibri's "demand blocks, never drops" rule — colibri-sched §1.4).

### 1.3 Synchronization objects — complete inventory

| object | where | type | writer→reader |
|---|---|---|---|
| per-slot `st` + `wake` | `PinnedRing::SlotCtl` (insignia_streaming.hpp:183, `alignas(64)`) | `std::atomic<u32>` FREE/FILLING/READY/IN_USE + auto-reset event | reader `publish` (release store, streaming.cu:336) → consumer `acquire` (CAS + acquire, :341-351); consumer `release` (:353) → reader `try_claim` (CAS, :331) |
| feeder `mtx_`, `next_submit_`, `released_`, `arm_[]` | streaming.cu:373-450 | std::mutex + plain u32s (T0 sole release writer; on_done under same mutex) | guards the window invariant; `released_ = epoch_index+1` (:448) — strictly sequential contract |
| IOCP `port_`, per-unit `left` countdown + manual-reset `event` | streaming.hpp:108-120, 143-144 | kernel objects + atomics | last block completion sets event + queues callback (:197-206) |
| `outstanding_`, `fatal_`, `stop_` | reader | atomics | QD accounting; fatal publishes READY-with-garbage so blocked acquirers wake and get nullptr (:440) |
| `CpuPool::claim_` | insignia_cpu.hpp:333 | `std::atomic<u64>` (gen<<32 \| ticket) | dispatcher publishes gen (release), workers CAS-claim tickets — PipePool discipline (colibri.c:3294-3424) transplanted |
| activation handoff | **NEW** (this design) | pinned ping-pong `h_host[2][2][5120]` + 2 `cudaEvent_t` (`ev_h2d[2]`, `ev_gpu_out`) | GPU→CPU: event recorded after layer i's last kernel; T0 spin-waits. CPU→GPU: stream order (memcpy then kernels on same stream); the `__sync` flag the mission mentions is **not needed** because `CpuPool::launch` blocks T0 until every ticket returned (left==0, insignia_cpu.hpp:242-251) — the flag degenerates to "launch returned" |
| WDDM posture | — | `cudaSetDeviceFlags(cudaDeviceScheduleSpin)` at startup | T0 spins instead of parking on every sync (pcie-pipeline §6: WDDM submission spikes 1–2 ms; spin ≈ 5 µs vs 10 ms layer) |

### 1.4 Epoch lifecycle and the two scheduling rules that make the numbers

1. **Last N layer = layer 63.** With a naive Bresenham spread ending at ~61, the reader
   finishes the epoch plan and idles while the consumer walks the remaining Z/C layers +
   lm_head + MTP (~90–100 ms of reader idle = ~6% throughput). Putting the last N at 63 and
   the first N at 19 caps the non-N tail at lm_head + argmax + draft ≈ 12–13 ms.
2. **`begin_epoch(s+1)` fires at the last release, not at step end.** The instant T0 calls
   `release_layer(n−1)` (layer 63, mid-tail), the prior epoch is fully released and the
   feeder accepts `begin_epoch` (precondition checked at streaming.cu:375). The reader then
   gets a 12–15 ms head start on next step's slot 0 while the GPU finishes lm_head/MTP.
   Steady state carries this head start across steps (simulated: reader reaches step s+1's
   first N layer 15 ms "early"; consumer still waits ~55 ms for it — correct, the disk is
   the pace car).

Ring depth: **K=2 (2×368 MiB)** is the v2 default (RAM ledger leaves 694 MB spare;
MASTER-Plan §2.4). One slot being consumed + one being filled is sufficient because the
consumer of an N layer (CPU, 10.8 ms) is 10× faster than the fill (115.4 ms) — the consumer
never laps the reader, the reader never stalls on a busy slot. **K=3 stretch** (probe-gated,
+368 MB RAM → 13,174 MB total, 326 MB spare) adds jitter margin for HMB FTL variance on the
DRAM-less 980 (nvme-reader §1) — take it only if the R10 endurance/stability run shows
>1σ acquire-wait jitter.

### 1.5 Handoff cost budget (µs)

| boundary | mechanism | cost |
|---|---|---|
| GPU→CPU (before C/N layer) | `cudaEventSynchronize` (spin) on layer i's last kernel + D2H 20 KB pinned into `h_host[i&1]` on copy stream + event spin | event wake 2–5 µs + 20 KB/24 GB/s ≈ 0.8 µs + WDDM submit 5–10 µs → **≤ 20 µs** |
| CPU→GPU (after C/N layer) | `launch` returned ⇒ data visible; `cudaMemcpyAsync` H2D 20 KB pinned + kernels enqueued after it on the same stream (no extra sync for GPU-side ordering) | ~1 µs bus + 5 µs submit → **≤ 10 µs** |
| ping-pong reuse guard | before CPU writes `h_host[b]` again, T0 spin-checks `ev_h2d[b&1]` (recorded after the previous H2D) | 0 when GPU is ≥1 layer behind (always true — chain is serial) |
| slot acquire wake | `PinnedRing::acquire` spin (kAcquireSpin=65536 `YieldProcessor` ≈ 20–60 µs) then 20 ms event cap | free vs the ~60 ms paced wait it terminates |
| **total/step** | 31 GPU↔CPU switches (counted on the §0 interleave) × ~25 µs | **≈ 0.8 ms/step (0.05%)** |

---

## 2. The CPU↔GPU dependency chain

### 2.1 Activation buffers (ping-pong)

- **Device**: keep the current single-buffer in-place discipline (`x_.hidden` residual
  updated in place, decode.cu:128-129) — GPU kernels are stream-ordered, no ping-pong
  needed on device. Per-layer scratch stays in `DecodeWorkspace` (27B-sized in Phase B).
- **Host**: `h_host[2][2][5120]` f32 pinned (two buffers × two verify rows; 80 KB total —
  "host f32 [1..2,5120]" as briefed). CPU layers run in place on `h_host[b]` with ~0.5 MB
  of L2-hot scratch (norm 5120, qkv 10240, z 6144, gate/up 17408…), mirroring how
  `cpu::` kernels already operate on caller buffers. Buffer `b = cpu_boundary_count & 1`;
  the reuse guard is `ev_h2d[b]` from two boundaries ago (never fires in practice — the
  GPU consumes the buffer within one layer-time of the H2D).
- The existing pinned-scalars pattern (`next_host`, `pos_host`, `host_committed`,
  decode.cu:16-18) extends: add `h_host`, `logits_tail[2]` pinned, `embed_row[2][5120]`.

### 2.2 The two dependency edges per engine switch

1. **GPU layer i → CPU layer i+1**: T0 has already enqueued layer i; before launching CPU
   work it records `ev_gpu_out` after layer i's last kernel and spin-waits it, then enqueues
   the 20 KB D2H (20 KB × 2 rows for the T=2 verify path) on the copy stream and waits the
   copy event (~2 µs). Only then `cpu_layer()` reads `h_host[b]`.
2. **CPU layer i → GPU layer i+1**: `CpuPool::launch` returned (all tickets done — the
   caller-is-progress guarantee, insignia_cpu.hpp:229-252, means even a fully-parked pool
   cannot deadlock this edge). T0 issues `cudaMemcpyAsync(h_dev, h_host[b], …, H2D,
   compute_stream)` then layer i+1's kernels on the **same stream** — stream order makes
   them observe the copy; no flag, no fence, no event needed for correctness (the mission's
   "__sync flag for the host buffer" collapses into launch()'s return; keep a
   `std::atomic<u32>` done-counter only if a future variant moves CPU issue off T0).

### 2.3 State and KV ownership (follows the layer's engine)

- **DeltaNet recurrent state** (48×128×128 f32 = 3.15 MB/lin layer): device-side for V and
  Z layers, host-side (VirtualLock'd, 107.9 MB total — already in MASTER-Plan's RAM ledger)
  for C/N layers. No cross-device sync needed: the state's owner never changes tier
  mid-run (static placement), and each engine's kernels (`deltanet_decode` /
  `cpu::deltanet_step_cpu`, insignia_cpu.hpp:721) touch only its own copy.
- **KV caches**: V fulls → VRAM; Z fulls → **pinned host**, GPU `gqa_decode` reads K/V via
  UVA (8.39 MB @ctx2048 → 0.47 ms at 18 GB/s, +0.45 ms/layer ≈ +2.3 ms/step over VRAM —
  noise) and `store_kv` writes one 4 KB row via UVA per token (the one UVA *write*; 4 KB is
  far below the threshold where the non-coherent write path matters — pcie-pipeline §6).
  C/N fulls → locked pageable host KV, consumed by `cpu::gqa_decode_cpu`
  (insignia_cpu.hpp:925, token-range split so K/V streams once). MTP KV → VRAM (untouched).
- **Conv state**: 120 KB/layer, owner = layer's engine (conv is tiny either way).

### 2.4 What the chain forbids (be honest)

GPU and CPU never compute concurrently — the activation dependency is total (dense, single
sequence). The mission's "GPU layer i−1 executes while CPU computes layer i" is achievable
only in the degenerate sense that T0 enqueues layer i−1 and the *enqueue* of layer i's CPU
work overlaps the GPU's execution of i−1 by the ~10 µs it takes to detect "no more GPU
work until CPU returns". The genuine overlaps are: reader ‖ all compute; copy-stream
logits/embed tails ‖ compute; MTP draft ‖ next-epoch reader fill; and CPU-layer compute ‖
the GPU's *idle* (GPU sits parked during C/N layers — 26% GPU duty, 15% CPU duty; both
fine, the disk binds). Any deeper overlap requires batching tokens (verify T=2 already
gives it: weights read once, two rows — the second row is free bandwidth-wise) or multiple
sequences — out of scope.

---

## 3. UVA consumption of pinned/ring memory by the fp8 GEMV kernels

### 3.1 Verification from src/fp8.cu — the weight stream is UVA-clean

`fp8_gemv_kernel` (fp8.cu:14-51) touches the weight pointer in exactly one place:

```cpp
const uint4 packed = __ldcs(reinterpret_cast<const uint4 *>(row_w + c0));   // fp8.cu:32
```

- One 16-byte vector load per lane per 512-col round ⇒ 512 B per warp round: perfectly
  coalesced, sequential per row — **the near-best-case UVA pattern** (pcie-pipeline §6:
  coalesced 128 B-aligned streaming reads are the top of the 15–20 GB/s band). `__ldcs`
  (streaming hint) is legal on mapped host memory; weights are read exactly once so the
  uncached nature of zero-copy costs nothing.
- **Alignment is the sharp edge**: every 27B row stride (5120/6144/10240/12288/17408) is a
  multiple of 16, so 16 B alignment of the tensor base makes every `uint4` legal. Shard
  `data_start ≡ 8 (mod 16)` (2,600 for linear shards) puts raw F8 bases at ≡8 mod 16 →
  misaligned-address crash (MASTER-Plan risk #7). The streaming layer has **no pad/gap
  primitive today** (gap list in §6.4): `build_blocks_locked` (streaming.cu:121-150)
  concatenates physical extents gaplessly and `map()` derives logical offsets from the
  running cursor (streaming.cu:452-456). Required: an 8-byte pad at the BF16→F8 boundary,
  expressed either as a `dst_gap` field on `ReadRequest` or by letting INSIDX02's
  `in_slot_off` drive `map()` (the index already stores it — MASTER-Plan Phase A.2). Assert
  `(f8_base & 15) == 0` at acquire; for **pinned Z copies** simply re-base at startup load
  (one-time memmove, MASTER-Plan Phase G knob).
- **The x vector must stay in VRAM**: the smem staging loop loads x with `__ldg`
  (fp8.cu:17-22) — mapped-host `__ldg` works but is uncached and latency-bound; keep
  activations device-side (this design does; pcie-pipeline §6 caveat).

### 3.2 A finding: scales must NOT be UVA-read — keep them in VRAM

The per-round scale is a scalar bf16 load (fp8.cu:33, 77):
`__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(row_s + (c0>>7)))`. Same address
across the warp (one broadcast transaction), but re-read per 512-col round per row: for
`gate/up [17408,5120]` that is 17408×10 = 174 K independent 2 B loads per layer. Zero-copy
accesses bypass L1/L2, so every one of those is a PCIe round trip — ~19 MB of pure TLP
overhead plus latency pressure on exactly the loop we are trying to run at 18 GB/s.
**Design rule: the BF16 smalls arena (scales, norms, a/b, conv1d, A_log, dt_bias ≈ 65 MB
for all 64 layers) is VRAM-resident for GPU-consumed layers (V/Z and v1's N)** — 65 MB of
the 292 MB VRAM spare, and it deletes the entire class. CPU-consumed layers use their own
host copies with scales pre-widened once via `cpu::fp8_prepare_scales` (insignia_cpu.hpp:101).

### 3.3 Two launch-shape gaps found while verifying (fix in Phase C alongside F1/F5)

1. `fp8_gemv` (fp8.cu:52-55) launches with dynamic smem `cols*4` but never calls
   `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)`. At `cols=17408` (down_proj) that is
   69,632 B > the 48 KB default ⇒ **launch failure on exactly one 27B matrix**. Only
   `fp8_gemv2` has the 99 KB opt-in (fp8.cu:100). Add the attribute + launch-error checks
   to `fp8_gemv`.
2. `fp8_gemv2` **throws** at `cols=17408` (2·cols·4 = 139 KB > 99 KB, fp8.cu:98-99 — the
   throw is correct, the message says "use fp8_gemm"). The T=2 verify dispatch must route
   `down_proj` (and only it) to `fp8_gemm` (rows 5120 % 32 == 0 ✓; y must be 64-row padded
   — provide a 64×5120 f32 scratch, rows ≥ T never read, fp8.cu:182-184). All other pair
   matrices (cols 5120/6144/10240) fit gemv2.

### 3.4 Zero-copy read vs async memcpy to device — the verdict

| path | per-layer time | VRAM cost | end-to-end effect (v2) |
|---|---|---|---|
| **Z: read-only direct UVA** (chosen) | 381 MB @ 18 (15–20) GB/s = **21.2 (19–25.4) ms**; kernel time ≡ transfer time | **0** | GPU-serial 461 ms, hides under NVMe 1,727 ms with 3.8× slack |
| Z via copy stream (S-tier) | 15.2–15.9 ms copy + 0.78 ms VRAM GEMV | +381–768 MB staging (1–2 slots) = 1–2 L-layers | saves 5.2 ms × 21 = 109 ms of GPU-serial time that **is not binding**, pays +115–230 ms/step of NVMe (L drops) — strictly dominated (MASTER-Plan Appendix A.1) |
| N v1: UVA from pinned ring | 21.2 ms | 0 (ring already pinned in v1) | v1 chain 981 ms « 5,193 ms NVMe ✓ |
| N v2: CPU-compute from locked ring | 10.8 ms | 0 pinned (frees the cap for Z) | CPU-serial 254 ms ✓ 15% duty |

So: **read-only direct-from-pinned is correct for both Z and (in v1) N**. Do not put bulk
weight copies on the copy stream — beyond being dominated, WDDM's DMA batching would let a
15 ms monster memcpy starve the latency-sensitive logits D2H / embed H2D behind it
(pcie-pipeline §7.2). The copy stream carries only: activation ping-pong (20–40 KB),
logits tail (2 ints… or 2×994 KB if full logits are needed for sampling — keep argmax
device-side and D2H 2 ints, as `argmax_fast`+`next_host` already do, decode.cu:135), delta
tails (none — states don't migrate, §2.3), embed rows (20 KB).

The 21.2 ms/layer estimate is consistent: 380.97 MB / 18 GB/s = 21.2 ms; the kernel keeps
thousands of 16 B loads in flight (8 warps/block × grid rows/8 ⇒ ~1,280 blocks at
[10240,5120]), saturating the 15–20 GB/s band (pcie-pipeline §6). Verify at startup with
the 200 ms UVA microbench (Phase G solver input), C-shift as the pressure valve.

---

## 4. MTP draft (all-VRAM ≈ 6.7 ms) + T=2 verify — schedule sketch

Steady-state spec step (D=1/T=2, same weight pass for both rows — bandwidth-bound, second
row free; `fp8_gemv2` on GPU / `cpu::fp8_gemv2_mt` on CPU, insignia_cpu.hpp:470):

```
t0        last N layer (63) released -> feeder.release_layer -> begin_epoch(s+1)   [reader gets 12-15 ms head start]
t0+0      final rms_bf16 + lm_head bf16 GEMM T=2 (VRAM, 5.4 ms) + merged argmax (2 launches)
t0+5.7    D2H [next2,next] (2 ints, copy stream) ; spec_commit / spec_rollback kernels  (decode.cu:225-226 semantics)
t0+6      mtp_layer() draft — VRAM-only (fc bf16 0.2 + attn layer 0.7 + lm_head 5.4 + ε ≈ 6.7 ms)   (decode.cu:137-192 shape)
          ‖ reader filling step s+1 slot 0 (≈ 58 MB of it during the draft)
t0+12.7   embed rows for [pending, draft] pread (10 KB ×2, buffered twin / mmap page cache, ~1 µs warm)
          + H2D into embed_row staging; step ends; loop to layer 0 (all-VRAM band, 14.8 ms) — reader still filling
```

The draft hides under the stream exactly as MASTER-Plan §1.3 claims — not by overlapping
*compute* (nothing to overlap; the chain is done), but by giving the reader a continuous
work queue across the tail. The verify's T=2 rides the *same* treadmill with zero extra
I/O: every GEMV/GEMM reads weights once for both rows (`fp8_gemv2` fp8.cu:58-95,
`cpu::fp8_gemv2_rowrange` insignia_cpu.hpp:408-447 — dequant once, two FMA chains). D=4/T=5
(MASTER-Plan Phase F.2, +300 LOC) reuses this skeleton unchanged: drafts chain on the GPU
back-to-back (~27 ms), still hidden under 1.7 s of stream; only the snapshot buffers grow
(host-pinned 628 MB). SPEC_PIN rule applies: draft T=1 and verify T≥2 must stay on the same
kernel family (colibri #163 — accumulation-order divergence collapses acceptance), i.e. use
`fp8_gemv2`-shaped math for the draft row too (it already is: single-row lm_head GEMV +
`argmax_fast` are the same kernels the verify uses row-wise).

Watchdog: all kernels ≤ 21.2 ms ≪ 2 s TDR — no WDDM watchdog exposure (pcie-pipeline §7.3).
The 60–900 ms human-scale waits live on the *host* (`acquire_layer`), never in a kernel.

---

## 5. Failure / lifecycle

### 5.1 Ring underrun (reader slower than consumption) — throttle policy

- **Structural backpressure**: consumption rate is limited by `acquire_layer` blocking —
  the decoder throttles itself; there is nothing to drop (dense, no speculation — the
  colibri pilot/hint machinery is deliberately absent, colibri-sched §8.1). This is the
  steady state, not an error: 912 ms/step of paced wait in the sim.
- **Transient slow disk** (HMB FTL variance, background traffic): absorbed by ring depth
  (K=2 covers one 115 ms fill vs ~56 ms average consumption cadence; K=3 stretch doubles
  jitter margin).
- **Watchdog + telemetry** (add — gap, §6.4): accumulate per-step `acquire_layer` wait
  time; if step wall-clock > 1.5× predicted, log `Data Units Read` SMART delta; > 3× ⇒
  run the health probe and stop with a diagnostic (a dying 980 is the expected failure
  mode at 12 TB/h — nvme-reader §5.1). Hard reader failure: 3 bounded retries per block
  (`kMaxTries`, streaming.hpp:104) ⇒ unit failed ⇒ `fatal_` ⇒ failed slot still publishes
  so blocked acquirers wake and `acquire_layer` returns **nullptr** (never garbage, never
  hang — streaming.cu:199, 440); T0 aborts the step with an exception.
- **The opposite (reader faster)**: impossible to matter — window math caps read-ahead at
  slots−1; a full ring blocks submits implicitly (`arm_locked` stops claiming).

### 5.2 VRAM OOM probe (startup, before any allocation)

`cudaMemGetInfo` → require free ≥ 11,073 MB (11,008 + 65 smalls arena) for L=19; ladder
down one layer at a time (each drop frees 372–384 MB): L=18 ⇒ the demoted layer becomes Z
(pinned 8,003→8,377 ≤ 8,531 ✓, but re-check the plan cap 8,048 — else C). Degradation is a
**manifest re-solve** (tier tables are data — AGENTS "modify constant data directly"), no
engine code path changes. Mid-run VRAM pressure cannot occur: static placement, no CUDA
graphs at 27B (MASTER-Plan §2.2), no VRAM churn, copy stream carries KB-scale traffic.

### 5.3 Pinned-cap probe ladder (startup)

Allocate 1 GiB `cudaHostAlloc` chunks until failure (or one chunked arena with running
total); usable_pinned = floor(success_total − margin). Expected ≈ 8,531 MB hard. If
usable_pinned < manifest Z bytes: shift excess Z→C (engine flip GPU→CPU per layer; scales
move VRAM-arena → host `fp8_prepare_scales` arrays; state/KV owner flips per §2.3).
Allocation **order** matters on the cap: Z arena first, ring second — then if the cap is
exhausted the ring's `cudaHostRegister` fails cleanly into its existing VirtualLock
fallback (streaming.cu:305-312) which is *exactly what v2 wants anyway* (the ring must not
consume pinned cap in v2). Make it explicit with a ring mode flag rather than relying on
failure (§6.4).

### 5.4 Teardown (verified order)

1. Step loop exits (EOS / max_new); T0 holds no slot (last release done).
2. `LayerFeeder` dtor: `reader_.shutdown()` FIRST — stop flag → `CancelIoEx` per handle →
   wake posts → ABORTED completions reaped → join ≤5 s (`TerminateThread` last resort) —
   callbacks suppressed under `stop_` so nothing fires into the dying feeder; then members
   destroy in order (ring `cudaHostUnregister` / `VirtualUnlock` / `VirtualFree`)
   (streaming.cu:364-371, 58-87, 321-329). Smoke-tested with units in flight
   (streaming-impl §6).
3. `CpuPool` static dtor joins its 6 workers (insignia_cpu.hpp:326-329).
4. `DecodeWorkspace` dtor: `cudaStreamSynchronize` as first line (safety C5 fix, Phase 0),
   then frees (decode.cu:29).
5. Optional `cudaDeviceReset`. GPU contexts tear down after the ring is unregistered
   (ordering guaranteed by 2-before-4).

---

## 6. Thread + affinity map; LayerFeeder API mapping and gaps

### 6.1 Affinity map (5600X: LP0-5 physical primaries, LP6-11 SMT siblings)

| thread(s) | mask | priority | source / rationale |
|---|---|---|---|
| T0 sequencer | **0x3 (LP0-1)** | ABOVE_NORMAL | pcie-pipeline §7.1 ("T0+GPU driver cores 0-1"); WDDM user-mode submits originate on T0 — keep them off SMT siblings for launch latency; `cudaSetDeviceFlags(ScheduleSpin)` so T0 never parks mid-step |
| CpuPool workers (6) | 0x3F (LP0-5), one per core | normal | existing: insignia_cpu.hpp:319-323. **T0 intentionally shares LP0-1 with workers 0-1**: during C/N layers T0 is a compute participant (`caller_helps`); during V/Z layers the workers are parked (spin 4096 pauses ≈ 10-20 µs then cv — insignia_cpu.hpp:285-291) and T0 owns the pair |
| NvmeReader workers (2, stretch 3) | **0xFC0 (LP6-11)** | ABOVE_NORMAL | existing: streaming.cu:32, 47-49. They sleep in `GetQueuedCompletionStatusEx`; stornvme DPCs land on the issuing (sibling) cores, off the GEMV primaries (nvme-reader §7). ABOVE_NORMAL, never HIGHEST/TIME_CRITICAL — DPC/ISR starvation is the anti-pattern on an I/O-bound box |
| GPU DMA / copy stream | — | — | stream order; only KB-scale traffic (§3.4) |

WDDM notes: no TDR exposure (§4); submission spikes (1-2 ms) absorbed by the ring and by
spin handoffs; `ScheduleSpin` + `_mm_pause` spins keep every wait under ~60 µs except the
deliberate paced waits.

### 6.2 LayerFeeder → treadmill mapping (what maps 1:1 today)

| treadmill need | existing API | file:line |
|---|---|---|
| N-tier plans per step | `begin_epoch(std::vector<ReadPlan>)` (N shards in layer order; whole-shard `[0, size)` requests; gapless prefix ⇒ tensor ptr = slot base + abs offset) | streaming.cu:373-393; nvme-reader §2.3 |
| pacing wait + slot ptr | `acquire_layer(i)` (spin+event; nullptr on fatal) | streaming.cu:430-441 |
| tensor pointers | `map(i, r)` / `map_[e]` precomputed offsets | streaming.cu:452-456 |
| free + read-ahead re-arm | `release_layer(i)` → `released_=i+1` → `arm_locked` (window `next_submit_ < released_+slots`) | streaming.cu:443-450, 399-408 |
| reader continuity across steps | `begin_epoch` called at last release (§1.4 rule 2) | — |
| teardown | dtor ordering (reader first) | streaming.cu:364-371 |
| CPU consumption of ring slots | works as-is: slot memory is readable in place (pinned or VirtualLock'd) | streaming.cu:297-312 |

### 6.3 Engine-side integration points that already exist

- `Qwen35Weights::matrix()` already returns `WKind{fp8, bf16}` (qwen35.cu:19-28) — the
  dispatch on kind is the hook for tier/engine routing (Phase C.3).
- `Qwen35Decode::layer/delta_layer/attention_layer` (decode.cu:126-132) is the loop to
  fork per tier; `mtp_layer` (:137-192) and `spec_step` (:219-237) are the tail; the eager
  path (no `capture_*`) is what 27B uses (MASTER-Plan Phase D.4).
- The whole `insignia::cpu` layer body already mirrors the GPU flow op-for-op
  (cpu-fp8 §4 wiring sketch ≈ decode.cu:126-130).

### 6.4 Gaps (ranked; all are Phase D line items)

1. **Plan-level padding/rebasing** (blocks the 16 B `uint4` UVA loads): no gap primitive in
   `ReadPlan`/`build_blocks_locked` (streaming.cu:121-150); add `u64 dst_gap` to
   `ReadRequest` or drive `map()` from INSIDX02 `in_slot_off`. **Blocking.**
2. **`PinnedRing` forced-locked mode**: ctor always tries `cudaHostRegister` first
   (streaming.cu:302-304); v2 needs "deliberately VirtualLock" so the cap is provably
   reserved for Z (and no stderr noise). Add ctor mode `try_pin|force_locked`.
3. **Cross-epoch read-ahead**: `begin_epoch` requires the prior epoch fully released
   (streaming.cu:375) — fine with the §1.4 rule, but a queued `begin_next_epoch(plans)`
   (arms automatically at full release) would make the head start structural instead of
   call-site-timed. Optional.
4. **Acquire telemetry/timeout** for the §5.1 watchdog + R10 (`tok/s ≥ 80% predicted` gate).
5. **`read_once`** (buffered one-shots: MTP shard 477 MB, embed rows, outside text) — the
   streaming layer only has ring submits; nvme-reader §2.3 mandates one-shots stay off the
   ring (`span > slot_bytes` is even asserted against, streaming.cu:389-390).
6. **`fp8_gemv` smem opt-in** (§3.3-1) + pair-routing for `down_proj` (§3.3-2) — engine
   kernel gap surfaced by this audit, not a streaming gap.
7. Prefill sweep holds: `release_layer`'s strictly-sequential bookkeeping tolerates long
   holds only within the window; add the documented `sweep_acquire` alias when Phase F
   lands (streaming-impl §7.2).
8. Dual-drive per-file QD (OPTION-G only; streaming-impl §7.1).

---

## 7. Step-function pseudocode (v2; ~100 lines, real names)

```cpp
// tier tables baked at startup from the placement manifest + probe ladder (Phase G solver)
//   mem_of[64] ∈ {V, Z, C, RING}; eng_of[64] ∈ {ENG_GPU, ENG_CPU}; idxN[l] = feeder epoch index
struct Treadmill {
    insignia::LayerFeeder feeder;                 // 2 x 368 MiB slots, 2 readers (zero_copy mode)
    float *h_host[2][2];                          // pinned ping-pong [buf][T] x 5120
    float *h_dev;                                 // device activation (in-place, as x_.hidden)
    cudaEvent_t ev_out, ev_h2d[2];                // GPU->CPU fence; H2D ping-pong reuse guard
    cudaStream_t cs, ks;                          // compute, copy
    int cpu_bnd;                                  // ping-pong cursor
    // resident pointers: dev_v[l] (VRAM), pin_z[l] (pinned), lock_c[l] (pageable), plus the
    // VRAM smalls arena (scales/norms/a/b/conv/A_log/dt_bias for V+Z layers, 65 MB)
};

int step_spec(Treadmill& T, Qwen27Weights& W, DecodeWorkspace27& x, int pending, int draft) {
    // ---- verify forward, T=2 rows [pending, draft], weights read once (pair kernels) ----
    embed_rows_to_dev(T, x, pending, draft);                        // 2x10 KB pread'd a step ahead, H2D on ks
    if (!T.armed_next) T.feeder.begin_epoch(W.nvme_plans());        // normally armed at last release (§1.4);
    T.armed_next = false;                                           // begin_epoch is NOT idempotent (re-arming
    int e = 0;                                                      // resets next_submit_ -> double submit)
    for (int l = 0; l < 64; ++l) {
        const bool cpu = T.eng_of[l] == ENG_CPU;
        if (cpu && T.eng_of[l-1] == ENG_GPU) {                      // GPU -> CPU edge (§2.2-1)
            cudaEventRecord(T.ev_out, T.ks);
            cudaEventSynchronize(T.ev_out);                          // spin (ScheduleSpin)
            cudaMemcpyAsync(T.h_host[T.cpu_bnd&1], x.hidden, 2*5120*4, cudaMemcpyDeviceToHost, T.cs);
            cudaEventRecord(T.ev_h2d[T.cpu_bnd&1], T.cs); cudaEventSynchronize(T.ev_h2d[T.cpu_bnd&1]);
        }
        if (T.mem_of[l] == RING) {                                   // N layer: paced acquire
            const void* slot = T.feeder.acquire_layer(T.idxN[l]);
            if (!slot) throw std::runtime_error("feeder fatal");
            layer_cpu(T, W, l, (const u8*)slot, T.h_host[T.cpu_bnd&1]);   // v1: layer_gpu_uva(...) instead
            T.feeder.release_layer(T.idxN[l]);
            if (T.idxN[l] == W.nvme_plans().size()-1) {
                T.feeder.begin_epoch(W.nvme_plans());                // LAST-RELEASE rule (§1.4-2)
                T.armed_next = true;
            }
        } else if (cpu) {
            layer_cpu(T, W, l, T.lock_c[l], T.h_host[T.cpu_bnd&1]);  // C layer
        } else {
            const u8* w = (T.mem_of[l] == V) ? T.dev_v[l] : T.pin_z[l];
            layer_gpu(T, W, l, w, x.hidden, T.ks);                   // fp8_gemv/fp8_gemv2 w/ UVA ptrs
        }
        if (!cpu && T.eng_of[l+1] == ENG_CPU)                        // CPU -> GPU edge (§2.2-2)
            cudaMemcpyAsync(x.hidden, T.h_host[T.cpu_bnd++ &1], 2*5120*4, cudaMemcpyHostToDevice, T.ks);
    }                                                                // stream order: next kernels see it
    // ---- tail: lm_head + argmax + commit + draft, reader already filling next epoch ----
    rmsnorm_bf16(x.hidden, W.model_norm_w, x.norm, 2, 5120, true, T.ks);          // zero-centered (R4)
    bf16_gemm(W.lm_head, x.norm, x.logits, 248320, 5120, 2, T.ks);                // one sweep, both rows
    argmax_rows(x.logits, 248320, 2, x.next2_dev, T.ks);
    spec_commit_T(x.pos_dev, x.committed, T.ks); spec_rollback_T(..., T.ks);      // decode.cu:225-226 semantics
    cudaMemcpyAsync(x.tail_host, x.next2_dev, 2*sizeof(int), cudaMemcpyDeviceToHost, T.cs);
    mtp_layer27(W, x, T.ks);                                                      // 6.7 ms VRAM draft ‖ reader
    cudaStreamSynchronize(T.cs);
    return x.tail_host[1];                                                        // next pending
}

// one CPU layer (linear-attn flavor; full-attn swaps the middle block) — all real fns:
void layer_cpu(Treadmill& T, const LayerBlob& b, float* h /*[2][5120] in place*/) {
    using namespace insignia::cpu;
    rmsnorm_cpu(h, b.in_ln_w, T.n, 5120, true);
    fp8_gemv2_mt(b.qkv_w, b.qkv_s256, T.n, T.y_qkv, 10240, 5120);                 // pair: one weight pass
    fp8_gemv2_mt(b.z_w,   b.z_s256,   T.n, T.y_z,    6144, 5120);
    bf16_gemv_mt(b.a_w, T.n, T.a, 48, 5120); bf16_gemv_mt(b.b_w, T.n, T.b, 48, 5120);
    deltanet_parameters_cpu(T.a, T.b, b.A_log, b.dt_bias, 48);
    causal_conv4_silu_cpu(T.y_qkv, b.conv_state, b.conv_w_f32, 10240);
    deltanet_step_cpu(b.delta_state, T.y_qkv, T.y_qkv+2048, T.y_qkv+4096, T.a, T.b, T.core, 48, 3);
    gated_rmsnorm_per_head_cpu(T.core, b.norm_w, T.y_z, T.core, 48, 128);
    fp8_gemv2_mt(b.out_w, b.out_s256, T.core, T.dn, 5120, 6144);  residual_add_cpu(h, T.dn, 5120);
    rmsnorm_cpu(h, b.post_ln_w, T.n, 5120, true);
    fp8_gemv2_mt(b.gate_w, b.s, T.n, T.g, 17408, 5120); fp8_gemv2_mt(b.up_w, b.s, T.n, T.u, 17408, 5120);
    silu_mul_cpu(T.g, T.u, T.g, 17408);
    fp8_gemm_cpu_or_split(b.down_w, b.down_s256, T.g, T.dn, 5120, 17408);         // cols=17408: 2 cols-block passes
    residual_add_cpu(h, T.dn, 5120);
}
// GPU Z layer middle is the existing body of delta_layer (decode.cu:126-130) with
// linear()/linear2() routed to fp8_gemv / fp8_gemv2 (fp8.cu:52/96) on host-pinned weight
// pointers; down_proj pair -> fp8_gemm (fp8.cu:186) per §3.3-2.
```

---

## 8. Numbers recap + risks

| quantity | value | check |
|---|---|---|
| placement | V19 (0-18) / Z21 = 8,003 MB pinned / C9 = 3,432 MB locked / N15 = 5.70 GB per step | ≤ 8,048 MB pin plan; RAM 12,806 MB ≤ 13,500 |
| T_step (sim, reader-paced) | **1,736 ms** (NVMe floor 1,727; paced waits 912) | MASTER-Plan 1.63–1.75 s ✓ |
| GPU-serial / CPU-serial / PCIe | 461 / 254 / 445 ms | all ≤ 0.27× NVMe ✓ |
| tok/s | 0.58 single / 0.92 MTP-D1 / ~1.44 MTP-D4 (p=.6) | MASTER-Plan v2 ✓ |
| GPU↔CPU handoffs | 31 × ≤25 µs ≈ **0.8 ms/step** | §1.5 |
| ring | 2×368 MiB (K=2), stretch K=3 probe-gated | §1.4 |
| reader idle/step | ~0 (last-N-at-63 + begin_epoch-at-last-release) | §1.4 — was ~100 ms naive |

Top risks with designed catches: (1) F8 16 B misalignment in ring slots — pad primitive +
`(base&15)==0` assert (R3); (2) UVA scale re-reads — smalls arena in VRAM (§3.2); (3)
`fp8_gemv` smem launch failure at cols=17408 (§3.3-1) — Phase C fix; (4) WDDM pinned-cap
probe returning < manifest — Z→C shift ladder (§5.3); (5) disk endurance/health at 12 TB/h —
SMART-per-session logging + watchdog (§5.1). Parity gates R3→R10 (MASTER-Plan §3 Phase E)
remain the correctness spine; none of this ships before R9.
