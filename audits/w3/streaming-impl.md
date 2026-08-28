# w3 — NVMe→pinned-RAM streaming layer: implementation report

Date 2026-08-25. Deliverable: `include/insignia_streaming.hpp` (249 lines) +
`src/streaming.cu` (555 lines, host-only, compiled as .cu for the CUDA runtime
link). Compile-verified for sm_89 via the known-good nvcc path; smoke-verified
byte-exact against a plain buffered read on the real checkpoint. **Not yet
wired into the engine** — this is the standalone module the TieredStorage v2
integration will consume.

- Build check (no smoke): `nvcc -ccbin <MSVC 14.51> -arch=sm_89 -O3 -std=c++20
  -Iinclude -c src\streaming.cu -o build\streaming-check.obj` → clean, 0
  warnings/errors. Artifact left at `build\streaming-check.obj` (NOT installed
  into mk.py's cache; named to avoid collisions).
- Smoke: same command + `-DINSIG_STREAMING_SMOKE ... -o build\streaming-smoke.exe`,
  run once from the repo root (default path `Qwen3.8-27B-FP8\layers-0.safetensors`).
  The smoke `main` is `#ifdef`-guarded; the TU compiles clean without it.

## 1. Smoke results (the one live run)

```
[smoke] feeder up: 2 slots x 80 MiB, cuda_pinned=1
[smoke] round 0 slot0: 2x32 MiB MATCH MATCH (1.79 GiB/s fill, span 67117056 B)
[smoke] round 0 slot1: 64 MiB MATCH
[smoke] round 1 slot0: 2x32 MiB MATCH MATCH (2.44 GiB/s fill, span 67117056 B)
[smoke] round 1 slot1: 64 MiB MATCH
[smoke] destroying feeder with units in flight (CancelIoEx teardown path)
SMOKE PASSED
```

What that exercises: two-request plans concatenated into one slot (logical
data at in-slot offset 2600 = the shard header phase), 64 MiB requests with
the last 8 KiB block non-2MiB, second slot read-ahead while slot0 is consumed,
epoch re-arm (round 1 = handle recycling + slot reuse), and teardown with
units in flight (CancelIoEx → ABORTED completions → clean join, no 5 s stall).
`cudaHostRegister` succeeded (ring is true pinned memory, DMA-able). Fill rate
on the 67 MB span is 1.8–2.4 GiB/s — this is a correctness smoke, not the
throughput bench; the span is small (33 blocks; QD16 never fully ramps) and the
980 1TB is DRAM-less. The dedicated bench (nvme-reader.md §6 io_bench matrix,
acceptance E: ≥ 3.0 GB/s on a 20 GB span) is still the gate for rate claims.

Two real bugs were found and fixed during bring-up (both would have been
annoying later; both now encoded in comments):
1. `submit()` popped a recycled handle whose `Unit` had been `reset()` to null
   by `retire()` → null-deref on the second epoch. Fix: `units_[h].reset(new
   Unit{})` on the free-list path.
2. Stale-handle ambiguity: `retire()`→`submit()` recycles stream handles, so
   `on_done`'s search of `arm_[]` could match a stale entry (the recycled
   slot's *previous* epoch) and publish a slot that is mid-fill for the new
   epoch — wrong-epoch corruption. Fix: descending search + clear the matched
   `arm_[e].stream` (0xFFFFFFFF) after publish; feeder-mutex ordering makes the
   clear strictly precede any reuse.

## 2. API

```cpp
struct ReadRequest { const wchar_t* path; uint64_t offset; uint64_t len; }; // logical extent
using ReadPlan = std::vector<ReadRequest>;  // one plan = one contiguous ring region
```

Alignment math is shared by reader/feeder/consumers (single source of truth,
header): physical extent = `[align_down_4096(off), align_up_4096(off+len))`
clamped to EOF; logical bytes at region_base + `req_head(r)` = `off &
4095`. **The reader consumes generic (file, offset, len) plans only — zero
knowledge of INSIDX/INSIDX02.** Rounding: offsets down, lengths up, to 4096
(satisfies 512e and 4Kn). Only an EOF-clamped final chunk can be sub-512B; that
tail is read synchronously through a buffered twin handle at plan-build time
(colibri twin-tail pattern). A request whose logical extent exceeds the file
size is rejected at submit (strict caller contract).

### NvmeReader (ctor: `NvmeReader reader(2)`)
- `u32 submit(const ReadPlan&, void* ring_slot, DoneFn = nullptr, void* ctx = nullptr)`
  → stream handle. Files opened on demand and cached (direct
  `FILE_FLAG_NO_BUFFERING|FILE_FLAG_OVERLAPPED` + buffered twin + size + IOCP
  association per file). Blocks: 2 MiB, FIFO stream order = submit order (disk
  stays sequential), global QD = 16 outstanding blocks (≥ 8×2 MiB mandated)
  maintained by a self-arming `top_up()` after every completion batch.
- `ready(h)` / `wait(h)` / `done_event(h)` (manual-reset) / `retire(h)`
  (frees the handle for recycling; only when done).
- Completion callback `void (*)(void* ctx, u32 stream, bool ok)` fired once per
  unit on a reader thread, outside all locks. Bounded retry: 3 attempts per
  block (sync-fail loop + async-error reissue), same destination.
- `healthy()`, `shutdown()` (idempotent: stop flag → CancelIoEx per handle →
  PostQueuedCompletionStatus wake → join 5 s → TerminateThread last resort).

### PinnedRing (ctor: `PinnedRing(bytes, slot_count)`)
- `VirtualAlloc` (MEM_RESERVE|COMMIT) + `cudaHostRegister(cudaHostRegisterDefault)`
  after `cudaSetDevice(0)`; slot bytes = floor(bytes/slots) rounded to 4096.
  WDDM pins at ~50% of RAM (~7.95 GiB) — the 4×368 MiB default sits far under.
  Registration failure → VirtualLock-per-slot fallback (CPU GEMV unaffected;
  H2D pays a staging bounce) + one stderr line.
- Per-slot atomic state machine (one cache line per slot); `try_claim` /
  `publish` / `acquire` / `release` / `state`; `acquire` spins 65536
  `YieldProcessor` iterations then waits on an auto-reset event with 20 ms
  timeout re-check (no lost-wakeup hang possible).

### LayerFeeder (owns one NvmeReader + one PinnedRing)
```cpp
LayerFeeder feeder;                                     // 4 slots x 368 MiB, 2 reader threads
feeder.begin_epoch(plans);                              // decode epoch: N-tier shards in layer order
const void* slot = feeder.acquire_layer(i);             // blocks until slot READY (nullptr on fatal)
const void* w    = feeder.map(i, r);                    // logical ptr of request r (between acquire/release)
feeder.release_layer(i);                                // IN_USE -> FREE, arms next submit
```
- `begin_epoch` requires the prior epoch fully released (throws otherwise);
  recomputes per-epoch `map_[]`/`span_[]`, asserts `span ≤ slot_bytes`
  (mtp/outside one-shots do NOT belong in the ring — they are read_once
  material per nvme-reader.md §2.3), then arms.
- Read-ahead depth = slots−1: submit(epoch e) allowed while
  `e < released + slots`. Slots assigned cyclically (`e % slots`); the window
  invariant guarantees the previous occupant is released before re-claim (a
  failed claim is treated as fatal — it means a caller bug).
- Consumer contract: **strictly sequential** acquire 0..n−1 then release
  (decode is token-serial and layer-serial — baked in, per AGENTS spirit; an
  out-of-window acquire returns nullptr instead of hanging on a wrong epoch's
  slot).
- `ConsumeMode { zero_copy, copy_out }` — v1 returns the pinned pointer for
  both; the engine decides at the call site (CPU GEMV reads the slot in place,
  or cudaMemcpyAsync's from it). `copy_out` is the reserved hook where acquire
  will additionally issue the slot→VRAM chunked copy chain (pcie-pipeline.md
  §7.2 STREAM_FEED, dormant).

## 3. State diagrams

Ring slot (atomic, one cache line per slot):

```
                 try_claim (CAS)                publish (release store)             acquire (spin+event, CAS)
   FREE ───────────────────────────▶ FILLING ───────────────────────────▶ READY ───────────────────────────▶ IN_USE
    ▲                                                                       (reader DMA + tail complete)      │
    └────────────────────────────────────────────────────────────────────────────── release (release store)──┘
```
Transitions are written by exactly one role each: reader claims/publishes,
consumer acquires/releases. Data visibility: DMA writes precede the completion
that triggers `publish` (release); the consumer's `acquire` CAS is the
acquire-side pair — slot bytes are safe to read/GEMV/memcpy immediately.

Reader stream unit (guarded by the reader mutex):

```
   submit: build blocks (open files on demand, split extents, twin-tail read)
      └─▶ ISSUING ──(top_up: while outstanding<16, next block)──▶ ISSUED
              │  each ReadFile completion: success → left--; failure → retry ≤3
              ▼                                                   └─3 strikes→ failed=true, left--
   left==0 ──▶ DONE(ok=!failed): SetEvent, queue callback ──▶ (worker fires cb outside lock) ──▶ retire(h) frees handle
```

LayerFeeder epoch:

```
   begin_epoch ──arm──▶ submit(e) ──reader fills slot e%slots──▶ on_done: publish + retire + re-arm
   consumer: acquire_layer(i) [block] ─▶ compute from map(i,r) ─▶ release_layer(i) ─▶ released_=i+1 ─▶ arm next
   window: next_submit_ < released_ + slots        (read-ahead = slots−1)
```

## 4. TieredStorage v2 integration contract (loader-gaps.md §3.3)

For **nvme-tier layers**, TieredStorage2 maps onto the feeder 1:1:

| TieredStorage2 | streaming module | Notes |
|---|---|---|
| placement manifest → ordered list of N-tier layer shards | `std::vector<ReadPlan>` built once at load (one plan per N-tier layer, requests = whole-shard `[0, file_size)` or per-tensor extents; both work) | paths must outlive the epoch (wchar_t* are borrowed, not copied) |
| `poll_completions()` / io thread | NvmeReader worker threads | storage v2's "exactly one io thread" becomes "2 reader threads" here — see §5 |
| `prefetch(name)` | implicit: `begin_epoch` + self-arming top-up keeps slots−1 units ahead | decode's prefetch is structural, not name-driven (dense, cyclic) |
| `acquire_host_blocking(name)` for an N-tier layer | `acquire_layer(epoch_index_of(layer))` + `map(epoch, request)` for each tensor window | returns pinned host view; CPU kernels read in place |
| `acquire_blocking(name)` (vram-tier upload of an N-fed layer) | `acquire_layer` + engine-side `cudaMemcpyAsync` from the pinned slot (real DMA — this is why the ring is cudaHostRegister'd, not just VirtualLock'd) | the `copy_out` ConsumeMode hook will absorb this |
| `release(name)` | `release_layer(epoch_index_of(layer))` | pin decrement ≡ IN_USE→FREE |
| graph-capture rule | unchanged: N-tier layers break the graph (slot pointers are per-epoch) | pcie-pipeline.md §7.2 |

In-slot tensor addressing: tensor at `slot + in_slot_off` where `in_slot_off =
req_head(request) + Σ prior req_spans` — precomputed per epoch in `map_[]`;
equivalently `map(epoch, request_index)`. INSIDX02's `in_slot_off` for
whole-shard plans equals the safetensors absolute offset (shard reads start at
byte 0), so consumers add `abs_off` to the slot base with zero remapping.

## 5. Thread / affinity budget

| Thread | Count | Affinity | Priority | Job |
|---|---|---|---|---|
| NvmeReader workers | 2 | LP 6–11 (mask 0xFC0, SMT siblings; non-fatal if it fails) | ABOVE_NORMAL | GQCSEx batches of 16, 1 s poll; process completions, retry, top_up, fire callbacks |
| Consumer (decode T0) | engine's | LP 0–5 with the GEMV team (engine side) | engine's | begin_epoch / acquire / release + compute |

Rationale (nvme-reader.md §7): GEMV threads own the physical primaries;
reader threads sleep in GQCSEx and cost ~nothing on siblings; ABOVE_NORMAL
preempts normal-priority noise without starving DPC/ISR (stornvme completions
land on the issuing cores = siblings, off the compute primaries). All lock
ordering is feeder-mutex → reader-mutex (callbacks run outside reader locks),
so no inversion is possible between consumer and reader threads.

## 6. Logical review checklist (no unit-test harness; verified by review + smoke)

- **State machine transitions**: every ring transition is a CAS or a
  release-store by a unique role (claim/publish = producer, acquire/release =
  consumer). No FREE→READY or FILLING→IN_USE shortcut exists. Missed-event
  waits are impossible: acquire re-checks the atomic after every 20 ms timeout.
- **Handle lifecycle**: a stream handle is live for exactly one unit; recycle
  (retire→free_→submit) is ordered behind the feeder mutex and the matched
  `arm_[]` entry is cleared at publish, so a callback can never address a
  recycled unit (bug 2 in §1 was exactly this, now structurally prevented).
- **Shutdown ordering** (LayerFeeder dtor): reader threads joined FIRST
  (stop → CancelIoEx per handle → wake posts → ABORTED completions reaped →
  workers exit; callbacks suppressed under stop so nothing fires into the
  dying feeder), then the ring unregisters/frees. Member order (reader before
  ring) makes the implicit destructor pass a no-op after the explicit
  shutdown. Verified live by the smoke's destroy-with-units-in-flight case —
  clean exit, no join stall.
- **Error paths**: bad plans (past-EOF, >slot, unaligned slot) throw at
  submit/begin_epoch (caller-visible, no engine state corrupted); IO failures
  retry ≤ 3 then mark the unit failed → global fatal → failed unit still
  publishes its slot so blocked acquirers wake and `acquire_layer` returns
  nullptr (never garbage data, never a hang); ERROR_OPERATION_ABORTED is
  accepted silently during teardown. Known minor: a completion that fires
  only via the synchronous-fail path inside submit() can have its callback
  delayed up to the workers' 1 s GQCS poll — the unit event is set
  immediately, so `wait()` callers are unaffected (feeder uses callbacks; the
  delay only postpones a publish on an already-fatal path).
- **Alignment**: every ReadFile uses 4096-aligned file offset, 4096-multiple
  length (except twin tails <512 B), and a destination at slot_base +
  4096-multiple offset (slot base is page-aligned). Sector-size guard pending:
  see TODO (ioctl query omitted in v1 — 4096 covers 512e/4Kn by construction).

## 7. Known TODOs

1. **Dual-drive split/mirror (E:+C:)**: the plan format already supports it
   (requests are per-path; a layer can be split across drives as two requests
   into one slot), but the reader issues strictly FIFO — a split plan
   interleaving two drives serializes on QD16 global. Needs per-file QD or
   two reader pools (nvme-reader.md §5.1 mitigation 2: ~2.3× aggregate).
2. **Prefill sweep mode**: same machinery layer-major (weights-once,
   FlexGen-style), but the consumer wants the slot pointer held across many
   tokens per layer and acquire order relaxed vs decode. Add a
   `sweep_acquire(sweep_index)` alias + longer holds (window math already
   permits: holds shrink read-ahead, not correctness).
3. **`copy_out` ConsumeMode**: chunked 32 MB cudaMemcpyAsync chain + event on
   a dedicated copy stream when a future config wants S-layers (dormant by
   design — pcie-pipeline.md kills staged-copy for v1).
4. **Sector-size ioctl guard** (`IOCTL_STORAGE_QUERY_PROPERTY`) at file open —
   belt-and-braces assert that logical sector divides 4096.
5. **IoRing**: skipped for the same reasons as nvme-reader.md §7 (≤ +2%,
   second I/O stack); revisit if profiling shows >2% CPU in the issue path.
6. **Rate bench**: run the nvme-reader.md §6 matrix (E: ≥ 3.0 GB/s acceptance
   on a 20 GB span, QD sweep) before any placement math is tuned against this
   reader; the smoke's 1.8–2.4 GiB/s on a 67 MB span is not the steady-state
   number.
7. **mk.py wiring**: none done (out of scope by mission rules);
   `build\streaming-check.obj` is a standalone compile artifact only.

## 8. Files

- `E:\coding\Insignia\include\insignia_streaming.hpp` — API (new)
- `E:\coding\Insignia\src\streaming.cu` — implementation + guarded smoke main (new)
- `E:\coding\Insignia\build\streaming-check.obj`, `build\streaming-smoke.exe` — build artifacts

No other files touched; no git operations.
