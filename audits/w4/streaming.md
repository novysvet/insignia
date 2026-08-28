# w4 deep audit — src/streaming.cu + include/insignia_streaming.hpp

Date 2026-08-25. Read-only audit; the only file written is this one. Everything below was
verified against the live code (both files read end-to-end, all line refs current), the
live checkpoint headers (66 shards parsed with Python), the w3 design docs
(`audits/w3/MASTER-PLAN.md` §Phase D, `audits/w3/nvme-reader.md`, `audits/w3/io-bench-results.md`,
`audits/w3/loader-gaps.md` §3.3, `audits/w3/pcie-pipeline.md` §7.2), and one empirical
Win32 probe (ctypes `ReadFile` on a synchronous handle with an `OVERLAPPED` offset — see §2.4).
No builds, no engine binaries run.

Legend: **[V]** = verified from code/measurements; **[H]** = hypothesized (reasoned, not
reproduced). Severity P0..P3.

---

## 0. Verdict in one paragraph

The reader core is solid: IOCP + GQCSEx, `OVERLAPPED`-first `Req` layout
(static_asserted, streaming.cu:37), bounded retries (kMaxTries=3 shared counter across both
failure paths), consistent feeder→reader lock order, callbacks fired outside locks
(streaming.cu:276-279), correct CancelIoEx-first teardown ordering, and a smoke test that
genuinely exercises head-offsets, multi-request slots, epoch re-arm and teardown-in-flight.
Four real defects: **(1)** the master plan's 16-byte F8 rebasing is implemented nowhere and
is in fact *inexpressible* in the current `ReadPlan` API — but it only affects 8/66 shards,
all of which sit inside the always-VRAM layers 0–18 of every planned manifest; **(2)** a
kernel-handle leak of one event per retired unit; **(3)** the empty-plan fast path never
fires the completion callback (contract violation → slot never publishes); **(4)** a
defensive branch in the worker skips the `outstanding_` decrement. Beyond that, the
generate27 integration surface (tier dispatch, `read_once`, pinned-cap probe ladder, epoch
cancel, deadline acquire) is entirely absent — consistent with Phase D.3/D.4 not having
started, but it means nothing in the tree consumes streaming yet (verified: zero references
to `LayerFeeder`/`NvmeReader`/`PinnedRing` outside the two files; no mk.py/bat target
compiles streaming.cu).

---

## 1. Lifecycle and races

### 1.1 NvmeReader — verified issues

**[V] [P1] Event-handle leak on every `retire()` — streaming.cu:239-248.**
`Unit` holds a raw `HANDLE event` with no destructor. `retire()` does
`units_[stream].reset()` (destroys the `Unit`, leaking the handle), and the recycle path in
`submit()` (streaming.cu:157) does `units_[h].reset(new Unit{})` → `u.event == nullptr` → a
fresh `CreateEventW` (streaming.cu:162). Net: one leaked kernel handle per completed unit =
per streamed layer per epoch. At v2 (N=15, ~0.57 tok/s) that is ~27 K handles/hour
(~650 K/day, paged pool each). The R3 1000-token endurance run would leak ~64 K handles —
visible in Task Manager, harmless for short tests, fatal for a long session. Fix is
3 lines: `CloseHandle(u->event)` in `retire()` before `reset()`, or reuse the `Unit` (don't
`reset(new Unit{})`; just `ResetEvent` + clear `blocks/reqs`).

**[V] [P2] Empty-plan completion skips the callback — streaming.cu:166-168.**
The degenerate path sets `u.ok/done/SetEvent` directly but never queues `u.fn` into
`pend_cbs_`, violating the header contract "fired once, on a reader thread"
(insignia_streaming.hpp:83, 91-94). In LayerFeeder terms: `arm_locked()` claimed the slot
(streaming.cu:403), submitted, and `on_done` never runs → slot stays FILLING forever →
`acquire_layer` blocks forever (PinnedRing::acquire has no timeout, streaming.cu:341-351).
Unreachable with real shard plans (whole-file ≥ 372 MB ⇒ 178+ blocks), but it is an
API-level hang trap. Fix: push to `pend_cbs_` there too, or fire inline.

**[V] [P2] `outstanding_` leak on the defensive retired-unit branch — streaming.cu:262-263.**
`if (!units_[ui]) continue;` runs *before* `outstanding_.fetch_sub(1)` (line 267). If a
stale packet for a retired-but-not-yet-resubmitted handle ever arrived, the count would
leak permanently and `top_up_locked` would stall at `outstanding_ >= qd_`
(streaming.cu:215) forever. Under correct Win32 semantics stale packets cannot exist
(one packet per initiated `ReadFile`; a unit reaches `done` only after every block was
counted exactly once), so the branch is dead code — but dead code that breaks the
invariant it defends. Move the decrement before the null-check.

**[V] [P2] `TerminateThread` fallback can abandon `mtx_` → destructor deadlock —
streaming.cu:68, 73.** A worker holds `mtx_` across completion processing + `top_up`
(including `ReadFile` issue syscalls, streaming.cu:257-277). If the storage stack wedges
inside an issue for >5 s, `WaitForSingleObject(t,5000)` times out and `TerminateThread`
kills the mutex owner; the subsequent `std::lock_guard<std::mutex> lk(mtx_)` in `shutdown()`
(line 73) then blocks forever (SRWLOCK has no abandoned-owner recovery). Probability low
(requires a >5 s hang *inside* the lock), impact total. Prefer `INFINITE` wait after
CancelIoEx, or drop to one thread before the last resort.

**[V] [P3] Worker early-exit can close files under in-flight IRPs — streaming.cu:280-282.**
`if (stop_ && (!got || outstanding_ == 0)) return;` — `!got` is also the 1 s GQCSEx
*timeout*. If CancelIoEx's ABORTED completions take >1 s (wedged drive), both workers exit
with `outstanding_ > 0`, then `shutdown()` closes the file handles (lines 77-78) with IRPs
in flight — documented-UB territory. The normal path (cancel → µs/ms ABORTED packets →
drain) is correct and exercised by the smoke teardown; this is the corner. Note the exit
condition's comment claims "port dead or fully drained" but timeout is neither.

**[V] [P3] `wait()` handle-close race — streaming.cu:228-237.** `done_event()` returns
`u->event` under `mtx_`, then `WaitForSingleObject` runs without the lock; a concurrent
`retire()`/`shutdown()` could invalidate the handle. Safe under the single-consumer
LayerFeeder discipline (nothing calls `wait()`); flagged for API hygiene.

### 1.2 NvmeReader — things checked and found CORRECT (verified)

- **OVERLAPPED-first layout**: static_assert `offsetof(Req, ov) == 0` + standard-layout
  (streaming.cu:37) — the container_of cast at streaming.cu:260 is sound.
- **Double completion**: one packet per initiated read; retries re-issue and the shared
  `r->tries` counter (incremented in both the worker path streaming.cu:271 and the issue
  path streaming.cu:190) bounds total attempts at kMaxTries=3 from either side; no
  infinite-retry, no double-count on the paths that exist. The one implicit Win32
  assumption — a *synchronously failing* `ReadFile` (error ≠ ERROR_IO_PENDING) queues no
  packet — is standard; if it were ever false the `left` countdown would underflow, but
  MSDN semantics say no packet when no I/O was initiated. **[V] code / [H] semantics**
- **Lock order**: feeder `mtx_` → reader `mtx_` everywhere (on_done: streaming.cu:418-427;
  arm_locked→submit: 404; release_layer→arm_locked: 447-449). Workers fire callbacks
  *outside* the reader lock (276-279) → no reader→feeder inversion. No deadlock cycle. **[V]**
- **Teardown ordering**: stop_ (idempotent CAS, 58-60) → CancelIoEx both handle sets
  (61-64) → wake posts (65-66) → join (67-70) → close files/events under lock (72-84) →
  close port (85) → fatal_ (86). Port closed only after threads joined. Correct order. **[V]**
- **Port release**: `files_.clear()` before `port_` close; IOCP key (file index+1,
  streaming.cu:108) is stored but never read — dead weight, harmless.
- **Handle recycling race** (retire→submit reusing a stream id): LayerFeeder's on_done
  does the descending arm_ search, stale-proofs the entry (`0xFFFFFFFFu`), *then* retires
  (streaming.cu:416-428, with the rationale comment at 412-415). Within this design's usage
  (only on_done retires) a recycled id cannot be confused with a stale one. **[V]**
- **EOF-tail read through the buffered twin** (streaming.cu:137-141): the twin is a
  *synchronous* handle and the code passes an `OVERLAPPED` purely for the offset — I
  verified empirically (ctypes, correct x64 `OVERLAPPED` layout) that `ReadFile` on a
  synchronous handle honors `ov.Offset` and blocks until complete. Semantics correct;
  note the smoke test never reaches this path (§6).
- **QD accounting**: `top_up_locked` walks `order_` FIFO and issues under `mtx_`, so
  `outstanding_` tracks exactly; drive sees mostly-sequential 2 MiB blocks. Minor deviation
  from the design's "one file streaming at a time" (nvme-reader.md §4): the window keeps up
  to `slots` units from different files interleaved at block granularity — at 2 MiB on a
  Gen3 980 this is irrelevant (io-bench: flat in QD/blk).

### 1.3 PinnedRing

- **State machine / ABA**: FREE→FILLING (CAS, 331-334), FILLING→READY (release store +
  SetEvent, 336-339), READY→IN_USE (CAS after acquire-load, 341-351), IN_USE→FREE (release
  store + SetEvent, 353-356). One cache line per slot (`alignas(64)`, hpp:183). The real
  ABA danger — consumer acquires epoch e while the slot has been recycled to epoch e+slots —
  is prevented *only* by the sequential contract: `arm_locked` can claim slot s for e+slots
  only after `release_layer(e)` set `released_ > e` (window inequality, streaming.cu:400),
  and a consumer that already returned from `acquire_layer(e)` cannot still be pre-acquire.
  **[V] safe under contract.** What is missing is defense against contract violations:
  - **[V] [P3] `release_layer` double-call corrupts a refilled slot** — `release()` stores
    FREE unconditionally (streaming.cu:353-355); a second release after the slot was
    re-claimed FILLING would mark it FREE mid-fill (two units, one slot). A CAS
    IN_USE→FREE would make it a no-op instead.
  - **[V] [P3] `acquire_layer` has no lower-bound guard** — it refuses only
    `epoch_index >= next_submit_` (streaming.cu:436); re-acquiring an already-released
    epoch spins on a slot now owned by e+slots and eventually returns *that* epoch's bytes
    with no error. Add `epoch_index < released_` → nullptr.
- **Memory ordering**: publish = release store, acquire = acquire load + acq_rel CAS; the
  DMA'd slot data is written before the completion packet is observed by the worker
  (kernel handoff), so publish/consume is a proper release/acquire edge; the twin tail
  read happens at submit time (same or earlier thread) before publish. Sound. **[V]**
- **No-epoch-tag**: slots carry no generation counter (contrast pcie-pipeline §7.2's
  `Slot.epoch` design, where consumers match epochs). Fine given the above; noted because
  the w3 design deliberately had it and this implementation traded it for the window guard.
- **Timeout paths**: `acquire` = 65536 pause spins then 20 ms auto-reset-event waits,
  forever (documented "no timeout: unit always ends", hpp:177). Consequence: if the reader
  is wedged (drive hang, no completions) or during teardown with in-flight units
  (callbacks suppressed under stop_, streaming.cu:278-279 → on_done never runs → slot never
  publishes), a blocked consumer hangs forever. Acceptable for the single-owner decode loop
  (the owner is the one calling the destructor); **[H]** hazard only for multi-consumer use
  that the CPU tier (v2) will eventually want.
- **Epoch recycling**: `begin_epoch` requires `released_ == n_` (streaming.cu:375) — with
  the sequential contract that also proves every slot is FREE. Correct. Note `released_` is
  assigned `epoch+1` (448), so out-of-order releases *reduce* it — again contract-gated.

### 1.4 LayerFeeder

- **Self-arming / deadlock**: `arm_locked` (399-408) only ever *submits* (submit never
  blocks on I/O; it opens files + reads ≤512 B tails under the two mutexes — a first-epoch
  startup hiccup of ~64×2 `CreateFileW`s on the worker/consumer thread, then steady-state
  clean). No cycle (lock order above), no blocking wait inside a lock that another path
  must take. `on_done` wraps `arm_locked` in try/catch → `fatal_` (427); same in
  `release_layer` (449). **`begin_epoch` does not** wrap its `arm_locked()` (line 392) —
  a bad plan throws out of begin_epoch with a partially-armed epoch; `healthy()` stays
  true (reader `fatal_` unset by a throw), so subsequent `acquire_layer(e)` for un-armed
  e returns nullptr (out-of-window guard) rather than hanging — degrades, doesn't hang.
  **[V]** Still: wrap it for symmetry.
- **Over-read**: impossible beyond the window — `next_submit_ < released_ + slots`
  bounds in-flight + ready units to `slots`; at epoch end no further reads; total bytes
  read per epoch = plan spans exactly. On *abandonment* mid-epoch (EOS/give-up), up to
  slots−1 already-read layers are wasted and there is **no cancel API** — the next
  `begin_epoch` dies ("prior epoch not fully released", streaming.cu:375) unless the
  consumer mechanically acquires+releases the rest. See §4.
- **Slot leak on exception**: if `submit` throws after `try_claim` succeeded
  (e.g. `build_blocks_locked` dies on "extent past EOF", streaming.cu:129), the slot stays
  FILLING forever — but `fatal_` is set by every caller's catch and `acquire_layer` checks
  `healthy()` *before* touching the ring (streaming.cu:433), so nothing hangs; the feeder
  is simply dead. Bounded and acceptable. **[V]**
- **`map()` unlocked reads of `map_`/`n_`** (452-460): both are only written in
  `begin_epoch` under `mtx_`; a consumer racing a new `begin_epoch` while still holding
  stale mapped pointers (contract violation: map valid only between acquire and release)
  is UB-by-contract. Technically a C++ data race; benign on x64. **[V]** minor.

---

## 2. The 16-byte alignment issue (task item 2) — measured, and different from the plan

### 2.1 What the master plan says

MASTER-PLAN.md:362-368 (Phase D.2, "CRITICAL addition — 16-byte rebasing") claims
`data_start ≡ 8 (mod 16)` *in every shard*, so F8 tensor bases land ≡8 mod 16 in a slot;
fix = "insert one 8-byte pad at the BF16→F8 boundary in the plan builder"; assert
`(f8_base & 15) == 0` at acquire. Risk #7 (MASTER-PLAN.md:473) rates it crash-at-first-
streamed-GEMV.

### 2.2 What is actually true (all 66 headers parsed live)

| shard family | count | data_start | data_start mod 16 | F8 tensors mod 16 |
|---|---|---|---|---|
| linear 1-digit (layers 0,1,2,4,5,6,8,9) | **8** | 2600 | **8** | **all 6 per shard ≡ 8** |
| linear 2-digit (10..63 minus fulls) | 40 | 2624 | 0 | all ≡ 0 |
| full-attn 1-digit (3, 7) | 2 | 2320 | 0 | all ≡ 0 |
| full-attn 2-digit | 14 | 2336 | 0 | all ≡ 0 |
| mtp | 1 | 2480 | 0 | all ≡ 0 |
| outside | 1 | 38080 | 0 | (no F8 — pure BF16) |

So the misalignment is real but **confined to 8/66 shards — layers 0, 1, 2, 4, 5, 6, 8, 9**
(48 F8 tensors total). The master plan's "every shard" premise over-generalized from the
2600 family. **Crucially, all 8 affected layers lie inside 0–18**, which is the
VRAM-resident L=19 block in *every* manifest of the ladder (v1/v1.5/v2, MASTER-PLAN.md
§2.2/§2.4: "VRAM contiguous at the start"). Under the planned placements, streamed layers
are 19–63 — all 16B-aligned. The bug bites only if a manifest/solver ever puts one of
those 8 layers in the N tier (e.g., a degrade ladder after a VRAM-probe shortfall). The
smoke test streams layers-0 (a misaligned shard!) but only `memcmp`s — alignment-agnostic,
so it passes.

### 2.3 Is the fix implemented anywhere? No — and as specified it cannot be.

- **streaming.cu / insignia_streaming.hpp**: no pad, no dst-shift, no alignment assert
  anywhere in either file (verified by reading both in full; the only alignment checks are
  the 4096 slot check at streaming.cu:153).
- **tools/index27.py**: stores raw absolute offsets (`off = data_start + begin`,
  index27.py:181-182). The `align_base` / `in_slot_off` fields that loader-gaps.md:102-121
  designed (and that MASTER-PLAN.md:367 claims are "already precomputed in the index") **do
  not exist** in the emitted INSIDX02.
- **Kernels require 16B, no tolerance**: `fp8_gemv_kernel` loads
  `__ldcs(reinterpret_cast<const uint4*>(row_w + c0))` (fp8.cu:32) and `fp8_gemv2_kernel`
  likewise (fp8.cu:76) — row offsets (`row*cols`, cols%128==0) preserve the base's mod-16
  phase, so a base ≡8 faults with cudaErrorMisalignedAddress (loud, not silent). The
  prefill `fp8_gemm_kernel` B-side uses `cp.async ... , 16` on
  `weights + (n0+n)*cols + k + chunk*16` (fp8.cu:132) — same hard 16B requirement.
  Scales are scalar u16 loads — alignment-free. (gemm.cu's uint4/cp.async sites are the
  MXFP4/bf16-activation paths — activations, not streamed weights — irrelevant here.)
- **Why the plan-builder pad is inexpressible**: in-slot byte position ≡ absolute file
  offset (mod 16), *always*. Every byte at absolute X of request r lands at
  `slot + Σspans + (X − align_down_4096(r.offset))`; each span is a 4096 multiple (except
  EOF-clamped ones), and `align_down_4096 ≡ 0 mod 16` — so the phase is an invariant of
  the file layout. `ReadRequest` is just `{path, offset, len}` (insignia_streaming.hpp:55-59):
  no dst/shift/pad field, and a destination shift would break NO_BUFFERING's sector-aligned
  *buffer* requirement anyway. Splitting the plan at the BF16→F8 boundary (any offsets you
  like) cannot move the F8 phase by 8. The only latent mechanism that shifts cursor phase
  is the EOF clamp (cursor advances by `f.size − align_down(f.size)` ≡ 616 mod 4096 for
  the 2600 family), which is not a usable design lever.

### 2.4 Ranked fix options (pick one; the assert is mandatory regardless)

1. **[P2, recommended] Alignment-tolerant load path for the 8 shards**: the GEMV kernels'
   16B `uint4` read splits into two `uint2` reads — a base ≡8 mod 16 is *8B-aligned*, so
   `uint2` pairs are legal everywhere. Cost ≈ one extra load per 16 B on the decode path
   of 8 of 48 linear layers; measurable, likely ≪1% of step time. Add a base-phase
   template or runtime branch baked per-layer at startup (AGENTS spirit: bake the
   assumption per manifest).
2. **[P2] One-time +8 memmove at publish** (reader or consumer side) shifting the F8 region
   of affected shards left by 8 in the slot (~380 MB memmove ≈ 10 ms @ ~38 GB/s DRAM;
   needs 8 extra bytes read past region end). ~1% of a 115 ms layer; only for the 8 shards.
3. **[P3] Reject-at-plan-build**: the plan builder refuses/flags any F8 tensor that would
   land ≡8 mod 16 — turns the silent assumption into a loud error at startup, pairs with
   option 1.
4. The master plan's "pad in the plan builder" as written is **not implementable** without
   an API extension (per-request in-slot shift + post-read shift, since direct DMA cannot
   land misaligned) — recommend the plan doc be corrected along with the "every shard"
   premise.

Also add the missing assert the plan mandates: `(f8_base & 15) == 0` (or ==8 with the
tolerant kernel selected) at acquire/plan-build — currently absent everywhere.

---

## 3. Config vs io-bench recommendations (task item 3)

| parameter | code | bench recommendation (io-bench-results.md §8) | verdict |
|---|---|---|---|
| block size | `kBlock = 2 MiB` (hpp:66) | 2 MiB on E: (flat; −1% of best) | ✓ |
| queue depth | `kQD = 16` global (hpp:105, streaming.cu:144) | **QD 8** ("2M/QD8 = 3.22; QD16 = 3.14" — io-bench-results.md:53-54) | ~2.5% slower than measured optimum; QD is *latency insurance* on this drive (flat QD1–16). With 2 workers the global 16 is ~8/thread when both reap. Trivial knob; suggest 8, or keep 16 only if the C:-mirror future wants headroom (keep in-flight ≤128 MB — respected: 16×2 MiB = 32 MiB). |
| reader threads | default 2 (hpp:85, hpp:211) | 2 (1 suffices; warm spare + dual-slot arming) | ✓ |
| affinity | `0xFC0` = LP 6–11 (streaming.cu:32,47) | readers on SMT siblings, LP 6–11 | ✓ |
| priority | `THREAD_PRIORITY_ABOVE_NORMAL` (streaming.cu:48) | ABOVE_NORMAL (not HIGHEST/TIME_CRITICAL: DPC safety) | ✓ |
| flags | direct `NO_BUFFERING\|OVERLAPPED` + one IOCP + buffered twin (streaming.cu:97-103) | same; buffered = 3.4-core memcpy tax, 0% steady hits | ✓ |
| sector guard | kSector=4096 by construction; no IOCTL_STORAGE_QUERY_PROPERTY probe (nvme-reader.md §7 had it) | guard, not branch | acceptable omission — 4096-multiple blocks/offsets are 512e and 4Kn safe by construction |
| `FILE_FLAG_SEQUENTIAL_SCAN` | omitted on the direct handle (inert under NO_BUFFERING) | bench §4 confirmed inert + the cached-pass trap | ✓ correct to omit |
| smoke default ring | 4 × 368 MiB = 1.44 GiB (hpp:207-208) | 4×368 MiB default, 5 slots upper (nvme-reader.md §3.1) | ✓ (fits v1: pinned = ring 1,152 MB ≤ 8,531 MB cap) |

Config is 95% aligned with measurements; the only numeric delta is QD16 vs QD8 (−2.5%
bandwidth, no correctness impact).

---

## 4. API completeness for the decode loop / generate27 (task item 4)

What exists: `begin_epoch` / `acquire_layer(i)` (blocking, no deadline) / `release_layer`
/ `map(i,r)` / `plan_span` / `healthy` — the v1 zero-copy consumption pattern of
pcie-pipeline §7.2's hot-path sketch is expressible today. Missing, in integration order:

1. **Tier dispatch (MASTER-PLAN.md §3-D.4, 375-381) — entirely absent.** No
   `tier_of[64]`, no VRAM/Z/UVA/static-pointer path wired to streaming; grep confirms
   nothing outside streaming.cu references these classes; `ENGINE27` in tools/mk.py:28-29
   does not even compile streaming.cu. The decode loop integration is 0% started.
2. **`read_once` (nvme-reader.md §3, MASTER-PLAN D.2, 361)** — not implemented. mtp
   (477 MB, 228 blocks) exceeds `kDefaultSlotBytes` (184×2 MiB) and is rejected by
   begin_epoch's span check (streaming.cu:389-390, comment even says "mtp/outside are
   read_once material" — but the material's supply path doesn't exist). Startup loading of
   mtp/outside/embed must currently go through some other (unwritten) path.
3. **Embed row pread** (10 KB/token, buffered twin, issued a step ahead — MASTER-PLAN
   §2.4 mechanics): the twin handles are private to NvmeReader; no API. Missing.
4. **Epoch cancellation / early stop**: EOS mid-epoch leaves `released_ != n_`; the next
   `begin_epoch` throws (streaming.cu:375). No abort/flush API; the consumer must
   mechanically drain acquires/releases or destroy the feeder. Needed for generate27.
5. **Deadline/timeout on acquire**: none (hpp:177 documents "no timeout: unit always
   ends"). For a generation loop with a wall-clock budget (or watchdog), a
   `acquire_layer(i, deadline_ms)` variant returning nullptr would make stalls diagnosable
   instead of eternal.
6. **Back-pressure**: implicitly correct — the `slots`-window IS the back-pressure; if the
   GPU/consumer stalls, the ring fills, `arm_locked` stops submitting, the drive goes
   idle; on resume, READY slots are consumed with zero NVMe latency. No explicit
   mechanism needed. ✓ by construction (streaming.cu:400).
7. **`ConsumeMode::copy_out`** (hpp:199, 203): reserved hook only — acquire returns the
   pinned pointer in both modes. The S-tier is rejected by the master plan anyway
   (Appendix A.1); fine to leave dormant.
8. **Pinned-cap probe ladder** (MASTER-PLAN D.3, 373): absent — see §5.
9. **Drive-map hook for OPTION-G** (MASTER-PLAN D.2/Appendix B): generic per-request
   `path` in `ReadPlan` makes a mirror-dir override trivially expressible at the plan
   layer — no reader change needed. Effectively present by accident of the generic API. ✓

---

## 5. cudaHostRegister semantics and fallback (task item 5)

- **Registration**: one contiguous `VirtualAlloc(MEM_RESERVE|MEM_COMMIT)` region,
  `cudaSetDevice(0)` then `cudaHostRegister(base, bytes, cudaHostRegisterDefault)`
  (streaming.cu:295-303). `Default` (not `Mapped`) is correct on 64-bit UVA systems —
  device code and `cudaMemcpyAsync` can use the host pointer directly; `Mapped`+`GetDevicePointer`
  would be the pre-UVA ceremony. One register for the whole ring (not per-slot) is the
  right granularity. Unregister before VirtualFree in the dtor, correct order
  (streaming.cu:321-329). `cudaSetDevice(0)` hardcodes device 0 — fine for this rig,
  worth an assert.
- **Fallback**: on registration failure: print, clear the sticky error, `VirtualLock`
  every slot best-effort (failures silently ignored), set `locked_ = true`
  (streaming.cu:305-312). Consequences: H2D copies from the ring pay a staging bounce
  (functional, slower); CPU GEMV unaffected. **Gaps**: (a) `locked_` is set true even if
  every `VirtualLock` failed, and no `SetProcessWorkingSetSize` is attempted, so
  "locked pageable" is aspirational under WS pressure; (b) **no WDDM cap probe ladder**
  (MASTER-PLAN D.3: "1 GiB until fail → Z count") — registration is all-or-nothing over
  the whole 1.44 GiB ring. If pinned-Z layers (v1.5: 7.2 GiB) are registered *before* the
  ring, the ring registration can fail against the 7.95 GiB cap and the engine silently
  runs the streaming tier on pageable memory (stderr line only). The ladder belongs at
  startup orchestration level, but nothing in streaming.cu exposes partial registration
  (e.g., register slot-by-slot and fall back per slot) that the ladder would need.
  **[V] absent.**

---

## 6. Missing tests (task item 6) — what a streaming test suite should check

The existing smoke (`-DINSIG_STREAMING_SMOKE`, streaming.cu:472-555) covers: golden
byte-compare of 2×32 MiB + 64 MiB with a 2600 head (multi-request slot ✓), second epoch
re-arm ✓, teardown with units in flight ✓, real checkpoint file ✓. Not wired into any
build target (no mk.py/bat entry — has to be hand-compiled per the comment). Gaps, in R3
gate order (MASTER-PLAN.md:385-387):

1. **Whole-shard byte-equality for all 66 shards** (stream each as a one-request plan;
   memcmp slot vs plain buffered read) — currently only layers-0's first 128 MiB.
2. **EOF tail path** — every real tail shape: linear 86,528 B direct + 104/128 B twin,
   full-attn 1,117,696 + 144/160, mtp 1,148,416 + 304 (nvme-reader.md §2.3 table). The
   smoke's reads are all mid-file: `build_blocks_locked`'s tail branch
   (streaming.cu:135-142) has **zero test coverage** (its OVERLAPPED-offset-on-sync-handle
   semantics I verified manually via ctypes, but the integration path is untested).
3. **CRC32 of the streamed bytes vs `crc32.txt`** — verified present: 77-line manifest at
   `Qwen3.8-27B-FP8/crc32.txt`, entries for all shards incl. mtp/outside. The R3 gate's
   "all 66 CRC32s match" needs a mode that CRCs the ring slot contents after acquire.
4. **Alignment asserts**: `(f8_base & 15)` checked for every F8 tensor of every streamed
   shard at acquire — would have caught the 8-shard issue instantly (expected: 8 shards
   assert under the current kernels; see §2).
5. **Handle-count stress**: N epochs back-to-back, assert kernel handle count returns to
   baseline — catches the retire leak (§1.1) and any event/handle lifecycle regressions.
6. **Multi-file / multi-request plans with an EOF-clamped non-final request** — the
   `map()` drift case (see Findings-addendum below): map offsets are computed from
   un-clamped `req_span` (hpp:69-71) while `build_blocks_locked` advances the cursor by
   the clamped extent (streaming.cu:131) — for any plan where a non-final request's
   `align_up(off+len)` exceeds file size, subsequent requests' `map()` pointers are off by
   the clamp delta (≤4095 B). Unreachable with whole-shard single-request plans
   **[V] logic, [H] reachability]** — add a unit test pinning the intended semantics or
   fix `req_span` to clamp.
7. **Failure injection**: missing file (submit throws → feeder fatal → acquire returns
   nullptr, no hang); extent-past-EOF (die path, streaming.cu:129); read error mid-unit
   (harder — needs a fault-injection hook or a truncated twin; at minimum test that
   fatal_ propagates and waiters wake via the shutdown SetEvent path, streaming.cu:74-75).
8. **Teardown stress**: 100× begin-epoch-with-in-flight + destruct (the smoke does this
   once); assert no hang (TerminateThread/abandoned-mutex detector) and clean exit codes.
9. **Perf gates**: sustained ≥3.0 GB/s fill over a ≥6-layer sweep (bench says 3.22);
   reader CPU ≤0.5 cores (QueryThreadCycleTime, bench says 0.06-0.25); GQCSEx batch ~1.0.
10. **Ring fallback**: force `cudaHostRegister` failure (pre-register a large dummy to
    approach the 7.95 GiB WDDM cap) — assert VirtualLock path + `ring_pinned()==false`
    reported honestly (currently `locked_` can lie, §5).
11. **API-contract negatives**: double `release_layer`, re-acquire of a released epoch,
    out-of-window acquire — all currently either corrupt state silently or hang (§1.3);
    whatever hardening lands needs pinning tests.

---

## 7. Ranked fixes (summary)

| # | severity | fix | where |
|---|---|---|---|
| 1 | P2 (P0 if any of layers {0,1,2,4,5,6,8,9} ever streams) | F8 16B alignment: uint2-tolerant GEMV loads (or publish-time +8 memmove) + `(base&15)` assert at plan-build/acquire; correct the master plan's "every shard"/"pad in plan builder" text (inexpressible; 8/66 shards affected, all inside the planned VRAM block) | fp8.cu:32/76/132 + plan builder |
| 2 | P1 | Event-handle leak per retired unit: close or reuse the event in retire/recycle | streaming.cu:239-248, 157-162 |
| 3 | P2 | Empty-plan fast path must queue/fire the completion callback | streaming.cu:166-168 |
| 4 | P2 | `outstanding_` decrement before the retired-unit defensive skip | streaming.cu:262-267 |
| 5 | P2 | Generate27 API surface: `read_once` (mtp/outside/embed), epoch cancel for EOS, deadline acquire, tier dispatch §3-D.4, pinned-cap probe ladder + per-slot register fallback | new in streaming.cu / decode loop |
| 6 | P2 | Drop TerminateThread last resort (abandoned-mutex deadlock) and the `!got` early-exit-with-outstanding teardown corner | streaming.cu:68, 280-282 |
| 7 | P3 | Slot-state hardening: CAS IN_USE→FREE in release; reject re-acquire of released epoch; fix/flag `map()` drift on EOF-clamped non-final requests; wrap begin_epoch's arm_locked in try/catch | streaming.cu:353, 436, 448; hpp:69-71 |
| 8 | P3 | kQD 16→8 per io-bench (−2.5% free); honest `locked_`; mk.py target for the smoke | hpp:105, streaming.cu:305-312, tools/mk.py |

Everything else audited — IOCP worker structure, retry bounds, teardown ordering, lock
discipline, PinnedRing ordering, feeder window arithmetic, tail-read semantics, config vs
bench — is correct as written and reasonably matched to the w3 designs.
