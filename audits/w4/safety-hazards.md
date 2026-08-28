# w4 safety audit — hazard ledger + new-tier hazard hunt, 2026-08-25

Scope: every C1-C11 (w3/safety.md), H1-H8 (w3/diff-verify.md), graph-* item
(w3/graph-hazards.md) and master-plan §4 risk re-verified against the CURRENT tree
(HEAD `92e1028` + dirty; includes the landed fixes: gemm launcher throws, ctx>4090
refuse in generate.cu, v21 tail cp.async, NLL i4 branch), plus a fresh hazard hunt in
the uncommitted tier code (`src/streaming.cu`, `include/insignia_streaming.hpp`,
`include/insignia_cpu.hpp`, `src/fp8.cu`, `tools/index27.py`, `src/model_file.cpp`,
decode i4 branches, generate nll mode). Read-only: code reads + python analysis only;
no builds, no file changes except this report. 27B facts verified live against
`Qwen3.8-27B-FP8/` (66 shards, layers-0 = 383,865,448 B, header 8+2592 = 2600 B,
A_log is **BF16 [48]**, dt_bias BF16 [48], conv1d BF16 [10240,1,4], F8_E4M3 weights +
BF16 scale_inv [r/128,c/128]).

Verdict counts: 8 FIXED, 6 OPEN-real, 3 OPEN-hygiene, several N/A/latent.
Top new findings: **N1 27B A_log BF16 read as F32 (silent, confirmed on-disk dtype)**,
**N2 `NvmeReader::retire` leaks one event handle per layer-epoch (unattended-run kill)**,
**N3 no fp8/bf16 dispatch in decode.cu `linear*`/embed/lm_head (silent garbage on 27B
bring-up)**, **N4 feeder release-vs-in-flight-GPU-read contract unenforced**,
**R7 ring 16-byte misalignment is live (data_start 2600 ≡ 8 mod 16)**.

---

## Part 1 — Prior-hazard ledger (verified against code, not reports)

### C1 — graph replay overruns KV/score/committed — **FIXED (host), device belt still absent**

- FIXED: `src/generate.cu:112-115` now **throws** `"context overflow: prompt+max_new+16
  exceeds the 4090 cache cap"` instead of clamping. With ctx ≤ 4090: `want_total =
  ctx-16`, loop over-commit ≤ +7 → pos ≤ ctx-9 < ctx ≤ 4096 = `score[4096]` smem and
  KV slot bound; committed count ≤ ctx-9 ≪ 16384. Eager guards unchanged
  (`decode.cu:45,131,134`). The nll path (`DecodeWorkspace(4096)`) is fully eager and
  guarded; tokens = pos+1 ≤ 4096 exactly fits `score[4096]`.
- STILL OPEN (defense-in-depth, master-plan Phase D): no device-side bounds anywhere —
  `src/prefill.cu:88-98` `store_kv_batch_kernel` (`(void)max_context` at :96),
  `src/prefill.cu:287-301` `spec_commit_kernel` (`committed[c]`, `committed[c+1]`
  unbounded), `src/attention.cu:7` / `src/prefill.cu:104-120` `score[t]` indexed by
  unbounded `pos+t+1`. Any future caller that re-introduces a clamp (or a 27B commit
  loop with its own arithmetic) silently corrupts again. Cheap belt: pass max_context
  and clamp-with-sticky-flag in store_kv/store_kv_batch/spec_commit.

### C2 — v21 last-K-step cp.async race — **FIXED (all three kernels)**

`src/gemm.cu:275-276` (`mxfp4_gemm_v21`), `:432-433` (`mxfp4_gemm_v21_i4`),
`:524-525` (`mxfp4_gemm_ab_i4`): all three now do
`if (kb + 2 < ksteps) cp_async_wait_prev(); else cp_async_wait_all();`. No other
cp.async pipelines exist (only 3 wait sites in the tree; fp8.cu GEMM already had the
pattern).

### C3 — LRU eviction vs graph-baked pointers — **OPEN (latent, 9B-only; N/A at 27B)**

No `TieredStorage::pin()` API exists (`src/storage.cu` is unchanged: make_room /
acquire / release / clear only; `include/insignia_storage.hpp` has no pin). Still
protected by the two w3-documented accidents: (a) zero `acquire()` callers on the
replay loop (`generate.cu:179-189` only launches + memcpy), (b) 4.55 GiB checkpoint <
6 GiB budget so `make_room`'s eviction loop can never fire. `capture_spec`
(`decode.cu:238-248`) still has no warmup assert. Master plan rejects graphs at 27B
(§6c), so this is 9B-only; fix before anyone adds a second prompt/mid-generation
prefill.

### C4 — throws inside stream capture strand the stream — **OPEN**

`decode.cu:238-248` (`capture_spec`) and `:255-265` (`capture_step`) still have no
try/catch; any throw between Begin/EndCapture leaves `x_.stream` in capture mode and
leaks the partial graph. 9B-only (27B = no graphs).

### C5 — destructors free without stream sync — **OPEN**

`DecodeWorkspace::~DecodeWorkspace` (`decode.cu:29`): 41 frees, no
`cudaStreamSynchronize` first. `Qwen35Weights::~Qwen35Weights` (`src/qwen35.cu:5`):
`cudaFree(scratch_int_)` with no sync, and it runs **before** member `storage_`'s
synced `clear()` (storage_ declared first, `insignia_qwen35.hpp:19`). Error paths
("KV cache full", "device budget exhausted") unwind with ≤31 layers of kernels
enqueued → dtor frees live buffers → sticky context fault masks the real message.

### C6 — unchecked `cudaMalloc` for targets/logp — **OPEN (both copies)**

`src/generate.cu:63-66` (run_nll) and `src/nll.cu:63-66`: `logitsT` checked,
`targets`/`logp` returns ignored → OOM ⇒ nullptr ⇒ async IMA far from cause. Buffer
sizes themselves are consistent (T ≤ 64 enforced transitively by `prefill_chunk`'s
1..64 throw at `decode.cu:44`; `logitsT` sized 64·vocab; `row_logp_kernel` reads
`targets[row]`, row<T only).

### C7 — argv parsing: 1 MiB stack buffer + unchecked `wcstombs` — **OPEN (partially improved)**

Main generate path is now `char buf[1<<16]` (`generate.cu:104`) — stack risk reduced.
Still open: `run_nll` `char buf[1<<20]` on the host thread stack
(`generate.cu:51`, `nll.cu:50`), and **all three** `wcstombs` call sites ignore
`(size_t)-1` → `buf[(size_t)-1]=0` wild write on any non-ASCII wide char in argv
(generate.cu:52-53, :105-106; nll.cu:51-52). `atoi` garbage still feeds C9.

### C8 — host position mirror drift in graph mode — **FIXED**

`committed_count()` now also D2H-copies `pos_dev` and sets `x_.position = pos`
(`decode.cu:207-213`), and the graph loop calls it every 4 replays
(`generate.cu:182`). `spec_graph_step`'s `+=2` guess (`decode.cu:253`) is corrected
at each host read as its comment claims.

### C9 — token-id range unvalidated outside `forward_token` — **OPEN**

`forward_token` validates (`decode.cu:134`); `prefill_chunk`/`prefill_chunk_device`
(`decode.cu:110-115`) and the spec paths still pass raw tokens to `embed_gather*`
whose `row = __ldg(tokens+t)` is unbounded (`prefill.cu:11-12, 28-30`). argv `atoi`
garbage (typo `-5`, `999999999`) ⇒ ~2 TB GPU OOB read ⇒ garbage or device fault.
Device clamp (`row = (unsigned)row >= vocab ? 0 : row`) is still the cheap closure.

### C10 — ab2 pair kernels hard-require cols==4096 unenforced — **FIXED**

`src/mxfp4.cu:578` (`mxfp4_gemv_ab2_q8`), `:670` (`mxfp4_gemv_ab2_q8g`),
`src/mxfp4_i4.cu:239` (`mxfp4_gemv_ab2_q8_i4`) all throw
`"ab2 pair kernel is 9B-specialized (cols must be 4096)"`. Additional new guards
since w3: `mxfp4_gemm_v21_i4` T>64 throw (`gemm.cu:452`), ab GEMM T∈1..64
(`gemm.cu:543-544`), `fp8_gemm` T>64 (`fp8.cu:188`).

### C11 — leaks / minor hygiene — **OPEN (all sub-items)**

- `Qwen35Decode` still has no destructor: `graph_`/`spec_graph_` never
  `cudaGraphExecDestroy`ed (`include/insignia_decode.hpp:50`); double-capture leaks
  the old exec.
- `DecodeWorkspace` never destroys its owned stream; ctor leaks all earlier allocs if
  a later `alloc` throws (`decode.cu:10-27`, no RAII).
- run_nll/nll: `logitsT/targets/logp` + events `a`/`b` leak (C6 companions).
- `TieredStorage::release` is `noexcept` but allocates a `std::string` for lookup
  (`storage.cu:10`).
- `quantize_q8_groups` still silently returns on bad dims (`mxfp4.cu:242`); ~7
  bench/aux launchers still guard-free (`mxfp4.cu:272,278,401,461,510,577`) —
  bench-only, acceptable.
- No post-launch `cudaGetLastError` anywhere — accepted convention, noted.

### H1 — `reference_pf_i4.py` conv history missing for T>1 — **OPEN (parity-hunt poisoner)**

`tools/reference_pf_i4.py:34-35` still uses only the current-token tap
(`qkv = qkv * cw[:, 3]`) — no in-chunk t-3..t-1 history, no conv-state roll. Token 0
matches; every DeltaNet seam at t>0 in dump-pf comparisons is depressed by a
**reference** bug. Highest-priority fix before the next parity session (the
multistep script rolls state correctly; port that).

### H2 — `dump_multistep.cu` lm_head u8-only — **FIXED**

`src/dump_multistep.cu:33` now branches `if (z.insig4) mxfp4_gemv_v2_i4(...)`.

### H3 — `red[0]` reuse race in gqa/row_logp — **FIXED everywhere; sweep clean**

- `src/attention.cu:7` `gqa_decode_kernel`: dedicated `__shared__ float red[8], smx,
  sden;` — scale/den live in their own slots, written once between barriers.
- `src/prefill.cu:107` `gqa_prefill_kernel`: same `red[8], smx, sden` fix.
- `row_logp` twins: `src/nll.cu:24-27,36-41` and `src/generate.cu:27-30,39-44` use
  `red[8]` (max) and `red[9]` (sum) — written once, read after `__syncthreads()`.
- Sweep of every other `__shared__` reduction: `rms_kernel` partial[0]
  (`ops.cu:5`), `rms_bf` p[0] (`qwen_kernels.cu:5`), `argmax_kernel` bv/bi (:20),
  `argmax_fast` (memset-0 scratch + total-order atomicMax key, memset captured as a
  graph node — replay-safe, `qwen_kernels.cu:30-60`), `bf16_gemv` p[8] (:67),
  `deltanet_decode` sq/sk written once per launch, `deltanet_prefill`
  (`prefill.cu:219-263`) iteration-scoped with the loop-tail `__syncthreads()` at
  :257 before sq/sk rewrite, mxfp4 partials write-once, ab2 staging
  (`mxfp4.cu:588-668`) is phase-separated by one barrier before any cross-thread
  read. **No shared-slot reuse race remains in the tree.**
- RoPE race fix confirmed landed in both kernels: `ops.cu:9` and
  `prefill.cu:54-83` each carry `__shared__ float nsc` with staging in `mem[0..63]`.

### H4 — link-broken bats — **OPEN (all four unchanged)**

`build/test-qwen35.bat`, `build/oldgen.bat`, `build/shim-only.bat`,
`build/test-pair-chain.bat` still have **zero** `mxfp4_i4.cu` references while
compiling decode.cu/generate.cu/qwen35.cu (grep verified). 8 more pre-existing broken
+ smoke step 3 (unchanged). None are on the 27B path; do not use them as build gates.

### H5 — ab2 dim guards — **FIXED** (see C10).

### H6 — stale u8-only debug tools — **OPEN (debug-only)**

`src/dump_attention.cu`, `src/test_checkpoint.cu` still have no `insig4` branch and no
assert (grep: zero matches). Pointed at an i4 index they silently misread scales.

### H7 — mtp.fc hardcoded 4096/8192 — **OPEN**

`src/decode.cu:154-155` still passes literals, not `fc.rows/fc.cols`. Consistent for
9B; footgun for 27B (whose fc is [5120,10240] bf16).

### H8 — checkpoint hygiene — **OPEN (unchanged)**

`build/qwen35-insig4.safetensors` (6.03 GB, fp16-as-BF16, format-broken) and
`qwen35-insig4-v2.safetensors` (4.89 GB, pre-fix) still on disk. The default
`qwen35-insig4.insignia-index` still points at `-text.safetensors`
(double-rounded); `-good.safetensors` + its own index exist and are best. Re-point or
delete; unattended runs should assert the index path ends in `-good`.

### graph-hazards.md items — summary

| item | verdict |
|---|---|
| #1 eviction vs graphs | = C3, OPEN-latent (accident-protected) |
| #2 KV-full bypass | = C1, FIXED host-side / device belt pending |
| #3 spec-reject stale KV row | structurally safe (store-before-read on every path; the doc comment at prefill.cu:305 is still missing — cosmetic) |
| #4 capture legality / exception gap | capture bodies clean (kernel+memset+D2D only); exception gap = C4 OPEN |
| #5 evict→re-acquire same-VA silent wrong weights | latent, unchanged (no pooling, no generation counter) |
| #6/#7 pin-at-capture design | NOT implemented (no pin API) |

### master-plan §4 risk register vs current tree

| # | risk | current evidence |
|---|---|---|
| 1 | zero-center (1+w) norms | engine 9B paths pass `false` (`decode.cu:49,89,96`...); 27B undecided until R4 — OPEN (designed catch R4) |
| 2 | GQA kvh | GPU `attention.cu:7`/`prefill.cu:103` still `head>>2` (9B 16/4). CPU tier is correct for 27B (group-major, 6 q-heads per kv group, `insignia_cpu.hpp:807-810`). OPEN for the GPU 27B port |
| 3 | DeltaNet kshare | GPU `deltanet.cu` decode kernel `kh=head>>1` (9B share 2); CPU tier `kshare=3` correct (`insignia_cpu.hpp:717-725`). OPEN for GPU port |
| 4 | A_log BF16 read as F32 | **CONFIRMED LIVE for 27B**: on-disk A_log is BF16 [48] (verified in layers-0 header); `tensor()` performs no dtype check and `deltanet_parameters*` cast to `const float*` (`decode.cu:82,128`; cpu twin `insignia_cpu.hpp:634-643`). Silent α≈garbage. See N1 |
| 5 | weight_scale_inv semantics | multiply+x256-fold everywhere (`fp8.cu:33,77`; cpu `bf16_scale_x256`) — gated by R1, unresolved |
| 6 | fp8_gemm 64-row y contract | still a caller contract only (`fp8.cu:182-184` store guard is per 16-row tile `wm*16 < T`, so T=3 still stores rows 0..15 → y must be 64·rows; test allocates correctly, no launcher assert) — OPEN |
| 7 | ring 16B misalignment | **OPEN and armed**: shard data_start = 2600 ≡ 8 (mod 16); first F8 tensor base in-slot ≡ 8 mod 16 → `uint4` GEMV misaligned. No pad-at-F8 code exists in streaming.cu; smoke test memcmp-only (can't catch). See N5 |
| 8 | mtp.fc dims/orientation | = H7, OPEN |
| 9 | v21 cp.async race | = C2, FIXED |
| 10 | KV/committed overrun | = C1, host-fixed; Phase D device guards pending |

---

## Part 2 — NEW hazards in the uncommitted tier code (ranked)

### N1 — 27B A_log is BF16 on disk; engine reads it as F32 — CRITICAL (certain × silent)

`Qwen3.8-27B-FP8/layers-N.safetensors`: `A_log` dtype BF16, shape (48,).
`TieredStorage`/tensor() path has **no dtype validation** (only `matrix()` validates
scale tensors, `qwen35.cu:8-22`); `deltanet_parameters` /
`deltanet_params_batch` (`decode.cu:82,128`) and the CPU twin
`deltanet_parameters_cpu` (`insignia_cpu.hpp:641`, takes `const float* A_log`)
bit-reinterpret bf16 pairs as f32 → α = −exp(garbage) — exactly the "state saturates
silently" failure. Note `tools/quantize_insig4.py` FIXED this for 9B by emitting F32
A_log, and index27.py deliberately keeps checkpoint dtypes — so the 27B loader must
either convert at index time or assert `dtype==F32` at load. Pre-flight assert #6
below. (dt_bias/conv1d/norm weights ARE consumed as bf16 — correct.)

### N2 — `NvmeReader::retire` leaks one event handle per completed unit — HIGH for unattended runs

`src/streaming.cu:239-248`: `units_[stream].reset()` destroys the `Unit` **without
`CloseHandle(u->event)`**. Events are created in `submit` (`:162`) and only closed in
`shutdown` for still-live units (`:82`). Every epoch cycle × 64 layers leaks 64
handles; a multi-day unattended 27B run at ~1 epoch/token-step walks toward handle
exhaustion (plus kernel object pressure). One-line fix: close the event in `retire`
(or move it into the free-list entry for reuse). Probability certain over long runs;
damage: eventual failure of CreateEvent/CreateThread → die() cascade mid-generation.

### N3 — no fp8/bf16 dispatch in decode.cu linear paths — CRITICAL 27B bring-up tripwire (silent)

`linear`/`linear2`/`linear_batch` (`decode.cu:31-41`) branch only on `m.insig4`
(u8-MLX vs f16-i4). `matrix()` returns `WKind::fp8` with `insig4=false`
(`qwen35.cu:14-18`) → 27B F8 qkv/o/mlp matrices fall into the u8-scales MXFP4 kernels
→ **silent garbage** (weights reinterpreted as nibbles; scales pointer is the bf16
scale_inv — wrong but non-null, so no crash). Same class: embed branch
(`decode.cu:46`) and lm_head branches (:97-105) have only u8/i4 arms — 27B embed is
BF16 (`outside.safetensors`) → `embed_gather` u8 misread, silent. Only two loud
tripwires exist: ab2 cols!=4096 throw (pair path) and empty `scales` DeviceView for
bf16 matrices on the per-token ab path (nullptr deref). The 27B port must add
`kind==fp8 → fp8_gemv/fp8_gemm` and `kind==bf16 → bf16_gemv` arms (or assert kind at
every dispatch). Recommend a `switch(m.kind)` with `default: throw` — this is the
single most likely source of "27B runs but emits plausible garbage".

### N4 — feeder release vs in-flight GPU async read of the slot — HIGH (design contract, unenforced)

`LayerFeeder::release_layer` (`streaming.cu:443-450`) marks the slot FREE on the host;
the next epoch's `ReadFile` DMA can then start filling it. **CUDA stream ordering
does not protect host-side DMA**: if the consumer enqueued `cudaMemcpyAsync` (H2D of
the layer's weights) from the slot and calls `release_layer` before that copy
completes, disk writes race the PCIe read → torn weights, silent. Nothing in the code
or header states the "no pending GPU reads at release" precondition. The 27B driver
must `cudaStreamSynchronize` (or event-query) the consuming stream before
`release_layer`. Cheapest enforcement: fold a `cudaStreamSynchronize(copy_stream)`
into `release_layer` (or document + assert). Related: `ConsumeMode::copy_out` hook is
reserved but unimplemented — when it lands, this contract becomes mandatory.

### N5 — F8 tensor bases land ≡ 8 (mod 16) in ring slots — HIGH (certain, but loud)

layers-N `data_start = 2600` (8+2592 JSON) ≡ 8 mod 16. A shard-level plan maps
logical data at `slot_base + 2600`; the first F8 tensor (and every tensor whose
offset ≡ 8 mod 16) violates the 16-byte alignment required by the `uint4` loads in
`fp8_gemv`/`fp8_gemv2` (`fp8.cu:32,76` — `__ldcs(reinterpret_cast<const uint4*>)`).
Failure mode: misaligned-address fault at the first streamed GEMV (loud crash) or,
for the CPU tier, unaligned `__m256i` loads (movdqu — tolerated, just slower...
actually `_mm256_loadu` is used everywhere in the CPU tier — CPU is safe; GPU uint4
is not). Fix per master plan: pad at F8 boundary when building plans (start the plan
at the first 16-aligned offset ≥ tensor start), or assert `(base&15)==0` in the
consumer and refuse. Put the assert in the pre-flight.

### N6 — teardown with blocked consumer / in-flight units — MEDIUM (abnormal-shutdown UB)

- `PinnedRing::acquire` (`streaming.cu:341-351`) has no timeout by design ("unit
  always ends"), but during `shutdown()` callbacks are **suppressed under `stop_`**
  (`:278-279`) → a consumer blocked in `acquire_layer` never sees READY and spins
  forever — and then `~PinnedRing` (`:321-329`) closes the `wake` events and
  VirtualFree's the base while the consumer may be inside
  `WaitForSingleObject(ctl_[i].wake, 20)` → wait on a closed handle = UB.
  Rule for the 27B driver: **finish or abandon the epoch (all acquires returned)
  before destructing the feeder**; the smoke test does this correctly.
- `NvmeReader::shutdown` last resort `TerminateThread` after 5 s (`:68`): if the
  worker holds `mtx_` when terminated, every later `mtx_` lock deadlocks
  (`wait/retire/on_done`). Acceptable as last resort; keep the 5 s grace.
- `wait(stream)` hands out the event handle and waits outside the lock; a concurrent
  `shutdown` may `CloseHandle` it (`:82`) → wait on closed handle. Edge race; prefer
  duplicating the handle or waiting under a shared stop check.
- IOCP double-free specifically: **not present**. retire requires `done` (all blocks
  counted), and every queued completion is counted exactly once before its unit can
  be retired, so a post-retire completion cannot exist; the `if (!units_[ui])
  continue` guard at `:263` is genuinely defensive (note: if it ever fired it would
  skip the `outstanding_` decrement — latent accounting bug, currently unreachable).
  Handle recycling cannot alias a stale completion for the same reason. Handle/file
  closes all happen under `mtx_` after thread join.

### N7 — acquire/release pairing unenforced (silent wrong-epoch data on caller bug) — MEDIUM

- `release_layer(e)` without a successful `acquire_layer(e)`: slot may be FILLING;
  release stores FREE (`streaming.cu:353-356`), `arm_locked` then claims the slot for
  epoch e+slots while the old unit's ReadFiles are still in flight → two DMA streams
  interleaving into one slot → silent garbage.
- Double-acquire of a released epoch: the stale acquirer eventually CASes the slot
  IN_USE when the **next** epoch publishes it and returns the wrong epoch's bytes —
  silent. Slots carry no epoch tag.
- Both are caller-bug classes; today's only consumer discipline is the header comment
  ("strictly sequential"). Cheap hardening: assert slot state == IN_USE inside
  `release_layer` (return early / set fatal otherwise) instead of blindly storing
  FREE.

### N8 — `build_blocks_locked`/`submit` overflow guards exist only on the feeder path — LOW/MEDIUM

`LayerFeeder::begin_epoch` dies when a plan's span exceeds `slot_bytes()`
(`streaming.cu:389-390` — verified: max shard 383,865,448 B + alignment expansion ≤
385,875,968 B slot = 184×2 MiB, ~2.0 MB slack). But `NvmeReader::submit` is public:
a direct caller with an oversized plan gets silent OOB writes into the ring (dst is
u32 cursor, wraps past 4 GiB only for absurd plans). `r.offset+r.len > f.size` is
checked (`:129`) — good shard-bounds check; `r.offset+r.len` itself can wrap u64
(caller-supplied). Keep feeder-only usage or add the same span check to submit.

### N9 — `PinnedRing` ctor failure paths — LOW/MEDIUM

- `cudaHostRegister` failure → VirtualLock fallback **silently ignores per-slot
  VirtualLock failure** (`streaming.cu:309-311`, `/* best effort */`) and sets
  `locked_=true` regardless. Correctness is preserved (pageable + staged copies are
  valid; CPU GEMV unaffected) — the cost is performance and standby-list paging.
  Pre-flight should log/warn when `ring_pinned()==false`.
- Any `die()` in the ctor after `cudaHostRegister` (e.g. `CreateEvent` failure at
  `:317`) leaks the registration + VirtualAlloc + earlier events (dtor never runs on
  a throwing ctor). For a retry-capable driver this pins memory up to the WDDM cap.
  Also `cudaSetDevice(0)` return ignored (fine on the 1-GPU contract).

### N10 — model_file.cpp parses the index without any bounds checks — LOW (corrupt/truncated index ⇒ AV, not silent)

`src/model_file.cpp:21-37`: `take<T>` reads are never checked against the blob size —
a truncated `INSIDX01` index reads past the vector (host AV, loud). `h.count` is
unbounded → the tensor loop walks off the blob. The per-tensor **mapping** bound
check exists (`:36` `payload_offset_+off+bytes > mapped_bytes_` → throw — good shard
bounds check) but `off+bytes` can wrap uint64 (corrupt index ⇒ wild `t.data`).
Path handling: the embedded path is opened verbatim — no canonicalization/traversal
guard (by design for local indexes; the index is trusted input). Fix when touching
this file: check `p` stays within `blob.data()+n` at each take; compute the bound as
`off <= mapped-payload && bytes <= mapped-payload-off`. NOTE: **no C++ INSIDX02
consumer exists yet** (grep: zero hits outside tools) — index27.py's INSIDX02
(magic/version/shape-header/66-shard table with crc32/1606-vision tensor table,
self-read verified) is currently write-only; the Phase-A loader must replicate these
checks from day one, including re-verifying shard crc32 at startup (crc is embedded
but nothing reads it yet).

### N11 — `CpuPool` (worker pool) — races audited: SAFE as written; contract notes

- Generation packing `(gen<<32)|ticket` in one atomic + `launch_mut_` serialization:
  a straggler's `drive(g)` CAS can never touch gen g+1's fn (different slot), and
  slot g is not republished until every ticket of g has **returned** (left hits 0
  only after `fetch_sub` post-fn), so fn/ctx lifetime is immutable during execution.
  No ABA, no stale-fn. Verified `launch`/`drive`/`worker_main`
  (`insignia_cpu.hpp:229-301`).
- Deadlock notes (documented in-code, restated): `launch()` from inside a job
  self-deadlocks on `launch_mut_` (non-recursive); `~CpuPool` joins workers — a job
  blocking forever wedges exit. Nested-launch detection is a one-line
  `assert(!in_job)` away if wanted.
- `SetThreadAffinityMask(1<<i)` failure ignored (perf-only). INSIG_CPU_THREADS ≤ 32
  enforced; shift stays < 32 — no UB.
- `gqa_decode_cpu`'s `thread_local GqaScratch` + tickets write disjoint slices;
  merge order fixed → deterministic. `gqa_head_range` smem layout verified: rows of
  `sbuf[6][72]` are 288 B (32-aligned), `pbuf[6][64]`/`obuf[6][256]` 32-divisible; n
  ≤ 64 ⇒ padded loop bound ≤ 64 — no overflow. Neutral partials (NINF m) correctly
  weighted 0 in the merge.

### N12 — fp8.cu at odd T — contract-correct, one assert short

- Alignment: launchers require `cols % 128 == 0` (`fp8.cu:53,97,187`) ⇒ row stride
  multiple of 128 B ⇒ `uint4`/`float4` loads aligned **iff the base is 16-aligned**
  (see N5 — base alignment is the caller's/ring's problem).
- Odd T: `fp8_gemm` A-tile always loads 64 rows — caller must zero-pad x16 to 64 rows
  (`linear_batch`'s stage_a does; `test_fp8.cu:103-105` does). Output: store guard is
  per 16-row tile (`wm*16 < T`), so T=3 writes rows 0..15 → **y buffer must be
  64·rows floats**; only the comment at `fp8.cu:182-183` says so. Add a launcher
  assert or document in the 27B prefill (risk register #6 half-open).
- `fp8_gemv2` staging 2·cols·4 B throws > 99 KB (cols 17408 ⇒ pair path must use
  fp8_gemm) — good, matches the guard.

### N13 — i4 scale-dtype confusion — CLEAN today, one asymmetric dispatch

All i4 scale reads are `__half` (`mxfp4_i4.cu:11-13`, `gemm.cu:330,406,498`,
`prefill.cu:30`); `matrix()` keys `i4` on `s.dtype==f16` and byte-validates
(`qwen35.cu:10-13`) — note i4 (rows·cols/64·2) and u8 (rows·cols/32) have **identical
byte counts**, so the dtype tag is the only discriminator (an index that mislabels
F16-as-u8 would pass the size check and silently misdecode). Residual asymmetry: the
pair ab path (`decode.cu:67-69`) dispatches on `ma.insig4` only — if a/b ever had
different scale dtypes, b would be misread (same-quantizer assumption; footgun).

### N14 — run_nll / nll residual odds and ends — LOW

run_nll duplicates nll.cu ~90 lines (incl. the fixed row_logp); both share C6/C7.
`chunk` argv in nll.cu: `_wtoi` garbage → 0/negative → `prefill_chunk` 1..64 throw —
safe. T is capped by `remain` and chunk; buffers sized by chunk ⇒ consistent.
`logitsT` = 64·vocab·4 ≈ 63.5 MB per run (leaked on exit — C11).

---

## Part 3 — VRAM/RAM ledger overrun paths (mission item 3)

1. **cudaMalloc fails mid-load (device)**:
   - `TieredStorage::acquire`: frees the partial `d` and rethrows (`storage.cu:9`) —
     clean; `make_room` throws "tensor exceeds device budget" / "budget exhausted by
     pinned tensors" — loud. Evictions already synced before free — no UAF.
   - `DecodeWorkspace` ctor: Nth-alloc throw leaks allocs 1..N-1 (C11) — process
     exits, acceptable; a long-lived host loading/unloading models repeatedly would
     leak VRAM until context reset.
   - Mid-load failure inside a capture: impossible-by-warmup today, but if it ever
     fires it hits C4 (stranded stream). Pre-touch + assert resident-set before
     capture remains the right hardening.
2. **Pinned cap during ring registration**: `cudaHostRegister` of 1472 MiB against
   the ~7.95 GiB WDDM cap — far under; on failure the VirtualLock fallback keeps
   correctness (N9). The hazard is a *later* second registration (e.g. a second
   feeder or pinned staging buffers for the CPU tier) stacking toward the cap — the
   driver should account all `cudaHostRegister`ed bytes in one place. WDDM cap hit ⇒
   fallback path ⇒ staged copies ⇒ throughput drop, not corruption.
3. **ctx growth past KV during graph-free 27B decode (C1 class)**: every eager entry
   re-checks (`decode.cu:45,131,134`) → loud throw, no overrun. The exposure is (a)
   any 27B-specific commit loop with its own arithmetic (add the device belt:
   max_context into a pos_dev slot + clamped store_kv/spec_commit — spare slots
   exist, pos_dev is a 16-int allocation with 8 used), and (b) the 9B graph loop,
   now bounded by the ctx>4090 refusal. 27B KV sizing note: at ctx 4096, f32 KV for
   16 full-attn layers = 16·2·4096·1024·4 B ≈ 512 MiB (kvrow 1024) — must be in the
   VRAM/RAM ledger with the 21-layer pinned block (~8 GiB) + ring + workspace
   against 12,282 MiB.
4. **Committed-stream overrun**: bounded by want_total+7 ≤ ctx-9 ≪ 16384 at 9B
   (host refusal). A 27B driver reusing `host_committed[16384]` must re-derive the
   bound from its own want_total.

## Part 4 — Determinism: remaining flakiness sources (mission item 4)

- The historical smem-race class is **fully closed** (H3 table above; verified by
  read, every `__shared__` reduction slot is write-once-between-barriers or
  barrier-isolated). RoPE `nsc` in both kernels; `smx/sden` in both gqa kernels;
  `red[8]/red[9]` in both row_logp twins.
- Read-uninitialized sweep: v21/v21_i4/ab_i4 A-tail rows are zeroed by stage_a
  (`decode.cu:36,73`); ab_i4 stale y rows T..15 live inside 64-row `pf_a/pf_b` and
  are never read (params_batch reads t<T); `argmax_fast` 0-scratch sentinel is valid
  under its total-order key. fp8_gemm y rows T..63 stale-by-contract (N12).
- CPU tier: fixed ticket→disjoint-output mapping; deterministic merge order;
  Remez exp is deterministic; `thread_local` scratch does not affect results.
  `INSIG_CPU_FP8_LUT` A/B changes numerics by design (debug flag).
- Remaining nondeterminism-in-parity-runs risks are all *reference/tooling* side:
  **H1** (reference_pf_i4 conv bug — will make dump-pf cosines look flaky/wrong at
  t>0), H6 (stale tools), H8 (wrong default checkpoint). Engine-side, the only
  order-dependent float behavior is the documented per-block promote order (fixed).

## Part 5 — Pre-flight checklist for the 27B driver (concrete asserts)

Index / loader (before any weights touch VRAM/RAM):
1. `magic == "INSIDX02" && version == 2` (loader must reject INSIDX01 for 27B);
   shape header == `(5120, 64, 248320, 24, 4, 48, 16, 17408, 4)`; `shard_count == 66`;
   `tensor_count == 1606 - vision_skipped` (matches index27 invariants).
2. Every shard: `file_bytes` matches on-disk size; crc32 (embedded) matches a
   startup sweep (full sweep once per install; spot-check per run). CRC is embedded
   but no C++ reads it yet — wire it or run `tools/index27.py --no-crc=false`-style
   verification before unattended launches.
3. Parser hardening when writing the INSIDX02 loader: every `take` bounds-checked
   against blob end; `off/bytes` checked without u64 wrap (model_file.cpp's
   `:36` check needs the wrap fix — N10).

Dtypes / geometry (per layer, first load):
4. `A_log.dtype == F32` — **will FAIL today** (on-disk BF16, N1): convert at index
   time or reject. `dt_bias/conv1d/*.norm/q_norm/k_norm == BF16`;
   F8 weights have `.scales` BF16 `[ceil(r/128), ceil(c/128)]`; embed/lm_head kind
   declared and dispatched (N3: `switch(kind)` with throw on unknown).
5. `WKind::fp8/bf16` matrices route to fp8/bf16 kernels — assert the dispatch table
   is total before the first token (N3).
6. Post-parameters sanity: dump `a[h]`: `exp(a[h]) ∈ (0,1)` and `!= 1.0f` for all
   heads (R4 catch for N1-class mistakes).

Alignment / ring:
7. For every F8 weight base: `(base & 15) == 0` — expected to FAIL on raw shard
   plans (data_start 2600 ≡ 8 mod 16) until pad-at-F8 lands (N5).
8. `ring.slot_bytes() (368 MiB = 385,875,968) ≥ max_plan_span + 4096` (max shard
   383,865,448 B — ~2.0 MB headroom); `ring_pinned() == true` else log the VirtualLock
   fallback and derate the throughput budget; slot base 4096-aligned (submit asserts).
9. No pending GPU reads of a slot at `release_layer` — assert via
   `cudaStreamQuery(copy_stream) == cudaSuccess` or sync (N4).

Budgets:
10. 9B gate (unchanged): `prompt+max_new+16 ≤ 4090` (throws already); weights
    4,887,547,392 B < 6 GiB budget; workspace ~430 MB.
11. 27B ledger: pinned RAM = ring 1472 MiB + host activations + any staged buffers
    ≤ ~7.95 GiB WDDM cap (account ALL cudaHostRegister/cudaHostAlloc in one
    counter); VRAM = pinned layer block + KV (16·2·ctx·1024·4 B f32, or halve for
    bf16) + workspace + logits 2×vocab ≤ 12,282 MiB with ≥ 1 GiB slack.
12. `DecodeWorkspace ctx ∈ 1..4096` (ctor throws); `want_total + 7 < 16384` if the
    committed stream is reused.

Process hygiene for unattended runs:
13. Handle count stable across epochs (N2 — fix before launch); feeder epoch fully
    released before teardown; no thread parked in `acquire_layer` at shutdown (N6);
    4 link-broken bats (H4) excluded from build gates; default 9B index points at
    `-good` (H8) if the 9B regression gate runs.

---

## Ranked summary (probability × damage)

| # | hazard | prob | damage | class |
|---|---|---|---|---|
| 1 | N1 A_log BF16-as-F32 (27B) | certain | silent wrong tokens | CRITICAL |
| 2 | N3 missing fp8/bf16 dispatch in decode.cu linear/embed/lm_head | certain on 27B wiring | silent garbage | CRITICAL |
| 3 | N4 release_layer vs in-flight GPU async read | certain if unrestrained driver | silent torn weights | HIGH |
| 4 | N5 F8 base ≡8 mod 16 in ring slots | certain at first streamed GEMV | misalign crash (loud) | HIGH |
| 5 | N2 retire() event-handle leak | certain over long runs | unattended run death | HIGH |
| 6 | C1-belt device guards absent (store_kv/spec_commit/score) | low (host refuse holds) | critical if regressed | MED |
| 7 | H1 reference_pf_i4 conv bug | certain at T>1 | poisons parity hunts | MED (tooling) |
| 8 | N6/N7 teardown + pairing contracts unenforced | edge cases | UB / silent wrong epoch | MED |
| 9 | C4/C5 capture & dtor error paths | error-path only | masks real error | MED |
| 10 | C7 wcstombs -1 wild write | non-ASCII argv only | host memory corruption | MED-LOW |
| 11 | C6 unchecked targets/logp mallocs | OOM only | deferred IMA | LOW-MED |
| 12 | N10 model_file parse OOB / u64 wrap | corrupt index only | AV (loud) | LOW |
| 13 | C9 unvalidated token ids | garbage argv | GPU OOB read | LOW-MED |
| 14 | C3 graph-pointer eviction (9B, latent) | accident-protected | silent wrong weights | LOW (9B) |
| 15 | C11/H4/H6/H8 leaks, broken bats, stale tools, bad checkpoints | certain (present) | hygiene/friction | LOW |

File:line index (new code): streaming.cu — submit :152-172, retire :239-248 (N2),
worker_loop :250-284, PinnedRing ctor :289-319 (N9), acquire :341-351, dtor :321-329
(N6), arm/on_done :399-428, acquire/release_layer :430-450 (N4/N7); insignia_cpu.hpp
— CpuPool :217-339 (N11), gqa :794-952, deltanet params :634-643 (N1 twin); fp8.cu —
:53,97,99,182-189 (N12); model_file.cpp — :21-37 (N10); qwen35.cu — :5 (C5),
:8-22 (matrix kinds); decode.cu — :31-41 (N3), :46,97-105 (N3), :82,128 (N1),
:154-155 (H7), :207-213 (C8 fix); generate.cu — :112-115 (C1 fix), :51-53/104-106
(C7), :62-66 (C6); prefill.cu — :62,107 (H3 fix), :88-98/287-301 (C1 belt),
:287-301 spec_commit; nll.cu/generate.cu row_logp red[8]/red[9] (H3 fix).
