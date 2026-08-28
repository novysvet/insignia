# w3 audit: CUDA graph × tiered-storage hazards — 2026-08-25

Scope: interaction of `capture_step`/`capture_spec` graph replay with `TieredStorage`
acquire/evict, position tracking, spec-reject rollback, and stream-capture legality.
Files read in full: `src/decode.cu`, `src/storage.cu`, `include/insignia_storage.hpp`,
`src/generate.cu`, `src/nll.cu`; supporting: `src/qwen35.cu`, `src/prefill.cu`,
`src/ops.cu`, `src/qwen_kernels.cu`, `src/model_file.cpp`, `include/insignia_decode.hpp`,
`include/insignia_qwen35.hpp`, checkpoint header of `build/qwen35-insig4-good.safetensors`.
Read-only audit; no builds, no git changes. Code state = commit `92e1028` + dirty tree
(`src/decode.cu`, `src/generate.cu`, `include/insignia_decode.hpp` modified — this audit
describes the current working tree).

## 0. Verdict table (wave-1 findings vs current code)

| # | Wave-1 claim | Verdict today | Severity |
|---|---|---|---|
| 1 | Post-capture LRU eviction frees pointers baked into graphs | **Latent, not live** — two independent reasons (§1) | design hazard |
| 2 | KV-full guard bypassed by graph replay | **LIVE memory-corruption path** via unbounded `max_new` + host-mirror drift (§2) | critical |
| 3 | Full-attn KV not restored on spec reject | **Structurally safe on all paths** — stronger invariant than wave-1 assumed (§3) | none (fragile) |
| 4 | Pageable memcpy / illegal ops inside capture | **Capture bodies are clean**; miss-path acquire would fail fast; exception-safety gap (§4) | robustness |
| 5 | Storage frees/reallocs vs graph pointers | Fresh `cudaMalloc` per miss; same-VA reuse after `cudaFree` = silent wrong-weights hazard (§5) | latent |
| 6/7 | Fix design + pinned-during-graph API | §6, §7 | — |

---

## 1. Eviction vs graphs: full lifecycle trace

### 1.1 Storage semantics (all of `src/storage.cu`, 12 lines of logic)

- `acquire(name)` — storage.cu:9. **Hit**: `pins++; tick++` (pure host map ops, no CUDA
  call) → returns existing device pointer. **Miss**: `make_room(t->bytes)` → fresh
  `cudaMalloc` → `cudaMemcpyAsync` H2D from the mmap view + `cudaStreamSynchronize` →
  insert entry with `pins=1`.
- `release(name)` — storage.cu:10: `pins--` (floor-guarded). Entry **stays resident**.
- `make_room(bytes)` — storage.cu:8: evicts **lowest-tick entry with `device && pins==0`**;
  `cudaStreamSynchronize(stream_)` then `cudaFree(victim)`. Pinned entries are never
  chosen → *pinning is already an eviction shield*.
- `clear()` — storage.cu:11: frees everything **regardless of pins** (dtor path only).
- Host source pointers: `ModelFile` mmaps the payload once (`model_file.cpp:30-37`);
  `TensorView::data` is stable for process lifetime. Only the **device** side churns.

`Qwen35Weights::matrix` (qwen35.cu:7) acquires **two** entries per call
(`base+".weight"`, `base+".scales"`); `release(base)` releases both (qwen35.cu:11).

### 1.2 Capture-time lifecycle (what actually runs)

```
generate.cu flow:
  prefill loop            (123-127)   -> touches embed + all 32 layers + lm_head + norm
  append_committed_host   (130)       -> no acquire
  warm = d.spec_step(first) (143)     -> EAGER: mtp_layer + pair verify
                                       -> touches mtp.* tensors (ONLY place they're touched)
  d.capture_spec()        (169)       -> BeginCapture(ThreadLocal) [decode.cu:233]
                                       -> re-runs the same code path
                                       -> every acquire() HITS (entry exists, pins++/tick++)
                                       -> NO make_room, NO cudaMalloc, NO sync inside capture
                                       -> EndCapture -> Instantiate [decode.cu:240-242]
                                       -> after EndCapture all pins are back to 0
  replay loop             (178-188)   -> see 1.3
```

Key facts:

1. **During capture, every graph-referenced tensor is already resident** because the eager
   warmup (`generate.cu:143`) executes the identical code path (`mtp_layer` +
   `prefill_chunk_device(T=2)` + spec kernels) immediately before `capture_spec`
   (`decode.cu:232-243` mirrors `spec_step` `decode.cu:213-231` exactly). Acquire-on-hit
   performs no CUDA API call, so capture is clean (§4).
2. **After capture all entries have `pins==0`** — every acquire inside the captured body
   is matched by a release before `EndCapture`. The graph exec holds raw device pointers
   into `entries_`, but *nothing pins them*.

### 1.3 Enumerate EVERY acquire() caller between graph replays

Replay-loop instruction inventory (`generate.cu:177-188`):

| Call | Code | acquire calls |
|---|---|---|
| `d.spec_graph_step()` ×4 | decode.cu:244-248 → `cudaGraphLaunch` only | **0** |
| `d.committed_count()` | decode.cu:203-208 → D2H memcpy + sync | **0** |
| `d.read_committed()` | decode.cu:209-212 → D2H memcpy + sync | **0** |
| EOS scan, printf | host only | **0** |

Also zero between `capture_spec` (169) and loop entry (172-176: event record/sync), and
after the loop (191-197: count readback + printf). **No thread, no other model, no
prefill, no `clear()` runs between replays.** Therefore:

> **Eviction between replays is impossible in the current program** — not because the
> design is safe, but because nothing calls `acquire()` on the replay path.

### 1.4 Budget math: eviction cannot fire even if something did acquire

- Checkpoint payload (`build/qwen35-insig4-good.safetensors` header, 700 tensors):
  **4,887,547,392 B = 4.553 GiB**. Breakdown: 32 layers 3,677 MB, embed 540.4 MB,
  lm_head 540.4 MB, mtp.* 129.3 MB, final norm ~0.
- Budget: `6ull << 30` = 6 GiB (`generate.cu:115`, `nll.cu:58`).
- `make_room` evicts only `while(used_+bytes>budget_)`; max possible `used_` = 4.553 GiB
  < 6 GiB → **the eviction loop body never executes for the 9B model. `used_` never
  exceeds the budget by construction.**

So wave-1 item 1 is **latent, protected by two independent, undocumented accidents**:
(a) zero acquire callers on the replay path, (b) checkpoint < budget. Either a second
sequence prefilled mid-generation, a budget cut, or the 27B (25.65 GB weights, synthesis
model facts) turns it live.

### 1.5 Does capture itself evict mid-capture? (the dangling-already-captured-node question)

Structurally, the dangerous sequence exists in the code: during capture, layer L's
tensors are acquired then released (`pins` back to 0, entry resident); if a *later*
acquire inside the same capture **missed**, `make_room` would be allowed to pick the
just-released layer-L entry (pins==0, oldest tick) and `cudaFree` a pointer already
baked into a captured kernel node → that node dangles.

In the current program this **cannot happen and would not be silent if it did**:

- It cannot happen because the warmup guarantees all hits (§1.2). The only tensors
  prefill doesn't touch are `mtp.*`, and the warmup `spec_step` touches exactly those.
- If it did happen (fresh process path calling `capture_spec` without warmup — note
  `capture_step`, decode.cu:249, currently has **zero callers** and no built-in warmup):
  the miss-path runs `cudaStreamSynchronize(stream_)` on the *capturing* stream
  (storage.cu:9) and `cudaMalloc` — both are "potentially unsafe" APIs under
  `cudaStreamCaptureModeThreadLocal` (decode.cu:233) → the capture is **invalidated**;
  `check()` throws mid-body, `EndCapture` is never reached with a valid graph, the graph
  is never instantiated nor launched. Fail-fast, not silent. Caveats: (i) the exception
  leaves the stream **still in capture mode** (no cleanup — §4); (ii) under
  `cudaStreamCaptureModeRelaxed` the same sequence *would* instantiate a graph holding a
  freed pointer → silent IMA/corruption at replay. ThreadLocal is an accident shield,
  not a design.

### 1.6 Lifecycle pseudocode (state after each phase)

```
T0 prefill+warmup:  entries_ = {all 700 tensors}, pins=0 all, used_=4.55GiB < 6GiB
T1 capture_spec:    for each kernel: acquire(hit)->pins 1->0 ... graph_exec{ptr_k}
                    entries_ unchanged; nothing pins graph_exec's pointers
T2 replay N:        cudaGraphLaunch(graph_exec)   // reads entries_ pointers raw
                    (no acquire anywhere on path)
T3 hypothetical acquire(miss) at T2:  make_room -> pins==0 victims INCLUDE every
                    graph-referenced tensor -> cudaFree -> graph_exec nodes dangle
                    // cannot happen today: (no caller) AND (used_+bytes<=budget)
T4 process exit:    ~Qwen35Decode -> graph exec leak? (no cudaGraphExecDestroy —
                    minor; process exit reclaims) ; ~TieredStorage -> clear() frees all
```

`cudaGraphExec_t graph_, spec_graph_` (insignia_decode.hpp:50) are never destroyed —
cosmetic today (process-lifetime), matters if decode objects ever get torn down and
recreated while storage lives on.

---

## 2. KV-full guard bypass — LIVE bug (upgraded from wave-1)

### 2.1 Where the guard lives and who checks it

- Host-side guard: `prefill_chunk_device` — decode.cu:45
  `if(x_.position+T>x_.max_context)throw std::runtime_error("KV cache full")`. Also
  decode.cu:127 (`attention_layer`) and decode.cu:130 (`forward_token`).
- Device-side truth: `pos_dev[0]` (`x_.pos_dev`), mutated by `addi_kernel` (decode.cu:103,
  prefill.cu:271), `bumpi_kernel` (decode.cu:119/255), and `spec_commit_kernel`
  `pos[0] -= 1` on reject (prefill.cu:300). Layout documented at prefill.cu:275-276.
- The graph contains the *kernels* but the guard is **host code that runs once, at
  capture time**. Replays never re-evaluate it.

### 2.2 Host-mirror drift across replays

- Eager `spec_step`: host mirror is *corrected from device* — decode.cu:222-224
  (`tail` D2H of `pos_dev`, `x_.position=tail[0]`). No drift in eager mode.
- Graph `spec_graph_step`: decode.cu:247 `x_.position+=2;` — **assumes accept**. On every
  reject the device advanced +1 (prefill.cu:300), so the mirror over-counts by 1 per
  reject. The comment ("corrected to the device truth at each host read") is wrong for
  the graph loop: `committed_count()` (the only host readback in the loop) reads
  `count_dev`, never `pos_dev`; `generate.cu` never reads `d.position()`. Drift persists
  for the whole generation. Consequence today: none *inside* the loop (nothing consumes
  the mirror there), but any post-loop mixed-mode use (eager `prefill_chunk`,
  `forward_token`, `set_position` — which would also clobber device pos with the stale
  host value, decode.cu:120) inherits a wrong position: spurious "KV cache full" throws
  or wrong RoPE/KV slots.

### 2.3 The live corruption: `max_new` is unbounded, `ctx` is capped

`generate.cu:112-114`:

```c
int ctx = int(tokens.size()) + max_new + 16;
if (ctx > 4090) ctx = 4090;            // cap from DecodeWorkspace's 1..4096 guard
                                       // (decode.cu:12, gqa score[4096] smem, prefill.cu:106)
```

`want_total = tokens.size() + max_new` (generate.cu:150) is **not** capped. In graph mode
position advances in lockstep with the committed count (`pos == count` after the warmup;
both +2 on accept, both +1 on reject), and the loop only stops at
`count >= want_total` or EOS (generate.cus 179-187). The inner 4-step batch over-commits
by at most +7 (`generate.cu:179` comment; worst case count = want_total+7 →
`pos = want_total+7`).

So for `prompt_len + max_new >~ 4083`, e.g. prompt 10 + `max_new 5000`:

- `ctx = 4090`, `want_total = 5010` → the loop happily replays past position 4090.
- **Eager path**: `prefill_chunk_device` throws "KV cache full" at pos+2>4090 — loud.
- **Graph path**: no guard runs. `store_kv_batch_kernel` writes
  `kc[pos*1024+i] / vc[pos*...]` (prefill.cu:88-93) with **no bounds check** → writes
  past the `ai=7` slot of `kv_keys`/`kv_values` (sized `8*4090*1024` floats, decode.cu:14)
  into adjacent allocations — for prompt 10 / max_new 5000 that is ~920 positions ×
  8 attn layers × 2 × 1024 floats = **~60 MB of out-of-bounds GPU writes**, silent.
  Additionally `gqa_prefill_kernel`'s `__shared__ float score[4096]` (prefill.cu:106)
  is indexed by absolute token count (`for j < tokens`, `score[j]=z`, prefill.cu:120) →
  shared-memory overflow past 4096 → wrong softmax or illegal address.

**Fix (minimal, host):** bound the loop by both counters — stop when
`count >= want_total` **or** a position readback ≥ `ctx-2`. **Fix (real, graph-internal):**
write `max_context` into a device slot (there is spare room: `pos_dev` is a 16-int
allocation, decode.cu:15, 8 used) and make `spec_commit_kernel` / `store_kv` honor it
(e.g., commit kernel sets a "full" flag and the host loop drains) — the guard must live
inside the graph to exist at replay time. Also make `spec_graph_step` not touch the host
mirror (or refresh it from `count_dev`: `pos == count` invariant makes the mirror
redundant in graph mode).

---

## 3. Full-attn KV not restored on spec reject — verified safe (and why)

`spec_rollback_kernel` (prefill.cu:305-311) restores `delta_state`, `conv_state`,
`hidden` — **not** `kv_keys`/`kv_values` (nor `mtp_keys`/`mtp_values`). A rejected step
leaves the phantom row-1 K/V at slot `p+1` (pair stored at `p`,`p+1`; reject commits one
token so new pos = `p+1`).

Exit-path enumeration (graph loop, generate.cu:178-188):

1. `count >= want_total` → break (line 181): no further model execution (only
   read_committed + printf). Stale row never read. Safe.
2. EOS → break (line 186): same. Safe.
3. Exception (`spec_graph_step`/memcpy throws) → unwinds to catch (line 208), process
   exits. Safe.
4. Host abort mid-stream: process dies. Safe.

Continuing paths: the only continuation is another spec step, and even a **mode switch**
is safe, because the stale row sits exactly at the *next-free-slot* index (`pos`), and
every reader of that slot first writes it: `attention_layer` does `store_kv` then
`gqa_decode` (decode.cu:127); the pair path does `store_kv_batch` (both rows) then
`gqa_prefill` reads `j <= pos+t` (prefill.cu:104,114); mtp uses its own cache and
rewrites `pos[7]` every step (`spec_prologue`, prefill.cu:277). Since slot indices
advance monotonically with committed tokens, **stale KV rows are always overwritten
before their first read, on every path** — a stronger property than wave-1's "every
reject is followed by another pair". Not a bug; add a one-line comment at
prefill.cu:305 documenting the invariant, because it silently depends on
store-before-read ordering inside every consumer.

---

## 4. Stream-capture legality audit (both capture regions)

Inventory of the captured bodies:

- `capture_spec` (decode.cu:233-240): `spec_prologue`, `mtp_layer`, `spec_setup`,
  `prefill_chunk_device(pf_tokens,2)`, `spec_commit`, `spec_rollback` — all kernel
  launches, plus **one** `cudaMemcpyAsync` D2D (`hidden <- pf_x`, decode.cu:102 — legal
  memcpy node) and **one** `cudaMemsetAsync` (`argmax_fast` scratch zero, qwen_kernels.cu:60
  — capturable). `pair==true` for T=2 so `linear_batch`'s memset tail path (decode.cu:37)
  is not captured (and would be legal anyway).
- `capture_step` (decode.cu:250-256): `copyi_kernel`, `embed_dev`, `forward_body`,
  `argmax_fast`, `bumpi` — kernels only. The pageable H2D copy feeding it
  (`forward_token`'s `&token` copy, decode.cu:130; `step`'s copy, decode.cu:260) is
  correctly **outside** the capture.
- **No pageable-host memcpy inside either capture region.** The spec graph is
  deliberately fed device-side (`pos[1]` pending token, `spec_setup_kernel`
  prefill.cu:280-284), so no fixed host address is baked in — the "replay reads stale
  host address" hazard does not exist today. `prime_spec`'s H2D from pinned
  `next_host` (decode.cu:189-192) runs only in eager mode.
- **No cudaEventRecord inside capture** (events in generate.cu:117-120/172-176/189 are
  outside capture). Kernel launchers contain no runtime API calls except the memset
  above (grep over mxfp4.cu, mxfp4_i4.cu, gemm.cu, qwen_kernels.cu, attention.cu,
  deltanet.cu). `deltanet_prefill`'s lazy `cudaFuncSetAttribute` (prefill.cu:266) was
  already triggered by the warmup and is not capture-unsafe regardless.
- Acquire-on-hit inside capture performs no CUDA call (storage.cu:9 hit path) — the
  resident-set precondition is what makes capture legal; it is enforced only by
  generate.cu's call order (warmup at 143 before capture at 169), **not by any assert**.
- **Exception-safety gap:** no try/catch around either capture body. Any throw between
  Begin and End (e.g., `matrix()` dtype check, qwen35.cu:8-9; a miss-path acquire,
  §1.5) leaves `x_.stream` permanently in capture mode — every subsequent stream op
  errors out. Fix: wrap body in try { } catch { cudaStreamEndCapture(stream,&g);
  cudaGraphDestroy(g); throw; }. Related pin-leak: `matrix()` acquires both entries
  *before* its dtype checks; a throw leaks pins (retention, not corruption) — check
  dtypes before the second acquire.

---

## 5. Storage pointer stability (evict → re-acquire)

- No pooling: every miss is a fresh `cudaMalloc` (storage.cu:9); eviction is a raw
  `cudaFree` (storage.cu:8). Two failure shapes if a live graph's tensor is evicted and
  later re-acquired:
  1. **New address**: graph nodes read freed VA → IMA or stale bytes → crash/garbage.
  2. **Same address** (common on Windows — the driver frequently reuses a just-freed
     equal-size VA range, and evict/reload cycles same-size layer tensors): the graph
     *silently reads the replacement tensor's weights*. No crash, wrong logits. This is
     the worst mode and it is plausible in any future budget-tight loop.
- Host views: mmap stable for process lifetime (model_file.cpp:30-37) — no host-side
  hazard.

---

## 6. Fix design

### (a) 9B — graph-safe = pin every graph-referenced tensor for the graph's lifetime

Graph-referenced set (spec graph) = **the entire checkpoint** — `mtp_layer` +
`prefill_chunk_device(2)` + `forward_body` touch embed, all 32 layers, final norm,
lm_head, and all `mtp.*`:

| component | bytes |
|---|---|
| 32 layers (weights+scales) | 3,677 MB |
| embed_tokens (w+s) | 540.4 MB |
| lm_head (w+s) | 540.4 MB |
| mtp.* | 129.3 MB |
| final norm | ~0 |
| **total pinned** | **4,887.5 MB = 4.553 GiB** |

Feasibility against the 6 GiB budget: 4.553 GiB pinned → 1.447 GiB slack for any future
acquire (`make_room` then throws "device budget exhausted by pinned tensors" — loud —
rather than evicting). Workspace is *outside* the budget (plain `cudaMalloc`,
decode.cu:10-26): ~430 MB at ctx 4090 (kv 268 MB, delta+snap 101 MB, mtp kv 33 MB,
pf_* 19 MB, logits 2 MB, misc). Total VRAM ≈ 4.55 + 0.43 + ~0.4 (context+graph exec,
no graph memory pool since all buffers are external) ≈ **5.4 GiB of 12 GiB** — fits with
room to spare. Pinning all resident entries is the *simplest correct* policy for 9B
(resident set == referenced set == whole checkpoint after warmup).

### (b) Eviction → graph invalidation protocol (when pinning is partial)

Primary: pins (above) — eviction of graph tensors becomes impossible by construction
(`make_room` skips `pins>0`, storage.cu:8). Secondary, if a future budget ever forces
partial pinning: keep a `generation_` counter in TieredStorage, incremented on every
`cudaFree` in `make_room`; `Qwen35Decode` records the generation at capture; before each
`cudaGraphLaunch` compare — on mismatch, re-acquire-and-release the referenced set (so
entries are resident at *new* addresses), re-capture the graph, and
`cudaGraphExecUpdate(exec, newGraph)` to patch pointers into the existing exec
(cheaper than re-instantiating ~600 nodes; topology is identical). Never launch a stale
exec. This is the protocol to reach for *only* when the pinned set can't fit; for 9B it
cannot trigger, and the simpler invariant (pin-all) should ship first.

### (c) 27B — NO whole-model graph; per-layer micro-graphs not worth it

- Whole-model graph is impossible: the spec graph references every layer's weights
  (§6a logic scales: 25.65 GB referenced), which cannot be resident — the model streams
  per token through the tier hierarchy. A captured graph would need every tensor pinned
  → dead on arrival.
- And it is pointless: decode step ≈ 1.5 s/token at L21/M23/N21 (synthesis.md:57-58),
  dominated by NVMe/RAM streaming. Launch overhead ≈ ~700 kernels/step × ~5-7 µs ≈
  3.5-5 ms → **≤ 0.3 % of step time**; even the CPU-tier v2 (~0.78 s) → ≤ 0.6 %.
- Per-layer micro-graphs for the VRAM-resident block (21 layers pinned): save ~20
  launches × ~6 µs ≈ 0.12 ms/layer → ~2.5 ms of 1.5 s ≈ 0.17 %, while importing the
  entire invalidation machinery of (b) per layer. Verdict: **no CUDA graphs for 27B
  decode.** Revisit only if a step ever drops below ~50 ms (then graph the VRAM-resident
  block, pinned for graph lifetime, generation-checked per (b)).

---

## 7. Pinned-during-graph design, concretely

API on TieredStorage (include/insignia_storage.hpp):

```cpp
// sticky pin: pins++ and record in the graph pin-set; MUST hit an existing entry
// (caller warms up first) — a miss uploads here, pre-capture, which is legal.
void acquire_pinned(std::string_view name);          // pins++ ; names_ push_back
void release_graph() noexcept;                        // for each recorded name: pins--
uint64_t generation() const noexcept;                 // ++ on every make_room cudaFree
```

Who calls: `Qwen35Decode::capture_spec` / `capture_step` — after the mandatory warmup,
immediately **before** `cudaStreamBeginCapture`:

```
warmup (eager spec_step)               // generate.cu already does this (143)
for name in referenced_set:            // = every matrix()/tensor() name the body uses
    w_.storage().acquire_pinned(name)  // asserts hit; resident set now pinned
assert(storage.device_bytes() <= storage.budget_bytes())   // loud, not evict-silently
BeginCapture(ThreadLocal) ... body ... EndCapture -> instantiate
// pins stay held for the exec's lifetime
~Qwen35Decode: cudaGraphExecDestroy(graph_); cudaGraphExecDestroy(spec_graph_);
               w_.storage().release_graph();
```

Invariants table:

| # | Invariant | Enforced where |
|---|---|---|
| 1 | capture-time resident set ⊇ replay-time referenced set | warmup-before-capture (generate.cu:143→169) + `acquire_pinned` asserts hit |
| 2 | pinned entries are never evicted | `make_room` victim filter `pins==0` (storage.cu:8) — already correct |
| 3 | no CUDA API on the capture path except kernels/memset/D2D-memcpy | acquire-on-hit only; a miss throws *before* any capture-unsafe call is reached post-pin (§1.5); add try/catch around capture body (§4) |
| 4 | graph exec lifetime ⊆ pin lifetime | pin before BeginCapture, release after ExecDestroy |
| 5 | host position mirror never consulted stale | graph mode: derive pos from `count_dev` (`pos==count` invariant) or D2H pos_dev at the existing committed_count sync; delete `position+=2` guess (decode.cu:247) |
| 6 | replay bounded by KV capacity, not just want_total | host: break at pos ≥ ctx−2; device: max_ctx slot honored by spec_commit/store_kv (§2.3) |

Also cheap hardening while in here: `cudaGraphExecDestroy` in the dtor; dtype checks in
`Qwen35Weights::matrix` before the second acquire (pin-leak, qwen35.cu:7-9); a comment at
prefill.cu:305 documenting the KV overwrite-on-reject invariant (§3).

---

## nll.cu (read in full — graph relevance)

No graphs anywhere in `src/nll.cu`: it drives `prefill_chunk` (eager, guarded by the
host-side KV check each chunk, decode.cu:45) and per-chunk
`w.matrix("language_model.lm_head")` acquire/release (nll.cu:78-81) — LRU-refreshing a
resident entry; budget 6 GiB > 4.55 GiB so nothing ever evicts. The only latent overlap
with graph hazards: if nll-style teacher forcing were ever run *between* graph replays
in one process, its acquires are all hits today (no eviction) but would become the
acquire-caller that arms finding #1 under a tighter budget. (Synthesis items 4 noted
nll's missing i4 branch — that is fixed in the current tree, nll.cu:78-80 has both
branches.)

## File:line index

- src/decode.cu: 12 (ctx guard), 14-27 (workspace allocs), 30-41 (tensor/linear acquire-release), 45 (KV guard), 102-104 (D2D + pos bump + mirror), 119 (bumpi), 120-121 (set_position), 127 (attention_layer guard + store-then-read), 129-130 (forward_body/forward_token), 133-188 (mtp_layer), 189-192 (prime_spec), 213-231 (spec_step, 224 mirror fix), 232-243 (capture_spec), 244-248 (spec_graph_step, 247 drift), 249-259 (capture_step, no callers), 260 (step)
- src/storage.cu: 8 (make_room), 9 (acquire), 10 (release), 11 (clear)
- src/generate.cu: 112-115 (ctx cap + budget), 123-131 (prefill + commit + prime), 143 (warmup), 150 (want_total), 169 (capture_spec), 177-188 (replay loop, 179 +7 comment), 191-207 (drain/trim), 208 (catch)
- src/prefill.cu: 88-98 (store_kv, no bounds check), 102-165 (gqa, score[4096] smem), 271-272 (addi), 275-276 (pos layout), 277-302 (spec prologue/setup/commit, 300 pos[0]-=1), 305-313 (rollback, no KV restore)
- src/qwen35.cu: 7-11 (matrix: two acquires, dtype checks, release)
- src/qwen_kernels.cu: 60 (memsetAsync in argmax)
- src/model_file.cpp: 30-37 (stable mmap host views)
- include/insignia_decode.hpp: 50 (graph_ / spec_graph_ never destroyed)
- build/qwen35-insig4-good.safetensors: 700 tensors, 4,887,547,392 B payload
