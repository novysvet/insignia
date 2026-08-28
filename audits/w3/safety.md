# w3 safety audit — fresh eyes, 2026-08-25

Scope: recent/uncommitted work in `src/generate.cu` (run_nll + wmain), `src/nll.cu`,
`src/decode.cu` (prefill_chunk_device overloads, seam, mtp_layer, spec_step, capture_*),
`src/storage.cu`, `src/gemm.cu`, `src/mxfp4.cu`, `src/mxfp4_i4.cu`, `src/fp8.cu`,
`src/prefill.cu`, `src/ops.cu`, `include/insignia_decode.hpp` (+ dataflow into
`src/qwen35.cu`, `src/attention.cu`, `src/qwen_kernels.cu`, `src/model_file.cpp`,
`src/deltanet.cu`, build bats). Read-only audit; no builds; no git changes.

Model arithmetic used for worst cases: 9B = vocab 248320, hidden 4096; 27B deltas
(vocab 248320 × hidden 5120, inter 17408, qkv 10240) checked where the code is shape-generic.

---

## Ranked findings

### C1 — CRITICAL: graph-replay generation overruns KV cache / score buffer / committed stream (silent device-memory corruption)

The **only** capacity guards are host-side and run at enqueue time:

- `src/decode.cu:45` — `if(x_.position+T>x_.max_context) throw "KV cache full"` (prefill path, eager only)
- `src/decode.cu:127` — `if(x_.position>=x_.max_context) throw` (attention_layer, eager only)

But the production generate path (`src/generate.cu:168-187`) **replays `spec_graph_` in a
loop with no guard at all**:

- `src/generate.cu:112-113` — `ctx = prompt+max_new+16; if (ctx>4090) ctx=4090;` — ctx is
  **silently clamped**, it does not fail the request.
- `src/generate.cu:150` — `want_total = prompt + max_new` — **not** clamped to ctx.
- `src/generate.cu:179` — 4 graph replays per count check (count grows ≤2/replay, pos ≤2/replay).

When `prompt+max_new ≳ 4081` (e.g. prompt 100 + `max_new` 4000 — an ordinary CLI call),
`pos_dev` passes 4090 while the loop is still replaying. There is no device-side bound
anywhere:

- `src/prefill.cu:91-93` `store_kv_batch_kernel`: `kc[pos*1024+i]` — pos unbounded
  (`(void)max_context` at :96 explicitly discards the capacity).
- `src/prefill.cu:287-301` `spec_commit_kernel`: `committed[c] = pos[1]` — c (= `pos[5]`)
  unbounded; `committed` is 16384 ints (`src/decode.cu:20`), `host_committed` pinned 16384
  (`src/decode.cu:18`), so `want_total > ~16370` additionally overruns the committed stream
  and `read_committed`/`append_committed_host` D2H/H2D.
- `src/attention.cu:7` `gqa_decode_kernel`: `__shared__ float score[4096]` written for
  `t < tokens = pos+1` — pos ≥ 4090 is an **out-of-bounds shared-memory write** in every
  full-attn block of every replay, plus global OOB reads `kc[(size_t(t)*4+kvh)*256+d]`
  past the end of `kv_keys` (last layer's slot runs to the end of the allocation).
- `mtp_keys`/`mtp_values` (`src/decode.cu:169-170` via `mtp_pos_dev = pos[0]-1`) overrun the
  same way.

Consequences: silent garbage generation at best; device-heap corruption and sticky CUDA
context errors at worst; nondeterministic, load-dependent. The eager path is safe (it
throws via decode.cu:45), which makes the bug look like a "graph-mode flakiness".

**Minimal fix (3 pieces, all cheap):**
1. `src/generate.cu`: refuse instead of clamping — `if (int(tokens.size())+max_new+16 > 4090) return 2;`
   (or clamp `want_total` to `ctx-8`).
2. Device-side belt: in `spec_commit_kernel` wrap the writes with `if (c < 16383)`; in
   `store_kv`/`store_kv_batch` pass `max_context` and wrap with `if (pos < max_context)`
   (branch is uniform, ~zero cost — set a sticky error flag instead of writing if you want
   it observable).
3. Optionally bound `gqa_decode`'s loop: `const int tokens = min(pos+base+1, 4096);`.

### C2 — HIGH: `mxfp4_gemm_v21` cp.async pipeline race on the last K-step (nondeterministically wrong prefill GEMM)

`src/gemm.cu:272-276`:

```cpp
for (int kb = 0; kb < ksteps; kb++) {
    if (kb + 1 < ksteps) { prefetch(kb + 1, buf ^ 1); cp_async_commit(); }
    cp_async_wait_prev();            // cp.async.wait_group 1 — UNCONDITIONAL
    __syncthreads();
    dequant(kb, buf);
```

`wait_group 1` only guarantees completion of all but the **most recent** group. On the last
iteration there is no new commit, so the single outstanding group **is** the tile being
consumed: `dequant` reads `Braw[buf]` / `wmma` reads `As[buf]` before the final cp.async
lands. On all earlier iterations the just-committed prefetch is the "most recent" group and
the current tile is correctly awaited — this is purely a final-iteration bug.

Proof it's unintended: the twin kernel `src/fp8.cu:161-166` already has the fix:

```cpp
if (kb + 2 < ksteps) wait_group 1;
else                 wait_group 0;   // drains the last group
```

`mxfp4_gemm_v21` is the engine for every non-pair batched prefill GEMM
(`linear_batch`, `src/decode.cu:37-40`), i.e. all prefill chunks at T=64. The race window
is one iteration of mma work, so it usually wins — latent nondeterministic corruption of
exactly the kind that poisons parity hunting.

**Minimal fix:** copy the fp8.cu pattern into gemm.cu (replace line 275):
`if (kb + 2 < ksteps) cp_async_wait_prev(); else cp_async_wait_all();`

### C3 — HIGH (latent, one call away): LRU eviction frees pointers baked into instantiated CUDA graphs

`TieredStorage::make_room` (`src/storage.cu:8`) may `cudaFree` any cached tensor with
`pins==0`. Every `linear()` releases (unpins) its matrix **immediately after enqueue**
(e.g. `src/decode.cu:31-41`), so after `capture_step`/`capture_spec`
(`src/decode.cu:249-259`, `232-243`) **all** entries are unpinned and evictable — while
`graph_`/`spec_graph_` embed their raw device pointers as kernel-node params
(`cudaGraphInstantiate`, decode.cu:257/241). The graphs are replayed onto `x_.stream`
(decode.cu:245, 260).

Any post-capture acquire that misses cache (or any new tensor name) can evict a baked
tensor; a later re-acquire likely returns a different address; the next replay reads
freed/reused memory. `make_room`'s `cudaStreamSynchronize` does **not** protect against
this (it covers in-flight kernels, not future replays of stale pointers). Today's
`generate.cu` happens to issue no acquires after `capture_spec` (the replay loop only calls
`spec_graph_step`/`committed_count`/`read_committed`), so this is currently dormant — but a
single `prefill_chunk` (e.g. a second prompt) after capture is enough.

**Minimal fix:** at the end of a successful `capture_*`, walk the tensors the graph used and
set `pins=1` permanently (add `TieredStorage::pin(name)`), releasing/unpinning only when the
graph exec is destroyed. Alternative: keep a `graphs_live_` count in storage and make
`make_room` throw (or `cudaGraphExecUpdate`) instead of evicting while > 0.

### C4 — MED: throws inside stream-capture leave the stream broken and graphs half-built

`capture_step`/`capture_spec` (decode.cu:249-259 / 232-243) call `forward_body()` /
`prefill_chunk_device()`, which call launchers that throw on dim mismatch (gemm.cu:78/182/
293/354, mxfp4_i4.cu:70/151, qwen35.cu:7-9) and `TieredStorage::acquire`. If anything
throws between `BeginCapture` and `EndCapture`:

- `cudaStreamEndCapture` is never called → `x_.stream` stays in capture mode → every later
  CUDA call on it fails with a confusing capture error; the captured `cudaGraph_t` leaks.
- Worse, a cache-miss **inside** capture is guaranteed to throw: `acquire` does
  `cudaMemcpyAsync` from pageable mmap memory (illegal to capture) and
  `cudaStreamSynchronize(stream_)` (`src/storage.cu:9`), which returns
  `cudaErrorStreamCaptureUnsupported` while capturing. Same for `make_room`'s sync. So
  capture currently only works if every tensor is already resident — an undocumented
  precondition with no enforcement.

**Minimal fix:** wrap the capture body in try/catch: on exception call
`cudaStreamEndCapture(x_.stream, &g)` + `cudaGraphDestroy(g)` (or at minimum
`cudaStreamEndCapture` with a discarded graph), then rethrow. Optionally pre-touch all
needed tensors before capture so acquire is guaranteed to be a cache hit, and assert it.

### C5 — MED: destructors free device/pinned memory without synchronizing the stream (UAF on the exception path)

- `DecodeWorkspace::~DecodeWorkspace` (`src/decode.cu:29`): 40 `cudaFree`/`cudaFreeHost`
  calls, **no** `cudaStreamSynchronize` first.
- `Qwen35Weights::~Qwen35Weights` (`src/qwen35.cu:6`): `cudaFree(scratch_int_)` with no
  sync — and it runs **before** member `storage_`'s dtor (`clear()` does sync,
  storage.cu:11), so scratch is freed while its consuming kernel may still be in flight.

Happy path is safe only because `generate.cu`'s last ops (`read_committed`,
`cudaEventSynchronize`) happen to leave the stream idle. But the intended error path —
"KV cache full" / "device budget exhausted" thrown from mid-layer (decode.cu:45,
storage.cu:8) — unwinds with up to 31 layers of kernels enqueued; dtors then free buffers
those kernels are reading. `cudaFree` is not stream-ordered: the host-side free invalidates
immediately. Result: error reporting itself turns into a sticky CUDA context fault that
masks the original message.

**Minimal fix:** first line of both dtors: `cudaStreamSynchronize(stream);` (ignore errors —
best effort). Also note `DecodeWorkspace`'s ctor leaks all earlier allocations if a later
`alloc` throws (no RAII; LOW).

### C6 — MED: `run_nll`/`nll` unchecked `cudaMalloc`s for `targets`/`logp`

`src/generate.cu:62-66` and `src/nll.cu:62-66`: `logitsT` is checked; `targets` and `logp`
returns are ignored. On OOM they are `nullptr`; the subsequent unchecked
`cudaMemcpyAsync`/`row_logp_kernel` launch "succeeds" (async) and the kernel then reads
`targets == nullptr` → device IMA far from the cause. (Incidental protection: with a
negative/huge `chunk` argv in nll.cu:56, `prefill_chunk`'s 1..64 throw fires first.)

**Minimal fix:** check all three the same way; free them (and the `a`/`b` events,
generate.cu:117-119) before return — LOW otherwise.

### C7 — MED: 1 MiB stack buffer + unchecked `wcstombs` in the argv parsers

`src/generate.cu:51-53` (run_nll) and `src/nll.cu:50-52`:

```cpp
char buf[1 << 20];
size_t n = wcstombs(buf, argv[3], sizeof(buf) - 1);
buf[n] = 0;
```

- These are DLLs loaded into python.exe (`src/dllshim.cu`, build bats use `-shared`);
  `wmain` runs on the **host thread's** stack. A 1 MiB local + frames sits at/over the
  Windows default 1 MiB stack reserve. Works only by grace of the host thread's stack size.
  A plain exe build (MSVC default) would overflow at function entry.
- `wcstombs` returns `(size_t)-1` on a conversion failure (non-ASCII wide char in argv) →
  `buf[(size_t)-1] = 0` is a wild write.
- `atoi` on garbage silently yields 0 or UB on overflow — feeds C9 (unvalidated token ids).

**Minimal fix:** `static char buf[1<<20];` (or std::string), and
`if (n == size_t(-1)) return 2;`.

### C8 — MED (unbounded runtime): host `x_.position` mirror drifts in graph mode

Device truth: `spec_commit_kernel` rewinds `pos[0] -= 1` on reject
(`src/prefill.cu:300`); `bumpi`/`addi` advance on other paths. Host mirror:
- `forward_token` decode.cu:130: host `++` + device `bumpi` — in sync.
- `prefill_chunk_device` decode.cu:103-104: device `addi(+T)` + host `+=T` — in sync.
- `spec_step` decode.cu:224: host = device truth read-back — in sync.
- `spec_graph_step` decode.cu:247: host `+= 2` **unconditionally**; on reject device only
  advanced 1 → host is permanently ahead by 1 per reject. The comment says "corrected at
  each host read", but nothing in the graph loop reads `pos_dev` (`committed_count` reads
  only `count_dev`), so the drift is never corrected.

Current main never reads `x_.position` after capture, and drift is in the safe direction
(guards would fire early), but any post-graph `prefill_chunk`/`attention_layer` gets a
wrong position → wrong RoPE/KV slots.

**Minimal fix:** fold a position read-back into `committed_count()` (it already syncs), or
track `position_awaiting_resync` and refresh lazily in `position()`.

### C9 — MED: token-id range never validated except in `forward_token` → device OOB reads on garbage input

`forward_token` validates (`src/decode.cu:130`). Every other entry does not:
- `prefill_chunk` / `prefill_chunk_device` (`src/decode.cu:106,46`): tokens go straight to
  `embed_gather*` whose `const size_t row = __ldg(tokens + t)` is unbounded
  (`src/prefill.cu:11-12, 28-30`).
- `mtp_layer` / `prime_spec` / `spec_step` (`src/decode.cu:137-138, 189-192, 213`): the
  draft token into `embed` via `x_.token_dev`, unvalidated.
- `step(int token)` (`src/decode.cu:260`): writes the raw token into `next_dev`; the
  replayed graph embeds it unvalidated.
- Same class: `mxfp4_get_row_mlx` (`src/mxfp4.cu:277`), `mxfp4_get_row_i4`
  (`src/mxfp4_i4.cu:244`), `bf16_get_row` (`src/fp8.cu:188`).

Tokens originate from `atoi` of argv (C7) — a typo like `-5` or `999999999` becomes a GPU
OOB read (~2 TB offset), i.e. garbage hidden states or a device fault, not a clean error.
Within the engine's own loop the values are argmax outputs (< vocab), so this is an
input-validation hole, not an engine-logic bug.

**Minimal fix (pick one):** host-side loop in `prefill_chunk`/`spec_step`/`step`
(`if (t<0||t>=vocab) throw`), or a 2-instruction device clamp at each gather site
(`row = (unsigned)row >= vocab ? 0 : row`). Given AGENTS.md's spirit, the device clamp is
cheaper than the throw.

### C10 — LOW/MED (27B tripwire): `ab2` pair kernels hard-require `cols==4096` with no launcher check

Staging in `mxfp4_gemv_ab2_q8_kernel` (`src/mxfp4.cu:594`) and
`mxfp4_gemv_ab2_q8_i4_kernel` (`src/mxfp4_i4.cu:163`) splits threads as
`r = tid>>7, g = tid&127` — i.e. exactly 128 groups (cols 4096). Launchers
(`src/mxfp4.cu:668-671`, `src/mxfp4_i4.cu:237-240`) check nothing:
- cols < 4096: staging reads past the 2-row activation region (in-allocation for pf_n, but
  wrong data; a small x buffer would be OOB).
- cols > 4096 (27B hidden 5120 → 160 groups): groups 128..159 are never staged; the GEMV
  loop consumes uninitialized shared memory → silently wrong a/b heads.

Called today only with hidden=4096 (`src/decode.cu:66-70`). First 27B bring-up hits this.

**Minimal fix:** `if (cols != 4096) throw ...` in both launchers (and the `q8g` twins),
or generalize the staging loop to `for (rg = tid; rg < 2*groups; rg += 256)` like
`mxfp4_gemv2_q8_kernel` already does.

### C11 — LOW: leaks and minor error-path hygiene

- `Qwen35Decode` has **no destructor**: `graph_`/`spec_graph_` are never
  `cudaGraphExecDestroy`ed; double-capture also leaks the old exec (decode.cu:50, 232-259).
- `DecodeWorkspace` never `cudaStreamDestroy`s its stream (it owns it when constructed with
  `stream=nullptr`, decode.cu:13).
- `run_nll`/`nll`: `logitsT/targets/logp`, events `a`/`b` leak (process exit cleans up).
- `TieredStorage::release` is `noexcept` but constructs a `std::string` for the lookup
  (`src/storage.cu:10`) — allocation failure would `std::terminate`. Take `const char*`
  key or prehash.
- `quantize_q8_groups` silently returns on bad dims (`src/mxfp4.cu:242`) — inconsistent
  with the throw-everywhere convention reintroduced this session (good: gemm.cu:354,
  mxfp4_i4.cu:70/151 now throw; the old synthesis item 5 is fixed).
- No post-launch `cudaGetLastError` checks anywhere (kernel-launch config failures surface
  much later at an unrelated sync). Acceptable for this codebase; noting for completeness.
- GEMM launchers check `T > 0` but not `T <= 64` (gemm.cu:78/182/293/354). All current
  callers bound T (decode.cu:44 throws 1..64; run_nll's loop caps at 64); nll.cu's
  argv-provided `chunk > 64` is stopped transitively by `prefill_chunk`'s throw before the
  GEMM is reached. Note-only; add `T<=64` to the four checks for cheap insurance.

---

## Hunt-item verdicts (explicit answers)

### 1. Stream-ordering UAF in TieredStorage — **NOT present as hypothesized; three residual hazards**

`make_room` (`src/storage.cu:8`) **does** synchronize before every free:
`check(cudaStreamSynchronize(stream_),"synchronize before eviction");` precedes each
`cudaFree(victim->second.device)` inside the eviction loop, and `clear()` (:11) syncs too.
All kernel/copies/graph launches touching these buffers are enqueued on `x_.stream`
(= `stream_`; `cudaGraphLaunch(spec_graph_, x_.stream)` decode.cu:245/260), and
`cudaStreamSynchronize` on the launch stream does cover graph executions launched into it.
So the "token N+1 acquire frees buffers while token N's kernels run" scenario is correctly
prevented. Release-after-enqueue (pins drop to 0 immediately) is safe because eviction
re-syncs.

Residuals:
- **C3** (graph-baked pointers — sync doesn't help future replays) — HIGH, latent.
- **C4** (a sync or eviction during capture throws and strands the stream) — MED.
- Perf note: eviction costs a full-stream sync per victim. If that ever matters, the
  zero-cost hardening is `cudaFreeAsync(ptr, stream_)` instead of sync+`cudaFree` — free
  becomes stream-ordered, no host stall, same guarantees, available since CUDA 11.4 on
  this driver. Keep the sync in `clear()` (dtor) as-is.

### 2. Host-buffer lifetime with async copies — **safe by documented contract, but subtle; don't "optimize" the sources to pinned**

Sites: `run_nll` `tgt`/`lp` vectors (generate.cu:73-85, nll.cu:75-86);
`prefill_chunk`'s caller-owned `tokens` (decode.cu:107, 114); stack scalars
`&token` (decode.cu:130), `tail[3]` (decode.cu:222), `int c` (decode.cu:226),
`trip[7]`/`draft` (generate.cu:136, 146).

Per the CUDA Programming Guide's async-copy semantics: **H2D from pageable memory returns
only after the source has been fully staged** (it is effectively synchronous w.r.t. the
source buffer, at the cost of a host stall + a stream sync before the copy is even
initiated); **D2H to pageable memory returns only after the copy has completed**. Hence a
dying `std::vector` or a stack `int` cannot be read-after-free today. Two caveats worth a
comment at the call sites:
- The guarantee evaporates if anyone makes these buffers **pinned** (then the copy is truly
  async and the temp-buffer pattern becomes a real UAF) — e.g. someone "optimizing" `tgt`
  to a pinned staging buffer.
- Pageable memcpys cannot be captured into graphs (relevant to C4) and are pseudo-async
  (host stalls) — the `tgt` copy in run_nll pays an avoidable host stall per chunk; a
  pinned, explicitly-managed staging buffer would be both faster and lifetime-explicit.

### 3. Integer-overflow sweep at 27B/lm_head shapes — **all clean; every hot index is size_t**

| Site | Expression | Worst case (27B) | Type | Verdict |
|---|---|---|---|---|
| prefill.cu:12 `embed_gather` | `w + row*512` (row=size_t) | 248319×512 = 127.0M words | size_t | OK (fits int too) |
| prefill.cu:30 `embed_gather_i4` | `w + row*512`, `s + row*64` | 127.0M / 15.9M | size_t | OK |
| 27B-shape equivalent | `row*640` u32/row (hidden 5120) | 158.9M | size_t | OK (int32 would also hold) |
| mxfp4.cu:111 / mxfp4_i4.cu:35 | `weights + row*groups*4` | 248319×160×4 = 158.9M | size_t | OK |
| mxfp4.cu:50,52; gemm.cu:51-52,117-119,328-329 | `(n0+n)*groups*4`, `(n0+n)*groups` | 158.9M / 39.7M | size_t | OK |
| gemm.cu:73-74,177-178,289,350-351 | `y + size_t(warp_m<<4)*rows + n0` | 48×248320 = 11.9M | size_t | OK |
| generate.cu:62 / nll.cu:62 | `size_t(64)*vocab*4` | 254 MB alloc | size_t | OK |
| row_logp (generate.cu:17) | `logits + size_t(row)*vocab` | 63×248320 = 15.6M | size_t | OK |
| fp8.cu:27-28,71-72 | `size_t(row)*cols`, `(row>>7)*kblocks` | 1.27G (bytes) | size_t | OK |
| fp8.cu:190 `bf16_get_row` | `size_t(row)*cols` | 248319×5120 | size_t | OK |
| attention.cu:7, qwen_kernels.cu:15 | `(size_t(t)*4+kvh)*256`, `pos*1024` (pos=size_t) | 4096×1024 | size_t | OK |
| prefill.cu:91-93 | `size_t(pos)*1024` | bounded by C1 (not by types) | size_t | C1 |
| prefill.cu:227-229,256 etc. | `size_t(t)*8192`, `(size_t(t)*32+head)*128` | 63×8192 | size_t | OK |
| decode.cu:14-26 allocs | `24*32*128*128`, `24*8192*3`, `248320*2`, `size_t(8)*ctx*1024` | 12.58M etc. | int consts → size_t args | OK (< 2^31) |
| prefill.cu:308 | `const int n = 24*32*128*128` | 12,582,912 | int | OK |
| decode.cu:197-201 | `host_committed[c+i]`, `committed+c` | c<16384 by C1 only | int | C1 |
| mxfp4.cu:370, mxfp4_i4.cu:153,239 | smem size `16*groups*4 + 2*groups*4 + 2048` | cols 8192 → 33.3 KB | size_t | OK (< 48 KB, no opt-in needed) |
| mxfp4.cu:154 / mxfp4_i4.cu:74 | dyn smem `cols*4+64` | 8192→32.8 KB (mx GemV), 99 KB opt-in set | size_t | OK; opt-in failure would surface as launch failure (unchecked — see C11 note) |

No 32-bit multiplication overflows exist at 27B worst case; the `(warp_m<<16)*rows`-style
sites in the brief are actually `size_t(warp_m<<4)*rows` (gemm.cu:73) — safe. The only
unbounded indices are the C1 (pos/count) ones, which are logic bugs, not type bugs.

### 4. Bounds/contract items

- **prefill seam callback** (decode.cu:90): fires immediately after
  `cudaStreamSynchronize(x_.stream)` — the stream is **idle**, so the callback may legally
  use `x_.stream` (any enqueue just becomes the next queued op before the following
  layer's kernels). Real hazards: (a) the contract is undocumented; (b) a throwing seam
  aborts mid-model with layers 0..l-1 already KV-written (re-invoking the same chunk is
  recoverable only because store_kv is idempotent); (c) a seam passed during a capture
  would hit the illegal-sync path of C4 — currently impossible (spec path passes nullptr,
  decode.cu:42). Fix: one doc comment on `prefill_chunk_seam` (insignia_decode.hpp:25)
  stating "called with the stream idle; may enqueue on x_.stream; must not throw".
- **prefill T bounds**: decode.cu:44 throws for T outside 1..64 — verified correct.
- **token id range**: see C9 (accepted-risk question — currently a hole; device clamp is
  the cheap closure).
- **run_nll argv parsing**: see C7 (wcstombs `-1`, 1 MiB stack buffer, atoi garbage);
  `argc>3 && argv[1]=="nll"` dispatch order in wmain (generate.cu:94-98) means the plain
  generate path still requires argc≥4 — consistent with old behavior; no new hazard.

### 5. Error-path leaks/state

- run_nll malloc checks: logitsT checked, targets/logp NOT → C6.
- Exception during prefill: stream state + dtor UAF → C5; graph half-build → C4.
- `DecodeWorkspace` destructor: frees **everything** it allocated (verified line-by-line
  against the ctor: all 41 device/host pointers, including pos_dev which covers the
  embedded token_dev..mtp_pos_dev aliases, and snap_*/pf_* ) — complete, but unsynchronized (C5)
  and with no graph-exec destruction (that belongs to Qwen35Decode, which has no dtor — C11).
- `Qwen35Weights` member/dtor order: dtor body frees `scratch_int_` **before** `storage_`
  (declared first in insignia_qwen35.hpp:19) runs its synced `clear()` — the scratch free
  is the unsynced one (C5). `storage_.clear()` itself is correct (sync then free).
- `TieredStorage::acquire` failure path frees the partial `d` and rethrows (storage.cu:9) — good.
- `DecodeWorkspace` ctor leaks earlier allocations if a later `alloc` throws (C11, LOW).

### 6. Host/device position drift

Answered in C8. Every writer of `pos_dev`: `set_position`/`set_mtp_position` (host→device
async), `bumpi_kernel` (forward_token, mirrored by host `++`), `addi_kernel`
(prefill_chunk_device, mirrored by host `+=T`), `spec_prologue` (mtp slot = pos-1, per-step,
self-correcting), `spec_commit` (rewind -1 on reject — the only place device moves
**back**), `spec_rollback` (does not touch pos). Host mirror keepers: forward_token ✓,
prefill_chunk_device ✓, spec_step ✓ (device read-back tail[0]), spec_graph_step ✗ (+2
assumed, drift on reject, never resynced — C8). Probe path sets mtp position manually
(generate.cu:133) and memsets mtp KV after — consistent.

### 7. Throws crossing stream capture

Answered in C4: yes, `capture_step`→`forward_body`→`layer`→`tensor`/launchers can throw
mid-capture (dim checks, storage acquire/sync/memcpy, "tensor not found"); the capture is
never ended → broken stream + leaked graph. Not acceptable as-is; cheap try/catch fix.
Note the *reverse* also holds: the graph path can no longer surface dim errors at all
(baked at capture), so throwing launchers are only protective for eager calls.

---

## Things that are deliberately fine (so nobody "fixes" them wrong)

- `make_room`'s per-eviction `cudaStreamSynchronize` (storage.cu:8) is load-bearing for
  eager correctness; don't remove it (upgrade to `cudaFreeAsync` is fine).
- Pageable async copies from temporaries (see §2) are safe by documented staging semantics.
- `argmax_fast`'s 0-initialized `am_scratch` is a valid sentinel under its total-order key
  transform (all finite floats map > 0); the memset is captured into the graph as a node —
  replay-safe.
- `conv_roll_state_kernel`'s `state[c*3+3+j]` negative-j indexing is the intended shift
  for T<3.
- DecodeWorkspace ctx guard (`1..4096`, decode.cu:12) exactly matches `score[4096]` in
  gqa kernels — keep them coupled.
- delta/conv slot math (`di = l-l/4` ∈ 0..23, `ai = l/4` ∈ 0..7) exactly fills the 24/8-slot
  allocations; no off-by-one at l=30/31.
- The f16-scale fix from synthesis item 1 is correctly applied everywhere (mxfp4_i4.cu:11,
  gemm.cu:329, prefill.cu:30 read `__half`); mtp embed + nll i4 branches (synthesis 3/4)
  are in (decode.cu:137, generate.cu:78, nll.cu:79); launcher throws (synthesis 5) are in —
  except the ab2 twins (C10) and quantize_q8_groups (C11).

## Minimal-fix checklist (priority order)

1. generate.cu:113 → fail instead of clamp; clamp `want_total` to ctx-8. + device guards in
   spec_commit / store_kv (C1).
2. gemm.cu:275 → conditional `wait_group 1/0` tail like fp8.cu:164-165 (C2).
3. `TieredStorage::pin()` + pin-at-capture (C3); optional `cudaFreeAsync` hardening (§1).
4. try/catch around both capture bodies with EndCapture+Destroy on throw (C4).
5. `cudaStreamSynchronize` first line of `~DecodeWorkspace` / `~Qwen35Weights` (C5).
6. Check targets/logp mallocs; free them + events (C6/C11).
7. static/heap argv buffer + wcstombs error check (C7).
8. Device-side token clamp in the four gather kernels, or host validation in
   prefill_chunk/spec_step/step (C9).
9. Position resync in `committed_count()` (C8).
10. `cols==4096` throws in ab2 launchers (C10); `~Qwen35Decode` destroying graph execs and
    the owned stream (C11).
