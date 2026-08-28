# Audit: llama.cpp `--n-gpu-layers` partial offloading + mmap mechanics

Clone: `E:\coding\Insignia\llama.cpp\` @ `c060ca974` ("model : support MTP in GLM-4.5-Air", 2026-era master).
Note: this tree has `llama-sched` merged into `llama-context.cpp`, and `ggml_backend_sched` merged into
`ggml/src/ggml-backend.cpp` (there is no separate `ggml-backend-sched.cpp` / `llama-sched.cpp`).
All paths below are relative to the clone root; line numbers are exact for this commit.

Audience: Insignia engine. Purpose: enumerate every sync + copy llama.cpp performs per
offloaded-layer-per-token, to quantify its overhead vs a colibri-style SPMC prefetch design.

---

## 0. Executive summary (the mechanical answer)

llama.cpp does **not** stream weights per token. Placement is static after load:

1. **GPU layers' weights are copied once at load time** from the mmap page cache into VRAM
   (blocking `cudaMemcpyAsync` + `cudaStreamSynchronize` per tensor).
2. **CPU layers' weights are zero-copy pointers into the mmap region** (`ggml_backend_tensor_alloc`
   over the mapping). Per token they are read by *CPU kernels directly from page-cache pages*.
   No H2D copy, no staging, no prefetch.
3. Per token, only **activations and graph inputs cross the CPU/GPU boundary**, at *graph-split*
   boundaries (not per layer — contiguous runs of same-backend layers form one split).
   With the canonical "bottom k layers on CPU, rest on GPU" layout there is **1 CPU→GPU boundary
   per token**; each boundary costs: 1 failed async-copy attempt, 1 full `cudaStreamSynchronize`
   of the GPU stream (no events in single-GPU mode), and 1 **blocking** `cudaMemcpyAsync` on
   `cudaStreamPerThread` followed by another `cudaStreamSynchronize`.
4. Nothing is double-buffered in single-GPU partial-offload mode. Double buffering
   (`n_copies == 2`, events, ping-pong copies) exists but is gated on multi-GPU pipeline
   parallelism with full offload (`llama-context.cpp:428-433`).
5. There is **no inference-time readahead/prefetch of weights anywhere**. The only prefetch
   machinery is load-time: `MAP_POPULATE`/`POSIX_FADV_SEQUENTIAL` (Linux), `PrefetchVirtualMemory`
   (Windows, only for explicit prefetch>0, which the normal load path never passes), and a
   4-buffer pinned staging ring that is used **only when mmap is disabled**.

So llama.cpp's partial-offload overhead per token is dominated by (a) CPU GEMV speed on the
non-offloaded layers themselves and (b) ~2 full-device syncs + 1-2 blocking memcpys per
CPU↔GPU boundary. It never attempts to overlap CPU compute with H2D weight transfer the way a
prefetcher would, because weights never move after load.

---

## 1. Placement: which layer goes where (`--n-gpu-layers`)

### 1.1 Layer split arithmetic

`src/llama-model.cpp:1308-1410` (`llama_model_base::load_tensors`):

- `src/llama-model.cpp:1313-1314` — `n_layer_all` includes the output "layer"; `n_gpu_layers()` returns
  `params.n_gpu_layers >= 0 ? params.n_gpu_layers : n_layer_all + 1` (`src/llama-model.cpp:1776-1778`).
- `src/llama-model.cpp:1385` — `i_gpu_start = max(n_layer_all + 1 - n_gpu_layers, 0)`.
- `src/llama-model.cpp:1387-1397` — `get_layer_buft_list(il)`: if `il < i_gpu_start` (or beyond the
  count of GPU layers) the layer is assigned to `cpu_dev` + `cpu_buft_list`; otherwise to
  `devices[layer_gpu]` via `upper_bound` over normalized free-memory splits. **Offload is counted
  from the tail**: the last `n_gpu_layers` repeating layers (plus the output head if it fits) go to
  the GPU; the bottom layers stay on CPU.
- `src/llama-model.cpp:1399-1410` — the *input* layer is always pinned to CPU
  ("there is very little benefit to offloading the input layer"); `dev_output` follows the split
  arithmetic (il == n_layer_all).
- Multi-GPU only: row-split (`LLAMA_SPLIT_MODE_ROW`) introduces a split buffer type
  (`src/llama-model.cpp:1006-1031`). Not relevant for the 4070 SUPER single-GPU case.

### 1.2 Buffer types for weights

- CPU list: `make_cpu_buft_list` (`src/llama-model.cpp:944-1003`). Order: ACCEL bufts, then the
  **first GPU device's host buffer type** (`CUDA_Host`, pinned, `src/llama-model.cpp:965-973`),
  then extra bufts, then plain CPU. Comment at `960-963`: host buffers exist so that "processing of
  large batches offloaded to a GPU" copies faster.
- **mmap overrides the pinned choice**: `src/llama-model-loader.cpp:1212-1220` — "avoid using a host
  buffer when using mmap": if the selected buft is a host buffer type, swap it for the plain CPU
  buffer type, so CPU weights can sit zero-copy over the mapping.
- GPU list: `make_gpu_buft_list` (`src/llama-model.cpp:1006-1052`) — device default buft + extras.

### 1.3 Load: mmap → VRAM (GPU tensors) and mmap → zero-copy (CPU tensors)

`src/llama-model-loader.cpp:load_all_data` (fn starts `:1426`), per tensor:

- **mmap path** (`src/llama-model-loader.cpp:1556-1583`):
  - `data = mapping->addr() + weight->offs` (`:1562`).
  - CPU-resident tensor: `ggml_backend_tensor_alloc(buf_mmap, cur, data)` (`:1572`) — the tensor's
    `data` pointer is set directly into the file mapping. **Zero-copy; the page cache IS the weight
    storage.** `mlock` optionally grows over it (`:1573-1576`).
  - GPU-resident tensor: `ggml_backend_tensor_set(cur, data, 0, n_size)` (`:1582`) → CUDA buffer
    `set_tensor` = `cudaMemcpyAsync(H2D, cudaStreamPerThread)` + `cudaStreamSynchronize`
    (`ggml/src/ggml-cuda/ggml-cuda.cu:786-792`). **Blocking per-tensor upload; page faults are
    serviced by the DMA read (pageable source, driver-staged).** There is no batching/ring on this
    path.
- **non-mmap path** (`src/llama-model-loader.cpp:1584-1639`): host tensors are `read_raw` directly
  (`:1587-1589`); GPU tensors use a **4-buffer pinned staging ring + events**:
  - `src/llama-model-loader.cpp:1443-1459` — `n_buffers = 4`, pinned (`host_buft` from the device,
    i.e. `CUDA_Host`), buffer size 1 MiB (64 MiB when the file requires alignment)
    (`:1454`), one event per buffer.
  - Loop `:1611-1639`: `ggml_backend_event_synchronize(events[buffer_idx])` before reusing a slot
    (`:1618`), aligned `read_raw_unsafe` into pinned memory (`:1621`), then
    `ggml_backend_tensor_set_async` (`:1639`) → `ggml_backend_cuda_set_tensor_async`
    (`ggml-cuda.cu:2435-2442`) = `cudaMemcpyAsync` on the backend stream, no sync. This is the only
    genuine producer/consumer pipeline in the whole pipeline — **and it is load-time-only, disabled
    under mmap** (`src/llama-model-loader.cpp:1460-1463`).

### 1.4 mmap advice flags (question 5)

`src/llama-mmap.cpp`, constructed from `init_mappings(prefetch, ...)`:

- Normal model load passes `prefetch = true` → `llama_mmap(file, -1, is_numa)`
  (`src/llama-model.cpp:1599`, `src/llama-model-loader.cpp:1348-1364`); llama-quant passes `false`
  → 0 (`src/llama-quant.cpp:902`).
- **Linux** (`src/llama-mmap.cpp:445-475`):
  - `posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)` always (`:451`).
  - `MAP_POPULATE` when `prefetch != 0` — i.e. **yes for normal loads (prefetch == -1)** (`:455`).
  - `POSIX_MADV_WILLNEED` only when `prefetch > 0` — **never in the normal path** (`-1`), so no
    explicit willneed range (`:462-467`).
  - `POSIX_MADV_RANDOM` over the whole mapping **only under NUMA** (`numa` disables populate and
    asks for random access, `:449`, `:468-473`).
- **Windows** (`src/llama-mmap.cpp:536-578`): `CreateFileMappingA` + `MapViewOfFile`
  (`:543-550`); `PrefetchVirtualMemory` only when `prefetch > 0` (`:558-577`) — again **never in
  the normal path**, so on the audit box (win32) the mapping is faulted lazily page by page during
  the blocking per-tensor H2D uploads.
- `unmap_fragment` (`:490-524` POSIX only) — used by `unmap_weight`
  (`src/llama-model-loader.cpp:1398-1401`) e.g. after quantization-time surgery.

**No `MADV_RANDOM`/`MADV_DONTNEED`/`fadvise(DONTNEED)` traffic management at inference time.**
Eviction of clean file-backed pages is left entirely to the OS.

---

## 2. Runtime: what happens per token (question 1)

### 2.1 Graph construction & scheduling overview

- `llama_context::graph_compute` → `ggml_backend_sched_graph_compute_async(sched, gf)`
  (`src/llama-context.cpp:2494`) → `ggml_backend_sched_alloc_graph` +
  `ggml_backend_sched_compute_splits` (`ggml/src/ggml-backend.cpp:1961-1974`).
- `ggml_backend_sched_split_graph` (`ggml/src/ggml-backend.cpp:1055-1540`):
  - Pass 1 assigns backends by weight location: ops follow their weight tensors
    (`ggml_backend_sched_backend_id_from_cur`, `:910-974`; cause `"1.wgt%d"` at `:967`).
  - Passes 2-4 expand/upgrade assignments; GPU runs stretch over adjacent CPU ops where possible
    (down/up expansion skips CPU, `:1119-1191`), so **splits are maximal same-backend runs**.
  - Pass 5 (`:1287-1425`) cuts the node list into splits; for every src that lives on an
    incompatible backend it creates a *copy tensor* on the split's backend
    (`:1399-1420`) and rewires `node->src[j]` to the copy (`:1419`). The comment at `:1321-1322`
    is the smoking gun for weight copies: "by starting a new split, the memory of the previously
    offloaded weights can be reused" — i.e. when `op_offload` pulls a CPU-weight op onto the GPU,
    **the whole weight tensor becomes a split input and is re-copied H2D every time the graph runs**
    (see §4).
- The activation buffers for the CPU side are allocated in the **pinned** host buffer type:
  `src/llama-context.cpp:407-417` swaps the CPU backend's default buft for
  `ggml_backend_dev_host_buffer_type(devices[0])` ("faster transfer of the intermediate state").
  So **boundary copies are pinned-source**, while CPU weights (mmap) are pageable-source.

### 2.2 The exact per-token sequence, canonical partial offload (k bottom layers CPU, rest GPU)

`ggml_backend_sched_compute_splits` (`ggml/src/ggml-backend.cpp:1594-1790`), split loop:

**Split 0 = CPU part** (embedding `GET_ROWS` + layers `0..k-1` + their KV updates):
1. Graph inputs (tokens/positions/seq_ids/mask) were written with plain `ggml_backend_tensor_set`
   → host `memcpy` into the pinned CPU input buffers (`src/llama-graph.cpp:73-113`).
   `GET_ROWS` on the token-embedding table stays on CPU: weights on CPU pin the op to CPU
   (`ggml-backend.cpp:956-969`), and op-offload never applies to `GET_ROWS`
   (`ggml-cuda.cu:5328-5329` returns batch 0).
2. `ggml_backend_graph_compute_async(cpu_backend, split)` (`ggml-backend.cpp:1743`) →
   `ggml_backend_cpu_graph_compute` (`ggml/src/ggml-cpu/ggml-cpu.cpp:170-191`) →
   `ggml_graph_compute` — **fully synchronous on the caller thread + threadpool**. Weights are
   touched as raw mmap pointers (pageable); first-touch of evicted pages = major fault inside the
   CPU kernel loop. There is no pinned staging for CPU-layer weights and no readahead.

**Split boundary CPU→GPU** (the "when a layer is NOT in VRAM" answer for activations):
3. For each split input (the hidden-state activation `n_embd` floats/token, plus whichever graph
   inputs the GPU part consumes — e.g. kq_mask, positions):
   - Fast path attempt: `split_backend->iface.cpy_tensor_async(...)`
     (`ggml-backend.cpp:1729`) = `ggml_backend_cuda_cpy_tensor_async`
     (`ggml-cuda.cu:2475-2532`) — **returns false** because it requires both buffers to be CUDA
     (`:2483-2485`). Only D2D/peer copies are async.
   - Fallback (`ggml-backend.cpp:1726-1737`):
     a. `ggml_backend_synchronize(input_backend)` — CPU backend's `synchronize` is `NULL`
        (`ggml-cpu.cpp:201`), a no-op (CPU split already blocked).
     b. `ggml_backend_synchronize(split_backend)` (or event sync — see §3.3 why there are no
        events here) — **full `cudaStreamSynchronize` of the GPU compute stream**
        (`ggml-cuda.cu:2534-2538`). *Sync point #1.*
     c. `ggml_backend_tensor_copy(input, input_cpy)` (`ggml-backend.cpp:477-498`): src buffer
        `is_host` → `ggml_backend_tensor_set(dst, src->data, ...)` →
        `ggml_backend_cuda_buffer_set_tensor` (`ggml-cuda.cu:786-792`):
        `cudaMemcpyAsync(H2D on cudaStreamPerThread)` + `cudaStreamSynchronize(cudaStreamPerThread)`.
        **Blocking copy, *Sync point #2*** (copy completion). Note the copy runs on
        `cudaStreamPerThread`, *not* the backend's compute stream — the immediate sync is what
        makes ordering correct; nothing is pipelined.
   - Exception — graph *inputs* (user-provided, `GGML_TENSOR_FLAG_INPUT`): copied "immediately to
     prevent the user overwriting" (`ggml-backend.cpp:1625-1632`): event/backend sync of the split
     backend then the same blocking copy. So **each GPU-consumed input tensor adds its own
     blocking H2D** per token.
   - MoE weights host→GPU get a specialized *partial* copy — see §4.2.

**Split 1 = GPU part** (layers `k..N-1`, final norm, output head if offloaded, argmax/samplers):
4. `ggml_backend_graph_compute_async(cuda_backend, split)` (`ggml-backend.cpp:1743`) →
   `ggml_backend_cuda_graph_compute` (`ggml-cuda.cu:4247-4304`): kernels are enqueued **asynchronously**
   on the backend's dedicated `cudaStreamNonBlocking` stream (`ggml/src/ggml-cuda/common.cuh:1488-1496`);
   with CUDA graphs enabled the split is captured/replayed (map keyed by first node pointer so
   "CPU/GPU split (e.g. with --n-cpu-moe)" works, `common.cuh:1427-1430`). The sched-level
   boundary memcpys happen *outside* any capture, so they remain sync points even under CUDA graphs.
5. Event record for the split — skipped when no events (single-GPU) (`ggml-backend.cpp:1781-1784`).

**Tail — logits readback:**
6. Logits/sampler tensors are read with `ggml_backend_tensor_get_async` on the backend stream
   (`src/llama-context.cpp:1862-1875`, `:1958-1966`, `:2216-2218` → `ggml-cuda.cu:2444-2451`,
   `cudaMemcpyAsync` D2H, **no sync issued here**; destination is a pageable malloc'd buffer).
7. The sync lands either when the app calls `llama_synchronize()` (`src/llama-context.cpp:705-710` →
   `ggml_backend_sched_synchronize` → per-backend `cudaStreamSynchronize`,
   `ggml-backend.cpp:1976-1980`) or implicitly at the *next* token's immediate-input copy
   (step 3's backend sync). **Effectively ≥1 full GPU sync per token. *Sync point #3*.**

**GPU→CPU boundaries** (if the layout puts CPU layers *after* GPU layers, e.g. output head on CPU
with `-ngl` covering only some layers, or `--no-kv-offload` patterns): step 3's fallback runs in the
other direction — `ggml_backend_tensor_get` D2H via `cudaMemcpyAsync+Sync`
(`ggml-cuda.cu:794-800`) after `ggml_backend_synchronize(input_backend)` on the *CUDA* backend —
**two full GPU syncs per boundary** (one for src readiness, one for copy completion).

### 2.3 Is anything double-buffered? (question 1, sub-point)

- **No, in the configuration that matters here.** `sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1`
  (`ggml-backend.cpp:1816`); events are only created `if (sched->n_copies > 1)` (`:1849-1853`).
  `parallel` = `cparams.pipeline_parallel`, which requires **`model.n_devices() > 1` AND full
  offload AND split-mode layer AND `offload_kqv` AND no tensor overrides**
  (`src/llama-context.cpp:428-433`). Single-GPU partial offload → `n_copies == 1`, zero events,
  every "event_wait/event_synchronize" branch degenerates to full `ggml_backend_synchronize`.
- The plumbing for ping-pong exists: per-backend per-copy events (`ggml-backend.cpp:1849-1853`),
  `cur_copy` rotation (`:1941-1942`), input copy tensors per copy index (`:1375-1397`),
  `ggml_backend_cuda_event_record/wait` = `cudaEventRecord`/`cudaStreamWaitEvent`
  (`ggml-cuda.cu:4306-4329`), and CUDA `cpy_tensor_async` records a `copy_event` on the src stream
  and makes the dst stream wait (`ggml-cuda.cu:2517-2526`) — D2D only.

---

## 3. Pinned buffers & async copy patterns in ggml-cuda (question 2)

### 3.1 Inventory

| Mechanism | Where | Notes |
|---|---|---|
| `CUDA_Host` buffer type | `ggml-cuda.cu:1263-1325` | `cudaMallocHost` (`:1277-1293`), freed with `cudaFreeHost` (`:1273-1275`); **not a pool** — one `cudaMallocHost` per `alloc_buffer` call, falls back to plain CPU buffer on failure or `GGML_CUDA_NO_PINNED` env (`:1278-1280`). Alignment/max_size/is_host delegate to the CPU buffer type (`:1315-1318`). |
| Host buft usage at runtime | `src/llama-context.cpp:407-417` | CPU backend's activation/input buffers live in `CUDA_Host` (pinned) so boundary H2D copies are pinned-source. |
| Host buft avoided for weights under mmap | `src/llama-model-loader.cpp:1212-1220` | CPU weights = zero-copy mmap, pageable. |
| Load-time staging ring (non-mmap only) | `src/llama-model-loader.cpp:1443-1539` | 4 pinned buffers (1/64 MiB) + 4 events; event-sync before slot reuse; async `set_tensor` on backend stream. The closest thing to a colibri-style SPMC ring in the codebase. |
| `cudaHostRegister` | `ggml-cuda.cu:4641-4662` | **Opt-in only** via `GGML_CUDA_REGISTER_HOST` env (`Portable|ReadOnly`); exported as proc address `ggml_backend_register_host_buffer` (`:5486-5487`); unregister at `:4664-4674`. |
| Per-backend events | `ggml-cuda.cu:5347-5376` | `cudaEventCreateWithFlags(cudaEventDisableTiming)`; record/wait/sync used only by multi-copy sched. |
| `copy_event` on context | `ggml-cuda.cu:2517-2526`, `common.cuh:1418` | cross-stream ordering for D2D `cpy_tensor_async`. |

### 3.2 Streams

- Backend compute: lazily-created `cudaStreamNonBlocking` per device per stream index
  (`common.cuh:1420`, `:1488-1496`); CUDA-graph capture on this stream (`ggml-cuda.cu:4298`).
- Buffer-level set/get (`set_tensor`, `get_tensor`, 2d variants, `cpy_tensor`, `clear`, `memset`):
  all use **`cudaStreamPerThread` + immediate `cudaStreamSynchronize`**
  (`ggml-cuda.cu:778-852`). Every buffer-interface copy is therefore serializing, regardless of
  caller intent.
- Backend-level async variants (`set_tensor_async`, `get_tensor_async`, 2d): use `cuda_ctx->stream()`
  and **no sync** (`ggml-cuda.cu:2435-2473`) — but they are only reachable through
  `ggml_backend_tensor_set/get_async` with an explicit backend handle (MoE expert copies at
  `ggml-backend.cpp:1695-1700`, logits readback, load ring). The sched's generic boundary path does
  **not** use them (it calls the buffer-level sync path via `ggml_backend_tensor_copy`).

### 3.3 Why events don't help single-GPU partial offload

Events are allocated per backend only when `n_copies > 1` (`ggml-backend.cpp:1849-1853`), which
requires multi-GPU pipeline parallelism (`llama-context.cpp:428-433`). With one GPU every
`event_wait`/`event_synchronize` branch in `compute_splits` (`:1611-1617`, `:1627-1639`,
`:1731-1735`) degrades to `ggml_backend_synchronize` = full `cudaStreamSynchronize`
(`ggml-cuda.cu:2534-2538`).

---

## 4. `op_offload` and MoE: the only per-run weight movement (questions 1/2 spill-over)

### 4.1 `--op-offload` (prompt processing only, never decode)

- Flag origin: `cparams.op_offload` (`src/llama-context.cpp:272`, default `true`,
  `:3542`), passed into `ggml_backend_sched_new` (`:604`, `:639`).
- Mechanics: `ggml_backend_sched_backend_id_from_cur` (`ggml-backend.cpp:959-966`): when an op's
  weights are on the CPU (host buffer) and `sched->op_offload`, a higher-priority (GPU) backend may
  claim the op **if** `ggml_backend_offload_op` — CUDA: `get_op_batch_size(op) >= 32`
  (`ggml-cuda.cu:5341-5345`, threshold from `GGML_OP_OFFLOAD_MIN_BATCH` env, default 32, `:5515`).
- Batch sizes (`ggml-cuda.cu:5326-5339`): `MUL_MAT` → `ne[1]` (n_tokens), `MUL_MAT_ID`/`ROPE` →
  `ne[2]`, `GET_ROWS` → **0 (never offloaded)**.
- Consequence: with batch ≥ 32 (prompt ubatches) the op runs on GPU **with weights still on
  CPU/mmap** → pass 5 inserts the *weights tensor* as a split input → per graph run the fallback at
  `ggml-backend.cpp:1726-1737` does a **blocking full-tensor H2D copy** (pageable mmap source).
  The new-split heuristic at `:1321-1329` exists so the GPU-side staging allocation for these
  weights is reused across splits rather than accumulating. For decode (n_tokens = 1 < 32) nothing
  is offloaded this way — CPU layers execute entirely on CPU.

### 4.2 MoE expert partial copies (`--cpu-moe` / `--n-cpu-moe`)

`ggml-backend.cpp:1641-1725`: when a split input is a *weights host buffer* consumed by
`GGML_OP_MUL_MAT_ID` (i.e. `--cpu-moe`, tensor-buft-overridden experts — flags registered at
`common/arg.cpp:2740-2747`):

1. `ggml_backend_synchronize(input_backend)` (`:1653`).
2. The routing `ids` tensor is fetched D2H and **synchronized** (`ggml_backend_tensor_get_async` +
   `ggml_backend_synchronize`, `:1669-1672`) — a host round-trip per run.
3. Used experts are computed via bitset (`:1674-1686`) and copied as contiguous ranges with
   `ggml_backend_tensor_set_async` (async, backend stream, `:1689-1700`), with ≤512 B of padding
   per range "to ensure there are no NaNs ... necessary for MMQ" (`:1692-1699`).

This is the **only** weight-paging llama.cpp does per inference run, and it is triggered by
batch-size-qualified op offload (prompt processing); decode-time expert GEMMs just run on the CPU
against mmap pages. There is no residency cache, no LRU, no reuse of the previous run's copy
decision (the copy repeats every run even if the same experts stay hot).

---

## 5. KV cache placement interaction (question 4)

- Construction: per-architecture `llama_kv_cache*` factories receive `cparams.offload_kqv`
  (`src/llama-model.cpp:2158`, `:2185`, `:2210`, ... `:2304`).
- Per-layer device: `src/llama-kv-cache.cpp:210-219` — `buft = offload ?
  ggml_backend_dev_buffer_type(model.dev_layer(il)) : cpu_buffer_type`. **KV cache for layer `il`
  lives on the same device as layer `il`'s weights.** With partial offload, CPU layers get CPU
  (pageable, malloc'd — not mmap, not pinned) K/V tensors; GPU layers get VRAM K/V.
- `--no-kv-offload` (`offload_kqv=false`): KV stays on CPU **for all layers**, and the attention
  output chain is forced onto the CPU backend (`src/llama-graph.cpp:2665-2668`:
  "all nodes between the KV store and the attention output are run on the CPU"), adding a
  GPU→CPU→(kqv)→GPU zig-zag per offloaded attention layer per token — i.e. two extra boundaries
  and 2 extra full syncs each.
- Because KV tensors are pre-allocated with a specific buft, they pin ops to their device:
  `ggml_backend_sched_backend_id_from_cur` finds them via `backend_from_buffer`
  (`ggml-backend.cpp:912`, `:956-969`), and `supports_buft` for discrete CUDA only accepts CUDA
  buffers (`ggml-cuda.cu:5320-5324`), so a CPU KV cache feeding a GPU softmax would force split
  copies of K/V views per token — avoided only by the explicit `set_tensor_backend` pinning above.
- Pipeline-parallel gating uses `offload_kqv` as a precondition (`src/llama-context.cpp:432`).

---

## 6. Prefetch / readahead inventory (question 3)

| What | Where | Active when |
|---|---|---|
| `POSIX_FADV_SEQUENTIAL` | `llama-mmap.cpp:451-454` | always (Linux, mmap) |
| `MAP_POPULATE` | `llama-mmap.cpp:455` | Linux, normal load (prefetch=-1 → nonzero) |
| `POSIX_MADV_WILLNEED` | `llama-mmap.cpp:462-467` | never in normal load (needs prefetch>0) |
| `POSIX_MADV_RANDOM` | `llama-mmap.cpp:468-473` | NUMA only |
| `PrefetchVirtualMemory` | `llama-mmap.cpp:558-577` | Windows, needs prefetch>0 → never in normal load |
| Load-time pinned ring + events | `llama-model-loader.cpp:1443-1539` | **non-mmap loads only** |
| Inference-time weight prefetch | — | **does not exist** |
| `cudaMemAdvise`/UVM, `cudaMemcpyPeerAsync` prefetch | — | not used for weights (peer copy only in D2D async, `ggml-cuda.cu:2513`) |
| `GGML_CUDA_REGISTER_HOST` (`cudaHostRegister`) | `ggml-cuda.cu:4641-4662` | opt-in env, registration only (no prefetch semantics) |

---

## 7. Sync/copy enumeration per offloaded boundary per token

Canonical single-GPU partial offload, 1 decode token, k CPU layers at the bottom, output head on
GPU, `offload_kqv=1`, no MoE, CUDA graphs on or off (boundary is outside capture):

| # | Event | Mechanism | file:line |
|---|---|---|---|
| 1 | CPU split executes | blocking `ggml_graph_compute` on threadpool; weights read from mmap pages (pageable, may major-fault) | `ggml-cpu.cpp:170-191` |
| 2 | GPU stream quiesce before boundary copy | `cudaStreamSynchronize(compute_stream)` (no events, `n_copies=1`) | `ggml-backend.cpp:1729-1737` → `ggml-cuda.cu:2534-2538` |
| 3 | Activation H2D | `cudaMemcpyAsync(H2D, cudaStreamPerThread)` + `cudaStreamSynchronize` — blocking, single-buffered, pinned source (`CUDA_Host` activations) | `ggml-backend.cpp:1736` → `:485` → `ggml-cuda.cu:786-792` |
| 3a | × per GPU-consumed graph input (mask, positions, …) | immediate-copy variant of #2+#3 | `ggml-backend.cpp:1625-1632` |
| 4 | GPU split enqueued | async on dedicated non-blocking stream / CUDA-graph replay | `ggml-backend.cpp:1743` → `ggml-cuda.cu:4247-4304` |
| 5 | Logits D2H | async `cudaMemcpyAsync` on backend stream, pageable dest, **unsynced here** | `llama-context.cpp:1873` → `ggml-cuda.cu:2444-2451` |
| 6 | End-of-token sync | full `cudaStreamSynchronize` via `llama_synchronize` or next token's step 2 | `llama-context.cpp:705-710`; `ggml-backend.cpp:1976-1980` |

Per-boundary cost (CPU→GPU direction): **2 full-stream syncs + 1 blocking memcpy (≈n_embd×4 B) +
1 failed cpy_tensor_async attempt + split-input bookkeeping**. GPU→CPU boundaries cost 2 syncs +
1 blocking D2H. `--no-kv-offload` adds 2 boundaries per offloaded attention layer. Prompt
processing additionally re-uploads op-offloaded CPU weights per ubatch (§4.1) and, with `--cpu-moe`,
the used-expert ranges per ubatch plus an ids D2H round trip (§4.2).

---

## 8. What this means vs a colibri-style SPMC prefetch design

llama.cpp's structural overheads that Insignia would not have to replicate:

1. **Boundary serialization**: every cross-backend tensor triggers a full
   `cudaStreamSynchronize` before the copy and a blocking memcpy after; copies run on
   `cudaStreamPerThread` disconnected from the compute stream, so no overlap is even attempted
   (`ggml-cuda.cu:786-800` vs `common.cuh:1488-1496`).
2. **No double buffering where it counts**: ping-pong copies + events exist but are gated behind
   multi-GPU full-offload pipeline parallelism (`ggml-backend.cpp:1816`, `llama-context.cpp:428-433`).
3. **No inference-time weight movement**: paging = static placement; the only dynamic weight copy
   is the MoE used-expert upload (batch≥32) with no residency tracking (`ggml-backend.cpp:1641-1725`).
   A colibri-style SPMC producer (NVMe→pinned ring→VRAM ahead of use) has no analog.
4. **Load-time uploads under mmap are per-tensor blocking** from pageable file pages
   (`llama-model-loader.cpp:1582`); the good ring pipeline is unreachable with mmap
   (`:1460-1463`).
5. **CPU layers read weights straight from the page cache with no madvice policy** (no RANDOM to
   keep readahead from wasting NVMe bandwidth in steady state, no WILLNEED hints) — Linux relies on
   the one-time `MAP_POPULATE` + FADV_SEQUENTIAL; Windows gets no prefetch at all in the default path.
6. **KV cache placement is welded to weight placement** (`llama-kv-cache.cpp:214-219`), so any
   future weight migration scheme would need to decouple them; `--no-kv-offload` demonstrates the
   zig-zag penalty (2 extra syncs per attention layer per token,
   `llama-graph.cpp:2665-2668`).

Quantifiable baselines to cite when comparing: per token, canonical partial offload costs
(2 + #GPU-consumed-inputs) blocking H2D copies of activation-sized tensors, 2-3 full GPU stream
syncs, one of which sits on the critical path between the CPU threadpool finishing and the first
GPU kernel launching. For Qwen3.5-9B-class n_embd (≈16-32 KiB/token), the *copy* is negligible
(≈2-4 µs at 8 GiB/s effective pinned BW) — the syncs and the serialized handoff (tens of µs each,
plus CUDA-graph-unfriendly boundaries) are the real tax.
