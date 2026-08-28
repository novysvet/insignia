# w4 — Colibri heterogeneous-execution extract (primary-source, clone present)

Date 2026-08-25. The colibri clone IS present at `E:\coding\Insignia\colibri` (v1.7.0-era,
git 33e67a9). Everything below was re-verified at file:line in the clone itself, not from
the w2/w3 audits (those were cross-checked and agree; both cite the same lines). Read-only
audit; the only file written is this report.

Clone inventory relevant to us (all under `colibri/c/`):
`colibri.c` (10,477 ln, GLM engine + pipe/pilot/LFRU core), `qwen36.c` (2,549 ln, Qwen3.5
MoE+DeltaNet engine — closest cousin to our 27B), `qwen36_tier.c/h` (446/81 ln, CUDA VRAM
expert tier), `tier.h` (60 ln, LFRU scoring), `olmoe.c` (LFRU + pilot variant),
`compat.h` (540 ln, Windows shims), `st.h` (961 ln, shard/twin-fd/mirror layer),
`backend_cuda.cu` (2,757 ln), `uring.h`, `expert_store.h`. Docs: `docs/tuning.md`,
`docs/qwen36-cuda-tier.md`, `docs/CACHE_ROUTE.md`.

---

## 1. The SPMC load pool (PIPE) — exact mechanics

### 1.1 Data structure — `colibri.c:3343-3354`

```c
typedef struct {
    _Atomic uint64_t cur;                         /* (gen<<8)|index; gen main-only, index 0..njobs (≤64) */
    _Atomic int njobs;                            /* current batch job count */
    _Atomic int eids[64];                         /* current batch expert ids */
    _Atomic int layer;                            /* current batch layer */
    _Atomic int ready[64];                        /* per-slot load-done flag */
    pthread_mutex_t mx; pthread_cond_t cv;        /* ONLY for parking/waking idle workers */
    pthread_cond_t cv_done;                       /* COLI_PIPE_BLOCK: signals ready[] transitions */
    Model *m;
    pthread_t th[16]; int nw; int started;
} PipePool;
static PipePool g_pp;
```

The whole protocol is documented in the block comment `colibri.c:3303-3320`: the main
thread is the **sole writer of `gen`** (monotonic bump → no ABA); workers grab jobs by
CAS-advancing the low 8 bits. The invariant: *a worker reads `eids[i]/layer` only AFTER
its winning CAS, and the CAS's comparand carries the generation* — a straggler preempted
anywhere can never grab a wrong-generation job. `dispatch` publishes batch state with
relaxed stores then **RELEASE-stores `cur`**; workers ACQUIRE-load `cur`. The mutex/condvar
park idle workers only, never for correctness. Old `active` counter and end-of-block drain
barrier were deleted as redundant with the per-slot `pipe_wait(ready[q])`.

### 1.2 Worker claim — `colibri.c:3356-3385`

```c
for(;;){
    uint64_t c=atomic_load_explicit(&p->cur,memory_order_acquire);
    seen=c>>8;
    uint32_t i=(uint32_t)(c & 0xFF);
    if(i >= (uint32_t)atomic_load_explicit(&p->njobs,memory_order_relaxed))
        break;                                /* batch drained → re-park */
    if(atomic_compare_exchange_weak_explicit(&p->cur,&c,c+1,
            memory_order_acq_rel,memory_order_relaxed)){
        int L  =atomic_load_explicit(&p->layer,memory_order_relaxed);
        int eid=atomic_load_explicit(&p->eids[i],memory_order_relaxed); /* AFTER winning CAS */
        expert_load(p->m,L,eid,&p->m->ws[i],1,1);  /* demand=1: moe()'s own miss path */
        atomic_store_explicit(&p->ready[i],1,memory_order_release);
        ...
    }
    /* CAS failed → another worker advanced index (or gen advanced): re-loop */
}
```

Parking is gen-watching: `while((cur>>8)==seen) pthread_cond_wait(&p->cv,&p->mx)` (3360).
Default pool `g_pipe_nw=8` workers (3325), capped 16. **PIPE defaults ON on Windows**
(3321-3324: `getenv("PIPE")?:1 on _WIN32, :0 elsewhere`).

### 1.3 Dispatch publish order — `colibri.c:3404-3424`

`pipe_dispatch(m,layer,eids,njobs)`: store `njobs`, `layer`, `eids[]` relaxed; **reset
`ready[]` to 0 BEFORE publish**; `gen = (cur>>8)+1`; `cur = (gen<<8)` with **release**
("PUBLISH"); then `pthread_cond_broadcast(&cv)` under `mx`. Backpressure: **none, by
design** — a batch is ≤64 jobs, the main thread is both sole publisher and sole
consumer-waiter, so dispatch rate is self-limiting (see w3/colibri-sched-deep §1.4).

### 1.4 Slot lifecycle (ws[] scratch → ecache LRU promotion)

- Each job i loads into scratch slab `m->ws[i]`; `expert_load` fills `ws[i].slab/q4/s`
  and sets `ws[i].eid`.
- Consumers `pipe_wait(q)` per slot (`colibri.c:3435-3454`): spin `sched_yield` by
  default; `COLI_PIPE_BLOCK=1` switches to condvar `cv_done` (~5 µs wake vs 0.5-3 ms
  reads, issue #159 — a yield storm fights the OpenMP team).
- **End-of-block LRU promotion** (`colibri.c:5596-5609`, FASE D): after the matmul loop
  has waited every dispatched slot, the `nmiss` ws[] slabs are swapped into the per-layer
  `ecache[layer]` — grow if `ecn<ecap`, else `eslot_lru_victim()` (free-with-slab first,
  then LRU); swapped-out slot becomes the new ws scratch (`ESlot tmp=*dst; *dst=m->ws[q];
  m->ws[q]=tmp`), fresh `used=++eclock` stamp. Correctness rests on "every dispatched slot
  waited before the swap + the gen-tagged cursor keeps a still-spinning worker off a
  wrong-generation slot" (comment 5594-5598).

### 1.5 Early-issue policy — how far ahead the loads run

Three horizons, all dispatch-side; the pool itself only ever holds the *current* block:

1. **Current-block misses, dispatched at routing time** (`colibri.c:5039-5051`): right
   after FASE A routing computes the union of experts and before any matmul,
   `pipe_dispatch(m,layer,eids,nmiss)` fires. Workers' preads overlap the resident-expert
   GPU submit and the whole matmul loop. This is the "early" — it is not a fixed depth, it
   is "as soon as the ids are known".
2. **Same-layer next-block WILLNEED** (`colibri.c:5052-5065`): while computing block of
   64 unique experts, `expert_prefetch` (fadvise WILLNEED, `colibri.c:3510-3518`) is issued
   for the *next* 64-expert chunk of the same layer (weights skipped when `g_direct` —
   O_DIRECT bypasses the page cache, only the buffered `.qs` scales are hinted).
3. **Cross-layer PILOT lookahead, 1 layer** (`pilot_prefetch`, `colibri.c:6055-6120`):
   after layer L's attention, run layer L+1's router on the post-attention state
   (**recall 71.6% of true top-8 vs 41.3% for "same experts as last token"** —
   `docs/tuning.md:134-139`), enqueue the top-`PILOT_K` (default 8) predictions into a
   second SPMC ring `pilot_q[4096]` (1 producer/1..N consumers, `pilot_ring_claim` CAS at
   `colibri.c:5925-5942`; payload read *before* the claim, torn read discarded if CAS
   loses). `PILOT_REAL=1` makes the pilot do real cross-layer loads into `ecache[L+1]`
   with a **visible reservation** `dst->eid=-(eid+2)` and `used=(uint64_t)-1` sentinel
   (`colibri.c:5777-5844`); the safety invariant is two-part (comment 1377-1388): pilot
   writes only `ecache[layer > g_cur_moe_layer]`, scan reads only the future layer under
   the same `g_pilot_mx`. olmoe's `PILOT=2/3` (2-/3-layer lookahead) exists
   (`olmoe.c:12,792`) but the mainline default is 1 layer; qwen36.c adds EMA smoothing of
   router logits and a confidence-limited widened top-k (`qwen36.c:1994-2012`,
   `pilot_conf_limit`, `g_wide`).

### 1.6 The depth-1 speculation note — `colibri.c:10227-10233` (verbatim)

```c
/* Auto depth = 1, not 3. A GLM-5.2 744B sweep (DRAFT=0/1/2/3, streaming and
 * fully-resident) showed single-token speculation is the only depth that pays:
 * acceptance ~85% at depth 1 vs ~44-62% at 2-3, and every extra draft token
 * both costs verify compute and (when streaming) faults experts that evict the
 * LRU working set. Depth 1 was the fastest MTP setting in every measured
 * configuration; 2-3 never beat it anywhere. DRAFT=n still forces any depth. */
g_draft = (m.has_mtp && (!g_cuda_enabled || cuda_mtp)) ? 1 : 0;
```

Adjacent caveat (`colibri.c:10218-10224`): MTP is **off by default under CUDA** because
cold/streaming experts still run on CPU where S==1 and S>=2 kernels diverge in FP
accumulation order, collapsing acceptance (#163); `COLI_CUDA_MTP=1` opts in (30-50%
acceptance even with the mismatch). Adaptive pause: below 70% MTP / 50% corpus acceptance
over a 24-proposal window, drafts pause 256 tokens (w2/colibri-sched.md:243-245).

---

## 2. CPU/GPU task dispatch

### 2.1 Per-expert decision inside moe()

Assignment is **not** a static layer table in the GLM engine — it is per-slot capability:
an `ESlot` whose g/u/d weights were uploaded has `cuda_eligible` set and its CUDA tensors
resident; those join a per-device **group**; everything else is computed on the CPU by the
main thread (`expert_ffn`, internally OpenMP-parallel). The qwen36 engine is the cleaner
tier: `qt_ready()` → `qt_issue`/CPU-miss-loop/`qt_take` (`qwen36.c:1622-1669`).

### 2.2 The early-issue + take pattern — "t_emm = max(cpu,gpu), 1 sync/device"

CUDA Inc.4, `colibri.c:5104-5181`: *pass 1* packs all VRAM-resident experts of the block
per device and **issues** them async (`coli_cuda_expert_group_issue`, one call per device)
*before* the CPU loop starts; *pass 2* is the CPU loop over RAM-tier/miss rows, which
skips `done_j[]` experts already on the GPU (5377) and `pipe_wait`s only the slots it
needs; *pass 3* is the take (`colibri.c:5448-5478`): one
`coli_cuda_expert_group_take(device)` per device — i.e. **one `cudaStreamSynchronize`
per device per block** (`backend_cuda.cu:2106-2113`), then the weighted accumulate; a
failed take recomputes those rows on CPU (`expert_host_ensure` reloads the released
slab). Gated by `COLI_GROUP_ASYNC`, `S<=4` (decode only), `!omp_in_parallel()`. Issued
experts are refcounted (`eslots_acquire` before issue / `eslot_release` after take) so the
LFRU/swap machinery cannot free an in-flight tensor; the qtier has the same
`issue_open` guard (`qwen36_tier.c:44,81,358,399-402`: uploader frees a swap victim only
when no group is in flight).

qwen36's canonical version (`qwen36.c:1622-1669`) is the 20-line template:

```c
for (kk..) { expert_get(...); if (e->g4) qt_note(layer, idx[kk], ...); }  /* heat + hint */
uint32_t qmask = qt_issue(layer, idx, K, xs);      /* async groups on all devices */
for (kk..) if (!(qmask & (1u<<kk))) { ...matmul_qe CPU miss expert... }
/* shared expert computed NOW so it overlaps the GPU groups */
matmul_d(sh...); matmul_d(shu...); ...silu...; matmul_d(shd...);
qt_take(qmask, val, K, out + s*D);                 /* one sync/device + accumulate */
```

The **shared expert is deliberately computed between issue and take** (comment
`qwen36.c:1644-1645`) to fill the overlap window — same trick as issuing the slower
device first (`colibri.c:5292-5301`, vk2 submit on a worker thread so dev2's submission
cost overlaps dev0 issue + CPU share).

### 2.3 Activation handoff

- Expert level: plain `std::memcpy` into a **pinned host_x staging buffer** then
  `cudaMemcpyAsync` H2D on the device stream (`backend_cuda.cu:2031-2036`); result via
  async D2H into pinned host_y, consumed after the take sync (2095-2096). **No events, no
  flags** — stream order + the single take sync is the whole protocol. `reserve_pinned` =
  `cudaMallocHost` (`backend_cuda.cu:1076-1079`); weights are never in these buffers.
- Layer level (GPU-resident sparse layers): `pipe_layer_sparse` (`colibri.c:6128-6260`)
  keeps the residual `x_dev` **on device across layers**; only (a) the post-attention
  `nrm` is downloaded (sync `pipe_download`) for the CPU router + cold experts, (b) new KV
  records, (c) pre-attention nrm on DSA layers cross to host. The shared expert is issued
  on GPU before `moe()` runs on CPU; the routed result is uploaded and both residual adds
  are ordered on one stream — **"no pipe_sync at the end: the next layer's pipe_download
  provides the implicit sync point"** (comment 6217-6220). The residency chain and its
  teardown (stale-host-x hazard, peer-copy fallback) is at `colibri.c:6402-6449`.

### 2.4 Thread/affinity layout

Main decode thread = sole pipe-gen writer + all CPU matmul (OpenMP-parallel inside) + GPU
issue/take + sampling. PIPE workers (≤16) run preads only — they must "stay off the omp
team and off these I/O threads" (`colibri.c:3299-3301`) because `matmul_qt` dispatches to
GPU only when `!omp_in_parallel()`. Pilot worker(s): 1 (URING/hint) or N
(`PILOT_WORKERS`, blocking PILOT_REAL only). qtier: 1 uploader thread + its queue
(depth 48, ~1.6 MB staged copies/entry, `qwen36_tier.c:11,33`). **No thread affinity
pinning anywhere in colibri itself** — only an OMP env workaround that re-execs with a
full affinity mask (`colibri.c:9831-9848`); `omp_tune.h:157` sets thread *count* to
physical cores. (Insignia already went further: `CpuPool` pins LP0-5 per core,
`insignia_cpu.hpp:319-324`; readers on LP6-11, `streaming.cu:32,47`.)

---

## 3. LFRU residency scoring

### 3.1 Score — `tier.h:27-33` (verbatim; olmoe copy `olmoe.c:115-119`)

```c
/* LFRU: frequency is the primary signal; recency breaks close calls. A recent
 * access contributes at most 255 points while one frequency count is worth
 * 256, so a merely recent expert cannot displace a genuinely hotter one. */
static uint64_t tier_lfru_score(uint32_t heat, uint32_t last, uint32_t clock){
    uint32_t age=clock-last, recent=age<255?255-age:0;
    return ((uint64_t)heat<<8)|recent;
}
```

`heat` = `eheat[layer][eid]`, incremented on every demand routing hit
(`colibri.c:4750`: `if(m->eheat[layer][idx[kk]]<UINT32_MAX) m->eheat[layer][idx[kk]]++`);
`last` = `elast[layer][eid]`, stamped `++m->eaccess_clock` per access (4751). Speculative
pilot loads are **never classified** (demand=0 path) — they don't bump heat, and a
separate private clock `eaccess_clock_dc`/`elast_dc` exists for DISK-CLASS bookkeeping so
classification recency can't pollute LFRU (Model fields, `colibri.c:448-471`).

### 3.2 Swap selection + hysteresis — `tier.h:35-54`

Coldest resident (`pinned[]` scan) vs hottest non-resident; swap only if
`hs > cs + (cs>>2) + (4u<<8)` — the 25% + 4-frequency hysteresis in score units. The
pilot's **eviction guard** reuses the same margin to protect a warm victim from a
speculation (`colibri.c:5800-5811`: victim protected only when `heat>=2` AND
`vs+(vs>>2)+(4u<<8) > cs`; otherwise the speculation is *dropped*, not the resident).
Note #490 (comment 5802-5803): the un-narrowed guard dropped ~all speculations on a full
cache — hysteresis narrowing matters.

### 3.3 Promotion cadence

- **GLM ecache:** promotion happens per MoE block (FASE D swap-in, §1.4 above); LFRU
  scoring is used at victim selection (`eslot_lru_victim`) and by the pilot guard.
- **qtier (VRAM):** `qt_lfru_tick_locked` runs **every 16 ticks** — one tick per token,
  checked once per token at layer 0 (`qwen36_tier.c:327-329` + gate at 357
  `if(layer==0) qt_lfru_tick_locked()`): per device, coldest resident vs hottest
  non-resident with the same `hh<=ch+(ch>>2)+4` hysteresis; a swap is budget-neutral
  (victim freed only when `issue_open==0`, uploader waits on `cv_take`,
  `qwen36_tier.c:79-87`).
- **Pinned caps interaction:** the `pinned[]`/resident lists are inputs to the scans
  (residency check `for z<npin if pinned[z]==e` — pins are never victims); the qtier
  requires **full RAM residency** — `qt_init` refuses when `cap != n_experts`
  (`qwen36_tier.c:116-119`: "needs full RAM residency") and per-device `budget[i]` =
  free VRAM − 1 GB or `CUDA_EXPERT_GB`; on upload failure the budget is clamped to what
  actually fit (`G.budget[hd]=G.used[hd]`, 106-107). Learned heat persists across runs via
  `HEAT_FILE` (`qt_init` load 154-169 / `qt_shutdown` save 427-439), and warmstart fills
  budgets in heat-descending order (`qt_fill_next`/`qt_plan_fill`, 242-297).

---

## 4. Windows I/O specifics

### 4.1 O_DIRECT twin — `compat.h:296-311` (verbatim)

```c
/* --- O_DIRECT -> FILE_FLAG_NO_BUFFERING ---
 * Apre il fd "gemello" senza cache del file system, come il twin O_DIRECT di
 * st.h su Linux e F_NOCACHE su macOS. Stesso contratto: offset, lunghezza e
 * buffer del chiamante devono essere allineati a 4K ...; richieste non
 * allineate falliscono con -1, mai dati corrotti. */
static inline int compat_open_direct(const char *path){
    HANDLE h = CreateFileA(path, GENERIC_READ,
                           FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,
                           NULL, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, NULL);
    if(h == INVALID_HANDLE_VALUE) return -1;
    int fd = _open_osfhandle((intptr_t)h, _O_RDONLY|_O_BINARY);
    if(fd < 0){ CloseHandle(h); return -1; }
    return fd;
}
```

Every shard gets **both** fds eagerly at open: buffered `fds[i]` + direct `dfds[i]`
(`st.h:141-158`); `st_direct_fd` maps between them (`st.h:160-168`). Measured rationale
(st.h:161): "buffered read on ext4-in-VHDX chokes at ~0.8 GB/s, O_DIRECT reaches 2.3+".
Mirrors (multi-SSD read replicas) duplicate the twin per copy (`st.h:44-52,170-182`).
Sub-512B EOF tails and misaligned one-shots go through the buffered twin.

### 4.2 The other shims (contract notes for us)

- `compat_pread` = `ReadFile` + `OVERLAPPED` on the raw OS handle, thread-safe, >4 GB
  offsets, 2 GB chunking, per-thread `GetLastError` preserved (`compat.h:158-193`).
- `compat_fadvise(WILLNEED)` = a **blocking overlapped read into a scratch buffer whose
  only purpose is to populate the standby list**, capped at 64 MB
  (`compat.h:134-156`) — the later synchronous pread then faults from RAM. DONTNEED and
  friends are no-ops on Windows.
- `lseek(SEEK_END)` returns -1 on UCRT NO_BUFFERING fds → use `GetFileSizeEx`
  (`compat.h:313-323`, `compat_fsize`).
- `posix_memalign`→`_aligned_malloc` + matching `compat_aligned_free`
  (223-231) — plain `free()` on an aligned pointer corrupts the CRT heap (0xC0000374),
  the bug class their own expert path hit (`colibri.c:3466-3471`).
- Avail-RAM sizing via `GlobalMemoryStatusEx` `ullAvailPhys` (237-246) for cache caps.

---

## 5. CUDA-side consumption of host-resident weights

**Colibri never has the GPU read host-resident weights.** There is no `cudaMallocManaged`,
no `cudaHostRegister` of weights, no zero-copy mapped host pointers anywhere in
`backend_cuda.cu` (grep verified; only activations use pinned staging). Two tier styles:

1. **VRAM tier (expert tensors):** weights `cudaMalloc`'d once
   (`backend_cuda.cu:1370,1386`), uploaded via `cudaMemcpyAsync` from a *staging copy*
   the uploader thread prepared (`stage()` XOR 0x88 offset-binary pack +
   `coli_cuda_tensor_upload(_g)`, `qwen36_tier.c:53-66,89-101`), then kept resident and
   LFRU-swapped **on device**. Kernels read only device pointers (`GroupDesc` carries
   `g->weights` device pointers, `backend_cuda.cu:2007-2009`).
2. **CPU tier (cold/host experts):** computed on the **CPU**, full stop. The GPU never
   touches them; their slabs are `munlock`ed/`MADV_DONTNEED`ed or freed once a slot goes
   VRAM-resident (`expert_host_release`, `colibri.c:3457-3489`).

Why: their per-expert working set (~10-20 MB) amortizes upload against hundreds of
resident uses, and decode GEMV kernels from host memory over PCIe (~13-25 GB/s practical)
would lose to both VRAM (~500+ GB/s) and to just doing the matvec on the CPU (they have
AVX2/NEON int4/IDOT kernels). The ANS-compressed variant likewise stages: pread into a
pinned host arena → async H2D → GPU-side DietGPU ANS decompress on load
(`backend_cuda.cu:1483-1518`).

For activations, every device has `host_x`/`host_y` pinned (cudaMallocHost) staging +
`group_desc` device buffer; issue does memcpy→H2D→kernels→D2H, take does the single
`cudaStreamSynchronize` (§2.2-2.3). Multi-GPU: device chosen by `home(eid)=eid%ndev`
sharding, per-device input replicas (`qwen36_tier.c:51,370-374`), issue to all devices
then take each — the "1 sync/device/step" pattern.

---

## 6. What colibri does NOT do that we plan (v2 CPU-tier deviation)

1. **Whole dense layers computed on CPU, alternating with GPU layers.** Colibri's unit of
   CPU compute is the *expert* (or shared expert) inside a layer whose attention runs on
   GPU; a *dense* layer is either fully VRAM-resident (`pipe_layer_sparse`) or fully CPU
   (no-GPU fallback). Our v2 plan (`audits/w4/tier-dispatch.md`: V/Z→GPU, C/N→CPU whole
   layers, eng_of[64] baked table) has no colibri precedent. **But** colibri does prove
   every ingredient separately: (a) activation handoff host↔device per layer with pinned
   staging and at most one sync (§2.3); (b) the residual chain staying on device across
   consecutive GPU layers with the *implicit* sync of the next download
   (`colibri.c:6217-6220`); (c) the main thread acting as both coordinator and CPU
   compute (OpenMP inside its matmuls); (d) "layers alternate engines but never split a
   layer" — our own w3/colibri-sched-deep §8.4 phrasing, which colibri respects at expert
   granularity within moe but we extend to layer granularity. The throughput claim for
   CPU dense FP8 GEMV comes from our own `audits/w3/cpu-fp8.md` + `w4/cpu-tier.md`
   (~37 GB/s, 10.8 ms/layer), not from colibri.
2. **GPU reading host-resident weights (our Z-tier UVA plan).** Rejected by colibri by
   omission (§5). Their Z-equivalent (qtier) *uploads* to VRAM instead. Our deviation is
   justified by our different ratio: a 381 MB dense layer used once per token has zero
   reuse to amortize an upload, and 18 GB/s UVA beats 10.8 ms CPU when the CPU is busy
   with C layers. Colibri's evidence neither supports nor refutes UVA GEMV performance —
   it is silent (see `audits/w3/pcie-pipeline.md` + `w4/tier-dispatch.md` for our own
   numbers). Colibri's pinned-staging-only discipline does warn: if UVA stalls, their
   pattern (stage→H2D→compute) is the fallback shape we already implemented in
   `PinnedRing` (cudaHostRegister'd, `streaming.cu:289-319`).
3. **Static baked placement.** Colibri's placement is dynamic (heat-driven LFRU swaps,
   pilot predictions). Our v2 bakes eng_of[64] at startup (dense model, uniform layers →
   nothing to differentiate by heat; the LFRU signal colibri exploits exists only because
   MoE routing is skewed). Correct deviation; keep `tier_lfru_score` in the back pocket
   if we ever add dynamic residency.
4. **io_uring / Linux paths** (`uring.h`, `colibri.c:3123-3292`): N/A on Windows; our
   IOCP reader is the equivalent and already outperforms the pread-pool shape for
   sequential layer streaming (their pool serves *random* expert misses).

---

## 7. Steal map — every colibri mechanism → Insignia target

| # | Colibri mechanism (file:line) | Insignia target | Verdict | One-line reason |
|---|---|---|---|---|
| 1 | Gen-tagged SPMC cursor `(gen<<8)\|idx`, sole-writer gen, CAS-claim, release-publish (`colibri.c:3303-3424`) | `include/insignia_cpu.hpp:212-278` CpuPool `claim_=(gen<<32)\|ticket` | **ADOPTED (already in)** | Ours is the same pattern with a 32/32 split — colibri's ≤64-job 8-bit index would overflow our ticket counts; keep ours. |
| 2 | per-slot `ready[]` release-flag + spin→condvar wait (`colibri.c:3348,3435-3454`, #159) | `PinnedRing::acquire` spin→WaitForSingleObject (`src/streaming.cu:341-351`) | **ADOPTED (already in)** | Same spin-first-then-park; our Windows auto-reset event is the cv_done analog and cheaper to reason about. |
| 3 | dispatch publish order (state relaxed → ready reset → gen release → broadcast) (`colibri.c:3416-3423`) | `LayerFeeder::begin_epoch`/`arm_locked` (`streaming.cu:373-408`) | **ADOPTED (adapted)** | We publish per-slot (claim→submit→publish on IOCP callback) instead of one gen; the ordering discipline (reset-before-publish) is preserved in `try_claim` state machine. |
| 4 | Early-issue GPU groups, CPU share between, one take-sync per device (`colibri.c:5104-5478`; `qwen36.c:1622-1669`) | future `generate27.cu` / `Qwen35Decode::forward_body` (`src/decode.cu:133`) — issue V+Z kernels on the compute stream, run C/N layer on `CpuPool` during, single stream-sync at engine boundary | **ADOPT — the headline steal** | This is exactly our treadmill's V/Z-ahead, C-behind structure; qwen36.c:1622-1669 is a 40-line reference implementation including the "shared expert between issue and take" filler trick. |
| 5 | Shared-expert / slow-device-first ordering to widen the overlap window (`qwen36.c:1644-1663`, `colibri.c:5292-5301`) | generate27 loop filler between Z-issue and take | **ADOPT** | Free throughput: insert independent CPU work (next C layer's first GEMV, MTP draft prep) into the issue→take gap. |
| 6 | Activation handoff: pinned host_x/host_y + memcpy→async H2D, async D2H, no events (`backend_cuda.cu:2031-2096,1076-1079`) | generate27 C↔GPU layer boundary (CpuPool writes into pinned staging, async H2D on copy stream) | **ADOPT** | Stream-order + one sync beats event bookkeeping; our `PinnedRing` slots are already cudaHostRegister'd so activations can ride the same allocation class. |
| 7 | Residual stays on device across consecutive GPU layers; implicit sync via next download (`colibri.c:6128-6260,6402-6449`) | generate27 V-then-Z and Z-then-Z runs (hidden vector never round-trips) | **ADOPT** | Removes a per-layer H2D/D2H pair (16 KB×2) and a sync for our 40-of-64 GPU-run majority; copy their stale-host-x hazard comment verbatim. |
| 8 | `tier_lfru_score = (heat<<8)\|max(0,255-age)` (`tier.h:30-33`) | (potential) dynamic residency scorer in `include/insignia_streaming.hpp` | **REJECT for v2** | Dense layers have no per-layer heat skew; static baked placement (w4/tier-dispatch) is strictly simpler. Keep formula cited for any future NVMe-page-level residency. |
| 9 | LFRU swap every 16 tokens @layer0 + hysteresis `cs+(cs>>2)+(4<<8)` + pilot eviction guard (`qwen36_tier.c:325-345`; `colibri.c:5800-5811`) | same as #8 | **REJECT for v2** | Nothing to swap in a static plan; the *hysteresis margin idea* (25%+ε before any swap) is worth keeping if #8 ever activates. |
| 10 | HEAT_FILE persist + heat-descending warmstart (`qwen36_tier.c:154-169,427-439`) | n/a | **REJECT** | No routing heat in a dense model. |
| 11 | PILOT router-lookahead prefetch (71.6% recall, 1-layer; EMA + confidence top-k in `qwen36.c:1980-2042`) | replaced by `LayerFeeder` deterministic cyclic read-ahead (`slots-1`, `streaming.cu:395-408`) | **REJECT (superseded)** | Dense layers are all needed every token — prediction is unnecessary; the cyclic schedule is the degenerate-but-exact version (w3/colibri-sched-deep §8.1 agrees). |
| 12 | Visible reservation `eid=-(eid+2)` + `used=(uint64_t)-1` sentinel for in-flight fills (`colibri.c:5813-5818`) | `PinnedRing` slot states FREE/FILLING/READY/IN_USE (`streaming.cu:331-356`) | **ADOPTED (equivalent)** | Our explicit FILLING state encodes the same "never a victim while loading" contract without the sentinel-value trick. |
| 13 | O_DIRECT twin handles, eager open, buffered twin for tails (`compat.h:296-311`; `st.h:141-168`) | `NvmeReader::file_index_locked` direct+twin (`streaming.cu:93-109`), tail via twin (`137-142`) | **ADOPTED (already in)** | We added FILE_FLAG_OVERLAPPED on the direct handle (theirs is sync-pread); both derive from the same 4K-alignment contract. |
| 14 | WILLNEED→standby-list read analog (`compat.h:134-156`) | n/a | **REJECT** | Our reader is already full-throughput async on the direct handle; a blocking read to warm a cache we deliberately bypass (NO_BUFFERING) is a no-op. |
| 15 | PIPE default-ON on Windows, PIPE_WORKERS=8 (16 max) (`colibri.c:3321-3339`) | reader thread count 2-3 (`streaming.cu` ctor, `kReaderAffinity`) | **ADAPT (thread count only)** | Their 8 pread workers serve random expert misses (QD via threads); our IOCP QD16 + 2-3 threads saturates the E: drive for sequential streams (w3/io-bench-results). |
| 16 | I/O workers must never join the compute team / run `matmul_qt` (`colibri.c:3296-3301`) | readers pinned to LP6-11 SMT siblings, pool owns primaries (`streaming.cu:32-48`; `insignia_cpu.hpp:319-324`) | **ADOPTED (stronger)** | We formalized their comment into hard affinity masks. |
| 17 | Depth-1 MTP only; deeper never won; CUDA cold-kernel divergence kills acceptance (`colibri.c:10216-10236`) | MTP draft-1 in generate27 (draft layer already ~4 ms warm) | **ADOPT** | Confirms our D1-first plan; also adopt their warning: keep draft+verify on one engine (our draft is all-VRAM, so the #163 class of divergence can't bite). |
| 18 | qtier uploader thread, queue depth 48, staging copies, `issue_open` no-free-while-in-flight (`qwen36_tier.c:33-111`) | v1 N→GPU-UVA variant: H2D tails on a copy stream while next slot fills | **ADAPT** | Borrow the queue+staging-copy shape and the in-flight guard if v1 streams N layers to GPU instead of v2's CPU compute; v2 needs neither. |
| 19 | Weight staging with format conversion at upload time (XOR 0x88 int4 pack, `qwen36_tier.c:53-66`) | n/a | **REJECT** | MXFP4/FP8 device decode already exists here (`mxfp4_gemv_v2*`); no upload-time transform wanted. |
| 20 | CPU-share reordering: pipe-ready/resident experts first, in-flight last (ordering hint only) (`colibri.c:5309-5324`) | generate27 pacing: consume the N slot that's READY while another still fills | **ADAPT** | The treadmill already fixes order statically; if a fill is late, the "hint-only class" idea (reorder to whatever is ready, correctness unaffected) is a cheap reschedule fallback. |
| 21 | `qt_take` failure → CPU recompute fallback (`colibri.c:5465-5472`; `qwen36.c` qmask clearing) | generate27 engine fallback (UVA stall → CPU layer) | **ADOPT** | Losing-device recovery by recompute maps directly to our UVA-timeout → CpuPool path; keeps the decode alive rather than aborting the step. |

**Insignia targets referenced:** `src/streaming.cu` (NvmeReader/PinnedRing/LayerFeeder —
smoke-verified), `include/insignia_cpu.hpp` (CpuPool §5.3 + FP8/BF16/Delta/GQA jobs),
`src/decode.cu:133` `forward_body` (the loop generate27 will replace with the
tier-dispatched treadmill; no generate27.cu exists yet — it is the future file per
w4/tier-dispatch.md).

---

## 8. Corrections / confirmations vs the w2/w3 audits

- w2/colibri-sched.md §4 and w3/colibri-sched-deep §1 quote the pool identically to the
  clone — no drift. The "85% vs 44-62%" numbers trace to `colibri.c:10227-10233` (above).
- w2/colibri-io.md §3.1 cites `compat.h:296-311` — still exact (current file has it at
  those lines).
- w3/colibri-sched-deep §8's "layer-granular streaming has no colibri precedent" is
  **confirmed** (§6.1 here). Its thread-roles table matches §2.4, with the addition that
  the qtier uploader thread exists only in the qwen36 engine.
- One nuance neither audit headlines: `PIPE` (the load pool) is **Windows-default-ON**
  (`colibri.c:3321-3324`) — colibri ships the SPMC pread pool as the *primary* Windows
  path, io_uring only on Linux. Their Windows I/O stack (twin fds + pread pool) is the
  direct ancestor of our NvmeReader design, which replaced threads-with-pread by
  IOCP+OVERLAPPED for sequential layer streams.
