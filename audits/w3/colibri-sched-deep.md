# Colibri heterogeneous scheduling — deep extraction for Insignia 27B-FP8 (w3)

Source of truth: `E:\coding\Insignia\colibri\` (read-only). Every claim below was re-read
at the cited `file:line` in this checkout; w2 report claims that survived re-verification
are folded in, and the handful that needed correction are flagged. Companion reports:
`audits/w2/colibri-io.md` (storage), `audits/w2/colibri-sched.md` (w2 pass),
`audits/synthesis.md` (27B model facts + feasibility math).

Model target from synthesis.md: Qwen3.8-27B-FP8, 25.65 GB text weights, 64 layers
(48 linear-attention + 16 full-attention), ~383.87 MB/layer uniform shards, embed +
lm_head 2.543 GB bf16 each, MTP 477 MB. Rig: 4070 SUPER 12 GB + 5600X 6C/12T + 16 GB
RAM + Gen4 NVMe (~6.8 GB/s at QD8). Tier costs per layer: VRAM ~0.76 ms (0.5–1.4),
RAM→CPU compute ~9.6–10 ms, RAM→PCIe stream 15.4 ms, NVMe ~56.5 ms.

---

## 1. The PIPE load pool — the `(gen<<8)|idx` SPMC (GLM engine)

### 1.1 Struct + the invariant block (`colibri.c:3294-3354`)

```c
/* ============================ PIPE: load ‖ matmul ============================
 * Overlap NVMe expert-weight loads with expert matmul. A small persistent pool
 * of I/O worker pthreads runs the misses' pread (expert_load) into distinct
 * ws[] slabs and sets a per-slot `ready` flag; the MAIN thread walks the block's
 * experts in order, waiting on ready[q] only for the expert it needs right now,
 * and does all matmul_qt on itself ...                                                  */
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
```

The load-bearing invariants are the header comment (`colibri.c:3303-3319`):

- **gen is written only by the main thread**, monotonic bump ⇒ **no ABA**: a straggler
  worker preempted anywhere re-enters with a gen-checked CAS and can never grab a
  wrong-generation job.
- **Payload (`eids[i]`, `layer`) is read only AFTER the winning CAS**, and the CAS
  comparand carries the generation — torn batch-state reads are structurally impossible.
- Dispatch writes all batch state RELAXED, then **RELEASE-stores `cur`** to publish;
  workers ACQUIRE-load `cur`. `ready[]` reset happens BEFORE the publish store.
- The per-expert `pipe_wait(ready[q])` inside the consumer loop guarantees every grabbed
  job completes before the block ends — "no grab outlives its generation". This is why
  the old `active` counter and end-of-block drain barrier were **deleted** as redundant.
- **Mutex/condvar exist ONLY to park idle workers, never for correctness.**

### 1.2 Worker claim loop (`colibri.c:3356-3385`) — exact

```c
static void *pipe_worker(void *arg){
    (void)arg; PipePool *p=&g_pp; uint64_t seen=0;
    for(;;){
        pthread_mutex_lock(&p->mx);
        while((atomic_load_explicit(&p->cur,memory_order_relaxed)>>8)==seen)
            pthread_cond_wait(&p->cv,&p->mx);
        pthread_mutex_unlock(&p->mx);
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
                expert_load(p->m,L,eid,&p->m->ws[i],1,1);  /* needed-now load: fatal on I/O error; demand=1 */
                atomic_store_explicit(&p->ready[i],1,memory_order_release);
                ...
            }
            /* CAS failed → another worker advanced index (or gen advanced): re-loop */
        }
    }
}
```

Note what the CAS actually claims: **`cur+1`** — one 64-bit word simultaneously (a)
reserves job `i` for this worker, (b) advances the index for everyone else, (c) would
fail atomically if the generation moved. 8 bits of index ⇒ **hard cap 64 jobs/batch**
(matches `eids[64]`, `ready[64]`, `ws[64]`); gen occupies the upper 56 bits.

### 1.3 Dispatch (`colibri.c:3404-3424`) — exact publish protocol

```c
static void pipe_dispatch(Model *m,int layer,const int *eids,int njobs){
    g_pp.m=m;
    atomic_store_explicit(&g_pp.njobs,njobs,memory_order_relaxed);
    atomic_store_explicit(&g_pp.layer,layer,memory_order_relaxed);
    for(int q=0;q<njobs;q++) atomic_store_explicit(&g_pp.eids[q],eids[q],memory_order_relaxed);
    for(int q=0;q<njobs;q++) atomic_store_explicit(&g_pp.ready[q],0,memory_order_relaxed); /* reset BEFORE publish */
    uint64_t g=(atomic_load_explicit(&g_pp.cur,memory_order_relaxed)>>8)+1;
    atomic_store_explicit(&g_pp.cur,(g<<8),memory_order_release);                          /* PUBLISH */
    pthread_mutex_lock(&g_pp.mx); pthread_cond_broadcast(&g_pp.cv); pthread_mutex_unlock(&g_pp.mx);
}
```

### 1.4 Full-pool backpressure — there is none, by design

There is **no queue-full condition** on this pool. Backpressure is structural:

1. `njobs ≤ 64` is enforced by the *producer* (the moe() block is processed in
   64-expert chunks, `colibri.c:4923`: `for(int base=0;base<nu;base+=64)`).
2. The consumer's per-slot `pipe_wait(qof[j])` (`colibri.c:5385`) is the drain: before
   the end-of-block LRU swap recycles a `ws[]` slab, every dispatched slot has been
   waited. Under METAL the block-level drain is mandatory *earlier* (`colibri.c:5077-5087`)
   because the GPU path would otherwise hand a half-loaded slab to a kernel and would
   skip the CPU loop's waits entirely (comment lists both reasons verbatim).
3. A new batch cannot be published until the consumer has consumed the old one — the
   main thread is both sole publisher and sole consumer-waiter, so dispatch rate is
   self-throttled by the matmul loop.

The *other* two queues in the engine do have explicit full policies:
- **pilot ring** (4096 slots, 1P/SPC): full ⇒ **drop the hint silently**
  (`colibri.c:6110-6113`: `if(w-__atomic_load_n(&pilot_r,__ATOMIC_ACQUIRE)<4096){...}`);
  "un hint perso non e' un errore" — safe because hints are advisory.
- **qwen36_tier upload ring** (48 slots): full ⇒ **skip + count** for demand-note
  (`qwen36_tier.c:194`: `if(G.qn>=QT_QCAP){ G.q_full_skips++; return 0; }`), but
  **blocking** on the warmstart path (`qt_note_block`, `qwen36_tier.c:230`:
  `while(G.qn>=QT_QCAP && !G.th_stop) pthread_cond_wait(&G.cv_take,&G.mx);`). That
  split — *admissible drop for speculation, mandatory block for demand* — is the
  transferable rule.

### 1.5 Waiting (`colibri.c:3435-3454`)

Default `pipe_wait` = `sched_yield()` spin on `ready[q]` (legacy byte-identical);
`COLI_PIPE_BLOCK=1` switches to mutex+condvar with double-check (worker stores ready
RELEASE *before* taking mx to broadcast ⇒ no lost wakeup). Measured justification
(`colibri.c:3327-3332`): "a yield storm on the main thread fights the OpenMP team ...
the condvar wake costs ~5 us against reads that cost 0.5-3 ms (#159)".

The **pilot SPMC claim** (`colibri.c:5922-5942`) is the second CAS pattern — payload
read *before* CAS, `pilot_r` CAS-advanced, proven by
`c/tests/test_pilot_ring.c` (300k items / 8 threads, exactly-once, no tears). Covered
in w2-io; unchanged on re-read.

### 1.6 Knobs (verified `colibri.c:9945-9977`)

`PIPE` default **1 on `_WIN32`, 0 elsewhere** (comment: "default ON: overlap expert
load ‖ matmul ... PIPE=0 opts out"); `PIPE_WORKERS` default 8 clamp [1,16];
`pipe_workers_imply_pipe()` — setting `PIPE_WORKERS>0` with `PIPE` unset forces the
pipe on, because "a whole campaign had it set with the pipe off without noticing".
`docs/tuning.md:22`: PIPE overlaps pread with matmul, **−18% disk service**;
`docs/tuning.md:23`: `DIRECT=1` measured **+65% alone** on a Strix Halo.

---

## 2. Layer→device assignment + thread inventory + early-issue

### 2.1 How assignment is decided (all colibri engines)

Colibri assigns **experts**, not layers, and the answer is: **profile-driven static
placement computed at startup, then (optionally) slow live re-pin between turns.**

- **RAM tier (pins):** `pin_load` builds `PinRec{l,e,c}` from the persisted
  `.coli_usage` routing histogram (`route_trace.h`; `rt_load` accumulates
  `<layer> <expert> <count>` rows, `rt_save` writes them back with an optional
  per-save decay `COLI_USAGE_DECAY`, default 1.0, so history has a configurable
  half-life "in turns" — measured motivation at `route_trace.h` (~line 300): after
  18.2M recorded selections a turn moves the ranking by 0.2%). Sorted
  frequency-descending (`pin_rec_cmp`, `colibri.c:8823-8825`), sliced to the byte
  budget by `pin_count_for_budget` (`colibri.c:8842-8852`), which prices each row at
  its true width (`expert_bytes_row`) — comment #885: pricing mixed-width containers
  at the widest row bought the wrong 1,352 experts.
- **VRAM tier:** the **budget-sized head of the same ranking**. `colibri.c:8967-8991`:
  per-device `remaining[i] = free − dense_projected − reserve_gb`; `CUDA_EXPERT_GB`
  explicit or `auto = Σ remaining`; `prefix_est = budget/eb` experts. The per-expert
  device pick in the load loop (`colibri.c:9064-9102`) is **greedy min-bytes** (default)
  or **min-weighted load-balance** (`CUDA_EXPERT_LOAD_BALANCE=1`, compares `placed_w`
  built from the usage counts). Upload failure sets `remaining[best]=0` and tries the
  next device — graceful degradation instead of OOM (#491 comment above it).
- **qwen36 tier:** placement is `home(eid) = eid % ndev` (`qwen36_tier.c:51`) — static
  round-robin across GPUs — with the *residency set* still heat-ranked from a persisted
  `HEAT_FILE` (`qt_plan_fill`, heat-descending qsort, `qwen36_tier.c:272-297`; saved
  back in `qt_shutdown`, loaded with a one-shot `>>1` decay, `:163`).
- **Vulkan tier:** same usage-ranked fill with priority classes (`colibri.c:8510-8538`).
- **Live re-pin (REPIN, default off):** between turns, `repin_pass` → `tier_pick_lfru`
  per layer, **max 4 swaps/pass** (`colibri.c:7585`), executed at the safe point after a
  reply ("all device work is synchronized", `colibri.c:7010`).

**No engine places by measured runtime.** It is always *access-frequency history*
(persisted) + *byte budgets* (probed free memory) + *greedy balance* at load. The
startup placement is explicitly NOT LFRU — LFRU only governs later dynamics.

### 2.2 Thread inventory (whole tree, verified)

| Thread(s) | Count | Created at | Job | Synchronization |
|---|---|---|---|---|
| Main decode thread | 1 | — | routing, all CPU matmul, GPU issue/take, sampling, spec loop; **sole writer of pipe gen** | — |
| PIPE I/O workers | 8 def, [1,16] | `pipe_init` `colibri.c:3386-3400` | demand expert preads into `ws[]` | gen-tagged CAS cursor + per-slot `ready[]` |
| Pilot workers | 1 def, [1,16] | `colibri.c:5965-5979` | speculative cross-layer preads from 4096-slot SPMC ring | CAS ring claim + `eid=-(eid+2)` slot reservation |
| qtier uploader | 1 | `qwen36_tier.c:171` | drains 48-slot staging ring, per-device tensor uploads, LFRU victim free | mutex + `cv` (work) + `cv_take` (space/issue_open) |
| vk2 issue worker | transient per block | `colibri.c:5296-5301` | offloads the 2nd Vulkan device's ~0.8 ms submit | join before take |
| Mirror stripe workers | per-read | `colibri.c:2514` | dual-SSD striped chunks of one expert | pthreads, joined |
| V4 loader lanes | 9 def, [1,16] | `deepseek_v4.c:4498-4539` | dual-expert O_DIRECT load pool | per-lane ready flags, `usleep(50)` wait |
| OMP pool | all cores | implicit | expert loads (PIPE=0), dense prefill, kernel-internal row parallelism | `omp_in_parallel()` gates all GPU dispatch |

Crucial role separation (`colibri.c:3299-3301`): "matmul_qt parallelises internally via
OpenMP and checks !omp_in_parallel() for GPU dispatch — so it must stay off the omp
team and off these I/O threads". The main thread is a *coordinator + CPU compute*
thread; I/O never runs OpenMP-parallel work.

### 2.3 Early-issue overlap — "1 sync per device per step"

Two complementary mechanisms, both verified:

**(a) Issue-all-then-take-all** (`colibri.c:5500-5533`):

```c
/* Inc.4: at decode scale, issue every device's group WITHOUT syncing, then take
 * them all — one stream sync per device per layer instead of a full staged
 * round-trip per call (measured: ~70% of the sync call is host-side wait).
 * Any issue failure drains what was issued and the whole layer falls back to the
 * sync path below, which recomputes from group_x (idempotent). */
```

The device side (`backend_cuda.cu:1992-2113`): `coli_cuda_expert_group_issue` packs
`GroupDesc{g,u,d ptrs, scales, fmts, rows[c], offset}` H2D, uploads the packed
activation block through **pinned staging** (`host_x` → `cudaMemcpyAsync`), launches
the whole layer's expert set as one `dim3(I, max_rows, count)` grid per projection,
enqueues the D2H into `host_y`, sets `group_pending=1`, **returns without sync**.
`take` is literally `cudaStreamSynchronize` + return `host_y` (`backend_cuda.cu:2106-2113`).
Gated `total>8 → return 0` — **decode-scale only** (`backend_cuda.cu:2022`), and
`ctx->group_pending` forbids two in-flight groups per device (`:2023`).

**(b) Early-issue BEFORE the CPU loop** (`COLI_GROUP_ASYNC`, `colibri.c:5104-5182`):
VRAM-resident experts of the current block are packed and issued first; the CPU loop
then computes RAM-tier/miss experts (skipping GPU-owned rows via `done_j[j]`,
`colibri.c:5377`); the take phase (`colibri.c:5452-5478`) syncs each device and
accumulates. Comment `colibri.c:5105-5110`: "t_emm becomes max(cpu, gpu) instead of the
sum". Failure ⇒ idempotent CPU recompute of exactly those rows (`colibri.c:5465-5472`).
Windows: `g_ovl_issue/g_ovl_cpu/g_ovl_take` timers bracket the three phases.

The same shape everywhere: qwen36 `qt_issue → CPU misses + shared expert inline →
qt_take` (`qwen36.c:1622-1669`, comment: "Compute the shared expert NOW so it overlaps
with the GPU groups"); Vulkan 2-GPU variant **issues the slower device first** so it
gets the longest overlap window (`colibri.c:5250-5372`, transient submit thread);
V4 pipelined refill runs "the lookups of group g+1 (parallel O_DIRECT reads) ...
concurrently with the uploads of group g (single-stream cudaMemcpy)" in one
dynamic-scheduled OMP loop whose item 0 is the previous group's upload
(`deepseek_v4.c:9778-9824`, group default 6, pipe cap 2/3·16).

**"Weights for layer l+1 issued while l computes"** in colibri is *I/O-side*, not
compute-side: next-block WILLNEED readahead (`colibri.c:5052-5066`) and the pilot's
router-predicted L+1 (or L+2/L+3) loads run during layer L's matmul; GPU upload of a
future expert happens through `qt_note` ("RAM-resident layer-L+1 candidates go to
VRAM asynchronously", qwen36 pilot lookahead). Colibri never streams *dense layer*
weights at all — dense weights are pread-once into RAM at startup and (under CUDA)
uploaded once, tracked as `g_cuda_dense_projected` and subtracted from the expert
budget *before* placement (`colibri.c:2064-2101, 8967-8971`). Insignia's
layer-granular streaming has no colibri precedent — but the degenerate cyclic version
(§8) is much simpler than anything colibri does.

### 2.4 Thread affinities — a deliberate negative finding

**Colibri pins no threads.** The only `sched_setaffinity` in the tree is a *reset to
all CPUs* before the OMP re-exec, undoing a user-exported `OMP_PROC_BIND` mask that
would jail the fresh team on one core (~20x slowdown, #471, `colibri.c:9831-9852`).
NUMA policy is memory-side only: `mbind(MPOL_INTERLEAVE)` on expert slabs (#82,
`colibri.c:1297-1369`; +40% on a 4-socket) with the #419 two-arenas-per-layer packing
to stay under `vm.max_map_count` (`colibri.c:8854-8901`). Wake-latency tuning is done
through OMP env instead (`colibri.c:9798-9828`): `OMP_WAIT_POLICY=active`,
`GOMP_SPINCOUNT=200000`, `KMP_BLOCKTIME=200` (libomp burns 100%·nthreads forever
without it, #341), `OMP_PROC_BIND=close` (pack the team on adjacent cores), `OMP_DYNAMIC=FALSE`.
V4's loader-lane sizing explicitly reasons "lanes block in pread and do not need whole
CPUs" (`deepseek_v4.c` comment near `:4530`) — readers share cores with the OMP team.

---

## 3. Staging rounds, ring management, Windows prefetch analog

### 3.1 The min(4GB, budget/8) staging cap (`colibri.c:9019-9045`) — exact

```c
int stage = pre_n;                       /* default: all at once */
#ifdef COLI_CUDA
    if(g_cuda_enabled && g_cuda_release_host && gpu_prefix>0 && budget>0){
        /* Tetto di staging: il piu' piccolo tra 4 GB e un ottavo del budget del tier,
         * mai meno di un esperto ... Un ottavo mantiene i round abbastanza grandi da
         * tenere occupati i thread del carico parallelo; il tetto assoluto protegge
         * chi ha poca RAM e un budget enorme */
        double cap = 4e9; if(budget/8.0 < cap) cap = budget/8.0;
        int st = (int)(cap/eb);
        if(st < 1) st = 1;
        if(st < stage) stage = st;
        if(stage < pre_n)
            fprintf(stderr,"[CUDA] tier staging: %d experts per round (%.1f GB host peak) "
                           "instead of %d at once (%.1f GB) — CUDA_RELEASE_HOST frees each "
                           "round before the next (#730)\n", ...);
    }
#endif
```

Origin #730: hosts with more VRAM than RAM (96 GB VRAM / 64 GB RAM) OOM'd because the
whole VRAM prefix was host-staged in aggregate before any release. The loop
(`colibri.c:9046-9107`) alternates: OMP parallel pread of `stage` experts → CUDA
upload + budget check → `expert_host_release` frees each round's host slab (an
`madvise(MADV_DONTNEED)`/`compat_munlock` + detach on the arena slice,
`colibri.c:3457-3489`) → next round. **Peak host RSS = one round.** Rationale of the
two caps: budget/8 keeps rounds big enough to keep the parallel loader threads
saturated; the absolute 4 GB protects small-RAM/big-budget hosts.

### 3.2 Ring management (qwen36_tier — the cleanest specimen, whole file read)

One uploader thread; ring of `QT_QCAP 48` entries, each owning a malloc'd staging copy
(`w=malloc(3*mb)`, `:198`); `stage()` (`:55-66`) converts packed int4 two's-complement
→ offset-binary (**XOR 0x88** per 64-bit word) and concatenates scales — *the staging
copy exists because a format conversion happens between RAM format and upload format*.
`qh/qt_/qn` cycle under one mutex with two condvars: `cv` = work available,
`cv_take` = queue space (broadcast by the uploader at `:78` right after popping). The
in-flight guard: `G.issue_open` is set by `qt_issue` (`:358`) and cleared by `qt_take`
(`:399-402`); the uploader **waits on `cv_take` before freeing an LFRU victim**
(`:81`: `while(G.issue_open && !G.th_stop) pthread_cond_wait(&G.cv_take,&G.mx);`)
because `coli_cuda_tensor_free` on a tensor whose group is executing would corrupt the
stream. LFRU tick every 16 tokens, on layer 0 (`:327-346, :357`), per device, raw-heat
hysteresis `hh<=ch+(ch>>2)+4` (`:340`).

Warmstart (`qwen36.c:2415-2456`): `qt_plan_fill` **reserves the whole budget once**
(atomic `used+=` under the lock), then any number of OMP loader threads call
`qt_note_planned` (blocking only on queue space), `qt_fill_wait` drains. Two measured
lessons embedded there: load **all** experts to RAM, not just the VRAM set — "otherwise
the first touch of a CPU-fallback expert triggers a ~12 ms container read in the middle
of decode (measured: 139 ms/token on a single-GPU run)" (`qwen36.c:2431-2434`); and
free the int8 RAM copy of a VRAM-resident expert right after staging ("never shows up
in peak RSS; rematerialized from g4 on eviction", `:2444-2448`).

### 3.3 Windows prefetch analog (`compat.h:114-156`) — verified verbatim

`posix_fadvise(WILLNEED)` on Windows = allocate a scratch buffer (cap **64 MiB**),
issue ONE blocking overlapped `ReadFile` into it, free it — the point is not the
data, it is that the read **populates the standby page cache** so the later demand
pread faults from RAM. Two policy rules baked into comments: called only from the
dedicated PILOT thread / next-block readahead, never inline ("inline fadvise submit
measured ~0.5 ms x 169k calls = +92 s / 48 tok", `compat.h:118-121`); **DONTNEED is a
no-op** ("Windows' standby-list trimming self-regulates under pressure, and on a
low-RAM host keeping the pages is what we want for reuse", `compat.h:122-125`).
Insignia's O_DIRECT primary path makes WILLNEED mostly moot (it warms the cache the
direct reads bypass — colibri skips weight WILLNEED under DIRECT=1, `colibri.c`/w2-io
§5.5); keep it only for small buffered sidecar reads.

---

## 4. LFRU — exact code, in-flight coordination, pins

### 4.1 Score (`tier.h:27-33`) and swap pick (`tier.h:35-54`) — verified verbatim

```c
/* LFRU: frequency is the primary signal; recency breaks close calls. A recent
 * access contributes at most 255 points while one frequency count is worth
 * 256, so a merely recent expert cannot displace a genuinely hotter one. */
static uint64_t tier_lfru_score(uint32_t heat, uint32_t last, uint32_t clock){
    uint32_t age=clock-last, recent=age<255?255-age:0;
    return ((uint64_t)heat<<8)|recent;
}
...
    uint64_t cs=tier_lfru_score(heat[pinned[cold]],last[pinned[cold]],clock);
    /* Retain the existing 25%+4-frequency hysteresis in score units. */
    if(hs<=cs+(cs>>2)+(4u<<8)) return 0;
```

`heat` = saturating per-(layer,expert) access counter, bumped once per routed
(token, expert) at FASE A (`colibri.c:4742-4751`); `last`/`clock` = monotonic access
tick; decay `heat>>=1` for all experts once per live-repin pass (`tier.h:56-58`,
called `colibri.c:7774`). The hysteresis means: a challenger must beat the victim by
25 % of the victim's score **plus 4 full frequency counts** (`4<<8`), or no swap.
Three consumers of the same two arrays: `repin_pick` (max 4 swaps/pass),
the pilot eviction guard, the qwen36 VRAM tick.

### 4.2 Eviction vs in-flight — the layered guards (all verified)

1. **Speculative pilot must not evict warm victims** (#441, narrowed by #497,
   `colibri.c:5800-5811`): victim protected only when genuinely warm (`heat>=2`) AND
   `vs + vs>>2 + (4u<<8) > cs`; otherwise the speculation is *dropped*
   (`g_pilot_drops++`) instead of thrashing. The narrowing matters: the un-narrowed
   test "dropped ~all speculations on a full cache (#490)".
2. **Reservation encoding** (`colibri.c:5813-5819`): slot marked `eid=-(eid+2)`
   (visible reservation ≠ any real eid and ≠ −1 free) and **`used=(uint64_t)-1`** —
   "in charge" sentinel, never an LRU victim until the real eid is published. The
   pread runs **outside** `g_pilot_mx`; only slot choice/publication is locked.
3. **Failed speculation must reset `used=0`** (`colibri.c:5828-5837`): leaving the
   sentinel made failed loads *permanently* un-evictable — "every failed speculation
   silently subtracted ~19 MB of cache" (fixed bug, worth remembering).
4. **`in_flight` refcount for GPU borrowers** (`colibri.c:385-392`): every slot
   passed to a group issue is `eslot_acquire`d; `eslot_lru_victim` and `rss_guard`
   skip `eslot_busy()` slots; underflow aborts.
5. **Victim scan order** (`eslot_lru_victim`, `colibri.c:399-411`): skip busy and
   reservations (`eid<-1`, which count as live); prefer a freed-slot-still-owning-slab
   (`eid==-1 && slab`) over LRU eviction; a slab-less empty slot only while the row's
   live count is under `ecap` (#1034).
6. **Free-under-lock discipline** (`rss_guard`, `colibri.c:7673-7683`): the `free`
   stays under `g_pilot_mx` — unlocking between `eid=-1` and `slab=NULL` let the pilot
   reuse the slot and pread *into the slab being freed* (use-after-free).
7. **VRAM side**: uploader frees a victim only when no group is in flight
   (`issue_open` + `cv_take`, §3.2); the LFRU tick skips `queued` slots and reverts
   `resident` if enqueue fails (`qwen36_tier.c:336-344`).

### 4.3 Pins

Startup pins are a *separate class* from the LRU: allocated from the ranked list,
NUMA-bound arenas (#419), never counted in `resident_bytes` together with LRU
("avoid double budgeting", `colibri.c:8943-8951`), never evicted by `rss_guard`
(#403 — it evicts only LRU ecache slots; pins, dense, KV untouched). Live re-pin is
the *only* mechanism that demotes a pin, 4 swaps/pass at turn boundaries. The RSS
guard itself: checked every 16 emitted tokens, tolerance 2 % + 300 MB, frees LRU
slots and ratchets `ecap` down ("niente ricrescita").

---

## 5. Windows O_DIRECT twin handles (`compat.h:296-311`) — exact + why two

```c
/* --- O_DIRECT -> FILE_FLAG_NO_BUFFERING ---
 * Apre il fd "gemello" senza cache del file system, come il twin O_DIRECT di
 * st.h su Linux e F_NOCACHE su macOS. Stesso contratto: offset, lunghezza e
 * buffer del chiamante devono essere allineati a 4K (gli slab expert usano
 * posix_memalign(4096) e il percorso DIRECT=1 del motore allinea gia' offset
 * e len); richieste non allineate falliscono con -1, mai dati corrotti.
 * Il fd si usa con la normale pread() (compat_pread -> ReadFile+OVERLAPPED). */
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

**Why two handles on the same file** (`st.h:160-161`): the buffered fd's page-cache
reads "on ext4-in-VHDX chokes at ~0.8 GB/s, O_DIRECT reaches 2.3+; measured". The
twin is opened **eagerly at index time** (`st.h:149-155`) so the later lock-free
`st_direct_fd()` lookup is thread-safe; the buffered fd stays open alongside for
scales/metadata (small reads where cache helps) and for `DIRECT=0` hosts. Twin is
per-mirror too (`mdfds`). Kimi measured 7.1 GB/s direct vs 2.9 buffered; V4 makes
DIRECT default; GLM keeps it opt-in (`DIRECT=1`, +65 % measured, tuning.md:23).

Companions that must ship with it (all re-verified):
`compat_fsize` (`compat.h:317-323`) — CRT `lseek(SEEK_END)` **returns −1 on
NO_BUFFERING fds (measured on UCRT)**, size must come from `GetFileSizeEx`;
`compat_pread` (`compat.h:158-193`) — positioned blocking `ReadFile` with OVERLAPPED
as a 64-bit offset carrier (no IOCP anywhere; QD = blocked worker threads);
`_FILE_OFFSET_BITS=64` guard (`compat.h:85-90`) — 32-bit off_t "silently wraps >4 GB
offsets into the first 4 GB → reads wrong weight bytes → silent token corruption";
alignment contract 4K offset/length/**destination** (slabs `posix_memalign(4096)`,
+8 KiB slack); `_aligned_free` discipline (plain `free` corrupts the CRT heap,
0xC0000374 — hit twice: `compat.h:214-231` and `colibri.c:3466-3469`).

---

## 6. Speculative depth — why depth 1 only

The auto-default (`colibri.c:10227-10233`) — exact comment:

```c
/* Auto depth = 1, not 3. A GLM-5.2 744B sweep (DRAFT=0/1/2/3, streaming and
 * fully-resident) showed single-token speculation is the only depth that pays:
 * acceptance ~85% at depth 1 vs ~44-62% at 2-3, and every extra draft token
 * both costs verify compute and (when streaming) faults experts that evict the
 * LRU working set. Depth 1 was the fastest MTP setting in every measured
 * configuration; 2-3 never beat it anywhere. DRAFT=n still forces any depth. */
g_draft = (m.has_mtp && (!g_cuda_enabled || cuda_mtp)) ? 1 : 0;
```

So their stated reasons: (1) acceptance collapse at depth 2–3 (85 % → 44–62 %), which
is a property of chained MTP draft heads — each extra link multiplies the
per-position miss probability; (2) verify compute scales with batch; (3) **the I/O
cost per draft**: at partial residency a deeper verify batch routes through more
experts, and in a MoE every added row *faults experts that evict the LRU working
set*. README.md:124 (w2): "MTP has also measured a 32% loss around 85% expert hit".
So yes — it is explicitly an IO-cost-per-draft argument in the streaming regime.

Two structural mitigations worth copying:
- **SPEC_PIN (#163)** (`colibri.c:541-543, 6904-6909`): while drafts are live, the
  kernel family is pinned so draft (S=1) and verify (S=1+g) forwards compute *the same
  function* — FP accumulation-order divergence between the S==1 fused-pair kernel and
  the S>=2 IDOT kernel collapsed acceptance. Under CUDA, MTP is off by default for
  exactly this reason (cold CPU experts diverge from GPU ones; opt-in acceptance
  30-50 %, `colibri.c:10218-10225`).
- **Adaptive pause** (`colibri.c:6910-6975`): 24-proposal sliding window; below 70 %
  MTP acceptance (or 50 % corpus) drafts pause for 256 tokens and re-arm, instead of
  a permanent latch.

Verify mechanics (`colibri.c:6985-7008`): one batched `step_all(m,batch,S=1+g,kv)`,
greedy-accept while argmax matches (rejection-sample `p(draft)` at temperature>0),
`mtp_absorb` replays verified tokens through the MTP head's own KV, `hlast` rewound to
the last **accepted** position (not end-of-batch), `repin_pass` at the loop's safe
point.

---

## 7. KV / recurrent state placement + prefill over tiers

### 7.1 State is a separate, non-evictable class

- **Weights (evictable, budgeted):** expert slabs in three stores — `pin[]`
  (startup, ranked, NUMA arenas), `ecache[]` (dynamic, `ecap` ceiling), transient
  `ws[]` promoted into ecache at end-of-block. Dense weights: separate budget class,
  VRAM-projected and subtracted before expert placement (`colibri.c:2064-2101,
  8967-8971`).
- **KV (never evicted, host-canonical):** CUDA device *shadow* `kv_dev_L/R` +
  watermark `kv_dev_valid`; host rows stay canonical, the shadow re-uploads only the
  tail `[v, upto)` (`kv_dev_sync`, `colibri.c:3699-3717`), invalidated on row
  rewrite/rewind or KV rebind. Vulkan: identical watermark scheme. Vulkan allocator
  priorities make the class explicit (`colibri.c:8526-8535`): **experts 0.4
  (evictable), dense 0.75, scratch/KV 1.0**, plus `COLI_VK_RESERVE_GB` (default 3)
  stopped on the expert fill so "dense weights + KV mirror + staging (measured
  ~1.7 GB at 4k ctx) always fit".
- **Recurrent state:** qwen36 DeltaNet `DN_rec/DN_conv` always host-resident, zeroed
  per request (`reset_recurrent`, `qwen36.c:2052-2059`), never paged; `ensure_kv`
  grow-only (`qwen36.c:2064-2090`); attention score scratch per-OMP-thread sized with
  the KV. MTP head has its own KV with a decode-only window from
  `kv_start[n_layers]` (`colibri.c:2323`); `mtp_absorb` keeps it consistent.
- The RSS guard (#403) touches only LRU ecache — pins, dense, KV are structurally safe.

### 7.2 Prefill over tiers

- **The batch union goes through the GPU groups too** (PIPE Inc.1b,
  `colibri.c:4903-4911`): `group_enabled = S<=64 || (g_cuda_pipe && S<=4096)` —
  "before this, 9343 experts in VRAM stayed UNUSED during prefill (measured: 81 s of
  expert-matmul all on CPU, GPU groups 21 ms total)". Lesson: the tier must serve
  prefill or the whole VRAM investment idles.
- **Attention prefill on device**: `attn_pipe_prefill` (`colibri.c:3724+`,
  `COLI_CUDA_PIPE=1`), full chain resident on the layer's home device, downloads only
  `out[S,D]` + new KV rows; any failure falls back to idempotent CPU.
- **EXPERT_BUDGET (I/O capping) is decode-only** (`S<=4`,
  `colibri.c:4790-4795`): during prefill the batch union is 30–100+ experts; capping
  to 4–8 "drops 80–90% of them, each with non-trivial gate weight → corrupted prefill
  hidden state → wrong KV cache → repetitive garbage decode" (#292). Plus a rescue
  rule: no position may end with zero routed experts (`colibri.c:4835-4847`).
- **KV-prefix reuse** (`kv_prefix.h`): turn N prefills only the new suffix against the
  persisted slot state; a chunked prefill's completion is tracked so an aborted chunk
  doesn't silently poison the prefix.

---

## 8. Mapping to Insignia 27B (dense, token-serial, single stream)

### 8.1 The degenerate-simplicity verdict — CONFIRMED, with two corrections

Colibri's scheduling is complex for exactly one reason: **MoE routing is
data-dependent**, so *which* weights the next token needs is unknown — hence router
lookahead (71.6 %/75.8 % recall), pilot rings, LFRU learning, EXPERT_BUDGET, per-expert
pin/ecache/ws stores. Insignia's decode is dense and single-stream: **every token
needs layers 0..63 in order, forever.** All of that machinery collapses:

- No pilot, no router prediction, no WILLNEED heuristics: the NVMe reader's schedule
  is a **fixed cyclic list** (NVMe-tier layers in model order, wrapping). It can and
  should run **flat out** — synthesis already shows the NVMe tier is the *binding*
  constraint (~8.06 GB of NVMe-tier weights per token at 6.8 GB/s = 1.19 s vs ~0.25 s
  of compute for the rest), so there is nothing to "schedule": the reader is the
  pace car and read-ahead exists only to absorb jitter, not to hide a faster device
  behind a slower one.
- No LFRU in decode: dense heat is uniform across layers; placement is a **static
  budget split decided at startup** (colibri's own precedent: startup placement is
  history-ranked *static*, LFRU only governs later dynamics — and here there is
  nothing to learn). Keep LFRU only if VRAM slots ever get oversubscribed/cycled.
- Colibri never streams layer-granular dense weights at all (dense = pread once,
  pinned; VRAM-dense projected and subtracted from the expert budget). So Insignia's
  layer streaming is new — but it is the *easy* special case of their problem.

**Correction 1 — placement must be interleaved, not blocked.** With contiguous tier
blocks (L layers, then M, then N), the consumer demands the whole 8+ GB NVMe block in
a ~210 ms burst (21×10 ms CPU compute) = ~38 GB/s instantaneous — no reader survives.
Place tiers **round-robin** (e.g. repeating L,M,N or a fixed ratio pattern matching
budgets) so NVMe-layer consumption is one 384 MB layer every ~56 ms on average —
exactly the reader's natural cadence. Colibri doesn't need this (19 MB expert
granularity is already smooth); a layer-granular engine does. Same logic for the
VRAM/RAM boundary if RAM→GPU streaming layers exist.
**Correction 2 — RAM accounting is tighter than synthesis assumed.** 23 RAM-pinned
(8.8 GB) + a ring large enough to hold 21 NVMe layers (8.1 GB) does not fit in
15.9 GB with OS + activations + embed. The ring is a *window*, not a second cache:
size it for jitter (a few slots), not for a full token; the reader re-reads every
NVMe layer every token by design. Budget example: M=14 pinned RAM (5.4 GB) + ring 8
slots (3.1 GB) + embed bf16 pinned 2.5 GB (or mmap it, §8.2) + host KV shadow +
activations ≈ 12–13 GB with ~2 GB OS headroom on a trimmed box. Every extra pinned
RAM layer steals ring slots; measure, don't assume.

### 8.2 embed / lm_head / MTP placement specials (dense-model wins colibri can't have)

- **lm_head 2.54 GB: VRAM-pinned, always** (synthesis: over PCIe it would be
  102 ms/token). It is the one "dense weight" colibri also always keeps resident.
- **embed 2.54 GB: does NOT need VRAM.** Decode reads exactly one 10 KB row per token.
  Gather the row on the CPU from a pinned/mmap'd RAM copy and ship the 5120-element
  activation vector with the token — frees 2.5 GB of VRAM (~6 extra layers) for the
  cost of nothing. (Prefill gathers T rows once — still trivial.) This is the
  "modify constant data directly / bake assumptions in" school: the embed is a
  lookup, not a matmul, at decode.
- **MTP 477 MB: VRAM-pinned**, runs on GPU between target steps; verify S=2 shares the
  main weights (bandwidth-bound ⇒ the second row is ~free).

### 8.3 Thread roles table (5600X: 6C/12T)

| Thread | Colibri analog | Job | Sync | Affinity (colibri pins none; we should) |
|---|---|---|---|---|
| T0 main/GPU | main thread | token loop; per-layer kernel submit; `cudaMemcpyAsync` H2D from pinned ring slot into VRAM double-buffer; event waits; sampling; MTP draft/verify; **sole writer of consumer cursor** | CUDA events; "1 sync-equivalent (event wait) per layer, upload(l+1) issued before compute(l) launches" | core 0–1 (with the GPU-interrupt affinity the driver already uses) |
| T1..T3 NVMe readers (3, range 2–4) | PIPE workers / V4 lanes | cyclic pread of next ring slot chunks (O_DIRECT twin fd, 4K-aligned 8–16 MB chunks, QD = thread count; one slot fully read ⇒ mark READY release) | per-slot state machine {EMPTY→READING→READY→BUSY→EMPTY} + SRWLOCK/condvar for producers (readers) — full ring = **block** (demand, never drop); chunk claim = atomic fetch_add | spread on cores 2–5; they block in ReadFile, share cores with the GEMV team (V4's "lanes do not need whole CPUs") |
| T4..T10 CPU GEMV team (7) | OMP pool (keep it off T0 and off readers) | RAM-tier + ring-fed layer compute (~10 ms/layer, DRAM-bound 40 GB/s beats PCIe 15.4 ms streaming — compute on CPU, don't stream to GPU) | persistent pool, spin-then-park (GOMP_SPINCOUNT/KMP_BLOCKTIME=200 lesson); one parallel region per layer | cores 2–11 minus reader load; `OMP_PROC_BIND=close` |
| (none) | pilot, uploader, LFRU ticker, vk2 worker | **not needed** — see §8.1 | — | — |

### 8.4 Queue types

1. **NVMe→RAM ring** (the only real queue): N slots × 384 MB pinned (`VirtualLock` via
   colibri's grow-working-set shim, `compat.h:195-212`), cyclic. Producers = readers
   (multi), consumer = T0. **No gen-tagged CAS needed** — the cyclic order is fixed,
   so a per-slot sequence number (slot `epoch` = monotonically increasing layer-cycle
   id) plus the state machine suffices; but if Insignia ever batches multiple
   sequences, lift `PipePool` verbatim (§1) — its 64-job/8-bit-index cap maps to
   "≤64 chunks per layer batch" cleanly.
2. **RAM→VRAM**: no queue, no uploader thread. Unlike qwen36 (whose staging copy
   exists for the XOR-0x88 format conversion), Insignia uploads **raw e4m3 bytes +
   bf16 scales unchanged** — T0 issues `cudaMemcpyAsync` from the pinned slot into the
   layer's VRAM double-buffer, stream-ordered, with an event. VRAM slots for streamed
   layers are double-buffered (compute on buffer a%2, upload into (a+1)%2).
3. **CPU↔GPU handoff**: none — token-serial means layers alternate engines but never
   split a layer. Colibri's `done_j/mask` pattern applies only if a layer is ever
   split (e.g. fp32 out_proj on CPU while attention on GPU — don't).

### 8.5 Prefetch distance per tier

- **VRAM-pinned layers: distance 0** (no I/O).
- **RAM→VRAM streamed (if any): 1 layer** — upload l+1 while l computes; this is
  colibri's "1 sync per device" generalized to uploads: 15.4 ms transfer vs
  ≥10 ms compute ⇒ double-buffer suffices, nothing deeper helps (PCIe is
  half-duplex-ish shared with nothing else here).
- **NVMe→RAM ring: distance = K slots of jitter, not a fixed time.** Reader runs
  flat out cyclically; K sized so reader never laps the consumer (K≥2: one slot
  being read + one ready) and consumer never laps the reader: with round-robin
  placement the average consumption cadence of NVMe layers (1 per ~56 ms) equals the
  reader's production cadence (384 MB per 56 ms) — the system is a just-in-time
  treadmill, so K=3–4 (1.2–1.6 GB) absorbs scheduler/disk jitter and the MTP verify
  burst (verify re-reads nothing, but the *next* token's demand resumes immediately).
  Fill the ring at startup (warmstart lesson: cold first-touch mid-decode cost
  139 ms/token in a measured colibri run) and keep it full across turns.
- **Windows prefetch analog**: not used on the O_DIRECT path (colibri skips WILLNEED
  under DIRECT — it warms a cache the direct reads bypass). Keep a buffered twin fd
  for safetensors headers/small scales only.

### 8.6 KV / state placement for tiered decode

Copy colibri's class separation verbatim:
- fp32 DeltaNet state 48 layers × 3.15 MB = 151 MB → **VRAM, always** (it is both
  read and written per token; 302 MB r/w at 504 GB/s = 0.6 ms — and CPU-computed
  layers need it too, so VRAM-resident with a device→host shadow only when a linear
  layer runs on CPU; simpler: keep canonical on the device that owns the layer's
  compute — but layers alternate CPU/GPU, so: **host-canonical + device shadow +
  watermark tail sync** is exactly `kv_dev_sync` (`colibri.c:3699-3717`) and is the
  right port; the state slice per layer is 3.15 MB, so the tail sync after a CPU
  layer is 3.15 MB H2D ≈ 0.13 ms — acceptable).
- Full-attn KV (16 layers × 4 KV × 256 × 2 × bf16 × ctx): ~2.1 GB at 32k ctx →
  VRAM if it fits after weights; otherwise host-canonical + shadow watermark
  (append-only tail upload — never re-upload the whole cache). **Never evictable,
  never budgeted against weights** — reserve it up front (the Vulkan 1.0-priority /
  RESERVE_GB lesson: an unreserved KV silently starves the weight tier as ctx grows).
- MTP head KV: own window, `mtp_absorb`-style replay on accept; on reject, roll back
  the watermark (Insignia bug list already flags the KV-not-restored-on-reject
  hazard — colibri's rewind/invalidation points, `colibri.c:4074-4075`, are the fix
  pattern).
- Prefill over tiers: **layer-major weight-stationary** (FlexGen-style, per
  synthesis): stream each layer's weights once through the same ring/double-buffer
  machinery, compute all chunks for that layer, checkpoint activations in RAM; the
  reader executes the same cyclic schedule, just consumed as one 0→63 sweep. Colibri's
  Inc.1b lesson applies: the VRAM tier must serve prefill (group GEMMs at T=64–256),
  or 6+ GB of VRAM idles during the most expensive pass. And the #292 lesson: never
  cap/skip weights during prefill to save I/O — a corrupted prefill hidden state
  poisons the whole KV.

### 8.7 Recommended Insignia scheduler skeleton (~100 lines)

```
// ---- placement (startup, static; colibri: budget-split + greedy, no runtime learning)
const int  NL        = 64;                    // layers 0..63
const u64  LAYER_B   = 384ull<<20;            // 383.87 MB rounded
int  dev_of[65];                             // 0=VRAM-pinned,1=RAM-pinned,2=NVMe-fed, MTP=64→0
u64  vram = free_vram() - LM_HEAD_B - MTP_B - KV_RESERVE_B - ACT_B;   // lm_head+MTP pinned
int  L = vram / LAYER_B;                      // e.g. 14–16 (embed stays in RAM!)
int  M = (ram_total - OS_RESERVE - EMBED_B - KV_HOST - ACT_HOST - RING_B) / LAYER_B; // e.g. 14
int  N = NL - L - M;                          // e.g. 34 — NVMe-fed, computed on CPU
// interleave round-robin: for l in 0..63: tier[l] = pattern[l % 3] adjusted to counts
// (guarantees NVMe layers are spaced ~every 3rd layer — smooths demand to reader rate)

// ---- ring (the only queue; demand-loads block, nothing is ever dropped)
enum { EMPTY, READING, READY, BUSY };         // per-slot state
struct Slot { u8* buf; _Atomic u32 st, epoch; int layer; };   // buf = 4K-aligned pinned
Slot ring[RING_K];                            // RING_K = 3..8 (jitter budget, §8.5)
u32   ring_head;                              // next slot the reader fills (readers own)
u32   ring_tail;                              // next slot the consumer needs (T0 only)

// ---- reader threads (3): fixed cyclic schedule, flat out
void reader(int id) {
  for (;;) {
    Slot* s = lock_and_pick_next_empty();     // ring_head slot; wait (condvar) if full
    s->layer = nvme_order[ring_head % N];     // cyclic layer list, model order
    s->epoch = ++global_epoch;
    s->st = READING;                          // visible reservation (colibri -(eid+2))
    // chunked parallel pread: each reader takes 8-16MB chunks via atomic fetch_add
    for (chunk = fetch_add(&s->chunk_cursor, CHUNK); chunk < LAYER_B; ...)
      pread(direct_fd(shard_of(s->layer)), s->buf+chunk, CHUNK, off(chunk));  // O_DIRECT twin
    if (fetch_add(&s->chunks_left,-1)==0 on last) { __atomic_store_n(&s->st, READY, __ATOMIC_RELEASE); }
  }
}

// ---- decode (T0): token-serial; 1 event-wait per layer; upload(l+1) ‖ compute(l)
void decode_token(int tok) {
  embed_row_cpu(embed_ram, tok, h_host);      // embed lives in RAM (§8.2)
  cudaMemcpyAsync(h_dev, h_host, 5120*4, H2D, stream);
  for (int l = 0; l < NL; ++l) {
    Weights* w = acquire_layer(l);            // below
    if (tier[l]==0) {                         // VRAM-pinned: pointer is static
      if (dbuf[l&1].busy) cudaEventWait(dbuf[l&1].done);   // double-buffer guard
      launch_layer(stream, l, w->dev);
    } else {                                  // RAM or ring slot: compute on CPU team
      cpu_layer(w->host_or_slot, h_host);     // OMP team inside; ~10ms; readers keep running
      if (state_owner(l) != CPU) upload_state_tail(l);     // kv_dev_sync pattern
    }
    issue_next_upload(l);                     // for l+1 streamed-layer: cudaMemcpyAsync into dbuf
    cudaEventRecord(layer_done[l], stream);
  }
  final_norm(); logits = lm_head_pinned();    // lm_head never leaves VRAM
}

Weights* acquire_layer(int l) {               // demand path — blocks, never drops
  if (tier[l] != 2) return resident(l);       // pinned RAM or VRAM
  Slot* s = &ring[ring_tail % RING_K];
  u32 want = epoch_of(l);
  while (__atomic_load_n(&s->st, __ATOMIC_ACQUIRE) != READY || s->epoch != want)
    _mm_pause();                              // spin; ~5µs wake vs 10ms layer ⇒ spin is fine (#159)
  s->st = BUSY;
  return (Weights*){ .host_or_slot = s->buf };
}
void release_layer(int l) {                   // after the layer's kernels/threads are done
  if (tier[l] != 2) return;
  Slot* s = &ring[ring_tail % RING_K];
  __atomic_store_n(&s->st, EMPTY, __ATOMIC_RELEASE);   // reader may refill
  ring_tail++;                                // T0 sole writer (the PipePool gen discipline)
}

// ---- MTP: depth-1 only (see 8.8); verify shares weights; SPEC_PIN rule:
// draft (T=1) and verify (T=2) MUST use the same kernel family/accumulation order.
void spec_step() {
  d  = mtp_draft_vram(h_dev);                 // MTP layer pinned VRAM, ~4ms
  run verify forward on [tok, d] batched T=2; // same weights, one pass, bandwidth-bound
  if (argmax(row0)==d) { emit d; k=1; } else { kv_rollback(1); k=0; }  // absorb/rewind
}
```

Startup sequence: open twin fds + index shards → static placement → **fill ring**
(readers run immediately) → pin lm_head/MTP → build KV/state reserves → prefill sweep
(layer-major through the same acquire/release, weights-once).

### 8.8 Accept-depth recommendation: **depth 1 (verify T=2), never deeper**

Direct colibri evidence (`colibri.c:10227-10233`): 85 % acceptance at depth 1 vs
44–62 % at 2–3; "depth 1 was the fastest MTP setting in every measured configuration;
2–3 never beat it anywhere". The Insignia translation:

- Bandwidth-bound verify makes T=2 nearly free (weights read once regardless of rows)
  ⇒ at p≈0.6–0.85 tokens/step ≈ 1.6–1.85 → matches synthesis's ×1.6 estimate.
- Depth ≥2 needs chained drafts (acceptance compounds multiplicatively — the 44–62 %
  measurement), extra MTP forwards, and a T≥3 verify whose KV-rollback window grows —
  and Insignia *already* has the reject-KV hazard flagged in the bug list.
- The colibri I/O argument inverts for us but still bites: deeper drafts don't read
  more *weight* bytes (dense), but every rejected draft wastes a full 1.2–2 s NVMe
  token cycle; at 60 % per-step acceptance the expected waste of a second draft level
  (≈0.6·0.55 ≈ 0.33 marginal hit rate) never pays against a 2× longer stall on miss.
- **SPEC_PIN (#163) is mandatory**: draft T=1 and verify T=2 must run the *same*
  kernel family (batched decode kernels at T∈{1,2}, one code path, no S==1 special
  case), or accumulation-order divergence collapses acceptance — colibri measured
  this as a real bug class twice (#8/#163, and the CUDA-MTP 30–50 % acceptance note).
- Steal the **adaptive pause**: 24-proposal window, pause 256 tokens below ~70 %
  acceptance, re-arm — costs nothing and rescues a degenerate prompt.

---

## 9. Port checklist (colibri → Insignia, ranked)

1. **Copy verbatim**: `compat_open_direct` + eager twin open + `compat_fsize` +
   `_FILE_OFFSET_BITS=64` + 4K alignment contract + `_aligned_free` discipline
   (§5); `compat_mlock` working-set growth (§3.3/w2); LFRU score + `25%+4<<8`
   hysteresis *if* VRAM cycling is ever added (§4.1).
2. **Copy the pattern, not the code**: per-slot READY flags with release/acquire and
   T0 as sole cursor writer (PipePool discipline, §1); host-canonical KV/state +
   device shadow + watermark tail sync (§7.1); staging rounds min(4 GB, budget/8)
   for any bulk VRAM fill from NVMe (§3.1); warmstart-everything-before-token-1 (§3.2).
3. **Do NOT port**: pilot/router lookahead, LFRU learning in decode, mirror
   striping (single SSD), io_uring (Windows), EXPERT_BUDGET (dense has no routing),
   multi-sequence mux. The cyclic reader + ring + double-buffered VRAM replaces all
   of it (§8.1).
4. **New vs colibri** (they never needed it): round-robin tier interleaving to smooth
  NVMe demand (§8.1 correction 1); embed-in-RAM row gather (§8.2); layer-granular
  streaming itself.
5. **Bugs to pre-empt** (all fixed late in colibri — learn from them): failed
   speculation/reservation must reset the slot to evictable (`used=0`,
   colibri.c:5828-5837); free-under-lock for recycled slabs (colibri.c:7673-7683);
   no victim free while a group is in flight (`issue_open`); CRT lseek on
   NO_BUFFERING fds; draft/verify kernel-family divergence (#163); prefill must not
   cap weights (#292).
