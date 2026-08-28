# Audit: colibri storage/I/O machinery (for Insignia's 25.65 GB FP8 heterogeneous engine)

Source: `E:\coding\Insignia\colibri\` (read-only reference clone). All paths relative to
`E:\coding\Insignia\colibri\`. Line numbers verified against this checkout.
Scope: storage + I/O only (SPMC load pools, O_DIRECT twins, staging slabs/rings, async read
issuance, prefetch heuristics, fadvise analogs, mmap-vs-read policy). Scheduling/tiering
(LRU/LFRU/REPIN/heat) is covered by the second agent and only referenced where it touches I/O.

Colibri is *the* existence proof for the Insignia mission: it runs a **370 GB** int4 GLM-5.2
MoE from disk on 16 GB-RAM boxes, and a DeepSeek-V4 FP8/FP4 checkpoint through the same
`st.h` layer. Everything below is read-path; the model dir is never written.

---

## 1. Layer map — who owns what

| Concern | File | Key symbols |
|---|---|---|
| Shard index + tensor pread + fadvise + O_DIRECT twins + mirrors | `c/st.h` | `shards`, `st_pread_full`, `st_prefetch`, `st_direct_fd`, `st_mirror_add` |
| Platform shims (Windows/macOS) | `c/compat.h` | `compat_pread`, `compat_fadvise`, `compat_open_direct`, `compat_mlock`, `compat_ro_map` |
| GLM engine: PIPE SPMC pool, PILOT ring, io_uring batches, slabs | `c/colibri.c` | `PipePool`, `pilot_q`, `UringBatch`, `expert_load_impl` |
| Minimal io_uring (Linux only) | `c/uring.h` | `ColiUring`, `coli_uring_prep_read` |
| Kimi K3 engine: its own loader pool + O_DIRECT windows | `c/kimi_k3.c` | `g_lp`, `expert_read` |
| DeepSeek-V4 store: direct-window reads + N loader lanes | `c/deepseek_v4.c` | `v4_read_direct_window`, `DualExpertLoaderPool` |
| Qwen3.6 CUDA upload ring (staging) | `c/qwen36_tier.c` | `QT_QCAP`, `uploader`, `stage` |
| Disk microbench | `c/iobench.c` | 19 MB-block random-read benchmark |

---

## 2. The shard layer (`st.h`) — pread-first, mmap opt-in

Design decision stated at `st.h:1-6`: tensors are read with **pread, not mmap**, plus
`posix_fadvise(DONTNEED)` after streaming experts, so peak RSS stays "dense + cache" instead
of the whole model (the "mmap RSS bug"). mmap exists but is an opt-in mode (§7).

### 2.1 Index struct + twin fds

`st.h:42-73`:
```c
typedef struct {
    st_tensor *t;  int n, cap;
    int        fds[512];      /* buffered fds */
    int        dfds[512];     /* O_DIRECT twins (opened eagerly): -2 = not yet tried */
    char      *paths[512];  int64_t sizes[512];
    int        nfd;
#define ST_MAX_MIR 4          /* extra read replicas beyond the primary (multi-SSD) */
    int        mfds[ST_MAX_MIR][512];   /* mirror buffered fds, -1 = absent */
    int        mdfds[ST_MAX_MIR][512];  /* mirror O_DIRECT twins */
    ...
} shards;
#define ST_MAX_SHARDS 512
```
Constants: **512 shards max**, **4 mirrors max** (5 replicas incl. primary). The O_DIRECT
twin is opened **eagerly at index time** so the later `st_direct_fd()` lookup is
thread-safe with zero locking (`st.h:149-155`, comment "eager: lookup poi thread-safe"):

```c
#ifdef O_DIRECT
    S->dfds[S->nfd] = open(path, COMPAT_O_RDONLY | O_DIRECT);
#elif defined(__APPLE__) || defined(_WIN32)
    S->dfds[S->nfd] = compat_open_direct(path);   /* macOS: F_NOCACHE; Windows: NO_BUFFERING */
#else
    S->dfds[S->nfd] = -1;
#endif
```

**Why a twin at all** (`st.h:160-161`): *"bypasses the page cache: the buffered read on
ext4-in-VHDX chokes at ~0.8 GB/s, O_DIRECT reaches 2.3+; measured"*. The buffered fd stays
open alongside for scales/metadata and for the DIRECT=0 default on hosts where buffered wins.

### 2.2 Full pread loop

`st.h:258-281`: `ST_PREAD_CHUNK = 1<<30` (1 GiB) because one pread caps at ~2^31 bytes;
EINTR retry; honest short-read errors (exit(1), like every st.h reader). On Windows `pread`
is `#define`d to `compat_pread` (§3.2).

### 2.3 Prefetch primitives

`st.h:738-755`:
```c
static void st_prefetch(shards *S, const char *name) {
    st_tensor *t = st_find(S, name);
    if (t) posix_fadvise(t->fd, t->off, t->nbytes, POSIX_FADV_WILLNEED);
}
```
`st_prefetch_rep` targets the *same replica fd* the later demand pread will hit — required
because replica routing is deterministic (§6.3) and a WILLNEED on the wrong drive warms a
cache nobody reads.

### 2.4 DONTNEED discipline

Every streaming read path takes a `drop` flag: `st_read_f32` (`st.h:790`), `st_read_raw`
(`st.h:891`), slices (`st.h:958`) all end with
`if (drop) posix_fadvise(fd, off, nbytes, POSIX_FADV_DONTNEED);`.
GLM's `g_drop` **defaults to 0** (`colibri.c:1176-1178`): *"leave them in page-cache
(buff/cache, NOT RSS) as a free L2 — exploits MoE routing imbalance (a few 'hot' experts
reused)"*. So DONTNEED is per-host policy, not law.

### 2.5 Mirrors + split (multi-SSD)

- `st_mirror_add` (`st.h:200-243`): a mirror file is accepted **only if size AND safetensors
  header are byte-identical** to the primary (data_offsets then match by construction).
  Partial mirrors allowed (a smaller SSD holding only expert shards). Mirrors never written.
- `st_init_multi(..., extra_dirs)` (`st.h:466+`): `COLI_MODEL_DIRS` **split** mode — each
  shard lives on exactly ONE drive (dedup by basename, first-listed wins), concurrent expert
  loads parallelize across drives. This is the "no duplication" complement to mirrors.
- Startup bandwidth probe `mirror_probe_bw` (`colibri.c:8716-8742`): **8 reads x 19 MiB**
  at deterministic spread offsets on the largest shard, O_DIRECT twin preferred ("buffered
  would measure the page cache"), ~150 MB/drive, few hundred ms. Feeds the hash-cut split.

---

## 3. Windows compat layer (`compat.h`) — the part Insignia copies verbatim

### 3.1 O_DIRECT twin = FILE_FLAG_NO_BUFFERING (`compat.h:296-311`) — exact code

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
Notes: shares READ|WRITE|DELETE (files stay openable by other processes); wrapped back into
a CRT fd so the whole engine keeps calling `pread()`. Contract = Linux O_DIRECT: 4K-aligned
offset, length, *and destination buffer* (slabs are `posix_memalign(4096)`).

Companion `compat_fsize` (`compat.h:313-323`): CRT `lseek(SEEK_END)` **returns -1 on
NO_BUFFERING fds (measured on UCRT)** — size must come from `GetFileSizeEx`.

Also `compat.h:85-90`: compile-time guard `_FILE_OFFSET_BITS=64` is *mandatory* — "model is
370 GB... 32-bit off_t silently wraps >4 GB offsets into the first 4 GB → reads wrong weight
bytes → silent token corruption."

### 3.2 pread → ReadFile + OVERLAPPED (`compat.h:158-193`)

There is **no IOCP / ReadFileEx / Registered I/O anywhere on Windows**. Async-ness is
threads, not completion ports. `compat_pread` is a *synchronous, thread-safe, positioned*
read on the raw OS handle:

```c
static inline ssize_t compat_pread(int fd, void *buf, size_t n, off_t off){
    intptr_t osfh = _get_osfhandle(fd);
    ...
    while(total < n){
        DWORD chunk32 = (chunk > 0x7FFFFFFF) ? 0x7FFFFFFF : (DWORD)chunk;
        OVERLAPPED ov = {0};
        ov.Offset     = (DWORD)( (off + (off_t)total)        & 0xFFFFFFFFULL);
        ov.OffsetHigh = (DWORD)(((off + (off_t)total) >> 32) & 0xFFFFFFFFULL);
        DWORD rd = 0;
        if(!ReadFile(h, (char*)buf + total, chunk32, &rd, &ov)){ ... }
```
The OVERLAPPED struct is used purely as a **64-bit offset carrier** (the handle was not
opened OVERLAPPED, so ReadFile blocks). Comment (`compat.h:68-72`): NEVER `_read/_lseeki64`
— racy AND corrupts 0x0A bytes via CRT text-mode translation. A `__thread DWORD
compat_pread_lasterr` preserves the real GetLastError() for field diagnosis (#307).

Queue depth on Windows therefore = number of worker threads blocked in `compat_pread`
(default `PIPE_WORKERS=8`, §4), not an io_uring/IOCP depth.

### 3.3 MADV_WILLNEED analog (`compat.h:114-156`)

`posix_fadvise(WILLNEED)` on Windows = **fire-and-forget readahead**: allocate a scratch
buffer (cap **64 MiB** per call), issue one blocking overlapped ReadFile into it, free it:

```c
static inline int compat_fadvise(int fd, off_t off, off_t len, int advice){
    if(advice!=POSIX_FADV_WILLNEED || len<=0) return 0;
    ...
    size_t rdlen = (len>(off_t)(64*1024*1024)) ? (size_t)(64*1024*1024) : (size_t)len;
    char *buf=(char*)_aligned_malloc(rdlen, 4096);
    ...
    OVERLAPPED ov={0};  ov.Offset=...; ov.OffsetHigh=...;
    /* ... ReadFile still accepts lpOverlapped (it carries the 64-bit offset) and blocks
     * until the read completes — but crucially it populates the standby page cache for
     * this region, so the later synchronous pread on the same offsets faults from RAM
     * not disk. */
    DWORD got=0;  ReadFile(h, buf, (DWORD)rdlen, &got, &ov);
    _aligned_free(buf);  return 0;
}
```
Two policy notes baked into comments:
- Called **only from the dedicated PILOT I/O thread / next-block readahead in moe(),
  NEVER inline on the hot path** — "inline fadvise submit measured ~0.5 ms x 169k calls =
  +92 s / 48 tok" (compat.h:118-121).
- `DONTNEED` is a **no-op**: "Windows' standby-list trimming self-regulates under pressure,
  and on a low-RAM host keeping the pages is what we want for reuse" (compat.h:122-125).
macOS analog (`compat.h:28-51`): WILLNEED→`fcntl(F_RDADVISE)` (capped 2^31-1), DONTNEED→no-op.

### 3.4 mlock → VirtualLock (`compat.h:195-212`)

```c
static inline int compat_mlock(const void *addr, size_t len){
    HANDLE p = GetCurrentProcess();
    SIZE_T mn = 0, mx = 0;
    if(GetProcessWorkingSetSize(p, &mn, &mx)){
        SIZE_T need = len + (SIZE_T)(1u<<20);
        SetProcessWorkingSetSize(p, mn + need, mx + need);   /* best effort */
    }
    return VirtualLock((LPVOID)addr, len) ? 0 : -1;
}
```
VirtualLock fails beyond the *minimum* working set (default a few hundred KB), so the shim
first grows min+max by len+1 MiB. Best-effort like mlock. Note: per-call growth means N
separate pin_wire locks do N working-set grows — fine for a one-time pin pass.

### 3.5 Read-only mapping primitive (`compat.h:398-467`)

`compat_map_readonly`: `CreateFileMappingA(PAGE_READONLY)` + `MapViewOfFile(FILE_MAP_READ)`
with **allocation-granularity** alignment (dwAllocationGranularity, fallback 64 KiB — NOT
the 4 KiB page size, a classic Windows trap); POSIX side is plain `mmap(PROT_READ,
MAP_SHARED)` page-aligned. Used by `st_map_raw` (`st.h:917-935`) and K3_MMAP (§7.2).
Comment: it "does not prefetch or lock pages — mapping changes ownership, not the working
set."

### 3.6 Alignment / allocator discipline

`compat.h:214-231`: `posix_memalign` → `_aligned_malloc`, freed via `compat_aligned_free`
(= `_aligned_free`); plain `free()` on such a block corrupts the CRT heap (0xC0000374) —
colibri hit this in production. `O_BINARY` everywhere (0x0A corruption defense, §103-112).

---

## 4. The PIPE SPMC load pool — the `(gen<<8)|idx` CAS pattern (GLM)

### 4.1 The pool

`colibri.c:3294-3354`. Header comment (abridged):
*"A small persistent pool of I/O worker pthreads runs the misses' pread (expert_load) into
distinct ws[] slabs and sets a per-slot `ready` flag; the MAIN thread walks the block's
experts in order, waiting on ready[q] only for the expert it needs right now, and does all
matmul on itself."*

```c
typedef struct {
    _Atomic uint64_t cur;      /* (gen<<8)|index; gen main-only, index 0..njobs (<=64) */
    _Atomic int njobs;         /* current batch job count */
    _Atomic int eids[64];      /* current batch expert ids */
    _Atomic int layer;         /* current batch layer */
    _Atomic int ready[64];     /* per-slot load-done flag */
    pthread_mutex_t mx; pthread_cond_t cv;     /* ONLY for parking/waking idle workers */
    pthread_cond_t cv_done;                    /* COLI_PIPE_BLOCK: signals ready[] transitions */
    Model *m;
    pthread_t th[16]; int nw; int started;
} PipePool;
```

**Exact CAS claim** (`colibri.c:3356-3385`, `pipe_worker`):
```c
for(;;){
    uint64_t c=atomic_load_explicit(&p->cur,memory_order_acquire);
    seen=c>>8;
    uint32_t i=(uint32_t)(c & 0xFF);
    if(i >= (uint32_t)atomic_load_explicit(&p->njobs,memory_order_relaxed))
        break;                                /* batch drained -> re-park */
    if(atomic_compare_exchange_weak_explicit(&p->cur,&c,c+1,
            memory_order_acq_rel,memory_order_relaxed)){
        int L  =atomic_load_explicit(&p->layer,memory_order_relaxed);
        int eid=atomic_load_explicit(&p->eids[i],memory_order_relaxed); /* AFTER winning CAS */
        expert_load(p->m,L,eid,&p->m->ws[i],1,1);
        atomic_store_explicit(&p->ready[i],1,memory_order_release);
        ...
    }
    /* CAS failed -> another worker advanced index (or gen advanced): re-loop */
}
```

**Generation protocol** (`colibri.c:3303-3319` comment, the load-bearing invariants):
- main thread is the **sole writer of gen** (monotonic bump ⇒ no ABA);
- a worker reads `eids[i]/layer` **only AFTER its winning CAS**, and the CAS's comparand
  carries the generation — a straggler preempted anywhere can never grab a wrong-generation
  job; its first act is a gen-checked CAS;
- dispatch writes all batch state RELAXED, then **RELEASE-stores `cur`** to publish:
```c
uint64_t g=(atomic_load_explicit(&g_pp.cur,memory_order_relaxed)>>8)+1;
atomic_store_explicit(&g_pp.cur,(g<<8),memory_order_release);   /* PUBLISH */  // colibri.c:3421-3422
pthread_mutex_lock(&g_pp.mx); pthread_cond_broadcast(&g_pp.cv); pthread_mutex_unlock(&g_pp.mx);
```
- `ready[]` reset happens BEFORE the publish store (`colibri.c:3420`);
- the per-expert `pipe_wait(ready[q])` in the matmul loop makes every grabbed job complete
  before the block ends — "no grab outlives its generation", which is why the old `active`
  counter AND the end-of-block drain barrier were deleted;
- mutex/condvar exist ONLY to park/wake idle workers, never for correctness.

8 bits of index ⇒ **max 64 jobs per batch** (matches `eids[64]`/`ws[64]`); gen in the upper
56 bits. Weak CAS + retry; torn payload reads are impossible because payload is read only
after CAS success, and workers that lose re-read `cur`.

### 4.2 Waiting

`pipe_wait` (`colibri.c:3435-3454`): default is a **spin on `sched_yield()`** acquiring
`ready[q]` (byte-identical legacy behavior); `COLI_PIPE_BLOCK=1` switches to mutex+condvar
with the classic double-check — *"the condvar wake costs ~5 us against reads that cost
0.5-3 ms (#159)"* (`colibri.c:3327-3332`). URING mode has no flag to peek: `pipe_ready`
returns 0 and the wait does the completion work (`colibri.c:3429-3441`).

### 4.3 Knobs & defaults (`colibri.c:9964-9977`)

| Env | Default | Meaning |
|---|---|---|
| `PIPE` | **1 on `_WIN32`, 0 elsewhere** | "default ON [Windows]: overlap expert load ‖ matmul (byte-identical; reorders I/O)" |
| `PIPE_WORKERS` | 8 (clamp [1,16], `th[16]`) | I/O worker threads = disk queue depth |
| `COLI_PIPE_BLOCK` | 0 (spin) | blocking pipe_wait |
| `PIPE_WORKERS>0` with PIPE unset | implies PIPE=1 | sizing the pool declares intent (a whole campaign had it set with the pipe off) |

Call site (`colibri.c:5039-5050`): misses of a 64-expert block are dispatched as one batch;
PIPE=0 runs the original `#pragma omp parallel for schedule(dynamic,1)` blocking parallel
load. Docs (`docs/tuning.md:22`): PIPE overlaps pread with matmul, **−18% disk service**.

---

## 5. The PILOT ring — second SPMC (prefetch lane)

### 5.1 Ring

`colibri.c:5759-5772`:
```c
/* I WILLNEED partono da un THREAD I/O dedicato: con la coda disco satura la submit
 * del fadvise BLOCCA (~0.5ms x 169k chiamate = +92s/48 token, misurato) — inline
 * il pilota costava piu' di quanto rendesse. Ring lock-free 1P/1C; pieno = scarta
 * (un hint perso non e' un errore). */
static struct { _Atomic int l,e; } pilot_q[4096];
static volatile unsigned pilot_w=0, pilot_r=0;
```
**4096 slots**, single producer (main thread = `pilot_w`), N consumers. Payload is `_Atomic`
*pair* {layer, expert} because "la claim SPMC legge speculativamente prima della CAS e
scarta se perde -> senza _Atomic sarebbe una data race C11 col produttore. int e' sempre
lock-free: stessa size/align" (`colibri.c:5771`).

Producer enqueue (`colibri.c:6110-6113`, and same in couple_prefetch 6045-6050):
```c
unsigned w=__atomic_load_n(&pilot_w,__ATOMIC_RELAXED);
if(w-__atomic_load_n(&pilot_r,__ATOMIC_ACQUIRE)<4096){
    atomic_store_explicit(&pilot_q[w&4095].l,lnext,memory_order_relaxed);
    atomic_store_explicit(&pilot_q[w&4095].e,best,memory_order_relaxed);
    __atomic_store_n(&pilot_w,w+1,__ATOMIC_RELEASE);
}
```
Full ring = drop the hint silently.

### 5.2 SPMC claim (`colibri.c:5922-5942`) — exact code

```c
/* SPMC ring claim: each of N pilot workers grabs a UNIQUE ring index via CAS, never
 * advancing past pilot_w. ... The producer stays single — this only splits the consumer. */
static int pilot_ring_claim(int *out_l, int *out_e){
    for(;;){
        unsigned r=__atomic_load_n(&pilot_r,__ATOMIC_ACQUIRE);
        unsigned w=__atomic_load_n(&pilot_w,__ATOMIC_ACQUIRE);
        if(r==w) return 0;                              /* empty */
        /* Read the payload BEFORE committing the claim. While pilot_r==r the slot at
         * index r cannot be overwritten (the producer's w-r<4096 guard keeps pilot_w
         * below r+4096). If the CAS succeeds, pilot_r was r the whole time -> the read
         * is valid. If it fails, another worker advanced pilot_r; we discard and retry. */
        int l=atomic_load_explicit(&pilot_q[r&4095].l,memory_order_relaxed);
        int e=atomic_load_explicit(&pilot_q[r&4095].e,memory_order_relaxed);
        if(__atomic_compare_exchange_n(&pilot_r,&r,r+1,/*weak=*/1,
                                       __ATOMIC_ACQ_REL,__ATOMIC_ACQUIRE)){
            *out_l=l; *out_e=e; return 1;
        }
    }
}
```
Racy-consumer safety argument is in the comment: payload is read *before* the CAS; the
producer can't wrap onto slot `r` while `w-r<4096`; a losing worker's torn read is thrown
away. Proven by `c/tests/test_pilot_ring.c` (300k items, 8 threads, exactly-once claims,
no tears).

### 5.3 Workers & modes

`pilot_worker` (`colibri.c:5943-5961`): idle = `usleep(200)`; three modes:
1. **URING+PILOT_REAL** (Linux): single worker drains a whole batch via
   `pilot_uring_batch` ("URING drains a whole batch itself → single worker (nw==1)");
   hard invariant: *"con URING attivo nw DEVE restare 1 — pilot_uring_batch e' un
   consumatore SINGOLO (avanza pilot_r con uno store semplice); due drainer URING
   concorrenti corromperebbero pilot_r"* (`colibri.c:5972-5976`).
2. **PILOT_REAL** (blocking): `pilot_realload` — REAL pread into the future layer's ecache,
   QD=N with N = `PILOT_WORKERS` workers.
3. **PILOT hint-only**: `expert_prefetch` = fadvise WILLNEED only.

`pilot_spawn` (`colibri.c:5965-5979`): `nw = (g_pilot_real && !uring) ? g_pilot_nw : 1`,
clamp [1,16]; threads spawned once, detached.

### 5.4 pilot_realload slot reservation (how staging slots cycle)

`colibri.c:5777-5844`: under `g_pilot_mx`: skip if `layer <= g_cur_moe_layer` (main already
past it — that's the safety invariant: pilot only writes ecache[L+1] while matmul reads
ecache[L]); skip if already pinned/resident/reserved; pick slot (grow if `nn<ecap`, else
`eslot_lru_victim` with the #441/#497 LFRU eviction guard: a victim with heat>=2 is
protected unless the speculation beats it by 25%+4-freq); then **mark reservation visible
before unlocking**:
```c
dst->eid = -(eid+2);              /* visible reservation; dedup + victim-scan see it */
dst->used = (uint64_t)-1;         /* "in charge" sentinel: never an LRU victim until published */
g_pilot_inflight[layer]++;
pthread_mutex_unlock(&g_pilot_mx);
int rc = expert_load(m,layer,eid,dst,0,0);   /* REAL pread — OUTSIDE the lock */
```
On success: `dst->eid=eid` (set by expert_load), `dst->used=++eclock`; on failure the slot
must be reset to `eid=-1, used=0` — leaving `used=(uint64_t)-1` made failed speculations
*permanently* un-evictable (~19 MB cache lost per failure, fixed). URING variant
(`pilot_uring_batch`, `colibri.c:5846-5920`) batches up to `URING_LOAD_MAX` loads per drain
with the same reservation protocol, one submit, finalize each.

### 5.5 Prefetch prediction & distance (the heuristics)

- **Router lookahead** (`colibri.c:5698-5757` + `pilot_prefetch` 6055-6120): predict layer
  **L+1**'s top-K by running L+1's router on L's post-attention state. Measured recall
  **71.6%** of true top-8 vs **41.3%** for "same experts as last token" (`colibri.c:5759-5761`,
  `docs/tuning.md:135`). kind=2 "two-step" (`PILOT_TWO=1`) approximates MoE(L) with the
  *shared expert only* (resident, no disk) and adds it to the state before predicting:
  **75.8% recall (+2.3 points) for 3 small matmuls** (`colibri.c:5701-5702`).
- **`PILOT_K`** (`colibri.c:9951-9956`): hint-only default **8** ("WILLNEED hints are free");
  PILOT_REAL default **6** — "at ~28% mispredict a large K thrashes the cache — default to
  6 (best-measured)". K is also capped at the model's topk (`colibri.c:6057`).
- Gating: `pilot_prefetch` runs only for `S<=8` (decode / small-batch tail) and
  `li+1 < n_layers` sparse (`colibri.c:6345,6362`).
- **Next-block readahead inside moe()** (`colibri.c:5052-5066`): while computing the current
  64-expert block, issue `expert_prefetch` (WILLNEED) for the **next block's 64 experts**
  (`base+64<nu`) that aren't resident — distance = exactly one block = 64 experts.
- **Under DIRECT=1 the weight WILLNEED is skipped** (wasted: O_DIRECT reads bypass the page
  cache the readahead would warm); the small `.qs` scales are always buffered so their
  WILLNEED is kept (`expert_prefetch`, `colibri.c:3498-3518`).
- `PREFETCH=1` (cross-layer WILLNEED, "method C") is **default OFF**: "real parallel loads
  made it redundant, and under memory pressure speculative readahead got re-evicted"
  (`colibri.c:1179-1181`).
- Kimi K3 prefetch: same-layer, **all misses** WILLNEED before the pipelined loads, only
  when `!g_k3_direct` (`kimi_k3.c:1851-1857`); loads sorted **in disk-offset order**
  ("experts are NOT id-ordered inside the HF shards — measured 169/895", `kimi_k3.c:1841-1850`).
- V4 experimental prefetch: `COLI_V4_EXPERT_PREFETCH` gate (`deepseek_v4.c:4472-4481`),
  per-range WILLNEED on scale+weight ranges of non-resident experts, routed replica's fd
  (`deepseek_v4.c:7208-7264`).

---

## 6. io_uring backend (Linux) — batched expert I/O

### 6.1 Minimal ring (`c/uring.h`)

Raw syscalls, no liburing: `coli_uring_init(r, entries)` mmaps SQ/CQ/SQEs
(`MAP_POPULATE`, IORING_FEAT_SINGLE_MMAP honored). `coli_uring_prep_read`
(`uring.h:89-113`) sets **`sqe->flags = IOSQE_ASYNC`** with justification:
*"Cold regular-file reads are allowed to execute inline during io_uring_enter() unless
forced async. That serializes the submitter on filesystems without native nonblocking
buffered reads and destroys the intended I/O/compute overlap. io-wq gives the ring a real
bounded worker pool."*
`coli_uring_set_workers` registers `IORING_REGISTER_IOWQ_MAX_WORKERS = {n,n}`
(`uring.h:84-87`). `coli_uring_enter(min_complete)` passes `IORING_ENTER_GETEVENTS` when
waiting; `coli_uring_peek` reaps one CQE with acquire/release on cq_head/tail.

### 6.2 Batch layer (GLM)

`colibri.c:3103-3128`:
```c
#define URING_LOAD_MAX 64      /* loads (experts) per batch */
#define URING_REQ_MAX  512     /* read ops per batch = ring entries */
typedef struct {
    ColiUring ring;
    UringLoad load[URING_LOAD_MAX];   /* one expert: 3 weight reads + 3 scale reads */
    UringRead req[URING_REQ_MAX];     /* user_data = req index + 1 */
    int nload,nreq,started;
} UringBatch;
static UringBatch g_ub_pipe, g_ub_pilot;
```
Ring created with **512 entries** (`uring_batch_init`: `coli_uring_init(&b->ring,URING_REQ_MAX)`).
One expert = up to 6 READ ops (3 weights + 3 .qs scales) ⇒ 64 experts ≈ 384 ops < 512.
Completion: `uring_reap` decrements `load[r->load].pending` per CQE; `l->done` at 0;
`uring_finalize_load` resolves formats and publishes `s->eid` (release point). Init at
`colibri.c:9986-10004`: `URING=1` **implies PIPE=1**, is refused with `COLI_MMAP=1`,
registers **io-wq workers = min(PIPE_WORKERS, 64)**, and prints:
`[URING] queued expert I/O active (depth=512, workers=8, buffered|DIRECT)` plus a nudge
`cold NVMe: DIRECT=1 avoids page-cache copy/readahead bottlenecks`.
Linux-only (`URING=1` on Windows returns exit code 2, `colibri.c:10002`).

---

## 7. mmap vs pread decision logic

### 7.1 GLM `COLI_MMAP=1` (`g_mmap`, default 0 — `colibri.c:2410-2414`)

*"gli expert diventano VISTE dentro mmap dei file safetensors (niente pread, niente slab,
nessuna copia: la page cache del kernel E' la cache)"* — the kernel page cache becomes the
expert cache. `map_of_fd` (`colibri.c:2425-2444`): one **whole-file** `PROT_READ MAP_SHARED`
map per fd (length rounded to 16 KiB), cached in `g_maps[512]` under a mutex, registered
with Metal for zero-copy GPU reads. In `expert_load_impl` (`colibri.c:2663-2705`) the mmap
arm is tried FIRST when `g_mmap` is on:
- falls back to the slab path if any map is missing or a tensor offset is not 4-byte
  aligned (`(tw[k]->off)&3`);
- on success the QT views point straight into the map, **planarize is forbidden** ("bytes in
  the MAP are read-only/shared: MAI planarizzare");
- then **CPU pre-touch**: `madvise(MADV_WILLNEED)` on the 16 KiB-aligned range (async
  readahead) **plus** a `volatile` touch of every 4 KiB page (`colibri.c:2686-2692`):
  *"fault the pages in HERE (cheap, parallel, overlapped with the resident-experts GPU
  submit) so the GPU never demand-faults file-backed pages (measured catastrophic).
  madvise starts async readahead, the touch guarantees residency. This is pread's I/O
  without the copy and without the slab."*;
- mlock is deliberately NOT taken here (would leak wired pages for GPU-tier experts);
  `pin_wire()` wires the final resident set later (`qt_wire_mmap`, `colibri.c:8672-8679`,
  which skips `cuda_eligible` slots — an earlier version wired **363 GB instead of 231 GB**
  and thrashed the kernel, `colibri.c:8648-8655`).
- `URING=1` + `COLI_MMAP=1` is refused (`colibri.c:9989`) — uring reads into slabs, mmap
  has no slabs.

### 7.2 Kimi `K3_MMAP=1` (default 0)

`kimi_k3.c:478-499` `w_map_prepared`: maps prepared U8 weights + **F32 scale sidecars**
via `st_map_raw`; refuses loudly unless the container is *fully prepared* (no load-time
conversion possible), scale sidecar must be F32 and float-aligned; **CPU-only**
(`k3_mmap_backend_allowed`: mmap is incompatible with the VK/CUDA tiers because their
upload paths need host-owned buffers, `kimi_k3.c:474-476,759-763`).

### 7.3 Summary of the decision

colibri defaults to **pread into engine-owned slabs** because (a) RSS control (DONTNEED
drop), (b) O_DIRECT/uring/striping need engine buffers, (c) GPU tiers upload from host
slabs. mmap mode exists for CPU-only, prepared, aligned containers where the page cache can
double as the LRU. DENSE tensors (attention, embed, lm_head) are always pread-once into
RAM; only routed experts stream.

---

## 8. Staging buffers: slabs, working sets, arenas (sizes & cycling)

### 8.1 GLM `ESlot` (`colibri.c:372-383`)

```c
typedef struct { int eid; QT g,u,d; uint8_t *slab; float *fslab;
                 int64_t slab_cap, fslab_cap; uint64_t used;
                 unsigned in_flight;              /* async GPU readers borrowing this slot */
                 uint8_t *aslab; float *afslab; } ESlot;   /* pin-arena backing (#419) */
```
`slab` = one expert's 3 packed weight matrices read by (at most) one coalesced pread;
`fslab` = the float scales. `Model` has `ESlot ws[64]` — the per-block working set where
PIPE workers / OMP threads land misses (`colibri.c:445`), plus per-layer `ecache` rows and
`pin` rows (HOT-STORE).

**Sizing** (`expert_load_impl`, `colibri.c:2735-2792`):
- `want = wtot + 8192` (8 KiB slack); allocate `posix_memalign(4096, wtot+8192)`
  (16 KiB alignment + 16 KiB rounding under METAL);
- **shrink hysteresis** (#856): realloc-shrink only when `slab_cap > want + want/4` with
  floor **64 KiB** (`hyst=want/4; if(hyst<(1<<16)) hyst=1<<16`), because ws[] slots migrate
  between int4 rows and int8 MTP rows and a wide slot would bleed its width into every row
  (measured: cache halved 154→77 slots/row). Scale slab (floats): `fhyst=ftot/4` floor
  **16384 floats** (=64 KiB). `COLI_SLAB_SHRINK=0` disables (diagnostic).
- Never shrink arena slices (`aslab` set).

**Slot cycling**: demand miss → `ws[q]` (loaded by PIPE worker / OMP) → matmul consumes →
end of block, LRU promotion swaps hot `ws[]` slots into the layer's `ecache` row ("promozione
LRU"); `used = ++eclock` stamps recency; `eslot_lru_victim` (`colibri.c:399-411`) picks
free-with-slab first (only while live-slab count < ecap), else least-recent, skipping
busy/reserved (`eid<-1` reservations count as live). Reservation encoding: **`eid=-(eid+2)`**
so reserved ≠ any real eid and ≠ -1 (free).

**Pin arena (#419)** (`colibri.c:8854-8901`): a layer's pins pack into **two arenas**
(weights + scales) at fixed stride `ws=((wtot+8192+4095)&~4095)` — because 19,456 experts
x 2 mbind'd slabs each crossed `vm.max_map_count=65530` and posix_memalign died "with
terabytes free". 2 VMAs per layer instead of ~500. After a GPU H2D upload,
`expert_host_release` (`colibri.c:3457-3489`) `madvise(MADV_DONTNEED)`s the arena slice —
"the arena owns the virtual address, not the resident pages" — and re-attaches on
`expert_host_ensure`.

### 8.2 O_DIRECT read window (GLM)

`colibri.c:2803-2836`: the 3 weight tensors are sorted by file offset; if contiguous on one
fd, ONE read:
```c
int64_t base=off0 & ~4095LL, need=(off0-base)+wtot;
int64_t len=(need+4095)&~4095LL;
ssize_t r=mir_pread_striped(&m->S,tw[ord[0]]->fd,rep,(char*)s->slab,len,base);  /* multi-SSD first */
if(r<need){ r=pread(dfd, s->slab, len, base); }                                 /* O_DIRECT twin */
```
Slack: 8 KiB allocation slack covers the 4 KiB head pad + tail rounding. Scales always
buffered pread (3 small reads into fslab). Non-contiguous ⇒ 3 buffered preads.
Post-read `qt_resolve_fmt` + `qt_planarize` on slab-owned bytes.

### 8.3 Kimi K3 slots (`kimi_k3.c:1370-1412`)

`Slot { base, buf, eid, used }`; `base = posix_memalign(4096, e_slot+8192)` where
`e_slot = 2*(w1p+w1s)+w2p+w2s` (per-expert bytes). DIRECT window read with explicit
sub-4K tail: head slack `pad`, aligned `dlen` via O_DIRECT, remainder via **buffered tail
pread** on the normal fd ("O_DIRECT wants aligned lengths"). `ws[64]` working set
(`kimi_k3.c:180`), LRU `LCache` per layer, ecap sized `ram_budget / e_slot`
(`kimi_k3.c:930`). Measured: **7.1 GB/s direct vs 2.9 buffered** (1.8 effective with
resident weights eating cache headroom) — `kimi_k3.c:1365-1369`. K3_DIRECT **default 1**.

### 8.4 K3 loader pool (`kimi_k3.c:1531-1583`)

```c
#define LP_MAX 64
static struct {
    pthread_t th[16]; int nth, started;
    pthread_mutex_t mx; pthread_cond_t cv;
    LJob job[LP_MAX];  _Atomic int ready[LP_MAX];
    _Atomic int next;  int count;
} g_lp;
```
Simpler than GLM's PipePool: workers `atomic_fetch_add(&next,1)` under the mutex (no
generation counter — *"One batch in flight at a time (the submitter consumes every job
before the next submit), so the flags need no generation counter"*, `kimi_k3.c:1533-1535`).
Compute waits per-expert with `usleep(50)` spin on `ready[]` (`kimi_k3.c:1640-1646`),
counting only un-hidden time into `t_eload`. `K3_LOAD_THREADS` default **4**, clamp [1,16].
`K3_PIPE` default 1; fallback = `omp parallel for` all-up-front.

### 8.5 DeepSeek-V4 store (FP8/FP4 — closest to Insignia's mission)

- Slab **is** the final cache slot; aligned hot-store slabs:
  `posix_memalign(4096, record_bytes + 8192)` (`deepseek_v4.c:7862-7871`) — "+8192: the
  extra two pages let an unaligned safetensors range be expanded to an O_DIRECT window
  safely" (`deepseek_v4.c:7528-7530`). Cold ssd-io store uses plain `malloc(record_bytes)`
  slabs + a per-call 4-KiB-aligned bounce buffer (`coli_st_read_at_streaming`,
  `deepseek_v4.c:139-200`).
- `v4_read_expert_record` (`deepseek_v4.c:7585-7624`): FLOCK-packed checkpoints
  ([scales][weights] contiguous) ⇒ **one** direct window request; standard HF layout ⇒
  weights direct + scales buffered; any direct error falls back to fully buffered.
- `COLI_V4_DIRECT` env gates it (`deepseek_v4.c:119-137`), per-replica twin check.
- **Loader lanes** (`deepseek_v4.c:4498-4539`): dual-expert loader pool,
  `V4_LOADER_LANES` default **9**, clamp [1,16] (`DUAL_EXPERT_LOADER_MAX=16`).
  Justification, measured on a 12-core box streaming V4-Flash from a VHDX: one 13.4 MB
  cold expert read costs **~48 ms at 3 lanes vs ~29.6 ms at 10**, because the disk scales
  almost linearly with queue depth (**86 MB/s at QD1 → 696 MB/s aggregate at QD8**,
  measured with O_DIRECT dd); decode **6.2 → 4.9 s/token**. And 9-vs-3 lanes on the real
  checkpoint: decode **147.3 s → 104.8 s (1.41x mean, 1.46x median)**, zero OOM in 8/8
  runs. The CPU-reservation for OpenMP deliberately still subtracts the *compile-time*
  count ("lanes block in pread and do not need whole CPUs").
- **Pipelined GPU refill** (`deepseek_v4.c:9755-9819`): `V4_MOE_REFILL_GROUP_MAX=16`,
  group default **6** (`V4_MOE_REFILL_GROUP`), pipe group capped at 2/3*16=10: "the
  lookups of group g+1 (parallel O_DIRECT reads) run concurrently with the uploads of
  group g (single-stream cudaMemcpy)"; two groups' views held at once, so the size is
  halved against the pin-slot budget.

### 8.6 Qwen3.6 CUDA upload ring (`qwen36_tier.c`)

```c
#define QT_QCAP 48            /* upload queue depth (staging ~1.6 MB/entry) */   // qwen36_tier.c:11
struct { int layer, eid; uint8_t *w; float *s; int v_layer, v_eid; } q[QT_QCAP]; // :33
```
One uploader thread; each entry owns a **malloc'd staging copy** (not pinned memory):
`w=malloc(3*mb)`, `sc=malloc(scales)` (`:198`), filled by `stage()` which converts packed
int4 two's-complement → offset-binary (**XOR 0x88** per 64-bit word, the fmt=2 upload
format) and concatenates scales (`:55-66`). Ring cycles via qh/qt_/qn under one mutex with
two condvars (work available / space available); full ring = skip (counted) or block
(warmstart `qt_note_block`). Victim tensors are freed only when no expert group is in
flight (`issue_open` + `cv_take`, `:79-87`). LFRU swap check every 16 ticks/tokens (`:325-346`).

---

## 9. Multi-SSD routing & striped reads

- **Deterministic replica routing** (`colibri.c:2461-2472`): `expert_route(layer,eid)`
  hashes (layer,eid) (Knuth-ish multipliers + xorshift finish) into [0,256) and walks
  cumulative cuts `g_mir_cut[]`. *"Determinism is a requirement: the readahead/PILOT
  WILLNEED and the demand pread must hit the same fd/page-cache, and in buffered mode an
  expert must never be cached twice."* Cuts derive from `COLI_DISK_WEIGHTS` or the startup
  probe (§2.5); every replica keeps a non-empty slice.
- **Striped single-expert read** (`mir_pread_striped`, `colibri.c:2479-2525`): only when
  `>= 2 replicas` and `len >= 4 MiB` and O_DIRECT ("no page cache involved, so striping
  cannot double-cache"); splits the coalesced window into 4-KiB-aligned disjoint chunks,
  one pthread per chunk (chunk 0 starts on the routed replica to keep hash balance);
  any short stripe fails the whole attempt → single-replica fallback. *"a cold ~19 MB
  expert read is single-thread latency-bound (~4 GB/s on one NVMe) ... N drives reading
  stripes of the SAME expert cut it ~N-fold."* `COLI_MIR_STRIPE=0` disables.
- V4 mirror note (`deepseek_v4.c:336-346`): the XOR hash didn't spread small hot subsets
  evenly → replaced with a linear mix there.

---

## 10. Windows-native measured numbers & posture

- `docs/windows.md:159`: Core Ultra 9 285K / RTX 5080 / 128 GB / NVMe at **5.85 GB/s
  random-read (19 MB blocks, iobench)**: 0.26 tok/s cold CPU → 0.30 warm → **0.42 GPU tier
  + auto-pin, 66% expert hit, ~65% of wall in expert-disk**. "Disk-bound is the expected
  shape at ~25% residency."
- `PIPE` **defaults ON on Windows** (`colibri.c:9964-9970`) precisely because there is no
  uring there; the thread pool is the QD mechanism.
- `iobench.c` (the measurement harness): defaults 19 MB block, 64 reads, 8 threads,
  direct=1; 4 KiB-aligned offsets from a 30-bit rand (Windows RAND_MAX=32767 caveat);
  `compat_fsize` on Windows because CRT lseek fails on NO_BUFFERING fds; aligned frees.
- Windows fadvise WILLNEED is real but *blocking* — hence the rule that only the PILOT
  thread / next-block readahead may call it (§3.3).

---

## 11. Constant inventory (with justification where present)

| Constant | Where | Value | Why |
|---|---|---|---|
| `ST_MAX_SHARDS` / fd arrays | st.h:45-53,74 | 512 | max shard files indexed |
| `ST_MAX_MIR` | st.h:50 | 4 | max mirror replicas (+primary = 5 drives) |
| `ST_MAX_HEADER` | st.h:28 | 512 MiB | cap on safetensors header malloc (hostile file guard) |
| `ST_PREAD_CHUNK` | st.h:259 | 1 GiB | single pread caps ~2^31 B on Linux |
| `ST_FMT_STAMP_MAX` | st.h:297 | 4096 | stamp-map scan bound (resident-tensor convention) |
| O_DIRECT align contract | compat.h:296-311, st.h:299-301 | 4096 | offset/len/buffer must be 4 K-aligned |
| fadvise readahead cap | compat.h:141 | 64 MiB | "pathological huge len would spike transient memory" |
| mlock working-set margin | compat.h:205 | len + 1 MiB | VirtualLock needs grown min working set |
| pread chunk | compat.h:175 | 0x7FFFFFFF | DWORD per ReadFile |
| `PipePool` jobs | colibri.c:3344-3348 | 64 (`eids[64]`,`ready[64]`) | one moe block; index fits 8 bits of `cur` |
| `cur` layout | colibri.c:3344 | `(gen<<8)\|idx` | gen main-only monotonic (no ABA), idx ≤ 64 |
| pipe workers | colibri.c:3325,9975 | 8 (env `PIPE_WORKERS`, clamp [1,16]) | disk queue depth (threads) |
| pipe wait | colibri.c:3453 | sched_yield spin (default) | legacy byte-identical; condvar opt-in (~5 us wake vs 0.5-3 ms reads, #159) |
| `pilot_q` | colibri.c:5771 | 4096 slots {layer,expert} | 1P/SPC ring; full = drop hint |
| pilot idle poll | colibri.c:5950,5956 | usleep(200) | park latency vs burn |
| `PILOT_K` | colibri.c:9955 | 8 hint / 6 REAL | head of ranking more reliable; ~28% mispredict thrash at high K |
| `PILOT_WORKERS` | colibri.c:9960-9961 | 1 (clamp [1,16]) | QD on blocking REAL path only; URING already batches |
| pilot layer distance | colibri.c:6345 | L+1 | recall 71.6% (75.8% two-step) vs 41.3% token-repeat |
| next-block distance | colibri.c:5054-5056 | +64 experts | one moe block of WILLNEED during current block |
| `URING_LOAD_MAX` | colibri.c:3107 | 64 | experts per uring batch |
| `URING_REQ_MAX` (= ring entries) | colibri.c:3108,3127 | 512 | 64 experts x up to 6 reads |
| io-wq workers | colibri.c:9994 | min(PIPE_WORKERS,64) | bounded io-wq pool |
| uring SQE flag | uring.h:104 | IOSQE_ASYNC | prevent inline-exec serialization of the submitter |
| ws / ecache victim hysteresis | colibri.c:399-411 | 25% + 4-freq (tier.h:53) | anti-thrash |
| slab slack | colibri.c:2735 | +8192 B | O_DIRECT head pad + tail rounding |
| slab shrink hysteresis | colibri.c:2736 | want/4, floor 64 KiB | stop width bleed across rows (#856); clears 16 K METAL rounding |
| fslab hysteresis | colibri.c:2755 | ftot/4, floor 16384 floats | same, in floats |
| K3 slot slack | kimi_k3.c:1377 | e_slot+8192 | same purpose |
| K3 load threads | kimi_k3.c:1564 | 4 (clamp [1,16]) | overlap reads with compute |
| K3 wait | kimi_k3.c:1644 | usleep(50) spin | per-expert readiness |
| V4 hot slab | deepseek_v4.c:7862 | record_bytes+8192, align 4096 | "extra two pages ... O_DIRECT window safely" |
| V4 loader lanes | deepseek_v4.c:4527,4533 | 9 (env `V4_LOADER_LANES`, [1,16]) | 86 MB/s QD1 → 696 MB/s QD8; 1.41x decode measured |
| V4 refill group | deepseek_v4.c:9761-9773 | 6 (max 16, pipe ≤10) | QD vs bounded pin slots; two groups in flight |
| qwen36 upload ring | qwen36_tier.c:11 | 48 entries (~1.6 MB each) | staging depth for single uploader thread |
| mirror probe | colibri.c:8723 | 8 x 19 MiB | engine-shaped bandwidth probe, ~150 MB/drive |
| stripe threshold | colibri.c:2498 | ≥ 4 MiB & ≥2 replicas & O_DIRECT | below that, stripe overhead dominates |
| LRU swap tick | qwen36_tier.c:328 | every 16 tokens | amortize the O(n) scan |

---

## 12. Transferable lessons for Insignia (25.65 GB FP8, 4070S + 5600X + 16 GB)

1. **Copy the twin-handle pattern as-is** (`compat_open_direct` + eager open + GetFileSizeEx
   + `_FILE_OFFSET_BITS=64` guard). It is the only measured-correct Windows O_DIRECT story
   in any of the reference clones. Alignment contract: 4K offset/len/buffer.
2. **Windows async = N threads in blocking positioned ReadFile(OVERLAPPED-as-offset)**.
   Colibri never built IOCP — with 8 pread workers the disk (not completion dispatch) is
   the bottleneck; V4 measured QD8→696 MB/s on a DRAM-less VHDX. For Insignia's NVMe
   (Gen4, likely 7 GB/s at QD≥8) start with the PIPE pool (gen-tagged CAS) at 8 workers.
3. **The (gen<<8|idx) cursor is cheap and correct**: single-word publish, no ABA, no drain
   barrier, mutex only for parking. If Insignia's decode block ever exceeds 64 jobs per
   layer it needs a wider index field (the layout is the only 64-cap).
4. **Slab = final destination**: read the expert's whole [scales][weights] window in one
   4-KiB-aligned O_DIRECT read into the slot it will be used from (+8 KiB slack); scales
   buffered. Never bounce-and-copy per miss (V4 removed it).
5. **Prefetch distance that measurably works**: layer+1 router lookahead (71.6%/75.8%
   recall) with K≈6-8, issued from a dedicated thread (Windows WILLNEED blocks!), plus
   next-block (64-expert) readahead; skip weight WILLNEED under DIRECT (it warms a cache
   the direct reads bypass).
6. **fadvise policy asymmetry**: keep DONTNEED a no-op on Windows (standby list = free L2
   for hot experts); only drop under measured memory pressure.
7. **mmap is an opt-in CPU-only mode** — engine-owned slabs are what make O_DIRECT, uring,
   striping, and VRAM uploads possible. For a 25.65 GB model on 16 GB RAM, slab streaming
   with a RAM budget + pin arena is the architecture that fits.
8. **Deterministic (layer,eid)→device routing** whenever hints and demand reads must meet
   on the same fd/cache.
9. **Measure with the engine's own pattern**: `iobench` (19 MB random blocks, 8 threads,
   O_DIRECT) is the right shape for MoE expert streaming; ASYNC-queue-depth scaling is the
   single biggest Windows I/O lever measured in this codebase.
