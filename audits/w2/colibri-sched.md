# Colibri scheduling/execution audit (Week 2)

Scope: `E:\coding\Insignia\colibri\c\` — scheduling, placement, overlap, speculation,
threading, KV-vs-weights. Raw I/O (pread/O_DIRECT/uring/mirror) covered by the other
agent. All references are `file:line` in that tree. Engines: `colibri.c` (GLM-5.2 MoE
 MLA, 10477 lines), `qwen36.c` (Qwen3.5 MoE + GatedDeltaNet, 2549 lines),
`qwen36_tier.c` (CUDA expert tier, 446 lines), `olmoe.c` (OLMoE, 1530 lines),
`tier.h` (LFRU, 60 lines), `backend_cuda.cu` (CUDA group API).

Colibri's model is **expert-granular** placement, not layer-granular: "layers" are
always fully visited per token; what moves between NVMe/RAM/VRAM are individual MoE
expert slabs (gate/up/down, ~19-30 MB each) plus optional dense-weight promotion.
Their own summary of the philosophy is `qwen36_tier.h:3-5`:
"route -> place -> overlap -> learn".

---

## 1. LFRU placement score — exact code

`tier.h:27-33` (the canonical version; duplicated verbatim at `olmoe.c:113-118`):

```c
/* LFRU: frequency is the primary signal; recency breaks close calls. A recent
 * access contributes at most 255 points while one frequency count is worth
 * 256, so a merely recent expert cannot displace a genuinely hotter one. */
static uint64_t tier_lfru_score(uint32_t heat, uint32_t last, uint32_t clock){
    uint32_t age=clock-last, recent=age<255?255-age:0;
    return ((uint64_t)heat<<8)|recent;
}
```

So score = `(heat << 8) | (255 - min(age,255))`. `heat` is a saturating per-(layer,
expert) access counter; `last`/`clock` are a monotonically bumped access tick.

Heat bookkeeping (the "learn" half):

- `colibri.c:4742-4751` — routing FASE A bumps, once per routed (token, expert):
```c
m->eusage[layer][idx[kk]]++;
...
if(m->eheat[layer][idx[kk]]<UINT32_MAX) m->eheat[layer][idx[kk]]++;
m->elast[layer][idx[kk]]=++m->eaccess_clock;
```
  (same bump in the pre-routed path `colibri.c:4598-4603`). A separate private
  clock `eaccess_clock_dc`/`elast_dc` exists for the DISK-CLASS heuristic
  (`colibri.c:448-471`, `4745-4748`) so classification recency never pollutes
  placement recency.
- `qwen36.c:1616-1619` — same idea, `freq_l[idx[kk]]++` when not hot_pinned.
- Decay: `tier.h:56-58` `heat[e]>>=1` for all experts, run once per live-repin
  pass (`colibri.c:7774`), so heat is an exponential window over passes/tokens.

Swap selection with hysteresis — `tier.h:35-54` (`tier_pick_lfru`): pick coldest
pinned slot and hottest non-resident expert by score, then require

```c
/* Retain the existing 25%+4-frequency hysteresis in score units. */
if(hs<=cs+(cs>>2)+(4u<<8)) return 0;    /* tier.h:52 */
```

i.e. the challenger must beat the victim by 25% of the victim's score plus 4 full
frequency counts (`4<<8`), or the swap is refused (anti-ping-pong). The pure-heat
ancestor `tier_pick_swap` (`tier.h:8-25`) uses `fh<=fc+(fc>>2)+4`.

Where LFRU actually drives decisions:

1. **Live re-pin between turns** (`REPIN=n`, default off): `repin_pick`
   `colibri.c:7593-7626` calls `tier_pick_lfru` per layer, max 4 swaps/pass
   (`colibri.c:7585`), executed at the safe point after a reply (`repin_pass`
   called from `spec_decode` at `colibri.c:7010` — "all device work is
   synchronized").
2. **Speculative-pilot eviction guard** (`#441`, on by default):
   `colibri.c:5805-5811` and `5874-5880` — a prefetch may evict a *warm* resident
   (heat>=2) only if the prefetch's LFRU score beats `vs + vs>>2 + (4u<<8)`;
   otherwise the speculation is dropped (`g_pilot_drops++`) instead of thrashing
   a warm slab.
3. **qwen36_tier VRAM swaps**: `qt_lfru_tick_locked` `qwen36_tier.c:327-346`,
   every 16 tokens (tick incremented in `qt_issue` when layer==0,
   `qwen36_tier.c:357`), per device: coldest resident (raw `heat`, not the
   composed score) vs hottest non-resident with `hh<=ch+(ch>>2)+4` hysteresis;
   victim freed only when no group is in flight (`issue_open` condvar,
   `qwen36_tier.c:80-87`).

Startup placement (before any token) is **not** LFRU: `pin_load` reads persisted
usage history (`.coli_usage` / route_trace), sorts frequency-descending
(`pin_rec_cmp` `colibri.c:8823-8825`), then `pin_count_for_budget` slices the list
to the RAM budget (`colibri.c:8993-8994`). The VRAM prefix is the budget-sized
head of that same ranking (`colibri.c:8954-8991`), with per-device
load-balance/greedy assignment in the load loop (`colibri.c:9064-9102`).
`qwen36_tier` warmstart likewise fills VRAM heat-descending from a persisted
`HEAT_FILE` (`qt_plan_fill` `qwen36_tier.c:272-297`; heat saved back in
`qt_shutdown` `qwen36_tier.c:427-439`, loaded with a one-shot `>>1` decay at
`qwen36_tier.c:163`).

---

## 2. Early-issue CPU/GPU overlap — "1 sync per device"

Two-layer design in the CUDA MoE path of `colibri.c` `moe()`:

**(a) Issue-all-then-take-all group calls** (`colibri.c:5500-5529`, comment):

```c
/* Inc.4: at decode scale, issue every device's group WITHOUT syncing, then take
 * them all — one stream sync per device per layer instead of a full staged
 * round-trip per call (measured: ~70% of the sync call is host-side wait).
 * Any issue failure drains what was issued and the whole layer falls back to
 * the sync path below, which recomputes from group_x (idempotent). */
```

Backed by `backend_cuda.cu`: `coli_cuda_expert_group_issue` (1992-2104) packs
per-expert `GroupDesc {g,u,d ptrs, scales, fmts, rows, offset}` H2D, uploads the
packed activation block `x` through pinned staging, launches the grouped
gate+up+silu and down kernels for the whole layer's expert set as one
`dim3(I, max_rows, count)` grid per projection on the device stream, enqueues the
D2H of `y`, sets `group_pending`, and **returns without sync**. `take`
(`backend_cuda.cu:2106-2113`) is just `cudaStreamSynchronize` + return host `y`.
Decode-scale only (`total>8` refused, `backend_cuda.cu:2022`).

**(b) Early issue before the CPU expert loop** (`COLI_GROUP_ASYNC`, "Inc.4 overlap
stash", `colibri.c:5104-5182`): VRAM-resident experts of the block are packed and
issued *first*; then the CPU loop computes the RAM-tier/miss experts (`done_j[]`
skips GPU-owned ones, `colibri.c:5377`); then the take phase
(`colibri.c:5452-5478`) syncs each device and accumulates. Comment at
`colibri.c:5105-5110`: "t_emm becomes max(cpu, gpu) instead of the sum". Failure
anywhere -> CPU recompute of exactly those rows (`colibri.c:5465-5472`), because
`group_x` packing is idempotent. Overlap is measured via `g_ovl_issue/g_ovl_cpu/
g_ovl_take` windows (`colibri.c:5157,5454,5476`).

The same pattern in the other engines:

- **Vulkan two-GPU variant** (`colibri.c:5250-5372`): partition by registry
  residency; *issue the slower device first* so it gets the longest overlap
  window, with dev2's submit on a transient worker thread
  (`vk2_issue_worker` `colibri.c:4456-4469`; join before `take2`, comment
  5293-5295); compute the CPU share between issue and take; take dev0, accumulate,
  *then* take dev2 (comment 5352-5353: "the slower card gets the extra overlap").
- **qwen36 tier** (`qwen36.c:1622-1669`): `qt_issue` (async, all devices) ->
  CPU computes the *miss* experts (`matmul_qe` int8) *and the shared expert*
  inline (comment 1644-1645: "Compute the shared expert NOW so it overlaps with
  the GPU groups") -> `qt_take` accumulates. Timed as `g_qt_iss / g_qt_cpu /
  g_qt_tak` (`qwen36.c:1666-1669`).
- **Metal variant** (`colibri.c:5019-5103`): resident subset submitted async
  (`coli_metal_moe_block_begin`), misses pread in between, missed subset
  submitted, both collected at end-of-block.

Failure containment is uniform: any issue/take failure drops the affected subset
to an idempotent CPU recompute; nothing about the schedule is unsafe to retry.

---

## 3. Layer N+1 prefetch overlapping layer N execution

Three mechanisms, router-driven rather than blind:

**(1) Pilot prefetch (the big one).** After layer i's attention residual is
produced, and again after its MoE residual, the *next layers' routers* are
evaluated speculatively on the current hidden state:

```c
if (g_pilot >= 1 && S <= 8 && i + 1 < c->n_layers) pilot_prefetch(m, i + 1, x, S);  /* qwen36.c:1910-1911 */
...moe...
if (g_pilot >= 2 ...) pilot_prefetch(m, i + 2, x, S);   /* qwen36.c:1918-1919 */
if (g_pilot >= 3 ...) pilot_prefetch(m, i + 3, x, S);   /* qwen36.c:1920-1921 */
```

`pilot_prefetch` (`qwen36.c:1980-2042`) rmsnorms x with *layer i+1's* post_ln,
runs the actual router GEMV (int8 copy — comment 1988: "f32 may be freed"),
blends with an EMA of past router logits (`momentum_logits`, `pilot_smooth`),
softmaxes, and selects candidates until cumulative probability reaches
`pilot_conf_limit` with a floor of top-K and a ceiling of `topk*g_wide`
(`qwen36.c:2004-2012`). Candidates already RAM-resident are promoted to VRAM
asynchronously (`qt_note`, `qwen36.c:2022-2026` — "Lookahead: RAM-resident
layer-L+1 candidates go to VRAM asynchronously"); non-resident ones are pushed on
a 4096-entry SPMC ring (`pilot_q`, `qwen36.c:2027-2038`) to the pilot worker
thread (`pilot_worker` `qwen36.c:1967-1978`), whose `pilot_realload`
(`qwen36.c:1946-1965`) preads the expert into an LCache slot (LRU by `used`
clock, pinned slots respected).

The GLM version (`colibri.c`) has the same pilot with more knobs: `PILOT`,
`PILOT_K` (default 8 hint-only, 6 under `PILOT_REAL` — comment 9951-9955: at
~28% mispredict a large K thrashes the cache), `PILOT_WORKERS` (SPMC fan-out
1..16, `colibri.c:5922-5978`), `PILOT_TWO` (shared-expert-corrected router
prediction, +2.3% recall for 3 matmuls, `colibri.c:9949`), `g_looka` measuring
routing predictability (previous-token routing overlap, `colibri.c:4763-4769`),
and `couple_prefetch` (`colibri.c:4761-4762`). The pilot respects
`g_cur_moe_layer`: stale queue entries for layers already reached are dropped
(`colibri.c:5856-5858`).

**(2) Expert-block readahead inside moe()** (`colibri.c:5052-5066`): while the
current 64-expert union block computes, the *next* 64-expert union block's
non-resident members get `expert_prefetch` (POSIX_FADV_WILLNEED-style hints) —
comment: "il kernel legge in background, le pread dopo trovano cache calda".

**(3) PIPE async demand loads** (`colibri.c:3294-3454`): the current block's
misses are dispatched to I/O workers *before* the matmul loop; see §6.

---

## 4. Speculative decode — depth-1 MTP, acceptance ~85%

Central loop `spec_decode` `colibri.c:6899-7021` (single-slot; mux/serve variants
set `g_mux_stop/cancel` checked at 6898/6921):

1. Emit target token from `logit` (`pick_tok`, with `carry_ban` excluding a
   rejected draft token from resampling, 6903/6997).
2. Choose a draft source, priority: grammar-forced (`grammar_draft`, ~free
   acceptance, 6934-6938) -> corpus n-gram span (`corpus_draft` 6939-6962) ->
   **MTP head** (`mtp_draft`, 6977-6979) -> plain n-gram fallback when no MTP
   head exists (`ngram_draft` 6660-6669).
3. One batched verify forward: `step_all(m, batch, S=1+g, kv)` (6985-6987).
4. Greedy accept while argmax matches (or rejection-sample `p(draft)` when
   temperature > 0) — 6992-7001; accepted count `k`, `kv += 1+k` (7008).
5. `mtp_absorb` (7005, def 6719-6746+) replays the *verified* tokens through the
   MTP head's own KV so the next draft sees consistent state; `hlast` is rewound
   to the last *accepted* position, not end-of-batch (7007).
6. `repin_pass(m)` at 7010 — safe point for live re-pin.

Draft depth auto-default — `colibri.c:10216-10236`, the exact source of the
"depth-1 / ~85%" numbers:

```c
/* Auto depth = 1, not 3. A GLM-5.2 744B sweep (DRAFT=0/1/2/3, streaming and
 * fully-resident) showed single-token speculation is the only depth that pays:
 * acceptance ~85% at depth 1 vs ~44-62% at 2-3, and every extra draft token
 * both costs verify compute and (when streaming) faults experts that evict the
 * LRU working set. Depth 1 was the fastest MTP setting in every measured
 * configuration; 2-3 never beat it anywhere. DRAFT=n still forces any depth. */
g_draft = (m.has_mtp && (!g_cuda_enabled || cuda_mtp)) ? 1 : 0;
```

Note the interplay with placement (README.md:124): "MTP and grammar drafts work,
but MTP has also measured a 32% loss around 85% expert hit" — a deeper verify
batch routes through *more* experts per forward, so at partial residency
speculation actively evicts the working set. Two structural mitigations ship:

- `SPEC_PIN` (#163, `colibri.c:541-543, 6904-6909`): while drafts are live, the
  kernel family is pinned so draft (S=1) and verify (S=1+g) forwards compute
  *the same function* — FP accumulation-order divergence between the S==1
  fused-pair kernel and the S>=2 IDOT kernel collapsed acceptance (#8/#163).
  Under CUDA, MTP is off by default for exactly this reason (cold CPU experts
  diverge from GPU ones; comment `colibri.c:10218-10225`, opt-in
  `COLI_CUDA_MTP=1`, "acceptance can still reach 30-50%").
- Adaptive pause guards: a 24-proposal sliding window; below 70% MTP acceptance
  (env-tunable) or 50% corpus acceptance, drafts pause for 256 tokens and
  re-arm (`colibri.c:6910-6975`), instead of the old permanent latch.

Draft mechanics (`mtp_draft` `colibri.c:6679-6715`): DeepSeek-V3 style chain
`h' = eh_proj[ enorm(emb(tok)) ; hnorm(h) ]`, one `layer_forward` of the extra
`mtpL` row (stored as layer `n_layers`, int8-only — int4 MTP heads collapse to
0-4% acceptance, README.md:381), argmax over the shared lm_head; state carried
across drafts via `m->hlast`. The MTP row has its own KV with a decode-only
window starting at `kv_start[n_layers]` (`colibri.c:2323`).

---

## 5. Staging rounds: min(4 GB, budget/8)

`colibri.c:9019-9045` (issue #730 — hosts with more VRAM than RAM OOM'd when the
whole VRAM-prefix was host-staged at once):

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
```

The loop (`colibri.c:9046-9107`) then alternates: OMP parallel pread of `stage`
experts -> CUDA upload + budget check -> `expert_host_release` frees each round's
host slab before the next is read. Peak host RSS becomes `stage*expert_bytes`
instead of the whole VRAM tier.

Same discipline in `qwen36_tier`: upload queue depth `QT_QCAP 48`
(`qwen36_tier.c:11`, ~1.6 MB staging per entry), queue-full drops
(`q_full_skips`, 194), warmstart fills via `qt_plan_fill` (reserve budget once,
`qwen36_tier.c:291-292`) then any number of OMP loader threads call
`qt_note_planned` (blocking only on queue space, 301-315), `qt_fill_wait` drains.
Crucially the qwen36 warmstart loads **all** experts to RAM, not just the VRAM
set — comment `qwen36.c:2431-2434`: otherwise "the first touch of a CPU-fallback
expert triggers a ~12 ms container read in the middle of decode (measured:
139 ms/token on a single-GPU run)". And `qwen36.c:2444-2448`: the int8 RAM copy
of a VRAM-resident expert is freed immediately after staging (rematerialized by
`slot_ensure_int8` from the kept int4 `g4` on eviction).

---

## 6. Threading model — who exists, who does what

Persistent pthreads (plus the OpenMP pool):

| Thread(s) | Created | Job |
|---|---|---|
| Main decode thread | — | Token-serial: routing, all CPU matmul (`matmul_qt` must stay off the OMP team and off I/O threads — comment `colibri.c:3299-3301`), GPU issue/take, sampling, spec loop. Sole writer of pipe generation counter. |
| PIPE I/O workers xN (default 8, max 16) | `pipe_init` `colibri.c:3386-3400` | Demand expert preads into distinct `ws[]` slabs, set per-slot `ready[]` flags. Synchronization: single generation-tagged lock-free cursor `cur=(gen<<8)|idx`; workers CAS-advance the index, main RELEASE-publishes batches (`colibri.c:3343-3385, 3404-3424`). Mutex/condvar only parks idlers. `pipe_wait` = sched_yield spin by default, optional condvar (`COLI_PIPE_BLOCK`, ~5us wake vs 0.5-3ms reads, #159). Linux alternative: io_uring backend (`URING=1`). |
| Pilot worker(s) x1..16 | `colibri.c:5976-5978`, `qwen36.c:630-634` | Speculative cross-layer expert preads from the SPMC ring; slot reservation visible as `eid=-(eid+2)` with `used=(uint64_t)-1` sentinel so a loading slot is never an LRU victim (`colibri.c:5814-5818`). |
| qtier uploader x1 | `qwen36_tier.c:171` | Drains the staging ring, XOR 0x88 offset-binary conversion + scale copy (`stage`, 55-66), per-device tensor uploads, LFRU victim free (waits `issue_open`). |
| vk2 issue worker (transient per block) | `colibri.c:5296-5301` | Off-loads the slow Vulkan device's ~0.8ms submit cost so it overlaps dev0 issue + CPU share. |
| Mirror stripe workers | `colibri.c:2514` | Dual-SSD weighted striping (raw-I/O agent's domain). |
| OMP pool | — | Expert loads (`#pragma omp parallel for schedule(dynamic,1)`), dense prefill, XEXP phases, kernel-internal row parallelism. GPU dispatch is gated on `!omp_in_parallel()` everywhere (e.g. `colibri.c:5402, 5413`). |

CPU-share ordering trick worth stealing (`colibri.c:5315-5323`): the CPU
experts of a block are run *serially* (their kernels are already OMP-parallel;
one-task-per-expert measured -23% because ~5 CPU experts ran single-threaded at
~2.5 GB/s, A/B 2026-07-20) but **reordered** — pipe-ready first, cache-resident
second, sync-miss last — "so a still-loading expert gets its I/O hidden behind
their matmuls instead of head-of-line blocking".

---

## 7. KV / recurrent state vs weights

Clean class separation:

- **Weights (evictable, budgeted):** expert slabs are the only paged class —
  three stores per layer: pinned `pin[]` (startup, history-ranked, NUMA-bound
  arenas `colibri.c:8883-8899`), LRU `ecache[]` (dynamic, `ecap` ceiling), and
  transient `ws[]` swap slabs promoted into ecache at end-of-block
  (`colibri.c:5599-5609`). Dense weights are a separate budget class: CUDA
  dense upload tracked via `g_cuda_dense_projected[]` and subtracted from the
  expert budget *before* placement (`colibri.c:2064-2101, 8967-8971`).
- **KV / state (never evicted, host-canonical):**
  - CUDA: device *shadow* `kv_dev_L/R` + watermark `kv_dev_valid`; host rows
    stay canonical, shadow re-uploads only the tail `[v, upto)`
    (`kv_dev_sync` `colibri.c:3699-3717`); invalidated when a mirrored row is
    rewritten (rewind at `colibri.c:4074-4075`) or the KV buffer is rebound
    (`colibri.c:6471, 6512-6516`).
  - Vulkan: same watermark scheme `vk_kv_valid` (`colibri.c:4292-4310`,
    "rows appended incrementally; watermark, invalidated like the CUDA shadow").
  - Vulkan heap classes make the priority explicit (`colibri.c:8526-8535`):
    experts priority 0.4 (evictable), dense 0.75, scratch/**KV 1.0**, plus a
    `COLI_VK_RESERVE_GB` (default 3) stop on the expert fill so "dense weights +
    KV mirror + staging (measured ~1.7 GB at 4k ctx)" always fit.
  - qwen36: `ensure_kv` grow-only, freed only on growth (`qwen36.c:2064-2090`);
    DeltaNet recurrent state `DN_rec/DN_conv` zeroed per request
    (`reset_recurrent` `qwen36.c:2052-2059`) — always host-resident, never
    paged. Attention score scratch is per-OMP-thread, sized with the KV
    (`qwen36.c:2078-2088`).
  - Persistent KV across serve turns: `kv_persist.h` / `.coli_kv`; MLA
    compressed latent+rope rows (57x smaller state, README).
  - The RSS guard (#403, `colibri.c:7647-7689`) evicts **only** LRU ecache
    slots — pins, dense, KV are never touched; `resident_bytes` counts only
    pin+dense, never LRU, to avoid double budgeting (`colibri.c:8943-8951`).

---

## 8. Lessons mapped to Insignia's 25.65 GB FP8 / 64-layer mission

1. **Steal the LFRU score verbatim** — one u64, no floats, comparable across
   heat and recency; the `+25%+4<<8` hysteresis is what makes dynamic swaps
   stable. Colibri burns it in three places (repin, pilot-evict guard, VRAM
   tick) from the same two arrays (`eheat`, `elast`+clock).
2. **"1 sync per device" is cheap and big** — measured ~70% of a sync group
   call is host-side wait (`colibri.c:5502-5503`). Issue the whole layer's
   resident expert set async, do the CPU/RAM tier in between, take once.
   Insignia's token-serial decode has the same shape (per-token = sum of tiers,
   so overlap is the only lever).
3. **Overlap windows are ordered deliberately**: slower device issued first
   (VK), shared expert + CPU misses computed between issue and take (qwen36),
   next-block readahead during current-block matmul, pilot router eval on
   layer N's residual for N+1..N+3.
4. **Depth-1 MTP only, and beware the residency interaction**: ~85% acceptance
   at depth 1 vs 44-62% at 2-3 on GLM-5.2; deeper verify batches route more
   experts and evict the working set at partial residency (32% loss at 85%
   expert hit). Draft/verify must run the *same kernel family* or acceptance
   collapses (#163). Insignia's MTP draft layer should be pinned kernel-stable.
5. **Staging cap min(4GB, tier_budget/8)** directly transfers to Insignia's
   16 GB RAM box uploading a ~10 GB VRAM tier from a 25.65 GB model: peak host
   RSS = one round, not the whole prefix.
6. **Warmstart everything before token 1** (all experts to RAM, hot to VRAM,
   heat persisted across sessions via HEAT_FILE/.coli_usage) — cold first
   touches cost ~12 ms each and destroyed a measured run (139 ms/token).
7. **CPU tier is a first-class compute tier, not a fallback**: int8/IDOT
   kernels with internal OMP, XEXP single-parallel-region two-barrier MoE
   (`colibri.c:5183-5249`), order-by-readiness to hide remaining I/O.
8. **KV/state classes must be un-evictable and budget-reserved** (priority
   class + explicit reserve GB) or a growing context silently starves the
   expert tier.
