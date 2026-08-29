# P7 — host-tier admission under bursty verify unions: analysis + A/B plan

Scope note: no engine runs were performed for this document. All code refs are
`src/glm53_generate.cu` at the current working tree; all measured numbers are
quoted from `audits/s6-open-problems.md` section 0, `audits/mla-latent-session.md`
section 2, and `progress.md` (session 4/6 entries).

## 1. What the code actually does (line refs)

Host tier = `ExpertStager` (src/glm53_generate.cu:547).

- Slots: `window_count_ = clamp(host_cache_bytes / kWindowBytes, 64, 4096)`
  (line 560); `kWindowBytes = 13.504 MiB` (549-554) so the 32 GiB default
  (line 1941, `INSIGNIA_GLM53_EXPERT_CACHE_MB`) = **2425 slots**.
- Halve-retry: constructor loop lines 561-573 — on `cudaHostAlloc` failure the
  attempt count is **halved** (`attempt /= 2`) and retried down to 64 slots.
  A 40 GiB request that fails allocates **20 GiB = 1480 slots**, silently
  (only visible in the final stats line's slot count, line 3985-3989).
- Eviction: `take_window()` (1249-1293) — free list first, else evict the
  minimum-`stamp` completed/unclaimed/unpinned window = **global plain LRU**
  across all 42 layers. The code comment (1256-1260) records that
  frequency-based eviction was measured worse and pure LRU wins.
- Pin exclusion: windows loaded from `INSIGNIA_GLM53_PIN_LIST` get
  `state.pinned = true` (line 679) and are skipped as victims (1264-1266);
  default 8/layer host + 2/layer VRAM keys (lines 649-653).
- Lock granularity: one `pool_mutex_` for the whole stager (queues, done
  flags, drive cache). Readers hold it only to dequeue / mark done
  (1335-1361); I/O runs outside the lock. At ~4 readers x ~70 records/s this
  is not a factor in the mystery.
- TinyLFU-style door: exists (lines 751-769, `INSIGNIA_GLM53_ADMIT/N`,
  default **off**), compares `sight_count_` (8192-key FIFO, lines 715-724)
  against the coldest resident's `hits`. History: measured harmful
  (progress.md admission-control saga) and left disabled.
- **Scalar decode inserts** (`Runner::step` -> `load_batch(layer, {8 experts})`,
  line 2903): populate defaults true, mask 0xff -> every demanded record is
  admitted, door off. Lookup stream = 8/layer/token; `cache_hits_` counts
  completed-resident matches only (731-736, 771-772).
- **Verify inserts** (`moe_multi` in the kda_archive_ branch, i.e. both
  `verify_round` batch mode and `verify_token` sequential mode — both call
  `prefill(..., capture=true)`, lines 3217/3238, which sets `kda_archive_`,
  line 3687):
  1. `stage_layer(layer, distinct, n)` (line 3489 -> 786-800) starts reads
     for the **whole deduplicated union** at demand priority. It consults no
     admission policy — it only skips keys already resident/in-flight.
  2. The union is consumed in first-seen batches of 8 via `load_batch(layer,
     batch, n, populate, populate_mask)` (line 3599). Because `stage_layer`
     already put every missing key in flight, `load_batch` takes the
     `resident != flight_index_.end()` branch (726-745): **the TinyLFU door
     at 751-769 never executes for verify traffic** — `batch_admit_` keeps
     the `fill(true)` value from line 702.
  3. Retention quota (3563-3585): `quota = (cache_slots() - 16) / 42`, and
     `retained` fills from consecutive verify positions starting at
     `chosen_token` (default 0). Non-retained records are released right
     after upload (875-891).
  4. **At 2425 slots the quota is 57/layer and never binds**: U(7) ~ 41 < 57.
     The comment "the default 379 slots retain 8 experts/layer"
     (3558-3560) and "far beyond the 379-record host tier" (3688-3690) are
     stale text from the 5 GiB era, where quota was indeed 8/layer
     ((379-16)/42 = 8). **In the current regime every verify-union record,
     including the whole rejected draft tail, is admitted.**
- Acceptance context: `main()` skips a round entirely when d1 != truth0
  (4166-4183), so **position 0 of every executed round is guaranteed
  accepted**; acceptance beyond that is EMA-tracked (4122, adaptive k at
  4150-4153). Batch verify processes all k rows before acceptance is known
  (3215-3227 + 4213-4217); sequential mode (`DF_SEQ_VERIFY`, 4192-4209)
  stops at first mismatch and never reads the tail.

## 2. The 80.3% vs 28-31% mystery — mechanism

Provenance correction first: the 80.3% figure is from session 4 on the
**campaign prompt** (DFlash2 k4 at 2425 slots; audits/mla-latent-session.md:29,
cache-slot sweep 488->50%, 999->72%, 1819->79.5%, 2425->80.3%, flat after).
The 28-31% is session 6 on **real GSM8K/MATH-500-class text**. s6 P7 calls
the 80.3% "scalar decode"; the sweep shows scalar plateaus at the same ~80%,
so the mode confound is real but secondary — the dominant confound is prompt
repetitiveness (campaign acceptance 3.7-4.0/round vs real-text 2.64).

### 2.1 The insert-rate / horizon law

For a global LRU of C slots over 42 layers, the per-layer retention horizon
in committed tokens is

    T = (C / 42) / r_ins ,   r_ins = records admitted per layer per token.

A key with per-token pick share p is re-referenced about every 1/(8p) tokens;
it stays resident iff 1/(8p) < T, so the hit rate is M(p > 1/(8T)) where M is
the per-layer coverage curve (measured: top-1 9.46%, top-8 41.07%, top-28
91.49%).

- **Scalar steady state** (self-limiting): r_ins = 8(1-h); at h = 0.803,
  r_ins = 1.58, **T = 57.7/1.58 = 36.6 tokens** -> p_min = 0.34% -> deep
  inside the top-28 (91.5% mass) -> ~80% after the nonstationarity discount.
  Matches the documented plateau and its saturation (more slots buy only the
  drifting cold tail: "flat after").
- **DFlash2 k4 on real text**: every union record is admitted (Section 1.4),
  so r_ins = U(4)/E[M] = 26.40/2.64 = **10.0 records/layer/token** ->
  **T = 57.7/10 = 5.8 tokens** -> p_min = 2.2% -> about the top-8 boundary
  (41% mass) minus burst jitter and draft-noise -> **28-31% measured**.

The horizon collapses by **6.3x**; one number explains both measurements with
no free parameters. Equivalently, in slot units: one k4 verify union is
42 x 26.4 = 1109 records = 46% of the whole tier; cross-round reuse must
survive >= 1 full union (1109) and never 3 (3327 > 2425) — exactly the 2-round
/ ~5.3-committed-token horizon above.

### 2.2 Verdict on the two candidate causes

- **"Union bursts evict the scalar working set"** — YES, but the precise
  mechanism is insert-rate aging, not dramatic eviction: the burst is
  admitted unconditionally (quota non-binding, door bypassed via the
  in-flight branch), and each insert consumes one LRU position, so the hot
  set's recency depth shrinks 36.6 -> 5.8 tokens. The hot set is not gone;
  its ranks 9-28 (50% of mass) simply cannot re-reference in time.
- **"Per-layer quota caps admission at ~8/layer when the union needs 26+"** —
  NO in the current regime: (2425-16)/42 = 57 > U(7) ~ 41, so the quota binds
  nothing at k <= 7. It did bind at the 379-slot tier (quota = 8); the
  in-code comments describing that regime are stale (lines 3558-3560,
  3688-3693).

### 2.3 The 40 GiB -7%

Two stacked causes, both grounded:

1. **Halve-retry collapse (likely dominant).** With the parallel session
   pinning 32 GiB (s6 non-math blockers record exactly this hazard), a
   40 GiB `cudaHostAlloc` fails; the constructor halves to 20 GiB ->
   **1480 slots** -> T = (1480/42)/10 = 3.5 tokens -> hits collapse further.
   The run that "measured 40 GiB" may have actually run a 20 GiB tier; the
   final stats line's slot count (3985-3989) is the check to run.
2. **Even when the alloc succeeds, +536 slots buy almost nothing without a
   policy change**: T moves 5.77 -> 7.05 tokens (+22%), worth a few hit
   points in-model, while adding pin/TLB pressure on a box whose WSL guest
   has 60 GiB. Meanwhile scalar marginal value is flat because scalar sat
   on the coverage plateau already (sweep: 1819 -> 79.5%, 2425 -> 80.3%).

Capacity knee arithmetic: serving both streams wants C >= pins (336) +
~2 burst residues ((1-h) x U(k) x 42; ~610 at k4, ~940 at k7) + hot set
(28/layer = 1176) ~ 2730-3450 slots. 2425 is 11-30% below the knee — which
is exactly why slot count alone flails while an admission policy that halves
the effective residue (prefix cap: U(3)/U(4) = 0.78) or shields the hot set
(SLRU) is worth more than the RAM.

## 3. P7 formalization

Per sparse layer l, committed-token time n:

- Token-routing process: S_n^l in 288 choose 8, marginal p_l with entropy
  4.54-5.22 bits, adjacent overlap I/8 = 0.19-0.27 (repetitive half 0.266).
- **High-locality stream** (scalar / fallback / accepted prefix): requests
  X = S_n^l, rate 8/layer/token.
- **Bursty stream** (batch verify round r, k positions, acceptance M_r,
  E[M]=2.64): requests = the deduplicated union
  U_r^l = union_{j<k} S^l_{t_r+j} (accepted-prefix positions, real-text
  routing) **joined with** draft-noise sets D^l_{r,j}, j >= M_r
  (drafter-conditioned routing: correlated with text but off-manifold,
  effectively never re-referenced). Measured |U_r^l| = U(K):
  14.45 / 20.61 / 26.40 / 31.40 for K = 2..5; |U(7)| ~ 41.
- Cache: C = 2425 global slots, admission policy A, retention R.

Objective: maximize combined saved I/O = 13.56 MiB x (hits) — equivalently
weighted hits h = (lambda_s h_s + lambda_v h_v) with lambda by lookup volume —
subject to determinism (A, R may not alter FP order; they only choose
residency of byte-identical payloads).

Optimal-policy structure (why the winners win):

- An admission decision at insert time has two information channels:
  **history** (has this key been re-referenced before?) and **context**
  (which stream / which in-block position issued the request?). The
  request identity alone is uninformative at first sight.
- **TinyLFU door** uses count comparisons; under near-uniform routing all
  counts converge, the door congeals (measured: 9.4% hits). Rejected.
- **Per-layer quota raise** is a no-op (57 >= U(7)). Rejected.
- **Segment LRU / 2Q (Megiddo-Modha)** separates a stationary hot component
  from transient scans with the *minimal* evidence rule: one re-reference
  promotes; no counts, no comparisons; eviction inside each segment stays
  pure LRU (preserving the measured "plain LRU wins" property). Scalar-only
  steady state degenerates to plain LRU automatically (probation drains),
  so h_s = 80.3% is preserved by construction.
- **Acceptance-prefix admission** uses the context channel, which is free
  information the cache is currently ignoring: position 0 is guaranteed
  accepted (truth0 pre-check), positions < ~E[M] are the likely-real prefix,
  positions beyond are likely-rejected draft tail. Capping admission at
  ~ceil(E[M])+1 positions cuts burst residue by U(k)/U(cap): at k4/cap3 =
  -22% inserts; at k7/cap5 = -23%.
- **Frequency-decay aging**: rejected — same family as the measured-bad
  frequency eviction, and it re-introduces the congealing failure mode.

Fixed points (estimates; the model over-predicts plain-LRU verify hits
~45% vs 28-31% measured, so treat as upper bounds and validate on trace):

- Plain LRU (today): h_v ~ 28-31% (measured).
- Soft SLRU: probation population ~ unre-referenced residue of the last
  ~2 bursts (~610 at k4) + in-flight; protected ~ C - 610 - 336 pins ~ 1479
  (35/layer, enough for the top-28 hot set = 91.5% mass); promotion requires
  one re-reference within the probation age (~1 round ~ 2.6 tokens), so
  ranks 1-8 promote nearly always and mid ranks partially ->
  **h_v ~ 45-60%**, h_s unchanged.
- SLRU + prefix cap 3: residue 0.55 x 20.6 x 42 ~ 476; probation age +~28%;
  k7 case stronger. Combined is the patch below.
- Sequential verify (existing knob) already avoids tail reads entirely
  (union = 8/position); its cost is one 45-layer forward per position. The
  adaptive switch selects it when accept_ema < 0.7k — on real text
  (2.64 < 2.8 at k4) it is frequently selected already; the policy patch
  mainly rescues **batch** mode, which wins when acceptance is high.

## 4. Patch sketch

`scratch/host-tier/p7-slru-prefix.patch` (unified diff, NOT applied; src/ is
read-only for this task). Delta: ~30 lines in `ExpertStager` + the retention
loop in `moe_multi`:

1. `WindowState::probation` flag; `start_read` sets it; a completed-resident
   hit in `load_batch` clears it (promotion = one re-reference).
2. `take_window` victim scan keeps two LRU ranks (probation / protected) and
   prefers the probation victim; protected is touched only when probation is
   dry. Pins/claimed/releasing exclusions unchanged. Eviction stays
   stamp-pure inside each segment (no frequency, no min-hits — the measured
   failure modes are avoided).
3. `INSIGNIA_GLM53_TIER_SEGMENTS=0` disables (plain-LRU control arm).
4. Retention loop bound by `INSIGNIA_GLM53_VERIFY_RETAIN_POS` (default 0 =
   current retain-everything behavior); fills from `chosen_token` exactly as
   today, so position-0-first ordering is preserved.

Determinism argument: admission/retention only decides which windows stay
resident after upload. The verify GEMV chain is driven by the `distinct`
first-seen order and per-expert `users`/`out_ids`/`combine` lists (3600-3649)
— none of that moves. Payload bytes are identical whether read fresh or hit
(byte-verified store, deterministic AVX2 scale expansion), so greedy IDs and
top-10 logits cannot move. The acceptance histogram is therefore also fixed
(drafter inputs unchanged). Still gated empirically per the determinism law.

## 5. A/B design

Box: glm-box only (dev box lacks the RAM). Build: `build/glm53-gen.sh`
inside WSL Arch. Always `pgrep -af glm53-generate` first (parallel-session
pinning corrupts both arms and can trigger the halve-retry). Record the
printed slot count every run; discard any run where 2425 was not delivered.

Arms (same binary, env-switched; ABAB-interleaved, 3 repetitions each,
medians — run-to-run swing is ~2x):

| arm | TIER_SEGMENTS | VERIFY_RETAIN_POS | verify mode |
|---|---|---|---|
| A0 baseline | 0 | 0 | adaptive (shipping) |
| A1 SLRU | 1 | 0 | adaptive |
| A2 SLRU+cap | 1 | 3 | adaptive |
| A3 cap-only | 0 | 3 | adaptive |
| B0/B1/B2 | mirror A0/A1/A2 | — | DF_BATCH_VERIFY=1 (the regime the policy targets) |
| S | 0 | 0 | DF_SEQ_VERIFY=1 (reference) |

Common env: `EXPERT_CACHE_MB=32768`, `DFLASH2=1`, `DFLASH2_FP8=
/var/lib/insignia/glm53-dflash2-fp8-fixed`, `PIN_LIST` both on and off
(separate sub-arms), CCT off, PREFETCH off for the clean read.

Workloads:
- GSM8K pilot 10 cases x 100 gen tokens (tools/benchmark_math.py) — the
  established pilot; then 40 cases for the winning pair.
- MATH-500 half, k4 and k7 (verify_k) — k7 stresses the burst hardest
  (U(7) ~ 41/layer).
- Campaign prompt 100-240 tokens for continuity with the 80.3% / 187.7 ms/tok
  documented numbers.
- Parity gate every arm: standard 12-token prompts (greedy IDs + top-10
  logits digit-identical) + 30/40/100/240-token sequence checks. Any
  divergence: reject the arm regardless of speed.

Metrics per arm: ms/token median; host-tier hit % (engine print); verify-round
wall; accepted/round histogram and empty-round fraction (must be IDENTICAL
across arms — they are the built-in determinism canary); NVMe bytes
(io_bytes). Success criteria: verify-mode tier hits 28-31% -> >= 45% at
unchanged acceptance histogram, and >= 8% ms/token improvement in at least
one verify-mode arm with parity clean. If A2 > A1 > A0 but A3 ~ A0, the
history channel is doing the work; if A3 alone moves, the context channel
suffices and SLRU can stay off (smaller behavioral delta ships).

Pre-validation before GPU time (offline, no engine changes): replay a
ROUTE_TRACE from a real-text run through plain LRU vs soft-SLRU vs
SLRU+prefix simulations per layer (extend the existing
tools/glm53_route_analysis.py simulator on the box; draft-tail positions in
verify rounds are identifiable from DF_DUMP round boundaries). The sim
predicts the hit delta; only arms with simulated delta >= +8 points earn
GPU runs.

Rollout note: if A2 wins, wire VERIFY_RETAIN_POS to ceil(accept_ema)+1
(one line in main() next to the adaptive-k clamp, 4150-4153) instead of a
static env value, and re-run the parity gate.
