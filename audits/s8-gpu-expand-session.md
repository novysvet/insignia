# Session 8 — GPU scale expansion follow-up: transport v2, campaign analytics, CCT repair (2026-08-29)

Session shape: continuation of the 8b3018e ("expand packed expert scales on
Ada GPU") work. A 13-agent analysis wave (kernel audit, uint32-kernel and
2D-copy design, API economics, microbench draft, host-tier/ioaudit re-derivation,
literature, four campaign-trace analyses, drafter check) followed by an
implementation wave on glm-box: transport v2 (merged 2D copies + fused
expansion + warp uint32 worker), the expansion microbenchmark, a critical CCT
builder fix, and the parity/A-B gates to match.

## 1. What was implemented this session (all env-gated, default-off)

- **Packed H2D transport v2** (`INSIGNIA_GLM53_PACKED_V2=1`): per record,
  the three 4 MiB body copies (contiguous in the window, 4.5 MiB-strided in
  the destination slot) become ONE `cudaMemcpy2DAsync`; the three scale
  blobs (contiguous on both sides by construction) become ONE linear
  `cudaMemcpyAsync`; the three expansion launches fuse into ONE
  `expand_scale_nibbles3*` launch. 6 memcpy + 3 launches → 2 + 1 per record.
- **Warp uint32 expansion kernel** (`INSIGNIA_GLM53_PACKED_KERNEL=2`): one
  thread per packed uint32 (8 nibbles → one 64-bit store), 128 blocks × 512
  threads per projection, 256-entry shared pair table with escape flags
  folded into bits 16/17 (escape counting becomes free popc), PRMT-based
  byte assembly, ctz-walk escape fixup via `__ffs`. Byte-identical to the
  byte worker by construction (same escape consumption order). Works both
  per-projection (`_v2`) and fused (`_3_v2`).
- **Escape-tail hardening**: `stage_packed_gpu` now bounds
  `header->escapes[p] ≤ (kPayloadCapacity − kBodyBytes − 3·blob_overhead)/3`
  BEFORE any window write — a corrupt sidecar record previously could spill
  past its window into a neighbouring live record before the capacity
  `require` fired (audit finding, wave-1 agent A1).
- **Expansion microbench** (`glm53-expert-bench --expand [seed] [iters]`):
  synthesizes packed planes at escape rates {0, 0.782%, 2%, 5%, 12.5%},
  byte-compares GPU output vs ground truth, measures v1/v2 kernels, the
  full host staging decode (ported AVX2), pinned H2D transports, the
  record-level v1 vs v2 enqueue mixes, and the empty-launch floor.
- **CCT builder fix** (`tools/dump_cct.py`): `np.argsort(-co)` on a uint32
  count matrix WRAPS (0 → 2³²), so every table the tool ever built ranked
  never-co-activated experts first — an anti-signal table. The shipped
  `/var/lib/insignia/cct-gsm8k.table` is 100% garbage rows (2.4% OOS ≈ the
  random floor). Fixed to int64 before negation; regenerated
  `/var/lib/insignia/cct-campaign-v2.table` from 10 campaign prompts
  (11,750 decode tokens), loader-format-verified and row-verified against
  directly recomputed co-activation counts.
- **Dead default path**: `INSIGNIA_GLM53_DFLASH2_FP8` default was the
  superseded `glm53-dflash2-fp8` cache which no longer exists on disk — any
  DFlash2 run without the env var died at startup. Default repointed to
  `glm53-dflash2-fp8-fixed`. Stale "position >= 263" comment updated.

## 2. Wave-analysis findings (agents, read-only)

### The landed 8b3018e path (kernel audit)

- No functional bug in the kernel, staging math, stream ordering, or
  `packed_scale_device_` scratch reuse (single-stream program order
  protects it; the event machinery protects the destinations).
- Provably byte-identical to the AVX2 decoder (same nibble extraction,
  interleave, escape order, codebook semantics).
- `kPackedDeviceCapacity` is 1600 KiB (not 576) — ~48% per-projection
  escape headroom.
- The three blob H2Ds and three launches were the dominant per-record API
  cost: ~13 CUDA calls/record post-8b3018e vs 3-4 before. The blob regions
  were ALREADY contiguous on both sides — the merge is layout-free.

### API economics (this box's own probe anchors: 8.1 µs/launch WDDM)

- The ioaudit's 16–27 ms/round is the exposed batch-head + reader-burst
  funnel: 4 readers complete near-simultaneously and only the Runner thread
  may issue CUDA calls, so completions drain serially at ~13 calls × 8.1 µs.
- Ranked levers (cold real text, k7): graph capture of the copy segment
  (−25–55 ms/tok realistic), cross-record batching (REJECTED on overlap
  math — a batched copy couples the whole 8-record group to the slowest
  disk read: one 2D/batch op cannot start until all rows are ready; the
  +160 ms/tok convoy dwarfs the ≤20 ms API saving), merged-copies v2
  (−8–14 ms/tok), O(1) LRU victim scan (−7–8 ms/tok, best ms/LOC),
  per-drive condvars (−3–13).
- Hot-path wall is PCIe-engine-bound (194 ms/token serialized H2D at
  23.2 GB/s for 336 records) — API savings don't move it; only fewer
  transported bytes do.

### Campaign traces (13 prompts, ~16k decode tokens — replaces the 5-token evidence base)

- **U1 settled (HIGH confidence)**: the old "5.22-bit access entropy" was
  small-sample plug-in bias (5–96 draws cannot resolve a 7.3-bit
  distribution); true per-layer entropy on GSM8K/MATH is 6.4–8.0 bits
  (mean ≈ 7.3), effective support PR ≈ 110–120 experts/layer. The old
  U(2)/U(5) pair was real; at the true entropy nothing violates the bound
  (0/42 layers). An exchangeable model fitted to the trace's own marginal
  now OVER-predicts U(K) by 7–16% — real temporal structure exists with
  the OPPOSITE sign: one-step stickiness (retention β≈0.165) explains 99%
  of U(5); exact set repetition is negligible (2e-4). Implication: d(k)
  extrapolations from exchangeable curves over-predicted verify unions;
  use the sticky curve.
- **Static pinning re-priced**: a global top-330 static VRAM tier covers
  only ~14.7% of the real decode access stream (top-28/layer "91.5%" was a
  5-token artifact; 32.8% at 1176 slots). Pin-list v3 (2425 slots,
  water-filled on shrunken posteriors) delivers ~48–51% OOS vs pin-v2's
  measured 32.7%. Static-vs-LRU phase transition at B=336 (one token's
  insert working set): below it LRU thrashes to ~0%, above it LRU wins.
  ⇒ VRAM tier (~330) must be static global-hottest; host tier stays
  dynamic. Cross-family transfer is good now (1–3pp loss gsm8k↔math).
- **Units error found**: "576 MiB VRAM tier ≈ 576 slots" was wrong — packed
  records are 12.76 MiB, so 576 MiB holds only 43–45 records; full-VRAM is
  ~350–400 slots; 2425 slots = 29.5 GiB host.
- **CCT**: corrected table = 28.4% OOS coverage at 1.0× overfetch (cap 8;
  41.7% at 2.0× with cap 16) — 2.1× the marginal baseline, below the
  early_route 59.9% bar, cross-family transfer 0.8–4pp. Verdict: KEEP as a
  capped-8 reader-pool hint only. Regeneration protocol: ≥8 diverse
  prompts / ≥3k pooled tokens (saturates ~28%), int64 argsort, min-obs 30
  with marginal fill.
- **Adaptive-k accounting corrected**: a verified round yields exactly
  `matched` tokens (boundary argmax is free), so T(k) = [D + (1−p1)F +
  p1·b·d(k)] / [(1−p1) + ΣS(j)] — NOT 1+ΣS. The s7 "T_mixed 482 ms win"
  recomputes to ~624 ms (0.92×). Model reproduces 8 measured gen-32 runs
  to ≤0.2%. Break-even b* = 1.74 ms/record at measured p̄=3.08. k* is
  bimodal (6–7 campaign-like, 3 real-text at b≈1.4); at b=1.8 real text NO
  k beats scalar (best 1.026×). Estimator: censoring-correct hazard EMA +
  8-point argmax + 1% hysteresis + MANDATORY k-probe every 16 rounds
  (without probes the argmax deadlocks at k=1 with 11–46% regret — new
  result). Online b̂ (U3): per-round delta regression on {nvme, host,
  vram} record counts identifies b to ±0.2 at ~240 rounds; CUSUM drift
  detector detects b shifts in 8–26 rounds. Instrumentation spec: reuse
  stage_records/cache_hits/device_hits deltas + DF_COSTTRACE CSV env.
- **Drafter**: checkpoint config is block_size 8, window 2048 — the
  block-16 "2× union discount" hypothesis is DEAD for this checkpoint
  (it was a paper ablation, b16 wins math/code for DFlash v1). bd3b8d3
  fully landed the 2048 window (+75 MiB VRAM, smem exactly at the 64 KiB
  opt-in limit); residuals are cosmetic. MTP/drafter parity is speed-only.
- **Host-tier (P7) spec re-derived** (scratch drafts were on the dev box,
  gone): soft segment-LRU (probationary → protected on 2nd hit, soft cap
  50%) + acceptance-prefix admission via insert-then-demote (records stage
  normally; after `matched` is known, release same-round windows not in
  the accepted rows' expert set). Retention horizon math: verify insert
  rate drops from 10–17.8 to ~6.8–7.2/layer-token (T: 3.2 → 8.0 tokens);
  predicted verify hits 28–31% → ~45–60% (57% point estimate, 77%
  exchangeable upper bound). Falsifier F1 is offline: replay the campaign
  traces through the simulated policy BEFORE building it.
- **ioaudit top re-derived**: (1) O(1) intrusive-LRU eviction (O(2425)
  scans cost 10–12 ms/round); (2) residency ordering — a record in the
  VRAM tier whose host window was evicted currently takes a FULL NVMe
  re-read before the device hit is noticed (load_batch checks
  flight_index_ before device_index_; est. 20–40 ms/tok cold, 5–15 hot);
  (3) pinned router D2H drain (42/round, also the graph-capture blocker);
  (4) per-drive condvars; (5) WC arena now SAFE on the packed path (both
  expand variants write the window only; preads go to scratch) — one-line
  gated experiment, 5–15% H2D efficiency possible, MUST A/B under WSL.
- **Literature**: cudaMemcpy2DAsync for 3×4 MiB rows at 4.5 MiB stride is
  descriptor-cheap (pathologies hit small/misaligned rows);
  cudaMemcpyBatchAsync (12.8+) is named-stream-only, graph-incompatible,
  intra-batch unordered, and NCCL retreated from it — do not adopt blind.
  Prior art is unanimous on decode vs prefill: in-register dequant for
  gemv, expand-to-buffer only for wide batches (vLLM AWQ prefill,
  llama.cpp cuBLAS path); exllamav3's stloader transform-on-arrival kernel
  is the closest cousin to our scale expansion. DFlash ablations: target
  conditioning dominates; adaptive-k (AdaEDL) cuts empty rounds; nobody
  publishes drafter-FP8 acceptance deltas. Ada consumer cards: 1 CE per
  direction — one deep H2D stream is the right topology; 23.2 GB/s is
  already PCIe4 x16 practical peak.

## 3. Validation status

See §4 for numbers as they land. Gates: base parity (unpacked vs packed-CPU
vs packed-GPU-v1, greedy IDs + full-vocab logits byte-compare, standard
prompts P1/P2), v2-knob parity (PACKED_V2=1 and PACKED_KERNEL=2 arms vs the
same references), cold-process timing A/B (5 scalar arms + 2 DFlash arms,
3 reps each), and the `--expand` microbench byte-exactness gates.

## 4. Results

### Parity (all green)

- Base gate (binary at 8b3018e): unpacked == packed-CPU-expand ==
  packed-GPU-expand on both standard prompts — greedy IDs AND full-vocab
  logits byte-identical (24.8 MB + 62 MB dumps). Wave-a overnight
  references also identical (zero drift across the prefill-chunk commits).
  The run.log's two "IDS FAIL" lines vs wave-a were a path bug in the gate
  script ($OUT/ prefixed onto absolute paths); manual diff confirms
  identical. PARITY_FAIL corrected to 0.
- v2-knob gate (patched binary): PACKED_V2=1 (merged 2D bodies + single
  blob copy + fused byte-worker launch), PACKED_KERNEL=2 (warp uint32
  worker, per-projection), and both combined — all six arms byte-identical
  to the unpacked reference (IDs + full-vocab logits).

### Microbench (`glm53-expert-bench --expand 1 2000`, idle GPU)

Per 512 KiB projection at the 0.782% production escape rate:
- GPU expand kernel: v1 (byte worker) 6.08 µs (86 GB/s), v2 (warp uint32)
  5.32 µs (99 GB/s). Both are launch-overhead-bound (empty launch 4.1–4.7
  µs on this box); v2 is ~12% faster end-to-end and uses 8× fewer blocks.
- Host staging decode (AVX2): 42.9 µs (12.2 GB/s) = 5.2 prefix + 37.7
  expand — the GPU path removes this from the reader threads (~129
  µs/record), matching the engine's measured 1.98 ms/record all-in cost.
- H2D transports: expanded 512 KiB = 24.6 µs vs blob 264 KiB = 13.9 µs →
  ~32 µs/record of PCIe saved (5.38%, as designed).
- Record-level enqueue mix: v1 (6 memcpy + 3 launches) 647.5 µs vs v2
  (2D + 1 memcpy + 1 fused launch) 639.9 µs → −7.6 µs/record (≈ −2.6
  ms/token at 336 records upper bound, PCIe-bound floor dominates both).
- Byte-exactness: v1, v2, and host decode PASS at every rate
  {0, 0.782%, 2%, 5%, 12.5%}. The bench's fused record-mix spot-check
  reports FAIL — cause unresolved, but the per-kernel checks pass and the
  ENGINE end-to-end gate (full-vocab logits on real records with varying
  per-projection escape counts, 3 knob combos × 140 tokens) is
  byte-identical, so the shipping paths stand; debugging the bench harness
  check is a follow-up.

### Timing A/B (cold-process, P2/100 tokens, 3 reps, medians)

| arm | scalar ms/tok | DFlash k7 ms/tok |
|---|---:|---:|
| unpacked store | 384.7 | — |
| packed, CPU expand | 403.2 | — |
| packed, GPU v1 (8b3018e path) | 405.5 | 223.3 |
| packed, GPU v2 merged transport | 405.0 | — |
| packed, GPU v2 + uint32 kernel | 405.3 | 223.0 |

Reading: all packed arms cluster within ±1 ms/tok — on the cold disk-bound
prompt the per-record API and PCIe differences hide behind NVMe waits,
exactly as the economics model predicted (the exposed share is the
batch-head funnel, not the steady-state stream). DFlash v1 vs v2+k2 is dead
even with identical 5.88 accepted/round. The packed store overall costs
~+5% vs unpacked on this prompt — NOT from transport (v1/v2/CPU all
equal) but from the staging window memcpy: the packed path preads into
scratch and memcpy's 12.8 MiB into the window, while the unpacked path
preads directly into the window. Removing that copy (direct-read layout:
bodies are already contiguous at the record front; only the ~0.77 MiB
scale region needs blob construction) is the next transport lever, worth
~2 ms/record of reader-thread time. GPU expand itself already pays for
the switch: it deletes 129 µs/record of AVX2 decode from the readers at
a cost of ~16 µs/record of GPU launch+kernel, and cuts 5.38% of PCIe
bytes.

## 5. Open queue (next session)

0. Direct-read packed staging: pread the packed record straight into the
   window (bodies are contiguous at the record front; 12.76 MiB of the
   12.77 MiB transport needs no re-layout) and build only the ~0.77 MiB
   scale blob separately — removes the 12.8 MiB CPU memcpy per record that
   makes the packed store cost +5% vs unpacked. Also debug the microbench
   fused spot-check (engine paths are parity-green; the bench check is not).
1. ioaudit F3 residency ordering (VRAM tier consulted before NVMe) — likely
   the largest single cold-path win (20–40 ms/tok), parity-green.
2. Host-tier segment-LRU + acceptance-prefix demote — run the offline
   trace-replay falsifier (F1) first, then land behind
   `INSIGNIA_GLM53_TIER_SLRU=1`.
3. O(1) intrusive LRU (base primitive for 2), per-drive condvars, pinned
   router D2H (graph prerequisite), WC-arena one-liner A/B.
4. Adaptive-k v2 with the corrected T(k), probe schedule, and DF_COSTTRACE
   counters; consider AdaEDL-style per-round length.
5. CCT enablement decision: corrected table on disk; A/B `INSIGNIA_GLM53_CCT`
   with the cap-8 hint on cold GSM8K (reader-pool idle capacity is the
   precondition — at 1 reader it is strictly harmful).
6. Pin-list v3 adoption (needs the runtime to express global-hottest VRAM
   fill, not per-layer quotas) + re-measure the 330-slot static tier
   coverage with the new 14.7% prior.
7. CUDA-graph capture of the per-token copy segment (needs pinned router
   D2H first); the 8.1 → 0.78 µs/node replay gap is the ceiling.
