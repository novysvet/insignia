# Handoff 01 — byte-packed variable-size pinned expert cache

## Objective

Determine whether the host tier can retain packed-v2 expert records at their
actual byte sizes instead of reserving one expanded-size fixed window per
record. The design must remain exact and must upload directly from pinned
storage on a hit. A design that inserts a host-to-host unpack/copy on every hit
has missed the objective.

This is a cache-layout and allocator task, not a new replacement-policy task.
Use the current production request order and compare like-for-like LRU/SLRU
policies.

## Repository and fresh-agent setup

- Repository: <https://github.com/novysvet/insignia.git>
- Branch: `glm53-dflash2-4070ti-super`
- Dispatch HEAD: `e48f633430c679ac6a30aae248159c887ac41601`
- Read, do not copy: `AGENTS.md`, `progress.md`,
  `audits/s8-gpu-expand-session.md`, `audits/s9-reclaim-session.md`,
  `src/glm53_generate.cu`, and `tools/relayout_glm53_experts_v2.py`.
- Relevant production object: `ExpertStager` in `src/glm53_generate.cu`.
  Inspect `kPayloadCapacity`, `kWindowBytes`, the packed-v2 layout structs,
  `stage_packed_v2_gpu`, `enqueue_record_copy`, event ownership, and the
  intrusive LRU/SLRU lists at the frozen commit.
- Make a fresh clone or worktree at the dispatch HEAD. Do not work in the
  owner's dirty checkout and do not edit production source.

H8 cross-head FP8 decode (`78e1a1c`) and fused H4×Q8 cross-head FP8 prefill
(`e48f633`) are completed production work. This task produces a host-cache
blueprint only and must not modify or reanalyze either MLA path.

## Verified anchors

- Model: 42 sparse layers, 288 experts/layer, top 8, 12,096 records total.
- Each record has three 4 MiB packed NVFP4 bodies: 12 MiB body bytes total.
- The expanded host-window layout reserves approximately 13.5 MiB/record.
- The exact `IG53XPK1` v2 sidecar is 150.77 GiB logical for all 12,096
  records, about 12.765 MiB/record on average. Its measured scale escape rate
  is 0.782%.
- The 32 GiB pinned tier holds 2,425 current fixed windows on glm-box.
  Pure average-size arithmetic suggests about 2,566 packed records before
  allocator/reserve costs: roughly 141 additional records, not a promise.
- Pinned H2D is 23.2 GB/s. The single NVMe is 3.7–4.7 GB/s typical; four
  readers are optimal. Packed-v2 direct-read staging, GPU scale expansion,
  VRAM-before-NVMe lookup, and O(1) LRU are already implemented.
- Session 9 expert VRAM slots were 383 scalar, 281 normal DFlash, and 292
  forced-sequential DFlash before the completed H8 decode and H4×Q8 prefill
  kernels. Do not treat those historical VRAM counts as current host-cache
  capacity.

## Required non-git artifacts

Both files live in the user's copied handoff directory. They are data only;
use repository source from git instead of any source copies in older bundles.

1. `glm53-scale-sample-504-records.tar`
   - SHA-256:
     `e7ef9f702e8ff54dcffccde23a6854d1e0f6f2728bde9b564a604ca0e0db58da`
   - Contains 12 evenly spaced experts (`0,24,...,264`) for every sparse layer
     3–44, 504 records total.
   - Each directory contains the exact 128-byte `XPR1` header and all three
     padded v2 scale regions. `scale-balanced-12x42/manifest.csv` supplies
     geometry and per-file SHA-256.
   - Read `SAMPLE_FORMAT.md` inside the tar before parsing.

2. `s9-campaign-handoff-20260829.tar.zst`
   - SHA-256:
     `819dcdb9e611f73a34c535292fed2a34da4b7fca5924cd6ca7bc69d817c94e56`
   - Relevant members:
     `var/lib/insignia/tracecampaign/merged/route-merged.trace`,
     `route-merged.manifest.tsv`, and `tracecampaign/traces/*.txt`.
   - Trace rows are `token layer e0..e7 s0..s7`. A complete token has exactly
     layers 3–44; discard partial tokens rather than repairing them.

Verify the outer hashes first, then every hash in the scale manifest. Treat
headers, lengths, expert IDs, escape counts, and trace values as untrusted and
bounds-check them before allocation.

## Work

### 1. Measure the real size distribution

Parse every supplied `XPR1` header and scale region. Add the invariant 12 MiB
body and required 4 KiB alignment to obtain per-record packed resident sizes.
Report mean, median, p95, p99, maximum, number of distinct padded sizes, and
variation by layer/projection. Bootstrap by layer because the sample is
systematic rather than random.

Cross-check the sample estimate against the verified full-sidecar average.
Explain any discrepancy instead of silently rescaling it.

### 2. Model address-stable allocators

Compare at least:

- current fixed `kWindowBytes` slots;
- exact 4 KiB-granular extents in one pinned arena;
- a small size-class allocator derived from the observed distribution;
- a compact-entry arena with reserve pools of 8, 16, and 24 concurrent demand
  reads.

An admitted entry's address must remain stable until its final H2D event has
completed. Misses should O_DIRECT-read directly into their final pinned
extent. Hits should issue the existing body/blob H2D operations directly from
that extent. Account for metadata, alignment, free-list storage, fragmentation,
temporarily claimed entries, copy-in-flight entries, and reserve space.

Use a byte-capacity replay, not a slot-count shortcut. Run the scalar trace
exactly and a clearly labelled k7 consecutive-token union proxy for burst
sensitivity. Compare the same eviction/admission rule on both layouts. Report
leave-one-prompt-out and family-held-out results; never fit and score on the
same prompt.

### 3. Produce an integration blueprint

Specify allocation/free state transitions, locks, O_DIRECT alignment,
reader ownership, H2D-event lifetime, eviction guards, error rollback, and
shutdown. Show how current `Layout` offsets remain valid for variable base
pointers. Include pseudocode and an invariant table, not a production patch.

Quantify projected time using both byte traffic and conservative achieved
bandwidth. One percentage point of avoided packed-record NVMe traffic is
roughly 9–10 ms/token at 336 requests/token and 4.7 GB/s before overlap; use a
discrete-event or queueing correction rather than claiming that raw value as
wall time.

## Deliverables

- `verify_inputs.py` and machine-readable verification report.
- `record_sizes.csv` and distribution summary.
- Reproducible byte-capacity replay with unit tests.
- `capacity_results.csv` covering allocator, reserve, fragmentation, prompt,
  and policy.
- `ALLOCATOR_BLUEPRINT.md` with state machine and invariants.
- `REPORT.md` with a clear build/kill decision and uncertainty.
- A manifest and SHA-256 for every new deliverable. Do not include repository
  source or audit copies.

## Gates

Advance only if all are true:

- retained capacity improves by at least 4% after realistic reserves and
  fragmentation;
- leave-one-prompt-out host-hit gain is at least 1.5 percentage points or the
  conservative modeled decode gain is at least 1%;
- p99 internal fragmentation is at most 2%;
- the hit path remains zero-copy on the host and preserves existing H2D byte
  order;
- no workload can evict/reuse an extent before all reads and H2D events finish.

Kill the task if the gain depends on the 5-token legacy trace, on unbounded
compaction pauses, or on a hit-path memcpy. Accuracy is unchanged only if the
canonical device bytes and accumulation order remain identical; prove that
property in the blueprint.

## Forbidden duplication

Do not redesign LRU/SLRU, causal prediction, scale coding, cross-expert body
compression, VRAM reclaim, full layer-major prefill, CPU expert execution, H8
cross-head FP8 decode (`78e1a1c`), fused H4×Q8 cross-head FP8 prefill
(`e48f633`), or other MLA kernels. This task changes representation and
allocation of the pinned host tier only.
