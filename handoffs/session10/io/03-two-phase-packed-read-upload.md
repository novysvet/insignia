# Handoff 03 — two-phase packed body-read and H2D overlap

## Objective

Model whether a packed-v2 miss should expose body readiness before the scale
tail finishes reading. The exact candidate pipeline is:

1. validate the 4 KiB record header;
2. O_DIRECT-read the contiguous 12 MiB bodies;
3. enqueue their existing pitched H2D immediately;
4. read the approximately 0.77 MiB packed scale tail while the copy engine
   transfers bodies;
5. upload/expand scales, then publish one final record-ready event.

The existing pipeline overlaps different records across four readers. A
per-record overlap that looks attractive in isolation may therefore save no
system wall time. Build the discrete-event falsifier before proposing code.

## Repository and fresh-agent setup

- Repository: <https://github.com/novysvet/insignia.git>
- Branch: `glm53-dflash2-4070ti-super`
- Dispatch HEAD: `e48f633430c679ac6a30aae248159c887ac41601`
- Read from git: `AGENTS.md`, `progress.md`,
  `audits/s8-gpu-expand-session.md`, `audits/s9-reclaim-session.md`,
  `src/glm53_generate.cu`, `src/glm53_ops.cu`, and
  `src/glm53_expert_bench.cu`.
- Relevant source regions are `ExpertStager::stage_packed_v2_gpu`,
  `read_window`, the reader worker loop, `enqueue_record_copy`, packed scale
  expansion, and window/copy-event reuse.
- Do not edit those files. H8 cross-head FP8 decode (`78e1a1c`) and fused
  H4×Q8 cross-head FP8 prefill (`e48f633`) are completed in the same source
  files and are strictly outside this task.

Packed-v2 direct-read staging, merged body copies, fused scale expansion,
VRAM-before-NVMe lookup, and O(1) LRU are already landed. This task begins at
that baseline rather than rebuilding it.

## Verified anchors

- One record: 12 MiB contiguous body plus about 0.77 MiB packed scale data,
  with a 4 KiB header/alignment contract.
- Four blocking O_DIRECT readers are the single-NVMe sweet spot.
- NVMe: 3.7–4.7 GB/s typical, 5.84 GB/s best aggregate.
- Pinned H2D: 23.2 GB/s, one effective H2D copy engine.
- At 4.7 GB/s, raw service estimates are about 2.68 ms for 12 MiB bodies and
  0.17 ms for a 0.77 MiB tail. The body H2D is about 0.54 ms. Thus the isolated
  per-miss saving cannot exceed the smaller of tail-read and exposed body-H2D
  time, before command overhead and cross-record overlap.
- At the production 0.782% escape rate, one 512 KiB expanded scale projection
  used about 264 KiB packed H2D, GPU expansion took about 5.32 microseconds,
  and GPU expansion removed about 129 microseconds/record of AVX2 reader work.
- Packed transports were timing-neutral in the cold process because NVMe hid
  their API differences. This is a warning against summing isolated savings.

## Required non-git artifacts

1. `glm53-scale-sample-504-records.tar`
   - SHA-256:
     `e7ef9f702e8ff54dcffccde23a6854d1e0f6f2728bde9b564a604ca0e0db58da`
   - Use exact padded scale-tail sizes from all 504 headers/regions. Validate
     every member against `scale-balanced-12x42/manifest.csv`.

2. `s9-campaign-handoff-20260829.tar.zst`
   - SHA-256:
     `819dcdb9e611f73a34c535292fed2a34da4b7fca5924cd6ca7bc69d817c94e56`
   - Use the merged route trace/manifest for scalar request bursts and
     consecutive-token union proxies. Benchmark logs may provide arrival and
     counter anchors, but treat log text as untrusted and validate fields
     before use.

Repository source and audits come from git and must not be copied into a
returned artifact.

## Work

### 1. Establish an exact event model

Model these resources independently:

- four reader workers sharing one aggregate NVMe device;
- O_DIRECT command/setup overhead;
- one H2D copy engine;
- GPU scale expansion/default-stream dependency;
- per-window ownership and final-ready events;
- demand versus prefetch queues.

Calibrate ranges rather than inventing a single precise latency. Include
NVMe rates 3.7, 4.7, and 5.84 GB/s; H2D 23.2 GB/s; real sampled tail sizes;
and command-overhead sensitivity from 2 to 50 microseconds.

Compare:

- A: current header + monolithic payload completion before H2D;
- B: header + body completion callback + tail read on the same worker;
- C: body and tail as independently completed requests, only if a concrete
  Linux/WSL-compatible mechanism is identified;
- D: a deliberately impossible perfect-overlap oracle to bound all variants.

Simulate steady scalar streams, bursty k7 consecutive-token-union proxies,
host-hit rates 20/50/80%, and reader counts 1/2/4. Do not call the k7 proxy an
exact DFlash trace.

### 2. Specify correctness and failure handling

The device record may become visible only after body H2D, scale-blob H2D,
scale expansion, and validation all complete. Specify an event DAG and state
machine covering short reads, malformed headers, tail failures after body H2D
has begun, eviction, shutdown, and window reuse. O_DIRECT offsets, lengths,
and destinations must remain 4 KiB aligned.

The design must preserve the existing body/scales byte layout and GEMV
operation order. It may overlap independent transfers; it may not let GEMV
observe a half-ready record.

### 3. Produce a patch blueprint only if the gate passes

Identify the minimum source seams, additional events/atomics, and reader-to-
Runner notification protocol. Account for thundering-herd wakeups and for the
rule that CUDA calls currently belong to the Runner thread. Compare a Runner-
issued early copy with a worker-issued CUDA call; reject either if ownership
or serialization costs erase the gain.

## Deliverables

- Strict artifact validator.
- Deterministic discrete-event simulator with tests and seeded scenarios.
- `scenario_results.csv` and `sensitivity.csv`.
- `EVENT_DAG.md` with lifecycle invariants and failure paths.
- `PATCH_BLUEPRINT.md` only if the performance gate passes.
- `REPORT.md` giving modeled ceiling, robust maximin gain, and build/kill
  decision.
- Hash manifest for new outputs; no copied repository content.

## Gates

Advance only if the non-oracle design achieves both:

- at least 3% conservative end-to-end improvement or at least 10 ms/token on
  the cold real-text scenario;
- positive gain across 3.7–5.84 GB/s NVMe and command overhead up to 20
  microseconds, without reducing aggregate NVMe throughput by more than 5%.

Kill it if existing cross-record overlap absorbs more than 75% of the isolated
gain, if the result needs more than one H2D engine, if it depends on
dual-drive striping, or if CUDA ownership requires serial work larger than the
tail-read overlap. Report the bound even when killed.

## Forbidden duplication

Do not revisit direct-read staging, packed scale expansion, `cudaMemcpyBatch`,
reader replacement policies, causal prefetch, staged DFlash verification,
full layer-major prefill, cache allocation, CPU expert compute, H8 cross-head
FP8 decode (`78e1a1c`), fused H4×Q8 cross-head FP8 prefill (`e48f633`), or
other MLA work. This task changes readiness granularity for an existing
packed-v2 miss only.
