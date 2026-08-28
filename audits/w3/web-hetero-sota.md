# web research: heterogeneous / NVMe LLM inference SOTA (w3, 2026-08-25)

Scope: 8 assigned topics, web-only, read-only. Rig context: RTX 4070 SUPER 12GB +
Ryzen 5600X (Zen3, AVX2+FMA, dual-channel DDR4) + 16GB RAM + Gen4 NVMe ~7GB/s,
Qwen3.8-27B-FP8 dense (25.65GB text weights), custom CUDA engine. Cross-checked
against `audits/synthesis.md` tier plan (L≈21 VRAM / M≈23 RAM / N≈21 NVMe,
CPU tier ~10ms/layer, MTP T=2 ×1.6, IOCP NO_BUFFERING reader, shard-major staging).

---

## 1. "PipeLLaMA" — the cited paper does not exist (negative result)

Searched: arXiv API full-text (`all:"PipeLLaMA"`, `all:"PipeLlama"`,
`ti:"Pipelined Fast Large Language Model Inference"`, `abs:"NVMe" AND abs:"double buffer"`,
`abs:"NVMe" AND abs:"inference"` sorted desc), Hugging Face papers, Semantic Scholar,
and general web. **Zero hits for any arXiv paper named PipeLLaMA/Pipellama about
pipelined LLM inference via direct NVMe streaming.** The closest real paper by name is
[PipeLLM: Fast and Confidential LLM Serving (arXiv 2411.03357)](https://arxiv.org/abs/2411.03357)
— but that is AES-GCM encryption pipelining for confidential computing, not NVMe.
The coordinator's citation appears garbled; the *description* (double-buffered weights,
IO/compute overlap, direct NVMe streaming, 70B on consumer rig) matches a real **GitHub
project, not a paper: nTransformer / SLEP**.

### 1a. nTransformer SLEP — the real "pipelined fast LLM inference via direct NVMe streaming"

Source: https://github.com/xaskasdf/ntransformer (+ HN thread
https://news.ycombinator.com/item?id=47104667, coverage
https://news.lavx.hu/article/ntransformer-enables-llama-70b-inference-on-single-consumer-gpu-with-novel-streaming-architecture).

Technique (their exact structure, verified from README):
- **SLEP** (Streaming Layer Engine Pipeline): only a few layers live in VRAM at once;
  two GPU compute buffers (`gpu0`/`gpu1`) + two pinned staging buffers (`stg0`/`stg1`)
  alternate — classic double buffering: while layer N computes in one buffer, layer N+1
  DMAs from pinned RAM into the other (`[L29→gpu0][L30→gpu1][L31→gpu0]`).
- 3 overlapping stages: NVMe read → pinned staging, staging→GPU DMA, compute.
- **3-tier placement**: Tier A VRAM-resident (auto-sized from `cudaMemGetInfo`),
  Tier B pinned RAM double-buffered (H2D DMA only), Tier C NVMe/mmap fallback;
  sizes auto-computed from free VRAM/RAM. Example split 70B Q6_K: 26 VRAM + 54 RAM + 0 NVMe.
- **gpu-nvme-direct**: a userspace NVMe driver (separate repo) where the *GPU* performs
  MMIO doorbell writes directly to NVMe controller queues via VFIO; path is
  NVMe→DMA→pinned staging→PCIe H2D→GPU. 3.35 GB/s achieved; ~670MB layer = 670 NVMe
  commands in ~202ms. Invasive setup (IOMMU off, DKMS patch, VFIO unbind) — **Linux-only,
  impossible on Windows**.

Measured (RTX 3090 24GB, 48GB RAM, WD SN740 NVMe, PCIe Gen3 x8 ≈ 6.5GB/s H2D — the cited
bottleneck):
- Llama3.1 8B Q8_0 resident: 48.9 tok/s.
- **Llama3.1 70B Q6_K tiered: 0.2 tok/s; Q4_K_M: 0.3; + layer-skip: 0.5 tok/s** (23GB VRAM used).
- 83x over their 0.006 tok/s mmap baseline (53GB model > 48GB RAM → page-cache thrash).

Steal list for us: the 2xGPU-buffer+2xstaging ring is exactly our shard-major staging
plan — validated. Their Tier B/C split-from-free-memory logic mirrors colibri. Their
numbers also bound expectations: 70B Q4 (~40GB) at 0.3 tok/s single-GPU ≈ NVMe+PCIe
bound; our 27B FP8 (25.65GB) with 12GB VRAM + ~8.8GB RAM pinned should land ~2-4x better
than their placement because we keep L≈21 layers never moving.

### 1b. Other near-misses found while hunting (real, cited for completeness)

- ladebw/PipeLLM (https://github.com/ladebw/PipeLLM): poster-titled "PipeLLM: Pipeline
  LLM Inference on Heterogeneous Devices"; **all numbers simulated, no measurements** —
  ignore.
- M2Cache (arXiv 2410.14740, https://arxiv.org/abs/2410.14740): SSD holds all FFN
  weights, DRAM 2-level cache, HBM neuron cache; RTX 3090 + 64GB DRAM + PCIe3 SSD:
  LLaMA-2-70B **0.3835 tok/s**, Falcon-40B 0.312 tok/s; SSD-resident ≈ 8x slower than
  DRAM-resident. Sparse-predictor based (DejaVu) — accuracy-caveated, not lossless.
- LOIP (arXiv 2512.21835, https://arxiv.org/abs/2512.21835): dense LLaMA3.3-70B on
  5 Jetson edge nodes, offloading-based interleaved pipeline parallelism overlapping
  weight streaming + compute + inter-device comm; 8.8-20.3x over SOTA baselines
  (multi-device scenario, not ours, but confirms "loading bubbles" as the core enemy).
- NeuroPrefetcher (arXiv 2608.22643, ICPP 2026, https://arxiv.org/abs/2608.22643):
  delta prefetching of sparse weights; 82-85% of active neurons persist token-to-token;
  GPU-resident predictor (2.86% of model params) forecasts downstream layer activity;
  explicit NVMe reads for delta rows only; **7.9-12x over llama.cpp** on unified-memory
  edge hardware. Requires activation sparsity (predictor-driven) → lossy/sparse-model
  assumption; **not applicable to dense FP8 Qwen3.8 (SwiGLU, dense activations)**, but
  see §5 steal on prefetch distance.
- Prima.cpp (arXiv 2504.08791, https://arxiv.org/abs/2504.08791): 30-70B on 4 consumer
  devices; pipelined-ring parallelism overlapping disk I/O, compute, Wi-Fi comms;
  mmap-based offloading with prefetch-release conflict solved; 70B at **674 ms/token
  (~1.48 tok/s)** under <6% memory pressure; 32B+spec-decode 26 tok/s; 5-17x lower TPOT
  than llama.cpp/exo/dllama. This is the best published "big dense model, tiny memories,
  NVMe in the loop" number — our 1.05-1.25 tok/s single-box target for 27B FP8 is
  same-class (they use 4 devices).
- INF² near-storage (arXiv 2502.09921), DUAL-BLADE NVMe-direct KV offload (arXiv
  2604.26557), SSD MoE energy analysis (arXiv 2508.06978), ActiveFlow DRAM weight
  swapping (arXiv 2504.08378), gdsllm GDS weight streaming
  (https://github.com/rscunha13/gdsllm) — adjacent; none beat the tier math for a
  single Windows box.

**Verdict vs synthesis:** nothing here invalidates the plan; SLEP is existence proof for
double-buffered tier streaming; Prima.cpp sets the single-node-ish SOTA bar (~1.5 t/s
70B). Our numbers stand.

---

## 2. FlexGen (arXiv 2303.06865) — what to actually copy for 27B prefill

Sources: https://arxiv.org/abs/2303.06865, full text
https://ar5iv.labs.arxiv.org/html/2303.06865, repo https://github.com/FMInference/FlexGen.

Methodology (the parts worth stealing):
1. **Zig-zag block schedule** (their Fig. 3): iterate the (layer × batch-chunk) grid in
   blocks so a layer's weights are loaded once and reused across many batch chunks —
   weight-stationary — **proven within 2x of optimal I/O complexity** (Theorem 4.1).
   This is exactly our "layer-major prefill with batch-chunk reuse": stream each layer's
   383.88MB once per chunk-of-prompts instead of re-streaming 25.6GB per 64-token chunk.
2. **Overlap schedule (Algorithm 1)**: while chunk computes, parallel streams do:
   next layer's weight load + previous chunk's activation/KV store + next chunk's
   activation/KV load, joined by one sync per block.
3. **Placement as LP**: percentages of weights/activations/cache across GPU/CPU/disk
   (wg,wc,wd / hg,hc,hd / cg,cc,cd), layer-granularity for weights, tensor-granularity
   for cache — i.e., our L/M/N split but formally optimized. For us the LP would be
   3 variables (trivially enumerable) — worth doing as a check script, not a system.
4. **CPU attention delegation**: when KV lives in CPU, compute attention scores on CPU so
   only activations cross PCIe (cuts I/O by factor s). Relevant later if KV outgrows VRAM.
5. **4-bit compression** doubled feasible batch (we stay FP8 — different tradeoff).

Measured (Table 2; 1x T4 16GB, 208GB DRAM, 1.5TB NVMe SSD ~2GB/s read):
- OPT-6.7B 25.26 tok/s (4bit 29.12); OPT-30B **7.32** (4bit 8.70); OPT-175B **0.69**
  (4bit 1.12) at seq512, batch 144-256. Baselines at 175B: DeepSpeed/Accelerate ~0.01.
- Overlap gain reported ~1.17-1.25x (hides latency, not bandwidth) — matches our
  wave-1 note.

**Correction to synthesis.md:** FlexGen does **not** use or mention "activation
checkpointing" (verified in full text; its memory levers are offload placement, 4-bit
quant, sparse attention). Checkpoint-recompute is a training technique (ZeRO world).
Our prefill plan (keep per-layer activations/checkpoints in RAM while streaming weights
layer-major) is still correct engineering — just cite zig-zag block schedule, not
"FlexGen activation checkpointing".

---

## 3. KTransformers (2025-2026) — AVX2/Zen numbers; dense-FP8 applicability

Sources: https://github.com/kvcache-ai/ktransformers,
AVX2 tutorial
https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/AVX2-Tutorial.md,
leaderboard https://ktransformers.net/en/benchmarks,
https://www.phoronix.com/news/KTransformers-0.5.3 (v0.5.3 added AVX2-only MoE kernels
for BF16/FP8/quantized),
issue https://github.com/kvcache-ai/ktransformers/issues/504,
AMX docs https://ktransformers.net/en/docs/optimization-techniques/amx,
LMSYS deep-dive https://www.lmsys.org/blog/2025-10-22-KTransformers/.

Numbers:
- Same-rig ISA comparison (leaderboard, EPYC 9355 + 1x RTX 5090, Qwen3.5-35B-A3B FP8):
  86.2 t/s decode full-ISA vs **60.9 t/s in AVX2 mode → AVX2 ≈ 71%** of
  AMX/AVX512 throughput on bandwidth-bound MoE decode.
- Genuine AVX2-only Zen3 server (2x EPYC 7C13 Milan + 1x RTX 4090):
  Qwen3.5-35B-A3B FP8 **19.3 t/s**, Qwen3-Coder-Next FP8 20.6 t/s,
  Qwen3-30B-A3B BF16 19.4 t/s decode.
- Xeon 8255C (Cascade Lake, forced AVX2) + 4x A10: DeepSeek-V3 **12-16 t/s** (issue #504);
  reporter notes AVX2 unexpectedly comparable to AVX512 in practice.
- Community (Reddit https://www.reddit.com/r/LocalLLaMA/comments/1kqz9uu/deepseek_v3_benchmarks_using_ktransformers/):
  consumer non-AMX rigs ~5-10 t/s decode (Q5_K_M 9.7 t/s @short ctx).
- AMX reference points: dual Xeon4 + 24GB GPU: 500+ t/s prefill DeepSeek-V3; AMX kernels
  21.3 TFLOPS sustained per Xeon socket (LMSYS).

Applicability to us: **they are MoE-only in practice** (expert placement CPU/GPU by
`--kt-num-gpu-experts`; models on leaderboard all MoE; docs say memory bandwidth is the
bottleneck, threads = physical cores, NUMA-aware threadpool). No dense-27B numbers
exist. Steal list: (a) the AVX2≈0.71x-of-best-ISA factor confirms Zen3 AVX2 is
viable for a bandwidth-bound CPU tier (our ~10ms/layer estimate needs ~38GB/s effective
— llamafile data in §4 says 77% of theoretical is achievable, so 0.77×51.2 ≈ 39GB/s,
consistent); (b) chunked-prefill knobs (`--chunked-prefill-size` 4096/8192) and
GPU-prefill-token-threshold (~400) as scheduling hints; (c) their KT_RAWINT4/F16C-style
dequant in-register philosophy. Linux-only; their kernels are GPL-ish territory we
reimplement anyway.

---

## 4. llamafile / llama.cpp CPU matmul ("up to 5x", Jan-Mar 2024) — steal list for AVX2 FP8 GEMV

Source: Justine Tunney, "LLaMA Now Goes Faster on CPUs"
https://justine.lol/matmul/ (fetched in full; Mozilla announcement
https://hacks.mozilla.org/2024/01/llamafile-a-new-way-to-distribute-and-run-llms/
is 404; llamafile itself: https://github.com/Mozilla-Ocho/llamafile, kernels in
`tinyblas_cpu.h`).

What was done: 84 new hand-written matmul kernels contributed to llama.cpp (PRs
#6412, #6414), covering f16/q4_0/q8_0 with SSE2/AVX/AVX2/AVX512-VP2INTERSECT.

Headline numbers:
- i9-14900K (Alderlake): f16 prompt processing **50 t/s vs 13 (5x)** — the famous 5x;
  q8_0 prompt 63 vs 40.
- HP i9-9900 (Skylake-Vega AVX2, DDR4-2200): q4_0 prompt 28 vs 17 t/s; **eval (decode)
  ~7 t/s for q4_0 AND ~4 t/s q8_0** — decode is memory-bandwidth-bound and quants
  stop mattering at n=1 (their words: decode performance is "capped by memory
  bandwidth" — ~27-27.5 GB/s effective on a 35.2GB/s-theoretical box ≈ **77%**).
- Threadripper 7995WX (Zen4): f16 prompt 485 vs 197; "run llama.cpp 2.8x faster on
  Zen4"; 790-810 GFLOPS vs MKL 295 at 512x512.

Techniques (steal list for our AVX2 FP8 GEMV):
1. **Unroll OUTER loops, not inner** — 3x4 register blocking: one row of A reused in
   registers across 4 accumulator rows; shares loads of a0 across 3 dot columns
   (this is where the "up to 5x" came from — outer-product tiling, not wider vectors).
2. **Tile packing (mnpack)**: recursive bitonic-sort-based packing so the inner loop is
   a straight line of FMA on hot rows/columns (we do this GPU-side with cp.async; same
   idea CPU-side with plain loads).
3. **No BLAS, no OpenMP**: persistent threads with spinlock barriers (a custom barrier
   struct); pin threads to cores, avoid scheduler drift; on hybrid parts avoid E-cores.
   For 5600X: 6 physical cores, 12 threads — pin 6, spin-wait.
4. For decode GEMV (n=1): bandwidth is king — layout weights for pure sequential
   streaming, no gathers (matches our Zen3 note: vpgatherdd ~1µop/elem, LLVM fastGather
   disabled on zen3).
5. e4m3 decode: our F16C trick (`(b&0x7f)<<7` bitcast fp16 ×256, `_mm256_cvtph_ps`,
   1µop) is the same family as their f16 kernels — validated approach.

Implication for synthesis: 5600X DDR4-3200 dual = 51.2GB/s theoretical; 77% → ~39GB/s;
384MB/layer → **~9.8ms/layer CPU tier** — the synthesis "~10ms/layer, 40GB/s DRAM"
row is confirmed by two independent sources (llamafile 77% factor; KTransformers AVX2
0.71 factor is an upper-bound sanity not a contradiction, different bottleneck mix).

---

## 5. ZeRO-Infinity / ZeRO-Inference, PowerInfer, and the 2025-26 tiered-decode landscape

### ZeRO-Infinity (arXiv 2104.07857) — bandwidth-centric design + P2P prefetch

Source: https://arxiv.org/abs/2104.07857, full text
https://ar5iv.labs.arxiv.org/html/2104.07857.

- **Bandwidth-centric philosophy**: CPU/NVMe capacity is ~50x GPU memory; bandwidth is
  the constraint, so partition *each parameter* across ranks (allgather) instead of
  whole-tensor broadcast — aggregate NVMe→GPU bandwidth scales linearly with DP degree
  (DGX-2 node: ~48GB/s CPU, ~25GB/s NVMe aggregate; 64 nodes: >1.5TB/s). Single-box
  moral: **parallelize the links you have** — for us: both PCIe H2D AND NVMe reads AND
  CPU compute must run concurrently, no serialized hops.
- **P2P prefetch pipeline, depth 3**: before op i runs, issue NVMe→CPU (nc) for i+3,
  CPU→GPU (cg) for i+2, GPU-to-GPU (gg) for i+1, all overlapping compute of i.
  Direct steal: our IOCP reader should prefetch **2-3 layers ahead** through pinned
  staging, not 1 (synthesis says "~1 token ahead" for NVMe — keep the ring deeper:
  latency hiding needs the pipe filled to bandwidth-delay product).
- DeepNVMe implementation notes: completion-based bulk async read/write; "aggressive
  parallelization of I/O requests" (queue many outstanding reads); pinned-memory layer
  with buffer reuse to avoid fragmentation; write in-place from pinned tensors with no
  extra copies. Maps 1:1 to colibri's slab ring and our 8-16×1-2MB slot plan.
- ZeRO-Inference (https://www.deepspeed.ai/2022/09/09/zero-inference.html): batch-1
  reality check — 43 t/s CPU-offload vs **30 t/s NVMe-offload** on a big model with
  batch amortization; without batching NVMe offload collapses.

### PowerInfer — CONFIRMED irrelevant for dense Qwen3.8

Sources: https://arxiv.org/abs/2312.12456 (hot neurons, power-law activation locality,
GPU hot + CPU cold split); slides https://adsl-rg.github.io/slides/241210-PowerInfer.pdf
(sparsity defined on ReLU zero-activations; SwiGLU GLU models lack natural zeros);
Apple relufication https://machinelearning.apple.com/research/relu.
PowerInfer requires ReLU-trained or relufied models (Falcon, ReLU-Llama). Qwen3.8-27B
is dense SwiGLU (MLP) + gated DeltaNet — no exploitable zero-activation sparsity
without retraining/relufication (which we refuse: un-requantized weights is the whole
point). PowerInfer-2 (arXiv 2406.06282) same story on smartphones (neuron-aware SSD
streaming). **Dense = irrelevant, confirmed.** Same conclusion kills NeuroPrefetcher
(§1b) and M2Cache's neuron-caching half for us (M2Cache's *prefetch-timing* half still
applies, below).

### 2025-2026 "SSD/RAM/VRAM tiered dense decode" corpus (searched: "NVMe LLM inference
2025", "tiered memory LLM decode", "SSD offload LLM windows")

| System (date) | What | Dense? | Number | URL |
|---|---|---|---|---|
| nTransformer SLEP (2025) | 2-buffer NVMe/PCIe/compute pipeline, 3 tiers | dense | 70B Q4: 0.3 t/s (3090) | github.com/xaskasdf/ntransformer |
| Prima.cpp (2025-04) | mmap prefetch + pipelined-ring, 4 devices | dense | 70B 674ms/t; 32B+spec 26 t/s | arxiv.org/abs/2504.08791 |
| LOIP (2025-12) | interleaved pipeline + offload cost model, 5 Jetsons | dense 70B | 8.8-20.3x vs baselines | arxiv.org/abs/2512.21835 |
| M2Cache (2024-10) | mixed-precision + SSD/DRAM/HBM cache | dense (sparse preds) | 70B 0.38 t/s; SSD tier 8x DRAM tier | arxiv.org/abs/2410.14740 |
| NeuroPrefetcher (2026-08) | delta-row NVMe prefetch, predictor | sparse | 7.9-12x vs llama.cpp | arxiv.org/abs/2608.22643 |
| INF2 (2025-02) | near-storage processing | MoE | high-throughput | arxiv.org/abs/2502.09921 |
| SSD MoE offload energy (2025-08) | SSD expert streaming analysis | MoE | — | arxiv.org/abs/2508.06978 |
| DUAL-BLADE (2026-04) | dual-path NVMe-direct KV offload | KV cache | -42.4% decode latency | arxiv.org/abs/2604.26557 |
| AirLLM (2024) | layer-wise NVMe streaming, no overlap | dense | users 5-35 s/tok | wavect.io/blog/airllm-layer-wise-inference-low-vram/ |

Steal: **M2Cache's prefetch-distance rule** — "one-layer SSD→DRAM preload time ≈ 2x
one-layer inference time" and next-layer predictability ~100% (falls to ~80% two layers
out) → issue NVMe reads **2 layers ahead** with certainty; our 56.5ms NVMe layer vs
~1-16ms compute layers says the same: read-ahead depth 2-3, ring of 3-4 layer slots.

**Verdict vs synthesis:** ZeRO-Infinity depth-3 prefetch and M2Cache depth-2 rule both
say our "~1 token/layer ahead" NVMe pipelining is **too shallow** — widen the IOCP
ring to 2-3 layers (memory cost 3×384MB ≈ 1.1GB pinned, affordable inside M-tier
budget or borrowed against it). Otherwise plan stands.

---

## 6. llama.cpp -ngl on Windows: mmap partial offload — verified behavior + real numbers

Mechanics (verified first-hand in wave-2 code audit `audits/w2/llamacpp-offload.md`,
clone at c060ca974): placement is static after load; CPU-layer weights are zero-copy
mmap pointers read directly by CPU kernels (no per-token H2D weight copies, no
readahead, no double buffering in single-GPU mode — that path exists only for
multi-GPU full offload); per token ~2 device syncs + 1-2 blocking memcpys at the single
CPU↔GPU graph-split boundary; load-time PrefetchVirtualMemory exists but normal loads
never pass prefetch>0. So **llama.cpp does NOT prefetch-hide H2D or NVMe faults during
decode** — confirmed. Corroborating web evidence: ideas discussion "Force all
computations to run on GPU during partial offload"
(https://github.com/ggml-org/llama.cpp/discussions/11442 — no maintainer response, no
implementation; a user's Linux unified-memory hack ran *worse* than hybrid); an
experimental community PR "prefetch weights when offloading to CPU" exists only as a
Reddit post (https://www.reddit.com/r/LocalLLaMA/comments/1s5xcmw/); mmap-thrash
explainer https://news.ycombinator.com/item?id=35426679; partial-loading study
https://tinycomputers.io/posts/partial-llm-loading-running-models-too-big-for-vram.html.

Quantified user reports (4070-class, our exact model family):
- **Qwen3.8-27B on RTX 4070 Ti 12GB, Windows 25H2, llama.cpp b10448, CUDA 13.3**
  (reproducible study, https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/discussions/65):
  - UD-IQ2_XXS (fits, 66/66 GPU): **43.6 tok/s** gen (Q2_K_XL 38.0; 16K ctx 39.2).
  - **IQ4_XS 14.63GB, 45/66 GPU layers (partial offload): 5.98 tok/s** — 7.3x slower
    than IQ2 full-GPU; at ~64K ctx: 1.57 tok/s, 301s TTFT.
  - Same box, llama.cpp `draft-mtp` (n_max=2): acceptance **55.2% prose / 90.1% code**,
    speedup **+47.3% / +92.7%** (see §7).
- Qwen3.8-27B on 4070 12GB fits-VRAM quant: ~25 t/s @80K ctx
  (https://www.reddit.com/r/LocalLLaMA/comments/1vqjeub/); a 4070 SUPER user hit a
  14x slowdown until discovering silent full-CPU offload
  (https://www.reddit.com/r/LocalLLaMA/comments/1vweszt/).
- Qwen3.5-27B: 32-38 t/s while it fits; **7-10 t/s once layers spill**
  (https://www.reddit.com/r/LocalLLaMA/comments/1rq8l0x/).
- Model > RAM via mmap: 0.5-3 t/s typical (HN 35426679; issue
  https://github.com/ggml-org/llama.cpp/issues/864).

Takeaway: llama.cpp partial offload of a dense 27B on our class of box ≈ **6 t/s**
(6-8 CPU layers of IQ4_XS), collapsing to ~1.5 t/s at long context. Our plan
(21 GPU + 23 RAM + 21 NVMe FP8, un-quantized) targets 0.66-1.25 t/s — i.e., we trade
llama.cpp's ~6 t/s IQ4 partial-offload for exact FP8 at ~1 t/s. Both numbers are
honest; the 43.6 t/s IQ2 full-GPU figure is the "just quantize harder" ceiling we
are deliberately refusing.

---

## 7. MTP / speculative decoding for Qwen-style MTP heads

Acceptance-rate literature (per draft depth, greedy/topk=1 chains):

| Source | Model / setting | depth-1 | depth-2 | depth-3 | URL |
|---|---|---|---|---|---|
| DeepSeek-V3 report | 2nd-token acceptance 85-90% | 0.85-0.90 | — | — | arxiv.org/abs/2412.19437 |
| SGLang blog 2025-07-17 | DeepSeek, 3-tok window topk=1 | avg accepted len **2.18** | | 4-tok: **2.44** | lmsys.org/blog/2025-07-17-mtp/ |
| FastMTP | vanilla MTP baselines | ~70% | ~11% | ~2% | arxiv.org/abs/2509.18362 |
| FastMTP | after training aligned to inference | **81%** | **56%** | **36%** | same |
| vLLM issue 35387 | Qwen3-Next-80B FP8, num_spec=1 | **~95%** | — | — | github.com/vllm-project/vllm/issues/35387 |
| llama.cpp draft-mtp | **Qwen3.8-27B** (our model) | **55.2% prose / 90.1% code** | — | — | huggingface.co/unsloth/Qwen3.8-27B-GGUF/discussions/65 |
| SpecDecode-Bench | GLM-4.5-Air MTP | speedup 1.3-1.8x | | | specdecode-bench.github.io |
| Qwen official | Qwen3-Next MTP claim | ~3x decode speedup claim | | | qwen.ai blog (Qwen3-Next-80B-A3B) |

Verify-batch cost model: verification is one batched forward over T rows — bandwidth
free at batch≤T (weights read once); cost is only extra activation + lm_head rows.
SGLang: parallel verify "replacing n sequential decode steps with a single parallel
verification pass"; +59.8% throughput at 3-token window on 16xH200 (81.5 vs 51.0 t/s
per rank); recommended default draft_token_num=2. **Critical caveat**: vLLM on
Qwen3-Next measured 95% acceptance but only **1.1x speedup** (scheduling/prefill
interactions, even a 76% latency regression in some configs) — proof that draft-side
host overhead can eat the entire win; our graph-captured draft loop (replayed 4x per
host check) is the right shape. Community Qwen3.6-27B spec-decode benchmark: MTP/NEXTN
2.2-2.8x in vLLM (https://www.reddit.com/r/LocalLLaMA/comments/1v22hu9/ — Reddit
blocked to bots; figures from search index).

Production implementation details (draft head plumbing):
- **DeepSeek-V3 MTP module shares BOTH the embedding table AND the output head (full
  lm_head) with the main model** (verified in report text: "its embedding layer is
  shared with the main model... its output head is shared with the main model";
  arxiv.org/abs/2412.19437). Qwen3-Next MTP and GLM-4.5 MTP follow the DeepSeek recipe
  (same concat-projection + shared TRM design; GLM-4.5 card: --speculative-num-steps 1,
  num-draft-tokens 4; https://huggingface.co/zai-org/GLM-4.5).
- So production systems **do run draft logits through the full lm_head** — no
  vocab truncation trick in the reference implementations; top-k appears only as
  EAGLE-style tree branching (topk=1 = plain chain). Practical tricks they DO use:
  CUDA-graph the draft+verify loop, share KV/cache between MTP layer and target
  (vLLM MTP docs: "wires the assistant layers to share KV cache with the target
  model"; https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/), and
  keep depth shallow (default num_speculative_tokens=1, i.e., T=2 — exactly our plan).
- llama.cpp's Qwen3.8-27B draft-mtp note: greedy outputs **diverged at token 16 on
  prose** (0/5 exact match) despite acceptance 55% — their MTP sampling path isn't
  output-preserving; our verify must reject-exact (pos[6] accept flag) to stay greedy-identical.

Verdict vs synthesis: **MTP T=2 ×1.6 @ p=0.6 is validated** — measured on our exact
model+GPU class: +47% (prose, p=0.55) to +93% (code, p=0.90) with weights already
resident; on our tiered engine the verify batch is bandwidth-free so net ≈
2/(2-p) - draft cost ⇒ ×1.4-1.8. Depth 2-3 drafts (T=3-4): only worth it if MTP is
retrained FastMTP-style (vanilla d2 acceptance ~11% is useless, matching colibri's
44-62% d2-3 observations being charitably sampled from code-heavy workloads).
lm_head VRAM residency confirmed necessary (shared head, 2.543GB, used by both draft
and verify).

---

## 8. Windows-specific: IOCP/IoRing streaming for ML, GPUDirect Storage status

- **IoRing vs IOCP**: Microsoft's own IoRing_Demos (Yarden Shafir,
  https://github.com/microsoft/IoRing_Demos) measured I/O rings only ~2% faster than
  IOCP (~3% vs sync reads; up to 5-10% in some IOPS tests); deep-dive comparison:
  https://windows-internals.com/ioring-vs-io_uring-a-comparison-of-windows-and-linux-implementations/
  (Windows IoRing is a submission/completion queue modeled on io_uring but fundamentally
  built on IOCP machinery underneath). RavenDB 7.1 production adoption write-up:
  https://ravendb.net/l/uvzbeu/hub/2024/04/01/one-io-ring-to-rule-them-all.
  → Our IOCP NO_BUFFERING choice is right; IoRing is a +2% polish, not a redesign.
- **No published Windows ML streaming engine** was found (2024-2026) that beats IOCP
  semantics; the model-loading literature is Linux-centric: fastsafetensors (arXiv
  2505.23072, https://arxiv.org/abs/2505.23072) gets 4.8-7.5x GPU load speedup over
  safetensors deserialization via batched multithreaded reads + pinned buffer reuse +
  (on Linux) GDS; repo supports Windows for the CPU-staging path
  (https://github.com/foundation-model-stack/fastsafetensors); vLLM integrates it
  (https://docs.vllm.ai/en/stable/models/extensions/fastsafetensor/). Alluxio and
  NVIDIA Run:ai Model Streamer are S3/network-focused. **Our IOCP reader would be the
  fastest published Windows NVMe weight-streaming path for inference we can find.**
- **GPUDirect Storage on Windows consumer GPUs: NOT AVAILABLE.**
  - Official: "GDS is currently supported on Linux x86-64 distributions; it is not
    supported on Windows" (https://developer.nvidia.com/gpudirect-storage; kernel
    module https://github.com/NVIDIA/gds-nvidia-fs).
  - GeForce-class cards are **compatibility mode only even on Linux** ("All other
    cards are supported only in compatibility mode",
    https://docs.nvidia.com/gpudirect-storage/release-notes/index.html; forum: "The
    GeForce line doesn't currently support GPUDirect RDMA, on which GDS depends...
    no particular plans" https://forums.developer.nvidia.com/t/gpudirect-storage-requirement-questions/157984).
  - Windows consumer alternative is DirectStorage + RTX IO (GPU *decompression*,
    not storage→VRAM DMA; stages through system memory;
    https://devblogs.microsoft.com/directx/directstorage-1-1-now-available/,
    https://developer.nvidia.com/rtx-io). Not useful for uncompressed FP8 weights.
  → Confirms synthesis: NVMe→pinned RAM→PCIe H2D is the only path; nTransformer's
  GPU-doorsbell NVMe trick is Linux/VFIO-only.

---

## 9. Cross-check vs synthesis plan — threats and adjustments

**Validated (no change):**
1. Tier split L≈21/M≈23/N≈21 → 0.66-1.05 tok/s: consistent with every measured
   system (SLEP 0.3 t/s on 70B-Q4/3090; Prima.cpp 1.48 t/s on 70B/4 devices; M2Cache
   0.38 t/s 70B/3090; llama.cpp mmap-spill 0.5-3 t/s).
2. CPU AVX2 tier ~10ms/layer (≈39-40GB/s effective on DDR4-3200): confirmed by
   llamafile's 77%-of-theoretical decode bandwidth factor and KTransformers AVX2
   leaderboard (AVX2 ≈ 71% of AMX on bandwidth-bound decode; Zen3 EPYC 7C13 sustains
   19-21 t/s on 3B-active MoE).
3. MTP T=2 net ×1.6: confirmed by measured acceptance on the same model
   (55% prose / 90% code → +47-93% with resident weights) and SGLang default
   draft_token_num=2, avg accepted length 2.18.
4. lm_head VRAM-resident: confirmed — production MTP heads share the FULL lm_head
   (DeepSeek-V3 report); no truncated draft vocab exists in reference impls.
5. IOCP NO_BUFFERING + pinned ring: confirmed best Windows path (IoRing only +2%;
   GDS impossible on Windows/GeForce; DirectStorage not for raw weights).
6. Shard-major staging / double buffering: SLEP does exactly this (2 GPU + 2 staging
   buffers) with an 83x-over-mmap result.

**Adjustments (flagged):**
- **A1 — NVMe prefetch depth should be 2-3 layers, not ~1.** ZeRO-Infinity runs a
  depth-3 pipeline (nc i+3 / cg i+2 / gg i+1); M2Cache: SSD layer load ≈ 2x layer
  compute → prefetch ≥2 ahead. Cost: ~0.8-1.2GB extra pinned ring. (56.5ms NVMe layer
  vs sub-16ms compute layers means depth-1 stalls on every layer.)
- **A2 — FlexGen citation fix:** zig-zag block schedule (2x-optimal I/O proof) is the
  steal; FlexGen has no activation checkpointing. Our prefill fix is still right.
- **A3 — Draft overhead discipline:** vLLM's Qwen3-Next 95%-acceptance-but-1.1x result
  proves host-side draft overhead can annihilate MTP; keep draft path graph-captured
  and host-free (we do), and keep greedy verify exact (llama.cpp's draft-mtp diverges
  on prose — do not copy its sampling path).
- **A4 — Honest positioning:** llama.cpp already gives Qwen3.8-27B 43.6 t/s (IQ2) /
  6 t/s (IQ4_XS partial) on a 12GB 4070-class Windows box. Our engine's reason to
  exist is exact FP8 weights; expect ~1 t/s class, and say so.

**Nothing found that beats or invalidates the plan.** The only genuinely superior
techniques (GPU-doorbell NVMe = SLEP's gpu-nvme-direct; GDS = gdsllm; near-storage =
INF2) are all Linux/datacenter-gated. On Windows consumer hardware, the synthesis
design is at or beyond published SOTA for dense-FP8 tiered decode.

---

## 10. Consolidated steal list for the rig/model

1. SLEP ring: 2 device buffers × 2 pinned staging buffers, alternating layers
   (github.com/xaskasdf/ntransformer).
2. ZeRO-Infinity depth-3 prefetch (nc i+3, cg i+2) mapped to IOCP: issue layer+2/+3
   reads immediately after layer i starts computing (arxiv 2104.07857).
3. M2Cache rule: prefetch distance 2 layers (SSD layer ≈ 2x layer compute; next-layer
   predictability ~100%) (arxiv 2410.14740).
4. FlexGen zig-zag block schedule for prefill: layer-major weights × batch-chunk
   reuse, one sync per block, 2x-optimal I/O (arxiv 2303.06865).
5. llamafile outer-loop unrolling (3x4 register block), tile packing, spinlock-barrier
   pinned threads, no BLAS/OpenMP for the AVX2 FP8 GEMV tier (justine.lol/matmul).
6. KTransformers knobs: threads = physical cores, NUMA-pool = 1, chunked prefill
   4096-8192 (ktransformers AVX2 tutorial).
7. MTP: T=2 default, chain topk=1, shared embed + shared lm_head, CUDA-graphed
   draft+verify, exact greedy verify (DeepSeek-V3 report; SGLang blog; vLLM MTP docs).
8. IOCP completion-based bulk reads with pinned-buffer reuse layer (DeepNVMe pattern)
   — our 8-16×1-2MB slot ring is the right shape.
9. Do NOT bother: PowerInfer/NeuroPrefetcher hot-neuron tricks (dense SwiGLU — dead
   end), GDS on Windows (doesn't exist), IoRing rewrite (+2%), AirLLM-style no-overlap
   streaming (5-35 s/tok).

— end of report —
