# Paper research digests (session 3, 2026-08-28) — MoE offloading/prefetch/spec-decode

Swarm-researched for the 12 tok/s push. One section per paper: mechanism,
reported numbers, and a verdict for THIS engine (RTX 4070 SUPER, 5600X, 16 GB
host RAM, 180 GiB NVFP4 model on NVMe, 42 sparse layers top-8-of-288,
13.5 MiB/expert, near-uniform routing entropy 7.98/8.17 bits, measured LRU
cliff 26% @512 slots / 69% @672 / 82% @1024).

## 1. DFlash / DFlash2 (the adopted lever)

- Blog: https://inco.ai/blog/dflash2/
- DFlash paper (ICML 2026): https://arxiv.org/abs/2602.06036 (HTML: https://arxiv.org/html/2602.06036v1)
- Reference impl: https://github.com/z-lab/dflash (dflash/model.py)
- Checkpoint: https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2 (2.34 GB BF16, 81 tensors)
- SGLang port studied: branch `xinyuan/glm-5.3-flash-support`, models/dflash.py + speculative/dflash_worker_v2.py (PR #36708)
- Mechanism + our port status: see `audits/dflash2-session.md` section 1.
- Key reported numbers (GLM-5.3-Flash, block 8, 7 drafts/round): acceptance
  length 4.03 (MT-Bench) to 5.86 (MATH-500); 1.73-2.79x vs autoregressive at
  concurrency 1 on 4xGB300. vs native MTP: +0.3 to +0.9 accepted/round.
- Block-size ablation (DFlash v1): b8 full-accept rate only 35.7%; b16 wins
  on math/code; draft latency is block-size-insensitive (one parallel pass).
- Selector ablation: Recall@1 85.4% -> Recall@16 99.5%; oracle-over-top16
  acceptance 4.27 -> 6.79; beats +77.8M-param sequential corrector.
- Feature ablation: KV-injection beats EAGLE-style input fusion (4.2x vs
  3.5x GSM8K); 5 target layers > 3; capture layers spread over [1, N-3].

## 2. FlashMoE (learned cache replacement) — https://arxiv.org/abs/2601.17063

- Policy: per-expert features recency r (1 on access, ++ otherwise) and
  lifetime frequency f; input [1/r, f/max_f] to a 3-layer 128-hidden SiLU
  FFN regressing Belady-oracle "routing distance"; evict argmax. ~113 KB per
  layer, 158 us/op, trained on 512 TriviaQA traces in ~2 h.
- Headline "+51%" is **vs LFU**; vs LRU it is +21% hit rate, on same-domain
  evals, at cache sizes already in the 70-85% hit regime.
- Verdict: frequency features are pure noise under our near-uniform
  marginals (our TinyLFU/count/second-chance experiments all falsified this
  axis). Expected upside single-digit %. **Cheapest kill test: compute the
  Belady ceiling on our 200-token trace — HitRate(Belady) - HitRate(LRU)
  bounds every learned policy including this one.** Not done yet.

## 3. HOBBIT (mixed-precision offloading) — https://arxiv.org/abs/2411.01433

- iMoE tail rule: with router probs sorted desc, u_i = tail mass;
  u > 0.9 skip expert, u > 0.6 use low-precision copy. ~3% skip rate on
  chat traces. Our precision floor (NVFP4) kills the low-precision tier —
  only the skip rule survives, worth ~nothing at batch 1.
- Layer-level prefetch: next-layer gates predicted from this layer's hidden
  (a d x N matvec "stacking computer" over p layers): ~96%/90%/… accuracy
  for +1/+2/+3. **Adaptive-depth rule: keep extending prefetch depth only
  while predicted experts are MISSING from cache** (stop when predictions
  land on cached entries). Pin prefetched entries against eviction.
- Engineering caveat we must copy: **cudaMemcpy is non-cancellable** — a
  wrong prefetch blocks the copy queue. Mitigation: be aggressive at the
  NVMe->pinned-RAM stage (file reads abandonable) and conservative at the
  pinned->VRAM commit. Applies directly to ExpertStager's copy_stream_.
- Their cache priority: w1*LRU + w2*LFU + w4*FLD (farthest-layer-distance,
  experts recur at fixed layer period) calibrated by a miss-penalty model;
  2-9% better than plain LRU/LFU. Marginal for us.
- Numbers: up to 9.93x vs MoE-Infinity, 3.2x vs MoE-Offloading (4090,
  Mixtral/Phi-MoE). Assumes host RAM holds both precision copies — we
  cannot (16 GB vs 180 GiB).

## 4. MoE-Infinity — https://arxiv.org/abs/2401.14361

- Request-level trace metrics: activation probability, entropy, expert reuse
  R = E_within/E_across. Findings on THEIR models: uniform across requests,
  skewed within request -> LRU GPU cache + selective/group activation.
- Verdict: SKIP mechanism. Premises inverted for us: host DRAM << model,
  our routing uniform within and across requests (R ~ 1), no batch.
- KEEP: the trace methodology — log expert IDs, compute lag-1 conditional
  entropy / Jaccard between consecutive tokens before building anything
  smarter than capacity-LRU. (Our route_trace already dumps this.)

## 5. Pre-Attention Expert Prediction (ETH) — https://arxiv.org/abs/2511.10676

- Insight: softmax/LayerNorm/attention are ~ranking-preserving, so a cheap
  linear map of the **pre-attention hidden (post-LN, same layer)** reproduces
  the expert ranking. Per-layer predictor, best variant:
  Linear(4096->2048) -> SiLU -> Linear(2048->288) (~9.4M params/layer).
- Training: multi-label BCE (label = in-true-top-8) with class weights 3.0
  (top-10) / 1.5 (11-30) / 0.5 (rest) + 0.3 * pairwise ranking hinge
  (margin 0.1) on raw scores; 10M samples harvested from real runs; single
  24 GB GPU, 30 epochs.
- Accuracy: exact top-k 93.0% (DSv2-Lite 64E k6), 94.7% (Qwen3-30B 128E k8 —
  closest analog to us), 97.6% (Phi-mini 16E k2). +4 overfetch -> 98.7% hit.
- Cost: 0.15 ms/layer on CPU, overlapped with attention. Timing window =
  one attention block of the SAME layer (works on layer 0, unlike
  cross-layer methods).
- Verdict: the strongest prediction paper for our geometry. Requires trace
  harvesting (engine instrumentation: dump pre-attention hidden + true
  top-8 per sparse layer — extend INSIGNIA_GLM53_ROUTE_TRACE).

## 6. ST-MoE (spatio-temporal prefetch) — https://arxiv.org/abs/2606.15453

- Training-free branch-predictor style: **CCT** (per adjacent-layer pair,
  co-occurrence top-K successors per selected expert, 2-bit saturating
  confidence, online +/-1 with replacement of dead entries) + **HT**
  (previous token's actual top-8 for this layer, fixed score 2).
  PrefetchSet = {f : sum of CCT votes + HT >= 2}.
- Pipeline: layer i's gate result triggers prediction for i+1; transfers
  overlap i+1's attention; misses demand-fetch at the gate (correctness
  preserved).
- Fit with our data: our static CCT measured 73.7% coverage @ 2.36x
  overfetch; our adjacent-token overlap 2.19/8 is 4-9x their models'
  ~2x-over-random, so **HT union alone should add ~27% coverage nearly
  free**; the >=2 aggregated threshold + online replacement should push
  toward their 85%.
- **Pollution warning (theirs is bufferless, ours is not): stage prefetches
  OUTSIDE the LRU; admit on verification only.** Otherwise 2.4x overfetch
  evicts live entries and the LRU cliff eats the gain.
- Verdict: ADOPT — cheapest prefetch upgrade, loader already landed
  (Runner::load_cct/cct_prefetch, INSIGNIA_GLM53_CCT); still need
  tools/dump_cct.py output + HT union + threshold.

## 7. APEX (adaptive overfetch) — https://arxiv.org/abs/2608.11688

- On top of a predictor (theirs: router-distilled single linear, KL loss,
  0.05% params): choose overfetch delta per token via **ordinal logistic
  CDF** on the predictor's own score vector: delta_hat = min{d :
  sigma(theta_d - w.q) >= tau}, tau 0.9-0.97; monotone thresholds trained
  with cumulative BCE against oracle-delta labels from a held-out trace.
- >99% overlap at tau=0.97; adaptive beats static k+4/k+8 by 20-40% because
  overfetch traffic itself becomes the bottleneck. Correctness-preserving
  mode confirmed: prefetch is I/O only; misses compute-available-first then
  second-pass accumulate stragglers with true router weights.
- Bandwidth bound to copy: T_prefetch(delta) = delta * 13.5 MiB / B_free <=
  T_attention; clamp delta and skip already-resident experts.
- Verdict: ADOPT the delta-scheduler once a predictor exists (pairs with
  paper 5). Simulator-based absolute ms values — trust percentages only.

## 8. DAOP — https://arxiv.org/abs/2501.10375 — SKIP

- Cache-line CPU/GPU placement + CPU pre-compute of predicted experts.
  Needs routing skew (their Mixtral top-1 acc 84%); at our entropy the
  predictor lands near chance and every mispredict costs a full
  NVMe->RAM->VRAM round trip. Pre-decoding to fp16 in host RAM halves our
  scarcest resource (coverage). Their overlap schedule is generic
  double-buffering we already do.

## 9. Fiddler — https://arxiv.org/abs/2402.07033 — SKIP compute, ADOPT rule

- "Move the activation, not the weights": per-expert CPU-vs-GPU decision by
  measured latency model. Break-even favors CPU only for DRAM-resident
  experts; our DRAM holds <5% of the model and CPU GEMV competes with the
  same DDR4 bandwidth the NVMe staging needs. Net < 1%.
- Durable rules: batch any CPU GEMVs (never batch-1 matmul); expert
  popularity placement buys only 3-5 pp even on skewed Mixtral (their own
  data) — ~0 for us.

## 10. HeteGen — https://arxiv.org/abs/2403.01164 — ADAPT the pipeline

- Tensor alpha-split rejected (our CPU is ~3.5% of GPU FLOPs; split rows
  would sit in DRAM we don't have). The implementable finding is the
  **stage decomposition**: pinning was 72% of naive runtime, unpinned
  streaming 96.9% — i.e. a pinned ring of 2-3 buffers per group, one thread
  issuing large queued NVMe reads into buffer i+1 while the copy engine
  DMAs buffer i and the GPU computes buffer i-1, last module of layer l
  pre-pins layer l+1's first weights. Balance equation: T_consume(max(VRAM,
  PCIe)) = T_NVMe sets the prefetch depth (~3-5 layers of lookahead at our
  sizes, which 16 GB can almost hold).
- Verdict: directly extends ExpertStager; candidate for the demand-path
  stall cleanup AFTER spec-decode lands (stalls are currently hidden by
  waits anyway on empty rounds).

## 11. SPICE — https://arxiv.org/abs/2608.21240 — partial

- Low-rank SVD expert surrogates on prediction miss: lossy below our NVFP4
  floor and never corrected -> OUT (user constraint).
- Transferable: the H_min = ceil(B_per_token_PCIE / beta) lookahead-depth
  calculus; bounded-depth low-priority prefetch queue always preempted by
  demand misses (fail-open); measured PCIe utilization 82-91% sustained.
- Their CPU-exact-residual track: exact but DDR4-bound and competes with
  staging bandwidth on our 2-channel 5600X -> < 1% net; only revisit if
  the NVMe tier is ever hot enough that misses are rare.

## 12. Cross-paper synthesis for this machine (agent's ladder, our numbers)

Per-token bottleneck floor for 4.43 GiB of expert bytes:

| stage | rate | floor/token |
|---|---|---|
| NVMe -> RAM (virtio O_DIRECT, 4-8 deep) | ~5.5-5.8 GB/s | **~0.77 s — the wall** |
| DRAM write+read | ~35 GB/s | ~0.25 s |
| PCIe pinned H2D | 23.2 GB/s | ~0.19 s |
| VRAM GEMV consume | ~500 GB/s | ~0.01 s |

- Every offloading paper assumes host DRAM >= model; ours is 16 GB/180 GiB.
  Their CPU-compute tricks attack the THIRD bottleneck (PCIe), not the wall.
- Near-uniform routing (8/8.17 bits) kills statistical placement (Fiddler
  popularity, MoE-Infinity grouping, DAOP layer-ahead). Only CAUSAL
  prediction survives: (a) same-layer pre-attention predictors (papers 5/7),
  (b) short-range temporal correlation (our 2.19/8 overlap — strongest
  measured anywhere in these papers), (c) spec-decode draft states
  (DFlash2!) driving routing prefetch.
- Lever ranking for 12 tok/s: (1) DFlash2 acceptance (multiplicative,
  ceiling ~5-6 tok/round); (2) tier sizing past the 672-slot cliff
  (69%->82%+ hits); (3) HT/CCT prefetch + APEX-style overfetch to warm the
  tier; (4) bytes/token cuts below NVFP4 are FORBIDDEN by the user except
  FP8/NVFP4-class; (5) deep NVMe pipelining (HeteGen ring) to protect the
  5.5 GB/s; (6) dual-SSD striping to raise the wall itself.

## Sources cross-verified by the DFlash2 semantics agent

- SGLang dflash.py / dflash_worker_v2.py / glm5_next.py on branch
  `xinyuan/glm-5.3-flash-support`; flashinfer/triton backends for the
  ENCODER_ONLY + window_left semantics; memory_pool commit_len masking;
  z-lab dflash model.py conv/selector/lattice einsum
  ("blpr,blcr->blpc"); HF configs of both draft and target. All quotes and
  the GLM-specific delta list are in `audits/dflash2-session.md` section 1.
