# Insignia mathematics frontier

Date: 2026-08-30
Repository: https://github.com/novysvet/insignia
Branch at creation: `glm53-dflash2-4070ti-super`

These are research problems, not benchmark chores. Every problem can be worked
on a generic CPU-only computer without the model, CUDA, or access to glm-box.
Repository traces are optional empirical witnesses; a valid submission must
state and solve the abstract mathematical problem even when no trace is
available.

## Fixed system facts

- GLM-5.3-Flash has 45 target layers: 42 sparse MoE layers after the first
  three dense layers, 288 routed experts, and Top-8 routing. One decode token
  therefore requests 336 layer-expert records.
- One on-disk NVFP4 expert record is approximately 13.5 MiB. A cache-free token
  would request about 4.4 GiB.
- The large machine has one NVMe device, roughly 3.7--4.7 GB/s, a 32 GiB pinned
  host cache holding about 2,425 records, and a 576 MiB device expert cache.
  The observed host hit rate plateaus around 80.3% at that capacity.
- The 4070 Ti SUPER is Ada, with about 800 GB/s observed overclocked VRAM
  bandwidth. Ada has FP8 tensor cores and no native FP4 MMA.
- DFlash2 proposes a block and normally verifies four rows at once. Exact
  DFlash has measured around 5.1--5.3 token/s; the long-term target is 12
  token/s. Approximate policies may trade quality for I/O.
- User quality policy: up to +3.5% same-prefix perplexity is acceptable only
  if difficult free-running outputs remain useful. MSE, cosine, KL, JS, Top-1
  agreement, collapse, and decoded outputs all matter.
- A 10.09M-parameter, 99.21875%-inactive controller is under construction. Its
  complete native i7-14700KF AVX-VNNI ceiling is 3.1849 ms per four-row verify
  round. It sees previous target and DFlash logits, routing/cache state, hidden
  sketches, and causal history.
- Exact FP8 weight compression measured total ratios 0.906244 for dense and
  0.907438 for DFlash. It is accepted for deferred fused-decoder work.
- A roughly 120 GB `UD-IQ3_XXS` checkpoint is downloading for future hybrid
  precision experiments. Do not assume undocumented details of that format;
  parameterize any layout-specific term.

## Ranked problems

1. [Adaptive-k risk control](01-adaptive-k-risk-control.md) -- constrained stochastic control for
   adaptive expert count.
2. [Cache-route hypergraphs](02-cache-route-hypergraph.md) -- joint cache-aware routing and layer-union
   minimization.
3. [Contextual Belady](03-contextual-belady.md) -- online hierarchical caching with imperfect
   route predictions.
4. [FP8 codec break-even](04-fp8-codec-break-even.md) -- fused exact codec and system-level
   compute-for-bandwidth break-even.
5. [Hybrid precision allocation](05-hybrid-precision-allocation.md) -- globally optimal placement of
   UD-IQ3/NVFP4/FP8 representations.
6. [Routing discontinuity](06-routing-discontinuity.md) -- error propagation across discontinuous
   Top-k routing and recurrent state.
7. [DFlash renewal control](07-dflash-renewal-control.md) -- optimal draft/verify/retry policy.
8. [Falsifier sufficiency](08-falsifier-sufficiency.md) -- sufficient causal state and calibrated
   multi-horizon failure control.
9. [Logit-sketch limits](09-logit-sketch-limits.md) -- information limits and optimal sketches for
   full-vocabulary risk.
10. [MLA FP8 stability](10-mla-fp8-stability.md) -- long-context absorbed-MLA FP8 error theory.
11. [Heterogeneous scheduling](11-heterogeneous-schedule.md) -- optimal CPU/GPU/NVMe overlap schedule.
12. [Prefetch information](12-prefetch-information.md) -- value of early information for exact
    expert prefetch.
13. [Hierarchy separability](13-separability-counterexample.md) -- prove or destroy a tempting global
    cache separability conjecture.

## Submission standard

A strong solution contains:

1. precise definitions and any additional assumptions;
2. a theorem, impossibility result, counterexample, or exact/approximate
   algorithm with a proved guarantee;
3. adversarial examples showing why easier approaches fail;
4. a small deterministic reference implementation using only ordinary CPU
   time;
5. numerical evaluation on synthetic instances spanning easy and pathological
   regimes;
6. a section translating the result into an engine decision and listing the
   measurements still required on hardware.

Unsupported intuition, a neural-network proposal without a falsifiable bound,
or a solver that works only on one supplied trace is not a complete solution.
