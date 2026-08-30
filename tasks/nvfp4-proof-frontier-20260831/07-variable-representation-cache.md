# Problem 7: online caching with packed and expanded NVFP4 representations

Repository: https://github.com/novysvet/insignia  
Branch: `codex/glm53-dflash2-4070-super`  
Hardware required: none

## Engine facts

There are `42 * 288 = 12,096` sparse layer-expert keys. An expanded execution
record is 13.5 MiB. The exact packed-scale representation averages 0.94532 of
expanded record bytes on the complete store and has record-dependent escape
length. A packed device slot saves 651,264 bytes (4.60%) relative to the
current expanded execution slot, but consuming it incurs scale decode work.
Measured capacity increased from 281 to 294 DFlash slots (+13). The existing
expand-on-use implementation did not improve wall time.

A request can arrive as a scalar token or as a DFlash union. If an expert is
requested by multiple rows, one body/scale decode can be reused. Rejected draft
tail experts can pollute caches. Host and VRAM have different capacities and
costs; a disk miss, H2D transfer, packed-to-expanded conversion, and GEMV can
overlap only when their dependency/buffer constraints permit it.

## Mathematical problem

For every expert `e`, the cache may hold no copy, a packed copy of size `p_e`,
an expanded copy of size `B=13.5 MiB`, or both. A packed hit has conversion or
inline-decode cost `d_e(r)` depending on row multiplicity `r`; an expanded hit
has lower compute cost but uses more capacity. Copies may be promoted,
demoted, or evicted. Design the offline optimum and an online policy minimizing
committed-token time, not hit count.

## Proof obligations

1. Formulate the offline problem with variable object sizes, representation
   conversion, DFlash burst requests, finite conversion scratch, and overlap.
   Establish hardness and give an exact solver for small instances.
2. Characterize the single-expert promotion threshold when future uses follow
   a known renewal law. Include multiplicity and the opportunity cost of lost
   cache capacity.
3. Show why ordinary equal-slot LRU and byte-LRU can each be arbitrarily bad.
   Construct explicit traces using packed/expanded choices.
4. Derive a competitive, regret, or resource-augmentation guarantee for an
   online policy with imperfect route predictions. Prediction false positives
   must pay conversion, I/O, and eviction pollution.
5. Determine whether a two-threshold hysteresis policy
   (packed -> expanded after `N_up` uses, expanded -> packed after cooling) is
   optimal in a nontrivial stochastic special case. If not, give the smallest
   counterexample.
6. Incorporate host and VRAM jointly. Prove or disprove separability of
   representation choice from admission/eviction and from H2D issue order.
7. Include scale-code escape heterogeneity: records have different `p_e` and
   `d_e`. Derive when it is optimal to favor less-compressible records despite
   their larger size because they are cheaper to decode.
8. Give a robust decision rule when service times and reuse laws are estimated
   with confidence intervals rather than known exactly.

## CPU deliverables

- A deterministic simulator and an offline optimal solver for small traces.
- Implementations of LRU, byte-LRU, packed-only, expand-on-first-use,
  hysteresis, and the proposed policy.
- Adversarial traces and stochastic sweeps over capacity, compression ratio,
  decode cost, burst multiplicity, prediction error, and prompt shift.
- Regret decomposed into extra disk bytes, H2D bytes, conversions, evictions,
  and GPU work.
- An engine-ready admission/promotion state machine with explicit invariants
  for async copies and conversion scratch.

## Decision rule

The result must predict a measurable operating region where +13 slots repay
inline/one-time decoding. Hardware promotion requires increased committed
token/s in a repeated median A/B plus scale byte parity and unchanged exact
outputs. Kill any policy optimized only for hit rate or one stationary trace.
