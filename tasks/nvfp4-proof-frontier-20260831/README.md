# Insignia NVFP4 proof frontier

Date: 2026-08-31  
Repository: https://github.com/novysvet/insignia  
Working branch at creation: `codex/glm53-dflash2-4070-super`

This packet contains mathematics and algorithm-design assignments, not requests
to benchmark the private model. Every assignment is self-contained and can be
completed on an ordinary CPU-only computer. A solver may clone the public
repository for implementation details, but must not require either RTX machine,
the 180 GiB model store, private traces, or access to an SSH host.

## Fixed model and format facts

- GLM-5.3-Flash has 45 target layers. The first three MLPs are dense and the
  remaining 42 are sparse MoE layers with 288 routed experts and Top-8 routing.
  A scalar decode token therefore requests `42 * 8 = 336` expert records before
  cache hits or speculative amortization.
- Every routed expert contains three matrices: gate and up are `2048 x 4096`;
  down is `4096 x 2048`. Each matrix has 8,388,608 weights.
- ModelOpt NVFP4 stores two signed E2M1 values per byte. For every consecutive
  group of 16 weights it stores eight body bytes and one unsigned E4M3 scale;
  each matrix also has one FP32 global multiplier. The exact magnitude alphabet
  is `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` and the high nibble bit is the sign.
- One matrix body is 4 MiB and its scale plane is 512 KiB. One expanded expert
  record is therefore 13.5 MiB: 12 MiB of packed bodies plus 1.5 MiB of scale
  bytes, ignoring the three negligible FP32 globals.
- Ada `sm_89` has no native block-scaled FP4 MMA. The current decode path
  quantizes an activation independently in groups of 16 to signed INT8, expands
  each E2M1 code to twice its value in signed INT8, performs four `DP4A`
  operations per 16 products, and applies
  `0.5 * activation_scale * E4M3_scale * global_scale` in FP32.
- The current multi-row kernel loads one expert weight group once and applies it
  to as many as eight DFlash verification rows. The pair kernel evaluates gate
  and up together so their shared activation is quantized only once. Floating
  accumulation order is model-visible because tiny changes can flip later
  Top-8 routes.
- The exact packed-scale sidecar is 0.94532 times the expanded expert size on a
  complete 12,096-record store, with 0.782% scale escapes. Keeping packed scales
  in VRAM saves 651,264 bytes per expert slot (4.60%) and creates 13 additional
  DFlash slots in the measured 16 GiB budget, but the current expand-on-use cost
  erased that capacity benefit. It is not yet a default path.
- The large-machine measurements to use only as supplied constants are: pinned
  H2D bandwidth about 23.2 GB/s; one NVMe about 3.7--4.7 GB/s; 32 GiB pinned host
  cache = 2,425 expanded records; observed host hit plateau about 80.3%; and
  overclocked VRAM bandwidth about 800 GB/s. The current exact DFlash frontier is
  about 5.1--5.3 committed tokens/s. Twelve tokens/s is the target to certify or
  refute under explicit assumptions.
- Approximation is permitted, but the engine must report MSE, relative L2,
  cosine, forward and reverse KL, JS, same-prefix perplexity delta, Top-1, route
  changes, DFlash acceptance, and readable hard-prompt outputs. A same-prefix
  PPL increase up to 3.5% is allowed only if hard free-running answers remain
  useful. An exact proposal must also preserve the existing accumulation order.

## Ranked assignments

1. [Minimal exact E2M1 compute embedding](01-e2m1-compute-embedding.md) - find
   or rule out a better DP4A/tensor-core algebra for Ada.
2. [Multiplicity-aware multi-row schedule](02-multirow-multiplicity-schedule.md)
   - spend registers and compute only where DFlash row sharing pays.
3. [Execute directly from packed scales](03-packed-scale-direct-execution.md) -
   eliminate expanded scale residency without paying an expand-on-use tax.
4. [Bit-exact split-K frontier](04-bitexact-splitk-frontier.md) - prove when
   more parallelism is possible without changing the effective model.
5. [Certified residual-corrected INT8 activation](05-residual-corrected-int8.md)
   - buy back quality with excess compute and a falsifiable error certificate.
6. [Optimal exact prefill schedule](06-prefill-io-compute-schedule.md) - minimize
   expert traffic while respecting layer, recurrent-state, and spill costs.
7. [Variable-size representation cache](07-variable-representation-cache.md) -
   jointly choose packed/expanded representation, eviction, and decode work.
8. [Twelve-token/s feasibility certificate](08-twelve-tps-certificate.md) - give
   a rigorous constructive certificate or impossibility bound for the target.

## Advanced research assignments

These are deliberately harder than the first wave. They couple numerical
analysis, online control, information theory, and machine scheduling; a useful
negative theorem is an acceptable result.

9. [Discontinuous router-cascade certificate](09-router-cascade-certificate.md)
   - turn local approximation error into a non-vacuous 42-router risk bound.
10. [Certified partial expert execution](10-certified-partial-execution.md) -
    decide when remaining expert work provably cannot change a route or token.
11. [Cache/speculation semi-Markov control](11-cache-speculation-control.md) -
    solve adaptive draft length and two-tier caching under policy feedback.
12. [On-policy tiny-MoE falsifier](12-on-policy-tiny-moe-falsifier.md) - train a
    compute-heavy selector with a finite-sample selective-risk guarantee.
13. [Cross-expert rate-distortion code](13-cross-expert-rate-distortion.md) -
    exploit structure shared by 288 experts without assuming it exists.
14. [Dual-SSD deadline scheduler](14-dual-ssd-deadline-scheduler.md) - jointly
    place, replicate, and dispatch routed records across unequal drives.
15. [MoE red-blue pebble bound](15-moe-io-pebble-bound.md) - prove the minimum
    traffic of exact layer-major prefill with finite sidecar memory.
16. [Floating-point equivalence certificate](16-fp-equivalence-certificate.md)
    - mechanically certify bit-exact CUDA reduction rewrites.

## Submission contract for every assignment

A complete response contains:

1. explicit definitions and assumptions, separating proved statements from
   empirical parameters;
2. at least one theorem, lower bound, impossibility result, or counterexample;
3. a constructive algorithm when the result is not purely negative;
4. a deterministic CPU reference implementation and tests on synthetic easy,
   boundary, and adversarial cases;
5. sensitivity plots or tables with every free parameter exposed;
6. an engine decision rule: exact code seam, required measurements, pass/fail
   thresholds, and a kill criterion;
7. no invented GPU throughput, route distribution, cache hit rate, or quality
   result. Unknown hardware constants must remain symbolic.

The most valuable answer can be a proof that a tempting optimization cannot
win. That prevents an expensive CUDA sidequest and is therefore a performance
result.
