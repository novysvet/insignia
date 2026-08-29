# Session 10 - adaptive router-mass pruning verdict

Date: 2026-08-30  
Deliverable: `adaptive-router-mass-pruning-20260830.tar.zst`  
Archive SHA-256: `a982b88e505492a4a0063e1b46bf0c7ecb3382c19f9498e2e2a822432a326292`

## Verdict

**Reject offline. Do not implement and do not spend glm-box time on this
policy.** The trace shows that selected expert weights are not small enough for
mass-based pruning to remove material traffic safely. Even a noncausal
corpus-global oracle misses the task gate by more than fivefold.

At a 1% mean omitted normalized-mass budget, the optimistic oracle removes only
2.8776% of expert records (9.669 records/token). For a 15% record reduction,
the minimum possible mean omitted mass is 7.4248%. No online policy can beat
that oracle.

## Independent validation

- All 23 files in the returned SHA-256 manifest verify.
- The supplied parser/policy tests pass 10/10 locally.
- The original 608,044-row trace archive was extracted independently. The
  analyzer accepted 608,034 rows / 14,477 complete tokens and rejected the one
  known ten-row interrupted tail.
- A fresh run against the current repository reproduced `oracle_bound.csv`,
  `score_semantics.csv`, `per_layer_thresholds.csv`,
  `offline_gate_results.csv`, `policy_prompt_metrics.csv`, and
  `traffic_projection.csv` exactly after ignoring CRLF/LF differences.
- The fresh run stopped only after producing the core results because the local
  bundled Python lacks optional Matplotlib. The returned archive already
  contains checksum-valid plots; plotting does not affect the verdict.
- Current `src/glm53_generate.cu` confirms the assumed semantics: corrected
  sigmoid-plus-bias scores choose the top eight, while the mixture coefficients
  normalize the eight raw sigmoid scores and preserve slot accumulation order.

## Falsifying numbers

| Policy | Records removed | Mean omitted mass | p99 omitted mass | Records saved/token |
|---|---:|---:|---:|---:|
| fixed top-7 | 12.5000% | 7.6087% | 11.1091% | 42.000 |
| fixed top-6 | 25.0000% | 15.8561% | 22.5366% | 84.000 |
| per-row 1% budget | 0.0034% | 0.0001% | 0.0000% | 0.011 |
| per-token 1% budget, 4% layer cap | 0.3863% | 0.1002% | 3.6770% | 1.298 |
| leave-one-prompt-out 1% target | 0.1519% | 0.0527% | 0.0000% | 0.510 |

The lowest selected expert alone carries 7.609% mean normalized mass and
11.109% at p99. This is why conservative threshold policies almost never
prune, while fixed top-6/top-7 policies remove substantial model contribution
at every sparse layer.

## Aggressive-mode check

The user's speed-first quality tolerance does not rescue this proposal. At the
production 80% host-hit assumption and 4.7 GiB/s NVMe, fixed top-6 has only a
45.135 ms/token transfer-channel upper bound. Against the representative
~500 ms/token real-prompt decode, that is at most about 9% before routing
divergence and pipeline effects, while deleting 15.86% of routed coefficient
mass across all 42 sparse layers. Fixed top-7 has roughly half the traffic
ceiling and still drops 7.61% mean mass.

That is not an extreme enough speed ceiling to justify the expected cascading
quality loss. Revisit only if a future architecture makes expert transfer a
much larger fraction of wall time or supplies a learned residual/surrogate for
the omitted experts. Plain mass pruning is closed.

## Integration decision

No production source, environment knob, or benchmark campaign was added. The
smallest correct integration is this negative-result audit and marking the
handoff complete.
