# Problem 11: Joint cache/speculation control as a semi-Markov process

Repository: https://github.com/novysvet/insignia  
Branch: `codex/glm53-dflash2-4070-super`  
Hardware required: none; all experiments must be synthetic/CPU.

## Mission

Solve—or sharply delimit—the joint online decision problem for DFlash draft
length, NVMe/RAM/VRAM expert residency, prefetch, and packed-versus-expanded
representation. These controls interact: a longer speculative block changes
the expert union, which churns the cache, changes round cost, changes future
acceptance, and changes the data used to estimate the policy.

## Fixed constants and observations

- 42 sparse layers, 288 experts/layer, Top-8; 336 scalar requests/token.
- Expanded expert record: about 13.5 MiB. Packed sidecar ratio: 0.94532.
- Host tier: 2,425 records at 32 GiB; scalar hit plateau about 80.3%.
- VRAM expert tier: roughly 281--383 records depending on mode.
- Pinned H2D: about 23.2 GB/s. Single NVMe: about 3.7--4.7 GB/s.
- DFlash supports up to seven drafted candidates. Real acceptance varies
  strongly by prompt. An accepted round yields the matched prefix, while a
  rejection still changes cache contents and estimator observations.
- The current exact DFlash frontier is about 5.1--5.3 committed tokens/s.

Let the action include `k in {0,...,7}`, admission/eviction/prefetch choices,
and per-slot representation. Let elapsed wall time be the renewal cost and
committed tokens the reward. Observations are censored by the chosen `k`, and
the policy affects future state and data distribution.

## Required result

1. Formulate the smallest defensible semi-Markov/POMDP state. Prove which
   summaries are sufficient under explicit assumptions and give
   indistinguishable histories showing when they are not.
2. Derive an optimal policy for at least one nontrivial special case and a
   hardness or non-indexability result for the full coupling.
3. Construct an online policy with a finite-time regret, competitive, or
   robust-control guarantee under censored acceptance and drifting I/O costs.
   Include safe exploration; always choosing the currently estimated `k` is
   known to deadlock without probes.
4. Account for cache pollution caused by rejected rows and for variable-size
   packed/expanded slots. The cost model must separate NVMe, H2D, expansion,
   dense compute, and overlap rather than fitting one opaque scalar.
5. Analyze stability and oscillation under policy feedback. Supply an explicit
   counterexample for any unjustified two-timescale assumption.

## CPU deliverable and gate

Submit `smdp_controller.py`, deterministic event-driven simulation, exact
dynamic programming on tiny instances, scalable policy code, and adversarial
regime-change tests. Compare against scalar, fixed-k, oracle Belady, oracle-k,
LRU, and a clairvoyant joint optimum. Report committed ms/token, regret,
cache/H2D/NVMe counts, probes, oscillations, and confidence calibration.

Completion requires recovery of the exact tiny-instance optimum, declared
assumptions for every theorem, and a deployable O(1) or O(log C) per-event
policy with a kill criterion. Never invent a throughput result; leave unknown
hardware quantities symbolic or use only the constants above.
