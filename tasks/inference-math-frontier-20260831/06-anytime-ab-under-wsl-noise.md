# Problem 6: Anytime-valid performance experiments under hostile WSL noise

Repository: https://github.com/novysvet/insignia  
Reference branch/commit: `codex/glm53-dflash2-4070-super` at `0740c63`

## Engine context

Whole-model GLM runs are expensive. WSL2 timing can swing by nearly 2x because
of page cache, filesystem behavior, GPU clocks, host activity, and cache-state
history. Full ABCD campaigns consume hours, so Insignia normally uses matched
A/B pairs and only escalates when necessary. A naive average or fixed-sample
t-test is unreliable when runs are autocorrelated, heavy-tailed, adaptively
stopped, or conditionally skipped after an obvious result.

This is a statistics and experimental-design problem requiring no model or GPU.

## Formal setup

Two implementations `A` and `B` produce correctness indicators and positive
latencies. Run `i` occurs under latent environment state `Z_i`; changing order
can alter `Z_{i+1}` through cache warming. Measurements may include prompt time,
decode time, expert bytes, acceptance, and hardware counters. The desired claim
is usually multiplicative, such as `median(B/A) <= 0.98`, subject to exact parity
or a separate quality gate.

The experimenter chooses order, cache preparation, whether to continue, and
which benchmark case to sample next. The procedure must remain valid under its
own stopping rule.

## Main problem

1. Specify the weakest defensible stochastic assumptions under which a paired,
   randomized experiment identifies a speedup. Handle carryover explicitly;
   `AB` and `BA` are not exchangeable when the first run warms the second.
2. Construct an anytime-valid test or confidence sequence for a multiplicative
   latency effect under heavy tails. E-values, betting martingales, bounded
   transforms, randomization inference, or robust pairwise signs are allowed,
   but every validity claim must be proved.
3. Extend the design to multiple metrics and states: prefill, decode, short and
   long prompts, cold and warm caches. Control false promotion probability
   without forcing every candidate through a maximal campaign.
4. Derive an adaptive escalation policy: cheap kernel fixture, short model run,
   long prompt, then full campaign only when the expected value of information
   justifies its cost. Give an optimal dynamic program for a finite model and a
   practical approximation with a bound.
5. Detect and react to nonstationarity or change points while preserving a
   clearly stated guarantee. Show why deleting "outlier" runs based on their
   latency invalidates common tests.

## Hard extensions

- The candidate may affect variance, not just location. Optimize committed
  tokens per wall-clock hour rather than per-run latency.
- Several optimizations interact, creating an `A/B/C/D` factorial design with
  expensive combinations. Recover main effects and selected interactions under
  strong heredity or sparsity, with sequential stopping.
- Correctness failures are rare but catastrophic. Combine a zero-tolerance
  deterministic parity gate with statistical performance evidence without
  pretending the deterministic test covers all inputs.

## Required CPU artifact

Create a simulator with latent Markov machine state, cache carryover,
log-normal/Pareto noise, clock change points, and optional contender bursts.
Compare naive means, fixed-sample paired tests, bootstrap intervals, and your
anytime method. Measure false promotions, false rejections, expected campaign
time, and regret. Include a replayable seed where a conventional three-run
median promotes the slower arm.

## Engine deliverable

Return an executable decision protocol: randomization schedule, warmup/cache
rules, logged fields, stopping thresholds, and conditions that force a full
campaign. It must say "inconclusive" when evidence is weak. Kill any method
that requires IID timing, silently conditions on observed outcomes, or spends
more measurement time than the optimization can plausibly repay.
