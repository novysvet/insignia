# Engine handoff: joint adaptive compute controller

## 1. Smallest schema

### 1.1 Finite research state

The exact CPU artifact needs only:

```text
belief_index       0..6     posterior token-difficulty bin
cache_level        0..2     cold / partial / warm
io_queue           0..2     empty / occupied / saturated
quality_debt       0..1     request monitor has seen harmful divergence
phase              0..1     exact metric still available / already consumed
```

This is the 252-state finite controller. It is sufficient for the synthetic model, not for production calibration or OOD detection.

### 1.2 Minimal online observation

The smallest production observation that preserves the artifact's causal update and safety gate is:

| field | minimum representation | source |
|:---|:---|:---|
| previous-logit log-likelihood ratio | signed 16-bit fixed point | previous committed target logits and route churn |
| posterior token difficulty | unsigned 16-bit fixed point | online Bayes filter |
| posterior request hardness | unsigned 16-bit fixed point | request-level filter; reset at request start |
| previous accepted prefix | 4 bits | DFlash commit record |
| cache level or hit predictor | 2 to 8 bits | expert cache counters |
| I/O queue level | 2 to 8 bits | reader and H2D queue counters |
| quality-debt/risk monitor | 1 bit plus remaining risk budget | request monitor |
| metric-used flag | 1 bit | block controller |
| likelihood surprise | 8 to 16 bits | OOD filter |
| support/overlap score | 8 to 16 bits | logged-policy support model |
| previous block time residual | signed 16-bit fixed point | measured minus predicted time |
| three virtual quality queues | 3 fixed-point words | primal-dual controller |

Raw logits do not need to enter the controller. The previous-logit likelihood ratio is the sufficient statistic for the calibrated binary likelihood. Retain raw margin and route-churn bins in telemetry for recalibration and OOD audits.

A 64-byte cache-line-aligned controller record is adequate. Do not place variable-length route lists in this hot record; store a compact churn/overlap statistic and keep the list in existing route telemetry.

### 1.3 Action encoding

The current finite action set fits in one byte:

```text
bit 0    expert_k       0 = 4, 1 = 8
bit 1    draft_k        0 = 2, 1 = 4
bit 2    verify_policy  0 = approximate delta, 1 = exact full
bit 3    precision      0 = FP8, 1 = BF16
bit 4    prefetch       0 = none, 1 = one synthetic budget unit
bit 5    measurement    0 = act, 1 = acquire exact metric first
bit 6    exact_fallback set by safety gate
bit 7    reserved
```

Engine code should use an enumerated action table rather than branches over each coordinate. The table contains prevalidated kernel IDs, checkpoint layout, capture mask, prefetch limits, and retry behavior. Expanding draft widths or verification modes changes the table, not the controller ABI.

The block-level safe action is:

```text
expert_k=8, draft_k=2, verify=exact_full,
precision=bf16, prefetch_budget=0, measurement=0
```

## 2. Offline data requirements

Every logged row must be causally ordered and contain the behavior-policy propensity. Required fields are:

- request ID, block ID, request boundary, prompt class, and hard-answer label when available;
- pre-action observation, posterior, OOD score, overlap score, virtual queues, and remaining risk allocation;
- full selected tuple, behavior propensity, safety-gate result, rejected proposal, and fallback reason;
- action start and finish timestamps for controller, metric, snapshot, draft, I/O, verify, rollback, and commit;
- accepted-prefix length, committed tokens, exact fallback occurrence, retry, and verification outcome;
- previous and next route summaries, expert union size, cache hits, cache promotions, queue depths, and bytes by device;
- next causal logit features and target-logit checksum;
- request-level excess NLL or exact-shadow divergence, hard-answer violation, catastrophe label, and monitor-state transition;
- exact shadow action and outcome on the paid evaluation subset;
- software revision, model hash, checkpoint layout, GPU clock regime, storage device, and thermal state.

The minimum identification requirement is overlap:

\[
P_{\rm log}(A=a\mid O=o)>0
\]

for every state-action region the new policy may use. Clip-free off-policy evaluation is invalid when this fails. Stratified exact shadows are required near action boundaries, high request-hardness posterior, debt states, queue saturation, and every OOD bucket.

Estimate unknown parameters with uncertainty:

- logit likelihood parameters and calibration intervals;
- multinomial belief, cache, queue, and debt transitions;
- accepted-prefix survival distributions by tuple and state;
- latency and byte distributions with p95 and p99 tails;
- request-end, hard-request, quality-damage, and catastrophe probabilities;
- controller and exact-metric cost distributions;
- action-dependent next-route and cache effects.

Use confidence sequences or held-out bootstrap regions. Do not replace unmeasured interactions with zero.

## 3. Online update

At each committed block anchor:

1. Read previous-logit margin and route churn. Convert them to calibrated log-likelihood ratio.
2. Predict token regime through the action-dependent transition model.
3. Add the observation log-likelihood ratio to prior log odds and project to the finite bin or retain the continuous posterior.
4. Update request-hardness posterior separately. Reset it only at request start.
5. Update cache and queue state from completed operations, not submitted operations.
6. Update quality debt, remaining request risk, and three virtual queues from committed outcomes.
7. Update timing and likelihood surprise. Inflate the uncertainty set when residuals exceed calibration.
8. Form the admissible action set using robust quality, overlap, economic-gain, and deadline checks.
9. Acquire the exact metric only when its net value is positive and its deadline is feasible. Charge its time and bytes.
10. Select the joint tuple. If no nonexact tuple is admissible, emit the safe exact action.
11. Log the proposal, certificate, propensity, selected tuple, and eventual outcome.

The posterior update for observation `o` is

\[
\operatorname{logit}q_{n+1}
=
\operatorname{logit}\hat q_{n+1|n}
+
\log\frac{P(o\mid D=H)}{P(o\mid D=E)}.
\]

The online Dinkelbach estimate can use cumulative committed utility divided by cumulative wall time. Virtual queues update with global PPL, hard-request, and catastrophe increments separately.

## 4. Safety envelope

A proposed tuple is allowed only when:

```text
uncertainty_radius <= configured maximum
support_probability >= p_min
worst_case_ppl_ratio <= reserved PPL envelope
worst_case_hard_fail <= reserved hard-answer envelope
worst_case_catastrophe <= reserved catastrophe envelope
LCB(incremental transformed value over exact) > 0
controller_p99 + optional_metric_p99 fits every deadline
request risk allocation remains nonnegative
```

Fallback triggers include:

- likelihood surprise or timing residual outside the calibrated envelope;
- support below `p_min`;
- missing exact-shadow overlap for the proposed action region;
- quality certificate failure;
- controller, metric, I/O, or GPU deadline miss;
- debt monitor entering a forbidden state;
- NaN, stale epoch, model-hash mismatch, or telemetry loss;
- empty robust action set.

Fallback must be branchless at the final dispatch boundary: overwrite the proposed action code with the prevalidated exact action code.

Global controller kill conditions are:

1. p99 controller plus paid-metric cost consumes the lower-confidence predicted saving.
2. Safe exploration lacks overlap for any action needed by the candidate policy.
3. No uncertainty-certified policy improves on the best fixed configuration by at least the materiality threshold, set to 3% in the artifact.
4. The OOD latch rate or exact-fallback rate exceeds the operating ceiling.
5. Held-out hard-answer or catastrophe confidence bounds exceed budget.
6. Model calibration fails after a software, checkpoint, clock, storage, or prompt-distribution change.

A global kill selects the best fixed configuration that passed the same robust quality audit. A per-block certificate failure selects conservative exact inference.

## 5. GPU and I/O deadlines

Let `t0` be the previous block's commit, `t_io` the expert-I/O submission point, `t_snap` the recurrent snapshot decision, `t_draft` the DFlash launch, `t_verify` the target verification launch, and `t_commit` the irreversible commit. All numeric margins remain hardware measurements. The controller stores symbolic p99 requirements until those measurements exist.

| decision | last causal evidence | must finish before | late behavior |
|:---|:---|:---|:---|
| update posterior and OOD score | previous committed logits, routes, acceptance, timing | `min(t_io, t_snap) - controller_p99_margin` | exact action |
| decide whether to buy exact metric | posterior, cache, queues, risk state | metric launch slot before any irreversible action-specific work | skip metric and exact action |
| exact metric acquisition | pre-action state only | `min(t_io, t_snap) - metric_p99_margin` | discard metric result, exact action |
| expert count and precision | posterior after optional metric | expert union construction and kernel/graph selection | exact action code |
| prefetch budget and device assignment | routes, cache, queue state | async read submission `t_io` | no new prefetch; keep existing reads |
| DFlash width and checkpoint layout | posterior after optional metric | snapshot allocation `t_snap` | width 2, full safe snapshot |
| capture feature mask | selected action and target graph | capture flag publication before target layers execute | safe capture set |
| verification policy and depth | selected tuple | target verify launch `t_verify` | exact full verification |
| retry or rollback | verification result and exact metric | recurrent/cache commit `t_commit` | rollback and exact token |
| cache promotion or demotion | accepted prefix and verified route | cache metadata publication after commit | conservative demotion |
| next-block speculative prefetch | newly committed routes | next I/O slack deadline | omit prefetch |

For every action-specific deadline `d`, require

\[
c_{\rm controller}^{p99}+1_{\rm metric}c_{\rm metric}^{p99}
\le d-t_0-m_d,
\]

where \(m_d\) is the measured scheduling margin. A tuple that cannot meet this inequality is absent from the admissible set.

## 6. Initial integration order

1. Add telemetry and exact-shadow logging with the controller disabled.
2. Fit likelihood and transition confidence sets offline.
3. Replay the finite solver and robust gate on logs.
4. Enable shadow decisions only; compare predicted action against fixed execution.
5. Enable exact metric acquisition on a bounded paid budget.
6. Enable nonexact actions for states with overlap and a positive robust gain certificate.
7. Expand support only through anytime-safe exploration.
8. Kill the controller immediately when any global gate fails.

The hot path should contain a table lookup, a small posterior update, fixed-point certificate comparisons, and one final action-code overwrite. LP solving, confidence-set fitting, and policy-table generation remain offline.
