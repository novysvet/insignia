# Problem 5: certified residual-corrected INT8 activation for NVFP4

Repository: https://github.com/novysvet/insignia  
Branch: `codex/glm53-dflash2-4070-super`  
Hardware required: none

## Engine facts and exact algebra

The checkpoint weights remain NVFP4. In a 16-element group, write

```text
w_i = alpha * phi(u) * z(c_i) / 2,
```

where `alpha` is a matrix FP32 global, `u` is the group's E4M3 scale byte,
and

```text
z(c) in {0, +/-1, +/-2, +/-3, +/-4, +/-6, +/-8, +/-12}.
```

The fast kernel quantizes an FP32 activation group with absmax scale `delta`:

```text
q_i = round_to_nearest(x_i / delta),  q_i in [-127,127],
x_i = delta*q_i + r_i.
```

It reads and expands the weight group once, uses four DP4A operations for
`sum z_i*q_i`, and applies the floating scale. The only approximation in this
identity is the activation residual `r`. Because the GPU has excess integer
compute relative to expert-byte supply, a second residual term can reuse the
already-loaded weights:

```text
r_i = eta*t_i + e_i,
dot2 = delta * sum z_i*q_i + eta * sum z_i*t_i.
```

This costs four additional DP4A operations per group but no additional expert
body or scale read. It may also permit a coarser first-stage activation scale
than group 16, reducing quantization and scale traffic.

## Mathematical problem

Find the optimal multi-stage residual quantizer and a cheap certificate that
chooses zero, one, or more correction stages per activation group or output
tile. The objective is end-to-end time under a quality/routing-risk constraint,
not merely activation MSE.

Consider:

- one-stage group-16 Q8 (the current path);
- one-stage group-32/64/128 Q8;
- Q8 plus a second signed INT8 residual stage;
- a sparse exact-FP32 correction for selected elements or groups;
- bitplane residuals or a shared correction across multiple DFlash rows;
- an exact fallback on rows whose output/router margin certificate fails.

## Proof obligations

1. Derive the exact deterministic output error for one row:

   ```text
   E_r = sum_g alpha*phi(u_rg)/2 * sum_i z(c_rgi)*e_gi,
   ```

   and give useful absolute, relative, and data-dependent upper bounds. Compare
   L1, L2/Cauchy, interval, and signed-cancellation bounds.
2. Prove the worst-case contraction factor of one residual quantization stage
   for absmax rounding. Treat zero groups, saturation, ties, subnormals, and
   FP32 scale rounding.
3. For a fixed compute budget, solve the allocation of correction stages among
   groups or output rows to minimize a certified error norm. Prove when greedy
   selection by bound reduction per operation is optimal and give a
   counterexample when it is not.
4. Connect output error to the next Top-8 router. If router score intervals are
   `[z_j-epsilon_j,z_j+epsilon_j]`, prove the exact set-stability test and
   derive the familiar margin/2 rule as a corollary. Ties use a fixed expert-ID
   rule.
5. Establish a break-even inequality for spending another four DP4As while
   reusing weight bytes versus using the FP32-activation kernel or a narrower
   first-stage group. Include registers, dependency depth, and multi-row `r`.
6. Prove or disprove:

   > Two-stage group-64 Q8 residual quantization always dominates one-stage
   > group-16 Q8 in both worst-case error and activation-side byte traffic.

7. Construct examples with excellent cosine/MSE but a Top-8 flip, and examples
   where a coarse deterministic bound rejects a genuinely safe correction.

## CPU deliverables

- An exhaustive small-group integer/FP32 reference for one- and two-stage
  quantization, including all rounding boundaries.
- A solver for stage allocation under operation and error budgets.
- Synthetic Gaussian, heavy-tail, outlier, cancellation, and router-near-tie
  sweeps reporting MSE, relative L2, cosine, KL, JS, PPL proxy, Top-1, and
  Top-8 changes.
- A concrete fused inner-loop sketch showing that both DP4A chains consume one
  weight expansion; list expected additional accumulators and loads.
- An engine fallback rule and exact measurements needed from real logits.

## Acceptance and kill rules

Promote only if the fused kernel reduces activation error materially at less
than 10% NVFP4-kernel latency cost, or if it enables a faster wider-group path
with at least 10% net gain. It remains approximate unless it reproduces the
FP32 activation DAG. The full engine gate must include hard free-running output
and all listed numerical metrics; same-prefix PPL alone cannot approve it.
