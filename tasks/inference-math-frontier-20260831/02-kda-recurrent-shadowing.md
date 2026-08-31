# Problem 2: Shadowing theory for quantized KDA recurrent state

Repository: https://github.com/novysvet/insignia  
Reference branch/commit: `codex/glm53-dflash2-4070-super` at `0740c63`

## Engine context

GLM-5.3-Flash has 34 gated-DeltaNet/KDA layers. Each uses 64 heads of width 128,
a depth-4 causal convolution, and an FP32 recurrent state. Dense weights and
some activations may be FP8, while approximate verification or future state
compression could perturb the inputs or the state itself. Unlike a feed-forward
layer error, a state perturbation is carried into every later token. The engine
needs to know whether errors contract, remain bounded, or occasionally explode,
and how often an exact replay/checkpoint is worth paying for.

No private weights are supplied. Work with the following abstract per-head
recurrence, broad enough to cover gated delta rules:

`S_t = A_t S_{t-1} B_t + beta_t u_t v_t^T`,

where `S_t` is `d x d`, `d=128`, `A_t` and `B_t` may be diagonal or structured
gates, `beta_t in [0,1]`, and `u_t,v_t` are functions of normalized token
features. An approximate path produces `S_hat_t` using perturbed coefficients
and a finite-precision map `Q_t`. State any stronger assumptions explicitly.

## Main problem

Develop a non-vacuous finite-horizon shadowing theory for
`Delta_t = S_hat_t - S_t` that can drive an engine checkpoint policy.

1. Derive exact error recurrences separating propagated state error, coefficient
   error, low-rank update error, and rounding/quantization error.
2. Prove the sharpest bound you can under only operator-norm gate bounds. Then
   construct a sequence showing that the naive product-of-norms bound is tight
   or explain the missing structure that makes it loose.
3. Exploit at least one real structural feature—diagonal gates, rank-one
   updates, normalization, shared left/right singular directions, or bounded
   variation—to replace a worst-case exponential bound by a Lyapunov,
   path-length, or energy bound.
4. Define an online statistic computable in `O(d^2)` or less per head that
   upper-bounds future error without comparing against the exact state every
   token. Prove when it is safe and provide an adversarial false-confidence
   example for at least two tempting alternatives.
5. Given exact-reset cost `C_r`, per-token approximate saving `C_s`, and a loss
   budget, derive an optimal or approximation-guaranteed checkpoint/replay
   schedule. The horizon and gate sequence are revealed causally.

## Hard extensions

- Allow exact state snapshots only at DFlash block boundaries and permit a
  rejected block to restore an earlier state. Couple numerical error with
  random accepted lengths.
- Analyze FP8 block scaling where the quantizer scale is itself a discontinuous
  function of the current block maximum.
- Determine whether a backward-error statement is possible: is the approximate
  trajectory exactly equal to the recurrence for nearby inputs and gates, and
  are those perturbations causally realizable?

## Required CPU artifact

Provide a deterministic simulator for `d <= 32` plus exact/symbolic checks for
`d <= 4`. It must generate contractive, marginal, switching-nonnormal, and
adversarial gate sequences; compare observed error with every proposed bound;
and solve the finite-horizon reset problem exactly on small instances. Include
a case where every individual step has spectral radius below one but switching
causes large transient amplification.

## Engine decision

End with a concrete rule specifying which scalars the CUDA KDA path must expose,
where snapshots live, how much work a reset saves or costs, and what event
forces exact replay. Kill the idea of compressed/replayed KDA state if the only
sound certificate requires scanning more data than the exact update or is
vacuous at horizons of a few thousand tokens.
