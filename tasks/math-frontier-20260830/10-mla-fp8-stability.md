# Problem 10: stability of absorbed MLA with FP8 latent history

Repository: https://github.com/novysvet/insignia
Hardware required: none

## Geometry

GLM attention uses 64 heads of width 256 and a 512-wide compressed KV latent.
The engine can absorb per-head K/V projections into the query/output path and
store group-scaled E4M3 latents for contexts up to 8,192. A hybrid exact bridge
retains exact expanded K/V for the first 256 positions. Cross-head FP8 compute
is an optional faster approximation.

For one head, write the exact absorbed attention as

```text
s_i = q^T U_k z_i / sqrt(d),
p_i = exp(s_i) / sum_j exp(s_j),
o = U_v sum_i p_i z_i.
```

Latents, absorbed queries, scales, exponentials, and online-softmax merge states
may all be perturbed or rounded.

## Main problem

Derive nonvacuous forward-error bounds for `o` at long context which exploit
score gaps, latent norms, and softmax concentration rather than multiplying
worst-case Lipschitz constants 8,192 times.

## Required results

1. Bound output error from groupwise latent quantization and separate query
   quantization, allowing per-token scales.
2. Analyze stable online-softmax partial/merge arithmetic, including error in
   maxima and denominators. Compare sequential, tiled, and tree merge orders.
3. Show how error depends on attention entropy or the gap between dominant and
   nondominant scores. Identify regimes where context length drops out of the
   bound and regimes where it cannot.
4. Extend the analysis to cross-head quantization groups. Determine when sharing
   a scale across `H` heads is better or worse than independent scales after
   accounting for tensor-core utilization.
5. Construct adversarial latent/query sequences that make tiny per-element FP8
   error cause a large attention-output change.
6. Optimize scale grouping or mixed precision under a byte/compute budget. A
   useful result would assign higher precision only to tokens/heads near an
   attention decision boundary.

## Quality translation

Relate attention-output error to hidden cosine/MSE and, with explicitly stated
assumptions, to probability of changing a later MoE route. Do not claim a PPL
bound without addressing the discrete routing boundary.

## Deliverables

- Error theorems and tightness examples.
- Arbitrary-precision CPU reference for exact and quantized absorbed attention.
- Numerical sweeps to context 8,192 over concentrated, diffuse, and adversarial
  attention.
- A proposed runtime precision rule with quantities the engine can measure
  cheaply before attention.
