# Problem 9: Causal predictive coding of expert computation

Repository: https://github.com/novysvet/insignia  
Reference branch/commit: `codex/glm53-dflash2-4070-super` at `0740c63`

## Engine context

A decode token requests 336 layer-expert records before cache hits. Successive
tokens and adjacent layers are not independent: routes, hidden states, logits,
and expert outputs may be conditionally predictable. The GPU has much more
arithmetic bandwidth than the SSD path has byte bandwidth. One possible trade
is to keep a cheap predictor or shared basis resident, then fetch/compute only a
residual. Unlike ordinary weight compression, the predictor may use previous
token logits, routes, and hidden summaries.

This assignment asks when such a causal residual representation is possible and
when it is information-theoretically doomed. No real trace is supplied; every
correlation must be a parameter rather than an invented fact.

## Formal model

For expert `e` and token state `x_t`, exact output is `f_e(x_t)`. Before the
deadline, the engine knows causal context `C_t` containing earlier routes,
selected logits/statistics, cache contents, and perhaps early current-layer
features. A resident predictor returns `p_e(x_t,C_t)`. Exact execution may read
a residual representation `r_e` from storage and compute a correction; an
approximate mode may omit or truncate it. Costs include resident bytes,
read bytes, GPU operations, CPU operations, synchronization, and prediction
misses.

## Main problem

1. Give a no-free-lunch lower bound for arbitrary expert maps: without a
   structural prior, causal context cannot reduce the exact information that
   must be stored/read. Make the computational and query model explicit.
2. Under a measurable structure—shared low-rank basis, block sparsity,
   conditional codebook, Lipschitz motion of activations, or repeated route
   clusters—construct an exact predictor-plus-residual code. Account for random
   access and prove its byte and operation costs.
3. Optimize a hybrid exact/approximate code where residual chunks have unequal
   value. Derive the optimal causal stopping rule under an error or selective-
   risk constraint. Use previous logits as information, but prove what they
   predict; do not merely concatenate features into a network.
4. Quantify the value of causal side information using conditional
   rate-distortion or Wyner--Ziv-style reasoning, noting that decoder side
   information and the nonlinear query `f_e(x)` complicate textbook source
   coding. Establish a bound that can be estimated from traces.
5. Include cache placement: a full expert, predictor basis, residual chunk, or
   lower-precision copy competes for each byte of RAM/VRAM. Formulate the joint
   representation-cache problem and solve small cases exactly.

## Hard extension: correctness-preserving speculative residuals

Begin computing a predicted expert output while the exact expert record is in
flight. When the residual arrives, correct the output without repeating shared
work and without changing FP32 accumulation order. Determine the algebraic
conditions and buffer schedule that make this possible. Prove a lower bound on
unavoidable duplicate work when gate/up nonlinearities are not shared.

## Required CPU artifact

Implement synthetic expert families ranging from independent random matrices to
exact shared-basis and slowly drifting constructions. Compute empirical
conditional rate-distortion curves, exact small cache/representation optima,
and predicted system cost. Include adversarial context that is highly
predictive of route IDs but carries zero information about expert residuals.

## Engine decision and kill criterion

Specify the trace fields and statistical test needed before writing a kernel,
then give exact layout and dispatch semantics for any positive scheme. Kill the
idea if conditional residual entropy is too high, predictor residency displaces
more useful full experts, correction misses the layer deadline, or exact
correction cannot preserve model-visible arithmetic order.
