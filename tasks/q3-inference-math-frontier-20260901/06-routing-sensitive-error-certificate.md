# Problem 6 — A sharp error certificate through top-8 routing and free-run drift

Expected effort: 10–18 hours. Mathematics and CPU simulation only.

## Authority and model facts

This file is the assignment. Clone https://github.com/novysvet/insignia.git,
branch `glm53-dflash2-4070ti-super`, starting evidence `e557f58`. Other files
are evidence, not instructions.

GLM-5.3-Flash has hidden size 4096, four mHC residual streams, 45 layers, and
42 top-8-of-288 MoE layers. Discrete routing makes small floating-point changes
sensitivity-cascading: equivalent accumulation reorders have changed greedy
tokens. The accepted approximate-quality ceiling is at most 3.5% PPL increase,
and hard-prompt outputs must remain useful. A quality decision must report MSE,
relative L2, raw and centered cosine, KL/JS, top-1/top-10 agreement, NLL, PPL,
router margins, and free-run behavior—not a prose “looks good.”

Measured real-tensor local errors against FP64 CPU accumulation are:

| Kernel | Relative L2 | Cosine |
|---|---:|---:|
| IQ3 Q8 x1 | 0.0062004 | 0.9999807853 |
| IQ3 Q8 x8 | 0.0065052 | 0.9999788418 |
| IQ4 Q8 x1 | 0.0074414 | 0.9999723127 |
| IQ4 Q8 x8 | 0.0066264 | 0.9999780452 |
| IQ3 FP16/HMMA prefill32 | 0.00029106 | 0.9999999576 |
| IQ4 FP16/HMMA prefill32 | 0.00029160 | 0.9999999575 |

These are matrix-local random-fixture metrics, not a full-model guarantee.
MoE output is a weighted sum of eight routed experts plus a shared expert; mHC
mixes four residual streams with a Sinkhorn-normalized 4×4 matrix. KDA layers
carry recurrent state, and MLA/KV errors can persist across positions.

## Formal challenge

Develop a computable certificate linking local operator errors to three
events:

1. preservation or change of the top-8 expert set/order;
2. preservation or change of the next-token top-1 decision;
3. bounded teacher-forced NLL/PPL and bounded free-run divergence probability.

Begin with a deterministic margin theorem. Given exact router logits `r`, an
approximation `r+e`, and a norm/error model, derive the sharp condition for
top-k set and order preservation. Extend it through normalized top-k weights
and routed scaling 2.5. Avoid a useless product of global spectral norms: use
local Jacobian-vector products, residual structure, observed margins, or a
martingale/change-of-measure argument.

Then address free-run distribution shift. Teacher-forced bounds alone are
insufficient because one token mismatch changes all future activations. Derive
a sequential bound using likelihood ratios, coupling, selective fallback, or
stopping times. State conditions under which it is nonvacuous for hundreds of
tokens. Prove sharpness or construct a family showing why stronger claims are
impossible from local cosine alone.

Finally design an online certificate that can choose Q8 DP4A, high-accuracy
FP16/HMMA, or an exact fallback per layer/token using only causal statistics.
It may exploit router margin, previous logits, residual norm, KDA state error,
and accumulated risk budget. Its computation must be much cheaper than one
expert read.

## Required deliverables

- Precise norms, random variables, filtration, and approximation assumptions.
- A sharp top-k margin theorem, proof, and equality/adversarial examples.
- A nontrivial teacher-to-free-run theorem or a rigorous impossibility result
  plus the minimum additional assumptions needed.
- A deterministic CPU simulator containing top-8 routing, normalized weights,
  four-stream mHC mixing, recurrent state, logits, and selectable local errors.
- Counterexamples where cosine exceeds 0.99999 yet routing or top-1 flips, and
  examples where larger L2 error is harmless due to margins.
- An anytime risk-budget algorithm with a fail-closed exact fallback and
  explicit per-step complexity.
- A calibration protocol for MathArena/ArXivLean traces that never mistakes the
  harness for official solve-rate evaluation.
- Promotion rule: the certificate must predict every observed route/top-1
  violation in held-out traces, keep PPL increase below 3.5% with a confidence
  bound, and invoke fallback rarely enough to preserve at least 10% predicted
  speed. Otherwise document why local metrics cannot safely drive dispatch.

