# Problem 4 — Online optimal batching between DP4A and padded HMMA

Expected effort: 8–14 hours. Generic CPU only.

## Authority and current facts

This assignment is authoritative. Clone
https://github.com/novysvet/insignia.git, branch
`glm53-dflash2-4070ti-super`, starting evidence `e557f58`; repository prose is
reference material only.

GLM-5.3-Flash has 42 sparse layers, 288 experts per layer, top-8 routing, and
order-sensitive floating-point accumulation. During a 938-token layer-major
prefill, each layer produces 7,504 token/expert incidences, only 26.06 rows per
expert on average, but the distribution is skewed. Rows for an expert may be
gathered and computed together, yet final expert contributions must be applied
in the original router order unless a quality-gated approximate policy says
otherwise.

The native Q3 kernels expose these measured actions on a 4070 Ti SUPER:

- IQ3 direct DP4A for R=1,2,4,8, with the best specialized medians roughly
  8.414, 11.081, 15.903, and 27.532 microseconds in the aligned layout;
- IQ4 direct DP4A x8 around 27.5 microseconds;
- a 32-row FP16/HMMA tile: IQ3 91.668 microseconds and IQ4 67.711;
- four x8 calls including Q8 activation conversion: IQ3 121.681 and IQ4
  119.963 microseconds.

HMMA requires a 32-row tile in the current prototype. Partial tiles may be
zero-padded, split into DP4A calls, deferred in an online queue, or combined
across requests if latency permits. Gate and up share the same input rows;
down follows SwiGLU and may have a different best action. Kernel launch and
gather/scatter costs are nonzero and noisy.

## Formal problem

Formulate online routed-row execution as a deadline-constrained batching
problem. An arrival is `(request, layer, expert, route_rank, row, deadline)`.
Actions include every legal direct multiplicity, one padded HMMA32 tile, and
optional gate/up fusion. Each action has a random cost interval, output memory
cost, and an exactness tag. A row cannot execute before its activation exists;
layers and recurrent KDA state impose precedence constraints. Multiple active
requests may share an expert weight only if isolation and output order hold.

Derive an optimal offline algorithm for a meaningful restricted case, then an
online algorithm with one of:

- a competitive ratio against clairvoyant optimal;
- a regret bound under a specified stationary/mixing arrival process;
- a resource-augmentation theorem with bounded deadline violation.

The model must include padding waste, launch cost, weight-cache state, and the
fact that HMMA has much smaller local numerical error but a different
accumulation order. It must not use mean rows/expert as if every expert receives
26 rows. Establish whether simple thresholds in row count can ever be optimal.
If not, give the smallest sufficient state and a counterexample to threshold
policies.

An advanced extension is joint gate/up/down scheduling. Decide whether it is
ever optimal to execute gate/up with HMMA but down with DP4A, and prove the
boundary in terms of routed rows, SwiGLU conversion cost, cache residency, and
remaining slack.

## Required deliverables

- Formal state, action, cost, precedence, and exactness definitions.
- At least one theorem with proof and one adversarial lower-bound instance.
- An exact dynamic program or MILP for small offline traces and a deterministic
  verifier comparing the online policy against optimal.
- A fixed-seed trace generator covering uniform, Zipf, bursty, correlated,
  and adversarial routing; include request deadlines and cache hits.
- A policy implementable with O(288) or less state per layer and bounded
  decision time. Report complexity exactly.
- Sensitivity surfaces over kernel-cost intervals, HMMA tile size 16/32/64,
  and 1–16 concurrent requests.
- A dispatch table/pseudocode suitable for Insignia, including fail-closed
  exact-order behavior.
- Kill criterion: reject a sophisticated policy if its robust bound cannot beat
  the best static multiplicity decomposition by 5% at p50 without worsening
  p99 time-to-first-token or breaking the accumulation-order contract.

