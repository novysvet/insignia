# Problem 4: Can coded expert storage beat exact striping?

Repository: https://github.com/novysvet/insignia  
Reference branch/commit: `codex/glm53-dflash2-4070-super` at `0740c63`

## Engine context

Each of GLM-5.3-Flash's 42 sparse layers has 288 routed experts and selects
eight per token. One NVFP4 expert record is about 13.5 MiB. On the local machine
the records can be placed across two unequal SSDs; cache misses have deadlines
set by the layer execution order. Ordinary striping stores each record whole.
The GPU often has surplus arithmetic relative to storage bandwidth, suggesting
an uncomfortable question: can parity, linear coding, shared bases, or coded
computation reduce bytes or tail latency while preserving the exact expert
sum?

This problem can be solved on a generic CPU. Do not assume that independently
trained expert weights are low-rank or correlated unless that property appears
as an explicit parameter and is tested adversarially.

## Mathematical model

At one layer, expert `e` implements

`f_e(x) = W_down,e [silu(W_gate,e x) elementwise_mul (W_up,e x)]`.

The router requests a set `R` of eight experts and weights `a_e`; the required
output is `y = sum_{e in R} a_e f_e(x)` in the engine's prescribed accumulation
order. Records live on devices with rates `B_1,B_2`, queueing/service-time laws,
and optional failures. A cache reveals some records for free. The GPU may spend
at most `C` additional operations and `M` bytes of temporary VRAM.

A storage code may place arbitrary functions of expert blocks on either drive,
but total stored bytes and write amplification must be counted. Exact means the
same decoded FP32 values and accumulation order, not merely close real-number
arithmetic.

## Main problem

1. Prove an information-theoretic or algebraic lower bound for arbitrary expert
   tensors: how many source-equivalent bytes must be read to answer every
   possible `(R,a,x)` exactly? State the word/RAM model and whether preprocessing
   is allowed. Determine whether generic MDS parity can ever reduce normal-case
   bytes rather than merely survive a failed/slow drive.
2. Identify the strongest nontrivial structure under which coding can win.
   Candidate structures include a shared basis plus sparse residuals, exact
   duplicate blocks, a low-dimensional subspace of router coefficient vectors,
   or blockwise integer relations among quantized weights. Give a constructive
   code and account for decode compute, metadata, alignment, random access, and
   exact floating-point order.
3. Because the expert contains nonlinear gate/up operations, determine which
   linear combinations can be pushed through the computation and which cannot.
   Give explicit counterexamples to invalid "read a coded expert and decode the
   weighted sum" arguments.
4. Add two-device deadlines and caches. Jointly choose uncoded placement,
   replicas, parity blocks, and read dispatch to minimize expected layer stall
   or a high percentile. Give an exact algorithm for small instances and a
   proved approximation or competitive bound for a scalable restriction.
5. Analyze systematic codes as straggler hedges: start ordinary reads, launch
   parity reads after a causal timeout, and cancel losers. Derive the optimal
   timeout under a specified service distribution and include cancellation
   waste in the objective.

## Hard extension: coded routed sums

Let `g(x)=(f_1(x),...,f_E(x))` be treated as an unknown vector over a finite
field or exact integer ring after quantized block decoding. The query asks for
`a^T g(x)` where `a` is eight-sparse and comes from a known family `A`. Derive
the minimum number of stored linear measurements needed to answer every query
in `A` with at most `r` reads. Relate the answer to the rank, spark, or covering
structure of `A`, then explain what breaks when returning to nonlinear experts
and FP32 order.

## Required CPU artifact

Build a block-level simulator with configurable expert correlations, two
service-time distributions, cache state, query family, and code. Verify exact
reconstruction byte-for-byte on integer fixtures. Compare uncoded optimal
placement, replication, MDS straggler protection, shared-basis coding, and your
method. Include incompressible random experts as a mandatory no-free-lunch
case.

## Engine decision and kill criterion

Report the exact on-disk layout and read plan a CUDA implementation would need,
or prove why ordinary miss-weighted striping is optimal under the measured
structure. Kill coded storage if its gain depends on unmeasured cross-expert
relations, if parity recovery reads more bytes than a direct record in the
nonfailure regime, or if reconstruction changes the model-visible accumulation
order.
