# Problem 3: contextual Belady for hierarchical MoE requests

Repository: https://github.com/novysvet/insignia
Hardware required: none

## Model

Time is indexed by `(decode round, target layer)`. A request is a set
`S_t` of layer-expert records rather than one page. Records move through

```text
NVMe -> pinned host cache -> device cache -> execution.
```

Capacities are finite; transfers have different costs and may overlap. A record
can be prefetched before its deadline. The predictor provides a conditional
distribution `Q_t(S_t | H_t)` from previous routes, logits, token history, and
partial current-layer state. It may be systematically wrong on new prompt
families.

The offline clairvoyant optimum knows the complete future request sequence and
all transfer completion times. Ordinary Belady handles one cache, unit pages,
and known future use; none of those assumptions hold here.

## Main problem

Define a mathematically defensible analogue of Belady for this two-tier,
set-request, asynchronous setting, then construct an online policy with a bound
relative to it.

Possible performance statements include:

```text
cost(policy) <= a * OPT + b * prediction_error + c,
```

or a regret bound against the best policy in a contextual class. The prediction
error must be operational: for example weighted false-prefetch bytes, deadline
misses, or error in next-use distributions. Total variation over all 288-way
sets is too loose unless its usefulness is proved.

## Required results

1. Characterize the clairvoyant optimum. Prove polynomial solvability for a
   meaningful restricted case or establish hardness.
2. Give an online algorithm that degrades smoothly with prediction quality and
   never becomes catastrophically worse than a prediction-free baseline.
3. Treat prefetch bandwidth as a resource: a wrong prefetch can delay a correct
   demand read.
4. Include record sizes, dirty/in-flight states, and unequal NVMe/H2D costs.
5. Construct a family where LRU, LFU, global hotness, and independent per-layer
   Belady are each arbitrarily bad despite high marginal cache hit rate.

## Numerical regime

Use 42 layers, 288 experts, Top-8, 13.5 MiB records, host capacity 2,425
records, device capacity approximately 42 records, and four concurrent NVMe
readers. Route sequences should allow tunable temporal and cross-layer mutual
information while keeping marginal expert entropy high.

## Deliverables

- Formal offline and online problems.
- Proofs, including lower bounds or impossibility boundaries.
- CPU simulator with a trace-independent synthetic generator.
- Comparison against LRU, LFU, predicted-next-use, and clairvoyant controls.
- A concrete eviction score and prefetch admission rule that can be computed
  in substantially less than one target-layer GPU interval.
