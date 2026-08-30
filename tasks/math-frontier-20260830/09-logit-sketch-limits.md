# Problem 9: optimal sketches for full-vocabulary trajectory risk

Repository: https://github.com/novysvet/insignia
Hardware required: none

## Existing compression

The vocabulary has 154,880 logits. Moving several full vectors into every
controller event is impossible. The current representation uses three
64-dimensional signed CountSketches after centering and standardizing:

```text
S(prior target), S(current DFlash), S(current DFlash - prior target),
```

plus Top-32 IDs/values and scalar entropy, probability, margin, JS, cosine,
overlap, and temporal-change features. The total sketch payload is 192 floats.

## Main problem

Characterize what can and cannot be inferred about risk-relevant relations
between two or three logit vectors from this representation. Design a better
fixed-budget sketch if one exists.

Important target functionals include:

- centered/raw cosine;
- softmax KL and Jensen-Shannon divergence;
- Top-1 and Top-k disagreement;
- entropy and margin collapse;
- mass entering/leaving a candidate set;
- emergence of a dangerous low-ranked token.

## Required results

1. Prove distortion bounds for the current CountSketch under explicit norm,
   tail, or margin assumptions.
2. Give adversarial pairs with identical or nearly identical 192-float sketches
   and Top-32 summaries but radically different softmax risk. Establish a lower
   bound on any sketch that must detect such pairs.
3. Design a fixed-size alternative, possibly combining random projections,
   polynomial/exponential moments, heavy hitters, quantiles, or learned hashes.
   Its computation must be causal and cheaper than transferring full logits.
4. Treat softmax's exponential nonlinearity directly; a Johnson-Lindenstrauss
   result for Euclidean distance alone is insufficient.
5. Determine how previous-round logits should be incorporated without doubling
   memory. The current encoder has no unused dimensions after dataset v3.

## Concrete game

An encoder observes `x in R^154880` and emits at most 768 bytes plus 32 token
IDs and values. An adversary chooses `y` after seeing the encoding scheme but
not its private random seed. The decoder must estimate `JS(softmax(x),
softmax(y))` and whether argmax changes. Derive upper/lower bounds on required
space for additive error `epsilon` under several margin/tail classes.

## Deliverables

- Information-theoretic bounds and explicit collision examples.
- Proposed sketch with a proof for at least one useful logit class.
- CPU implementation benchmarked on synthetic heavy-tail, flat, multimodal,
  and adversarial logits.
- Recommendation: retain CountSketch, modify it, or replace it, with the exact
  additional bytes and operations stated.
