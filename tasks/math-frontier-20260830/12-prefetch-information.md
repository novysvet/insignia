# Problem 12: value of early information for exact expert prefetch

Repository: https://github.com/novysvet/insignia
Hardware required: none

## Setting

The exact router decision is known only after the current layer's attention and
router projection. Starting a missing 13.5 MiB expert read then is often too
late. Earlier signals include previous-layer routes, previous decode tokens,
draft states, pre-attention hidden state, partial attention statistics, and
target/DFlash logits. Prefetching a wrong expert can delay a correct read on the
single NVMe device.

## Decision-theoretic problem

At time `tau` before the exact route, observation `X_tau` induces posterior
probabilities that experts will be requested. A prefetch set `P_tau` consumes
bandwidth now. Later, exact routing preserves model correctness; unused reads
are merely wasted work.

Define the value of information at each possible observation time and derive an
optimal confidence-aware prefetch policy with deadlines and queue interaction.

## Required results

1. For a restricted queue model, prove the optimal admission threshold in
   terms of posterior request probability, remaining transfer time, miss stall,
   cancellation cost, and displacement of other reads.
2. Show why independent expert thresholds can fail for Top-8 set requests and
   shared layer-group unions. Give a minimal counterexample.
3. Quantify the benefit of earlier but less accurate information versus later
   accurate information. A useful expression should involve conditional mutual
   information or Bayes risk, but must translate into milliseconds/bytes.
4. Give a robust policy under posterior miscalibration with a no-worse-than-
   demand-read guarantee or bounded competitive loss.
5. Determine the optimal allocation of a fixed speculative bandwidth budget
   across layers with different slack and predictability.
6. Analyze whether an early low-rank surrogate should predict exact expert IDs,
   a candidate superset, next-use time, or the marginal stall reduction.

## Synthetic process

Generate route sets with independently controllable temporal correlation,
cross-layer correlation, marginal entropy, and conditional entropy. Simulate
four NVMe readers, one device-upload channel, finite cache space, and
cancelable/noncancelable reads.

## Deliverables

- Theorem and counterexamples.
- CPU event simulator and calibrated policy comparisons.
- A feature-value ranking based on incremental Bayes risk reduction.
- An engine-facing rule that never changes the target router's selected experts
  and therefore preserves output correctness.
