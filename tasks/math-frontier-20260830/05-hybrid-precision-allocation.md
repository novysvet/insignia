# Problem 5: robust hybrid-precision placement across NVMe, RAM, and VRAM

Repository: https://github.com/novysvet/insignia
Hardware required: none

## Setting

Future work may have at least three weight representations:

- an approximately 120 GB `UD-IQ3_XXS` checkpoint;
- the current approximately 180 GiB NVFP4 compact store;
- exact or near-exact FP8/BF16 compute caches.

Do not assume undocumented UD-IQ3 layout details. Represent format `p` by
measured parameters `(bytes, decode cost, transfer cost, error law)` supplied
later. Experts have highly nonuniform conditional activation probabilities and
different quality sensitivity. A miss may load a cheap low-precision version
immediately, wait for a better version, or combine a surrogate with a residual.

## Optimization problem

For each tensor/expert `e` and tier `z in {NVMe,RAM,VRAM}`, choose zero or more
representations `p`. Let

```text
s[e,p,z]  occupied bytes,
t[e,p,z]  expected service time,
d[e,p,x]  output-damage random variable in causal state x,
pi[e|x]   conditional request probability.
```

Capacities couple choices across all experts. Runtime may select a
representation after observing state `x`, but placement happens earlier.
Minimize expected accepted-token latency subject to a trajectory-level quality
constraint, or characterize the Pareto frontier.

## Required results

1. Establish complexity for the static and adaptive problems. Identify when
   the static problem reduces to multiple-choice knapsack, generalized
   assignment, or a submodular placement problem.
2. Derive an exact algorithm for a meaningful special case and an FPTAS or
   approximation for a larger one.
3. Give a robust solution when `pi` lies in an uncertainty set estimated from
   prompt families. It should not spend all high precision on yesterday's hot
   experts.
4. Allow correlated expert requests: Top-8 sets and multirow union costs make
   marginal probabilities insufficient. Quantify the value of pairwise or
   higher-order statistics.
5. Model low-precision-on-miss as an option with immediate latency but random
   recurrent damage. Determine when waiting for the exact representation is
   optimal.
6. Derive a water-filling law, Lagrange multiplier interpretation, or a
   counterexample showing why no scalar "sensitivity per byte" score can be
   optimal.

## Synthetic instance

Use 42 layers x 288 experts, Top-8 requests, one 32 GiB RAM tier, one roughly
576 MiB VRAM expert tier, and three parameterized formats. Generate conditional
route clusters and rare high-sensitivity experts. Sweep capacity, precision
error tails, route shift, and transfer bandwidth.

## Deliverables

- Formal stochastic placement/control problem.
- Proofs and adversarial instances.
- CPU solver producing a complete placement manifest and Pareto curve.
- A measurement schema for learning `t`, `d`, and conditional request-set
  statistics without requiring accuracy labels from easy benchmarks.
