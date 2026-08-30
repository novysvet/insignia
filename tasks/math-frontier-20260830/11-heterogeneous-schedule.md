# Problem 11: optimal overlap of CPU control, GPU compute, NVMe, and H2D

Repository: https://github.com/novysvet/insignia
Hardware required: none

## Resources and tasks

One four-row DFlash verify round contains 42 sparse layer groups. Each group has
a precedence DAG containing:

```text
GPU attention / router GEMV,
small GPU-to-host feature transfers,
CPU AVX-VNNI controller work,
NVMe reads on four reader queues,
host-to-device uploads,
GPU expert compute,
recurrent-state commit or rollback.
```

The complete controller ceiling is 3.1849 ms per round, about 0.0758 ms per
layer group when four P-core workers synchronize. The i7-14700KF has 8 P cores,
12 E cores, and 28 logical threads but no AVX-512/AMX. The GPU and NVMe operate
asynchronously, subject to dependencies and finite buffers.

## Main problem

Find a schedule minimizing round makespan while preserving exact causal action
dependencies. Decide which work belongs on P cores, E cores, GPU streams,
reader queues, and copy engines; how far to precompute/prefetch; and when a
barrier is avoidable.

## Required results

1. Formulate the scheduling problem with renewable resources, communication
   delays, deadlines, and optional/speculative tasks. Classify its complexity.
2. Derive a lower bound stronger than `max(total work/resource capacity)` by
   using the critical path, queue coupling, and causal release times.
3. Give an exact or fixed-parameter algorithm for a reduced `42 x 4` instance,
   or a heuristic with a proved approximation under restricted assumptions.
4. Model wrong prefetches as optional tasks that consume bandwidth and buffer
   capacity but may be cancelled.
5. Determine when dedicating four P cores to the controller is superior to
   using more rows/threads, SMT siblings, or E cores, including barrier cost and
   AVX frequency effects as symbolic parameters.
6. Prove conditions under which the entire 3.1849 ms controller cost can be
   hidden, and conditions under which at least some fixed amount is necessarily
   serial.

## Synthetic scheduler

Create task durations as distributions, not constants. Include WSL timing
variance, NVMe long tails, cache hits/misses, 0--4 accepted tokens, and exact
retry. Compare list scheduling, earliest-deadline, critical-path priority,
integer programming on small cases, and the proposed method.

## Deliverables

- Complexity result, lower bound, and scheduling algorithm.
- Gantt-style output from a deterministic CPU simulator.
- Sensitivity analysis showing the value of reducing each task class by 10%.
- A concrete engine schedule and the instrumentation needed to validate every
  assumed overlap on glm-box later.
