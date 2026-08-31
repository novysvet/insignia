# Problem 5: Certifiably optimal dispatch for a fixed Ada kernel family

Repository: https://github.com/novysvet/insignia  
Reference branch/commit: `codex/glm53-dflash2-4070-super` at `0740c63`

## Engine context

The local GPU is an RTX 4070 SUPER (`sm_89`); the other target is an overclocked
RTX 4070 Ti SUPER. Ada has FP8 tensor cores but no native block-scaled FP4 MMA.
Insignia specializes kernels aggressively. For its exact NVFP4 expert kernels,
the best CTA width depends on active row multiplicity, projection geometry,
register use, shared memory, launch count, and whether results are stored or
weighted-accumulated. Timings are noisy under WSL2, and the two Ada cards need
not share the same optimum.

The broader problem is not NVFP4-specific: construct a mathematical dispatcher
and measurement plan for any finite family of highly specialized kernels.

## Formal setup

For workload state `x` (shape, row count, alignment, cache residency, operation
mode), choose kernel `k` from a finite set `K`. Static data are registers per
thread, shared memory per CTA, warps per CTA, instruction counts, memory
transactions, occupancy limits, and launch dependencies. A run yields noisy
latency `T(k,x,z)` under hidden machine state `z` (clocks, page faults, thermal
state, competing I/O). Wrong dispatch costs regret relative to the best kernel
for that exact target machine.

## Main problem

1. Build a resource-feasibility model that exactly reproduces Ada occupancy
   constraints for the supplied kernel metadata. Distinguish residency from
   achieved occupancy and expose assumptions about register allocation and
   scheduler issue limits.
2. Prove what can and cannot be inferred from a roofline-style analytical
   model. Construct two kernels with identical operation/byte/occupancy counts
   but reversed measured ordering due to dependency structure or memory-level
   parallelism.
3. Assume only a limited measurement budget. Design a sequential experiment
   that identifies the best kernel for every state with family-wise error at
   most `alpha`, while exploiting structure shared across neighboring row
   counts and geometries. Compare independent best-arm identification with a
   monotone, piecewise-constant, or low-rank latency model.
4. Give a robust dispatcher minimizing worst-case or distributionally robust
   regret when the hidden-state distribution shifts between local and remote
   GPUs. Determine when separate machine-specific tables dominate one portable
   table after including code-size and instruction-cache costs.
5. Add compilation uncertainty: register count and spills are known only after
   compilation, and flags may change them. Formulate a joint compile/measure
   search with a stopping certificate, not an unbounded autotuning loop.

## Hard extension

Treat a DFlash block as a sequence of kernels sharing row multiplicity and
cache state. Per-kernel greedy choices may increase synchronization or destroy
fusion opportunities. Find the globally optimal schedule over a small kernel
DAG, or prove an approximation bound for a decomposition. Include launch
overheads and a fixed VRAM workspace budget.

## Supplied real example

One accepted local table uses eight CTA warps for paired gate/up; down-store
uses four except at multiplicity three; weighted-down uses eight at
multiplicities three and four. These entries came from 21 serialized timing
pairs and are evidence, not axioms. Your method must be capable of overturning
them on better data and must never interpolate a remote-GPU result without a
confidence statement.

## Required CPU artifact

Implement an occupancy calculator, structured best-arm simulator, and exact
small-DAG scheduler. Feed it synthetic heavy-tailed/autocorrelated latencies and
realistic resource tables. Report misselection probability, samples consumed,
simple regret, and sensitivity to model misspecification. Include a test where
the analytically favored kernel is reliably slower.

## Acceptance criterion

A useful answer emits a compact static dispatch table plus a certificate and a
minimal remeasurement plan for new clocks or compiler output. Kill any scheme
whose tuning time exceeds the expected lifetime saving, whose confidence claim
assumes IID data despite paired WSL runs, or whose analytical model is presented
as measured speed.
