# Problem 15: A red-blue pebble lower bound for exact MoE prefill

Repository: https://github.com/novysvet/insignia  
Branch: `codex/glm53-dflash2-4070-super`  
Hardware required: none; proofs and CPU schedule simulation only.

## Mission

Prove the minimum NVMe, host-to-device, and temporary-memory traffic required
for exact layer-major prefill when compute is effectively free. Then construct
a schedule attaining the bound, or characterize its approximation gap. This
must decide when the new whole-layer expert sweep is I/O-optimal and how it
should tile when its down-output sidecar no longer fits.

## Fixed graph

For each of 42 sparse layers and prompt length `P`, causal attention/KDA is
processed in ascending chunks of at most 128 rows. Routing is exact Top-8 over
288 experts. An expert record contains gate/up/down NVFP4 tensors totaling
about 13.5 MiB. A token-expert edge consumes one normalized 4096-vector and
produces one 4096-vector down result. Routed results must be replayed in the
legacy per-64-row first-seen-expert order; shared expert and mHC order are
fixed. The current candidate stores normalized inputs plus one down vector per
token/pick (`P * 8 * 4096 * 4` bytes), uploads each layer-union expert once,
then replays exact sums. At `P=938`, its principal device temporaries are about
132 MiB; at `P=8192`, the sidecar alone is about 1 GiB.

There are three memory colors: disk, pinned host, and VRAM. Capacities and
bandwidths are symbolic. Recomputation is allowed and charged separately;
floating-point operation order may not change.

## Required result

1. Define an appropriate weighted red-blue/three-level pebble game for this
   routed DAG, including unknown routes until normalized inputs exist.
2. Prove lower bounds on record reads, H2D bytes, D2H/H2D residual spill, and
   sidecar space as functions of prompt-route incidence, capacities, and
   permitted recomputation.
3. Determine necessary and sufficient conditions for “one upload per distinct
   layer expert” to be attainable. Treat the exact accumulation-order sidecar
   as a dependency, not an implementation detail.
4. When memory is insufficient, derive the optimal or approximately optimal
   tiling over tokens, experts, and output coordinates. Compare expert-major,
   token-major, chunk-major, and hybrid schedules.
5. Include lazy permanent VRAM-arena allocation: transient prefill buffers may
   not silently reduce the later decode expert tier. Account for allocation
   order or a headroom constraint.

## CPU deliverable and completion

Submit `moe_pebble.py` that accepts a token-expert incidence matrix, memory
budgets, and costs; finds exact optimal schedules for tiny cases; evaluates
constructive schedules for realistic symbolic sizes; and emits a verifiable
event trace with live-memory and traffic ledgers. Include worst-case, uniform,
sticky, clustered, and adversarial routes.

Completion requires machine-checkable lower bounds on tiny cases, a theorem
stating the candidate schedule's optimality conditions, and an engine rule for
prompt-length preflight/fallback. Any claimed speedup must be derived from
bytes and supplied bandwidth, never an invented benchmark.
