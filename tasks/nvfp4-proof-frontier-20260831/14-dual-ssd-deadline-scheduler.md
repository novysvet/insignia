# Problem 14: Routing-aware dual-SSD placement and deadline scheduling

Repository: https://github.com/novysvet/insignia  
Branch: `codex/glm53-dflash2-4070-super`  
Hardware required: none; solve with an event-driven CPU simulator.

## Mission

Derive an online placement, replication, and read-dispatch policy for the local
machine's two unequal SSDs. The workload is not a bulk RAID stream: each
13.5 MiB expert record has a layer deadline, cache state, reuse probability,
and speculative confidence. Incorrect prefetch can consume the bandwidth that
a later demand read needs.

## Fixed system facts

The model has 42 sparse layers, 288 experts/layer, Top-8. A scalar token asks
for 336 records before cache hits; DFlash verification asks for correlated
unions. The local WSL instance lives on the C: 980 PRO; E: is a separate
DRAM-less SSD carrying the repository/checkpoints. Existing code can map an
alternate shard directory and stripe records, but drive rates, tails, thermal
behavior, and WSL interference vary. The large single-drive box is out of
scope. Four reader threads are the observed WSL sweet spot. Reads are large,
aligned, O_DIRECT requests; host and VRAM caches sit above the drives.

## Required result

1. Model two non-identical, time-varying service processes with queueing,
   correlated tail latency, and shared CPU/WSL overhead. State the minimal
   observations needed online.
2. Jointly choose static placement, optional replication under a byte budget,
   demand dispatch, prefetch admission, cancellation, and reader allocation.
   A record may live on one or both drives.
3. Prove an optimal policy for a tractable case and a competitive/regret or
   stability result for the online case. Include adversarial counterexamples
   to naive round-robin, hash striping, shortest-queue, and duplicate racing.
4. Respect layer deadlines and cache hits. Optimize committed token latency,
   not aggregate bandwidth; quantify head-of-line blocking and wasted reads.
5. Derive a safe controller for regime changes such as E: garbage collection
   or C: contention, including hysteresis and recovery.

## CPU artifact and completion gate

Submit `dual_ssd_scheduler.py` with deterministic discrete-event simulation,
trace/synthetic request generators, tiny-instance dynamic-programming oracle,
and policies above. Sweep rate ratios, tail distributions, correlation,
replication budget, cache size, DFlash union size, and prediction precision.
Report deadline misses, committed ms/token, useful/wasted bytes per drive,
queueing delay, fairness, and regret to clairvoyant scheduling.

Completion requires matching the tiny oracle, no invented hardware numbers,
and a concrete manifest/dispatch format usable by `tools/stripe_copy.py` and
the runtime. The report must state when single-drive placement is optimal and
when two-drive striping is provably beneficial.
