# Problem 11 heterogeneous scheduler: deferred implementation note

Date: 2026-08-30

## Decision

Retain the package as a scheduling hypothesis, trace vocabulary, and correctness
checklist.  Do not treat its simulator output as an expected Insignia speedup and
do not wire its policy into the local Ryzen path.

The immediately useful part is the demand-first/cancellation discipline.  The
P-core controller gang, row-local CUDA-event pipeline, and host-only speculative
prefetch experiments belong on `glm-box`, after native-controller parity and
shadow instrumentation pass.

## Provenance and integrity

- Source: `C:\Users\Pufos\Downloads\problem11_scheduler.zip`
- Archive SHA-256:
  `4143a0e363ae63e955d925608253659f74ff108a72ec3f82cf5434f8073b8e14`
- 30 ZIP entries, 585,027 bytes uncompressed; CRC clean.
- No absolute/traversal paths, duplicate names, symlinks, encryption, or
  suspicious compression ratios were found.
- All 27 payload files present in the archive match `SHA256SUMS.txt`.
- The manifest names 11 files that were not shipped, including
  `REPORT_TEMPLATE.md` and build/run/validation logs.  The report therefore
  cannot be rebuilt exactly from the bundle.
- Review extraction lives only under
  `scratch/problem11_scheduler_triage_4143a0e3/`; no delivered patch was applied.

## What is measured and what is synthetic

The only hardware-backed timing used by the package is the existing Raptor Lake
Falsifier-VNNI result:

| quantity | measured value |
|---|---:|
| complete four-row controller | 3.1849 ms |
| synchronized group | 0.07583 ms |
| observed complete range | 3.1562--3.3057 ms |
| matrix-only work | 1.2106 ms |

The rest is a synthetic simulator.  Its assumptions include 15% device hits,
20% host hits, 0.69/0.46 one-/two-step prediction accuracy, four independent
NVMe servers, 4 MiB expert records, twelve host slots, 0.43 ms NVMe service, and
0.175 ms H2D service.  None of those values is a live measurement of the current
engine.

For reproducibility, the package reports this 60-round synthetic comparison:

| scheduler | synthetic mean round time |
|---|---:|
| proposed | 44.8982 ms |
| critical-path heuristic | 49.2049 ms |
| EDF | 49.2156 ms |
| FIFO/list | 51.9748 ms |

The proposed arm also generated 524.8 MiB of wrong-prefetch traffic per round
and only a 24.52% speculative-prefetch hit rate.  These numbers are recorded as
package evidence, not as performance claims.

Its synthetic 10% sensitivity sweep ranks NVMe first (`-4.7640%` mean time),
then H2D (`-2.9988%`), router (`-0.970%`), expert compute (`-0.377%`), D2H
features (`-0.208%`), state (`-0.148%`), and controller (`-0.0597%`).  This is
directionally consistent with prioritizing miss-weighted C:/E: striping now,
but the percentages are not transferable.

## Model mismatches that block adoption

1. The simulator schedules one expert record per layer-row.  GLM selects top-8,
   and DFlash verification needs a cross-row expert union with multiplicity and
   deduplication.
2. A 4 MiB record conflicts with the measured approximately 4.4 GiB / 336-touch
   decode footprint, roughly 13 MiB per touched record.
3. Independent cache-hit draws do not reproduce the measured approximately 80%
   hit rate of the 32 GiB glm-box host tier or its reuse correlations.
4. Four independent NVMe servers do not represent glm-box's one physical drive.
   They also do not represent the local box's asymmetric two-drive overlay.
5. D2H and H2D are modeled independently even though they share PCIe.  CUDA
   stream priority is not preemption and does not guarantee overlap.
6. The policy checkpoint is a smoke artifact.  Native/PyTorch controller parity,
   causal-state parity, and production integration are still prerequisites.
7. Sixty rounds cannot support the reported p99 or 2.5%-tail conclusions.
8. The two-group MILP is not discriminating: every heuristic reaches the same
   67-tick optimum.  The described frontier dynamic program is not implemented.
9. Duplicate one-/two-step predictions are not deduplicated before I/O, and some
   completed duplicate reads can escape waste accounting.

## Useful now on the Ryzen / dual-SSD box

- Keep the scheduler's causal timestamps and invariants as names for future
  telemetry: release, deadline, exact-read ready, H2D ready, expert start/end,
  barrier, cancellation, and commit.
- Unit-test strict demand priority, queued-read promotion, cancellation residual,
  buffer ownership, and duplicate-prefetch suppression with synthetic traces.
- Continue the route-miss-weighted C:/E: stripe overlay.  This attacks the only
  resource the synthetic sensitivity study also ranks first, and it already has
  a fail-closed implementation and deterministic plan.
- If a local reader scheduler is tested later, reserve at least one demand
  opportunity per physical drive.  Do not infer the reservation count from the
  single-drive simulator.

The Ryzen 5600X has no P/E topology and no Raptor Lake AVX-VNNI controller
contract.  Local measurements can validate ordering and accounting, but they
must not choose glm-box affinity or expected speedups.

## First glm-box experiment

1. Pass native/PyTorch Falsifier fixture, state, and output parity.
2. Add shadow-only timestamps around router, D2H, controller, exact read, H2D,
   expert compute, barriers, and commit.  Do not let the controller schedule yet.
3. On one fixed k4 prompt, compare seven repeated runs of baseline versus a
   four-physical-P-core controller gang, then global barriers versus row-local
   CUDA events/futures.
4. Only if controller time is hidden, test one-layer host-only speculation with
   one exact-demand opportunity reserved.  Keep two-layer prediction and
   speculative H2D disabled in this first gate.
5. Require digit-identical logits and greedy IDs, no exact-demand latency
   regression, repeated-median wall-time improvement, and measured bytes, hit
   rate, cancellation residual, APERF/MPERF, and CUDA/NVMe overlap.

Only a passing shadow trace promotes this from a design note to an implementation
candidate.
