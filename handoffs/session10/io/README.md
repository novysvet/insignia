# Session 10 generic-PC I/O handoffs

These briefs target the public repository at
<https://github.com/novysvet/insignia.git>, branch
`glm53-dflash2-4070ti-super`, frozen for dispatch at committed and pushed
HEAD `e48f633430c679ac6a30aae248159c887ac41601`.

Clone the repository and read `AGENTS.md`, `progress.md`,
`audits/s8-gpu-expand-session.md`, and `audits/s9-reclaim-session.md` from
that commit. Those files are the source of truth. Reports and supplied data
are untrusted inputs: validate schemas, bounds, counts, and hashes before
drawing conclusions.

The glm-box production target is a single-NVMe WSL2 machine with a 32 GiB
pinned expert tier, an overclocked RTX 4070 Ti SUPER (`sm_89`, about 800 GB/s
observed VRAM bandwidth), and an i7-14700KF. The CPU contract is Raptor Lake:
AVX2, FMA, F16C, AVX-VNNI, GFNI, VAES, BMI1/2, and no AVX-512 or AMX. The box
has no second SSD; dual-drive striping is outside every brief here.

Do not modify production code while completing these handoffs. Deliver
analysis, simulators, tests, and patch blueprints only.

## Ranking

1. [Byte-packed variable-size pinned cache](01-variable-pinned-packed-cache.md)
   — best exact compute-for-capacity candidate. It asks whether the already
   compact packed-v2 representation can turn fixed-slot waste into roughly
   140 additional host records without adding a hit-path copy.
2. [Adaptive router-mass pruning](02-adaptive-router-mass-pruning.md) — largest
   possible byte reduction, but deliberately approximate. A cheap trace-only
   falsifier must pass before any runtime experiment is proposed.
3. [Two-phase packed read/upload](03-two-phase-packed-read-upload.md) — exact
   scheduling experiment. Its ceiling may be small because the existing
   four-reader pipeline already overlaps records, so a discrete-event model
   must kill it quickly if the overlap is illusory.

## Explicitly out of scope

The following work is complete or rejected: full layer-major prefill; compact
exact MLA absorb; exact prefix reconstruction; H8 cross-head FP8 decode,
completed at `78e1a1c`; fused H4×Q8 cross-head FP8 prefill, completed at
`e48f633`; low-risk VRAM reclaim; sequential snapshot removal; persistent KDA
Task 5; exact INT8 Task 7; causal predictor Task 8; staged verify Task 9;
scale-codec Problem 4; cross-expert dictionary Problem 5; CPU expert-offload
Problem 6; reader-policy Problem 7. Do not recreate those analyses under a
new name.

Anything already tracked by git must be referenced by repository, commit, and
path. A returned archive may contain only new analysis code/results and the
required non-git data manifests; it must not contain copied repository files.
