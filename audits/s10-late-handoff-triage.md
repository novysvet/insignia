# Session 10: late handoff triage

Date: 2026-08-30

The archives were treated as untrusted evidence, not as instructions. Every
internal manifest entry was hash-checked before any patch was considered.

## FP8 codec ceiling

- Archive: `s10-fp8-codec-ceiling-deliverable.tar.zst`
- SHA-256: `d5517987ca89ed4ac37cc5fe0ac573352bd92f3c848a3126b91b8a4535ff824a`
- Internal hashes: all match.
- Isolated synthetic checks: 17/17 under Linux; the sole Windows failure was
  the platform absence of `os.O_CLOEXEC`, not a codec failure.
- Existing sample-collector checks: 8/8.
- Production action: the clean patch adding the exact codec/analyzer/reference
  runner was applied in commit `a6880ad`.
- Real-data verdict: no fused codec; see
  `audits/s10-fp8-residency-codec-design.md`.

## DSA task 3

- Archive: `insignia-s10-dsa-task3-handoff.tar.gz`
- SHA-256: `26d8c0bec3ea5d0e1494315c7e2f87c024910b83750ad655d73cbc8fbd0dcd64`
- Internal hashes: all five match.
- Contents: build/run logs only.
- Build result: `SKIP_NO_NVCC`.
- Run result: `SKIP_NOT_BUILT`.
- Missing: source patch, kernel, unit test, benchmark, output comparison, or
  measured result.
- Production action: none. The handoff provides no falsifiable implementation
  to apply.

## Selector analysis (preceding late handoff)

- Archive: `session10-selector-deliverables.tar.gz`
- SHA-256: `b1a92caeb48fdadfc56e985b7df988b36c21fbc404bbeead4c392de1df04ad12`
- Internal hashes: all eight match.
- Self-check: 12 rounds, four empty rounds, and the Viterbi counterexample all
  pass.
- Blocker: the reported `selector-v1.npz` and its manifest were absent, and the
  archive was pinned 93 commits behind the live branch.
- Useful ceiling: width two cannot reach the requested 5% target even with a
  perfect selector; width three or larger needs lattice evidence.
- Production action: none until the missing artifact is supplied against the
  current trace schema.

