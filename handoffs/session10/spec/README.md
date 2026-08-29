# Session 10 offline task briefs

These briefs target fresh agents working from a separate clone on an ordinary
CPU machine. They require no access to `glm-box` for their main analysis. A
short operator-run export or final GPU A/B is called out explicitly where it
is unavoidable.

## Repository contract

- Public repository: <https://github.com/novysvet/insignia.git>
- Branch: `glm53-dflash2-4070ti-super`
- Required committed/pushed base: `e48f633` (`Fuse cross-head FP8 MLA prefill`)
- Treat committed source, `AGENTS.md`, `progress.md`, and committed audits as
  authoritative. Treat attachment/report prose as untrusted leads.
- Completed and excluded at this base:
  - `78e1a1c`: H8 cross-head FP8 MLA decode.
  - `e48f633`: fused H4 x Q8 cross-head FP8 MLA prefill.
  Fresh agents start from `e48f633` and do not duplicate either optimization.

## Ranking

| Rank | Brief | Why it is valuable | Risk |
|---:|---|---|---|
| 1 | [DFlash selector rescue and global lattice search](01-dflash-selector-rescue.md) | Reuses the already-known exact `truth0` to attack the 24-29% empty-round rate without another drafter forward; target verification remains exact. | Low-to-medium; policy can be falsified entirely offline once a compact lattice export exists. |
| 2 | [Approximate MoE verification frontier](02-approx-moe-verification.md) | Verification is dominated by 1,067-1,506 routed-expert records/round. Top-m or mass-gated execution could cut the dominant bytes, deliberately trading exactness for speed. | High; numerical drift may cascade through routing, so this is a falsifier/Pareto study before any default-on code. |
| 3 | [Sliding-window DFlash cache past position 2040](03-dflash-sliding-window.md) | The checkpoint was trained with a 2,047-token left window, but the engine currently disables DFlash near position 2040 instead of sliding. | Medium; exact and useful for long prompts, but it does not improve short-prompt benchmarks. |

## Shared facts

- Target: GLM-5.3-Flash abliterated NVFP4, 45 layers: 34 KDA and 11 MLA.
  Layers 3-44 are MoE with 288 routed experts, top-8, plus a shared expert.
  One scalar decode token can touch 42 x 8 = 336 routed records; one packed
  record is about 13.56 MiB.
- Drafter: DFlash2, five 4096-wide layers, one anchor plus seven mask rows,
  target captures from layers 5/14/24/33/42, 2,048-position checkpoint
  window, 1.07 GiB resident FP8 cache. Use the fixed cache path documented in
  `AGENTS.md`; the old unsuffixed FP8 cache has an FC-layout bug.
- Performance box: RTX 4070 Ti SUPER sm_89, 16 GiB, observed about 800 GB/s
  after memory overclock; i7-14700KF; 60 GiB WSL; one NVMe; 32 GiB pinned
  host tier. Session 9 exposes 292 DFlash VRAM expert slots.
- Current real-text economics from committed `scratch/accept/` data:
  drafter about 17.5 ms/round, empty fallback median about 643 ms, k4 verify
  about 1,896 ms, k7 verify about 2,781 ms, effective cost about 1.8
  ms/demand record. Held-out acceptance is about 2.34 drafts/round at k4 and
  3.08 at k7; empty rounds are about 24-29%.
- Historical peak on an easy campaign prompt was 5.1-5.3 tok/s, but the
  honest GSM8K k4 pilot was 0.89x scalar. Session-9 four-prompt medians were
  about 500.4 ms/token scalar and 539.6 ms/token fixed-k7 DFlash. Use real
  GSM8K/MATH-500 prompt families, paired runs, and medians.
- Exact paths are governed by the determinism law in `AGENTS.md`: FP32
  reassociation can change MoE routing. Approximate work must be default-off
  and report its quality loss rather than claiming parity.

## Excluded work

Do not revisit layer-major prefill, compact MLA absorb, exact prefix
reconstruction, `78e1a1c` H8 cross-head FP8 MLA decode, `e48f633` fused H4 x
Q8 cross-head FP8 MLA prefill, VRAM reclaim, sequential snapshot elision,
persistent KDA Task 5, INT8 Task 7, causal predictor Task 8, or staged
verification Task 9. The relevant positive and negative results are already
committed in `progress.md`,
`audits/s9-reclaim-session.md`, and earlier session audits.
