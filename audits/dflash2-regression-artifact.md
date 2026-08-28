# DFlash2 acceptance "regression" resolved: prompt artifact, not a code defect (2026-08-28, session 5)

Session 4 closed with an open alarm: on the exact-prefix MLA shadow bridge
(`fc046a3`) DFlash2 acceptance had "collapsed" to 1.43 tokens/round (516.7
ms/tok) vs the pre-bridge 5.0/round, 187.7–239.7 ms/tok. Session 5 took the
box apart over this. Verdict: **there is no regression.** The 1.43 number is
a property of the 5-token oracle prompt, not of the bridge.

## Evidence (all runs on glm-box, 32 GiB expert tier, Q8 10 GiB pin, fp8-fixed drafter cache)

1. Reproduced the session-4 reading exactly on HEAD (`c295638`, bridge):
   oracle prompt `154820,13,171,1496,2343`, k4, 30 gen → **1.43 accept/round,
   15/21 empty, histogram 0:15 1:3 4:3**, 599.3 ms/tok.
2. `INSIGNIA_GLM53_MLA_LEGACY=1` on HEAD, same prompt: **bit-identical**
   histogram, IDs, 1.43/round (563.9 ms/tok). Legacy mode changes nothing at
   positions < 256 — the bridge already runs the same exact kernels there; it
   only removes the latent shadow store. (This was also the conjectured
   experiment left running by the other session; its result is a null.)
3. **Pre-bridge binary** (`/var/tmp/insignia-build-raptor/glm53-generate`,
   built 15:49, predates `fc31d7a`; lacks the "legacy MLA", "exact
   FlashAttention-2", and "shadow store" strings) on the oracle prompt:
   **identical 1.43/round, 15 empty, 0:15 1:3 4:3**, 511.4 ms/tok. A binary
   from before the entire latent-MLA war reproduces the "regression"
   exactly. Code cause: eliminated.
4. Control — HEAD on the 16-token **campaign prompt** (`build/bench-df.sh`:
   `154820,11,301,2745,941,1516,87,29871,526,1052,374,123,77,918,1520,25`),
   100 gen:
   - k4: **3.70 accept/round**, histogram 0:1 1:1 2:1 4:24 (24/27 rounds
     full), **228.7 ms/tok**.
   - k7: **5.88 accept/round**, histogram 0:1 1:1 2:1 5:1 7:13 (13/17 full),
     **227.8 ms/tok**.
   - Matches the pre-bridge campaign numbers (k4 239.7; k7 187.7–194.4 at
     235 gen) within WSL run-to-run swing and gen-length amortization.

## Mechanism

The 5-token oracle prompt greedy-decodes into a `200 200 200 ...` parrot
loop. The DFlash2 block drafter fails on it from the anchor: 15/21 rounds die
at the `d1 != truth0` short-circuit, and the fallback `step()` costs
~450–670 ms each. On realistic structured text the drafter anchors fine and
acceptance saturates the block (k4: 4/4 in ~89% of verified rounds; k7: 7/7
in ~76%). Acceptance is prompt-distribution-dependent by orders of magnitude;
**the 5-token oracle prompt must never be used to judge DFlash2** — it is a
parity-gate prompt only. The 16-token campaign prompt is the acceptance
reference.

Greedy IDs were identical in every A/B pair (bridge == legacy == pre-bridge
binary): parity intact throughout.

## Operational notes from this session

- The campaign prompt + k7/100-gen is the cheapest acceptance smoke test
  (~1 engine minute).
- glm-box cannot push to GitHub (no creds); to land glm-box-made commits:
  `git bundle create X.bundle <base>..branch` → scp → local
  `git fetch X.bundle ...` → local push. Done for `bf577e6` + `c295638`
  (logits-dump comparator, PPL scorer, parameterized DFlash bench) — now on
  origin.
- glm-box WSL VM recycled during the session (~19:48 boot); `/var/tmp`
  persists in the ext4 vhdx, `/tmp` does not. Both engine binaries survived.
- An orphaned benchmark from another session was racing my runs on the
  single GPU; killed with user's blessing. Concurrent engine runs pin
  2×32 GiB host RAM and invalidate each other's timings — always check
  `pgrep -af glm53-generate` before benchmarking.

## Unchanged open work

1. GSM8K/MATH-500 honest campaign (`tools/benchmark_math.py`, staged) — real
   prompts, scalar vs k4/k7, acceptance histograms decide defaults. Cold-start
   single-sample readings (e.g. 2.31/round at 640 ms/tok on one GSM8K prompt)
   are dominated by cache warm-up; wait for medians.
2. Latent MLA validation beyond position 256; `tools/compare_logits.py` +
   `tools/ppl.py` (c295638) are the A/B tooling.
3. CCT cross-layer prefetch integration — the ~6→8.5 GB/s serialization gap
   remains the biggest I/O lever.
4. Prefill is expert-I/O-bound (~700 ms/token cold at 16 tokens): expert-read
   scheduling, not GEMV, is the prefill lever.
