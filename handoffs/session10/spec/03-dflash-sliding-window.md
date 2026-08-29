# Task 3: Exact sliding-window DFlash2 beyond position 2040

## Mission

Replace the engine's hard DFlash cutoff near absolute position 2040 with the
checkpoint's intended 2,047-token sliding-left window. Design and prove a
power-of-two ring cache for the drafter K/V state, including the exact
per-query attention mask for an eight-position bidirectional block. This task
is primarily index algebra, a CPU oracle, and compile/static review; final GPU
validation can be handed to an operator.

Target generation must remain exact because target verification is unchanged.
Drafter logits before the old cutoff must be bit-identical. At later positions
the new output is compared against a contiguous CPU sliding-window oracle.

## Checkout and authority

- Repository: <https://github.com/novysvet/insignia.git>
- Branch and required committed/pushed HEAD: `glm53-dflash2-4070ti-super` at
  `e48f633`.
- Completed exclusions at this base: `78e1a1c` H8 cross-head FP8 MLA decode
  and `e48f633` fused H4 x Q8 cross-head FP8 MLA prefill. Use a fresh clone
  and do not duplicate either.
- Read `AGENTS.md`, `progress.md`, `audits/dflash2-session.md`,
  `audits/dflash2-fixes-session.md`, `audits/quality-cct-session.md`, and
  `audits/s9-reclaim-session.md`. Verify window semantics against the cited
  DFlash/SGLang reference; attachment summaries are not authority.

## Current implementation

The five-layer drafter uses 32 query heads, eight KV heads, head dimension
128, block size eight, and absolute neox RoPE. `kMaxCtx=2048`. Its FP32 K and
V caches are each `[5][2048][8*128]`, about 40 MiB apiece.

Source anchors at pinned base `e48f633`:

- `include/insignia_glm53_dflash2.cuh:33` and the K/V member comments:
  `kMaxCtx`, `forward`, and `commit` contracts.
- `src/glm53_dflash2.cu:141-206`, `df_attn_kernel`: contiguous context plus
  eight block keys, online softmax, shared score storage.
- `src/glm53_dflash2.cu:242-272`, `df_kv_append_kernel`: absolute contiguous
  writes.
- `src/glm53_dflash2.cu:372-376`: cache allocation and sm_89 shared-memory
  opt-in.
- `src/glm53_dflash2.cu:438-515`, `DFlash2Drafter::forward`: `ctx_len`,
  absolute RoPE, and attention launch.
- `src/glm53_dflash2.cu:587-628`, `commit`: rows beyond 2048 are dropped.
- `src/glm53_generate.cu:5457`: main loop breaks when the anchor
  cannot fit a full eight-row block below `kMaxCtx`.
- `tools/dflash2_oracle.py`: contiguous NumPy reference to extend with a
  separate ring/sliding mode.

At the largest current round, score shared memory is approximately
`8 * (2048 + 8) * 4 = 65,792` bytes, below the roughly 99 KiB sm_89 opt-in
limit. A sliding implementation should keep the active key count bounded and
must not grow this allocation with absolute position.

## Questions to settle first

1. Does `window_left=2047` mean at most 2,048 keys including the current key,
   and how is it applied when the eight block rows are bidirectional?
2. Does each block query require a different oldest committed position, or
   does the reference construct one shared context slice for the whole block?
3. Are all eight block keys always visible to all eight queries, independent
   of the left-window cutoff?
4. What absolute positions are passed to q/k RoPE after the cache wraps?

Answer from reference code and encode the result as a small truth table for
anchors 2038-2050. Do not start a CUDA patch while any mask question remains
ambiguous.

## Proposed representation to test

Use absolute positions for RoPE and metadata, but map committed K/V to
`slot = absolute_position & 2047`. Since 2048 is a power of two, the hot-path
mapping needs no division. Iterate logical keys from oldest to newest and map
each logical key to its slot; online-softmax order must match a contiguous
reference. If queries have different lower bounds, mask scores per query
without changing the order of retained keys.

The design must handle:

- prompt commits in batches up to 128;
- a commit spanning slot 2047 -> 0;
- multiple complete wraps;
- absolute positions used by RoPE and dump diagnostics;
- no overwrite of a logical key still visible to any block query;
- context lengths shorter than the window;
- partial final generation rounds.

## Staged experiment plan

1. **Reference mask:** document the exact reference mask and key-count formula
   for every query row. Completion: a table for anchors 0, 2039, 2040, 2047,
   2048, 2055, and 4096.
2. **Index simulator:** implement a dependency-free CPU model that appends
   integer-tagged K/V rows in chunks 1, 7, 8, 127, and 128 and compares the
   ring's logical sequence with a deque/contiguous oracle after every write.
3. **Attention simulator:** use small random vectors and reproduce contiguous
   online-softmax outputs with ring mapping and the derived masks. Compare
   key visitation sequence exactly and numerical output bitwise when the same
   FP32 operations are used.
4. **Boundary replay:** extend a copy of `tools/dflash2_oracle.py` under
   `scratch/session10-df-window/` to run both contiguous and ring modes at
   positions around every wrap. Existing positions below 2040 must not
   change.
5. **Resource review:** calculate cache bytes, dynamic shared memory,
   registers, modulo/mask instruction count, and expected draft overhead.
   Compile for sm_89 and inspect PTX/SASS if CUDA is available; a GPU is not
   required for compilation.
6. Only after all CPU gates pass, prepare a minimal default-off patch and a
   box-operator test matrix using prompts crossing 2040 and 4096. Keep the
   old cutoff as an A/B switch until long-context acceptance is known.

## Deliverables

- `scratch/session10-df-window/WINDOW-SEMANTICS.md` with cited reference
  locations, formulas, and boundary truth tables.
- `scratch/session10-df-window/test_ring.py`, stdlib/NumPy only, with
  deterministic self-tests and randomized chunk sequences.
- `scratch/session10-df-window/RESOURCE-MODEL.md`.
- `scratch/session10-df-window/dflash-ring.patch` only if CPU/reference gates
  pass; no unrelated cleanup.
- An operator runbook for paired old-cutoff/new-window tests at positions
  below 2040, 2040-2064, and beyond 4096.

No non-git artifact is required for the proof. If a live `DF_DUMP` is supplied
for validation, require its documented tag format from
`audits/dflash2-session.md`, a manifest containing prompt ids/command/git
commit/byte count, and SHA-256. Provide a parser command that extracts only
the first round on each side of positions 2040 and 4096; never trust an
unmanifested binary dump.

## Gates and kill criteria

- Ring/deque logical key ids and per-query masks match on every deterministic
  and at least 100,000 randomized append/query operations.
- CPU contiguous and ring attention visit identical logical keys in identical
  order and are bit-identical under identical arithmetic.
- Live drafter logits and selector path below position 2040 are digit/bit
  identical to the shipping path.
- Target greedy ids and top-10 logits remain exact because verification is
  unchanged.
- Draft-round overhead below the old cutoff is <=3%; no cache or shared-memory
  growth with absolute context.
- On real prompts past the cutoff, accepted drafts/round must exceed 1.5 and
  paired median decode must beat scalar fallback by at least 10% before the
  window becomes default-on.

Kill if the checkpoint/reference actually requires a fixed prefix rather than
a sliding window, if the exact per-query mask cannot fit the current online
softmax structure, if dynamic shared memory exceeds the sm_89 limit, or if
long-context acceptance fails the 1.5 gate. A correct but unprofitable ring
may remain as a default-off capability.

## Forbidden duplication

Do not alter target MLA, `78e1a1c` H8 cross-head decode, `e48f633` fused H4 x
Q8 cross-head prefill, exact prefix reconstruction, compact absorb, prefill
scheduling, adaptive-k, verification mode, expert cache policy, or snapshot
allocation. This task is only the DFlash drafter's K/V address/mask horizon.
