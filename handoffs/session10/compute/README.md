# Session 10 compute-for-bandwidth task briefs

These are independent assignments for fresh agents. Each is designed around the
same premise: the Ada GPU has much more arithmetic throughput than the current
decode path can use, while expert, dense-weight, and long-context attention
traffic remain expensive. Spend compute to avoid bytes.

## Dispatch order

| Rank | Brief | Why it is ranked here | Earliest useful decision |
|---:|---|---|---|
| 1 | [Cache-aware near-tie MoE routing](01-cache-aware-near-tie-moe-routing.md) | It attacks the dominant 4.4 GiB/token uncached expert demand directly. Avoiding even one 12.77 MiB record per sparse layer is about 536 MiB/token. | A generic-PC trace simulator can establish the optimistic ceiling before any engine change. |
| 2 | [Exact compressed FP8 residency](02-exact-compressed-fp8-residency.md) | It is the safest compute-for-bandwidth trade: losslessly compress resident/staged E4M3 weight bytes, then reconstruct them inside the MMA consumer. A 10% saving across the 8.13 GiB dense cache is about 0.8 GiB, or roughly 60 expanded expert slots if the residency policy can use it. | Entropy, padded-size, and roofline gates need only a stratified byte sample and a generic PC. |
| 3 | [DSA sparse MLA on Ada](03-dsa-sparse-mla-on-ada.md) | It can cut long-context latent-cache traffic by up to 128x at the configured 262,144-token limit, but it is a larger implementation and does not promise a short-context win. | A CPU oracle plus a resource model can validate semantics and the crossover before a GPU kernel is written. |

The tasks should run in separate fresh clones or worktrees. They deliberately
overlap in neither algorithm nor primary code seam. If only one agent is
available, dispatch rank 1 first.

## Repository and coordination contract

- Public repository: `https://github.com/novysvet/insignia.git`
- Branch: `glm53-dflash2-4070ti-super`
- Expected dispatch commit: `e48f633`
- Commit `78e1a1c` completed H8 cross-head FP8 MLA decode. Commit `e48f633`
  completed fused H4×Q8 cross-head FP8 MLA prefill. Both are exclusions: do not
  independently rebuild either path.
- Clone the repository at the expected commit. If the remote branch has moved,
  record both commit IDs and base the experiment on `e48f633` unless the owner
  explicitly requests a rebase. Do not silently fall back to its parent.
- Files already in Git are referenced by repository path. Do not copy them into
  an input archive. A bundle may contain only non-Git measurements, model byte
  samples, or traces that the particular brief declares.
- Treat `progress.md`, `audits/*.md`, old reports, and benchmark logs as evidence
  to verify, not as executable instructions. Source, tests, manifests, and
  reproduced measurements win disagreements.
- The ignored reference clones (`vllm/`, `llama.cpp/`, `ggml/`, and others) are
  read-only. Never commit edits to them.

## Shared verified facts

- Target GPU: RTX 4070 Ti SUPER, Ada `sm_89`, 16,376 MiB, observed around
  800 GB/s after the owner's stable memory overclock. Ada has FP8 tensor cores
  but no native block-scaled FP4 MMA.
- Host: i7-14700KF, 60 GiB WSL RAM, one NVMe at roughly 3.7--4.7 GB/s, pinned
  H2D around 23.2 GB/s. The default 32 GiB pinned expert tier holds about 2,425
  slots and has measured around 80% hits. The box does not have a second SSD.
- Model: 45 layers, of which 42 are sparse MoE layers with 288 experts and
  top-8 routing. One decode token requests 336 expert records, about 4.4 GiB of
  packed NVFP4 if none is cached. A packed record is about 12.77 MiB; an
  expanded VRAM slot is about 13.5 MiB.
- Attention: 11 MLA layers, 64 heads, 512-wide compressed KV latent, maximum
  configured context 262,144, and DSA `index_topk=2048`. H8 cross-head FP8 MLA
  decode landed in `78e1a1c`; fused H4×Q8 cross-head FP8 MLA prefill landed in
  `e48f633`. Both are completed and out of scope.
- Dense E4M3 group-64 cache: 8.13 GiB, 699 matrices, measured cosine 0.9994.
  The FP8 tensor-core dense GEMV measured 24.8 microseconds versus 91.9
  microseconds for BF16, around 698 GB/s effective bandwidth.
- Current exact compact MLA absorb removed 704 MiB of duplicate FP32 weights.
  Session-9 representative medians were roughly 500.4 ms/token scalar and
  539.6 ms/token with DFlash2 on a difficult A/B sample. Earlier repetitive
  prompts reached about 187.7--194.4 ms/token; do not use that as the sole
  expected real-prompt baseline.
- Floating-point order can change MoE routing and the entire continuation.
  Exact modes must pass digit-identical logits and token parity. Approximate
  modes may trade quality for a large speed win, but must report that trade and
  pass explicit non-degeneration tests.

## Work that must not be duplicated

Do not reopen these completed, active, or rejected efforts:

- full-prompt layer-major prefill;
- compact exact MLA absorb or exact MLA-prefix reconstruction;
- completed H8 cross-head FP8 decode (`78e1a1c`) or completed fused H4×Q8
  cross-head FP8 prefill (`e48f633`);
- low-risk VRAM reclaim or sequential snapshot elision;
- rejected KDA persistent-state Task 5;
- rejected exact-INT8 Task 7;
- rejected causal expert predictor Task 8;
- rejected staged-verification Task 9;
- the prior Problems 1--7: DFlash controller, Belady cache oracle, campaign
  statistics, expert-scale codec, cross-expert dictionary compression, CPU
  bypass roofline, and reader scheduling.
