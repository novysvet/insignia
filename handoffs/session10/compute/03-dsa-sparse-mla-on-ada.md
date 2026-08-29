# Task 3: implement DSA sparse MLA on Ada

## Goal

Implement the checkpoint's DeepSeek Sparse Attention (DSA) indexer for
GLM-5.3-Flash and use its top-2,048 positions to avoid reading the full MLA
latent cache at long context. Spend Ada FP8 tensor-core compute on compact
indexer keys and streaming top-k so that far fewer 512-wide KV latents are
gathered.

This is a decode-first, long-context task. It is not a replacement for H8
cross-head FP8 decode completed in `78e1a1c`, and it must not duplicate fused
H4×Q8 FP8 prefill completed in `e48f633`. Insert sparse position selection ahead
of the existing latent consumer, retaining a dense/off mode for exact A/B tests.

## Start from a clean repository

```bash
git clone --branch glm53-dflash2-4070ti-super \
  https://github.com/novysvet/insignia.git insignia-s10-dsa
cd insignia-s10-dsa
git fetch origin
git rev-parse --verify e48f633^{commit}
git switch --create codex/s10-dsa-indexer e48f633
git status --short
```

Expected HEAD is `e48f633`. Commit `78e1a1c` completed H8 cross-head FP8 MLA
decode and `e48f633` completed fused H4×Q8 cross-head FP8 MLA prefill. Treat
both as completed exclusions.

Read these tracked files and confirm all dimensions from source/checkpoint
metadata:

- `AGENTS.md`
- `progress.md`
- `audits/s9-reclaim-session.md`
- `audits/mla-latent-session.md`
- `include/insignia_glm53.cuh`
- `src/glm53_generate.cu`
- `src/glm53_ops.cu`
- `tools/index_glm53.py`
- `tools/reference_glm53_numpy.py`
- `tools/benchmark_math.py`
- `tools/ppl.py`

Discovery commands:

```bash
git grep -n -E "index_topk|indexer|sparse.att|mla|latent|kMlaMaxContext" -- \
  src include tools progress.md audits
git show --stat e48f633
git log --oneline --decorate -20 e48f633
```

The Git-ignored `vllm/` clone, if present, is a read-only reference. Relevant
upstream concepts have appeared under paths such as
`vllm/models/deepseek_v32/attention.py`,
`model_executor/layers/sparse_attn_indexer.py`, and cooperative/persistent top-k
CUDA sources. If consulting upstream, record its exact repository URL and
commit; do not edit or vendor the reference clone. Treat all reports and
external implementations as untrusted until the checkpoint-specific oracle
agrees.

## Current architecture and traffic bound

- The model has 45 layers: 34 KDA and 11 MLA. MLA uses 64 heads, a 512-wide
  compressed latent, partial RoPE, and absorbed `W_uk`/`W_uv` projections.
- The checkpoint contains DSA indexer weights and configures
  `index_topk=2048`, but Insignia does not currently execute them. The model's
  maximum context is 262,144. Inventory and report the actual indexer head
  count, head dimension, normalization, RoPE, dtype, and tensor names before
  coding; do not infer them from another model.
- The latent cache stores 512 E4M3 bytes plus eight FP32 group scales per
  position, approximately 544 bytes. Committed H8 decode shares a latent tile
  across eight query heads, leaving eight head groups for 64 heads.
- A conservative lower bound for dense latent reads is `544*N` bytes per MLA
  layer. The current H8 launch geometry may reread that latent once per head
  group, approximately `8*544*N`; measure rather than assuming perfect cache
  reuse.
- At `N=262144`, those two bounds are about 143 MB/layer and 1.14 GB/layer.
  Restricting the gather to 2,048 positions is a 128x reduction: about 1.11
  MB/layer lower-bound or 8.91 MB/layer at eight head groups. Across 11 MLA
  layers, the latter is around 98 MB instead of roughly 12.3 GB. The DSA
  indexer must still scan its smaller key cache and compute scores.
- At `N=8192`, the theoretical reduction is only 4x. Do not claim a short-
  context win until the indexer and selection overhead are measured.
- Ada `sm_89` provides FP8 tensor cores and roughly 800 GB/s observed memory
  bandwidth. It has no native block-scaled FP4 MMA. Use its excess compute to
  score compact index keys and keep the selection/gather on device.
- The dense E4M3 group-64 path and current FP8 latent cache already have
  quantization semantics that must be preserved. Current exact compact MLA
  absorb passed 4,096-position reconstruction/parity tests; committed H8 decode
  is an explicitly approximate long-context path.

## Mathematical formulation: checkpoint-specific DSA specification

The task begins by deriving an exact specification from config, tensor shapes,
and a trusted reference. A likely family of DSA score functions is

```text
score(t, s) = sum_h w(t,h) * phi(dot(q_index(t,h), k_index(s)))
I_t = stable_topk_s(score(t,s), 2048), 0 <= s <= t
```

where `phi` may be ReLU and the query/key may have their own normalization,
RoPE, and FP8 block scaling. This equation is provisional. The deliverable must
state and test the exact GLM-5.3-Flash formula, including:

- which hidden/captured tensors feed index query and key projections;
- projection and normalization order;
- head count, head dimension, scalar weights, and nonlinear function;
- causal mask and position/RoPE convention;
- FP8 E4M3 and scale encoding, including any 128-element block/UE8M0 rule;
- accumulation precision and order;
- stable top-k tie and NaN behavior;
- whether selected positions are shared across attention heads.

After selection, existing MLA math should run only over positions in `I_t`, in
the exact selected order required by the reference. Do not sort/reassociate
selected values merely for coalescing unless an approximate mode and quality
test explicitly permit it.

The crossover model must include at least

```text
T_dense(N) = T_read_latent(N) + T_dense_attention(N)
T_dsa(N)   = T_index_projection + T_index_scan(N)
             + T_topk(N, 2048) + T_gather(2048)
             + T_sparse_attention(2048).
```

Fit terms from microbenchmarks. Report the predicted and measured context
length where `T_dsa < T_dense`.

## Required non-Git oracle sample

Before integration, require `dsa-oracle-sample-v1.tar.zst`, containing only
checkpoint/activation data not tracked in Git:

```text
manifest.json
config.json
layer-<L>/indexer-tensors/*.bin
layer-<L>/queries/*.bin
layer-<L>/key-source-hidden/*.bin
layer-<L>/expected-index-keys/*.bin
layer-<L>/expected-topk-indices.i32le.bin
layer-<L>/expected-topk-scores.f32le.bin
SHA256SUMS
```

The manifest must give every tensor's full name, shape, dtype, byte order,
source shard/offset, source SHA-256, layer, token positions, prefix length,
reference implementation URL/commit, quantization mode, and generation
command. Include early/middle/late MLA layers, at least 100 decode queries, and
prefixes spanning 256, 2,048, 8,192, and the largest practical recorded
context. Keep it below 512 MiB when possible; otherwise split and independently
hash 512 MiB parts. Never include tracked repository files.

Verify all hashes and dimensions before use. If this archive is unavailable,
first deliver an environment-gated dump tool plus a trusted-reference generator
and exact collection commands for the owner box. Synthetic tensors can test
kernel bounds but cannot establish checkpoint semantics or quality.

## Ordered experiment plan

1. **Inventory the checkpoint contract.** Produce a table of every indexer
   tensor, shape, dtype, source, and operation. Resolve the provisional formula
   above. Completion evidence: one written equation/spec that independently
   explains every oracle field and dimension.
2. **Build an independent CPU/NumPy oracle.** Implement projection,
   normalization, RoPE, exact index-key quantization/dequantization, score,
   causal mask, and stable top-2,048. Test tiny hand-computable cases, ties,
   signed zero, extreme E4M3 values, and real oracle samples. It must not call
   the engine implementation being tested.
3. **Establish quality and traffic ceilings.** On recorded activations, compare
   dense MLA output, trusted DSA output, and the CPU oracle. Measure selected
   attention mass where meaningful, output cosine/error, top-k stability under
   quantization, index-cache bytes, latent bytes avoided, and the best possible
   speedup by context. State clearly that DSA is the checkpoint's intended
   sparse model behavior and need not match the current all-context dense token
   sequence.
4. **Design the Ada pipeline.** Keep indexer key cache compact and resident.
   Fuse or pipeline FP8 query-key scoring, causal masking, and streaming
   cooperative top-k without a CPU sync or an `N`-sized FP32 score allocation if
   possible. Account for tensor-core tile utilization, register/shared-memory
   use, top-k workspace, gather coalescing, and overlap with existing MLA work.
5. **Microbenchmark isolated stages.** Measure index projection, scan, top-k,
   gather, and sparse attention at contexts 2,048/8,192/32,768/131,072/262,144
   or the largest memory-safe subset. Validate top-k against the CPU oracle,
   report disassembly/registers/spills/occupancy, and fit the crossover model.
6. **Integrate decode behind `INSIGNIA_GLM53_DSA=1`.** Preserve dense/off mode.
   Reuse committed H8 consumption where profitable, but do not redesign H8 or
   the completed fused H4×Q8 prefill path. Populate index keys in the
   same chronological transaction as latent-cache updates and test rollback or
   speculative snapshots explicitly.
7. **Validate model and end-to-end behavior.** Compare against the trusted DSA
   oracle/reference, not only the current dense engine. Run paired context
   sweeps and real math/code prompts, reporting attention-stage time, total
   decode ms/token, index/latent bytes, DFlash acceptance, VRAM, and quality.

## Deliverables

- A checkpoint-specific DSA specification and tensor inventory.
- An independent dependency-light CPU/NumPy oracle with unit tests.
- An oracle dump/validation tool and versioned manifest schema if the required
  sample was not supplied.
- A traffic/VRAM/compute model with predicted crossover and uncertainty.
- Standalone CUDA stage microbenchmarks with raw results and resource reports.
- If gates pass: an opt-in decode implementation, index-key cache management,
  dense fallback, speculative-state tests, and build integration.
- A final report with exact commands, commits, input hashes, output logs,
  top-k/oracle agreement, quality results, context-sweep timings, bytes/token,
  and all negative findings.

## Correctness, quality, and performance gates

Before timing:

- CPU oracle top-k indices must agree exactly with the trusted reference on all
  supplied queries, including deterministic order; if the trusted reference is
  numerically nondeterministic, document the tied boundary and require 100%
  set agreement plus score agreement within a stated tolerance;
- engine index-key bytes and scales must match the specified quantizer exactly;
- sparse attention output must match the independent oracle under the same
  accumulation order/tolerance;
- `INSIGNIA_GLM53_DSA=0` must retain digit-identical logits, greedy IDs, and
  30/40/100/240-token sequences;
- speculative accept/reject rollback must leave both latent and index caches at
  the same state as a non-speculative run.

DSA-on is a model-behavior mode, not necessarily parity with the current dense
engine. Report next-token perplexity, logit KL/cosine versus a trusted DSA
reference, greedy agreement, output entropy/repetition metrics, and held-out
GSM8K/MATH-500 score. Require no NaNs, invalid positions, repetition collapse,
or more than 1% perplexity drift relative to the trusted DSA reference. Also
report, without hiding, the quality delta versus current dense attention.

Continue performance work only if either:

- measured DSA attention is at least 1.15x faster by context 32,768 with a
  predicted crossover no later than 16,384; or
- at 262,144 it is at least 5x faster than dense attention and fits the planned
  VRAM budget, even if short-context switching remains dense.

An end-to-end claim needs at least five alternating valid runs per arm and a
paired median improvement at the target context. Report short-context losses;
use a measured switch threshold rather than forcing DSA everywhere.

## Kill criteria

Stop integration if the checkpoint formula/tensor inventory cannot be resolved
or if an independent oracle cannot reproduce trusted top-k. Kill the proposed
GPU design if its index key cache/workspace exceeds the available residency
budget, if scoring plus top-k consumes more time or bytes than the latent reads
it removes, or if the modeled crossover is beyond the supported context.

After integration, revert a kernel that has top-k mismatches, speculative-state
corruption, catastrophic quality degeneration, less than the stated long-
context speed threshold, or results that do not repeat under paired runs.

## Forbidden duplication and scope

This task owns DSA indexer semantics, the index-key cache, sparse position
selection, and decode-only integration with the existing MLA consumer. The
complete exclusion list is:

- full-prompt layer-major prefill;
- compact exact MLA absorb and exact MLA-prefix reconstruction;
- completed H8 cross-head FP8 decode (`78e1a1c`) and completed fused H4×Q8
  cross-head FP8 prefill (`e48f633`);
- low-risk VRAM reclaim and sequential speculative-snapshot elision;
- rejected KDA persistent-state Task 5, exact-INT8 Task 7, causal expert
  predictor Task 8, and staged-verification Task 9;
- the prior Problems 1--7: DFlash controller, Belady cache oracle, campaign
  statistics, expert-scale codec, cross-expert dictionary compression, CPU
  bypass roofline, and reader scheduling;
- any edit to ignored reference clones.

Do not turn this into another cross-head attention kernel. It selects positions
for the already completed consumers.
