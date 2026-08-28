# DFlash2 session audit (2026-08-28, session 3)

Paper-research digests with links (DFlash2 + the ten MoE offloading/prefetch
papers) live in `audits/papers-session3.md`.

Goal context: 12 tok/s decode on GLM-5.3-Flash abliterated (from 690 ms/tok =
1.45 tok/s) without going below FP8/NVFP4 for weights, plus faster prefill.
Strategy: DFlash2 block-diffusion speculative decoding as the multiplicative
lever (acceptance ceiling ~5-6 tok/round per the inco.ai evals), then tier
sizing, dual-SSD striping, prefill work.

Status at time of writing: **machinery complete and parity-exact, acceptance
0 extra tokens/round; root cause narrowed to two candidate bugs (engine
drafter-forward kernel divergence from the NumPy oracle at layer 0, and/or the
engine's known deep-layer parity issue poisoning the captures).**

## 1. What DFlash2 is (verified against z-lab/dflash model.py + SGLang PR #36708 branch `xinyuan/glm-5.3-flash-support`)

- Block-diffusion drafter, **one forward pass per decode round, no diffusion
  iterations**. Block = [anchor embed, mask_token 154856 x7], positions are
  absolute S..S+7. Outputs read at mask positions 1..7 only.
- 5-layer Qwen3-style backbone: hidden 4096, 32q/8kv heads, head_dim 128,
  SwiGLU 12288, RMSNorm eps 1e-5, **RoPE theta 1e4, neox (rotate-half), full
  128 dims, applied AFTER per-head q_norm/k_norm**. Draft theta is unrelated
  to the target's.
- **Target-feature injection is KV-only**: per drafter layer,
  K/V = k_proj/v_proj(RMSNorm(fc([H5;H14;H24;H33;H42]))) of the context
  tokens; Q never sees target features; O-proj/FFN bypassed. fc is
  20480->4096 (one shared), hidden_norm RMSNorm once at model level.
- GLM capture semantics (SGLang glm5_next.py `_prepare_aux_hidden_state`):
  under mHC, residual is folded in, so capture = **hc_contract = mean over
  the 4 mHC streams of the layer's completed widened output** (capture
  before layers 6/15/25/34/43 = outputs of layers 5/14/24/33/42). Plain
  mean, NO norm at capture. This is exactly the engine's
  average_streams_kernel.
- Two-tap grouped dynamic conv (`conv_kernel_size 2, conv_group_size 16` ->
  256 groups) wrapped around BOTH sublayers:
  `y = x + Finish(SubLayer(Prepare(LN(x))))`. Both sides' dynamic
  coefficients come from ONE kernel_projection(x_ln) of the sublayer INPUT,
  row layout `(side*taps + tap)*groups + group` = side*512 + tap*256 + g.
  `out[t,c] = (base[side,tap,c] + dyn[t,side,tap,g(c)]) * x[t-tap,c]`, tap-1
  zeroed at block-local position 0 (causal within block only — block resets
  every round). base_kernel [2(side),2(tap),4096] initialized tap0=1.0;
  trained ckpt has tap0 per-channel mean ~0.97-1.09, tap1 mean ~0 (verified
  healthy in the actual checkpoint).
- Attention mask: `is_causal: false` => **bidirectional within block**, committed
  prefix limited by window_left=2047 (never bites below that). No softcap, no
  alibi, scale 1/sqrt(128), GQA repeat 4.
- Noise embedding scale: `get_dflash_noise_embedding_scale` defined by NO
  model on the branch => **1.0, raw embed rows** (anchor = its own raw
  embedding).
- Selector: top-16 of the logits row per position; score =
  `unary + <A[pred] * hp[t], B[cand]>` (A=predecessor_codebook,
  B=successor_codebook, both [vocab,256] bf16, hp = hidden_projection
  [4096->256] of the drafter hidden); **greedy left-to-right walk** (not
  Viterbi), predecessor of slot 0 = anchor id. Unary = target lm_head on the
  drafter's final-normed hiddens (draft borrows BOTH embed table and
  lm_head; no multiplier, no softcap for GLM).
- Verification: plain flat-block greedy — target argmax over the 8 block
  positions; accept prefix of drafts that match; bonus = argmax at the first
  mismatch. Output identical to plain greedy by construction. Draft K/V
  cache holds ONLY committed tokens' projected captures (block K/V are
  scratch, cropped after each round). Reported acceptance (inco.ai, base
  model, GB300x4): 4.03-5.86 accepted/round across tasks, 1.7-2.8x vs AR.

## 2. Infrastructure landed this session

- Checkpoint: `C:\models\GLM-5.3-Flash-DFlash2` (2.34 GB BF16 single
  safetensors, 81 tensors) copied to
  `/var/lib/insignia/glm53-dflash2.safetensors`, sha256 prefix verified
  `8931dc522be0aa31` on both sides.
- `tools/index_dflash2.py` — writes stock IGLMIDX1 index for the single
  file (geometry block carries drafter values; nothing validates them).
  Output: `/var/lib/insignia/glm53-dflash2.index` (81 tensors).
- `tools/quantize_dflash2.py` — g64 FP8 cache in the stock IGLMF8A1 format
  (same encoder math as quantize_glm53_q8.py --format fp8: per-64-group
  absmax/448 fp16 scales, torch e4m3, NaN-code check). fc.weight [4096,20480]
  exceeds the kernel's 256-group column cap => split into fc.a/fc.b
  [4096,10240] column halves. Names: L{n}.{q,k,v,o,gate,up,down,akp,mkp},
  fc.a, fc.b, hp. Output `/var/lib/insignia/glm53-dflash2-fp8.{bin,index}`
  (48 matrices, 1073.5 MiB, 7.1 s). Norms/base kernels/codebooks stay BF16
  in the safetensors.
- `src/glm53_dflash2.cu` + `include/insignia_glm53_dflash2.cuh` — the
  drafter: FP8 tensors all pinned in VRAM at construction (~1.07 GiB), k/v
  caches [5][264][8*128] fp32 (~2.7 MB), block forward with
  fp8_tc_gemv_batch / fp8_tc_gemv2, custom kernels: df_rms_rows (rows x
  4096 RMSNorm), df_norm_rope (per-head RMS + neox rope, warp per head),
  df_attn (warp per (pos,head) against ctx+block keys), df_conv (2-tap
  grouped dynamic conv), df_kv_append (k_norm+rope+cache write),
  df_gather (5 captures -> fc.a/fc.b column-split inputs), host selector
  walk (top-16 insert-sort scan + rank-256 bilinear dot per candidate).
- Runner integration (src/glm53_generate.cu): env `INSIGNIA_GLM53_DFLASH2=1`
  (paths overridable via `INSIGNIA_GLM53_DFLASH2_INDEX`,
  `INSIGNIA_GLM53_DFLASH2_FP8`), pinfed host logits/hp buffers, capture
  hooks after the FFN mhc_mix at kDfCaptureLayers {5,14,24,33,42} in BOTH
  step() (token 0) and prefill() (per token), commit points:
  prompt chunks + step() commit their own tokens; verify rounds commit the
  accepted prefix via df_commit(matched, position+1) from the main loop.
  Draft loop `df_draft()` stages embed rows, forwards, runs the target
  lm_head over the 7 draft hiddens, downloads logits+hp, selector walk.
  Main loop branch mirrors the MTP flow: d1 vs truth0, prefix-match,
  rollback_kda on partial accept, empty-round falls back to one exact
  step().
- CCT compile break fixed: `load_cct()`/`cct_prefetch()` now implemented
  (static cross-layer top-8-per-expert successor tables,
  `INSIGNIA_GLM53_CCT=<file>`, magic "CCT0", u32 header layers/experts/
  topk, uint16 tables per adjacent sparse pair; prefetch = union of routed
  experts' successor lists capped at 16 records via the existing
  ExpertStager::prefetch). `if (cct_)` -> `if (!cct_.empty())`.
- Build: glm53-gen.sh links glm53_dflash2.cu. Do NOT link src/ops.cu into
  this target (missing `<cstdint>` include in insignia_ops.cuh breaks
  nvcc; drafter carries its own df_silu_mul_kernel instead).
- Debug/dump env vars: `INSIGNIA_GLM53_DF_DEBUG` (per-round anchor/truth0 +
  top-5 of first 5 draft logits rows to stderr),
  `INSIGNIA_GLM53_DF_DUMP=<path>` (binary oracle dump, tags below),
  `INSIGNIA_GLM53_DF_LTRACE` (adds per-layer x_block snapshots).
- Dump format (df_dump_file() — ONE shared static FILE*, the two-per-class
  static-fopen version truncated itself): tag 1 commit {u8, i32 count, i32
  pos0, f32[count*5*4096] captures}, tag 2 draft {u8, i32 anchor, i32
  position}, tag 3 forward result {u8, i32 pos, i32 n, f32[8*4096]
  x_block, i32 n, f32[7*4096] hidden}, tag 4 layer trace {u8, i8 layer,
  i32 n, f32[8*4096]}. Records interleave commit/draft/trace in time order.
- `tools/dflash2_oracle.py` — independent NumPy replay: reads the dump,
  rebuilds drafter context K/V from captures in fp64-fp32 numpy (NO FP8),
  runs the block forward, prints top-5 per position, truth rank (pass the
  greedy sequence as arg 6, comma-separated), engine-vs-oracle cosines for
  x_block/hidden (tag 3) and per-layer (tag 4 + LTRACE), zero-context
  ablation (arg 7 `zero`), gaussian capture-noise ablation (arg 8, relative
  sigma). lm_head is mmap'd from the store shards.

## 3. Measured state and evidence

- Parity gate: 60-token run with the modified binary (DFlash2 off) matches
  bench-base.txt exactly (greedy IDs + top-10 logits digit-identical).
  With DFlash2 ON the committed sequence ALSO matches plain greedy exactly
  (verify machinery is correctness-preserving as designed) — 30/30 tokens
  identical on the seed-154820 loop prompt.
- Acceptance: **1.00 accepted/round, 30/30 rounds empty** (draft never
  matches truth0, or matches nothing before the first mismatch). Round cost
  ~4.6-4.9 s/token (empty round = wasted draft + full verify prefill of 8
  positions + exact step fallback). This is the MTP symptom all over again
  BUT with richer diagnostics (below).
- Engine draft logits shape: t0 (first mask position) low-confidence wrong
  (top ~3.8 logit); t1+ confidently wrong (10-15) — i.e. the drafter runs
  and produces fluent-ish garbage, context-blind-like.
- NumPy oracle (independent, no FP8, no CUDA): ALSO predicts wrong tokens.
  Truth ranks at t0: 1076 (round 0), 81 (round 1), 1729 (round 2). So the
  failure is NOT purely a CUDA kernel bug — the semantics/data feed itself
  produces wrong predictions in a clean-room reimplementation.
- BUT engine and oracle disagree with each OTHER too: engine-vs-oracle
  x_block cos 0.010 at layer 4 output, engine |x_block| max 4.05e6 vs
  oracle 4.0e5. Per-layer (LTRACE): layer 0 cos 0.665, layer 1 0.696,
  layer 2 0.748, layer 3 0.404, layer 4 0.010 — **engine drafter forward
  diverges from the oracle already inside layer 0** (separate kernel bug,
  must fix regardless).
- Both implementations explode inside layer 0's attention finish-conv:
  oracle stats x0 max 4.1e-2, xn 5.4, dyn 61.2, att_in 96, q 6.6, o 334,
  attn_fin 2.19e4. dyn (kernel_projection output) reaching ~61 and getting
  multiplied with o (334) gives e4-scale outputs; subsequent RMSNorms keep
  re-normalizing so the model "runs" but the finish-conv deltas dominate.
  The z-lab source confirms our formula; possibility remains that the
  trained magnitudes are genuinely like this and something UPSTREAM feeds
  wrong activations into kernel_projection.
- Context flows: zeroing the committed K/V changes predictions completely
  (top-5 becomes uniform-ish low-confidence). So K/V are connected.
- Capture-noise ablation: +5% and +30% relative gaussian noise on the
  captured features barely moves predictions (same top-1 in most rounds).
  => the drafter is NOT hypersensitive to capture perturbation; the
  "abliteration shifted the manifold" hypothesis is WEAKENED as the sole
  explanation (contrast with the MTP layer-45 failure which had sharp
  confident wrong predictions from tiny input changes).
- Weight sanity: all 81 tensors load with sane stats (fc std 0.16,
  q_proj std 0.09, base_kernel tap0 means ~1.0 = trained-init signature,
  hidden_norm mean ~1.01). Loading is not the bug.

## 4. Candidate root causes, ranked

1. **Engine deep-layer parity issue (AGENTS.md known-unresolved)**: if the
   engine's residual stream at layers 5+ has drifted from HF/transformers
   semantics (MLA/KDA/mhc subtlety), the captured features are off-manifold
   for a drafter trained on the base model's true residual stream. The
   noise-robustness result argues the drafter tolerates *noise*, but not
   *systematic semantic drift*. TEST PENDING: engine per-layer mean-of-
   streams vs tools/reference_glm53*_numpy.py independent reference
   (cosines at layers 1,2,3,4,5,14 for the seed token). Agent dispatch for
   this failed (ticket error) — needs rerun. NOTE: the engine matches its
   OWN parity gate (greedy IDs + logits) which is self-consistent, not
   externally validated beyond layer 0.
2. **Drafter-forward kernel bug(s) in glm53_dflash2.cu**: engine vs oracle
   diverge at layer 0 already. Bugs found and fixed so far in this file:
   - conv/residual launch grids 32 blocks -> 128 (8x4096 needs 32k threads)
   - finish-conv read the stashed dyn1 buffer with stride 1024 instead of
     512 (found via an oracle indexing crash of the same shape)
   - df_kv_append grid (was (count*8+7)/8 blocks with warp-per-row body)
   More may remain; next bisect step is sublayer-level trace dumps
   (post-attention, post-MLP) and comparing against oracle intermediates;
   prime suspects: df_attn warp-per-(pos,head) tiling, fp8_tc_gemv_batch
   workspace reuse across consecutive calls (workspace is sized for
   cols=12288 batch 32 — verify activation quantize layout matches the
   kernel's expectation for x rows vs the later GEMV that consumes it),
   and the df_rms_rows smem reduction (256 threads, 8 warps, ws[32]).
3. **Oracle bug** (it shares my reading of the conv/rope/selector; the
   z-lab source cross-check confirmed the math, but e.g. its lm_head is
   fp32-widened bf16 with no final-norm bf16 round-trip, and SGLang does a
   bf16 round-trip after qk-norm before RoPE — precision details, unlikely
   to explain rank-1000 failures).
4. Abliterated-manifold shift: weakened by the noise-robustness result but
   not excluded (would predict oracle-correct-after-bugfix still wrong).
   The decisive experiment after fixes: run the drafter against captures
   produced by tools/reference_glm53_numpy.py (true-model features) and
   see if acceptance appears.

## 5. Also measured this session (perf)

- WSL config: memory=14GB in .wslconfig; guest reports 13 GiB total, 12
  free at idle; C: has 58 GB free (vhdx lives on C:, 980 PRO).
- E: drive: 464 GB free. ABLITERATED copy on E: = 380,260,018 KB
  (~380 GB, still present; the vhdx inside holds a byte-verified compact
  copy, so the E: copy is deletable per user when needed).
- DFlash2 empty-round decode cost: ~4.6-4.9 s/tok (vs 0.69 s plain) —
  draft+verify8+fallback step on an all-miss round is ~6.8x WORSE than
  plain decode, because verify prefill pays 8-token expert I/O without
  reusing the host LRU well (NVFP4 cache hit rate fell to 0.4-0.7% on
  those runs vs 29.3% baseline: the verify union admits everything,
  thrashing the 379-slot tier). **Do not ship speculative decoding until
  acceptance > ~1.5** — break-even needs accepted/round > ~1.4 just to
  match plain decode at current per-round costs (draft ~50 ms + verify ~8
  positions x ~0.55 s/position-of-expert-IO amortized by dedup).
- Hierarchy bandwidth steady at 5.45-5.5 GB/s across these runs.

## 6. Files touched (session 3)

- NEW: include/insignia_glm53_dflash2.cuh, src/glm53_dflash2.cu,
  tools/index_dflash2.py, tools/quantize_dflash2.py, tools/dflash2_oracle.py
- MODIFIED: src/glm53_generate.cu (include, ctor DFLASH2 block, members
  df_/dflash2_on_/df_logits_host_/df_hp_host_, kDfCaptureLayers constant +
  average_streams_kernel hook in step()/prefill(), df_draft/df_commit,
  main-loop DFLASH2 branch, load_cct/cct_prefetch implementations,
  cct_ bool fixes, DF_DEBUG prints), build/glm53-gen.sh (dflash2.o),
  .gitignore (audits/ visible? — audits/ was already untracked-visible)
- Data in /var/lib/insignia: glm53-dflash2.safetensors (+config.json),
  glm53-dflash2.index, glm53-dflash2-fp8.{bin,index}, df-dump.bin (debug)

## 7. Next steps (in order)

1. Fix the engine-vs-oracle layer-0 divergence (sublayer-granularity trace;
   suspects listed in section 4.2). Gate: engine x_block vs oracle cos
   > 0.999 at every layer for round 0.
2. Run the engine-vs-independent-NumPy-reference layer parity check
   (layers 1..5,14) to validate/culpate the capture stream. If the engine
   has drifted, the DFlash2 project becomes downstream of fixing the
   deep-layer parity issue (which ALSO explains the parked MTP failure).
3. If both pass and acceptance is still ~1: test the oracle against
   true-model features (reference-generated captures) to separate
   drafter-training mismatch (abliteration) from feed mismatch; consider
   fine-tuning fc/hidden_norm only (cheap, 84M params) on engine captures.
4. Only after acceptance > 1.5: optimize the round (draft is ~50 ms; verify
   dedup already works; look at LRU admission policy for verify unions).
5. Parallel non-DFlash2 levers still open: host-tier bump toward the 672+
   slot cliff (needs the WSL pinned-memory ceiling re-measured at 14 GB
   guest RAM), VRAM expert tier, dual-SSD striping via
   INSIGNIA_GLM53_ALT_SHARD_DIR (stripe_copy.py rate-limited), CCT prefetch
   table generation (tools/dump_cct.py exists; loader now implemented),
   prefill union-I/O improvements.
