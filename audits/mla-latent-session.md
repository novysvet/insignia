# 4070 Ti SUPER bring-up + latent MLA + DFlash2 regression audit (2026-08-28, session 4)

Scope: glm-box hardware bring-up, the DFlash2 remote baseline campaign, the
latent-MLA rework diagnosis and hybrid bridge, and the currently-open DFlash2
acceptance regression. All numbers from `run-engine.sh`-style invocations on
glm-box WSL unless noted. Commits: `5afebc5..fc046a3` on
`glm53-dflash2-4070ti-super`.

## 1. glm-box bring-up

- Branch deployed to a dedicated worktree `C:\coding\Insignia-glm53-dflash2`
  (original `C:\coding\Insignia` checkout is a stale dirty snapshot — never
  touched). First build broke on CRLF-converted `.sh` files; fixed
  repo-side with `.gitattributes` (`*.sh text eol=lf`).
- DFlash2 FP8 cache regeneration on the clean host: PyTorch dependency removed
  from `tools/quantize_dflash2.py` in favor of a native E4M3FN encoder
  (verified byte-identical to torch on 8,000,001 samples); the same encoder
  then replaced torch in `tools/quantize_glm53_q8.py`. Full 699-matrix FP8
  cache builds without any training framework installed.
- FA2/DFlash boundary bug fixed: `verify_round` always verified all k drafts,
  so the final round at ≥236 generated tokens launched speculative rows past
  the 256-row KV allocation and `mla_flash2_prefill` correctly refused. Fix:
  round width = `min(k, remaining_output)`.

## 2. Remote performance campaign (single SSD, OC validated)

Scalar baseline ~447 ms/tok. DFlash2 k4: 414.7 → 399.6 (1024 slots) → 337.0
(24.57 GiB / 1819 slots) → 239.7 ms/tok over 100 tokens at 32 GiB (2425
slots, 80.3% expert hits; 40 GiB regressed to 244.5 — pinning pressure
outweighs hits). Raptor Lake `-march` build: ~0.8% (kept for AVX-VNNI
enabling, not for the headline). Six readers regress; four stays optimal.

Best sustained: **k7 at 235 generated tokens: 187.7 ms/tok (5.33 tok/s),
7-of-7 accepted in 32/35 rounds**; 240-token endurance run 194.4 ms/tok
(5.14 tok/s), 56.5% faster than scalar, bit-exact output. k4 acceptance
saturates (4/4 in 59/62 rounds) which is why k7 wins once the cache covers
the verify union.

Cache-slot sweep (hit rate): 488→50%, 999→72%, 1819→79.5%, 2425→80.3%, flat
after. The 672-slot cliff from the local-box simulation is confirmed but the
RAM to blow past it only exists on this machine.

## 3. Latent MLA: diagnosis chain (the long war)

Goal: replace the expanded 16384-wide K/V cache (~11.8 GiB FP32 at 8192 ctx)
with the 512-wide compressed latent (~50 MiB FP8), absorbed W_uk/W_uv
attention, context cap 256→8192. Initial state produced degenerate text
(`220 16 220 17...`).

Ruled out, in order, with evidence:

1. **Batched prefill kernel**: forcing scalar prefill gave bit-identical wrong
   results — exonerated.
2. **Weight-source mismatch** (absorbed weights loaded from raw BF16 while the
   old path executed the FP8 cache): real but secondary — BF16-only control
   still diverged (`220 13` vs `287 7326`).
3. **Independent NumPy recomputation** of both parenthesizations from dumped
   tensors: expanded vs absorbed differ by only ~2.6e-7, CUDA matches both —
   formula, live cache contents, and decode kernel correct.
4. **Root mechanism**: FP8-latent quantization perturbs attention by ~1e-6,
   and the discrete top-8 router amplifies it. First expert-set flip: layer 6
   (FP8 latent), layer 12 (FP32 latent). Layer-3 hidden RMS 1.66e-4.

Fixes landed and kept:

- **Group-scaled FP8 latents** (8×64-wide scales per token, matching the dense
  FP8 convention): layer-3 RMS 1.66e-4→1.61e-4. Objectively better, not the
  fix.
- **Match absorbed MLA to dense FP8 weights**: absorbed kernels now consume
  the exact dequantized FP8 `kv_b_proj` operands the old path executed. Token
  2 restored exactly; first-MLA-layer RMS 1.61e-4→4.56e-5. First real fix.
- **Exact 256-token oracle restored** (`INSIGNIA_GLM53_MLA_LEGACY=1`): the
  other agent's online-softmax rewrite of the legacy kernels was itself a
  regression; the two-pass operation order is bit-exact vs the old binary and
  is the reference for every A/B since.
- **Canonical-order MoE accumulation probe: REJECTED.** Sorting the same
   top-8 set before summation changed legacy tokens immediately. The
   accumulation order is part of the effective model (→ determinism law in
   AGENTS.md).
- **Decomposition experiment**: score-side absorption contributes 1.90e-6 RMS,
  value-side reassociation 6.67e-6 (~78% of error energy). An exact-FP32-value
  prefix hybrid (176 MiB) hit only 1.49e-6 at the first MLA layer yet flipped
  the first expert set EARLIER (layer 9 vs 12) — error magnitude does not
   predict route stability. Rejected as default.
- **Shadow-prefix bridge (current coherent path, `fc046a3`)**: exact expanded
  K/V for positions < 256 while the 8k FP8 latent cache is populated.
  352 MiB. Reproduces all 12 oracle IDs exactly. This is the shipping path;
  latent attention beyond 256 remains unvalidated.

## 4. DFlash2 on the bridge: open regression

With the exact-prefix path active, DFlash2 output stays greedy-exact (30/30)
but acceptance collapsed to 1.43 tokens/round (15/21 empty rounds, 516.7
ms/tok vs the pre-bridge 187.7–194.4). Not a memory issue — drafter/verify
alignment on the restored exact path. Investigation state at session end:
prior acceptance-fix commit `90fb255` reviewed, verify-state replay and
capture feed on the exact-prefix path are the suspects; nothing changed yet.

## 5. Benchmark infrastructure (staged, not yet run)

GSM8K official test Parquet + MATH-500 official `test.jsonl` staged on
glm-box; `tools/benchmark_math.py` drives scalar/DFlash pairs on identical
samples (real tokenizer + chat formatting, 256-token cap), reporting prompt
throughput, decode throughput, acceptance, and exact parity. Deliberately no
answer grading. The alternating-math prompt overstates acceptance (repetition
loops); the campaign is the honest measurement.

## 6. Next steps (in order)

1. DFlash2 acceptance regression on the exact-prefix bridge: diff capture
   feed + verify replay between pre-bridge and bridge builds for the same
   seed; suspect the drafter's layer-5/14/24/33/42 captures changed
   numerically under the exact-prefix path.
2. Run the GSM8K/MATH-500 campaign once acceptance is healthy — real-prompt
   acceptance histograms decide k4-vs-k7 and seq-vs-batch defaults.
3. Validate latent MLA beyond position 256 (long-context run + MSE/PPL/
   cosine A/B tooling — still missing).
4. CCT cross-layer prefetch integration (loader landed, tables generated via
   `tools/dump_cct.py`) — the ~6→8.5 GB/s serialization gap is the biggest
   remaining I/O lever on glm-box.
5. Re-measure the local-box pinned ceiling under the 14 GiB `.wslconfig`
   before any host-tier work there.
