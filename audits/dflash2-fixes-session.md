# DFlash2 root-cause fixes audit (2026-08-28, session 3 continued)

Scope: the debugging arc that took DFlash2 from "all rounds reject" (session 3
end state) to "greedy-exact, ~5 accepted tokens/round on realistic prompts",
plus the decode-side cache/verify optimizations measured on the local box
before the glm-box migration. Local 4070 SUPER, 6.6 GiB tier unless noted.
Backfills the gap between `audits/dflash2-session.md` and
`audits/seqverify-session.md`.

## 1. The four real bugs (found via the one-round dump replay red loop)

Red loop: replay one dumped block against `tools/dflash2_oracle.py`. Initial:
layer-0 cosine 0.664869, CUDA activations exploding to 4.04e6 (oracle 9.08e4).

1. **Paired FP8 API was batch-1 only; DFlash2 passed 8 rows** for block K/V
   and gate/up → rows 1–7 uninitialized. Fix: batched paired-FP8 path
   (`fp8_tc_gemv2` family), bit-exact for all 8 rows. Kept independently of
   the rest.
2. **Dump serializer stride bug** (diagnostic only): `capture_` is layer-major
   with a 32-token stride; the dump copied `count×5` rows token-major, so the
   oracle received one real capture + four garbage rows instead of layers
   5/14/24/33/42.
3. **`df_gather_kernel` column split**: the 20480-wide `[c0..c4]` vector was
   split with an if/else that omitted the upper halves of captures 0/1, wrote
   only half of captures 3/4, and used a wrong channel offset — thousands of
   uninitialized columns fed the shared FC projection.
4. **Offline FP8 cache builder slice bug** (the big one): `fc.weight`
   [4096,20480] is row-major; `fc.a`/`fc.b` were quantized as two contiguous
   80 MiB slabs, but the column halves are strided per row, so both halves
   interleaved wrong rows/columns. Fix: strided slice in
   `tools/quantize_dflash2.py`, regenerate the 1.07 GiB cache. **This is why
   `INSIGNIA_GLM53_DFLASH2_FP8` must point at `glm53-dflash2-fp8-fixed`.**

After 3+4: committed context K/V parity 0.038/−0.017 → 0.999837/0.999241;
layer-0 cosine 0.664 → 0.999477; the 19M-outlier collapse. Second round
predicts the target's first token exactly.

## 2. Acceptance recovery

- Repetition-loop seed: 1.00 → 1.62 accepted/round (3/8 empty). BF16 replay
  of two rounds matched FP8 acceptance → precision was not the ceiling.
- **Realistic prompt: 5.00 accepted/round**, 12 tokens in 3 rounds — the
  paper's reported range. The alternating-math prompt is the pathological
  case; do not tune policy on it alone.

## 3. Verification correctness (near-tie flip at token 20)

Symptom: block-4 and block-7 both flip a 0.0046-logit tie (`13` vs `448`) vs
scalar. Scalar replay proven bit-stable across runs first. Two real bugs:

- **`moe_multi` accumulated experts in union order**, not each token's
  router top-k order. Fix: ordered accumulation inside verification only
  (preserving the exact fmaf sequence); prompt prefill unchanged.
- **`archive_kda_rows` wrote batched rows contiguously into a
  token-interleaved archive** — every partial-acceptance replay restored
  overlapping garbage. Full and empty rounds hid it; first partial round
  poisoned later logits. Fix: scatter each token into its true stride.

Result: exact 30/30 scalar IDs at every block size; acceptance bimodal
(`0:1 4:8`). Block sweep: k3 705.3, **k4 628.2**, k5 784.0, k7 765.7 ms/tok —
k4 wins; the k7 verifier's expert union outgrows its accepted-token yield on
a 6.6 GiB tier.

## 4. Decode-side optimizations (all parity-exact, kept unless noted)

- **Empty-round short-circuit**: `d1 != truth0` is known before verification
  (truth0 carried from the previous exact pass); those rounds skip the ~3.5 s
  verify entirely. 30-token decode 1503.6 → 1011.1 ms/tok (32.8%).
- **Verify cache admission = first 8 experts per layer**: hits 0.3% → 5.7%,
  1109.2 → 985.1 ms/tok. "Cache the last verified token" anchor REVERTED
  (hits 4.4%, 996.8 ms/tok); retained-token position stays a sweepable knob.
- 6.6 GiB tier = 488 records as per-layer quota 11 (vs global LRU).
- Rejected on measurement: pinned-host three-way expert tier on the local
  box (collapses the 8-way O_DIRECT queue, ~3.7% slower despite 11% host
  hits and 4.69 GiB NVMe avoided).

Session-end state (local box): DFlash2 k4 628.2 ms/tok vs 667.5 scalar at the
same tier — first time speculative beat plain decode. The striping work that
followed (dual-SSD `ALT_SHARD_DIR`, 843.8 → 401.4 ms/tok on the math prompt)
and the adaptive seq/batch verify restructure are in
`audits/seqverify-session.md`; the glm-box campaign that superseded both is
in `audits/mla-latent-session.md`.
