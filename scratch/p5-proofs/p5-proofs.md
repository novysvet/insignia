# P5 — Bit-exactness proofs for the shared MLA split-tile reduction tree

Write-up for problem P5 of `audits/s6-open-problems.md` (256K unblock). All line
numbers refer to `src/glm53_ops.cu` at HEAD of `glm53-dflash2-4070ti-super`;
constants from `include/insignia_glm53.cuh` (`kMlaLatentDim = 512` :162,
`kMlaMaxContext = 262144` :161); tile size `kMlaDecodeTile = 512` (:775);
exact window `kMlaExactContext = 256` (:456). Compilation contract:
`build/glm53.sh:10` — `nvcc -arch=sm_89 -O3 --use_fast_math -lineinfo`.

Verdicts up front: **(a) holds conditionally — it is NOT a bitwise no-op as
literally stated; the unique deviation is canonicalization of acc elements
equal to −0, plus a load-bearing axiom that `__expf(+0)` is exactly `1.0f`;
the exact fix is a control-flow predicate on tile activity, not a value test.
(b) is proved by induction on the merge sequence; "same kernel ⇒ one
compilation" is what kills the FMA-contraction risk. (c) worst-case divergence
is ~7.2e-2 of the denominator at 256K and the 500-token evidence is vacuous
(T = 1 there), so the determinism law REQUIRES the shared-tree construction
for decode↔verify; the bound's legitimate role is quality certification of the
new prefill tree and sizing the one-time parity battery.**

## 0. The code under proof

Stage 1 — `mla_decode_latent_partial_kernel` (lines 1018–1096), one block per
(head, tile of 512 keys), fresh online-softmax state per tile:

```cuda
1051      float maximum = -3.402823466e38F;
1052      float denominator = 0.0f;
1053      float acc0 = 0.0f, acc1 = 0.0f;
...
1030      const int first = tile * kMlaDecodeTile;
1031      const int last = min(position, first + kMlaDecodeTile - 1);
...
1080          const float maximum_new = fmaxf(maximum, score);
1081          const float correction = expf(maximum - maximum_new);
1082          const float weight = expf(score - maximum_new);
1083          maximum = maximum_new;
1084          denominator = denominator * correction + weight;
1085          acc0 = fmaf(weight, k0, acc0 * correction);
1086          acc1 = fmaf(weight, k1, acc1 * correction);
```

Stage 2 — `mla_decode_latent_merge_kernel` (lines 1101–1134), one block per
head, sequential merge over tile partials:

```cuda
1110      float maximum = -3.402823466e38F;
1111      float denominator = 0.0f;
1112      float acc0 = 0.0f, acc1 = 0.0f;
1113      for (int tile = 0; tile < tiles; ++tile) {
1114          const float *src = base + size_t(tile) * (latent_dim + 2);
1115          const float tile_max = src[0];
1116          const float tile_denominator = src[1];
1117          const float maximum_new = fmaxf(maximum, tile_max);
1118          const float correction = expf(maximum - maximum_new);
1119          const float tile_correction = expf(tile_max - maximum_new);
1120          maximum = maximum_new;
1121          denominator = denominator * correction + tile_denominator * tile_correction;
1122          acc0 = fmaf(src[2 + element], tile_correction, acc0 * correction);
1123          acc1 = fmaf(src[2 + element + 256], tile_correction, acc1 * correction);
1124      }
1125      const float inverse = 1.0f / denominator;
```

Launcher (line 1301): `tiles = (position + kMlaDecodeTile) / kMlaDecodeTile` —
decode launches exactly the tiles covering keys 0..position.

The causal-skip mechanism that P5 lifts to tile granularity already exists
per-key in `mla_prefill_latent_kernel` (lines 1217–1228): the ENTIRE state
update sits inside
`if (slot < query_count && key <= position_base + query_base + slot) { ... }`
— a skipped key issues no update at all. (The FA2-style
`mla_flash2_prefill_kernel`, lines 535–617, is the two-pass, order-preserving
kernel for the ≤256 exact expanded-K/V window; it is untouched by P5 and its
header comment, lines 529–533, is the on-record precedent for the determinism
law.) The plan: one 8-row generalization of stage 1 + stage 2 serves decode
(query_count = 1), verify and prefill (query_count ≤ 8) with a fixed grid;
tiles beyond a row's causal limit write the neutral partial
(m = −FLT_MAX, den = +0, acc = +0).

## 0.1 Floating-point contract and axioms

- Round-to-nearest, ties-to-even everywhere; unit roundoff u = 2⁻²⁴ (each
  correctly-rounded FMUL/FADD/FFMA has relative error ≤ u, i.e. ≤ ½ ulp).
- `--use_fast_math` = `--ftz=true --prec-div=false --prec-sqrt=false
  --fmad=true` plus intrinsic remapping; in particular `expf` → `__expf`,
  which lowers to an FMUL by log2e followed by `ex2.approx.ftz.f32` (MUFU.EX2).
  FTZ-qualified FP ops flush subnormal operands/results to signed zero.
- IEEE facts used (all valid in CUDA FP32 RN): x − x = +0 exactly for every
  non-NaN x, including ±0; (+0)+y = y unless y = −0 (then +0); (−0)+y = y
  unless y = +0 (then +0); (−0)+(−0) = −0; y·1.0f = y bitwise for every
  finite y; fmaxf suppresses NaN (returns the non-NaN operand) and is exact;
  MUFU functions are deterministic given identical input bits on a fixed
  architecture; there are no atomics or order-nondeterministic reductions in
  the path (`warp_sum` is a fixed xor-butterfly, lines 11–16; the cross-warp
  sum is a serial 8-term loop by thread 0, lines 1072–1075).

**Axiom E (exp-zero).** `__expf(+0.0f) = 1.0f` bitwise on sm_89. This is true
of the hardware — 2⁰ is the interpolation anchor of the MUFU.EX2 table — and
is load-bearing for every online softmax in existence (every non-record key
multiplies by correction = expf(0)); but it is NOT implied by the documented
accuracy bound (max ulp error 2 + ⌊|1.16x|⌋ permits 2 ulp at x = 0). Every use
below is marked; §1.4 removes the dependency for neutral merges and §4 adds a
one-line startup probe to pin it for the rest.

**Precondition P (finite data).** All scores s = fl(warp-reduced q·k)·rsqrtf(256)
are finite. Justification: operands are FP8-E4M3 (≤448) times FP32 group
scales and BF16-derived `W_uk` entries, all O(1)–O(10²); hitting ±Inf needs
products ~3.4e38, thirty-plus orders of magnitude beyond reach, and any
NaN/Inf would fail the engine's digit-verified logit gates loudly. Under P:

- **Invariant I1:** every weight and every correction is `expf` of a
  nonpositive argument (score − m_new ≤ 0 because m_new = fmaxf(m, score) ≥
  score; likewise for corrections and tile corrections), hence lies in [0, 1].
  No `expf` overflow can occur, and NaN would require (+Inf) − (+Inf) or
  0·Inf, both excluded by P. In particular the running max m is finite.
- **Invariant I2:** den ≥ +0 and den ≠ −0 always (init +0; updates combine
  nonnegative values; sums/products of nonnegative signed zeros give +0).
- **Invariant I3:** no subnormal is ever resident in the state or the partials
  buffer: every partial component is produced by an FTZ-qualified FFMA/FMUL/
  FADD in registers (lines 1084–1086), so subnormal results are flushed at
  production and the merge never reads a subnormal operand. Residual flush
  perturbations are absolutely ≤ 2⁻¹²⁶ ≪ u·den* (den* ≥ 1, see Lemma D) —
  folded into constants.

Notation: den* = Σ_j exp(s_j − m*) with m* the global max (den* ≥ 1); S⁰ =
(−FLT_MAX, +0, 0⃗) is the initial state of both stage 1 (per tile) and
stage 2; N = number of keys, T = N/512 tiles.

## 1. (a) Neutral-partial no-op

### 1.1 The literal fmaf question

**Lemma 1 (exact characterization of fmaf(x, +0, y)).** The fused multiply-add
computes fl(fl(x·(+0)) + y); the product is exact with sign = sign(x) and
magnitude 0. Consulting the zero-sign rules:

```
fmaf(x, +0, y) = y bitwise   iff   (x < 0, any y)  or  (x ≥ 0 finite, y ≠ −0)
```

It FAILS in exactly three cases: (i) x = NaN → NaN·0 = NaN → NaN + y = NaN;
(ii) x = ±Inf → 0·Inf is an invalid operation → NaN; (iii) x ≥ 0 (including
+0 and positive finite) and y = −0 → +0. So the answer to the question as
posed is **no** — not for all x, and explicitly not for x = NaN.

In the merge (line 1122), x = src[2+element] is the incoming partial's acc
element, which the construction fixes to +0, so only case (iii) survives.

### 1.2 The merge on a neutral partial

Let the incoming state be (m, den, acc) with m finite (Invariant I1), and the
neutral partial (m_n, den_n, acc_n) = (−FLT_MAX, +0, +0). Walk lines 1117–1123:

1. `maximum_new = fmaxf(m, −FLT_MAX) = m`. −FLT_MAX is the most-negative
   finite FP32; fmaxf is exact; the only tie is m = −FLT_MAX, which returns
   that value. Bitwise, always. (m = NaN would return −FLT_MAX and destroy
   the NaN — unreachable under P.)
2. `correction = expf(m − m)`. By IEEE, m − m = +0 exactly for every non-NaN
   m (including m = ±0); +0 · log2e = +0 exactly; so correction = __expf(+0)
   = 1.0f **by Axiom E [Use 1]**.
3. `tile_correction = expf(−FLT_MAX − m) = +0`. The argument rounds to
   −FLT_MAX (|m| < ½ulp(FLT_MAX) ≈ 1e31) or overflows to −Inf; either way the
   result underflows and is flushed to +0. Note tile_correction ≥ +0 always —
   it is an expf of a nonpositive argument (I1).
4. `m' = m` — bitwise.
5. `den' = fl(fl(den·1.0f) + fl((+0)·tc)) = fl(den + +0) = den` bitwise,
   since den·1.0f is exact, (+0)·tc = +0 (tc ≥ 0), and +0 + den = den except
   den = −0, which Invariant I2 excludes. If ptxas contracts the line into
   `fmaf(tile_denominator, tile_correction, denominator*correction)`, Lemma 1
   with x = +0, y = den gives the same result. ✓
6. `acc'_e = fmaf(+0, tc, fl(acc_e·1.0f)) = fmaf(+0, tc, acc_e) = acc_e`
   bitwise for every acc_e ≠ −0, by Lemma 1 with x = +0 ≥ 0; acc_e·1.0f is
   exact for finite acc_e and no FTZ flush can occur (I3: no subnormals
   resident). **The unique deviation: acc_e = −0 maps to +0.**

**Theorem 1 (neutral no-op, conditional).** Under P (hence I1–I3) and Axiom E,
merging (−FLT_MAX, +0, +0) into any state (m, den, acc) with m finite returns
bitwise (m, den, acc), with one exception: acc elements equal to −0 are
canonicalized to +0. Merging neutral into the initial state S⁰ returns S⁰
bitwise. If den or acc elements are NaN (unreachable under P), the neutral
merge preserves them (NaN·1 = NaN; +0 + NaN = NaN); a NaN in m is not
preserved (fmaxf suppresses it) but is likewise unreachable.

### 1.3 Reachability of the −0 case

acc_e = −0 requires every per-key fused update over all active tiles to yield
−0: fmaf(w, k_e, −0) = −0 needs the exact product w·k_e = −0, i.e. per key
either (k_e = −0, any w ≥ +0) or (w = +0 flushed and k_e < 0, since
(+0)·(negative) = −0). Some tile's argmax key has w = 1 exactly (Axiom E),
forcing k_e = −0 there. So the construction needs an entire latent column
(one of 512 dims, one head) that is −0 (FP8 0x80 column) or negative-with-
all-flushed-weights across all N keys. Representable, unreachable from real
checkpoints, and annihilated by the first downstream multiply with a nonzero
operand — but the determinism law is bitwise, so it counts.

### 1.4 The exact construction that fixes it

**Value-sniffing is wrong.** `if (tile_max == −FLT_MAX) continue;` misfires on
a REAL tile whose every score is exactly −FLT_MAX: such a tile legitimately
carries den = 512, acc = Σk (all its weights are expf(0) = 1), and dropping it
changes the model. The predicate must be tile ACTIVITY — a function of tile
index and the row's causal limit only:

- per row r with causal limit p_r (its position): active tile count
  T_r = (p_r >> 9) + 1 (tile = 512 keys);
- stage 1 on a fixed grid of T_max tiles: if tile·512 > p_r for all rows of
  the block, write the neutral partial for every row and return (never enter
  the compute path); otherwise clamp per row, last_r = min(p_r, first + 511)
  (line 1031 with `position` generalized to the per-row limit), and guard
  each row's update with `if (key <= p_r)` — the line-1219 pattern verbatim;
- stage 2: `for (tile = 0; tile < T_r; ++tile)` — or T_max iterations under
  `if (tile < T_r)`; identical effect.

Inactive tiles then issue no merge at all: Theorem 1's residue (−0, and the
Axiom E dependency for neutrals) is bypassed and part (b) becomes purely
structural.

**Remark (why this neutral triple is forced).** Among partials that are
no-ops in every state INCLUDING S⁰, (−FLT_MAX, +0, +0) is essentially unique:
any den_n > 0 would corrupt states with m = −FLT_MAX (there tile_correction =
expf(+0) = 1, so den_n would be added in full); den_n = +0 with
m_n = −FLT_MAX contributes (+0)·tc = +0 in every state. Score-masking
(feeding score = −FLT_MAX instead of skipping the key) is also a per-key no-op
for states with m > −FLT_MAX (correction = 1, weight = +0) but FAILS in the
all-(−FLT_MAX) degeneracy, where a masked key gets weight = expf(0) = 1 and
leaks k into acc — one more reason the plan's SKIP (not mask) is correct.

## 2. (b) Causal-skip equivalence

### 2.1 Setup and determinism of the transitions

Fix a head, a row r, its causal limit p, and the shared latent key stream.
Define:

- **U** — the per-key state transition compiled from lines 1080–1086. U is a
  deterministic function of (state; key data): `warp_sum` is a fixed
  xor-butterfly (lines 11–16), the cross-warp reduction is a serial 8-term
  sum by thread 0 (1072–1075), the score reaches all threads through shared
  memory (ordering only, no arithmetic), and FMA/MUFU are deterministic per
  input bit pattern.
- **F(t)** — the stage-1 partial of tile t: U applied over keys
  first..min(p, first+511) in ascending order (1055) starting from the fresh
  state S⁰ (1051–1053). F(t) depends only on the tile's keys (the q·W_uk
  absorption, 1044–1049, reads only q and w_uk) — partials are independent.
- **M** — the merge transition compiled from lines 1117–1123.
- The merged trajectory: S_0 = S⁰ (1110–1112), S_{t+1} = M(S_t, P_t), tiles
  ascending (1113).

**Row-isolation lemma.** No component of row r's recurrence reads any other
row's values (per-slot qe 1162–1180, per-slot scores 1205–1215, per-slot
guarded update 1217–1228). A row's output bits depend only on (its query, its
p_r, the shared latent cache) — not on how many rows the launch carries, and
not on the grid's tile count beyond §1.4.

### 2.2 Theorem 2 (mode equivalence)

Let execution A be the decode-mode launch of the ONE shared kernel for this
row (query_count = 1, T_A = T_r active tiles) and execution B any other mode
(verify or prefill, query_count ≤ 8) of the SAME cubin with a fixed grid of
T_max ≥ T_r tiles and neutral writes (or §1.4 predicates) for inactive tiles.
Then A and B produce bitwise-identical merged states and outputs.

*Proof.* By row isolation it suffices to track row r.

Claim 1 (partials agree): for t < T_r, tile t is partially or fully active in
both executions, and stage 1 clamps the key range by the SAME per-row limit p
(line 1031 generalized per row), so the sequence of U-applications — keys
first..min(p, first+511), ascending — is identical; U is one compiled
function; identical inputs give identical bits. Hence P^A_t = P^B_t.

Claim 2: for t ≥ T_r, P^B_t is the neutral partial by construction (or absent
under §1.4).

Claim 3: merging a neutral partial is the identity on states (Theorem 1);
under §1.4 the iteration is skipped outright and the claim is vacuous.

Induction on t with invariant S^A_t = S^B_t (pad A's merge sequence with
identities for t ≥ T_r; T_A = T_r). Base: both initialize S⁰ (1110–1112).
Step: S_{t+1} = M(S_t, P_t) is the same compiled function applied to equal
bit vectors, and M is deterministic (§2.1), so S^A_{t+1} = S^B_{t+1}. At the
terminal states the common tail — `1.0f / denominator` (1125), scaling
(1127–1128), the ascending-c W_uv fmaf chain (1130–1132) — is the same
compiled function on equal bits. ∎

**Corollary (greedy-exactness of speculation).** Every verify position
executes row-wise the same update sequence as a scalar decode of that
position over the same latent cache (the store kernel, lines 777–808, is
shared and mode-independent), so the committed sequence equals plain greedy
decode — bit-for-bit, hence greedy-ID and top-10-logit identical.

### 2.3 Why "same kernel ⇒ same compilation" is load-bearing

- **FMA contraction.** The only FP expression in the recurrence without a
  pinned form is `denominator * correction + weight` (1084/1225) and its
  merge twin `denominator * correction + tile_denominator *
  tile_correction` (1121). Whether ptxas emits {FMUL, FADD} or contracts to
  FFMA is decided at compile time, per function, under `--fmad=true`
  (implied by `--use_fast_math`), as a function of the surrounding schedule
  and register pressure. In the current tree these lines live in three
  different kernels with different loop bodies and unroll factors — their
  contraction decisions are independent accidents of compilation, and
  bit-equality between decode and prefill would rest on an unverifiable
  coincidence. In the shared-tree construction there is ONE function, hence
  ONE instruction stream: the comparison "decode vs verify vs prefill"
  executes literally the same SASS, and contraction freedom is neutralized
  rather than assumed away. (The explicit `fmaf(...)` calls are already
  single fused ops and cannot be re-associated.)
- **Launch geometry.** Block/grid dimensions do not change the per-thread
  instruction stream of a `__launch_bounds__(256)` kernel (no
  occupancy-adaptive code paths), so a T_max grid cannot alter bits versus a
  T_r grid — and §1.4 makes the extra tiles vacuous regardless.
- **Frozen binary.** ptxas output is fixed at build time; both target boxes
  are sm_89 and run the embedded SASS (no PTX JIT). The two build scripts
  differ only in host-side `-march/-mtune` (`glm53-gen.sh` adds
  raptorlake), which cannot affect device FP. What remains is operational,
  not mathematical: same nvcc version + flags for anything that must be
  bit-compared (treat toolchain changes as parity-gate events), and forbid
  mode-specific fast paths (e.g. special-casing T = 1 to skip the merge
  loop) — they would silently fork the tree again.
- **Reduction determinism.** No atomics, no cross-block reductions, fixed
  butterfly order, serial warp-partial order, `__syncthreads` used only for
  ordering. Same input bits ⇒ same output bits, run to run.

## 3. (c) The merge reassociation bound

### 3.1 What differs

The single-pass tree (today's `mla_prefill_latent_kernel`: one running max
over all keys, per-key rescale, 1221–1227) and the tiled tree (per-tile
running max in stage 1 + one merge rescale per tile) compute the same
mathematical quantity — the corrections telescope in exact arithmetic — but
with different rounding sequences and different `__expf` arguments (the
running max at key j is tile-local in one tree, global in the other). We
bound their divergence.

### 3.2 Model and lemmas

RN, u = 2⁻²⁴; each FMUL/FADD/FFMA contributes ≤ u relative (½ ulp). `__expf`
contributes relative error ε_e(x) ≤ (2 + 1.16|x|)·u for x ≤ 0 (documented
bound); arguments below −103.28 produce flushed +0, so WLOG the score
excursion R := m* − min-relevant-max satisfies |x| ≤ R ≤ 103.3 (real MLA
latent scores are O(10); R = 100 used for the tables).

**Lemma D (domination — no blow-up).** In either tree, at any step the
partial denominator rescaled to the terminal max, V_j = D_j·exp(m_j − m*),
equals Σ_{k≤j} exp(s_k − m*) ≤ den* in exact arithmetic (and ≤ (1+
accumulated error)·den* in FP, since corrections ≤ 1 and weights ≤ 1). Every
rounding at a step is ≤ u times the partial then present, whose
terminal-rescaled value is ≤ den*, so each surviving error contribution is
≤ u·den*. The bound therefore counts STEPS, not intermediate magnitudes: the
naive fear that a partial can exceed den* by e^R (early keys under a small
running max) is real, but the later tiny corrections shrink the error
together with the value, and the surviving error is the rescaled one.

**Lemma T (excursion telescoping).** The |x|-dependent part of `__expf`'s
error, applied along a chain of corrections, telescopes: Σ_events
1.16·|Δm|·u ≤ 1.16·R·u, because the max increments along a chain sum to at
most the total excursion R. Likewise the rounding of each argument
(m − m_new) contributes ≤ u·|Δm| and telescopes the same way. This is what
keeps the bound at ~(count)·u instead of the vacuous count·ε_e (which at
N = 262144 would exceed 1).

**Lemma W (weights).** All weights are nonnegative, so their individual
`__expf` relative errors θ_j combine as |Σ w_j θ_j| ≤ ε_e·Σ w_j =
ε_e·den*, independent of N — one ε_e per tree, not per key.

### 3.3 Theorem 3 (bound)

Count roundings on the den chain: per key, den·correction + weight = 2 ops
(mul + add, or 1 if contracted — 2 is conservative); per merge step, 3 ops
(den·c, t_den·t_c, add). Per acc element the count is ≤ the same. With
γ_n := n·u/(1 − n·u) (valid while (2N + 3T)·u < 1, i.e. up to N ≈ 4·10⁶ ≫
262144), the two trees' denominators satisfy

    |den₁ − den₂| / den*  ≤  2·γ_{2N+3T}  +  2·κ(T, R),
    κ(T, R) := ε_e + u·R·(2.16 + 1.16·T)   [ + flush terms ≤ 3·2⁻¹²⁶·(N+T) — negligible ]

and each normalized output element satisfies
|o₁ − o₂| ≤ (same bound)·Σ_j w̄_j|k_{j,e}|/den* ≤ (same bound)·max_j|k_{j,e}|
(elementwise RELATIVE error of acc is unbounded under cancellation — acc
elements are signed sums; the absolute-vs-max|k| form is the honest one).

| context | N | T | worst Δden/den* | RMS estimate |
|---|---|---|---|---|
| 500 tokens | 500 | 1 | 0 (identical tree) | 0 |
| 8K | 8192 | 16 | 2.2e-3 | ~4e-6 |
| 32K | 32768 | 64 | 8.8e-3 | ~9e-6 |
| 256K | 262144 | 512 | 7.2e-2 | ~2.5e-5 |

RMS model: roundings treated as independent ±½ulp on partials of size ≤ den*;
σ ≈ u·√(N/3) per tree, ×√2 for the pair. (Example check, 256K:
2·γ_{525824} = 6.47e-2; 2κ = 2·(ε_e + u·103·(2.16 + 1.16·512)) =
2·(7.3e-6 + 3.66e-3) = 7.3e-3; total 7.2e-2.)

### 3.4 Reading the bound against the evidence

1. **The 500-token zero-flip observation is vacuous for this question.** At
   N ≤ 500 ≤ 512 the launcher (line 1301) yields T = 1: there is no inter-
   tile merge, and the "different summation tree" does not exist — the tiled
   path IS the single-pass sequence. The merge reassociation has never been
   A/B-tested. Moreover, unless the standard parity prompts exceed 512
   tokens, the 30/40/100/240-token sequence battery also never crosses a
   tile boundary; the only merges ever exercised are the ≤8192 latent-cache
   oracle runs.
2. **The cos 0.9994–0.9997 figures measure a different quantity.** They are
   distances-from-model of DATA approximations (FP8 dense/latent), not parity
   between two reduction orders. They establish that the engine's routing
   tolerated percent-scale logit-vector perturbations on the tested prompts
   (0 flips at 500 tokens) — i.e. typical top-k logit gaps are large relative
   to that noise class. They say nothing about two different trees producing
   digit-identical top-10 logits, which is a bitwise requirement.
3. **Propagation.** The bound is per attention-output element per layer;
   between it and the logits sit 45 layers of mHC mixing, RMS collapse, and —
   decisively — 42 discrete top-8-of-288 routers with measured entropy
   4.5–5.2 bits. The determinism law exists precisely because
   mathematically-equivalent reassociations were observed to flip routes
   immediately (the canonical-order MoE probe). A worst case of 7.2e-2 and
   an RMS of ~2.5e-5 per element at 262144 keys, times 11 MLA layers, times
   262144 tokens of in-context state accumulation, cannot support zero
   greedy flips: even a per-decision flip probability as low as the
   battery's own resolution (1/240, the strongest un-flip any existing test
   could certify) compounds over 262144 tokens to
   1 − (1 − 1/240)^262144 ≈ 1. No finite ULP bound composes to bitwise
   digit-identity through a discrete routing cascade.

### 3.5 Verdict (decisive)

**The determinism law requires the shared-tree construction, and the shared
tree is sufficient.**

- **decode ↔ verify: bitwise equality is required** (verify's committed
  tokens must equal plain greedy, and the KV/latent trajectory they induce
  must be the same bits), and the shared tree delivers it STRUCTURALLY
  (Theorems 1 + 2: same cubin + same per-row update sequence + no-op or
  skipped neutrals) at zero recurring validation cost. A ULP-bound argument
  cannot substitute: the gate is a bitwise specification, and the bound is
  both too large (worst case 7.2e-2 at 256K) and of the wrong type
  (probabilistic vs structural) to certify it.
- **prefill: the hard requirement is weaker** — prefill sets the initial
  state for both arms of any decode/verify A/B, so its tree choice does not
  perturb mode-consistency; the bound certifies the new tree's QUALITY
  (≤ 8.8e-3 worst, ~1e-5 realistic at 32K — inside the noise class already
  accepted for the FP8 dense cache at cos 0.9994 ⇒ ~3.5e-2 vector-relative),
  so the shared tree is expected to clear the existing parity battery.
  Nevertheless route prefill through the SAME kernel: it costs nothing (the
  kernel is mode-agnostic; only query_count and per-row limits change),
  collapses three kernels into one proof obligation, and removes the current
  prefill kernel's O(position) sequential scan — the P5 goal.

## 4. Obligations checklist (the construction, precisely)

1. Stage 2 per row: iterate `tile < T_r`, T_r = (p_r >> 9) + 1 — control
   flow keyed on tile index, NEVER a value test on tile_max. Keep lines
   1117–1123 verbatim, including the `fmaf` spelling.
2. Stage 1, 8-row generalization: keep the q·W_uk absorption ascending in j
   with `fmaf` (1044–1049) so qe bits are launch-invariant; guard the WHOLE
   per-key update block with `key <= p_r` (1219–1228 pattern); clamp
   `last_r = min(p_r, first + 511)`; fully-inactive tiles write
   (−FLT_MAX, +0, +0) per row and return before the compute path.
3. One kernel, one cubin for decode/verify/prefill (query_count = 1/≤8/≤8);
   one per-row-generalized merge kernel. No mode-specific fast paths.
4. Axiom E probe at startup (device-side `__expf(0.0f) == 1.0f`, fail
   loudly). It is already load-bearing for every online softmax in the
   engine (correction = 1 at every non-record key); pin it once.
5. Toolchain pinning: same nvcc + device flags for every box that runs
   parity (device flags already aligned in build/glm53*.sh); any toolchain
   change is a parity-gate event.
6. Extend the sequence battery across tile boundaries (≥600, ≥2048, ≥8192
   tokens): the current 30/40/100/240 battery is single-tile unless the
   standard prompts exceed 512 tokens.
7. Mirror the tiled recurrence in the NumPy reference (tile size 512,
   per-key online update, ascending merge) so the oracle gate compares
   against the engine's actual tree — however the references already handle
   the __expf-vs-exp discrepancy today, the ORDER structure is what must be
   mirrored.
