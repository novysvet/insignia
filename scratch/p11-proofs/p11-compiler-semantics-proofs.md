# P11 — KDA smem-transplant bit-exactness: compiler-semantics proofs

Scope: the fused chunk-recurrence design (FP32 state in 64 KiB dynamic smem, in-kernel
T-loop, one launch per KDA layer per chunk) vs the current per-token kernels, under
`nvcc -O3 --use_fast_math -arch=sm_89`. Companion context: `scratch/kda-fusion/`.

## 0. The code under proof (exact locations)

Per-token decode path (`Runner::kda`, `src/glm53_generate.cu:2474-2503`):

1. `linear(q_proj/k_proj/v_proj)` (batched GEMV, outside the fusion),
2. `kda_conv_silu3` — one launch, all three streams (`src/glm53_ops.cu:416-454`),
3. `kda_gate_kernel` (`src/glm53_generate.cu:338-355`),
4. `kda_decode_kernel<128>` (`src/glm53_ops.cu:319-383`), grid = 64 heads,
   128 threads, state RMW in **global** (`kda_states_`, `DeviceBuffer<float>`),
5. `kda_output_kernel` (`src/glm53_generate.cu:357-383`),
6. `linear(o_proj)` (outside the fusion).

Prefill per-token launches (`Runner::kda_multi`, `src/glm53_generate.cu:3264-3314`),
chunk `kMaxChunk = 64` (`src/glm53_generate.cu:2084`): 3·T × `kda_conv_silu` (one loop
per stream, lines 3272-3286) + T × `kda_gate_kernel` (3299) + T × `kda_decode`
(3301-3304) + T × `kda_output_kernel` (3310) = **6T launches per KDA layer per chunk** —
exactly the "~6T" P11 wants removed. The fused kernel replaces exactly these 6T; the
projections stay batched outside (they are not per-token launches).

The state size for one head is 128×128 FP32 = 65,536 B = **64 KiB**, hence one head per
block with dynamic smem (`cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, 65536)`).

Existing bit-carry evidence *inside the current engine*: `kda_decode_kernel` already
round-trips `norms`, `sq`, `sk`, `decay`, `delta` through **shared memory** between
`__syncthreads()` (glm53_ops.cu:339-373), and already round-trips the recurrent state
through **global memory** across launches; `rollback_kda` / `archive_kda_rows`
(glm53_generate.cu:3166-3172, 2941-2958) round-trip full state/history through device
copies. The fusion only *lengthens* these round-trips, never changes their nature.

---

## (a) FP32 round-trip through shared/global memory: PROVEN

**Claim.** For IEEE-754 binary32 values, `ld ∘ st = id` on bit patterns, for
`st.shared.f32`/`ld.shared.f32`, `st.global.f32`/`ld.global.f32` (including local-memory
spill traffic and `cudaMemcpy`), across kernel boundaries, on sm_89. There is no
extended-precision carry.

**Proof.**

1. *The ISA defines ld/st as pure data movement.* In the PTX ISA, `ld` and `st` move data
   between registers and a state space; the type qualifier (`.f32` vs `.u32` vs `.b32`)
   selects size and register class, not a transformation. No rounding, canonicalization,
   denormal flush, or NaN quieting is defined for loads/stores — those are properties of
   *arithmetic* instructions only. Memory (shared, global, local) is untyped byte storage;
   `st.shared.f32 [%a], %f1; ld.shared.f32 %f2, [%a];` yields `%f2` bit-identical to
   `%f1` for every 32-bit pattern including sNaN payloads, denormals, ±0, ±∞.
   Corollary: SASS `STS/LDG/STG/LDL/STL/MOV` and the shuffle
   (`__shfl_xor_sync`, used by `warp_sum`, glm53_ops.cu:11-16) are the same class of
   bit-moves.

2. *Registers hold exactly binary32; no x87-style extended precision exists.* The
   x87 hazard — 80-bit registers holding a 64-bit-significand intermediate whose value
   depends on whether it stayed in a register or was stored to memory — requires a
   register file wider than the stored type plus an implicit widening convention.
   NVIDIA's floating-point whitepaper (docs.nvidia.com/cuda/floating-point/) states the
   contrast explicitly: on x87 "results of a computation ... can depend on whether an
   intermediate result was allocated to a register or stored to memory", whereas on the
   GPU the precision of every operation "is encoded in the instruction" (FLT_EVAL_METHOD
   = 0 semantics). Device `float` expressions never evaluate in a wider format; any
   f32↔f64 movement is an explicit `cvt` instruction. sm_89 SASS FP32 registers are
   32-bit containers. Therefore the value a register holds, the value it stores, and the
   value a reload produces are the same 32 bits. (One archival phrasing of the same
   point: unlike SSE-vs-x87, there is no double-rounding and no register-pressure-dependent
   precision for FP32 on the GPU.)

3. *Across kernel boundaries.* A value that leaves a kernel through global memory is a
   bit pattern in DRAM; the next kernel's `ld.global.f32` returns those bits. The PTX
   memory consistency model additionally guarantees naturally-aligned 32-bit accesses are
   single-copy atomic (no torn reads), and `__syncthreads()` gives causality for the
   intra-block smem staging — the fused kernel's per-token `sq/sk/decay/delta` staging is
   the same pattern the per-token kernel already executes between its barriers. ∎

**The one caveat class (and its boundaries).** A smem slot has no rounding provenance of
its own; bits can only differ between "register-resident" and "stored" versions if the
*producing instruction* differs between the two compilations — e.g. one compilation
emits FFMA where the other emits MUL+ADD for the same source expression. That is a
part-(b) failure mode, not a storage failure, and it is decided before the store ever
happens. Sub-cases to keep straight:

- **FTZ is a property of the producing arithmetic instruction.** `--use_fast_math`
  implies `--ftz=true --prec-div=false --prec-sqrt=false --fmad=true` (nvcc
  documentation; CUDA Programming Guide §5.5). FTZ flushing of denormal *inputs/outputs*
  happens inside FADD/FMUL/FFMA (and approximate div/sqrt). It never happens at an
  ld/st/mov/shuffle. A denormal that enters the pipeline from memory (e.g.
  `bf16_to_float`, which is a bit shift, glm53_ops.cu:33-35) survives every round-trip.
- The compiler may *eliminate* a redundant smem round-trip and keep the value in a
  register — permitted by as-if and harmless: the register already holds the exact bits.
- Register spilling (`STL/LDL`) is bitwise; performance-only.
- Explicit format conversions (FP32→BF16/FP8) around a store would break identity by
  construction; the KDA state/history buffers are pure `DeviceBuffer<float>`
  (`kda_states_`, `conv_history_`, glm53_generate.cu:1976-1977) — no conversion exists
  on the transplanted path.

**Verdict (a): PROVEN at the ISA level.** No test needed beyond the end-to-end gate.

---

## (b) FMA-contraction invariance across the two compilations: PROVEN CONDITIONALLY

**Claim.** If the fused kernel transplants every expression tree *verbatim* (same
operators, same operand order, same intrinsic spellings, same helper functions, same
unroll pragmas) into the same compilation unit, compiled with the same flags, then nvcc's
contraction decisions are identical for both kernels.

**Argument.**

1. *What the standard licenses.* The C++ as-if rule ([intro.execution]: a conforming
   implementation may perform any transformation that preserves observable behavior)
   explains why **loop restructuring alone cannot change arithmetic**: replacing T
   kernel launches with an in-kernel T-loop changes control flow, launch boundaries, and
   instruction *scheduling* — none of which are arithmetic. Note that contraction
   (`a*b+c` → FFMA) is *not* justified by as-if — it changes rounded results and is a
   separate floating-point license (the C-family FP_CONTRACT allowance; C++ defines no
   standard pragma). On nvcc this license is the documented `-fmad` switch: "-fmad=true
   ... enables the contraction of floating-point multiplies and adds/subtracts into
   floating-point multiply-add operations (FMAD, FFMA, DFMA)" (CUDA Programming Guide
   §5.5; `--use_fast_math` implies `--fmad=true`).

2. *Where the decision is made.* Contraction is a compile-time, expression-local
   decision. Two stages can apply it — the nvcc frontend (NVVM/LLVM pattern-matching
   `mul.f32` + `add.f32` def-use pairs into `fma.rn.f32` in PTX) and ptxas at SASS level
   (NVIDIA staff, njuffa: "both the compiler frontend and backend can apply the
   contraction FMUL + FADD → FMAD/FFMA"; "operations with default rounding mode are
   eligible"; "operations with explicit rounding mode are not eligible"). Crucially,
   eligibility is determined by the *operator spelling*: `__fmul_rn`/`__fadd_rn` lower to
   `mul.rn.f32`/`add.rn.f32` and can never be merged; `fmaf()` lowers to one
   `fma.rn.f32`; plain `*`/`+` carry default rounding and are the contractible ones.

3. *The invariance chain.* Identical source trees → identical NVVM def-use graphs →
   identical frontend contraction decisions → identical PTX for the tree → ptxas
   deterministic on identical input → identical SASS arithmetic. Loop structure lives
   *outside* the tree; frontend contraction pattern-matching is not cost-modeled or
   profile-guided, so surrounding code cannot flip it for a given def-use subgraph.

**Residual risks (must be respected / tested).**

- **R1 — ptxas-level contraction of plain `*`/`+` trees can in principle key off
  scheduling context**, which the different loop structures *do* change. In this
  codebase the contractible-by-spelling trees inside the fusion zone are few; the audit:
  - All hot arithmetic in `kda_decode_kernel` (glm53_ops.cu:370, 379-380) and both conv
    kernels (glm53_ops.cu:408, 448) is already `fmaf` — **pinned by spelling,
    context-free, immune to R1**.
  - One genuinely contractible plain pattern: `rsqrtf(square * (1.0f / float(head_dim))
    + 1.0e-5f)` (kda_output_kernel, glm53_generate.cu:377) — mul-then-add; under
    fmad=true it is exactly the license's target. It will contract or not *identically*
    only if transplanted token-for-token.
  - Two heuristic-pinned shapes: `warp_sum(q*q)` / `warp_sum(value*value)`
    (glm53_ops.cu:335-338, glm53_generate.cu:367-368) — the mul feeds both the local
    reduction add and the shuffle (two uses), which blocks contraction under the
    single-use rule; this is stable but heuristic, so the fused kernel must inline the
    *same* `warp_sum` helper over the *same* shape.
  - Everything else near the trees (`(v - memory) * beta[head]`,
    `current * taps[3]`, `q * q_scale`, `__expf(a_log) * (forget + dt_bias)`) has no
    mul whose result feeds a plain add — the adds are either inside an operand, consumed
    by an explicit intrinsic, or separated by stores — not contractible in either
    compilation.
- **R2 — fast-math flag drift between TUs** would silently break the premise
  (transcendental selection `expf`→`__expf`→MUFU.EX2, `div.approx.f32`, FTZ all come
  from the same flag set). Compile both kernels in one TU with one command line, and
  keep the source spelling of transcendentals verbatim (`expf` in
  `kda_decode_kernel:362`, `__expf` in the gate/output kernels — same final instruction
  under this flag set, but the spelling must survive flag drift).
- **R3 — helper-inlining shape changes** (rewriting `warp_sum` or re-associating any
  expression, e.g. folding `q_scale * rsqrtf(head_dim)` into one constant) are
  refactorings, not transplants; forbidden by the invariant.

**On "just add `__fmul_rn`/`__fadd_rn` to be safe": REJECTED.** Inserting them changes
the expression tree (forcing separately-rounded MUL/ADD where the current compilation
may have contracted to FFMA), so it *guarantees* bit-divergence from the shipped
per-token kernels — it breaks both verbatim-transplant status and the engine's
determinism law against the validated behavior. The safe alternative for any spot
deemed too heuristic is: keep the source identical and verify the SASS
(`cuobjdump -sass`, diff the FMA/MUL/MUFU sequence) plus the Tier-A state walk below.

**Verdict (b): PROVEN for the intrinsic-pinned trees (everything in the recurrence and
conv); PROVEN-CONDITIONAL for the plain-operator trees under the verbatim + same-TU +
same-flags invariant; R1's tail risk (ptxas scheduling influence) is not provable from
documentation and is closed empirically by the SASS diff + state bit-compare.**

---

## (c) Tap order in the register-ring rewrite: PROVEN-BY-CONSTRUCTION (pattern-following)

**Actual current order — quoted** (`kda_conv_silu3_kernel`, glm53_ops.cu:443-451; the
prefill twin `kda_conv_silu_kernel` is identical at 403-411):

```cuda
        float value = current * taps[3];
#pragma unroll
        for (int lag = 1; lag <= 3; ++lag) {
            if (position >= lag) {
                const int slot = (position - lag) % 3;
                value = fmaf(segment_history[slot * count + within], taps[3 - lag], value);
            }
        }
        segment_history[(position % 3) * count + within] = current;
```

So the accumulator is seeded with the **newest** tap (`current * taps[3]`, a bare MUL)
and lags are folded in oldest-last, each an explicit `fmaf`:

    value = fmaf(h3, t0, fmaf(h2, t1, fmaf(h1, t2, current*t3)))   // nesting, not evaluation order of fmaf args

Addition order (left-to-right accumulation): current-first, then lag 1, lag 2, lag 3.
This is *not* the NumPy oracle's order (see §Empirical gate), so it cannot be "fixed" to
match anything — it is part of the effective model.

**The pattern that guarantees preservation** (per channel; the fused kernel's thread owns
its channel across the whole T-loop, so the ring is thread-local and program order
replaces inter-launch ordering):

```cuda
// loaded once at chunk entry (position_base - 1/-2/-3), guarded like the originals:
float h1 = history_at(t0 - 1), h2 = history_at(t0 - 2), h3 = history_at(t0 - 3);
...
#pragma unroll
for (int t = 0; t < T; ++t) {
    const float current = projection[t * count + within];
    float value = current * taps[3];        // (1) newest first — bare MUL, seeds accumulator
    if (age_ok_1) value = fmaf(h1, taps[2], value);   // (2) lag 1
    if (age_ok_2) value = fmaf(h2, taps[1], value);   // (3) lag 2
    if (age_ok_3) value = fmaf(h3, taps[0], value);   // (4) lag 3 — OLDEST LAST
    h3 = h2; h2 = h1; h1 = current;         // shift strictly AFTER value is complete
    projection[t * count + within] = value / (1.0f + expf(-value));
}
```

Guarantees: the same textual left-to-right `fmaf` chain (newest-seeded, oldest-added-
last), same guards, same final division. For `position_base > 0` the three age guards
are uniformly true and hoist cleanly; the *first* chunk (`position_base == 0`) must keep
per-position guards — positions 0/1/2 have 1/2/3 taps.

**Anti-patterns that silently break bit-identity:**

1. **Oldest-first accumulation** — seeding with `h3 * taps[0]` or looping
   `for (lag = 3; lag >= 1; --lag)`. Reassociates the sum
   (h3·t0 + h1·t2 … order swapped) → different rounding at every position →
   state divergence via the determinism law. This is exactly the NumPy oracle's order
   (reference_glm53_numpy.py:142-146, oldest tap first from a zero seed) — matching it
   would *change* the engine.
2. **Iterating ring *slots* instead of *ages*.** Ages live at `(p - a) % 3`; slot order
   visits ages in a rotated order once `p % 3` wraps. Worked examples: p = 3 → slots
   {2,1,0} hold ages {1,2,3} (accidentally oldest-last-reversed, still wrong order:
   visiting slots 0,1,2 = ages 3,2,1); p = 5 → slots {1,0,2} = ages {1,2,3}, so
   visiting slots 0,1,2 = ages 2,1,3. Every residue class gives a wrong order — and it
   *changes per position*, so divergence starts at the first wrap and compounds.
3. **Zero-filling missing taps and executing the fmaf unconditionally.** The original
   *skips* the fmaf for `position < lag`. `fmaf(0.0f, t, value)` equals `value` for all
   finite t **except** it flips `-0.0` to `+0.0` (0 + (−0) = +0 in RN). Under FTZ,
   `current * taps[3]` underflowing to −0.0 is a real occurrence, silu preserves the
   sign of zero, and the sign of zero in q/k/v can propagate into the FP32 state. Keep
   the guard; do not substitute neutral arithmetic for control flow.
4. **Rotating tap indices** (`taps[(3 - lag + something) % 4]`) or "normalize" the
   nesting to right-fold the fmafs — any change to the accumulated *sequence* of rounded
   operations.

**Cross-thread note.** In the fused (decode-geometry) kernel each thread computes its
own channel's conv for every t before the recurrence consumes it; the ring never crosses
threads. The per-token `sq/sk/decay/delta` smem staging between conv/recurrence phases
is covered by (a). Verdict (c): **provable by construction** — it reduces to "emit the
same token sequence of explicit-fmaf source statements"; anything else is a counterexample
generator.

---

## Empirical gate — settles every unprovable residual

What is *not* provable from documentation: the actual frontend/ptxas output for the
plain-operator trees (R1/R2), and any transcription slip. The decisive comparator is
**engine-vs-engine** (per-token path vs fused kernel on identical inputs). The NumPy
oracle is **not** a bitwise oracle: `check()` (tools/reference_glm53_numpy.py:80-92)
compares rel/cos/max-abs per seam (tolerance), and its KDA math differs from the
engine's *by construction* — oldest-first zero-seeded conv accumulation
(reference_glm53_numpy.py:142-146), `np.sum` pairwise reductions, `np.exp` vs the
engine's fast-math `expf`, and separate mul/add vs the engine's `fmaf` chains
(165-170 vs glm53_ops.cu:368-381). Its role is the unchanged *numerical band*, not bits.

**Tier A — bit-exact, decisive (toy model `/var/lib/insignia/glm53-tiny`: 4 KDA layers,
4 heads × 64 head_dim, hidden 256; per-layer FP32 state 4·64·64 = 16,384 floats = 64 KiB;
conv history 9·width):**

1. Run the same 64-token prompt through (arm 1) current `kda_multi` per-token path and
   (arm 2) fused kernel — same binary, same flags (`-O3 --use_fast_math -arch=sm_89`),
   same weights stream, greedy decode, deterministic single stream.
2. **State granularity: bit-compare (uint32 equality / memcmp, first-divergence report
   as (layer, position, element)) the FP32 recurrent state at EVERY position** t = 0..63
   — 65 snapshots (initial + 64) × 4 KDA layers × 16,384 floats ≈ 16 MiB of dumps — plus
   (i) conv history after each position, (ii) per-token recurrence output (`core_`
   pre-o_norm), (iii) per-token post-`kda_output_kernel` rows. Vehicle: the existing
   snapshot machinery (`kda_snap_`/`conv_snap_` at glm53_generate.cu:3680-3685,
   `archive_kda_rows` 2941-2958) extended with a per-position D2H dump hook behind an
   `INSIGNIA_GLM53_*` env knob, per repo convention; dumps land outside `/tmp` (WSL
   wipe) e.g. `/var/lib/insignia/p11/`.
3. Repeat for a **second chunk at position_base = 64** (exercises ring wrap at every
   residue class and the full-lag path) and for **position_base = 0** (exercises the
   partial-tap guards of positions 0-2 — the zero-sign hazard of anti-pattern 3).
4. End-to-end: greedy IDs digit-identical AND top-10 logits digit-identical on the
   standard prompts plus the 30/40/100/240-token sequence checks (the repo's determinism
   gate).
5. **SASS audit (one-time):** `cuobjdump -sass` both kernels; diff the FMA/MUL/ADD/MUFU
   sequence of each transplanted tree (expect: identical FFMA chains; the
   `rsqrtf(square * inv + 1e-5f)` tree contracted identically; `warp_sum(q*q)`
   uncontracted identically). This converts R1 from "trust" to "checked".

**Tier B — independent numerical band (not bitwise):** the NumPy oracle must show the
same seam rel/cos/max and top-8 logit digits as the recorded per-token baseline. If
Tier A passes bit-for-bit, Tier B cannot move (identical outputs); Tier B exists to
catch wiring errors where the fused path feeds some seams and not others.

**Decision rule.** Tier A bit-equality at every position for both chunks + token/logit
parity ⇒ the fusion is exact; any single differing (layer, position, element) with
identical inputs falsifies the transplant and localizes the slip (conv order vs
contraction vs staging — the dump points at which stage first diverges: conv-history
mismatch ⇒ (c); core mismatch with clean history ⇒ (b)/(recurrence); core clean but
proj mismatch ⇒ output-kernel tree).

---

## Verdict summary

| Sub-claim | Status |
|---|---|
| (a) FP32 smem/global round-trip bit-identity, no extended precision | **PROVEN** (PTX/SASS ISA: ld/st are bit moves; GPU registers hold binary32 only; FTZ belongs to producing arithmetic, not the slot) |
| (b) Contraction invariance for verbatim trees | **PROVEN** for all `fmaf`-pinned trees (recurrence + conv: the entire hot path); **conditional** (same TU, same flags, token-for-token copy incl. `warp_sum` shape and the one contractible tree at glm53_generate.cu:377) for plain-operator trees; residual ptxas-scheduling risk closed by SASS diff + Tier A. Adding `__fmul_rn`/`__fadd_rn` would break verbatim status and is rejected |
| (c) Tap order in register ring | **PROVEN by construction** given the quoted pattern (newest-seeded bare MUL, fmaf chain lag 1→3, shift after, guards kept for positions < 3); three named anti-patterns falsify |
| Overall P11 claim | Sound *iff* the transplant is literal; the only unprovable components are compiler-output facts, settled by the 64-token × every-position state bit-compare + SASS diff on the toy model |
