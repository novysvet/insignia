# w3: working-tree diff verification — 2026-08-25

Scope: every hunk of the uncommitted diff (14 files, +159/-32) against the 7 wave-1
bug claims in `audits/synthesis.md`, plus completeness sweeps (dispatch, launchers,
bats, seam chain). Read-only audit; no builds run. HEAD = `92e1028`.

## Per-claim verdicts

### 1. INSIG4 scale dtype (F16 on disk, read as bf16) — **FIXED**

Every site that touches an INSIG4 scale now reads `__half`:

| site | evidence |
|---|---|
| `src/mxfp4_i4.cu:11-13` | `i4_scale()` = `__half2float(*reinterpret_cast<const __half*>(...))`; sole scale-read helper for all 4 kernels in the file (`mxfp4_gemv_v2_i4_kernel`:54, `mxfp4_gemv2_q8_i4_kernel`:142, `mxfp4_gemv_ab2_q8_i4_kernel`:225, `mxfp4_get_row_i4_kernel`:249) |
| `src/gemm.cu:329` | `mxfp4_gemm_mlx_i4_kernel` scale load = `__half` (single read site in kernel) |
| `src/prefill.cu:30` | `embed_gather_i4_kernel` = `__half` |
| headers | `include/insignia_layout.cuh:67-71`, `include/insignia_prefill.cuh:7` — all i4 prototypes take `const uint16_t*` |
| `src/test_i4.cu:35-37` | synthetic scales built via `__float2half` (F16) — consistent |
| dump tools | `dump_i4_chunk.cu`, `dump_i4_seams.cu` call the i4 kernels (no direct scale reads) — consistent |
| benches | `bench_mxfp4_mlx.cu`, `bench_gemm.cu` have no i4 paths at all (synthetic u8 only) — N/A |

Remaining `__bfloat162float` casts near scales: `src/fp8.cu:33,77` — those are the
Qwen3.8-FP8 checkpoint's BF16 `weight_scale_inv` (different format, correct as-is);
`qk_norm`/`conv1d`/`dt_bias` casts are real-BF16 tensors (correct).

Disk side verified end-to-end: `tools/quantize_insig4.py:129` emits
`scales.astype(np.float16)` with dtype string `'F16'`; `tools/index_safetensors.py:8`
maps `F16->3`; `include/insignia_model.hpp:10` `DType::f16=3`. `Qwen35Weights::matrix`
(`src/qwen35.cu:7-9`) keys dispatch on `s.dtype==DType::f16` and validates scale
bytes (`rows*(cols/64)*2` == `rows*(cols/32)` u8 bytes, so both formats pass the size
check). All dtypes (F32/BF16/F16/U8/U32/I8) handled by the indexer; unknown dtypes raise.

### 2. RoPE shared-mem race — **FIXED** (both kernels); same *pattern* survives elsewhere

- `src/ops.cu:9` (`qk_norm_rope`, decode path): `__shared__ float nsc` added; scale
  written by warp0/lane0 before the 2nd `__syncthreads()`, read by all 256 threads
  after it, and **never written again** — `mem[0..63]` roped staging can no longer
  clobber the scale. Sync placement: both `__syncthreads()` unconditional, staging
  write (tid<64) between sync2 and sync3, staging read after sync3 — race-free.
- `src/prefill.cu:54-83` (`qk_norm_rope_batch_kernel`, prefill path): identical fix
  with comment. Same trace, race-free.

Other kernels with reduction-then-reuse of the same shared slot (pattern-similar,
NOT fixed by this diff):

| kernel | read | overwrite | between |
|---|---|---|---|
| `src/attention.cu:7` `gqa_decode_kernel` | `mx=red[0]` right after sync | `red[0]=den` (warp0/lane0) after exp loop | no barrier — exp loop is only `tokens/256` iters (as short as ~1 iter at pos 0) |
| `src/prefill.cu:131→138` `gqa_prefill_kernel` | `mx=red[0]` | `red[0]=den` | no barrier — `tokens/8` iters |
| `src/nll.cu:27→31` and new copy `src/generate.cu:40-53` `row_logp_kernel` | `mx=red[0]` | `red[0]=sum` | no barrier — but ~vocab/256 ≈ 970 exp iterations |

All are formally racy (unsynchronized read/write of `red[0]` between two barriers).
Practical risk: the read is the first post-barrier instruction and the write comes a
full loop later, so the window is far wider than the RoPE bug's (~600-cycle
`__ldg(pos_dev)`-gated window that demonstrably fired). gqa_decode at position 0-1 is
the tightest (single-iteration loop). Worth the same one-slot fix while the full-attn
parity hunt is live; softmax max corruption there produces exactly the flaky-cos
signature being chased. The final `red[0]` reads (post-`1/den` / post-sum) in all
three kernels are properly after barriers and never rewritten — safe.

Checked and safe (writes fenced before reuse or never reused): `rms_kernel`
(ops.cu:5), `rms_bf` (qwen_kernels.cu:5), `argmax_kernel` (:20), `deltanet_decode`
(deltanet.cu:7-12, `sq[0]/sk[0]` written once, `delta[tid]` same-thread), 
`deltanet_prefill` (prefill.cu:233-257, iteration-scoped with barrier at loop tail),
`partial` reductions in mxfp4.cu (:22,:59,:269).

### 3. MTP draft embed missing i4 branch — **FIXED** (+ bonus)

`src/decode.cu:136-139`: `mtp_layer()` embed now branches
`embed_gather_i4(..., (const uint16_t*)m.scales.data, ...)` vs `embed_gather` (u8).
Bonus in the same hunk: `mtp.fc` (`src/decode.cu:148-153`) gained an i4 branch
(`mxfp4_gemv_v2_i4`, rows=4096/cols=8192 hardcoded — arg order matches
`bf16_gemv(w,x,y,rows,cols)` and the quantized [4096,8192] shape; fine for 9B, will
silently break on any other geometry since dims are literals, not `fc.rows/cols`).

Full dispatch enumeration (engine paths) — **complete**:

- `linear()` decode.cu:31 i4 ✓, `linear2()` :32 ✓, `linear_batch()` :33-41 ✓
  (i4 path skips the bf16 staging/pad, correct — `mxfp4_gemm_mlx_i4` reads x directly
  with a `tid<T` guard).
- `prefill_chunk_device`: embed :46 ✓; q/k/v/o via linear2/linear_batch ✓;
  in_proj_qkv/z via linear2/linear_batch ✓; in_proj_a/b pair path :68
  (`mxfp4_gemv_ab2_q8_i4`) ✓ and per-token path :72-73 ✓; mlp gate/up/down :86-88 ✓;
  lm_head all three paths :95-100 ✓.
- `mtp_layer`: embed ✓, fc ✓, q/k/v/o + mlp + lm_head via `linear()` ✓.
- `forward_body`/`forward_token`: embed via `Qwen35Weights::embed_dev`
  (`src/qwen35.cu:13`, i4 branch added in this diff) ✓; lm_head via `linear()` ✓.
- `nll.cu:79` ✓, `generate.cu run_nll:76-78` ✓ (both lm_head GEMMs branch).
- Graph/spec paths reuse `prefill_chunk_device(T=2)`/`mtp_layer` ✓.

Stale tools with no i4 branch (debug/bench only): `dump_attention.cu:10` (`mat()`
lambda, u8), `dump_multistep.cu:31-35` (`mat()` for lm_head — see hazard H2),
`test_checkpoint.cu:10` (u8), `dump_multistep` layer seams are fine (they go through
`d.layer()`). None of these can silently corrupt the engine itself; `matrix()` throws
only on scale-dtype surprises for the u8 path when scales are f16... actually it
returns `i4=true` and the u8-cast call misreads — silent garbage (see H2/H6).

### 4. nll.cu lm_head missing i4 — **FIXED**

`src/nll.cu:78-81`: `if (lh.insig4) mxfp4_gemm_mlx_i4(... uint16_t ...)` else u8
`mxfp4_gemm_mlx`. The end-to-end check now actually exercises i4 because
`build/nll.bat` adds `src\mxfp4_i4.cu` to the link (was omitted — the audit's "never
ran" finding). Chunk/target indexing verified: all-positions logits from `x.pf_n`
(written for all T rows at decode.cu:92), targets `tokens[done+1..]`, NLL math
(`tgt - mx - log(sumexp)`) correct.

### 5. Silent early-returns → throws — **FIXED at every flagged site; long tail remains**

Now throwing (`std::runtime_error` with rows/cols): `src/mxfp4.cu:35,71,147,215,370`;
`src/mxfp4_i4.cu:70,151`; `src/gemm.cu:78,182,293,354`; `src/fp8.cu:53,97,182`
(already threw). Required includes added (`<stdexcept>`, `<string>`, `<cuda_fp16.h>`)
in all three modified TUs. `prefill.cu` launchers never had guards (nothing to fix).

Remaining silent / absent guards (none can fire with current 9B-hardcoded callers;
all would silently corrupt on a 27B port):

| launcher | state |
|---|---|
| `mxfp4.cu:242` `quantize_q8_groups` | still silent `return;` |
| `mxfp4.cu:272` `mxfp4_gemv_dp4a`, `:278` `get_row_mlx`, `:401` `quantize_x8`, `:461` `gemv2_q8g`, `:510` `gemv_q8g`, `:577` `gemv_ab2_q8g` | no guard at all (bench/aux paths) |
| `mxfp4.cu:668` `mxfp4_gemv_ab2_q8` | **no guard** — audit's "hard-requires cols==4096" still unenforced; hot decode path (linear2 pair) |
| `mxfp4_i4.cu:237` `mxfp4_gemv_ab2_q8_i4` | **no guard** — same, hot path (decode.cu:68) |
| `mxfp4_i4.cu:252` `mxfp4_get_row_i4` | no guard (cols<=0 → 0-block launch, y unwritten) |
| `mxfp4.cu:148` `gemv_v2` cols%1024!=0 → mlx fallback | intentional redirect, fine |

### 6. Quantizer emitted fp16 bytes labeled BF16 — **FIXED** (RNE math verified)

`tools/quantize_insig4.py:67-71`:
```python
bits = a.astype(np.float32).view(np.uint32).copy()
bits += np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
return (bits >> np.uint32(16)).astype('<u2').tobytes()
```
Textbook RNE-to-bf16. Edge cases checked:
- carry into exponent (0x3F7FFFFF → 0x3F80 = 1.0) correct;
- max finite 0x7F7FFFFF → 0x7F80 = +Inf (ties-to-even, matches `cvt.rn.bf16.f32`);
- ±0 preserved (0x80000000 → 0x8000); tiny negatives round to -0, not +0;
- quiet NaN survives; **signaling NaN → Inf** and **negative NaN with full payload
  wraps uint32 → +0.0** (0xFFFFFFFF + 0x8000 mod 2^32) — never applies to real weights;
- `<u2` little-endian ✓ (matches device reads).
Coverage: the else-branch (`quantize_insig4.py:134`) catches every non-quantized
tensor — conv1d [8192,1,4] (row-major flatten matches the kernel's `w + c*4 + i`),
all norms / q_norm / k_norm / dt_bias / mtp pre_fc norms / mtp.norm (1D, engine reads
bf16 ✓). `in_proj_a/b`, all `*proj*`, `fc.weight`, `lm_head`, `embed_tokens` are
INSIG4-quantized by the predicate (`:126`) — they never were bf16 emissions, and the
engine's a/b i4 paths match. Embed stays INSIG4 for 9B ✓ (predicate
`'embed_tokens' in name`; consumed by `embed_gather_i4` / `mxfp4_get_row_i4`).

`tools/fix_insig4_bf16.py` (read): one-shot stream repair of a v1/v2 checkpoint —
reads every BF16-labeled tensor as fp16 (`<f2`), re-emits real bf16 RNE, promotes
`.A_log` to F32, rewrites header offsets. Still *works*, but **no longer needed**: 
`qwen35-insig4-good.safetensors` (18:06, after the fixed quantizer) supersedes any
repaired file with single-rounded values (the fix path double-rounds f32→f16→bf16).

Checkpoint timeline (mtimes, `build/`):
| file | mtime | verdict |
|---|---|---|
| qwen35-insig4.safetensors (5.62GiB) | 08-25 02:50 | v1, fp16-as-BF16 — **BAD, delete** |
| qwen35-insig4-v2.safetensors | 08-25 05:56 | still pre-fix (fix script 07:39) — **BAD, delete** |
| qwen35-insig4-text.safetensors | 08-25 07:52 | fix-script repair of -v2 (real bf16, A_log F32, double-rounded) — usable |
| qwen35-insig4.insignia-index | 08-25 13:46 | → points at **-text** (verified embedded path) — current default i4 index |
| quantize_insig4.py | 08-25 15:18 | this diff |
| qwen35-insig4-good.safetensors + .insignia-index | 08-25 18:06 | fresh requant, single-rounded — **best** |

Size forensics corroborate: -text is 1542 B larger than -v2 ≈ 24 delta layers ×
(A_log f32 128B - bf16 64B = 64B) = 1536 B + header — i.e. -v2 predated even the
A_log-F32 format, and -good lands at exactly -text's size.

### 7. A_log emitted F32 — **FIXED**

`tools/quantize_insig4.py:131-132`: `elif eng.endswith('.A_log'): emit_raw(eng,
w.astype('<f4').tobytes(), 'F32', ...)`. Engine reads A_log as `const float*`
(`deltanet_parameters`, `deltanet_params_batch` — prefill.cu:208/210 `ar[h] +
bf16(dt_bias)` with separate bf16 dt ✓). Reference scripts parse F32 via dtmap ✓.
Size-delta evidence above confirms -v2/-text boundary.

### 8. Other hunks

- `include/insignia_decode.hpp`: `prefill_chunk_seam` decl + private seam-overload
  decl — match definitions (decode.cu:42-43,113-117). DecodeWorkspace members used by
  the new tools (`pf_n`, `pf_tokens`) are public ✓.
- `build/dump-layers.bat`, `build/nll.bat`: both now compile the full closure
  (mxfp4_i4 + prefill + gemm) — link-correct, and dump-layers.exe still runs the u8
  MXFP4 checkpoint (dispatch handles both) ✓.
- `src/generate.cu` `run_nll` (new): logic mirrors nll.cu (T fixed at 64, no chunk
  arg; `argc>3` guard ✓; lm_head i4 branch ✓; `Qwen35Shape::vocab` used for the
  reduce ✓). Includes `insignia_layout.cuh`/`insignia_prefill.cuh` for the kernel
  decls ✓. Note it *duplicates* nll.cu's row_logp_kernel — including its latent
  `red[0]` reuse pattern (hazard H3).
- `tools/reference_all_layers.py` q/k-norm change: algebraically a no-op
  (`sqrt(mean+eps)*128` ≡ `sqrt(sum+eps)*sqrt(128)`; k's old `sqrt(mean)*sqrt(128)`
  ≡ `sqrt(sum)`) — differs only in eps placement (negligible). Cosmetic cleanup, not
  a bug fix, not harmful.

## build/*.bat link-closure verdicts

Requirement for any bat compiling `src/decode.cu`: model_file.cpp, storage.cu,
mxfp4.cu, **mxfp4_i4.cu**, qwen35.cu, qwen_kernels.cu, ops.cu, attention.cu,
deltanet.cu, **prefill.cu**, **gemm.cu**. `qwen35.cu` alone additionally needs
mxfp4.cu + **mxfp4_i4.cu**.

| bat | verdict |
|---|---|
| dump-layers.bat (diff) | OK — full closure added |
| nll.bat (diff) | OK — mxfp4_i4.cu added |
| dump-i4-chunk / dump-i4-seams / dump-layers-i4 / dump-multistep / dump-pf | OK — full closure |
| generate.bat | OK — full closure |
| test-pair.bat / test-prefill.bat | OK — full closure |
| test-i4.bat | OK — test_i4.cu only needs mxfp4_i4+gemm+prefill (self-checked) |
| test-fp8.bat | OK — fp8.cu + test_fp8.cu closed |
| bench-gemm.bat / bench-gemm-blocked.bat | OK (still byte-identical twins — dedupe backlog item unchanged) |
| bench-mxfp4.bat | OK (has gemm.cu) |
| test-checkpoint.bat / test-model.bat / test-argmax / test-attention / test-deltanet / test-ops | OK (don't compile qwen35/decode) |
| **test-qwen35.bat** | **BROKEN — NEW** (qwen35.cu now calls `mxfp4_get_row_i4`; no mxfp4_i4.cu; linked fine at HEAD) |
| **oldgen.bat** | **BROKEN — NEW** (decode.cu+generate.cu i4 refs; no mxfp4_i4.cu) |
| **shim-only.bat** | **BROKEN — NEW** (same) |
| **test-pair-chain.bat** | **BROKEN — NEW** (same) |
| dump-attention.bat, dump-layer0.bat, dump-layer3.bat | BROKEN — pre-existing (decode.cu already needed prefill/gemm at HEAD; also now i4) |
| test-full-model.bat, test-generate.bat, test-layer.bat, test-mtp.bat, generate-ids.bat | BROKEN — pre-existing (same cause) |
| smoke.bat | step 3 (bench-mxfp4-mlx.exe) BROKEN — pre-existing (bench_mxfp4_mlx.cu calls `mxfp4_gemm_v21`/`gemm_v2` in gemm.cu; smoke omits gemm.cu; true already at HEAD 92e1028); steps 1-2 OK |
| mk.bat / tiny.bat / sani.bat / shim-only (as shim builder) | not model links — N/A |

The diff updated only 2 of the ~10 affected bats. The 4 newly-broken ones are the
regression directly caused by adding i4 references to decode.cu/generate.cu/qwen35.cu.

## Seam chain (prefill_chunk_seam → dump_pf.cu → reference scripts)

Contract verified consistent:
- `prefill_chunk_seam(tokens,T,seam,user)` (decode.cu:113) → H2D copy, seam-overload,
  final sync. Seam fires at end of every layer body with `(l, x_.pf_x, T, user)` **after
  `cudaStreamSynchronize`** (decode.cu:90), so dump_pf's blocking `cudaMemcpy` inside
  the callback is safe; the host stack `tok_dev[64]` outlives the async copy because
  the wrapper synchronizes before returning.
- `dump_pf.cu` writes 32 seams of [T,4096] f32 (seam-major) + a 33rd block = `x.pf_n`
  (final model.norm output, computed inside prefill_chunk_device). `reference_pf_i4.py:74`
  reshapes `[33, T, 4096]` — exact match, seam l ↔ layer l, seam 32 ↔ final norm.
- argv/build: dump_pf expects argc==5 (index, tokens, T, out) — matches
  `rundll.py` conventions (dllshim `dll_run` → wmain, argv[0] dummy). dump-pf.bat
  compiles the full closure incl. dllshim.cu ✓.
- Reference math checked against kernels: rope64 pairs (i,i+32)/signs match
  `qk_norm_rope*`; 1/16 attention scale matches `.0625f`; kvh=h>>2 (9B) matches;
  F16-scale dequant (`repeat(s,64)`) matches i4_scale; BF16 tensors parsed as real
  bf16 (correct for -text/-good checkpoints only).
- `reference_all_layers_i4.py` ↔ dump-layers-i4.dll ([32,4096] single-token, via
  `d.layer()` — i4 dispatch correct) ✓.
- `reference_multistep_i4.py` ↔ dump-multistep.dll ([steps+1,33,4096] via
  `d.layer()` + own conv-state roll + KV cache) — layout matches, **but** see H2.

## Remaining hazards (ranked)

1. **H1 — `reference_pf_i4.py` conv history missing for T>1.** Its `delta()` uses only
   the current-token conv tap (`qkv*cw[:,3]`), while the engine's `conv_prefill_kernel`
   uses in-chunk tokens t-3..t-1. Token 0 (zero state) matches; tokens t>=1 diverge in
   every DeltaNet layer. Since the whole point of dump-pf is T>=4 chunks (first
   attention layer is l=3), the per-token seam cosines for t>0 will be depressed by a
   reference bug, not an engine bug — it will poison the parity hunt. The multistep
   script rolls conv state correctly; port that.
2. **H2 — `dump_multistep.cu:33` lm_head still u8-only.** On i4 checkpoints the
   argmax feeds `next` for the +1 generated step → native extra step embeds a garbage
   token while `reference_multistep_i4.py` uses its own correct argmax → guaranteed
   false mismatch on the last step. Needs the same `if (z.insig4)` branch nll.cu got.
3. **H3 — `red[0]` reuse race in `gqa_decode` (attention.cu:7) and `gqa_prefill`
   (prefill.cu:131/138)** (+ row_logp copies). Formally racy, narrow practical window,
   but gqa_decode at pos 0-1 has a ~single-iteration loop between read and write, and
   attention is where the residual parity bug lives. Cheap to fix with a dedicated slot.
4. **H4 — 4 newly link-broken bats** (test-qwen35, oldgen, shim-only, test-pair-chain):
   add `src\mxfp4_i4.cu`. 8 more are pre-existing broken (missing prefill.cu/gemm.cu).
5. **H5 — `mxfp4_gemv_ab2_q8_i4` / `mxfp4_gemv_ab2_q8` have no dim guard at all**
   (audit's cols==4096 hard-requirement unenforced) — fine while 9B dims are hardcoded
   at call sites, silent OOB on a 27B port (cols=5120 also breaks the kernel's
   128-thread staging assumption).
6. **H6 — stale u8-only debug tools** (`dump_attention.cu`, `test_checkpoint.cu`):
   pointed at an i4 index they misread scales silently (matrix() returns i4=true,
   caller casts to u8). Harmless if kept on the MXFP4 index; worth an assert.
7. **H7 — mtp.fc i4 dims hardcoded** (4096,8192 literals instead of `fc.rows/cols`) —
   consistent today, footgun for 27B.
8. **H8 — checkpoint hygiene**: -insig4.safetensors and -insig4-v2.safetensors are
   format-broken (fp16-as-BF16); both still on disk. Default index
   `qwen35-insig4.insignia-index` points at -text (double-rounded) rather than -good
   (single-rounded); re-point or regenerate the index for the best file.
9. Minor: run_nll in generate.cu duplicates nll.cu (~90 lines) incl. chunk=64 only;
   bench-gemm/bench-gemm-blocked twins still not deduped; `dump-layers.bat` output
   (`layers-native.f32`) is the u8 path while `reference_all_layers.py` (not _i4)
   expects that checkpoint — consistent, but easy to mix up with the _i4 pair.

## TL;DR (10 lines)

1. All 7 claimed bug fixes are genuinely in the diff and are correct: i4 scales read
   `__half` everywhere; both RoPE kernels got the dedicated `nsc` slot; MTP embed/fc,
   nll lm_head, generate-nll lm_head all have i4 branches; flagged launchers throw;
   quantizer now emits true RNE bf16 and F32 A_log (math verified, incl. edge cases).
2. Dispatch completeness: every engine-reachable matrix branches on `insig4` — no
   reachable gap. Stale no-branch tools are debug-only (dump_attention, test_checkpoint,
   dump_multistep's lm_head).
3. Bug 5 is only partially done: `ab2_q8`/`ab2_q8_i4` (hot pair path) still have *no*
   guard, and ~7 bench/aux launchers remain silent.
4. The RoPE race class survives in gqa_decode/gqa_prefill/row_logp (`red[0]` reuse
   between barriers) — formally racy, narrow window, but sits exactly on the flaky
   full-attn parity path; cheap to fix.
5. Seam chain (prefill_chunk_seam → dump_pf.cu → reference_pf_i4.py) contract matches
   ([33,T,4096], layer index, sync-before-callback, rundll argv), build path closed.
6. But reference_pf_i4.py omits in-chunk conv history for t>0 → false parity failures
   for every DeltaNet seam beyond token 0. Highest-priority fix for the parity hunt.
7. dump_multistep.cu's lm_head is still u8-only → its generated step argmax is garbage
   on i4, mismatching reference_multistep_i4.py's correct argmax (H2).
8. Bats: dump-layers.bat and nll.bat were fixed, but 4 more bats are newly
   link-broken by the i4 references (test-qwen35, oldgen, shim-only, test-pair-chain —
   all missing mxfp4_i4.cu); 8 others + smoke's bench step were already broken.
9. Checkpoints: -insig4 and -insig4-v2 are format-broken leftovers (fp16-as-BF16);
   -text (fixed via fix_insig4_bf16.py, double-rounded) is what the default index
   points at; -good (18:06, new quantizer) is the best file — re-point the index.
10. reference_all_layers.py hunk is an algebraic no-op cleanup, not a fix; everything
    else in the diff does what it claims.
