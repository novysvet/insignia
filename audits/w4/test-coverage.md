# w4 test-coverage — Insignia test inventory, coverage matrix, gaps, and minimum safety net

Date 2026-08-25. Read-only audit; the only file written is this one. Sources read end-to-end:
all `src/test_*.cu|.cpp`, `tools/mk.py`, all `build/test-*.bat`, `src/smoke.cu`, `src/streaming.cu`
(smoke main), `src/model_file.cpp`, `src/fp8.cu` (guards), `src/attention.cu`, `src/decode.cu`
(kernel dispatch), `include/*.cuh/.hpp` surfaces, `tools/reference*.py`, `tools/nll_compare.py`.
No builds; only prebuilt binaries were run (all cheap, none read >1 GB except nothing — the
6-GiB-budget whole-model tests were skipped by rule).

---

## 1. Test inventory

Invocation key: `IDX = build\qwen35.insignia-index` (INSIDX01 → external HF-cache
`models--sleepyeldrazi--Qwen3.5-9B-MXFP4-MTP\...\model.safetensors`, 6.03 GB, 699 tensors).
DLL targets run as `python tools/rundll.py build/<name>.dll [args]`; exe targets run directly
(mirrored by `build/<name>.bat`, which also rebuilds).

| Test | Binary (state) | Covers | Shapes | Tolerance / gate | How invoked | Result this session |
|---|---|---|---|---|---|---|
| smoke | build/insignia-smoke.exe (old) | CUDA present, sm_89 check, trivial kernel | 1 block | rc-gated (device+arch) | `build\insignia-smoke.exe` | **PASS** (<0.1 s) |
| test-mxfp4 | build/test-mxfp4.exe | `mxfp4_gemv` (MxFp4Block/E8M0 path) + 1000-iter timing | 257x4096 | max_rel<2e-4, rc-gated | `build\test-mxfp4.exe` | **PASS** max_rel=0, 49 GiB/s (~1 s) |
| test-ops | build/test-ops.exe | `rmsnorm_zero_centered` (GPU) only | 3x4096 | max_abs<2e-6, rc-gated | `build\test-ops.exe` | **PASS** 3.6e-7 (<1 s) |
| test-attention | build/test-attention.exe (STALE, see §4) | `gqa_decode` vs double ref | T=257,H=16,KV=4,D=256 | max_abs<2e-5, rc-gated | `build\test-attention.exe` | **PASS** 2.6e-9 — but binary predates current API (source no longer compiles) |
| test-deltanet | build/test-deltanet.exe | `deltanet_decode` single step vs double ref + 1000-iter timing | H=32,K=V=128,kh=16 | max_rel<3e-3, rc-gated | `build\test-deltanet.exe` | **PASS** max_rel=3.3e-5, 0.009 ms (~1 s) |
| test-argmax | build/test-argmax.dll | `argmax_logits` vs `argmax_fast` (two-stage atomic), ties, -0.0, 200 randomized rounds (implicit nondeterminism check) | n=248320 (real vocab) | exact-value agreement, rc-gated | `python tools/rundll.py build/test-argmax.dll` | **PASS** (~1 s) |
| test-i4 | build/test-i4.dll | INSIG4 kernels vs host f64: `gemv_v2_i4`, `gemm_mlx_i4` (T=3), `get_row_i4` (exact), `gemv2_q8_i4`, `gemm_v21_i4` T=33 (sampled + cross-kernel vs mlx_i4), `gemm_ab_i4` vs v21(32 rows) | 8192x4096 | **PRINT ONLY — always rc=0** (only gemv has a `bad` counter, unused for exit) | `python tools/rundll.py build/test-i4.dll` | rc=0; cos 1.0/0.999998–0.999985; per-el max_rel up to 0.755 on near-zero dots (1.7 s) |
| test-fp8 | build/test-fp8.dll | `fp8_gemv`, `fp8_gemv2`, `fp8_gemm` T=3 + T=33 tile-boundary + T=65 throw, host e4m3/bf16 quantizers | 10240x5120 (=27B in_proj_qkv) | **PRINT ONLY — always rc=0** (T=65 prints "NO (BUG)" but exits 0) | `python tools/rundll.py build/test-fp8.dll` | rc=0; cos 1.0 / 1.0 / 0.9999987 / 0.9999987; throw=yes (5.2 s) |
| test-cpu | build/test-cpu.exe (+ stale dup build/test_cpu.exe) | CPU tier (27B RAM path): e4m3+bf16 exhaustive decoders, fp8_gemv/gemv2 (3 shapes) + st==mt bit-exact, bf16_gemv, rmsnorm(±zc), gated_rmsnorm, silu/sigmoid_mul, conv1d+silu+state, deltanet params/step (h48/k3 2-step chained, h32/k2), qk_norm_rope pos 0/1234, split_q_gate, store_kv f32/bf16, gqa f32/bf16 t∈{2048,77}; bench mode = pool scaling + 27B layer shapes | real 27B shapes (10240x5120 etc.) | GATED: rel/cos/abs triple (1e-4/0.999999/1e-5 GEMV; 1e-5 norms; 1e-3 deltanet/gqa), rc=1 on failure | `build\test-cpu.exe test` (or `all`) | **PASS 40/40 checks** (~5 s) |
| test-model | build/test-model.exe | ModelFile INSIDX01 mmap: 699-tensor count, qkv weight/scales sizes, first bytes | n/a | rc-gated: tensors==699 exactly | `build\test-model.exe build\qwen35.insignia-index` | **PASS** (0.1 s) |
| test-checkpoint | build/test-checkpoint.exe | TieredStorage acquire/release, 32 MiB budget → LRU eviction/reload, `mxfp4_gemv_mlx` on real weights (checksum only, NO numeric ref) | 8192x4096 qkv | weak gate (isfinite && maxv>0) | `build\test-checkpoint.exe build\qwen35.insignia-index` | **PASS** (0.3 s) |
| test-qwen35 | build/test-qwen35.exe | Qwen35Weights matrix acquire (i4+mlx), `mxfp4_get_row_i4` embed, release; device bytes report | embed 1 row; q_proj 8192x4096 | rc-gated rows/cols + mx>0 | `build\test-qwen35.exe build\qwen35.insignia-index` | **PASS** (0.6 s, 532 MiB prefetch) |
| test-layer | build/test-layer.exe | 768 MiB budget: real layer-0 DeltaNet + layer-3 attention decode, warm/cold timing | full 4096 hidden | weak gate (finite, max>0) | `build\test-layer.exe build\qwen35.insignia-index` | **PASS** warm 1.41 ms (0.7 s) |
| test-pair | build/test-pair.dll | whole-model spec pair path: `prefill_chunk([760,6511])` row0/row1 argmax vs known greedy (2614/314) + sequential comparison | full 9B model | **PRINT ONLY — rc=0 always** | `python tools/rundll.py build/test-pair.dll build\qwen35.insignia-index` | **NOT RUN** (6 GiB budget ⇒ reads whole 6 GB checkpoint) |
| test-pair-chain | build/test-pair-chain.dll | chains greedy pairs through commit/rollback; every step must accept; localizes state corruption; also single-token chain sanity | 7-token prompt, 17-token greedy chain | prints DIVERGED but **rc=0 always** | `python tools/rundll.py build/test-pair-chain.dll build\qwen35.insignia-index` | **NOT RUN** (>1 GB, same reason) |
| test-prefill | build/test-prefill.dll/.exe | prefill_chunk warm/cold timing, 64-token chunks | n tokens arg | timing print; rc=0 | `python tools/rundll.py build/test-prefill.dll build\qwen35.insignia-index <n>` | **BROKEN RUN LINE** — mk.py `run=[IDX]` gives argc=2 ⇒ immediate rc=2. NOT RUN otherwise (>1 GB) |
| test-full-model | build/test-full-model.exe | all 32 layers x2, projected tok/s, residency | full 9B, 6 GiB budget | weak gate (finite, max>0) | `build\test-full-model.exe build\qwen35.insignia-index` | **NOT RUN** (>1 GB) |
| test-generate | build/test-generate.exe | 3 greedy `decode_token` steps + timing | full 9B, 6 GiB budget | gate: token>=0 only | `build\test-generate.exe build\qwen35.insignia-index` | **NOT RUN** (>1 GB) |
| test-mtp | build/test-mtp.exe | `forward_token` + `logits_argmax` + `mtp_draft` cold/warm | full 9B, 6 GiB budget | gate: draft>=0 only (any valid token passes) | `build\test-mtp.exe build\qwen35.insignia-index` | **NOT RUN** (>1 GB) |
| streaming smoke | build/streaming-smoke.exe (NOT a mk.py target; manual nvcc line in src/streaming.cu:467) | NvmeReader+PinnedRing+LayerFeeder: 2 requests/slot, epoch re-arm (2 rounds), teardown with units in flight, byte-compare vs buffered golden read (192 MiB golden, 128 MiB streamed) | 2x80 MiB ring, 32/64 MiB requests | rc-gated memcmp MATCH | `build\streaming-smoke.exe [path]` (default `Qwen3.8-27B-FP8\layers-0.safetensors`) | **PASS** — all MATCH, 1.5–2.4 GiB/s fill (1.5 s, ~320 MB read) |
| test-cpu (mk target) | future=True in mk.py:76 but sources+binary exist — flag is stale | (same as test_cpu.cpp) | | | | PASS (above) |
| FUTURE 27B targets | none (generate27, nll27, dump-layers27, io-bench listed future) | | | | | **NOT-BUILT** |

### Reference-comparison flows (manual, not rc-gated, all read the full checkpoint)

- `dump-layers` → `build\layers-native.f32` → `tools/reference_all_layers.py` (per-layer,
  MXFP4 u8-scale ground truth); i4 variant `dump-layers-i4` + `reference_all_layers_i4.py`.
- `dump-layer0`/`dump-layer3` → `reference_layer0.py` / `check_layer3.py` /
  `reference_layer3_native_input.py` (native-input isolation) — layer0 DeltaNet cos ~0.9999998.
- `dump-attention` → `reference_attention_seams.py` (18 seams incl. q/k norm, RoPE, core, gated).
- `dump-multistep` → `reference_multistep.py` / `_i4.py`: per-step worst layer cos. **Current
  state from build\multistep-parity.log (MLX path): worst 0.99978 — near parity.
  build\i4-ref.log (INSIG4 path): worst_layer_cos 0.972 at step 5 — the i4 path is still
  WRONG (consistent with AGENTS.md's unresolved full-attention parity issue and
  w4/quantizer.md's "active INSIG4 file doubly broken").**
- `dump-pf` → `reference_pf_i4.py` (prefill seams, i4 only — **no MLX-path prefill reference**).
- `dump-i4-chunk`/`dump-i4-seams` → chunked prefill (`embed_gather_i4`) + 64-token seam audit.
- `nll` DLL via `tools/nll_compare.py`: engine NLL mxfp4-vs-insig4, engine-vs-engine only,
  **no NumPy ground truth on 9B**. `tools/reference27.py` provides `selftest/layer/seams/nll/
  greedy` ground truth for 27B — ready, waiting for an engine (no consumer built yet).

---

## 2. Coverage matrix (engine component → tested-by)

| Component | Unit test | Indirect / flow test | Verdict |
|---|---|---|---|
| mxfp4_gemv (E8M0 blocks) | test-mxfp4 (exact, gated) | — | COVERED |
| mxfp4_gemv_mlx (u8 scales) | none numeric (test-checkpoint checksum only) | dump-layer0/layer3 + reference flows; bench-mxfp4 | WEAK (no gated unit test) |
| mxfp4_gemv_v2 (u8) — MLX decode linear | none | dump flows via decode.cu:31 | WEAK |
| mxfp4_gemv2_v2 (u8) | none | dead code (no caller found) | UNTESTED (unused) |
| mxfp4_gemv2_q8 / _q8g / gemv_q8g / ab2_q8g / ab2_q8 (u8, q8 activation path) | none | q8 via dump flows (decode.cu:32,68); **q8g prequant variants unused+untested** | PARTIAL/UNTESTED |
| quantize_x8 / quantize_q8_groups / f32_to_bf16 (activation staging) | none | only inside whole-model flows | UNTESTED (unit) |
| i4 gemv family: gemv_v2_i4, gemv2_q8_i4, get_row_i4 | test-i4 (f64 ref, exact row gather) | test-qwen35, dump flows | COVERED but UNGATED |
| i4 gemm: gemm_mlx_i4, gemm_v21_i4, gemm_ab_i4 | test-i4 (T=3/33, cross-kernel) | dump-pf flows | COVERED but UNGATED |
| **gemv_ab2_q8_i4 (pair-prefill fused a/b, decode.cu:68)** | **NONE** | only whole-model dump-pf | **UNTESTED (unit)** |
| mxfp4_gemm_v21 / gemm_ab (u8-scale MLX prefill) | none (i4 twins tested) | dump-pf (engine), **no MLX prefill NumPy reference** | WEAK |
| fp8_gemv / fp8_gemv2 / fp8_gemm | test-fp8 (cos ~1.0, T=3/33/65-throw) | — | COVERED but UNGATED |
| fp8 alignment guards (cols&127, rows&31 throws, smem>99KB) | only T=65 throw exercised | — | WEAK (guard paths untested) |
| bf16_get_row (fp8.cuh, 27B embed) / bf16_gemv (GPU, MTP head, decode.cu:155) | **NONE on GPU** (CPU twins tested in test-cpu) | whole-model only | **UNTESTED (unit)** |
| embed gathers: get_row_mlx, embed_gather (u8), embed_gather_i4 (batch) | get_row_i4 single-row only | dump-i4-chunk (i4 batch) | PARTIAL; MLX gather UNTESTED (unit) |
| gqa_decode (graph-replay API) | test-attention source **bitrotted** vs header (passes int where `const int*` expected, missing base/max_context) — stale exe passes | test-layer, dump flows | BROKEN TEST |
| gqa_prefill, store_kv_batch, qk_norm_rope_batch, split_q_gate_batch | none | dump-pf + reference_pf_i4 | WEAK (manual) |
| deltanet_decode | test-deltanet (gated) | test-layer, dump-layer0 | COVERED |
| deltanet_prefill, conv_prefill_silu, deltanet_params_batch | none | dump-pf flows | WEAK |
| rope/norm GPU ops: rmsnorm_gated_silu, silu_mul, residual_add, qwen35_qk_norm_rope_gate; qwen_kernels: rmsnorm_bf16, gated_rmsnorm_bf16, causal_conv4_silu, deltanet_parameters, expand_gate_heads, sigmoid_mul, split_q_gate, store_kv, concat | only rmsnorm_zero_centered (test-ops) | layer0 parity cos 0.9999998 + CPU twins unit-tested in test-cpu | WEAK on GPU (mirrored on CPU) |
| argmax (both kernels) | test-argmax (gated, 200 rounds) | every whole-model run | COVERED |
| logsoftmax / fused logsumexp (nll.cu) | **NONE** — nll_compare is engine-vs-engine | reference27.py nll ready, no 27B engine | **UNTESTED vs ground truth** |
| spec_setup/commit/rollback (device kernels, decode.cu:221-245) | **NONE** | test-pair, test-pair-chain, dump-multistep (all 6-GB, all ungated) | **UNTESTED (cheap)** |
| storage/TieredStorage LRU | test-checkpoint (1 eviction cycle), test-qwen35 | every IDX run | PARTIAL (no stress, no concurrency) |
| model loader INSIDX01 | test-model (gated on 699) | — | COVERED (9B only) |
| **model loader INSIDX02 (27B, 66 shards)** | **NONE — ModelFile rejects INSIDX02 outright (model_file.cpp:23); no loader exists** | index27.py crc32 at build time only | **UNTESTED / MISSING** |
| streaming NvmeReader/PinnedRing/LayerFeeder | streaming-smoke (gated, byte-exact, re-arm, teardown) — but NOT a mk.py target; happy path only | io_bench (bench only) | PARTIAL (prior "no test" claim now half-fixed; error paths untested — see w4/streaming.md defects: empty-plan callback violation, event leak, retry path) |
| CPU tier kernels + CpuPool | test-cpu 40/40 gated, st==mt bit-exact | — | COVERED; **pool STRESS missing** (no oversubscription/concurrency/destructor test; bf16-gqa only t=2048) |
| quantizer (quantize_insig4.py) | **NONE** (w4/quantizer.md: uncommitted quantizer never run; active INSIG4 file is a bad encode) | insig4 SQNR measured in audits only | **UNTESTED** |
| generate.cu CLI + CUDA-graph loop | none (manual) | generate.bat / chat.py manual | UNTESTED (automated) |
| MTP draft | test-mtp gate is draft>=0 (toothless) | dump-multistep ref_draft column | WEAK |

Prior-audit claims verified: "streaming.cu has NO test" — **outdated**: streaming-smoke.exe
exists and passes (but is outside mk.py, happy-path only). "fp8 alignment untested" —
**confirmed** (guards exist at fp8.cu:53/97/187, only T=65 path exercised). "CPU pool stress
missing" — **confirmed** (st==mt checks only, no stress).

---

## 3. Missing tests, ranked by blast radius

Silent-wrong-token class (worst — ships garbage confidently):
1. **Gated unit test for test-i4 / test-fp8** (both always exit 0). A regression in any INSIG4
   or FP8 kernel currently turns the suite green. Spec: convert prints to asserts — cos>0.99999,
   sampled max_rel<1e-2 (floored like test_cpu's Parity), T=65/misaligned-dims throws; runtime <10 s.
2. **gemv_ab2_q8_i4 unit test** (pair-prefill fused a+b): two v2_i4 GEMVs over rows 0..31/32..63
   as reference (recipe already in test_i4 §6 for the gemm twin); assert max_rel<1e-2; <5 s.
   The pair path is the production MTP verify path — corruption = silent wrong tokens.
3. **spec commit/rollback unit test, checkpoint-free**: drive spec_prologue/setup/commit/rollback
   on synthetic pos/state buffers; assert pos arithmetic and snap restore bit-exact after a
   rejected draft; then a full `prefill_chunk(pair)` on synthetic weights. <2 s. (i4-ref.log's
   0.972-cos divergence at step 5 is exactly the class of bug this would localize cheaply.)
4. **NLL ground-truth gate**: 9B NumPy NLL (reference_all_layers pattern + chunked logsumexp)
   or reuse reference27.py when nll27 lands; assert |ΔNLL|<0.01 on a fixed 128-token probe.
5. **GPU bf16_gemv + bf16_get_row + get_row_mlx + embed_gather unit tests**: exact-row
   assertions vs host bf16 (error budget 0 for gathers; cos>0.999999 for gemv at 4096x8192); <3 s.
6. **MLX (u8-scale) prefill reference**: extend reference_pf_i4.py pattern to u8 scales so the
   non-i4 prefill path (v21/gemm_ab u8) has a NumPy ground truth; seam cos>0.9999.

Crash/hang class:
7. **INSIDX02 loader test** (once the loader TU lands — currently nothing can even open the 27B
   index): assert 407 F8+scales links, shard crc32 spot-check, dtype tags (F8_E4M3=7), tensor
   bounds rejection ("tensor escapes mapping"), wrong-magic rejection; <1 s.
8. **CpuPool stress**: 10k varied-size ticket launches from main + 2 extra threads, results
   bit-equal serial, clean shutdown; plus gqa bf16 t=77 and deltanet 2-step kshare=3 already
   present; <30 s budget.
9. **Streaming error paths as mk.py target**: empty-plan epoch (callback must fire — currently
   violates contract per w4/streaming.md), retry path (simulate failing read), slot starvation,
   66-shard interleaved epoch byte-compare; gate rc; budget 60 s.
10. **fp8 guard-path test**: cols=4160 (cols&127≠0), rows=1000 (rows&31≠0), T=65 must throw.

Stale/toothless (test suite lies):
11. Fix test_attention.cu for the current gqa_decode API (pos_dev device buffer, base,
    max_context) — today the target cannot build; the passing exe is from Aug 24 02:07.
12. mk.py test-prefill run line missing `<n>` (immediate rc=2 today).
13. Gate test-pair/test-pair-chain (return 1 on divergence) and make a sub-1-GB variant
    (768 MiB budget like test-layer) so token-parity can run unattended.

---

## 4. Test hygiene findings

- **Duplicated build systems**: all 19 `build/test-*.bat` rebuild via raw nvcc and duplicate
  mk.py TARGETS; several have drifted: test-i4.bat compiles `prefill.cu` (mk.py doesn't),
  test-qwen35.bat omits `mxfp4_i4.cu` (mk.py includes it), test-deltanet.bat lacks
  `--use_fast_math`, and test-attention.bat would now fail to compile (API bitrot above).
- **Ungated tests**: test-i4, test-fp8, test-pair, test-pair-chain always return 0; the only
  "assertions" are printed strings ("NO (BUG)", "DIVERGED"). CI-style use is impossible.
- **9B index dependencies** (all >1-GB class tests): `build\qwen35.insignia-index` → external
  HF cache `models--sleepyeldrazi--Qwen3.5-9B-MXFP4-MTP` model.safetensors (path baked into
  the index; if the cache is pruned every IDX test, dump flow, and nll mxfp4 leg dies).
  `qwen35-insig4.insignia-index` → local `qwen35-insig4-text.safetensors` (currently a bad
  encode per w4/quantizer.md). `qwen38-27b-fp8.insignia-index` → local 66 shards (no consumer).
  `nll_compare.py` additionally needs the `models--Qwen--Qwen3.5-9B` tokenizer from HF cache.
- **Hardcoded expectations tied to one checkpoint**: test-model asserts exactly 699 tensors;
  test-pair/pair-chain hardcode greedy ids (2614/314/…) valid only for this checkpoint+quant.
- **Flaky timing**: no test asserts on timing (all prints) — good; but conversely nothing
  guards performance regressions either. argmax's 200 randomized rounds is the only
  run-to-run nondeterminism probe; no bit-exact repeat-run check for any GEMM.
- **Determinism**: test-cpu's st==mt memcmp is the only bit-exactness check in the repo.
  No GPU kernel has a repeat-run variance check (atomic argmax partially covered via agreement).
- **Stale binaries**: build/test_cpu.exe vs build/test-cpu.exe (two names, same tool, 1 h
  apart); test-attention.exe predates its header; test-cpu is future=True in mk.py though it
  builds fine; io-bench likewise. Root dir litter: hello.obj, model_file.obj, nul.obj,
  test_cpu.obj, build$t.dll, wl_O0/O1/O2.exe, gdb*.log.

---

## 5. Minimum safety net before the 27B engine runs unattended (≤8)

1. **Gate test-fp8** (exit≠0 on cos<0.99999 for gemv/gemv2/gemm T∈{1,3,33,64}; assert the
   T=65 + misaligned-dims throws). Runs 5 s, no checkpoint. (Wrong FP8 math = silent garbage.)
2. **Gate test-i4** (same policy; plus the gemv `bad` counter gates rc). 2 s, no checkpoint.
3. **Fix + gate test-attention** to the current gqa_decode API; add T=4096 (score-buffer
   boundary). <5 s. (KV decode bug = silent wrong tokens.)
4. **test-attention27 / gqa shape check** for 27B head counts, or extend test-cpu's gqa_ref
   recipe to the GPU kernel at 27B (24q/4kv/256d) — exact softmax double ref, max_abs<2e-5.
5. **INSIDX02 loader smoke**: open build\qwen38-27b-fp8.insignia-index, assert shard table
   crc32 + 407 F8 links + dtype tags + bounds rejection. <1 s once the loader exists.
6. **Streaming correctness as a mk.py target** (today's streaming-smoke + empty-plan epoch +
   retry simulation), rc-gated; 60 s. A hung slot = unattended run stalls forever.
7. **27B layer-parity gate**: dump-layers27 single linear-attn layer + single full-attn layer,
   T=2..33, vs reference27.py `layer/seams`; assert per-seam cos>0.9999 AND exact greedy token
   match on a fixed 7-token prompt (recipe: test_pair_chain, budgeted to 1 GiB so it stays runnable).
8. **test-cpu + CpuPool stress** (40 existing checks + pool stress + gqa bf16 t=77): the CPU
   tier executes 27B layers; a pool race = wrong tokens or hang under load. <30 s.

Prerequisite zero-cost fix: make test-pair-chain/test-pair return failure on divergence and
repair the test-prefill run line — otherwise even the existing 9B safety story is illusory.
