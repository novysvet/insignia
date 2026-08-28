# W4 — 27B parity-ladder execution plan (R0–R10 work orders)

Date: 2026-08-25/26. Read-only audit + python arithmetic only; no builds, no
mk.py/nvcc/cl invocations, no file changes except this report. Inputs read in
full: `audits/w3/parity-ladder.md`, `audits/w3/diff-verify.md`,
`audits/w3/MASTER-PLAN.md` (§2, §3 Phases 0–F), `audits/w4/reference27.md`,
`audits/w3/reference27.md`, all 8 dump tools in `src/`, all reference scripts
in `tools/`, `tools/mk.py`, `tools/index27.py`, `src/{decode,qwen35,fp8,
test_fp8,attention,nll,mxfp4_i4,streaming}.cu`, `include/*` headers, live
`build/qwen38-27b-fp8.insignia-index` header bytes, `Qwen3.8-27B-FP8/crc32.txt`.

Mission answers, in order: §1 9B inventory, §2 R0–R10 work orders,
§3 golden-token bootstrap, §4 H1–H8 + new traps, §5 R5/R6 cost math,
§6 run-order script.

---

## 1. Current 9B implementation inventory (rung by rung)

Build path today is `python tools/mk.py <target>` (mk.py:38-81; `build\*.bat`
are legacy — see H4). DLLs run via `python tools/rundll.py build\<t>.dll <args>`
(rundll.py:1-25, keeps a dummy argv[0] so tool argc conventions hold).

| rung (9B analogue) | dump tool (src) | CLI (after mk.py target) | output layout | reference pair (tools) | thresholds |
|---|---|---|---|---|---|
| R0 index (9B) | `tools/index_safetensors.py` | `python tools/index_safetensors.py model.safetensors out.insignia-index` | INSIDX01, names verbatim, dtypes {U32,U8,BF16,F32,F16} — **no F8_E4M3** | — | raise-on-unknown-dtype |
| R1 kernel units | `src/test_fp8.cu` (target `test-fp8` = fp8.cu + test only) | `python tools/mk.py test-fp8 && python tools/rundll.py build/test-fp8.dll` | stdout: cos lines | in-binary f64 host ref | none enforced (print-only); ladder: cos>0.999999 |
| R3 IO (9B→27B) | `src/streaming.cu` (NvmeReader+PinnedRing+LayerFeeder, built-in main) + `src/io_bench.cu` | `python tools/mk.py io-bench`; `streaming-smoke` | byte-equality vs gold buffer at DATA_START | in-tool memcmp | pass/fail prints |
| R4/R5/R6 prefill seams | `src/dump_pf.cu` (target `dump-pf`) | `rundll build/dump-pf.dll <index> 760,3712,... 14 out.f32` | 33 × T × 4096 f32, seam-major (32 layer outputs + final norm) | `tools/reference_pf_i4.py model.safetensors ids dump.f32` | print-only worst-seam cos (H1 poison for t>0) |
| R6 single-token all-layer | `src/dump_layers.cu` (target `dump-layers-i4`) | `rundll build/dump-layers-i4.dll <index> out.f32` | 32 × 4096 f32 (token 42) | `tools/reference_all_layers_i4.py model.safetensors out.f32` | print-only per-layer max/mean/cos |
| R7 multistep | `src/dump_multistep.cu` (target `dump-multistep`) | `rundll build/dump-multistep.dll <index> 760,3712,314,23470 out.f32 [gen_extra=1]` | (steps+1) × 33 × 4096 f32 + stdout argmax/draft per step | `tools/reference_multistep_i4.py model ids dump.f32` | print-only; ref argmax vs engine stdout compared by eye |
| sub-op debug | `src/dump_i4_seams.cu` (layer-0, 14 seams), `src/dump_i4_chunk.cu` (T=2 chunk path, 14 seams + all-layer + 2×248320 logits), `src/dump_attention.cu` (layer-3 sub-ops, **u8-only**), `src/dump_layer0.cu`/`dump_layer3.cu` (single vector via `d.layer()`) | `rundll build/dump-i4-seams.dll <index> out.f32` etc. | tool-specific | `tools/reference_layer0.py`, `check_layer3.py`, `reference_attention_seams.py`, `compare_vec.py` | print-only |

Key facts established from the current tree:

- **No 9B reference script enforces a threshold programmatically** — all print
  worst cosines and exit 0. Gates live in `parity-ladder.md` §6 and are applied
  by the operator. The 27B compare tooling (§2 R4) must add exit-code gates or
  the run-order script (§6) must grep the printed minima.
- `dump_pf.cu:28` caps T ≤ 64; `dump_multistep.cu:14` takes optional gen_extra.
- `dump_multistep.cu:31-36` now branches i4/u8 for lm_head (H2 9B form fixed).
- `dump_attention.cu:10` and `test_checkpoint.cu:10` remain u8-scale-only (H6).
- `dump_i4_chunk.cu:27-33` writes conv/delta carried states to argv[2] and then
  **reopens the same path `"wb"` and truncates** — the state dump is always
  destroyed (new finding N3; 9B-only footgun).

## 2. R0–R10 work orders for 27B

Context that reshapes the ladder: **R0 is mostly already done** —
`tools/index27.py` exists (INSIDX02, engine-name map §1.1, F8_E4M3=7, config
verify, 66-shard crc32 gate, byte accounting, self-read) and
`build/qwen38-27b-fp8.insignia-index` is built (119,017 B, header verified to
reference `..\Qwen3.8-27B-FP8\layers-N.safetensors`). What is NOT done:
`src/model_file.cpp:23` still rejects anything but INSIDX01/v1 — the engine
cannot load the index yet; `Qwen35Shape` is still 4096/12288/32
(`include/insignia_qwen35.hpp:7`); `decode.cu` loops/sizes are all 9B
(decode.cu:14-26, 47); `linear/linear2/linear_batch` (decode.cu:31-41) have no
fp8/bf16 arm even though `qwen35.cu:19-28` matrix() already returns
`WKind::fp8`/`bf16`. So R4+ clones ride MASTER-PLAN Phases A–D first.

### R0 — loader/index bring-up (gate for R3+/R4+)
- Index side: **DONE**. Command (re-runnable): `python tools/index27.py
  Qwen3.8-27B-FP8 build/qwen38-27b-fp8.insignia-index` (crc gate included).
- Engine side work order: `src/model_file.cpp` + `include/insignia_model.hpp`
  → INSIDX02 parse (66-shard table, per-shard bounds, `find_linked`),
  keep INSIDX01 path for 9B regression. Update `src/test_model.cpp` table diff
  vs ladder §1 census. `f8_e4m3=7` dtype already wired in qwen35.cu:19.
- Metric/threshold per ladder §6 R0: 100% of §1 tensors mapped, zero orphans,
  dtype/shape diff empty. (index27's self-read + matrix() probe already prove
  the index side; the engine-side resolution test is the remaining gate.)

### R1 — fp8 GEMV/GEMM units vs f64 (gate R4+)
- Clone/extend: **`src/test_fp8.cu` in place** (target `test-fp8` exists,
  mk.py:46). Current state: bias-7 host reference **fixed** (test_fp8.cu:15
  `ldexpf(1+m/8, e-7)`), single shape (10240×5120), T∈{3,33}, T=65 throw test.
- Changes: loop over all 7 27B geometries {(10240,5120),(6144,5120),
  (5120,6144),(17408,5120),(5120,17408),(12288,5120),(1024,5120)} × T∈{3,33,64}
  for `fp8_gemm` + `fp8_gemv`/`fp8_gemv2`; add max|Δ|/‖y‖ next to cos (ladder
  §6 R1 — cosine alone hid the old bias bug); keep T>64 throw assertions.
- **Blocking new trap N1**: `fp8_gemv` (fp8.cu:52-55) sets no
  `cudaFuncAttributeMaxDynamicSharedMemorySize`; cols=17408 needs
  cols*4 = 68 KB dynamic smem > 48 KB default → launch fails (y stays stale).
  `fp8_gemv2` has the 99 KB opt-in (fp8.cu:100) but its guard throws over
  99 KB (fp8.cu:97-99) so the pair path can't take down_proj either. Fix by
  adding the attribute to fp8_gemv (and a launch-error check). The shape sweep
  above catches this deterministically.
- Command: `python tools/mk.py test-fp8 && python tools/rundll.py build/test-fp8.dll`.
- Threshold: cos > 0.999999 AND max-rel < 1e-4, every shape/T.

### R2 — CPU kernels: future, does not block R3–R10 (ladder §6 R2). Skip here.

### R3 — IO integrity (gate R4+)
- Already covered by: index27 crc32 verification (66/66 shards vs crc32.txt at
  index build; crc32.txt has 77 lines incl. config files — first lines verified
  live, e.g. `e3a49ca5 layers-0.safetensors`), and `streaming-smoke`
  (src/streaming.cu:481-541 main does gold-buffer byte-equality + ring tests).
- Remaining work order: once TieredStorage2 (Phase D3) exists, one sweep
  streaming every shard through the reader vs `np.memmap` byte compare + the
  `(f8_base & 15) == 0` 16-byte alignment assert (MASTER-PLAN risk #7 —
  data_start ≡ 8 mod 16 in every shard; the index/plan builder must insert the
  8-byte pad at the BF16→F8 boundary). `io-bench` acceptance: E: ≥ 3.0 GB/s.

### R4 — layer-0 (DeltaNet) prefill seam (gate R5)
- Clone `src/dump_pf.cu` → **`src/dump_pf27.cu`** (new mk.py target
  `dump-pf27`, ENGINE27 closure). Changes:
  - seam count 32→65 blocks, T×**5120** rows per seam (dump_pf.cu:12,37-40);
  - index arg = `build\qwen38-27b-fp8.insignia-index`; names stay engine-side
    (`language_model.model.layers.N.*`) because index27 already remaps;
  - rides the Phase B workspace (attn_gate/core 6144, a/b 48, delta
    48×48×128×128, conv 48×10240×3, kv 16·ctx·1024, pf_bf16 **64×17408×2B** —
    N4: decode.cu:26 is still 64×12288×2B) and the Phase A7 zero-center flips
    (`rmsnorm_bf16(..., true)` at 9 sites + `qk_norm_rope` +1;
    `linear_attn.norm` stays raw — rmsnorm machinery already parameterized,
    qwen_kernels.cu:5 `rms_bf<Z,G>`, insignia_qwen_kernels.cuh:5);
  - dispatch: `linear_batch` → `fp8_gemm` (f32→bf16 staging, zero-padded tail),
    embed → `bf16_get_row`/`embed_gather_bf16`, a/b → bf16 pair GEMV,
    A_log bf16 (deltanet params dtype dispatch), deltanet `<<<48,128>>>`,
    conv 10240 — i.e. the Phase A–C engine changes, with
    `prefill_chunk_seam` (decode.cu:117) cloned at 27B shapes;
  - streaming acquisition: parity dumps ride the **v1 all-stream placement**
    (MASTER-PLAN §2.4/Phase E): every layer shard acquired through the
    TieredStorage2/LayerFeeder path, L=19 resident layers only if budget set;
    sweep cost §5 below.
- Reference: **`tools/reference27.py seams <ids> <out.npy>` already implements
  the R4 reference math** (conv roll correct — reference27.py:239-243, the H1
  class is closed there; w4 audit 26/26 PASS). Missing piece: an engine-dump
  compare. Work order: add a `compare` subcommand (or `tools/compare27.py`)
  reading engine `.f32` [65,T,5120] + model dir + ids → per-seam cosine table
  with **exit-nonzero gate** at the ladder thresholds.
- Command pair:
  `python tools/mk.py dump-pf27 && python tools/rundll.py build/dump-pf27.dll build\qwen38-27b-fp8.insignia-index 760,3712,314,23470,25044,369,303,919,279,3712,314,16465,33633,13 14 build\27b\pf-seams.f32`
  then `python tools/reference27.py --model Qwen3.8-27B-FP8 compare 760,...,13 build\27b\pf-seams.f32`.
- Threshold: every layer-0 seam cos > 0.99999 over T=14 (9B layer 0 hit
  0.9999998); plus the α∈(0,1) sanity (deltanet-27b §7.3 signature check).

### R5 — full-attn layer-3 seam (gate R6)
- Same dump as R4 (seams 0..3 include layer 3); compare row l=3, tokens 0..13.
- Exercises: RoPE pairs (i,i+32) statically resolved by w4 audit — runtime
  kill-shot stays), GQA h/6 (head>>2 would silently mis-group — MASTER-PLAN
  Phase B correction), q/k norm (1+w), output gate, `kvh=h//6` kernels.
- Threshold: cos > 0.99999 for all 14 tokens, **5 repeats, zero run-to-run
  variance** (protocol in §5). Hard stop on flicker.

### R6 — all-64-layer seams (gate R7)
- Same dump; reference = `reference27.py seams` (layer-major, one shard at a
  time, ~1.5 GB f32 per LayerSet — reference27.py:168-198, within ladder §5
  policy) + the compare command.
- Threshold: min ≥ 0.9999 and median ≥ 0.99999 over 65×14 = 910 vectors,
  zero NaN seams.
- Note: `seams` writes `.npy` while the engine dump is raw f32 — the compare
  command handles both (np.fromfile vs np.load).

### R7 — multistep decode states, 4 steps (gate R8)
- Clone `src/dump_multistep.cu` → **`src/dump_multistep27.cu`** (target
  `dump-multistep27`). Changes: 64 layers × 5120 (dump_multistep.cu:42-47),
  final norm name via index (engine name unchanged), **lm_head must branch
  bf16** (`bf16_gemv`, qwen_kernels.cu:68 — rows=248320 blocks — or Phase C5
  bf16 GEMM): this is the 27B reincarnation of H2; the i4 branch pattern to
  copy is dump_multistep.cu:31-36. Embed via `bf16_get_row` (fp8.cu:198).
  MTP part (dump_multistep.cu:63-65) needs 27B MTP wiring (fc 5120×10240 —
  H7) or is omitted from the R7 gate (ladder R7 does not gate on draft ids;
  reference MTP is absent anyway — §2 gaps).
- Reference: **gap** — reference27.py has no multistep-seam compare. Work
  order: add `multistep <ids> <dump.f32>` subcommand: token-major, States
  persist across steps (States class already does exactly this for `greedy`,
  reference27.py:205-226, 502-519), report per-step worst-layer cos + final
  cos + ref argmax, diff engine argmax from stdout.
- Command: `rundll build/dump-multistep27.dll build\qwen38-27b-fp8.insignia-index 760,3712,314,23470 build\27b\multi27.f32` (args 5 = 4 teacher + 1 generated, layout (4+1)×65×5120).
- Threshold: worst-layer cos ≥ 0.9999 every step, final-norm ≥ 0.99999,
  ref argmax == engine argmax 4/4 steps (argmax tie-break: reference27
  `greedy_argmax` keeps lowest index on '>' ties — engine `argmax_fast`
  behavior must be spot-checked once).

### R8 — NLL on GOLDEN-128 (gate R9)
- Clone `src/nll.cu` → **`src/nll27.cu`** (mk.py future target `nll27` already
  reserved, mk.py:79; needs src file + ENGINE27 closure). Changes: 27B chunk
  loop (T=64 chunks ×2 for 128 ids), fp8 body, **bf16 lm_head branch**
  (nll.cu's current lm_head GEMM is mxfp4 i4/u8 only — nll.cu:78-81 pattern
  extended with a bf16 arm), vocab 248320 unchanged, targets ids[1..].
- Reference side: **`reference27.py nll <ids>` exists** (f64 online logsumexp,
  teacher-forced; reference27.py:342-360, 485-499). Pairing runner: extend
  `tools/nll_compare.py` (currently 9B two-index MXFP4-vs-INSIG4; point at the
  27B index + reference27 output, drop the 9B cross-compare per ladder §6 R8).
- Threshold: |ΔNLL| < 0.02 nat and ppl ratio ∈ (0.98,1.02) on first clean run;
  tighten to 0.005 after R1–R7 green.

### R9 — greedy 8-token continuation (gate R10 + correctness claim)
- Clone `src/generate.cu` → **`src/generate27.cu`** (future target
  `generate27` reserved, mk.py:78). Eager only, NO CUDA graphs (Phase D4).
- Command: `python tools/chat.py build\qwen38-27b-fp8.insignia-index
  Qwen3.8-27B-FP8 "<prompt A>"` — note chat.py:44 hardcodes
  `build\generate.dll`; either add a dll-path arg or have the runner invoke
  `rundll build/generate27.dll <index> <ids> <max_new>` directly.
- Reference: **`reference27.py greedy <ids> 8` exists and is the ground truth
  by ladder §0** (state persistence + absolute positions verified by w4 audit
  aspects 39-40). Bootstrap of the expected ids: §3.
- Threshold: 8/8 exact ids; any divergence → bisect with R6/R7 dumps at the
  first diverging step.

### R10 — endurance/stability (release gate)
- `chat.py` max_new=1000 + stats hook (finiteness of sampled seams/logits,
  self-NLL < 5 nat/token, committed monotonicity, VRAM/RAM budget drift,
  tok/s vs placement prediction ≥ 80%). No 9B tool to clone — wrapper work on
  chat.py/generate27 output. Thresholds exactly ladder §6 R10.

### reference27.py vs ladder — precise gap list
Already implemented (verified): shard reader + `dq_f8` (§3 semantics), e4m3
selftest, layer trajectory, **full 65-seam prefill with correct in-chunk conv
history** (H1 class absent), all-layer math incl. attention (R4/R5/R6),
teacher-forced NLL (R8), greedy ids + logits + decode (R9), tokenizer glue.
Gaps:
1. **No engine-dump compare command** — `seams` writes a trajectory npy but
   never reads an engine dump or computes/cosines/gates (R4/R5/R6 blocker).
2. **No multistep seam-compare** (R7 blocker) — states machinery exists in
   `greedy`; needs the dump-reading, per-step report variant.
3. **No MTP reference at all** — grep "mtp" = 0 hits; `mtp.fc`, `pre_fc_norm_*`
   (measured 100% negative → (1+w) mandatory), `mtp.norm`, `mtp.layers.0` all
   absent (w4 audit aspects 35-36, HIGH). Blocks draft-path parity (R7 draft
   column) and Phase F acceptance. Work order: `mtp` subcommand per w4 §3.1:
   `fc @ concat(rms(embed,pre_fc_emb), rms(hidden,pre_fc_hid))` — embed half
   first (ladder §1.4), mtp.layers.0 full-attn block, mtp.norm (1+w), shared
   bf16 lm_head.
4. No pairing runner for NLL (engine side nll27 + compare harness) — script
   side complete.
5. Minor/dead: `--attn` flag is a no-op (reference27.py:533, 562-564), QSCALE
   dead (line 52), greedy reloads all 64 shards per step (~26.9 GB reads/step —
   documented slow-but-correct).

## 3. Golden tokens — where they come from, exact bootstrap

Ladder §6 references GOLDEN-14 / GOLDEN-128; the actual ids are specified in
ladder §8 ("hardcode in tests"). Provenance + bootstrap for 27B (no torch):

1. **Token id provenance** = the 27B's own tokenizer via
   `python tools/reference27.py enc "<text>"` (reference27.py:387-393 loads
   `<model_dir>/tokenizer.json`; tok.py/chat.py are the same path). §7 of the
   ladder already verified 9B-vs-27B tokenizers are byte-different but
   probe-id-identical. Bootstrap = re-encode and diff:
   - Prompt A → must equal `[760, 3712, 314, 23470, 25044, 369, 303, 919, 279, 3712, 314, 16465, 33633, 13]`;
   - B `"Hello!"` → `[9419, 0]`; C = A[:4];
   - GOLDEN-128 = first 128 ids of the `tools/nll_compare.py` probe text —
     re-encode the full text with reference27 enc and diff against the §8
     128-id list; also assert `min(ids) ≥ 0 and max(ids) < 248320`
     (parse_ids already enforces this, reference27.py:395-401).
   Mismatch ⇒ tokenizer drift ⇒ stop before any parity run.
2. **Expected-output provenance** (greedy ids for R9, NLL value for R8): there
   is no external oracle — transformers is installed but torch is not
   (ladder §0), so reference27.py IS ground truth by charter, with an
   independence argument: separate NumPy implementation from raw checkpoint
   bytes + HF `_ref_modeling_qwen3_5.py` source, math-audited 26/26 (w4),
   e4m3 table self-tested bitwise, α∈(0,1) and finite growing norms on a live
   4-token smoke (w4 §2.7). Bootstrap procedure (one-time, reference-only):
   - `python tools/reference27.py selftest` (gate);
   - `python tools/reference27.py seams 760,...,13 build\27b\ref-seams.npy`
     (records reference seam norms — also the R6 ‖seam‖ baseline for R10);
   - `python tools/reference27.py nll <GOLDEN-128>` → record mean NLL/ppl to
     `build/27b/golden-nll.txt`;
   - `python tools/reference27.py greedy 760,...,13 8` → record ids + logits
     to `build/27b/golden-greedy.txt`; decode the text — human-coherence
     sanity (the continuation of "The history of computing machinery…" must
     read as English; garbage ⇒ reference bug, stop);
   - **reference self-determinism check**: rerun greedy with `LM_CHUNK` 8192 →
     4096 (and optionally f64 argmax) and diff ids — catches chunk-boundary
     argmax ties inside the reference itself. Only then lock the constants.
3. First-run engine outputs are never golden — R9 compares engine ids to the
   locked reference ids; on mismatch, bisect via R6/R7 dumps (ladder §6 R9).

## 4. Diff-verify traps H1–H8 — current-tree verdicts (+ new traps)

| id | claim (diff-verify.md) | verdict now | evidence |
|---|---|---|---|
| H1 | `reference_pf_i4.py` conv history missing for T>0 | **OPEN (9B)** — `reference_pf_i4.py:34-35` still `qkv = qkv * cw[:, 3]`, no conv state, no roll; per-token loop at :79-89 has Δ-state carry only. 9B mitigation was T=1 dumps (`build/pf-i4-v21-t1.f32`). **H1-class is closed for 27B**: reference27.py:239-243 rolls conv correctly and persists across steps |
| H2 | `dump_multistep.cu` lm_head u8-only | **FIXED (9B i4)** — dump_multistep.cu:31-36 `mat()` branches `insig4 → mxfp4_gemv_v2_i4`. **27B form still open**: no bf16 arm — the dump_multistep27 clone must add `bf16_gemv`/bf16 GEMM for the BF16 lm_head (§2 R7) |
| H3 | `red[0]` reuse race in gqa_decode / gqa_prefill / row_logp twins | **FIXED everywhere** — attention.cu:7 (`red[8],smx,sden` dedicated slots), prefill.cu:107 (`red[8], smx, sden`), nll.cu:15-41 (`red[8]`/`red[9]`), generate.cu:18-44 (`red[8]`/`red[9]`); grep `red[0]` in those files → 0 hits |
| H4 | 4 newly + 8 pre-existing link-broken bats | **SUPERSEDED, cleanup open** — mk.py is the canonical path and every dump/test target compiles the full closure (mk.py:24-29, 38-81); the stale bats (dump-attention.bat, dump-layer0/3.bat, test-qwen35.bat, oldgen.bat, shim-only.bat, test-pair-chain.bat, tiny.bat, smoke.bat, bench-gemm-blocked.bat …) are still on disk; MASTER-PLAN Phase 0 "delete 11 stale bats" not executed |
| H5 | `ab2_q8`/`ab2_q8_i4` no dim guard | **FIXED (throws)** — mxfp4_i4.cu:237-238 and mxfp4.cu:670-671 both throw `cols != 4096` ("9B-specialized"). 27B a/b must never route there (they're bf16 there — different kernel) |
| H6 | stale u8-only debug tools misread i4 scales silently | **OPEN** — dump_attention.cu:10 (`mat()` lambda u8), test_checkpoint.cu:10 (u8 `mxfp4_gemv_mlx`). Debug-only; keep pointed at the MLX index or add an assert |
| H7 | mtp.fc i4 dims hardcoded 4096/8192 | **OPEN** — decode.cu:154-155 literals (i4 arm takes 4096,8192; bf16 arm `bf16_gemv(...,4096,8192)`); 27B needs 5120,10240 (or `fc.rows/fc.cols`) |
| H8 | checkpoint hygiene (broken insig4 v1/v2 on disk; default index → -text not -good) | **OPEN (9B-only, moot for 27B)** — `build/qwen35-insig4.safetensors` and `-v2.safetensors` still present (fp16-as-BF16, per diff-verify §6 timeline); `qwen35-insig4.insignia-index` → -text; `-good` + `-good` index exist and are best |

New traps found in this pass (not in diff-verify):

- **N1 — fp8_gemv smem opt-in missing** (fp8.cu:52-55): no
  `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)`. cols=17408 (down_proj)
  ⇒ 68 KB dynamic smem > 48 KB default ⇒ launch rejected, y stale garbage.
  fp8_gemv2 opts in at 99 KB (fp8.cu:100) but throws >99 KB (fp8.cu:97-99).
  Blocks R1's (5120,17408) shape and the 27B decode down_proj path. Fix in R1.
- **N2 — engine cannot read the 27B index yet** (model_file.cpp:23 accepts
  only INSIDX01 v1). Phase A item 3 outstanding; R0 engine-side gate.
- **N3 — dump_i4_chunk.cu state-dump clobber** (dump_i4_chunk.cu:27-33):
  writes carried conv/Δ states to argv[2], then reopens argv[2] `"wb"`
  (truncate) for the seam dump — the state file is always destroyed.
- **N4 — pf_bf16 staging still 64×12288×2B** (decode.cu:26): 27B down_proj
  staging needs 64×17408×2B (ladder §6 R1 note; Phase B2). Assert in R0/R4.
- **N5 — Phase B not started**: `Qwen35Shape` 4096/12288/32
  (include/insignia_qwen35.hpp:7); decode.cu hardcodes l<32, 4096, 8192,
  12288, 24×32×128×128, 24×8192×3, 8×ctx×1024 (decode.cu:14-26, 47, 128).
- **N6 — zero-center norm call sites**: `rms_bf<Z>` machinery + bool param
  exist (qwen_kernels.cu:5, insignia_qwen_kernels.cuh:5) but every call site
  passes `false`; 27B needs `true` at 9 sites + qk_norm_rope +1;
  `linear_attn.norm` stays raw (MASTER-PLAN A7, w4 audit aspects 8-11).
- **N7 — linear dispatch lacks fp8/bf16 arms** (decode.cu:31-41): matrix()
  can return `WKind::fp8`/`bf16` (qwen35.cu:19-28) but linear/linear2/
  linear_batch would miscast to u32/u8. Phase C3 outstanding.

## 5. R5 5×-repeat protocol + R6 sweep cost on the streaming rig (3.22 GB/s)

Byte model (from live shard sizes): linear shard 383,865,448 B ×48, attention
shard 372,313,744 B ×16 → **64 layers = 24.383 GB**; text outside (embed
2.545 + lm_head 2.545 + norm) = 5.086 GB; mtp 0.477 GB; text grand total
29.945 GB. Cold-read at 3.22 GB/s (=3,220 MB/s decimal):

| item | bytes | cold @3.22 GB/s |
|---|---|---|
| one full dump sweep (64 layers + lm_head argmax in prefill epilogue) | 26.93 GB | **8.4 s** |
| 5× R5 repeats (engine side; reference runs once) | 134.6 GB | **41.8 s** |
| R6 reference seams sweep (I/O part; numpy dequant dominates wall-clock, ladder §5 "minutes") | 24.38 GB | 7.6 s I/O |
| R7 engine 5 steps | 137.0 GB | 42.6 s |
| R8 engine NLL-128 (2×64-token chunks) | 53.9 GB | 16.7 s |
| R9 engine prefill-14 + 8 greedy steps (~9 sweeps) | 242.3 GB | 75.3 s |
| v1-placement steady decode step (45 layers streamed, 19 resident) | 17.1 GB | 5.3 s (matches MASTER-PLAN v1 ≈5.21 s) |

Cross-check: MASTER-PLAN Phase E "one sweep ≈ 9–30 s per dump" — the 8.4 s
cold floor agrees; the upper end covers first-touch pinning, ring fills and
lm_head embed-row gathers. Warm page-cache repeats (RAM 15.9 GB < 24 GB
working set — only partial caching) sit between.

**R5 determinism protocol (exact)**:
1. Build once, dump five times with identical args to distinct files
   `pf-seams-r{1..5}.f32` (total ≈42 s cold; reboots not needed — page cache
   partial-warm makes later runs faster, which is fine: determinism is about
   bitwise engine output, not timing).
2. Gate A (engine self-determinism): `cmp` the five files pairwise — must be
   **byte-identical** (any diff = residual race; hard stop per ladder §6 R5).
3. Gate B (engine vs reference): run the compare command on r1 only — layer-3
   seam cos > 0.99999 for all 14 tokens (reference is single-threaded NumPy,
   deterministic, run once).
4. Gate C: no NaN/Inf anywhere in seams 0..3; ‖seam‖ within 3× of the
   reference27 `seams` trajectory norms.
5. Record: min/median cos per run, file sha256s, driver/clock snapshot.

**R6 sweep cost**: engine dump 8.4 s (once) + reference `seams` pass — I/O
7.6 s + numpy dequant of 24.4 GB f8→f32 with per-layer 1.5 GB working set;
expect single-digit minutes wall-clock (ladder §5 estimate "minutes"
confirmed). The compare command itself is seconds. Total R6 ≈ one reference
pass dominates; reruns of the engine dump are ~free after page-cache warm-up
of the resident layers.

## 6. Run-order script sketch (mechanically executable)

```bash
# Insignia 27B parity ladder runner — every gate hard-blocks the next rung.
set -euo pipefail
M=E:/coding/Insignia; IDX='build\qwen38-27b-fp8.insignia-index'; MD=Qwen3.8-27B-FP8
G14=760,3712,314,23470,25044,369,303,919,279,3712,314,16465,33633,13
G128=760,3712,314,23470,25044,369,303,919,279,3712,314,16465,33633,13,14496,417,54595,47434,264,6463,4560,421,1000,5474,6134,45735,5568,1472,279,1654,314,33093,11461,11,17068,3611,1412,494,279,5492,314,86844,20569,12269,13,357,9018,2843,11,13934,60875,12215,51373,11,321,279,9476,1957,1801,279,5484,19565,13,4236,5924,279,491,16698,25,7233,998,5158,11,1301,11,17736,888,11,321,30687,1040,22133,13,17722,13771,2878,8754,11391,314,10895,303,279,854,264,21461,30805,7481,2957,11,3482,279,7326,5154,314,24332,11,55404,11,321,12293,6922,49666,391,3611,13,561,1786,23438,4089,264,18826,279,1560,264,6700,30697,264,1647
cd "$M"

# --- R0: index + loader ------------------------------------------------------
python tools/index27.py "$MD" "$IDX" | grep -q 'SELF-READ PASS'          # crc66+byte-accounting+self-read
python tools/mk.py test-model "$IDX" && ./build/test-model.exe "$IDX"    # INSIDX02 parse + name/shape diff (needs Phase A)
# --- R1: fp8 units -----------------------------------------------------------
python tools/mk.py test-fp8 && python tools/rundll.py build/test-fp8.dll | tee build/27b/r1.log
grep -E 'cos=0\.9999(9[5-9]|[9-9])' build/27b/r1.log                    # gate: cos>0.999999, max-rel<1e-4 (N1 fix required)
# --- R3: IO ------------------------------------------------------------------
python tools/mk.py io-bench && python tools/rundll.py build/io-bench.dll | tee build/27b/r3-io.log   # E: >= 3.0 GB/s
python tools/mk.py streaming-smoke && ./build/streaming-smoke.exe        # byte-equality + alignment assert
# --- Golden bootstrap (reference only) ---------------------------------------
python tools/reference27.py selftest
[ "$(python tools/reference27.py enc 'The history of computing machinery is in part the history of automatic arithmetic.')" = "[$G14]" ]
python tools/reference27.py seams "$G14" build/27b/ref-seams.npy
python tools/reference27.py nll "$G128" | tee build/27b/golden-nll.txt
python tools/reference27.py greedy "$G14" 8 | tee build/27b/golden-greedy.txt   # + rerun with LM_CHUNK=4096, diff ids
# --- R4/R5/R6: prefill seams (one dump serves all three) ----------------------
python tools/mk.py dump-pf27
python tools/rundll.py build/dump-pf27.dll "$IDX" "$G14" 14 build/27b/pf-seams.f32
python tools/reference27.py compare "$G14" build/27b/pf-seams.f32 --layer 0 --min 0.99999   # R4 gate
for r in 1 2 3 4 5; do python tools/rundll.py build/dump-pf27.dll "$IDX" "$G14" 14 build/27b/pf-r$r.f32; done
cmp build/27b/pf-r1.f32 build/27b/pf-r2.f32 && cmp build/27b/pf-r1.f32 build/27b/pf-r3.f32 \
 && cmp build/27b/pf-r1.f32 build/27b/pf-r4.f32 && cmp build/27b/pf-r1.f32 build/27b/pf-r5.f32   # R5 gate A
python tools/reference27.py compare "$G14" build/27b/pf-r1.f32 --layer 3 --min 0.99999          # R5 gate B
python tools/reference27.py compare "$G14" build/27b/pf-seams.f32 --min 0.9999 --median 0.99999 # R6 gate
# --- R7: multistep ------------------------------------------------------------
python tools/mk.py dump-multistep27
python tools/rundll.py build/dump-multistep27.dll "$IDX" 760,3712,314,23470 build/27b/multi27.f32 | tee build/27b/r7-engine.txt
python tools/reference27.py multistep 760,3712,314,23470 build/27b/multi27.f32 --min 0.9999 | tee build/27b/r7.log
grep -q 'argmax 4/4' build/27b/r7.log
# --- R8: NLL ------------------------------------------------------------------
python tools/mk.py nll27
python tools/rundll.py build/nll27.dll "$IDX" "$G128" | tee build/27b/r8-engine.txt
python - <<'PY'  # |ΔNLL| < 0.02 (tighten 0.005 once R1-R7 green)
eng=float(open('build/27b/r8-engine.txt').read().split('NLL mean ')[1].split()[0])
ref=float(open('build/27b/golden-nll.txt').read().split('NLL mean ')[1].split()[0])
assert abs(eng-ref) < 0.02, (eng, ref)
PY
# --- R9: greedy 8 ---------------------------------------------------------------
python tools/mk.py generate27
python tools/rundll.py build/generate27.dll "$IDX" "$G14" 8 | tee build/27b/r9-engine.txt
diff <(grep -o 'ids:.*' build/27b/r9-engine.txt) <(grep 'generated ids' build/27b/golden-greedy.txt)  # 8/8 gate
# --- R10: endurance ---------------------------------------------------------------
python tools/chat.py "$IDX" "$MD" "<prompt A>" --max-new 1000 --stats | tee build/27b/r10.log
# gates: zero non-finite, self-NLL < 5 nat/token, seam norms within 3x of ref-seams.npy, tok/s >= 0.8 * placement
```

Notes for the executing session: `compare`/`multistep` subcommands and the
`--stats`/`--max-new` chat flags are the §2 work items (they do not exist
yet); `test-model`'s 27B mode and all `*27` mk.py targets (currently FUTURE,
mk.py:76-80) land with Phases A–D; every gate above is a hard stop per
AGENTS.md ("wait for coherent token parity before claiming the engine is
correct").

## 7. Sources (current-code claims)

src/dump_pf.cu:28,33-40; src/dump_multistep.cu:14,31-36,42-47,63-65;
src/dump_layers.cu:5; src/dump_i4_seams.cu:22-70; src/dump_i4_chunk.cu:19-33
(N3),95-161; src/dump_attention.cu:10; src/dump_layer0.cu:5;
src/dump_layer3.cu:5; src/decode.cu:14-26 (N4),31-41 (N7),47,117-121,154-155
(H7); src/qwen35.cu:7-29; src/attention.cu:7; src/prefill.cu:107;
src/nll.cu:15-41,78-81; src/generate.cu:18-44; src/mxfp4_i4.cu:237-238;
src/mxfp4.cu:670-671; src/fp8.cu:52-55 (N1),96-103,186-190,198;
src/test_fp8.cu:11-17,50,130-146; src/model_file.cpp:23 (N2);
src/qwen_kernels.cu:5,67-68; src/streaming.cu:34-541;
include/insignia_qwen35.hpp:7 (N5); include/insignia_qwen_kernels.cuh:5;
include/insignia_fp8.cuh:23-26; tools/mk.py:20-29,38-81,183-195;
tools/index27.py:33-57,142-229,275-348; tools/reference27.py:37-57,130-140,
168-198,205-226,236-262,265-300,303-327,331-373,449-519,533-564;
tools/reference_pf_i4.py:34-35 (H1); tools/reference_multistep_i4.py:53-54;
tools/reference_all_layers_i4.py; tools/nll_compare.py:1-25; tools/chat.py:44;
tools/rundll.py; Qwen3.8-27B-FP8/crc32.txt; build/qwen38-27b-fp8.insignia-index
(header bytes read); audits/w3/parity-ladder.md §0-§9; audits/w3/diff-verify.md
H1-H8; audits/w3/MASTER-PLAN.md §2.4,§3 Phases 0-F; audits/w4/reference27.md;
audits/w3/reference27.md.
