# W4: full 9B-constant sweep of src/*.cu/.cpp + include/*.hpp/.cuh — THE Phase B work order

Date 2026-08-25. Every file in `src/` (27 files) and `include/` (13 files) read in full;
line numbers are CURRENT tree (post-w2: `mxfp4_i4.cu`, `gemm.cu` ab_i4, `streaming.cu`,
`io_bench.cu`, `insignia_cpu.hpp`, `insignia_streaming.hpp` all exist now; decode.cu
mtp_layer/spec machinery edited). Read-only audit; the only file written is this one.

**Classification legend** (per mission):
- **[MUST-27B]** — literal/behavior that breaks Qwen3.8-27B; must be edited (or the kernel
  redesigned/dispatched away) in Phase B/C.
- **[PARAM]** — already parameterized; the shape flows in from the caller / QuantMatrix.
  Includes "single-sourced through Qwen35Shape" (constant edit counted once, in the header).
- **[9B-ONLY]** — dump/test/bench tooling for the 9B; acceptable to leave, EXCEPT the ones
  the parity ladder needs (flagged **(R-gate)**: Phase B item 5 requires their update).
- **[UNRELATED]** — numeric coincidence with model dims (grid caps, sector sizes, vocab-arg
  plumbing), or 27B-native code.

**Counting rule**: one site = one literal/role occurrence that must change (finer than w2's
bundled rows). "Stays" anchors (values verified unchanged at 27B) are listed separately —
they are load-bearing and must be re-verified, not ignored.

**Dimension map** (ground truth: w2 loader census + attn-27b §0 + deltanet-27b §0):
hidden 4096→5120, inter 12288→17408, layers 32→64 (48 delta + 16 full), qkv row 8192→10240
(q|k 2048 each **stay**, v 4096→6144), z rows 4096→6144, q heads 16→24 (q_proj rows
8192→12288, per-head 512 interleave **stays**), kv heads 4 (**stay**, cache row 1024
**stays**), GQA group 4→**6 (`kvh=head/6`, NOT `>>2`)**, a/b rows 32→48, delta v:k 2:1→
**3:1 (`kh=head/3`, NOT `>>1`)**, delta k-heads 16 (**stay**), head dims 128/128 & 256
(**stay**), conv 8192→10240 (width 4 stays), full-attn layers 8→16 (`l/4` spans 0..15),
vocab 248320 (stays), rope 64/256 theta 1e7 (stays), mtp.fc [4096,8192]→[5120,10240].

---

## 0. Corrections to w2 shape-constants.md (verified in current source)

1. **`kvh = head>>2` is WRONG at 27B — confirmed still in the tree**, exactly as attn-27b
   §0 says: `src/attention.cu:7` and `src/prefill.cu:103` (gqa_prefill). w2 lines 169/365
   ("stays") are FALSE: 24 q / 4 kv ⇒ group 6 ⇒ `kvh=head/6`; `>>2` sends heads 16–23 OOB
   into the NEXT token's KV rows (silent future-attention) and past the layer slice at the
   last slot. Reclassified from "stays" to **[MUST-27B]** ×2.
2. **`kh = head>>1` → `head/3`** — `src/deltanet.cu:5` and `src/prefill.cu:221` (w2 had
   these right; unchanged classification).
3. **The three ab2 pair launchers are now GUARDED** (tree changed since w2):
   `mxfp4.cu:578` (`ab2_q8g`), `mxfp4.cu:670` (`ab2_q8`), `mxfp4_i4.cu:239` (`ab2_q8_i4`)
   all `throw` on `cols != 4096`. The failure is now LOUD, not silent — but the kernels are
   live engine paths (`decode.cu:68-69` pair in_proj_a/b), so they are still **[MUST-27B]**
   (redesign for 160 groups / 96 rows, or Phase C dispatches 27B to fp8/bf16 paths).
4. **NEW kernel not in w2**: `gemm.cu:461-546 mxfp4_gemm_ab_i4` ("a/b monster-killer") —
   the prefill a/b path that replaced w2's per-token GEMV sites (old decode.cu:72-73 are
   GONE). It bakes **exactly 32 rows** per tensor (store ldm 32, `<<<2,256>>>` grid)
   → 3 new **[MUST-27B]** sites.
5. **spec_rollback latent 9B bug is FIXED in this tree**: `prefill.cu:308-310` are now
   grid-stride loops (the old `threadIdx.x<4096`-in-a-256-thread-block partial copy is
   gone). Only the totals (24*32*128*128 / 24*8192*3 / 4096) still need the 27B values.
6. **gqa smem-race fix landed**: `attention.cu:7` and `prefill.cu:107` use dedicated
   `smx, sden` slots next to `red[8]` (the Phase 0 "second red[0] reuse" item). Preserve
   when porting grids 16→24.
7. **v21 GEMM tail-wait + fp8 T-guard landed** (Phase 0 items): `gemm.cu:275-276`,
   `gemm.cu:432-433`, `gemm.cu:524-525`, `fp8.cu:167-168` all do
   `if (kb+2<ksteps) wait_group 1 else wait_group 0`; `fp8.cu:184` store is guarded
   `wm*16 < T`, `fp8.cu:188` throws T>64; same guard in `gemm.cu:447`. test_fp8.cu has the
   bias-7 e4m3 host fix and the T=33/T=65 tests. (Not shape sites — noted so Phase B
   doesn't reintroduce them.)
8. **generate.cu:115 now THROWS** on ctx>4090 (was a clamp; safety C1 quick win landed).
9. `pf_xq8`/`pf_xs8` (decode.cu:26) are **dead staging** — allocated, freed, used by
   nothing (engine stages pair inputs in-block; `quantize_x8` is bench-only). 9B-sized
   6144/768; dormant, resize only if revived.

---

## 1. Engine headers

### include/insignia_qwen35.hpp (5 sites)
| Line | Literal | Meaning | Class | 27B |
|---|---|---|---|---|
| 7 | `hidden=4096` | hidden (single source; used by qwen35.cu embed cols) | **MUST-27B** | 5120 |
| 7 | `intermediate=12288` | inter (dead today — every loop hardcodes its own) | **MUST-27B** + WIRE | 17408 |
| 7 | `layers=32` | layers (dead today — loops hardcode 32) | **MUST-27B** + WIRE | 64 |
| 7 | `vocab=248320` | vocab | [PARAM] anchor | stays |
| 7 | `full_attention(i){(i&3)==3}` | full-attn interval | [PARAM] anchor | stays (now selects 16) |

### include/insignia_deltanet.cuh (2)
| 4 | `DELTA_HEADS=32` | delta v-heads (documentation only — kernels hardcode) | **MUST-27B** | 48 (+ add DELTA_QK_HEADS=16) |
| 4 | `DELTA_K=128, DELTA_V=128` | head dims | [PARAM] anchor | stays |

### include/insignia_decode.hpp (1)
| 7 | `max_context=4096` default | context cap, numerically == hidden by coincidence; bounded by `score[4096]` | [UNRELATED/CTX] | policy decision (keep ≤4096) |

### include/insignia_attention.cuh / prefill.cuh / qwen_kernels.cuh / ops.cuh / layout.cuh / fp8.cuh / model.hpp / storage.hpp
0 shape sites. **Dtype note (Phase A, not shape)**: `deltanet_parameters` and
`deltanet_params_batch` take `const float* A_log` — 27B ships BF16 A_log; needs
`const void* + bool a_log_f32` (deltanet-27b §4).

### include/insignia_cpu.hpp — 27B-NATIVE (0 MUST)
CPU tier written for 27B: `deltanet_step_cpu(heads=48, kshare=3)`, `qk_norm_rope_cpu
(qheads=24, kvheads=4)`, `split_q_gate_cpu(qheads=24)`, GQA group 6 hard-baked
(`h0=hg*6, hh<6, hg<4`, scratch heads=24, `kvh=h/6` — the CORRECT mapping), kvrow 1024.
All shapes flow in as defaulted args. [PARAM/27B-ready]. Hardcoded 6/24/4 would break a
9B CPU run — by design, fine.

### include/insignia_streaming.hpp — 27B-NATIVE (0 MUST)
Ring default `kDefaultSlotBytes = 184*2MiB = 368 MiB` sized for the 383.87 MB 27B linear
shards; `kSector=4096` is a disk constant [UNRELATED]. [PARAM/27B-ready].

---

## 2. src/decode.cu — 82 MUST-27B sites (the densest file)

### DecodeWorkspace ctor (lines 11-28) — 33 MUST
| Line | Buffer | 9B | Class | 27B |
|---|---|---|---|---|
| 12 | ctx guard `ctx>4096` | tied to score[4096] | [UNRELATED/CTX] | stays (policy) |
| 14 | hidden, norm | 4096 | **MUST** ×2 | 5120 |
| 14 | qkv | 8192 | **MUST** | 10240 |
| 14 | attn_gate | 4096 | **MUST** | **6144** (24×256, NOT 5120 — trap) |
| 14 | key, value | 1024 | [PARAM] stays ×2 | 4 kvh×256 |
| 14 | z | 4096 | **MUST** | **6144** (48×128 — trap) |
| 14 | a, b | 32 | **MUST** ×2 | 48 |
| 14 | core | 4096 | **MUST** | **6144** (max(48×128, 24×256)) |
| 14 | gate, up | 12288 | **MUST** ×2 | 17408 (also covers q_proj 12288 ✓) |
| 14 | down | 4096 | **MUST** | 5120 (also mtp embed output) |
| 14 | logits | 248320*2 | [PARAM] stays | vocab×pair |
| 14 | delta_state | 24·32·128·128 | **MUST** | 48·48·128·128 |
| 14 | conv_state | 24·8192·3 | **MUST** | 48·10240·3 |
| 14 | kv_keys, kv_values | `size_t(8)*ctx*1024` | **MUST** ×2 | **`size_t(16)*ctx*1024`** (16 full layers) |
| 14 | mtp_keys/values | ctx·1024 | [PARAM] stays ×2 | 1 mtp layer, 4×256 |
| 15 | pos_dev 16 ints / pf_tokens 64 | state slots / chunk cap | [UNRELATED] | stays |
| 18,20 | committed/host_committed 16384 | token-stream cap | [UNRELATED/CTX] | policy (safety C1) |
| 22 | pf_x, pf_n | 64·4096 | **MUST** ×2 | 64·5120 |
| 22 | pf_qkv | 64·8192 | **MUST** | 64·10240 |
| 22 | pf_scratch | 64·8192 | **MUST** | **64·12288** (q_proj rows, NOT conv's 10240 — trap) |
| 22 | pf_z | 64·4096 | **MUST** | 64·6144 |
| 23 | pf_q, pf_g | 64·4096 | **MUST** ×2 | 64·6144 |
| 23 | pf_k, pf_v | 64·1024 | [PARAM] stays ×2 | 4×256 |
| 23 | pf_core | 64·4096 | **MUST** | 64·6144 |
| 24 | pf_down | 64·4096 | **MUST** | 64·5120 |
| 24 | pf_gate, pf_up | 64·12288 | **MUST** ×2 | 64·17408 |
| 24 | pf_a, pf_b | 64·32 | **MUST** ×2 | 64·48 |
| 25 | snap_delta, snap_conv | 24·32·…/24·8192·3 | **MUST** ×2 | 48·48·…/48·10240·3 |
| 26 | pf_xq8 6144 u32, pf_xs8 768 f32 | dead pair staging | [9B-ONLY dormant] | 8704/1088 if revived |
| 26 | pf_bf16 | 64·12288·2B | **MUST** | 64·17408·2B |
| 27 | memsets delta/conv | 24·32·…·4 / 24·8192·3·4 | **MUST** ×2 | track allocs |

### Engine body — 49 MUST
| Line | Site(s) | Class → 27B |
|---|---|---|
| 47 | `for l<32` (prefill loop) | **MUST** → 64 |
| 49 | rmsnorm cols 4096 | **MUST** → 5120 |
| 59 | `ai=l/4` + `ai*max_context*1024` | [PARAM] stays (ai→0..15 via l<64; 1024 stays) |
| 62 | sigmoid_mul `T*4096` (attn out) | **MUST** → T·6144 |
| 66-77 | pair/batch a+b via `mxfp4_gemv_ab2_q8[_i4]` / `mxfp4_gemm_ab_i4(ma.cols)` | call sites [PARAM] — kernels carry the MUST (see those files) |
| 79 | conv_state `di*8192*3` + snap `di*8192*3` | **MUST** ×2 → 10240·3 |
| 82 | `(const float*)A.data` (params_batch) | **DTYPE (Phase A)** — 27B A_log is BF16 |
| 84 | delta_state `di*32*128*128` + snap | **MUST** ×2 → 48·128·128 |
| 85 | gated rmsnorm `T*32,128` | **MUST** → T·48 (128 stays) |
| 88,93 | residual `T*4096` ×2 | **MUST** ×2 → T·5120 |
| 89 | rmsnorm 4096 (post LN) | **MUST** → 5120 |
| 91 | silu `T*12288` | **MUST** → T·17408 |
| 96 | final model.norm 4096 | **MUST** → 5120 |
| 103,104 | lm_head input `pf_n+(T-1)*4096` (i4 + mlx arms) | **MUST** ×2 → 5120 |
| 106 | D2D copy `4096*4` | **MUST** → 5120·4 |
| 122-124 | (spec helper — no literals) | — |
| 126 | delta_layer guard `l>=32` + LN 4096 | **MUST** ×2 → 64, 5120 |
| 128 | conv n=`8192` + state `di*8192*3` | **MUST** ×2 → 10240 |
| 128 | `deltanet_parameters(...,32,...)` heads | **MUST** → 48 |
| 128 | `delta_state+di*32*128*128` | **MUST** → 48·128·128 |
| 128 | `qkv, qkv+2048, qkv+4096` | [PARAM] stays — q/k sections 2048 rows each; v head count covers 6144 |
| 128 | gated `32,128` | **MUST** → 48,128 |
| 128 | `(const float*)A.data` (decode params) | **DTYPE (Phase A)** |
| 129 | post-LN 4096; silu 12288; residual 4096 ×2 | **MUST** ×4 |
| 131 | attention_layer guard `l>=32`; LN 4096; sigmoid 4096; silu 12288; residual 4096 ×2 | **MUST** ×6 → 64; 5120; **6144**; 17408; 5120 ×2 |
| 133 | forward_body `l<32`; norm 4096 | **MUST** ×2 → 64, 5120 |
| 147,148 | mtp pre-fc norms 4096 ×2 | **MUST** ×2 → 5120 |
| 151 | `concat(...,4096)` | **MUST** → 5120 (fc cols 10240) |
| 154 | mtp.fc i4 gemv `4096,8192` | **MUST** → 5120,10240 |
| 155 | mtp.fc bf16_gemv `4096,8192` | **MUST** → 5120,10240 |
| 162 | mtp LN 4096 | **MUST** → 5120 |
| 172 | sigmoid_mul 4096 (mtp attn gate) | **MUST** → **6144** |
| 178,186 | residual 4096 ×2 | **MUST** ×2 → 5120 |
| 180 | mtp post-LN 4096 | **MUST** → 5120 |
| 184 | silu 12288 | **MUST** → 17408 |
| 188 | mtp.norm 4096 | **MUST** → 5120 |
| 238-249 / 255-265 | `capture_spec` / `capture_step` | **MUST** ×2 (re-capture after ANY dim change — graphs freeze grids/smem/pointers) |

decode.cu verify/stays anchors: key/value/logits/mtp_kv (5), pf_k/pf_v (2), qkv offsets
2048/4096 (2), ai formula (1), 16384 caps (2), ctx guard (1) ≈ 13.

---

## 3. src/attention.cu (2 MUST)
| Line | Site | Class → 27B |
|---|---|---|
| 7 | `kvh=head>>2` | **MUST** → `head/6` (attn-27b §0; compiler emits mul-magic) |
| 7 | `score[4096]` | [UNRELATED/CTX] stays (context cap) |
| 7 | `(size_t(t)*4+kvh)*256` K/V rows; `scale=.0625f` | [PARAM] stays (4×256, 1/√256) |
| 7 | `smx,sden` dedicated slots | race fix — preserve in port |
| 8 | `<<<16,256>>>` | **MUST** → `<<<24,256>>>` |

## 4. src/deltanet.cu (2 MUST)
| 5 | `kh=head>>1` | **MUST** → `head/3` |
| 5,10,12 | `kh*128`, `128` dims, state `head*128*128` | [PARAM] stays (16 k-heads, 128/128) |
| 8 | `0.08838834764831845f` | [PARAM] stays (=1/√128) |
| 14 | `<<<32,128>>>` | **MUST** → `<<<48,128>>>` |

## 5. src/ops.cu (3 MUST)
| 7 | silu grid cap `4096` | [UNRELATED] (max-blocks clamp) |
| 9 | `isq=head<16`; `k+(head-16)*256` | **MUST** ×2 → `<24`, `head-24` |
| 9 | theta 1e7, `tid<64`, `mem[64]`, `ss/256`, eps | [PARAM] stays (27B config verified) |
| 10 | `<<<20,256>>>` | **MUST** → `<<<28,256>>>` (24q+4k) |

## 6. src/qwen_kernels.cu (5 MUST + 1 dtype)
| 5-6 | `rms_bf` rows/cols generic | [PARAM] |
| 7-8 | `conv4` n-generic grid-stride | [PARAM] — only call-site literals (decode.cu:128) |
| 9-10 | `params` kernel `const float* A` + `<<<1,32>>>` | **DTYPE** + **MUST** → `<<<1,48>>>` |
| 15-16 | `store_kv` `i<1024`, `pos*1024`, `<<<4,256>>>` | [PARAM] stays (verified, attn-27b §5) |
| 61 | argmax `<<<64,512>>>` | [UNRELATED] (grid-stride) |
| 67-68 | `bf16_gemv(rows,cols)` | [PARAM] |
| 73-74 | `split_q_gate` `i<4096` + `<<<16,256>>>` | **MUST** ×2 → `i<6144`, `<<<24,…>>>` (`h=i>>8` stays) |
| 78-79 | `expand_gate` `i<4096` + `<<<16,256>>>` | **MUST** ×2 → `i<6144`, `<<<24,…>>>` |

## 7. src/prefill.cu (45 MUST + 1 dtype)
| Line | Site(s) | Class → 27B |
|---|---|---|
| 12 | `w + row*512` (u32/row) | **MUST** → 640 |
| 13 | `s[row*128+g]` | **MUST** → 160 |
| 14 | `out + t*4096` | **MUST** → 5120 |
| 22 | `<<<T,128>>>` (one thread/group) | **MUST** → `<<<T,160>>>` |
| 29-31,39 | i4 twin: `row*512`→640, `row*64+(g>>1)`→80, `t*4096`→5120, `<<<T,128>>>`→`<<<T,160>>>` | **MUST** ×4 |
| 44 | `t=blockIdx.x>>4, h=&15` | **MUST** ×2 → `/24`, `%24` (bit tricks illegal at 24) |
| 45 | `base=t*8192` (q_proj rows) | **MUST** → t·12288 |
| 46,47 | q/gate `t*4096` ×2 | **MUST** ×2 → t·6144 |
| 50 | `<<<T*16,…>>>` | **MUST** → T·24 |
| 56 | `isq=head<16` | **MUST** → 24 |
| 57 | q `(t*16+head)*256`; k `head-16` | **MUST** ×2 → `t*24`, `head-24` |
| 78 | theta/64/256 literals | [PARAM] stays |
| 85 | `dim3(20,T)` | **MUST** → `dim3(28,T)` |
| 90-93,97 | store_kv_batch `1024`, `dim3(4,T)` | [PARAM] stays |
| 103 | `kvh=head>>2` | **MUST** → `head/6` (the §0 correction) |
| 106 | `score[4096]` | [UNRELATED/CTX] |
| 109,160 | q/out `(t*16+head)*256` ×2 | **MUST** ×2 → `t*24` |
| 115,150 | `(j*4+kvh)*256` ×2 | [PARAM] stays |
| 164 | `dim3(16,T)` | **MUST** → `dim3(24,T)` |
| 171-179 | conv: `idx/8192`,`%8192`,`(t-(3-i))*8192`,`t*8192` | **MUST** ×4 → 10240 |
| 184-199 | roll: `c>=8192`, `j*8192`, `n=T*8192`, grid `(8192+255)/256` | **MUST** ×4 → 10240 |
| 203 | `params_batch` `const float* A_log` | **DTYPE (Phase A)** |
| 205,206 | `h>=32`, `t*32` ×2 (ar/br) | **MUST** ×3 → 48 |
| 213 | `<<<T,32>>>` | **MUST** → `<<<T,48>>>` |
| 221 | `kh=head>>1` | **MUST** → `head/3` |
| 227-229 | qkv strides `t*8192` ×3 (q,k,v) | **MUST** ×3 → t·10240; offsets `2048`/`4096` **stay** |
| 239 | q-norm constant | [PARAM] stays |
| 244,245 | `a[t*32]`, `b[t*32]` | **MUST** ×2 → t·48 |
| 256 | out `(t*32+head)*128` | **MUST** → (t·48+head)·128 |
| 266 | smem opt-in `64*1024+512` | **[PARAM] STAYS — verified**: 128×128 f32 state staging + 512B; head dims unchanged at 27B, so **66,048 B is still the request** (fits the 99 KB sm_89 opt-in; 48 blocks ≤ 56 SMs = exactly 1 wave, same as 9B's 32) |
| 268 | `<<<32,128,64*1024+512>>>` | **MUST** → `<<<48,128,66,048>>>` (smem literal unchanged) |
| 305-310 | rollback totals `24*32*128*128`, `24*8192*3`, `4096` | **MUST** ×3 → 48·48·…, 48·10240·3, 5120 (grid-stride loops already correct) |

## 8. src/mxfp4.cu (11 MUST — all in the two ab2 pair kernels)
| 35,71,147,368 | `cols&31`/`cols&1023` guards + fallback | [PARAM] verify (5120/6144/10240/17408 all pass; mlx fallback only for non-%1024) |
| 90-155 | `mxfp4_gemv_v2(rows,cols)`, smem `cols*4+64` | [PARAM] (5120→20.8KB, 17408→69.7KB < 99KB ✓) |
| 159-222 | `mxfp4_gemv2_v2` pair fp32: smem `cols*2*4+64` = **139,264B > 99KB at 17408** | [9B-ONLY dormant] — no engine caller (bench only); would need dp4a-style staging at 27B |
| 215-221 | guard + smem | verify (throws loudly, nothing routes here) |
| 287-371 | `mxfp4_gemv2_q8(rows,cols)` dp4a pair | [PARAM] — engine pair path for MLX models |
| 408-464 / 467-513 | `q8g` pair / single prequant GEMVs | [PARAM] |
| 521-580 | `ab2_q8g`: `rr<32`, `row=rr-32`, `ya[32+row]`, `yb[32+row]`, guard `cols!=4096` (line 578), `<<<1,256>>>` | **MUST** ×5 — 96 rows > 64-row warp capacity; redesign (Phase C) or never route 27B |
| 589-673 | `ab2_q8`: staging `r=threadIdx.x>>7,g=&127` (bakes 128 groups = hidden 4096), `rr<32`, `row-32`, `ya[32+row]`, `yb[32+row]`, guard (line 670) | **MUST** ×6 — same redesign; staging breaks at 160 groups |

## 9. src/mxfp4_i4.cu (6 MUST)
| 16-75 | `gemv_v2_i4(rows,cols)` | [PARAM] (cols&1023 gate: 5120 ✓) |
| 78-154 | `gemv2_q8_i4(rows,cols)` dp4a pair | [PARAM] — LIVE spec path for INSIG4 models |
| 157-242 | `ab2_q8_i4`: staging `threadIdx.x>>7,&127` (163), `rr<32` (197), `row-32` (198), `ya[32+row]` (232), `yb[32+row]` (233), guard `cols!=4096` (239) | **MUST** ×6 — the LIVE `linear2` pair path (decode.cu:68); loud throw at 27B today |
| 245-256 | `get_row_i4(cols)` grid-stride, cap 128 blocks | [PARAM] safe |

## 10. src/gemm.cu (3 MUST + 6 verify)
| 78 | `rows&63 / cols&31` (mlx gemm) | verify: 27B rows all %64 ✓, cols all %32 ✓ |
| 182 | `rows&63 / cols&63` (v2) | verify ✓ |
| 275-276 | v21 tail cp.async wait | [UNRELATED] — race fix present, keep |
| 293-296 | `rows&31 / cols&64` (v21); A zero-padded 64 rows | verify ✓ — pf_bf16 growth counted in decode.cu:26 |
| 354-357 | mlx_i4 gates | verify ✓ |
| 371-454 | `v21_i4(rows,cols,T)` + guarded store + T>64 throw | [PARAM] |
| 461-546 | **`gemm_ab_i4`** — "EXACTLY 32 rows baked": store `y + wm*16*32 + wn*16, acc, 32` (line 539, ×2 literals: offset+ldm), `<<<2,256>>>` (545) | **MUST** ×3 — 48 rows needs grid 3 blocks or NT=48/64; ya/yb stride `T,32`→`T,48` (buffers counted in decode.cu) |
| 543-545 | guards `cols&1023`, T 1..64 | verify ✓ |

## 11. src/fp8.cu (0 MUST — fully parameterized)
| 53,97,187 | `cols&127` gates | verify ✓ (5120/6144/10240/17408 all %128) |
| 98-99 | gemv2 smem `2*cols*4` + 99KB throw | verify ✓ (cols≤10240 ⇒ ≤80KB; nothing calls it with cols=17408 — inter mats have cols=5120) |
| 184,188 | guarded store + T>64 throw | [UNRELATED] — F1 fix present |
| all kernels | rows/cols/T args | [PARAM] — 27B dispatch target per Phase C |

## 12. src/qwen35.cu (0 MUST)
Single-sourced: `embed_dev` passes `Qwen35Shape::hidden` (constant edit in the header);
`matrix()` kinds mlx/i4/fp8/bf16 with shape checks `[ceil(r/128)][ceil(c/128)]` — [PARAM].
Note: fp8 scale acquired as `base+".scales"`; Phase A wires the `weight_scale_inv` name.

## 13. src/generate.cu / nll.cu (0 MUST)
generate.cu: 57 `x(4096)` [CTX]; 62 `64*vocab` [PARAM]; 115 `ctx>4090` throw [CTX policy];
142-143 mtp memset `ctx*1024` ×2 [PARAM stays]. nll.cu: 57 `x(4096)` [CTX]; `row_logp`
takes `vocab` arg [PARAM]. (Both carry 9B token-id expectations in golden comments only.)

## 14. Clean files (0 sites)
storage.cu, model_file.cpp (INSIDX01 — Phase A moves it), streaming.cu (4096 = sector
size [UNRELATED]; smoke reads the real 27B shards), io_bench.cu (already points at
`Qwen3.8-27B-FP8\layers-0..5`), generate_ids.cu (ctx=512), smoke.cu, dllshim.cu,
dllshim_c.cu.

---

## 15. Instrumentation — [9B-ONLY] unless flagged (R-gate)

| File | Sites | What changes (hidden→5120, qkv/conv→10240, z→6144, a/b→48, l<32→64, inter→17408) |
|---|---|---|
| dump_multistep.cu | 7 | h(4096) L30; `l<32` L42; copies `16384`→20480 L44,51; norm 4096 L49; header comment L10 **(R-gate R7)** |
| dump_layer0.cu | 1 | h(4096) **(R-gate R4)** |
| dump_layer3.cu | 1 | h(4096); `l<4` stays (layer 3 = first full attn at 27B too) **(R-gate R5)** |
| dump_layers.cu | 3 | `l<32`→64; h(4096); "32 layer seams" print **(R-gate R6)** |
| dump_attention.cu | 3 | hidden dumps 4096 (×~19 uses); gate/up 12288→17408 (×3); key/value 1024 stays; `gqa_decode(...,16,...)` = ctx, stays **(R-gate R5)** |
| dump_pf.cu | 4 | seam `T*4096` ×2 (L12,37); `x(4096)` CTX; "32 prefill seams" print **(R-gate R4/R6)** |
| dump_i4_seams.cu | 12 | hidden ×6; qkv 8192→10240 ×2 (L39,44); z 4096→**6144**; a/b 32→48 ×4; `deltanet_parameters(...,32)`→48; delta memset 24·32·…→48·48·…; gated 32→48; silu 12288→17408; rows_probe `{0,1,4096,8191}`→`{0,1,4096,10239}` (4096 stays = k|v boundary); `qkv+2048/qkv+4096` stay **(R-gate R4)** |
| dump_i4_chunk.cu | 12 | `2*4096` ×8; `8192*3`/`di*8192*3`→10240·3; state `32*128*128`→48·…; `2*12288` ×2; `T*32`→48; `l<32`→64; `ai*max_context*1024` stays; `64*248320`/`2*248320` stay **(R-gate R4/R7)** |
| test_i4.cu | 1 | `rows=8192,cols=4096` → 10240,5120 (kernels take dims; optional) |
| test_fp8.cu | 0 | already 27B (10240×5120); bias-7 + T=33/T=65 tests present |
| test_mxfp4.cu | 1 | cols=4096 (optional) |
| test_argmax.cu | 0 | n=248320 vocab anchor [PARAM] |
| test_attention.cu | 2 **(R-gate R5)** | `H=16`→24, reference `h/4`→**`h/6`**; ALSO STILL STALE: 5-arg `gqa_decode(dq,dk,dv,dy,T)` doesn't compile against the 8-arg signature — rewrite per attn-27b §9 (poisoned-KV per (t,kvh) check) |
| test_deltanet.cu | 3 **(R-gate R4)** | `H=32`→48, state H·K·V→48·…, reference `kh=h/2`→**`h/3`** (ratio change); KH=16 stays |
| test_checkpoint.cu | 1 | `x(4096),y(8192)`, gemv `8192,4096` → 10240,5120 |
| test_qwen35.cu | 2 | alloc 4096 ×2; assert `rows==8192&&cols==4096` → **12288 && 5120** (q_proj) |
| test_model.cpp | 1 | `tensors().size()==699` → 27B index count (Phase A) |
| test_full_model.cu | 3 | `l<32` ×2; h(4096); "all 32 layers" print |
| test_layer.cu | 1 | h(4096) |
| test_ops.cu | 1 | C=4096 (optional) |
| test_pair / pair_chain / prefill / generate | 0 | no model literals (ctx consts only) |
| test_mtp.cu | 0 | STALE: calls removed `mtp_draft` — does not compile [UNRELATED] |
| test_cpu.cpp | 0 | 27B-native (48/3, 24/6, 5120/6144/10240/17408) [PARAM] |
| bench_gemm.cu | 2 | shape tables `{8192,4096},{12288,4096},{4096,12288},{248320,4096}` ×2 (L9, L359) → `{10240,5120},{17408,5120},{5120,17408},{248320,5120}` (+{6144,5120}) |
| bench_mxfp4_mlx.cu | 6 | shape tables ×2; gates `rows==8192||rows==248320` (L514), `cols==4096` (L568); `gemv_f` staging `&127` assumes 128 groups (L276) + `cols==4096` gate; ab2 bench `dw+32*groups*4`, `ref[32+r]`, `ya/yb 64*4` → 48-row split, 96 alloc |

Instrumentation total ≈ **67 sites** (w2: 56 — delta = literal-level counting + dump_i4
twins). Mandatory-for-parity subset (R-gate): dump_layer0/layer3/layers/pf/attention/
multistep/i4_seams/i4_chunk + test_deltanet + test_attention ≈ 37.

---

## 16. Parameterization audit — who takes rows/cols vs who hardcodes (mission item)

**Already parameterized (shape flows in; only call-site literals change):**
- GEMV family: `mxfp4_gemv`, `mxfp4_gemv_mlx`, `mxfp4_gemv_v2`, `mxfp4_gemv2_v2`(bench-
  only), `mxfp4_gemv2_q8`, `mxfp4_gemv2_q8g`, `mxfp4_gemv_q8g`, `mxfp4_gemv_dp4a`,
  `mxfp4_gemv_v2_i4`, `mxfp4_gemv2_q8_i4`, `fp8_gemv`, `fp8_gemv2`, `bf16_gemv`,
  `mxfp4_get_row_mlx/i4`, `bf16_get_row`, `quantize_x8/q8_groups` — all (rows, cols).
- GEMM family: `mxfp4_gemm_mlx`, `mxfp4_gemm_v2`, `mxfp4_gemm_v21`, `mxfp4_gemm_mlx_i4`,
  `mxfp4_gemm_v21_i4`, `fp8_gemm` — all (rows, cols, T).
- Elementwise: `rmsnorm_bf16`, `gated_rmsnorm_bf16`, `rmsnorm_zero_centered`,
  `silu_mul`, `residual_add`, `sigmoid_mul`, `causal_conv4_silu(n)`, `concat(n)`,
  `argmax_logits/argmax_fast(n)` — all n/rows/cols.

**Hardcoded (kernel-side dims — every Phase B/C kernel edit):**
- Attention: `gqa_decode` (grid 16, kvh>>2), `gqa_prefill` (dim3(16,T), kvh>>2, t*16
  strides), `qk_norm_rope` (20-grid, <16), `qk_norm_rope_batch` (dim3(20,T), <16),
  `split_q_gate`/`expand_gate_heads` (16-grid, i<4096), `split_q_gate_batch` (>>4/&15,
  8192/4096 strides), `store_kv`/`store_kv_batch` (1024 — stays by verification).
- DeltaNet: `deltanet_decode` (grid 32, kh>>1), `deltanet_prefill` (grid 32, kh>>1,
  t*8192, t*32), `params`/`params_batch` (32 heads + F32 A_log).
- Embed: `embed_gather`/`embed_gather_i4` (512 words/row, 128/64 scale groups, 128
  threads, t*4096 out — all hidden/32 or /64 in disguise).
- Spec machinery: `spec_rollback` (24·32·128·128, 24·8192·3, 4096).
- ab2 pair family: `mxfp4_gemv_ab2_q8`/`_q8g`/`_q8_i4` (cols-only signature; 128-group
  staging + 32+32 rows; guarded) and `mxfp4_gemm_ab_i4` (T, cols — but 32-row bake).

## 17. Mission's specific questions, answered
- **Deltanet prefill smem**: the scan is in `src/prefill.cu:266-268` (not deltanet.cu).
  Dynamic smem = `64*1024+512` = **66,048 B — UNCHANGED at 27B** (128×128 state staging is
  per-head; head dims stay 128/128). Only the grid moves 32→48 (≤56 SMs = 1 wave). The
  `cudaFuncSetAttribute` opt-in (prefill.cu:266) stays as-is.
- **KV cache sizing**: `decode.cu:14` — `size_t(8)*ctx*1024` ×2 must become
  **`size_t(16)*ctx*1024`** (16 full-attn layers × 4 kvh × 256 = 1024 floats/token/layer);
  `mtp_keys/mtp_values ctx*1024` stay (1 MTP layer); `ai=l/4` formula stays, range 0..15.
- **Corrected totals**: see below.

---

## 18. Totals (corrected, vs w2's 148 engine / 56 instrumentation / 204 grand)

| Scope | MUST-27B (this sweep) | w2 estimate | Delta drivers |
|---|---|---|---|
| include/ engine headers | 4 (qwen35.hpp 3 + deltanet.cuh 1) | 7 | dtype/CTX sites moved out |
| src/decode.cu | 82 | 73 | literal-level counting; +2 graph re-captures kept |
| src/prefill.cu | 45 | 31 | literal-level split of conv/scan/rollback rows |
| src/attention.cu | 2 | 5 | `kvh>>2` "stays"→MUST (w2 error); score stays |
| src/deltanet.cu | 2 | 4 | anchors separated |
| src/ops.cu | 3 | 3 | — |
| src/qwen_kernels.cu | 5 | 7 | store_kv/argmax anchors separated |
| src/mxfp4.cu | 11 | 5 | ab2 internals itemized; now guarded (loud, not silent) |
| src/mxfp4_i4.cu | 6 | 4 | guard + staging itemized |
| src/gemm.cu | 3 | 3(verify) | **NEW `gemm_ab_i4` 32-row bake (post-w2 kernel)** |
| src/fp8.cu / qwen35.cu / generate.cu / nll.cu | 0 | 5 | all verify/CTX |
| **Engine total** | **163** | **148** | +2 kvh/6 corrections, +3 new ab_i4 kernel, finer literal counting, mtp.fc branch split |
| A_log BF16 dtype sites (Phase A) | 4 (decode 82+128, qwen_kernels 9-10, prefill 203) | (separate) | |
| Instrumentation | ≈67 (37 R-gate) | 56 | literal-level; i4 dumps itemized |
| **Grand total** | **≈230 (163 engine + 4 dtype + 67 instr)** | 204 | |

**Bottom line for Phase B planning**: the honest engine edit count is **163 MUST-27B
sites** (not 148) plus 4 A_log dtype sites and ~55 verify anchors; ~37 instrumentation
sites are parity-gate-mandatory. The five silent-corruption clusters (unchanged ranking,
updated locations): (1) `kvh>>2` ×2 → `/6` (attention.cu:7, prefill.cu:103); (2) `kh>>1`
×2 → `/3` (deltanet.cu:5, prefill.cu:221); (3) DecodeWorkspace trap values (6144/48/
48·48/16·ctx/64·12288 pf_scratch); (4) 24-head non-pow2 grids + `>>4/&15` bit tricks;
(5) embed gather strides (512/128/64/128-threads). The ab2 family now fails LOUD
(guards) instead of silently — redesign or dispatch remains Phase C work.
