# W2 Audit: Model-shape constants blocking Qwen3.5-9B -> Qwen3.8-27B

Sweep of every file in `src/` and `include/` (.cu/.cuh/.hpp/.cpp). Verified 27B dims
against `E:\coding\Insignia\Qwen3.8-27B-FP8\config.json` (text_config):
hidden 5120, inter 17408, 64 layers (16 full-attn at l%4==3), vocab 248320 (unchanged),
head_dim 256, 24 q / 4 kv heads, linear attention 48 value heads / **16 key heads** /
128+128 dims, conv kernel 4, rope_theta 10000000, partial_rotary_factor 0.25 (64 of 256),
mtp 1 layer, mtp.fc [5120, 10240].

**Dimension map (9B -> 27B)**

| Dim | 9B | 27B |
|---|---|---|
| hidden | 4096 | 5120 |
| layers | 32 | 64 |
| intermediate | 12288 | 17408 |
| vocab | 248320 | 248320 (no change) |
| delta layers (l%4!=3) | 24 | 48 |
| delta v-heads (a/b rows, in_proj_z rows) | 32 | 48 |
| delta q/k heads | 16/16 | 16/16 (unchanged!) |
| delta v:k head ratio | 2 (`kh=head>>1`) | **3 (`kh=head/3`)** |
| qkv rows (conv channels) | 8192 = q2048\|k2048\|v4096 | 10240 = q2048\|k2048\|v6144 |
| in_proj_z rows | 4096 | 6144 |
| full-attn q heads | 16 | 24 (GQA group 4 -> 6) |
| full-attn kv heads | 4 (1024 cache row) | 4 (unchanged) |
| full-attn layers | 8 (l/4 in 0..7) | 16 (l/4 in 0..15) |
| attn out / gate per token | 4096 (16x256) | 6144 (24x256) |
| q_proj rows (q+gate interleaved) | 8192 | 12288 |
| mtp.fc | [4096, 8192] | [5120, 10240] |
| per-head dims 128/256, conv width 4, interval 4 | - | unchanged |

Counting rule: a **site** = one literal/role occurrence that must be edited or re-verified
for 27B (grouped buffers count per buffer; "stays" anchors are counted because they are
load-bearing assumptions that must be re-verified, not ignored).

Legend for category: **K**=kernel assumption, **W**=workspace alloc (DecodeWorkspace,
decode.cu 10-29), **KV**=KV/cache sizing, **IDX**=index naming/validation, **G**=graph
capture, **T**=test/dump instrumentation, **CTX**=context-length policy (coincides with
model numbers but is not a model dim).

---

## include/insignia_qwen35.hpp (5 sites)

| Line | Literal | Dim | Cat | 27B value |
|---|---|---|---|---|
| 7 | `hidden=4096` | hidden | K/W (single source used by qwen35.cu embed) | 5120 |
| 7 | `intermediate=12288` | inter | reference only (not used elsewhere!) | 17408 |
| 7 | `layers=32` | layers | reference only (loops hardcode 32 separately!) | 64 |
| 7 | `vocab=248320` | vocab | IDX/validation | 248320 (no change) |
| 7 | `full_attention(i){return (i&3)==3;}` | interval 4 | K | unchanged, but now selects 16 layers (0..63) |

Note: `layers`/`intermediate` here are dead constants today — every loop hardcodes its own
32/12288. Either wire them up or they will drift.

## include/insignia_decode.hpp (1 site)

| Line | Literal | Dim | Cat | 27B value |
|---|---|---|---|---|
| 7 | `max_context=4096` | context cap | CTX/KV | numeric coincidence with hidden; bounded by `score[4096]` smem in attention.cu/prefill.cu. Independent decision; not hidden. |

## include/insignia_deltanet.cuh (2 sites)

| Line | Literal | Dim | Cat | 27B value |
|---|---|---|---|---|
| 4 | `DELTA_HEADS=32` | delta v-heads | K (but NOT used by kernels - they hardcode 32/48 inline) | 48 |
| 4 | `DELTA_K=128, DELTA_V=128` | head dims | K | unchanged (anchor) |

Missing constant: no DELTA_QK_HEADS exists; the 2:1 v:k ratio is baked into kernels (see
deltanet.cu / prefill.cu below) and becomes 3:1 for 27B.

## src/decode.cu (73 sites)

### DecodeWorkspace ctor (lines 11-28) — 43 sites, all W unless noted

| Line | Buffer | 9B | 27B | Dim |
|---|---|---|---|---|
| 12 | ctx guard `ctx>4096` | 4096 | tied to score[4096] smem | CTX |
| 14 | `hidden` | 4096 | 5120 | hidden |
| 14 | `norm` | 4096 | 5120 | hidden |
| 14 | `qkv` | 8192 | 10240 | in_proj_qkv rows |
| 14 | `attn_gate` | 4096 | 6144 | q heads 16->24 x256 |
| 14 | `key` | 1024 | 1024 (stays) | 4 kv heads x256 |
| 14 | `value` | 1024 | 1024 (stays) | 4 kv heads x256 |
| 14 | `z` | 4096 | 6144 | in_proj_z rows = 48x128 |
| 14 | `a` | 32 | 48 | delta heads |
| 14 | `b` | 32 | 48 | delta heads |
| 14 | `core` | 4096 | 6144 | max(delta 48x128, attn 24x256) = 6144 |
| 14 | `gate` | 12288 | 17408 | inter (also receives q_proj rows 12288 for 9B / would need 12288<=17408 for 27B q_proj=12288: fits) |
| 14 | `up` | 12288 | 17408 | inter |
| 14 | `down` | 4096 | 5120 | hidden (also mtp embed output) |
| 14 | `logits` | 248320*2 | unchanged | vocab x pair rows |
| 14 | `delta_state` | 24*32*128*128 | 48*48*128*128 | delta layers x v-heads |
| 14 | `conv_state` | 24*8192*3 | 48*10240*3 | delta layers x conv channels x (width-1) |
| 14 | `kv_keys` | 8*ctx*1024 | 16*ctx*1024 | KV: full-attn layer count x 4 kv x256 |
| 14 | `kv_values` | 8*ctx*1024 | 16*ctx*1024 | KV |
| 14 | `mtp_keys` | ctx*1024 | unchanged | KV: 1 mtp layer, 4 kv x256 |
| 14 | `mtp_values` | ctx*1024 | unchanged | KV |
| 22 | `pf_x` | 64*4096 | 64*5120 | hidden x chunk 64 |
| 22 | `pf_n` | 64*4096 | 64*5120 | hidden |
| 22 | `pf_qkv` | 64*8192 | 64*10240 | in_proj_qkv rows |
| 22 | `pf_scratch` | 64*8192 | 64*12288 | max(conv 10240, q_proj rows 12288) |
| 22 | `pf_z` | 64*4096 | 64*6144 | in_proj_z rows |
| 23 | `pf_q` | 64*4096 | 64*6144 | 24 q heads x256 |
| 23 | `pf_g` | 64*4096 | 64*6144 | 24 gate x256 |
| 23 | `pf_k` | 64*1024 | unchanged | 4 kv x256 |
| 23 | `pf_v` | 64*1024 | unchanged | 4 kv x256 |
| 23 | `pf_core` | 64*4096 | 64*6144 | attn out rows |
| 24 | `pf_down` | 64*4096 | 64*5120 | hidden |
| 24 | `pf_gate` | 64*12288 | 64*17408 | inter |
| 24 | `pf_up` | 64*12288 | 64*17408 | inter |
| 24 | `pf_a` | 64*32 | 64*48 | delta heads |
| 24 | `pf_b` | 64*32 | 64*48 | delta heads |
| 25 | `snap_delta` | 24*32*128*128 | 48*48*128*128 | spec rollback = delta_state size |
| 25 | `snap_conv` | 24*8192*3 | 48*10240*3 | = conv_state size |
| 26 | `pf_xq8` | 6144 u32 | 8704 u32 (2 rows x 544 groups x 8) | int8 pair staging sized for inter 17408; currently unused by engine path - verify before relying |
| 26 | `pf_xs8` | 768 f32 | 1088 (2 x 544) | staging scales |
| 26 | `pf_bf16` | 64*12288*2 B | 64*17408*2 B | bf16 GEMM A-scratch for inter (biggest cols) |
| 27 | memset `delta_state` | 24*32*128*128*4 | 48*48*128*128*4 | must track alloc |
| 27 | memset `conv_state` | 24*8192*3*4 | 48*10240*3*4 | must track alloc |

### Engine body — 30 sites (K unless noted)

| Line | Literal(s) | Dim | Cat | 27B |
|---|---|---|---|---|
| 47 | `for l<32` | layers | K (prefill loop) | 64 |
| 49 | rmsnorm cols 4096 | hidden | K | 5120 |
| 59 | `ai=l/4`, stride `ai*max_context*1024` | attn layer idx; KV row width | KV | formula stays; ai range 0..15; 1024 stays |
| 62 | sigmoid_mul `T*4096` | attn out | K | T*6144 |
| 72 | `pf_n+t*4096`, `pf_a+t*32` (in_proj_a loop) | hidden, heads | K | 5120, 48 |
| 73 | `pf_n+t*4096`, `pf_b+t*32` (in_proj_b loop) | hidden, heads | K | 5120, 48 |
| 75 | `conv_state+di*8192*3`, `snap_conv+di*8192*3` | conv channels/layer stride | K | 10240*3 |
| 80 | `delta_state+di*32*128*128`, snap | per-layer delta state stride | K | 48*128*128 |
| 81 | gated_rmsnorm rows `T*32`, head 128 | delta heads | K | T*48 (128 stays) |
| 84 | residual `T*4096` | hidden | K | T*5120 |
| 85 | rmsnorm 4096 | hidden | K | 5120 |
| 87 | silu_mul `T*12288` | inter | K | T*17408 |
| 89 | residual `T*4096` | hidden | K | T*5120 |
| 92 | final rmsnorm 4096 | hidden | K | 5120 |
| 99-100 | lm_head input `pf_n+(T-1)*4096` | hidden | K | 5120 |
| 102 | D2D copy `4096*4` | hidden | K | 5120*4 |
| 122 | `l>=32` guard, rmsnorm 4096 | layers, hidden | K | 64, 5120 |
| 124 | conv `8192`, `di*8192*3` | conv channels | K | 10240 |
| 124 | `deltanet_parameters(...,32,...)` | delta heads | K | 48 |
| 124 | `delta_state+di*32*128*128` | state stride | K | 48*128*128 |
| 124 | `deltanet_decode(qkv, qkv+2048, qkv+4096, ...)` | qkv section offsets q=2048, q+k=4096 | K | **stays** (q,k rows unchanged); v section size 4096->6144 handled by head count |
| 124 | gated_rmsnorm `32,128` | heads | K | 48,128 |
| 125 | rmsnorm 4096, silu 12288 | hidden, inter | K | 5120, 17408 |
| 127 | attention_layer: guard `l>=32`, LN 4096, `ai*max_context*1024`, sigmoid 4096, silu 12288 | layers/hidden/KV/attn-out/inter | K/KV | 64, 5120, ai->0..15 (1024 stays), 6144, 17408 |
| 129 | forward_body `l<32`, norm 4096 | layers, hidden | K/G (body is captured in graph) | 64, 5120 |
| 143-144 | mtp pre-fc norms 4096 (x2) | hidden | K | 5120 |
| 147 | `concat(...,4096)` | hidden (mtp.fc input = 2x hidden) | K | 5120 (fc cols 10240) |
| 150-151 | mtp.fc gemv `4096,8192` | fc rows, fc cols | K | 5120,10240 |
| 158/176/184 | mtp LNs 4096 | hidden | K | 5120 |
| 172 | sigmoid_mul 4096 | attn out (mtp has 24 heads too) | K | 6144 |
| 180 | silu 12288 | inter | K | 17408 |
| 182 | residual 4096 | hidden | K | 5120 |
| 249-259 | `capture_step` | G: graph freezes every launch above (grid dims, smem sizes) | G | re-capture mandatory after any dim change |
| 232-243 | `capture_spec` | G: same for spec graph | G | re-capture |

(33 bullet rows; several rows bundle 2 literals of one statement — counted as the listed
site total 73 = 43 W + 30 K.)

## src/attention.cu (5 sites)

| Line | Literal | Dim | Cat | 27B |
|---|---|---|---|---|
| 7 | `__shared__ float score[4096]` | max context (NOT hidden — numeric coincidence) | CTX/K | stays unless context cap changes; bounds DecodeWorkspace ctx<=4096 |
| 7 | `kvh=head>>2` | 4 kv heads | K | stays (24/4=6 group works) |
| 7 | `kc[(size_t(t)*4+kvh)*256+d]`, `vc[...+tid]` | KV layout 4 kv x256 | K/KV | stays |
| 7 | `scale=.0625f` | 1/sqrt(256) | K | stays |
| 8 | launch `<<<16,256>>>` | 16 q heads | K | **24** |

## src/deltanet.cu (4 sites)

| Line | Literal | Dim | Cat | 27B |
|---|---|---|---|---|
| 5 | `kh=head>>1` | v:k head ratio 2 | K | **BREAKS**: 48 v / 16 k => `kh=head/3` (or table) |
| 5 | `q16[kh*128+tid]`, `k16[kh*128+...]`, `v[head*128+tid]` | 128 head dims | K | stays |
| 8 | `0.08838834764831845f` | 1/sqrt(128)/sqrt(2) q-norm scale | K | stays |
| 14 | launch `<<<32,128>>>` | 32 v heads | K | **48** |

## src/prefill.cu (31 sites)

| Line | Literal | Dim | Cat | 27B |
|---|---|---|---|---|
| 12 | `w + row*512` | 512 u32/row = hidden 4096 | K | **640** (5120 cols) |
| 13 | `s[row*128+g]` | 128 scale groups = hidden/32 | K | **160** |
| 14 | `out + t*4096` | hidden | K | 5120 |
| 22/39 | launch `<<<T,128>>>` (both embed kernels) | one thread per group | K | **T,160** |
| 29 | i4 `w + row*512` | hidden | K | 640 |
| 30 | `s + row*64 + (g>>1)` | 64 fp16 super-groups = hidden/64 | K | **80** |
| 31 | i4 out `t*4096` | hidden | K | 5120 |
| 44 | `t=blockIdx.x>>4, h=blockIdx.x&15` | 16 q heads (power-of-2 bit trick) | K | **24: &15/>>4 illegal** -> modulo/div by 24 |
| 45 | `base = t*8192 + h*512` | q_proj rows 8192; 512 = 256q+256gate per head | K | **t*12288** (h*512 stays) |
| 46-47 | `q[t*4096+h*256+d]`, gate same | q/gate stride | K | **t*6144** |
| 50 | launch `<<<T*16,256>>>` | q heads | K | **T*24** |
| 57 | `isq = head<16`; q ptr `(t*16+head)*256` | q heads | K | **24**, `(t*24+head)*256` |
| 57 | k ptr `(t*4+head-16)*256` | q-head offset into combined grid | K | **head-24** |
| 85 | launch `dim3(20,T)` | 16q+4k blocks | K | **dim3(28,T)** |
| 92-93 | store_kv `i>=1024`, `pos*1024` | KV row width | KV | stays |
| 97 | launch `dim3(4,T)` | kv heads | KV | stays |
| 103 | `kvh=head>>2` | 4 kv heads | K | stays |
| 107 | `__shared__ float score[4096]` | max context | CTX | as attention.cu |
| 109/160 | q row `(t*16+head)*256`, out same | q heads | K | **t*24+head** |
| 115/150 | kv rows `(size_t(j)*4+kvh)*256` | KV layout | KV | stays |
| 164 | launch `dim3(16,T)` | q heads | K | **dim3(24,T)** |
| 171-179 | `t*8192`, `idx/8192`, `%8192`, `w+c*4+i` | conv channels; conv width 4 | K | **10240** (width 4 stays) |
| 184-194 | `c>=8192`, `j*8192+c`, `state[c*3+i]` | conv channels, state 3 | K | 10240 |
| 196-199 | `n=T*8192`, launch `(8192+255)/256` | conv channels | K | T*10240 |
| 205-206 | `h>=32`, `a+t*32`, `b+t*32` | delta heads | K | **48** |
| 213 | launch `<<<T,32>>>` | delta heads | K | **T,48** |
| 221 | `kh=head>>1` | v:k ratio 2 | K | **BREAKS -> head/3** |
| 227-229 | `qkv + t*8192 + kh*128`, `+2048+kh*128`, `+4096+head*128` | qkv row stride; q/k/v section offsets | K | stride **10240**; offsets 2048 (q), 4096 (q+k) **stay**; v now 48x128 |
| 244-245 | `a[t*32+head]`, `b[t*32+head]` | heads | K | t*48 |
| 256 | out `(t*32+head)*128` | heads | K | **(t*48+head)*128** |
| 266-268 | smem `64*1024+512`, launch `<<<32,128>>>` | 128x128 state/head; 32 heads | K | smem stays; **<<<48,128>>>** |

Plus rollback (graph-captured):
| 307 | `n = 24*32*128*128` | delta state total | K/G | **48*48*128*128** |
| 309 | `24*8192*3` | conv state total | K/G | **48*10240*3** |
| 310 | `threadIdx.x<4096` hidden restore | hidden | K/G | **5120** |

## src/qwen_kernels.cu (7 sites)

| Line | Literal | Dim | Cat | 27B |
|---|---|---|---|---|
| 10 | `params<<<1,32>>>` | delta heads (guarded by n param, decode.cu passes 32) | K | <<<1,48>>> + caller passes 48 |
| 15-16 | store_kv `i<1024`, `pos*1024`, launch `<<<4,256>>>` | KV row width 4x256 | KV | stays |
| 73 | split_q_gate `i<4096`, `h=i>>8`, `src[h*512+d]`, `src[h*512+256+d]` | 16 heads x256; 512 per head | K | **i<6144** (h=i>>8 stays) |
| 74 | launch `<<<16,256>>>` | q heads | K | **24** |
| 78 | expand_gate `i<4096` | q heads x256 | K | **6144** |
| 79 | launch `<<<16,256>>>` | q heads | K | **24** |
| 24-25 | argmax comment "248K-vocab", grid <<<64,512>>> | vocab (grid unrelated) | IDX | vocab unchanged |

## src/ops.cu (3 sites)

| Line | Literal | Dim | Cat | 27B |
|---|---|---|---|---|
| 9 | `isq=head<16`, `k+(head-16)*256` | 16 q heads (+4 k blocks) | K | **24** / `head-24` |
| 10 | launch `<<<20,256>>>` | 16+4 blocks | K | **28** |
| 9 | rope base `10000000.0f`, `tid<64`, `mem[64]`, `ss/256` | rope_theta, partial rotary 64/256, head dim | K | all stay (verified in 27B config) |

## src/mxfp4.cu (5 sites)

| Line | Literal | Dim | Cat | 27B |
|---|---|---|---|---|
| 148 | `if (cols & 1023)` -> fallback to mlx kernel | fast-path needs cols%1024==0 | K | 5120/6144/10240/17408 all %1024==0 — **verify only** (silent 2x slowdown if not) |
| 215-221 | `cols & 1023` throw + smem `cols*2*4+64` | pair fp32 GEMV | K | 5120 OK; **17408 needs 139KB smem > 99KB cap -> launch fails** (bench-only path today; mxfp4_gemv2_q8 covers inter via dp4a) |
| 594 | `r = threadIdx.x>>7, g = threadIdx.x&127` (ab2_q8 kernel) | staging 2 rows x 128 groups = **hidden 4096** | K | **BREAKS at 5120 (160 groups): silently half-staged activations** |
| 537-540, 572-573 | `is_a = rr<32`, `row = rr-32`, `ya[32+row]`, `yb[32+row]`, comment "64 rows (32 a, then 32 b)" | a/b rows 32 each; 8 warps x 8 rows = 64-row capacity | K | **48+48=96 > 64: kernel redesign required** |
| 521-579 | same two defects in `mxfp4_gemv_ab2_q8g_kernel` (`rr<32`, 128-group assumption via xq layout) | a/b fused pair | K | same fix |

## src/mxfp4_i4.cu (4 sites)

| Line | Literal | Dim | Cat | 27B |
|---|---|---|---|---|
| 70 | `cols & 1023` throw | i4 fast GEMV divisibility | K | 5120 = 5*1024 OK — verify only |
| 163 | `r = threadIdx.x>>7, g = threadIdx.x&127` (ab2_i4 kernel) | staging = hidden 4096 | K | **BREAKS at 5120 — this is the LIVE spec-decode pair path (`linear2` -> `mxfp4_gemv_ab2_q8_i4`)** |
| 197-198, 232-233 | `rr<32`, `ya[32+row]`, `yb[32+row]` | a/b rows; 64-row capacity | K | **48 each, capacity 96 > 64 — redesign** |
| 253 | get_row_i4 grid cap 128 blocks | grid-stride loop | K | safe (loops) — verify only |

## src/gemm.cu (3 sites — all verify-only)

| Line | Check | Constraint | 27B dims |
|---|---|---|---|
| 78 | `mxfp4_gemm_mlx`: rows%64==0, cols%32==0, Y padded to 64 rows | tile | rows 5120/6144/10240/12288/17408/248320 all %64==0; cols 5120/10240/12288/17408 %32==0 — OK |
| 182 | `mxfp4_gemm_v2`: rows%64, cols%64 | tile | 5120%64=0, 17408%64=0 — OK |
| 293 | `mxfp4_gemm_v21`: rows%32, cols%64; A zero-padded to 64 rows | tile + W (pf_bf16 64xcols) | OK; pf_bf16 must grow to 64x17408 (counted in decode.cu) |

## src/fp8.cu (1 site — verify-only)

| Line | Check | 27B |
|---|---|---|
| 53/97/182 | `cols & 127` (fp8 128-col scale blocks) | 5120/10240 %128==0 — OK; `test_fp8.cu` already exercises 10240x5120 |

## src/qwen35.cu (0 sites)

Single-sourced through `Qwen35Shape::hidden/vocab` — the only file that is already clean.

## src/generate.cu (3 sites)

| Line | Literal | Cat | 27B |
|---|---|---|---|
| 57 | `DecodeWorkspace x(4096)` (nll mode) | CTX | context cap, not hidden |
| 113 | `if (ctx > 4090) ctx = 4090;` | CTX | hard 4090 context ceiling (score[4096] headroom); policy decision |
| 140-141 | `cudaMemsetAsync(x.mtp_keys, ..., ctx*1024)` x2 | KV | 1024 stays |

(6ull<<30 device budget is capacity, not shape — but 27B weights ~3x; revisit.)

## src/nll.cu (1 site)

| Line | Literal | Cat | 27B |
|---|---|---|---|
| 57 | `DecodeWorkspace x(4096)` | CTX | context cap |

## Instrumentation (src tests/dumps/benches) — 56 sites

| File | Sites | Model literals -> 27B |
|---|---|---|
| dump_multistep.cu | 5 | `h(4096)`->5120 (x2 uses); `l<32`->64; copy `16384`->20480 bytes; argmax `248320` stays; header comment "32 layer outputs...33 rows of 4096" |
| dump_layer0.cu | 1 | `h(4096)`->5120 |
| dump_layer3.cu | 1 | `h(4096)`->5120 (`l<4`, layer 3 still first full-attn — OK) |
| dump_layers.cu | 2 | `l<32`->64; `h(4096)`->5120; "32 layer seams" print |
| dump_attention.cu | 4 | hidden dumps `4096`->5120 (~9 uses); key/value `1024` stays; gate/up `12288`->17408 (x3); `gqa_decode(...,16,...)` is ctx=16 not heads |
| dump_pf.cu | 3 | seam `T*4096`->T*5120; `DecodeWorkspace(4096)` CTX; "32 prefill seams" print -> 64 |
| dump_i4_seams.cu | 12 | `4096`->5120 (hidden x6); `8192`->10240 (qkv/conv x3); z dump `4096`->6144; a/b dumps `32`->48 (x4); `deltanet_parameters(...,32)`->48; delta memset `24*32*128*128`->48*48*128*128; gated `32,128`->`48,128`; silu `12288`->17408; `rows_probe {0,1,4096,8191}`: 4096 stays (=q+k boundary), 8191->10239; `deltanet_decode(qkv, qkv+2048, qkv+4096)` offsets stay |
| dump_i4_chunk.cu | 12 | `2*4096`->2*5120 (x8); `8192*3`/`di*8192*3`->10240*3; `32*128*128` state->48*128*128; `2*12288`->2*17408 (x2); `T*32`/`pf_a+t*32`->48; `l<32`->64; `ai=l/4` stays (range 0..15); kv stride `ai*max_context*1024` stays; logits `64*248320`/`2*248320` stay (vocab) |
| test_i4.cu | 1 | `rows=8192, cols=4096` -> 10240,5120 (would then also cover conv-sized matrices) |
| test_fp8.cu | 0 | already 27B shapes (`rows=10240, cols=5120`) |
| test_mxfp4.cu | 1 | `cols=4096`->5120 optional |
| test_argmax.cu | 1 | `n=248320` stays (vocab anchor) |
| test_attention.cu | 1 | `H=16`->24 (reference uses `h/4` group -> `/6`); file also stale: calls 5-arg `gqa_decode`, no longer compiles |
| test_deltanet.cu | 1 | `H=32`->48, `KH=16` stays — reference `kh=h/2` -> `/3` (ratio change!) |
| test_checkpoint.cu | 1 | `x(4096), y(8192)`, gemv `8192,4096` -> 10240,5120 |
| test_qwen35.cu | 1 | alloc `4096*4`->5120*4; **assert `a.rows==8192 && a.cols==4096` -> 12288 && 5120** (q_proj rows) |
| test_model.cpp | 1 | **`m.tensors().size()==699` -> 27B index tensor count** (IDX validation; must match new index_safetensors output) |
| test_full_model.cu | 2 | `l<32` x2 loops ->64; `h(4096)`->5120; "all 32 layers" print |
| test_layer.cu | 1 | `h(4096)`->5120 |
| test_ops.cu | 1 | `C=4096`->5120 optional |
| bench_gemm.cu | 2 | shape tables `{8192,4096},{12288,4096},{4096,12288},{248320,4096}` -> `{10240,5120},{17408,5120},{5120,17408},{248320,5120}` (+{6144,5120}); gates `rows==8192||rows==248320`, `cols==4096` |
| bench_mxfp4_mlx.cu | 3 | shape table x2 (main + gemm bench); `gemv_f` explicitly "Assumes cols==4096 (groups==128)" (`threadIdx.x&127` staging); ab2 bench `dw+32*groups*4`, `ya/yb 64*4` -> 48-row split, 96 alloc |

(test_pair.cu, test_pair_chain.cu, test_prefill.cu, test_generate.cu, test_mtp.cu, smoke.cu,
generate_ids.cu, dllshim.cu, dllshim_c.cu, model_file.cpp, storage.cu: no model-shape
literals. test_mtp.cu uses removed `mtp_draft` API — stale, unrelated to shapes.)

---

## Totals

- Engine core (include/ + engine src): **148 sites** (decode.cu 73, prefill.cu 31,
  qwen_kernels 7, mxfp4 5, qwen35.hpp 5, attention 5, deltanet.cu 4, mxfp4_i4 4,
  gemm 3, ops 3, generate 3, decode.hpp 1, deltanet.cuh 2, nll 1, qwen35.cu 0)
- Instrumentation (dumps/tests/benches): **56 sites**
- **Grand total: 204 sites**

## Top risk clusters (silent-failure ordered)

1. **DeltaNet v:k head ratio** — `kh=head>>1` in src/deltanet.cu:5 and
   src/prefill.cu:221. 27B has 48 v-heads over 16 k-heads (3:1) vs 9B 2:1. Division
   shift reads the wrong k-vector for heads 32..47 -> garbage output, no crash.
2. **Fused in_proj_a+b pair kernels** — mxfp4_i4.cu:157-240 (`ab2_q8_i4`, the live
   `linear2` spec path) and mxfp4.cu:519-679 (`ab2_q8`/`ab2_q8g`): staging
   `threadIdx.x>>7 / &127` bakes hidden=4096 (128 groups), `rr<32`/`ya[32+row]` bake
   32 a/b rows, and 8 warps x 8 rows = 64-row capacity < 96 needed. Three coupled
   redesigns; wrong answers guaranteed at 5120/48.
3. **DecodeWorkspace alloc block** — decode.cu:12-27: 43 buffers; the traps are
   attn_gate/z/core/pf_q/pf_g 4096->6144 (not 5120!), a/b/pf_a/pf_b 32->48,
   delta_state/conv_state/snap 24x32 -> 48x48 and 24x8192 -> 48x10240, kv caches
   8->16 layers, pf_scratch 8192->12288 (must exceed q_proj 12288, not just conv
   10240), pf_bf16 12288->17408. Any miss = OOB writes into the next buffer.
4. **Q-head count 16 -> 24 (non-power-of-2)** — split_q_gate (`<<<16,256>>>`, `i<4096`,
   `h*512`), expand_gate_heads, split_q_gate_batch (`>>4`/`&15`/`t*8192`),
   qk_norm_rope (+batch, `head<16`, grids 20/28, 16/T -> 24/T), gqa_decode/prefill
   (`<<<16,256>>>`/`dim3(16,T)` -> 24). The `&15`/`>>4` bit tricks silently
   mis-decode at 24 heads.
5. **Embedding gather strides** — prefill.cu:12-31: `row*512` (u32/row), `row*128`
   (scale groups), `row*64` (i4 super-groups), block size 128 threads, out stride
   `t*4096` — all are hidden/32 or hidden/64 in disguise; every embedding read
   misindexes at 5120. Adjacent: index validation (test_model.cpp 699-tensor count,
   test_qwen35 8192x4096 assert) will reject the 27B index until updated.

Also note (verify-only but load-bearing): all 27B GEMV cols (5120/6144/10240/17408)
satisfy the `cols&1023`/`%64`/`%128` gates, and 24 q heads keep GQA `head>>2` valid;
gqa score buffers `score[4096]` are context caps (not hidden) — decide context policy
separately. CUDA graphs (capture_step/capture_spec) freeze every launch config: any dim
change requires deleting and re-capturing both graphs.
