# w4 — Speculative decode (MTP) for Qwen3.8-27B-FP8 in the tiered engine

Date 2026-08-25. Read-only audit; the only file written is this report.
Read: `AGENTS.md`, `audits/w3/MASTER-PLAN.md` (esp. §1.3, §2.2, Phase F), `audits/w3/spec-deepen.md`,
`audits/w3/spec-phase1.md`, `audits/w3/graph-hazards.md` (§2, §6c), `audits/w3/deltanet-27b.md` §5a,
`audits/w3/fp8-kernels.md` §F5, `audits/w4/tier-dispatch.md` (§4, §7), `audits/w4/engine27-gap.md` §0;
sources `src/decode.cu`, `src/prefill.cu`, `src/generate.cu`, `src/qwen_kernels.cu`, `src/attention.cu`,
`src/deltanet.cu`, `src/fp8.cu`, `src/mxfp4.cu` (pair kernel decls), `src/storage.cu`, `src/streaming.cu`,
`src/qwen35.cu`, `include/insignia_decode.hpp`, `include/insignia_prefill.cuh`, `include/insignia_qwen35.hpp`,
`include/insignia_streaming.hpp`, `tools/reference_multistep.py:90-135`, git `777f55f` (+ its parent tree),
and the 27B checkpoint headers themselves (`Qwen3.8-27B-FP8/{mtp,outside,layers-0,layers-3}.safetensors`,
`config.json`) — every 27B shape below was read out of the checkpoint this session.

27B ground truth (checkpoint-verified):

| object | shape / bytes | note |
|---|---|---|
| layers | 64, `full_attention_interval=4` → fulls at `(i&3)==3`, 16 full / 48 linear | config.json; `Qwen35Shape::full_attention` formula survives |
| full-attn | q_proj F8 `[12288,5120]` (24 q + 24 gate × 256), k/v `[1024,5120]` (4 kvh × 256), o `[5120,6144]` | layers-3 header |
| linear-attn | in_proj_qkv F8 `[10240,5120]` (q 2048 / k 2048 / v 6144), in_proj_z `[6144,5120]`, a/b **BF16** `[48,5120]`, conv1d BF16 `[10240,1,4]`, A_log/dt `[48]` | layers-0 header |
| mlp | gate/up `[17408,5120]`, down `[5120,17408]`, all F8 + bf16 128×128 `weight_scale_inv` | both headers |
| lm_head / embed | bf16 `[248320,5120]` each (2,542.8 MB) | outside.safetensors (6.007 GB incl. vision tower, skipped) |
| MTP shard | **477,202,224 B** on disk; `mtp.fc` **bf16 `[5120,10240]`** ([out,in]); 1 full-attn layer with the same 24/4 head geometry as main fulls; `mtp_num_hidden_layers=1` | mtp.safetensors header |
| vocab | 248,320 — **identical to 9B** | both |

---

## 1. The 9B spec machinery, piece by piece

### 1.1 Device-state plumbing (reusable nearly verbatim)

`pos_dev` = 16 ints allocated at `src/decode.cu:15` with aliases `token_dev=pos_dev+1 … mtp_pos_dev=pos_dev+7`
(DecodeWorkspace ctor `src/decode.cu:11-28`). Documented layout at `src/prefill.cu:275-276`:

| slot | alias | role |
|---|---|---|
| 0 | pos_dev | next unprocessed main position P |
| 1 | token_dev | pending token (occupies position P) |
| 2 | next_dev | MTP argmax dst (draft-first order); verify row-1 argmax t\*_1 after |
| 3 | next2_dev | verify row-0 argmax t\*_0 ("t2") |
| 4 | draft_dev | draft (copied from pos[2] by spec_setup) |
| 5 | count_dev | cursor into `committed[]` |
| 6 | accflag_dev | accept flag (0/1) — doubles as rollback selector |
| 7 | mtp_pos_dev | MTP KV slot for this step's invocation |

- `spec_prologue` (`src/prefill.cu:277-278`): `pos[7] = pos[0]-1`. **No shapes — reusable as-is.**
- `spec_setup` (`src/prefill.cu:280-285`): `pos[4]=pos[2]; pf_tokens=[pos[1],pos[4]]`. No shapes —
  reusable; generalized form in §5.
- `spec_commit` (`src/prefill.cu:287-302`): 1 thread; `acc = pos[4]==pos[3]`; accept → commit
  `[pending,draft]`, `pending←pos[2]`, count+=2; reject → commit `[pending]` only, `pending←pos[3]`,
  count+=1, **`pos[0] -= 1`** (prefill.cu:300 — undo the second row's addi). **No shapes — reusable
  verbatim at 27B for D=1/T=2.**
- `prime_spec` (`src/decode.cu:193-196`), `append_committed_host` (`src/decode.cu:197-206`, pinned
  staging slot at index 16383), `committed_count` (`src/decode.cu:207-214`, refreshes `x_.position`
  from device truth), `read_committed` (`src/decode.cu:215-218`): shape-agnostic host plumbing,
  reusable as-is.
- `addi_kernel` (`src/prefill.cu:271-272`): adds T to pos[0]. Reusable.

### 1.2 The step itself (`spec_step`, `src/decode.cu:219-237`)

`prime_spec(t0)` → `spec_prologue` → `mtp_layer()` → `spec_setup` → `prefill_chunk_device(pf_tokens, 2)`
→ `spec_commit` → `spec_rollback` → 5 small D2H readbacks (pos/pending/acc, count, last committed,
token — `src/decode.cu:227-236`). `capture_spec` (`src/decode.cu:238-249`) replays the identical body
inside `cudaStreamBeginCapture(ThreadLocal)`; `spec_graph_step` (`src/decode.cu:250-254`) launches it
and guesses `x_.position += 2` (corrected later by `committed_count`, decode.cu:212).

`generate.cu`: prefill loop (:123-129) → `append_committed_host` (:132) → `prime_spec(first)` (:133) →
probe variant (:134-144) → eager warmup `spec_step` (:145) → eager loop (:154-169) or `capture_spec`
(:171) + 4-replay batches (:179-190) → drain/trim (:193-199).

`mtp_layer()` (`src/decode.cu:137-192`): embed gather of `token_dev` (:139-144) → two rmsnorms →
`concat` at 4096 (`:151`, qwen_kernels.cu:69) → fc GEMV `bf16_gemv(w,qkv,hidden,4096,8192)` (`:152-157`)
→ the MTP full-attn layer at `mtp_pos_dev` with its own KV (`:159-186`: q/k/v linear, split_q_gate at
:165, qk-norm+rope :170, `store_kv` :173 **before** `gqa_decode` :174, gate sigmoid :175-176, o_proj,
MLP) → `mtp.norm` → shared `lm_head` GEMV + `argmax_fast` (:190-191, full-sweep draft — spec-deepen F1).

### 1.3 Verify path = `prefill_chunk_device(T=2)` (`src/decode.cu:43-109`)

`const bool pair = T==2` (`:50`): full-attn layers route q/k/v/o through `linear2` (dp4a pair kernel
`mxfp4_gemv2_q8`, `src/mxfp4.cu:287-366`, decl `include/insignia_layout.cuh:79`); linear layers pair
qkv/z via `linear2` and a/b via the fused `mxfp4_gemv_ab2_q8` (`src/decode.cu:66-70`); MLP pairs via
`linear2`. lm_head T==2 branch (`src/decode.cu:97-104`): one `mxfp4_gemv2_q8` weight pass writes
`logits[2][248320]`, then two `argmax_fast` (row0 → pos[3], row1 → pos[2]). Both weight-schemes read
each weight byte once for both rows (spec-deepen F2/F3). Hidden export `x_.hidden ← pf_x[T-1]`
(`:106`) + `addi(pos, T)` (`:107`).

### 1.4 Snapshots + rollback (the 777f55f machinery, §7 details)

- `snap_delta` alloc `24*32*128*128` f32 = 50.33 MB, `snap_conv` `24*8192*3` = 2.36 MB
  (`src/decode.cu:25`) — **9B-hardcoded** (24 linear layers × 32 v-heads; conv width 8192).
- `deltanet_prefill` snapshots the recurrent state **after row t==0** inside the row loop
  (`src/prefill.cu:258-261`, launch `:265-269` with `row0_snap` arg threaded from decode.cu:84).
- `conv_roll_state_kernel` writes the conv window **rolled to just after row 0** — `snap=[s1,s2,x0]`
  (`src/prefill.cu:184-195`) — then rolls the live state to end-of-chunk.
- `spec_rollback_kernel` (`src/prefill.cu:305-314`): `if (pos[6]) return;` (accept keeps live state);
  else grid-stride restore `delta_state ← snap_delta` (n = `24*32*128*128`), `conv_state ← snap_conv`
  (`24*8192*3`), `hidden ← pf_x[0..4095]`. All three loops are grid-stride in the current tree (the
  256-of-4096 partial-hidden bug that deltanet-27b §5a found in the old `if (blockIdx.x==0 &&
  threadIdx.x<4096)` form is already fixed here); only the **size literals** are 9B.

### 1.5 Shape audit — hardcoded vs reusable

**Reusable with zero or trivial change (shape-agnostic):**
`spec_prologue`, `spec_setup` (T=2 form), `spec_commit`, `addi`, `prime_spec`, `append_committed_host`,
`committed_count`, `read_committed`, `argmax_fast` (two-stage, `src/qwen_kernels.cu:25-63` — vocab is
the same 248,320), `concat` (qwen_kernels.cu:69), `sigmoid_mul`, `residual_add`, `causal_conv4_silu`
(n-generic; call-site literal 8192→10240, `src/decode.cu:128`), `store_kv`/`store_kv_batch`
(`src/qwen_kernels.cu:15-16`, `src/prefill.cu:88-98` — KV row is 4 kvh × 256 = **1024 floats in both
models**, MASTER-PLAN Phase C.1 "verified UNCHANGED"), `rmsnorm_bf16` (cols param; zero-center flag
flips to `Z=true` per MASTER-PLAN Phase A.7 — `linear_attn.norm` stays raw), `TieredStorage`
acquire/release/make_room (`src/storage.cu:8-10`), `LayerFeeder` (`src/streaming.cu:373/430/443/452`),
`Qwen35Shape::full_attention` cadence `(i&3)==3` (interval 4 in both configs).

**Hardcoded 9B shapes that must change (site → 27B value):**

| site (file:line) | 9B literal | 27B value |
|---|---|---|
| `decode.cu:14` hidden/norm/down/z/a/b/core/gate/up/qkv/attn_gate/key/value | 4096/12288/8192/32/32/1024 | 5120 / **6144** (attn_gate, z, core: 24 heads × 256) / 17408 / 10240 / **48** (a,b) / key,value stay 1024 |
| `decode.cu:14` delta_state, conv_state | `24*32*128*128`, `24*8192*3` | **`48*48*128*128`** (150,994,944 B), **`48*10240*3`** (5,898,240 B) |
| `decode.cu:14` kv_keys/kv_values | `8*ctx*1024` (8 fulls) | **`16*ctx*1024`** (16 fulls; 268.4 MB @ctx2048 f32) |
| `decode.cu:14` mtp_keys/values | `ctx*1024` | same (4 kvh × 256) — only memset grows with ctx |
| `decode.cu:25` snap_delta/snap_conv | `24*32*128*128` / `24*8192*3` | `48*48*128*128` / `48*10240*3` (151.0 + 5.9 MB) |
| `decode.cu:22-26` pf_* rows | 4096/8192/12288/32 widths @64 rows | 5120/10240/17408/48 per MASTER-PLAN Phase B trap list (`pf_qkv 64*10240`, `pf_gate/up 64*17408`, `pf_a/b 64*48`, `pf_bf16 64*17408*2B`, `pf_q 64*6144`, …) |
| `decode.cu:14` logits | `248320*2` | `248320*8` (T-headroom; vocab unchanged) |
| `decode.cu:15` pos_dev | 16 ints | widen to 32 (D=4 needs ~16 used; one-line) |
| `decode.cu:47` layer loop | `l<32` | `l<64` (wire the `Qwen35Shape::layers` constant — today dead) |
| `decode.cu:150-156` mtp fc | `concat(...,4096)`, `bf16_gemv(...,4096,8192)` | concat 5120, `bf16_gemv(w,x,hid,5120,10240)` — orientation `[out,in]` verified from the mtp.safetensors header |
| `decode.cu:165` split_q_gate | 4096 split, 16 heads | 6144 split, 24 heads (`split_q_gate_kernel` qwen_kernels.cu:73-74: `i<4096`→`i<6144`, `h<16`→`h<24`, grid 16→24) |
| `decode.cu:184` mlp silu | 12288 | 17408 |
| `prefill.cu:43-51` split_q_gate_batch | src stride `t*8192`, h&15, grid `T*16` | `t*12288`, 24 heads, grid `T*24`, q/gate dst `t*6144` |
| `prefill.cu:54-86` qk_norm_rope_batch | grid `dim3(20,T)`, `isq=head<16`, k dst `(t*4+head-16)` | grid `dim3(28,T)`, `isq=head<24`, k dst `(t*4+head-24)` (attn-27b §1) |
| `prefill.cu:102-165` gqa_prefill | grid `dim3(16,T)`, `kvh=head>>2` | grid `dim3(24,T)`, **`kvh=head/6`** (24/4=6; `>>2` sends heads 16–23 OOB — risk #2), q row `(t*24+head)*256`; `score[4096]` stays = ctx cap |
| `attention.cu:7-8` gqa_decode | grid 16, `kvh=head>>2` | grid 24, `kvh=head/6` |
| `deltanet.cu:5,14` | `kh=head>>1`, `<<<32,128>>>` | `kh=head/3` (16 k-heads), `<<<48,128>>>`; **qkv offsets 0/2048/4096 stay** (q 2048 / k 2048 / v 6144 = 10240) |
| `prefill.cu:219-269` deltanet_prefill | 32 blocks, qkv stride 8192, a/b `[t]*32` | 48 blocks, stride 10240, `[t]*48`; smem 66,048 B stays (1 wave on 56 SMs) |
| `prefill.cu:170-199` conv kernels | 8192 literals ×4 | 10240 (pure literals) |
| `prefill.cu:203-214` params_batch | 32 heads, `A_log` f32 | 48 heads, **A_log bf16 dispatch** (Phase A.6; α≈1 signature, risk #4) |
| `prefill.cu:305-314` spec_rollback | `24*32*128*128`, `24*8192*3`, 4096 | `48*48*128*128`, `48*10240*3`, 5120 (grid-stride already correct) |
| `decode.cu:31-32,99-100` linear/linear2 | mxfp4/dp4a pair (9B formats) | 27B: `linear`→`fp8_gemv`, pair→`fp8_gemv2`, batch→`fp8_gemm` (dispatch on `WKind`, hook already exists `src/qwen35.cu:19-28`); `linear2`/dp4a never fires (guards per Phase C.4) |

**27B-specific pair routing (verified from `src/fp8.cu`):** `fp8_gemv2` (fp8.cu:58-103) is exactly
T=2 — it stages `2*cols` smem (fp8.cu:61,98), keeps `acc0/acc1` chains off each loaded weight group
(fp8.cu:73-91), writes `y[row]` and `y[rows+row]` (fp8.cu:94). Smem ceiling after the in-tree 99 KB
opt-in (fp8.cu:99-100, fixes fp8-kernels F5): `2*cols*4 ≤ 99 KB → cols ≤ 12,672`. All 27B pair
matrices fit (cols 5120, 6144, 10240→80 KB ✓) **except `down_proj [5120,17408]` (136 KB ✗)** → route
down_proj (and any cols>12,672) through `fp8_gemm` with the Phase C.3 F1 guarded epilogue (write only
`t<T` rows — the raw 64-row store contract is at fp8.cu:182-184); weights still stream once, T=2 rides
the same pass. `a/b` are bf16 `[48,5120]` at 27B (0.49 MB each): two `bf16_gemv` calls (qwen_kernels.cu:67-68)
or the Phase C.4 `bf16_gemv_ab2_pair`; the 9B fused dp4a ab2 kernel is mxfp4-only and must be guarded off.

---

## 2. The 27B eager spec step (D=1 / T=2) — design

### 2.1 Why eager, and what "eager" changes

No CUDA graphs at 27B (MASTER-PLAN Phase D.4 / graph-hazards §6c: the spec graph references every
layer's weights — 25.65 GB — which can never be resident; launch overhead ≈ 3.5–5 ms of a 1.7 s step
≤ 0.3%). Consequences:

1. `capture_spec`/`spec_graph_step` are simply **not called** by the 27B driver (the 9B paths stay
   untouched — dual-instantiation via the `<QH>` template convention per Phase B).
2. The `x_.position += 2` mirror guess (decode.cu:253) and its drift problem disappear: the eager
   step reads true pos from the tail D2H every step (decode.cu:227-230 pattern, fused into one
   readback).
3. The graph-bypassed KV-full hazard (graph-hazards §2) becomes a plain per-step host check (§3.3).

### 2.2 Order change: draft at the TAIL (not before the verify)

9B runs draft→verify (draft-first) because everything is device-resident and graph-replayed. At 27B
the draft must run **after** commit/rollback, at the step tail, for one structural reason: **embed is
host-side** (NVMe row-pread, 10 KB/token, MASTER-PLAN §2.5) and the verify's two embed rows must be
pread by the host — the draft id is only known after the draft's argmax, so a draft-first order would
force a mid-step D2H+pread+H2D round trip (host sync inside the treadmill). Tail order batches it into
the single step-end readback. This matches the sibling schedule in `audits/w4/tier-dispatch.md` §4
(draft ‖ next epoch's reader fill, one D2H `[next2,next]`).

Step skeleton (v1 all-stream; v2 swaps N-layer consumption to CPU per tier-dispatch §7):

```
input:  pending (host int), draft (host int), pos (host int, exact)
0  pread embed rows [pending, draft] via buffered twin (2 × 10,240 B, issued a step ahead);
   H2D into embed_stage[2][5120] on copy stream ks
1  copy embed_stage → pf_x rows 0..1                     (replaces decode.cu:46 embed_gather)
2  for l = 0..63 (tier dispatch per tier-dispatch §1):
     V: fp8 pair kernels on static device ptrs       (fp8_gemv2 / fp8_gemm(down) / bf16 a,b)
     Z: same kernels, UVA ptrs into pinned copies
     N: acquire_layer(idxN) → same kernels on ring-slot UVA ptr (16B-aligned F8 base, R3) → release_layer
        [v2: CPU kernels instead, activation ping-pong per tier-dispatch §2.2]
     full-attn l: pair q/k/v (fp8_gemv2) → split_q_gate_batch(24) → qk_norm_rope_batch(28 grid,
                  pos_dev) → store_kv_batch(T=2) → gqa_prefill(dim3(24,2), kvh=h/6) → sigmoid_mul
     linear  l: pair qkv/z (fp8_gemv2) → a/b bf16 → conv_prefill_silu(10240, snap_conv)
                  → deltanet_params_batch(48, A_log bf16) → deltanet_prefill(48 blocks, snap_delta)
                  → gated_rmsnorm(48×128) → out_proj pair
     residual + post-norm (Z=true) + MLP pair → residual
3  model.norm (Z=true) → lm_head bf16 GEMM T=2 (one 2.54 GB sweep, both rows) → argmax_fast ×2
   (row0→pos[3], row1→pos[2])                             [decode.cu:97-104 structure]
4  hidden ← pf_x[1] (D2D), addi(pos,2)                    [decode.cu:106-107]
5  spec_commit(pos, committed)                            [prefill.cu:287-302, unchanged]
6  spec_rollback_27(snap_delta, snap_conv, delta_state, conv_state, pf_x, hidden, pos)
                                                          [prefill.cu:305-314 with 27B sizes]
7  F7b MTP hole fill (unconditional, headless — §4.3)
8  spec_prologue(pos); mtp_layer_27(staged row = embed(pending), hid = x.hidden, dst = pos[4])
   — the 6.7 ms all-VRAM draft, ‖ reader already filling next epoch's slots
9  ONE tail D2H on ks: pos_dev[0..7] + count (+ accept flag) → sync → host: EOS scan, ctx guard,
   next-step preads for [t*_a, draft]
```

Cost check (v1 all-stream, MASTER-PLAN §2.4): verify body bound by NVMe 45 × 115.4 = 5,193 ms; tail
lm_head 5.4 + draft 6.7 + fill 1.0 ms — all hidden under the stream; **T_step ≈ 5.21 s → 0.31 tok/s
at p=0.6 (1.6 tok/step)**. v2 (L19/Z21/C9/N15): T_step ≈ 1.75 s → 0.92 tok/s D=1. Draft hot set
pinned in VRAM at load: mtp shard 477 MB + lm_head 2,542.8 MB (both inside the MASTER-PLAN §2.2
fixed block 3,678 MB).

`mtp_layer_27` = the `mtp_layer_impl(token_src, hid, argmax_dst, with_head)` refactor of spec-phase1
§4.1 with 27B call sites: concat 5120, `bf16_gemv(fc, x, hid, 5120, 10240)`, split at 6144/24 heads,
`gqa_decode` grid 24 with `kvh=head/6` over `mtp_keys/values`, MLP 17408, norms Z=true, lm_head
`bf16_gemv`-family single row + `argmax_fast` (same kernels the verify uses row-wise — the SPEC_PIN
rule from tier-dispatch §4: draft and verify must share kernel family/accumulation order or
acceptance collapses).

### 2.3 KV-full guard without graphs

See §3.3.

---

## 3. KV bookkeeping for the pair at 27B

### 3.1 Store-2-then-rollback-by-position (no KV restore, ever)

The 9B invariant (graph-hazards §3, spec-deepen §2.1) carries over unchanged because the KV **row
shape is identical** (4 kvh × 256 = 1024 f32) and both caches are position-indexed append-only:

- `store_kv_batch(T=2)` (`src/prefill.cu:88-98`) writes rows at `pos` and `pos+1` **before**
  `gqa_prefill` launches (decode order at `src/decode.cu:60-61`); row t reads exactly slots
  `0..pos+t` (`gqa_prefill_kernel` `tokens = pos_dev[0]+t+1`, prefill.cu:104) — never beyond its own.
- On **accept**: pos += 2 — both rows are true; nothing to undo.
- On **reject**: `spec_commit` does `pos[0] -= 1` (prefill.cu:300). Slot `pos+1` holds the rejected
  draft's k/v — stale but dead: the next step's row-0 `store_kv_batch` overwrites that exact slot
  before any gqa read (store-before-read within the layer call). No memcpy, no rewind kernel.
- Per-layer caches: `kv_keys/kv_values = 16 (full layers) × ctx × 1024` f32 (268.4 MB @ ctx 2048);
  placement follows the layer (V→VRAM, Z/N→pinned host, MASTER-Plan §2.5) — the pointer table
  `kc_of[l]` is baked at startup; `store_kv_batch`/`gqa_prefill` already take arbitrary pointers.

### 3.2 Recurrent-state rollback at 27B row shapes

`spec_rollback_27` restores, on reject only (`if (pos[6]) return;`):
`delta_state` ← snap (48·48·128·128 = 151.0 MB), `conv_state` ← snap (48·10240·3 = 5.9 MB),
`hidden` ← `pf_x[0..5119]`. Grid-stride copy as today (`src/prefill.cu:305-314` with the three size
literals swapped); ~157 MB write ≈ 0.4 ms at 504 GB/s — noise. v1: all states VRAM. v2 (C/N layers
CPU-consumed): state owner follows the consuming engine — CPU-owned layers' delta/conv live in
host memory and are restored by the same-sized host memcpy (or a CPU grid-stride twin) — the restore
kernel takes per-tier `(state,snap)` pointer pairs from the baked tables; snapshots for CPU-owned
layers never leave host memory (the deltanet kernel writes the snapshot only for GPU-owned layers —
see §5.2 for the mechanism at T>2).

### 3.3 KV-full guard, no graphs (fixes the graph-hazards §2 class by construction)

- Eager-only means the guard at `src/decode.cu:45` (`position+T > max_context` → throw) executes on
  every step — the replay-bypass path is gone. Keep it as the backstop.
- Host-side per-step gate in the driver (pos is exact from the tail readback): issue the T=2 verify
  only while `pos + 2 + 1 ≤ ctx`; in the last 2 slots fall back to eager single-token steps
  (T=1 GEMV path, `src/decode.cu:103-104` analog), then stop at `pos == ctx`. No clamp: refuse or
  degrade, never clamp+overshoot (safety C1; generate.cu:113-115's up-front refuse can stay as a
  coarse first line, but the per-step gate is now real, not decorative).
- Device-side belt (Phase D.5): `store_kv_batch_kernel` currently ignores `max_context`
  (`(void)max_context`, `src/prefill.cu:96`) — make it `if (pos >= max_context) return;` per row;
  same guard in `spec_commit_kernel` for `c+1+a > 16384` (committed cap, `src/decode.cu:18-20`).
  Cost: one scalar compare; benefit: bounded device writes even if the host gate is ever wrong.
- Overshoot bound without graphs: ≤ 1 commit batch (≤2 ids) past `want_total` — the existing trim
  (`src/generate.cu:196-199`) already cuts it.

---

## 4. MTP KV fill (spec-phase1 F7) at 27B

The reference (`tools/reference_multistep.py:104-119`) zero-inits `mtp_kvc` and fills it densely at
every position; the engine's drafts are NOT reference-equal today (two hole classes, spec-deepen
F7 / spec-phase1 §1.1). Fix components at 27B:

### 4.1 F7c — determinism memset (1 line)

Append to the ctor memset chain (`src/decode.cu:27`):
`cudaMemsetAsync(mtp_keys,0,size_t(ctx)*1024*4,stream); cudaMemsetAsync(mtp_values,0,...)` — 16.8 MB
@ctx2048, once. (And delete the probe path's post-probe memsets at `src/generate.cu:142-143`, which
destroy correctly-filled state once F7a lands — spec-phase1 §1.3 wiring note.)

### 4.2 F7a — prompt fill inside the weight-stationary prefill

27B prefill is weight-stationary (`prefill_layer_stationary`, MASTER-PLAN Phase F.1: outer loop
layers 0–63, inner 64-row tiles, activations ping-pong `h_A/h_B [S,5120]` f32 in VRAM). After the
last layer the ping-pong buffer holds the raw residuals h_0..h_{S-1} of the whole turn — the fill's
hidden inputs are already there (the 9B analogue used `pf_x` rows, spec-phase1 §1.3 fact 1).

Batched fill per turn (no lm_head, no proposals — k/v side effects are the product):

1. E = embed(tokens[1..S-1]): S−1 preads of 10 KB through the buffered twin (prompt ids are host
   data; ~µs each warm) → H2D to `pf_up` (compact [R,5120]).
2. En = rmsnorm(E, `mtp.pre_fc_norm_embedding`, **Z=true**); Hn = rmsnorm(h rows 0..R−1,
   `mtp.pre_fc_norm_hidden`, Z=true) → `concat_batch` [R,10240] (spec-phase1 kernel A).
3. fc: one bf16 GEMM [R,10240]→[R,5120] (Phase C.4 `bf16_gemm` / spec-phase1 `bf16_gemm_tn`;
   weight 104.9 MB read once for the whole turn).
4. The MTP layer in batch form — mirror of the full-attn branch of the stationary prefill with
   `pos = c0+r`, `kc/vc = mtp_keys/mtp_values`: pair of `fp8_gemm` calls for q/k/v/o (T=R),
   `qk_norm_rope_batch` reading a device slot set to the chunk base (pass `x_.mtp_pos_dev` after a
   `set_mtp_position(c0)` — spec-phase1 §1.3 pattern), `store_kv_batch`, `gqa_prefill` (grid
   `dim3(24,R)`, `kvh=head/6`), gate sigmoid, o_proj, MLP. Row r ropes/stores/attends exactly slot
   `c0+r` — the reference triad (spec-phase1 "Correctness notes"). Causality is free:
   `store_kv_batch` writes all R rows before `gqa_prefill`; row r reads ≤ own slot.
5. `gqa_prefill`'s `score[4096]` bounds the window: max read slot = N−1 ≤ 4095 ✓ (ctx cap 4096,
   default 2048).

Cost: fc 104.9 + layer 372.2 MB VRAM sweeps ≈ 0.95 ms + preads ≈ **~1 ms per turn** — 0.03% of the
~3.0–3.5 s prefill. Wiring: the turn driver calls `prefill_layer_stationary(...) → mtp_prefill_fill27(...)`
in place of the 9B `prefill_chunk_mtp` (spec-phase1 §1.3) — chunk granularity is per-turn (S = whole
turn ≤ ctx), no cross-chunk dependency (last row slot N−1 deliberately left to the first draft's own
store_kv).

### 4.3 F7b — the accept-hole fill under tail-draft order (one-step-shifted rule)

With the draft at the tail (§2.2), each step writes exactly ONE mtp slot, `pos_after_commit − 1`.
Reject (a=0): next pos = P+1 → next draft slot P — dense. Accept (a=1): next pos = P+2 → next draft
slot P+1 — **slot P is skipped**, and its invocation `(embed(t_{P+1}), h_P)` is exactly the
never-run "second draft". The hole is in the *next* step's draft read window.

**Rule (derived, matches spec-phase1 §1.4 with names shifted one step): in every step, between
`spec_rollback` and the tail draft, unconditionally run one headless MTP invocation writing slot
`entry_pos = pos_at_step_start`, with inputs (embed(staged row 1), pf_x[0]).**

- Inputs are exactly available at that point: staged row 1 = embed(this step's draft-under-
  verification = proposal for position entry_pos+1 ✓); `pf_x[0]` = h at entry_pos ✓ (rollback just
  anchored it for a=0; for a=1, pf_x[0] is still the row-0 residual — the copy at decode.cu:106 took
  pf_x[T-1] into `hidden`, pf_x itself is untouched).
- Slot targeting device-side: `spec_fill_slot_kernel<<<1,1>>>(pos): pos[7] = pos[0] - 1 - pos[6];`
  (post-commit pos minus (1+accept) = entry_pos), then `mtp_layer_impl(staged_row1_dev,
  mtp_fill_h /* = pf_x[0] saved D2D first, spec-phase1 ORDERING HAZARD note */, nullptr, false)`,
  then `spec_prologue` re-derives `pos[7]=pos[0]-1` for the tail draft (idempotent, no cross-step
  state).
- Trace: previous step accepted → this slot is the exact missing invocation (reference-equal).
  Previous step rejected → wrong-token entry, but the tail draft's own `store_kv` (slot entry_pos =
  its own slot when this step rejects, or overwritten later) precedes its gqa — harmless overwrite,
  nothing valid clobbered (same argument as spec-phase1 §1.4 v2).
- Cost: fc 104.9 + layer 372.2 MB ≈ **0.95 ms/step unconditional** — 0.05% of v2's 1.75 s, fully
  hidden under the reader. Ship the exact fill (v2) as default; the zero-patch fallback
  (`spec_mtp_patch_kernel`, spec-phase1 §1.4 v1 — zero 1024 floats at slot entry_pos) stays as a
  one-launch emergency variant.

Post-fix invariant (spec-phase1 §1.4): every slot a draft's gqa can read (0..mtp_pos) is
prefill-filled, written by a previous draft, written by the draft's own store_kv, or hole-filled.
MTP cache append-only; no rewind; no MTP-side rollback state besides `mtp_pos_dev` (re-derived every
step by the prologue).

---

## 5. Depth-4 generalization (optional, spec-deepen §2/§4 at 27B shapes)

### 5.1 The memory finding (new): D=4 breaks the v2 pinned ledger; D=3 fits

Snapshots at 27B are 151.0 + 5.9 = 156.9 MB per row. T=5 needs rows 0..3 → **627.7 MB**; T=4 (D=3)
→ 470.7 MB. The WDDM pinned cap is 8,531 MB and in v2 the ring is deliberately *not* registered
(tier-dispatch gap 2), so pinned = Z + snapshots:

| config | pinned total | verdict |
|---|---|---|
| v2 Z=21 (8,003 MB) + D=1 snaps (156.9) | 8,160 | ✓ |
| v2 Z=21 + D=3 snaps (470.7) | 8,474 | ✓ (57 MB spare) |
| v2 Z=21 + D=4 snaps (627.7) | **8,631** | ✗ 100 MB over the 8,531 cap |
| v2 **Z=20** + D=4 snaps | 8,250 | ✓ (one Z layer → C/N; +~115.4 ms on the NVMe-binding step, ≈ −6% T_step) |

Recommendation: **D=3/T=4 is the depth default** (E[tok/step] 2.31 vs 2.44 at p=0.6 — 94% of the gain,
zero placement cost); D=4/T=5 only if the §6 probe shows E[a] ≥ 2.3, and then at Z=20. Host-pinned
snapshots restore in 156.9 MB / ~13 GB/s ≈ **12 ms** — hidden under the 1.7 s stream (MASTER-PLAN
§2.2). D=1 keeps snapshots in VRAM (157 MB inside the §2.2 fixed block).

### 5.2 Snapshot mechanics: per-row recurrence instead of intra-kernel host writes

`deltanet_prefill` snapshots inside its sequential row loop (prefill.cu:258) — at T=2 writing to
VRAM `snap` is one extra global write; at T>2 with host-pinned snaps, a UVA write from inside the
recurrence would serialize the kernel over PCIe 604 MB/step. Instead run the **recurrent part of the
T=5 verify in per-row mode**: for the 48 linear layers, call the existing decode-path kernels once
per verify row (`causal_conv4_silu` + `deltanet_decode`-family, `src/qwen_kernels.cu:7-8`,
`src/deltanet.cu:4-14`, at 27B `<<<48,128>>>`, kh=h/3) and snapshot between rows with plain async
memcpys on a copy stream:

```
after verify row t (t = 0..T-2), for each linear layer group:
    cudaMemcpyAsync(snap_delta_h + t*nd, delta_state, 151.0 MB, D2H, copy_stream)   // pinned dst
    cudaMemcpyAsync(snap_conv_h  + t*nc, conv_state,   5.9 MB,  D2H, copy_stream)
```

Double-buffer not needed if issued on a second stream ordered after row t's kernels and the kernel
for row t+1 reads/writes `delta_state` only after an event wait on the copy — or accept the simpler
single-stream variant (copies serialize the recurrence: 604 MB D2H ≈ 46 ms at 13 GB/s pinned =
2.6% of v2's step — acceptable v1; pipeline later). All GEMV/GEMM weight traffic stays batched T=5
(weights read once — the recurrence split only affects state traffic, which the scan would pay in
snapshots anyway). Full-attn layers need no snapshots (KV append-only, §3.1): `store_kv_batch(T)`
+ `gqa_prefill dim3(24,T)` unchanged.

Correctness note: per-row decode ≡ prefill scan recurrence (same ops, same order per row); gate with
a unit test against the R4/R7 references before trusting it (AGENTS.md measurement rule).

### 5.3 Kernel signatures (new/changed, compile-ready)

```cpp
// ---- prefill.cu / insignia_prefill.cuh (all <<<1,1>>> unless noted) ----
// T-row setup: pf_tokens[t=0]=pos[1]; pf_tokens[t]=pos[draft_slot(t)] for t>=1.
// Draft slots: pos[4]=d1, pos[10..12]=d2..d4 (spec-phase1 §5.3 layout, widened pos_dev to 32 ints).
__global__ void spec_setup_T_kernel(int *__restrict__ pos, int *__restrict__ pf_tokens, int T);
void spec_setup_T(int *pos, int *pf_tokens, int T, cudaStream_t stream = nullptr);

// Chain-rule commit, D<=8 serial loop (spec-deepen §4; spec-phase1 §5.4 generalized):
//   a = max k: pos[draft_slot(k)] == pos[tstar_slot(k-1)] for all k<=a   (tstar_0..T-1 in pos[3],pos[2],pos[8],pos[9],pos[13])
//   committed[c..c+a] = [pending, d1..da]; pos[1]=tstar_a; pos[5]=c+1+a;
//   pos[0] += 1+a-T;  pos[6]=a;  guard: c+1+a <= 16384 else truncate (C1 device belt).
__global__ void spec_commit_T_kernel(int *__restrict__ pos, int *__restrict__ committed, int T);
void spec_commit_T(int *pos, int *committed, int T, cudaStream_t stream = nullptr);

// Batched argmax: grid dim3(64, T); each row reduces n floats into one monotonic u64
// (order-bits<<32|idx) via atomicMax on scratch[row]; stage-2 kernel unpacks T winners.
__global__ void argmax_rows_stage1_kernel(const float *__restrict__ x, int n, unsigned long long *__restrict__ scratch);
__global__ void argmax_rows_stage2_kernel(const unsigned long long *__restrict__ scratch, int *__restrict__ out, int T);
void argmax_rows(const float *logits /*[T][248320]*/, int vocab, int T,
                 int *out /*[T]*/, unsigned long long *scratch /*[T]*/, cudaStream_t stream = nullptr);

// Restore at accept-length a < T-1 (host-orchestrated; sources host-pinned at 27B, VRAM at 9B):
void spec_restore_T(const float *snap_delta /*[T-1][48*48*128*128]*/,   // host-pinned
                    const float *snap_conv  /*[T-1][48*10240*3]*/,      // host-pinned
                    float *delta_state, float *conv_state,             // device (v1) / host (v2 CPU-owned)
                    const float *pf_x, float *hidden,                  // hidden <- pf_x + a*5120 (D2D)
                    const int *pos, int T, cudaStream_t compute, cudaStream_t copy);
//   = 2 × cudaMemcpyAsync H2D (a-indexed) + 1 small D2D hidden copy + event join; ~12 ms @T=5.

// Rollback at D=1/T=2 (27B sizes; grid-stride, early-exit on pos[6]):
__global__ void spec_rollback27_kernel(const float *snap_delta, const float *snap_conv,
        float *delta_state, float *conv_state, const float *pf_x, float *hidden, const int *pos);
//   n1 = 48*48*128*128; n2 = 48*10240*3; hidden width 5120. Launch <<<512,256>>>.

// F7b hole-fill slot targeting (tail-draft order):
__global__ void spec_fill_slot_kernel(int *__restrict__ pos);  // pos[7] = pos[0] - 1 - pos[6]
// followed by mtp_layer_impl(staged_row1, mtp_fill_h, nullptr, /*with_head=*/false)

// Device belts (C1): store_kv_batch_kernel gains `int max_context` use: if (pos >= max_context) return;
//                                              spec_commit guards the 16384 committed cap.

// ---- decode.cu (27B driver) ----
int  spec_step27(int pending, int draft);   // eager tail-draft step (§2.2); one fused tail D2H
void mtp_layer_impl27(const float *embed_row /*staged [5120]*/, float *hid, int *argmax_dst, bool with_head);
void mtp_chain27(int D);                    // D tail drafts, chained in-place on hid (§5.4)
int  prefill_turn_mtp27(const int *tokens, int S);  // stationary prefill + F7a fill (§4.2)
```

Draft chain (D≥2) is spec-phase1 §4.2's in-place residual chain: draft 1 consumes (embed(pending),
h_anchor); draft i>1 consumes (embed(a_{i-1}), R_{i-1}) where R is the MTP residual carried in `hid`
— verified safe in-place (rmsnorm reads are non-destructive; concat materializes fc inputs before fc
overwrites `hid`). At 27B each chain step needs embed(a_{i-1}) — the previous draft's argmax is
device-side but embed is host-side! Two options: (a) stage the D−1 inner embed rows at step start
from the *previous* step's chain ids (they are exactly the previous chain's a_1..a_{D-1} — known to
the host from the tail D2H; the chain re-derives them deterministically), plus one row for the last
draft from this step's own D2H — i.e. pread D+1 rows/step total; (b) keep a 1-row on-device fallback
via `bf16_get_row` into a *small VRAM embed slice* — rejected (655 MB, embed-lmhead §6). Use (a);
note the chain's first draft embed row = staged row 0 (pending) — already prefetched.

### 5.4 Timing at D=4/T=5 (v2)

Verify T=5 ≡ T=2 in cost (tier-bandwidth-bound; weights stream once — spec-deepen §3). Tail: lm_head
T=5 GEMM 5.4 ms + argmax_rows ~0.1 + commit/restore (12 ms on a<4, hidden) + 4 drafts ≈ 27 ms +
fill 0.95 ms — **~45 ms of tail against 1.7 s of stream, all overlapped with the next epoch's
reader fill** (drafts touch only VRAM-pinned tensors — no load-path contention). Snapshot D2H 604 MB
(46 ms serial, pipelined later) rides the copy stream during the verify. Expected 2.44 tok/step at
p=0.6 → ~1.4 tok/s; 2.77 at p=0.7 (post-F7) → ~1.6 tok/s (MASTER-PLAN §1.3 multipliers ÷ T_step).

---

## 6. Acceptance-rate instrumentation (measuring p to gate depth)

What exists: the eager loop already prints the per-step accept flag (pos[6], `src/generate.cu:162`)
and the probe path drives one post-prefill draft (`src/generate.cu:134-144`). At 27B:

1. **Online p₁ (always on, free):** the tail D2H already returns pos[6]; accumulate
   `accepts/steps` over the session, print at drain. This is the D=1 acceptance p and the drift
   monitor (a p collapse after a change = F7-class regression or kernel-family mismatch).
2. **Depth histogram probe (the D-gate; exact, engine-side):** measuring the true acceptance-length
   distribution needs greedy tokens conditioned on the draft prefix — that is precisely a T=D+1
   verify. Probe mode: every K-th step (K≈8, ≥200 samples), run the D=4 chain + T=5 verify +
   `spec_commit_T` and histogram `pos[6]` (=a) before continuing (the probe step is itself a legal
   spec step — outputs stay greedy-exact). Cost: (K−1)/K of steps pay D=1 tail, 1/K pays the D=4
   tail (~45 ms) — throughput-neutral. Report `E[a] = Σ a·freq`, p_i (prefix rates), and the
   projected E[tok/step] at each D from the same histogram.
   **Gate: enable D=3/D=4 when measured E[a|D=4] ≥ 2.2** (vs 1.6 at D=1) and pinning fits (§5.1);
   re-run after F7 lands — spec-deepen §1.2 predicts p 0.6→~0.7 once drafts stop attending garbage.
3. **Reference cross-check (offline):** `tools/reference27.py` currently has **no MTP subcommand**
   (engine27-gap §0) — port `reference_multistep.py:104-130` (dense zero-init `mtp_kvc`, teacher-forced
   fills, chain recursion per spec-deepen §8 item 4) to validate the F7a fill and the chain input
   convention before trusting engine-side p.

---

## 7. The 777f55f reject-path fix — what it fixed, and the eager-port invariant

Commit `777f55f` ("fix reject-path state corruption") introduced the whole `src/prefill.cu` spec
machinery (its parent `1ccba12` had no spec step in-tree; the two "pre-existing bugs" were bugs of
the in-development spec path it replaced). The fix has two halves:

1. **Rollback anchor = post-committed-row state, not pre-step state.** The commit message: "reject
   rollback erased the committed row-0 conv/DeltaNet contributions (kernels now checkpoint
   post-row-0 state inline)". Mechanics in the tree: `deltanet_prefill` snapshots S **after**
   processing t==0 (`src/prefill.cu:258-261`); `conv_roll_state_kernel` writes the window **rolled
   past row 0** (`snap=[s1,s2,x0]`, prefill.cu:187-189); `spec_rollback` restores those + `hidden =
   pf_x[0]` (prefill.cu:308-310). A reject therefore means "keep row 0 (the pending token), drop
   row 1" — row 0's recurrent contributions survive. (A pre-step snapshot would have erased them
   and silently corrupted every subsequent token.)
2. **Reject double-commit of t2.** `spec_commit_kernel` on reject commits ONLY `[pending]`, sets
   `pending ← pos[3]` (t\*_0 is *decided*, not *processed*), `count += 1`, `pos[0] -= 1`
   (prefill.cu:295-300). The old path wrote [t0, t2] and re-committed t2 next step.

**Invariant the eager 27B port must keep:** after every step, the anchor triple
`(delta_state, conv_state, hidden)` equals the state immediately after processing the last
*committed* row (row a at accept length a), and `pos[0]` equals the count of processed rows —
enforced by the same three pieces: post-row-0 snapshots at 27B sizes (§1.5), `spec_commit(_T)`'s
`pos += 1+a−T` correction, and the restore kernel's `snap[a]`/`pf_x[a]` selection with early-exit at
a = T−1. The port introduces two new state-mutating pieces that must respect it:
(a) the **F7b hole fill runs after `spec_rollback` and reads `pf_x[0]` via the `mtp_fill_h` staging
copy** (spec-phase1 §1.4 ORDERING HAZARD — reading pf_x[0] directly through the MTP layer would
scribble the anchor hidden before/while rollback restores it; the staging D2D copy makes the fill
order-free); (b) the **tail draft runs after commit/rollback** so it consumes the anchor hidden
(`x_.hidden`), never a speculative residual — with draft-first order this was guaranteed by
`prefill_chunk_device`'s hidden export happening after `mtp_layer` had finished reading
`x_.hidden` (decode.cu:106 after :222's layer body); with tail order it is guaranteed by
construction. Greedy-exactness is unaffected by any of this in the emitted stream (verify gates
everything; spec-deepen §4 argument) — the invariant protects *state*, i.e. future acceptance and
the reference-equality of drafts.

Also preserved: eager==graph id-stream equality at 9B (commit message) has its 27B analogue in
"eager spec == non-spec greedy" — R9 is the gate; the probe path (§6) is the acceptance diagnostic.

---

## 8. Validation gates (per AGENTS.md: measurement + parity)

1. **F7a/F7b parity:** probe draft at slot N−1 after a filled prompt == reference27 MTP teacher-forced
   argmax (port the reference first — §6.3); multistep eager ids == non-spec greedy ids.
2. **No state drift:** 100-step spec run: finiteness, NLL < 5 nat/token (R8 harness), pos == count
   invariant at every tail readback, no mtp slot read before write (assert via compute-sanitizer
   initcheck on a 3-step run).
3. **p measurement:** §6.1 counter ≥ 0.55 over ≥ 500 steps before declaring F7 done (Phase F gate).
4. **Perf:** v1 step within 5% of 5.21 s + spec tail hidden (nvtx/event the tail); D=3/D=4 probe
   per §6.2 before enabling; snapshot D2H ≤ 50 ms measured.
5. **9B regression:** the 9B graph path untouched (no `capture_*` calls removed; template
   instantiations `<16>`/`<24>` coexist per Phase B).

## 9. File:line index (primary)

- src/decode.cu: 11-28 (ctor allocs/sizes), 15 (pos aliases), 27 (memset chain), 31-41 (linear/
  linear2/linear_batch), 45 (KV guard), 50 (pair flag), 53-92 (pair layer body), 78-84 (conv+delta
  with snap), 96-105 (lm_head T==2), 106-108 (hidden/addi), 133-134 (forward), 137-192 (mtp_layer),
  193-218 (host plumbing), 219-237 (spec_step), 238-249 (capture_spec), 250-254 (spec_graph_step +2)
- src/prefill.cu: 43-51, 54-86, 88-98, 102-165 (batch kernels + grids), 170-200 (conv + roll/snap),
  203-214 (params), 219-269 (deltanet_prefill + snap 258), 271-272 (addi), 277-314 (spec kernels)
- src/qwen_kernels.cu: 5-10 (norms/conv/params), 15-16 (store_kv), 25-63 (argmax_fast), 67-69
  (bf16_gemv/concat), 73-79 (split/expand)
- src/attention.cu: 7-8 · src/deltanet.cu: 4-14 · src/fp8.cu: 14-55, 58-103 (gemv2 T=2, 99 KB cap),
  109-190 (gemm + F1 store contract), 193-200 (bf16_get_row)
- src/generate.cu: 112-115 (ctx guard), 123-133, 134-144 (probe), 145 (warmup), 154-190, 193-199
- src/storage.cu: 8-10 · src/streaming.cu: 373/430/443/452 (LayerFeeder) · src/qwen35.cu: 7-31 (WKind)
- include/insignia_prefill.cuh: 17-20 · include/insignia_decode.hpp: 8-15 (workspace members)
- git 777f55f (spec machinery birth + the two reject-path fixes; parent 1ccba12 has none of it)
