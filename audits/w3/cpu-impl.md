# CPU FP8 tier — implementation report (w3, `insignia_cpu.hpp` landed)

Date: 2026-08-25. Implements the design in `audits/w3/cpu-fp8.md` §4. Files:

- `include/insignia_cpu.hpp` — header-only CPU tier, `namespace insignia::cpu`, AVX2/FMA/F16C only.
- `src/test_cpu.cpp` — correctness harness (f64 references) + benchmarks. **40/40 checks PASS**.
- Build (the known-good path used): `call vcvars64.bat` then
  `cl /nologo /arch:AVX2 /O2 /std:c++20 /fp:precise /EHsc /Iinclude src\test_cpu.cpp /Fe:build\test_cpu.exe`
  (`/EHsc` added vs the mission sketch — the kernels throw on bad dims; MSVC needs it for unwinding).
  Run: `build\test_cpu.exe` (all), `test`, or `bench`. `INSIG_CPU_THREADS=n` overrides pool size.

Headline: the 6-worker pool sustains **35.3–35.9 GB/s** on every 27B layer shape (audit model:
37 GB/s), a 27B linear-attention layer's F8 GEMVs take **10.75–10.83 ms** measured (model: 10.34),
the deltanet step runs in **31 µs** (12.5× under the audit's 260 µs DRAM estimate — the 3.15 MB
state is L3-resident), and GQA decode at ctx 2048 runs **0.49–0.68 ms** f32 after two rewrites.

---

## 1. Changes vs the design doc (§4 of cpu-fp8.md), with reasons

The design was implemented almost verbatim; everything below is a deviation, and every one is
either a correctness fix against the actual GPU kernels or an MSVC reality.

1. **`bf16_scale_x256`: subnormal scales break the bit-add** (real bug, found by the exhaustive
   65536-code test). For bf16 exponent field 0 the mantissa is subnormal; `b + 0x04000000`
   reinterprets it as a normal mantissa → wrong by ~(128/m)·2^8. The audit's "never rounds"
   claim only holds for normals. New guard: `e == 0 || e >= 0xF7` takes the exact-multiply path
   (also covers ±0, overflow-to-inf, NaN). Exhaustive: 65536/65536 bit-exact. Real block scales
   (~1e-3..1) never hit the guards; the discipline is free.
2. **`gated_rmsnorm_per_head_cpu`: norm weight is `[128]` SHARED across heads** (fix). The
   engine kernel (`src/qwen_kernels.cu` `rms_bf`, called as `gated_rmsnorm_bf16(..., 32, 128)`)
   indexes `w + i` within each 128-wide row — one `[128]` vector for all heads. The design draft
   advanced `w += hd` per head, i.e. assumed `[heads][128]`. Caught by reading `decode.cu` +
   the per-head test reference.
3. **`qk_norm_rope_cpu`: `qw`/`kw` are `[256]` SHARED across heads** (fix). Same story in
   `src/ops.cu` `qk_norm_rope` (`w + tid`, tid ∈ [0,256)). The draft advanced them per head.
4. **RoPE angles computed in f64, cast once** (fix). `pos·θ` reaches ~1.2e3 rad; float-side
   `pow` injects ~1e-4 absolute angle error → up to 3e-2 relative on cancelled roped elements.
   Double angles drop q-rope parity to 3.7e-6 rel. 64 transcendentals per token, negligible.
   (The GPU uses `__powf` float — CPU-vs-GPU difference stays ≪ the 1e-3 CPU/GPU bar.)
5. **MSVC portability**: `/arch:AVX2` does NOT define `__FMA__`/`__F16C__` (GCC conventions) —
   the `#error` guard is now MSVC-aware (requires `__AVX2__` only). MSVC has no `_cvtsh_ss` —
   added `f16_to_f32()` via `_mm_cvtph_ps`+`_mm_cvtss_f32`.
6. **Dequant register form `e4m3x32_rr`** (perf, disasm-verified). MSVC spills the `__m256
   wv[4]` array of the design's chunk macro to stack before the FMAs (verified in dumpbin:
   `vmovups [rbp+40h]` round-trips per 32 weights — audit risk §8.2 realized). Named-register
   outputs keep the GEMV cluster fully in ymm6–ymm15, zero spill stores (re-verified). 1T rate
   unchanged at 8.4 GB/s (single-core was not spill-bound), 6T still DRAM-pinned at ~35.8.
7. **GQA decode rewritten twice** (the design's per-head loop was its weakest part):
   - v1 (design): online softmax per token with `o[32]` array + 2 scalar `expf` per head-token
     → stack traffic every token; measured 1.53 ms f32 @ctx2048 (audit predicted 0.454).
   - v2: block-of-64 batching — scores first (register-light), one `vexp256_ps` per 8 scores,
     V-accumulation in chunks of 4–8 live registers → 2.2× (0.69 ms best).
   - v3 (final): **kv-group-major** — the 6 q-heads of a kv group share each K/V row walk
     (row loaded once, L1 re-reads). v1/v2 re-walked K/V 6× per thread with a 4 KB stride
     (~100 MB of L3/DRAM traffic instead of 16.8 MB, TLB-hostile — this was also the source of
     the wild run-to-run variance). Final: f32 0.49–0.68 ms (24.5–34.0 GB/s), bf16 0.41–0.60 ms.
8. **`deltanet_step_cpu` gained a `kshare` parameter** (default 3 for 27B: kh = head/3).
   Layout matches `src/deltanet.cu` exactly: `S = state + head*128*128`, `S[k*128+v]`
   (k = key index, v = value index), raw q/k vectors scaled per head, q-norm folds 1/√128,
   eps inside the rsqrt, two passes (dot pass then fused update+output pass). The test also
   runs the 9B shape (32 heads, kshare=2) to mirror the GPU kernel's `head>>1`.
9. **Additions**: `fp8_gemv_st` (serial entry: 1-thread bench + fallback), `store_kv_bf16_cpu`
   + `f32_to_bf16_bits` (bf16 KV cache store side), `INSIG_CPU_THREADS` env override,
   `Parity.max_abs_rel` (scale-normalized error) in the reference comparator.
10. **Parity metric formalized** (see §2 rationale): `max_rel` is floored at 1e-3·refmax for
    cancellation-prone kernels and a scale-normalized `max_abs_rel = max|y−ref|/refmax` gate
    was added. Mission bars as landed: GEMV rel ≤ 1e-4 & cos > 0.999999 (measured ≤ 5.6e-5 /
    1.0000000000); norms/deltanet/gqa "exact-ish 1e-5" landed as abs ≤ 1e-5·scale &
    cos ≥ 0.9999999 (measured ≤ 6e-7). Element-wise 1e-5 rel is not achievable for ANY fp32
    kernel on cancellable 128+-term sums (chained SSM state after 2 steps reaches 1.5e-4 rel
    on 1e-3-cancelled elements with abs error 3e-7·scale) — the abs gate is the honest form.

Non-change worth noting: the thread pool (packed-gen atomic tickets, caller-participates
progress guarantee, spin-then-park, LP0–5 affinity) is the design's, verbatim; it proved stable
across every bench (GEMV timings repeat to ±0.1 GB/s).

## 2. Correctness results (all 40 checks PASS)

| check | result |
|---|---|
| `e4m3x32_f32` exhaustive (all 256 codes) | exact, incl. subnormals, ±480 codes |
| `bf16_scale_x256` exhaustive (all 65536 bf16) | bit-exact (254 NaN codes agree as NaN) |
| fp8 GEMV 2560×5120 / 256×384 / 1024×5120 | cos=1.0000000000, rel ≤ 2.8e-5, abs ≤ 2.3e-7 |
| fp8 GEMV pair (T=2), same shapes | cos=1.0000000000, rel ≤ 5.6e-5, abs ≤ 2.0e-7 |
| fp8 `st == mt` bit-identical | pass (serial path = pool path) |
| bf16 GEMV 48×5120, 512×640 | cos=1.0000000000, abs ≤ 6.9e-7 |
| rmsnorm (plain + zero-centered) 5120 | rel ≤ 1.7e-7 |
| gated per-head rmsnorm 48×128 (shared w) | rel ≤ 2.5e-6 |
| silu_mul / sigmoid_mul 17408 | rel ≤ 3.0e-6 |
| conv1d+silu ×2 steps + state shift | rel ≤ 6.0e-6, state bit-exact |
| deltanet parameters (48) | rel ≤ 1.1e-7 |
| deltanet step 48h/k3 (×2 chained) + 32h/k2 | cos=1.0000000000, abs ≤ 4.1e-7 |
| qk_norm_rope pos=0 / 1234 (q and k) | rel ≤ 3.7e-6 |
| split_q_gate, store_kv f32/bf16 | exact |
| gqa decode f32/bf16, t=2048 / 77 | cos=1.0000000000, abs ≤ 2.2e-6 |

References are double-precision mirrors of the GPU kernel semantics (deltanet ref replicates
the exact GPU index/order structure; gqa ref is a plain softmax like `attention.cu`).

## 3. Measured performance (5600X, 6 pinned workers + participating main, min-of-iters)

**Pool scaling, fp8 GEMV 10240×5120 (52.43 MB):**

| threads | ms | GB/s |
|---|---|---|
| 1 (serial `fp8_gemv_st`) | 6.24 | 8.4 |
| 2 (1 worker + main) | 3.14 | 16.7 |
| 3 (2 + main) | 3.00 | 17.5 |
| 4 (3 + main) | 2.04 | 25.7 |
| 5 (4 + main) | 1.97 | 26.6 |
| 7 (6 + main) | 1.46 | **35.9** (4.3× vs 1T) |

**27B layer shapes (6 workers + main):**

| matrix | shape | MB | ms | GB/s |
|---|---|---|---|---|
| in_proj_qkv | 10240×5120 | 52.43 | 1.46–1.48 | 35.5–35.9 |
| in_proj_z | 6144×5120 | 31.46 | 0.88–0.92 | 34.0–35.7 |
| out_proj | 5120×6144 | 31.46 | 0.87–0.89 | 35.3–36.1 |
| mlp gate/up | 17408×5120 | 89.13 | 2.49–2.52 | 35.4–35.8 |
| mlp down | 5120×17408 | 89.13 | 2.49–2.55 | 35.0–35.8 |
| **linear layer total** | 382.73 MB | | **10.75–10.83** | 35.3–35.6 avg |

Pair GEMV (MTP verify, T=2): 1.81 ms → 28.9 GB/s stream, **57.8 GB/s effective** (second token
free, as the synthesis argued).

Non-GEMV ops: deltanet step 48 heads **31 µs** (state is L3-resident between layers in this
bench; the audit's 0.26 ms DRAM figure is the cold-cache upper bound); GQA decode ctx2048
**0.49–0.68 ms** f32 / 0.41–0.60 ms bf16 (residual run-to-run variance tracks fresh-page
placement of the 25 MB test KV — pre-fault/large-pages would pin it; see gaps).

**Expected per-layer / per-token (from measured numbers):**

- linear-attention layer: 10.8 (GEMV) + ~0.09 (conv ~0.03 est + deltanet 0.031 + norms ~0.03)
  ≈ **10.9 ms**
- full-attention layer: 372.24 MB F8 ≈ 10.5 + GQA 0.5–0.7 + qk-rope/store ~0.02 ≈ **11.1–11.3 ms**
- 23 CPU layers (17 linear + 6 full at the 3:1 census ratio) ≈ **253 ms/token clean** — the
  audit's 245 ms model number, now measurement-backed. Under NVMe staging contention (model:
  −5 GB/s) expect ~290 ms. MTP verify on CPU layers rides the pair GEMV at ~2× effective rate.

## 4. Known gaps (what is still missing for full CPU-layer execution)

1. **Nobody calls this yet.** The decode loop (`src/decode.cu`) has no CPU-layer branch; the
   engine's model index has no `DType::f8e4m3` entry (loader gap per w2 audit). CPU layers
   need: host-resident weight views (mmap or staged), per-layer `s256` + expanded conv weights
   prepared once at load, host f32 activation buffers, and the 20 KB activation handoff at
   CPU↔GPU layer boundaries (~8 µs each over PCIe, negligible).
2. **State residency**: deltanet state (3.15 MB/layer) and KV caches for CPU full-attn layers
   must live in host RAM (pinned if the GPU reads them back). KV is f32 today; the bf16 path
   (halves GQA traffic + store side `store_kv_bf16_cpu`) is ready but unused.
3. **Prefill**: CPU kernels are decode-shaped (T=1, pair=2). Weight-stationary prefill through
   CPU layers would need a T-batched GEMM — not implemented (out of scope; prefill stays GPU).
4. **Pool overlap**: workers idle while GPU/NVMe tiers run; wiring them as staging/copy muscle
   (colibri early-issue) is future work. `launch()` blocks the caller (by design, caller =
   progress guarantee) — never call it from a thread that must not block (IOCP callbacks).
5. **Small ops are serial** (norms, conv, params, merge) — ~90 µs/layer total, 0.8% — fine.
   Conv1d is scalar by engine state layout `[ch][3]` (~30 µs est, unmeasured).
6. **GQA page-placement variance**: fresh 25 MB buffers vary 0.49→0.68 ms between runs;
   production should pre-fault the KV cache once at allocation (or use large pages).
7. **L3-residency effects**: k/v_proj (5.24 MB) and a/b (0.98 MB) go L3-resident after the
   first token (~25 µs steady-state vs 0.14 ms cold) — not re-measured with neighbors resident;
   the layer totals above conservatively use the full-DRAM GEMV table.
8. Per AGENTS.md: adoption bar is bench + NumPy-reference parity + disasm. Bench ✓, disasm ✓
   (spill-free GEMV cluster), NumPy end-to-end layer parity pending — naturally blocked on
   gap 1 (plumbing), not on the kernels.

## 5. Integration notes for the decode-loop engineer

Everything is in `#include "insignia_cpu.hpp"`, `namespace insignia::cpu`. The pool spawns +
pins on first use (`CpuPool::get()`, 6 workers on LP0–5; override with `INSIG_CPU_THREADS`).

**One-time per CPU layer (at load):**

```cpp
// scales: bf16 weight_scale_inv [ceil(r/128)][ceil(c/128)] -> f32 (x256 folded)
std::vector<float> s256(nblocks);  fp8_prepare_scales(scale_bf16_ptr, s256.data(), nblocks);
// conv1d [ch=10240][4] bf16 -> f32 [4][ch]
std::vector<float> conv_f32(4*10240);  expand_conv_weights(conv_bf16_ptr, conv_f32.data(), 10240);
```

**Per token, linear-attention layer (27B shapes; buffers host f32):**

```cpp
rmsnorm_cpu(x, input_norm_w_bf16, n, 5120, /*zero_centered=*/false);      // engine uses false today
fp8_gemv_mt(qkv_w, qkv_s256, n, qkv, 10240, 5120);
fp8_gemv_mt(z_w,   z_s256,   n, z,    6144, 5120);
bf16_gemv_mt(a_w, n, a, 48, 5120);  bf16_gemv_mt(b_w, n, b, 48, 5120);
deltanet_parameters_cpu(a, b, A_log_f32, dt_bias_bf16, 48);
causal_conv4_silu_cpu(qkv, conv_state, conv_f32.data(), 10240);
deltanet_step_cpu(delta_state, qkv, qkv+2048, qkv+4096, a, b, core, 48, /*kshare=*/3);
gated_rmsnorm_per_head_cpu(core, norm_w_bf16 /*[128], shared*/, z, core, 48, 128);
fp8_gemv_mt(out_w, out_s256, core, t, 5120, 6144);  residual_add_cpu(x, t, 5120);
rmsnorm_cpu(x, post_norm_w_bf16, n, 5120, false);
fp8_gemv_mt(gate_w, g_s256, n, gate, 17408, 5120);  fp8_gemv_mt(up_w, u_s256, n, up, 17408, 5120);
silu_mul_cpu(gate, up, gate, 17408);
fp8_gemv_mt(down_w, d_s256, gate, t, 5120, 17408);  residual_add_cpu(x, t, 5120);
```

**Full-attention layer adds** (q_proj is [q|gate] interleaved per head):

```cpp
fp8_gemv_mt(q_proj_w, q_s256, n, qg, 12288, 5120);
split_q_gate_cpu(qg, q, gate, 24);                    // q,gate: [24][256]
fp8_gemv_mt(k_w, k_s256, n, k, 1024, 5120);  fp8_gemv_mt(v_w, v_s256, n, v, 1024, 5120);
qk_norm_rope_cpu(q, k, q_norm_w_bf16 /*[256] shared*/, k_norm_w_bf16 /*[256] shared*/, pos, 24, 4);
store_kv_cpu(k, v, kc, vc, pos, 1024);                // or store_kv_bf16_cpu for bf16 KV
gqa_decode_cpu(q, kc, vc, pos+1, attn_out, /*kv_bf16=*/false, 6);
sigmoid_mul_cpu(attn_out, gate, 6144);
fp8_gemv_mt(o_w, o_s256, attn_out, t, 5120, 6144);  residual_add_cpu(x, t, 5120);
// then the same MLP block as above
```

**MTP verify (T=2)** on a CPU layer: swap the five `_mt` GEMVs for `fp8_gemv2_mt(w, s256,
x2 /*[2,cols]*/, y2 /*[2,rows]*/, rows, cols)` — one weight pass, both tokens (57.8 GB/s
effective). Norms/conv/deltanet/GQA are per-token (call twice); that matches the GPU flow.

Constraints enforced by throws: fp8 GEMV `cols % 128 == 0`; bf16 GEMV `cols % 16 == 0`;
rmsnorm `cols % 32 == 0`; gated norm `hd % 16 == 0`; `silu_*`/`sigmoid_mul`/`residual_add`
`n % 8 == 0` (all 27B dims qualify). GQA is hard-shaped 24q/4kv/256hd, `kvh = head/6`,
scale 1/16 — 27B only by construction.

## 6. Repro

```
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cl /nologo /arch:AVX2 /O2 /std:c++20 /fp:precise /EHsc /Iinclude src\test_cpu.cpp /Fe:build\test_cpu.exe
build\test_cpu.exe            # 40 checks + benches (~15 s total)
```

Bench numbers above: repeated 3× (`pool scaling` ±0.1 GB/s, shape table ±0.5 GB/s; GQA as
noted). Disassembly check: `dumpbin /disasm build\test_cpu.exe` — the GEMV dequant+ FMA cluster
(around `vpmovzxbw`/`vcvtph2ps`) is register-resident (no `[rbp]` spill traffic; the one
array-form cluster that remains is the exhaustive-test code path, where the array is the API).
