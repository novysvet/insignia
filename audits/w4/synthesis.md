# w4 swarm synthesis — 2026-08-25

23 agents (1 network-failed: decode-wiring; coverage folded into this synthesis via
test-coverage + direct runs). Full reports in audits/w4/*.md.

## State correction (what w3 master plan got wrong)

- INSIDX02 index is BUILT and byte-perfect (66 shards, 1273 tensors, CRC-verified,
  R0 data-green) — but `model_file.cpp:23` parses INSIDX01 ONLY. Loader-red.
- Only 8/66 shards (layers 0,1,2,4,5,6,8,9, data_start=2600) have F8 ≡8 mod 16 — all
  inside the always-VRAM L=19 block. Master plan's "every shard" was wrong; the
  "pad in plan builder" fix is inexpressible — fix = publish-time +8 memmove or
  uint2-tolerant loads + acquire assert.
- streaming smoke EXISTS and passed byte-exact (not a mk target).
- Active INSIG4 file is the BROKEN encode (12.01 dB SQNR, 6.2 dB worse than MLX;
  scales 0.55x optimal) with un-shifted norms. `-good` (20.13 dB, +1.9 dB over MLX)
  is correct encode but norms still raw. The fixed quantizer (with +1 bake) has
  NEVER been run. Source of truth for quantization = Qwen--Qwen3.5-9B BF16 HF repo
  (re-quantizing from MLX-decoded f32 costs ~2 dB).

## Critical fix list (P0 — blocks everything)

1. fp8_gemv: 68 KB dynamic smem at cols=17408, no cudaFuncSetAttribute, no launch
   check → fails every 27B MLP layer. fp8_gemv2 exceeds hw cap at 17408 → route to
   fp8_gemm. [w4/fp8-kernels N1, engine27-gap, tier-dispatch]
2. ModelFile INSIDX02 parser (follow index27.py bytes exactly — no scale_idx
   field; records HBB,name,shape,HQQ). [w4/index-loader]
3. A_log BF16→F32 widen dispatch (decode.cu:82/128, prefill.cu:212,
   qwen_kernels.cu:9; 27B has BF16 on disk). [w4/safety N1]
4. Zero-center: engine rmsnorm Z=false at 9 sites + qk_norm_rope kernels raw;
   27B needs (1+w) at runtime (bf16 checkpoint is raw). 9B INSIG4 bakes +1 at
   quantize (regen required). linear_attn.norm stays RAW everywhere. [w4/quantizer]
5. kvh=head>>2 → head/6 at attention.cu:7 + prefill.cu:103. [w4/shapes-sweep]
6. deltanet params<<<1,32>>> skips heads 32-47 at n=48. [w4/attn-deltanet]
7. CpuPool::launch recursive-mutex UB (hpp:243-251, one-line unlock). [w4/cpu-tier]
8. NvmeReader::retire leaks 1 event HANDLE/layer/epoch (streaming.cu:239); empty-plan
   path never fires callback (hang trap, :166). [w4/streaming]
9. mk.py flags-hash wipe guard deletes all objs but rebuilds only stale subset →
   link FileNotFoundError; io-bench wrong shim + stale future flags; cl /utf-8 +
   /Fo:build\obj\host. [w4/build-system]
10. WKind dispatch missing at 12 decode.cu sites (fp8/bf16 → mxfp4 path = garbage).
    [w4/index-loader, safety N3]
11. test_attention.cu + test_mtp.cu don't compile (stale signatures). [w4/shapes]
12. row0_snap=nullptr conv fallback corrupts state at tail tiles T=1/2
    (prefill.cu:187-199) — dormant until stationary prefill. [w4/prefill27 P1]

## INSIG4 optimization verdicts (measured, w4/insig4-quality + insig4-evolution)

- Current good encode: 20.13 dB mean SQNR vs BF16, +1.9 dB over MLX MXFP4 @ 4.25 bpw.
- **#1 WIN — Hessian-weighted scale fitting (quantizer-only, zero bytes, zero kernel
  changes)**: s* = Σ h w ĝ / Σ h ĝ² with per-column activation 2nd moments h,
  2-3 Lloyd iters. Measured +9.4…+14.3 dB OUTPUT SQNR on DeltaNet projections/attn.q,
  +4-6 dB on MLP gate/up. Fit/eval split validated. ~3x faster than grid search.
- **#2 — top-0.5% fp16 outlier sidecar** (+0.16 bpw, +5 dB on top of #1): outliers
  → code 0 in main matrix (kernels untouched) + ~30-line CSR GEMV. Phase 2.
- Rejected by measurement: e8m0 scales (−1.8 dB), bf16≡fp16, int4-sym < E2M1,
  g32 (+287 MB for +0.6 dB), low-rank residual (10-30x worse dB/bpw).
- Most sensitive tensors: mlp.gate/up L0/L3, in_proj_a/b, late k_proj.
- Degradation grows with depth; DeltaNet ≈ attn on average.

## Perf verdicts (measured where marked)

- CPU pool: 9-11 workers → 49-54.5 GB/s (vs 35.9 @ 6) — SMT DOES add BW (w3 wrong);
  layer 7.77-12.8 ms honest range; pair GEMV spills (fix via 16-w chunking).
- GEMV v2: 436 GiB/s @ [248320,4096] (93% DRAM), 334 @ L2-resident 8192 rows;
  small rows collapse 139 GiB/s (wave quantization → split-K).
- gqa_decode 27B ctx2048: 1.4-5.1 ms/token as-is (scattered K) → split-K (dim3(24,4),
  warp-per-row, 98.7 KB scratch) → ~37-42 µs/layer. Crossover ctx≈700.
- DeltaNet 27B decode: 0.60-0.72 ms/token; bf16 state → 0.36 (needs drift experiment).
- fp8 GEMM at T≤64 is DRAM-bound (crossover T≈282) — TRT-LLM port buys precision
  (+5-10% exactness via f32 promote), not TFLOPS. Keep GEMV.
- Tier treadmill simulated: T_step ≈ 1,736 ms v2 (Z21/C9/N15) = 0.58 single /
  0.92 MTP-D1 tok/s. Two scheduling rules save ~100 ms: last N layer = 63;
  begin_epoch(s+1) at last release.
- Colibri steal applied: early-issue/take treadmill (qt_issue→CPU-miss→qt_take,
  1 sync/device) for generate27; LFRU/PILOT rejected (static baked placement wins).

## 27B architecture decisions (w4 consensus)

- New src/decode27.cu (shape-specialized), NOT in-place retarget of decode.cu —
  9B file is a frozen regression asset mid-parity-hunt; kernels shared via
  template<int QH>. DecodeWorkspace27 ≈ 598 MiB (delta_state 144 MiB etc.).
- TieredStorage2: manifest as data (INSIGM01, v1/v1.5/v2 files), baked tier_of[64] +
  TagPtr table; startup order lm_head+MTP→VRAM (refuse if short) → smalls arena →
  V slabs (16B rebasing at load) → Z pinned → C VirtualLock → N feeder plans.
- Spec: eager D=1/T=2, draft at step TAIL (embed D2H fused); D=3/T=4 (NOT D=4 —
  628 MB snaps + Z21 exceeds WDDM cap; D=3 = 471 MB fits).
- Prefill: weight-stationary; C layers stage to GPU (CPU GEMM too slow: 2.6 s/layer);
  bench fp8_gemm T=64 first (12-23 TF measured on mxfp4 v21 chain).
- Parity: reference27.py needs MTP path + dump-compare subcommands; golden ids via
  reference27 enc + self-determinism bootstrap.

## Execution order (this session)

P0 fixes → quantizer Hessian-fit + regen INSIG4 (background: minutes) → 9B
regression green → bf16 kernels → ModelFile v2 → TieredStorage2 + decode27 +
generate27 (v1 all-stream) → parity ladder R3→R9 → perf tuning (split-K GQA,
CPU pool 9-11, MTP eager) → v1.5/v2 manifests.

## SESSION ADDENDUM (implementation findings)

- **UVA zero-copy is DEAD on this box (WDDM)**: kernels fault reading cudaHostRegister'd
  AND cudaHostAlloc'd memory (src/test_uva.cu probe, plain loads, both paths illegal).
  The master plan's "GPU consumes ring via UVA zero-copy" and the w4 tier-dispatch
  "verified in fp8.cu" claim (static analysis only) are wrong at runtime. Architecture
  pivoted: every non-VRAM tier stages slot->VRAM via cudaMemcpyAsync before compute
  (acquire_staged). Cost ~19 ms/layer over PCIe ~= the UVA 21 ms estimate; v1/v1.5/v2
  step predictions survive. Z-tier (v1.5) must stage the same way.
- fp8_gemm cp.async on host memory: also illegal (sanitizer-confirmed) — staging covers it.
- 27B v1 all-stream ENGINE RUNS: prefill + greedy decode, 0.148 tok/s bring-up
  (staged, naive); first-next on "760,3841" = 6186 (engine), reference27 pending.
- ModelFile v2 (INSIDX02) landed with a 12-byte header over-read fix (v1 Header struct
  consumed count+payload fields that don't exist in v2 — tensor count read garbage).
- nvcudart_hybrid64.dll (CUDA 13.3 default for new exes) is NOT loadable on this box —
  exes fail with ENTRYPOINT_NOT_FOUND; DLLs (delay-load) work via rundll. 27B driver
  stays a DLL.
