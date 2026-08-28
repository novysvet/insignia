# web research (swarm wave 1, 2026-08-25)

## sm_89 FP8 MMA (Ada)

- instruction: mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 — Ada-only (sm_90 uses
  wgmma). A frag 4xb32 (16 e4m3/thread, 4 consecutive K per reg), B 2xb32, C/D 4xf32; same
  layout table as u8 m16n8k32. rate = 2x bf16 per SM; 4070 SUPER (AD104, 56 SM, 7168 cores
  @2.475GHz) = 70.96 bf16 / 141.93 fp8 dense TFLOPS. NOTE: libraries accumulate fp32 on
  GeForce Ada capping at ~half marketing; full rate needs fp16 accum (4090: 473-587 TF by
  Bakhshaliyev vs 330 cuBLASLt).
- ldmatrix is 16-bit granular: fp8 needs K-major/swizzled layout or ld+PRMT reshuffle
  (ThunderKittens 8x16/16x32 tiles via ldmatrix.x4).
- cvt: cvt.rn.f16x2.e4m3x2 / cvt.rn.satfinite.e4m3x2.f16x2 (sm_89+, 2 lanes per instr).
- CUTLASS: SM89 fp8 only via 2.x API (example 58_ada_fp8_gemm, CUTLASS≥3.5, CUDA≥12.4);
  3.x collective builders have NO sm_89 fp8 specialization → manual PTX or 2.x.

## e4m3 decode tricks

- exact incl. subnormals: (b&0x7f)<<7 bitcast fp16, ×256 (2^8 uniform across normal+subnormal;
  intel-xpu-triton#7028, -21..-35% kernel time vs LUT in vLLM). fp32 route ((b&0x7f)<<20
  +0x3C000000) is subnormal-wrong — Qwen fp8 (RTN, scale=448/amax) has ~1e-4 subnormal
  rate; treat-as-zero acceptable. cvt.rn.f16x2.e4m3x2 = exact, 2/instr.
- Zen3: no AVX512-FP8; arithmetic fp32 decode ~4 ops/8elem or the fp16 trick + _mm256_cvtph_ps
  (F16C 1µop); vpgatherdd ~1µop/elem + LLVM fastGather disabled on zen3 → avoid gathers.

## windows NVMe IO

- ReadFile+OVERLAPPED+IOCP + FILE_FLAG_NO_BUFFERING is the workhorse; IoRing (Win11 22H2+,
  ioringapi.h) only ~2% faster at high IOPS (Yarden Shafir benchmarks); DirectStorage usable
  from non-game apps but targets D3D12 resources. NO_BUFFERING: offsets/lengths/buffers
  sector-aligned (4096 safe); avoids cache-manager double-buffering + standby pollution —
  required when working set > RAM. realistic: 5.5-6.5GB/s sustained from 7GB/s-class drive
  with QD 8-16 × 1-4MB; ~1 core per 6-7GB/s IO + 0.7 core DPC; mmap read-ahead only ~64KB
  (Raymond Chen), PrefetchVirtualMemory better but still cache-backed.

## NVMe LLM engines (measured)

- AirLLM 70B: users 5-35 s/tok (NVMe-bound); FlexGen OPT-175B 1xT4+208GB+SSD 0.69 tok/s
  (batch 256), 30B 7.32 tok/s; DeepSpeed Zero-Inference 43 vs 30 tok/s CPU vs NVMe offload
  (batch-amortized); KTransformers 671B ~9-14 tok/s (2 Xeon + 24GB, MoE); Colibri GLM-5.2
  744B 0.05-0.1 cold → 10.05 with CUDA expert tier.
- dense decode = full weights read per token; spec verification free (batched), each draft
  forward re-reads streamed layers; FlexGen overlap gain only 1.17-1.25x (hides latency not
  bandwidth); prefill = block-pipelined (load layer once, run whole block of positions;
  utilization 82% vs 13% decode).

## quant frontier mid-2026 (INSIG4 evolution)

- NanoQuant (arxiv 2602.06694): W ≈ s1 ⊙ (U·V^T) ⊙ s2^T, U/V ±1 packed, rank r = compression
  knob (1.0/0.8/0.55 bpw); K-FAC LB-ADMM + STE, 0.26M tokens; L2-7B ppl 5.47→10.34 @1.0bpw;
  inference = two chained binary matvecs, FMA-only; traffic shrinks <16x since both factors
  read. thin citation graph so far.
- leaders 2-4bpw: QTIP (trellis, 7.4 @2bpw, bitshift decode), EXL3 (QTIP+2nd-order scales,
  4bpw beats EXL2-6bpw, DRAM-bound at 4bpw on 4090), QuIP# (8.5 @2, Hadamard+LUT), AQLM
  (8.5 @2, +10-20% residual matmul cost), VPTQ (~9 @2.1, LUT), D2Quant (2-bit dual-scale,
  overhead-free), SSVQ. no published e4m3-base mixed formats — greenfield.
- ranked for INSIG4: (1) Hessian-guided scale fitting (EXL3-style, calib-side only);
  (2) smaller scale dtypes; (3) Hadamard incoherence (pow2 issue at 5120);
  (4) D2Quant dual-scale for down_proj; (5) NanoQuant binary residual below 3bpw;
  (6) trellis only if sub-3bpw becomes a goal.

## 27B FP8 budget model (computed, verified vs shard headers)

- per-layer weights: linear 383.88MB (fp8 382.7 + scales 0.047 + bf16 misc 1.1); full-attn
  data 372.33MB in 383.87MB padded slot — STREAM BY TENSOR BYTES NOT FILE SIZE.
- text total 29.95GB incl. bf16 embed+lm_head (2.543GB each); backbone 24.38GB; mtp 477MB.
- per-token state: DeltaNet S 151MB fp32 (const, ctx-free), conv 5.9MB, KV 64KiB/tok
  (0.54GB@8k). state r+w per token = 302MB = 0.6ms @504GB/s.
- VRAM plan: 12,879MB − 1,600 (kv@8k + state + ws) − 2,543 lm_head − 477 mtp ≈ 8.26GB
  → L=21 resident; RAM M=23 (8.8GB safe; embed stays mmap'd, 10KB/tok); N=21-24 NVMe.
- decode tok/s: plain 0.66-0.72; CPU-GEMV tier v2 0.78; +MTP ×1.6 → 1.05-1.25. GPU compute
  never binds. NVMe tier is the ~15x bottleneck vs pure-GPU ceiling (20 tok/s).
- MTP: net WIN at every placement (×1.6 @ p=0.6; break-even p<0.02).
