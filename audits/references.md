# reference-clone studies (swarm wave 1, 2026-08-25)

## colibri (heterogeneous CPU+GPU+disk engine, C monolith + python planner)

- c/colibri.c (~642KB), backends via loader (cuda/metal/vulkan). MoE-centric: dense weights
  RAM-resident int4; routed experts on disk promoted by rank.
- placement: planner c/resource_plan.py:798-980 — VRAM = free-2GB reserve, RAM = avail*0.88,
  cache slots = cache_bytes/per-layer-median; C side pin_load sorts (layer,expert,count)
  from routing history (.coli_usage); VRAM prefix + RAM suffix; staging rounds
  min(4GB, budget/8) experts, host slabs freed after upload (colibri.c:9027-9044).
- prefetch: PILOT router predicts layer L+1 top-K from L's post-attn state (71.6% recall vs
  41.3% prev-token), lock-free 1P/1C ring pilot_q[4096] to dedicated IO thread (inline
  fadvise blocks ~0.5ms when disk saturated); PILOT_K=8. COUPLE learned pair-file prefetch K=8.
- windows IO (compat.h): pread→ReadFile+OVERLAPPED raw handles (:158-193), fadvise WILLNEED →
  thread-safe overlapped background ReadFile (:114-156), O_DIRECT → FILE_FLAG_NO_BUFFERING
  twin fd (:296-306); PIPE async load pool ON by default on Windows.
- steal-list: SPMC load pool (atomic (gen<<8)|idx CAS claim, 8 workers clamp 16, 64-slot
  batch, release-store publish + cond-broadcast) (colibri.c:3343-3424); LFRU score
  (freq<<8)|min(255-age) with 25%+4 swap hysteresis (c/tier.h:27-58); early-issue overlap —
  issue all device batches, compute CPU experts meanwhile, 1 take() per device (:5501-5533);
  mmap pre-touch MADV_WILLNEED 16KB-aligned + volatile touch per 4096B (:2679-2692);
  dual-SSD striped reads ≥4MB; autotune anti-drift (reverse-order rerun, ≥3%, p99 ≤1.2x).
- defaults: PIPE_WORKERS=8, DRAFT depth-1 only (85% acceptance vs 44-62% d2-3), scratch
  27 slots/device, expert batch 64.

## TensorRT-LLM (FP8 on sm_89)

- CUTLASS via CPM v4.4.2, header-only. SM89 e4m3 MMA hand-rolled PTX:
  mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32, A=4xu32 B=2xu32 C=4xf32
  (cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/ada_blockwise_gemm/
  sm89_utils.cuh:41-82); TiledMMA 32x32x32, 4 warps.
- blockwise GEMM 1d1d (sm89_fp8_gemm_1d1d.cuh:344-445): per 128-K slab cute::gemm on RAW
  fp8 into temp, then promote(): accum += temp * scale, scale = SFA(row)*SFB(0) computed
  once per k-tile — dequant-then-MMA avoided AND epilogue-only scaling avoided.
  granularity M=1,N=128,K=128; scales f32; SFA col-major (M padded 4), SFB row-major.
- launch: tile 32x128x128, 3 stages, 128 threads, grid (ceil(M/32), ceil(N/128))
  (fp8_blockscale_gemm_kernel.cuh:681-711); cp.async multistage ring + predicated small-M
  residue; LDSM + Swizzle<3,4,3>.
- fused 1x128 activation quant: warp/row, 4x128 per warp, amax via 3 shfl_xor(width 8),
  scale = amax/448, optional UE8M0 rounding, writes 128x4 swizzled scale layout
  (fp8_blockscale_quant_packed.cu:66-133,215).
- FP8 GEMV batch≤16: CUDA-core kernel, int4 loads, NumericArrayConverter<float,e4m3,4>,
  scalar FMA, TILE_N=2, BLOCK 128, k%16==0 (weightOnlyBatchedGemv/cudaCoreGemm.cu:38-135).
- per-token quant: scale=448/rowMax clamped by min 1/448 (quantization.cuh:188-266).
- FP8 KV: decoder MMHA template, kv_scale_quant_orig/dequant pairs per head.

## vLLM (block-fp8 = the checkpoint's quant scheme)

- sm_89 CANNOT use blockwise CUTLASS (SM90+ only, scaled_mm_entry.cu:161-172); Ada falls
  back to MarlinFP8 weight-only W8A16 bf16-acts (marlin.py:41-45) — block scales [K/128,
  N/128] broadcast to group-wise via repeat_interleave + 2^120 exponent-bias folded into
  scales (marlin_utils_fp8.py:34-47,191-195).
- sm_89 per-tensor fp8 CUTLASS tile table exists (scaled_mm_c2x_sm89_fp8_dispatch.cuh:
  343-382): M16→16x64x128/16x128x64, M64→64x64x128 5-stage, 128x128x128 etc.
- sm90 blockwise reference impl: scales<1,128,128>, tile 128x128x128 cooperative; SFA/SFB
  separate mainloop args; per-K-tile promote-accum (scaled_mm_blockwise_sm90_fp8_dispatch).
- per-token-group quant kernel (per_token_group_quant.cu:100-164): 16thr/group, absmax shfl,
  y_s = absmax/448, eps 1e-10, clamp ±448, f32 scales; register fast path GROUP_SIZE==128
  8thr/group no smem (:300-455); PDL launch.
- padding: B padded to K,N mult 16 with scale rows 1.0 (cutlass.py:206-247).
- lm_head/embed stay bf16 via modules_to_not_convert (fp8.py:160-185); weight_scale_inv
  param name honored (DeepSeek-style, :350-351).

## ggml (CPU+GPU split)

- scheduler: ops follow weights (buffer usage WEIGHTS picks backend); split ranges contiguous
  per backend; event double-buffer ring instead of syncs; INPUT tensors copied sync
  (ggml-backend.cpp:910-1740). GGML_OP_OFFLOAD_MIN_BATCH=32 (ggml-cuda.cu:5510).
- AVX2 quant GEMV (Zen3 path): per QK=32 block, scale fp16 mul broadcast, nibble expand,
  _mm256_sign_epi8+maddubs+madd → cvtepi32_ps, fmadd fp32 8-lane, hsum at end
  (arch/x86/quants.c:718-741). threadpool: chunk 16 (64 if single-dim), work-stealing
  atomic_fetch_add, spin barrier (ggml-cpu.c:1396-1451).
- pinned host buffer type for activations (cudaMallocHost) enables async DMA both ways;
  mmap'd caller memory → cudaHostRegister ReadOnly opt-in (ggml-cuda.cu:4637-4658).
- NO GGML_TYPE_F8; e4m3 only as MXFP4 scale format.

## llama.cpp

- n_gpu_layers split by per-GPU free memory (llama-model.cpp:1385-1410); embed always CPU;
  --override-tensor regex placement (model-loader.cpp:1178-1203).
- windows mmap: CreateFileMappingA + PrefetchVirtualMemory via GetProcAddress, prefetch=-1
  whole file (llama-mmap.cpp:543-576); mlock = VirtualLock grow-loop; unmap_weight after
  load for >RAM.
- activation crossing per boundary ≈ hidden*4B ≈ 20KB/token (n_embd 5120 f32).
- threads: physical cores on Windows (common.cpp:118-148). no FP8 type at all.
- spec: simple/eagle3/dflash/mtp impls; MTP reuses trunk nextn layer (speculative.cpp:1281+).

## exllamav3

- EXL3 QTIP-style: barrier-free warp-split-K GEMV (warps own K-chunks, zero block syncs;
  CFG0 512thr/WK16, CFG1 256thr/WK8); __ldcs streaming + compile-time register prefetch
  ring PF stages; in-warp dequant via __shfl + funnel shifts, mma.m16n8k16 fp16-acc folded
  to fp32 every FOLD tiles; dp4a codebook decode (mul1 affine in bytesum → single dp4a
  decodes+MACs); fused Hadamard epilogue/prologue with packed half4 scales + grid.sync;
  GEMM: cp.async SH_STAGES 4-6 / FRAG_STAGES 3-5, XOR-swizzled staging, split-K via global
  lock array; per-arch shape map sm_89 CC_ADA; graph param patching via
  cudaGraphExecKernelNodeSetParams memcmp-diff (graph.cu:30-184).
- no fp8 CUDA kernels; fp8 only dequant-on-load in python (linear.py:133-217).
