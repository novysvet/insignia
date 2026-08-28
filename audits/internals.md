# engine internals audit (swarm wave 1, 2026-08-25)

## INSIG4 format + kernels

- format: E2M1 nibbles 8/u32 LE (quantize_insig4.py:59-63); per-64-elem super-group one
  MSE-optimal scale, golden-section over [0.6,1.6]x amax/6, 10 iters (py:26-38); scales F16
  [rows, cols/64] (py:64,118). byte-identical to MLX MXFP4 4.125bpw, +5.9dB SQNR
  (mxfp4_i4.cu:5-6, quant_study.py:132-137). kernels index scale s[g>>1]
  (mxfp4_i4.cu:8, gemm.cu:326, prefill.cu:29).
- GEMV v2_i4: 256thr, 1 warp/row, x staged smem-transposed, 16-float LUT, 1 uint4/group/lane
  __ldcs, 4 FMA partials + per-group scale FMA, shfl reduce (mxfp4_i4.cu:12-64);
  needs cols%1024==0 (:66). pair q8_i4 dp4a: 256-entry u64 byte-broadcast LUT + __byte_perm,
  4 dp4a/group, scale*0.5 fold (:74-150). fused ab2_q8_i4: <<<1,256>>>, hardwires 2x128 groups
  (:153-159,235). get_row_i4: grid<=128 (:239-249). GEMM mlx_i4 = v1 clone, KT=32/NT=64,
  wmma 16x16x16, single-buffered, no cp.async (gemm.cu:298-353).
- no unit test or bench touches any i4 kernel; only end-to-end check is NLL (nll_compare.py)
  but build/nll.bat:5 omits src\mxfp4_i4.cu → unlinkable. bench numbers: ~150GiB/s @4096x4096
  (L2-resident, meaningless), 379-435GiB/s real streaming shapes, 54GiB/s legacy GEMV.
- correctness risks: F16-scales-read-as-bf16 (fixed this session); ab2/embed_i4/lm_head-stride
  hardcode 4096 (blocks 27B); silent early-return launchers; GEMM writes 64 rows regardless
  of T (Y must have 64 rows).

## MXFP4 GEMV path

- production: mxfp4_gemv_v2_kernel (mxfp4.cu:88-142) 1 warp/row, 128-bit __ldcs, smem x stage,
  E8M0 scale via `__int_as_float(code<<23)` (:128). dp4a family :244-665.
- variant sweep commit 4bd0513: multi-row warps win in bench (L2-resident <=36MB) but lose
  end-to-end on DRAM streaming. 248320x4096 ≈ 379GiB/s ≈ 75% of 504GB/s peak.
- why 4096x4096 shows only ~150GiB/s: shape fits 48MB L2; launch+staging bound; x re-read
  rows/8 times; __ldcs actively evicts L2-resident matrix; scattered stride-`groups` smem
  stores; no cp.async; 6 GEMV launches per layer fully drained (decode.cu:31-35).
- ranked opts: persistent grid-stride GEMV (stage x once); L2 accessPolicyWindow + __ldca for
  small shapes; cp.async x staging; u32-vectorized scale fetch; minBlocks occupancy; PDL
  between the 4-6 GEMVs per layer.

## prefill GEMM (v2.1)

- v2.1 mxfp4_gemm_v21_kernel (gemm.cu:207-292): 64x32 tile, KT=64, cp.async.cg 16B double
  buffer, 256-entry bf16 pair-LUT, wmma 16x16x16, 8 warps 4x2, ~30.7KB smem → 3 blocks/SM.
  dequant Braw→Bs sits on critical path between two __syncthreads (:274-275).
- analytic ceiling at T=64: 241 FLOP/weight-byte → ~121 TFLOPS (73% of bf16 peak).
  4096-row down_proj = 128 blocks / 56 SM = 2.29 waves (30% tail waste).
- ranked opts: T=128/256 chunks; dequant off critical path (interleave with MMA or decode
  straight into B fragments); kill fp32 round-trip (packed __hmul2 with pow2-scale int add);
  3-4 stages; persistent kernel / NT=64; fuse silu_mul/sigmoid_mul/residual + f32_to_bf16
  into rmsnorm; weight upload overlap (acquire syncs per tensor, storage.cu:8-9).
- correctness: host double reference 8 sampled rows x every 7th token, rel-err 2e-2
  (bench_gemm.cu:39-65); multistep parity worst_layer_cos 0.9998-0.99999 (build log).

## decode + spec + graphs

- spec step: MTP draft (1 token) → verify pair T=2 through all 32 layers → lm_head pair GEMV
  (one weight sweep, both rows) → argmax row0=t2, row1=after → spec_commit (accept if
  draft==t2) → rollback (delta+conv state from row-0 snapshots; full-attn KV left stale,
  safe only because next pair overwrites the same slots).
- capture_spec: prologue+mtp+setup+pair+commit+rollback captured; everything device-state
  driven (pos_dev slots), replay self-feeding; host only checks committed ids every 4 steps.
- single stream for everything incl readback; per-4-steps full sync.
- opts ranked: deeper verify T=3-4 (pf buffers already sized 64); zero-copy host-visible
  commit buffer (kill both syncs); skip MTP lm_head sweep 0.51GB/step (fuse into verify
  lm_head); conditional graph nodes; second stream for readback.
- 121 tok/s ≈ 12-16ms/step ≈ ~5.5GB reads at 400-500GB/s effective (weight-traffic bound).

## storage/residency

- mmap: CreateFileW FILE_FLAG_RANDOM_ACCESS + CreateFileMapping + MapViewOfFile
  (model_file.cpp:27-31). No NO_BUFFERING/OVERLAPPED/threads. Reads fault via page cache.
- acquire-miss: cudaMalloc + cudaMemcpyAsync(pageable mmap) + cudaStreamSynchronize per
  tensor (storage.cu:9). LRU+pin: pins only held around each op → nearly all evictable.
  budget device-only; host_pinned tier declared (insignia_storage.hpp:11) never implemented.
- missing for 25.4GB > VRAM+RAM: NO_BUFFERING+IOCP read ring; pinned staging ring; per-layer
  prefetch pipeline; page-cache discard (OfferVirtualMemory); graph-pin set; VRAM slot reuse
  (pool); dual budget; layer-execution-ordered index.
- capture hazard: acquire-miss during capture would call cudaMalloc/sync inside capture
  (fails); warmup ordering is the only protection. Post-capture LRU eviction → dangling
  graph pointers.

## model loader

- DType {f32,bf16,f16,u8,u32,i8} (insignia_model.hpp:10); index_safetensors.py:29-31 RAISES
  on F8_E4M3. INSIDX01 single-file, name-sorted, exact-match binary search find().
- 27B needs: f8 dtype id, multi-shard (66 payloads), HF naming (model.language_model.layers.N
  vs MLX language_model.model.layers.N; scales .weight_scale_inv vs .scales), QuantMatrix
  fp8 variant (weight e4m3 + bf16 [r/128,c/128] scales) + bf16-dense GEMV for unquantized
  tensors (in_proj_a/b, conv1d, A_log/dt_bias, mtp.fc, embed/lm_head).
- shard offsets: data starts at 8+header, tensors contiguous, no padding, all in one file
  per layer; layout groups bf16 tensors first then F8.

## attention/deltanet kernels

- decode qk dot fully uncoalesced (attention.cu:7: per-thread d-loop, stride-4KB lane
  addresses); grid 16 blocks uses 16/56 SMs; KV fp32 268MB@ctx4096; deltanet_decode 32x128
  threads ≈3% occupancy, 2MB state traffic/layer; per-token a/b GEMV loop in prefill
  (decode.cu:71-72).
- parity hunt cleared: kv strides, chunk masking, softmax scale, gate broadcast (fixed
  7ccfe57), rope pairing, mrope (no-op for text). prime suspect = the RoPE smem race
  (confirmed). secondary: bf16 activation staging drift in prefill GEMMs, __powf/__cosf
  fast-math rope at high pos, dp4a int8 quant drift.
- NOTE: HF ref Q/K-norm is zero-centered (1+w) with zeros-init weights; engine multiplies
  raw w and matches local numpy ref — checkpoint stores effective weights (benign).

## build system

- all 30 .bat: identical template, `-arch=sm_89 -O3 --use_fast_math -std=c++20 -Iinclude`;
  no -lineinfo/-Xptxas -v/--threads 0/rdc/lto. exes blocked by Smart App Control →
  python+ctypes rundll.py loads DLLs (dllshim.cu exports dll_run→wmain).
- bench-gemm.bat ≡ bench-gemm-blocked.bat (byte-identical, both emit bench-mxfp4.dll);
  bench-gemm.dll stale. no build-all. mk.bat cl /LD /O2. test-model.bat cl /O2.
- known-good generate: nvcc ... src\{model_file,storage,mxfp4,mxfp4_i4,qwen35,qwen_kernels,
  ops,attention,deltanet,decode,prefill,gemm,generate}.cu(pp) src\dllshim.cu -o build\generate.dll

## parity/test infra

- numpy 2.5.2, NO torch/mlx. pattern: CUDA dumps f32 → numpy recomputes from checkpoint →
  max/mean/cosine, eyeballed (no asserts in python; CUDA tests have thresholds:
  test_mxfp4 2e-4, test_attention 2e-5, test_deltanet 3e-3, test_ops 2e-6).
- MXFP4 9B checkpoint: ~/.cache/huggingface/hub/models--sleepyeldrazi--Qwen3.5-9B-MXFP4-MTP.
- nll_compare.py = A/B mxfp4 vs insig4 indexes, no external oracle.
- FP8 parity plan: numpy ref with manual e4m3 decode + 128x128 scale (extend dq() pattern in
  reference_all_layers.py), per-layer cosine ≥1-1e-6, layer3 seams script, greedy-chain
  regression freeze. no bf16 master weights exist → no external oracle except one-time
  vLLM/TRT NLL run.
