# swarm audit synthesis — 2026-08-25

21-agent audit wave (10 engine internals, 6 reference codebases, 5 web research, 1 budget model).
mission: push INSIG4 further; run Qwen3.8-27B-FP8 (25.65GB text weights) un-requantized on
4070 SUPER (12GB) + 5600X (16GB RAM) + NVMe, heterogeneous NVMe/RAM/VRAM.

## critical bugs found (engine, verified firsthand)

1. **INSIG4 scale dtype mismatch**: quantizer writes F16 scales (tools/quantize_insig4.py:118)
   but every kernel reinterprets as bf16: src/mxfp4_i4.cu:8 (`i4_scale`), src/gemm.cu:326,
   src/prefill.cu:29. Only end-to-end check (build/nll.bat) omits src\mxfp4_i4.cu from the link
   so it never ran. FIX: read as `__half` (checkpoint stays valid).
2. **full-attention parity bug = RoPE smem race**: src/ops.cu:9 + src/prefill.cu:70-72 — after
   `__syncthreads()`, all 256 threads read `mem[0]` (rsqrt norm scale) but threads 0..63 then
   overwrite `mem[0..63]` with roped values before warps 2..7 have read the scale. Fires only
   pos>0, attention-only, nondeterministic — exactly the "later full-attn layers flaky"
   symptom. FIX: dedicated shared slot for the scale.
3. **MTP draft embed has no i4 branch** (src/decode.cu:130): always calls `embed_gather`
   (u8 MXFP4 scales) → INSIG4 spec-decode drafts read garbage.
4. nll.cu:78 lm_head GEMM has no i4 branch (generate.cu WIP does) — nll.dll wrong for INSIG4.
5. Silent early-returns on dim mismatches in launchers (mxfp4_i4.cu:66,147; gemm.cu:351):
   cols%1024!=0 or cols%32!=0 → y unwritten, no error. ab2_q8_i4 hard-requires cols==4096.
6. Graph hazards: post-capture LRU eviction frees pointers baked into instantiated graphs
   (storage.cu:8 vs decode.cu:230); KV-full guard bypassed by graph replay; full-attn KV not
   restored on spec reject (safe only because every reject is followed by another pair).

## model facts (Qwen3.8-27B-FP8, from shard headers — verified)

- Qwen3_5ForConditionalGeneration, text-only engine skips vision (0.92GB).
- 64 layers, full_attention_interval 4 → 48 linear_attention + 16 full_attention ((i&3)==3).
- hidden 5120, intermediate 17408, vocab 248320. embed/lm_head bf16 [248320,5120] (2.543GB each).
- full attn: 24 Q heads (q_proj [12288,5120] = q+gate interleaved per head), 4 KV heads
  (k/v [1024,5120]), head_dim 256, partial rope 64 dims, theta 1e7, q/k norm, output gate,
  GQA group = 24/4 = 6 (kernel must map kvh = head/6, NOT head>>2).
- linear attn (gated DeltaNet): in_proj_qkv [10240,5120] (q 2048 | k 2048 | v 6144),
  in_proj_z [6144,5120], in_proj_a/b [48,5120] bf16, conv1d [10240,1,4] bf16, A_log/dt_bias [48],
  norm [128], out_proj [5120,6144]. k heads 16x128, v heads 48x128 (k-sharing = head/3),
  fp32 ssm state 48x128x128 = 3.15MB/layer.
- quant: F8_E4M3 weights + BF16 weight_scale_inv [ceil(r/128), ceil(c/128)] (0.012% overhead),
  activation_scheme=dynamic. mtp.fc bf16 [5120,10240]; MTP layer = full-attn layer, 1 layer.
- shards: layers-N.safetensors uniform 383.87MB slots (linear 383.88MB data, full-attn 372.33MB
  + pad), mtp.safetensors 477MB, outside.safetensors = embed/lm_head/norm + vision.
- 9B→27B deltas for kernels: hidden 4096→5120, layers 32→64, inter 12288→17408,
  qkv row 8192→10240 (q@0,k@2048,v@6144), a/b 32→48, delta state 24x32x128x128→48x48x128x128,
  conv 8192→10240, q heads 16→24, kvh mapping >>2 → /6, deltanet kh>>1 → /3, q_proj out 8192→12288.

## feasibility math (25.65GB text weights; per-layer ms by tier)

| tier | ms/layer | note |
|---|---|---|
| VRAM (504GB/s GEMV) | 0.76 | also state 302MB r/w per token = 0.6ms |
| RAM→PCIe stream (25GB/s) | 15.4 | |
| RAM CPU-GEMV (40GB/s DRAM) | 9.6 | DRAM-bound beats PCIe; needs CPU layer kernels |
| NVMe (6.8GB/s) | 56.5 | dominates; read-ahead hides latency not bandwidth |

- decode is token-serial (autoregressive) → per-token time = Σ tiers, NVMe reads pipelined
  ~1 token ahead. placements (VRAM L / RAM M / NVMe N): L=21,M=23,N=21 ≈ 1.5s/t → 0.66 tok/s;
  with MTP ×1.6 ≈ 1.05. CPU-compute tier v2: ~0.78 → ~1.2 tok/s. all-stream: 0.27 tok/s.
- lm_head MUST be VRAM-resident (2.5GB, 5.1ms/token; over PCIe it would be 102ms).
- MTP verify batch=2 reads weights once (bandwidth-bound ⇒ free) → net ×1.6 at p=0.6. WIN.
- prefill chunk-major re-streams 25.6GB per 64-tok chunk (16 tok/s); weight-stationary
  (layer-major with activation checkpoints, FlexGen-style) fixes to ~1 pass total.
- GPU compute never binds decode (0.005ms/layer) — this model on this rig is an I/O problem.

## kernel plans (from TRT-LLM / vLLM / exllamav3 / ggml / colibri studies)

- sm_89 FP8 MMA: `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` (Ada-only, 2x bf16
  rate; 4070S ≈ 142 dense TFLOPS). TRT-LLM hand-rolled blockwise sm89 GEMM
  (cpp/.../fp8_blockscale_gemm/): tile 32x128x128, 3 stages, promote-accumulate pattern —
  raw fp8 MMA per 128-K slab, then accum += temp * (SFA[row]*SFB) in registers. Scales f32.
- vLLM on Ada runs block-fp8 as Marlin W8A16 (weight-only, bf16 acts) — block scales
  broadcast to group-wise + 2^120 exponent-bias fold into scales. Confirms w8a16-dequant
  is the pragmatic Ada decode path; native e4m3 MMA is the optimization on top.
- e4m3→fp16 bit trick: `(b&0x7f)<<7` bitcast to fp16 then ×256 — EXACT incl. subnormals
  (uniform 2^8 ratio). e4m3→fp32: `((b&0x7f)<<20) + 0x3C000000` (subnormal-wrong; Qwen fp8
  subnormal rate ~1e-4, treat-as-zero acceptable). CUDA: `cvt.rn.f16x2.e4m3x2` (2/instr, sm_89).
  Zen3: bytes→fp16 lanes<<7 (shuffle+shift) + `_mm256_cvtph_ps` ×256 — exact, F16C 1µop.
- activation dynamic quant (config says activation_scheme: dynamic): per-token-group 1x128,
  scale = absmax/448 (e4m3 max), clamp ±448, floor 1e-10 (vLLM per_token_group_quant.cu).
- INSIG4 format evolution backlog (ranked): EXL3-style Hessian scale fitting (calib-side
  only), e4m3/f16 per-block scales, Hadamard incoherence (needs pow2 — 5120 isn't),
  D2Quant dual-scale down-proj, NanoQuant low-rank binary residual (below 3bpw),
  QTIP trellis (multi-month, only if sub-3bpw needed).
- Windows NVMe: ReadFile+OVERLAPPED+IOCP with FILE_FLAG_NO_BUFFERING, 8-16×1-2MB pinned
  slots, QD 8-16 → 5.5-6.5GB/s on a 7GB/s drive; ~2 cores of overhead. IoRing ≈ +2% (Win11
  22H2+). mmap+PrefetchVirtualMemory NOT reliable for line rate; page-cache double-buffers
  (fatal for >RAM working sets).
- colibri steal-list: SPMC load pool (atomic (gen<<8|idx) CAS claim), O_DIRECT twin handles
  on Windows (compat.h:296-306), early-issue CPU/GPU overlap with 1 sync/device, LFRU score
  (freq<<8|255-age), MADV_WILLNEED pre-touch analog, staging rounds min(4GB, budget/8),
  depth-1 speculation only (acceptance 85% vs 44-62% d2-3).

## engine optimization backlog (9B/INSIG4, ranked by audit)

1. i4 GEMM was unpipelined v1 clone → port v2.1 cp.async pipeline (done this session).
2. Persistent grid-stride GEMV + stage x once; L2 accessPolicyWindow; drop __ldcs on
   L2-resident shapes (bench lies at <=12288 rows: L2-resident).
3. Deeper verify (T=3-4) in spec decode; zero-copy host-visible commit buffer; conditional
   graph nodes; skip MTP lm_head sweep (fuse into verify lm_head GEMM).
4. GEMM: kill fp32 round-trip in dequant (packed __hmul2 bf16x2); decode into B fragments;
   3-stage pipeline; T=128+ chunks for attention+MLP GEMMs (241 FLOP/B at T=64 → ~121 TF ceiling).
5. Attention decode: coalesced Q·K (warp-per-key-row), split-K over tokens, bf16 KV cache.
6. Fused epilogues (silu_mul/sigmoid_mul/residual into GEMM), f32_to_bf16 folded into rmsnorm.
7. Build: -lineinfo, -Xptxas -v, --threads 0, dedupe bench-gemm-blocked.bat (byte-identical
   twin of bench-gemm.bat), bench-mxfp4.dll overwrite bug.

detailed reports: audits/internals.md, audits/references.md, audits/research.md
