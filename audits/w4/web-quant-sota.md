# Web audit: quant / FP8 / CPU-GEMV / heterogeneous / MTP-spec SOTA (w4, 2026-08-25)

Web research for Insignia (RTX 4070 SUPER sm_89, Ryzen 5600X Zen3 AVX2, Windows, CUDA 13.3,
Qwen3.5-9B MTP, INSIG4 ≈ 4.25 bpw = E2M1 nibbles MLX-packed + fp16 scale per 64-elem
super-group). Scope: 2025-2026 state of the art, actionable extraction. Companions:
`audits/research.md` (wave-1 quant frontier), `audits/w3/web-fp8-ada.md` (Ada FP8 roofline),
`audits/w3/web-hetero-sota.md` (FlexGen/AirLLM/Colibri lineage), `audits/w4/quantizer.md` +
`audits/w4/insig4-quality.md` (our encoder state). Rig numbers cited from those + AGENTS.md:
MXFP4 matvec ~150 GiB/s, CPU pooled fp8/e4m3 GEMV 35.9 GB/s, MTP draft ~4 ms warm.

## TL;DR (10 lines)

1. **ScaleSweep (arXiv 2606.07618, Jun 2026)**: bounded fp8-bit-pattern sweep around the
   absmax base scale for FP4 block formats, WMSE-weighted by diag(X^T X). **Quantizer-only,
   format unchanged, +1-3.5 acc-recovery points.** The single highest value/cost item for
   INSIG4's fp16 super-group scales.
2. **Non-pow2 rotation is a solved problem**: QuIP# factorizes n = p·q (p = largest pow2)
   and uses block-diagonal diag(H_p,...,H_p). Our 4096 = clean H_4096; 5120 = 5 × H_1024
   block-diagonal. KronQ (2607.07964) adds *output-side* (gradient-Hessian) incoherence —
   rescues extreme column outliers GPTQ cannot touch. FlatQuant/SingleQuant are the
   learnable/closed-form alternatives; all fold into weights, engine sees only a cheap
   activation FWHT.
3. **GPTAQ (2504.02692, ICML 2025)** = "GPTQv2": asymmetric calibration objective on top of
   GPTQ + act-order; W4 Llama-3-8B 6.42 vs GPTQ 8.61 ppl. GPTQModel (modelcloud) ships
   first-order error compensation. Quantizer-only for us (E2M1 grid works fine as the
   codebook).
4. **FP8 blockwise on Ada in 2026: nothing new below Hopper.** CUTLASS 4.5.2 changelog
   confirms sm89 FP8 = 2.x API only; 2D blockwise scaling kernels are sm90/sm100/sm120.
   Machete is WGMMA/Hopper-only. Best T=1 path on Ada remains Marlin-style W8A16 (w3 audit).
   Measured (SqueezeBits, H100): at batch-1 memory-bound, weight-only W4A16 gains 80-100%
   vs W8A8's ~40% — fewer weight bytes wins; W8A8 only overtakes at compute-bound batch.
   At equal 1 B/weight (w8a8 vs w8a16) T=1 GEMV is a wash (both <12% of compute peak);
   no public Ada blockwise-w8a8 GEMV GB/s number exists to this date.
5. **CPU FP8 GEMV**: llama.cpp upstream has NO F8 weight type (types 41/42 = Q1_0/Q2_0 in
   master ggml.h); a DeepSeek-V4 fork ships `F8_E4M3_B128` (e4m3 + e8m0 per 128 block)
   with an **AVX2-only** dot kernel. llama.cpp's MXFP4 CPU kernel (in our local clone) =
   pshufb LUT E2M1→int8 + vpmaddubsw/madd + scalar fp16/ue4m0 scale FMADD. Zen3 practical
   ceiling ≈ 80-85% of 57.6 GB/s DDR4-3600 → our 35.9 GB/s pooled has ~1.3-1.45× headroom.
6. **WDDM pinned cap is real and quantified**: cudaHostAlloc page-locked allocations cap
   at **~50% of physical RAM on Windows 10/11** (NVIDIA forum). 15.9 GiB → budget ≤ ~7.9
   GiB pinnable RAM tier; WDDM mode itself also costs measurable PCIe throughput (community
   FLUX/LLM reports).
7. **llama.cpp merged MTP (PR #22673, 2026-05-16) for exactly our family** (Qwen3.5/3.6
   NextN heads, tested on dense 27B + MoE 35B-A3B): 70-90% acceptance at temp 0, ~2× tok/s.
   The killer add-on: **`--spec-draft-p-min` (PR #22397)** — skip low-confidence drafts →
   Qwen3.6-27B Q4 on 3090Ti went 29 → 48.9 tok/s (+68%), holding flat at long outputs
   where DFlash (separate drafter, own KV cache) collapses 46.9 → 30.1 tok/s.
8. **FastMTP (2509.18362)**: fine-tune the single MTP head self-distilled, shared across
   steps, decay-weighted loss → acceptance 2nd-token 11%→56%, 3rd 2%→36%; 2.03× vs 1.21×
   for vanilla MTP reuse. Needs training (out of our scope) but proves the draft head is
   the lever, not the verifier.
9. **KV quant**: KIVI asymmetry (K per-channel, V per-token) is the baseline; 2025-2026
   SOTA = Kitty (dynamic channel boost), OSCAR (~2.28 bits, first INT2 not collapsing at
   128K), XQuant, RotateKV. For us KV is small (hybrid DeltaNet: KV only on full-attn
   layers) → low priority; KIVI asymmetry is the cheap takeaway.
10. Bottom line for INSIG4: adopt **scale sweep (WMSE) + Hadamard incoherence (block-diag
    for 5120) + GPTQ-style compensation** — all quantizer-side, format intact — and copy
    llama.cpp's **draft-confidence gating** into our MTP loop. FP8-MMA and DFlash-class
    drafter models are explicitly *not* worth it here.

---

## 1. ~4-bit weight quantization, 2025-2026: what beats EXL3/GPTQ/AWQ/QTIP

### 1.1 EXL3 status (reference point, already known)

EXL3 (turboderp, https://github.com/turboderp-org/exllamav3,
[doc/exl3.md](https://github.com/turboderp-org/exllamav3/blob/master/doc/exl3.md)) is a
QTIP variant: procedural codebook + tail-biting trellis, "deviates from QTIP in how
tensors are regularized and packed"; coherent at 2.5-3 bpw, q8 considered lossless.
Our wave-1 note (`audits/research.md`): EXL3's edge over QTIP is second-order scale
fitting. Still the quality king at 2-4 bpw for GPU-local inference; but trellis decode
adds per-value work that our E2M1-nibble streaming decode doesn't have — not the format
to copy for bandwidth, only the *scale-fitting discipline*.

### 1.2 ScaleSweep — bounded scale search for FP4 block formats (ACTIONABLE, #1 pick)

- What: NVFP4 (E2M1 + e4m3 scale per 16 + global fp32) PTQ. AbsMax/4-6 scale init "leaves
  a noticeable gap to optimal". Per block: fix base s_base = max|x|/6 floored to fp8, then
  **sweep fp8 bit patterns in a provably-bounded neighborhood** ([−3,+7] patterns for MSE,
  [−8,+7] for WMSE; upper bound (12/7)·s_base, MSE lower bound (4/5)·s_base for block 16)
  and keep the loss-minimizing scale. WMSE weights = squared input-channel norms
  (diag of X^T X); Q/K/V get softmax-structure-aware weights. Works with RTN and GPTQ
  (scales fixed pre-GPTQ).
- Numbers: Llama-3.1-8B-Ins GPTQ WAKVQ: 93.92% bf16 recovery vs 90.46 (AbsMax)/92.16
  (4/6); GSM8K 77.33 vs 73.16. Qwen3-8B WA RTN: 99.50%. Gap to the FP32-scale oracle
  stays <10%.
- Why here: INSIG4's fp16 scale per 64-group is the same two-level structure; the sweep is
  a pure `tools/quantize_insig4.py` change (per-super-group loop over a handful of fp16
  neighbor values, WMSE from one calibration pass of diag(X^T X)). No file-format, no
  kernel, no engine change.
- Cost: quantizer-only. Calibration needs ~128×2048 tokens (we already have numpy
  reference tooling); WMSE variant 1.37× slower than the quant op (irrelevant offline).
- Source: https://arxiv.org/html/2606.07618

### 1.3 LATMiX — learnable affine transforms that keep MX formats stock (ACTIONABLE)

- What: arXiv 2602.17681 (Feb 2026). Learns invertible affine T(x)=Ax+v (LU or QR
  parameterization) applied at residual stream + attention, plus a fixed online
  block-Hadamard; **the MX format itself (MXFP4/MXINT4, block size 32) stays untouched** —
  "focus on outlier mitigation without modifying the PTQ method or data format". Weights
  quantized with blockwise GPTQ afterwards; KL-distilled 256 calib samples, ~2 h/Qwen3-14B
  on 4×A100.
- Numbers: beats RTN/GPTQ/QuaRot/SpinQuant/OSTQuant/FlatQuant/MR-GPTQ on 7 zero-shot tasks
  + wiki2; Llama-3.2-1B MXFP4 93.88% recovery (vs 93.07 MR-GPTQ); Qwen3-1.7B 95.62%.
  Transforms fold into adjacent weights/embeddings → "negligible inference overhead"
  (verified on RTX 6000 Ada + vLLM).
- Why here: proof that MXFP4 4.25-bpw-class formats gain real accuracy from
  rotation/outlier-flattening *without* changing the bit layout — exactly our constraint.
- Cost: training step is the price (needs GPU distillation); the *fixed block-Hadamard*
  half is free (see 1.5). Adopt the Hadamard now, consider the learned affine only if
  parity targets demand it.
- Source: https://arxiv.org/html/2602.17681v2

### 1.4 KronQ — Kronecker/gradient Hessian + bidirectional incoherence (2026)

- What: arXiv 2607.07964. K-FAC: H ≈ H_X ⊗ H_G (input AND gradient covariance). Key result:
  H_G *cancels* in the GPTQ update loop (so the rounding loop stays GPTAQ-identical); it is
  used for (a) **bidirectional incoherence**: diagonal rescale S_G =
  diag(H_G,ii/‖W_i‖²)^{1/4} + randomized Hadamard on both sides; (b) analytic
  mixed-precision scoring tr(H_G)·tr(H_X) that separates Q/K/V (which share H_X).
- Numbers (wiki2, per-channel): Llama-2-7B W4 5.56 / W3 5.83 / W2 8.19 (GPTQ: 5.82/6.74/
  31.11; QuIP#: 5.66/6.19/12.30). Llama-3-8B W4 6.42 vs GPTQ 8.61. g=128 W2: 7.60 vs 274.
- Cost: +8-11 s/layer calib; inference = QuIP#-style online rotation
  (Θ(d log d), FWHT-class, no stored matrix) — kernel touch but small.
- Why here: the output-side (gradient) incoherence is the piece nobody in our notes has;
  it targets exactly the "later full-attention layers parity" class of sensitivity
  (column outliers). We cannot run backward passes easily in numpy — but the *idea*
  (row/column variance flattening before quantizing) is testable with diag stats only.
- Source: https://arxiv.org/html/2607.07964v2

### 1.5 Non-pow2 Hadamard (the 5120 = 1024×5 question) — ANSWER

- QuIP# (https://arxiv.org/html/2402.04396v1, § on RHT): "for dimensions n that are not
  powers of 2 ... factorize n = p·q where p is the largest power of 2" and use the
  **block-diagonal transform diag(H_p, ..., H_p)** (q blocks) with random signs. No
  padding (padding wastes memory/compute and dilutes incoherence at the seam); no radix-5
  Hadamard needed (complex-entry generalized Hadamards exist but nobody uses them in
  engines). So: hidden 4096 → full H_4096; hidden 5120 → diag of five H_1024. Block size
  1024 is plenty for incoherence at our widths.
- Engines handle it exactly this way: QuIP#/QTIP ship block-diagonal Hadamard + random
  signs; FlatQuant (ICML 2025, https://arxiv.org/abs/2410.09426) parameterizes the
  transform as a **Kronecker product** (works for any n, params cheap, learned to flatten
  sharp regions; W4A4 <1% acc drop, +7.5% over SpinQuant on Llama-3-70B; runtime cost
  claimed ~<5% via low-rank folding); SingleQuant/"OSbR" (https://arxiv.org/abs/2511.22316)
  computes rotations in **closed form via Givens transforms** — no learned optimization,
  and diagnoses STE-rotation-learning instability (SpinQuant-style) as a failure mode.
- For Insignia: quantizer computes W' = W · diag-sign · BlockDiagH offline; the GPU/CPU
  GEMV applies the same FWHT to the activation vector in registers before the dot
  (butterfly, O(d log d), no weight matrix read); the fp16 scale grid then sees
  near-Gaussian rows. Kernel change = one fused FWHT on the x-vector per linear (can be
  folded into the RMSNorm→GEMV hand-off or done on the fly per K-slice since H is an
  involution up to 1/sqrt scaling).

### 1.6 GPTQ lineage 2025-2026 (error compensation on a fixed grid)

- GPTAQ ("GPTQv2", ICML 2025, https://arxiv.org/pdf/2504.02692): asymmetric calibration
  objective + act-order (Hessian-diag-sorted columns) = SOTA PTQ/pruning baseline set.
- GPTQModel (https://github.com/modelcloud/gptqmodel) is the maintained stack (AutoGPTQ
  archived Apr 2025, AutoAWQ deprecated → llm-compressor) and adds first-order error
  compensation.
- For INSIG4: GPTQ's Cholesky-based compensation works on ANY reconstruction grid —
  including E2M1×fp16-scale — because the grid only changes the quantize() callback.
  Combined recipe per ScaleSweep: fix scales by sweep first, then run compensation
  (they measured that ordering; static Hessian-diag reordering).
- Cost: quantizer-only. The numpy reference (`tools/reference_*.py`) can host the Hessian
  accumulation.

### 1.7 KV-cache quantization (low priority for this hybrid model)

- Baseline: KIVI (https://arxiv.org/abs/2402.02750): **K per-channel / V per-token**
  asymmetry; 2-bit ≈ <1% avg drop, 2.6× KV memory cut.
- 2025-2026: Kitty (MLSys'26, https://arxiv.org/pdf/2511.18643) — dynamic channel-wise
  precision boost, ~8× KV cut, fixes 2-bit long-context degradation; OSCAR (Together AI,
  https://arxiv.org/html/2605.17757v1) — offline spectral covariance rotation, ~2.28
  eff. bits, first INT2 stable at 128K, ~8× memory, up to 7× throughput at batch
  (open-sourced 2026-05, https://oscar-quantize.github.io/); XQuant (EMNLP 2025,
  https://arxiv.org/abs/2510.11236) cross-layer; RotateKV (IJCAI 2025,
  https://www.ijcai.org/proceedings/2025/0690.pdf) ~0.1 ppl better than KVQuant at 2-bit;
  NVIDIA NVFP4 KV cache for long context
  (https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/).
- Why here / cost: our KV lives only on full-attention layers (DeltaNet layers carry
  recurrent state instead) and is small at 8K; if 128K ever matters, adopt KIVI asymmetry
  (engine change, quantize K along channels at cache-write) + a Hadamard on the head dim
  before cache-write (OSCAR-lite). Not now.

### 1.8 Adjacent (noted, not recommended)

- HyperQuant (https://arxiv.org/html/2606.23406): data-free per-tile RHT + lattice +
  Rice entropy coding; great rate (3.9× weight cut) but decode 51.8 → 7.8 tok/s on H100 —
  variable-length de-entropy per forward kills bandwidth-bound decode. Anti-pattern for
  us; confirms fixed-rate streaming formats are right at T=1.
- LCQ low-rank codebook (https://arxiv.org/abs/2405.20973), BTC-LLM sub-1-bit
  (https://arxiv.org/abs/2506.12040, 1.6× over FP16 at 0.8 bit, no sparse masks),
  LittleBit (NeurIPS 2025, https://neurips.cc/virtual/2025/poster/115061, 0.1 bpw latent
  factorization): the sub-1-bit lane beyond NanoQuant; all need codebook lookups or extra
  matvecs → only relevant when we chase <3 bpw tiers.

## 2. FP8 blockwise (128×128) on Ada/sm89 in 2026

(Extends `audits/w3/web-fp8-ada.md`, which fixed the roofline: 4070S = 71/142 TF
bf16/e4m3 f32acc, weights-bound until T*≈70/141.)

- **CUTLASS**: changelog 4.5.2 (https://docs.nvidia.com/cutlass/4.5.2/CHANGELOG.html)
  still: "Support for Ada (SM89) FP8 tensor cores via the 2.x API. Requires CUDA 12.4+";
  2D-scaling-tensor blockwise FP8 GEMMs are Hopper/Blackwell (sm120 grouped GEMM w/
  blockwise FP8/NVFP4/MXFP8 exists — CUTLASS issue #3263). Ada row-wise scaling only via
  EVT epilogue hacks (issue #1937). No 3.x/4.x collective-builder sm89 FP8 arrived through
  2026.
- **Machete** (https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel):
  Neural Magic's mixed-input GEMM, built on Hopper WGMMA + CUTE — **not portable to sm_89**
  (no wgmma). On Ada the ceiling remains Marlin-class SIMT+mma.sync kernels
  (https://github.com/IST-DASLab/marlin; ideal speedup to batch 16-32, ~85-90% of peak BW).
- **w8a8 vs w8a16 for memory-bound GEMV — measured**: SqueezeBits
  (https://blog.squeezebits.com/vllm-vs-tensorrtllm-7-weightactivation-quantization-34461,
  H100, Llama-3.1-8B): batch-1 decode/prefill gains — weight-only W4A16: 80-100%;
  W8A8: ~40%. "W8A8 demands more memory access than W4A16" — at equal bytes-per-weight
  the two W8 paths tie at T=1 (both <12% of compute), and W8A8 additionally needs the
  activation group-quant (one extra kernel or a fused RMSNorm tail) plus per-tensor scale
  plumbing on Ada (cuBLASLt blockwise = Hopper-only). vLLM serves DeepSeek-style 128×128
  checkpoints on sm89 as Marlin W8A16 today (PRs #5975, #16850; w3 audit). GIGAGPU 4090
  Llama-3.1-8B (https://gigagpu.com/rtx-4090-24gb-llama-3-8b-benchmark/): FP8 ≈ 2× FP16
  single-stream at ~95 t/s fp16 baseline — i.e. both W8 paths ride bandwidth.
  **Conclusion unchanged from w3: implement W8A16-style dequant streaming for T≤32;
  e4m3-MMA (w8a8) only buys the T≈64-141 prefill window. No public Ada blockwise-w8a8
  GEMV GB/s measurement exists as of 2026-08 — if we build one, benchmark it ourselves
  and publish.**
- Community confirmation that kernel choice, not w8a8-vs-w8a16, dominates: "W8A8
  performed worse than Float W8A16 in some cases" (Medium amineka9), Marlin 712 t/s vs
  461 fp16 vs 276 (non-Marlin GPTQ) on identical weights.

## 3. CPU AVX2 fp8/e4m3 dequant GEMV — published kernels & the number to beat

- **llama.cpp upstream (verified in our local clone + master)**: no FP8 weight type
  (master `ggml.h`: type 39 MXFP4, 40 NVFP4, 41/42 Q1_0/Q2_0). FP8 checkpoints are
  converted (bf16/int) at load. CPU MXFP4 dot (local:
  `llama.cpp/ggml/src/ggml-cpu/quants.c:298` generic, `arch/x86/quants.c:918` AVX2;
  upstream mirrors) = **pshufb LUT `kvalues_fp4` to int8 domain + vpmaddubsw/vmadd
  integer dot + per-32-block scalar scale FMADD** (fp16 y-scale × ue4m0/ue4m3 x-scale via
  a 1 KB f32 LUT `ggml_table_f32_ue4m3`). I.e., the mainstream CPU answer to 4-bit float
  formats is *integer-domain dot with table dequant*, not float decode.
- **F8_E4M3_B128 fork kernel**: DeepSeek-V4 Flash support added a native e4m3-weights
  ggml type ("type 42", e4m3 + e8m0 scale per 128×128 block) in the
  nsparks/blobfish-class forks (https://huggingface.co/nsparks/DeepSeek-V4-Flash-FP4-FP8-GGUF:
  "requires a build with native F8_E4M3_B128 ... stock upstream cannot load it"; tracking:
  https://github.com/ggml-org/llama.cpp/issues/22319, implementation PR
  https://github.com/ggml-org/llama.cpp/pull/24162; port blog:
  https://blog.teamblobfish.com/posts/deepseek-v4-flash-llama-cpp/). Notable issue
  comment: the FP8 CPU dot kernel "was written for AVX2 only and never extended" and is
  the hot kernel for V3-MoE too. No GB/s published.
- **ik_llama.cpp** (https://github.com/ikawrakow/ik_llama.cpp, docs
  https://ikawrakow-ik_llama-cpp.mintlify.app/inference/hybrid-cpu-gpu): "fastest
  quantized matrix multiplications on the planet" claim, AVX2/AVX512/Zen5-tuned GEMV;
  no native FP8 — its low-bit int quants instead; community evidence of closer-to-peak
  saturation than mainline.
- **Bandwidth anchors (Zen3-class, DDR4-3600 dual channel = 57.6 GB/s theoretical)**:
  llama.cpp generation ≈ memory bandwidth rule (tok/s ≈ GB/s at 8 bpw;
  https://mlechner.substack.com "Memory Bandwidth Ladder"); practical efficiency
  80-85% (≈ 46-49 GB/s) is the achievable ceiling for streaming decode
  (https://www.reddit.com/r/LocalLLaMA/comments/18ybtnn/ — 5600G data point + "4 threads
  saturate" bandwidth-bound confirmation; 5600H tg ≈ 10 t/s 7-8B Q4 both CPU and iGPU,
  shared-BW bound). Justine Tunney's llamafile matmul (https://justine.lol/matmul/)
  shows 30-500% CPU matmul gains available from better microkernels (prefill-oriented).
- **Our 35.9 GB/s pooled = 62% of theoretical → ~1.3-1.45× headroom**. Techniques to
  close it, ranked: (a) integer-domain dot for the MXFP4/INSIG4 CPU tier (pshufb LUT,
  matches llama.cpp's choice — note AGENTS.md says int8/DP4A lost on *GPU*; CPU Zen3
  vpmaddubsw is a different machine, worth its own measurement); (b) e4m3 decode via the
  f16-bit-trick + F16C vcvtph2ps chain from `audits/research.md` (12 ops/8 weights, no
  gathers); (c) NUMA-free 6-thread placement, one GEMV per core over contiguous row
  slabs, reverse-loop-to-zero per project rules.

## 4. Heterogeneous CPU+GPU+NVMe single-node inference, 2025-2026

(w3/web-hetero-sota.md covered FlexGen/AirLLM/DeepSpeed/Colibri + Windows IO
(IOCP/NO_BUFFERING/IoRing). New items:)

- **llama.cpp tensor-granular offload became mainstream** (the "GPU-poor" playbook):
  `--override-tensor` (PR #11397) then `--cpu-moe`/`--n-cpu-moe N` (2025,
  https://www.reddit.com/r/LocalLLaMA/comments/1mi7bem/) — keep attention/shared on GPU,
  push N expert FFNs to CPU until the model fits. Guides:
  https://gist.github.com/Docshotgun/a02a4c0c0a57e43ff4f038b46ca66ae0. For dense
  models the analogue is layer-granular `-ngl` plus per-tensor regex — i.e., exactly our
  budgeted-residency layer treadmill; nothing new to copy beyond validation of the
  approach at community scale.
- **ik_llama.cpp hybrid CPU/GPU** specializes dense+MoE mixed execution with dual cache
  prefetch (weights prefetched ahead of CPU compute while GPU busy) — the closest open
  analog to our treadmill; docs above.
- **InfiniGen (OSDI '24, https://www.usenix.org/conference/osdi24/presentation/lee,
  arXiv 2406.19707)**: KV-offload systems get 3× by *predictive* prefetch — use the
  partially-computed current-layer attention (top-K token selection) to prefetch only
  essential KV entries one layer ahead. Lesson for us: for any future KV tier, prefetch
  driven by attention scores beats blind streaming; for weight tiers the analogue is
  layer-order certainty (we have it — treadmill by construction).
- **PowerInfer-2 (arXiv 2406.06282, SJTU-IPADS)**: smartphone CPU+GPU+NNAPI pipeline,
  polymorphic neuron engine (different strategies prefill vs decode), **burst I/O
  prefetch from flash**, neuron placement in flash. 47B on a phone, 11.68 t/s, up to 29×.
  Transferable ideas: burst-prefetch scheduling (issue big sequential reads ahead, not
  on-demand 4 KB faults) and stage-dependent compute placement — both already in our
  design; confirms the pattern at another extreme.
- **KVDrive (ACM 2026, https://dl.acm.org/doi/10.1145/3802077)**: holistic multi-tier
  (VRAM/RAM/NVMe) KV management; latest citation anchor for SSD-tier KV.
- **Windows/WDDM pinned cap (ACTIONABLE planning number)**: cudaHostAlloc-style pinned
  memory caps at **~50% of physical RAM on Win10/11 under WDDM**
  (https://forums.developer.nvidia.com/t/change-limit-of-50-for-cudahostalloc-pinned-memory-on-windows-10-11/228235;
  older 512 MB-era thread
  https://forums.developer.nvidia.com/t/unable-get-over-512mb-of-page-locked-memory-with-cudahostregister-or-cudamallochost/27064;
  WDDM 2.x model: https://learn.microsoft.com/en-us/windows-hardware/drivers/display/gpu-virtual-memory-in-wddm-2-0).
  With 15.9 GiB RAM: **plan the pinned RAM tier ≤ ~7.9 GiB**, keep the rest pageable
  (our residency layer's pinning must degrade gracefully). WDDM mode itself also costs
  real RAM↔GPU throughput vs TCC (community:
  https://www.reddit.com/r/LocalLLaMA/comments/1ommahm/) — pinning + staged async
  copies are the mitigation; there is no TCC on GeForce.
- **DeepSeek-V4-Flash in llama.cpp (PR #24162, Apr-May 2026)** — the 2026 treadmill
  reference points: Epyc 9374F + RTX PRO 6000: ~748 t/s PP @8k → 282 t/s @524k, TG 10-11
  → 20+ t/s after graph-reuse; DGX Spark IQ2_XXS: pp2048 148 t/s, tg32 6 t/s. Also
  documents the failure modes we must avoid: illegal memory access with expert
  offloading at ubatch ≥32, OOM at 57k, multi-pass prompt token duplication. MTP was
  still TODO there — our engine is ahead on that axis for Qwen3.5.

## 5. Speculative decoding with MTP — acceptance-rate levers 2025-2026

- **llama.cpp MTP landed for our exact family** (PR #22673, merged 2026-05-16): loads
  NextN/MTP draft heads from the same GGUF, tested on Qwen3.5/3.6 (dense 27B, MoE
  35B-A3B). Community numbers: acceptance 70-90% at temp 0, ~2× tok/s typical; regression
  caution — issue #23230 (acceptance 0.82 → 0.33 after later merges; watch draft-cache
  invalidation bugs). Discussion #23735 tracks porting to other architectures.
- **Draft-confidence gating = the single biggest engine-side win** (`--spec-draft-p-min`,
  PR #22397, 2026-04-28): skip drafting when the head's top prob < threshold (0.75).
  Measured (Qwen3.6-27B Q4, RTX 3090 Ti, https://www.neoteric.no/blog/llamacpp-spec-draft-p-min-mtp-qwen3
  and https://ianlpaterson.com/blog/3090-ti-qwen-speedup-dflash-mtp/): MTP 44.6/39.1/
  44.4/48.9 t/s at 100/500/1000/2000 output tokens vs 29.0 autoregressive (+68%), holding
  FLAT with length. Mechanism: unfiltered MTP "speculatively generates and then rejects
  tokens at a rate that collapses throughput on longer contexts". For our engine: one
  comparison against the draft head's softmax max — trivial, engine-only, zero risk to
  correctness (gating only decides whether to draft at all).
- **Window sizing at T=2-5 (verify is memory-bound ⇒ verification is nearly free)**:
  LMSYS/SGLang MTP blog (https://www.lmsys.org/blog/2025-07-17-mtp/, DeepSeek V3):
  topk=1 everywhere; 3-token window → acceptance length 2.18, 81.5 t/s/rank; 4-token →
  2.44, 82.0 t/s/rank (+60% over no-MTP). Large-batch advice: draft 2-3 only; "raise
  draft size only if acceptance length stays consistently above 2". AMD serving:
  1.25-2.11× (https://rocm.blogs.amd.com/software-tools-optimization/mtp/README.html).
  Tree verification (EAGLE-2 dynamic trees 3.05-4.26×, https://www.alphaxiv.org/overview/2406.16858;
  EAGLE-3 https://arxiv.org/html/2503.01840v1) pays off on batch servers, but at our
  T≤5 and single user the SGLang data says **chains with topk=1 beat trees** — trees add
  KV rows and verify-width without more weight-stream reuse. Our MTP break-even analysis
  (`audits/research.md`: net win if p>0.02, ×1.6 at p=0.6) is conservative vs these
  measured 1.6-2×.
- **DFlash (2026, llama.cpp `--spec-type dflash`, BeeLlama fork)**: separate drafter
  model, up to 16 tokens/round, up to 6× on narrow tasks (linked-list coding 43→148 t/s)
  BUT it carries its own KV cache that grows with output — "DFlash decode collapse"
  46.9 → 30.1 t/s by 2000 tokens, +17.8% prefill tax, and hardware sensitivity (issue
  #25792: acceptance ~0.15 for official drafters on CPU+Vulkan). Verdict for a 12 GB
  card: a second model's weights + KV do not fit; MTP heads (shared trunk, tiny) are the
  right architecture — our choice of MTP-first is validated by 2026 deployment data.
- **FastMTP (arXiv 2509.18362, Tencent)**: fine-tune the single MTP head with shared
  weights across steps, self-distilled data, decay-weighted CE, +frequency-compressed
  draft vocab: acceptance 1st/2nd/3rd draft token 70→81%, **11→56%**, 2→36%; 2.03× vs
  1.21× for vanilla MTP-head reuse on MiMo-7B (A10, SGLang). Cost: 1 day on 1× H20 for
  a 211 M head. Out of scope for us (no training), but it flags that our 4 ms draft
  layer's acceptance is the entire lever — and that *draft-vocabulary restriction during
  drafting only* (lossless) is a cheap inference-side trick we could mimic by masking
  the draft softmax to top-N frequent tokens (engine-only, no training).
- Overlap: SGLang overlap scheduling alone gave +20.4% and is NOT yet composable with
  MTP in their stack — for us, overlapping the draft layer with NVMe prefetch of the
  next main-layer slab is the direct analogue (engine-only).

## Ranked top-5 adoption list for THIS repo

1. **INSIG4 scale sweep with WMSE (ScaleSweep-style)** — `tools/quantize_insig4.py`:
   per-64-super-group fp16 scale chosen by bounded bit-pattern neighborhood sweep
   minimizing Hessian-diag-weighted MSE (diag from one calibration pass), scales fixed
   before any later compensation. Evidence: +1-3.5 acc-recovery pts on NVFP4; quantizer
   -only, format/kernel/engine untouched. Do this before any kernel work.
2. **Block-diagonal randomized Hadamard incoherence** — offline W·diag(sign)·
   BlockDiagH (H_4096 full; five H_1024 for 5120), online FWHT on the activation vector
   fused at GEMV entry (registers, no stored matrix; QuIP#/LATMiX/KronQ lineage).
   Evidence: LATMiX/QuIP#-class gains at unmodified MX formats; kills the outlier-driven
   scale waste that motivates bigger scale dynamic range. Cost: quantizer + small kernel
   touch; parity-checkable layer-by-layer with the numpy reference.
3. **GPTQ/GPTAQ error compensation on the E2M1 grid** (after #1): Cholesky-updated
   rounding with act-order in numpy at quantize time; zero inference cost. Evidence:
   GPTAQ ICML 2025 numbers (W4 6.42 vs 8.61 wiki2 on Llama-3-8B).
4. **MTP draft-confidence gating + window autotuning** (engine-only): gate drafting on
   draft-head top-prob ≥ ~0.75 (llama.cpp PR #22397 analog: +68% tok/s at flat profile),
   keep topk=1 chain of 3-4 with acceptance-length-feedback window adaptation (LMSYS
   tuning); optional lossless draft-vocab top-N mask (FastMTP-lite). Guard with a
   regression test for draft-cache invalidation (llama.cpp #23230 precedent).
5. **CPU-tier integer-domain dot + pinned-RAM budget discipline**: measure pshufb-LUT
   int8 dot (llama.cpp MXFP4 AVX2 pattern, local clone refs above) against our fp32
   decode path on the 5600X — target ≥ 45 GB/s pooled vs current 35.9; cap the pinned
   host tier at ~7.9 GiB (WDDM 50% rule) in the residency planner, rest pageable.

Explicitly rejected for this rig: entropy-coded weight formats (HyperQuant: 51.8→7.8
t/s), DFlash-class separate drafters (VRAM + KV collapse), w8a8 e4m3-MMA at T=1 (no
bandwidth win; w3 roofline), CUTLASS/Machete/cuBLASLt blockwise paths (Hopper-only),
2-bit KV (KIVI asymmetry only if long-context becomes a goal).

## Sources

Quant (4-bit, rotations, KV):
- https://arxiv.org/html/2606.07618 (ScaleSweep)
- https://arxiv.org/html/2602.17681v2 (LATMiX)
- https://arxiv.org/html/2607.07964v2 (KronQ)
- https://arxiv.org/pdf/2504.02692 (GPTAQ) ; https://github.com/modelcloud/gptqmodel
- https://arxiv.org/html/2402.04396v1 (QuIP# non-pow2 block-diagonal Hadamard)
- https://arxiv.org/abs/2410.09426 ; https://github.com/ruikangliu/FlatQuant (FlatQuant)
- https://arxiv.org/abs/2511.22316 (SingleQuant / closed-form rotations)
- https://arxiv.org/html/2606.23406 (HyperQuant)
- https://arxiv.org/abs/2406.11235 (QTIP) ; https://github.com/turboderp-org/exllamav3/blob/master/doc/exl3.md (EXL3)
- https://arxiv.org/abs/2405.20973 (LCQ) ; https://arxiv.org/abs/2506.12040 (BTC-LLM) ; https://neurips.cc/virtual/2025/poster/115061 (LittleBit)
- KV: https://arxiv.org/abs/2402.02750 (KIVI) ; https://arxiv.org/pdf/2511.18643 (Kitty) ; https://arxiv.org/html/2605.17757v1 + https://oscar-quantize.github.io/ (OSCAR) ; https://arxiv.org/abs/2510.11236 (XQuant) ; https://www.ijcai.org/proceedings/2025/0690.pdf (RotateKV) ; https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/

FP8 / kernels / Ada:
- https://docs.nvidia.com/cutlass/4.5.2/CHANGELOG.html ; https://github.com/NVIDIA/cutlass/issues/1937 ; https://github.com/NVIDIA/cutlass/issues/3263
- https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel ; https://github.com/IST-DASLab/marlin ; https://arxiv.org/html/2408.11743v1 (MARLIN paper)
- https://blog.squeezebits.com/vllm-vs-tensorrtllm-7-weightactivation-quantization-34461
- https://gigagpu.com/rtx-4090-24gb-llama-3-8b-benchmark/ ; https://docs.vllm.ai/en/stable/features/quantization/llm_compressor/fp8/
- (local) `audits/w3/web-fp8-ada.md`, `audits/w2/trtllm-sm89-fp8.md`, `audits/w2/vllm-marlin-fp8.md`

CPU / FP8 GEMV:
- https://huggingface.co/nsparks/DeepSeek-V4-Flash-FP4-FP8-GGUF ; https://github.com/ggml-org/llama.cpp/issues/22319 ; https://github.com/ggml-org/llama.cpp/pull/24162 ; https://blog.teamblobfish.com/posts/deepseek-v4-flash-llama-cpp/
- (local clone, read this session) `llama.cpp/ggml/src/ggml-cpu/quants.c:298` (generic MXFP4/NVFP4 dot), `llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c:918` (AVX2 pshufb-LUT), `ggml-cpu.c:85` (ue4m3 f32 LUT); upstream master `ggml/include/ggml.h` type enum (no F8 type)
- https://github.com/ikawrakow/ik_llama.cpp ; https://ikawrakow-ik_llama-cpp.mintlify.app/inference/hybrid-cpu-gpu
- https://www.reddit.com/r/LocalLLaMA/comments/18ybtnn/ ; https://mlechner.substack.com (bandwidth ladder) ; https://justine.lol/matmul/

Heterogeneous / Windows:
- https://www.reddit.com/r/LocalLLaMA/comments/1mi7bem/ (--n-cpu-moe) ; llama.cpp PR #11397 (--override-tensor) ; https://gist.github.com/Docshotgun/a02a4c0c0a57e43ff4f038b46ca66ae0
- https://www.usenix.org/conference/osdi24/presentation/lee (InfiniGen) ; https://arxiv.org/abs/2406.06282 (PowerInfer-2) ; https://dl.acm.org/doi/10.1145/3802077 (KVDrive)
- https://forums.developer.nvidia.com/t/change-limit-of-50-for-cudahostalloc-pinned-memory-on-windows-10-11/228235 ; https://learn.microsoft.com/en-us/windows-hardware/drivers/display/gpu-virtual-memory-in-wddm-2-0 ; https://www.reddit.com/r/LocalLLaMA/comments/1ommahm/

Speculative / MTP:
- llama.cpp PR #22673 (MTP heads) ; https://github.com/ggml-org/llama.cpp/discussions/23735 ; https://github.com/ggml-org/llama.cpp/issues/23230
- llama.cpp PR #22397 (--spec-draft-p-min) ; https://www.neoteric.no/blog/llamacpp-spec-draft-p-min-mtp-qwen3 ; https://ianlpaterson.com/blog/3090-ti-qwen-speedup-dflash-mtp/
- DFlash: https://moazharu.medium.com/dflash-how-llama-cpp-got-a-6-lossless-speedup-by-firing-the-autoregressive-drafter-bacb654cce21 ; https://xhinker.medium.com/dflash-just-landed-in-llama-cpp-worth-to-upgrade-to-get-speed-boost-a20db434e8f7 ; https://huggingface.co/z-lab/Qwen3.6-27B-DFlash/discussions/1 ; https://github.com/ggml-org/llama.cpp/issues/25792
- https://www.lmsys.org/blog/2025-07-17-mtp/ ; https://rocm.blogs.amd.com/software-tools-optimization/mtp/README.html
- https://arxiv.org/html/2509.18362v1 (FastMTP) ; https://arxiv.org/html/2503.01840v1 (EAGLE-3) ; https://www.alphaxiv.org/overview/2406.16858 (EAGLE-2) ; https://docs.vllm.ai/en/latest/features/speculative_decoding/
