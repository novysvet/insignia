# Insignia Context

## Mission

Insignia is a deliberately specialized inference engine for RTX 40xx NVIDIA
GPUs (sm_89). The primary target is text-only **GLM-5.3-Flash abliterated
NVFP4** (~180 GiB) with DFlash2 block speculative decoding; the older
Qwen3.5-9B MXFP4 + MTP path remains in tree. Vision is deferred.

The optimization rule: spend complexity, portability, and elegance to buy
measured latency and throughput on the author's RTX 4070 SUPER (dev box) and
RTX 4070 Ti SUPER (`ssh glm-box`). Unsafe tricks are not automatically good
tricks: undefined behavior, self-modifying code, and register fantasies are
only allowed when a benchmark and disassembly prove they help.

## Hardware contract

- Dev box: RTX 4070 SUPER 12 GiB, Ryzen 5 5600X, 15.9 GiB RAM (14 GiB WSL),
  dual SSD (C: 980 PRO + E:), pinned ceiling 6.6–9.25 GiB, 4-reader virtio
  sweet spot.
- glm-box: RTX 4070 Ti SUPER 16 GiB, overclocked +150 MHz core /
  +2000 MHz memory (~800 GB/s observed, verified stable), i7-14700KF
  (AVX2/FMA/AVX-VNNI-256; no AVX-512/AMX), 60 GiB WSL RAM, single NVMe,
  32 GiB pinned expert cache sweet spot.
- Both: CUDA 13.3, Arch WSL2, sm_89. Ada has no block-scaled FP4 MMA:
  NVFP4 decodes through an Ada path (FP32-accum-from-nibbles beat DP4A);
  FP8 E4M3 group-64 tensor-core GEMV is the dense compute format.

## GLM-5.3-Flash model contract

From `config.json` (text_config): 45 layers — 34 gated-DeltaNet (KDA)
linear-attention + 11 MLA full-attention layers; first 3 dense, 42 sparse
MoE (288 routed experts, top-8, `noaux_tc` + `norm_topk_prob`, scaling 2.5,
expert intermediate 2048, plus shared expert). Hidden 4096, 64 heads × 256,
q_lora_rank 1536, 512-wide KV latent, partial RoPE, 4 hyper-connection
streams (mHC), vocab 154880. DSA indexer weights present (topk 2048),
unimplemented. MTP layer 45 parked (predicts wrong on the abliterated
checkpoint). DFlash2 drafter: 5 layers, target captures at 5/14/24/33/42,
KV-only feature injection, borrows target embed + lm_head.

## Memory budget

Text-only compact store 180.2 GiB NVFP4 (120 shards). Dense FP8 cache
8.13 GiB (699 matrices). Routed experts stream 4.4 GiB/token uncached;
the pinned host LRU (default 32 GiB on glm-box, halve-and-retry) reaches
~80% hits; 576 MiB VRAM expert cache covers all sparse layers. KV: exact
expanded K/V for the first 256 positions (352 MiB), 512-wide FP8 latent
beyond (~50 MiB per 8192 context across the 11 MLA layers). KDA recurrent
state is sequence-length independent.

## The determinism law

Discrete top-8 routing amplifies ~1e-6 floating-point perturbations into
different experts and different tokens. Expert accumulation order, two-pass
vs online softmax, and FP32 reassociation are part of the effective model —
mathematically equivalent rewrites are behavior changes. Every optimization
must pass the parity gate: greedy IDs plus digit-identical top-10 logits on
the standard prompts and long-sequence checks, against the exact oracle
(`INSIGNIA_GLM53_MLA_LEGACY=1`) where applicable.

## Current focus

1. DFlash2 drafter/verify alignment on the exact-prefix MLA bridge
   (acceptance regressed to 1.43/round; `audits/mla-latent-session.md`).
2. GSM8K/MATH-500 benchmark campaign on glm-box (`tools/benchmark_math.py`).
3. Latent MLA validation beyond position 256 (quality A/B tooling missing).
4. CCT cross-layer expert prefetch integration (the 6→8.5 GB/s I/O gap).
5. Qwen3.5/insig4 path is dormant, not dead — Windows `build\*.bat` targets.
