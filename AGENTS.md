# AGENTS.md

## What this project is

Insignia is an alternative inference engine in the spirit of ggml/llama.cpp and
exllamav3, built by someone who wants the most extreme optimization available and
considers the process a form of pure masochism. Most bleeding-edge research is to
be used. The engine is **only compatible with RTX 40xx GPUs and above** (the
author owns an RTX 4070 SUPER and an RTX 4070 Ti SUPER box and does not want to
rent cloud GPUs).

Target capabilities, in the order the project fights for them:

- CPU + GPU mixed compute support.
- NVMe / RAM / VRAM hierarchical memory support (a model may be partially mapped
  from disk, paged through pinned host memory, and executed from VRAM).
- Aggressive but quality-gated quantization: NVFP4 weights on disk, FP8 (E4M3
  group-64) for dense-matrix compute and KV latents. Sub-1-bit (NanoQuant-style)
  was evaluated and rejected for now — see `progress.md` (sub-4-bit verdict).
- Primary model: **GLM-5.3-Flash abliterated NVFP4** (~180 GiB text-only compact
  store), running text-only with **DFlash2 block speculative decoding**.
- Secondary/legacy target still in tree: Qwen3.5-9B MXFP4 + MTP (`qwen35*`,
  `insig4` code and the Windows `build\*.bat` path). Vision towers are skipped.

The spirit of the project: be utterly ridiculous and silly, use deliberately
insane code, and sacrifice everything in the name of performance.

## The rules of this project

These are the author's stated rules. They are the project's constitution:

- Align to cache lines (e.g., add padding to structs so data lines up with
  hardware cache lines).
- Make every assumption you can (bake hardware- and context-specific assumptions
  into the code and tell the compiler).
- Use local copies that match the hardware's register typing (avoids extra
  casting/truncation instructions on smaller data types).
- TRUE doesn't have to be 1... (bind TRUE to an existing non-zero register like
  the GP pointer to save instruction cycles).
- Consider reserving a special register (globally reserve specific registers for
  the most frequently accessed variables).
- Abuse "compare to 0" hardware instructions (reverse loops to count down to 0,
  which CPUs can check faster than comparing against arbitrary numbers).
- Use goto statements for branch prediction / locality (rearrange code structure
  in memory to optimize instruction cache usage).
- Manipulate your pointers to spoof a struct if necessary (offset pointers to
  reuse existing functions on different data structures without writing new
  routines).
- Keep the engine as specialized as possible (avoid generic engine overhead and
  modify constant data directly to implement features without engine checks).
- Consider replacing if conditions with self-modifying code (memcpy instructions
  over functions at runtime when conditions rarely change or to handle hardware
  revision bugs).
- Bit manipulate floats if it makes sense (directly manipulate raw
  binary/mantissa representations with integer bitwise operations instead of
  slow float conversions/divisions).
- Use inline assembly when the compiler misbehaves (force exact machine
  instructions when compiler heuristics generate suboptimal assembly).

## Ignore these completely, in the sake of performance

- **Readability & maintainability.** The code becomes almost unreadable
  ("spaghetti code"), making it extremely difficult for other programmers (or
  even the original author months later) to understand and maintain.
- **Type safety & compiler guarantees.** Techniques like spoofing structs via
  pointer offsetting, raw bit manipulation of floats, and manual register
  overrides bypass compiler warnings, type checking, and memory safety.
- **Debuggability.** Self-modifying code, inline assembly, and unconventional
  register allocations make using standard debuggers and tracking down
  edge-case bugs a nightmare.
- **Modularity & clean architecture.** Reusable, generic engine architecture is
  discarded in favor of hyper-specialized, hardcoded logic.
- **Extensibility.** Adding new features or altering existing data structures
  risks silently breaking hardcoded offsets, alignment assumptions, and
  register bindings across the codebase.

## Hardware contract (verified)

Two machines, both Ada Lovelace sm_89, both CUDA 13.3, both running Arch Linux
under WSL2 for the GLM path:

**Dev box (local, where edits happen):**

- GPU: NVIDIA GeForce RTX 4070 SUPER, 12282 MiB VRAM, driver 610.47.
- CPU: AMD Ryzen 5 5600X, 6 cores / 12 threads. Host RAM: 15.9 GiB
  (`.wslconfig` caps the WSL guest at 14 GiB).
- Storage: two SSDs — C: (980 PRO, carries the WSL vhdx) and E: (DRAM-less,
  carries the repo + original checkpoints). Dual-SSD expert striping exists
  (`INSIGNIA_GLM53_ALT_SHARD_DIR` + `tools/stripe_copy.py`).
- Pinned host memory ceiling: 6.6–9.25 GiB (Windows' ~50%-of-RAM lockable law;
  cudaHostRegister and GDS are dead on WSL2). Reader-pool sweet spot: 4 threads
  (virtio-blk ceiling ~5.8 GB/s).

**glm-box (remote, where the big cache lives):**

- Reach: `ssh glm-box` (→ `desktop-hlvh09q` over Tailscale, user `PC`, key
  auth). Windows OpenSSH + Tailscale are auto-start services.
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16376 MiB VRAM, driver 610.47, mild OC
  (core ~2.95–2.97 GHz, mem 12.251 GHz ≈ 784 GB/s observed; validated stable
  under sustained load — no throttle/PCI/token faults).
- CPU: Intel i7-14700KF (Raptor Lake). AVX2 + FMA + 256-bit AVX-VNNI + F16C +
  GFNI. **No** AVX-512, AMX, AVX-VNNI-INT8, native BF16/FP16 arithmetic. See
  `audits/14700kf-isa.md`.
- RAM: 60 GiB usable in WSL (62 GiB limit). Pinned expert cache sweet spot
  measured at 32 GiB (`INSIGNIA_GLM53_EXPERT_CACHE_MB=32768`, the engine
  default; requests above ~32–40 GiB fail `cudaHostAlloc` and halve).
- Storage: single NVMe (~3.7–4.7 GB/s typical, 4 readers optimal). **No second
  SSD — striping code must stay off here.**
- WSL Arch carries the 180.2 GiB compact text store + FP8 caches in its ext4
  vhdx.

Ada has no native block-scaled FP4 MMA (that is Blackwell-only). Insignia stores
NVFP4/MXFP4 weights but executes them through an Ada-supported decode path;
direct FP32 accumulation from packed nibbles currently beats the Q8/DP4A
experiment by a wide margin on decode GEMV. FP8 (E4M3 group-64, tensor-core
GEMV) is the accepted compute format for dense matrices; E2M1-Q4 was measured
slower than FP8-TC and rejected.

## GLM-5.3-Flash model contract (from config.json)

- 45 layers: 34 gated-DeltaNet (KDA) linear-attention layers + 11 MLA
  full-attention layers. First 3 layers dense MLP; the other 42 are sparse MoE.
- MoE: 288 routed experts, top-8 per token (`noaux_tc` routing,
  `norm_topk_prob`, routed scaling 2.5), expert intermediate 2048, plus a
  shared expert. One decode token touches 42 × 8 = 336 expert records
  (~4.4 GiB NVFP4) if nothing is cached.
- MLA: hidden 4096, 64 heads × 256, q_lora_rank 1536, 512-wide compressed KV
  latent (kv_a/kv_b projections), partial RoPE. The engine's absorbed path
  extracts per-head `W_uk`/`W_uv` blocks from `kv_b_proj` at startup.
- KDA: 64 heads × 128, FP32 recurrent state, depth-4 causal conv, FP32 taps.
- Hyper-connections: 4 residual streams (mHC) with Sinkhorn-normalized 4×4
  mixer + RMS collapse.
- Vocabulary 154880; DSA indexer weights present (index_topk 2048) but not
  implemented — long-context sparse attention is future work.
- MTP layer 45 exists and its machinery works, but the draft layer predicts
  confidently wrong tokens on this ABLITERATED checkpoint (parked; see
  `progress.md` MTP outcome).
- DFlash2 speculative drafter: 5-layer block drafter reading target captures
  from layers 5/14/24/33/42; ~2.2 GiB BF16 checkpoint, 1.07 GiB FP8 cache,
  target embed/lm_head shared.

## Current engine state

- Whole-model 45-layer streaming execution and greedy decode run end-to-end
  from the compact ext4 store (`/var/lib/insignia/glm53-flash-text`, 120
  shards, 180.2 GiB, byte-verified; indexed by `tools/index_glm53.py
  --text-only`).
- Dense FP8 cache (`glm53-fp8-g64`, 8.13 GiB, 699 matrices, cos 0.9994) is the
  default dense path; produced by the PyTorch-free native E4M3 encoder
  (`tools/quantize_glm53_q8.py`, tested by `tools/test_e4m3fn.py`).
- Hierarchy in working form: O_DIRECT reader pool → pinned host-RAM expert LRU
  (default 32 GiB with safe halve-and-retry; ~80% hit rate at 2425 slots on
  glm-box) → 576 MiB VRAM expert cache covering all 42 sparse layers → GPU.
- Batched FP8 prefill GEMV (8-row tiles, ~4.6× vs scalar at 2048×8192) and an
  order-preserving FA2-style tiled MLA prefill (bit-exact vs scalar attention,
  ~5.2× at 16 tokens).
- MLA strategy (hybrid shadow bridge, current coherent path): **exact expanded
  K/V for the first 256 positions** (kLegacyMlaContext, 352 MiB) while the
  **512-wide group-scaled FP8 latent cache + absorbed attention** is populated
  for contexts up to 8192. Reproduces the exact oracle 12/12 greedy tokens;
  beyond-position-256 latent attention has NOT yet been validated at scale.
- DFlash2 speculative decoding: greedy-exact (committed sequence always equals
  plain greedy), k4 default, adaptive sequential/batch verify. Best sustained
  numbers: 187.7–194.4 ms/token (5.1–5.3 tok/s) on glm-box with the 32 GiB
  tier, vs ~447 ms/token scalar. **Open regression**: on the exact-prefix MLA
  bridge the drafter acceptance collapsed to 1.43 tokens/round (516.7
  ms/token) — drafter/verify alignment investigation in progress; see
  `audits/mla-latent-session.md`.
- Benchmark harness for real prompts staged: GSM8K + MATH-500 canonical data on
  glm-box, driver `tools/benchmark_math.py` (performance + parity only, no
  accuracy grading).
- The toy 84M oracle (`/var/lib/insignia/glm53-tiny`) and the independent
  NumPy references (`tools/reference_glm53_numpy.py`, `tools/dflash2_oracle.py`,
  `tools/mtp_oracle.py`) remain the parity ground truth.

### The determinism law (learned the hard way)

This engine's discrete MoE routing is so sensitivity-cascading that
mathematically equivalent rewrites change the output: expert accumulation
order, two-pass vs online softmax, and FP32 reassociation of attention are all
part of the *effective model*. A canonical-order MoE probe (same sets, sorted
summation) changed greedy tokens immediately and was rejected. Any
optimization that touches floating-point order must preserve operation order
bit-for-bit or be rejected by the parity gate: greedy IDs AND digit-identical
top-10 logits on the standard prompts, plus 30/40/100/240-token sequence
checks. Wait for coherent token parity before claiming the engine is correct.

## Working conventions

- Do not modify the reference clones (`llama.cpp`, `ggml`, `exllamav3`,
  `colibri`, `_mlx`, `vllm`) - they are read-only references and are gitignored.
- **GLM builds go through `build/glm53.sh` / `build/glm53-gen.sh` inside the
  Arch WSL distro** (`glm53-gen.sh` adds `-march=raptorlake -mtune=raptorlake`
  for glm-box host code). Binaries land in `/var/tmp/insignia-build{-raptor}/`.
  Windows `build\*.bat` scripts (vcvars64 + nvcc `-arch=sm_89`) are the
  Qwen-era path and still work for those targets.
- `.gitattributes` forces LF on `*.sh` — Windows checkouts CRLF-break bash
  otherwise.
- **Remote workflow**: edit locally → commit → `git push origin
  glm53-dflash2-4070ti-super` → `ssh glm-box` → `git -C
  C:\coding\Insignia-glm53-dflash2 pull --ff-only` → build/run inside
  `wsl -d Arch`. **Never touch `C:\coding\Insignia` on glm-box — it is a stale
  dirty snapshot kept deliberately.**
- Env knobs are the A/B surface (`INSIGNIA_GLM53_*`): expert tier
  (`EXPERT_CACHE_MB`), FP8 residency (`Q8_BUDGET_MB`), BF16 residency
  (`VRAM_BUDGET_MB`), readers (`READERS`), speculative mode (`DFLASH2`,
  `DFLASH2_FP8`, `DF_VERIFY_K`, `DF_SEQ_VERIFY`/`DF_BATCH_VERIFY`), MLA modes
  (`KV_FP8`, `MLA_LEGACY` exact oracle, `SCALAR_MLA_PREFILL`,
  `MLA_BF16_ABSORB`), striping (`ALT_SHARD_DIR`), prefetch (`CCT`,
  `PREFETCH`), and the trace/dump hooks (`ROUTE_TRACE`, `LAYER_DUMP`,
  `MLA_DUMP`, `DF_DUMP`, `LOGITS_DUMP`).
- `INSIGNIA_GLM53_DFLASH2_FP8` must point at
  `/var/lib/insignia/glm53-dflash2-fp8-fixed` — the default
  (`glm53-dflash2-fp8`) is the superseded cache with the FC layout bug.
- WSL `/tmp` is wiped on VM recycle: anything that must survive goes to
  `/var/lib/insignia`. From Git Bash use `wsl -d Arch -- bash -c '...'`
  (single-quoted payloads still eat `$var` — prefer script files under
  /mnt/e).
- Performance optimizations must be backed by a measurement and by the parity
  gate (or a unit test). Insignia's spirit is deliberate specialization, not
  expensive superstition: an optimization that survives benchmark + parity +
  disassembly is welcome; an unsafe trick with no evidence is not
  automatically accepted. Timings swing ~2× run-to-run on WSL — trust repeated
  medians, never single readings.
