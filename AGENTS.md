# AGENTS.md

## What this project is

Insignia is an alternative inference engine in the spirit of ggml/llama.cpp and
exllamav3, built by someone who wants the most extreme optimization available and
considers the process a form of pure masochism. Most bleeding-edge research is to
be used. The engine is **only compatible with RTX 40xx GPUs and above** (the
author owns an RTX 4070 SUPER and does not want to rent cloud GPUs).

Target capabilities, in the order the project fights for them:

- CPU + GPU mixed compute support.
- NVMe / RAM / VRAM hierarchical memory support (a model may be partially mapped
  from disk, paged through host memory, and executed from VRAM).
- Sub-1-bit quantization support (see NanoQuant:
  https://arxiv.org/html/2602.06694v1 - low-rank binary factorization reaching
  binary and sub-1-bit compression levels).
- First model: **Qwen3.5 9B with MTP (multi-token prediction) and MXFP4
  weights**, running text-only. Vision towers are intentionally skipped for now.

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

- GPU: NVIDIA GeForce RTX 4070 SUPER, Ada Lovelace, compute capability 8.9,
  12282 MiB reported VRAM, driver 610.47.
- CPU: AMD Ryzen 5 5600X, 6 cores / 12 threads. Host RAM: 15.9 GiB.
- Build: CUDA 13.3.73, MSVC 19.51 x64, Windows SDK 10.0.22621.0.
- Target architecture: sm_89. No pretense of broad GPU portability.

Ada has no native block-scaled FP4 MMA (that is Blackwell-only). Insignia stores
MXFP4 weights but executes them through an Ada-supported decode path; direct
FP32 accumulation from packed nibbles currently beats the Q8/DP4A experiment by
a wide margin on decode GEMV.

## Current engine state

- Zero-copy mmap reader for the cached 4.8 GB MLX-format MXFP4 + MTP checkpoint
  (indexed via `tools/index_safetensors.py`; no conversion step required).
- Budgeted GPU residency layer with pinning + LRU eviction, so the NVMe/RAM/VRAM
  hierarchy already exists in a working form.
- Native MLX-format MXFP4 matvec (~150 GiB/s on 4096x4096).
- Fused gated-DeltaNet recurrent decode kernel.
- Full-attention GQA decode, Q/K norm + partial RoPE, gated RMSNorm, SwiGLU,
  MLP, residual paths, per-layer state, KV caches.
- Whole-model 32-layer execution and greedy token decode run; MTP draft layer
  runs in ~4 ms warm.
- An independent NumPy reference exists for layer-by-layer and
  sub-operation-by-sub-operation parity checking (`tools/reference_*.py`).

Layer 0 DeltaNet matches the independent reference at cosine ~0.9999998.
Later full-attention layers still have an unresolved parity issue; the
instrumentation and reference scripts are the intended tools for finding it.
Wait for coherent token parity before claiming the engine is correct.

## Working conventions

- Do not modify the reference clones (`llama.cpp`, `ggml`, `exllamav3`,
  `colibri`, `_mlx`) - they are read-only references and are gitignored.
- Builds go through `build\*.bat` scripts that call vcvars64 and nvcc with
  -arch=sm_89; those scripts are the only known-good build path.
- Performance optimizations should be backed by a measurement and by a
  correctness check against the NumPy reference (or a unit test) wherever
  possible. Insignia's spirit is deliberate specialization, not expensive
  superstition: an optimization that survives benchmark + parity + disassembly
  is welcome; an unsafe trick with no evidence is not automatically accepted.
