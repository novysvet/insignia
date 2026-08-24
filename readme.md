# Insignia

An experimental LLM inference engine in the spirit of llama.cpp / exllamav3,
built for one machine and one GPU architecture: NVIDIA Ada Lovelace
(`sm_89`, RTX 40-series). No pretense of portability.

## Goals

- CPU + GPU mixed compute.
- Tiered memory: models map from NVMe, page through pinned host RAM, and
  execute from VRAM under a budgeted residency layer (pinning + LRU
  eviction).
- Extreme quantization, down to sub-1-bit via low-rank binary factorization
  (NanoQuant).
- First target: Qwen3.5-9B in MXFP4 with multi-token prediction (MTP),
  text-only.

## Status

Work in progress. The engine runs all 32 layers and greedy decode; layer 0
gated DeltaNet matches the independent NumPy reference to cosine
~0.9999998. Full-attention parity is still being chased. Correctness claims
wait on coherent token parity.

## Building

MSVC + CUDA via the scripts in `build/` (`-arch=sm_89`).

## License

Apache 2.0 — see `license.txt`.
