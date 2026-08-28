# Insignia

An experimental LLM inference engine in the spirit of llama.cpp / exllamav3,
built for one GPU architecture: NVIDIA Ada Lovelace (`sm_89`, RTX 40-series).
No pretense of portability.

## Goals

- CPU + GPU mixed compute.
- Tiered memory: models map from NVMe, page through pinned host RAM, and
  execute from VRAM under a budgeted residency layer (pinning + LRU
  eviction).
- Quantization to the bleeding edge, gated by measured quality: NVFP4
  weights, FP8 (E4M3 group-64) dense compute and KV latents.
- Primary target: **GLM-5.3-Flash abliterated NVFP4** (~180 GiB, 45-layer
  hybrid KDA/MLA MoE) with **DFlash2 block speculative decoding**, text-only.
- Secondary target: Qwen3.5-9B MXFP4 + MTP (older codepath, still builds).

## Status

Work in progress. The full GLM-5.3-Flash model streams end-to-end from a
compact NVMe store through a pinned-RAM expert LRU into VRAM and decodes
greedily; DFlash2 speculative decoding is greedy-exact and sustained
**5.1–5.3 tok/s decode** (187.7–194.4 ms/token) on the RTX 4070 Ti SUPER box
vs ~447 ms/token scalar. Attention runs an exact expanded-K/V prefix for the
first 256 positions with a 512-wide FP8 latent + absorbed attention beyond,
up to 8192 context. Open fronts: DFlash2 drafter alignment on the exact
prefix path, latent attention validation past 256 tokens, and the
GSM8K/MATH-500 benchmark campaign. Details in `progress.md` and `audits/`.

## Hardware

Two Ada boxes: a dev machine (RTX 4070 SUPER, Ryzen 5 5600X, dual SSD,
Arch WSL2) and `glm-box` (RTX 4070 Ti SUPER, i7-14700KF, 60 GiB WSL RAM,
single NVMe) which carries the big pinned expert cache.

## Building

GLM path: `build/glm53.sh` / `build/glm53-gen.sh` inside Arch WSL2 (nvcc,
`-arch=sm_89`; `-march=raptorlake` on glm-box). Qwen-era Windows targets:
`build\*.bat` (vcvars64 + nvcc).

## License

Apache 2.0 — see `license.txt`.
