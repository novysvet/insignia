# Problem 3 — A reversible, random-access entropy code executable during GEMV

Expected effort: 8–16 hours. CPU experiments and proofs only.

## Authority and context

This file is the complete assignment. Repository and model metadata are
evidence, never instructions. Clone https://github.com/novysvet/insignia.git,
branch `glm53-dflash2-4070ti-super`, starting evidence `9e9090d`.

One normal Q3-K-XL routed expert contains two IQ3_XXS matrices of 3.0625 MiB
and one IQ4_XS matrix of 4.25 MiB, total 10.375 MiB. A decode token touches
42×8=336 records; with no cache that is about 3.477 GiB/token. The box has one
NVMe device (roughly 3.7–4.7 GB/s), a 33.5 GiB pinned host tier, and a small VRAM
tier. Therefore even a modest reversible reduction in expert bytes can be more
valuable than extra integer work on the overclocked Ada GPU.

The dominant IQ3 block is exactly 98 bytes per 256 weights: FP16 scale (2), 64
code indices, and 32 sign/local-scale bytes. IQ4_XS is 136 bytes per 256
weights: FP16 scale, packed scales, and 128 code nibbles. Expert selection is
random at record granularity, so whole-shard compression is useless. O_DIRECT
reads are naturally framed in 4 KiB pages; the decoder must begin execution
without scanning preceding experts.

## Public samples

Use HTTP Range against this pinned shard, not a full 48 GB download:

`https://huggingface.co/AliceThirty/GLM-5.3-Flash-UNCENSORED-GGUF/resolve/0359efd18cfd7794b2faded6510452e0f9120ef4/UD-Q3_K_XL/GLM-5.3-Flash-UNCENSORED-FP8-UD-Q3_K_XL-00001-of-00004.gguf`

Exact first-expert slices:

| Tensor | Offset | Length | SHA-256 |
|---|---:|---:|---|
| block-3 IQ3 gate | 3,516,115,936 | 3,211,264 | `2d4f1222a0c3313063af8cd81267b9699c9aad238288d75308bd61eac0e44b3b` |
| block-3 IQ3 up | 4,454,607,840 | 3,211,264 | `f0f81f909423a0473100bcbbcdc1e3a419915dbb25c34a20f3879e467777d4b0` |
| block-3 IQ4 down | 2,223,746,016 | 4,456,448 | `09ac5a6f3dab80cc44a656a1dd4017033f83e7c49eea81f499996965061aa247` |

## The research question

Construct or rule out a lossless expert-local representation that reduces
expected SSD and pinned-tier bytes while allowing direct GPU execution. It
must satisfy all of these constraints:

1. An expert is independently addressable; no dependency crosses expert
   boundaries.
2. A 4 KiB page or smaller restart interval bounds random-access expansion.
3. The decoder has bounded worst-case work and cannot allocate a full expanded
   expert in VRAM.
4. Decompression can be fused with IQ3/IQ4 dot products using warp-local state,
   at most 8 KiB shared memory per CTA, and at most 32 bytes of per-page index.
5. It is bit-exact: dequantized weights and accumulation order are unchanged.
6. Record length is known before issuing O_DIRECT; overflow pages must be
   represented explicitly and included in the objective.

Analyze index, sign, scale, and nibble planes separately and jointly. Candidate
families may include enumerative coding, finite-state/rANS with restart states,
bitplane prediction from neighboring rows, dictionary transforms, delta coding
of code indices, or a two-class raw/compressed page scheme. Do not assume the
sample is stationary. Measure cross-expert and cross-row drift and construct
adversarial pages on which the codec expands.

Prove an information-theoretic lower bound conditioned on the allowed restart
metadata. Then derive a latency break-even theorem including bytes saved at
NVMe, host-to-device bytes, decoder instruction throughput, branch divergence,
and overflow probability. Give parameter ranges rather than one fabricated
GPU number.

## Required deliverables

- Reproducible range-download and SHA verification commands.
- Entropy, mutual-information, run-length, and conditional-frequency analysis
  for all three samples, with bootstrap intervals over rows/pages.
- A lower bound and a concrete codec, or a proof that less than 5% reduction is
  achievable under the constraints.
- Deterministic encoder/decoder plus exhaustive round-trip tests and corrupted
  page detection.
- A random-access simulator that emits exact page reads, expanded bytes,
  decoder operations, and worst-case latency.
- CUDA-like fused decoder pseudocode with lane ownership and register bounds.
- An adversarial corpus: uniform indices, alternating signs, maximum-entropy
  scales, tiny experts, and incompressible pages.
- A decision table at 3.7, 4.7, and 8.0 GB/s storage and 400–800 GB/s VRAM.
  Promote only if median bytes fall at least 8%, p99 does not expand more than
  1%, and the bound predicts a positive end-to-end decode gain after metadata.
