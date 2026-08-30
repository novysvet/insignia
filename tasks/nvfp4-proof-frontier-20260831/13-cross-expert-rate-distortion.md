# Problem 13: Cross-expert rate-distortion coding with useful decode economics

Repository: https://github.com/novysvet/insignia  
Branch: `codex/glm53-dflash2-4070-super`  
Hardware required: none; use synthetic or user-supplied matrix samples only.

## Mission

Find—or rule out—shared structure among the 288 experts of one sparse layer
that permits fewer transferred bytes than independent NVFP4 records while
spending abundant GPU compute to reconstruct each requested expert. This is a
rate-distortion and online reconstruction problem, not merely a low-rank SVD
exercise.

## Fixed format and geometry

Each expert has gate/up `2048 x 4096` and down `4096 x 2048`, for 25,165,824
weights total. NVFP4 stores E2M1 bodies (two weights/byte), one E4M3 scale per
16 values, and three FP32 globals: about 13.5 MiB/expert expanded. Existing
weight samples appeared approximately Gaussian after NVFP4 dequantization
(kurtosis about 3.1), and generic Hadamard preprocessing did not help a prior
sub-4-bit pilot. Do not assume low rank, clusters, or a common basis without a
statistical test and an out-of-sample guarantee.

## Required result

1. Define a family such as shared dictionary plus sparse code, shared base plus
   quantized residual, tensor factorization, low-rank correction, or learned
   conditional generator. Derive total bits including dictionaries, indices,
   scales, alignment, and error-correction metadata.
2. Prove approximation/error bounds under explicit structural assumptions and
   give minimax lower bounds or counterexamples outside them.
3. Derive reconstruction work, scratch, and transferred bytes for a single
   randomly requested expert and for an R-row prefill batch. Shared metadata
   that must remain in VRAM counts against expert-cache capacity.
4. Design a streaming encoder that need not hold a layer in RAM and a decoder
   compatible with Ada sm_89. Reusing NVFP4's exact hot representation while
   coding cold experts differently is permitted.
5. Give a sample-complexity/generalization analysis across layers and prompts;
   choosing the best dictionary on the evaluation experts is not evidence.

## Quality and economic gates

Approximate proposals report weight and output MSE/relative-L2/cosine plus
forward/reverse KL, JS, same-prefix PPL, routes, Top-1, DFlash acceptance, and
hard outputs at engine time. The maximum PPL increase is +3.5% conditional on
useful hard outputs. A format wins only if expected saved NVMe/H2D time exceeds
decode/reconstruction cost while accounting for reduced VRAM capacity.

## CPU deliverable

Submit `cross_expert_codec.py` with streaming encode/decode, exact bit ledger,
Gaussian nulls, planted shared-structure families, adversarial experts, held-
out selection, and a symbolic break-even solver. Report the entire rate-
distortion-compute frontier and confidence intervals. Completion requires a
theorem or lower bound, byte-exact round trips of the coded representation,
and a crisp engine sampling plan. A strong null result is valuable.
