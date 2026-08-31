# Session 13: Q3-K-XL native kernel wave

Date: 2026-09-01  
Branch: `glm53-dflash2-4070ti-super`  
Measurement host: `glm-box-wsl`, RTX 4070 Ti SUPER, sm_89, CUDA 13.3  
Fixture: real block-3 expert tensors from shard 1 of
`AliceThirty/GLM-5.3-Flash-UNCENSORED-GGUF/UD-Q3_K_XL`

## Outcome

The artifact name is not its useful dispatch type. The live routed MoE is
primarily IQ3_XXS gate/up plus IQ4_XS down. Native kernels now exist for both
formats, including decode-style 1--8 row DP4A paths and a 32-token FP16/HMMA
prefill path. Literal Q3_K support remains for the parked MTP layer only.

The best measured improvements in this wave are:

- IQ3 scalar decode CTA: 10.221 -> 9.345 us raw (1.094x) and
  10.046 -> 8.414 us in the aligned layout (1.194x).
- IQ3 two-row streaming decode: 12.082 -> 11.581 us raw (1.043x) and
  12.791 -> 11.081 us aligned (1.154x).
- IQ3 32-token prefill: 121.681 -> 91.668 us versus the four-x8 Q8 pipeline
  (1.327x).
- IQ4 32-token prefill: 119.963 -> 67.711 us (1.772x).

All timings are seven-run medians of serialized CUDA-event measurements with
the Hugging Face download paused and weights resident in VRAM. The Q8 pipeline
comparison includes activation quantization. The HMMA kernel includes its own
FP32-to-FP16 conversion, so this is an end-to-end compute comparison rather
than a conversion-excluded headline.

## Format-specific design

IQ3_XXS is 98 bytes per 256 weights: one FP16 super-scale, 64 codebook-index
bytes, then 32 bytes containing parity-compressed signs and eight local scales.
The kernel does not expand the expert. Eight-lane cohorts decode signed byte
vectors from the 256-entry codebook and issue DP4A against Q8-per-32
activations. A byte-neutral optional row layout stores all FP16 scales, then
all index planes, then all sign/scale planes. It removes the pathological
98-byte AoS stride without increasing SSD, RAM, or VRAM bytes.

IQ4_XS is 136 bytes per 256 weights. Its nonlinear sixteen-value codebook is
implemented with `PRMT`/`__byte_perm` immediates, so no lookup table load is
needed. The accumulating down-projection epilogue is present for weighted MoE
combination.

Decode and prefill deliberately use different machines. Decode uses direct
packed-weight DP4A and never materializes a matrix. Prefill expands one 16x32
tile to FP16 shared memory and reuses it across two 16-token HMMA warps. IQ3
uses 128 decode lanes because codebook/sign expansion, not tensor-core math,
was the bottleneck. IQ4 needs only 64 lanes because one packed word yields
both 16-value halves.

## Quality gates

The independent CPU oracle fully dequantizes the real tensor and accumulates
FP64 products before the final FP32 cast.

| Path | MSE | Relative L2 | Cosine | Max abs |
|---|---:|---:|---:|---:|
| IQ3 Q8 x1 | 2.406383e-6 | 0.006200428 | 0.9999807853 | 0.005269289 |
| IQ3 Q8 x8 | 2.873962e-6 | 0.006505243 | 0.9999788418 | 0.007628888 |
| IQ4 Q8 x1 | 1.932986e-6 | 0.007441363 | 0.9999723127 | 0.004995264 |
| IQ4 Q8 x8 | 1.370503e-6 | 0.006626402 | 0.9999780452 | 0.005544052 |
| IQ3 HMMA prefill32 | 6.050547e-9 | 0.0002910598 | 0.9999999576 | 0.0003473461 |
| IQ4 HMMA prefill32 | 2.757042e-9 | 0.0002915997 | 0.9999999575 | 0.0002539158 |

The raw and byte-neutral IQ3 layouts produce identical output metrics. The
prefill path is about twenty times better in relative matrix error than Q8
decode because it converts reconstructed weights and FP32 activations to FP16
instead of quantizing activations to int8.

## Rejected arms

- A 1 KiB shared-memory copy of the IQ3 codebook regressed x1 by about 3.9%
  and x8 by about 11%; the read-only/global path won.
- A 16 KiB sign-folded codebook removed packed-byte compare/subtract work but
  increased lookup pressure. It regressed the repacked gate+up x8 median
  55.160 -> 55.793 us (1.15%) and raw 55.550 -> 56.446 us (1.61%). It was
  deleted.
- The first 64-thread IQ3 HMMA kernel was slower than four x8 calls
  (135.021 versus 121.767 us). K=32 retile alone reached 133.660 us. Doubling
  the expansion lanes to 128 was the decisive change, reaching 91.668 us.

## Compiler evidence

The promoted IQ3 HMMA kernel uses 48 registers, one barrier, 3,072 bytes of
shared memory, and no spills. The K=32 64-thread predecessor used 56 registers.
The scalar two-warp CTA and two-row streaming specializations also compile
without spills. `build/glm53-q3.sh` records `ptxas -v` output for every
specialization.

## Remaining integration work

The kernels and real-tensor harness are complete, but the production engine
still expects its NVFP4 compact-store record schema. The Q3-K-XL downloader is
still fetching shard 3; shards 1, 2, and 4 are final. Next steps are a typed
GGUF index/repacker, byte-neutral IQ3 sidecars for the dispatches that win,
Q6_K support for blocks 11/12/44, Q8_0 shared/dense handling, MoE gather/scatter
integration, and then full-sequence PPL/cosine/parity and throughput gates.

