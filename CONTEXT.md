# Insignia Context

## Mission

Insignia is a deliberately specialized inference engine for RTX 40xx and newer NVIDIA GPUs. The first target is text-only Qwen3.5-9B with its one-layer MTP head and MXFP4 weights. Vision is intentionally deferred.

The optimization rule is simple: spend complexity, portability, and elegance to buy measured latency and throughput on the author's RTX 4070 Super. Unsafe tricks are not automatically good tricks: undefined behavior, self-modifying code, and register fantasies are only allowed when a benchmark and disassembly prove they help.

## Hardware contract

- GPU: NVIDIA GeForce RTX 4070 SUPER, Ada Lovelace, compute capability 8.9, 12282 MiB reported VRAM.
- CPU: AMD Ryzen 5 5600X, 6 cores / 12 threads.
- Host RAM: 15.9 GiB.
- CUDA: 13.3. Host compiler: MSVC x64 19.51.
- Primary build target: sm_89; no pretense of broad GPU portability.

Ada has no native block-scaled FP4 MMA. Native mxf4 block-scale MMA is a Blackwell path. Insignia therefore stores MXFP4 but dispatches an Ada path that expands or maps values to an Ada-supported operand type. The first candidate is INT8 DP4A/IMMA; FP8 MMA is a competing candidate and must be benchmarked rather than assumed superior.

## Qwen3.5-9B text contract

From the upstream config:

- hidden size 4096, intermediate size 12288, vocabulary 248320.
- 32 decoder layers, with full attention at layers 3, 7, 11, 15, 19, 23, 27, 31. The other 24 layers are gated DeltaNet linear attention.
- Full attention: 16 query heads, 4 KV heads, head dimension 256, Q/K RMSNorm, partial RoPE (64 rotary dimensions), interleaved MRoPE sections [11, 11, 10], theta 10000000, and a query-projection gate passed through sigmoid before output projection.
- DeltaNet: 16 key heads x 128 and 32 value heads x 128. Q/K are repeated from 16 to 32 value heads. QKV convolution width is 8192 with depthwise causal kernel 4. Projections are in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, and out_proj.
- DeltaNet recurrence uses FP32 state. Per step: decay state by exp(g), compute kv_mem = sum(state * k), delta = (v - kv_mem) * sigmoid(b), update state with k * delta, then output sum(state * q). Q and K use L2 normalization with epsilon 1e-6 and query scale 1/sqrt(128).
- DeltaNet gate parameter is g = -exp(A_log) * softplus(a + dt_bias) evaluated in FP32.
- DeltaNet output uses gated RMSNorm: FP32 variance, RMS normalization, learned weight, then SiLU(z).
- MLP is SwiGLU: down(silu(gate(x)) * up(x)).
- RMSNorm weights use Qwen zero-centered convention: normalized output is multiplied by (1 + weight).
- MTP assets exist in the checkpoint: mtp.fc, one attention layer, mtp.norm, mtp.pre_fc_norm_embedding, and mtp.pre_fc_norm_hidden. The serving reference concatenates normalized token embedding and normalized hidden state, applies mtp.fc, runs one decoder layer, normalizes, and produces logits through the shared LM head.

## Memory budget

Approximate text-only model size is 9.2B parameters. MXFP4 uses 17 bytes per 32 weights (16 packed E2M1 bytes plus one E8M0 scale byte), or 4.55 GiB for all parameters. A practical first layout keeps embeddings in BF16, the LM head in 8-bit, norms/state in FP32/BF16, and quantizes matmul weights: approximately 6.4 GiB before runtime workspaces.

Full-attention KV cache costs 32 KiB per token in BF16: 1 GiB at 32K context and 8 GiB at 262K. DeltaNet recurrent state is sequence-length independent: approximately 48 MiB FP32 for 24 layers, plus about 2.25 MiB of convolution state.

## Implementation phases

1. Build and execute a CUDA smoke test for sm_89.
2. Implement a correctness-first MXFP4 codec and CPU/GPU reference matvec.
3. Implement specialized decode GEMV, then prompt GEMM. Compare INT8 and FP8 expansion paths.
4. Implement DeltaNet recurrent decode and chunked prefill, validating against the Python reference.
5. Implement full-attention GQA with paged or contiguous KV cache and fused Q/K norm + RoPE where profitable.
6. Implement SwiGLU, RMSNorm, residual scheduling, and the one-layer MTP path.
7. Add a hierarchy manager for VRAM, pinned host RAM, and NVMe mappings only after profiling demonstrates a real capacity need.
8. Add NanoQuant-style sub-1-bit weights as an experimental format after the MXFP4 baseline has accuracy and performance tests.

## Performance law

No optimization is accepted from the joke list unless it survives correctness comparison, guard testing where possible, disassembly inspection, and an end-to-end benchmark. Deliberate specialization is welcome. Deliberate UB without evidence is just expensive superstition.
