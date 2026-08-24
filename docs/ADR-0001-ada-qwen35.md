# ADR 0001: Ada-specialized Qwen3.5 execution plan

## Status
Accepted

## Decision
Build Insignia as a narrow native CUDA engine for sm_89, with an internal tensor/weight layout designed around Qwen3.5-9B rather than a general graph runtime. The first supported model is text-only Qwen3.5-9B with MTP. Store MXFP4 weights in the OCP-style E2M1 + E8M0 block format, but do not claim native FP4 execution on Ada. Benchmark INT8 and FP8 expansion paths and select per workload.

Use separate prefill and decode implementations. Prefill may use chunked DeltaNet and tiled attention; decode is dominated by recurrent DeltaNet state updates and GEMV. Keep DeltaNet state in FP32 until profiling proves a lower-precision state is accurate.

## Why
The model is hybrid: 24 gated DeltaNet layers and only 8 full-attention layers. A generic transformer engine would either misrepresent the model or add dispatch overhead at every layer. The 4070 Super has 12 GiB VRAM, so a mixed weight layout fits, but long-context KV cache can consume the remaining capacity quickly.

## Rejected
- Native MXFP4 MMA on Ada: unavailable in hardware.
- Vision-first implementation: increases scope without helping the initial text inference thesis.
- Generic portability layer: no second GPU target exists yet.
- NVMe paging as the first feature: it cannot repair a slow kernel and should follow measurement.
- Self-modifying code and global register reservations as default techniques: the toolchain and GPU execution model make these unreliable; only measured, isolated experiments can justify them.

## Consequences
The code will be intentionally specialized and unpleasant in places, but the external engine interface must remain small enough to test. Every layout assumption gets a static assertion and a validation test. Every fast path has a reference fallback until numerical parity is established.
