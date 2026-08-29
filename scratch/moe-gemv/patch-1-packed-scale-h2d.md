# Patch sketch 1 — ship packed scale bytes over PCIe, expand on device

Rank 1 in ranked-opportunities.md. Expected: b 610 -> 577 us/record (−5.5% of
the per-record PCIe floor), i.e. −64 ms on a k=8 verify round (1930 records),
−11 ms per scalar token. Determinism-safe by construction (integer-only
expansion, byte-identical scale codes, GEMV kernels and arithmetic untouched).

## What exists today

- Packed sidecar (env `INSIGNIA_GLM53_PACKED_EXPERTS`): per projection the
  512 KiB of E4M3 scale codes are nibble-packed to 256 KiB via a 16-entry
  codebook + escape bytes (nibble 15 = "take the next escape byte").
  `expand_scale_nibbles` (src/glm53_generate.cu:1017-1051) is the AVX2
  decoder; `stage_packed` (1052-1094) runs it in the reader thread and the
  window ends up holding the EXPANDED 14.156 MB record, so the H2D
  (`upload()`, 841-844 / 865-866) still crosses PCIe at full size. The disk
  already benefits (−5.5%); PCIe does not.
- `Layout{body[3], scales[3], bytes}` (glm53_generate.cu:921-925) carries
  per-projection offsets into the slot; GEMV launchers just take pointers
  (`gate_weight()` etc. wrap `active_device_ + active_.body/scales`), so the
  in-slot layout is free to change.

## Design

1. **Stage raw** (new `stage_packed_raw` next to `stage_packed`): pread the
   bodies straight to window offsets `body[p] = p * (4 MiB)`; pread the three
   packed scale blocks to a contiguous zone `ps[p] = 12 MiB + p*256 KiB`;
   escapes + codebooks to the tail `[13.5 MiB, 13.5 MiB + Σ + 48 B)`
   (kPayloadCapacity = 14.156 MB leaves 0.66 MiB — the escape count is tiny by
   construction; keep the existing under/overflow `require` checks).
   `layout.bytes = 12 MiB + 768 KiB + Σ + 48 B` (~13.37 MB). No CPU expansion,
   no extra memmoves (positioned preads only).

2. **New kernel** `expand_scale_codes_kernel` in src/glm53_expert_bench.cu
   (namespace insignia::glm53), a byte-exact GPU port of
   `expand_scale_nibbles`. Integer work only — nibble UNPCK + codebook PRMT +
   escape substitution with an output-order escape cursor (block-wide prefix
   count of nibble==15 occurrences; escapes are consumed in output index
   order exactly as the CPU loop does):

```cuda
// One block of 512 threads per projection. src/esc are in the freshly
// copied slot; dst is the canonical scales[p] region of the same slot.
// scratch is a persistent 256 KiB device buffer used to snapshot src
// because dst regions overlap later projections' src (see ordering note).
__global__ __launch_bounds__(512) void expand_scale_codes_kernel(
    const uint8_t *__restrict__ src,      // 256 KiB packed nibbles
    const uint8_t *__restrict__ escapes,  // escape bytes, output order
    uint32_t escape_count,
    const uint8_t *__restrict__ codebook, // 16 entries
    uint8_t *__restrict__ scratch,
    uint8_t *__restrict__ dst) {          // 512 KiB expanded codes
    // phase 1: snapshot src -> scratch (256 B/thread, uint4)
    for (int i = threadIdx.x * 16; i < (256 << 10); i += 512 * 16)
        *reinterpret_cast<uint4 *>(scratch + i) =
            *reinterpret_cast<const uint4 *>(src + i);
    __syncthreads();
    // phase 2: each thread expands 32 outputs (16 packed bytes) and counts
    // its own escapes; block-exclusive prefix gives each thread its slice
    // of the escape stream (same consumption order as the CPU loop).
    __shared__ uint32_t prefix[513];
    const int chunk = threadIdx.x;            // 512 chunks x 32 outputs
    uint8_t out[32]; uint32_t mine = 0;
    // ... nibble unpack + codebook lookup, escape nibble (15) marked ...
    prefix[chunk] = mine; __syncthreads();
    // block-exclusive scan over prefix (single pass, 512 threads)
    uint32_t base = /* scan result */;
    for (int i = 0; i < 32; ++i)
        if (escaped[i]) out[i] = escapes[base + used++];
    // phase 3: write dst (uint4 stores)
}
```

   (Hand-waved scan/detail level is intentional — it is a sketch; the
   contract that matters is: `dst` is byte-identical to the AVX2 output, and
   escape consumption is in output order.)

3. **upload() branch** (glm53_generate.cu:817-855): when
   `state.layout.packed_scales`, H2D `state.layout.bytes` (13.37 MB) as
   today, then — on the DEFAULT stream, immediately after the existing
   `cudaStreamWaitEvent(nullptr, state.copy_done, 0)` — launch the three
   expansions (p = 2, 1, 0 order) writing the canonical
   `scales[p] = 12 MiB + p * 512 KiB` regions, using the shared 256 KiB
   scratch. Ordering/safety:
   - expands and GEMVs serialize on the default stream, so scratch reuse
     across records is safe while the next record's H2D proceeds on the copy
     stream;
   - expansion destinations overwrite packed-scale sources only of
     projections already consumed (p=2 first), hence the scratch snapshot;
   - VRAM-tier hits (`device_index_`) skip both H2D and expand — slots always
     hold the canonical expanded layout, so GEMV pointers never change.

4. **Costs added per record**: 3 kernel launches (+3 x 8.1 us CPU-side,
   hidden under the 577 us copy), ~0.79 MB snapshot + 1.57 MB expanded writes
   (~3 us D2D at 800 GB/s). Removed: the reader-thread AVX2 expand
   (already metered by `packed_expand_nanoseconds_`).

## Net arithmetic

| item | per record | per k=8 round (1930) | per scalar token (336) |
|---|---|---|---|
| H2D saved | 0.786 MB / 23.2 GB/s = 33.9 us | 65 ms | 11.4 ms |
| D2D added | ~3 us + 24 us launches (hidden) | ~0 | ~0 |

## Gate

Parity: greedy IDs + digit-identical top-10 logits + 30/40/100/240-token
sequences vs the expanded-transport build (both arms on the same store with
the sidecar present). The scale codes are bit-equal by construction, so any
failure indicates an escape-order port bug, not a law violation.

## Prerequisite

The packed sidecar must exist for the glm-box 180.2 GiB store
(`INSIGNIA_GLM53_PACKED_EXPERTS=/var/lib/insignia/...`); without it this
patch has nothing to ship. (Sidecar geometry is validated at open,
glm53_generate.cu:960-1005.)
