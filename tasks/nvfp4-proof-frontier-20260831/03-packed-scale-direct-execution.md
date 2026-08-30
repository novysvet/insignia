# Problem 3: execute NVFP4 directly from exact packed scales

Repository: https://github.com/novysvet/insignia  
Branch: `codex/glm53-dflash2-4070-super`  
Hardware required: none

## Engine facts

An expanded expert contains 12 MiB of nibble bodies and 1.5 MiB of E4M3 scale
bytes. Insignia's exact XPR1 sidecar compresses scale planes with a nibble
codebook plus an ordered escape stream. Across all 12,096 expert records it
stores 0.94532 times the expanded bytes; only 0.782% of scale entries escape.
The decoder is byte-exact.

Existing paths either expand scales on the CPU before H2D or copy packed data
and expand the complete scale plane on the GPU. A newer device-slot layout can
keep scales packed, saving 651,264 bytes (4.60%) per resident expert and fitting
13 more expert slots in the measured DFlash VRAM budget. Its current
expand-on-use scratch pass costs enough to erase the cache benefit. The desired
path consumes packed scale codes inside the NVFP4 GEMV so the expanded 1.5 MiB
plane never exists in a device slot or global scratch.

For this problem, model a scale stream as blocks of packed four-bit symbols.
Most symbols index a record-local exact byte codebook; one distinguished symbol
consumes the next byte from a stable ordered escape stream. A prefix table may
give the escape count before each fixed-size block. The precise sidecar parser
is in `src/glm53_generate.cu`; a proof must parameterize codebook and block size
rather than relying on private model bytes.

The current v2 projection region is concrete enough to use as a baseline. It
has 262,144 packed symbol bytes, `E` ordered escape bytes, a 16-byte codebook,
alignment to four bytes, and 1,025 uint32 exclusive-prefix entries (one per
256 packed bytes plus the final total). Its disk span is

```text
align_4096(align_4(262144 + E + 16) + 4100).
```

Any claimed byte saving must include those 4 KiB pads and prefix bytes.

## Mathematical problem

Construct a parallel random-access decoder that supplies one exact E4M3 scale
per 16-weight group directly to a row-major DP4A or MMA kernel. It must avoid a
serial scan from the start of the scale plane, retain coalesced body traffic,
and bound extra registers/shared memory. Multiple DFlash rows may reuse the
same decoded scale.

## Proof obligations

1. Specify the packed grammar formally and prove that `(block prefix, local
   popcount/rank)` addresses every escape exactly once and in original order.
   Include the last partial block and corrupt-input bounds.
2. Prove that selecting the fifteen most frequent bytes minimizes logical
   length within the current single-palette/one-escape-code grammar. Treat
   ties and show exactly when 4 KiB padding changes the set of co-optimal
   palettes.
3. Derive the minimum auxiliary index size needed for `O(1)` or bounded
   random access under an escape rate `p`. Compare dense prefixes, two-level
   prefixes, succinct rank, and warp-cooperative scans.
4. Give an information-theoretic lower bound under the declared independent-
   access contract. It must count the directory/rank structure and alignment,
   not merely zero-order entropy.
5. Prove coalescing and non-overlap for a concrete mapping from warp lanes to
   scale symbols, escape bytes, and 16-weight groups.
6. Derive the break-even inequality among:

   ```text
   651,264 bytes of slot capacity saved,
   additional resident-slot hit probability,
   packed H2D bytes saved,
   per-use rank/decode instructions,
   occupancy loss, and reuse over r activation rows.
   ```

7. Solve the cache/use asymmetry: a resident expert can be used many times, so
   repeated inline decode may eventually cost more than one expansion. Derive
   the optimal promotion rule among always-packed, expand-on-first-use,
   expand-after-N-uses, and dual-form storage.
8. Prove or disprove:

   > If the packed representation increases resident capacity and its symbols
   > are independently decodable, inline decode must improve end-to-end time on
   > a compute-rich GPU.

## CPU deliverables

- A byte-exact encoder/decoder and random-access rank oracle with exhaustive
  tests over small streams, every escape position, corrupt prefixes, and
  adversarial all-escape blocks.
- A symbolic optimizer for prefix granularity and promotion threshold.
- Synthetic traces showing regions where always-packed, lazy expansion, and
  expanded storage are each optimal.
- A precise proposed fused-kernel layout and list of changes to
  `src/glm53_expert_bench.cu`, `src/glm53_generate.cu`, and slot accounting.

## Engine gate

Before default-on, require byte-identical decoded scales, no out-of-bounds
under sanitizer fixtures, zero spills or a measured justification, and a
matched A/B that separately reports H2D bytes, inline decode time, slot count,
VRAM hit rate, round wall, decode/prefill throughput, and full quality metrics.
Kill any design whose index metadata gives back most of the 651,264-byte gain.
