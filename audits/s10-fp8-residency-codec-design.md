# Session 10 — exact E4M3 fixed-block codec and fused-consumer design

Date: 2026-08-30
Requested base: `e48f633430c679ac6a30aae248159c887ac41601`
Primary cache: dense `IGLMF8A1`, 699 matrices
Secondary cache: fixed DFlash2 `IGLMF8A1`, 48 matrices

## Current status and evidence boundary

The real-byte archive has been produced and independently validated on
`glm-box`. Its complete SHA-256 is
`6c9917a2f97508928f8d96899262bca7a4e0da47540f27099af709107b478e71`.
The archive is 528,788,854 bytes, which exceeds the analysis environment's
256 MiB Drive transfer limit. The collection inventory and validation reports
are available, but the weight payload is not yet locally readable. Therefore:

- the exact CPU reference format and analyzer are implemented and tested;
- no real-byte entropy, padded-ratio, winning-family, CUDA-speed, or allocator
  claim is made in this note;
- CUDA implementation remains gated on the real held-out result;
- split archive parts below 256 MiB are the remaining data-transfer dependency.

This preserves the task's prohibition against conclusions from synthetic
Gaussian or generated E4M3 bytes.

## Why the physical coding tile is two-dimensional

The current Ada consumer does not consume a contiguous 1024-byte row slice. A
warp stages one `16 x 64` weight slab for one group-64 column:

```text
16 output rows x 64 E4M3 bytes = 1024 bytes
```

The raw kernel issues 64 `uint4` loads. Four consecutive lanes form one aligned
64-byte row transaction, the slab lands in per-warp shared memory, and the same
slab feeds two `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`
instructions. Batched prefill reuses that weight slab across up to eight token
rows.

The codec consequently maps the requested nominal tile sizes as follows:

| Nominal bytes | Matrix region | Blocks needed by one current MMA slab |
|---:|---|---:|
| 128 | 2 rows x one group-64 column | 8 |
| 256 | 4 rows x one group-64 column | 4 |
| 512 | 8 rows x one group-64 column | 2 |
| 1024 | 16 rows x one group-64 column | 1 |

This is materially different from compressing arbitrary contiguous byte
windows. A 1024-byte codec tile is already in the exact shared-memory order the
consumer needs. Smaller tiles may improve locality statistics, but pay more
directory loads, mode decisions, exception scans, and synchronization per MMA
slab. Equal byte ratios therefore prefer 1024, then 512, 256, and 128.

FP16 group scales remain untouched. The codec neither transforms nor claims
savings from them. They retain the current row-major one-scale-per-64-weight
layout and are loaded at the existing accumulation point.

## Reference container `IF8XTC01`

The reference stream is deterministic, little-endian, independently
addressable, and contains no global dictionary or serial prefix.

### Header

The header is 64 bytes. It records:

- magic `IF8XTC01`;
- version 1 and 64-byte header size;
- nominal tile bytes;
- matrix rows and columns;
- group size 64;
- canonical-zero-padding flag;
- exact raw byte count and tile count;
- directory and payload offsets.

### Directory

Every tile has one eight-byte descriptor:

```text
uint32 offset_in_16_byte_units
uint16 unpadded_payload_bytes
uint8  mode
uint8  parameter
```

The directory is padded once to 16 bytes. Every payload begins on a 16-byte
boundary. Offsets are absolute within the matrix payload region, so any tile is
reachable with one descriptor load. The 32-bit offset field addresses up to
64 GiB per matrix at 16-byte granularity.

### Raw and matrix escapes

Each compressed candidate is compared after 16-byte payload padding. RAW wins
all equal-sector ties, preventing extra bit work with no byte saving. The
analysis additionally models a tensor-level raw escape: a matrix remains in the
existing representation when its complete compressed container is not smaller
than its raw weight span. Thus incompressible tensors do not pay a directory or
header expansion in the versioned cache estimate.

### Exact tile modes

The implemented candidate set is deliberately small and warp-decodable:

| Mode | Representation |
|---|---|
| `BYTE_PALETTE4/5/6` | Per-tile full-byte palette, packed indices, dense literal exceptions |
| `MAG_PALETTE4/5/6` | Raw sign bitplane, per-tile 7-bit magnitude palette, packed indices and exceptions |
| `EXP_PALETTE2/3` | Sign bitplane, raw three-bit mantissas, exponent palette and four-bit exceptions |
| `MAG_XOR4` | Sign bitplane, one modal magnitude base per 32 values, four-bit XOR residuals and literals |
| `ZERO_SPARSE` | Magnitude-nonzero bitmap, signed-zero bits, exact nonzero literals |
| `BITPLANE_CONST` | Omit bitplanes that are all zero or all one inside a tile |
| `RAW` | Original bytes, unchanged |

Palette entries are ordered by descending frequency and then byte value.
Palette length minimizes padded bytes, then stored bytes, then palette count.
All bit streams are LSB-first. Signed zero and both E4M3FN NaN byte encodings
are ordinary bit patterns and round-trip unchanged.

The decoder rejects unknown modes, absent palette entries, duplicate palette
values, wrong exception counts, noncanonical literals, nonzero packed-bit or
sector padding, overlapping/gapped payloads, truncation, trailing data, invalid
geometry, and directory/header disagreement.

## Analyzer and held-out protocol

`tools/analyze_fp8_residency.py` consumes a sample directory, `.tar`, `.tar.zst`,
or the first part of a consecutively numbered split archive. It first invokes
the independent sample validator and then emits:

- `tensor_inventory.csv` — per-tensor H0, byte support, signed zeros, NaN bytes,
  bitplane frequencies, horizontal and vertical pair statistics;
- `family_entropy.csv` — family/stage/split histograms and correlations;
- `tensor_tile_stats.csv` — distinct-count and signed-zero tile distributions;
- `tensor_codec_results.csv` — actual padded payload, directory, header, raw
  tile fraction, matrix raw escape, and raw-scale-inclusive ratios;
- `family_codec_results.csv` — family aggregates for every format/tile pair;
- `heldout_results.json` — train choices, held-out generalization, bootstrap
  intervals, disk-layout model, allocator model, and gate decision;
- `summary.json` — machine-readable top-level receipt.

The split is by whole tensor:

```text
sha256(source_kind + NUL + tensor_name) modulo 100
```

No bytes from one tensor occur in both train and held-out sets. Training selects
one tile size and format per family. Equal ratios prefer the 1024-byte MMA tile
and then the reference decoder's fixed mode-cost order. Families with no train
or held-out member use an explicitly reported source-kind fallback.

Whole-cache estimates use the full 699/48-tensor inventory, not sampled-byte
frequency. Bootstrap replicates resample held-out tensors within family and
then reweight by full family bytes. Scale bytes remain raw in every total ratio.
The disk estimate replays the current quantizers' 4096-byte weight/scale span
alignment. The VRAM estimate uses logical allocation bytes and reports potential
13.5 MiB expanded-expert slots, but labels them modeled until the live allocator
actually consumes the reclaim.

## Decision gates encoded by the analyzer

For each source cache, the analyzer reports:

- weighted held-out weight ratio and 95% tensor-bootstrap interval;
- weighted held-out ratio including unchanged scale bytes;
- average held-out per-tensor H0, reweighted by full family volume;
- strongest directly held-out family with at least 5% of source weight bytes
  and at least 64 MiB;
- logical VRAM and 4096-aligned disk bytes saved;
- potential expanded expert slots under the stated 13.5 MiB assumption.

The pre-CUDA kill condition is exactly represented as:

```text
weighted held-out H0 > 7.6 bits/byte
AND
best selected fixed-block held-out ratio > 0.94
```

Eligibility for the normal GPU microbenchmark requires:

```text
weighted held-out ratio <= 0.90
AND
one high-volume directly held-out family <= 0.85
```

An allocator-only probe may remain interesting when the modeled logical reclaim
is at least 384 MiB and 28 assumed slots, but it is not treated as allocator
proof. The live engine must print the resulting Q8 bytes and expert-slot count.

## Fused warp decode mapping

The 1024-byte path is the primary implementation target if it wins held-out
bytes.

1. The current warp computes `(first_row, group)` exactly as before.
2. It loads the eight-byte descriptor for tile
   `(first_row / 16) * groups + group`.
3. Lanes cooperatively load the aligned compressed payload.
4. Every lane reconstructs its 32 owned output bytes into the existing
   1024-byte per-warp shared slab.
5. Warp prefix counts locate dense exception literals. No global prefix is read.
6. The existing `__syncwarp()` makes the slab visible.
7. The current register loads, MMA instructions, FP16 scale widening, and FMA
   accumulation order remain unchanged.

Decoded bytes never enter global memory. RAW tiles use vector loads into the
same slab. A palette can occupy a small per-warp shared tail or registers;
which choice survives depends on measured register pressure. For 4/5/6-bit
indices, each lane uses fixed bit offsets and at most two adjacent packed words
per extraction. Exception rank is a warp-local prefix over each lane's sentinel
count. The exception list remains dense and random-access at tile scope.

The 512/256/128 paths decode two/four/eight descriptors into successive row
subslabs. They require no global expansion, but their metadata and control cost
must beat any ratio advantage.

## First-order roofline

Let `R` include payload padding, the eight-byte descriptor per tile, matrix
header amortization, and raw escapes. With the measured raw FP8 kernel bandwidth
`BW_raw ~= 698 GB/s`:

```text
T_raw = B_raw / BW_raw
T_new = max(B_comp / BW_mem, Ops_decode / Throughput_decode)
        + metadata + synchronization + occupancy loss
```

An ideal 1.08x matrix speedup requires `T_new <= 0.9259 T_raw`. Memory alone
therefore needs `R <= 0.9259`; the task's `R <= 0.90` gate leaves limited room
for decode and occupancy costs. The decoder must sustain at least roughly
`698 / 0.9259 ~= 754 GB/s` of reconstructed-byte throughput when it is the
roofline term. This is a measurement target, not a claim about Ada integer
throughput. Metadata, instruction count, scoreboard stalls, register usage,
shared memory, and occupancy are measured in the standalone benchmark before
integration.

For a 1024-byte tile, one descriptor is 0.78125% of raw weight bytes before the
matrix header. At `R=0.85`, only about 146 bytes per tile remain as ideal memory
headroom after the descriptor. The implementation cannot afford a second
barrier, full-warp serial exception walk, local-memory spill, or global decoded
buffer.

## Standalone CUDA benchmark contract after the ratio gate

The benchmark should reuse the production MMA body and compare raw versus fused
exact decode on at least these source geometries:

### Dense

- `154880 x 4096` — `lm_head`;
- `8192 x 4096` — KDA q/k/v families;
- `4096 x 8192` and `4096 x 16384` — attention output families;
- `2048 x 4096` and `4096 x 2048` — shared expert gate/up/down;
- `16384 x 1536` — MLA q-b;
- `32768 x 512` — MLA kv-b.

### DFlash2, only after dense succeeds

- `12288 x 4096` and `4096 x 12288` — MLP;
- `4096 x 4096` — q/o;
- `4096 x 10240` — FC halves;
- `1024 x 4096` — k/v/dynamic-kernel projections.

For every case:

- use identical activation bytes, scale bytes, MMA instructions, and output
  accumulation order;
- compare decoded tile SHA-256 and output bits against raw;
- measure CUDA-event medians for warm repeated matrices and cold streaming
  after a cache-thrash pass;
- sweep tile size and only the formats that survive held-out bytes;
- report compressed bytes actually loaded, directory bytes, decoder time, MMA
  time, and total kernel time;
- compile with ptxas verbose output and record registers, stack, spills, shared
  memory, launch occupancy, and achieved bandwidth;
- save `cuobjdump --dump-sass` output and verify no local-memory spill or global
  decoded-weight store.

Only a benchmarked family at least 1.08x faster, or an allocator-backed reclaim
of at least 384 MiB/28 actual slots with no more than 2% kernel regression,
should proceed to the versioned reader and residency integration.

## Tests currently passing

The reference and analyzer test suite covers:

- all 256 E4M3 byte patterns, including signed zero and NaN encodings;
- every mode independently;
- deterministic full-matrix containers for all four tile sizes;
- random tile access and full SHA-equivalent round trips;
- malformed headers, directories, payloads, bit padding, sector padding,
  truncation, and trailing bytes;
- 600 random/adversarial property cases;
- vectorized size formulas equal to the exact encoder sector-for-sector;
- mixed-mode tie behavior equal to the reference mode priority;
- end-to-end sample-directory validation, entropy/ratio CSV generation,
  whole-tensor holdout selection, bootstrap output, disk alignment, and
  allocator modeling;
- reference CLI encode/inspect/decode behavior.

Synthetic matrices in these tests establish implementation correctness only.
They are not reported as compression evidence.

## Owner-box result-bundle path

The real archive can be analyzed in place without transferring its 504 MiB
payload. The wrapper validates the external archive receipt, runs the independent
sample validator and full analyzer, records tool/environment hashes, and emits a
small deterministic result bundle plus SHA-256 receipt:

```bash
cd C:\\coding\\Insignia-glm53-dflash2
wsl -d Arch -- bash -lc '
  set -euo pipefail
  cd /mnt/c/coding/Insignia-glm53-dflash2
  tools/run_fp8_residency_analysis.sh
'
```

Default outputs are:

```text
/var/lib/insignia/fp8-residency-analysis-v1/
/var/lib/insignia/fp8-residency-analysis-v1.tar.zst
/var/lib/insignia/fp8-residency-analysis-v1.tar.zst.sha256
```

Only the final result bundle and receipt need to leave `glm-box`.

## Commands

After the archive or its split parts are locally materialized and the external
checksums pass:

```bash
python3 tools/sample_fp8_cache.py validate \
  fp8-residency-sample-v1.tar.zst.part-000 \
  --minimum-dense-weight-mib 256 \
  --report-json audits/raw/s10/fp8-residency-sample-v1.validation.json

python3 tools/analyze_fp8_residency.py \
  fp8-residency-sample-v1.tar.zst.part-000 \
  --collection-json fp8-residency-sample-v1.collection.json \
  --output-dir audits/raw/s10/fp8-residency-analysis-v1 \
  --holdout-percent 20 \
  --bootstrap-replicates 2000 \
  --bootstrap-seed 5902584
```

A bounded smoke may use `--max-sample-weight-mib 16`; it must never be reported
as the coding decision.

Reference-container CLI:

```bash
python3 tools/fp8_residency_codec.py encode weights.e4m3.bin weights.if8 \
  --rows 4096 --cols 8192 --tile-bytes 1024 \
  --report-json weights.if8.encode.json

python3 tools/fp8_residency_codec.py inspect weights.if8 \
  --report-json weights.if8.inspect.json

python3 tools/fp8_residency_codec.py decode weights.if8 weights.roundtrip.e4m3.bin \
  --report-json weights.if8.decode.json

sha256sum weights.e4m3.bin weights.roundtrip.e4m3.bin
```

Test command:

```bash
cd tools
python3 -m unittest -v \
  test_sample_fp8_cache.py \
  test_fp8_residency_codec.py \
  test_analyze_fp8_residency.py
```

## Owner-box result (2026-08-30)

The checksum-locked 528,788,854-byte sample was analyzed in full with 2,000
bootstrap replicates. All 25 codec, analyzer, and collector tests passed under
Linux. The result bundle is
`/var/lib/insignia/fp8-residency-analysis-v1.tar.zst`, SHA-256
`ab2866b6ac9a9fa939bb4882f56ee9b170adfb149ba720464c79f95f4f51cb69`.

| family | held-out weight ratio | ratio incl. raw scales | best high-volume ratio | entropy |
|---|---:|---:|---:|---:|
| dense | 0.903314 | 0.906244 | 0.902110 | 6.464 bits/byte |
| DFlash | 0.904546 | 0.907438 | 0.904551 | - |

Neither family reached the required total-byte ratio of 0.90, and no large
family approached the 0.85 threshold that would justify a fused execution
codec. Do **not** build the decompressor hot path.

The exact dense container nevertheless models 818,675,361 logical bytes
(about 780.7 MiB) reclaimed, equivalent to roughly 57 existing expert slots;
the allocator-value gate passed. This is only an allocator ceiling. Current
dense kernels still require expanded E4M3 bytes, so the reclaim is not real
residency until a cold-matrix/decode-on-load allocation scheme demonstrates
that another consumer can occupy the saved VRAM with no more than 2% kernel
regression. DFlash saves only 104,194,871 logical bytes (about seven slots) and
is closed. The final analyzer decisions were
`eligible_for_allocator_only_probe` for dense and
`stop_or_revisit_fixed_block_formats` for DFlash.

## User decision override (2026-08-30)

The user accepts the measured 0.906244 dense and 0.907438 DFlash total ratios;
missing the original 0.90 gate by roughly 0.6--0.7 percentage points is not a
reason to abandon this compute-for-bandwidth path. The codec is now **PARKED
FOR IMPLEMENTATION**, not rejected.

Deferred order:

1. implement the exact 1024-byte slab decoder and standalone fused-MMA
   benchmark for dense matrices;
2. integrate versioned cold/resident allocation and measure whether the modeled
   780.7 MiB becomes usable expert-cache space;
3. implement the DFlash variant after the dense path is understood;
4. preserve decoded bytes and accumulation order exactly, and report output
   parity, registers, spills, occupancy, compressed bytes loaded, and total
   kernel/engine time.

The old 1.08x speed and 0.90 ratio thresholds become comparison baselines, not
implementation prerequisites. A slowdown is not to be hidden: keep the codec
available for residency-sensitive configurations and establish its actual
break-even point against extra expert-cache hits.
