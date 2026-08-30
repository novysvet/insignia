# Session 10 — exact FP8 residency sample gate

Date: 2026-08-30

Requested base: `e48f633430c679ac6a30aae248159c887ac41601`
Scope: dense and optional DFlash2 E4M3 weight bytes plus their unchanged FP16 scale bytes

## Status

The supplied shared folder did not contain `fp8-residency-sample-v1.tar.zst` or
an equivalent dense-cache byte sample. The available body and scale archives
belong to other work and cannot establish E4M3 weight entropy. No compression
ratio, coding decision, CUDA decoder, or speed claim is made from synthetic
weights.

The dependency-gate deliverable is therefore:

- `tools/sample_fp8_cache.py`: source-cache inspector, deterministic collector,
  archive builder/splitter, and offline/source-backed validator;
- `tools/test_sample_fp8_cache.py`: malformed-input, exact-byte, tamper, and
  archive round-trip tests;
- this manifest/layout specification and owner-box command.

## Confirmed source layout

The tracked `Q8Index` reader and both FP8 quantizers agree on the following
little-endian `IGLMF8A1` format.

### `<prefix>.index`

Header, with no C/C++ padding:

| Offset | Type | Meaning |
|---:|---|---|
| 0 | `char[8]` | `IGLMF8A1` |
| 8 | `uint32` | version, exactly 1 |
| 12 | `uint32` | group size, exactly 64 |
| 16 | `uint32` | tensor count |
| 20 | `uint64` | exact `<prefix>.bin` byte size |

Each tensor record is `<HIIQQQQ` followed immediately by the UTF-8 name:

| Field | Type |
|---|---|
| name length | `uint16` |
| rows | `uint32` |
| cols | `uint32` |
| weight offset | `uint64` |
| weight bytes | `uint64` |
| scale offset | `uint64` |
| scale bytes | `uint64` |
| tensor name | `name_length` raw UTF-8 bytes |

For every record:

- `cols % 64 == 0`;
- `weight_bytes == rows * cols`;
- `scale_bytes == rows * (cols / 64) * 2`;
- weights are row-major one-byte E4M3 codes;
- scales are row-major little-endian FP16, one scale per 64 weights;
- the tracked quantizers align each weight and scale span to 4096 bytes;
- alignment padding is not part of a tensor and is not sampled.

The collector rejects wrong magic/version/group size, duplicate names, invalid
geometry, range overflow, overlapping spans, trailing index bytes, misaligned
quantizer-format spans, and a `.bin` size different from the index declaration.
`--allow-unaligned` exists only for reader-valid caches not emitted by the
tracked quantizers.

## Archive contract

The archive contains only:

```text
manifest.tsv
dense/<tensor-id>.weights.e4m3.bin
dense/<tensor-id>.scales.f16le.bin
dflash/<tensor-id>.weights.e4m3.bin       # only when requested
dflash/<tensor-id>.scales.f16le.bin       # only when requested
SHA256SUMS
```

No source index, checkpoint tensor, repository file, cache padding, symlink, or
other metadata file is admitted. `SHA256SUMS` covers `manifest.tsv` and every
weight/scale file. An external `<archive>.SHA256SUMS` records the complete
archive hash and, when splitting is required, every part hash.

### `manifest.tsv` schema

Columns are exact and ordered:

| Column | Contract |
|---|---|
| `source_kind` | `dense` or `dflash` |
| `tensor_name` | exact source index name |
| `rows` | complete source tensor row count |
| `cols` | complete source tensor column count |
| `group_size` | exactly 64 |
| `source_index_sha256` | SHA-256 of the source `.index` |
| `sample_row_begin` | first copied source row, zero based |
| `sample_rows` | number of complete consecutive rows copied |
| `weight_file` | relative E4M3 payload path |
| `weight_bytes` | exactly `sample_rows * cols` |
| `weight_sha256` | SHA-256 of the copied weight bytes |
| `scale_file` | relative FP16 scale path |
| `scale_bytes` | exactly `sample_rows * (cols / 64) * 2` |
| `scale_sha256` | SHA-256 of the copied scale bytes |
| `sampling_reason` | URL-escaped key/value description of family, stage, selection mode, row band, layer, and source-stratum bytes |

The validator rejects duplicate files, duplicate or overlapping row intervals,
undeclared files, unsafe paths, malformed/truncated archives, hash disagreement,
source-index disagreement, geometry disagreement, and any byte mismatch against
an optionally supplied source cache.

## Deterministic sampling policy

Defaults are designed for the requested 512 MiB dense sample:

1. Include every matrix whose weight payload is at most 2 MiB in full.
2. Derive a family from the tensor name after removing the layer number and
   normalizing numeric path components. Global matrices such as `lm_head` form
   their own family/stage.
3. Partition existing layer numbers into deterministic early/middle/late bins.
4. Select the largest, stage-central representative for every family/stage
   stratum.
5. Take non-overlapping start/middle/end row bands from each large
   representative, with at least 0.5 MiB per band by default.
6. Fill the remaining target in proportion to each stratum's full source weight
   volume. Row rounding can exceed the target by less than one final source row;
   required complete-small and coverage slices may also raise the total.
7. Fail if dense sampled weight bytes are below 256 MiB.

Selection has no random seed and is repeatable for a fixed index and options.
The collector copies source bytes with `pread`; it never decodes or re-encodes
E4M3 or FP16. Signed zero, NaN encodings, infinities, subnormals, and otherwise
unusual payload bits therefore remain exact.

## Owner-box collection command

Dense first pass, including a machine-readable source/collection report:

```bash
cd C:\\coding\\Insignia-glm53-dflash2
wsl -d Arch -- bash -lc '
  set -euo pipefail
  cd /mnt/c/coding/Insignia-glm53-dflash2
  python3 tools/sample_fp8_cache.py collect \
    --dense-prefix /var/lib/insignia/glm53-fp8-g64 \
    --output /var/lib/insignia/fp8-residency-sample-v1.tar.zst \
    --target-dense-weight-mib 512 \
    --minimum-dense-weight-mib 256 \
    --complete-below-mib 2 \
    --coverage-band-mib 0.5 \
    --part-mib 512 \
    --report-json /var/lib/insignia/fp8-residency-sample-v1.collection.json
'
```

Add the fixed drafter cache to the same collection when desired:

```text
--dflash-prefix /var/lib/insignia/glm53-dflash2-fp8-fixed \
--target-dflash-weight-mib 64
```

`collect` validates the source index, writes the sample, revalidates every
sample byte against the source, builds a deterministic-metadata tar, runs
`zstd -t`, hashes the full archive, and splits only when the compressed archive
exceeds 512 MiB.

After transfer, validate without source-cache access:

```bash
python3 tools/sample_fp8_cache.py validate \
  fp8-residency-sample-v1.tar.zst \
  --report-json fp8-residency-sample-v1.validation.json
```

On the owner box, add `--dense-prefix` and optional `--dflash-prefix` to prove
that every file still matches the live cache byte for byte. For split output,
pass `.part-000`; the validator concatenates consecutively numbered parts in a
temporary directory. A manual reassembly is:

```bash
cat fp8-residency-sample-v1.tar.zst.part-* > fp8-residency-sample-v1.tar.zst
sha256sum -c fp8-residency-sample-v1.tar.zst.SHA256SUMS
```

## Tests run

```bash
cd tools
python3 -m unittest -v test_sample_fp8_cache.py
```

Covered cases:

- valid `IGLMF8A1` parsing and family/stage inventory;
- all 256 possible weight-byte values copied without interpretation;
- arbitrary FP16 raw payloads copied without interpretation;
- bad magic, trailing index bytes, wrong geometry, overlapping spans, and
  truncated data rejection;
- deterministic, non-overlapping sampling with complete-small and all
  family/stage coverage;
- source-backed byte comparison;
- tamper and undeclared tracked-file rejection;
- `.tar.zst` and split `.tar.zst.part-000` round trips.

The local result is 8/8 tests passing. A synthetic cache is used only to test
format handling and exact copying; no entropy or compression result is derived
from it.

## Next gate

Upload the generated archive or parts, the external archive `SHA256SUMS`, and
the collection JSON. Only after those real dense E4M3 bytes validate should the
entropy, padded fixed-block ratio, held-out tensor split, CPU codec, or CUDA
work begin.
