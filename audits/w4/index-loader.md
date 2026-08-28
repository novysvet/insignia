# w4 audit — 27B index→loader→weights path (INSIDX02 vs ModelFile)

Audit date 2026-08-25. Scope: `tools/index27.py` vs `src/model_file.cpp` byte
compatibility, engine name-map completeness, CRC chain, `matrix()` fp8/bf16
branches, `Qwen35Weights` VRAM-budget/LRU behavior at 30 GB, 9B INSIDX01
regression, and the Phase-A **R0 gate**. Method: read everything cited; all
verification done by parsing `build/qwen38-27b-fp8.insignia-index` and all 66
safetensors headers with Python (no builds run, no files touched except this
report).

---

## 0. Verdict in one line

**The index itself is byte-perfect and census-clean (R0 data side: PASS), but
the C++ loader is still INSIDX01-only: `ModelFile` throws on the 27B index
before reading a single tensor, so nothing downstream (budget/LRU, matrix(),
CRC) is even reachable at 27B today.** The uncommitted work landed the writer
side of Phase A and the `matrix()` kind branches, but not the parser (plan
item A.3) or the call-site kind dispatch (plan item C.3).

---

## 1. Q1 — Does ModelFile parse INSIDX02? **No. INSIDX01-only, and the two formats are structurally incompatible.**

The only magic/version check in the engine:

- `src/model_file.cpp:13` — `struct Header { char magic[8]; uint32_t version; uint32_t count; uint64_t payload; }` (packed, 24 B).
- `src/model_file.cpp:23` — `if(std::memcmp(h.magic,"INSIDX01",8)||h.version!=1) throw std::runtime_error("bad Insignia index");`

`INSIDX02` fails the memcmp and version 2 fails the check even if the magic
were patched. No other C++ source references INSIDX02, shards, or CRCs
(grep over `src/*.cu`, `src/*.cpp`, `include/*`): the only hits are
`model_file.cpp:23` and comments in `include/insignia_streaming.hpp:5-9`,
whose NvmeReader is explicitly *"NOT the INSIDX index format"* (generic
`(path, offset, len)` plans). `src/streaming.cu` confirms: plan-builder from
the index does not exist yet.

Structural incompatibility (beyond magic/version):

| field | INSIDX01 (`model_file.cpp:22-37`, `tools/index_safetensors.py:22-34`) | INSIDX02 (`tools/index27.py:236-251`) |
|---|---|---|
| fixed header | 8s magic + u32 ver + u32 tensor-count + **u64 payload_offset** | 8s magic + u32 ver + **9×u32 shape header** + u32 shard-count (no payload offset) |
| payload | **one** path (u32 len + bytes), one `CreateFileW`+`MapViewOfFile` (`model_file.cpp:25-31`) | **66** shard records: `<IQI` path_len, file_bytes, crc32 + path bytes |
| tensor record | `HBBQQ` name_len, dtype, rank, off, **bytes**, then name, then shape (`model_file.cpp:34-35`) | `HBB` name_len, dtype, rank, then **name, then shape**, then `HQQ` shard_idx, off, bytes (`index27.py:246-251`) |
| offsets | relative to `payload_offset_` into the single mapping; bound-checked vs `mapped_bytes_` (`model_file.cpp:36`) | `shard` (u16) + **absolute in-shard file offset** `data_start+begin` (u64) + bytes (u64) |
| scale links | none (name convention `base+".scales"` at call sites) | **no explicit `scale_idx` field either** — link stays the `base+".scales"` name convention, enforced at build time (`index27.py:196-206, 333-337`) |

Feeding INSIDX02 bytes to the current parser: `count` would read shape[0]
(5120), `payload` would read shape[1..2] as one u64, `path_n` would read 24
(q_heads) bytes of shape data as a path → `"cannot open model payload"`. Total
garbage, not a near-miss.

**Deviations from the master-plan/loader-gaps design** (benign, but whoever
writes ModelFile v2 must follow `index27.py`, not the w3 reports): no per-shard
`data_start`/`align_base`/`flags(skip_vision,is_layer,is_mtp)` in the shard
table (only path/file_bytes/crc); no tensor `scale_idx`/`in_slot_off` (absolute
offsets instead); added 9×u32 config-verified shape header
(`index27.py:38-50,119-129`).

### My independent byte-parse of `build/qwen38-27b-fp8.insignia-index` (119,017 B)

Parsed with the writer's struct format, decoded from raw bytes:

- magic `INSIDX02`, version 2, shape header `(5120, 64, 248320, 24, 4, 48, 16, 17408, 4)` = hidden/layers/vocab/q_heads/kv_heads/delta_v/delta_k/inter/interval — all equal `config.json` text_config.
- **66 shards**; relative paths are native Windows separators relative to the
  **index file's own directory** (`os.path.relpath(model, out.parent)`,
  `index27.py:240`): `..\Qwen3.8-27B-FP8\layers-0.safetensors` …
  `..\Qwen3.8-27B-FP8\outside.safetensors`. All 66 resolve from `build\`.
  Implication: the index is location-coupled; a v2 loader must resolve against
  the index's directory, not the CWD.
- **1273 tensors** = 1606 checkpoint − 333 vision (all `model.visual.*`
  skipped; matches loader-27b-spec §2.6's 336−3). dtype census: **407×
  F8_E4M3 (=7), 866× BF16 (=2)** — exactly the census minus vision
  (1199−333=866). File order is name-sorted (verified) — compatible with
  `ModelFile::find`'s `lower_bound` convention (`model_file.cpp:38,44`).
- Name samples: `language_model.model.layers.0.linear_attn.in_proj_qkv.weight`
  → F8 `[10240,5120]` shard 0 off 1,135,208 bytes 52,428,800;
  `...in_proj_qkv.scales` → BF16 `[80,40]` 6,400 B;
  `language_model.model.embed_tokens.weight` BF16 `[248320,5120]`;
  `language_model.lm_head.weight` BF16 `[248320,5120]`;
  `language_model.mtp.fc.weight` BF16 `[5120,10240]`.
- **Byte-compatibility vs the actual checkpoint: perfect.** Re-parsing all 66
  safetensors headers and re-deriving engine names independently: name-set
  equal (1273/1273), **0 mismatches in (dtype, shape, shard, off, bytes)** across
  every tensor. 0 tensors escape their shard's file_bytes. 0 trailing bytes.
- Alignment note (feeds master-plan risk #7): 48/407 F8 tensors sit at
  in-shard offsets ≡ 8 (mod 16) (the rest ≡ 0) — the Phase-D ring pad trick
  and the `(f8_base & 15) == 0` acquire assert are genuinely needed.

## 2. Q2 — Name-map completeness: every engine acquire has a 27B counterpart

Enumerated every `matrix(`/`tensor(`/`storage_.acquire` site in engine code
(`src/decode.cu:30-31,32,34,46,49,56,66-72,78,81,85,89,90,96,97,126-133,139-141,145-146,153,159-190`;
`src/qwen35.cu:8,13,21,33`; `src/generate.cu:76`; `src/nll.cu:78`;
`src/dump_multistep.cu:32,48`). Layer names expand with
`p="language_model.model.layers."+l`, full attention iff `(l&3)==3`
(config `layer_types` verified: full-attn = {(i&3)==3} for all 64, 16 of them —
the engine's `Qwen35Shape::full_attention` pattern matches the 27B exactly).

Built the full expected set (700 names for layers 0–31 + globals + MTP, with
unconditional `.scales` for every matrix base) and diffed against the decoded
index:

**R0 name diff (actual run):**

| check | result |
|---|---|
| engine-expected names (l∈0..31 + mtp + embed/lm_head/norm, `.scales` added unconditionally) | 700 |
| NOT in index | **51** — all of them `.scales` of BF16 matrices: 48× `linear_attn.in_proj_{a,b}.scales` (24 linear layers), `lm_head.scales`, `model.embed_tokens.scales`, `mtp.fc.scales` |
| of those, names `matrix()` would actually acquire | **0** — `qwen35.cu:25-28` only acquires `.scales` when the weight dtype is u32 (mxfp4, :13) or f8_e4m3 (:21); BF16 weights take the no-scale branch. The 51 are the exact bf16-no-scale families the writer's self-read probe reports (`index27.py:332-347`) |
| names in index the engine never acquires (l≥32, engine loops `l<32`) | 622 tensors of layers 32–63 (Phase B gap, not a loader gap) |

Conclusion: under the actual acquire semantics, **every name the engine asks
for resolves**. The `.scales`-on-demand design is what makes the 27B name map
complete with zero engine-side renames beyond what `index27.py` already bakes
in (`model.language_model.*` → `language_model.model.*`, bare `lm_head`, `mtp.*`).

## 3. Q3 — CRC

- **Build time: yes, hard gate.** `index27.py:221-229` recomputes crc32 of all
  66 shards and fails the build on any mismatch vs `crc32.txt` (skippable only
  with `--no-crc`, `index27.py:136-137`).
- **In the file: yes.** Per-shard crc u32 in the shard table; all 66 fields
  nonzero and **all 66 == crc32.txt** (verified byte-for-byte; crc32.txt has
  77 entries — the other 11 are text files, correctly excluded).
- **Independent recomputation: layers-3.safetensors (372,313,744 B) →
  `5d630060` — matches both crc32.txt and the index field.**
- **Load time: nothing.** No C++ code reads the crc field (or could — nothing
  parses INSIDX02). `ModelFile`/`TieredStorage` never checksum. First
  load-time check is deferred to Phase D's R3 (stream-all-shards CRC pass).

## 4. Q4 — `matrix()` branches + budget/LRU for a 30 GB model

### fp8 branch (`src/qwen35.cu:19-24`)

- Acquires `base+".scales"`, requires `s.dtype==bf16` and
  `s.bytes == ((rows+127)/128)*((cols+127)/128)*2`. **Verified against the
  real index: 407/407 F8 tensors pass the exact formula; 0 orphan scales.**
  Rows/cols come from the weight shape (no ×8 on cols — correct for F8,
  unlike the u32 branch's `*8` at :12).
- Minor hardening nit: the assert checks `s.bytes` only, not `s.shape`; a
  hypothetically transposed scale ([40,80] vs [80,40]) would pass with equal
  bytes. The writer guarantees the shape (`index27.py:198-200`); a v2 loader
  should check the shape since `DeviceView.shape` is available anyway.

### bf16 no-scale branch (`src/qwen35.cu:25-28`)

Correct and returns `WKind::bf16, has_scales=false` — this is what the 51
BF16 matrices (lm_head/embed/mtp.fc/in_proj_a/b at 27B) hit. **But it has
exactly one consumer**: `decode.cu:155` (`mtp.fc` → `bf16_gemv`, with 9B
hardcoded dims 4096/8192). Every other dispatch site still branches on
`m.insig4` only and would route fp8/bf16 matrices into mxfp4 kernels:

- `decode.cu:31,32` (`linear`/`linear2`), `:33-41` (`linear_batch`),
  `:46` and `:141` (embed gather), `:67-76` (in_proj a/b pair),
  `:97-105,133,190` (lm_head), `generate.cu:76-80`, `nll.cu:78-81`,
  `qwen35.cu:33` (`embed_dev`).
- For a BF16 matrix the scales `DeviceView` is default-constructed
  (`data==nullptr`, `include/insignia_storage.hpp:13`) → the mxfp4 kernel
  dereferences null → CUDA illegal address. For fp8 it would "only" produce
  garbage (valid scales pointer, wrong decode). Kernels for the fix exist:
  `fp8_gemv/fp8_gemv2/fp8_gemm`, `bf16_get_row` (`include/insignia_fp8.cuh:23-26`),
  `bf16_gemv` (used at `decode.cu:155`).
- Also dtype-trap: **`A_log` is BF16 `[48]` at 27B** (F32 `[32]` at 9B —
  confirmed in `build/qwen35-insig4.insignia-index`), while `decode.cu:82,128`
  cast `(const float*)A.data` → master-plan risk #4 (silent α≈1). Phase A
  item 6 is not implemented.

### `Qwen35Weights` ctor budget logic at 30 GB (`src/qwen35.cu:5`, `src/storage.cu`)

- Ctor just forwards to `TieredStorage(m, budget, stream)`; budgets in drivers:
  6 GiB (`generate.cu:58`, `dump_layers.cu:5`, `dump_multistep.cu:27`,
  `dump_i4_seams.cu:16`, `dump_pf.cu:30`, `test_full_model.cu:6`), 1–2 GiB
  (dump_layer0/3, dump_attention).
- `make_room` (`storage.cu:8`) throws only if a **single tensor** exceeds the
  budget; largest 27B tensor is lm_head/embed at 2,542,796,800 B ≈ 2.37 GiB —
  fits under every budget in use, so no hard throw. Otherwise LRU-evicts
  unpinned entries (min tick), `cudaStreamSynchronize` before `cudaFree`
  (no use-after-free), then `cudaMalloc` + `cudaMemcpyAsync` HtoD from
  `t->data` **(a host pointer into the single INSIDX01 mmap)** + sync
  (`storage.cu:9`).
- **What breaks first, in order:**
  1. `model_file.cpp:23` throws `"bad Insignia index"` — hard stop today;
     nothing else is reachable.
  2. With a hypothetical single 30.9 GB mmap: `acquire` copies from demand-
     paged file pages — a 29.95 GB/token working set on 15.9 GB RAM thrashes
     the OS pager (best case ~9 s/token of pure NVMe re-reads at 3.3 GB/s)
     plus ~500 acquire-miss cudaMalloc+H2D+sync per token after evictions
     (entries are erased, so every evicted matrix is a full re-upload).
     Mechanically "works", strictly worse than even v1 all-stream.
  3. First BF16 matrix use crashes on null scales (see above) — embed_tokens
     is literally the first matrix `forward_token` touches.
  4. Even past that: `Qwen35Shape` is still 9B (`hidden=4096, layers=32`,
     `include/insignia_qwen35.hpp:7`) — `embed_dev`'s get_row cols, the
     `l<32` layer loops, and `mtp.fc`'s hardcoded 4096/8192 are all wrong at
     5120/64/10240 (Phase B, expected).

## 5. Q5 — 9B INSIDX01 path still consistent? **Yes — both sides untouched.**

- `tools/index_safetensors.py` is **unchanged from HEAD** (git diff empty);
  still emits INSIDX01 v1, single payload, `HBBQQ`+name+shape records, and its
  `DTYPES` still lacks `F8_E4M3` (master plan A.1's `"F8_E4M3": 7` addition
  was not made — irrelevant for the MXFP4 9B model, but the 9B tool cannot
  index an FP8 single-file checkpoint; `index27.py` covers that need).
- `src/model_file.cpp` likewise unchanged; `include/insignia_model.hpp:10`
  gained `f8_e4m3=7` (additive, safe).
- Verified by parsing `build/qwen35-insig4.insignia-index` and
  `build/qwen35.insignia-index` with model_file.cpp's exact layout: magic
  INSIDX01, version 1, 700 records, consumed == file size, dtypes as expected
  (u32 mxfp4 + f16/u8 scales, BF16 norms, F32 A_log). The writer/parser pair
  remains byte-compatible; "9B INSIDX01 path still loads" holds at the format
  level.
- Watch-out: `src/test_model.cpp:4` still asserts `tensors().size()==699`
  (stale count even for some 9B variants — the current 9B index has 700) and
  uses the `.scales` string convention; Phase A item 8 (dtype/shape table
  diff vs census) is not implemented.

## 6. Q6 — R0 gate: actual result table

R0 (master plan Phase A gate): *"index builds for the 27B dir; every
engine-expected name resolves; dtype/shape diff vs census = empty; 9B INSIDX01
path still loads."*

| R0 clause | evidence from this audit | status |
|---|---|---|
| index builds for the 27B dir | `build/qwen38-27b-fp8.insignia-index` exists, 119,017 B, self-read pass, CRC-verified at build | **PASS** |
| every engine-expected name resolves | 700 expected; 0 unresolvable under actual `matrix()`/`tensor()` semantics (51 naive-missing are all BF16-no-scale names never acquired) | **PASS** |
| dtype/shape diff vs census empty | vs config-derived census (== loader-27b-spec §2): 0 not-in-index, 0 extra, 0 dtype/shape mismatch over all 1273; scale ceil(128) math 407/407 | **PASS** |
| 9B INSIDX01 still loads | writer+parser both unchanged, byte-compatible, 9B index parses clean | **PASS (format)** |
| *implicit: loadable by the engine* | `model_file.cpp:23` throws on INSIDX02; no shard table/multi-file support | **FAIL — R0's data checks pass but the engine cannot open the index** |

Honest R0 status: **data-plane PASS, loader-plane FAIL.** The gate as written
is satisfiable by Python alone and is effectively green; the blocking gap to
an actual 27B load is the missing ModelFile v2 (+kind dispatch, Phase C).

## 7. Ranked fixes

1. **ModelFile v2 (Phase A item 3, blocking everything).** Parse INSIDX02:
   shard table (resolve relative paths against the index's own directory,
   native separators), tensor table `HBB,name,shape,HQQ`, per-shard bounds
   check, 66 eager handles (O_DIRECT+OVERLAPPED for the reader, optional
   mapped set for parity tools), keep the INSIDX01 branch for 9B. Follow
   `index27.py`'s bytes (no data_start/flags/scale_idx fields — the w3 report
   §2.2 design does not match the emitted format). ~120 LOC.
2. **Wire `WKind` dispatch at the 12 mxfp4-only call sites** (listed in §4):
   fp8 → `fp8_gemv/gemv2/gemm`, bf16 → `bf16_gemv/gemm`+`bf16_get_row`/
   `embed_gather_bf16`. Until then the 27B crashes at the first embed with a
   null-scales mxfp4 call (bf16) or garbles fp8 weights. (Plan Phase C.3 —
   pull the dispatch skeleton forward, it's cheap.)
3. **A_log BF16 handling** (`decode.cu:82,128`): dispatch on
   `DeviceView.dtype` per plan A.6; silent α≈1 otherwise.
4. **Load-time integrity (optional now, R3 later):** nothing reads the 66
   stored CRCs; either verify at first open (~9 s full sweep at 3.3 GB/s —
   probably skip) or rely on the Phase-D stream-time CRC pass as designed.
5. **16B alignment:** 48/407 F8 tensors at in-shard off ≡ 8 mod 16 →
   implement the Phase-D pad-at-F8-boundary in the plan builder +
   `(f8_base & 15) == 0` assert at acquire (risk #7 quantified).
6. **`test_model.cpp`** (plan A.8): replace the `==699` assert and
   `.scales`-always finds with a dtype/shape table diff vs the census (this
   audit's script is a ready-made reference).
7. Minor: `matrix()` should assert scale *shape*, not just bytes
   (`qwen35.cu:22`); `index_safetensors.py` still can't tag F8_E4M3 (only
   matters if the 9B tool is ever pointed at an FP8 file).

### Note on "scale_idx links"

The mission brief and master plan describe the tensor table as carrying
`scale_idx` links. **The emitted format has none** — links are the
`base+".scales"` name convention, validated exhaustively at build time
(`index27.py:196-206` orphan/F8-pair checks + `:332-337` probe; re-validated
independently here: 407/407 linked, 0 orphans). Functionally equivalent and
it keeps `Qwen35Weights::matrix` unchanged; a future `find_linked` would just
be `find(base+".scales")`.
