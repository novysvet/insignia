# INSIG week 3 — loader gap analysis: Qwen3.8-27B-FP8 un-requantized

Audit date: 2026-08-25. Scope: what stands between the current engine and loading
`E:\coding\Insignia\Qwen3.8-27B-FP8\` (66 shards, 1606 tensors, 28.75 GiB) with no
checkpoint modification. Read-only audit; nothing built, nothing committed.

Sources verified firsthand this session (line numbers below are exact):

- `src/model_file.cpp` (45 lines), `include/insignia_model.hpp` (43), `src/storage.cu` (12),
  `include/insignia_storage.hpp` (33), `tools/index_safetensors.py` (38), `src/qwen35.cu` (14),
  `include/insignia_qwen35.hpp` (21), `src/fp8.cu` (196), `include/insignia_fp8.cuh` (29),
  `src/test_fp8.cu` (131), `src/decode.cu` (262), `src/qwen_kernels.cu` (conv4/bf16_gemv),
  `src/test_model.cpp` (4).
- `build/qwen35.insignia-index` parsed byte-by-byte (see §2.1). No 27B index exists yet:
  `ls build/` shows only `qwen35.insignia-index`, `qwen35-insig4.insignia-index`,
  `qwen35-insig4-good.insignia-index` (all 9B).
- Header-only re-verification of the checkpoint (8-byte len + JSON, no payloads): dtypes are
  exactly `{BF16: 1199, F8_E4M3: 407}`; all 407 `weight_scale_inv` shapes equal
  `[ceil(r/128), ceil(c/128)]` of their `X.weight` (0 violations, 0 orphans); shards are
  gapless and end flush; per-shard F8 phase residues 616/640/3728/3744/1840 confirmed;
  `outside.safetensors` has no F8 at all.
- `audits/w2/loader-27b-spec.md` (complete census) and `audits/synthesis.md` (week-2 21-agent
  synthesis) read in full; this document builds on them and does not repeat the census tables.

---

## 1. DType enum vs needed dtypes

**What exists** — `include/insignia_model.hpp:10`:

```cpp
enum class DType : uint8_t { f32=1, bf16=2, f16=3, u8=4, u32=5, i8=6 };
```

**What the 27B needs**: `F8_E4M3` (weights) + `BF16` (everything else). BF16 exists; F8 does not.
The 9B uses `u32` (MXFP4 packed nibbles) + `u8`/`f16` scales — none of those dtypes appear in
the 27B checkpoint at all (verified: zero U32/U8/F16 tensors).

**Where strings are parsed**: `src/model_file.cpp` parses **no dtype strings** — it reads the
u8 dtype tag from the binary index (`model_file.cpp:34`, `take<DType>(p)`). The only
safetensors-header string→enum mapping lives in `tools/index_safetensors.py:8`:

```python
DTYPES = {"F32": 1, "BF16": 2, "F16": 3, "U8": 4, "U32": 5, "I8": 6}
```

`"F8_E4M3"` is absent → `index_safetensors.py:29-31` raises
`ValueError: unsupported dtype F8_E4M3` on the very first weight tensor. The loader therefore
cannot even index the checkpoint today.

**Patch (both ends, keep 9B working)**:

```cpp
// include/insignia_model.hpp:10
enum class DType : uint8_t { f32=1, bf16=2, f16=3, u8=4, u32=5, i8=6,
                             f8_e4m3=7,   // Qwen3.8-27B-FP8 weights (1 B/elem)
                             f8_e5m2=8 }; // reserved, not present in this checkpoint
```

```python
# tools/index_safetensors.py:8
DTYPES = {"F32": 1, "BF16": 2, "F16": 3, "U8": 4, "U32": 5, "I8": 6, "F8_E4M3": 7}
```

Bytes-per-element table (needed by the builder's sanity check and by `matrix()`):
f8_e4m3 = 1, bf16 = 2. All 407 F8 tensors have `bytes == r*c` (multiples of 4096, verified).

---

## 2. Multi-shard: index format and file strategy

### 2.1 What the 9B index actually is (parsed from `build/qwen35.insignia-index`, 62,128 B)

INSIDX01 layout, little-endian packed (matches `model_file.cpp:13-36` exactly):

```
Header : char magic[8]="INSIDX01"; u32 version=1; u32 count=699; u64 payload=90413
Path   : u32 path_len; u8 path[]    -- ONE absolute path:
         C:\Users\Pufos\.cache\huggingface\hub\models--sleepyeldrazi--
         Qwen3.5-9B-MXFP4-MTP\snapshots\18fa...\model.safetensors
Tensor : u16 name_len; u8 dtype; u8 rank; u64 off; u64 bytes; char name[]; u64 dims[rank]
```

- **One payload file.** `ModelFile` opens exactly one `CreateFileW` + `CreateFileMappingW` +
  `MapViewOfFile` (`model_file.cpp:27-31`); every `t.data = base_ + payload_offset_ + off`
  (`model_file.cpp:35`); the only bounds check is a single-mapping escape check
  (`model_file.cpp:36`). There is no shard concept anywhere in the C++.
- Tensor table sorted by name; `find()` is `std::lower_bound` (`model_file.cpp:44`) — O(log n),
  signature `find(std::string_view) -> const TensorView*`.
- MXFP4 encoding visible in the 9B shapes: `weight` stored as u32 `[rows, cols/8]`
  (e.g. `in_proj_qkv.weight (8192,512)` = 8192×4096 nibbles), `.scales` u8 `[rows, cols/32]`
  (MLX) or f16 `[rows, cols/64]` (INSIG4). `matrix()` multiplies `cols` back by 8
  (`src/qwen35.cu:7`).
- dtype census of the 9B index: {u8: 257, u32: 257, bf16: 161, f32: 24}. The f32 ones are the
  32 per-layer `A_log` tensors — see §4: in the 27B A_log is **BF16**.
- `src/test_model.cpp:4` hard-asserts `tensors().size()==699` and looks up
  `...in_proj_qkv.scales` — both fail on a 27B index.

### 2.2 The 27B index (INSIDX02) — what `index_safetensors.py` must emit

Endorse `audits/w2/loader-27b-spec.md` §5.2 with these refinements (deltas from w2 marked):
shard table gains a whole-shard `align_base` (replaces `f8_align_base` — staging is shard-major,
see §7), and the per-tensor `in_slot_off` is stored precomputed instead of deriving window math
at runtime. Logical content per tensor is exactly `{name: {file, dtype, shape, offset, bytes}}`
plus the scale link; physical format stays packed binary (a JSON index of 1606 entries would be
~300 KB and ~ms to parse; the binary keeps `find()` at two cache-line-ish probes and changes
nothing in the C++ reader shape):

```
Header      : char magic[8]="INSIDX02"; u32 version=2;
              u32 shard_count(66); u32 tensor_count(1606);
Shard entry : u16 path_len; u64 file_size; u64 data_start;      // 8 + safetensors header len
              u64 payload_len;         // file_size - data_start (shards end flush; verified)
              u64 align_base;          // data_start & ~4095     [delta: whole-shard, not F8-only]
              u32 crc32;               // from crc32.txt, VERIFIED AT BUILD ONLY (§7)
              u8  flags;               // bit0 skip_vision, bit1 is_layer_shard, bit2 is_mtp
              u8  path[path_len];      // RELATIVE to index dir (9B index embedded an absolute
                                       //  HF-cache path — do not repeat that)
Tensor entry: u16 name_len; u8 dtype; u8 rank; u16 shard;
              u32 scale_idx;           // tensor idx of X.weight_scale_inv; 0xFFFFFFFF = none
              u32 in_slot_off;         // abs_off - shard.align_base   [delta: precomputed]
              u64 off;                 // begin relative to data_start (safetensors convention)
              u64 bytes; u64 dims[rank]; u8 name[name_len];
```

Builder requirements (all facts verified against headers this session):

- Loop over all 66 shards in the checkpoint dir (do not trust `model.safetensors.index.json`
  for offsets — it is a name→file map only; read each shard's own header). Skip nothing; index
  vision too, flagged `skip_vision`.
- Table sorted by name (binary search survives unchanged). Names stored **verbatim**
  (`model.language_model.layers.N.*`, `lm_head`, `mtp.*`) — the *engine's* name construction
  changes, not the data (§4).
- Link every F8 tensor to its BF16 `X.weight_scale_inv` in the same shard (407/407 exist;
  scale shape == `[ceil(r/128), ceil(c/128)]` re-verified with 0 violations). Reject orphans.
- Per-shard bound: `off + bytes <= payload_len` for every tensor (replaces the single-mapping
  escape check at `model_file.cpp:36`).
- Verify each shard CRC32 against `crc32.txt` **at build time** (w2 already matched all 66).
- Size estimate: 66×(~40 B + path) + 1606×(~40 B + ~45 B name) ≈ **115 KB**.

### 2.3 Open-file strategy: 66 persistent handles (recommendation)

- **66 open handles, eagerly at ModelFile construction.** Windows' per-process handle budget
  is effectively unlimited at this scale; 66 `CreateFileW` calls ≈ 1-2 ms once. Trivial.
- **LRU-of-handles**: pure complexity at 66 files — a cache to avoid a non-problem.
- **Reopen-per-read (~10-30 µs/open)**: the raw cost is negligible (a 383 MB shard is ~192
  2 MB-block reads; even 192×20 µs ≈ 4 ms ≈ 7% of one 57 ms shard read, and you would open
  once per shard, not per block). The real killers for the IOCP reader future: (a) you cannot
  keep 8 outstanding `ReadFile(OVERLAPPED)` calls pipelined across a handle you keep closing
  and reopening; (b) the open path traverses filter drivers (Defender) and has unbounded tail
  latency spikes; (c) `FILE_FLAG_NO_BUFFERING` handles are the reader's natural unit of
  queueing — one handle per shard, all 66 bound to **one IOCP port** on the io thread.
- Keep a *second*, optional set of plain mapped views (`FILE_FLAG_RANDOM_ACCESS` +
  `MapViewOfFile`) for warmup/parity tooling — mappings cost nothing but VA (~29 GiB, fine on
  x64) and are never on the serving path (§3).

**Verdict: open all 66 twice (O_DIRECT+OVERLAPPED set for the staging reader; mapped set
optional/lazy). No LRU. No reopen.**

---

## 3. Host access path: what storage.cu is today, and TieredStorage v2

### 3.1 Today (verified by reading both files end to end)

- `ModelFile` maps the single payload zero-copy: `CreateFileMappingW` + `MapViewOfFile`
  (`model_file.cpp:30-31`). Pages fill by 4 KB fault on first touch.
- `TieredStorage` (`src/storage.cu`, all 12 lines) is **device-only**: `entries_: name →
  {device ptr, bytes, pins, tick}`, one `budget_`, `make_room` LRU-evicts unpinned device
  copies (`storage.cu:8`), `acquire` does `cudaMalloc` + `cudaMemcpyAsync` **from the mmap**
  (`t->data`, `storage.cu:9`) and synchronizes. `MemoryTier { nvme_mapped, host_pinned,
  device }` is declared (`include/insignia_storage.hpp:11`) but `host_pinned` is unimplemented.
- So the entire NVMe tier today = demand paging through the file mapping. That is 4
  KB-granular, double-buffered through the page cache (fatal for a 28.75 GiB working set on a
  15.9 GiB box), and per the week-2 research tops out ~2-3 GiB/s with `mmap`+
  `PrefetchVirtualMemory` explicitly **not** reliable for line rate.

### 3.2 The decision (agrees with the task brief, argued)

**All tiers go through a pinned-RAM staging ring filled by an IOCP reader. mmap survives only
for index build and one-time warmup/parity scans.** Reasons:

1. The same 4 KB-fault pathology hits the *CPU-compute* tier too: a CPU GEMV reading weights
   via page faults is the identical problem one PCIe hop away. CPU-tier weights must live in
   **pinned** RAM (also required anyway for async H2D DMA of VRAM-tier tensors).
2. Page-cache double-buffering: 28.75 GiB of weights vs 15.9 GiB RAM — the standpage manager
   would thrash. `FILE_FLAG_NO_BUFFERING` into pinned slots is deterministic
   (5.5-6.5 GiB/s at QD 8-16 per week-2 research).
3. One mechanism, three tiers: a slot in the ring *is* the RAM tier; `cudaMemcpyAsync`
   slot→device is the VRAM tier; a slot not yet filled is the NVMe tier.

### 3.3 TieredStorage v2 interface

```cpp
// include/insignia_storage.hpp (v2, additive; ModelFile v2 supplies shard/off/bytes, not .data)
enum class Tier : uint8_t { nvme, ram, vram };

struct PlacementRule { int layer_lo, layer_hi; Tier tier; };   // manifest, see below

class TieredStorage2 final {
public:
    TieredStorage2(const ModelFile2 &mf, uint64_t ram_budget, uint64_t vram_budget,
                   cudaStream_t upload_stream);
    // ---- placement manifest: name -> tier hint -------------------------------
    // Rules attach to shard slots (layer granularity) and named specials
    // ("lm_head", "embed_tokens"). Weights inherit their layer's tier; the
    // manifest is data (a file), not code, per AGENTS.md "modify constant data
    // directly". Example for L=21/M=23/N=21 (synthesis §feasibility):
    //   layers 21..41 -> vram ; layers 0..20,42..63 -> ram ; default -> nvme
    //   lm_head -> vram (mandatory: 2.5 GB, 5.1 ms/token vs 102 ms over PCIe)
    //   embed_tokens -> vram ; mtp -> vram ; model.visual.* -> never (skip)
    void set_placement(std::vector<PlacementRule> rules);

    // ---- async prefetch (scheduler thread only) ------------------------------
    // Begins filling the slot holding `name`'s shard (or the named special's
    // window). Non-blocking, idempotent per slot generation: a prefetch for a
    // slot that is FILLING or READY is a no-op; for a slot being evicted it
    // re-arms it. Issues the read plan (§7) as 2 MB blocks at QD 8 on the
    // shard's O_DIRECT handle. Returns immediately.
    void prefetch(std::string_view name);

    // ---- acquire (compute threads) -------------------------------------------
    // Blocks until the slot is READY (if the slot wasn't prefetch-armed, this
    // performs a synchronous fill first). Pins the slot, then for vram-tier
    // names uploads slot->device (cudaMalloc + cudaMemcpyAsync from PINNED
    // memory on upload_stream, real DMA, no page-cache bounce) and returns a
    // DeviceView; for ram/nvme-tier names returns a host view into the pinned
    // slot itself (CPU-tier kernels read it directly at DRAM bandwidth).
    // Graph-capture rule: acquire() during stream capture is only legal for
    // vram-tier names whose pins are held for the capture's lifetime — captures
    // bake device pointers (week-2 bug 6: post-capture eviction freed graph
    // pointers). Enforce with an assert.
    DeviceView acquire_blocking(std::string_view name);
    HostView  acquire_host_blocking(std::string_view name);   // CPU-tier path
    void      release(std::string_view name) noexcept;        // unpin

    // ---- io thread ------------------------------------------------------------
    void poll_completions();   // GetQueuedCompletionStatus loop; FILLING->READY,
                               // wakes acquire_blocking waiters
private:
    enum class SlotState : uint8_t { empty, filling, ready, evicting };
    struct Slot {                              // granularity = ONE SHARD (§7)
        void  *pinned;                         // cudaHostAlloc, 4096-aligned
        SlotState state; uint32_t pins; uint64_t tick; uint32_t generation;
        uint16_t shard;                        // or named-special window
    };
};
```

**Threading contract (normative):**

- Exactly **one io thread** owns the completion port and all state transitions
  `filling→ready`; it never runs kernels.
- `prefill/prefetch` may only be called from the **scheduler (decode) thread**, ~1 layer ahead
  (decode is token-serial; read-ahead hides latency, not bandwidth — week-2 math).
- `acquire_blocking` may be called from the decode thread or a CPU worker; different slots are
  independent — a CPU worker holding layer K's slot does not block decode on layer K+1. A slot
  is filled by at most one reader at a time (state machine guarantees).
- `release` decrements pins; a slot with `pins==0` is LRU-evictable (oldest `tick` first).
  Eviction order across tiers: free device copies first (cheap re-upload from pinned), pinned
  slots last (NVMe round trip) — one `tick` counter drives both, as w2 §5.5 specified.
- Small BF16 layer params (~1.1 MB/linear layer, ~67 KB/full layer, ~65 MB all 64 layers) are
  **permanently pinned at load** (their bytes ride inside the layer slot; keep a dedicated
  small pinned arena so layer-slot eviction never drops norms).
- Backpressure: if `ram_budget` cannot hold `slot_size` (383.87 MB) plus the in-flight block,
  `prefetch` throws at manifest-parse time, not mid-decode.

---

## 4. Name mapping: every 27B tensor → engine name → dtype → consumer

### 4.1 Prefix rules (checkpoint → engine; engine strings from `src/decode.cu` / `src/qwen35.cu`)

| checkpoint | engine builds (today) | change needed |
|---|---|---|
| `model.language_model.layers.N.*` | `language_model.model.layers.N.*` | swap components (all `decode.cu:48,122,127` sites) |
| `model.language_model.embed_tokens.weight` | `language_model.model.embed_tokens(.weight/.scales)` | drop `.scales`; bf16 |
| `model.language_model.norm.weight` | `language_model.model.norm.weight` | name-only ✓ |
| `lm_head.weight` | `language_model.lm_head(.weight/.scales)` | drop `.scales`; bf16 |
| `mtp.*` | `language_model.mtp.*` | prefix trim; `.fc` loses `.scales` |
| `X.weight` F8 + `X.weight_scale_inv` BF16 | `X.weight` u32 + `X.scales` u8/f16 | `matrix()` FP8 branch (§4.3) |
| `model.visual.*` (332 tensors, 0.86 GiB) | — | indexed, flagged `skip_vision`, never staged |

### 4.2 Full per-tensor map (26 families; every 27B tensor is covered)

**Global**

| checkpoint tensor | dtype, shape | engine lookup | consumer kernel family |
|---|---|---|---|
| `model.language_model.embed_tokens.weight` | BF16 [248320,5120] | `language_model.model.embed_tokens` | **bf16 row-gather**: `bf16_get_row` exists (`src/fp8.cu:192`); needs a T-row batch gather for prefill |
| `model.language_model.norm.weight` | BF16 [5120] | `language_model.model.norm.weight` | `rmsnorm_bf16` ✓ dtype already bf16 |
| `lm_head.weight` | BF16 [248320,5120] | `language_model.lm_head` | **bf16 GEMV** decode (`bf16_gemv`, `qwen_kernels.cu:68`) + bf16 GEMM prefill (new; trivial wmma) — VRAM-resident mandatory |

**Linear-attention layer N (48 layers, `i%4 != 3`)** — prefix `model.language_model.layers.N.`

| tensor | dtype, shape | engine base | consumer |
|---|---|---|---|
| `input_layernorm.weight` | BF16 [5120] | `p+".input_layernorm.weight"` | `rmsnorm_bf16` |
| `linear_attn.in_proj_qkv.weight` **+** `.weight_scale_inv` | F8 [10240,5120] + BF16 [80,40] | `p+".linear_attn.in_proj_qkv"` | `fp8_gemv/fp8_gemv2/fp8_gemm` |
| `linear_attn.in_proj_z.weight` + scale | F8 [6144,5120] + BF16 [48,40] | `.in_proj_z` | fp8 |
| `linear_attn.in_proj_a.weight` | **BF16** [48,5120] | `.in_proj_a` | **bf16 GEMV** (was MXFP4 `ab2` path, `decode.cu:67-73`) |
| `linear_attn.in_proj_b.weight` | **BF16** [48,5120] | `.in_proj_b` | bf16 GEMV |
| `linear_attn.conv1d.weight` | BF16 [10240,1,4] | `.conv1d` | `causal_conv4_silu` n 8192→**10240**; flat layout `[c][0][k] == c*4+k` identical to 9B `[8192,4,1]` (`qwen_kernels.cu` conv4 reads `w[i*4+k]`) — shape-only change |
| `linear_attn.A_log` | **BF16** [48] | `.A_log` | **BROKEN consumer**: `deltanet_parameters(a,b,(const float*)A.data,...)` (`decode.cu:124`, batch `:78`) reinterprets bf16 as f32 (9B A_log was f32 [32]) → garbage. Add bf16 variant of the `params` kernel (dt path already reads u16 bf16); n 32→48 |
| `linear_attn.dt_bias` | BF16 [48] | `.dt_bias` | bf16 ✓ already |
| `linear_attn.norm.weight` | BF16 [128] | `.norm.weight` | `gated_rmsnorm_bf16` 32→**48** heads (`decode.cu:124`) |
| `linear_attn.out_proj.weight` + scale | F8 [5120,6144] + BF16 [40,48] | `.out_proj` | fp8 |
| `mlp.gate_proj.weight` + scale | F8 [17408,5120] + BF16 [136,40] | `.mlp.gate_proj` | fp8 |
| `mlp.up_proj.weight` + scale | F8 [17408,5120] + BF16 [136,40] | `.mlp.up_proj` | fp8 |
| `mlp.down_proj.weight` + scale | F8 [5120,17408] + BF16 [40,136] | `.mlp.down_proj` | fp8 |
| `post_attention_layernorm.weight` | BF16 [5120] | `p+".post_attention_layernorm.weight"` | `rmsnorm_bf16` |

**Full-attention layer N (16 layers, `(i&3)==3`)** — plus the shared layernorms+MLP rows above

| tensor | dtype, shape | engine base | consumer |
|---|---|---|---|
| `self_attn.q_proj.weight` + scale | F8 [12288,5120] + BF16 [96,40] | `.self_attn.q_proj` | fp8; **split offsets move**: q@0, gate@6144 (9B gate@4096; `decode.cu:127` `split_q_gate` reads halves) |
| `self_attn.k_proj.weight` + scale | F8 [1024,5120] + BF16 [8,40] | `.self_attn.k_proj` | fp8 |
| `self_attn.v_proj.weight` + scale | F8 [1024,5120] + BF16 [8,40] | `.self_attn.v_proj` | fp8 |
| `self_attn.o_proj.weight` + scale | F8 [5120,6144] + BF16 [40,48] | `.self_attn.o_proj` | fp8 |
| `self_attn.q_norm.weight` | BF16 [256] | `.self_attn.q_norm.weight` | `qk_norm_rope` ✓ |
| `self_attn.k_norm.weight` | BF16 [256] | `.self_attn.k_norm.weight` | ✓ |

**MTP (`mtp.safetensors`; `mtp_use_dedicated_embeddings: false` — shares embed/lm_head)**

| tensor | dtype, shape | engine base | consumer |
|---|---|---|---|
| `mtp.fc.weight` | **BF16** [5120,10240] | `language_model.mtp.fc` | `bf16_gemv(w,x,y, rows=5120, cols=10240)` — **orientation ✓** (see 4.4); current call hardcodes 4096,8192 (`decode.cu:150-151`) |
| `mtp.pre_fc_norm_embedding.weight` | BF16 [5120] | `language_model.mtp.pre_fc_norm_embedding.weight` | `rmsnorm_bf16` |
| `mtp.pre_fc_norm_hidden.weight` | BF16 [5120] | `language_model.mtp.pre_fc_norm_hidden.weight` | `rmsnorm_bf16` |
| `mtp.norm.weight` | BF16 [5120] | `language_model.mtp.norm.weight` | `rmsnorm_bf16` |
| `mtp.layers.0.*` (18 tensors) | = full-attn template | `language_model.mtp.layers.0.*` | same consumers |

**Vision**: `model.visual.*` — 332 tensors, all BF16, 921,460,192 B. Indexed + `skip_vision`;
never acquired, never staged, excluded from every budget.

### 4.3 Names the current engine looks up that DO NOT exist in the 27B (all throw)

1. **All 407 `.scales` lookups.** `matrix()` builds `base+".scales"` unconditionally
   (`src/qwen35.cu:7`, release at `:11`). In the 27B no tensor is named `*.scales` — the
   companion is `X.weight_scale_inv`. Every linear would throw `tensor not found`.
2. **`language_model.mtp.fc.scales` — latent bug that exists TODAY on the 9B MLX index too.**
   Parsed both 9B indexes: `qwen35.insignia-index` has `mtp.fc.weight` (bf16, 64 MiB) and **no**
   `mtp.fc.scales`; only the INSIG4 index has fc scales. So `matrix("language_model.mtp.fc")`
   throws on the MLX 9B checkpoint before the `bf16_gemv` else-branch at `decode.cu:151` can
   run — that bf16 branch is dead code today. Fix = make `.scales` conditional (below).
3. **`A_log` dtype** — name exists, consumer breaks (bf16 read as f32, §4.2).
4. **Prefix mismatch** — every `language_model.model.layers.*`, `language_model.model.embed_tokens`,
   `language_model.lm_head`, `language_model.mtp.*` lookup misses because the checkpoint says
   `model.language_model.*` / bare `lm_head` / bare `mtp.*`.

**`matrix()` patch (pseudocode)** — `QuantMatrix` gains a kind enum; bf16 matrices carry no scale:

```cpp
struct QuantMatrix { DeviceView weight, scales; int rows{}, cols{};
                     enum Kind { mxfp4_mlx, mxfp4_i4, fp8, bf16 } kind; };

QuantMatrix Qwen35Weights::matrix(const std::string &base) {
    auto w = storage_.acquire(base + ".weight");
    if (w.dtype == DType::u32) {                       // 9B MXFP4 family, unchanged
        auto s = storage_.acquire(base + ".scales");   // (still required there)
        ... existing checks (qwen35.cu:7-9), cols = shape[1]*8 ...
        return {w, s, rows, cols, s.dtype==DType::f16 ? mxfp4_i4 : mxfp4_mlx};
    }
    if (w.dtype == DType::f8_e4m3) {                   // 27B FP8 path
        auto s = storage_.acquire(base + ".weight_scale_inv");
        rows = shape[0]; cols = shape[1];              // NO *8: 1 byte/elem, row-major
        assert(s.dtype == DType::bf16);
        assert(s.shape == {ceil(rows/128), ceil(cols/128)});   // census-verified invariant
        return {w, s, rows, cols, fp8};
    }
    if (w.dtype == DType::bf16)                        // mtp.fc, in_proj_a/b
        return {w, {}, shape[0], shape[1], bf16};
    throw ...;
}
```

`linear()/linear2()/linear_batch()` (`decode.cu:31-41`) dispatch on `kind`: fp8 → the existing
`fp8_gemv`/`fp8_gemv2`/`fp8_gemm` (`src/fp8.cu` — kernels already exist and pass a
double-reference cosine test in `src/test_fp8.cu` at exactly 27B shapes 10240×5120); bf16 →
`bf16_gemv`/new bf16 GEMM; mxfp4 → unchanged 9B path.

### 4.4 Orientation convention — verified for every GEMV consumer

All engine GEMV kernels treat the weight as row-major `[rows=out, cols=in]`,
`y[row] = Σ_c w[row*cols+c]·x[c]`:

- `fp8_gemv` (`fp8.cu:27`): `row_w = weights + row*cols` ✓; `fp8_gemv2` same (`fp8.cu:71`);
  `fp8_gemm` B-operand loads `weights[(n0+n)*cols + k]` (`fp8.cu:129`) ✓;
- `bf16_gemv` (`qwen_kernels.cu:67`): `w[row*cols+i]` ✓; mxfp4 family identical.

PyTorch `Linear.weight` is `[out, in]` row-major, and **every** census shape is consistent
with out-major rows:

| tensor | shape [out, in] | out is | engine call |
|---|---|---|---|
| `q_proj` | [12288, 5120] | 24 heads×256 q **+ 24×256 gate** | rows=12288 ✓ |
| `k/v_proj` | [1024, 5120] | 4 KV heads×256 | rows=1024 ✓ |
| `o_proj` | [5120, 6144] | hidden ← attn 24×256 | rows=5120, cols=6144 ✓ |
| `in_proj_qkv` | [10240, 5120] | q2048 \| k2048 \| v6144 | rows=10240 ✓ — **but the segment offsets in `decode.cu:124` are hardcoded `qkv+0/+2048/+4096`; 27B needs `+0/+2048/+6144`** |
| `in_proj_z` | [6144, 5120] | z gate | ✓ |
| `out_proj` | [5120, 6144] | hidden ← 48 v-heads×128 | ✓ |
| `gate/up_proj` | [17408, 5120] | intermediate | ✓ |
| `down_proj` | [5120, 17408] | hidden ← intermediate | ✓ |
| `mtp.fc` | **[5120, 10240]** | hidden ← concat(embed 5120, hidden 5120) | `bf16_gemv(w,x,y, 5120, 10240)` ✓ — 9B fc was [4096,8192], same convention |

No orientation trap anywhere. The 27B-only hazards are **hardcoded dims**, not layout:
`decode.cu:150-151` (fc 4096/8192), `:147` (`concat(...,4096)` → 5120), `:124` (qkv split
offsets, `di*32*128*128` state stride → 48×48, conv n 8192→10240, a/b 32→48, gated norm
32→48), workspace allocs `decode.cu:14-27` (qkv 8192→10240, gate/up 12288→17408,
delta_state 24×32×128×128→48×48×128×128, conv_state 8192→10240, kv 8→16 full-attn slots).
These belong to `Qwen35Shape` (hidden 4096→5120, inter 12288→17408, layers 32→64 —
`include/insignia_qwen35.hpp:7`) and are inventoried in `audits/w2/shape-constants.md`.

---

## 5. `weight_scale_inv` semantics — verdict: MULTIPLY (DeepSeek-style), not inverted

`W_dequant = fp8(W) × scale_inv`, where `scale_inv = amax_block / 448` (448 = e4m3 max).
The name means "inverse of the quantization divisor", i.e. it IS the dequant multiplier.
Evidence, three independent sources:

1. **Local TRT-LLM clone** — `TensorRT-LLM/tensorrt_llm/quantization/utils/fp8_matrix_weight_dequant.py`:
   `return weight_fp8.to(dtype) * scale_expanded.to(dtype)` with
   `weight_scale_inv` `[ceil(N/128), ceil(K/128)]` expanded by `repeat_interleave(128)` —
   pure multiplication. Also
   `tensorrt_llm/_torch/auto_deploy/custom_ops/quantization/torch_quant.py:680-702`:
   "`weight_scale[0] = weight_scale_inv (per-block weight scale)`" fed straight into the
   block-fp8 matmul; and `modeling_deepseek.py:811` warns that ignoring it makes scores
   "~1000x too large" — a multiply-only magnitude correction.
2. **Local vLLM clone** — `vllm/vllm/model_executor/layers/quantization/utils/fp8_utils.py:112,197`:
   `scale = absmax / fp8_max` (fp8_max=448 for e4m3), `y_q = y / scale`. Quantize divides,
   dequant multiplies — the stored scale is the multiplier.
3. **This repo's own kernels already assume it** — `src/fp8.cu:46` `acc = fmaf(part, sc, acc)`
   (scale multiplies the block partial), and `src/test_fp8.cu:60-68` fabricates
   `sc = amax/448`, `wref = e4m3(code) × sc` and passes cosine vs a double reference.

**What the kernel must do**: exactly what `fp8.cu` does — accumulate raw e4m3 products per
128-column block, then multiply that block's partial by `scales[row>>7][col>>7]` and sum. No
division, no reciprocal anywhere.

---

## 6. Scale tensor shape and indexing — consistent, verified

- Checkpoint convention (re-verified programmatically this session, 407/407): PyTorch Linear
  weight `[out, in]` → `weight_scale_inv` `[ceil(out/128), ceil(in/128)]` — **first dim is
  rows = out**, row-major. Concretely: q_proj [12288,5120] → [96,40]; in_proj_qkv [10240,5120]
  → [80,40]; gate/up [17408,5120] → [136,40]; down [5120,17408] → [40,136]; out/o_proj
  [5120,6144] → [40,48]; k/v [1024,5120] → [8,40]; in_proj_z [6144,5120] → [48,40].
- Every F8 rows/cols pair in this checkpoint is an exact multiple of 128 (5120=40×128,
  17408=136×128, 12288, 10240, 6144, 1024) — ceil never rounds up, so
  `scale bytes = (rows/128)*(cols/128)*2` exactly, and the kernels' `cols%128==0` guard
  (`fp8.cu:53,97,182`) always passes.
- Engine kernels index `scales[row>>7][col>>7]` with out-major rows:
  `fp8_gemv`/`fp8_gemv2`: `row_s = scales + (row>>7)*kblocks; row_s[c0>>7]` with
  `kblocks = cols>>7` (`fp8.cu:28,72`); `fp8_gemm` dequant:
  `scales[((n0+n)>>7)*kblocks + (kb>>1)]` where kb steps 64 columns → `(kb*64)>>7 = kb>>1`
  (`fp8.cu:136`). **Consistent with the census — no transpose fix needed.** Scale dtype is
  read as bf16 (`uint16_t` reinterpreted as `__nv_bfloat16`, `fp8.cu:33`) matching the
  checkpoint's BF16 scales exactly.

---

## 7. Robustness: crc32, alignment, and the read plan

### 7.1 CRC32: skip at runtime

`crc32.txt` covers all 66 shards and was fully matched in the week-2 audit (§3 there). The
INSIDX02 **builder** verifies each shard while indexing (index build already streams headers
cheaply; adding a crc pass there is ~30-60 s once). Runtime re-hash of 28.75 GiB per load
would cost the same 30-60 s for zero protection against anything but disk rot — and O_DIRECT
reads bypass the page cache anyway, so there's no stale-cache hazard the CRC would catch.
**Decision: CRC at index build only; store the value in the shard entry for provenance.**

### 7.2 Alignment reality (verified): 45/1606 tensor starts 4096-aligned; all F8 sizes are
4096-multiples; shards are gapless and end flush; every shard has exactly one phase:

| shard class | count | data_start | F8 phase (mod 4096) |
|---|---|---|---|
| linear, 1-digit idx | 8 | 2,600 | 616 |
| linear, 2-digit idx | 40 | 2,624 | 640 |
| full-attn, 1-digit idx | 2 | 2,320 | 3,728 |
| full-attn, 2-digit idx | 14 | 2,336 | 3,744 |
| mtp | 1 | 2,480 | 1,840 |
| outside | 1 | 38,080 | no F8; BF16 bigs at 1,216 |

`ReadFile(FILE_FLAG_NO_BUFFERING)` needs offset, length, and destination all sector(4096 for
SSD-comfort)-aligned. Raw tensor offsets cannot be handed to it directly.

### 7.3 Read plan: SHARD-MAJOR staging (recommended over per-tensor)

The merge question dissolves because **one layer shard is exactly one layer** (verified: all
20 linear-layer tensors of layer N live in `layers-N.safetensors`, gapless, in two groups —
BF16 smalls alphabetically, then F8 bigs alphabetically). Therefore:

- **Slot granularity = layer shard.** A slot is `align_base = data_start & ~4095` through
  `file_size` ceil-rounded to 4096. Slot sizes: 383.87 MB (linear), 372.31 MB (full-attn),
  477.2 MB (mtp = fc 100 MB bf16 + one full-attn layer). Budget math on 15.9 GiB RAM:
  ~10 GiB pinnable → **26 layer slots**, which is exactly the RAM tier of the L=21/M=23/N=21
  placement (21 resident + 3-5 in-flight/eviction slack) from the synthesis feasibility table.
- **Per-tensor alignment handling disappears entirely.** Reading the whole aligned span means
  the only unaligned bytes ever read are the single ≤4,095 B header prefix (the phase) and
  the ≤4,095 B tail. Every tensor is then exposed at `slot + in_slot_off` with `in_slot_off =
  abs_off - align_base` precomputed in the index (§2.2). No head/tail splitting per tensor,
  no per-tensor window table, no special-casing the 5 phase classes.
- **Read plan builder** (per slot fill): `nblocks = ceil(payload_span / 2 MB)` descriptors
  `(shard_handle, file_off=align_base + i*2MB, len=min(2MB, end-file_off), dst=pinned +
  i*2MB)`; length of the first/last block is naturally 4096-multiple because align_base,
  2 MB, and ceil(file_size) all are. Issued at QD 8 on the io thread; ~192 blocks/shard →
  ~57 ms at 6.8 GB/s, fully hidden one layer ahead.
- **Per-tensor windows** (w2 §5.5's original design) would work — the F8 blocks are each
  4096-multiple-sized with a shared phase — but cost 407 window records, per-window head
  handling, and per-tensor acquire granularity that decode never uses (decode touches every
  tensor of a layer every token, in order). Sub-layer granularity buys nothing for
  token-serial decode and layer-serial prefill.
- **outside.safetensors is the exception** — it is NOT one layer (6 GB: lm_head + embed +
  norm + vision). Split into three named windows, each a single aligned span:
  `lm_head` [38,080 … 2,542,834,880), `embed` [… 5,085,631,680), `norm+vision`
  (never staged beyond the 10 KB norm). lm_head and embed are each single tensors, so their
  windows are per-tensor by construction: `win_off = off & ~4095`,
  `win_len = ceil((off & 4095 + bytes)/4096)*4096`, tensor at `slot + (off & 4095)`.
- Eviction is whole-slot (atomic layer in/out) — matches how the decoder consumes layers.

---

## 8. Consolidated patch plan (ordered, dependency-aware)

1. **DType** — add `f8_e4m3=7, f8_e5m2=8` (`insignia_model.hpp:10`) + `"F8_E4M3": 7` in the
   builder's `DTYPES` (`index_safetensors.py:8`). Trivial, unblocks everything.
2. **INSIDX02 builder** — `tools/index_safetensors.py`: directory mode, shard table (with
   `align_base`, crc-at-build, relative paths, `skip_vision` flag), tensor table with
   `shard/scale_idx/in_slot_off`, name-sorted, per-shard bounds + orphan-scale checks (§2.2).
3. **ModelFile v2** — multi-shard: per-shard mapping array + `shard_base(u16)`, `TensorView`
   gains `shard/off/scale_idx`, per-shard escape check replaces `model_file.cpp:36`, open-all
   66 (O_DIRECT+OVERLAPPED handle set for the reader, optional mapped set), `find()` unchanged,
   new `find_linked(view)` for scales (replaces the `base+".scales"` string convention).
   `test_model.cpp:4` loses the 699/`.scales` assertions.
4. **matrix()/QuantMatrix** — kind enum `{mxfp4_mlx, mxfp4_i4, fp8, bf16}`, FP8 branch
   (`weight_scale_inv`, shape check `[ceil(r/128),ceil(c/128)]`), bf16-no-scale branch
   (fixes the live `mtp.fc` latent throw), keep MXFP4 for the 9B (§4.3).
5. **Wire fp8.cu into decode** — `linear/linear2/linear_batch` dispatch fp8 → `fp8_gemv/
   fp8_gemv2/fp8_gemm` (already written + cosine-tested); bf16 branch for `in_proj_a/b`
   (drop the mxfp4 `ab2` calls `decode.cu:67-73`), embed via `bf16_get_row` + batch gather,
   lm_head bf16 GEMV/GEMM.
6. **Name remap + shape constants** — engine prefixes → `model.language_model.*`/`lm_head`/
   `mtp.*`; `Qwen35Shape` 5120/17408/64; the hardcoded offsets/dims in `decode.cu`
   (qkv split +0/+2048/+6144, fc 5120/10240, concat 5120, conv 10240, a/b 48, gated norm 48,
   state 48×48×128×128, kv slots 16) per §4.4 and `audits/w2/shape-constants.md`.
7. **A_log bf16** — bf16 variant of `deltanet_parameters`/`params` (`decode.cu:78,124`,
   `qwen_kernels.cu` params kernel) or upconvert bf16→f32 once at load into the pinned
   small-params arena (cheaper; loader-side fix).
8. **TieredStorage2** — staging ring + IOCP reader + placement manifest + prefetch/
   acquire_blocking/release with the §3.3 threading contract; mmap demoted to warmup/parity.
   Shard-major slots for layers/mtp; three windows for outside; vision never staged.
9. **Parity gate** — layer-0 DeltaNet vs `tools/reference_*.py` at the new dims (cosine
   ≥ 0.999999 target as on 9B) before any performance work, per AGENTS.md.

*Engine sources untouched by this audit; report only.*
