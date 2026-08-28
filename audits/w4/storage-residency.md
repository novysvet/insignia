# w4 — Storage & residency: TieredStorage2 spec for the 27B (V/Z/C/N static placement)

Date 2026-08-25. Mission: map every behavior of the 9B residency layer onto 27B needs and
produce the TieredStorage2 API spec — shard-granular slots, placement manifest as data,
static baked pointers, pinned/locked arenas, UVA access, startup probe ladders and loader
order, no-eviction verification, the embed NVMe row-pread service, and the 9B→27B
dual-mode migration. Read-only audit; the only file written is this report.

Sources read firsthand (all line refs current): `src/storage.cu` (12 lines),
`include/insignia_storage.hpp` (33), `include/insignia_model.hpp` (42), `src/model_file.cpp`
(45), `src/qwen35.cu` (34), `include/insignia_qwen35.hpp` (22), `src/decode.cu` (268),
`src/generate.cu` (211), `src/streaming.cu` + `include/insignia_streaming.hpp` (555/249),
`include/insignia_cpu.hpp` (CpuPool 217-336, fp8_gemv family 348-480),
`tools/index_safetensors.py` (INSIDX01), `tools/index27.py` (INSIDX02 as built),
`audits/w3/MASTER-PLAN.md` (§2, Phase D.3), `audits/w3/loader-gaps.md` (§2-§3, §7),
`audits/w3/pcie-pipeline.md` (§6-§7), `audits/w3/nvme-reader.md`, `audits/w3/embed-lmhead.md`
(§7-§8), `audits/w3/io-bench-results.md`, sibling w4 reports `tier-dispatch.md`,
`streaming.md`, `index-loader.md`. Ground numbers (MASTER-PLAN §1): layer 383.86 MB (lin) /
372.31 MB (full); lm_head 2,542.80 MB; MTP shard 477.20 MB; embed 2,542.80 MB (NVMe);
VRAM budget 11,300 MB; pinned cap 8,531 MB hard / 8,048 planned; usable RAM 13,500 MB;
E: 3.3 GB/s ⇒ 115.4 ms/layer; UVA 15-20 GB/s (plan 18) ⇒ 21.2 ms/layer; CPU 37 GB/s ⇒
10.8 ms/layer.

---

## 0. Verdict in one paragraph

The 9B `TieredStorage` is a **tensor-granular device LRU with on-demand mmap upload**
(`storage.cu:8-9`): unordered_map of name→{cudaMalloc'd copy, pins, tick}, budget-driven
eviction of unpinned entries, a `cudaStreamSynchronize` per cold acquire, and the NVMe tier
existing only implicitly as 4 KB page faults through `MapViewOfFile`
(`model_file.cpp:30-31`). Every one of those behaviors is wrong at 27B — per-tensor
`cudaMalloc`+sync×20 tensors×64 layers per token, LRU churn of 383 MB tensors against a
6 GB budget, and page-fault paging of a 30 GB working set on a 15.9 GB box. TieredStorage2
is a **new class, not an edit**: placement is a manifest resolved once at startup into
static pointers (VRAM slabs / pinned-UVA / VirtualLock'd host / NVMe feeder plans); there is
**no eviction, no per-token acquire/release, no name lookups, no cudaMalloc and no
cudaStreamSynchronize anywhere on the decode path**; the only dynamic tier (N) is the
already-smoke-verified `LayerFeeder` ring (`streaming.cu:373-456`). The 9B class, its
callers (`decode.cu`, `qwen35.cu`, 9 dump/test tools) and the INSIDX01 firewall
(`model_file.cpp:23` throws on INSIDX02) stay byte-identical.

---

## 1. What storage.cu is today — behavior-by-behavior 27B verdict

| # | 9B behavior | where (file:line) | 27B verdict |
|---|---|---|---|
| 1 | Per-tensor acquire by name; `std::string` construction + hash per call | `storage.cu:9`, `insignia_storage.hpp:21` | **KILL.** Decode touches ~20 names/layer × 64 layers × every token (hot loop at `decode.cu:126-133` builds the strings too). Replaced by a baked per-layer pointer table (§6). |
| 2 | Cold acquire = `cudaMalloc` + `cudaMemcpyAsync` **from the mmap** + `cudaStreamSynchronize` | `storage.cu:9` | **KILL.** Static startup load through explicit O_DIRECT reads (NvmeReader plans / `read_once`); a per-tensor host sync on the hot path is 1000+ syncs/token. |
| 3 | `make_room` LRU eviction of unpinned device tensors, victim = min `tick` | `storage.cu:8` | **KILL — no eviction at 27B.** Static placement; any evict path is a correctness hazard (pointers are baked into the step). Verified absent from the new design (§5). |
| 4 | Pin counting (`pins++`/`pins--`) to protect tensors from eviction | `storage.cu:9-10` | **KILL.** Lifetime = process; release() semantics shrink to feeder slot release for N only. |
| 5 | Single device budget (`budget_`, `used_`) | `storage.cu:6-8`, `insignia_storage.hpp:24-25` | Replace with three ledgers (VRAM / pinned / locked) + refuse-don't-degrade at startup (§4.4). |
| 6 | NVMe tier = demand paging through `MapViewOfFile` (`TensorView.data = base_+off`) | `model_file.cpp:30-35`, consumed at `storage.cu:9` | **KILL for serving.** 4 KB faults × double-buffered page cache on a 30 GB working set. mmap survives only for parity/dump tooling (`loader-gaps.md` §3.2). |
| 7 | `MemoryTier { nvme_mapped, host_pinned, device }` declared, `host_pinned` unimplemented | `insignia_storage.hpp:11` | Realized as Tier V/Z/C/N (below); never shipped in v1 form. |
| 8 | `DeviceView{data,bytes,dtype,shape}` returned per acquire | `insignia_storage.hpp:13` | Shape/dtype move to the baked table (static); hot path carries pointers only. |
| 9 | mmap'd single-file `ModelFile` (INSIDX01, one absolute HF-cache path) | `model_file.cpp:17-39` | ModelFile v2 over INSIDX02 (66 shards, absolute offsets, crc-at-build) — prerequisite, spec'd in sibling `w4/index-loader.md`; `index27.py` already emits it. |
| 10 | CUDA graphs over acquire'd pointers | `decode.cu:238-265` (`capture_step`/`capture_spec`) | Not called by the 27B driver at all (MASTER-PLAN §2.4: no graphs at 27B; graph-hazards §6c). Keeps working for 9B. |

**Dual-mode firewall (already free):** `model_file.cpp:23` hard-checks
`memcmp(magic,"INSIDX01") && version==1`, so a 27B index cannot be fed to the 9B stack —
the accidental "run 27B through TieredStorage" path (which would LRU-thrash 383 MB tensors
against a 6 GB budget at `generate.cu:117`'s `6ull<<30`) is structurally impossible as long
as ModelFile v1 stays INSIDX01-locked. Keep that check verbatim.

---

## 2. TieredStorage2 — full API spec

New files: `include/insignia_storage2.hpp` + `src/storage2.cu` (compiled into the ENGINE27
target only; no 9B file includes it). Prerequisite: ModelFile v2 (`find()` unchanged;
`TensorView2{shard u16, off u64 (absolute), bytes, dtype, shape[]}`; per-shard
`{rel_path, file_bytes, crc}` — exactly what `tools/index27.py:236-251` emits).

### 2.1 Tiers, manifest, rules

```cpp
// include/insignia_storage2.hpp
#include "insignia_model.hpp"          // v2
#include "insignia_streaming.hpp"      // NvmeReader/PinnedRing/LayerFeeder
#include <cuda_runtime.h>

namespace insignia {

enum class Tier : uint8_t { V=0, Z=1, C=2, N=3 };   // VRAM / pinned-UVA / locked-CPU / NVMe-stream

// ---- placement manifest: DATA, not code (AGENTS "modify constant data directly").
// v1 / v1.5 / v2 are three files; zero engine changes between them (MASTER-PLAN §2.4).
struct PlacementRule {                 // 6 B packed; layer range -> tier
    uint16_t layer_lo, layer_hi;       // inclusive, 0..63
    Tier      tier;
} __attribute__((packed));

enum class Special : uint8_t { lm_head=0, mtp=1, smalls=2, embed=3 };
struct SpecialRule { Special what; Tier tier; };   // lm_head/mtp MUST be V (checked; refuse otherwise)

// Manifest file INSIGM01 (binary, ~64 B): magic[8]; u8 version=1; u8 n_rules; u8 n_specials;
// PlacementRule[n_rules]; SpecialRule[n_specials]; char comment[32]. Loader requirements:
//   - rules must tile 0..63 exactly (no gap, no double-assign) -> else throw at parse;
//   - specials: lm_head=V, mtp=V enforced; smalls=Z (pinned arena) the only legal value
//     today; embed=3(N) the only legal value (embed is never resident, MASTER-PLAN App. A.2);
//   - optional u16 mirror_shard[66] drive-map tail (OPTION-G hook, ~50 LOC, off by default).
```

Manifest examples (one file each; engine identical):

```
v1        all-stream: rules {0-18:V, 19-63:N}                         Z=0  C=0  N=45
v1.5      pin-19:     rules {0-18:V, 19-37:Z, 38-63:N}                Z=19 C=0  N=26
v2        cpu-tier:   rules {0-18:V, Z/C interleave 19-63 per tier-dispatch §0}  Z=21 C=9 N=15
```

### 2.2 The class

```cpp
struct ProbeResults {                  // §4.1; filled by run_probes() BEFORE any allocation
    uint64_t vram_free, vram_budget;   // cudaMemGetInfo minus guard
    uint64_t pinned_cap_measured;      // 1 GiB ladder result, hard cap 8,531 MB clamp
    uint64_t pinned_plan;              // min(measured - 1 GiB guard, 8,048 MB)
    double   uva_gbps;                 // 200 ms microbench (15-20 expected)
    uint64_t ram_avail;                // GlobalMemoryStatusEx.availPhys at probe time
};

class TieredStorage2 final {
public:
    TieredStorage2(const ModelFile2& mf, cudaStream_t compute, cudaStream_t copy);
    ~TieredStorage2();                  // reverse order of load(); syncs streams first (safety C5)
    TieredStorage2(const TieredStorage2&) = delete;
    TieredStorage2& operator=(const TieredStorage2&) = delete;

    // ---- 1. manifest + probes -------------------------------------------------
    void load_manifest(const wchar_t* path);          // parse + tile check; throws on bad data
    ProbeResults run_probes();                        // VRAM free, pinned ladder, UVA bench
    // Fit check against the manifest's demand (computed from the shard table):
    //   VRAM: fixed(lm_head 2542.8 + mtp 477.2 + mtp_kv 8.4 + norm + workspace 250 + ctx 400)
    //         + sum(layer bytes of V rules)  <= vram_budget      (L=19 -> 11,008 of 11,300)
    //   pinned: sum(Z layers) + smalls arena ~67 MB + embed rows ~1 MB + pinned mirrors ~2 MB
    //         (+ registered ring in v1/v1.5: K*384 MB)  <= pinned_plan
    //   locked: sum(C layers) + ring (v2: ring is NOT registered) + host KV 108 MB
    //         + delta/conv states 108 MB  <= ram_avail - 250 MB margin
    // Returns false + fills `why[128]` with the first violated ledger -> caller (generate27)
    // either refuses to start or runs the degrade ladder (Z->C->N shift, ONE retry). Never
    // silently degrades, never evicts.
    bool fits(const ProbeResults&, char* why) const;

    // ---- 2. startup load (ordered, fail-fast; §4.2) ----------------------------
    void load();                                       // throws with tier+layer in the message

    // ---- 3. baked tables for the hot path (§6) ---------------------------------
    const struct WTable& table() const noexcept;       // 64 x WLayer, cache-line aligned
    Tier tier_of(int layer) const noexcept;            // tier_of_[64], one load+scale
    uint64_t vram_bytes()/pinned_bytes()/locked_bytes() const noexcept;   // ledgers, post-load

    // ---- 4. N-tier streaming (driven by generate27, NOT by this class) ---------
    // build_epoch_plans() maps N layers to LayerFeeder ReadPlans in layer order:
    //   per layer: one request {path (drive-map override), align_down_4096(first tensor off),
    //   span to align_up_4096(shard end)} -> one contiguous slot; consumers use feeder.map().
    std::vector<ReadPlan> build_epoch_plans() const;
    LayerFeeder& feeder() noexcept;                    // owned; ring sized by manifest tier mix:
                                                       // v1/v1.5 ring registered (counts vs cap);
                                                       // v2 ring VirtualLock'd (PinnedRing fallback
                                                       // path is the DESIGNED mode, streaming.cu:305-312)

    // ---- 5. cold-path acquire (parity tools, prefill sweeps, R3 byte-equality) --
    // Semantics (loader-gaps §3.3 contract, static amendment): blocks until the layer's
    // slot is READY (feeder.acquire_layer under the hood for N; V/Z/C return immediately);
    // returns a HOST pointer valid until release. Exactly one consumer at a time per layer;
    // strictly sequential release (feeder contract, streaming.cu:443-450).
    const void* acquire_host_blocking(int layer);
    void        release_host(int layer) noexcept;

    // ---- 6. embed row service (§7) ----------------------------------------------
    void        embed_stage_async(int token, int slot); // buffered-twin 10,240 B pread, dbl-buffered
    const void* embed_row_wait(int slot);               // event wait; returns registered row ptr

    // ---- 7. smalls arena accessors (kernels + CPU tier share these) --------------
    const uint16_t* smalls(int layer, SmallId which) const noexcept;  // pinned arena, UVA-readable
    const float*    a_log_f32(int layer) const noexcept;              // pre-widened bf16->f32 (R4 risk #4)

private:
    // arenas (all freed in ~, reverse construction order):
    //   vram_slab_   : one cudaMalloc per V layer + lm_head + mtp + scales arena (~2.7 MB)
    //   z_arena_     : VirtualAlloc(Z_total) + filled + cudaHostRegister in <=1 GiB pieces
    //   c_arena_     : VirtualAlloc(C_total) + filled + VirtualLock per layer (+working-set raise)
    //   smalls_arena_: VirtualAlloc + fill + register (~67 MB: bf16 smalls 64x~1.05 MB)
    //   embed_rows_  : 4 x 10,240 B registered slots + HANDLE events
    //   feeder_      : LayerFeeder (ring only used by N)
    const ModelFile2& mf_;  cudaStream_t compute_, copy_;
    std::vector<PlacementRule> rules_;  std::vector<SpecialRule> specials_;
    Tier tier_of_[64];  struct WLayer* table_{};  // baked (§6)
    // ... ledgers, arenas, probes
};

} // namespace
```

### 2.3 Threading contract (normative; carried from loader-gaps §3.3, amended for static)

- **One io path**: NvmeReader's 2 IOCP workers (`streaming.cu:34-51`, LP 6-11 affinity) own
  all NVMe traffic — Z/C startup fills, N epochs, embed preads use the **buffered twin**
  (different handle, cache path, no IOCP contention).
- **`load()` runs on the main thread before any decode thread exists.** No locks needed
  post-load; the baked table is read-only forever after.
- **N-tier**: `begin_epoch`/`release_layer` from the sequencer T0 only (feeder's
  strictly-sequential contract, `streaming.cu:448`); `acquire_layer` may block T0 (that is
  the pacing wait — tier-dispatch §1.2).
- **`acquire_host_blocking`** (cold path only) may be called by tool threads; different
  layers are independent slots; a slot is filled by at most one reader (PinnedRing state
  machine, `insignia_streaming.hpp:160-186`).
- **No acquire/release exists for V/Z/C on the hot path.** Their pointers are baked at
  startup and never invalidated (no eviction ⇒ no stale-pointer class at all).

---

## 3. Startup loader order (spec, with refuse conditions)

Ordered list implemented by `load()`; each step either completes or throws with a precise
message; nothing partially started survives (§4.4 unwind). Byte sources: INSIDX02 gives
absolute offsets; per-layer window = `[align_down_4096(first_off), align_up_4096(shard_end))`
computed at load (index27.py does NOT precompute `in_slot_off`/`align_base` — unlike the
loader-gaps §2.2 sketch; the math is 3 lines here instead).

| # | step | bytes | dest | fail behavior |
|---|---|---|---|---|
| 1 | `run_probes()` — see §4.1 | 0 | — | none (measurement) |
| 2 | `fits()` gate against manifest | — | — | refuse with the violated ledger; generate27 may retry once with a degraded manifest (Z→C→N), never a silent partial |
| 3 | **lm_head + MTP → VRAM FIRST** (MASTER-PLAN C.5: "refuse to start if short") | 2,542.8 + 477.2 | `cudaMalloc` slabs, filled by `read_once`-style buffered/O_DIRECT reads of `outside.safetensors` lm_head window + `mtp.safetensors` | `cudaMalloc` fail ⇒ refuse (a model that cannot verify/draft is useless; do NOT donate lm_head bytes to layers) |
| 4 | **BF16 smalls permanent arena** (norms, conv1d, A_log, dt_bias, q/k norm; ~1.05 MB/lin, ~0.55 MB/full ≈ 65 MB) + **F8 scales arena → VRAM** (~2.7 MB) + A_log pre-widened to f32 (kills risk R4-#4 `decode.cu:82/124`-class bf16-as-f32) | ~67 MB host + ~3 MB VRAM | one VirtualAlloc + register; filled per-shard at step 6/7/8 time (they ride the same layer reads) or by tiny `read_once`s | register fail ⇒ refuse (kernels need UVA reads of norms) |
| 5 | **V layers (0..L-1) → VRAM slabs**: one `cudaMalloc`/layer; tensors uploaded **individually into 16 B-aligned slab offsets** (see alignment note) from a pinned staging slot filled by one reader plan/layer | 383.86/372.31 each | device slabs | `cudaMalloc` fail at layer i ⇒ refuse (budget mispredicted — probe under-reported) |
| 6 | **Z layers → pinned + registered**: `VirtualAlloc(Z_total)` (one reservation), fill per layer with one NvmeReader plan into the layer's 4096-aligned slice, `cudaHostRegister` in ≤1 GiB pieces after each GiB fills (WDDM >2 GB single-register risk; pcie-pipeline §7.3) | 21×~381 = 8,003 MB (v2) | pinned, UVA-visible (same pointer host+device) | register fail ⇒ degrade ladder ONCE (move trailing Z layers to C), else refuse |
| 7 | **C layers → locked pageable**: `VirtualAlloc`, fill via reader plans, `SetProcessWorkingSetSize` raise, `VirtualLock` per layer. **NOT `cudaHostRegister` — deliberately outside the pinned cap** (MASTER-PLAN §2.3 fact 2; only CPU kernels ever read them) | 9×381 = 3,429 MB (v2) | locked host | VirtualLock fail ⇒ same degrade ladder |
| 8 | **N layers → nothing resident**; `build_epoch_plans()` emits one ReadPlan/layer (whole-shard gapless prefix, `nvme-reader.md` §3 unit); feeder ring allocated per mode: v1/v1.5 registered (counts: 3×384 / 2×384 MB), v2 `VirtualLock`'d (the PinnedRing fallback path at `streaming.cu:305-312` becomes the designed mode, not an error) | ring 768-1,152 MB | — | plan exceeds slot ⇒ manifest bug (assert at load, `streaming.cu:389-391` already dies) |
| 9 | **embed row arena**: 4×10,240 B registered slots + events; open `outside.safetensors` buffered twin (embed absolute base = 38,080 + 2,542,796,800 = **2,542,834,880**; row r at +r×10,240) | ~41 KB | — | refuse |
| 10 | **Warm**: pre-fill both ring slots + touch every C page before token 1 (colibri cold-start lesson, pcie-pipeline §7.4 step 3); `begin_epoch` fires so the reader gets its head start while VRAM finishes lm_head/MTP tail work | — | — | reader fatal ⇒ refuse |

Ordering rationale: VRAM-first fail-fast (steps 3+5) because VRAM is the tightest ledger
(11,008/11,300 at L=19, 292 MB spare); Z before C because the pinned cap is the second
hardest wall and its probe is the least trustworthy (§4.1); N last because it allocates
nothing but the ring; smalls (4) before layers so Z/C/N fills can memcpy smalls into the
arena in the same pass (one read of each shard, two consumers).

**Alignment note (corrects MASTER-PLAN D.2's premise, confirms w4/streaming §0):** only the
8 one-digit linear shards (`data_start` 2,600 ⇒ F8 absolute offsets ≡ 8 mod 16) are
misaligned; all other shard classes have `data_start`/F8 offsets ≡ 0 mod 16 (2,624 / 2,320 /
2,336 / 2,480 / 38,080 are all 16-multiples). Those 8 shards are layers 0-7 — always inside
the V range in every manifest, and step 5's per-tensor slab upload rebases them for free.
For ring slots (N tier) keep the plan-builder pad-at-F8-boundary + the acquire-time assert
`(f8_base & 15) == 0` (R3 gate); for Z/C permanent copies, per-tensor 128 B rebasing at
startup is a free Phase G knob (one-time copy), v1 can ship with slot offsets + the
misalignment-free shard set (all N-tier candidates 19-63 are two-digit ⇒ aligned).

---

## 4. Probes, budgets, and OOM mid-load

### 4.1 Probe ladders (mission-mandated)

1. **Free VRAM**: create context (ctx itself eats ~400 MB — probe AFTER), `cudaMemGetInfo`
   → `vram_budget = min(free − 300 MB guard, 11,300 MB)`. Print; L=19 needs 11,008.
2. **Pinned-cap ladder**: `cudaHostAlloc(1 GiB)` in a loop until failure (or 8 steps), sum
   successes, `cudaFreeHost` all; `pinned_cap_measured = min(sum, 8,531 MB)`;
   `pinned_plan = min(measured − 1 GiB, 8,048 MB)`. Guard rails: watch
   `GlobalMemoryStatusEx.availPhys` each step (WDDM may degrade the system before returning
   null — pcie-pipeline §8 reports fragmentation as low as 700 MB on some boxes); run this
   ladder FIRST, before any other allocation, so a poisoned state can't strand memory.
3. **UVA rate microbench (200 ms)**: register 256 MiB, grid-stride read-reduce kernel until
   200 ms wall ⇒ GB/s (expect 15-20; plan 18). Feeds the Z-vs-C split: the startup solver
   (Phase G) moves layers Z→C until `21.2 ms × Z_count(uva_rate) ≤ 1.1 × NVMe floor`
   (MASTER-PLAN §3-G.1). v1 hardcodes the manifest; the bench still runs and logs.
4. **NVMe rate**: acceptance belongs to `io_bench` (Phase D.1, E: ≥ 3.0 GB/s); the solver
   consumes its number; `run_probes` may re-run a 1 s in-process sequential read as a
   sanity clamp.

### 4.2 Ledger arithmetic the fit check encodes (v2 numbers)

- VRAM: 3,678.4 fixed (lm_head 2,542.8 + MTP 477.2 + MTP-KV 8.4 + workspace ~250 + ctx 400)
  + V layers 7,329.8 = **11,008 ≤ 11,300** (292 spare).
- Pinned (v2): Z 8,003 + smalls 67 + embed rows ~0.05 + mirrors ~2 = **~8,072 ≤ 8,531 hard**
  (planned 8,048 — Z=21 sits 24 MB over *plan*, under *hard*; solver clamps to Z=20 if the
  ladder returns ≤ 8,050; v1.5: Z 7,238 + registered ring 1,152 = 8,390 ≤ 8,531 ✓).
- Locked (v2): C 3,429 + ring 768 (v2, unregistered) + host KV 100.7 + delta/conv 107.9 +
  buffers 150 = **4,556; total RAM = 8,003+4,556+OS-visible margin ≈ 12,806 ≤ 13,500** ✓.

### 4.3 Every `cudaHostRegister`'d byte counted (hard rule)

pinned = Z weights + smalls arena + embed row arena + `DecodeWorkspace` pinned mirrors
(`next_host`/`pos_host`/`host_committed`, decode.cu:16-18 — trivial) + registered ring
(v1/v1.5 only) + optional host-pinned spec snapshots at D=4 (628 MB, Phase F — must re-gate
the ladder). NOT counted: C weights, v2 ring, host KV, host delta/conv states — all
VirtualLock'd pageable, CPU-only consumers.

### 4.4 OOM mid-load = deterministic unwind, never eviction, never degrade-silent

- All sizes are computed from the manifest + shard table BEFORE the first allocation
  (`fits()`); a mid-load failure therefore means the probe lied (driver overhead, fragmentation,
  another process) — handled by unwinding in reverse construction order
  (unregister→unlock→VirtualFree→cudaFree, streams synced first), throwing with
  `{step, tier, layer index, requested MB, ledger MB}` in the message.
- **There is no `make_room` equivalent.** No code path in TieredStorage2 frees or moves a
  resident object after `load()` returns. The only memory that changes state post-startup is
  the feeder ring (by design, transient) and the embed row slots.
- The one-shot degrade ladder (Z→C→N shift + re-`fits()` + re-`load()`) lives in generate27,
  runs at most once, and is logged as a manifest override — the running engine never
  re-places anything.

---

## 5. Eviction hazard audit (mission item 2)

At 27B **no path evicts**. Verification against every mechanism that could:

1. `TieredStorage::make_room` (`storage.cu:8`) — the only eviction code in the tree. The 27B
   binary does not instantiate `TieredStorage` (new class; `Qwen35Weights`/`decode.cu` are
   9B-only, and INSIDX01 magic-check keeps the stacks disjoint — §1 firewall).
2. `LayerFeeder`/`PinnedRing` — ring slots are *recycled*, not evicted: a slot's data is
   consumed and released within one step by contract (`release_layer` strictly sequential,
   `streaming.cu:443-450`); no resident pointer ever aliases a ring slot (N-tier kernel args
   are rebuilt from `feeder.map()` each acquire — §6). The `try_claim` failure path
   (`streaming.cu:403`, window-invariant bug) sets fatal rather than dropping data.
3. CUDA graphs — none at 27B (`capture_step`/`capture_spec` uncalled), so the
   week-2 bug-6 class (graph baked pointers + eviction) is void.
4. WDDM VRAM overcommit — impossible by construction: V bytes are committed at load into
   explicit `cudaMalloc`s within the probed budget; the only later device allocations are the
   fixed `DecodeWorkspace` buffers (already inside the 250 MB workspace line) — nothing
   allocates per token.
5. Standby-list pressure on C pages — mitigated by VirtualLock (that is its entire job);
   the working-set quota raise is part of step 7.
6. Residual risk (honest): a future `CUDA graph` revival or a "helpful" retry that reuses
   9B `Qwen35Weights` for the 27B target would reintroduce eviction silently. Guard: keep
   the INSIDX01 check, and add one assert in `generate27` main that
   `typeid(weights) != typeid(Qwen35Weights)` is trivially true by not linking storage.cu
   into ENGINE27 at all (mk.py closure — build-system §4).

---

## 6. Decode hot path: the baked per-layer pointer table (mission item 4)

The 27B decode loop (tier-dispatch §7) consumes ONE precomputed table; per-token storage
work is: `switch(tier_of[l])` on a u8, pointer loads, and (N tier only) `acquire/release`
+ slot-relative fixups. Zero name lookups, zero cudaMalloc, zero syncs, zero locks.

```cpp
// One WLayer = 4 cache lines (alignas(64)); table = 64 x WLayer, built once in load().
enum class WLoc : uint8_t { dev=0, uva=1, host=2, slot=3 };  // where the pointer lives
enum class WKindX : uint8_t { f8=0, bf16=1, f8scale=2, bf16small=3, f32small=4 };

struct TagPtr {                    // 16 B: the "tag + void* pair" the mission asks for
    void*   p;                     // V: device ptr; Z: pinned ptr (same value is the UVA
                                   //    device ptr under UVA); C: locked host ptr;
                                   //    N: NULL until per-step slot fixup
    uint64_t off;                  // N only: byte offset of this tensor inside the slot
                                   //    (slot base + off = pointer; fixup is 1 add)
    uint8_t kind : 3;              // WKindX
    uint8_t loc  : 2;              // WLoc
    uint8_t tier : 2;              // Tier (redundant with table tier; lets kernels assert)
    uint8_t rsv  : 1;
};                                  // static_assert(sizeof(TagPtr)==16)

struct WLayer {
    Tier     tier;                 // V/Z/C/N
    uint8_t  engine;               // 0=GPU, 1=CPU (V,Z->GPU; C->CPU; N->CPU(v2)/GPU-UVA(v1))
    uint8_t  n_full;               // full-attn? (i&3)==3 — baked, no per-token test
    uint8_t  _pad;
    TagPtr   qkv, z, a, b, out, gate, up, down;        // linear-attn bigs (a/b bf16 at 27B)
    TagPtr   q, k, v, o;                                // full-attn bigs (unused rows for lin)
    TagPtr   qkv_sc, z_sc, out_sc, gate_sc, up_sc, down_sc, q_sc, k_sc, v_sc, o_sc;  // F8 scales
    TagPtr   in_norm, post_norm, q_norm, k_norm, gated_norm, conv1d;  // bf16 smalls -> pinned arena
    TagPtr   a_log_f32, dt_bias;                       // pre-widened f32 / bf16
    // state owners follow the layer's engine (tier-dispatch §2.3): delta/conv/KV ptrs
    // live in DecodeWorkspace27 (device) or host arrays (C/N), NOT here — table stays const.
};

// Hot-path consumption sketch (generate27, per layer l):
//   const WLayer& w = table[l];
//   switch (table.tier_of(l)) {
//     case V: launch_layer_gpu(w, /*all pointers device*/);            break;  // static
//     case Z: launch_layer_gpu(w, /*weight ptrs = pinned (UVA); scales = dev*/); break;
//     case C: cpu_layer(w, /*all host*/);                               break;  // CpuPool launch
//     case N: { const u8* slot = feeder.acquire_layer(n_idx(l));
//               TagPtr fixed[16]; fixup_slot_ptrs(w, slot, fixed);     // base+off, ~16 adds
//               v2: cpu_layer(w, fixed);  v1: launch_layer_gpu(w, fixed); // slot UVA
//               feeder.release_layer(n_idx(l)); }                      break;
//   }
// lm_head/MTP: plain device pointers (never tier-dispatched).
```

Rules baked into the table (why each is safe/fast):

- **Scales always device** (VRAM arena, step 4 of load): Z/N GEMVs read 3-11 KB of scales
  per call — keeping them on-device removes the only small-but-frequent UVA reads from the
  Z path and makes `fp8_gemv*` see identical pointer classes for V and Z weights.
- **bf16 smalls always pinned-arena** (single copy, UVA-readable by GPU, directly readable
  by CPU tier) — one address space serves both engines; norm loads are ~2-10 KB/layer.
- **N-tier fixup cost**: 16 integer adds after a ~60-115 ms blocking acquire — noise. The
  fixup writes a stack array, never the table (table stays const → shareable, cache-hot).
- **tier_of[64] as data**: the interleave (Bresenham, last N at 63 — tier-dispatch §1.4) is
  computed at manifest load, so re-placement never touches the loop code.
- 9B keeps `Qwen35Decode::forward_body` (`decode.cu:133`) untouched; the 27B loop lives in
  generate27 and shares kernels only.

---

## 7. The embed NVMe row-pread path (mission item 3)

Decision (MASTER-PLAN §2.4/App. A.2): embed (2,542.80 MB bf16 [248320,5120]) **stays on
NVMe**; 10,240 B row-pread per token; whole-embed-in-RAM rejected (costs 6-7 RAM layers to
save ~1 ms). This overrides embed-lmhead §7.1's pinned-embed recommendation (its fallback
§7.1b is what we build).

- **Who issues**: the sequencer T0, on the **buffered twin** handle of
  `outside.safetensors` — deliberately NOT through the IOCP direct path: a 10 KB read is
  sub-sector, the twin has no alignment constraints, and io-bench shows buffered handles
  are fine for one-shot small reads (io-bench-results §1/§4: cache-copy tax only matters
  at bulk rates; `FILE_FLAG_SEQUENTIAL_SCAN` never assumed to cache). It does not contend
  with the NO_BUFFERING reader stream (separate handle class).
- **When (one step ahead)**: (a) `embed_stage_async(pending_{s+1})` at step s's commit —
  the pending id is known the moment the verify argmax lands (T0 already reads
  `next_host` each step, `decode.cu:233-235`); one full step (~1.6-5.2 s) of cover.
  (b) the spec draft row: known only after `mtp_layer`'s argmax; issued immediately on the
  D2H read of `next_dev`, covered by the remaining tail (lm_head sweep 5.4 ms + handoffs ≫
  0.2-1 ms cold pread). Prefill: T≤64 rows staged into a 64-row registered window by a
  loop of twin preads before each chunk sweep (~13 ms/chunk, hidden under the seconds-scale
  layer sweep).
- **Double-buffered row slots**: 4×10,240 B registered slots (2 logical buffers × 2 rows
  for the T=2 verify path), auto-reset event per slot; `embed_stage_async(token,slot)`
  issues `ReadFile(twin, base + token*10240, 10240, slot)` (base = 2,542,834,880 absolute);
  `embed_row_wait(slot)` blocks on the event (correctness first; by construction the read
  finished a step earlier). Consumer: `embed_gather_bf16` (Phase C.4) reads the row via UVA
  — the kernel signature already takes a host-pinned pointer (embed-lmhead §7.2 sketch).
- **Budget**: 41 KB pinned — rounds to zero against the 8,531 MB cap (§4.3).
- **MTP double-read note**: `mtp_layer` re-embeds the pending token (decode.cu:137-143);
  at 27B it reads the same staged row — free (embed-lmhead §7.2).

---

## 8. Migration risks — what is disabled at 27B, and how the 9B stays intact

| 9B behavior | disabled at 27B by | residual risk + guard |
|---|---|---|
| Tensor-granular LRU (`make_room`) | new class; no tick/pins fields at all | someone reuses `Qwen35Weights` for 27B → INSIDX01 magic check refuses the index (model_file.cpp:23); ENGINE27 doesn't link storage.cu |
| On-demand mmap upload (`acquire` memcpy-from-mapping + per-tensor sync) | explicit startup reads; mmap demoted to parity tools | tools that still mmap (dump_*) must never run concurrently with a serving epoch — document; they're dev-only |
| Name-based acquire/release on the hot path | baked WTable; strings exist only in the loader | none |
| `capture_step`/`capture_spec` CUDA graphs | 27B driver never calls them (no-graphs decision) | keep the calls out of shared code paths; 9B graphs untouched |
| `Qwen35Weights::matrix` `.scales` convention | 27B path uses `base+".weight_scale_inv"` (already implemented, qwen35.cu:19-23) + kind dispatch | 9B `matrix()` untouched; 27B matrix lives in storage2/generate27 layer-prep |
| single 6 GB device budget heuristic (`generate.cu:117`) | probe-driven three-ledger fit check | generate27 refuses with numbers instead of limping |
| PinnedRing auto-register at construction (streaming.cu:302-312) | v2 wants it UNregistered (locked); constructor gets an explicit `RegisterMode {register, lock_only}` parameter instead of fallback-by-error | today's fallback treats register-failure as surprise; v2 must treat lock-only as first-class (else the 8,531-cap accounting breaks) — small API edit in streaming.cu, flagged for Phase D |
| `read_once` missing (w4/streaming §4.2) | TieredStorage2::load implements its own one-shot reads via the buffered twins (or the designed `NvReader::read_once`, nvme-reader §3) — mtp/outside/lm_head load path must exist before Phase D.3 can load anything | currently nothing supplies startup reads; storage2.cu owns this |

**Dual-mode verdict: new class + new driver, zero edits to 9B files.** Files touched:

1. NEW `include/insignia_storage2.hpp` (~180 LOC): Tier, PlacementRule/SpecialRule,
   INSIGM01 format, ProbeResults, TieredStorage2, TagPtr/WLayer/WTable.
2. NEW `src/storage2.cu` (~450 LOC): manifest parser, probe ladders, ordered loader +
   unwind, arena management, embed row service, WTable bake, `build_epoch_plans`.
3. EDIT `src/streaming.cu`/`insignia_streaming.hpp` (+~30 LOC): `PinnedRing` explicit
   register/lock-only mode; (per w4/streaming P1) event-handle leak fix rides along.
4. EDIT `tools/mk.py` ENGINE27 closure: add storage2.cu, streaming.cu, model_file v2,
   generate27.cu; storage.cu stays in the 9B targets only.
5. NEW `src/generate27.cu` (tier-dispatch report's §7): wires ModelFile2 + TieredStorage2 +
   LayerFeeder + CpuPool + the baked-table loop; owns the degrade ladder.
6. UNTOUCHED: `storage.cu`, `insignia_storage.hpp`, `qwen35.cu`, `decode.cu`,
   `insignia_qwen35.hpp`, all 9B bats/tests (regression gate after every phase).

Fit with the plan: Phase A (ModelFile v2 / INSIDX02 — index-loader report) blocks storage2;
Phase D.3 (this spec) lands after D.1 io_bench + D.2 NvReader hardening; gates are R3
(byte-equality + alignment asserts through the ring) and the v1 3-layer smoke decode.

---

## 9. Risk register (storage-specific, honest)

| # | risk | severity | catch |
|---|---|---|---|
| 1 | Pinned-cap probe over-reports (fragmentation/driver); Z=21 register fails mid-load | high | ladder runs first with availPhys guard; one-shot degrade ladder; deterministic unwind (§4.4) |
| 2 | `VirtualLock` of 4.5 GB silently capped by working-set quota (PinnedRing's best-effort loop at streaming.cu:310 sets no quota) | med | `SetProcessWorkingSetSize` before C locks; assert locked-page count == requested |
| 3 | N-tier ring slot misalignment on the 8 one-digit shards if a manifest ever puts layers 0-7 in N | low today / med later | plan-builder pad + `(f8_base&15)==0` assert at acquire (R3); slab path rebases V/Z/C regardless |
| 4 | embed pread tail: draft-row read not done when verify needs it (slow cold QD1 read) | low | event wait (correct); issued at D2H of draft id, ≥5 ms cover; R10 watches step-time jitter |
| 5 | smalls arena filled from Z/C/N layer reads doubles as their only read — a bug there starves norms for ALL tiers | med | R4 seam (layer-0) exercises smalls; arena fill has its own byte-count check vs shard table |
| 6 | INSIDX02 lacks `align_base`/`in_slot_off` (vs w3 sketch) — window math re-derived in two places (storage2 + feeder plans) | low | single shared `window(shard)` helper; index-loader report owns format decisions |
| 7 | WDDM 1-2 ms submission spikes during Z-layer UVA reads | low | `cudaSetDeviceFlags(ScheduleSpin)`; C-tier pressure valve (pcie-pipeline §6/§7.3) |
| 8 | Host-pinned D=4 spec snapshots (628 MB, Phase F) forgotten in the cap ledger | med | §4.3 lists them; fits() must take an optional `extra_pinned_mb` from the spec config |

---

## 10. Summary

- TieredStorage2 = manifest-driven static placement; V (VRAM slabs, per-tensor 16 B
  rebasing) / Z (VirtualAlloc + register ≤1 GiB pieces, UVA) / C (VirtualAlloc + VirtualLock,
  never registered, outside the cap) / N (nothing resident; LayerFeeder plans).
- Startup order: probes → fit-gate → lm_head+MTP VRAM (refuse if short) → smalls+scales
  arenas → V slabs → Z pinned → C locked → N plans + ring → embed rows → warm.
  OOM mid-load = reverse unwind + precise throw + one logged degrade retry. No eviction
  anywhere; the only recycled memory is the ring by contract.
- Hot path: `tier_of[64]` + 4-cache-line `WLayer` of 16 B TagPtrs (kind|loc|tier + void*);
  N-tier adds 16 pointer fixups after the blocking acquire; embed = twin-handle 10 KB
  preads into 4 registered row slots, one step ahead, double-buffered.
- 9B stack untouched and structurally firewalled (INSIDX01 magic + link closure).
