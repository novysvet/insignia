# MASTER-PLAN — Qwen3.8-27B-FP8 un-requantized on the tiered rig

Date 2026-08-25. Consolidation of all 21 w3 reports + `audits/synthesis.md` +
`audits/w2/loader-27b-spec.md` + `audits/w2/shape-constants.md`. Read-only audit; the only
file written is this one. This is THE implementation blueprint: one recommended path,
alternatives in the appendix, every number recomputed from primary facts.

---

## 0. TL;DR — the one path

1. **Both wave-3 "champions" are invalid as published.** pcie-pipeline's L20/Z22/C13/N9 @
   2.9 tok/s assumed NVMe 6.5 GB/s (it's E: = Gen3 980, 3.3 GB/s) and its RAM table omits
   the CUDA host context and delta states (over budget ~0.4–1.0 GB). placement-final's
   L18/Z27/N19 @ 1.43 tok/s pins 10.3 GB — violating the WDDM 50%-RAM pinned cap (~7.95 GiB)
   that pcie-pipeline discovered, and also assumed 6.5 GB/s. Recomputed below.
2. **Honest achievable best on E: alone ≈ 1.6–1.75 s/step: ~0.6 tok/s single, ~0.9–1.0
   tok/s MTP depth-1, ~1.5 tok/s at MTP depth-4.** The binding resource is the Gen3 drive
   (115.4 ms per 381 MB layer); everything else (PCIe, CPU, GPU) hides under it with 2–3×
   slack. The second lever is the **CPU tier**, which escapes the pinned cap via
   VirtualLock'd pageable RAM — the only way to use RAM beyond ~8.4 GB of pinned Z-layers.
3. **Bring-up ladder (one architecture, three manifests):**
   v1 all-stream (correct, ~5.2 s/step) → v1.5 pin-19 (manifest flip, ~3.0 s) →
   v2 CPU tier Z21/C9/N15 (~1.75 s) → optional dual-drive E:+C: (~0.6–0.7 s, +150–190%,
   **requires explicit user approval** — copies ~6–15 GB to the SYSTEM drive).
4. VRAM: L=19 resident (layers 0–18) + lm_head 2.54 GB + MTP 0.48 GB + workspace ≈
   11.0 GB committed of an 11.3 GB budget (probe `cudaMemGetInfo` at startup; L=20 is a
   195-MB-spare stretch, off by default). **No CUDA graphs at 27B** (graph-hazards §6c:
   ≤0.3% upside, real pointer hazards).
5. embed stays on NVMe (10 KB row-pread/token, ~1 ms, prefetchable); lm_head VRAM
   (mandatory); MTP VRAM (drafts hide under the stream).
6. Phases: **A loader/INSIDX02 → B shapes → C kernels → D streaming/reader → E parity
   R0–R10 → F weight-stationary prefill + MTP KV fill + spec-decode → G placement
   solver/tuning.** io_bench + NvReader + Python reference scripts can start on day 1 in
   parallel. Budget ≈ 4,500–5,000 LOC, 9–13 sessions.
7. Top silent-wrong-token risks, all with a designed catch: zero-center (1+w) norms (R4),
   GQA `head/6` (R5 + unit test), deltanet `kh=head/3` (R4), A_log BF16-as-F32 (R4, α≈1
   signature), scale multiply semantics (R1), fp8_gemm 64-row OOB (R1 with fixed test),
   ring 16B misalignment (R3 assert — `data_start ≡ 8 mod 16`!), mtp.fc dims/concat (F,
   acceptance), v21 last-K-step cp.async race (fix pre-R4), KV/committed overrun (C1
   hardening, R10).

---

## 1. Ground truth and corrected inputs

All subsequent arithmetic uses only these verified facts.

### 1.1 Hardware (corrected; nvme-reader §1 is authoritative for drives)

| resource | value | source / correction |
|---|---|---|
| GPU | 4070 SUPER, AD104, **56 SMs**, 504 GB/s, 48 MB L2, 12,282 MiB | attn-27b §7 (the "128 SMs" premise was wrong) |
| VRAM app budget | **11,300 MB committed** (probe at startup; conservative 10,800 MB, aggressive 11,584 MB) | between placement-final §1 and pcie-pipeline §1; WDDM+display eat 0.5–1 GiB |
| Host RAM | 15.9 GiB physical; **13,500 MB usable** after OS+CUDA+HMB | placement-final §1; mission-given |
| **Pinned cap (WDDM)** | **7.95 GiB = 8,531 MB total** for cudaHostAlloc/Register (weights **and ring**); plan 7.5 GiB = 8,048 MB pending a startup probe ladder | pcie-pipeline §2 (NVIDIA-documented 50% of RAM) |
| PCIe 4 x16 | pinned memcpy 24–25 GB/s; **UVA zero-copy kernel read 15–20 GB/s (plan 18)** | pcie-pipeline §6 |
| CPU | 5600X Zen3 AVX2+FMA+F16C; DRAM ~37–40 GB/s; CPU GEMV ~10.3–10.8 ms/layer | cpu-fp8 §0/§2 |
| **E:** (model dir) | **Samsung 980 1TB, Gen3, DRAM-less HMB, ~3.3 GB/s effective, 600 TBW** | nvme-reader §1 — measured-class; NOT 6.5 GB/s |
| C: | 980 PRO 500GB, Gen4, ~6.4 GB/s effective, 300 TBW, **93.8 GB free, SYSTEM drive** | nvme-reader §1 |

### 1.2 Model bytes (byte-exact, loader-27b-spec §0 / placement-final §1)

| object | MB |
|---|---|
| linear layer ×48 (F8 382.73 + BF16 1.13) | 383.86 |
| full-attn layer ×16 | 372.31 |
| average layer | 380.97 |
| lm_head bf16 [248320,5120] (VRAM, mandatory) | 2,542.80 |
| embed bf16 (NVMe row-pread) | 2,542.80 |
| MTP shard (fc bf16 + 1 full-attn layer) | 477.20 |
| delta state /lin layer f32; conv state | 3.15; 0.12 |
| KV /full layer @ctx2048 bf16 / f32 | 8.39 / 16.78 |
| text weights total | 29,945 MB |

### 1.3 Per-layer per-token decode costs (bandwidth-bound, recomputed)

| tier | ms/layer | arithmetic |
|---|---|---|
| V VRAM @504 | 0.78 (lin) / 0.757 (full) | 383/504 + state 12.5 µs |
| Z pinned-UVA @18 GB/s | **21.2** | 380.97/18 (17.3 @22, 25.4 @15) |
| C CPU (locked pageable) | **10.8** | 380.97/37 + handoff ~0.3 |
| N E:-NVMe | **115.4** | 380.97/3.3 |
| N C:-NVMe | 59.5 | 380.97/6.4 |
| N split E:+C: (aggregate) | **39.3** | 380.97/(3.3+6.4) — see §2.5 |
| lm_head sweep (VRAM) | 5.4 | 2,542.8/470 |
| MTP draft step (all VRAM) | 6.7 | fc 0.22 + layer 0.74 + lm_head 5.41 + ε |

Step model (pipelined "treadmill", placement-final §4 / pcie-pipeline §3):
`T_step ≈ max(NVMe, GPU-serial, PCIe, CPU, DRAM) + ~10 ms tail`, where verify T=2..5 reads
weights once (bandwidth-bound ⇒ extra rows free) and MTP drafts (6.7 ms, VRAM-only) hide
under the stream. MTP multipliers: D=1/T=2 → 1.6 tok/step at p=0.6; D=4/T=5 → 2.52 (p=0.6)
/ 2.77 (p=0.7 after the F7 KV-hole fix lifts acceptance).

---

## 2. RESOLVED placement decision tree (arithmetic shown)

### 2.1 Why both published champions fail

**pcie-pipeline champion L20/Z22/C13/N=9 @ 545 ms (2.9 tok/s MTP):**
- NVMe term used 6.5 GB/s: 9×380.97/6.5 = 527 ms. On E:: 9×115.4 = **1,039 ms** → T_step
  ≥ 1,050 ms → max 1.6/1.05 = **1.52 tok/s MTP-D1**, not 2.9.
- RAM: their table (pinned Z 7.81 + locked C 4.62 + ring 0.73 + misc 0.15 + OS 2.4 = 15.71
  "of 15.9") **omits the 0.6 GiB CUDA host context** they themselves list, plus ~108 MB
  delta/conv states and 101 MB host KV → true ≈ 16.4–16.6 GiB > 15.9 physical. Over by
  ~0.4–0.7 GiB. Shaving 1–2 layers to Z/C brings N to 10–11 → 1,155–1,270 ms.
- Verdict: right architecture (CPU tier, no staging, embed mmap), wrong drive speed and an
  overdrawn RAM account.

**placement-final champion L18/Z27/N19 @ 1,120 ms (1.43 tok/s MTP):**
- Pinned: 27 layers = 10,286 MB **> 8,531 MB WDDM cap** (plus a 1.5 GB ring on top → 11.8
  GB pinned). Not allocatable on this box, period.
- NVMe: 19×380.97/6.5 = 1,115 ms assumed. On E:: 19×115.4 = **2,193 ms** → 0.46 single /
  0.73 MTP-D1.
- Verdict: right structure (treadmill, embed row-pread, interleaving), but Z must obey the
  pinned cap and every NVMe number must use 3.3 GB/s.

### 2.2 VRAM ledger → L = 19 (layers 0–18)

Fixed block: lm_head 2,542.8 + MTP 477.2 + MTP-KV 8.4 + norm 0.01 + workspace 250
(decode ws ~40, pf_* tiles ~27, snap_delta 151 + snap_conv 5.9 for D=1 — at D=4 the
T−1 = 4 snapshots = 628 MB go to **pinned host**, restore ≈ 12 ms hidden) + CUDA ctx 400
(no whole-step graphs) = **3,678.4 MB**.

| L | composition | total (bf16 KV) | vs 11,300 budget |
|---|---|---|---|
| 18 | 14 lin + 4 full | 3,678.4+5,419.8+1,522.8 = 10,621 | ✓ 679 spare |
| **19** | **15 lin + 4 full (0–18)** | 3,678.4+5,807.0+1,522.8 = **11,008** | **✓ 292 spare** |
| 20 | 15 lin + 5 full | 11,389 | ✗ (fits only the 11,584 aggressive budget with 195 spare — stretch, probe-gated) |

**L = 19 recommended.** Full-attn vs linear per-MB is a 1% wash at ctx 2048 (placement-final
§3); contiguous 0–18 contains 4 fulls naturally. At ctx ≥ 8k prefer linear layers in VRAM.
`score[4096]` caps ctx at 4096 regardless (safety C1 couples them); default ctx 2048.

### 2.3 RAM ledger and the pinned cap (the real constraint stack)

Usable 13,500 MB. Non-weight claims: ring (K×384: K=2 → 768, K=3 → 1,152), host KV bf16
for 12 non-VRAM fulls 100.7, delta+conv states for 33 non-VRAM linears 107.9, buffers 150,
margin ≥250. Weights budget = 13,500 − (ring + 608.6) → **Z+C ≤ 31 (ring 2) / 30 (ring 3)
/ 29 (ring 4)**. Pinned cap: 8,531 MB hard / 8,048 planned.

Two crucial structural facts:
1. **The NVMe ring counts against the pinned cap** when `cudaHostRegister`'d (needed for
   GPU-UVA consumption of streamed layers). Ring 768–1,152 MB ⇒ pinned Z ≤ (8,531−ring)/380.97
   = 18–20 layers.
2. **VirtualLock'd pageable RAM (CPU tier) does not count against the pinned cap** and is
   only consumable by CPU kernels. Beyond ~22 pinned layers, RAM is only useful via the
   CPU tier. This — not PCIe balance — is why the C tier exists.

### 2.4 The ladder (all configs: L=19, lm_head/MTP VRAM, embed NVMe, no graphs)

**v1 — bring-up, all-stream (BUILD THIS FIRST).** L=19 VRAM; all 45 non-VRAM layers stream
NVMe(E:)→pinned ring→**GPU consumes via UVA zero-copy reads** of the ring slot. No CPU layer
kernels, no pinned-Z residency, no VRAM staging (S-tier eliminated: pcie-pipeline §4 — it
buys PCIe efficiency on a non-binding bus and pays 2 VRAM layers). Pinned = ring 3×384 =
1,152 ≤ 8,531 ✓. RAM = 1,760 total ✓ enormous slack.
- NVMe 45×115.4 = 5,193 ms (binding); GPU-serial 14.8 + 45×21.2(954) + 12 ≈ 981 ✓ hides;
  PCIe 954 ✓; DRAM 34 GB/5.2 s = 6.6 GB/s ✓.
- **T ≈ 5.21 s → 0.19 single / 0.31 MTP-D1 / 0.48 MTP-D4.** Correct, minimal, and every
  mechanism (reader, ring, UVA consumption, tier dispatch) is reused unchanged by v1.5/v2.
  Parity dumps (R4–R9) ride v1: one 14-token sweep = 29.9 GB ≈ 9.1 s — perfectly fine.

**v1.5 — pinned-Z manifest flip.** Same binary; manifest pins Z=19 layers
(pinned 7,238 + ring 1,152 = 8,390 ≤ 8,531 ✓; conservative probe → Z=18). N = 26.
- NVMe 26×115.4 = 3,000 (binding); GPU 14.8+19×21.2+26×21.2+12 = 981 ✓.
- **T ≈ 3.01 s → 0.33 / 0.53 / 0.84.** One manifest file + a startup probe. No new code
  beyond v1 except the probe + residency loader.

**v2 — CPU tier (the E:-only optimum).** Ring becomes VirtualLock'd pageable, N-layers
CPU-consumed (frees the whole pinned cap for weights): **Z=21 pinned (8,000 MB) / C=9
locked (3,429) / N=15**, ring 2×384. RAM = 8,000+3,429+768+608.6 = 12,806 ✓ (694 spare).
Probe stretch: Z=22/C=8 (pinned 8,381 ✓). Ring-2 variant Z=21/C=10/N=14 → RAM 13,187 ✓.
- NVMe 15×115.4 = **1,731** (N=15) or 14×115.4 = **1,616** (N=14) — binding;
  GPU-serial 14.8 + 21×21.2 = 460 ✓ (N on CPU); CPU (9+15)×10.8 = 259 ✓ (15% duty);
  PCIe 21×21.2 = 445 ✓; DRAM 22.9 GB/1.73 s = 13.2 GB/s ✓.
- **T ≈ 1.63–1.75 s → 0.57–0.61 single / 0.92–0.98 MTP-D1 / 1.44–1.55 MTP-D4 (p=0.6);
  1.59–1.70 at p=0.7.**
- Bonus: pinned Z-layers are schedule-flexible — CPU can read pinned memory too, so each
  Z layer can be consumed by whichever engine (GPU-UVA 21.2 ms / CPU 10.8 ms) has slack.
  The startup solver rebalances from a 200 ms microbench, not hardcoded guesses.
- Prefill(512), weight-stationary: N layers staged via one 384 MB VRAM slot (drop to L=18
  for the prefill window): 15×115.4 + 21×15.9 + 9×~84 + V-compute 129 + tails 45 ≈
  **~3.0–3.5 s per turn** (first turn cold +1–2 s; startup Z+C fill 11.4 GB ≈ 3.5 s once).

**v2 + dual-drive (OPTION-G — flagged, requires explicit user approval; do NOT assume).**
Copy the N-set shards (14–15 shards, 5.3–5.7 GB; or all layers, 24.3 GB — 93.8 GB free
holds it) to C:, reader takes a per-shard drive map (INSIDX02 already stores per-shard
relative paths — a mirror-dir override list, ~50 LOC). Time-balanced assignment (C: takes
~65% of N by time; **aggregate = 3.3+6.4 = 9.7 GB/s; the mission's harmonic 4.5 GB/s is the
per-layer-stripe number — for layer-granular streaming the correct aggregate is the sum**,
nvme-reader §1/§5.1):
- NVMe 14×39.3 = 550; GPU 472; CPU 259; PCIe 445; **DRAM 22.5 GB/0.55 s = 41 GB/s —
  co-binding at the ~40 GB/s ceiling** → realistic T ≈ 0.60–0.70 s.
- **≈ 1.5–1.7 single / 2.4–2.7 MTP-D1 / 3.7–4.2 MTP-D4.** Prefill(512) ≈ 1.5–2.0 s.
- This is the single biggest lever on the rig (+150–190% over v2-E:). Why it needs
  approval: C: is the SYSTEM drive (pagefile/OS writes contend with a 100%-duty read
  stream; a wedged system drive wedges the box), the 980 PRO is 300 TBW (smaller), and
  ~6–30 GB of permanent disk. Endurance per drive halves (E: ~12 TB/h → ~6 TB/h each).
  Recommendation: **build the reader's drive-map hook in Phase D (it's free), ask the user,
  and only then copy.** If refused, v2-E: stands at ~1.0 tok/s MTP-D1.

### 2.5 Summary table (recomputed, E: = 3.3 GB/s unless noted)

| config | L | Z (pinned) | C (CPU) | N | T_step | single | MTP D=1 | MTP D=4 (p=.6) |
|---|---|---|---|---|---|---|---|---|
| v1 all-stream | 19 | 0 | 0 | 45 | ~5.21 s | 0.19 | 0.31 | 0.48 |
| v1.5 pinned-Z | 19 | 19 | 0 | 26 | ~3.01 s | 0.33 | 0.53 | 0.84 |
| **v2 CPU tier** | **19** | **21** | **9** | **15** | **~1.75 s** | **0.57** | **0.92** | **1.44** |
| v2′ ring-2 | 19 | 21 | 10 | 14 | ~1.63 s | 0.61 | 0.98 | 1.55 |
| v2+OPTION-G (E:+C:) | 19 | 21 | 10 | 14 split | ~0.60–0.70 s | 1.5–1.7 | 2.4–2.7 | 3.7–4.2 |

Placement mechanics (all tiers): VRAM contiguous at the start; Z/N interleaved in 19–63
with N every ~3rd layer (reader cadence ≈ consumption; Bresenham spread, pcie-pipeline §7.4);
KV follows its layer (V→VRAM, Z/N→pinned RAM); embed = buffered-twin 10 KB row-pread issued
a step ahead (target row known at commit time); small BF16 params (~65 MB all layers)
permanently pinned in a dedicated arena. f32 KV for parity (v1), bf16 flag later.

Endurance (honest framing, nvme-reader §5.1): TBW is a write rating — read-disturb wear is
far slower, so treat these as upper bounds. v2-E:: 5.7 GB/step streamed while generating →
~12 TB/h on E:; log SMART `Data Units Read`/`Percentage Used` per session (one-evening
experiment gates how hard to lean on N). OPTION-G halves per-drive traffic.

---

## 3. Ordered implementation phases

Conventions: builds only via `build\*.bat` (or the mk.py driver below) with vcvars64 +
nvcc `-arch=sm_89`; every phase ends with the 9B regression suite still green (kernels are
`template <int QH>` so `<16>` and `<24>` coexist); parity gates are hard blockers per
AGENTS.md.

### Phase 0 — build system + blocking bug fixes (prequel, half a session)

- **Files**: `build/mk.bat` + `tools/mk.py` (build-system §4, full source in the report:
  mtime-incremental obj cache, `--threads 0`, fixed output names, `ENGINE27` closure,
  future targets io-bench/generate27/nll27/dump-*-27b); delete `bench-gemm-blocked.bat`,
  `oldgen.bat`, `shim-only.bat`, `tiny.bat`, `mk.bat`-hello; 11 stale bats are superseded.
  Flags: add `-lineinfo -DNDEBUG`; `INSIG_PTXAS_V=1` opt-in.
- **Fixes that block everything** (safety.md, diff-verify.md):
  - `src/gemm.cu:275` — v21 last-K-step race: `if (kb+2<ksteps) wait_group 1 else
    wait_group 0` (C2; nondeterministically corrupts every prefill GEMM — poison for the
    parity hunt).
  - `src/test_fp8.cu:15` — e4m3_host bias-7 fix (`ldexpf(1+m/8, e-7)`) + add max-rel-err
    metric next to cosine; fix the `x` 2·cols/3·cols OOB (F2/F3). **R1 is meaningless
    until this lands.**
  - `src/attention.cu:7` + `src/prefill.cu:131-138` + `row_logp` twins — dedicated smem
    slot for the second `red[0]` reuse (same class as the fixed RoPE race; sits exactly on
    the flaky full-attn path).
  - Safety C1/C5 quick wins: `generate.cu` refuse-don't-clamp ctx; `cudaStreamSynchronize`
    first line of both dtors.
- **Gate**: all existing 9B tests green under mk.py; `test-fp8` prints honest rel-err.

### Phase A — loader: INSIDX02 + FP8 dtype + names + matrix kinds + norm centering

- **Files / function-level changes**:
  1. `include/insignia_model.hpp:10` — `f8_e4m3=7, f8_e5m2=8`; `tools/index_safetensors.py:8`
     — `"F8_E4M3": 7` (loader-gaps §1).
  2. `tools/index_safetensors.py` — directory mode, INSIDX02 emission (loader-gaps §2.2:
     shard table with `data_start/align_base/crc32-at-build/flags(skip_vision,is_layer,is_mtp)`
     + relative paths; tensor table with `shard/scale_idx/in_slot_off`; name-sorted; per-shard
     bounds; 407/407 scale links; ~115 KB index).
  3. `src/model_file.cpp` + `insignia_model.hpp` — ModelFile v2: 66 eager handles (O_DIRECT
     + OVERLAPPED set for the reader, optional mapped set for parity tools — loader-gaps
     §2.3: no LRU, no reopen), `TensorView{shard,off,scale_idx}`, per-shard bounds check,
     `find_linked(view)` replacing the `base+".scales"` string convention.
  4. `src/qwen35.cu` — `matrix()` kinds `{mxfp4_mlx, mxfp4_i4, fp8, bf16}` (loader-gaps
     §4.3 pseudocode): fp8 branch acquires `X.weight_scale_inv`, asserts
     `[ceil(r/128),ceil(c/128)]`, NO ×8 on cols; bf16-no-scale branch (fixes the live 9B
     `mtp.fc` latent throw); MXFP4 branch untouched.
  5. Engine name remap at all `decode.cu` acquire sites: `model.language_model.layers.N.*`,
     bare `lm_head`, bare `mtp.*` (parity-ladder §1.1 table).
  6. `A_log` BF16 — `deltanet_parameters`/`deltanet_params_batch` take `const void* +
     bool a_log_f32`, widen in-kernel (deltanet-27b §4; dispatch on `DeviceView.dtype`).
  7. **Zero-center norms** (qwen35-arch §7.1 — THE formula change): all `Qwen3_5RMSNorm`
     tensors (input_layernorm, post_attention_layernorm, q_norm, k_norm, model.norm,
     mtp.norm, pre_fc_norm_*) switch call sites to the existing `rms_bf<Z=true>` (1+w) path
     — 9 call sites in decode.cu + `qk_norm_rope` gains the +1; **`linear_attn.norm` stays
     RAW (Z=false)** — RMSNormGated is one-centered in this checkpoint (measured
     [0.79, 0.93]).
  8. `src/test_model.cpp` — drop the 699/`.scales` asserts; new dtype/shape table diff vs
     the census.
- **Gate (R0)**: index builds for the 27B dir; every engine-expected name resolves; dtype/
  shape diff vs loader-27b-spec §2 census = empty; 9B INSIDX01 path still loads.
- **Effort**: ~600 LOC, 1–1.5 sessions.

### Phase B — shape parameterization (9B → 27B constants)

- **Files**: per `audits/w2/shape-constants.md` (148 engine sites; the report is the
  checklist — with two corrections, see below) + attn-27b §8 + deltanet-27b §5c:
  1. `include/insignia_qwen35.hpp` — hidden 5120, inter 17408, layers 64 (and WIRE the
     constants; today every loop hardcodes its own 32/12288 — they are dead and will drift).
  2. `DecodeWorkspace` ctor/memsets (decode.cu:12–27): the trap list — `attn_gate/z/core/
     pf_q/pf_g/pf_core` **6144** (not 5120), `a/b/pf_a/pf_b` 48, `delta_state/snap_delta
     48×48×128×128`, `conv_state/snap_conv 48×10240×3`, `kv 16·ctx·1024`, `pf_scratch
     64×12288` (q_proj rows, not conv's 10240), `pf_bf16 64×17408×2B`, `pf_qkv 64×10240`,
     `pf_z 64×6144`, `pf_gate/up 64×17408`.
  3. Kernel grids/strides per attn-27b §1 table (qk_norm_rope grid 28, `isq=head<24`,
     `k+(head−24)*256`; gqa grids 24; split_q_gate `i<6144`, batch `/24 %24 t*12288`;
     sigmoid_mul 6144) and deltanet-27b (deltanet `<<<48,128>>>`, qkv stride 10240 with
     offsets 0/2048/4096 **staying**, a/b 48, gated norm rows 48 cols 128, conv 10240).
  4. `spec_rollback_kernel`: 48·48·128·128 / 48·10240·3 / hidden 5120 grid-stride (also
     fixes the latent 256-of-4096 partial-hidden 9B bug, deltanet-27b §5a).
  5. Instrumentation: 56 sites per shape-constants (dump/test/bench tables) — at minimum
     the R4/R7 dump tools + `test_attention`/`test_deltanet` (H=24/kvh=h/6; H=48/kh=h/3).
- **Corrections to shape-constants.md (attn-27b §0, authoritative)**: its lines 169/365
  claim `kvh=head>>2` "stays" at 24 q-heads — **FALSE**: group is 24/4 = 6, so
  `kvh = head/6` (write the plain division; nvcc emits mul-magic; `(h*171)>>10` verified
  equivalent but unnecessary). `head>>2` mis-groups 16/24 heads and sends heads 16–23 OOB
  into the next token's KV rows — silent future-token attention. Same class:
  deltanet `kh = head/3` (not `>>1`).
- **Gate**: unit tests (test_attention, test_deltanet, test_model, test_qwen35 asserts
  12288/5120) green; 9B `<16>`/`<32>` instantiations green; full ENGINE27 link closes.
- **Effort**: ~450 edited sites, 1 session (tedious, mechanical).

### Phase C — 27B kernels

Paste-ready code exists in the reports; this phase is mostly transcription + tests.

1. **Attention** (attn-27b §1a–§5, full replacements): `qk_norm_rope`/`_batch` as
   `template<int QH>` with the `nsc` race-fix preserved; `gqa_decode` grid 24 + `kvh=h/6`
   (body untouched for parity); `gqa_prefill` `dim3(24,T)` + `kvh` + `(t*24+head)*256`;
   `split_q_gate`/`expand`/`_batch`; `store_kv`/`_batch` verified UNCHANGED (4 kvh × 256).
   Split-K decode + bf16-KV (attn-27b §6–§7) deferred to Phase G behind a ctx>1024 bench.
2. **DeltaNet** (deltanet-27b §1–§5): decode kernel `<<<48,128>>>` + `kh=head/3`; prefill
   scan `<<<48,128,66,048>>>` (smem unchanged, exactly 1 wave on 56 SMs); conv prefill +
   `conv_roll_state` at 10240 (decode `conv4` is n-generic — call site literals only);
   params kernels with A_log dtype dispatch.
3. **fp8 fixes** (fp8-kernels F1/F5): `fp8_gemm` guarded epilogue (smem-staged tile, write
   only `t<T`) + `throw` on T>64 + optional `wm*16<T` MMA skip; `fp8_gemv2`
   `cudaFuncSetAttribute` 99 KB opt-in + launch-error checks. Dispatch
   `linear/linear2/linear_batch` on QuantMatrix kind (fp8 → `fp8_gemv/gemv2/gemm`).
4. **bf16 family** (embed-lmhead + ab2-redesign): `bf16_gemm` (v21 skeleton minus dequant;
   rows%32, cols 5120 ✓) — build FIRST and measure T=1 through it; `bf16_gemv` v2
   (persistent warp-per-row, fp32 accum — mandatory, §3.1) only if GEMM-T1 loses;
   `embed_gather_bf16` (T-row, reads the pinned staged row via UVA); `bf16_gemv_ab2_pair`
   (ab2-redesign design B — one launch replaces 4 GEMVs; exact `u<<16` bit surgery);
   guards `cols!=4096 → throw` on the three 9B ab2 launchers (never route 27B there).
5. **lm_head**: VRAM-pinned at load (before layer allocation, refuse to start if short);
   decode T=1 → bf16 GEMV/GEMM + `argmax_fast`; verify T≥2 → one bf16 GEMM + merged
   argmax launch; NLL T=64 → bf16 GEMM tiles; `nll.cu`/`generate.cu run_nll` bf16 branch.
6. **mtp_layer** 27B wiring: fc `bf16_gemv(w,x,y,5120,10240)` (orientation verified
   `[out,in]`, loader-gaps §4.4), concat 5120, embed-first order verified EXACT (§1.4).
- **Gates**: **R1** — `test-fp8` over all 7 matrix geometries × T∈{3,33,64} with the fixed
  reference: cos > 0.999999 AND max-rel < 1e-4; `test_bf16_ab2` (design §6) cos > 1−1e-6 +
  throw-tests; `test_attention`/`test_deltanet` updated references; bench bf16 GEMV/GEMM
  on [248320,5120] ≥ 400 GB/s cold-L2 (insig4-perf §2.4 protocol).
- **Effort**: ~1,200 LOC, 1.5–2 sessions.

### Phase D — streaming: reader, ring, tier dispatch, decode loop

1. **io_bench first** (nvme-reader §6): `tools/io_bench.cpp` — QD/block sweep, direct-vs-
  buffered cyclic A/B, tail-path CRC. **Acceptance: E: ≥ 3.0 GB/s sequential 2 MiB QD16
  NO_BUFFERING over a 20 GB span; if E: < 2.8, stop and re-plan the N-tier budget.**
   (Expect ~3.2–3.4.)
2. **NvReader** (nvme-reader §3, ~210 LOC complete in the report): per-file direct+buffered
   twin handles, one IOCP, 2 reader threads parked on `GetQueuedCompletionStatusEx`,
   self-arming top-up to QD, ring = `VirtualAlloc` + optional `cudaHostRegister`, unit =
   one layer shard read whole from byte 0 (gapless prefix verified), `<512 B` EOF tails via
   the buffered twin, `read_once` for mtp/outside startup, bounded retry + fatal flag.
   2 MiB blocks from offset 0 are 512e/4Kn-safe by construction.
   **CRITICAL addition — 16-byte rebasing**: `data_start ≡ 8 (mod 16)` in every shard, so
   F8 tensor bases in the slot are ≡8 mod 16 → `uint4`/`float4` loads in fp8 kernels get
   misaligned addresses (loud crash, or 1–3% penalty if scalarized). Fix in the block
   table: insert one 8-byte pad at the BF16→F8 boundary so every F8 tensor lands 16B-
   aligned in the slot (`in_slot_off` already precomputed in the index — make the pad part
   of the plan builder; BF16 smalls are scalar-u16 loads, alignment-free). Assert
   `(f8_base & 15) == 0` at acquire. R3 tests exactly this.
3. **TieredStorage2 + placement manifest** (loader-gaps §3.3 + pcie-pipeline §7.2): shard-
   major slots; `PlacementRule{layer_lo,layer_hi,tier}` manifest as **data** (v1/v1.5/v2
   are three manifest files, zero engine changes); one IO thread; prefetch ~1 layer ahead;
   `acquire_blocking/acquire_host_blocking/release` with the documented threading contract;
   permanent pin arena for BF16 smalls + lm_head + MTP; startup probe ladder for the pinned
   cap (1 GiB until fail → Z count) and free VRAM.
4. **Decode loop tier dispatch** (pcie-pipeline §7.2 hot-path sketch): `tier_of[64]` baked
   at startup; per layer V → static device pointer, Z → UVA pointer into pinned copy,
   N → `unit_wait` ring slot (UVA) [v2: or CPU]; logits D2H + embed row H2D on a dedicated
   copy stream; spin handoffs (`cudaSetDeviceFlags(ScheduleSpin)`); affinities T0+GPU 0–1,
   readers on SMT siblings, [v2: GEMV team 5–11].
   **NO CUDA graphs anywhere at 27B** — plain stream execution (graph-hazards §6c;
   spec-deepen §6). The existing `capture_*` paths simply aren't called by the 27B driver.
5. **Safety hardening in the loop**: C1 device-side guards (`store_kv`/`spec_commit`
   respect `max_context`/`committed` bounds — eager throws exist, but the spec loop's
   committed/KV overrun must be bounded in-device), token-id clamps in gather kernels (C9),
   position resync in `committed_count()` (C8).
- **Gate (R3)**: stream every shard through the reader; byte-equality vs `np.memmap`; all
  66 CRC32s match `crc32.txt`; alignment asserts pass; a 3-layer smoke decode runs v1.
- **Effort**: ~900 LOC, 1.5–2 sessions.

### Phase E — parity ladder R1→R10 (the correctness spine)

Exactly `parity-ladder.md` §6 (commands, metrics, thresholds, golden token ids all
specified there). Order and prerequisites:

- **Pre-req tool fixes**: `reference_pf_f8.py` must include in-chunk conv history for t>0
  (port the multistep script's conv roll — diff-verify H1, or every DeltaNet seam past
  token 0 shows false failures); `dump_multistep` bf16 lm_head branch (H2).
- **R0** (Phase A gate) → **R1** (Phase C gate) → **R3** (Phase D gate).
- **R4** layer-0 DeltaNet prefill seam (T=14 GOLDEN ids, cos > 0.99999): catches zero-center
  norms, A_log dtype, kh/3, scale semantics, conv layout, fp8 decode — layer 0 exercises
  everything except attention. Also check `alpha ∈ (0,1), alpha ≠ 1.0` after params
  (the BF16-as-F32 signature, deltanet-27b §7.3).
- **R5** layer-3 full-attn seam, run 5× (rope active, GQA /6, q/k norm +1, output gate):
  kills the RoPE-pairing risk (halves vs interleaved — still unvalidated anywhere) and any
  residual smem-race flicker. Hard stop on any run-to-run variance.
- **R6** all-64-layer seams (min ≥ 0.9999, median ≥ 0.99999, zero NaN) — rides the v1
  all-stream placement (one sweep ≈ 9–30 s per dump).
- **R7** 4-step multistep decode (states: conv [10240,3]×48, Δ (48,128,128)×48, KV 16×,
  bf16 embed slice, chunked lm_head argmax; argmax 4/4).
- **R8** NLL 128 tokens (|ΔNLL| < 0.02 first clean run → 0.005 tightened).
- **R9** greedy 8-token continuation vs the NumPy self-reference (no torch on this box —
  the reference IS ground truth): 8/8 ids. **This is the "engine is correct on 27B" gate.**
- **R10** 1000-token endurance/stability (finiteness, NLL < 5 nat/token, budget drift,
  tok/s ≥ 80% of the placement prediction).
- **Effort**: ~700 LOC (5 reference `_f8.py` clones + 3 dump tools + bats), 1–2 sessions
  **plus an open-ended budget of 0–3 sessions for the full-attn parity hunt** (the known
  open risk; R5 is the designed kill-shot and the smem-race/red[0] fixes from Phase 0
  remove the two known nondeterminism sources).

### Phase F — weight-stationary prefill + MTP KV fill + spec decode

1. **`prefill_layer_stationary(tokens, T_total)`** (prefill-27b §1/§6 TODO, ordered list):
   outer loop layers 0–63, inner 64-row tiles; activation ping-pong `h_A/h_B [S,5120] f32`
   in VRAM (21 MB @ 512 — NOT a RAM ring); position bumped ONCE per turn (the classic
   reorder bug); `row0_snap=nullptr` during turn prefill (saves 1.2 GB/sweep of pointless
   writes); tail-tile zero-padding of `pf_bf16` preserved; seam variant for parity;
   `prefill_chunk_device(T=2)` stays byte-identical (it is the spec verify path).
   Super-chunk policy: S = whole turn ≤ ctx (curve is flat above ~1000; C_io dominates).
2. **MTP KV prefill** (prefill-27b §4 + spec-deepen F7): memset mtp_keys/values at init
   (1 line); batched post-prefill pass filling slots 0..P−2 from
   `(embed(t_{p+1}), h^main_p)` — fc GEMM + qk-rope + `store_kv_batch`, skip gate/o/MLP,
   ~4 ms, no lm_head dependency. Fill-during-prefill option (b) rejected (report §4).
   This is the acceptance-rate lever (p 0.6 → ~0.7 expected).
3. **Spec decode at 27B**: v1 keep D=1/T=2 via the existing pair machinery, eager (no
   graph); `fp8_gemv2` covers pair GEMVs; bf16 ab2 pair for a/b. F2 (optional): D=4/T=5
   generalization per spec-deepen §2/§4 (chain residuals in-place, per-row snapshots
   `snap[t]` t≤T−2 — **host-pinned at 27B**, 628 MB, restore 12 ms hidden; `spec_setup_T`,
   `argmax_rows`, `spec_commit_T` chain rule, `spec_restore_T`; drafts issued while the
   next verify's NVMe fills are in flight). D=4 buys 1.44→2.52 tok/step at p=0.6.
- **Gates**: R7/R8/R9 re-run green through the new prefill; acceptance p ≥ 0.55 measured
  (probe path); no state drift across 100 spec steps.
- **Effort**: ~450 LOC (D=1) + ~300 (optional D=4), 1–1.5 sessions.

### Phase G — performance tuning + placement solver

1. **Startup solver** (~150 LOC): probe free VRAM, pinned-cap ladder, 200 ms UVA-rate
   microbench, `io_bench`-derived NVMe rate → solve the §2 LP (~20 lines: L from VRAM,
   Z = min(pinned/layer, RAM), C = RAM remainder, N = 64−L−Z−C; balance check
   `max-engine ≤ NVMe·1.1`); degrade ladder if the probe returns less (Z→C shift keeps the
   engine alive).
2. Knobs, each behind bench + parity re-run: ring depth K=2–4; read-ahead depth; Z-copy
   128B rebasing at startup load (one-time, free for pinned Z; ring keeps the pad trick);
   split-K GQA + `INSIG_KV_BF16` (ctx sweep 512–4096, attn-27b §6–§7); fp8 GEMM T-skip +
   L2-resident loads for k/v/a/b (fp8-kernels perf §1–2); merged argmax; fused
   `residual+rmsnorm(+bf16 out)` epilogue (insig4-perf §4.3); persistent GEMV.
3. **OPTION-G dual-drive** (§2.4): implement only after user approval — reader drive-map
   (~50 LOC, designed in Phase D), copy script, SMART logging before/after.
4. **R10 re-run** as release gate: tok/s ≥ 80% of the placement prediction, budgets flat.
- **Effort**: ~350 LOC + experiments, 1–2 sessions.

---

## 4. Risk register — top 10 silent-wrong-token risks at 27B

| # | risk | failure mode | caught by |
|---|---|---|---|
| 1 | **Zero-center (1+w) RMSNorms** (qwen35-arch §7.1): input/post/q/k_norm/model.norm/mtp norms are HF-convention (measured negative values); engine multiplies raw | every layer slightly-to-badly wrong; logits plausible | **R4** (layer-0 seam cos collapses), R6 |
| 2 | **GQA `kvh = head/6`** — `head>>2` (9B) mis-groups 16/24 heads; heads 16–23 read the NEXT token's KV rows (silent future-attention) + OOB at last slot | plausible garbage, no crash | unit test_attention (poisoned KV per (t,kvh)); **R5** |
| 3 | **DeltaNet `kh = head/3`** — `head>>1` at 48 v-heads/16 k-heads reads wrong/foreign head state | layer-0 wrong, state diverges | **R4**, test_deltanet |
| 4 | **A_log BF16 read as F32** (decode.cu:77/124 casts) | α ≈ exp(−1e-40) ≈ 1.0 → no forgetting → state saturates SILENTLY | **R4** + explicit α∈(0,1)≠1.0 dump check |
| 5 | **`weight_scale_inv` multiply semantics** (÷ instead of × gives 1e8 magnitudes) | absurd weights or absurd outputs | **R1** (kernel unit vs fixed f64 ref), R4 |
| 6 | **`fp8_gemm` 64-row y contract** (F1: `(void)T`, stores all 64 rows → 21× OOB write at T=3) | heap corruption that poisons unrelated parity runs | **R1** with T-shaped y + max-rel metric (after Phase 0 test fix) |
| 7 | **Ring 16B misalignment** — `data_start ≡ 8 mod 16` ⇒ F8 tensor bases ≡8 in slot ⇒ `uint4` misaligned-address crash (or scalarized 1–3% loss) | crash at first streamed GEMV (loud) or slow | **R3** byte-equality + `(base&15)==0` acquire assert (pad-at-F8-boundary fix in Phase D) |
| 8 | **mtp.fc dims/orientation + concat order** (hardcoded 4096/8192 at decode.cu:150-151; embed-normed half FIRST) | drafts garbage → acceptance collapses (verify keeps output greedy-correct, so it hides from token parity!) | Phase F acceptance probe + R7 draft-path comparison; NOT caught by R9 |
| 9 | **v21 GEMM last-K-step cp.async race** (safety C2) | nondeterministic prefill GEMM corruption — poisons the parity hunt itself | fixed in Phase 0; **R4/R6 determinism across 5 runs** |
| 10 | **KV/committed overrun in the generation loop** (safety C1: unbounded max_new vs ctx 4090/committed 16384) | silent device-heap writes past caches | Phase D device guards + eager throws; **R10** |

Honorable mentions (each has a designed catch): RoPE halves-vs-interleaved convention
(R5×5 runs); conv1d torch-vs-MLX layout (verified identical linearization; R4);
`score[4096]` = context cap ≠ hidden (keep ctx ≤ 4096, safety couple); graph hazards
(**moot — no graphs at 27B**); embed double-read MTP vs pf (10 KB, ignore); fast-math
`__powf/__cosf` in rope (2–3 ulp, precision-note only).

---

## 5. Dependency graph

```
Phase 0 (build+fixes) ─┬─────────────────────────────────────────────┐
                       ▼                                             │
Phase A (loader/INSIDX02) ── R0 ──┐                                  │
                                  ├──► Phase B (shapes) ──┐          │
io_bench (D.1, independent) ──► Phase D (reader/ring) ────┤          │
reference_*_f8.py (E-tools, independent) ─────────────────┼─────────►│
                                                            ▼        ▼
                                              Phase C (kernels) ── R1
                                                            │
                              A+B+C+D ──► Phase E (R3→R4→R5→R6→R7→R8→R9→R10)
                                                            │
                                                    Phase F (prefill+MTP+spec)
                                                            │
                                                    Phase G (solver+tuning+OPTION-G)
```

- **Parallel from day 1**: io_bench + NvReader (needs no engine changes), the five Python
  reference scripts (need only the checkpoint), Phase 0, Phase A.
- **B** is mechanical and can proceed concurrently with A's tail (only R0 needs A done).
- **C kernels** are compile-standalone (unit tests) — can be transcribed while A lands.
- **D** needs A (index/shard table) + io_bench acceptance. **E** needs A+B+C+D.
  **F** needs E (AGENTS: coherent token parity before any perf work). **G** needs F.
- Hard serialized spine: **R0 → R1 → R3 → R4 → R5 → R6 → R7 → R8 → R9 → R10**; a rung below
  threshold blocks everything above it.
- 9B regression gate after every phase (template coexistence + existing bats/mk targets).

---

## 6. Budget

| phase | LOC (new+edited) | sessions | gate |
|---|---|---|---|
| 0 build + blocking fixes | ~300 | 0.5 | 9B suite green, honest test-fp8 |
| A loader | ~600 | 1–1.5 | R0 |
| B shapes | ~450 sites | 1 | unit tests + link close |
| C kernels | ~1,200 | 1.5–2 | R1 + kernel benches |
| D streaming | ~900 | 1.5–2 | R3 + io_bench (E: ≥ 3.0 GB/s) |
| E parity | ~700 (+0–3 hunt) | 1–2 (+0–3) | R4–R9 |
| F prefill/MTP/spec | ~450 (+300 D4) | 1–1.5 | R7–R9 re-run, p ≥ 0.55 |
| G tuning | ~350 | 1–2 | R10, tok/s ≥ 80% predicted |
| **total** | **~4,500–5,000** | **9–13 (+parity hunt)** | R10 = release |

Hardware spends nothing; OPTION-G (if approved) copies 5.3–30 GB to C:.

---

## Appendix A — alternatives considered and rejected

1. **S-tier (staged-copy VRAM ring)**: buys PCIe efficiency (15.9 vs 21.2 ms) on a bus
   that never binds, pays 2×384 MB VRAM = 2 layers = +230 ms/step on E:. Strictly
   dominated (pcie-pipeline §4). Rejected — code exists only as a dormant reader mode.
2. **embed pinned in RAM (2.54 GB)**: costs 6–7 RAM layers ≈ +0.7 s/step to save ~1 ms of
   row-preads. Rejected (placement-final §3, pcie-pipeline §5).
3. **lm_head in RAM + VRAM slot refilled with layers** (embed-lmhead §2.2 scenario B):
   −10% step time on pure bandwidth arithmetic but needs 13.9 GB pinned/RAM — does not
   fit, stacks PCIe 13.9 GB/step on one link, and makes every draft sweep 116 ms.
   Revisit only if RAM grows to 32–64 GB.
4. **CUDA graphs at 27B**: ≤0.3% upside (launch overhead vs 1.6–5 s steps), real
   eviction/pointer hazards. Rejected (graph-hazards §6c, spec-deepen §6). 9B keeps its
   graphs untouched.
5. **OS file cache as N-tier cache**: LRU on a cyclic working set > cache = 0% hits
   (RRIP thrashing), plus standby can never hold the remainder. NO_BUFFERING default,
   buffered twins only for EOF tails / one-shot startup / embed rows (nvme-reader §5.2).
6. **lm_head draft slicing at 27B**: 655 MB VRAM or 30 ms PCIe to replace a 5.4 ms sweep.
   Rejected (embed-lmhead §6); it is the 9B lever, not ours.
7. **qkv+a/b fused GEMV**: saves ≤0.2% for a second dtype inside the hottest kernel.
   Rejected with arithmetic (ab2-redesign §4).
8. **IoRing**: +0–3% at 2 MiB blocks; the bottleneck is the Gen3 drive. Revisit only if
   profiling shows >2% CPU in the ReadFile issue path (nvme-reader §7, web-nvme-win §1).

## Appendix B — OPTION-G approval box (dual-drive E:+C:)

- **What**: copy the N-set layer shards (14–15 shards ≈ 5.3–5.7 GB, or all 64 for
  placement freedom ≈ 24.3 GB) to C: (980 PRO, 93.8 GB free); reader routes per-shard via
  a manifest; time-balanced assignment (C: ~65% of N).
- **Gain**: T_step 1.63–1.75 s → ~0.60–0.70 s; MTP-D1 0.98 → 2.4–2.7 tok/s; MTP-D4
  1.55 → 3.7–4.2 tok/s; prefill(512) ~3.0 s → ~1.5–2.0 s. DRAM becomes co-binding
  (~41 GB/s) — the largest single lever on this rig.
- **Costs/risks**: C: is the SYSTEM drive (read stream at ~100% duty contends with
  pagefile/OS writes; a system-drive problem is a box problem); 980 PRO 500GB is 300 TBW;
  ~6–30 GB permanently consumed; per-drive endurance traffic halves (~6 TB/h each while
  generating).
- **Decision**: ⬜ approved ⬜ refused — engine builds the 50-LOC drive-map hook regardless
  (Phase D); the copy happens only on approval.

## Appendix C — number cross-reference

Per-layer bytes and fixed blocks: loader-27b-spec §0/§5.6, placement-final §1. Pinned cap:
pcie-pipeline §2. Drive map and rates: nvme-reader §1 (authoritative; web-nvme-win §0 got
the drive roles inverted — E: is the Gen3 980, C: the Gen4 PRO). UVA rates: pcie-pipeline
§6. CPU tier: cpu-fp8 §0/§2/§6. MTP/spec math: spec-deepen §1.3 (D-tables), §2 (bookkeeping),
§6 (no graphs, snapshots host-staged at 27B). Prefill: prefill-27b §1/§7. Kernel sources:
attn-27b (attention pack), deltanet-27b (DeltaNet pack), ab2-redesign §3 (a/b pair),
embed-lmhead §4/§7/§9 (bf16 GEMV/embed/GEMM), fp8-kernels (F1–F8 + perf ranking).
Loader: loader-gaps §2/§3/§4/§7/§8. Shapes: shape-constants (with the two `head>>2`
corrections from attn-27b §0). Parity: parity-ladder §6–§8 (commands, thresholds, golden
ids). Safety: safety.md C1–C11, diff-verify H1–H8, graph-hazards. Builds: build-system §4.
