# W3 final placement — Qwen3.8-27B-FP8 (28.75 GiB ckpt, 29.95 GB text) on 4070 SUPER + 5600X + NVMe

Date 2026-08-25. Pure analysis (read-only; no builds, no git). Inputs: `AGENTS.md`,
`audits/synthesis.md`, `audits/w2/loader-27b-spec.md` (byte-exact shard census),
`audits/w3/colibri-sched-deep.md` (pipelining/thread model), `audits/w3/spec-deepen.md`
(MTP calibration). All arithmetic shown; MB = 10^6 B, MiB = 2^20 B.

## 0. TL;DR — THE manifest

**VRAM (L=18): layers 0–17. Pinned-RAM zero-copy (Z=27): 27 of layers 18–63. NVMe ring (N=19): the other 19. lm_head + MTP VRAM-pinned. embed: NOT in VRAM, NOT even pinned in RAM — one 10 KB NVMe row-pread per embed (target + MTP draft share the path).**

| metric | predicted |
|---|---|
| decode, single stream | **0.89 tok/s** (T_step ≈ 1.12 s) |
| decode with MTP (p=0.6, ×1.6/step) | **1.43 tok/s** (p=0.7 → 1.52) |
| prefill, 512-token prompt | **≈1.7 s** floor (1.9–2.1 s w/ checkpoint spill; 3.2 s naive 2-chunk) |
| VRAM used / budget | 10,521 / 10,800 MB (bf16 KV) — 279 MB spare |
| RAM used / usable | 13,158 / 13,500 MB — 342 MB spare |
| binding constraint | NVMe stream: 7,247 MB/token @ 6.5 GB/s = 1,115 ms; everything else hides under it |

```
idx:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15     V = VRAM, Z = pinned-RAM zero-copy,
tier: V  V  V  V  V  V  V  V  V  V  V  V  V  V  V  V     N = NVMe ring slot; * = full-attn
idx: 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31     (i%4==3). Full-attn: V*, Z*, N*
tier: V  V  N  Z* N  Z  Z  N* Z  N  Z  Z* N  Z  N  Z*
idx: 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
tier: Z  N  Z  N* Z  N  Z  Z* N  Z  N  Z* N  Z  Z  N*
idx: 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63
tier: Z  N  Z  Z* N  Z  N  Z* N  Z  Z  N* Z  N  Z  Z*
```
N = {18,20,23,25,28,30,33,35,37,40,42,44,47,49,52,54,56,59,61} (15 lin + 4 full),
Z = {19,21,22,24,26,27,29,31,32,34,36,38,39,41,43,45,46,48,50,51,53,55,57,58,60,62,63}
(19 lin + 8 full), V = 0–17 (14 lin + 4 full: 3,7,11,15).

---

## 1. Byte-exact inputs (from loader-27b-spec, §0/§2)

| object | exact bytes | MB |
|---|---|---|
| linear layer (48x) | 382,730,240 F8 + 1,132,608 BF16 | 383.86 |
| full-attn layer (16x) | 372,244,480 F8 + 66,944 BF16 | 372.31 |
| MTP shard (fc + 1 full layer) | 477,199,744 | 477.20 |
| lm_head bf16 [248320,5120] | 2,542,796,800 | 2,542.80 |
| embed bf16 | 2,542,796,800 | 2,542.80 |
| final norm | 10,240 | 0.01 |
| text weights total | 29,945,203,072 | 29,945 |
| delta state / linear layer f32 | 48·128·128·4 | 3.15 |
| conv state / linear layer | 10240·3·4 | 0.12 |
| KV / full layer @ctx2048 | 2·2048·4·256·4 (f32) / ·2 (bf16) | 16.78 / 8.39 |

Rig: 12,282 MiB = 12,880 MB VRAM, **app budget ≤ 10,800 MB** (WDDM headroom);
15.9 GiB RAM → **usable ≈ 13,500 MB**; NVMe 6.5 GB/s effective; PCIe uplink ≈ 22 GB/s
(kernel UVA reads of pinned host memory).

## 2. Per-layer per-token cost model (decode, bandwidth-bound GEMV; recomputed)

**Correction to synthesis.md confirmed:** synthesis's "VRAM 0.76 + 0.6 ms state" is wrong
per-layer. State traffic is 3.15 MB read + 3.15 MB written = 6.29 MB **per layer** →
6.29 MB / 504 GB/s = **12.5 µs/layer**. Synthesis's 0.6 ms is the 48-layer *total*
(48 × 6.29 MB = 302 MB → 0.6 ms across the whole stack).

| tier | linear (383.86 MB) | full-attn (372.31 MB) | arithmetic |
|---|---|---|---|
| VRAM @504 GB/s | **0.78 ms** | **0.77 ms** (f32 KV) / 0.757 (bf16) | 382.73/504 = 0.759 + state 6.29/504 = 0.0125 + elementwise ~0.01; full: 372.24/504 = 0.739 + KV 16.78/504 = 0.033 |
| zero-copy pinned RAM @22 GB/s | **17.45 ms** | 16.92 ms (+0.38 if KV in RAM bf16) | 383.86/22 = 17.45; state r/w over PCIe 6.29/22 = 0.29. Choice of 22: PCIe4 x16 pinned UVA reads cap ~20–25 GB/s; 22 is the measured-adjacent midpoint (sens. 15/20/25 in §6) |
| CPU AVX2 @37 GB/s | **10.8 ms** | ~10.9 ms | 383.86/37 = 10.37 + state/KV DRAM 0.17 + handoff/sync ~0.4 (synthesis's 9.6 @40 GB/s; 37 conservative for Zen3 dual-channel DDR4-3200) |
| NVMe-fed @6.5 GB/s | **59.1 ms** | 57.3 ms | 383.86/6.5 = 59.06. Consumption (CPU 10.8 or Z 17.4) overlaps the *next* layer's fill — hidden (both < 59.1) |

- lm_head sweep: 2,542.80/470 GB/s (bf16 GEMV realistic) = **5.41 ms** (5.05 at 504).
- MTP draft step = fc 104.86/470 = 0.22 + layer 372.31/504 = 0.74 + KV 8.39/504 = 0.02
  + embed row (10 KB UVA/pread) ≈ 0.01–0.10 + lm_head 5.41 + elementwise 0.25 ≈ **6.7 ms**.
- Verify (T=2) re-reads weights once (bandwidth-bound ⇒ 2nd row free) + one lm_head sweep.

## 3. Task 1 — VRAM ledger → L_max = 18

Fixed block: lm_head 2,542.80 + MTP 477.20 + MTP KV 8.39 (bf16; 16.78 f32) + norm 0.01
+ workspace 150 + CUDA context/graphs 400 = **3,578.4 MB** (bf16) / 3,586.8 (f32).

**embed decision — verified and pushed one step further.** 10 KB/token UVA read is
indeed ≈free, so embed never needs VRAM. But *pinned-RAM* embed costs 2,542.8 MB of the
RAM budget = 6.6 RAM-resident layers; an NVMe row-pread (10 KB, ~60–120 µs, issued while
the streaming reader is otherwise saturating the disk; can be prefetched for the draft)
costs ~0.1–1 ms/token. Moving 6–7 layers N→RAM saves 6 × 48–57 ms/token. **Winner: NVMe
row-pread** (§4 quantifies: 0.89 vs 0.63 tok/s). MTP's `x.down` embed gather uses the
same path — same zero VRAM, same row-pread ✓.

Per-layer VRAM cost (weights + state, KV follows layer): linear 383.86+3.15+0.12 =
**387.13**; full bf16-KV 372.31+8.39 = **380.70**; full f32-KV 389.09.

Layer-count solve, budget = 10,800 MB, natural mix (every 4th layer is full-attn):

| L | composition | total (bf16 KV) | fits? |
|---|---|---|---|
| 18 | 14 lin + 4 full: 14·387.13 + 4·380.70 + 3,578.4 = 5,419.8+1,522.8+3,578.4 | **10,521.0** | ✓ (279 spare) |
| 19 | +1 layer (best case all-lin +387.13) | 10,908.2 | ✗ (108 over; f32-KV: 10,979) |

**L_max = 18.** A 19th layer needs a 10.91 GB budget — inside the 12.88 GB physical card
but outside the WDDM-safe 10.8 GB target; listed as a stretch variant (§8).

Ordering inside VRAM: full-attn is cheaper per token (0.757 vs 0.78 ms) but also ~equal
per MB (0.757/380.7 = 1.99 µs/MB vs 0.78/387.1 = 2.01 µs/MB — 1% apart). **No meaningful
preference at ctx 2048**; take contiguous 0–17 (contains 4 full layers naturally).
The real full-attn caveat is ctx growth: a VRAM full layer locks +8.39 MB VRAM per 2048
ctx (bf16); linear layers lock nothing. At ctx ≥ 8k prefer linear layers for VRAM.

Staging slots: **0**. The N-tier feeds a *pinned-RAM* ring consumed by UVA zero-copy
reads (or CPU) — no VRAM staging, no double-buffer uploads. VRAM streaming slots
(2 × 384 MB) would only exist if NVMe/PCIe streamed *into VRAM* — rejected: it would put
46×384/22 = 803+ ms/token on the PCIe uplink that zero-copy weights already use.

## 4. Task 2 — RAM ledger and the LP

Usable 13,500 MB. Claims: Z weights + Z/N states&KV + ring + host buffers + OS margin.

- Z weights (19 lin + 8 full): 19·383.86 + 8·372.31 = 7,293.3 + 2,978.5 = **10,271.9**
- Z/N full-attn KV bf16 in RAM (12 × 8.39): **100.7** (f32: 201.3)
- Z/N delta states 23·3.15 + conv 23·0.12: **75.2**
- NVMe ring 4 slots × 384: **1,536** (treadmill needs ≥2–3; 4 = jitter margin)
- host activations/buffers: 150; OS margin: 1,024
- **Total 13,157.7 → spare 342.3** ✓ (f32 KV parity run: 13,258 → spare 242 ✓)

Cap check: 13,500 − 2,048(5-slot ring) − 1,024 = 10,428 ≈ 27.1 layers → **Z ≤ 27**
(prompt's bound). With embed *pinned* instead: 10,428 − 2,542.8 = 7,885 → only 20 layers
→ N grows to 26 → NVMe 26·~376/6.5 = 1,504 ms → 0.63 tok/s single / 1.01 MTP. **NVMe
row-pread embed: 0.89 / 1.43.** Verified: embed belongs on NVMe (row reads), not in RAM,
not in VRAM.

### The LP and the pipelined ("treadmill") solution

Naive serial LP (prompt's model, N-consumption hidden only):
T = 18·0.78 + 27·17.45 + 19·59.05 + lm_head 5.41 = 14.0 + 471.1 + 1,122.0 + 5.4 =
**1,612.6 ms** → 0.62 tok/s single, 0.99 MTP.

But the layers are interleaved (colibri-sched-deep §8.1 Correction 1), so the NVMe
reader is a *pace car*, not a blocking fetch: between two N-layers the consumer does
~2.4 non-N layers ≈ 14–31 ms ≪ 59.1 ms, i.e. the consumer always arrives early and waits
for the reader. All compute hides inside NVMe waits:

**T_step = max(NVMe stream, GPU-side serial work, PCIe uplink, DRAM) + tail**

- NVMe stream: 7,247.1 MB/token ÷ 6.5 GB/s = **1,114.9 ms** ← binding
- GPU-side serial: VRAM 14·0.78+4·0.757 = 14.0; Z 27·~17.6 = 475; N-consume 19·17.4 = 331;
  lm_head 5.4 + draft 6.7 → **≈ 832 ms** ✓ hides (75% of budget)
- PCIe uplink: Z 10,271.9 + ring reads 7,247.1 + KV/states ~250 = 17,769 MB ÷ 22 = **807 ms**
  (72% util) ✓
- DRAM: ring writes 7.25 + root-complex reads 10.3 GB ÷ ~44 achievable ≈ 400 ms ✓
- Tail (not hidden): embed row pread 0.1–1.0 + event/skew ~3–5 ms.

**T_step ≈ 1,120 ms → single 0.89 tok/s; MTP tokens/step = 1+p = 1.6 → 1.6/1.120 = 1.43 tok/s.**

Water-filling view (why this is optimal): every layer sits at a capacity bound —
VRAM full at L=18, RAM full at Z=27, residual N=19. Marginal moves: N→Z −57 ms/token,
Z→V −16.7 ms/token; both blocked by hardware capacity. The MTP draft (6.7 ms GPU) and
verify lm_head (5.4 ms) also hide under the treadmill (ring = 4 slots × 59.2 = 237 ms of
buffer ≫ 12.1 ms burst).

Key consequence: **CPU-tier kernels (v2) change nothing at this split** — CPU serial
would drop 475→291 ms, still ≪ 1,115. v1 zero-copy-only is sufficient; CPU GEMV is a
robustness upgrade (frees PCIe 807→~0 ms, removes DRAM/root-complex contention risk),
not a speed upgrade, until NVMe ≥ ~13 GB/s or N shrinks below ~12.

## 5. Solved split recap

L=18 (V, 0–17), Z=27 (pinned, zero-copy consumed), N=19 (NVMe ring, UVA-consumed from
pinned slot; fill 59.1 ms overlaps next fills via 4-slot ring). All non-NVMe work
overlaps the stream. Per-token bytes: NVMe 7,247 MB; PCIe 17.8 GB (72%); DRAM ~17.7 GB
(~40% of 44 GB/s achievable).

## 6. Task 3 — sensitivity (best split unchanged in every row: L18/Z27/N19)

| knob | value | binding resource | T_step ms | single tok/s | MTP tok/s (×1.6) |
|---|---|---|---|---|---|
| NVMe | 5.0 GB/s | NVMe 1,449 | 1,454 | 0.69 | 1.10 |
| NVMe | **6.5 GB/s** | NVMe 1,115 | **1,120** | **0.89** | **1.43** |
| NVMe | 8.0 GB/s | NVMe 906 | 911 | 1.10 | 1.76 |
| Z-read | 15 GB/s | **PCIe 1,184** > NVMe | 1,189 | 0.84 | 1.35 |
| Z-read | 20 GB/s | NVMe | 1,120 | 0.89 | 1.43 |
| Z-read | 25 GB/s | NVMe (PCIe 716) | 1,120 | 0.89 | 1.43 |
| CPU tier (v2) | 30 GB/s | NVMe (CPU serial 360) | 1,120 | 0.89 | 1.43 |
| CPU tier (v2) | 37 GB/s | NVMe (CPU serial 306) | 1,120 | 0.89 | 1.43 |
| CPU tier (v2) | 45 GB/s | NVMe (CPU serial 245) | 1,120 | 0.89 | 1.43 |
| accept | 1.4 tok/step | — | 1,120 | — | 1.25 |
| accept | 1.6 tok/step | — | 1,120 | — | 1.43 |
| accept | 1.8 tok/step | — | 1,120 | — | 1.61 |

Read: the design is NVMe-bound with 25–30% slack on every other resource; only a
pessimistic 15 GB/s zero-copy path flips the bottleneck to PCIe (still only −6%).
MTP upside dominates the user-visible range (accept 1.4→1.8 = +27%).

## 7. Task 4 — prefill (512 tokens, chosen split, layer-major weight-stationary)

Weights are read **once** (not per token); compute per layer = 2·512·384e6 = 393 GFLOP
≈ 2.8–3.9 ms at 100–142 TFLOPS FP8 ≪ every read tier:

| segment | per layer | total | arithmetic |
|---|---|---|---|
| 18 VRAM layers | max(0, 3.9 ms compute) | 70 ms | resident; compute-bound |
| 27 Z layers | 383.86/22 = 17.4 (UVA GEMM, no staging — VRAM spare 279 < 384 so no H2D copy possible anyway) | 470 ms | PCIe-bound |
| 19 N layers | 383.86/6.5 = 59.1 | 1,125 ms | NVMe-bound |
| lm_head T=512 | 2·512·2.543e9 = 2.60 TFLOP @ ~71–100 TF | 26–37 ms | resident |
| embed 512 rows | 512 × 10 KB preads, QD8 | ~10 ms | |
| **total** | | **≈ 1,700 ms** | weight-read-bound |

Activation checkpoints (512×5120×4 = 10.5 MB per layer boundary → 671 MB for 64 layers)
exceed the 342 MB RAM spare. Options: (a) NVMe-spill checkpoints: +671 MB write + read
back @6.5 = +206 ms → **≈1.9 s**; (b) 2×256-token chunks, weights read twice:
2·(470+1,125) ≈ **3.2 s** (v1-simple fallback); (c) bf16 checkpoints + 2 chunks → 336 MB,
fits spare, still 3.2 s. Recommend (a); quote **prefill 512 ≈ 1.7 s floor / ~2 s real**.

## 8. Task 5 — KV dtype and ctx

- **f32 KV for v1 parity** (the full-attn parity bug hunt needs reference-matching math);
  fits: f32 ledger = 10,563.0 MB (237 spare) with Z/N KV in RAM (201.3 MB, spare 242).
- **bf16 flag saves 8.39 MB × VRAM-resident full layers = 5 × 8.39 = 42 MB** (layers
  3,7,11,15 + MTP). With Z/N KV kept in VRAM instead of RAM it would save 12 × 8.39 =
  100.7 more (143 total) — our manifest keeps Z/N KV in RAM, so the flag's VRAM value is
  42 MB: **does not change L_max** (18 either way), but buys ctx headroom: 4 VRAM full
  layers + MTP reach ctx ~8k bf16 within the 279 MB spare (vs ~5k f32).
- RAM side bf16 saves another 100.7 MB (12 Z/N layers).
- **ctx cap 2048 default** ✓: all KV totals are trivial (17 layers × 8.4–16.8 MB =
  143–285 MB); max_position 262,144 is unreachable on this rig regardless — the 59 ms
  NVMe layer makes a 256k-context chat... an experience.
- DeltaNet f32 states stay f32 (mamba_ssm_dtype: float32 — config-mandated).

## 9. Task 6 — THE manifest (final)

| tier | count | indices | bytes resident | per-token cost |
|---|---|---|---|---|
| V (VRAM) | 18 | 0–17 | 6,942.6 MB (+ fixed 3,578.4 = 10,521.0 / 10,800) | 0.78/0.757 ms each = 14.0 ms (hidden) |
| Z (pinned RAM, UVA) | 27 | 19,21,22,24,26,27,29,31,32,34,36,38,39,41,43,45,46,48,50,51,53,55,57,58,60,62,63 | 10,271.9 MB | 17.4–17.8 ms each (hidden) |
| N (NVMe ring) | 19 | 18,20,23,25,28,30,33,35,37,40,42,44,47,49,52,54,56,59,61 | 0 (7,247.1 MB/token streamed) | **59.1 ms each = 1,115 ms (binding)** |
| VRAM-fixed | — | lm_head, MTP(fc+layer+KV), final norm, workspace, ctx | 3,578.4 MB | lm_head 5.4 + draft 6.7 (hidden) |
| RAM-fixed | — | ring 4×384, Z/N KV+states, buffers, OS | 2,885.8 MB | — |
| NVMe-resident | — | embed (row-pread 10 KB/embed), all 66 shards | 0 | ~0.1–1 ms/embed |

Placement rules used: VRAM contiguous at the *start* (no IO cadence concern; execution
order doesn't change the serial sum); Z/N **interleaved** in 18–63 with N every ~2.4
layers so reader cadence ≈ 59 ms matches consumption (colibri-sched Correction 1);
full-attn layers follow their tier (KV with layer: V→VRAM, Z/N→pinned RAM).

**Predictions: prefill(512) ≈ 1.7–2.0 s · decode single 0.89 tok/s · MTP 1.43 tok/s
(p=0.6) / 1.52 (p=0.7 after spec-deepen F7 KV-hole fix).**

Stretch variants (each +7%, WDDM/RAM-risky, off by default):
- Z=28, N=18 (ring 3×384): RAM 13,160.8 ✓ → NVMe 6,863 MB → T 1,061 ms → 0.94 / 1.51.
- L=19 (needs 10,908 MB = 10.91 GB budget, over the 10.8 target): N=18 → 0.94 / 1.51.
- Both together (L19/Z28/N17): T ≈ 1,002 ms → ~1.0 / 1.6 — the "if you like living
  dangerously" config; requires WDDM commit ≥ 10.9 GB AND ring 3 AND trimmed OS.

Risks / notes:
- WDDM: 12,282 MiB reported; 10,521 MB commit leaves 2.3 GB for driver/display — safe;
  the L=19 stretch is the only line that gambles here.
- Reader team: 3 threads (range 2–4), O_DIRECT twin-fd, QD 8–16 (colibri-io numbers);
  ~1.5–2 cores, leaving 4C/8T+ for the OMP team if CPU tier lands in v2.
- First token after prefill is warm (prefill leaves the ring full) — no cold-start 139 ms
  surprise (colibri measured lesson).
- MTP draft lm_head sweep (5.41 ms) is hidden today; spec-deepen's sliced head (0.83 ms
  at 9B scale) is irrelevant to throughput here but halves draft *latency* for
  interactive first-token feel.
- The engine's existing budgeted-residency layer (AGENTS.md) maps onto this directly:
  pin list = V∪fixed, pinned-RAM set = Z, ring = existing LRU machinery in window mode.

## 10. Verification checklist for implementation

1. Per-token NVMe bytes observed = 7,247 MB ± shard mix (15 lin + 4 full).
2. Ring slot dwell ≈ 59 ms; consumer wait on N-layers > 0 (pace-car signature).
3. PCIe GPU util counter ≈ 70–75% during steady decode (807/1,120).
4. VRAM commit 10,521 MB; RAM working set 13,158 MB.
5. Prefill single weight sweep: total NVMe reads ≈ 19 shards + Z/V never re-read.
