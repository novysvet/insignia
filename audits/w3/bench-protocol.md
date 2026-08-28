# The Insignia measurement protocol — w3 (2026-08-25)

Every future performance claim in this repo follows this document. Root cause it exists:
the audit wave found that every on-disk GEMV bench at <=12288 rows ran with the whole
weight matrix (8-47 MB) resident in the 48 MB L2 across 100 back-to-back reps — those
"150 GiB/s" numbers are L2 fiction. Production decode streams 4.99 GB/step through L2;
between two uses of any weight matrix, L2 is flushed many times over. Benches must
reproduce that condition or say loudly that they didn't.

Hardware anchor (audits/w3/insig4-perf.md:10-13): RTX 4070 SUPER, 56 SMs, L2 = 48 MB,
DRAM peak **504 GB/s** (audit contract; see §6.1 for the one-time calibration that
re-anchors every "%" claim). smem 100 KB/SM opt-in, bf16 tensor ~71-83 TFLOPS, fp8 MMA
~142 TFLOPS.

**Units rule:** all GB/s are decimal (1e9 B/s). 504 GB/s = 469.6 GiB/s. Old logs print
GiB/s — multiply by 1.0737 before comparing (156.2 GiB/s = 167.6 GB/s).

---

## 1. The L2-flush protocol (mandatory for every timed kernel rep)

### 1.1 Why both halves are needed

- A plain 256 MB read sweep displaces normal L2 lines but **cannot evict persisting
  lines** parked by an `accessPolicyWindow` (insig4-perf §2.3 experiments) — persisting
  lines resist normal eviction. `cudaCtxResetPersistingL2Cache()` demotes them first.
- `cudaCtxResetPersistingL2Cache()` alone does NOT flush ordinary L2 contents — it only
  resets the persistence attribute. The sweep does the displacement (256 MB = 5.33x L2).

Note: `cudaCtxResetPersistingL2Cache` takes **no arguments** (the sketch in
insig4-perf.md §2.4 wrote `(stream)` — that does not compile; the runtime signature is
`cudaError_t cudaCtxResetPersistingL2Cache(void)`).

### 1.2 The flush kernel (verbatim, add once to the bench harness)

```cuda
// 256 MB evict-first read sweep: displaces all 48 MB of L2; __ldcs lines never stick.
__global__ __launch_bounds__(256) void l2_flush_sweep(const float4* __restrict__ p,
                                                      size_t n4,
                                                      float* __restrict__ sink) {
    float4 a{0.f, 0.f, 0.f, 0.f};
    for (size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; i < n4;
         i += size_t(gridDim.x) * blockDim.x)
        a = __ldcs(p + i);                        // pure evict-first read stream
    if (a.x == 12345.678f) *sink = a.x + a.w;     // defeat DCE (never true)
}
```

Host setup once per process:

```cpp
float4* flush_buf; float* sink;
cudaMalloc(&flush_buf, size_t(256) << 20);        // 268,435,456 B = 64 Mi float4
cudaMalloc(&sink, 4);
cudaMemset(flush_buf, 0xCD, size_t(256) << 20);   // data irrelevant, pages mapped
const size_t flush_n4 = (size_t(256) << 20) / 16;
```

### 1.3 The exact per-rep sequence

```
// per timed COLD rep:
cudaCtxResetPersistingL2Cache();                                    // 1. demote persisting
l2_flush_sweep<<<56 * 8, 256, 0, stream>>>(flush_buf, flush_n4, sink); // 2. 448 blocks
cudaStreamSynchronize(stream);                                      // 3. L2 = flush data only
cudaEventRecord(t0, stream);
launch_kernel_under_test(...);                                      // 4. exactly ONE launch
cudaEventRecord(t1, stream);
cudaEventSynchronize(t1);
cudaEventElapsedTime(&ms_rep, t0, t1);
```

Rules:

1. **N=5 flushed reps, report the median** (plus min/max so spread is visible). Sweep
   cost ~0.5-0.7 ms/rep — total overhead < 5 ms per shape. Never average.
2. **Warm number, separately:** N=5 back-to-back reps with NO flush after one throwaway
   launch, median, always reported next to cold. A shape whose weight bytes < 48 MB must
   carry the suffix `[L2-RESIDENT]` on the warm (and any unflushed) number. Between
   48-96 MB (2x L2) label `[L2-PARTIAL]`. > 96 MB may go unflushed but flushing is
   cheap — do it anyway.
3. **NEVER report an unflushed number for a < 48 MB shape without the `[L2-RESIDENT]`
   label.** This is the whole point: the old bench loops
   (src/bench_mxfp4_mlx.cu:495 `for (100) run();`, src/bench_gemm.cu:58 `for (30)`) are
   back-to-back loop-averages — for <=12288-row shapes every rep after the first hits
   L2. The protocol replaces those loops.
4. One **untimed correctness launch** before timing (already in both benches for the
   host reference check) doubles as WDDM/driver warmup — first launch in a process pays
   module load + WDDM submission.
5. **Event-bracket overhead is 2-4 µs.** Fine for every matrix cell (>= 33 µs); note it
   on the 6144-row i4 cells where it is ~10% of the floor.
6. Record machine state with every table:
   `nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv`
   before and after the run. `-lgc` clock-locking usually fails on consumer WDDM —
   don't rely on it; do rely on 5-rep medians after the throwaway launch.
7. **Flush self-check (run once when the harness lands):** cold 4096x4096 i4 GEMV should
   drop far below the old 150 GiB/s toward the BW floor (8.7 MB → 17 µs @ 504). If
   cold ~= warm ~= 54 µs, either the flush is broken or the kernel is genuinely
   latency/launch-bound at that size — now you can tell which, and both facts get
   reported.
8. T=2 pair kernels get benched twice, labeled: `[FUSED-QUANT]` (today's
   `mxfp4_gemv2_q8_i4` re-quantizes both x rows in every one of the 31040 blocks —
   that per-block quantization is part of production cost) and `[PREQUANT]`
   (`quantize_x8` + pair kernel, the §2.2 insig4-perf redesign).

### 1.4 What "cold" means here

Cold = the weight matrix is NOT in L2 at kernel start (it is in DRAM/VRAM pages — the
checkpoint is already resident; we never measure PCIe/NVMe upload inside a kernel bench;
upload paths are benched separately per audits/w3/io-bench-results.md). This matches
production decode: any given weight tensor is touched once per 4.99 GB of traffic.

---

## 2. The 27B-shape bench matrix

### 2.1 Kernel families under test (current source of truth)

| family | kernel | file:line | T |
|---|---|---|---|
| fp8_gemv | `fp8_gemv_kernel` | src/fp8.cu:14 | 1 |
| fp8_gemv2 (pair) | `fp8_gemv2_kernel` | src/fp8.cu:58 | 2 |
| fp8_gemm | `fp8_gemm_kernel` | src/fp8.cu:109 | 4/16/64 (T<=64 hard cap) |
| i4 gemv | `mxfp4_gemv_v2_i4_kernel` | src/mxfp4_i4.cu:16 | 1 |
| i4 gemv2 (pair) | `mxfp4_gemv2_q8_i4_kernel` | src/mxfp4_i4.cu:78 | 2 |
| i4 gemm | `mxfp4_gemm_mlx_i4` (v1 clone; v21-i4 pending) | src/gemm.cu:302 | 4/16/64 |
| bf16 lm_head GEMV | `bf16_gemv_kernel` (1 block/row, 256 thr) | src/qwen_kernels.cu:67 | 1 (pair = 2nd sweep) |

Shapes (rows x cols, Qwen3.8-27B real projections): 10240x5120 (in_proj_qkv),
6144x5120 (in_proj_z), 5120x6144 (out_proj), 17408x5120 (gate/up), 5120x17408 (down),
12288x5120 (q_proj q+gate), 248320x5120 (bf16 lm_head). Transposes (RxC vs CxR) have
identical byte counts but different kernel behavior (row count sets the grid; col count
sets smem staging) — that is exactly why both orientations are in the matrix.

### 2.2 Weight-byte formulas (verified from source, not from memory)

- **F8 (e4m3 + BF16 tile scales):** `W = rows*cols` + `scales = (rows/128)*(cols/128)*2`
  (fp8.cu:40-41: `row_s = scales + (row>>7)*kblocks`, kblocks = cols>>7).
- **INSIG4 (E2M1 nibbles + F16 super-group scales):** `W = rows*cols/2` +
  `scales = rows * (cols/64) * 2` (= **cols/32 B per row**). Verified: quantizer writes
  `.scales F16 [rows, cols/64]` (tools/quantize_insig4.py:64,129); kernel indexes
  `row_s = scales + row*(groups>>1)` u16 (src/mxfp4_i4.cu:36, groups = cols/32).
  NOTE: the task brief's "+cols/128*2B per row" (= cols/64 B/row) undercounts by 2x;
  the code is the truth. Effective INSIG4 density = 0.53125 B/elt (17/32).
- **bf16:** `W = rows*cols*2`.

### 2.3 Expected-time tables — computed at 504 GB/s so deviations are visible

GEMV/gemv2 read the weight matrix exactly once regardless of T=1/2, so their floor is
bytes/504GB/s. GEMM floor = max(bandwidth floor, compute floor at 83 TF bf16-rate
dequant paths; 142 TF = native fp8 MMA future path). FLOPs = 2*T*rows*cols.

**fp8 family (bytes include scales):**

| shape | bytes (MB) | L2 class | gemv/gemv2 floor (µs) | gemm T=4 | T=16 | T=64 |
|---|---|---|---|---|---|---|
| 10240x5120 | 52.435 | partial | 104.0 | 104.0 | 104.0 | 104.0 (comp 80.9) |
| 6144x5120 | 31.461 | **<48MB** | 62.4 | 62.4 | 62.4 | 62.4 (comp 48.5) |
| 5120x6144 | 31.461 | **<48MB** | 62.4 | 62.4 | 62.4 | 62.4 |
| 17408x5120 | 89.140 | partial | 176.9 | 176.9 | 176.9 | 176.9 (comp 137.5) |
| 5120x17408 | 89.140 | partial | 176.9 | 176.9 | 176.9 | 176.9 |
| 12288x5120 | 62.922 | partial | 124.8 | 124.8 | 124.8 | 124.8 (comp 97.0) |

**INSIG4 family (bytes include scales):**

| shape | bytes (MB) | L2 class | gemv/gemv2 floor (µs) | gemm T=4 | T=16 | T=64 |
|---|---|---|---|---|---|---|
| 10240x5120 | 27.853 | **<48MB** | 55.3 | 55.3 | 55.3 | **80.9 (compute-bound)** |
| 6144x5120 | 16.712 | **<48MB** | 33.2 | 33.2 | 33.2 | **48.5 (compute)** |
| 5120x6144 | 16.712 | **<48MB** | 33.2 | 33.2 | 33.2 | **48.5 (compute)** |
| 17408x5120 | 47.350 | **<48MB (!)** | 93.9 | 93.9 | 93.9 | **137.5 (compute)** |
| 5120x17408 | 47.350 | **<48MB (!)** | 93.9 | 93.9 | 93.9 | **137.5 (compute)** |
| 12288x5120 | 33.423 | **<48MB** | 66.3 | 66.3 | 66.3 | **97.0 (compute)** |

The 17408x5120 i4 matrix is 47.35 MB — it *just* fits 48 MB L2, i.e. even the "big" i4
shapes are fiction-prone. **Every cell in this matrix except bf16 lm_head requires the
§1 flush.** Compute-bound cells at T=64: expected *effective* GB/s = bytes/floor =
~344 GB/s — an "effective BW" drop there is arithmetic intensity, not a bug.

**bf16 lm_head (248320x5120):** 2542.8 MB → **5045 µs** per sweep at 504 GB/s, at every
T (compute floor at T=64 is 1960 µs < BW floor; crossover for bf16 is T≈165). Today's
spec step does TWO sweeps (draft + verify): expected ≥ 10.09 ms/step just for lm_head —
the single biggest known lever (fuse draft sweep into verify: −5.07 MB... −2543 MB ≈
−5.0 ms/step).

**Crossover T (where GEMM leaves the BW floor), same for all shapes of a family:**
i4: T>44 (83 TF) / T>75 (142 TF); fp8-dequant: T>82; bf16: T>165.

### 2.4 Reporting format per cell

```
<family> <rows>x<cols> T=<t> [flags]: bytes=NN.N MB  expected=NN.N µs @504
  cold: median mm.m µs (min..max)  -> achieved NNN GB/s (NN.N% of measured_read_ceiling)
  warm: median mm.m µs [L2-RESIDENT|L2-PARTIAL|warm]
```

Health bands for cold achieved GB/s (vs measured_read_ceiling, §6.1): >= 90% tuned;
85-90% healthy; 70-85% investigate (occupancy/tail/launch); < 70% broken — find it
before optimizing anything downstream.

**Forbidden:** benching T>2 as T back-to-back GEMV launches and calling it a GEMM
number (that is T× the weight traffic — the decode.cu a/b anti-pattern). If measured,
label `[T-LOOP GEMV]`.

---

## 3. Baseline extraction — what the logs actually say, with trust labels

Searched: build/*.log (only `last-test.log` contains timing; `i4-ref.log` and
`multistep-parity.log` are parity-only), audits/internals.md, audits/w3/*.md, and the
nsys artifacts (build/spec2/spec3/spec_prof.nsys-rep + .sqlite — captures exist, no
exported numbers on disk).

| # | metric | value | provenance | verdict |
|---|---|---|---|---|
| 1 | 9B spec decode throughput | **121 tok/s = 8.26 ms/token** (13.2 ms/step, 1.6 tok/step @ p≈0.6) | src/generate.cu:178-207 CUDA events around graph-replay loop, warm, all-VRAM 9B | **TRUSTWORTHY** end-to-end (includes host sync every 4 steps). Implies 4.99 GB/step at 378 GB/s = 75% of 504 |
| 2 | 9B single-token (pre-spec) | 11.8 ms/token (≈83-85 tok/s), 4.33 GB | event timing; anchors in audits/w3/spec-deepen.md:85 | **TRUSTWORTHY** (superseded by #1) |
| 3 | lm_head GEMV 248320x4096 (e8m0) | **379 GiB/s = 407 GB/s = 75%** of peak | internals.md:18,28 (bench of commit 4bd0513 era); 508 MB matrix ≫ L2 | **TRUSTWORTHY** streaming |
| 4 | pair GEMV streaming shapes | 379-435 GiB/s (407-467 GB/s) | internals.md:18; spec-deepen.md:81 | **TRUSTWORTHY** (all > L2) |
| 5 | GEMV 4096x4096 warps=1/2/4/8 | 156.2/149.6/149.3/150.1 GiB/s | build/last-test.log:20-23; weights 8.25 MB ≪ 48 MB L2, 100-rep loop | **L2 FICTION** — re-run under §1 before citing |
| 6 | legacy GEMV 257x4096 | 0.010 ms, 54.0 GiB/s | build/last-test.log:14; 0.53 MB matrix | L2-resident AND launch-bound — measures launch+staging latency, not BW; cite only as "launch floor ≈ 10 µs" |
| 7 | i4 GEMM (any shape), v2/v21 GEMM T=64 | **no timing exists on disk** | i4-ref.log is parity-only; insig4-perf.md:7-8 confirms | **NO DATA** — the 250-350 µs @T=64 figures in insig4-perf §3.3 are analytic estimates |
| 8 | fp8 kernels (27B path) | none | no fp8 bench exists (test-fp8.bat is correctness-only) | **NO DATA** — §2 matrix is the first measurement |
| 9 | parity e8m0 multistep | worst_layer_cos 0.99978 | build/multistep-parity.log | parity gate reference (PASS) |
| 10 | parity INSIG4 multistep | worst_layer_cos **0.97229** (step 5, layer 26) | build/i4-ref.log | **GATE RED** — known full-attn issue (AGENTS.md); A/B rule applies until fixed |
| 11 | NVMe E:/C: streaming | 3.25 / 6.50 GB/s (dual 9.7) | audits/w3/io-bench-results.md — FILE_FLAG_NO_BUFFERING, cold by construction | **TRUSTWORTHY** (different subsystem; the model for honest methodology) |
| 12 | launch census | 554/token single (313 elementwise), ≈610/spec step | static count from src (insig4-perf.md:38-41), not nsys | trustworthy as count; re-derive via §4 KPI-3 |

Standing end-to-end targets derived from the honest numbers (insig4-perf §0): GEMV
family 75% → 90% of peak = 13.2 → ~11 ms/step ≈ 145 tok/s ceiling for kernel work; all
bigger wins are traffic elimination (lm_head dedup, deeper verify), not GB/s.

---

## 4. nsys / ncu protocol

Two tools, two jobs. NSYS = timeline + launch counts (matches the existing
build/spec3.nsys-rep artifacts). NCU = per-kernel hardware counters (its default
`--cache-control=all` flushes caches between replay passes, so NCU *bytes* counters are
cold-ish by construction — but replay distorts wall time: **never take ms from NCU**).

### 4.1 Capture (spec-style, on any new target)

```
:: timeline + DRAM bandwidth timeline (--gpu-metrics-device=0 is what makes cold/warm
:: visible end-to-end without touching kernels):
nsys profile --trace=cuda --sample=none --cpuctxsw=none --gpu-metrics-device=0 ^
    -o build/spec4 build\generate.exe <args>

:: per-kernel summary (time, instances) + export for SQL:
nsys stats -r cuda_gpu_kern_sum,cuda_gpu_mem_time_sum build\spec4.nsys-rep
nsys export -t sqlite -o build\spec4.sqlite build\spec4.nsys-rep
```

Same pattern for benches: `nsys profile ... build\bench-mxfp4.exe` etc. One capture per
optimization PR, artifact path cited in the commit message.

### 4.2 Counters (NCU) — the two that matter plus context

```
ncu -k "regex:mxfp4|fp8|bf16_gemv" --launch-skip 20 --launch-count 5 ^
  --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,^
lts__t_sector_hit_rate.pct,sm__inst_executed.sum,sm__warps_active.avg.pct_of_peak_sustained_active,^
launch__registers_per_thread,launch__occupancy_limit_registers ^
  build\bench-mxfp4.exe
```

- `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` — is the kernel actually
  streaming (target: matches §2 cold %);
- `dram__bytes_read.sum` — must equal §2 byte formula (± activation bytes); a mismatch
  means the kernel re-reads weights;
- `lts__t_sector_hit_rate.pct` ≈ 0 on a flushed rep = the §1 flush worked end-to-end;
- `sm__inst_executed.sum` (with `sm__sass_thread_inst_executed_op_*` when drilling into
  dequant) — instruction-count deltas validate things like the §1 insig4-perf packed
  `__hmul2` claim (4x dequant SASS cut) without trusting ms alone.

### 4.3 The 3 KPIs recorded per change

1. **ms/token end-to-end** — generate.exe CUDA-event median over ≥100 committed tokens,
   3 runs, warm (production condition, graph replay). Report median of 3.
2. **Achieved GB/s per kernel** — cold-L2 median from the §2 matrix cells the change
   touches (before AND after).
3. **Launch count per step** — `nsys stats -r cuda_gpu_kern_sum` instance count divided
   by steps; census today ≈610/spec step, ~140 fusable (insig4-perf §4.4). Trend must
   be down as fusions land.

---

## 5. Acceptance rule for optimization PRs (per AGENTS.md: measurement + parity)

A change is not "done" until every box checks. No change merges on warm-L2 numbers; a
change that wins warm but loses cold is REJECTED (production streams from DRAM).

```
[ ] (a) COLD-L2 BENCH TABLE, before AND after (same commit pair, same harness):
      [ ] §1 protocol: flush + N=5 + median + spread, per affected §2 matrix cell
      [ ] table columns: shape, bytes, expected µs @504, cold median, achieved GB/s,
          % of measured_read_ceiling; delta % vs before on every row
      [ ] warm row reported with [L2-RESIDENT]/[L2-PARTIAL] label where applicable
[ ] (a2) KPI-1: end-to-end ms/token re-measured (§4.3); regression > ±2% = investigate
      before merge (a kernel win eaten by launches/tails is not a win)
[ ] (b) PARITY GATE:
      [ ] tools/reference_multistep_i4.py (INSIG4): worst_layer_cos target ≥ 0.999.
          Current i4-ref.log = 0.972 (known full-attn issue) — until fixed the gate is
          A/B vs parent: |Δcos| ≤ 1e-3 per layer AND identical ref_argmax for all steps.
      [ ] e8m0 path (reference_multistep.py): worst_layer_cos ≥ 0.9998
          (multistep-parity.log = 0.99978 is the bar).
      [ ] tools/nll_compare.py: ΔNLL within ±1e-3 of parent. PRE-REQ: build/nll.bat
          must link src\mxfp4_i4.cu (currently omitted — synthesis bug #4; fix before
          this gate counts).
[ ] (c) SASS/occupancy note when compute-relevant (one line): from -Xptxas -v +
      cuobjdump -sass, e.g. "v21-i4: 42 reg, 41 KB smem, 2 blk/SM, HMUL2.BF16 per 2
      weights in hot loop". Not required for pure memory-layout changes.
[ ] (d) Environment row: GPU clocks/temp before+after, driver, nsys/ncu artifact path.
```

Build hygiene while touching benches (existing known traps): bench-gemm.bat and
bench-mxfp4.bat both emit `build\bench-mxfp4.dll` (silent overwrite — rename one);
bench-gemm-blocked.bat is a byte-identical twin of bench-gemm.bat (dedupe); rebuilds go
through build\*.bat with vcvars64 + `-arch=sm_89` only.

---

## 6. Calibration and honesty notes

### 6.1 measured_read_ceiling — measure once, cite forever

The % column is meaningless unless the denominator is real. Once, on an idle GPU, time
the flush sweep itself (256 MB / median ms over N=20) → `measured_read_ceiling`; all
"% of peak" numbers use it. Caveat on record: the 4070 SUPER marketing BW is 672 GB/s
(256-bit @ 21 Gbps); the audits' 504 GB/s matches a 192-bit bus. The sweep settles it
in 30 seconds — if the ceiling lands well above 504 GB/s, re-anchor every % (expected-ms
tables stay as computed at 504; just report both denominators). Until then 504 GB/s is
the repo contract per insig4-perf.md.

### 6.2 What cold numbers can and cannot tell you

A cold number that misses the floor is not automatically a bandwidth failure: at
<=10240 rows the grid is 1280 blocks (23 waves of nothing) and launch + x-staging
dominates (see baseline #6: 10 µs floor). The matrix surfaces this as "achieved GB/s
collapsing on small shapes" — the fix is structural (persistent grid-stride GEMV,
insig4-perf §2.2), not `__ldca`/L2 games. Cross-check every "kernel is slow" claim
with NCU `gpu__dram_throughput` before believing the GB/s arithmetic.

### 6.3 Reference weights for the 27B step budget (from §2 formulas)

fp8 linear layer = 52.44 + 31.46 + 31.46 + 2×89.14 + 89.14 ≈ 382.8 MB → 0.76 ms/layer
at 504 GB/s (matches the 383.87 MB shard slots, synthesis.md:41-42 — the byte formulas
cross-check against the checkpoint). bf16 lm_head 2542.8 MB → 5.05 ms/sweep: it must
stay VRAM-resident and be swept at most once per step.
