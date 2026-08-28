# CPU tier deep audit (w4) — `include/insignia_cpu.hpp` + `src/test_cpu.cpp`

Date 2026-08-25. Auditor scope: correctness vs GPU kernels, CpuPool concurrency, performance
(35.9 GB/s question), test-coverage gaps, w3 known-issue re-verification. Everything below was
read/run firsthand this session: the header (994 lines), test_cpu.cpp, `src/deltanet.cu`,
`src/attention.cu`, `src/ops.cu`, `src/qwen_kernels.cu`, `src/fp8.cu` (+`insignia_fp8.cuh`),
`src/decode.cu` (norm call sites), `src/streaming.cu` (reader affinity), `audits/w3/MASTER-PLAN.md`
§2.4, `audits/w3/cpu-fp8.md`, `audits/w3/cpu-impl.md`, `build/cpu_disasm.txt`, the shipped
`build/test-cpu.exe` (byte-level instruction scan), plus live runs of the test/bench binary and
python probes. No files modified except this report; nothing compiled.

**Verdict tags**: [V] = verified (test run / disasm bytes / arithmetic proof), [H] = hypothesized
(mechanism sound, needs the named measurement).

---

## 0. Headline results

1. **Correctness: clean.** 40/40 checks re-run PASS (`build/test-cpu.exe test`). e4m3 decode,
   bf16 handling, deltanet recurrence, GQA, RoPE convention all match the GPU kernels at the
   documented tolerances (§1). One lockstep obligation (qk-norm zero-center, §1.6) and one
   API-shape risk (A_log dtype, §1.7) to carry into Phase A.
2. **One real concurrency bug found** [V]: `CpuPool::launch()` calls `drive(g)` while holding
   mutex `m_` (insignia_cpu.hpp:243-251). If that drive executes the generation's last ticket it
   re-locks `m_` (hpp:274) → recursive lock of a non-recursive mutex → UB/self-deadlock on MSVC.
   One-line fix (§2.1). Trigger conditions are plausible in production (first-touch page faults
   on demand-paged weights make "slow straggler + unclaimed tickets" the norm, not the exception).
3. **The "35.9 vs ~37-40 GB/s theoretical" question had a wrong premise.** DDR4-3600 dual
   channel = **57.6 GB/s theoretical peak** [V: `Get-CimInstance Win32_PhysicalMemory` — 2×8GB
   Kingstons, ConfiguredClockSpeed 3600]. 35.9 GB/s is only 62% of peak. The lost cycles are
   **not** in the inner loop — they are the per-core memory-level-parallelism limit times the
   context count. Measured [V]: 9 workers+main = 48-49 GB/s layer avg, 11 workers = 49-54 GB/s
   (§3.1). The 6-worker "SMT adds streams not bandwidth" claim (cpu-fp8.md §2.5) is refuted on
   this box.
4. **The pair (T=2) GEMV spills in the shipped binary** [V: byte-level instruction scan of
   build/test-cpu.exe] — cpu-impl.md claim #6 ("zero spill stores") holds only for the
   single-token GEMV. Explains pair 23-29 GB/s stream vs 34-36 single (§3.3).
5. Bench methodology caveat [H]: bench mats are 31-89 MB vs 32 MB L3, min-of-N on the same
   buffer → some L3 service inflates GB/s; production streams 383 MB/layer with zero cross-token
   L3 reuse. Re-verify with a WS≫L3 cycling bench before trusting per-layer ms in the plan.

Measurement environment note: this audit ran concurrently with the other 23 swarm agents; my
absolute numbers (33.6-35.4 GB/s at 6T+main, 11.0-12.6 ms layer, deltanet 39 µs, GQA 0.67 ms)
are contention-depressed vs the mission's quiet-box figures (35.9 / 10.75 / 30 / 0.66). No
regression signal — deltas are consistent with DRAM/core stealing by the swarm.

---

## 1. Correctness

### 1.1 e4m3 decode bit-trick — exact, incl. subnormals [V]

`e4m3x32_f32`/`e4m3x32_rr` (hpp:127-156): fp16 pattern `mag=(b&0x7f)<<7`, sign→bit15, F16C
convert; ×256 folded into the scale (hpp:89-97). Analytic check per code class:
- Normal (E∈1..15): fp16 value (1+m/8)·2^(E−15), ×256 → (1+m/8)·2^(E−7) — exact E4M3, bias 7,
  3 mantissa bits drop in without rounding.
- Subnormal (E=0): fp16 subnormal m·2⁻¹⁷ ×256 = m·2⁻⁹ — exact (matches `e4m3_host`'s
  `m*(1/512)`, test_cpu.cpp:26).
- 0x7F/0xFF → ±480 by the same formula — consistent with the engine's `e4m3_host` convention
  (test_cpu.cpp:23-29) and with GPU `e4m3x2` (`insignia_fp8.cuh:12-19` multiplies by 256.f —
  the CPU folds that exact ×256 into `bf16_scale_x256`, verified below).
- Exhaustive test re-run: 256/256 exact (`e4m3x32_f32 exhaustive PASS bad=0/256`).
- Note [H, minor]: the exhaustive test covers the array form `e4m3x32_f32`; the GEMV uses
  `e4m3x32_rr` (same intrinsics, register outputs). End-to-end f64 GEMV parity effectively
  covers it; a 1-line `rr==f32` A/B assert would formally close it.

### 1.2 bf16 handling [V]

- `bf16_to_f32`/`bf16_widen8` (hpp:63-67, 106-108): (u16<<16) bitcast — exact widening.
- `f32_to_bf16_bits` (hpp:76-81): RNE via `bits += 0x7FFF + lsb` — standard, correct.
- `bf16_scale_x256` (hpp:89-97): the w3 subnormal-scale bug **is fixed** — guard
  `e==0 || e>=0xF7` routes ±0/subnormals/|s|≥2^120/inf/NaN to the exact multiply path; bit-add
  `+0x04000000` (exp+=8) for normals only. Exhaustive 65536/65536 bit-exact re-run
  (`bf16_scale_x256 exhaustive PASS bad=0/65536 nan=254`).
- GPU scale semantics match: `src/fp8.cu` reads raw bf16 scale, applies per-128-col-block
  (`acc = fmaf(part, sc, acc)`, fp8.cu:~40); CPU accumulates raw per 128-col block, promotes
  with the folded scale once per block (hpp:399-400) — same numerics class, different summation
  order, within the 1e-3 CPU/GPU bar.

### 1.3 deltanet CPU step vs GPU kernel — same recurrence [V]

Line-by-line vs `src/deltanet.cu:4-13`:
- State layout: GPU `S = state + head*128*128`, `S[i*128+tid]` (i=k index, tid=v); CPU pass A/B
  iterate `S + k*128` vectorized over v (hpp:686-707). Same [48][128][128] f32, S[k*128+v]. ✓
- q/k norm: GPU `rsqrtf(ss+1e-6)*0.08838834764831845` folds 1/√128 on q only, k gets plain
  rsqrt (deltanet.cu:8); CPU identical constants (hpp:676-677). ✓
- k-sharing: GPU 9B `kh = head>>1` (deltanet.cu:5) with `<<<32,128>>>`; CPU `kh = h/kshare`,
  default kshare=3 = 27B's head/3 (hpp:717-719), test also runs 32h/k2 mirroring the 9B shape
  (test_cpu.cpp:336). Matches MASTER-PLAN B.3 (27B `<<<48,128>>>`, kh=head/3). ✓
- Gating/alpha/beta order: params computed first (`deltanet_parameters_cpu` hpp:634-643 —
  bit-identical formula to `qwen_kernels.cu:9` incl. the z>20 softplus guard), then
  `decay = expf(g)` inside the step (hpp:680 = deltanet.cu:9); delta=(v−S·decay·k̂)·β then
  S'=S·decay+k̂·δ, out=ΣS'·q̂ (hpp:685-708 = deltanet.cu:10-12). ✓
- One ULP-level association difference [V, benign]: GPU computes `fmaf(S*decay, k̂, dot)`
  (rounds S·decay), CPU computes `fmaf(S, k̂*decay, d)` (rounds k̂·decay) — 2 roundings either
  way, different intermediates; measured chained-state parity abs ≤ 4.1e-7·scale, rel 1.5e-4 on
  cancelled elements — inside the stated 1e-3 CPU/GPU bar.

### 1.4 GQA CPU vs GPU [V]

- Grouping: CPU kv-group-major, heads h0=hg*6..h0+5 share kv-head hg — equivalent to
  `kvh = head/6` (hpp:820-831), matching the 27B correction in MASTER-PLAN §B (GPU
  `src/attention.cu:7` still has the 9B `head>>2`; the 27B GPU kernel will be /6 — CPU is
  already right).
- Layout: KV rows `[tokens][4 kvh][256]` — GPU `kc[(t*4+kvh)*256+d]` = CPU `t*1024+hg*256`
  (hpp:831). ✓ Scale 1/16 both (`0.0625f`). ✓
- Softmax numerics: GPU two-pass (max→exp→den→weighted V); CPU online-softmax over 64-token
  blocks with per-split (m,l,o) partials + flash merge (hpp:856-871, 934-951) — equivalent
  modulo fp; vector exp is the Remez deg-4 (max rel 2.6e-6) with ±87.3 clamp so far-out scores
  underflow like GPU `__expf`. Merge guards `m==NINF → w=0` for empty splits (hpp:942). ✓
- bf16 KV: exact widen path, template-dispatched (hpp:836-851, 885-897), tested
  (`gqa_decode bf16 t=2048 PASS`, abs 2.2e-6). Store side `store_kv_bf16_cpu` RNE. ✓
- Edge note [H, unreachable in practice]: tail padding uses −100 (hpp:858) which `vexp` clamps
  to e^−87.3 ≈ 1.3e-38, not exactly 0; it only matters if ALL real scores in a block are
  < −100 (impossible post-RMSNorm with scale 1/16: |s| ≲ 16·w̄). Pass 2 never reads padded p.

### 1.5 RoPE partial-64 pairing — consistent [V]

GPU `qk_norm_rope` (src/ops.cu:9): pairs (tid, tid±32) for tid<64 ⇒ (i, i+32), i<32; rotation
`v·c ∓ other·s`; applied AFTER per-head RMSNorm(256)+bf16-weight, first 64 dims only; gate not
roped. CPU (hpp:743-763): identical — norm+weight over 256, then `p[i]=a·cs−b·sn`,
`p[i+32]=a·sn+b·cs`, i<32, `if(pos)` guard matches GPU's `pos!=0`. θ=1e7, exponent −2i/64 both.
CPU computes angles in f64 (hpp:738-741) vs GPU float `__powf` — documented deliberate
improvement (cpu-impl.md #4), ≪ the 1e-3 CPU/GPU bar.

### 1.6 Zero-center (1+w) norms — matches today; ONE lockstep obligation [V + flag]

- `rmsnorm_cpu(x,w,y,cols,zero_centered)` implements both raw-w and (1+w) (hpp:547-553) —
  mirrors GPU `rms_bf<Z>` (qwen_kernels.cu:5). Both paths tested. Engine today calls
  `rmsnorm_bf16(..., false, ...)` everywhere (decode.cu:126-131,147-188) — CPU matches today.
- `gated_rmsnorm_per_head_cpu` uses raw w (hpp:579) — matches GPU `rms_bf<false,true>`
  (qwen_kernels.cu:6) AND the 27B rule ("linear_attn.norm stays RAW", MASTER-PLAN A.7). ✓
- **FLAG (must-do when Phase A lands)**: MASTER-PLAN A.7 switches q/k-norm (+ input/post/model
  norms) to (1+w) for the 27B checkpoint. `qk_norm_rope_cpu` has **no zero_centered parameter**
  (hpp:733-763) — it will silently diverge from the updated GPU `qk_norm_rope` (which gains the
  +1 per A.7). Add the flag + test at the same commit; likewise the integration sketch in
  cpu-impl.md §5 passes `zero_centered=false` for input/post norms ("engine uses false today") —
  that must flip to true at 27B. This is exactly the master-plan risk #1 class (silent
  plausible-looking wrongness), so wire a CPU-vs-GPU norm-parity check into R4.

### 1.7 A_log dtype shape risk [H → design note]

`deltanet_parameters_cpu` takes `const float* A_log` (hpp:635). The 27B checkpoint stores A_log
as BF16 (MASTER-PLAN risk #4: BF16-read-as-F32 → α≈1 → no forgetting, silent). The GPU side
gets a dtype dispatch (A.6); the CPU header needs the equivalent — either a bf16→f32 prepare
step at layer load (like `fp8_prepare_scales`) or an `a_log_f32` bool. Caller-side widening
must be tested (currently the test feeds f32 directly).

### 1.8 Other verified small ops

conv1d+silu (hpp:616-630) = `qwen_kernels.cu:7` math with weights pre-expanded to f32 [4][ch]
(`expand_conv_weights` — exact bf16 widen); state shift bit-exact in test. `split_q_gate_cpu`
(hpp:767-774) = `qwen_kernels.cu:73` layout q/gate per head (24-head default = 27B; GPU literal
4096 is the 9B shape — consistent with plan B). `sigmoid_mul_cpu`/`silu_mul_cpu`/`residual_add_cpu`
match `sigmul`/`silu_mul_kernel`/`add_kernel`. `store_kv_cpu` = `store_kv_kernel` layout
(qwen_kernels.cu:15). No bounds check on `pos` (see §4).

---

## 2. Concurrency (CpuPool, hpp:217-339)

### 2.1 BUG [V, code-reading]: `drive()` under lock in `launch()`'s cvd_ branch

insignia_cpu.hpp:243-251:
```cpp
else {
    std::unique_lock<std::mutex> lk(m_);
    cvd_.wait_for(lk, 500us, [&]{ return s.left... == 0; });
    drive(g);            // <-- m_ is HELD here (wait_for reacquires before returning)
}
```
`drive()`'s last-ticket path does `std::lock_guard<std::mutex> lk(m_); cvd_.notify_all();`
(hpp:273-276) → recursive acquisition of a non-recursive mutex = UB (MSVC std::mutex is
SRWLOCK-based: self-deadlock). Even on non-last tickets, `m_` is held across the whole ticket
`fn` execution, blocking every worker's park/unpark (their `cv_.wait` needs `m_`). Note the
asymmetry: `worker_main()` correctly does `lk.unlock(); drive(g);` (hpp:299).

Reachability: the cvd_ branch needs 32768 spin iterations with tickets still outstanding —
i.e. a straggler holding the last tickets for ≥ ~0.3 ms (32768 × pause+load). In the bench this
never fires (drive() is exhaustive, jobs finish inside the spin window). In **production it can**:
demand-paged weights (mmap/ring first touch) stall a worker for exactly that long, the
dispatcher times out on the 500 µs wait with tickets still unclaimed, claims them with `m_`
held, and if it finishes the last one → hang. Fix (1 line, mirror worker_main):
`lk.unlock();` before `drive(g);`. Rank: correctness-of-infrastructure #1; fix before wiring
the tier into decode.

### 2.2 Verified-sound aspects [V]

- **No ABA / no stale-fn**: packed `claim_ = (gen<<32)|ticket` (hpp:265-271); the dispatcher
  publishes g+1 only after `left==0` (all fn of g returned, hpp:235-251 under `launch_mut_`),
  so fn/ctx of gen g are immutable while any straggler might still CAS; a straggler's CAS
  against a moved-on gen fails on the reload. Slots double-buffered per gen parity.
- **Liveness**: caller participates (progress guarantee even with all workers parked); 500 µs
  cvd_ timeout self-heals a missed wakeup.
- **No false sharing**: `Slot` is `alignas(64)` (hpp:255); `claim_`/`stop_` share at most one
  line (stop_ written once at shutdown); `gen_` sits after the 128 B slots array. Layout
  verified by member order (MSVC does not reorder members). `claim_` CAS traffic ≈ 1/ticket
  (160-544/mat) vs 4.5 µs/ticket of work — negligible.
- **Affinity plan vs streaming.cu**: workers pinned LP 0-5 (`1<<i`, hpp:319-323); NvmeReader's
  2 threads pinned `0xFC0` (LP 6-11) at ABOVE_NORMAL (streaming.cu:32,47-48). Disjoint, matches
  MASTER-PLAN D.4. Workers park after 4096 pauses (hpp:285-290) so they don't fight readers
  during 115 ms NVMe windows.

### 2.3 Concurrency observations / minor warts [V]

- `caller_helps=false` is cosmetic: the wait loop calls `drive(g)` regardless (hpp:243-250), so
  the caller always participates once it starts waiting. Harmless today (nobody passes false).
- Hard-coded LP enumeration assumption "LP 0-5 = physical primaries" (hpp:321). True for this
  5600X in default enumeration, but CPPC/core-scheduler BIOS modes can reorder [H]. Cheap
  hardening: enumerate via `GetLogicalProcessorInformationEx` and pick first-LP-per-core.
- `INSIG_CPU_THREADS>6` pins workers into reader LPs (collision is opt-in, documented).
- Work-splitting: `rpt = clamp(rows/96,1,32)` → 320/160/544 tickets for the big mats; tail
  imbalance ≤ 1 ticket ≈ 160 KB ≈ 4.5 µs — negligible. deltanet (6 tickets) and GQA (nsplit=6)
  leave the 7th context idle; raise GQA nsplit with the worker count (it's a parameter,
  hpp:926).
- Spin-then-park (4096 pauses ≈ 30-40 µs) covers inter-op serial gaps (norms ~25 µs, conv ~30
  µs). If worker count goes to 9-12 (R1 below), SMT siblings spinning during gaps steal issue
  slots from their GEMV sibling — consider a shorter spin for LP≥6 workers.

---

## 3. Performance

### 3.1 The 35.9 GB/s question — answered: context count, not the inner loop [V]

Premise correction: DDR4-3600 dual channel = 2×8B×1.8GT/s = **57.6 GB/s peak** (RAM config
verified). The "~37-40 theoretical" band in the mission is a stream-class expectation (64-70%
of peak), not the ceiling. The inner loop is not the limiter (28 µops/32 weights ≈ ≤8 cycles of
port time vs the DRAM budget; single-GEMV cluster verified spill-free, §3.2). The limiter at
6 workers is per-core MLP (1-thread rate 8.2-8.4 GB/s measured) × 7 contexts.

Measured sweep (shipped binary, `INSIG_CPU_THREADS=n`, this session, contended box; shape =
10240×5120 / full linear layer):

| workers (+main) | qkv GB/s | linear layer ms |
|---|---|---|
| 6 (default) | 33.6-35.4 | 11.0-12.6 (quiet-box doc: 10.75) |
| 7 | 43.2 | 9.26-9.46 |
| 8 | 44.4 | 9.68 |
| 9 | 49.9 | 7.92 |
| 11 | 54.5 | 7.77 |
| 12 | 51.0 | 7.91 |

At 11 workers the pool sustains 54.5 GB/s = 95% of theoretical peak (48-54 GB/s = 83-94%).
All runs stable, 40/40 tests unaffected. **The w3 claim "SMT siblings add streams, not
bandwidth" (cpu-fp8.md §2.5) is refuted on this box** — more contexts = more outstanding
misses = deeper memory-controller queues.

Caveats before adopting: (a) [H] part of the gain may be L3 service — bench mats are 52-89 MB
vs 32 MB L3, min-of-N reuses the same buffer; production streams 383 MB/layer cyclically with
zero cross-token reuse. Re-bench with a ≥512 MB multi-mat cycling working set. (b) workers >6
pin onto LP 6-11, colliding with the 2 ABOVE_NORMAL reader threads; in the v2 pipeline GEMV and
staging run concurrently (DRAM co-budget: staging writes ~3.3 GB/s + reads 40+ GB/s < 57.6, but
write-turnaround costs ~5-10% read efficiency) — must be measured co-resident (Phase G). Even
discounting 30% of the gain, this is worth ~1-2 ms/layer.

### 3.2 Inner-loop disasm on the SHIPPED exe [V]

Byte-level scan of `build/test-cpu.exe` (python, no rebuild): 32 `vcvtph2ps` sites in 3 clusters —
- 0x5f47-0x60d0 file-off: GEMV loop **with** `prefetcht0 [r+ r*1+0x100]`, convert → `vmovups
  [rbp+xxh]` store → reload → FMA against **two** x base regs (r10/r11) ⇒ the **pair (T=2)
  GEMV, spilling** (accumulator/w round-trips through stack per 32-weight chunk).
- 0x6909-0x6a30: GEMV loop, `vcvtph2ps` → `vfmadd231ps [rcx+rax*4+..]` directly, single x base
  ⇒ single-token GEMV, **spill-free** — cpu-impl.md #6's claim holds here only.
- 0x10b61ff: the exhaustive-test array API (spills expected; it's the test).

So the record needs correcting: cpu-impl.md's "zero spill stores" is true for `fp8_gemv_rowrange`
only; `fp8_gemv2_rowrange` (pair) spills despite the rr form (16+ ymm pressure: 4 w + 8 acc +
consts + 2 x pointers). This, plus the doubled x loads, explains pair 23-29 GB/s stream vs
34-36 single. Note `build/cpu_disasm.txt` (22:09) predates the shipped exe (23:14) — its
spilling cluster is consistent with what I find in the shipped exe, so the fix never fully landed
for the pair path.

### 3.3 Ranked recommendations (27B linear layer: qkv 1.46-1.54, z 0.88-0.91, out 0.87-1.03,
gate/up 2.49-2.89 ×2, down 2.49-3.32 ms; total 10.75 clean / 11.0-12.6 contended)

| # | change | est. impact/layer | basis |
|---|---|---|---|
| R1 | **workers 6 → 9-11** (keep affinity `1<<i`, coordinate with reader mask; raise GQA nsplit to match) | **−1.5 to −3 ms (−15-28%)** | measured §3.1; verify co-resident + WS≫L3 |
| R2 | **fix launch() m_-held drive** (§2.1) | correctness (rare hang) + removes m_-held-across-fn stalls | verified bug |
| R3 | **pair-GEMV spill fix**: 16-weight chunks (2 w + 8 acc live) or two-pass a/b per 32 w | −0.3-0.4 ms per pair mat; ~1.5-2 ms per MTP-verify pass (5 mats/layer × CPU layers) | §3.2; pair is verify-path only |
| R4 | **fuse launches**: qkv+z (16384 rows, one ticket space), gate+up, a+b bf16 | −0.1-0.2 ms (~0.5-1.5%: 2 fewer wake/park cycles + halved tail imbalance) | audit §5.3 dispatch est. 20-60 µs/launch |
| R5 | **2 MB large pages** for weight residency (VirtualAlloc + SeLockMemoryPrivilege; ring slots 2 MB-aligned) | 0-5% (0-0.5 ms) [H] | 383 MB/token at 4 KB pages = ~93 k page touches; STLB pressure; needs A/B |
| R6 | **GQA nsplit 6→8-10** (engages idle contexts) + bf16 KV after the parity gate | −0.1-0.25 ms/full-attn layer | measured 0.43-0.68 ms @6 splits |
| R7 | **prefetch sweep** `INSIG_PREFETCH_DIST ∈ {0,128,256,512,1024}` | ≤1% [H] | one compile-time knob, needs rebuild A/B |

Non-issues verified: interleaved scale-apply is ~12 ops per 512 weights (hsum+2 mul per 128-col
block, hpp:399-400) — invisible; fp8 row-major access is sequential per ticket (rpt contiguous
rows), bank-friendly; F16C usage is convert-only (correct for Zen 3 — no fp16 FMA); NT stores
correctly avoided (outputs ≤70 KB, consumed next op).

### 3.4 Production per-layer budget correction [H]

With zero L3 reuse across tokens (383 MB/layer cyclic stream), the honest rate is the DRAM-only
figure, not the bench figure. If the WS≫L3 re-bench lands at, say, 30-40 GB/s, the layer is
9.6-12.8 ms and MASTER-PLAN §2.4's C-tier 10.8 ms (and "CPU 15% duty") shifts accordingly —
conclusions survive (NVMe still binds v2 at 1731 ms), but the solver (Phase G) must use the
de-inflated number. Deltanet state (3.15 MB/layer) and KV (16.8 MB/full layer) will also be
L3-evicted between tokens by the weight streams — the audit's 0.26 ms / 0.45 ms "cold" figures,
not the 31 µs / 0.5 ms L3-warm bench numbers, are the production ones.

---

## 4. Test coverage gaps (test_cpu.cpp)

1. **No pool stress test**: rapid launch churn, launch immediately after launch, stop-during-
   work, `INSIG_CPU_THREADS` >6 — none exercised. The gen/CV logic is the most subtle code in
   the header; a 5-second storm test (random shapes, 10k launches, verify every result) is cheap
   insurance and would have flushed §2.1's reachability conditions.
2. **GQA empty-range branch untested**: `nsplit=min(nsplit,tokens)` makes empty ranges possible
   (e.g. tokens=7, nsplit=6 → per=2, ticket 5 empty → neutral-partial path, hpp:916-919). Tests
   use 2048 (341/token/split) and 77 (13/split) — never an empty split.
3. **qk_norm_rope zero-centered variant**: missing (blocked on Phase A, §1.6) — add with the
   CPU/GPU lockstep test.
4. **A_log bf16 widening**: no helper, no test (§1.7).
5. **store_kv pos bounds**: no capacity check (OOB write if pos ≥ ctx), and no test for it —
   mirror the GPU-side C1 guards (MASTER-PLAN D.5) on the CPU store.
6. **No WS≫L3 bench mode**: all GB/s figures come from 31-89 MB mats reused across iterations
   (§3.4). Add a cycling multi-mat bench (aggregate ≥512 MB) and a `DRAM` mode that mixes the 5
   layer shapes in rotation so min-of-N can't hit L3.
7. **Odd T**: only T=1 and T=2 exist by design (fine for decode/D=1 verify). D=4/T=5 spec will
   need either T=4 batching or 2×pair+1 single — document the plan now.
8. Minor: no direct `e4m3x32_rr == e4m3x32_f32` A/B (covered end-to-end); no test that
   `deltanet_step_cpu` throws on heads%kshare≠0 (trivial); `INSIG_CPU_FP8_LUT=1` A/B build not
   exercised in CI.

Covered well (verified): bf16 KV decode+store, non-multiple-of-64 token counts (77 exercises
the sbuf tail padding), 2-step chained deltanet incl. state compare, both norm modes, conv state
shift bit-exactness, serial==pool bit-identity, pair y-layout.

---

## 5. w3 known-issues cross-check (cpu-fp8.md / cpu-impl.md → header)

| w3 item | status in header |
|---|---|
| bf16_scale_x256 subnormal guard (cpu-impl #1) | **fixed** hpp:92, exhaustive PASS [V] |
| gated norm w [128] shared (#2) | **fixed** hpp:564-587 (w never advanced), test uses [128] [V] |
| qk norm w [256] shared (#3) | **fixed** hpp:733-763 [V] |
| RoPE f64 angles (#4) | **landed** hpp:738-741 [V] |
| MSVC `__AVX2__` guard + f16_to_f32 (#5) | **landed** hpp:26-32, 71-73 [V] |
| rr register form, "zero spill" (#6) | **half-true** — single GEMV clean, **pair GEMV spills** in the shipped exe (§3.2) [V] |
| GQA v3 kv-group-major (#7) | landed, hpp:814-910, measured [V] |
| deltanet kshare param (#8) | landed hpp:721-727, both shapes tested [V] |
| max_abs_rel metric (#10) | landed hpp:975-992 [V] |
| cpu-impl gap #1 "nobody calls this yet" | **still true** — grep: only test_cpu.cpp includes insignia_cpu.hpp; decode.cu has no CPU branch [V] |

---

## 6. Bottom line

The kernels are correct and the parity harness is honest (exhaustive decode/scale proofs,
f64 references, floored rel + scale-normalized abs gates). The infrastructure has one real
(hard-to-hit, production-reachable) locking bug and one false performance record (pair-GEMV
spills). The big perf lever is embarrassingly simple: the box has 57.6 GB/s of DRAM and the
pool stops at ~35 with 7 contexts — 11 contexts measured 49-54. Fix order: R2 (bug) → R1
(workers) → re-bench WS≫L3 → R3 (pair spill) → R4-R7.
