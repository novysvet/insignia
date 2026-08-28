# W3 io_bench — measured NVMe streaming numbers (E: model dir + C: test file)

Date: 2026-08-25, single session, rig per AGENTS.md (5600X, 12 LPs, Win11 26200).
Artifacts: `src/io_bench.cu` (pure Win32 C++, no CUDA calls; named .cu for the build
system), `build/io_bench.exe` (MSVC `/TP /O2 /std:c++20`). Method: one IOCP,
`FILE_FLAG_OVERLAPPED`; direct mode adds `FILE_FLAG_NO_BUFFERING`; qd = outstanding
OVERLAPPED per thread; every config = one full sweep (E: 2.29 GB = layers-0..5.safetensors
whole-block floors, ~2.3 GB read per config; C: 6.44 GB = exactly 6 GiB test file).
Direct reads bypass the page cache, so every config is cold-from-disk by construction.
Full run ~1 min wall, ~34 GB read on E:, ~70 GB read + 6 GiB written-then-deleted on C:.
GB/s are decimal (1e9 B/s; 3.25 GB/s = 3.03 GiB/s). `cores` = worker-thread CPU cycles
(QueryThreadCycleTime) / elapsed / measured cycle frequency (4.19 GHz estimate).

## 0. TL;DR

| result | number |
|---|---|
| E: (980 1TB, Gen3) SEQ-DIRECT | **3.25 GB/s** max; **flat in QD and block size** (QD1 = QD16, 256K = 4M) |
| E: at engine config (2M, QD8) | 3.22 GB/s (=119 ms per 383.87 MB layer) |
| C: (980 PRO, Gen4) SEQ-DIRECT | **6.50 GB/s** max at 1M/QD8; 6.4-6.5 across 1M-4M except 4M/QD16 = 4.6 (anomaly) |
| DUAL E:+C: concurrent | E: 3.21 + C: 6.47 simultaneously → **~9.7 GB/s additive**, zero mutual degradation |
| E: buffered (cached pass 2) | 7.5-16.8 GB/s but **3.1-3.6 cores** of cache-copy CPU (vs 0.05-0.25 direct) |
| Per-completion CPU | direct: 54K-300K cycles/io (grows with blk); **0.06-0.25 cores total** — "2 cores" claim is 10-30x too high |
| GQCSEx avg batch (direct) | ~1.0 (drive-paced arrivals; batching never engages at streaming rates) |
| C: 6 GiB write (NO_BUFFERING, QD16x8MB) | 1.24-1.54 GB/s (82%-full PRO, SLC-limited), flushed + verified |

## 1. Hardware verification (Get-PhysicalDisk + Get-Partition, this session)

| drive letter | disk # | model | bus | role |
|---|---|---|---|---|
| **E:** | 2 | **Samsung SSD 980 1TB** | NVMe Gen3, DRAM-less (HMB), 600 TBW | model dir (Qwen3.8-27B-FP8) |
| **C:** | 1 | **Samsung SSD 980 PRO 500GB** | NVMe Gen4, 300 TBW | system + %TEMP% (86-92 GB free) |
| D: | 0 | WDC WD40EFAX-68JH4N1 4TB | SATA HDD | untouched |

Confirms `audits/w3/nvme-reader.md` §1 (the web-nvme-win agent's drive identification).
E: measured 3.25 GB/s = 3.03 GiB/s — slightly **under** the 3.3 GB/s planning number,
above the ≥3.0 GB/s acceptance bar from nvme-reader.md §6. C: behaves to spec (6.5 of
the 6.9-7.0 rated; 4070S host, real conditions).

## 2. [1] SEQ-DIRECT E: — model files, NO_BUFFERING (16 configs, 4 threads)

| blk | qd | GB/s | IOPS | batch | cyc/io | cores |
|---|---|---|---|---|---|---|
| 256K | 1 | 3.13 | 11,956 | 1.00 | 83,692 | 0.239 |
| 256K | 4 | 3.23 | 12,327 | 1.00 | 83,333 | 0.245 |
| 256K | 8 | **3.25** | 12,412 | 1.00 | 74,969 | 0.222 |
| 256K | 16 | 3.25 | 12,412 | 1.00 | 82,461 | 0.244 |
| 1M | 1 | 3.25 | 3,097 | 0.98 | 134,465 | 0.099 |
| 1M | 4 | 3.25 | 3,096 | 0.99 | 141,771 | 0.105 |
| 1M | 8 | 3.25 | 3,095 | 0.98 | 153,419 | 0.113 |
| 1M | 16 | 3.15 | 3,002 | 0.98 | 145,318 | 0.104 |
| 2M | 1 | 3.22 | 1,535 | 0.97 | 184,707 | 0.068 |
| 2M | 4 | 3.22 | 1,537 | 0.97 | 173,914 | 0.064 |
| 2M | 8 | 3.22 | 1,536 | 0.97 | 162,809 | 0.060 |
| 2M | 16 | 3.14 | 1,499 | 0.96 | 157,470 | 0.056 |
| 4M | 1 | 3.22 | 769 | 0.94 | 264,994 | 0.049 |
| 4M | 4 | 3.22 | 768 | 0.93 | 279,736 | 0.051 |
| 4M | 8 | 3.12 | 743 | 0.93 | 286,088 | 0.051 |
| 4M | 16 | 3.16 | 754 | 0.93 | 300,049 | 0.054 |

**The drive is the pacer at ~3.22-3.26 GB/s no matter what.** Queue depth 1 already
saturates it (HMB/SLC readahead); block size only moves CPU cost, not bandwidth.
Warmup run (uncounted): 3.22. Cross-run variance ~±0.05 GB/s.

## 3. [3] Thread probe at best E: config (256K, QD8)

| threads | GB/s | cores |
|---|---|---|
| 1 | 3.26 | 0.190 |
| 2 | 3.25 | 0.225 |
| 4 | 3.26 | 0.233 |
| 8 | 3.26 | 0.241 |

Thread count is irrelevant for bandwidth on E: — one thread with QD8 (or even QD1)
streams at full drive rate. Threads only add CPU.

## 4. [2] SEQ-CACHED-PAGEABLE control (buffered handle, pass 2 = standby-resident)

Buffered warmup pass 1 (cold): **3.22 GB/s** — identical to NO_BUFFERING (the disk is
the pacer cold). Matrix below is pass 2+ with the 2.29 GB span resident in the standby
list (RAM: 16 GB, span fits easily — this is the *best case* for buffered, NOT the
engine's steady state):

| blk | qd | GB/s | IOPS | batch | cyc/io | cores |
|---|---|---|---|---|---|---|
| 256K | 1 | 11.95 | 45,585 | 1.13 | 288,970 | 3.142 |
| 256K | 4 | **14.98** | 57,145 | 4.10 | 261,236 | 3.561 |
| 256K | 8 | 12.87 | 49,082 | 7.82 | 291,445 | 3.412 |
| 256K | 16 | 12.18 | 46,467 | 14.99 | 327,716 | 3.632 |
| 1M | 1 | **16.83** | 16,049 | 1.05 | 820,604 | 3.141 |
| 1M | 4 | 12.80 | 12,208 | 3.91 | 1,149,828 | 3.348 |
| 1M | 8 | 10.16 | 9,690 | 7.86 | 1,485,640 | 3.434 |
| 1M | 16 | 9.02 | 8,600 | 14.66 | 1,695,051 | 3.477 |
| 2M | 1 | 15.60 | 7,436 | 1.04 | 1,780,568 | 3.158 |
| 2M | 4 | 10.09 | 4,810 | 3.85 | 2,972,533 | 3.410 |
| 2M | 8 | 8.42 | 4,015 | 7.09 | 3,591,811 | 3.439 |
| 2M | 16 | 8.12 | 3,870 | 12.70 | 3,572,255 | 3.297 |
| 4M | 1 | 10.64 | 2,537 | 1.01 | 5,410,882 | 3.274 |
| 4M | 4 | 8.31 | 1,980 | 3.80 | 7,197,979 | 3.400 |
| 4M | 8 | 7.52 | 1,794 | 6.54 | 7,502,555 | 3.210 |
| 4M | 16 | 8.38 | 1,998 | 8.76 | 7,206,743 | 3.435 |

Findings:
- **Cache hits are ~4-5x disk rate but cost ~3.4 cores** (cache-manager copy at
  ~1.4-1.8 CPU cycles/byte, plus locking) vs 0.06-0.25 cores for direct. Note the
  cached ceiling (17 GB/s) is nowhere near RAM bandwidth — one memcpy through the
  system cache with shared-file locks is the bottleneck, and *bigger blocks are
  slower* when cached.
- Methodology trap found and fixed live: with `FILE_FLAG_SEQUENTIAL_SCAN` on the
  buffered handle, pass 2 tracked **disk** rate (3.2) — the hint tells the cache
  manager to recycle read-past pages immediately, hiding the cache effect entirely.
  Final numbers use the plain buffered handle. Engine implication: the flag is fine
  on the buffered *twin* for one-shot startup reads, but never assume it caches.
- GQCSEx batching works exactly as designed for bursty completions: batch ≈ qd
  (up to 16 entries per call) on cached bursts, ≈ 1.0 on drive-paced direct arrivals.
- This is the *knife-edge* regime of nvme-reader.md §5.2: it exists only because
  2.29 GB < free standby. The engine's real N-cycle (8.06 GB at N=21) and the 25.65 GB
  model cannot stay resident → LRU thrash → the direct numbers are the honest ones.

## 5. [4] C: — 6 GiB test file (write, then SEQ-DIRECT reads, then deleted)

`%TEMP%` = `C:\Users\Pufos\AppData\Local\Temp` (on C:, 91.5-92.0 GB free — passed the
≥10 GB gate). Write: exactly 6,442,450,944 B, NO_BUFFERING|OVERLAPPED, 8 MB × QD16,
pseudo-random pattern, `FlushFileBuffers` (SLC→NAND commit) before any read timing,
size verified. **1.24-1.54 GB/s** write (reproducible across two runs; drive is 82%
full, SLC cache limited). File deleted at the end (verified; free space restored).

| blk | qd | GB/s | IOPS | batch | cyc/io | cores |
|---|---|---|---|---|---|---|
| 1M | 4 | 6.47 | 6,173 | 0.99 | 102,107 | 0.150 |
| 1M | 8 | **6.50** | 6,199 | 0.99 | 95,866 | 0.142 |
| 1M | 16 | 6.49 | 6,191 | 1.00 | 103,307 | 0.153 |
| 2M | 4 | 6.48 | 3,090 | 0.98 | 170,404 | 0.126 |
| 2M | 8 | 6.47 | 3,083 | 0.98 | 170,767 | 0.126 |
| 2M | 16 | 6.45 | 3,076 | 0.98 | 179,432 | 0.132 |
| 4M | 4 | 6.49 | 1,548 | 0.96 | 296,475 | 0.109 |
| 4M | 8 | 6.38 | 1,522 | 0.96 | 330,285 | 0.120 |
| 4M | 16 | **4.60** | 1,097 | 0.96 | 361,744 | 0.095 |

Anomaly: **4M × QD16 collapses to 4.3-4.6 GB/s** (reproduced twice: 4.31, 4.60).
That is 64 outstanding × 4 MB = 256 MB in flight per 4 threads — too much for the
PRO/stornvme at this shape. Everything 1M-4M at QD4-8 sits at 6.38-6.50. Keep
in-flight bytes ≤ ~64-128 MB.

## 6. [5] DUAL — E: pool and C: pool simultaneously (independent NVMe drives)

E: pool at its best config (256K/QD8/4thr) repeating its 2.29 GB epoch; C: pool at
its best config (1M/QD8/4thr) reading the 6 GiB file once; both released by one
shared start event:

| pool | GB/s (dual) | GB/s (solo best) | degradation | cores |
|---|---|---|---|---|
| E: (980 1TB) | 3.21 | 3.25 | −1% | 0.237 |
| C: (980 PRO) | 6.47 | 6.50 | −0.5% | 0.190 |
| **aggregate** | **9.68 additive over the 1.00 s concurrent window** | | | 0.43 total |

(Program printed 7.71 GB/s over the 1.43 s span, which includes E:'s 0.43 s
finish-current-epoch tail after C: completed; both drives ran at full solo rate
during the actual concurrent window — the additive 3.21+6.47 ≈ **9.7 GB/s** is the
number to plan with. Slightly conservative for E:, which straddled one epoch
boundary at 50 ms revive latency.)

**Consequence for the split/mirror idea (nvme-reader.md §1, §5):** the two NVMe
drives are perfectly independent PCIe citizens — a checkpoint split E:+C: streams
at ~9.7 GB/s aggregate with 0.43 cores of reader CPU. Split halves E: traffic.

## 7. Per-completion CPU cost — the "2 cores of overhead" claim is dead

- Direct streaming at engine shape (2M/QD8, 3.22 GB/s): **0.060-0.068 cores**, 1536
  IOPS, ~163-185K cycles per completion (~40 µs core-time each, dominated by
  wake-from-park context switches + ReadFile/GQCSEx syscalls, growing with block
  size from MDL/page pinning: 75K cycles @256K → 300K @4M).
- Worst direct config measured (256K/QD8, 12.4K IOPS): **0.22-0.25 cores**.
- DUAL, both pools at full rate: **0.43 cores total**.
- So synthesis.md's "~2 cores of overhead" is **10-30x pessimistic** for IOCP at
  ≥256K blocks. The figure probably descends from colibri's blocking-pread thread
  pool; a GQCSEx park/wake pool costs ~1 wake + 2 syscalls per MB-scale completion.
- Buffered cached hits are the expensive path: 3.1-3.6 cores — the memcpy tax
  nvme-reader.md §5.2 warned about, now measured.
- GQCSEx batch size at streaming rates is **~1.0** (completions arrive drive-paced,
  80-650 µs apart at these bandwidths) — completion batching buys nothing here; it
  only engages on cache-burst patterns (batch ≈ qd, up to 16).

## 8. Recommendations for the engine reader

| parameter | E: (model dir) | C: (if split/mirror) | why |
|---|---|---|---|
| block | **2 MiB** | 1 MiB (2 MiB fine) | E: flat in blk; 2M matches the §3 reader slot math (184-block slots, 4096-mult, tail rule) at −1% of best; C: peaks at 1M |
| queue depth | **8 per thread** (QD1 suffices on E:) | 8 | E: saturates at QD1; QD8 is free slack. Never 4M×QD16 on C: (4.6 anomaly) — keep in-flight ≤ ~128 MB |
| threads | **2** (1 suffices; 4 harmless) | 4 (not probed below 4 on C:) | threads don't add bandwidth on E:, only CPU; 2 gives warm-spare + dual-slot arming |
| flags | NO_BUFFERING + OVERLAPPED, one IOCP | same | buffered = 3.4-core memcpy tax and 0% steady-state hits (§4) |

Expected engine numbers with these: E: 3.22 GB/s → **119 ms per 383.87 MB layer**
(nvme-reader.md's 116 ms estimate confirmed within 3%); N=21 all-E: = 8.06 GB/token →
**2.50 s/token → 0.40 tok/s** (+MTP ×1.6 ≈ 0.64). Split E:+C: (≈10.5 layers each):
E: 4.03 GB → 1.25 s, C: 4.03 GB → 0.62 s → **1.25 s/token → 0.80 tok/s** (1.3 with
MTP). Mirror (either drive serves each block): 8.06 / 9.7 = 0.83 s/token ≈ **1.2
tok/s** (2.0 with MTP) at the cost of doubling stored bytes.

## 9. Surprises / corrections to prior assumptions

1. **QD1 = QD16 on E:** — the 980's HMB readahead saturates the drive at depth 1.
   The reader's queue machinery is latency insurance, not a throughput lever.
2. **C: 4M×QD16 = 4.6 GB/s** (from 6.5) — in-flight volume, not QD alone, is the
   constraint; cap outstanding bytes per drive.
3. **E: at 3.25 GB/s decimal (3.03 GiB/s)** — below the 3.3 GB/s planning number,
   above the 3.0 acceptance bar. All per-layer budgets should use 119 ms/layer.
4. **Buffered cached hits are only 7.5-16.8 GB/s and cost 3.4 cores** — and
   `FILE_FLAG_SEQUENTIAL_SCAN` on a buffered handle makes even that disappear
   (pass 2 runs at disk rate because the hint recycles the pages). Buffered I/O
   is now measured-dead for the streaming tier in two independent ways.
5. **CPU overhead 0.06-0.43 cores** where 2 was budgeted — a full core worth of
   headroom is available for CPU-GEMV on the NVMe tier.
6. Write path (incidental): C: 6 GiB NO_BUFFERING write = 1.24-1.54 GB/s on the
   82%-full PRO — copying the 25.65 GB checkpoint to C: takes ~5 h at this rate;
   do it once, overnight, or stream in parallel.

## 10. Reproduction

```
cd E:\coding\Insignia
cl /TP /nologo /O2 /std:c++20 /W3 src\io_bench.cu /Fo:build\io_bench.obj /Fe:build\io_bench.exe   (after vcvars64)
build\io_bench.exe all        # phases: e = E: only, c = C:+dual only
```

Read-only on E: (model files opened GENERIC_READ with FILE_SHARE_READ|WRITE;
NO_BUFFERING leaves the standby list unpolluted). On C:: creates exactly 6 GiB at
`%TEMP%\insig_iobench.bin` (skipped if <10 GB free), deletes it after (verified),
and leaves the repo otherwise untouched. Source doubles as a reference
implementation of the reader mechanics the §3 `NvReader` needs: slot claiming
around a global cursor, cross-thread completion re-arm, epoch recycling, packet
always reaped via the port (IOCP queues packets even for synchronously-completed
overlapped reads — verified by the buffered rows).
