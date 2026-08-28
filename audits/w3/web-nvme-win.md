# W3 web research: Windows NVMe streaming for Insignia (25.65GB FP8 decode)

Date: 2026-08-25. Rig (identified read-only, no benchmarks run):
`Get-PhysicalDisk` / `wmic` output:

| Drive | Bus | Rated (mfr) | Endurance | Role |
|---|---|---|---|---|
| **Samsung SSD 980 PRO 500GB** | NVMe (PCIe Gen4) | 7000 MB/s seq read | **300 TBW** (5yr) | the engine drive (7GB/s-class) |
| Samsung SSD 980 1TB | NVMe (PCIe Gen3) | 3500 MB/s seq read | 600 TBW (1TB) | candidate 2nd tier / mirror |
| WDC WD40EFAX 4TB | SATA HDD | ~250 MB/s | — | cold archive only |

Note: the prompt assumed a "600TBW-class" drive; the actual Gen4 drive present is the
**500GB 980 PRO = 300 TBW** (250GB=150, 500GB=300, 1TB=600, 2TB=1200 TBW per Samsung spec
via reviews). Endurance math below uses both numbers. Windows build on this box is 26200
(≥22621), so IoRing APIs are *available* if wanted.

Context consumed before research: AGENTS.md, `audits/synthesis.md` (feasibility: NVMe tier
56.5 ms/layer at 6.8GB/s; placement L=21/M=23/N=21 → ~8.06 GB/token from NVMe),
`audits/w2/colibri-io.md` (twin NO_BUFFERING handles, thread-pool-as-QD, iobench 5.85GB/s).

---

## 1. ReadFile+OVERLAPPED+IOCP(NO_BUFFERING) vs IoRing vs NtReadFile/ReadFileEx

**Verdict: IOCP + FILE_FLAG_NO_BUFFERING at QD8-16 x 1-4MB reaches the same ~6.5-7GB/s
envelope as anything else on this drive class; IoRing is worth +0-3% here — it is a
syscall-amortization win for many *small* ops, not a large-block throughput win. Build the
IOCP path; treat IoRing as a later swap-in.**

Evidence:

- **Yarden Shafir's IoRing_Demos (Microsoft-reverse-engineering source, GPL-3.0)** — the
  origin of the "+2%" figure (colibri/synthesis cite almost certainly traces here):
  *"On average, I/O rings are ~2% faster than I/O ports [IOCP] and ~3% faster than
  synchronous read"*; *"other testing showed improvement of up to 5-10%"* (with registered
  buffers). Her IoRingPerf table over ~4000 file reads: ReadFile 20.3-24.6s, ReadFileEx
  20.2-23.8s, IoRing Win32 20.0-22.7s, IoRing NT 20.0-22.4s — a ~1-2% spread, dominated by
  many-file/small-op costs. https://github.com/yardenshafir/IoRing_Demos
- **fio issue #1711** (IoRing ioengine request) repeats the same ~2-3% expectation.
  https://github.com/axboe/fio/issues/1711
- Internals write-ups (same author) document *why*: IoRing batches submissions/completions
  through one shared ring, and registered buffers skip per-I/O probe+lock of the user
  buffer — i.e., it removes **per-operation CPU**, it does not make the device faster.
  https://windows-internals.com/i-o-rings-when-one-i-o-operation-is-not-enough/
  https://windows-internals.com/ioring-vs-io_uring-a-comparison-of-windows-and-linux-implementations/
  https://windows-internals.com/one-year-to-i-o-ring-what-changed/ (22H2 adds writes etc.)
- **Scale check for our shape**: at 1MB blocks and 6.9GB/s the app sees ~7,000 completions/s;
  IOCP dequeue cost is well under 1µs each → <1% of one core. IoRing's amortization has
  almost nothing to amortize at 1-4MB blocks. (At 64KB blocks it would matter more.)
- **Line-rate is reachable with the boring API**: Microsoft DiskSpd is itself a
  FILE_FLAG_NO_BUFFERING engine (`-Sh` = `-Suw`, NO_BUFFERING+WRITE_THROUGH per the wiki),
  and the standard saturating recipe on NVMe is QD≥8-32 with several threads:
  https://github.com/microsoft/diskspd/wiki/command-line-and-parameters,
  https://www.nutanix.com/kb/9653 (example `-Sh -b1M -t8 -o8`),
  https://nvmexpress.org/crystal-disk-marks-new-release-measures-true-performance-of-nvme/
- **Measured on a 980 PRO-class drive**: ServeTheHome's 980 PRO 500GB review measured
  ~6900 MB/s reads (ATTO, 256MB-8GB files) — the 7GB/s rating is real on Windows.
  https://www.servethehome.com/samsung-980-pro-500gb-pcie-gen4-nvme-ssd-benchmarks-review/2/
  https://www.storagereview.com/review/samsung-980-pro-pcie-4-0-nvme-ssd-review
- **Engine-shaped Windows datapoint (already in-house)**: colibri's `iobench`
  (19MB random-aligned blocks, 8 threads, O_DIRECT/NO_BUFFERING) measured **5.85GB/s** on
  their Gen4 box (docs/windows.md) and V4 measured 86MB/s at QD1 → 696MB/s at QD8 on a
  *DRAM-less VHDX* — QD is the lever, API flavor is not (audits/w2/colibri-io.md §8.5, §10).
- ReadFileEx (APC-based) has no throughput advantage (same table above) and is awkward with
  a thread pool; NtReadFile buys nothing user-mode-visible over ReadFile for buffered-path
  avoidance. RavenDB's IoRing adoption (7.1) is about syscall counts on DB-sized ops, not
  sequential streaming: https://ravendb.net/blog/ravendb-7-1-one-io-ring-to-rule-them-all

Expected on the 980 PRO 500GB: **~6-6.9GB/s** at QD8-16 x 1-2MB NO_BUFFERING (drive limit
~6.9-7.0GB/s; real engines see 5.5-6.5GB/s once per-layer scheduling interleaves). The
synthesis.md estimate of 5.5-6.5GB/s is consistent; "~2 cores of overhead" is pessimistic
at 1MB+ blocks (sub-1 core incl. memcpys out of the ring).

---

## 2. FILE_FLAG_NO_BUFFERING: alignment rules, workarounds, and the buffered-sweep tax

**Rules (Microsoft Learn, "File Buffering")** —
https://learn.microsoft.com/en-us/windows/win32/fileio/file-buffering:

- *File access sizes, including the optional file offset in the OVERLAPPED structure ...
  must be for a number of bytes that is an integer multiple of the volume sector size.*
- *File access buffer addresses ... should be physical sector-aligned* (logical vs physical
  sector: Advanced Format drives are 4096B physical emulating 512B; query
  `IOCTL_STORAGE_QUERY_PROPERTY` → `STORAGE_ACCESS_ALIGNMENT_DESCRIPTOR`. MS "strongly
  recommends" aligning unbuffered I/O to the **physical** sector size).
- MS's own suggested allocator: **VirtualAlloc** (page-aligned ⇒ sector-aligned because
  sector ≤ page on direct-access storage).
- **Does it bypass the cache?** Same doc, overview: the flag exists *"to disable system
  caching of data being read from or written to the file"* — with it, *"this local buffer
  is, in effect, the only file buffer that exists for this operation"* — i.e. the DMA lands
  in your buffer, no cache copy, no standby-list residency. Caveat from the doc: this says
  nothing about the **drive's own hardware cache** (DRAM/SLC), which you cannot turn off.
- Classic KB phrasing: "disk reads and writes must be done on sector boundaries, and buffer
  addresses must be aligned on disk sector boundaries"
  (archived KB https://ftp.zx.net.nz/pub/archive/ftp.microsoft.com/MISC/KB/en-us/99/794.HTM).

**Workarounds for unaligned offsets/lengths** (safetensors tensors are header-padded but
not 4K-guaranteed):
1. **Over-read + trim**: expand `[off, off+n)` to `[off & ~4095, (off+n+4095) & ~4095]`,
   read aligned, memmove within the ring slot. Cost: ≤8KB extra I/O + a memmove. This is
   colibri's window read (`base = off0 & ~4095`, `len = (need+4095) & ~4095`, +8KiB slack).
2. **Twin handles** (colibri `compat.h:296-311`, `compat_open_direct`): keep the
   NO_BUFFERING handle for bulk aligned reads and a normal buffered handle for tails,
   headers, and metadata; kimi_k3 does the aligned body via O_DIRECT + sub-4K tail via a
   buffered pread ("O_DIRECT wants aligned lengths"). Two `CreateFile` calls on the same
   path are legal and the file system handles coherence (read-only here anyway).
3. Never use CRT `lseek` on the NO_BUFFERING fd (returns -1 on UCRT — colibri measured;
   use `GetFileSizeEx`), never `_read` in text mode (0x0A translation corrupts weights).

**Buffered-sweep penalty on a 16GB box (why not just buffered reads):**
- **memcpy tax**: buffered read = device→system-cache page→memcpy into your buffer; with
  NO_BUFFERING the device DMAs straight into your buffer. On the cache-manager path you pay
  one full extra copy of every byte of the model, *plus* double-buffering against your own
  RAM tier (synthesis.md already flags this as fatal for >RAM working sets).
- **Cache pollution / standby churn**: a 25.65GB cyclic sweep through ~13GB of standby
  evicts everything continuously; available RAM collapses, other processes' standby gets
  trimmed, and any *actual* cache hit-rate for a dense cyclic working set is near zero
  anyway (see §7 model) — you pay the tax for nothing. Colibri's measured deltas: buffered
  0.8GB/s vs O_DIRECT 2.3GB/s on ext4-in-VHDX (st.h comment); 7.1 direct vs 2.9 buffered on
  kimi (1.8 effective once resident weights eat cache headroom) — audits/w2/colibri-io.md
  §2.1, §8.3. Public Windows datapoint: StarWind's DiskSpd `-Sh` on/off runs on Samsung NVMe
  show 8-23% deltas by pattern: https://www.starwindsoftware.com/blog/benchmarking-samsung-nvme-ssd-960-evo-m-2/
  (Linux writes analog, LWN: buffered sequential capped ~700-800MB/s on consumer NVMe:
  https://lwn.net/Articles/977526/). Expect the gap on a Gen4 7GB/s drive to be *larger*
  than 23%, not smaller — the memcpy and cache-manager overhead scale with bytes, and the
  16GB host cannot absorb 25.65GB.

**Conclusion**: NO_BUFFERING for all weight streaming; buffered twin handle only for
safetensors headers (~a few KB, unaligned) and any tails not worth over-reading.

---

## 3. Pinned destination ring: what's required, what breaks, and the right size

- **Pinning is NOT required for the ReadFile destination.** Any sector-aligned allocation
  (VirtualAlloc / _aligned_malloc) works as an unbuffered read target — the MS alignment
  contract (§2) is about alignment, not locking. Nothing in Win32 asks for page-locked
  read buffers.
- **Pinning IS required for the H2D copy source** if you want it asynchronous:
  `cudaMemcpyAsync` from **pageable** memory makes the runtime copy through an internal
  pinned staging buffer; the call is effectively synchronous ("returns once the pageable
  buffer has been copied to the staging memory"; runtime "will fall back to a synchronizing
  operation ... if pageable host memory is used"). NVIDIA forum confirmations:
  https://forums.developer.nvidia.com/t/clarification-on-cudamemcpy-synchronization-behavior-with-pageable-memory-and-non-blocking-streams/356640
  https://forums.developer.nvidia.com/t/cuda-8-0-cudamemcpy-with-pageable-memory/46606
  https://forums.developer.nvidia.com/t/synchronization-of-cudamemcpyasync-for-pageable-memory/191014
  and https://stackoverflow.com/questions/70760005/. So a plain VirtualAlloc ring *can* be
  read into, but every H2D from it then costs an extra CPU memcpy and serializes against
  the copy engine — exactly the overlap Insignia needs (read L+1 ‖ upload L).
- **Limits on Windows (WDDM)**: pinned host memory comes out of kernel resources — x64
  nonpaged pool is capped at min(128GB, ~75% of RAM)
  (https://techcommunity.microsoft.com/discussions/windows10space/about-limitations-of-nonepaged-pool-size-in-windows-10/3931195,
  https://woshub.com/huge-memory-usage-non-paged-pool-windows/). In practice users hit a
  **~50%-of-RAM wall** for cudaHostAlloc/cudaHostRegister on Win10/11:
  https://forums.developer.nvidia.com/t/change-limit-of-50-for-cudahostalloc-pinned-memory-on-windows-10-11/228235,
  https://forums.developer.nvidia.com/t/cudahostregister-strange-unexpected-behaviour-under-windows-10/77439;
  large `cudaHostRegister`s have caused **whole-system hangs** or CUDA OOMs:
  https://forums.developer.nvidia.com/t/arbitrary-device-limit-on-pinned-host-memory/34524,
  multi-GB failures even on 128GB hosts:
  https://forums.developer.nvidia.com/t/max-amount-of-host-pinned-memory-available-for-allocation/56053.
  On this 16GB box: **do not pin ≥8GB**; stay ≤4GB. CUDA docs also warn excessive pinned
  allocations degrade the whole system (paging headroom).
- **Recommendation**: one **2-4GB pinned ring** (`cudaHostAlloc`, or VirtualAlloc +
  `cudaHostRegister` registered once at startup before any GPU work — never re-registered
  per slot), 4K-aligned by construction, carved into 1-2MB slots (2GB = 1024 x 2MB slots;
  QD16 in flight, the rest as upload/consume runway). Justification: at 6.9GB/s disk and
  25GB/s PCIe, 2GB is ~290ms of runway vs a 56.5ms/layer NVMe budget — plenty for
  1-2-token lookahead incl. MTP verify; 4GB (25% of RAM) is the safe ceiling on 16GB.
  Colibri's qwen36 uploader uses *malloc* staging + sync copies (audits/w2 §8.6) — pinned
  is strictly better for our overlap goal; colibri's VirtualLock shim (compat.h:195-212)
  shows the WS-growth trick if we ever need host-pinned-without-CUDA.

---

## 4. PrefetchVirtualMemory (Win8+): not a line-rate primitive

- MS docs frame it as a pure optimization hint, no guarantees:
  https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-prefetchvirtualmemory
- **microsoft/Windows-Dev-Performance#108**: a 500MB file mapping, prefetch the whole view,
  then touch every byte: total page faults **identical** with/without (161,027); the
  prefetch call itself faulted only ~314 pages; *"the performance was slightly worse with
  PrefetchVirtualMemory"*; reporter's verdict: it "does not do its only job". No Microsoft
  response. https://github.com/microsoft/Windows-Dev-Performance/issues/108
- **microsoft/WindowsAppSDK#1992**: same shape — *"During PrefetchVirtualMemory I got 26
  page faults, and during the actual processing of the file I got 12,120"*.
  https://github.com/microsoft/WindowsAppSDK/issues/1992
- Nuance: it is not *always* useless — Chromium measured that it "gets more useful work
  done" for image pre-reads (https://issues.chromium.org/41212499), and Bruce Dawson
  describes it as the mmap analog of ReadFile prefetching
  (https://randomascii.wordpress.com/2014/12/10/hidden-costs-of-memory-allocation/). But
  even when it works it is page-granular readahead behind the memory manager, not a
  user-shaped large async device read — no QD control, no buffer ownership, no completion.
- **Verdict: colibri's claim is confirmed by independent evidence** — do not build the
  30GB streaming path on PrefetchVirtualMemory. Use it (if at all) only as a
  touch-accelerator for small mapped ranges. llama.cpp's load-time pain led to the same
  recommendation in the wild: *"Better to use an explicit hint: PrefetchVirtualMemory on
  Windows and madvise on Unix"* — and that is for *load*, not steady-state decode
  (https://github.com/ggml-org/llama.cpp/issues/705 thread).

---

## 5. llama.cpp mmap for >RAM models: why engines move to explicit IO

GitHub trail (ggml-org/llama.cpp):
- #91 — original mmap adoption ("files larger than RAM", address-space based).
- **#705** — *"Windows page fault disk I/O slow on first load"*: after PR #613, loads on
  disk went from 60-180s to **~15 minutes**, disk pegged at 100%; thread conclusion is
  explicit prefetch hints (quote above) or disable the change. This is the canonical
  "page-fault-driven loading is not line rate" incident.
  https://github.com/ggml-org/llama.cpp/issues/705
- #864 — mmap unsuitable when RAM < model (65B 4-bit): every generation pass streams
  through page faults. https://github.com/ggml-org/llama.cpp/issues/864
- #5207 — low-RAM recipe ends up at `--mlock` (load once, lock, OS swap is the overflow).
  https://github.com/ggml-org/llama.cpp/issues/5207
- **Discussion #18758** — the quantitative one: with model ≈ 100-150% of RAM, mmap+page
  cache beat O_DIRECT on *reload* (O_DIRECT reload is *"at least 10x longer"*); but
  replacing fault-streaming with **explicit pread of expert slices** gave **+13-14%
  end-to-end**, and reordering the file for co-activation cut reads/token 1418 → 775 →
  ~370 (2.23x) — i.e. the win comes from *engine-controlled* reads and layout, either IO
  path. Also: blanket prefetch on a model bigger than RAM *"causes thrashing ... minutes of
  delay"*. https://github.com/ggml-org/llama.cpp/discussions/18758
- #24037 — "mmap kills performance with Qwen 397B" (fragmentation/eviction),
  #19883 — mmap pageout risk without mlock.
- HN on "30B in 6GB RAM" — community pushback that per-token faults through disk cap far
  below disk line rate: https://news.ycombinator.com/item?id=35393284

**Effective page-fault bandwidth**: no public issue states a clean "2-3GB/s" figure
(found none in this pass — honest gap). What exists: relative numbers above plus the
granularity argument — demand paging serializes ~4KB faults with per-fault overhead;
4KB/1µs ≈ 4GB/s is the optimistic single-thread ceiling, degrading hard once eviction
interleaves with faulting (the 15-minute loads). ~2-3GB/s effective is a reasonable
planning number for Windows mapped sections with readahead clusters, but treat it as
derived, not measured. The direction is unanimous: **engines that must stream >RAM
eventually drop mmap for explicit IO** (llama.cpp's own hints/mlock knobs, colibri's
pread-first st.h, koboldcpp mmap complaints). Insignia's zero-copy mmap loader is fine for
the resident set; the NVMe tier should be explicit NO_BUFFERING reads into the pinned ring.

---

## 6. 66 files, one IOCP: queue depth, NCQ, handle sanity

- **Handles**: 66 concurrently-open read handles is nothing on Windows (per-process handle
  ceiling is ~16 million; the colibri shard index keeps 512 files open by design).
- **Device queue depth**: stornvme exposes `IoQueueDepth` under
  `HKLM\SYSTEM\CurrentControlSet\Services\stornvme\Parameters\Device` (present in the
  driver's registry reads — https://djdallmann.github.io/GamingPCSetup/CONTENT/RESEARCH/FINDINGS/registrykeys_stornvme.txt;
  community tweaks use 64). Storport above it has *"no predefined limits on outstanding
  requests per adapter"* (https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/storport-queue-management),
  and NVMe devices ship dozens of queues x up to 64K entries (Windows cert minimum 64
  queues — https://news.ycombinator.com/item?id=28708113). Practical reading: **excess QD
  doesn't fail, it queues**; the app-side global outstanding count is the control knob.
- **Interleaved sequential streams**: NVMe has no seek penalty; throughput is a function of
  **aggregate** QD across all streams, not per-file QD. Evidence: colibri iobench 8 threads
  x 19MB random-aligned blocks → 5.85GB/s (one physical device, global QD); V4 86MB/s QD1
  → 696MB/s QD8 on a weak device (audits/w2); NVM Express/CrystalDiskMark guidance
  (QD≥32, ≥8 workers to saturate): https://nvmexpress.org/crystal-disk-marks-new-release-measures-true-performance-of-nvme/.
  Caveat flagged honestly: I found no rigorous public benchmark of 66 interleaved
  *sequential* streams on one consumer Gen4 drive; expect 85-95% of single-stream rate
  (firmware/SLC bookkeeping), and QD8-16 global is comfortably in the flat region for 1MB+
  blocks. Our shards are read nearly layer-sequentially anyway (layers-N.safetensors), so
  interleave is shallow.
- **Pattern**: 66 CreateFile(GENERIC_READ, FILE_FLAG_NO_BUFFERING, FILE_FLAG_OVERLAPPED)
  handles → CreateIoCompletionPort(one port) → 1-2 issuer threads keep global outstanding
  ≈ 8-16 → completions dequeue in submission-agnostic order (IOCP is LIFO-ish by default,
  fine for a ring). No per-file threads (colibri's 8-blocking-pread-threads model is the
  fallback if we skip IOCP; audits/w2 §3.2 shows they never needed more).

---

## 7. SSD endurance: the honest math (and the correction)

**Worst case** (placement L=21/M=23/N=21, all 21 NVMe layers read every token, MTP verify
reads weights once — bandwidth-bound ⇒ free per synthesis.md):

- 21 layers x 383.87MB = **8.06 GB/token** from NVMe.
- At 30 tok/min sustained: 43,200 tokens/day → **~348 TB/day** (~14.5 TB/hour; equivalently
  a continuous 4.0GB/s read stream). 
- **Naive days-to-death if reads counted against TBW**: 980 PRO 500GB (300TBW) → 300/348 ≈
  **0.86 days (~21 hours)**; a 600TBW 1TB drive → ~1.7 days. Halved (~1.7 / 3.4 days) at a
  50% cache hit rate. This is the "drive as consumable" scenario.
- **Correction — TBW is *write* endurance.** Reads do not decrement TBW; host reads cause
  no NAND program cycles. The checkpoint is written once (25.65GB) and then only read.
  Read disturb exists but controllers scrub it with rare background rewrites — orders of
  magnitude below read volume. So the days-to-death panic math above is the *pessimistic
  fiction*; realistic wear from pure-NVMe decode is negligible even at 24/7. The real
  sustained-read risks are **thermals** (a 4-7GB/s stream throttles a heatsink-less M.2
  after minutes-tens-of-minutes; throttling halves bandwidth, not endurance) and, on
  DRAM-less drives, FTL pressure — the 980 PRO has DRAM, so neither is acute.
- TBW ratings and 0.33 DWPD for the 500GB model:
  https://www.storagereview.com/review/samsung-980-pro-pcie-4-0-nvme-ssd-review,
  https://www.servethehome.com/samsung-980-pro-500gb-pcie-gen4-nvme-ssd-benchmarks-review/2/,
  DWPD math per https://www.enterprisestorageforum.com/hardware/ssd-lifespan-how-long-will-your-ssd-work/.

**RAM-tier caching model (cyclic working set vs cache)** — W = 25.65GB (all-stream
cyclic), C = 13GB standby:
- Random-replacement / uniform bound: C/W = **50.7%** — the "classic C/W ≈ 50%" number.
- Strict LRU (and FIFO) on a loop larger than the cache: **0%** — the classic cyclic
  anomaly. A dense decoder is a *deterministic* loop (every layer every token), so
  Windows' standby list (priority-aging FIFO-ish, not LRU) lands near the 0% bound,
  nondeterministically. Colibri's 66% expert hit at 25% residency is **MoE routing skew**,
  which a dense model does not have — do not import that number.
- **Knife-edge specific to our placement**: the NVMe-tier cycle is only 21 x 383.87MB =
  **8.06GB**. If ≥8.06GB of standby survives, the next token's reads all hit cache (~100%
  at RAM speed); 1GB short → ~0%. Free standby on this box (16GB − engine WS − OS − GPU
  staging) is realistically 6-10GB → straddles the edge → **flapping** between "disk-free
  token" and "8GB/token token". Unpredictable latency is worse than either stable regime.
- **NO_BUFFERING** = 0% hits by construction, but *deterministic*: 8.06GB/token at line
  rate, no standby churn, no memcpy tax, stable placement optimizer.
- **Recommendation (default)**: NO_BUFFERING + **engine-managed** RAM tier (explicit
  residency, already built in Insignia). The hit rate then comes from placement (stationary
  RAM layers), not from standby luck; and since reads don't wear NAND (see correction), the
  RAM tier is justified by **speed** (15.4ms/layer RAM-stream vs 56.5ms NVMe) — not by
  endurance. If a future placement ever lets the RAM tier hold the *entire* 8.06GB NVMe
  cycle, that's a step-change (steady-state disk traffic → ~0); size placement to either
  clear 8.06GB+stationary or stay well under RAM — never straddle.

---

## 8. Open-source Windows C++ IOCP/streaming references beyond colibri

| Repo | What to steal | License |
|---|---|---|
| **colibri** (in-house clone) | twin NO_BUFFERING handles, PIPE SPMC pool, staging slabs, iobench harness | Apache-2.0 (LICENSE verified in clone) |
| **microsoft/diskspd** — https://github.com/microsoft/diskspd | the reference NO_BUFFERING + threads/QD/-o/-b/-Sh benchmark engine; steal target parameters & measurement hygiene | MIT |
| **ned14/llfio** — https://github.com/ned14/llfio | P1031 low-level file IO; async file handles on IOCP, alignment-aware; good test suite of Windows IO edge cases | Apache-2.0 |
| **Pagghiu/SaneCppLibraries (SC::Async)** — https://github.com/Pagghiu/SaneCppLibraries, https://pagghiu.github.io/SaneCppLibraries/libraries/async/ | completion-based event loop: files+sockets, **IOCP on Windows / io_uring on Linux / kqueue** — the cleanest small cross-backend file-async implementation to read | MIT (verify in-repo before copying) |
| **chriskohlhoff/asio** — https://github.com/chriskohlhoff/asio | windows_random_access_handle (OVERLAPPED+IOCP) backend; issue #79 documents the SetFileIoOverlappedRange/kernel-lock pain that IoRing later fixed — useful background | BSL-1.0 |
| **newbiediver/rioring** — https://github.com/newbiediver/rioring | Registered I/O (RIO) + io_uring backends — but **sockets only, no files**; study only | MIT |
| **libuv** — https://github.com/libuv/libuv | uv_fs work-queue: shows the "threads for file IO, IOCP for sockets" compromise most portable stacks make on Windows | MIT |
| **apriorit/win-iocp-copying** — https://github.com/apriorit/win-iocp-copying | minimal multi-file IOCP copying demo — a 1-hour teaching sample | (sample; check) |
| **yardenshafir/IoRing_Demos** — https://github.com/yardenshafir/IoRing_Demos | the IoRing perf harness (and the +2% number) if/when we A/B IoRing | GPL-3.0 (bench code only — do not copy into Insignia, read-only reference) |

Also: Windows Internals I/O Ring series (windows-internals.com links in §1) is the design
doc for IoRing internals; RavenDB 7.1 blog for a production IoRing adoption story.

---

## Recommended read stack (synthesis)

1. **66 NO_BUFFERING + FILE_FLAG_OVERLAPPED handles** (twin buffered handles kept for
   safetensors headers + sub-4K tails), all bound to **one IOCP**.
2. **1-2 issuer threads**, global outstanding **QD 8-16**, block size **1-2MB**
   (4K-aligned windows via over-read+trim; +8KiB slack per slot, colibri-style).
3. Destination = **2-4GB pinned ring** (cudaHostAlloc; 1-2MB slots) → `cudaMemcpyAsync`
   H2D is genuinely async, overlapping reads of L+1 with upload/consume of L.
4. **PrefetchVirtualMemory: no.** mmap: only for the already-resident/zero-copy portion.
5. Expected steady throughput on the 980 PRO 500GB: **~6-6.9GB/s cold-path** (5.5-6.5GB/s
   in-engine with per-layer scheduling), which prices the NVMe tier at ~56.5ms/layer
   (0.5 tok/s all-NVMe; 1.05-1.2 tok/s with MTP per synthesis.md placement).
6. IoRing: skip for v1 (worth ≤3% at our block sizes; available on this build 26200 if we
   A/B later, GPL demo numbers say ~2%).
7. Endurance: pure-read decode does **not** consume TBW (writes do); naive read-as-wear
   math says 0.86-1.7 days at 348TB/day, but real wear ≈ 0 — mind thermals instead. RAM
   tier stays for speed; make its sizing either clear or duck under the 8.06GB NVMe-cycle
   knife-edge; default NO_BUFFERING (deterministic 8.06GB/token) over standby-list luck.
