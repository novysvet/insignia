# w3 audit — async NVMe weight reader: pinned RAM ring + IOCP + shard-major staging

Audit date: 2026-08-25. Scope: complete implementable design of the NVMe tier's read path for
Qwen3.8-27B-FP8 on this exact rig (4070 SUPER / 5600X / 15.9 GiB RAM / model on E:).
Inputs: `audits/synthesis.md`, `audits/w2/colibri-io.md`, `audits/w2/loader-27b-spec.md`
(census + alignment sections re-verified), live header parse of `outside.safetensors`
(read-only), `Get-PhysicalDisk` / `Get-Partition` / `Get-PSDrive`, one 64 MiB read-only
timing probe. Nothing written outside this file; no builds; no git changes.

---

## 0. TL;DR

1. **Hardware correction that changes everything**: E: (model dir) is a **Samsung 980 1TB,
   PCIe Gen3, ~3.5 GB/s class, DRAM-less (HMB), 600 TBW** — NOT a 7 GB/s Gen4 drive. The
   Gen4 drive (980 PRO 500GB, 6.9 GB/s, 300 TBW) is C: with 93.8 GB free. Per-layer NVMe
   cost is ~116 ms, not 58 ms; the synthesis tier table's 56.5 ms/layer assumed a drive we
   don't host the model on.
2. Shard-major staging **verified**: every shard is gapless (0 inter-tensor gaps, ends
   flush), F8 sizes are 4096-multiples, per-shard constant phase — so each layer shard is
   ONE sequential stream of 2 MiB blocks from byte 0 (header rides along). Verified live:
   `outside.safetensors`' text tensors (lm_head, embed, norm) are a **contiguous prefix
   [0, 5,085,641,920)** and vision is a pure tail → outside needs **one** range, not two.
3. Reader = CreateFileW per file `NO_BUFFERING|OVERLAPPED`, one IOCP, 2 threads on
   `GetQueuedCompletionStatusEx`, ring = `VirtualAlloc` + `cudaHostRegister`, 4 slots ×
   368 MiB (184×2 MiB) = 1.44 GiB default (5 slots / 1.80 GiB if RAM allows). Full code in §3.
4. Scheduling is trivial because decode is strictly sequential: the reader just walks the
   epoch plan (N-tier shards in layer order) with unit-depth = slots−1 and ≥8×2 MiB in
   flight; re-arm per epoch; prefill reads the plan once per prompt.
5. Endurance: at N=21 (~8 GB/token) the drive streams continuously while generating —
   **~12 TB reads/hour; the 600 TBW rating ≈ 75,000 tokens total** (TBW is a write rating;
   read-disturb wear is slower but unbounded streaming is still the design killer).
   Mitigation ranked: minimize N (RAM is the deterministic cache), optionally split/mirror
   shards across E: + C: (halves wear, ~2.3× aggregate bandwidth, 93.8 GB free on C:).
6. The mission's "OS cache absorbs ~50% of re-reads" model is **wrong for LRU**: cyclic
   access with working set > cache gives LRU **0%** hits (classic thrashing; RRIP paper).
   Worse, on this box the standby list can never hold the whole M+N remainder
   (16.3 GB > ~10 GB available), so buffered mode is structurally in the thrash regime →
   NO_BUFFERING deterministic default; buffered handles kept for one-shot startup reads and
   as a bench A/B.
7. Micro-bench spec + harness sketch in §6; Windows specifics (sector query, flag myths,
   IoRing skip rationale, queue depth, thread affinity/priority) in §7.

---

## 1. Hardware reality check (measured this session)

`Get-PhysicalDisk` + `Get-Partition` mapping:

| disk | model | bus | drive letter | class | endurance |
|---|---|---|---|---|---|
| 0 | WDC WD40EFAX 4TB | SATA HDD | D: | irrelevant | — |
| 1 | Samsung SSD 980 PRO 500GB | NVMe **Gen4** | **C:** (system) | 6,900 MB/s seq read, 1M IOPS | **300 TBW** |
| 2 | **Samsung SSD 980 1TB** | NVMe **Gen3** | **E:** (model) | 3,500 MB/s seq read, 500K IOPS, **DRAM-less HMB** | **600 TBW** |

Sources: [TechPowerUp 980 1TB](https://www.techpowerup.com/ssd-specs/samsung-980-1-tb.d58),
[Samsung 980 product page](https://www.samsung.com/au/memory-storage/nvme-ssd/980-1tb-nvme-pcie-gen-3-mz-v8v1t0bw/)
(3,500 MB/s, DRAM-less/HMB, 600 TBW),
[TechPowerUp 980 PRO 500GB](https://www.techpowerup.com/ssd-specs/samsung-980-pro-500-gb.d46),
[Samsung 980 PRO announcement](https://news.samsung.com/us/memory-ssd-980-pro-gaming-pc).

Consequences:

- **All feasibility math must use ~3.2–3.4 GB/s effective** (Gen3 TLC sustained, QD≥8,
  2 MiB blocks) for the model dir as-is. One 383.87 MB linear layer = **~116–120 ms**;
  one 372.31 MB full-attn layer = ~113–117 ms. N=21 → **2.4 s/token** of pure NVMe.
- The 980 1TB is DRAM-less: HMB borrows up to ~64 MB host RAM for the FTL and adds some
  FTL latency variance under deep queues — irrelevant at our sizes, but it shaves another
  ~64 MB off the RAM budget and makes QD saturation slightly more important.
- **C: mirror/split opportunity (actionable, no new hardware)**: 93.8 GB free on the Gen4
  PRO. Copying or splitting the 30.9 GB checkpoint across E:+C: gives ~3.3+6.4 ≈ 9.7 GB/s
  aggregate (colibri's split/mirror mode is the existence proof — `st_mirror_add` /
  `COLI_MODEL_DIRS` split, audits/w2/colibri-io.md §2.5, §9) and halves per-drive read
  traffic. This is the single biggest lever after RAM-tier maximization. Cost: 300 TBW on
  the (smaller) PRO if it takes half the traffic; see §5.
- 64 MiB buffered probe (allowed live probe, layers-32.safetensors): **2,021 GiB/s at
  offset 0, 1,851 GiB/s at file middle** — i.e. both regions are currently sitting in the
  OS standby list from prior audit reads. Two lessons: (a) cache hits are *literally free*
  when they happen (epoch 1 after a warm run is fast), (b) they are not a plan — see §5
  for why steady-state hits are ~0 for our cyclic pattern.

---

## 2. Alignment analysis → shard-major staging (verified)

### 2.1 Census facts used (from audits/w2/loader-27b-spec.md §4, re-verified where cheap)

1. Only 45/1606 tensor starts are 4096-aligned (144/1606 512-aligned) — tensor-granular
   O_DIRECT windows need per-tensor phase math. **Not needed** under shard-major staging.
2. **All 407 F8 sizes are 4096-multiples**; every ≥1 MiB tensor is 4096-sized.
3. **Zero pad gaps in any shard** (all 66): every tensor begins exactly at the previous
   tensor's end and every shard ends flush at its last tensor's end. Therefore tensor
   offsets within a shard are **contiguous ascending with zero gaps** → a whole-shard read
   from byte 0 is a *perfectly sequential* stream. ✓ (mission's proposed verification).
4. Per-shard constant phase (first-F8 residue mod 4096): 616/640 (linear 1/2-digit),
   3728/3744 (full-attn 1/2-digit), 1840 (mtp). Irrelevant once we stream from offset 0.
5. Shard sizes: linear 383,865,448/472 B; full-attn 372,313,744/760 B; mtp 477,202,224 B;
   outside 6,007,102,112 B. data_start = header 8+JSON = 2,600/2,624 (linear), 2,320/2,336
   (full), 2,480 (mtp), 38,080 (outside). The ≤3 KB header rides along in block 0 and is
   skipped by consumers (tensor addresses stay *absolute file offsets*, so consumers add
   `abs_off` to the slot base with zero remapping).

### 2.2 The outside.safetensors question — settled by live header parse

Parsed this session (read-only, 8-byte len + JSON only):

```
data_start 38080, file_size 6007102112, 336 tensors, 0 gaps
tensor 0: [0, 2542796800)           lm_head.weight
tensor 1: [2542796800, 5085593600)  model.language_model.embed_tokens.weight
tensor 2: [5085593600, 5085603840)  model.language_model.norm.weight
tensor 3: [5085603840, ...)         model.visual.blocks.0...   <- vision starts here
last:     model.visual.pos_embed.weight ends at 6007064032
text bytes = 5,085,603,840; text region is a CONTIGUOUS PREFIX; vision is a pure tail
```

So outside's keep-set = **one range [0, 38,080 + 5,085,603,840) = [0, 5,085,641,920)** —
the mission's "two ranges" is unnecessary (vision is not interleaved; the census's
"2,304 B biases break the chain" comment concerned per-tensor window *phase*, not gaps).
The range builder below still handles the general N-range case for free.

### 2.3 Block math (2 MiB stream blocks, read from byte 0)

| shard | stream bytes | full 2 MiB blocks | EOF remainder | direct tail (floor512) | buffered tail (<512 B) | slot blocks |
|---|---|---|---|---|---|---|
| linear 1-digit (8) | 383,865,448 | 183 | 86,632 | 86,528 | 104 | 184 |
| linear 2-digit (40) | 383,865,472 | 183 | 86,656 | 86,528 | 128 | 184 |
| full-attn 1-digit (2) | 372,313,744 | 177 | 1,117,840 | 1,117,696 | 144 | 178 |
| full-attn 2-digit (14) | 372,313,760 | 177 | 1,117,856 | 1,117,696 | 160 | 178 |
| mtp | 477,202,224 | 227 | 1,148,720 | 1,148,416 | 304 | 228 |
| outside (text only) | 5,085,641,920 | 2425 | 48,320 | 48,128 | 192 | 2426 (one-shot) |

`NO_BUFFERING` requires offset, length *and* destination aligned to the volume's logical
sector size ([File Buffering](https://learn.microsoft.com/en-us/windows/win32/fileio/file-buffering);
[OpenFileById notes](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-openfilebyid):
"File access must begin at byte offsets within the file that are integer multiples of the
volume sector size"). 2 MiB blocks from offset 0 satisfy 512e and 4Kn alike. The only
misaligned quantity in the whole design is the **<512 B EOF tail** of each file — handled
colibri-style: sector-floor the last direct read, fetch the sub-sector remainder through a
buffered twin handle (≤1 tiny sync read per file per epoch; also warms exactly one page of
cache per shard — negligible pollution). `mtp` (228 blocks) and `outside` exceed the
184-block layer slot → they are **startup one-shots** (§3 `read_once`), not ring units —
which also matches policy: mtp + embed/lm_head want VRAM/RAM residency, not streaming
(synthesis: lm_head MUST be VRAM-resident; MTP draft runs 4 ms warm).

### 2.4 Range-plan builder (build-time, feeds INSIDX02 or runtime init)

Python (matches the tools/index_safetensors.py pipeline; emits prefix ranges + tails):

```python
def build_ranges(header: dict, data_start: int, keep_pred) -> list[tuple[int,int,int,int]]:
    """file -> list of (off, direct_len, tail_off, tail_len) block descriptors.
    keep_pred(name) selects wanted tensors (False for model.visual.*).
    Verified outcomes for this checkpoint: layer/mtp shards -> whole file, ONE range,
    0 gaps; outside -> ONE prefix range [0, 5_085_641_920)."""
    iv = sorted((v["data_offsets"][0], v["data_offsets"][1]) for k, v in header.items()
                if k != "__metadata__" and keep_pred(k))
    merged = []                       # gapless-merge adjacent intervals
    for lo, hi in iv:                 # gaps between kept tensors would just become range
        if merged and lo <= merged[-1][1]:   # boundaries (block stream skips them); here
            merged[-1][1] = max(merged[-1][1], hi)  # every shard collapses to ONE prefix.
        else: merged.append([lo, hi])
    BLK = 1 << 21
    out = []
    for lo, hi in merged:
        base = data_start + lo; end = data_start + hi; off = base
        while off < end:
            n = min(BLK, end - off); d = n & ~511           # direct: sector-floored
            if d: out.append((off, d, 0, 0))
            if n - d: out[-1] = (off, d, off + d, n - d)    # <512B tail via buffered twin
            off += n
    return out
```

The C++ consumer (`NvReader::init`) takes the same intervals and builds the identical
block table (code in §3), asserting the single-prefix-range property that
`map(abs_off) = slot_base + abs_off` relies on.

---

## 3. The reader (full C++/MSVC, project style)

Design in one breath: one `CreateFileW` direct handle (`FILE_FLAG_NO_BUFFERING |
FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN`) + one buffered twin per file; a single
IOCP associated with every direct handle (completion key = file index); 2 reader threads
park on `GetQueuedCompletionStatusEx` (batches of 16, 1 s poll for shutdown) and
**self-arm**: after draining completions they top up in-flight to QD by walking the epoch
plan; the ring is `VirtualAlloc` + `cudaHostRegister` (DMA-able for `cudaMemcpyAsync`,
page-locked against standby eviction, and directly readable by CPU GEMV threads); a
request is `{slot, block-in-slot}` → `ReadFile(direct, ring + pos*2MiB, len, NULL, &ov)`
with one preallocated `Req` per ring block (a ring block can only have one outstanding
read by construction: slots recycle only after the consumer releases the unit); unit
completion = per-unit manual-reset event (+ atomic countdown); consumer = PCIe copy thread
(`cudaMemcpyAsync` from the pinned slot on the copy stream) or CPU GEMV threads reading
the slot in place; teardown = stop flag + `CancelIoEx` per handle + drain; errors = bounded
retry, EOF-tail tolerance, fatal flag wakes all waiters.

```cpp
// include/insignia_nvme.hpp — async NVMe shard reader: shard-major staging into a pinned ring
#pragma once
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cuda_runtime.h>
#include <atomic>
#include <cstdint>
#include <utility>
#include <vector>
namespace insignia {
using u8=uint8_t; using u32=uint32_t; using u64=uint64_t;
struct NvFileSpec { const wchar_t* path; std::vector<std::pair<u64,u64>> intervals; };// keep-set, [off,end) rel. to data_start

class NvReader {
public:
  static constexpr u64  BLK=2ull<<21;      // 2 MiB: 4096-multiple, 512e/4Kn-safe, per census block math
  static constexpr u32  SLOT_BLOCKS=184;   // largest layer shard (383,865,472 B = 183.04 blocks)
  void init(const std::vector<NvFileSpec>&specs,u32 slots,u32 qd,u32 nthreads,DWORD affinity);
  void read_once(u32 file,u64 off,u64 len,void*dst);        // startup path (mtp/outside/embed): buffered twin, blocking
  u32  nunits()  const { return u32(units_.size()); }
  u8*  unit_wait(u32 unit);                                 // slot base == file byte 0; nullptr if fatal
  u8*  map(u32 unit,u64 abs_file_off) { return unit_wait(unit)+abs_file_off; } // valid: gapless prefix-from-0 plans
  void unit_release(u32 unit);                              // slot recyclable once PCIe/GEMV consumers are done
  void epoch_begin();                                       // re-arm: rewind plan cursor (one 64-layer sweep)
  bool healthy() const { return !fatal_.load(std::memory_order_acquire); }
  void shutdown();
  ~NvReader(){shutdown();}  NvReader()=default;
  NvReader(const NvReader&)=delete; NvReader&operator=(const NvReader&)=delete;
private:
  struct Blk { u64 off; u32 len; };                         // one direct read (EOF block is sector-floored)
  struct Unit{ u32 file,slot,first,nblk,tail_off,tail_len; std::atomic<u32>left; HANDLE done; };
  struct Req { OVERLAPPED ov; u32 unit,blk,tries; };        // indexed by ring block -> single outstanding read
  struct File{ HANDLE direct,buffered; u64 size; };
  u8* slot_base(u32 s) const { return ring_+u64(s)*slot_bytes_; }
  void build(const std::vector<NvFileSpec>&specs);
  void issue(u32 unit,u32 blk);  void top_up();  void complete(OVERLAPPED_ENTRY&e);  void worker();
  static DWORD WINAPI tramp(LPVOID p){ reinterpret_cast<NvReader*>(p)->worker(); return 0; }
  std::vector<File> files_; std::vector<Blk> blocks_; std::vector<Unit> units_; std::vector<Req> reqs_;
  u8* ring_=nullptr; u64 ring_bytes_=0,slot_bytes_=0; bool pinned_=false;
  u32 slots_=4,qd_=16,slot_blocks_=SLOT_BLOCKS,fill_u_=0,fill_k_=0;
  HANDLE port_=nullptr; std::vector<HANDLE> th_; DWORD aff_=0;
  std::atomic<u64> issued_u_{0},released_u_{0},outstanding_{0},fatal_{0};
};
} // namespace insignia
```

```cpp
// src/nvme_reader.cpp — the whole engine of the NVMe tier (~210 lines, MSVC/C++20)
#include "insignia_nvme.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
namespace insignia {
static void die(const char*w){ throw std::runtime_error(std::string("NvReader: ")+w); }
static void wchk(BOOL ok,const char*w){ if(!ok) die(w); }

void NvReader::build(const std::vector<NvFileSpec>&specs){
  for(u32 f=0;f<specs.size();++f){
    auto iv=specs[f].intervals; std::sort(iv.begin(),iv.end());
    for(size_t i=1;i<iv.size();++i) if(iv[i].first<iv[i-1].second) die("overlapping intervals");
    // this checkpoint: every keep-set collapses to ONE gapless prefix [0,end) (layers/mtp whole-file;
    // outside = text prefix, vision tail excluded) -> map(abs)=base+abs stays identity. Assert it.
    if(!(iv.size()==1&&iv[0].first==0)) die("plan not a single prefix range (general mapping TODO)");
    u64 len=iv[0].second; if(len>u64(slot_blocks_)*BLK) die("unit exceeds slot (mtp/outside -> read_once)");
    Unit u{}; u.file=f; u.first=u32(blocks_.size()); u.nblk=0; u.tail_len=0;
    u64 off=0;
    while(off<len){ u64 n=std::min(BLK,len-off),d=n&~511ull;   // direct: 512-floor; tail: <512 via twin
      if(d){ blocks_.push_back({off,u32(d)}); ++u.nblk; }
      if(n-d){ u.tail_off=off+d; u.tail_len=u32(n-d); }        // rides with the LAST block's unit (below)
      off+=n; }
    u.done=CreateEventW(nullptr,TRUE,FALSE,nullptr); if(!u.done) die("CreateEvent");
    u.left.store(0,std::memory_order_relaxed); units_.push_back(u);
    // blocks_ tail entry owns the sub-sector remainder; per census: linear 104/128B, full 144/160B, mtp 304B, outside 192B
  }
}

void NvReader::init(const std::vector<NvFileSpec>&specs,u32 slots,u32 qd,u32 nthreads,DWORD aff){
  slots_=std::max<u32>(slots,2); qd_=std::max<u32>(qd,8); aff_=aff;              // depth=slots-1 units ahead
  build(specs);
  STORAGE_PROPERTY_QUERY q{}; q.PropertyId=StorageAccessAlignmentProperty; q.QueryType=PropertyStandardQuery;
  for(u32 f=0;f<specs.size();++f){
    File F{}; F.direct=CreateFileW(specs[f].path,GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,
      nullptr,OPEN_EXISTING,FILE_FLAG_NO_BUFFERING|FILE_FLAG_OVERLAPPED|FILE_FLAG_SEQUENTIAL_SCAN,nullptr);
    if(F.direct==INVALID_HANDLE_VALUE) die("open direct");                        // SEQUENTIAL_SCAN is inert under
    F.buffered=CreateFileW(specs[f].path,GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,   // NO_BUFFERING (cache
      nullptr,OPEN_EXISTING,FILE_FLAG_OVERLAPPED,nullptr);                        // bypassed); kept, costs nothing
    if(F.buffered==INVALID_HANDLE_VALUE) die("open twin");
    LARGE_INTEGER sz{}; if(!GetFileSizeEx(F.direct,&sz)) die("size"); F.size=u64(sz.QuadPart);
    STORAGE_ACCESS_ALIGNMENT_DESCRIPTOR a{}; DWORD br=0;                          // 512e vs 4Kn: 4096-multiple blocks
    if(DeviceIoControl(F.direct,IOCTL_STORAGE_QUERY_PROPERTY,&q,sizeof q,&a,sizeof a,&br,nullptr) && a.BytesPerLogicalSector)
      if(4096%a.BytesPerLogicalSector) die("sector size not power-divisor of 4096");
    files_.push_back(F);
  }
  port_=CreateIoCompletionPort(INVALID_HANDLE_VALUE,nullptr,0,0); if(!port_) die("IOCP create");
  for(u32 f=0;f<files_.size();++f)
    if(CreateIoCompletionPort(files_[f].direct,port_,f+1,0)!=port_) die("IOCP associate"); // key=f+1 (0 reserved)
  slot_bytes_=u64(slot_blocks_)*BLK; ring_bytes_=u64(slots_)*slot_bytes_;
  ring_=reinterpret_cast<u8*>(VirtualAlloc(nullptr,SIZE_T(ring_bytes_),MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE));
  if(!ring_) die("VirtualAlloc ring");
  pinned_=cudaHostRegister(ring_,SIZE_T(ring_bytes_),cudaHostRegisterDefault)==cudaSuccess; // locked: standby can't touch it
  if(!pinned_) std::fprintf(stderr,"[nvme] cudaHostRegister failed -> unpinned ring (PCIe staging bounce, CPU GEMV unaffected)\n");
  reqs_.assign(size_t(slots_)*slot_blocks_,Req{});                                // 1 Req per ring block: unique writer
  for(u32 i=0;i<nthreads;++i){ HANDLE t=CreateThread(nullptr,0,tramp,this,0,nullptr); if(!t) die("thread");
    if(aff_) SetThreadAffinityMask(t,aff_);                                       // park on SMT siblings (LP 6-11)
    SetThreadPriority(t,THREAD_PRIORITY_ABOVE_NORMAL);                            // not HIGHEST/TIME_CRITICAL: DPC safety
    th_.push_back(t); }
}

void NvReader::read_once(u32 file,u64 off,u64 len,void*dst){                      // startup: mtp(477MB)/outside text(5.09GB)
  u8*p=static_cast<u8*>(dst);                                                    // buffered twin: cache reuse is free here
  while(len){ u64 n=std::min<u64>(len,8ull<<20); OVERLAPPED ov{}; ov.Offset=u32(off); ov.OffsetHigh=u32(off>>32); DWORD got=0;
    if(!ReadFile(files_[file].buffered,p,DWORD(n),&got,&ov)||got!=n) die("read_once"); p+=got; off+=got; len-=got; }
}

void NvReader::issue(u32 unit,u32 blk){                                          // blk = global block index
  Unit&u=units_[unit]; Blk&b=blocks_[blk]; u32 k=blk-u.first;                    // k = block-in-slot
  Req&r=reqs_[u.slot*slot_blocks_+k]; r.ov=OVERLAPPED{}; r.ov.Offset=u32(b.off); r.ov.OffsetHigh=u32(b.off>>32);
  r.unit=unit; r.blk=blk; r.tries=0;
  if(!ReadFile(files_[u.file].direct,slot_base(u.slot)+u64(k)*BLK,b.len,nullptr,&r.ov)
     &&GetLastError()!=ERROR_IO_PENDING) die("ReadFile issue");                   // pending is the normal path
  outstanding_.fetch_add(1,std::memory_order_acq_rel);
}

void NvReader::top_up(){                                                          // worker-thread only
  while(outstanding_.load(std::memory_order_acquire)<qd_){
    if(fill_u_>=units_.size()) return;                                            // epoch fully issued; epoch_begin() rewinds
    Unit&u=units_[fill_u_];
    if(u.left.load(std::memory_order_relaxed)==0){                                // unit start: claim slot (cyclic)
      if(issued_u_.load(std::memory_order_acquire)-released_u_.load(std::memory_order_acquire)>=slots_-1) return; // depth guard
      u.slot=fill_u_%slots_; ResetEvent(u.done); u.left.store(u.nblk,std::memory_order_relaxed);
      issued_u_.fetch_add(1,std::memory_order_release); fill_k_=0;
      if(u.tail_len){ OVERLAPPED ov{}; ov.Offset=u32(u.tail_off); ov.OffsetHigh=u32(u.tail_off>>32); DWORD got=0; // <512B EOF tail,
        if(!ReadFile(files_[u.file].buffered,slot_base(u.slot)+u64(u.nblk-1)*BLK+(blocks_[u.first+u.nblk-1].len), // buffered twin, sync,
                     u.tail_len,&got,&ov)||got!=u.tail_len) die("tail"); } }      // <=1/file/epoch, page-cached after epoch 1
    if(fill_k_<u.nblk){ issue(fill_u_,u.first+fill_k_); ++fill_k_; }              // blocks strictly in order (sequential!)
    else ++fill_u_;                                                               // unit fully issued -> next unit
  }
}

void NvReader::complete(OVERLAPPED_ENTRY&e){
  Req&r=*reinterpret_cast<Req*>(e.lpOverlapped); Unit&u=units_[r.unit]; DWORD got=0;
  BOOL ok=GetOverlappedResult(files_[u.file].direct,&r.ov,&got,FALSE);           // GQCSEx hides per-entry errors -> extract
  if(!ok){ DWORD err=GetLastError();
    if(err==ERROR_OPERATION_ABORTED){ outstanding_.fetch_sub(1); return; }        // CancelIoEx teardown: drop silently
    if(err!=ERROR_HANDLE_EOF&&++r.tries<3){                                      // bounded retry, same ring block
      u64 off=blocks_[r.blk].off; r.ov.Internal=0; r.ov.InternalHigh=0; r.ov.Offset=u32(off); r.ov.OffsetHigh=u32(off>>32);
      if(ReadFile(files_[u.file].direct,slot_base(u.slot)+u64(r.blk-u.first)*BLK,blocks_[r.blk].len,nullptr,&r.ov)
         ||GetLastError()==ERROR_IO_PENDING) return; }                            // outstanding_ unchanged on reissue
    fatal_.store(1,std::memory_order_release); for(auto&x:units_) SetEvent(x.done); // wake every consumer -> nullptr
  }
  if(u.left.fetch_sub(1,std::memory_order_acq_rel)==1) SetEvent(u.done);          // last block (tail already resident)
  outstanding_.fetch_sub(1,std::memory_order_acq_rel);
}

void NvReader::worker(){
  for(;;){ OVERLAPPED_ENTRY es[16]; ULONG n=0;
    if(!GetQueuedCompletionStatusEx(port_,es,16,&n,1000,FALSE)&&GetLastError()!=WAIT_TIMEOUT) continue;
    for(ULONG i=0;i<n;++i) complete(es[i]);
    if(fatal_.load(std::memory_order_acquire)) return;
    top_up();                                                                    // self-arming: keep QD saturated
    if(n==0&&outstanding_.load(std::memory_order_acquire)==0&&fill_u_>=units_.size()){ // idle+drained: check stop
      if(fatal_.load(std::memory_order_acquire)) return; }                        // (real stop handled by shutdown drain)
  }
}

u8* NvReader::unit_wait(u32 unit){ WaitForSingleObject(units_[unit].done,INFINITE); // copy thread AND gemv threads may wait
  return healthy()?slot_base(units_[unit].slot):nullptr; }
void NvReader::unit_release(u32 unit){ released_u_.fetch_add(1,std::memory_order_release); } // frees slot (issued-released<slots)
void NvReader::epoch_begin(){ if(released_u_.load()!=units_.size()) die("epoch re-arm with units in flight"); fill_u_=0; fill_k_=0; }

void NvReader::shutdown(){
  fatal_.store(1,std::memory_order_release);
  for(auto&F:files_) if(F.direct) CancelIoEx(F.direct,nullptr);                   // abort in-flight -> ABORTED completions
  for(HANDLE t:th_) if(WaitForSingleObject(t,10000)==WAIT_TIMEOUT) TerminateThread(t,0);  // drain first; terminate = last resort
  for(HANDLE t:th_) CloseHandle(t); th_.clear();
  for(auto&u:units_) if(u.done) CloseHandle(u.done); units_.clear();
  for(auto&F:files_){ if(F.direct) CloseHandle(F.direct); if(F.buffered) CloseHandle(F.buffered); } files_.clear();
  if(port_){ CloseHandle(port_); port_=nullptr; }
  if(ring_){ if(pinned_) cudaHostUnregister(ring_); VirtualFree(ring_,0,MEM_RELEASE); ring_=nullptr; }
}
} // namespace insignia
```

Consumer sketch (decode loop, tier-aware):

```cpp
// per token: for(l=0..63){ tier t=plan_tier(l);
//   if(t==nvme){ const u8*slot=reader.unit_wait(idx);            // waits ≤ (slots-1) layers of runout
//     cudaMemcpyAsync(vram_scratch[l], slot+data_start, layer_bytes, H2D, copy_stream);  // 383.87MB ≈ 15ms PCIe4
//     cudaStreamSynchronize(copy_stream); run_layer_gpu();        // -- OR --
//     cpu_gemv_layer(slot+data_start);                            // read pinned ring in place (~40GB/s DRAM)
//     reader.unit_release(idx); }                                 // slot recycles AFTER consumers done
//   else run resident path; }  reader.epoch_begin();              // re-arm for next token sweep
```

### 3.1 Ring size: 4 slots × 368 MiB = 1.44 GiB default (5 = 1.80 GiB upper)

RAM budget arithmetic (15.9 GiB physical, measured-class estimates):

| consumer | GiB |
|---|---|
| Win11 + desktop baseline (measured-class) | ~3.0 |
| CUDA context + WDDM overhead | ~0.6 |
| 980 HMB (FTL borrow) | ~0.06 |
| ring 4×368 MiB (or 5×) | **1.44 (1.80)** |
| engine C-tier RAM layers M×0.3673 avg (M=23 → 8.4; M=24 → 8.8) | 8.4 |
| KV (16 full-attn, short ctx) + 48×3.15 MB DeltaNet state + activations + lm_head margin | ~0.6 |
| **total** | **~14.1 (14.5)** |

M=23 (synthesis optimum) fits with 4 slots and ~1.4 GiB slack for browser/OS spikes; a 5th
slot (1.80 GiB) only fits if M drops to ~22 — a bad trade (a RAM layer is worth 380 MB/token
of NVMe traffic; a slot is worth only jitter absorption). **Prefetch depth justification**:
at Gen3 the disk produces a layer every ~116 ms while the consumer eats one every ≤58 ms
(PCIe 15 ms + GPU 0.005 ms, or CPU GEMV ~10 ms) — the disk is the pacer by ≥2×, so
runout beyond "1 consuming + 1 filled + 1 filling + 1 slack" (slots−1 = 3 ahead) buys
nothing: bandwidth is conserved, the ring cannot fill faster than the disk streams.
In-flight to saturate: 16×2 MiB = 32 MiB — one slot alone offers 184 outstanding-capable
blocks; QD16 is comfortable (colibri measured 86 MB/s QD1 → 696 MB/s QD8 on a weaker drive;
expect full rate at QD8–16 here — §6 verifies).

---

## 4. Scheduling

- **Decode consumption is strictly sequential layer 0→63**, and resident-VRAM (L) and
  RAM (M) layers are *skipped by the reader entirely* — their bytes never touch the ring.
  The per-epoch plan is just the N-tier shards in layer order; `top_up` walks it, so the
  scheduler *is* the plan cursor. No heuristics, no MoE-style prediction (dense model —
  colibri's PILOT machinery is deliberately absent; that's MoE-specific).
- **Per-token NVMe traffic = N×~380 MB and it dominates everything**:
  N=21 (≈15.75 linear + 5.25 full) = 8.0 GB/token →
  **2.42 s/token on E: (Gen3 3.3 GB/s)** → 0.41 tok/s; with MTP ×1.6 ≈ **0.66 tok/s**.
  (Synthesis's 56.5 ms/layer / 0.66 tok/s assumed 6.8 GB/s Gen4 — true only if the model
  lives on the 980 PRO; on E: the same config yields the *same* 0.66 tok/s figure only
  via MTP acceptance, at the cost of the drive running flat-out continuously.)
  Split E:+C: (9.7 GB/s agg): 0.83 s/token → ~1.3 tok/s with MTP.
- **Prefetch discipline**: keep ≥8×2 MiB in flight per active file (QD16 default, 32 MiB)
  and slots−1 = 3 units (≈1.1 GB) of read-ahead — 2–3 layers ahead as mission requires.
  Since only one file is open-streaming at a time (sequential sweep), per-file QD = global
  QD. Re-arm per epoch via `epoch_begin()` (asserts all units released — strict sequential
  consumption guarantees this on the decode path).
- **MTP interplay**: draft layer runs after the main sweep; if VRAM-resident (477 MB — it
  fits and synthesis says draft = 4 ms warm) it adds zero NVMe. If budget forces it to
  N-tier, its 228-block unit rides the plan between layer 63 and epoch end (needs slot
  bump to 228 blocks — flag, not default).
- **Prefill sweep**: weight-stationary (FlexGen-style, per synthesis) — the *same* plan
  streams once per prompt in layer order with activation checkpoints: each N-layer is read
  ONCE per prompt, not per token (8 GB/prompt = 2.4 s on E: + compute). Chunk-major
  re-streaming (25.6 GB per 64-token chunk) is the anti-pattern the plan structure makes
  impossible to regress into accidentally — the plan is a list of layer units, period.
- **Epoch 1 vs steady state**: the very first sweep is cold (full 2.42 s/token); if the
  standby list happens to hold shards (as it does right now — §1 probe), buffered twins
  would absorb it, but the ring path is NO_BUFFERING by default — cold is cold, by design
  (determinism; §5).

---

## 5. Endurance + caching (quantified, with the corrected model)

### 5.1 The endurance math (the feasibility-killer, stated three ways)

Constants: E: 980 1TB = 600 TBW; C: 980 PRO 500GB = 300 TBW; N=21 → 8.0 GB/token
(N=17 realistic minimum with L=21, M=26 → 6.5 GB/token).

| scenario | GB/token | sustained read rate (while generating) | TB/h on source | TBW-equivalent life | tokens per TBW-life |
|---|---|---|---|---|---|
| N=21, E: only | 8.0 | 3.3 GB/s (drive-bound, ~100% duty) | **11.9** | **~50 h** | **75,000** |
| N=21, mission's 100 tok/min hypothetical | 8.0 | 13.3 GB/s (unreachable) | 48 | ~12.5 h | 75,000 |
| N=17, E: only | 6.5 | 3.3 GB/s | 11.9 | ~50 h | 92,000 |
| N=21, split E:+C: | 8.0 | 3.3 + 4.7 | 5.9 each | E: ~100 h, C: ~51 h (300 TBW) | 75,000 |

Cleanest statement: **the drive streams continuously while generating, so its life in
hours is fixed by bandwidth (~50 h on E:), and total tokens = TBW ÷ GB/token.** Note a
correction to the mission's arithmetic: "8 GB/token at 100 tok/min" is 800 GB/min =
**48 TB/h** (→ 12.5 h), not 2.9 TB/h; the mission's 2.88 TB/h ↔ 200 h figures are
self-consistent only for ~0.48 GB/token (≈6 tok/min at 8 GB/token). Either way the
realistic regime is the first row: the drive saturates at 11.9 TB/h and lasts ~50 h of
continuous generation, delivering its ~75k tokens.

**Honest caveat that must travel with these numbers**: TBW is a *write* endurance rating.
Pure reads wear NAND via read-disturb (charge injection in unread cells → ECC pressure →
vendor-internal scrub/rewrite), which the controller absorbs passively; public data on
read-wear per TB read is scarce and it is *far* below write wear. So 75k tokens is an
upper-bound framing, not a countdown. Ground truth is measurable: log NVMe SMART
`Data Units Read` (0x02) and `Percentage Used` (0x05) per session (via
`Get-PhysicalDisk | Get-StorageReliabilityCounter` or smartctl) — the delta per TB read
on *this* drive is a one-evening experiment (§6.3) and should gate how hard we lean on
the N tier.

Mitigations, ranked:

1. **Minimize N — RAM is the cache.** Every layer moved N→M removes 380 MB/token =
   1.37 GB/min at 0.42 tok/s = 82 GB/h of flash wear. M is engine-pinned (deterministic),
   bounded only by the RAM budget table in §3.1. The endurance math *and* the latency math
   both point at the same lever: maximize M before anything else.
2. **Split/mirror the shards across E: + C:** (93.8 GB free on the Gen4 PRO): halves
   per-drive traffic, ~2.3× aggregate read rate, and colibri's mirror contract
   (size+header byte-identical; deterministic routing so hints and demand reads meet on
   the same replica — audits/w2/colibri-io.md §2.5, §8, §9) is the proven template.
   Cost: C: is the system drive (background writes contend; 300 TBW is lower) and the
   500GB PRO fills to ~124 GB used — fine at 93.8 GB free.
3. **Do nothing clever with the OS cache** — see 5.2; it does not work here.
4. Requantize below FP8 (INSIG4/NanoQuant direction) — shrinks *all* tiers including N's
   GB/token linearly; a separate workstream (synthesis backlog), noted for completeness.

### 5.2 The buffered-read "free cache" — modeled properly (mission's 50% is wrong)

Mission's model: cyclic access, cache C, working set W, hit = C/W = 13/25.65 ≈ 50%.
Three corrections:

1. **W is not 25.65 GB.** Only the N-tier shards are read through the cache (L is VRAM,
   M is engine-pinned RAM — not standby pages). W = N×~380 MB = 8.0 GB at N=21. The other
   43 layers never touch the file cache at all.
2. **LRU on a cyclic pattern with W > C yields 0% hits, not C/W.** Classic result — the
   RRIP paper spells it out: "a cyclic access pattern of length k that repeats N times.
   LRU receives zero cache hits due to thrashing" ([Jaleel et al., ISCA'10,
   §Re-Reference Interval Prediction](https://dl.acm.org/doi/pdf/10.1145/1815961.1815971);
   see also [Qureshi's DIP paper](https://safari.ethz.ch/architecture/fall2018/lib/exe/fetch.php?media=p381-qureshi.pdf)).
   C/W is the *MRU/random* bound; LRU evicts exactly the block needed soonest. Windows'
   standby list is aging-based approximate LRU — closer to the 0% regime than to 50% for
   a pure sweep.
3. **The standby list can never hold the remainder anyway — structural, not tunable.**
   Free standby after the engine's own allocations ≈ 15.9 − (3.0 OS + 0.6 CUDA + ring
   1.44 + 0.6 misc) − M×0.367 GiB. Cache-absorbing the N sweep needs
   C_free > W = (64−L−M)×0.367 ⇒ 10.26 > 0.367×(64−L−M) ⇒ L+M < 36.1 — but then
   N = 28+ and you've *un-pinned* 15 layers to make the OS hold them unpinned instead:
   strictly worse (same RAM, zero determinism, plus eviction risk from any background
   process). With the synthesis-optimal L+M = 44, C_free ≈ 1.3 GiB << W = 8.0 GB: deep in
   the thrash regime. In that regime buffered reads are **worse than nothing**: every page
   is filled, never hit, evicted — you pay the memcpy into the ring slot, pollute the
   standby list (evicting whatever else deserved it), and gain 0%.

**Recommendation**: default `FILE_FLAG_NO_BUFFERING` + maximize engine-owned RAM C-tier
(deterministic; pinned ring is immune to standby pressure — locked pages live outside the
standby lists). Keep the buffered twin handles for: (a) the <512 B EOF tails, (b) one-shot
startup reads (`read_once`: mtp + outside text 5.09 GB — here cache reuse across runs is
genuinely free and nothing is cyclic), (c) the §6 A/B bench, which will empirically confirm
the ~0% steady-state hit rate and quantify the buffered memcpy overhead. If a future config
ever runs small-N (say N≤6, W ≈ 2.3 GB) *and* leaves >W of standby free, flipping the ring
fills to the buffered handles is a two-line change worth benching — the reader keeps both
handles open per file precisely for that.

---

## 6. Micro-bench spec + harness (tools/io_bench.cpp sketch)

Purpose: (1) validate ≥6 GB/s expectation — on **C:** (PRO); E: should show ~3.2–3.4 GB/s,
which *is* the acceptance number for the real model dir; (2) QD/block-size sweep;
(3) buffered-vs-direct cyclic A/B to close §5.2 empirically; (4) tail-path correctness.
Acceptance for the engine: sequential 2 MiB QD16 NO_BUFFERING over a 20 GB span —
**E: ≥ 3.0 GB/s, C: ≥ 6.0 GB/s** (if E: measures below ~2.8, the N-tier time budget for
scheduling §4 must be recomputed before any tuning).

```cpp
// tools/io_bench.cpp — sequential QDn sweep, direct vs buffered, cyclic thrash probe
// usage: io_bench.exe <file> <span_GB> <blk_kb> <qd> <d|b> <cycles>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
static double now(){ LARGE_INTEGER c,f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f); return double(c.QuadPart)/f.QuadPart; }
int main(int argc,char**argv){
  const char*path=argv[1]; u64 span=u64(atof(argv[2]))<<30; u32 blk=u32(atoi(argv[3]))<<10,
        qd=u32(atoi(argv[4])); BOOL direct=argv[5][0]=='d'; int cycles=atoi(argv[6]);
  HANDLE h=CreateFileA(path,GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,
    FILE_FLAG_OVERLAPPED|(direct?FILE_FLAG_NO_BUFFERING:0),nullptr);
  LARGE_INTEGER fs{}; GetFileSizeEx(h,&fs); if(span>u64(fs.QuadPart)) span=fs.QuadPart;
  u8*buf=(u8*)VirtualAlloc(nullptr,SIZE_T(qd)*blk,MEM_COMMIT,PAGE_READONLY); // page-aligned: NO_BUFFERING-legal
  HANDLE port=CreateIoCompletionPort(INVALID_HANDLE_VALUE,nullptr,0,0); CreateIoCompletionPort(h,port,1,0);
  std::vector<OVERLAPPED> ov(qd); std::vector<u64>   off(qd);
  for(int c=0;c<cycles;++c){ u64 issued=0,reaped=0,infl=0; double t0=now();
    while(reaped<span/blk){
      while(infl<qd&&issued<span/blk){ u32 i=u32(issued%qd); ov[i]={}; ov[i].Offset=DWORD(issued*blk);
        ov[i].OffsetHigh=DWORD((issued*blk)>>32); off[i]=issued*blk;                 // sequential block order
        if(!ReadFile(h,buf+SIZE_T(i)*blk,blk,nullptr,&ov[i])&&GetLastError()!=ERROR_IO_PENDING){ printf("read err\n"); return 1; }
        ++issued; ++infl; }
      OVERLAPPED_ENTRY es[32]; ULONG n=0; GetQueuedCompletionStatusEx(port,es,32,&n,INFINITE,FALSE);
      reaped+=n; infl-=n; }
    printf("cycle %d: %.2f GB/s (%s, blk=%uK qd=%u)\n",c,span/now()-0+span/(now()-t0),
           direct?"direct":"buffered",blk>>10,qd); // report span/(elapsed) properly in real code
  }
  return 0; }
```

(Sketch: the timing print's elapsed must be `now()-t0` computed once — kept honest in the
real implementation; the structure — IOCP + qd-deep sequential issue + reap — is the part
that matters and mirrors the reader.) Bench matrix:

| # | test | expectation |
|---|---|---|
| 1 | layers-0..53 span ≈ 20 GB, 2 MiB, QD sweep 1/2/4/8/16/32, direct | E: saturates ≈ QD8: ~3.2–3.4 GB/s; QD1 ≈ 0.4–0.8 |
| 2 | block sweep 256K/1M/2M/4M at best QD, direct | 2 MiB within ~5% of 4 MiB; 256K drops (per-IOP overhead) |
| 3 | same span, buffered, 5 cycles (W=20 GB > C≈13 GB) | cycle 1 ≈ direct; cycles 2+ ≈ direct (thrash), NOT faster — §5.2 |
| 4 | small span 2 GB, buffered, 5 cycles (W < C) | cycles 2+ ≈ RAM speed — the only regime where buffering wins |
| 5 | tail path: read [span−86,656, +86,528) direct + 128 B twin | byte-exact vs plain buffered read (CRC vs crc32.txt) |
| 6 | (opt-in, hands-on) run generation for 1 h, read SMART Data Units Read + Percentage Used before/after | wear-per-TB-read ground truth for §5.1 caveat |

---

## 7. Windows specifics (practical)

- **Sector alignment**: query once per handle via
  `IOCTL_STORAGE_QUERY_PROPERTY` → `StorageAccessAlignmentProperty` →
  `STORAGE_ACCESS_ALIGNMENT_DESCRIPTOR.BytesPerLogicalSector`
  ([docs](https://learn.microsoft.com/en-us/windows/win32/fileio/file-buffering) — the
  canonical alignment contract: offset, length, and buffer must be sector-multiples).
  Both E: and C: are 512e (logical 512); our 2 MiB blocks from offset 0 into a
  page-aligned ring satisfy 512e and 4Kn simultaneously, so the query is a guard, not a
  branch. `GetFileSizeEx` for sizes (CRT `lseek` fails on NO_BUFFERING handles — colibri
  compat.h:313 measured).
- **FILE_FLAG_SEQUENTIAL_SCAN with NO_BUFFERING**: the hint tunes cache-manager readahead
  for *buffered* access; under NO_BUFFERING the cache manager is bypassed, so it is inert.
  Kept (harmless, documents intent); do not expect it to do anything
  ([CreateFile docs](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilea),
  [File Buffering](https://learn.microsoft.com/en-us/windows/win32/fileio/file-buffering)).
- **IoRing — skip rationale**: available on this build (26200 ≥ 22621) but v1/v2 is a
  minimal op set (read/write/flush/cancel) with mandatory buffer registration
  ([ioringapi](https://learn.microsoft.com/en-us/windows/win32/api/ioringapi/),
  [windows-internals IoRing deep-dives](https://windows-internals.com/i-o-rings-when-one-i-o-operation-is-not-enough/),
  [what changed](https://windows-internals.com/one-year-to-i-o-ring-what-changed/)).
  Our shape — 2 threads, one port, 16-deep large sequential reads, completion batching via
  GQCSEx — is already submission-cheap (16 `ReadFile`/ms at full rate); the synthesis's
  colibri figure was IoRing ≈ +2% best case. Not worth a second I/O stack while the
  bottleneck is the Gen3 drive itself. Revisit trigger: if profiling ever shows
  >2% CPU in ReadFile issue path.
- **Queue depth / storport**: NVMe on Win11 goes through stornvme (not storahci — that's
  the SATA HDD); app-side outstanding reads map ~1:1 to device commands; QD16×2 MiB is the
  saturation point per §6 test 1. MSI is enabled by default for stornvme on this platform;
  nothing to configure, nothing to tune from userland — kept out of the code.
- **Thread affinity**: 5600X LP0-5 = physical cores 0-5 primaries, LP6-11 = SMT siblings.
  Reader threads masked to `0xFC0` (LP 6-11) so GEMV threads own the primaries (0x3F) and
  each core still services I/O completions on its sibling — DPCs for stornvme land on
  whatever core issued, which now never collides with a full GEMV primary. Priority
  `THREAD_PRIORITY_ABOVE_NORMAL`: enough to preempt normal-priority noise, shy of
  HIGHEST/TIME_CRITICAL which can starve DPC/ISR work and the storage stack's own kernel
  threads — the exact anti-pattern on an I/O-bound box.
- **Pinned ring vs standby**: `cudaHostRegister`'d pages are locked (outside the standby
  lists) — cache pollution and eviction pressure cannot touch the ring, and the ring
  cannot evict anyone else's cache. The two memory pools coexist by construction
  (colibri's `pin_wire` discipline, compat.h:195).
- **Error paths recap** (in code, §3): `ERROR_IO_PENDING` = normal; EOF-tolerant tail;
  3× bounded retry on device errors; `ERROR_OPERATION_ABORTED` accepted silently during
  `CancelIoEx` teardown; fatal flag + wake-all on hard failure. One deliberate omission:
  no per-read timeout — NVMe resets are the controller's/driver's job; a wedged drive
  surfaces as a fatal completion, not a hang we can fix in userland.

---

## 8. Sources

- Census/alignment: `audits/w2/loader-27b-spec.md` (§0, §2, §4) — verified this session
  for outside.safetensors by direct header parse (§2.2 above).
- colibri I/O machinery: `audits/w2/colibri-io.md` (twin handles §3.1, tail-pread pattern
  §8.3, QD scaling measurements §8.5, mirrors/split §2.5/§9, Windows posture §10).
- Feasibility tiers + MTP math: `audits/synthesis.md`.
- [File Buffering (MS Learn)](https://learn.microsoft.com/en-us/windows/win32/fileio/file-buffering)
- [I/O Completion Ports (MS Learn)](https://learn.microsoft.com/en-us/windows/win32/fileio/i-o-completion-ports)
- [CreateIoCompletionPort](https://learn.microsoft.com/en-us/windows/win32/fileio/createiocompletionport),
  [GetQueuedCompletionStatusEx](https://learn.microsoft.com/en-us/windows/win32/fileio/getqueuedcompletionstatusex-func),
  [ReadFile](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-readfile)
- [ioringapi (MS Learn)](https://learn.microsoft.com/en-us/windows/win32/api/ioringapi/),
  [Windows Internals: IoRing](https://windows-internals.com/i-o-rings-when-one-i-o-operation-is-not-enough/),
  [One Year to IoRing](https://windows-internals.com/one-year-to-i-o-ring-what-changed/)
- [Jaleel et al., RRIP (ACm DL)](https://dl.acm.org/doi/pdf/10.1145/1815961.1815971) —
  LRU cyclic thrashing = zero hits; [Qureshi DIP (ETH)](https://safari.ethz.ch/architecture/fall2018/lib/exe/fetch.php?media=p381-qureshi.pdf)
- Drives: [TechPowerUp 980 1TB](https://www.techpowerup.com/ssd-specs/samsung-980-1-tb.d58),
  [Samsung 980 page](https://www.samsung.com/au/memory-storage/nvme-ssd/980-1tb-nvme-pcie-gen-3-mz-v8v1t0bw/),
  [TechPowerUp 980 PRO 500GB](https://www.techpowerup.com/ssd-specs/samsung-980-pro-500-gb.d46),
  [Samsung 980 PRO announcement](https://news.samsung.com/us/memory-ssd-980-pro-gaming-pc)

*Read-only audit; nothing outside this file was modified.*
