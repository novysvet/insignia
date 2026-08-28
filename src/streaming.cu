// =============================================================================
// src/streaming.cu — NVMe -> pinned-RAM streaming layer (w3). Host-only code
// (compiled as .cu so it links with the CUDA runtime for cudaHostRegister).
//
// NvmeReader   : IOCP + GetQueuedCompletionStatusEx workers, NO_BUFFERING direct
//                handles + buffered twin per file, 2 MiB blocks, QD16, FIFO
//                stream order, bounded retry, CancelIoEx teardown.
// PinnedRing   : VirtualAlloc + cudaHostRegister slot ring, atomic slot states.
// LayerFeeder  : decode-epoch feeder over the two (read-ahead = slots-1).
//
// Designs: audits/w3/nvme-reader.md §3 (reader), audits/w3/loader-gaps.md §3.3
// (TieredStorage v2 contract), audits/w3/pcie-pipeline.md §7.2 (LayerFeeder),
// audits/w3/colibri-sched-deep.md §8 (cyclic schedule, sequential discipline).
// =============================================================================
#include "insignia_streaming.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <type_traits>
#include <utility>

namespace insignia {

static void die(const char* w) { throw std::runtime_error(std::string("streaming: ") + w); }

// =============================================================================
// NvmeReader
// =============================================================================
static constexpr DWORD kReaderAffinity = 0xFC0;   // LP 6-11 = SMT siblings of the 5600X

NvmeReader::NvmeReader(u32 num_threads) {
    // IOCP hands us &Req.ov; ov must be the first member so the container_of
    // cast in worker_loop is exact.
    static_assert(std::is_standard_layout<Req>::value && offsetof(Req, ov) == 0, "OVERLAPPED must lead Req");
    if (num_threads == 0) num_threads = 2;
    port_ = CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 0);
    if (!port_) die("CreateIoCompletionPort");
    for (u32 i = 0; i < num_threads; ++i) {
        HANDLE t = CreateThread(nullptr, 0, &NvmeReader::worker_tramp, this, 0, nullptr);
        if (!t) { shutdown(); die("CreateThread"); }
        // Park readers on the SMT siblings so the GEMV team owns the physical
        // primaries (LP 0-5); stornvme DPCs then land off the compute cores.
        // audits/w3/nvme-reader.md §7. Failure (fewer LPs) is non-fatal.
        SetThreadAffinityMask(t, kReaderAffinity);
        SetThreadPriority(t, THREAD_PRIORITY_ABOVE_NORMAL);   // not HIGHEST/TIME_CRITICAL: DPC safety
        threads_.push_back(t);
    }
}

DWORD WINAPI NvmeReader::worker_tramp(LPVOID p) {
    static_cast<NvmeReader*>(p)->worker_loop();
    return 0;
}

void NvmeReader::shutdown() noexcept {
    bool expect = false;
    if (!stop_.compare_exchange_strong(expect, true, std::memory_order_acq_rel)) return;  // idempotent
    for (auto& f : files_) {                              // abort in-flight -> ABORTED completions
        if (f.direct) CancelIoEx(f.direct, nullptr);
        if (f.twin)   CancelIoEx(f.twin, nullptr);
    }
    for (size_t i = 0; i < threads_.size(); ++i)
        if (port_) PostQueuedCompletionStatus(port_, 0, 0, nullptr);   // wake parked workers NOW
    for (HANDLE t : threads_) {
        if (WaitForSingleObject(t, 5000) == WAIT_TIMEOUT) TerminateThread(t, 0);  // last resort
        CloseHandle(t);
    }
    threads_.clear();
    {
        std::lock_guard<std::mutex> lk(mtx_);
        for (auto& up : units_)
            if (up) { up->ok = false; up->done.store(true, std::memory_order_release); SetEvent(up->event); }
        for (auto& f : files_) {
            if (f.direct) CloseHandle(f.direct);
            if (f.twin)   CloseHandle(f.twin);
        }
        files_.clear(); paths_.clear(); order_.clear(); free_.clear();
        for (auto& up : units_)
            if (up) { CloseHandle(up->event); up.reset(); }
        units_.clear();
    }
    if (port_) { CloseHandle(port_); port_ = nullptr; }
    fatal_.store(true, std::memory_order_release);
}

u32 NvmeReader::file_index_locked(const wchar_t* path) {
    std::wstring key(path);
    for (size_t i = 0; i < paths_.size(); ++i)
        if (paths_[i] == key) return u32(i);
    // Open on demand: direct (NO_BUFFERING|OVERLAPPED, cache bypassed — cyclic
    // sweeps > standby list thrash, audits/w3/nvme-reader.md §5.2) + buffered
    // twin for sub-sector EOF tails and one-shots. IOCP key = file index+1.
    File f{};
    f.direct = CreateFileW(path, GENERIC_READ,
                           FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                           OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
    if (f.direct == INVALID_HANDLE_VALUE) die("open direct handle");
    f.twin = CreateFileW(path, GENERIC_READ,
                         FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                         OPEN_EXISTING, 0, nullptr);
    if (f.twin == INVALID_HANDLE_VALUE) { CloseHandle(f.direct); die("open twin handle"); }
    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(f.direct, &sz)) { CloseHandle(f.direct); CloseHandle(f.twin); die("GetFileSizeEx"); }
    f.size = u64(sz.QuadPart);
    if (CreateIoCompletionPort(f.direct, port_, DWORD(files_.size() + 1), 0) != port_)
        { CloseHandle(f.direct); CloseHandle(f.twin); die("IOCP associate"); }
    paths_.emplace_back(std::move(key));
    files_.push_back(f);
    return u32(files_.size() - 1);
}

// Build the block table for one plan into u: each request's physical extent
// [align_down(off), min(align_up(off+len), file_size)) is split into 2 MiB
// blocks (all 4096-aligned). Only an EOF-clamped final chunk can be sub-512B;
// that tail is read synchronously through the buffered twin right here
// (colibri twin-tail pattern, audits/w3/nvme-reader.md §2.3). Logical bytes
// land at slot + request head (see req_head()); consumers use LayerFeeder::map.
void NvmeReader::build_blocks_locked(const ReadPlan& plan, Unit& u) {
    u.blocks.clear(); u.reqs.clear(); u.next_blk = 0; u.failed = false; u.ok = false;
    u.done.store(false, std::memory_order_relaxed);
    u64 cursor = 0;                                       // running offset inside the ring region
    for (const ReadRequest& r : plan) {
        const u32 fi = file_index_locked(r.path);
        const File& f = files_[fi];
        if (r.len == 0) continue;
        if (r.offset + r.len > f.size) die("request extent past EOF");   // strict caller contract
        const u64 base = align_down_4096(r.offset);
        const u64 end  = std::min(align_up_4096(r.offset + r.len), f.size);
        for (u64 off = base; off < end; ) {
            u64 n = std::min(kBlock, end - off);
            u32 dlen = u32(n);
            if ((n & 511) != 0) {                         // EOF clamp created a sub-sector chunk
                dlen = u32(n) & ~511u;
                if (u32(n) - dlen) {                      // <512 B tail via the buffered twin, sync,
                    OVERLAPPED ov{}; ov.Offset = u32(off + dlen); ov.OffsetHigh = u32((off + dlen) >> 32);
                    DWORD got = 0;
                    if (!ReadFile(f.twin, u.slot + cursor + dlen, u32(n) - dlen, &got, &ov) || got != u32(n) - dlen)
                        die("tail read (buffered twin)");
                }
            }
            if (dlen) u.blocks.push_back({off, dlen, fi, u32(cursor)});
            off += n; cursor += n;
        }
    }
    u.reqs.assign(u.blocks.size(), Req{});
    u.left.store(u32(u.blocks.size()), std::memory_order_relaxed);
}

u32 NvmeReader::submit(const ReadPlan& plan, void* ring_slot, DoneFn done, void* ctx) {
    if (!ring_slot || (uintptr_t(ring_slot) & (kSector - 1))) die("ring slot not 4096-aligned");
    std::lock_guard<std::mutex> lk(mtx_);
    if (stop_.load(std::memory_order_acquire)) die("submit on stopped reader");
    u32 h;
    if (!free_.empty()) { h = free_.back(); free_.pop_back(); units_[h].reset(new Unit{}); }  // recycled slot
    else { units_.emplace_back(new Unit{}); h = u32(units_.size() - 1); }
    Unit& u = *units_[h];
    u.slot = static_cast<u8*>(ring_slot);
    u.fn = done; u.ctx = ctx;
    if (!u.event) { u.event = CreateEventW(nullptr, TRUE, FALSE, nullptr); if (!u.event) die("CreateEvent"); }
    else ResetEvent(u.event);
    build_blocks_locked(plan, u);                          // may die() on a bad plan (handle leaks; acceptable)
    order_.push_back(h);
    if (u.blocks.empty()) {                                // degenerate plan: complete immediately
        u.ok = true; u.done.store(true, std::memory_order_release); SetEvent(u.event);
    } else {
        top_up_locked();
    }
    return h;
}

// One ReadFile for block bi of unit ui. Sync failures retry inline (bounded);
// ERROR_IO_PENDING is the normal async path. Called with mtx_ held.
void NvmeReader::issue_block_locked(u32 ui, u32 bi) {
    Unit& u = *units_[ui];
    Blk& b = u.blocks[bi];
    Req& r = u.reqs[bi];
    HANDLE h = files_[b.file].direct;
    for (;;) {
        r.ov = OVERLAPPED{};
        r.ov.Offset = u32(b.off); r.ov.OffsetHigh = u32(b.off >> 32);
        r.unit = ui; r.blk = bi; r.len = b.len;
        outstanding_.fetch_add(1, std::memory_order_release);
        if (ReadFile(h, u.slot + b.dst, b.len, nullptr, &r.ov) || GetLastError() == ERROR_IO_PENDING)
            return;
        outstanding_.fetch_sub(1, std::memory_order_release);
        DWORD e = GetLastError();
        if (e == ERROR_OPERATION_ABORTED || ++r.tries >= kMaxTries) { count_block_locked(ui, false); return; }
        // transient device error: bounded retry of the same block/destination
    }
}

// Terminal accounting for one block. Last block finishes the unit: event set,
// callback queued (fired outside mtx_ by the worker / submitter).
void NvmeReader::count_block_locked(u32 ui, bool ok) {
    Unit& u = *units_[ui];
    if (!ok) { u.failed = true; fatal_.store(true, std::memory_order_release); }
    if (u.left.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        u.ok = !u.failed;
        u.done.store(true, std::memory_order_release);
        SetEvent(u.event);
        if (u.fn) pend_cbs_.emplace_back(u.fn, u.ctx, ui, u.ok);
    }
}

// Walk live units in submission order, issuing until QD is full. Strict FIFO
// keeps the drive streaming sequentially (the plan order IS the access order).
void NvmeReader::top_up_locked() {
    for (u32 h : order_) {
        Unit* u = units_[h].get();
        if (!u || u->done.load(std::memory_order_acquire)) continue;
        while (u->next_blk < u->blocks.size()) {
            if (outstanding_.load(std::memory_order_acquire) >= qd_) return;
            const u32 bi = u->next_blk++;
            issue_block_locked(h, bi);
        }
    }
}

bool NvmeReader::ready(u32 stream) const noexcept {
    std::lock_guard<std::mutex> lk(mtx_);
    const Unit* u = stream < units_.size() ? units_[stream].get() : nullptr;
    return u && u->done.load(std::memory_order_acquire);
}

void NvmeReader::wait(u32 stream) {
    HANDLE ev = done_event(stream);
    if (ev) WaitForSingleObject(ev, INFINITE);
}

HANDLE NvmeReader::done_event(u32 stream) const noexcept {
    std::lock_guard<std::mutex> lk(mtx_);
    const Unit* u = stream < units_.size() ? units_[stream].get() : nullptr;
    return u ? u->event : nullptr;
}

void NvmeReader::retire(u32 stream) noexcept {
    std::lock_guard<std::mutex> lk(mtx_);
    if (stream >= units_.size() || !units_[stream]) return;
    Unit* u = units_[stream].get();
    if (!u->done.load(std::memory_order_acquire)) return;   // retiring a live unit is a caller bug; ignore
    if (u->event) { CloseHandle(u->event); u->event = nullptr; }  // recycle path builds a fresh Unit+event; without this every retire leaks one
    units_[stream].reset();
    for (auto it = order_.begin(); it != order_.end(); ++it)
        if (*it == stream) { order_.erase(it); break; }
    free_.push_back(stream);
}

void NvmeReader::worker_loop() {
    OVERLAPPED_ENTRY es[16];
    for (;;) {
        ULONG n = 0;
        const BOOL got = GetQueuedCompletionStatusEx(port_, es, 16, &n, 1000, FALSE);
        std::vector<PendingCb> fire;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (got) {
                for (ULONG i = 0; i < n; ++i) {
                    Req* r = reinterpret_cast<Req*>(es[i].lpOverlapped);
                    if (!r) continue;                        // shutdown wake post
                    const u32 ui = r->unit, bi = r->blk;
                    if (!units_[ui]) continue;               // retired mid-flight (defensive)
                    Unit& u = *units_[ui];
                    DWORD got_bytes = 0;
                    const BOOL ok = GetOverlappedResult(files_[u.blocks[bi].file].direct, &r->ov, &got_bytes, FALSE);
                    outstanding_.fetch_sub(1, std::memory_order_acq_rel);
                    if (ok && got_bytes == r->len) { count_block_locked(ui, true); continue; }
                    const DWORD err = ok ? ERROR_HANDLE_EOF : GetLastError();
                    if (err == ERROR_OPERATION_ABORTED) { count_block_locked(ui, false); continue; }  // teardown
                    if (++r->tries < kMaxTries) { issue_block_locked(ui, bi); continue; }              // retry (re-adds outstanding)
                    count_block_locked(ui, false);           // hard failure -> fatal, waiters wake
                }
            }
            if (!stop_.load(std::memory_order_acquire)) top_up_locked();   // self-arming top-up
            fire.swap(pend_cbs_);                            // callbacks fire OUTSIDE mtx_
        }
        if (!stop_.load(std::memory_order_acquire))
            for (auto& c : fire) c.fn(c.ctx, c.handle, c.ok);
        if (stop_.load(std::memory_order_acquire) &&
            (!got || outstanding_.load(std::memory_order_acquire) == 0))
            return;                                          // port dead or fully drained
    }
}

// =============================================================================
// PinnedRing
// =============================================================================
PinnedRing::PinnedRing(u64 total_bytes, u32 slot_count)
    : slot_count_(slot_count ? slot_count : 1) {
    if (!total_bytes) die("PinnedRing: zero bytes");
    slot_bytes_ = align_down_4096(total_bytes / slot_count_);
    if (slot_bytes_ < kSector) die("PinnedRing: slot smaller than one sector");
    const u64 bytes = slot_bytes_ * slot_count_;
    base_ = static_cast<u8*>(VirtualAlloc(nullptr, SIZE_T(bytes), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!base_) die("PinnedRing: VirtualAlloc");
    // Pinned = DMA-able by cudaMemcpyAsync + page-locked against standby
    // eviction + readable in place by CPU GEMV. WDDM caps pinned at ~50% RAM
    // (~7.95 GiB here); the 4x368 MiB default sits far under. If registration
    // fails (no context yet / cap / fragmentation): VirtualLock fallback —
    // CPU GEMV unaffected, H2D pays a staging bounce. audits/w3/pcie-pipeline.md §2.
    cudaSetDevice(0);                                       // ensure a context exists before registering
    const cudaError_t e = cudaHostRegister(base_, SIZE_T(bytes), cudaHostRegisterDefault);
    pinned_ = (e == cudaSuccess);
    if (!pinned_) {
        std::fprintf(stderr, "[streaming] cudaHostRegister failed (%s) -> VirtualLock fallback\n",
                     cudaGetErrorString(e));
        cudaGetLastError();                                 // clear the sticky error
        for (u32 i = 0; i < slot_count_; ++i)
            if (!VirtualLock(slot_base(i), SIZE_T(slot_bytes_))) { /* best effort */ }
        locked_ = true;
    }
    ctl_.reset(new SlotCtl[slot_count_]());              // value-init: st=FREE, wake=nullptr
    for (u32 i = 0; i < slot_count_; ++i) {
        ctl_[i].st.store(FREE, std::memory_order_relaxed);
        ctl_[i].wake = CreateEventW(nullptr, FALSE, FALSE, nullptr);   // auto-reset; timeout re-check loop
        if (!ctl_[i].wake) die("PinnedRing: CreateEvent");
    }
}

PinnedRing::~PinnedRing() {
    for (u32 i = 0; i < slot_count_; ++i)
        if (ctl_ && ctl_[i].wake) CloseHandle(ctl_[i].wake);
    if (!base_) return;
    if (pinned_) cudaHostUnregister(base_);
    else if (locked_)
        for (u32 i = 0; i < slot_count_; ++i) VirtualUnlock(slot_base(i), SIZE_T(slot_bytes_));
    VirtualFree(base_, 0, MEM_RELEASE);
}

bool PinnedRing::try_claim(u32 i) noexcept {
    u32 e = FREE;
    return ctl_[i].st.compare_exchange_strong(e, FILLING, std::memory_order_acq_rel, std::memory_order_acquire);
}

void PinnedRing::publish(u32 i) noexcept {
    ctl_[i].st.store(READY, std::memory_order_release);     // publish slot data with the state
    SetEvent(ctl_[i].wake);
}

void PinnedRing::acquire(u32 i) noexcept {
    u32 spin = kAcquireSpin;
    for (;;) {
        u32 s = ctl_[i].st.load(std::memory_order_acquire);
        if (s == READY && ctl_[i].st.compare_exchange_strong(s, IN_USE, std::memory_order_acq_rel, std::memory_order_acquire))
            return;
        if (spin) { --spin; YieldProcessor(); continue; }   // spin first: ~us wake vs 10+ ms layer
        WaitForSingleObject(ctl_[i].wake, 20);              // then bounded event wait: no lost-wakeup hang
        spin = 64;
    }
}

void PinnedRing::release(u32 i) noexcept {
    ctl_[i].st.store(FREE, std::memory_order_release);
    SetEvent(ctl_[i].wake);                                 // wake claim re-checks (post-teardown)
}

// =============================================================================
// LayerFeeder
// =============================================================================
LayerFeeder::LayerFeeder(u64 ring_bytes, u32 slots, u32 reader_threads, ConsumeMode mode)
    : reader_(reader_threads), ring_(ring_bytes, slots), mode_(mode) {}

LayerFeeder::~LayerFeeder() {
    // Shutdown ordering (normative): join the reader threads FIRST (CancelIoEx
    // -> ABORTED completions -> workers exit; callbacks suppressed under stop_),
    // so no on_done can touch ring_ while it destructs. Member order (reader_
    // before ring_) then makes ring_ destruct first and reader_'s idempotent
    // shutdown() a no-op.
    reader_.shutdown();
}

void LayerFeeder::begin_epoch(const std::vector<ReadPlan>& layer_plans) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (n_ && released_ != n_) die("begin_epoch: prior epoch not fully released");
    if (!reader_.healthy()) die("begin_epoch: reader fatal");
    n_ = u32(layer_plans.size());
    plans_ = layer_plans;
    map_.assign(n_, {}); span_.assign(n_, 0); arm_.assign(n_, Arm{0xFFFFFFFFu, 0});
    next_submit_ = 0; released_ = 0; fatal_ = false;
    for (u32 e = 0; e < n_; ++e) {
        u64 cursor = 0;
        map_[e].reserve(plans_[e].size());
        for (const ReadRequest& r : plans_[e]) {
            map_[e].push_back(cursor + req_head(r));        // logical data position in the slot
            cursor += req_span(r);
        }
        span_[e] = cursor;
        if (cursor > ring_.slot_bytes())
            die("plan exceeds ring slot (mtp/outside are read_once material, not ring units)");
    }
    arm_locked();
}

// Submit while next_submit_ < released_ + slots  =>  read-ahead depth = slots-1
// beyond the unit the consumer currently holds. Slot assignment is cyclic
// (epoch % slots); the window invariant guarantees the previous occupant of a
// slot has been released before we claim it.
void LayerFeeder::arm_locked() {
    while (next_submit_ < n_ && next_submit_ < released_ + ring_.slots()) {
        const u32 e = next_submit_;
        const u32 slot = e % ring_.slots();
        if (!ring_.try_claim(slot)) { fatal_ = true; return; }   // window invariant violated (logic bug)
        const u32 h = reader_.submit(plans_[e], ring_.slot_base(slot), &LayerFeeder::on_done, this);
        arm_[e] = {h, slot};
        ++next_submit_;
    }
}

// Reader-thread trampoline: unit done -> publish slot (READY wakes consumers;
// on failure we publish anyway so blocked acquirers wake and observe !healthy),
// retire the stream handle, re-arm (harmless: capacity frees on release).
// The arm_ search runs DESCENDING and the matched entry is cleared: stream
// handles are recycled by retire()->submit(), so a stale ascending match could
// otherwise publish the recycled slot mid-fill (wrong-epoch corruption).
void LayerFeeder::on_done(void* ctx, u32 stream, bool ok) {
    LayerFeeder* self = static_cast<LayerFeeder*>(ctx);
    std::lock_guard<std::mutex> lk(self->mtx_);
    if (!ok) self->fatal_ = true;
    for (u32 e = self->n_; e-- > 0;)
        if (self->arm_[e].stream == stream) {
            self->ring_.publish(self->arm_[e].slot);
            self->arm_[e].stream = 0xFFFFFFFFu;             // stale-proof the handle
            break;
        }
    self->reader_.retire(stream);
    try { self->arm_locked(); } catch (...) { self->fatal_ = true; }
}

const void* LayerFeeder::acquire_layer(int epoch_index) {
    if (epoch_index < 0 || u32(epoch_index) >= n_) return nullptr;
    if (!healthy()) return nullptr;
    {   // out-of-window acquire (skipping layers) is a caller bug; the slot's
        // current unit would belong to another epoch — refuse instead of hang.
        std::lock_guard<std::mutex> lk(mtx_);
        if (u32(epoch_index) >= next_submit_) return nullptr;
    }
    const u32 slot = u32(epoch_index) % ring_.slots();
    ring_.acquire(slot);                                    // READY -> IN_USE (blocks; release/acquire pair)
    return healthy() ? ring_.slot_base(slot) : nullptr;     // failed unit publishes READY-with-garbage + fatal
}

void LayerFeeder::release_layer(int epoch_index) noexcept {
    if (epoch_index < 0 || u32(epoch_index) >= n_) return;
    const u32 slot = u32(epoch_index) % ring_.slots();
    ring_.release(slot);                                    // IN_USE -> FREE
    std::lock_guard<std::mutex> lk(mtx_);
    released_ = u32(epoch_index) + 1;                       // strictly sequential contract (decode is layer-serial)
    try { arm_locked(); } catch (...) { fatal_ = true; }
}

const void* LayerFeeder::map(int epoch_index, int request_index) const noexcept {
    if (epoch_index < 0 || u32(epoch_index) >= n_) return nullptr;
    if (request_index < 0 || u64(request_index) >= map_[epoch_index].size()) return nullptr;
    return static_cast<const u8*>(ring_.slot_base(u32(epoch_index) % ring_.slots())) + map_[epoch_index][u64(request_index)];
}

u64 LayerFeeder::plan_span(int epoch_index) const noexcept {
    return (epoch_index >= 0 && u32(epoch_index) < n_) ? span_[epoch_index] : 0;
}

} // namespace insignia

// =============================================================================
// Smoke test (host-only). Compile-verified out of the box WITHOUT this macro;
// built + run explicitly with -DINSIG_STREAMING_SMOKE:
//   nvcc ... -DINSIG_STREAMING_SMOKE src\streaming.cu -o build\streaming-smoke.exe
// Reads 64 MiB of the real checkpoint through NvmeReader+PinnedRing+LayerFeeder
// and byte-compares against a plain buffered read; exercises multi-request
// slots, epoch re-arm, and teardown with units in flight.
// =============================================================================
#ifdef INSIG_STREAMING_SMOKE
#include <cstdio>
#include <cstdlib>

static double now_s() {
    LARGE_INTEGER c, f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f);
    return double(c.QuadPart) / double(f.QuadPart);
}

int main(int argc, char** argv) {
    using namespace insignia;
    setvbuf(stdout, nullptr, _IONBF, 0);                    // crash-safe smoke logging
    const char* path8 = argc > 1 ? argv[1] : "Qwen3.8-27B-FP8\\layers-0.safetensors";
    wchar_t wpath[512];
    if (!MultiByteToWideChar(CP_ACP, 0, path8, -1, wpath, 512)) { std::puts("path too long"); return 1; }
    std::fprintf(stderr, "[smoke] file=%s\n", path8);

    // ---- golden: plain buffered read of the first 192 MiB -------------------
    const u64 GOLD = 192ull << 20;
    HANDLE g = CreateFileW(wpath, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (g == INVALID_HANDLE_VALUE) { std::puts("cannot open file"); return 1; }
    u8* gold = static_cast<u8*>(VirtualAlloc(nullptr, SIZE_T(GOLD), MEM_COMMIT, PAGE_READWRITE));
    if (!gold) { std::puts("golden alloc failed"); return 1; }
    {   u64 off = 0;
        while (off < GOLD) {
            OVERLAPPED ov{}; ov.Offset = u32(off); ov.OffsetHigh = u32(off >> 32);
            DWORD got = 0; u64 n = std::min<u64>(8 << 20, GOLD - off);
            if (!ReadFile(g, gold + off, DWORD(n), &got, &ov) || got != n) { std::puts("golden read failed"); return 1; }
            off += n;
        }
        CloseHandle(g);
    }
    std::fprintf(stderr, "[smoke] golden read done\n");

    const u64 DATA_START = 2600;              // layers-0.safetensors header (8 + 2592 JSON), audits/w3/loader-gaps.md §7.2
    const u64 LEN = 32ull << 20;
    int rc = 0;
    {
        // ring: 2 slots x 80 MiB (smoke-sized; production default is 4 x 368 MiB)
        LayerFeeder feeder(u64(2) * 80 << 20, 2, 2, LayerFeeder::ConsumeMode::zero_copy);
        std::fprintf(stderr, "[smoke] feeder up: %u slots x %llu MiB, cuda_pinned=%d\n",
                     feeder.slots(), (unsigned long long)(feeder.slot_bytes() >> 20),
                     int(feeder.ring_pinned()));
        ReadPlan p0 = { {wpath, DATA_START, LEN}, {wpath, (128ull << 20) + DATA_START, LEN} };  // 2 requests, 1 slot
        ReadPlan p1 = { {wpath, 64ull << 20, 2 * LEN} };                                       // 1 request, next slot
        const std::vector<ReadPlan> epoch = {p0, p1};

        for (int round = 0; round < 2; ++round) {   // round 1: cold NVMe; round 2: epoch re-arm path
            const double t0 = now_s();
            feeder.begin_epoch(epoch);
            std::fprintf(stderr, "[smoke] round %d: epoch begun\n", round);
            const void* s0 = feeder.acquire_layer(0);
            const double t1 = now_s();
            if (!s0) { std::puts("FAIL: acquire_layer(0) returned null"); return 1; }
            const u8* a = static_cast<const u8*>(feeder.map(0, 0));
            const u8* b = static_cast<const u8*>(feeder.map(0, 1));
            if (!a || !b) { std::puts("FAIL: map null"); return 1; }
            const bool ok0 = std::memcmp(a, gold + DATA_START, LEN) == 0;
            const bool ok1 = std::memcmp(b, gold + (128ull << 20) + DATA_START, LEN) == 0;
            const u64 span0 = feeder.plan_span(0);
            std::printf("[smoke] round %d slot0: 2x%llu MiB %s %s (%.2f GiB/s fill, span %llu B)\n", round,
                        (unsigned long long)(LEN >> 20), ok0 ? "MATCH" : "MISMATCH", ok1 ? "MATCH" : "MISMATCH",
                        double(span0) / (t1 - t0) / (1ull << 30), (unsigned long long)span0);
            rc |= (!ok0 || !ok1);
            feeder.release_layer(0);

            const void* s1 = feeder.acquire_layer(1);   // second slot, read-ahead while slot0 consumed
            if (!s1) { std::puts("FAIL: acquire_layer(1) returned null"); return 1; }
            const u8* c = static_cast<const u8*>(feeder.map(1, 0));
            const bool ok2 = c && std::memcmp(c, gold + (64ull << 20), 2 * LEN) == 0;
            std::printf("[smoke] round %d slot1: %llu MiB %s\n", round,
                        (unsigned long long)((2 * LEN) >> 20), ok2 ? "MATCH" : "MISMATCH");
            rc |= !ok2;
            feeder.release_layer(1);
        }
        // teardown with units in flight: begin a third epoch, destruct immediately
        feeder.begin_epoch(epoch);
        std::puts("[smoke] destroying feeder with units in flight (CancelIoEx teardown path)");
    }
    std::puts(rc ? "SMOKE FAILED" : "SMOKE PASSED");
    VirtualFree(gold, 0, MEM_RELEASE);
    return rc;
}
#endif // INSIG_STREAMING_SMOKE
