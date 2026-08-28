#pragma once
// =============================================================================
// insignia_streaming.hpp — NVMe -> pinned-RAM streaming layer (w3)
//
// NvmeReader : IOCP reader; FILE_FLAG_NO_BUFFERING|FILE_FLAG_OVERLAPPED direct
//              handles (opened on demand, cached, one buffered twin per file for
//              sub-sector tails); generic (file, offset, len) plans — NOT the
//              INSIDX index format. Sector-aligned splitting: offsets rounded
//              DOWN to 4096, lengths UP; consumers handle in-slot offsets.
// PinnedRing : VirtualAlloc + cudaHostRegister(Default) slot ring with an
//              atomic FREE/FILLING/READY/IN_USE state machine per slot.
//              (WDDM caps pinned at ~50% of RAM; 4x368 MiB is far under.)
// LayerFeeder: owns both; begin_epoch()/acquire_layer()/release_layer();
//              read-ahead depth = slots-1, self-arming top-up driven from the
//              reader completion callbacks.
//
// Designs: audits/w3/nvme-reader.md §3, audits/w3/loader-gaps.md §3.3
//          audits/w3/pcie-pipeline.md §7.2, audits/w3/colibri-sched-deep.md §8.
// Threading: reader threads pinned to LP 6-11 (SMT siblings, mask 0xFC0) at
//            THREAD_PRIORITY_ABOVE_NORMAL; decode/GEMV threads keep LP 0-5.
// =============================================================================
#ifndef INSIG_STREAMING_HPP
#define INSIG_STREAMING_HPP

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#include <windows.h>
#include <cuda_runtime.h>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace insignia {

using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;

// ---------------------------------------------------------------------------
// Plan types. A ReadPlan is an ordered list of logical extents that are read,
// physically aligned, and concatenated into ONE contiguous ring region in plan
// order. The reader never interprets paths/offsets (no index format knowledge).
// ---------------------------------------------------------------------------
struct ReadRequest {
    const wchar_t* path;   // UTF-16 file path (owned by the caller's plan)
    u64 offset;            // logical extent [offset, offset+len)
    u64 len;
};
using ReadPlan = std::vector<ReadRequest>;

// Alignment math — the single source of truth shared by reader, feeder and
// consumers. Physical extent of a request = [align_down(off), align_up(off+len))
// clamped to EOF; the logical bytes land at region_base + (off - align_down(off)).
constexpr u64 kSector = 4096;                 // 4096 satisfies 512e and 4Kn alike
constexpr u64 kBlock  = 2ull << 20;           // 2 MiB in-flight block (QD16 => 32 MiB queued)
inline constexpr u64 align_down_4096(u64 v) { return v & ~(kSector - 1); }
inline constexpr u64 align_up_4096(u64 v)   { return (v + kSector - 1) & ~(kSector - 1); }
inline constexpr u64 req_span(const ReadRequest& r) {  // physical bytes (pre-EOF-clamp)
    return align_up_4096(r.offset + r.len) - align_down_4096(r.offset);
}
inline constexpr u64 req_head(const ReadRequest& r) {  // logical bytes before region start
    return r.offset - align_down_4096(r.offset);
}

// ---------------------------------------------------------------------------
// NvmeReader — single IOCP, N worker threads on GetQueuedCompletionStatusEx,
// FIFO stream order (disk stays sequential), global QD16 (>= 8x2 MiB in flight
// whenever work exists), bounded retry, CancelIoEx teardown.
// ---------------------------------------------------------------------------
class NvmeReader final {
public:
    using DoneFn = void (*)(void* ctx, u32 stream, bool ok);  // fired on a reader thread

    explicit NvmeReader(u32 num_threads = 2);
    ~NvmeReader() { shutdown(); }
    NvmeReader(const NvmeReader&) = delete;
    NvmeReader& operator=(const NvmeReader&) = delete;

    // Fill ring_slot with the plan's concatenated physical extents. Returns a
    // stream handle; completion is observed via ready()/wait()/done_event() or
    // the optional callback (fired once, on a reader thread, outside locks).
    // slot must be 4096-aligned (PinnedRing guarantees this).
    u32   submit(const ReadPlan& plan, void* ring_slot, DoneFn done = nullptr, void* ctx = nullptr);

    bool  ready(u32 stream) const noexcept;          // non-blocking poll (done, any outcome)
    void  wait(u32 stream);                          // block until done / fatal
    HANDLE done_event(u32 stream) const noexcept;    // manual-reset; set on completion/failure
    void  retire(u32 stream) noexcept;               // free the handle for reuse (only when ready)

    bool  healthy() const noexcept { return !fatal_.load(std::memory_order_acquire); }
    void  shutdown() noexcept;                       // idempotent; CancelIoEx + drain + join

    static constexpr u32 kMaxTries = 3;              // bounded retry per block
    static constexpr u32 kQD       = 16;             // outstanding 2 MiB blocks (>= 8 mandated)

private:
    struct Blk { u64 off; u32 len; u32 file; u32 dst; };   // dst = offset inside ring region
    struct Req { OVERLAPPED ov{}; u32 unit; u32 blk; u32 tries{}; u32 len; };
    struct Unit {
        std::vector<Blk> blocks; std::vector<Req> reqs;
        u8* slot{};                       // ring region base (4096-aligned)
        u32 next_blk{};                   // issue cursor (guarded by mtx_)
        std::atomic<u32> left{};          // blocks still outstanding
        bool failed{};                    // guarded by mtx_
        std::atomic<bool> done{};         // terminal (ok or failed)
        bool ok{};                        // valid when done
        HANDLE event{};                   // manual-reset
        DoneFn fn{}; void* ctx{};         // completion callback
    };
    struct File { HANDLE direct{}, twin{}; u64 size{}; };

    u32   file_index_locked(const wchar_t* path);    // open-on-demand handle cache
    void  build_blocks_locked(const ReadPlan& plan, Unit& u);
    void  issue_block_locked(u32 ui, u32 bi);        // prep ov + ReadFile (retry loop)
    void  count_block_locked(u32 ui, bool ok);       // terminal accounting; finishes unit
    void  top_up_locked();                           // walk FIFO, keep outstanding_ < kQD
    void  worker_loop();
    static DWORD WINAPI worker_tramp(LPVOID p);

    struct PendingCb { DoneFn fn; void* ctx; u32 handle; bool ok; };  // fired outside mtx_

    mutable std::mutex mtx_;                          // guards everything below except atomics
    std::vector<std::unique_ptr<Unit>> units_;        // handle -> unit (null = retired)
    std::deque<u32> order_;                           // live units in submission order
    std::vector<u32> free_;                           // recycled handles
    std::vector<PendingCb> pend_cbs_;                 // completions awaiting callback fire
    std::vector<File> files_;                         // handle cache (index = Blk::file)
    std::vector<std::wstring> paths_;
    HANDLE port_{};
    std::vector<HANDLE> threads_;
    std::atomic<u32> outstanding_{0};
    std::atomic<bool> fatal_{false}, stop_{false};
    u32 qd_ = kQD;
};

// ---------------------------------------------------------------------------
// PinnedRing — total_bytes split into slot_count equal 4096-aligned slots.
// cudaHostRegister(cudaHostRegisterDefault) so slots are DMA-able by
// cudaMemcpyAsync AND readable in place by CPU GEMV; on WDDM the driver caps
// pinned at ~50% of RAM (~7.95 GiB here) — 4x368 MiB is the intended default.
// Fallback when registration fails: VirtualLock'd pageable (CPU GEMV unaffected,
// H2D pays a staging bounce).
//
// Slot state machine (atomic, one cache line per slot):
//   FREE --try_claim--> FILLING --publish--> READY --acquire--> IN_USE
//     ^                                                |
//     +------------------release-----------------------+
// ---------------------------------------------------------------------------
class PinnedRing final {
public:
    enum SlotState : u32 { FREE = 0, FILLING, READY, IN_USE };

    PinnedRing(u64 total_bytes, u32 slot_count);
    ~PinnedRing();
    PinnedRing(const PinnedRing&) = delete;
    PinnedRing& operator=(const PinnedRing&) = delete;

    void* slot_base(u32 i) const noexcept { return base_ + u64(i) * slot_bytes_; }
    u64   slot_bytes() const noexcept { return slot_bytes_; }
    u32   slots() const noexcept { return slot_count_; }
    bool  cuda_pinned() const noexcept { return pinned_; }
    SlotState state(u32 i) const noexcept { return SlotState(ctl_[i].st.load(std::memory_order_acquire)); }

    bool  try_claim(u32 i) noexcept;   // FREE -> FILLING (CAS); reader side
    void  publish(u32 i) noexcept;     // FILLING -> READY (release store) + wake waiters
    void  acquire(u32 i) noexcept;     // spin/wait until READY -> IN_USE (no timeout: unit always ends)
    void  release(u32 i) noexcept;     // IN_USE -> FREE (release store) + wake (for claim re-checks)

    static constexpr u32 kAcquireSpin = 1u << 16;    // pause iterations before event wait

private:
    struct alignas(64) SlotCtl { std::atomic<u32> st{FREE}; HANDLE wake{}; };  // one cache line/slot
    u8* base_{}; u64 slot_bytes_{}; u32 slot_count_{}; bool pinned_{}, locked_{};
    std::unique_ptr<SlotCtl[]> ctl_;                     // SlotCtl is non-movable (atomic)
};

// ---------------------------------------------------------------------------
// LayerFeeder — the decode-epoch state machine over NvmeReader + PinnedRing.
//   begin_epoch(plans) : decode epoch = ALL N-tier shards in layer order; every
//                        plan fills one ring slot (slot = epoch_index % slots).
//   acquire_layer(i)   : blocks until slot READY; returns pinned ptr (consumer
//                        cudaMemcpyAsync's from it or CPU-GEMVs it in place).
//   release_layer(i)   : IN_USE -> FREE; self-arms the next submit.
//   map(i, r)          : pointer to request r's LOGICAL bytes (valid between
//                        acquire_layer(i) and release_layer(i)).
// Read-ahead depth = slots-1: submit(epoch e) allowed while e < released+slots.
// ConsumeMode: v1 returns the pinned pointer for BOTH modes; `copy_out` is the
// reserved hook where acquire will issue the slot->VRAM cudaMemcpyAsync chain.
// ---------------------------------------------------------------------------
class LayerFeeder final {
public:
    enum class ConsumeMode : u8 { zero_copy = 0, copy_out = 1 };

    // ring defaults: 4 slots x 368 MiB (= 184 x 2 MiB; covers the 383.87 MB
    // linear shards of Qwen3.8-27B-FP8 with 4096 slack) — audits/w3/nvme-reader.md §3.1.
    static constexpr u64 kDefaultSlotBytes = 184ull * kBlock;
    static constexpr u32 kDefaultSlots = 4;

    LayerFeeder(u64 ring_bytes = u64(kDefaultSlots) * kDefaultSlotBytes,
                u32 slots = kDefaultSlots, u32 reader_threads = 2,
                ConsumeMode mode = ConsumeMode::zero_copy);
    ~LayerFeeder();
    LayerFeeder(const LayerFeeder&) = delete;
    LayerFeeder& operator=(const LayerFeeder&) = delete;

    void        begin_epoch(const std::vector<ReadPlan>& layer_plans); // prior epoch must be fully released
    const void* acquire_layer(int epoch_index);                       // blocks; nullptr on fatal error
    void        release_layer(int epoch_index) noexcept;              // strictly sequential (decode is layer-serial)
    const void* map(int epoch_index, int request_index) const noexcept; // logical data ptr (holding required)
    u64         plan_span(int epoch_index) const noexcept;              // physical bytes the epoch fills
    u32         plans() const noexcept { return n_; }
    u32         slots() const noexcept { return ring_.slots(); }
    u64         slot_bytes() const noexcept { return ring_.slot_bytes(); }
    bool        ring_pinned() const noexcept { return ring_.cuda_pinned(); }
    ConsumeMode mode() const noexcept { return mode_; }
    bool        healthy() const noexcept { return reader_.healthy() && !fatal_; }

private:
    static void on_done(void* ctx, u32 stream, bool ok);  // reader-thread trampoline
    void        arm_locked();                             // submit while next_submit_ < released_ + slots

    NvmeReader reader_;
    PinnedRing ring_;
    ConsumeMode mode_;
    std::mutex mtx_;
    std::vector<ReadPlan> plans_;         // [epoch]
    std::vector<std::vector<u64>> map_;   // [epoch][req] logical in-slot offsets
    std::vector<u64> span_;               // [epoch] physical bytes
    struct Arm { u32 stream; u32 slot; };
    std::vector<Arm> arm_;                // [epoch]
    u32 next_submit_{};                   // next epoch index to submit
    u32 released_{};                      // count of released epochs
    u32 n_{};                             // plans this epoch
    bool fatal_{};
};

} // namespace insignia
#endif // INSIG_STREAMING_HPP
