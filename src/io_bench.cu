// =============================================================================
// src/io_bench.cu — Insignia NVMe streaming microbenchmark (pure Win32, no CUDA).
//
// Phases:
//  [1] SEQ-DIRECT  E: model files (Qwen3.8-27B-FP8/layers-0..5.safetensors ~2.29GB),
//      FILE_FLAG_NO_BUFFERING|FILE_FLAG_OVERLAPPED, one IOCP, N reader threads,
//      blk {256K,1M,2M,4M} x qd {1,4,8,16} (qd = outstanding OVERLAPPED per thread).
//  [2] SEQ-CACHED  same matrix through a buffered handle (page-cache control).
//  [3] thread probe at the best direct config: threads {1,2,4,8}.
//  [4] C: %TEMP%\insig_iobench.bin — exactly 6 GiB NO_BUFFERING write (+flush),
//      then direct read matrix blk {1M,2M,4M} x qd {4,8,16}; file deleted after.
//  [5] DUAL: E: pool (best E: cfg, epochs repeated) || C: pool (best C: cfg).
//
// Per config: GB/s (decimal), IOPS, avg GetQueuedCompletionStatusEx batch size,
// CPU cycles/completion, worker cores consumed -> validates "2 cores of overhead".
//
// Model: one IOCP; each thread owns qd slots; a slot is claimed (atomic armed flag)
// around a global block cursor, so completions reaped by ANY thread can re-arm ANY
// slot; exhausted-cursor claims are released and retried on GQCSEx timeout, which
// also revives slots across dual-mode epoch resets. One packet per issue; slots
// re-arm only after their packet is reaped -> <=1 outstanding op per slot, always.
// With FILE_FLAG_OVERLAPPED + port-associated handles the completion packet is
// queued even when ReadFile returns TRUE synchronously, so every completion is
// reaped through the port. A 5s stall watchdog turns a pathological hang into a
// visible FAILED row instead of a deadlock.
// =============================================================================

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using u8 = uint8_t; using u32 = uint32_t; using u64 = uint64_t;

static double g_inv_freq = 0.0;   // QPC seconds
static double g_cycle_hz = 0.0;   // thread-cycle frequency estimate

static double now() { LARGE_INTEGER c; QueryPerformanceCounter(&c); return double(c.QuadPart) * g_inv_freq; }

static const char* blk_label(u32 blk) {
    switch (blk) {
        case 256u << 10: return "256K";
        case 1u  << 20: return "1M";
        case 2u  << 20: return "2M";
        case 4u  << 20: return "4M";
        default:         return "?";
    }
}

// ---------------------------------------------------------------- sweep engine

struct RunResult {
    bool ok = false; double sec = 0, gbps = 0, iops = 0, avg_batch = 0, cyc_io = 0, cores = 0;
    u64 bytes = 0; const char* note = "";
};

struct FileEntry { const char* path; HANDLE h = INVALID_HANDLE_VALUE; u64 nblocks = 0, base = 0; };

struct Sweep;

struct Slot {
    OVERLAPPED ov;
    Sweep* sw;
    HANDLE h;                 // file this op was issued against (for GetOverlappedResult)
    u8* buf;
    std::atomic<bool> armed;  // claim: exactly one issuer between packet-reaped -> next issue
};

struct Sweep {
    const char* tag = "";
    std::vector<FileEntry> files;
    u32 blk = 0, qd = 0, nthreads = 0;
    bool direct = true, repeat = false, own_start = true;
    u64 total_blocks = 0;
    HANDLE port = nullptr, start_evt = nullptr, done_evt = nullptr;
    std::atomic<u64> cursor{0}, done{0}, bytes_accum{0}, completions{0}, gqcs_calls{0}, cycles_sum{0};
    std::atomic<long> ready{0}, failed{0};
    std::atomic<int> dual_stop{0};
    std::atomic<const char*> fail_reason{nullptr};
    double t0 = 0, t_end = 0;
};

static void fail_sweep(Sweep* sw, const char* why) {
    sw->fail_reason.store(why);
    sw->failed.store(1);
    SetEvent(sw->done_evt);
}

static void on_complete(Sweep* sw) {
    sw->completions.fetch_add(1, std::memory_order_relaxed);
    sw->bytes_accum.fetch_add(sw->blk, std::memory_order_relaxed);
    u64 d = sw->done.fetch_add(1, std::memory_order_relaxed) + 1;
    if (d == sw->total_blocks) {
        if (sw->repeat && sw->dual_stop.load(std::memory_order_acquire) == 0) {
            sw->done.store(0, std::memory_order_relaxed);    // dual mode: recycle the epoch
            sw->cursor.store(0, std::memory_order_release);
            return;
        }
        sw->t_end = now();
        SetEvent(sw->done_evt);
    }
}

static bool arm_slot(Slot* s) {                              // returns true if an op was issued
    if (s->armed.exchange(true, std::memory_order_acq_rel)) return false;
    Sweep* sw = s->sw;
    u64 b = sw->cursor.fetch_add(1, std::memory_order_relaxed);
    if (b >= sw->total_blocks) { s->armed.store(false, std::memory_order_release); return false; }
    size_t f = sw->files.size() - 1;
    while (sw->files[f].base > b) --f;                       // base[0]==0 terminates
    FileEntry* fe = &sw->files[f];
    u64 off = (b - fe->base) * u64(sw->blk);
    OVERLAPPED* ov = &s->ov;
    memset(ov, 0, sizeof(*ov));
    ov->Offset = u32(off); ov->OffsetHigh = u32(off >> 32);
    s->h = fe->h;
    DWORD got = 0;
    BOOL ok = ReadFile(fe->h, s->buf, sw->blk, &got, ov);
    if (ok || GetLastError() == ERROR_IO_PENDING) return true;   // packet lands at the port either way
    fail_sweep(sw, "ReadFile");
    return false;
}

static DWORD WINAPI worker(LPVOID p) {
    Sweep* sw = (Sweep*)p;
    const u32 qd = sw->qd, blk = sw->blk;
    sw->ready.fetch_add(1, std::memory_order_acq_rel);       // first: main must never hang on a dead worker
    const size_t bufB = size_t(qd) * blk;
    u8* buf = (u8*)VirtualAlloc(nullptr, bufB, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!buf) { fail_sweep(sw, "VirtualAlloc"); return 1; }
    memset(buf, 0, bufB);                                    // pre-touch: no soft faults during DMA
    std::vector<Slot> slots(qd);                             // per-thread: total QD = nthreads * qd
    for (u32 i = 0; i < qd; ++i) { slots[i].sw = sw; slots[i].buf = buf + size_t(i) * blk; }
    u64 cy0 = 0; QueryThreadCycleTime(GetCurrentThread(), &cy0);
    WaitForSingleObject(sw->start_evt, INFINITE);
    for (u32 i = 0; i < qd; ++i) {                           // initial fill
        arm_slot(&slots[i]);
        if (WaitForSingleObject(sw->done_evt, 0) == WAIT_OBJECT_0) break;
    }
    OVERLAPPED_ENTRY es[64];
    double last_progress = now();
    for (;;) {
        if (WaitForSingleObject(sw->done_evt, 0) == WAIT_OBJECT_0) break;
        ULONG n = 0;
        BOOL ok = GetQueuedCompletionStatusEx(sw->port, es, 64, &n, 50, FALSE);
        sw->gqcs_calls.fetch_add(1, std::memory_order_relaxed);
        if (ok) {
            for (ULONG i = 0; i < n; ++i) {
                Slot* s = (Slot*)es[i].lpOverlapped;
                DWORD got = 0;
                if (!GetOverlappedResult(s->h, &s->ov, &got, FALSE) || got != blk) {
                    fail_sweep(sw, "GetOverlappedResult"); goto out;
                }
                s->armed.store(false, std::memory_order_release);   // packet reaped: slot re-claimable
                on_complete(sw);
                if (WaitForSingleObject(sw->done_evt, 0) == WAIT_OBJECT_0) goto out;
                arm_slot(s);
            }
            last_progress = now();
        } else if (GetLastError() == WAIT_TIMEOUT) {
            if (now() - last_progress > 5.0) { fail_sweep(sw, "stall>5s (no completions)"); goto out; }
            for (u32 i = 0; i < qd; ++i) arm_slot(&slots[i]);  // revive idle slots (epoch resets)
        } else { fail_sweep(sw, "GQCSEx"); goto out; }
    }
out: {
        u64 cy1 = 0; QueryThreadCycleTime(GetCurrentThread(), &cy1);
        sw->cycles_sum.fetch_add(cy1 - cy0, std::memory_order_relaxed);
    }
    VirtualFree(buf, 0, MEM_RELEASE);
    return 0;
}

static std::vector<HANDLE> g_threads;                        // joined by sweep_finish

static bool sweep_launch(Sweep* sw, HANDLE shared_start) {
    sw->port = CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 0);
    if (!sw->port) return false;
    for (auto& fe : sw->files) {
        // buffered control: plain FILE_FLAG_OVERLAPPED. An earlier build added
        // FILE_FLAG_SEQUENTIAL_SCAN here; that hint makes the cache manager recycle the
        // read-past pages immediately, so the pass-2 "cached" run tracked disk rate and
        // hid the very effect this control exists to show. No hint -> pages enter the
        // standby list and pass 2 measures RAM-resident reads.
        DWORD flags = FILE_FLAG_OVERLAPPED | (sw->direct ? FILE_FLAG_NO_BUFFERING : 0);
        fe.h = CreateFileA(fe.path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                           nullptr, OPEN_EXISTING, flags, nullptr);
        if (fe.h == INVALID_HANDLE_VALUE) { printf("  open failed %s GLE=%lu\n", fe.path, GetLastError()); return false; }
        if (CreateIoCompletionPort(fe.h, sw->port, 0, 0) != sw->port) return false;
        LARGE_INTEGER sz{};
        if (!GetFileSizeEx(fe.h, &sz)) return false;
        fe.nblocks = u64(sz.QuadPart) / sw->blk;              // floor: whole blocks only (aligned I/O)
        if (!fe.nblocks) { printf("  file < one block: %s\n", fe.path); return false; }
    }
    u64 base = 0; sw->total_blocks = 0;
    for (auto& fe : sw->files) { fe.base = base; base += fe.nblocks; sw->total_blocks += fe.nblocks; }
    sw->cursor.store(0); sw->done.store(0); sw->bytes_accum.store(0);
    sw->completions.store(0); sw->gqcs_calls.store(0); sw->cycles_sum.store(0);
    sw->ready.store(0); sw->failed.store(0); sw->dual_stop.store(0);
    sw->fail_reason.store(nullptr);
    sw->own_start = shared_start == nullptr;
    sw->start_evt = shared_start ? shared_start : CreateEventW(nullptr, TRUE, FALSE, nullptr);
    sw->done_evt = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!sw->start_evt || !sw->done_evt) return false;
    for (u32 i = 0; i < sw->nthreads; ++i) {
        HANDLE t = CreateThread(nullptr, 0, worker, sw, 0, nullptr);
        if (!t) { printf("  CreateThread failed\n"); return false; }
        g_threads.push_back(t);
    }
    while (sw->ready.load() < (long)sw->nthreads) Sleep(1);
    return true;
}

static RunResult sweep_finish(Sweep* sw) {
    RunResult r;
    WaitForSingleObject(sw->done_evt, INFINITE);
    for (HANDLE t : g_threads) { WaitForSingleObject(t, INFINITE); CloseHandle(t); }
    g_threads.clear();
    for (auto& fe : sw->files) if (fe.h != INVALID_HANDLE_VALUE) CloseHandle(fe.h);
    if (sw->own_start && sw->start_evt) CloseHandle(sw->start_evt);
    if (sw->done_evt) CloseHandle(sw->done_evt);
    if (sw->port) CloseHandle(sw->port);
    if (sw->failed.load()) { r.note = sw->fail_reason.load() ? sw->fail_reason.load() : "?"; return r; }
    double sec = sw->t_end - sw->t0;
    if (sec <= 0) { r.note = "bad elapsed"; return r; }
    u64 bytes = sw->bytes_accum.load(), ios = sw->completions.load();
    u64 calls = sw->gqcs_calls.load(), cyc = sw->cycles_sum.load();
    r.ok = true; r.sec = sec; r.bytes = bytes;
    r.gbps = double(bytes) / 1e9 / sec;
    r.iops = double(ios) / sec;
    r.avg_batch = calls ? double(ios) / double(calls) : 0.0;
    r.cyc_io = ios ? double(cyc) / double(ios) : 0.0;
    r.cores = g_cycle_hz > 0 ? double(cyc) / (sec * g_cycle_hz) : 0.0;
    return r;
}

static RunResult run_sweep(Sweep* sw) {
    if (!sweep_launch(sw, nullptr)) { RunResult r; r.note = "launch"; printf("  sweep_launch failed GLE=%lu\n", GetLastError()); return r; }
    sw->t0 = now();
    SetEvent(sw->start_evt);
    return sweep_finish(sw);
}

static void print_header(const char* title) {
    printf("\n%s\n%-10s %5s %3s %3s | %7s | %8s | %6s | %7s | %6s\n",
           title, "tag", "blk", "qd", "thr", "GB/s", "IOPS", "batch", "cyc/io", "cores");
}

static void print_row(const char* tag, u32 blk, u32 qd, u32 thr, const RunResult& r) {
    if (!r.ok) { printf("%-10s %5s %3u %3u | FAILED (%s)\n", tag, blk_label(blk), qd, thr, r.note); return; }
    printf("%-10s %5s %3u %3u | %7.2f | %8.0f | %6.2f | %7.0f | %6.3f\n",
           tag, blk_label(blk), qd, thr, r.gbps, r.iops, r.avg_batch, r.cyc_io, r.cores);
}

// ---------------------------------------------------------------- C: test file writer

static bool write_test_file(const char* path, u64 total_bytes, double* wgbps) {
    const u32 WBLK = 8u << 20, QD = 16;
    const u64 nblocks = total_bytes / WBLK;                  // exact for 6 GiB
    HANDLE h = CreateFileA(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                           FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
    if (h == INVALID_HANDLE_VALUE) { printf("  create %s failed GLE=%lu\n", path, GetLastError()); return false; }
    HANDLE port = CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 0);
    if (!port || CreateIoCompletionPort(h, port, 0, 0) != port) { CloseHandle(h); return false; }
    u8* buf = (u8*)VirtualAlloc(nullptr, size_t(QD) * WBLK, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!buf) { CloseHandle(h); CloseHandle(port); return false; }
    u32* w = (u32*)buf; u32 x = 0x1234567u;                  // pseudo-random pattern (no zero-compress)
    for (size_t i = 0; i < size_t(QD) * WBLK / 4; ++i) { x = x * 1664525u + 1013904223u; w[i] = x; }
    std::vector<OVERLAPPED> ovs(QD);
    u64 issued = 0, reaped = 0; double t0 = now();
    bool bad = false;
    while (reaped < nblocks && !bad) {
        while (issued < nblocks && issued - reaped < QD) {
            u32 i = u32(issued % QD);
            OVERLAPPED* ov = &ovs[i]; memset(ov, 0, sizeof(*ov));
            u64 off = issued * WBLK; ov->Offset = u32(off); ov->OffsetHigh = u32(off >> 32);
            DWORD put = 0;
            if (!WriteFile(h, buf + size_t(i) * WBLK, WBLK, &put, ov) && GetLastError() != ERROR_IO_PENDING)
            { printf("  WriteFile failed GLE=%lu\n", GetLastError()); bad = true; break; }
            ++issued;
        }
        if (bad) break;
        OVERLAPPED_ENTRY es[16]; ULONG n = 0;
        if (!GetQueuedCompletionStatusEx(port, es, 16, &n, 5000, FALSE)) { printf("  write GQCSEx failed\n"); bad = true; break; }
        for (ULONG i = 0; i < n; ++i) {
            DWORD put = 0;
            if (!GetOverlappedResult(h, es[i].lpOverlapped, &put, FALSE) || put != WBLK) { printf("  write result bad\n"); bad = true; }
        }
        reaped += n;
    }
    double sec = now() - t0;
    FlushFileBuffers(h);                                     // commit SLC -> NAND before benching reads
    LARGE_INTEGER sz{}; GetFileSizeEx(h, &sz);
    CloseHandle(h); CloseHandle(port); VirtualFree(buf, 0, MEM_RELEASE);
    if (bad || u64(sz.QuadPart) != nblocks * WBLK) {
        printf("  write FAILED (size=%llu expect=%llu)\n", (unsigned long long)sz.QuadPart, (unsigned long long)(nblocks * WBLK));
        DeleteFileA(path);
        return false;
    }
    *wgbps = double(nblocks * WBLK) / 1e9 / sec;
    printf("  wrote %s: %.2f GB/s (%.2fs), flushed, size verified\n", path, *wgbps, sec);
    return true;
}

// ---------------------------------------------------------------- main

int main(int argc, char** argv) {
    const char* phase = (argc > 1) ? argv[1] : "all";
    LARGE_INTEGER f{}; QueryPerformanceFrequency(&f); g_inv_freq = 1.0 / double(f.QuadPart);
    {   // thread-cycle frequency estimate (100ms spin)
        u64 a = 0, b = 0; volatile u64 sink = 0;
        QueryThreadCycleTime(GetCurrentThread(), &a);
        double t = now(); while (now() - t < 0.10) ++sink;
        QueryThreadCycleTime(GetCurrentThread(), &b);
        g_cycle_hz = double(b - a) / 0.10; (void)sink;
    }
    SYSTEMTIME st{}; GetLocalTime(&st);
    printf("== insignia io_bench (%04d-%02d-%02d %02d:%02d) — GB/s are decimal (1e9 B/s) ==\n",
           st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute);
    printf("cycle-freq est %.2f GHz; IOCP + FILE_FLAG_OVERLAPPED; qd = outstanding per thread\n", g_cycle_hz / 1e9);

    const u32 BLKS[] = { 256u << 10, 1u << 20, 2u << 20, 4u << 20 };
    const u32 QDS[] = { 1, 4, 8, 16 };
    const u32 NTHR = 4;
    const char* MDL = "E:\\coding\\Insignia\\Qwen3.8-27B-FP8";
    static char paths[6][256];
    std::vector<FileEntry> model_files;
    for (int i = 0; i < 6; ++i) {
        snprintf(paths[i], sizeof(paths[i]), "%s\\layers-%d.safetensors", MDL, i);
        FileEntry fe{}; fe.path = paths[i]; model_files.push_back(fe);
    }

    u64 e_span = 0;
    for (auto& fe : model_files) {                           // existence + span pre-check (read-only)
        HANDLE h = CreateFileA(fe.path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        if (h == INVALID_HANDLE_VALUE) { printf("FATAL: cannot open %s\n", fe.path); return 1; }
        LARGE_INTEGER sz{}; GetFileSizeEx(h, &sz); e_span += u64(sz.QuadPart) / (256u << 10) * (256u << 10);
        CloseHandle(h);
    }
    printf("E: span = %u.%02u GB over 6 layer shards; default threads=%u\n",
           u32(e_span / 1000000000ull), u32((e_span % 1000000000ull) / 10000000ull), NTHR);

    bool run_e = !strcmp(phase, "all") || !strcmp(phase, "e");
    bool run_c = !strcmp(phase, "all") || !strcmp(phase, "c");

    double bestE = -1; u32 best_blkE = 2u << 20, best_qdE = 8;
    double bestC = -1; u32 best_blkC = 2u << 20, best_qdC = 8;

    if (run_e) {
        {   // warmup (uncounted; pages in worker buffers + steady disk state)
            print_header("-- warmup (informational, not part of the matrix) --");
            Sweep s{}; s.files = model_files; s.blk = 2u << 20; s.qd = 8; s.nthreads = NTHR; s.direct = true;
            print_row("warmE", 2u << 20, 8, NTHR, run_sweep(&s));
        }
        print_header("-- [1] SEQ-DIRECT  E: model layers-0..5 (NO_BUFFERING|OVERLAPPED) --");
        for (u32 blk : BLKS) for (u32 qd : QDS) {
            Sweep s{}; s.files = model_files; s.blk = blk; s.qd = qd; s.nthreads = NTHR; s.direct = true;
            RunResult r = run_sweep(&s);
            print_row("E-direct", blk, qd, NTHR, r);
            if (r.ok && r.gbps > bestE) { bestE = r.gbps; best_blkE = blk; best_qdE = qd; }
        }
        printf("  best E: direct cfg: blk=%s qd=%u thr=%u -> %.2f GB/s\n", blk_label(best_blkE), best_qdE, NTHR, bestE);

        print_header("-- [3] thread probe at best E: cfg --");
        for (u32 thr : { 1u, 2u, 4u, 8u }) {
            Sweep s{}; s.files = model_files; s.blk = best_blkE; s.qd = best_qdE; s.nthreads = thr; s.direct = true;
            RunResult r = run_sweep(&s);
            print_row("E-thr", best_blkE, best_qdE, thr, r);
        }

        {   // buffered warmup: fills the standby list with the span so [2] measures cached
            print_header("-- buffered warmup (cold->standby; informational) --");
            Sweep s{}; s.files = model_files; s.blk = 2u << 20; s.qd = 8; s.nthreads = NTHR; s.direct = false;
            print_row("warmBuf", 2u << 20, 8, NTHR, run_sweep(&s));
        }
        print_header("-- [2] SEQ-CACHED-PAGEABLE E: same matrix, buffered handle --");
        for (u32 blk : BLKS) for (u32 qd : QDS) {
            Sweep s{}; s.files = model_files; s.blk = blk; s.qd = qd; s.nthreads = NTHR; s.direct = false;
            RunResult r = run_sweep(&s);
            print_row("E-buf", blk, qd, NTHR, r);
        }
    }

    if (run_c) {
        char tmp[MAX_PATH], tpath[MAX_PATH];
        GetTempPathA(MAX_PATH, tmp);
        snprintf(tpath, sizeof(tpath), "%sinsig_iobench.bin", tmp);
        printf("\n-- [4] C: test file %s (exactly 6 GiB) --\n", tpath);
        const u64 SIX_GIB = 6ull << 30;
        ULARGE_INTEGER freeb{}; BOOL have_space = GetDiskFreeSpaceExA(tmp, &freeb, nullptr, nullptr);
        if (!have_space || freeb.QuadPart < 10ull << 30) {
            printf("  free=%.1f GB (<10GB) -> C: phases SKIPPED per mission rules\n",
                   have_space ? double(freeb.QuadPart) / 1e9 : -1.0);
        } else {
            printf("  %s free=%.1f GB\n", tmp, double(freeb.QuadPart) / 1e9);
            double wgb = 0;
            if (!write_test_file(tpath, SIX_GIB, &wgb)) { printf("  C: write failed -> read/dual skipped\n"); }
            else {
                std::vector<FileEntry> cfiles; FileEntry fe{}; fe.path = tpath; cfiles.push_back(fe);
                print_header("-- [4b] SEQ-DIRECT C: 6GiB test file --");
                for (u32 blk : { 1u << 20, 2u << 20, 4u << 20 }) for (u32 qd : { 4u, 8u, 16u }) {
                    Sweep s{}; s.files = cfiles; s.blk = blk; s.qd = qd; s.nthreads = NTHR; s.direct = true;
                    RunResult r = run_sweep(&s);
                    print_row("C-direct", blk, qd, NTHR, r);
                    if (r.ok && r.gbps > bestC) { bestC = r.gbps; best_blkC = blk; best_qdC = qd; }
                }
                if (bestC > 0) {
                    printf("  best C: cfg: blk=%s qd=%u thr=%u -> %.2f GB/s\n", blk_label(best_blkC), best_qdC, NTHR, bestC);
                    if (bestE > 0) {
                        printf("\n-- [5] DUAL: E: pool (best cfg, epochs repeated) || C: pool (best cfg) --\n");
                        HANDLE shared = CreateEventW(nullptr, TRUE, FALSE, nullptr);
                        Sweep se{}; se.files = model_files; se.blk = best_blkE; se.qd = best_qdE; se.nthreads = NTHR;
                        se.direct = true; se.repeat = true;
                        Sweep sc{}; sc.files = cfiles; sc.blk = best_blkC; sc.qd = best_qdC; sc.nthreads = NTHR;
                        sc.direct = true;
                        bool l1 = sweep_launch(&se, shared), l2 = sweep_launch(&sc, shared);
                        if (l1 && l2) {
                            se.t0 = sc.t0 = now();
                            SetEvent(shared);
                            WaitForSingleObject(sc.done_evt, INFINITE);        // C: runs exactly 6 GiB
                            se.dual_stop.store(1, std::memory_order_release); // E: finishes current epoch
                            RunResult re = sweep_finish(&se), rc = sweep_finish(&sc);
                            if (re.ok && rc.ok) {
                                double tspan = re.sec > rc.sec ? re.sec : rc.sec;
                                printf("  E: %8.2f GB/s (%6.2fs, %6.2f GB read) cores=%.3f\n", re.gbps, re.sec, double(re.bytes) / 1e9, re.cores);
                                printf("  C: %8.2f GB/s (%6.2fs, %6.2f GB read) cores=%.3f\n", rc.gbps, rc.sec, double(rc.bytes) / 1e9, rc.cores);
                                printf("  AGGREGATE: %.2f GB/s over %.2fs (E:+C: concurrent)\n",
                                       double(re.bytes + rc.bytes) / 1e9 / tspan, tspan);
                            } else printf("  dual FAILED (E:%d C:%d)\n", (int)re.ok, (int)rc.ok);
                        } else printf("  dual launch failed\n");
                        CloseHandle(shared);
                    } else printf("  (dual skipped: no E: results — run with 'e' or 'all')\n");
                }
                BOOL del = DeleteFileA(tpath);
                ULARGE_INTEGER fa{}; GetDiskFreeSpaceExA(tmp, &fa, nullptr, nullptr);
                printf("  test file deleted: %s; %s free after: %.1f GB\n", del ? "yes" : "NO", tmp, double(fa.QuadPart) / 1e9);
            }
        }
    }

    FILETIME fc, fx, k0, u0; GetProcessTimes(GetCurrentProcess(), &fc, &fx, &k0, &u0);
    auto ft_s = [](const FILETIME& ft) { return (double((u64(ft.dwHighDateTime) << 32) | ft.dwLowDateTime)) / 1e7; };
    printf("\ntotal bench CPU: %.1fs (kernel+user) — done.\n", ft_s(k0) + ft_s(u0));
    return 0;
}
