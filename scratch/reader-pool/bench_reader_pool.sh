#!/usr/bin/env bash
# =============================================================================
# scratch/reader-pool/bench_reader_pool.sh — CPU-only reader-pool microbench.
# Runs INSIDE the Arch WSL distro on glm-box (wsl -d Arch). Read-only against
# the model store: never writes, truncates, or advises on the store files.
#
# Answers three questions before any engine change:
#   Q1  pread threads vs raw io_uring at the ENGINE's exact geometry
#       (4 KiB-aligned whole-record reads, ~12.76 MiB packed / 13.5 MiB
#       compact) — threads {1..8} vs depth {4..64}, fixed files, no SQPOLL.
#   Q2  request-size sensitivity (fio bs 4m/8m/13m/26m at qd 4/8) — is
#       fusing adjacent packed records worth anything?
#   Q3  is the 4-thread optimum still right now that the packed sidecar
#       shrank reads by ~5.5% and added a CPU expand stage?
#
# Usage:
#   ./bench_reader_pool.sh [packed-sidecar-path]
# Defaults the sidecar from $INSIGNIA_GLM53_PACKED_EXPERTS, then
# /var/lib/insignia/glm53-packed-experts*, else falls back to the largest
# shard of /var/lib/insignia/glm53-flash-text with synthetic 13.5 MiB records.
#
# Outputs: TSV rows to stdout; details under ./reader-pool-bench/.
# Wall time: ~8-10 minutes.
# =============================================================================
set -euo pipefail

OUT=reader-pool-bench
mkdir -p "$OUT"

SIDECAR="${1:-${INSIGNIA_GLM53_PACKED_EXPERTS:-}}"
if [[ -z "$SIDECAR" ]]; then
    for candidate in /var/lib/insignia/glm53-packed-experts* /var/lib/insignia/*packed*; do
        [[ -f "$candidate" ]] && SIDECAR="$candidate" && break
    done
fi
STORE_DIR=/var/lib/insignia/glm53-flash-text

echo "== environment =="
uname -r | tee "$OUT/env.txt"          # WSL kernel: io_uring needs 5.1+; modern WSL2 ships 6.x
fio --version | tee -a "$OUT/env.txt" || { echo "fio missing: pacman -S fio"; exit 1; }
nproc | tee -a "$OUT/env.txt"
grep -E "modelName|size" /sys/block/*/queue 2>/dev/null | head -4 || true

if [[ -n "$SIDECAR" && -f "$SIDECAR" ]]; then
    TARGET="$SIDECAR"; TARGET_KIND=sidecar
else
    TARGET=$(find "$STORE_DIR" -type f -printf '%s %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    TARGET_KIND=compact-shard
fi
echo "target: $TARGET ($TARGET_KIND)" | tee -a "$OUT/env.txt"

# --- Q1/Q3: exact-geometry harness (real sidecar record offsets) ------------
cat > "$OUT/pread_uring_sweep.c" <<'EOF'
/* pread_uring_sweep.c — engine-geometry expert-record read sweep.
 * gcc -O2 -D_GNU_SOURCE -o pread_uring_sweep pread_uring_sweep.c
 * Engines: pread (N threads, one whole-record blocking pread each, exactly
 * like ExpertStager) and uring (one thread, raw-syscall io_uring, fixed
 * files, no SQPOLL, depth D). Offsets come from the packed sidecar index
 * (real records, 4 KiB aligned) or are synthesized 4 KiB aligned. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <linux/io_uring.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* --- sidecar index parsing (mirrors open_packed_experts) ----------------- */
struct rec { uint64_t off; uint32_t stored, padded; };
static struct rec *recs; static size_t nrecs;

static void load_sidecar(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open sidecar"); exit(1); }
    struct { char magic[8]; uint32_t version, layers, experts, records;
             uint64_t index_offset, data_offset, file_bytes, source_bytes, stored_bytes; } h;
    if (pread(fd, &h, sizeof h, 0) != sizeof h || memcmp(h.magic, "IG53XPK1", 8)) {
        fprintf(stderr, "not a packed sidecar\n"); exit(1);
    }
    struct { uint64_t off; uint32_t stored, padded; } *e = calloc(h.records, sizeof *e);
    if (pread(fd, e, h.records * sizeof *e, h.index_offset) != (ssize_t)(h.records * sizeof *e)) {
        perror("index"); exit(1);
    }
    recs = calloc(h.records, sizeof(struct rec)); nrecs = 0;
    for (uint32_t i = 0; i < h.records; i++)
        if (e[i].off) recs[nrecs++] = (struct rec){e[i].off, e[i].stored, e[i].padded};
    free(e); close(fd);
    fprintf(stderr, "sidecar: %u records\n", h.records);
}

/* --- work list: seeded shuffle so every rep covers the same records ------ */
static uint64_t *order; static size_t norder;
static void build_order(size_t count, unsigned seed) {
    order = malloc(count * sizeof *order); norder = count;
    srand(seed);
    for (size_t i = 0; i < count; i++) order[i] = ((uint64_t)rand() << 32 | rand()) % nrecs;
}

/* --- pread engine: N threads, whole-record blocking reads ---------------- */
struct targ { int fd; size_t next; size_t end; uint64_t bytes; size_t bs; };
static void *pread_thread(void *arg) {
    struct targ *t = arg;
    for (;;) {
        size_t i = __atomic_fetch_add(&t->next, 1, __ATOMIC_RELAXED);
        if (i >= t->end) break;
        struct rec r = recs[order[i % norder]];
        uint32_t len = r.padded ? r.padded : (uint32_t)t->bs;
        uint64_t off = r.padded ? r.off : (order[i % norder] % 1024) * t->bs; /* synthetic */
        char *buf = aligned_alloc(4096, len);
        uint64_t done = 0;
        while (done < len) {
            ssize_t n = pread(t->fd, buf + done, len - done, (off_t)(off + done));
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) { perror("pread"); exit(1); }
            done += (uint64_t)n;
        }
        t->bytes += len;
        free(buf);
    }
    return NULL;
}

/* --- uring engine: one thread, fixed files, depth D, SQPOLL off ---------- */
static int ring_fd;
static unsigned *sq_head, *sq_tail, *sq_array, *sq_mask;
static unsigned *cq_head, *cq_tail, *cq_mask;
static struct io_uring_sqe *sqes;
static struct io_uring_cqe *cqes;
static char **slotbuf; static uint32_t slotlen;

static void ring_setup(int fd, int depth) {
    struct io_uring_params p; memset(&p, 0, sizeof p);
    ring_fd = (int)syscall(SYS_io_uring_setup, depth, &p);
    if (ring_fd < 0) { fprintf(stderr, "io_uring_setup: %s\n", strerror(errno)); exit(3); }
    unsigned char *sq = mmap(0, p.sq_off.array + p.sq_entries * sizeof(unsigned),
                             PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_fd, IORING_OFF_SQ_RING);
    unsigned char *cq = mmap(0, p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe),
                             PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_fd, IORING_OFF_CQ_RING);
    sqes = mmap(0, p.sq_entries * sizeof(struct io_uring_sqe),
                PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_fd, IORING_OFF_SQES);
    if (sq == MAP_FAILED || cq == MAP_FAILED || sqes == MAP_FAILED) { perror("mmap"); exit(3); }
    sq_mask = (unsigned *)(sq + p.sq_off.ring_mask); sq_head = (unsigned *)(sq + p.sq_off.head);
    sq_tail = (unsigned *)(sq + p.sq_off.tail); sq_array = (unsigned *)(sq + p.sq_off.array);
    cq_mask = (unsigned *)(cq + p.cq_off.ring_mask); cq_head = (unsigned *)(cq + p.cq_off.head);
    cq_tail = (unsigned *)(cq + p.cq_off.tail);
    cqes = (struct io_uring_cqe *)(cq + p.cq_off.cqes);
    if (syscall(SYS_io_uring_register, ring_fd, IORING_REGISTER_FILES, &fd, 1)) {
        fprintf(stderr, "register files failed (%s); plain fd\n", strerror(errno)); exit(3);
    }
    slotbuf = calloc(depth, sizeof *slotbuf);
    for (int i = 0; i < depth; i++) if (posix_memalign((void **)&slotbuf[i], 4096, slotlen)) exit(3);
}

static double run_uring(int fd, int depth, size_t count, uint64_t *bytes_out) {
    ring_setup(fd, depth);
    size_t issued = 0, done = 0; uint64_t bytes = 0;
    double t0 = now();
    while (done < count) {
        int submitted = 0;
        while (issued < count && submitted < depth) {
            unsigned tail = *sq_tail;
            if (tail - __atomic_load_n(sq_head, __ATOMIC_ACQUIRE) == *sq_mask + 1) break;
            unsigned idx = tail & *sq_mask;
            struct rec r = recs[order[issued % norder]];
            uint32_t len = r.padded ? r.padded : slotlen;
            uint64_t off = r.padded ? r.off : (order[issued % norder] % 1024) * slotlen;
            struct io_uring_sqe *s = &sqes[idx];
            memset(s, 0, sizeof *s);
            s->opcode = IORING_OP_READ; s->flags = IOSQE_FIXED_FILE; s->fd = 0;
            s->addr = (uint64_t)slotbuf[submitted]; s->len = len; s->off = off;
            s->user_data = issued++;
            sq_array[idx] = idx;
            *sq_tail = tail + 1;
            submitted++; bytes += len;
        }
        if (submitted) syscall(SYS_io_uring_enter, ring_fd, submitted, 0, 0, NULL);
        syscall(SYS_io_uring_enter, ring_fd, 0, 1, IORING_ENTER_GETEVENTS, NULL);
        for (;;) {
            unsigned head = *cq_head, tail = __atomic_load_n(cq_tail, __ATOMIC_ACQUIRE);
            if (head == tail) break;
            struct io_uring_cqe *c = &cqes[head & *cq_mask];
            if (c->res < 0) { fprintf(stderr, "cqe res %d\n", c->res); exit(3); }
            done++;
            __atomic_store_n(cq_head, head + 1, __ATOMIC_RELEASE);
        }
    }
    double dt = now() - t0;
    *bytes_out = bytes;
    for (int i = 0; i < depth; i++) free(slotbuf[i]);
    close(ring_fd);
    return dt;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s file [--engine pread|uring] [--threads N] "
            "[--depth N] [--records N] [--bs BYTES] [--reps R]\n", argv[0]); return 1; }
    const char *file = argv[1];
    const char *engine = "pread"; int threads = 4, depth = 8, reps = 3;
    size_t records = 640; uint32_t bs = 14156736; /* 13.5 MiB compact record */
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--engine")) engine = argv[++i];
        else if (!strcmp(argv[i], "--threads")) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--depth")) depth = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--records")) records = (size_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "--bs")) bs = (uint32_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "--reps")) reps = atoi(argv[++i]);
    }
    int fd = open(file, O_RDONLY | O_CLOEXEC | O_DIRECT);
    if (fd < 0) { perror("open O_DIRECT"); return 1; }
    /* sidecar offsets if the magic matches, else synthetic aligned offsets.
     * Sniff on a buffered fd: an 8-byte pread on an O_DIRECT fd is EINVAL. */
    int probe = open(file, O_RDONLY);
    char magic[8] = {0};
    int is_sidecar = probe >= 0 && pread(probe, magic, 8, 0) == 8 && !memcmp(magic, "IG53XPK1", 8);
    if (probe >= 0) close(probe);
    if (is_sidecar) load_sidecar(file);
    else {
        struct stat st; fstat(fd, &st);
        nrecs = (size_t)(st.st_size / bs); if (nrecs < 16) nrecs = 16;
        recs = calloc(nrecs, sizeof *recs);
        for (size_t i = 0; i < nrecs; i++) recs[i] = (struct rec){i * bs, bs, bs};
    }
    double best = 1e9; double med[8] = {0}; int nr = 0;
    for (int rep = 0; rep < reps; rep++) {
        build_order(records, 1234u + rep);
        uint64_t bytes = 0; double dt;
        if (!strcmp(engine, "pread")) {
            struct targ t = { fd, 0, records, 0, bs };
            pthread_t th[64];
            double t0 = now();
            for (int i = 0; i < threads; i++) pthread_create(&th[i], NULL, pread_thread, &t);
            for (int i = 0; i < threads; i++) pthread_join(th[i], NULL);
            dt = now() - t0; bytes = t.bytes;
        } else {
            slotlen = bs; dt = run_uring(fd, depth, records, &bytes);
        }
        med[nr++] = bytes / dt / 1e9;
        if (dt < best) best = dt;
        (void)best;
    }
    for (int i = 1; i < nr; i++) { double k = med[i]; int j = i - 1;
        while (j >= 0 && med[j] > k) { med[j + 1] = med[j]; j--; } med[j + 1] = k; }
    printf("%s\t%s\t%d\t%d\t%.2f\n", engine, file, threads, depth, med[nr / 2]);
    return 0;
}
EOF

gcc -O2 -pthread -o "$OUT/pread_uring_sweep" "$OUT/pread_uring_sweep.c"

echo; echo "== Q1/Q3: engine geometry (median of 3, GB/s decimal) =="
printf "engine\ttarget\tthreads\tdepth\tgb_s\n"
for t in 1 2 3 4 5 6 8; do
    "$OUT/pread_uring_sweep" "$TARGET" --engine pread --threads $t --records 640
done
for d in 4 8 16 32 64; do
    "$OUT/pread_uring_sweep" "$TARGET" --engine uring --depth $d --records 640
done | tee "$OUT/q1.tsv"

# --- Q2: fio sweep (broad matrix, request size & depth sensitivity) --------
echo; echo "== Q2: fio sweep =="
printf "engine\tbs\tiodepth\tnumjobs\tgb_s\n" | tee "$OUT/q2.tsv"
run_fio() { # engine bs qd jobs
    local out
    out=$(fio --name=s --filename="$TARGET" --rw=randread --bs="$2" --direct=1 \
        --ioengine="$1" --iodepth="$3" --numjobs="$4" --size=16G --time_based \
        --runtime=8 --group_reporting --readonly --offset_align=4096 \
        --output-format=json 2>/dev/null)
    printf "%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" \
        "$(echo "$out" | grep -o '"read":.*' | grep -o '"bw_bytes": [0-9]*' | head -1 | grep -o '[0-9]*' | awk '{printf "%.2f", $1/1e9}')"
}
for engine in psync io_uring; do
    for qd in 1 2 4 8 16 32; do
        if [[ "$engine" == psync ]]; then jobs=$qd; dq=1; else jobs=1; dq=$qd; fi
        run_fio "$engine" 13m "$dq" "$jobs" | tee -a "$OUT/q2.tsv"
    done
done
for bs in 4m 8m 26m; do
    for qd in 4 8; do run_fio io_uring "$bs" $qd 1 | tee -a "$OUT/q2.tsv"; done
done

echo; echo "done. medians in $OUT/{q1,q2}.tsv — compare pread@4t vs uring@d8 first;"
echo "then bs=13m vs 26m (fusion question), then the full thread/depth curves."
