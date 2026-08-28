#!/usr/bin/env bash
# insignia-nvme-probe.sh -- NVMe (virtio-blk) read-ceiling probe for Arch WSL2 guests.
# Usage: [sudo] ./insignia-nvme-probe.sh [--tune]
#   --tune   root only: scheduler=none, max_sectors_kb=max_hw_sectors_kb (prints before/after)
# Env: TESTDIR(/var/tmp) MODELDIR(/var/lib/insignia) FILESZ_G(32) RUNTIME(8s/job) RAMP(2s) PY_DUR(6s)
set -euo pipefail

TESTDIR=${TESTDIR:-/var/tmp}
MODELDIR=${MODELDIR:-/var/lib/insignia}
FILESZ_G=${FILESZ_G:-32}
RUNTIME=${RUNTIME:-8}
RAMP=${RAMP:-2}
PY_DUR=${PY_DUR:-6}
TUNE=0; for a in "$@"; do [[ $a == --tune ]] && TUNE=1; done

die() { echo "FATAL: $*" >&2; exit 1; }

echo "== insignia NVMe read-ceiling probe =="
uname -r
grep -qi microsoft /proc/version || echo "WARN: kernel does not look like WSL2"
command -v python3 >/dev/null 2>&1 || die "python3 required"

check_fs() {
  local line src fs tgt
  line=$(findmnt -T "$1" -no SOURCE,FSTYPE,TARGET) || die "findmnt failed for $1"
  read -r src fs tgt <<<"$line"
  [[ $fs == 9p || $fs == drvfs || $fs == tmpfs ]] && die "$1: fstype=$fs -- refusing"
  case $tgt in /mnt/*) die "$1 is on $tgt (drvfs) -- refusing" ;; esac
  echo "  $1 -> $tgt ($fs on $src)"
}
echo "-- mounts:"
check_fs "$TESTDIR"
[[ -d $MODELDIR ]] && check_fs "$MODELDIR"

PROBE=$MODELDIR; [[ -d $PROBE ]] || PROBE=$TESTDIR
SRC=$(findmnt -T "$PROBE" -no SOURCE)
DEV=$(lsblk -npo PKNAME "$SRC" 2>/dev/null | head -n1); DEV=${DEV:-$SRC}; DEV=$(basename "$DEV")
[[ -d /sys/block/$DEV ]] || die "cannot map $SRC -> /sys/block"
echo "-- device under test: /dev/$DEV (from $SRC on $PROBE)"

as_root() { if (( EUID == 0 )); then "$@"; elif command -v sudo >/dev/null 2>&1; then sudo "$@"; else return 1; fi; }
for p in liburing fio; do
  pacman -Qi "$p" >/dev/null 2>&1 || { echo "-- installing $p"; as_root pacman -S --noconfirm "$p" || echo "WARN: $p install failed"; }
done
HAVE_FIO=0; command -v fio >/dev/null 2>&1 && HAVE_FIO=1
(( HAVE_FIO )) || echo "-- fio unavailable: python3 pread fallback will run"

show_q() {
  echo "-- /sys/block/$DEV/queue ($1):"
  for f in scheduler rotational nr_requests max_sectors_kb max_hw_sectors_kb max_segments; do
    printf '     %-17s %s\n' "$f" "$(cat /sys/block/$DEV/queue/$f 2>/dev/null || echo n/a)"
  done
}
Q=/sys/block/$DEV/queue
show_q before
if (( TUNE )); then
  if (( EUID == 0 )); then
    if grep -qw none "$Q/scheduler" 2>/dev/null; then echo none > "$Q/scheduler" 2>/dev/null || true; fi
    hw=$(cat "$Q/max_hw_sectors_kb" 2>/dev/null || echo 0)
    (( hw > 0 )) && echo "$hw" > "$Q/max_sectors_kb" 2>/dev/null || true
    show_q after
  else
    echo "--tune given but not root: printing only"
  fi
fi

AVAIL_G=$(df -BG --output=avail "$TESTDIR" | tail -n1 | tr -dc '0-9')
USE_G=$(( FILESZ_G < AVAIL_G - 4 ? FILESZ_G : AVAIL_G - 4 ))
(( USE_G >= 4 )) || die "insufficient space under $TESTDIR (avail ${AVAIL_G}G)"
FILESZ_M=$(( USE_G * 1024 ))
F=$TESTDIR/insignia-nvme-probe.bin
CSV=$(mktemp)
cleanup() { rm -f -- "$F" "$CSV" 2>/dev/null || true; }
trap cleanup EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

[[ -e $F ]] && die "$F exists; remove it first"
echo "-- creating + write-filling ${USE_G} GiB at $F"
fallocate -l "${USE_G}G" "$F"
if (( HAVE_FIO )); then
  fio --name=fill --filename="$F" --rw=write --bs=4M --direct=1 --ioengine=libaio \
      --iodepth=8 --size="${USE_G}G" --end_fsync=1 --eta=never >/dev/null
else
  dd if=/dev/zero of="$F" bs=1M count=$((USE_G * 1024)) conv=notrunc status=none
fi
sync

fio_cfg() {
  local eng=$1 bs=$2 qd=$3 nj=$4 out
  local per=$(( FILESZ_M / nj ))
  out=$(fio --name=j --filename="$F" --rw=read --direct=1 --ioengine="$eng" \
        --bs="$bs" --iodepth="$qd" --numjobs="$nj" --size="${per}M" \
        --offset_increment="${per}M" --time_based=1 --runtime="$RUNTIME" \
        --ramp_time="$RAMP" --group_reporting --eta=never --gtod_reduce=1 \
        --output-format=json 2>/dev/null |
    python3 -c 'import json,sys
d = json.load(sys.stdin)
print(round(sum(j["read"]["bw_bytes"] for j in d["jobs"]) / 1e9, 2))') || out=ERR
  echo "${out:-ERR}"
}

if (( HAVE_FIO )); then
  ENGS=libaio
  if fio --enghelp=io_uring >/dev/null 2>&1; then ENGS="libaio io_uring"; else echo "-- fio lacks io_uring engine"; fi
  est=$(( $(wc -w <<<"$ENGS") * 27 * (RUNTIME + RAMP + 2) ))
  echo "== fio matrix: seq read O_DIRECT (est ~$((est / 60)) min) =="
  printf '%-9s %5s %4s %5s %8s\n' engine bs qd jobs GBps
  for eng in $ENGS; do for bs in 1M 4M 16M; do for qd in 8 16 32; do for nj in 1 4 8; do
    printf '%-9s %5s %4d %5d %8s\n' "$eng" "$bs" "$qd" "$nj" "$(fio_cfg "$eng" "$bs" "$qd" "$nj")" | tee -a "$CSV"
  done; done; done; done
  echo "-- top 5:"; sort -k5 -gr <(grep -v '^engine' "$CSV") | head -n5
  if (( TUNE )) && (( EUID == 0 )); then
    echo "post-tune sentinel (libaio 4M qd16 jobs8): $(fio_cfg libaio 4M 16 8) GB/s"
  fi
fi

echo "== python probe: N threads x 13.5 MiB O_DIRECT reads =="
python3 - "$F" "$PY_DUR" <<'PY'
import os, sys, time, random, threading, mmap, gc
path, dur = sys.argv[1], float(sys.argv[2])
CHUNK = 13*1024*1024 + 512*1024
fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
fsize = os.fstat(fd).st_size
n = max(1, (fsize - CHUNK) // 4096)
r = random.Random(0xC0FFEE)
OFFS = [r.randrange(n) * 4096 for _ in range(8192)]
del r
gc.disable()
modes = ["preadv"]
try: os.pread(fd, CHUNK, OFFS[0]); modes.insert(0, "pread")
except OSError: print("note: os.pread O_DIRECT -> EINVAL; pread variant skipped")
def bench(nthr, mode):
    start, stop = threading.Event(), threading.Event()
    cnt = [0] * nthr
    def w(i):
        buf = mmap.mmap(-1, CHUNK)
        k = i
        start.wait()
        while not stop.is_set():
            off = OFFS[(k * nthr + i) & 8191]; k += 1
            try:
                if mode == "pread": cnt[i] += len(os.pread(fd, CHUNK, off))
                else: cnt[i] += os.preadv(fd, [buf], off)
            except OSError: break
    th = [threading.Thread(target=w, args=(i,), daemon=True) for i in range(nthr)]
    for t in th: t.start()
    t0 = time.perf_counter(); start.set(); time.sleep(dur); stop.set()
    for t in th: t.join()
    return sum(cnt) / (time.perf_counter() - t0) / 1e9
for N in (4, 8, 16, 24, 32):
    print(f"threads={N:>2}  preadv {bench(N, 'preadv'):6.2f} GB/s")
PY
echo "== done; test file removed by trap =="
