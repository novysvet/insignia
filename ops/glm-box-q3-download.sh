#!/usr/bin/env bash
set -Eeuo pipefail

export LC_ALL=C

readonly REPO_ID="AliceThirty/GLM-5.3-Flash-UNCENSORED-GGUF"
readonly INCLUDE="UD-Q3_K_XL/*"
readonly DEST="${INSIGNIA_Q3_DIR:-/var/lib/insignia/glm53-q3-k-xl}"
readonly HF="${INSIGNIA_HF_BIN:-/root/.local/bin/hf}"
readonly LOG="${INSIGNIA_Q3_LOG:-/var/lib/insignia/q3-k-xl-download.log}"
readonly HOST_RESERVE_GIB="${INSIGNIA_Q3_HOST_RESERVE_GIB:-100}"
readonly TEMP_MARGIN_GIB="${INSIGNIA_Q3_TEMP_MARGIN_GIB:-32}"
readonly GUEST_RESERVE_GIB="${INSIGNIA_Q3_GUEST_RESERVE_GIB:-64}"
readonly GUARD_SECONDS="${INSIGNIA_Q3_GUARD_SECONDS:-30}"
readonly GIB=$((1024 * 1024 * 1024))

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

available_bytes() {
    df -B1 --output=avail "$1" | awk 'NR == 2 { print $1 }'
}

as_gib() {
    awk -v bytes="$1" 'BEGIN { printf "%.1f", bytes / 1073741824 }'
}

missing_bytes() {
    "$HF" download "$REPO_ID" \
        --include "$INCLUDE" \
        --local-dir "$DEST" \
        --max-workers 1 \
        --dry-run \
        --json |
    python3 -c '
import json, re, sys

units = {"": 1, "K": 10**3, "M": 10**6, "G": 10**9,
         "T": 10**12, "P": 10**15, "E": 10**18,
         "Ki": 2**10, "Mi": 2**20, "Gi": 2**30,
         "Ti": 2**40, "Pi": 2**50, "Ei": 2**60}

def parse_size(value):
    if value in (None, "", "-"):
        return 0
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTPE]?i?)(?:B)?", str(value))
    if not match:
        raise SystemExit(f"unrecognized hf size: {value!r}")
    return int(float(match.group(1)) * units[match.group(2)])

print(sum(parse_size(item.get("size")) for item in json.load(sys.stdin)))
'
}

preflight() {
    [[ -x "$HF" ]] || die "hf CLI not executable: $HF"
    command -v python3 >/dev/null || die "python3 is required for hf dry-run parsing"
    command -v flock >/dev/null || die "flock is required"
    [[ -d /mnt/c ]] || die "/mnt/c is unavailable; cannot measure the Windows backing volume"

    mkdir -p "$DEST"
    local missing host_free guest_free host_required guest_required
    missing="$(missing_bytes)"
    host_free="$(available_bytes /mnt/c)"
    guest_free="$(available_bytes "$DEST")"
    host_required=$((missing + (HOST_RESERVE_GIB + TEMP_MARGIN_GIB) * GIB))
    guest_required=$((missing + (GUEST_RESERVE_GIB + TEMP_MARGIN_GIB) * GIB))

    printf 'Q3_K_XL preflight: missing=%s GiB, C-free=%s GiB, ext4-free=%s GiB\n' \
        "$(as_gib "$missing")" "$(as_gib "$host_free")" "$(as_gib "$guest_free")"
    printf 'Required now: C >= %s GiB, ext4 >= %s GiB\n' \
        "$(as_gib "$host_required")" "$(as_gib "$guest_required")"

    ((host_free >= host_required)) ||
        die "Windows C: lacks download headroom; free space is the physical VHD limit"
    ((guest_free >= guest_required)) ||
        die "Arch ext4 lacks download headroom"
}

verify_complete() {
    local shards shard
    shopt -s nullglob
    shards=("$DEST"/UD-Q3_K_XL/*-0000?-of-00004.gguf)
    ((${#shards[@]} == 4)) ||
        die "expected four Q3_K_XL GGUF shards, found ${#shards[@]}"
    for shard in "${shards[@]}"; do
        [[ -s "$shard" ]] || die "empty shard: $shard"
    done
    printf 'repo=%s\ncompleted=%s\nshards=4\n' \
        "$REPO_ID" "$(date --iso-8601=seconds)" >"$DEST/.insignia-complete"
}

case "${1:-download}" in
    --preflight|preflight)
        preflight
        exit 0
        ;;
    download)
        ;;
    *)
        die "usage: $0 [download|--preflight]"
        ;;
esac

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

mkdir -p /var/lib/insignia/.locks "$DEST"
exec 9>/var/lib/insignia/.locks/q3-k-xl-download.lock
flock -n 9 || die "another Q3_K_XL download owns the lock"
preflight

printf 'Starting serial Q3_K_XL resume at %s\n' "$(date --iso-8601=seconds)"
"$HF" download "$REPO_ID" \
    --include "$INCLUDE" \
    --local-dir "$DEST" \
    --max-workers 1 &
download_pid=$!

stop_download() {
    kill -TERM "$download_pid" 2>/dev/null || true
    wait "$download_pid" 2>/dev/null || true
}
trap 'stop_download; exit 130' INT TERM HUP

while kill -0 "$download_pid" 2>/dev/null; do
    sleep "$GUARD_SECONDS"
    host_free="$(available_bytes /mnt/c)"
    guest_free="$(available_bytes "$DEST")"
    if ((host_free < HOST_RESERVE_GIB * GIB)); then
        printf 'ABORT: Windows C: crossed the %s GiB reserve\n' "$HOST_RESERVE_GIB" >&2
        stop_download
        exit 75
    fi
    if ((guest_free < GUEST_RESERVE_GIB * GIB)); then
        printf 'ABORT: Arch ext4 crossed the %s GiB reserve\n' "$GUEST_RESERVE_GIB" >&2
        stop_download
        exit 75
    fi
done

set +e
wait "$download_pid"
download_status=$?
set -e
trap - INT TERM HUP
((download_status == 0)) || exit "$download_status"

verify_complete
printf 'Q3_K_XL download complete: %s\n' "$DEST"
