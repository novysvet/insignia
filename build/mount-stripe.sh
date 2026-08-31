#!/usr/bin/env bash
# Mount and verify the E:-hosted stripe VHDX by filesystem identity.
# The Windows side must attach E:\stripe\stripe.vhdx before this runs.
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mount_point="${INSIGNIA_STRIPE_MOUNT:-/stripe}"
label="${INSIGNIA_STRIPE_LABEL:-stripe}"
expected_uuid="${INSIGNIA_STRIPE_UUID:-${1:-}}"
source_root="${INSIGNIA_GLM53_STORE:-/var/lib/insignia/glm53-flash-text}"

[[ "$mount_point" == /* && "$mount_point" != / ]] || {
    echo "unsafe stripe mount point: $mount_point" >&2
    exit 1
}
[[ ! -L "$mount_point" ]] || { echo "stripe mount point is a symlink: $mount_point" >&2; exit 1; }
[[ ! -e "$mount_point" || -d "$mount_point" ]] || {
    echo "stripe mount point is not a directory: $mount_point" >&2
    exit 1
}

if [[ -n "$expected_uuid" ]]; then
    device="$(blkid -U "$expected_uuid" || true)"
    [[ -n "$device" ]] || { echo "stripe UUID not attached: $expected_uuid" >&2; exit 1; }
    actual_label="$(blkid -s LABEL -o value "$device" || true)"
    [[ "$actual_label" == "$label" ]] || {
        echo "stripe UUID $expected_uuid has label '$actual_label', expected '$label'" >&2
        exit 1
    }
else
    mapfile -t label_devices < <(blkid -t "LABEL=$label" -o device 2>/dev/null || true)
    [[ "${#label_devices[@]}" -eq 1 ]] || {
        echo "stripe label '$label' resolves to ${#label_devices[@]} devices; expected exactly one" >&2
        exit 1
    }
    device="${label_devices[0]}"
fi

mkdir -p -- "$mount_point"
mounted_target="$(findmnt -rn -M "$mount_point" -o TARGET 2>/dev/null || true)"
if [[ "$mounted_target" != "$mount_point" ]]; then
    mount -t ext4 -- "$device" "$mount_point"
fi

check=(python "$repo/tools/stripe_mount.py" "$source_root" "$mount_point"
       --expected-label "$label" --expected-fs ext4)
if [[ -n "$expected_uuid" ]]; then
    check+=(--expected-uuid "$expected_uuid")
fi
"${check[@]}"
findmnt -rn -M "$mount_point" -o TARGET,SOURCE,FSTYPE,OPTIONS
