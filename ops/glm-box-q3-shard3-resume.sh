#!/usr/bin/env bash
set -Eeuo pipefail

readonly DIR="/var/lib/insignia/glm53-q3-k-xl/UD-Q3_K_XL"
readonly NAME="GLM-5.3-Flash-UNCENSORED-FP8-UD-Q3_K_XL-00003-of-00004.gguf"
readonly PART="$DIR/$NAME.partial"
readonly FINAL="$DIR/$NAME"
readonly URL="https://huggingface.co/AliceThirty/GLM-5.3-Flash-UNCENSORED-GGUF/resolve/main/UD-Q3_K_XL/$NAME"
readonly BYTES=48305374944
readonly SHA256="6817a6a3621a35c85b4d7f518402f6b803a35f58d9d4cdf8c76aaa8372e0af5f"

mkdir -p "$DIR"
if [[ -e "$FINAL" ]]; then
    [[ "$(stat -c %s "$FINAL")" == "$BYTES" ]]
    printf 'shard 3 already complete: %s\n' "$FINAL"
    exit 0
fi
[[ -f "$PART" ]] || { printf 'missing rescued partial: %s\n' "$PART" >&2; exit 2; }
(( $(stat -c %s "$PART") < BYTES )) || {
    printf 'partial is not smaller than expected shard size\n' >&2
    exit 3
}

curl --fail --location --continue-at - --output "$PART" \
    --retry 30 --retry-delay 2 --retry-all-errors "$URL"
[[ "$(stat -c %s "$PART")" == "$BYTES" ]] || {
    printf 'wrong completed size: expected %d, got %s\n' "$BYTES" "$(stat -c %s "$PART")" >&2
    exit 4
}
printf '%s  %s\n' "$SHA256" "$PART" | sha256sum --check --status || {
    printf 'shard 3 SHA-256 mismatch\n' >&2
    exit 5
}
mv -- "$PART" "$FINAL"
printf 'shard 3 complete and verified: %s\n' "$FINAL"
