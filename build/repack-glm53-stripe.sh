#!/usr/bin/env bash
# Build and fully verify the C:+E: expert overlay from route/miss weights.
# Usage: repack-glm53-stripe.sh LAYER_EXPERT_WEIGHT_FILE
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
weights="${1:?usage: repack-glm53-stripe.sh LAYER_EXPERT_WEIGHT_FILE}"
source_dir="${INSIGNIA_GLM53_STORE:-/var/lib/insignia/glm53-flash-text}"
source_index="${INSIGNIA_GLM53_INDEX:-/var/lib/insignia/glm53-flash-text.index}"
stripe_dir="${INSIGNIA_STRIPE_MOUNT:-/stripe}"
stripe_index="${INSIGNIA_GLM53_STRIPE_INDEX:-/var/lib/insignia/glm53-flash-text-striped.index}"
main_gbps="${INSIGNIA_STRIPE_MAIN_GBPS:-5.94}"
alt_gbps="${INSIGNIA_STRIPE_ALT_GBPS:-2.58}"

[[ -f "$weights" ]] || { echo "missing stripe weights: $weights" >&2; exit 1; }
bash "$repo/build/ensure-stripe.sh" >/dev/null

args=(
    "$source_dir" "$source_index" "$stripe_dir" "$stripe_index"
    --weights "$weights"
    --main-gbps "$main_gbps"
    --alt-gbps "$alt_gbps"
)
if [[ "${INSIGNIA_STRIPE_FORCE:-0}" == 1 ]]; then
    args+=(--force)
fi
if [[ -n "${INSIGNIA_STRIPE_UUID:-}" ]]; then
    args+=(--expected-uuid "$INSIGNIA_STRIPE_UUID")
fi
python "$repo/tools/stripe_repack.py" "${args[@]}"

verify=("$source_index" "$stripe_index" "$source_dir" "$stripe_dir")
if [[ -n "${INSIGNIA_STRIPE_UUID:-}" ]]; then
    verify+=(--expected-uuid "$INSIGNIA_STRIPE_UUID")
fi
# Full record parity and full shard hashes are deliberately the verifier's
# defaults.  Activation must not be based on a 16-record sample.
python "$repo/tools/stripe_verify.py" "${verify[@]}"
