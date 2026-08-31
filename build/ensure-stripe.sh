#!/usr/bin/env bash
# Compatibility entry point: all validation now lives in mount-stripe.sh.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$repo/build/mount-stripe.sh" "$@"
