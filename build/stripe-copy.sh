#!/usr/bin/env bash
set -euo pipefail
echo "build/stripe-copy.sh is disabled: its destination was the C:-backed root VHDX." >&2
echo "Mount E: with build/remount-stripe.bat, then use tools/stripe_repack.py." >&2
exit 64
