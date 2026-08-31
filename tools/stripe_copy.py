#!/usr/bin/env python3
"""Deprecated unsafe whole-shard copier; use stripe_repack.py."""

import sys


sys.stderr.write(
    "stripe_copy.py is disabled: it copied whole shards into the C:-backed "
    "root VHDX and cannot provide balanced per-layer dual-SSD traffic.\n"
    "Use tools/stripe_repack.py after build/remount-stripe.bat verifies /stripe.\n"
)
raise SystemExit(64)
