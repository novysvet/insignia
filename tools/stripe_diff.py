#!/usr/bin/env python3
"""Deprecated modulo-placement checker; use stripe_verify.py."""

import sys


sys.stderr.write(
    "stripe_diff.py is disabled: route-weighted placement is not expert-id modulo based.\n"
    "Use tools/stripe_verify.py ORIGINAL.index STRIPED.index SOURCE_DIR /stripe.\n"
)
raise SystemExit(64)
