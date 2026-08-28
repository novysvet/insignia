#!/usr/bin/env python3
"""Build the cross-layer co-activation table (CCT) for expert prefetching.

Reads a routing trace ("token layer e0..e7 s0..s7" per line, produced by
INSIGNIA_GLM53_ROUTE_TRACE — note every prefill row shares token index 0, so
tables are effectively decode-routing statistics), counts adjacent-sparse-layer
co-activations, and writes the binary table consumed by Runner::load_cct
(src/glm53_generate.cu): for each (L, L+1) sparse pair, the top-8 experts of
L+1 most co-activated with each expert of L.

Format: b"CCT0", u32 {layers, experts, topk}, then one experts*topk uint16
successor table per adjacent sparse pair in ascending layer order — no tags,
the loader reads positionally while walking its own is_sparse_ sequence.
Pairs absent from the trace are zero-filled to keep that alignment. Sparse
layers follow the GLM-5.3-Flash contract: 45 layers, 0-2 dense, 3+ sparse.
"""
import struct
import sys
from collections import defaultdict

import numpy as np

trace = sys.argv[1]
out = sys.argv[2]
topk = 8
LAYERS = 45
FIRST_SPARSE = 3
N_EXPERTS = 288

by_token = defaultdict(dict)  # token index -> layer -> [8 experts]
with open(trace) as f:
    for line in f:
        parts = line.split()
        if len(parts) < 11:
            continue
        token, layer = int(parts[0]), int(parts[1])
        experts = [int(x) for x in parts[2:10]]
        by_token[token][layer] = experts

tables = []
for a in range(FIRST_SPARSE, LAYERS - 1):
    co = np.zeros((N_EXPERTS, N_EXPERTS), dtype=np.uint32)
    for routing in by_token.values():
        if a in routing and a + 1 in routing:
            for e in routing[a]:
                co[e, routing[a + 1]] += 1
    # Top-8 per expert by count, ties broken by smaller id.
    order = np.argsort(-co, axis=1, kind="stable")[:, :topk].astype(np.uint16)
    tables.append(order)

with open(out, "wb") as f:
    f.write(b"CCT0")
    f.write(struct.pack("<III", LAYERS, N_EXPERTS, topk))
    for order in tables:
        f.write(order.astype("<u2").tobytes())
print(f"wrote {len(tables)} pair tables (layers {FIRST_SPARSE}..{LAYERS - 2}), "
      f"{N_EXPERTS} experts, top{topk} -> {out}")
