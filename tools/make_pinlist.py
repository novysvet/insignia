#!/usr/bin/env python3
"""Build a static hot-expert pin list from a ROUTE_TRACE dump.

Input: the engine's routing trace ("token layer e0..e7 s0..s7" rows).
Output: one "layer expert hits" line per (layer, expert), sorted by hits
descending within each sparse layer, so the loader can take the first N
entries per layer for the host tier and the first M for the VRAM tier.
"""

import argparse
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("output")
    args = parser.parse_args()

    counts = Counter()
    with open(args.trace, encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 10 or fields[0] == "token":
                continue
            layer = int(fields[1])
            for field in fields[2:10]:
                expert = int(field)
                if 0 <= expert:
                    counts[(layer, expert)] += 1

    by_layer = {}
    for (layer, expert), hits in counts.items():
        by_layer.setdefault(layer, []).append((expert, hits))
    lines = []
    for layer in sorted(by_layer):
        for expert, hits in sorted(by_layer[layer], key=lambda item: -item[1]):
            lines.append(f"{layer} {expert} {hits}")
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"{len(lines)} entries over {len(by_layer)} layers -> {args.output}")


if __name__ == "__main__":
    main()
