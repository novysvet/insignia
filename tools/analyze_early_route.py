#!/usr/bin/env python3
"""Summarize INSIGNIA_GLM53_EARLY_ROUTE_TRACE prediction quality."""

import argparse
import collections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    args = parser.parse_args()

    rows = []
    layers = collections.defaultdict(lambda: [0, 0])
    with open(args.trace, encoding="utf-8") as source:
        for line in source:
            fields = [int(value) for value in line.split()]
            if len(fields) != 19:
                continue
            token, layer, recorded = fields[:3]
            predicted, actual = fields[3:11], set(fields[11:19])
            hits = [expert in actual for expert in predicted]
            if sum(hits) != recorded:
                raise RuntimeError(f"bad overlap at token {token} layer {layer}")
            rows.append(hits)
            layers[layer][0] += recorded
            layers[layer][1] += 8

    if not rows:
        raise SystemExit("trace contains no early-route rows")
    print(f"rows={len(rows)} full-top8 recall={sum(map(sum, rows)) / (8 * len(rows)):.3%}")
    print(" N  precision  coverage  useful/row  wrong/row")
    cumulative = 0
    for count in range(1, 9):
        cumulative += sum(row[count - 1] for row in rows)
        useful = cumulative / len(rows)
        print(f"{count:2d} {useful / count:10.3%} {useful / 8:9.3%} "
              f"{useful:11.3f} {count - useful:10.3f}")

    print("\nlayer recall")
    for layer in sorted(layers):
        hits, total = layers[layer]
        print(f"L{layer:02d} {hits / total:7.3%} ({hits}/{total})")


if __name__ == "__main__":
    main()
