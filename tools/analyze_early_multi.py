#!/usr/bin/env python3
"""Rank/cap frontier for INSIGNIA_GLM53_EARLY_MULTI_TRACE dumps."""

import argparse


def ordered_union(rows, width, cap):
    result = []
    for rank in range(width):
        for row in rows:
            expert = row[rank]
            if expert not in result:
                result.append(expert)
                if cap and len(result) == cap:
                    return result
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--cap", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="keep batches no wider than this (0 keeps all)")
    args = parser.parse_args()

    groups = []
    with open(args.trace, encoding="utf-8") as source:
        for line in source:
            fields = [int(value) for value in line.split()]
            if len(fields) < 6:
                continue
            batch, layer, tokens, recorded, predicted_count, actual_count = fields[:6]
            if args.max_tokens and tokens > args.max_tokens:
                continue
            payload = fields[6:]
            if len(payload) != 16 * tokens:
                raise RuntimeError(f"bad payload for batch {batch} layer {layer}")
            predicted, actual_rows = [], []
            for token in range(tokens):
                row = payload[16 * token:16 * (token + 1)]
                predicted.append(row[:8])
                actual_rows.append(row[8:])
            actual = ordered_union(actual_rows, 8, 0)
            full = ordered_union(predicted, 8, 0)
            if (len(set(full) & set(actual)), len(full), len(actual)) != (
                    recorded, predicted_count, actual_count):
                raise RuntimeError(f"bad union metrics for batch {batch} layer {layer}")
            groups.append((predicted, actual))

    if not groups:
        raise SystemExit("trace contains no multi-route rows")
    print(f"groups={len(groups)} cap={args.cap or 'none'}")
    print(" N  precision    recall  useful/group  predicted/group  actual/group")
    for width in range(1, 9):
        hits = predicted_total = actual_total = 0
        for predicted_rows, actual in groups:
            predicted = ordered_union(predicted_rows, width, args.cap)
            hits += len(set(predicted) & set(actual))
            predicted_total += len(predicted)
            actual_total += len(actual)
        print(f"{width:2d} {hits / predicted_total:10.3%} {hits / actual_total:9.3%} "
              f"{hits / len(groups):12.2f} {predicted_total / len(groups):16.2f} "
              f"{actual_total / len(groups):12.2f}")


if __name__ == "__main__":
    main()
