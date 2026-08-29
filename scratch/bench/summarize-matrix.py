#!/usr/bin/env python3
"""Aggregate a bench-matrix run into one CSV: per-cell medians + IQR, parity
and acceptance-match status vs the baseline cell.

Usage: summarize-matrix.py /var/lib/insignia/bench-results/<date>-matrix
Reads (per cell dir):  rep*/results.json (tools/benchmark_math.py output),
parity/VERDICT + p*.diff, DONE/SKIP/FAIL markers, and greps the engine's
"NVFP4 cache ... N slots" line out of the per-case logs for tier sanity.
Protocol: N repeats per cell; report median + IQR (WSL timings swing ~2x,
single readings are meaningless). Determinism gate: every cell's DFlash2
greedy IDs must equal the baseline's; strict cells must additionally show
identical rounds / verify_k / accepted histograms (same k, same structure).
"""

import csv
import json
import pathlib
import re
import statistics
import sys


SLOTS_RE = re.compile(r"NVFP4 cache \d+/\d+ hits \([^;]*; (\d+) slots\)")


def rep_rows(cell: pathlib.Path):
    reps = {}
    for rep_dir in sorted(cell.glob("rep*")):
        try:
            rows = json.loads((rep_dir / "results.json").read_text())
        except (OSError, ValueError):
            continue
        if len(rows) == 4 and all(row.get("parity") for row in rows):
            reps[rep_dir.name] = rows
    return reps


def rep_metrics(rows):
    dflash_ms = [row["dflash"]["decode_ms_per_token"] for row in rows]
    scalar_ms = [row["scalar"]["decode_ms_per_token"] for row in rows]
    prefill = [row["dflash"]["prefill_tokens_per_second"] for row in rows]
    accepted = [row["dflash"]["accepted_per_round"] for row in rows]
    return {
        "dflash_ms": statistics.median(dflash_ms),
        "scalar_ms": statistics.median(scalar_ms),
        "speedup": statistics.median(scalar_ms) / statistics.median(dflash_ms),
        "prefill_tok_s": statistics.median(prefill),
        "accepted_per_round": statistics.median(accepted),
    }


def iqr(values):
    if len(values) < 2:
        return 0.0
    return statistics.quantiles(values, n=4)[2] - statistics.quantiles(values, n=4)[0]


def case_key(row):
    return f"{row['dataset']}#{row['index']}"


def cell_parity(cell: pathlib.Path):
    verdict = cell / "parity" / "VERDICT"
    if not verdict.is_file():
        return "n/a"
    text = verdict.read_text()
    if "DIVERGED" in text or "FAIL" in text:
        return "FAIL"
    if "top10+ids-identical" in text and "digit-identical" in text:
        return "pass"
    return "partial"


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    cells = []
    for cell in sorted(root.iterdir()):
        if cell.is_dir() and (cell.glob("rep*") or (cell / "DONE").exists()
                              or (cell / "SKIP").exists() or (cell / "FAIL").exists()):
            if cell.name in ("parity-base",):
                continue
            cells.append(cell)

    baseline = None
    for cell in cells:
        if cell.name == "baseline":
            baseline = cell
            break

    base_ids, base_struct = {}, {}
    if baseline is not None:
        for rep_name, rows in rep_rows(baseline).items():
            for row in rows:
                base_ids[case_key(row)] = row["dflash"]["ids"]
                base_struct[rep_name, case_key(row)] = (
                    row["dflash"]["rounds"], row["dflash"]["verify_k"],
                    row["dflash"]["accepted_histogram"])

    out = sys.stdout
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow([
        "cell", "status", "reps", "dflash_ms_median", "dflash_ms_iqr",
        "scalar_ms_median", "scalar_ms_iqr", "speedup_median",
        "prefill_tok_s_median", "accepted_per_round_median",
        "ids_match_baseline", "acceptance_match_baseline", "host_tier_slots",
        "parity_pack",
    ])
    for cell in cells:
        reps = rep_rows(cell)
        metrics = [rep_metrics(rows) for rows in reps.values()]
        ids_ok, struct_ok = True, True
        for rep_name, rows in reps.items():
            for row in rows:
                key = case_key(row)
                if baseline is not None and key in base_ids:
                    if row["dflash"]["ids"] != base_ids[key]:
                        ids_ok = False
                    if (rep_name, key) in base_struct and \
                            (row["dflash"]["rounds"], row["dflash"]["verify_k"],
                             row["dflash"]["accepted_histogram"]) != base_struct[(rep_name, key)]:
                        struct_ok = False
        slots = set()
        for log in cell.rglob("*.log"):
            if log.parent.name.startswith("rep"):
                for match in SLOTS_RE.finditer(log.read_text(errors="ignore")):
                    slots.add(int(match.group(1)))
        if (cell / "SKIP").exists():
            status = "SKIP: " + (cell / "SKIP").read_text().strip()
        elif (cell / "FAIL").exists():
            status = "FAIL"
        elif (cell / "DONE").exists():
            status = "done"
        else:
            status = "partial"
        if metrics:
            writer.writerow([
                cell.name, status, len(metrics),
                f"{statistics.median(m['dflash_ms'] for m in metrics):.1f}",
                f"{iqr([m['dflash_ms'] for m in metrics]):.1f}",
                f"{statistics.median(m['scalar_ms'] for m in metrics):.1f}",
                f"{iqr([m['scalar_ms'] for m in metrics]):.1f}",
                f"{statistics.median(m['speedup'] for m in metrics):.2f}",
                f"{statistics.median(m['prefill_tok_s'] for m in metrics):.1f}",
                f"{statistics.median(m['accepted_per_round'] for m in metrics):.2f}",
                "yes" if ids_ok else "NO",
                "yes" if struct_ok else ("n/a" if baseline is None else
                                         "ids-only(round-structure-differs)"),
                " ".join(str(s) for s in sorted(slots)) or "?",
                cell_parity(cell),
            ])
        else:
            writer.writerow([cell.name, status, 0, "", "", "", "", "", "", "",
                             "yes" if ids_ok else "NO",
                             "n/a" if baseline is None else "?", "", cell_parity(cell)])

    if baseline is not None:
        drift = [c for c in cells if c.name == "baseline-end" and rep_rows(c)]
        for cell in drift:
            metrics = [rep_metrics(rows) for rows in rep_rows(cell).values()]
            base = [rep_metrics(rows) for rows in rep_rows(baseline).values()]
            b = statistics.median(m["dflash_ms"] for m in base)
            e = statistics.median(m["dflash_ms"] for m in metrics)
            print(f"# drift control: baseline {b:.1f} vs baseline-end {e:.1f} ms/tok "
                  f"({(e - b) / b:+.1%}; >20% means the night drifted, re-run tight A/Bs)",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
