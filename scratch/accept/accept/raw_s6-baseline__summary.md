# GSM8K + MATH-500 inference performance

| dataset | row | prompt tok | scalar ms/tok | DFlash ms/tok | speedup | accept/round | parity |
|---|---:|---:|---:|---:|---:|---:|---|
| gsm8k | 895 | 84 | 634.5 | 1838.3 | 0.35x | 2.00 | yes |
| math500 | 162 | 78 | 1669.8 | 862.2 | 1.94x | 1.00 | yes |

Scalar median: 1152.1 ms/token.
DFlash2 median: 1350.2 ms/token.
Aggregate median speedup: 0.85x.

These are deterministic cold-process performance samples, not official accuracy scores.
