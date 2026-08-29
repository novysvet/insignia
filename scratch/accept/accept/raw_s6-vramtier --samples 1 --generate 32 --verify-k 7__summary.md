# GSM8K + MATH-500 inference performance

| dataset | row | prompt tok | scalar ms/tok | DFlash ms/tok | speedup | accept/round | parity |
|---|---:|---:|---:|---:|---:|---:|---|
| gsm8k | 893 | 76 | 509.1 | 625.6 | 0.81x | 4.57 | yes |
| gsm8k | 1159 | 93 | 543.5 | 570.3 | 0.95x | 3.20 | yes |
| math500 | 209 | 65 | 611.8 | 837.8 | 0.73x | 2.91 | yes |
| math500 | 212 | 91 | 598.3 | 629.5 | 0.95x | 3.20 | yes |

Scalar median: 570.9 ms/token.
DFlash2 median: 627.5 ms/token.
Aggregate median speedup: 0.91x.

These are deterministic cold-process performance samples, not official accuracy scores.
