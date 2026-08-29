# GSM8K + MATH-500 inference performance

| dataset | row | prompt tok | scalar ms/tok | DFlash ms/tok | speedup | accept/round | parity |
|---|---:|---:|---:|---:|---:|---:|---|
| gsm8k | 893 | 76 | 578.2 | 636.2 | 0.91x | 4.57 | yes |
| gsm8k | 1159 | 93 | 562.8 | 549.0 | 1.03x | 3.20 | yes |
| math500 | 209 | 65 | 609.7 | 736.3 | 0.83x | 2.91 | yes |
| math500 | 212 | 91 | 565.8 | 579.8 | 0.98x | 3.20 | yes |

Scalar median: 572.0 ms/token.
DFlash2 median: 608.0 ms/token.
Aggregate median speedup: 0.94x.

These are deterministic cold-process performance samples, not official accuracy scores.
