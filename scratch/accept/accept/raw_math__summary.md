# GSM8K + MATH-500 inference performance

| dataset | row | prompt tok | scalar ms/tok | DFlash ms/tok | speedup | accept/round | parity |
|---|---:|---:|---:|---:|---:|---:|---|
| gsm8k | 511 | 64 | 581.0 | 590.1 | 0.98x | 2.91 | yes |
| gsm8k | 404 | 69 | 562.0 | 678.4 | 0.83x | 2.67 | yes |
| gsm8k | 636 | 73 | 515.5 | 564.3 | 0.91x | 2.00 | yes |
| gsm8k | 2 | 78 | 501.1 | 683.9 | 0.73x | 2.67 | yes |
| gsm8k | 756 | 82 | 605.5 | 692.9 | 0.87x | 2.13 | yes |
| gsm8k | 1033 | 86 | 509.0 | 583.2 | 0.87x | 3.20 | yes |
| gsm8k | 537 | 92 | 646.2 | 574.4 | 1.12x | 3.20 | yes |
| gsm8k | 1207 | 97 | 523.4 | 507.8 | 1.03x | 2.67 | yes |
| gsm8k | 108 | 106 | 528.0 | 633.9 | 0.83x | 2.29 | yes |
| gsm8k | 151 | 119 | 1016.5 | 770.5 | 1.32x | 2.67 | yes |

Scalar median: 545.0 ms/token.
DFlash2 median: 612.0 ms/token.
Aggregate median speedup: 0.89x.

These are deterministic cold-process performance samples, not official accuracy scores.
