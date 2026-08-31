# Anytime-selective falsifier CPU report

This is a controlled synthetic result. Timing units model inference cost; they are not measurements from the RTX 4070 SUPER or GLM-5.3-Flash.

## Configuration

- runs: 16
- rounds per run: 12000
- base seed: 20260831
- drift change point: 4200
- initial exact-shadow pilot rows: 15000
- fail-closed reset rows: 2400
- selective target epsilon: 0.060
- anytime error budget delta: 0.050
- audit range: [0.18, 0.48]
- global startup reserve: 60.0 severity units

The audit probability is computed from causal context and the frozen score. Realized severity, latency, and cache transition do not enter the current coin.

## Results

| policy | intended risk | max local risk | reported CS failures | local risk violations | pathwise violations | abstention | audits | audit ms/round | reset ms/round | saved ms/round | throughput reward tok/s | catastrophes/run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| naive calibration | 0.0813 | 0.2194 | 16/16 | 16/16 | 16/16 | 30.4% | 0.0% | 0.0 | 0.0 | 325.8 | 1.054 | 536.06 |
| split conformal | 0.0756 | 0.1918 | 16/16 | 16/16 | 15/16 | 32.0% | 0.0% | 0.0 | 0.0 | 317.9 | 1.016 | 488.88 |
| nominal importance weighting | 0.0668 | 0.1944 | 16/16 | 16/16 | 6/16 | 42.7% | 11.5% | 113.4 | 0.0 | 185.9 | 0.476 | 285.38 |
| anytime random audit | 0.0084 | 0.0141 | 0/16 | 0/16 | 0/16 | 67.4% | 5.8% | 58.9 | 173.4 | 74.2 | 0.168 | 1.19 |

The anytime method had no observed reported-loss CS failure, committed-loss CS failure, local target violation, or global pathwise-budget violation.
That empirical record is separate from the mathematical coverage proof.

## Impossibility witness

- exact-only logs identical: `true`
- shared exact-log SHA-256: `d1c0199e9e710b09709d32c5fab389612644bfcd977a6839964793ce94f3480c`
- safe-world deployed fast risk: 0.0
- bad-world deployed fast risk: 1.0

## Frozen initial policy

- threshold: 0.117705414
- information price: 10.000
- historical stress-screen risk: 0.023736
- screen method: `adaptive_historical_cell_and_window_screen_no_coverage_claim`
- mean planned audit probability: 0.1800
- enabled support cells: 4

The historical screen is a tuning diagnostic and carries no coverage claim. Deployment coverage is supplied only by fresh post-freeze audit e-values.

## Reproduction

```bash
PYTHONHASHSEED=0 python tools/test_anytime_selective_falsifier.py
PYTHONHASHSEED=0 python tools/evaluate_anytime_selective_falsifier.py \
  --runs 16 --rounds 12000 --calibration-rows 15000 \
  --output-dir scratch/anytime-selective-falsifier
```
