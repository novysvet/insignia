# Anytime A/B CPU artifacts

Generated from the reference snapshot with:

```bash
TRIALS=1000 SEED=20260831 BOOTSTRAP_REPS=399 build/anytime-ab-cpu.sh
```

- `summary.csv` and `summary.json`: 1,000-campaign Monte Carlo comparison for each scenario.
- `run-output.json`: console-formatted copy of the summary rows.
- `replay-three-median-seed-0.csv` and `.json`: the first three guarded pairs for structural `B/A = 1.05`; the conventional three-run median promotes B.
- `protocol-example.json`: editable full decision protocol.
- `pair-log-template.csv`: header-only logging schema.
- `dp-demo.json`: exact finite escalation result and rollout bounds.

The report and proofs are in `audits/s11-anytime-wsl-noise.md`.
