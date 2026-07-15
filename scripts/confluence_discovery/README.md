# Confluence discovery (research tooling)

Rigor-first search for cross-signal interactions (technical × theme × news × calendar ×
short-flow × regime) on the meta-ranker matrix. Findings report:
`research/confluence_discovery_2026-07-07.md`. **Research only — never wire outputs into
live modules.**

## Pipeline

```bash
# 1. Rebuild the point-in-time dataset (meta matrix + 4H technical states
#    + FINRA short-flow, strictly-prior-day joins)
.venv/bin/python scripts/confluence_discovery/build_dataset.py

# 2. Mine pairs on TRAIN (thresholds fit on train), select with VAL.
#    Writes singles/pairs/shortlist CSVs + fitted_thresholds JSON to research/confluence/.
.venv/bin/python scripts/confluence_discovery/search.py mine --target meta_upside --triples
.venv/bin/python scripts/confluence_discovery/search.py mine --target meta_good  --triples

# 3. ONE test-set read of a frozen shortlist (CSV with A,B[,C] columns).
#    Do not iterate against this stage.
.venv/bin/python scripts/confluence_discovery/search.py confirm \
    --target meta_upside --shortlist research/confluence/shortlist_meta_upside.csv

# Power calibration: plants synthetic pure interactions and checks the gauntlet
# catches them (quantifies the minimum detectable effect).
cd scripts/confluence_discovery && ../../.venv/bin/python power_check.py
```

## Guarantees / conventions

- 60/20/20 temporal split via the house row-fraction convention
  (`family_backtest.compute_test_cutoff`).
- Condition thresholds fit on TRAIN only (`fitted_thresholds_*.json` records them).
- Significance: per-month block t-test of joint precision vs best marginal, BH-FDR across
  all pairs; super-additivity tracked as a log-odds interaction term; sample size reported
  as de-overlapped per-ticker episodes as well as rows.
- Banned features: `trend_persistence`, `earnings_in_fwd_window`, `meta_label`
  (forward-looking). A tripwire prints any condition column with |corr| > 0.15 to
  `fwd_max_alpha` on train.
- As of 2026-07 the labeled window is ~11.5 months; the measured certification floor is a
  +7–10pp precision interaction (see power_check). Re-run when ≥6 more labeled months exist.
