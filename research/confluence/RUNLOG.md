# Confluence discovery run log (2026-07-06/07, agent: Claude)

Chronological record of what was tried and what it showed, so dead ends are not re-run.
Full findings: `research/confluence_discovery_2026-07-07.md`.

1. **Inventory.** Meta ranker matrix chosen as spine (554,858 rows, 2025-05-29→2026-07-02;
   forward labels end 2026-05-14 with the momentum OOF build). Verified `mom_score`/
   `htf_score` are walk-forward OOF; `_asof_prior_day_ticker` is strictly backward
   (`allow_exact_matches=False`). Excluded for lack of PIT history: dealer snapshots
   (2 days), CBOE summary (~1 month dense), NASDAQ bi-monthly SI (yfinance snapshot-only,
   no stored history), USAspending (no dated ticker store), social attention (no data).
2. **Dataset build** (`build_dataset.py`): + technical states from features_4h (98.0%
   coverage), + FINRA short-flow features prior-day (99.5%). 554,858 × 98.
3. **Leakage tripwire** flagged `theme_crowding_frac` (corr 0.286 to fwd alpha). Audited:
   causal (same-bar share of theme members with mom_xs_rank>0.8) but momentum-derived —
   treated as momentum-family in interpretation. `theme_new` and `spy_uptrend` conditions
   degenerate on train → auto-dropped. `trend_persistence`/`earnings_in_fwd_window` banned.
4. **Mine meta_upside** (524 cross-axis pairs, 8 train months): p-values uniform
   (4.9%/4.1% ≤0.05 in precision/alpha families), min q_fdr 0.56/0.86. **0 survivors.**
5. **Mine meta_good**: same picture (2.4%/4.1% ≤0.05). **0 survivors.** Triples never
   triggered (no pair survivors) and were deliberately skipped — pair-level power floor
   already above realistic effect sizes.
6. **Power check** (`power_check.py`, planted pure interactions on real masks,
   near_52w_high × short_pct_high, n_joint≈3.5k): +10pp caught (q=0.074, 0 false
   positives), +5pp missed (q=0.25), +3pp missed (q=0.98). Certification floor ≈ +7–10pp.
7. **Relaxed (uncertified) screen** — pre-registered before test: raw p≤0.05, ≥300
   episodes, ≥75% positive months, val delta>0 → 16 unique pairs (≈ chance count).
8. **Single TEST read** (test = 2026-03-11→2026-05-14, base rates strongly inflated by
   recovery tape): 6 pairs failed outright (incl. the best-raw-p pair
   `mom_htf_agree & short_pct_high`, train p=0.0014, 8/8 months → test −1.0pp).
   Momentum × theme_crowded stacks: positive deltas, strongly negative interaction terms
   (redundancy). Best candidate: `theme_rank_top10 & short_vol_surge` — delta
   +3.7/+5.5/+9.7pp train/val/test, interaction positive everywhere, both targets,
   3/3 test months, 255 tickers / 52 themes (not concentrated).
9. **Fresh-context re-verification**: independent re-derivation (no search.py imports)
   reproduced the headline pair's numbers exactly; per-test-month deltas +8.4/+6.6/+10.9pp.
10. **Do not re-run `confirm` while exploring** — the test set is burned for this study.
    Next legitimate re-test: after ≥6 more labeled months (≈ early 2027).
