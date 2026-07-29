"""WS-E — market-regime/sector-context ablation harness for momentum_expansion.

See docs/superpowers/plans/2026-07-26-market-regime-and-sector-context.md §4
WS-E, and plan defect D6: no model training happens in this package. It
provides:

  * feature_blocks.py — the five ablation arms (baseline/+risk/+liquidity/
    +sector/+all) reusing REGIME_FEATURE_COLUMNS_4H.
  * folds.py           — the fixed walk-forward fold spec (reuses
    strategies.model_training.colab_competition.date_folds; does not
    reimplement it), sourced from WALK_FORWARD_CONFIG.
  * metrics.py          — rank IC, NDCG@10, top-N forward return, win rate,
    turnover, Sharpe, MaxDD.
  * bootstrap.py         — week-block bootstrap CIs, and BH-FDR re-exported
    from scripts.confluence_discovery.search (not reimplemented).
  * screen.py            — the training-free local screen: group-appropriate
    analysis (market-wide regime-conditional expectancy / sector-level rank
    IC / interaction rank IC) run on real data per walk-forward test period.
  * run_screen.py         — CLI entry point that runs screen.py end-to-end and
    writes CSV reports (never overwrites production feature/label/matrix
    files).
  * export_colab_ablation.py — Colab export bundling the five arms so they
    can actually be trained off-repo.
"""
