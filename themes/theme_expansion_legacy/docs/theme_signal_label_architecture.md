# Theme Signal And Label Architecture

This layer sits after `theme_scores.parquet`.

The goal is to keep live-available signals separate from forward-looking ML
training labels:

- `signal_*` columns are causal features available at the close of date `T`.
- `label_*` columns are forward outcomes from `T+1` through the label horizon.
- `fwd_*` and `future_*` columns are diagnostics/targets only and must not be
  used as live features.

## Playbook Layer

`signal_market_playbook` is rule-based and intentionally outside the first ML
target:

- `risk_on_continuation`: SPY/QQQ above long trend with broad market breadth.
- `recovery_rotation`: recent drawdown plus strong index/breadth rebound impulse.
- `risk_off_defense`: index trend/breadth/drawdown stress.
- `neutral_chop`: everything else.

The intended architecture is:

1. Rules determine the higher-timeframe playbook.
2. ML predicts which themes have the best forward outcomes inside that playbook.
3. Portfolio construction expresses the result as long, avoid, hedge, or short.

## Live Signals

Core live scores:

- `signal_theme_continuation_score`: strong themes likely to keep leading.
- `signal_theme_recovery_score`: weak/mid-ranked themes starting to rebound.
- `signal_theme_hedge_score`: weak themes useful as avoid/underweight/hedge legs.
- `signal_theme_short_score`: downside-continuation candidate score.
- `signal_theme_decay_score`: rank, breadth, and trend deterioration.
- `signal_theme_exhaustion_score`: extended leaders with slowing/breadth decay.

Portfolio role:

- `signal_pair_trade_side = long_continuation`
- `signal_pair_trade_side = long_recovery`
- `signal_pair_trade_side = hedge_short`
- `signal_pair_trade_side = short`
- `signal_pair_trade_side = avoid`
- `signal_pair_trade_side = neutral`

Important: current theme-only data still does not make weak ranks reliable naked
shorts. The builder therefore exposes `signal_short_candidate_flag` separately
from `signal_short_flag`. In risk-on regimes, weak absolute/relative breakdowns
are treated as hedge-short candidates, not automatic naked shorts.

## ML Labels

Labels are generated for 5d, 10d, and 20d horizons:

- `label_forward_top_decile_*d`: best forward benchmark-excess themes.
- `label_forward_bottom_decile_*d`: weakest forward benchmark-excess themes.
- `label_future_top5_rank_*d`: future top-5 regime-rank themes.
- `label_future_top10_rank_*d`: future top-10 regime-rank themes.
- `label_rank_improver_*d`: strongest future rank migration.
- `label_continuation_long_*d`: current strong theme kept working.
- `label_recovery_rebound_*d`: current weak/mid theme became a recovery winner.
- `label_hedge_underperformer_*d`: useful underperforming hedge leg.
- `label_true_short_*d`: negative absolute and benchmark-excess forward return.
- `label_drawdown_risk_*d`: forward adverse move exceeded drawdown threshold.
- `label_avoid_*d`: weak forward benchmark-excess outcome.

Primary output:

- `theme_expansion/outputs/theme_signal_labels.parquet`

Live review output:

- `theme_expansion/outputs/live_theme_signal_ranking.csv`

Diagnostics:

- `theme_expansion/outputs/reports/theme_signal_label_summary.csv`
- `theme_expansion/outputs/reports/theme_playbook_forward_summary.csv`
- `theme_expansion/outputs/reports/theme_signal_label_dictionary.csv`
