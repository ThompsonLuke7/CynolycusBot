# Location, Liquidity, Dealer, and Options Feature Audit

Date: 2026-06-14

## Why SPY Has ~1,600 Features But Swing Has ~200

SPY day trader uses a wide, auto-generated, multi-timeframe technical matrix. The current enriched export has 1,627 features:

- Existing SPY technical/context matrix: about 1,541 features
- Added SPY daily options/Greeks features: 79 features
- Added 9-second bid/ask-volume features: 7 features

Multi-ticker swing is intentionally curated and category-based. The enriched export has 183 features:

- Base swing features: 124
- Daily context features: 59

XGBoost can handle wide feature sets, but a wider matrix is not automatically better. Highly redundant features slow training, dilute split/importance attribution, and make ablations harder. The better target is not "more features"; it is better location/liquidity information with leakage-safe calculations.

## Existing SPY Day Trader Features

Source matrix: `Data/processed/spy/training_export/spy_daytrader_context_matrix.parquet`

Feature organization:

- Core feature code lives under `strategies/spy_intraday/Features/`.
- Support/resistance features are separated in `strategies/spy_intraday/Features/feature_sets/support_resistance_features.py`.
- The export manifest is `Data/processed/spy/training_export/spy_daytrader_context_manifest.json`.

### SPY Support / Resistance / Location

| Feature(s) | Description | Calculation | Data Source | Models Use It | Importance |
|---|---|---|---|---|---|
| `dist_to_recent_high_6/12/24`, `dist_to_recent_low_6/12/24` and MTF suffixes | Rolling local structure distance | Rolling high minus close; close minus rolling low | SPY OHLCV, multi-timeframe resamples | SPY day trader export, selected SPY models | Selected in long model: `dist_to_recent_high_6__30m`; selected in short model: `recent_high_touch_count__30m` |
| `pct_from_recent_high`, `pct_from_recent_low` and MTF suffixes | Position relative to rolling high/low | `(close - rolling_high) / rolling_high`, `(close - rolling_low) / rolling_low` | SPY OHLCV | SPY day trader export | Useful in correlation checks for big-move labels |
| `recent_high_touch_count`, `recent_low_touch_count` and MTF suffixes | Rolling count of touches near local highs/lows | Rolling count near max/min using tolerance | SPY OHLCV | SPY day trader export | Selected in SPY models |
| `dist_to_pdh`, `dist_to_pdl`, `dist_to_pwh`, `dist_to_pwl` and MTF suffixes | Distance to prior day/week high/low | Close minus prior period level, or level minus close depending feature | SPY OHLCV | SPY day trader export | Selected in SPY long/short/single models |
| `dist_to_vwap`, `dist_to_vwap_upper_1/2`, `dist_to_vwap_lower_1/2`, `above_vwap_flag`, `vwap_slope`, `vwap_dist_pct`, `vwap_slope_pct` and MTF suffixes | VWAP location and band context | Session VWAP and volume-weighted std bands | SPY OHLCV volume | SPY day trader export | Strong selected-feature presence; correlated with direction labels, especially later intraday |
| `dist_to_prior_day_vwap`, `dist_to_prior_week_vwap` and MTF suffixes | Higher-timeframe VWAP location | Prior day/week VWAP mapped to current bar | SPY OHLCV volume | SPY day trader export | Selected in SPY short/single models |
| `dist_to_round_5`, `dist_to_round_10` and MTF suffixes if enabled | Distance to psychological levels | Close minus nearest rounded price level | SPY OHLCV | SPY day trader export if generated | Importance not separately available |
| `dist_to_poc`, `inside_value_area_flag` if market profile enabled | Prior-day market profile location | Prior day volume-by-price POC/value area | SPY OHLCV volume | Optional in SPY support/resistance module | Not present in current export unless market profile enabled |

### SPY Reaction Features

| Feature(s) | Description | Calculation | Data Source | Models Use It | Importance |
|---|---|---|---|---|---|
| `upper_wick_ratio`, `lower_wick_ratio`, `range_normalized_wick`, `upper_wick_pct`, `lower_wick_pct` and MTF suffixes | Wick rejection anatomy | Candle wick length divided by total range or close | SPY OHLCV | SPY day trader export | Selected in SPY models |
| `rejection_near_vwap_flag`, `rejection_near_htf_flag` and MTF suffixes | Rejection near VWAP or prior HTF levels | Wick threshold plus proximity to VWAP/HTF levels | SPY OHLCV | SPY day trader export | Selected in SPY long/short models |
| `close_pos_in_range`, `range_pct` and related candle shape fields | Bar acceptance/rejection proxy | Close position inside high-low range; range normalized by price | SPY OHLCV | SPY day trader export | Selected in SPY models |

### SPY Options / Dealer-Derived Context

| Feature(s) | Description | Calculation | Data Source | Models Use It | Importance |
|---|---|---|---|---|---|
| `spyopt_oi_call`, `spyopt_oi_put`, `spyopt_oi_total`, `spyopt_oi_put_call_ratio` | Daily SPY open-interest pressure | Aggregated daily call/put OI and ratio | `drive-download-20260613T045727Z-3-001/spy_options_daily_dataset_3y_fixed_greeks.jsonl` | SPY context export | Correlations stronger for range/big-move than direction |
| `spyopt_vol_0930_*` | Early option volume pressure | 09:30 call/put option volume and ratio | Same options JSONL | SPY context export | Needs model importance after retrain |
| `spyopt_oiw_*_delta/gamma/vega/theta/rho/vanna/charm/vomma/color/...` | OI-weighted Greek exposure proxies | OI-weighted Greek aggregates and net/per-OI variants | Same options JSONL | SPY context export | Strong correlations to `spyopt_target_move_abs` and big-move labels |
| `spyopt_gamma_oi_within_1pct/2pct/5pct` | Gamma concentration near spot | Gamma OI near spot windows | Same options JSONL | SPY context export | Useful range/big-move candidates |
| `call_wall`, `put_wall`, `gamma_flip`, `nearest_magnet`, `next_magnet_*` | Live dealer levels | Computed from Schwab option chain ladder | `strategies/dealer_positioning/*`, `Data/dealer_positioning/*` | Dealer module only right now | Not yet model-facing in SPY day trader matrix |

Current SPY dealer gap: the dealer module computes wall/magnet/flip levels, but the SPY day trader matrix does not yet include live/historical proximity or confluence features such as `distance_to_call_wall`, `inside_put_wall_zone`, or support-putwall confluence.

## Existing Multi-Ticker Swing Features

Source matrix: `strategies/multi_ticker_swing/data/training_export/swing_context_colab_bundle.tgz`

Feature organization:

- Feature code: `strategies/multi_ticker_swing/features/build_features.py`
- Central categorized feature list: `strategies/multi_ticker_swing/config/pipeline_config.py`
- Categories include price, volatility, trend, oscillators, swing/setup, volume, gap, relative strength, location, time, daily, behavioral, and now context export features.

### Swing Support / Resistance / Location

| Feature(s) | Description | Calculation | Data Source | Models Use It | Importance |
|---|---|---|---|---|---|
| `dist_to_recent_swing_high`, `dist_to_recent_swing_low` | Distance to recent swing extremes | Uses causal 20-bar rolling high/low normalized by ATR | Per-ticker 30m OHLCV | Multi-ticker swing | `0.00749`, `0.00740` |
| `dist_20bar_high`, `dist_20bar_low` | Same underlying 20-bar high/low distances | Rolling 20-bar high/low normalized by ATR | Per-ticker 30m OHLCV | Multi-ticker swing | `0.00934`, `0.00982` |
| `range_pos_20` | Position inside 20-bar range | `(close - rolling_low) / rolling_range` | Per-ticker 30m OHLCV | Multi-ticker swing | `0.00440` |
| `dist_to_vwap` | Session VWAP location | Current close minus session VWAP, normalized by ATR | Per-ticker 30m OHLCV | Multi-ticker swing | `0.03330`, #3 current importance |
| `dist_to_pdh`, `dist_to_pdl`, `dist_to_pdc` | Prior day high/low/close location | Distance normalized by ATR | Per-ticker 30m OHLCV | Multi-ticker swing | `0.00503`, `0.00540`, `0.01284` |
| `dist_to_rolling_mean_20`, `zscore_close_20/64`, `percentile_close_20/64` | Mean-reversion location | Rolling mean/z-score/percentile | Per-ticker 30m OHLCV | Multi-ticker swing | `percentile_close_20` is #4 at `0.03244` |

### Swing Gap / Breakout / Reaction

| Feature(s) | Description | Calculation | Data Source | Models Use It | Importance |
|---|---|---|---|---|---|
| `gap_pct`, `gap_to_atr`, `overnight_ret` | Overnight gap context | Session open vs prior close; normalized by ATR | Per-ticker 30m OHLCV | Multi-ticker swing | `gap_to_atr` `0.01838`; `gap_pct` `0.01583`; `overnight_ret` `0.01686` |
| `gap_followthrough_1`, `gap_followthrough_2` | Early gap follow-through | Early session bar returns after gap | Per-ticker 30m OHLCV | Multi-ticker swing | `gap_followthrough_2` `0.00955` |
| `gap_frequency_rolling` | Ticker behavior profile | Rolling gap frequency | Per-ticker history | Multi-ticker swing | `0.00487` |
| `breakout_pressure_score` | Compression plus volume pressure | Weighted compression and relative-volume score | Per-ticker 30m OHLCV | Multi-ticker swing | `0.00724` |
| `close_location_in_bar` | Candle acceptance/rejection proxy | Close position in current bar range | Per-ticker 30m OHLCV | Multi-ticker swing | `0.01324` |

Current swing gap: it has basic location and gap features, but does not have full liquidity-zone features, 52-week high/low distances, recent 20d/60d daily high/low distances, nearest gap above/below, failed breakout/breakdown counts, or SPY dealer context.

## Redundancy Audit

### SPY Context Export

Rows: 52,528. Features: 1,627.

Findings:

- Constant columns: 12, all from options Greek families that are zero/empty in the source file.
- Exact duplicate columns: 11, all duplicates of the same constant-zero options columns.
- Non-constant pairs at 99.9% absolute correlation: none in the fast full-matrix audit.
- Highest approximate correlations are still very high, mostly indicator aliases or overlapping multi-timeframe levels:
  - `vol_z_20__4h` vs `dollar_vol_z_20__4h`: `0.998965`
  - `SQZ_ON` vs `SQZ_OFF`: `0.998503`
  - `TMO_14_5_3` vs `tmo_main`: `0.997226`
  - `TMOs_14_5_3` vs `tmo_signal`: `0.997481`
  - `NATR_14` vs `keltner_width_pct`: `0.995585`

Recommended SPY pruning before retrain:

- Drop the 12 constant `spyopt_*zomma/speed/vera*` columns unless the source is fixed to populate them.
- Do not aggressively prune below 99.9% yet; instead use XGBoost `colsample_bytree`, feature importance, and ablation groups.
- Consider a later "alias cleanup" pass for obvious indicator duplicates like `TMO_14_5_3`/`tmo_main`.

### Swing Context Export

Sample rows: 9,350. Features present in sample/context audit: 154 of 183.

Findings:

- Exact duplicates / redundant definitions:
  - `dist_to_recent_swing_high` = `dist_20bar_high`
  - `dist_to_recent_swing_low` = `dist_20bar_low`
  - `gap_pct` = `overnight_ret`
  - `dollar_volume_pctile` = `dollar_vol_pctile_rolling`
  - sparse macro high-impact fields duplicate all-macro fields in the current calendar
- 99.9% correlated pairs:
  - `bars_from_open` vs `bars_to_close`: `1.000000`
  - `ret_1` vs `log_ret_1`: `0.999759`
  - `volume_rel_20` vs `dollar_volume_rel_20`: `0.999221`
  - the exact duplicate pairs above
- CBOE context columns are effectively empty for the swing training window right now.

Recommended swing pruning before retrain:

- Remove one of each exact duplicate pair from the Colab manifest before training.
- Prefer keeping the more interpretable names:
  - Keep `dist_to_recent_swing_high/low`; drop `dist_20bar_high/low`.
  - Keep `gap_pct`; drop `overnight_ret`.
  - Keep `dollar_volume_pctile`; drop `dollar_vol_pctile_rolling`.
  - Keep `bars_from_open`; drop `bars_to_close`.
- Drop CBOE swing context until it has real history.

## Missing High-Priority Features

### SPY Day Trader

Already present:

- Rolling high/low distances and touch counts
- Prior day/week high/low distances
- VWAP and VWAP bands
- Wick ratios and VWAP/HTF rejection flags
- Daily options/Greeks aggregates
- Dealer module computes call wall, put wall, magnets, gamma flip

Missing:

- Liquidity zones built from clustered swing highs/lows, high-volume reversal candles, and wick rejections
- Zone width, touch count, days/bars since touch
- Failed breakout / failed breakdown counts
- Dealer proximity features in the SPY day trader matrix:
  - `distance_to_call_wall`
  - `distance_to_put_wall`
  - `distance_to_gamma_flip`
  - `distance_to_nearest_magnet`
  - `inside_call_wall_zone`
  - `inside_put_wall_zone`
  - `between_major_walls`
- Confluence features:
  - `support_and_putwall_confluence`
  - `resistance_and_callwall_confluence`
  - `distance_to_nearest_liquidity_cluster`
  - `liquidity_cluster_strength`

### Multi-Ticker Swing

Already present:

- Basic rolling 20-bar support/resistance location
- VWAP, prior day high/low/close
- Gap and follow-through
- SPY/QQQ/IWM/TLT/USO/GLD market context returns/correlation
- Earnings/news/theme/treasury context in Colab export

Missing:

- `distance_to_52w_high`
- `distance_to_52w_low`
- `distance_to_recent_20d_high`
- `distance_to_recent_20d_low`
- `distance_to_recent_60d_high`
- `distance_to_recent_60d_low`
- `distance_to_gap_above`
- `distance_to_gap_below`
- gap-fill probability/history features
- richer `breakout_proximity`, `support_proximity`, `resistance_proximity`
- SPY location context:
  - `spy_distance_to_support`
  - `spy_distance_to_resistance`
  - `spy_inside_support_zone`
  - `spy_inside_resistance_zone`
  - `spy_gamma_flip_distance`
  - `spy_dealer_position_score`

## Engineering Priority

1. Add a shared liquidity-zone engine with leakage-safe rolling calculations.
2. Wire SPY day trader first: support/resistance zones, reaction, dealer proximity, and confluence.
3. Wire swing second with daily/30m location features and SPY context, but do not build full per-ticker dealer positioning yet.
4. Add permanent daily dealer-level storage for SPY, QQQ, and IWM:
   - `timestamp`
   - `call_wall`
   - `put_wall`
   - `gamma_flip`
   - `nearest_magnet`
   - `next_magnet_above`
   - `next_magnet_below`
   - `total_gex`
   - `air_gap_above_score`
   - `air_gap_below_score`
   - `spot_price`

