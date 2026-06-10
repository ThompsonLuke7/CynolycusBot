# Intraday SPY Trading Agent: Research Summary So Far

Author: Luke Thompson  
Draft status: work-in-progress research summary  
Last updated: May 8, 2026

## Abstract

This project studies whether machine learning can support an intraday SPY trading agent that detects developing trend shifts, enters with less emotional bias than a discretionary trader, and manages trades without requiring constant human monitoring. The current best research direction is a 10-minute GA-selected XGBoost setup model paired with a 1-minute rule-based order policy. The model produces long, short, and neutral setup probabilities, while the order policy decides whether those probabilities are actionable based on short-term price confirmation and trade-management rules.

The strongest saved phase-4 ATR policy result currently found is a non-shift setup-area policy with long threshold 0.42, short threshold 0.20, 2-bar post-setup confirmation, 1.5 ATR target, and 1.0 ATR stop. It produced 281 trades, 2.08 trades per day, +0.4587 ATR expected value per trade, 56.80% long win rate, and 50.00% short win rate. A related body-and-close policy remains important because it is simpler and closer to the order-confirmation idea being validated; the best saved body-and-close row produced 309 trades, 2.29 trades per day, +0.3407 ATR expected value per trade, 66.67% long win rate, and 58.60% short win rate. These should be reported separately from the May 3 live-decision replay verification, which uses different return units and currently supports the live-style 0.35 long / 0.65 short candidate.

## Background

The project began from a practical trading problem: discretionary options trading can produce large gains, but the process is vulnerable to emotional decisions, poor risk control, overexposure, and missed entries or exits when the trader cannot watch the market continuously. The intended agent is not designed to predict every price movement. Instead, it attempts to detect when a tradable intraday move is forming and then "ride the wave" of larger market participants.

The research direction also follows the broader machine learning literature on market prediction. Prior work suggests that technical indicators, ensemble learning, hybrid models, and reinforcement learning can outperform simple benchmarks when the problem is framed carefully. However, this project has shown that good prediction metrics alone are not enough. Label design, leakage control, probability calibration, and the order execution policy matter at least as much as the model family.

## Purpose

The purpose of the project is to build and evaluate an automated intraday trading system for SPY that:

- Uses historical and live market data to identify long and short setup conditions.
- Converts model probabilities into tradable entries through a deterministic order policy.
- Reduces emotional trading and missed opportunities.
- Evaluates performance using trade-level statistics instead of only classification accuracy.
- Avoids future leakage and overly optimistic backtests.
- Leaves enough structure for live trading, option execution, feature engineering, and risk management to be improved outside the machine-learning model itself.

## Current System Overview

The current core pipeline is:

1. Fetch and cache intraday SPY data.
2. Build a feature matrix from OHLCV, technical indicators, regime features, volatility features, cross-asset features, and model-derived probability features.
3. Generate swing-support labels around tradable long and short setup regions.
4. Use GA plus XGBoost feature selection to reduce the feature set.
5. Train a single multi-class XGBoost model that predicts long, neutral, or short setup probabilities.
6. Convert setup probabilities into trades using a rule-based order policy.
7. Score results with classification metrics, trade metrics, and baseline comparison.

The current deployed-style model artifact is the full-fit `swing_support_single` model from April 13, 2026. It used 52,528 training rows, 1,598 available features, and 430 selected features. Because that artifact is full-fit, its training metrics are useful for deployment context, but the stronger validation reference is the April 11, 2026 run with an explicit test split.

## Model Summary

One-sentence summary: The model is a GA-selected XGBoost classifier that reads each 10-minute SPY market state and estimates whether the next useful setup is long, short, or neutral.

Expanded summary: The model acts like a fast technical trader. It looks at price action, volatility, trend, support/resistance context, time-of-day information, and other market features, then assigns probabilities to long and short setup conditions. The model does not directly place trades. It produces the signal layer that the order policy decides whether to act on. The current primary model is a single multi-class XGBoost model trained on the 10-minute `swing_support` label set. A genetic algorithm is used as a feature selector so the model does not have to rely on the full high-dimensional feature set. Side-specific thresholds are then applied to the long and short probabilities before the order policy looks for 1-minute confirmation.

## Order Policy Summary

One-sentence summary: The order policy waits for the 10-minute model to mark a setup, then requires 1-minute price-action confirmation before entering.

Expanded summary: The model is allowed to say, "A long or short setup may be forming." The order policy then asks, "Did price actually confirm this on the 1-minute chart?" This prevents the system from buying every raw probability spike. It also limits repeated entries from the same setup cluster and uses ATR-based profit and stop assumptions to evaluate trades. The most important policy family right now is the body-and-close confirmation family, because it requires the 1-minute candle to confirm with real price structure rather than only a loose threshold touch. The highest saved ATR row overall currently uses a mixed confirmation policy, so the paper should describe it as the best saved score while continuing to validate body-and-close as the cleaner order-policy candidate.

For the saved ATR scoreboards, the breakeven win rate for a +1.0 ATR target and -0.8 ATR stop is 44.44%. For a +1.5 ATR target and -1.0 ATR stop, the breakeven win rate is 40.00%. This matters because several current policy rows have win rates that clear breakeven even when they do not win a majority of all trades.

## Current Best Saved Results

### Validated XGBoost Model Metrics

The best clean validation reference remains `Data/models/ga_xgboost/10min/training_run_summary_20260411T075130Z_ga_xgboost_train.json`.

Long-side test metrics: threshold 0.3565, accuracy 82.77%, precision 46.03%, recall 44.88%, F1 45.45%, AUC 0.8276, and average precision 0.4780.

Short-side test metrics: threshold 0.3367, accuracy 83.15%, precision 45.93%, recall 32.38%, F1 37.98%, AUC 0.8014, and average precision 0.4134.

Multi-class test metrics: 72.34% accuracy, 0.4155 macro F1, and 0.6266 log loss.

The April 13 full-fit deployment artifact improved full-train multi-class accuracy to 78.12% with 0.6086 macro F1, but it does not replace the April 11 validation result because it was fit on the full available set.

### Best Order Policy Versus Baseline

The best current saved ATR policy result from `Data/models/ga_xgboost/model_competition_phase4/competition_best_scoreboards.csv` is:

- Model family: non-shift setup-area.
- Long setup threshold: 0.42.
- Short setup threshold: 0.20.
- Confirmation: asymmetric 1-minute post-setup confirmation, with long confirmation and short momentum confirmation.
- Post-setup confirmation window: 2 bars.
- Target/stop setting: 1.5 ATR target and 1.0 ATR stop.
- Trades: 281.
- Trades per day: 2.08.
- Total EV: +0.4587 ATR per trade.
- Long EV: +0.5215 ATR per trade.
- Short EV: +0.4083 ATR per trade.
- Long win rate: 56.80%.
- Short win rate: 50.00%.

The best saved body-and-close result, from `Data/models/ga_xgboost/model_competition_phase4_focused/nonshift_setup_area_l0.42_s0.15_lag2_h12_tp1.0_sl0.8/best_phase4_trigger_scoreboard.csv`, is:

- Model family: non-shift setup-area.
- Long setup threshold: 0.42.
- Short setup threshold: 0.15.
- Confirmation: body-and-close confirmation on both long and short entries.
- Post-setup confirmation window: 6 bars.
- Target/stop setting: 1.0 ATR target and 0.8 ATR stop.
- Trades: 309.
- Trades per day: 2.29.
- Total EV: +0.3407 ATR per trade.
- Long EV: +0.4092 ATR per trade.
- Short EV: +0.2954 ATR per trade.
- Long win rate: 66.67%.
- Short win rate: 58.60%.

The support/resistance breakout baseline remains a useful comparison because it shows what happens when the system trades simple breakouts without the learned setup layer. The saved lookback-24 baseline produced about -0.3380 ATR per trade, while the lookback-12 baseline produced about -0.3707 ATR per trade. In plain English, the current model-plus-policy approach moved the system from a negative breakout baseline to positive expected value in the saved ATR framework.

### Current Live-Style Verification

The most recent setup verification file is `Data/inference/spy/10min/setup/verification_current_best_2026-05-03.md`. This did not run a new training job. It replayed existing live-decision data and checked a narrow 2024-plus robustness window.

The recommended live-style candidate from that verification is:

- Long threshold: 0.35.
- Short threshold: 0.65.
- Setup max bars: 3.
- New-entry cutoff: 15:00.
- Opposite-exit thresholds: 0.40 long and 0.75 short.
- Entry/exit quote model: mid entry and bid exit.
- Stop loss: 1.0, effectively disabled in that replay.
- Trail: arm at 1.0, give back 0.20.
- Time decay: 60 minutes with 0.5 progress threshold.
- Scalp overlay: disabled.
- Candidate entries: disabled.

On the 2024-plus narrow robustness check from January 2, 2024 through April 1, 2026, this row produced 801 trades, 0.1018 average return, 36.70% win rate, 81.51 total return units, 0.7217 average MFE, 695 long trades, and 106 short trades. This should not be mixed directly with ATR EV numbers because it is a broader replay-style check with different units.

## Probability Normalization

Probability normalization was an experiment to make the short-side probability easier to compare across market regimes. Instead of saying, "short probability must be above one fixed raw number," the experiment asked whether the current short probability was unusually high compared with its own recent or regime-specific history. The idea was reasonable because short setups often behave differently from long setups, and raw short probabilities can mean different things in different volatility regimes.

The best saved probability-normalization experiment used regime-history percentile normalization for the short side. It produced 1,308 trades, 0.0648 average return, 34.02% win rate, and 84.82 total return units on the selected 2024-plus replay. It also increased short participation, with 664 short trades and 29.24 short-side return units.

The current interpretation is cautious: probability normalization may be useful as a calibration layer, especially for short-side behavior, but it is not the current default policy and should not be presented as a final result. It increased trade count and reduced win rate, so it needs stricter walk-forward validation before it becomes part of the main method.

## Statistics Used

Accuracy measures the fraction of bars classified correctly. It is easy to understand but can be misleading when neutral bars dominate.

Precision measures how many predicted setup bars were actually setup labels. This helped most when false entries were the main risk.

Recall measures how many true setup bars the model found. It helped identify whether the model was missing too many tradable regions.

F1 balances precision and recall. It was useful for comparing model variants, but it was not enough by itself because a good F1 score does not guarantee profitable trades.

AUC measures whether the model ranks positive examples above negative examples across thresholds. It helped identify whether a model had real separation before choosing a threshold.

Average precision was often more useful than AUC because long and short setup labels are relatively sparse.

Log loss measures probability quality. It matters because the order policy depends on probability thresholds, not just hard class labels.

Macro F1 averages F1 across long, neutral, and short classes so the neutral class cannot hide weak long/short behavior.

Event +/- 1 metrics give credit when a prediction lands within one bar of the true event. This matters because a setup one bar early or late can still be tradable.

EV per trade in ATR became one of the most important statistics because it connects model output to trading performance while normalizing for volatility.

MFE and MAE show how far trades moved in the favorable or adverse direction after entry. These helped evaluate whether exits, stops, and trailing rules were leaving too much money on the table.

The least useful statistics by themselves were plain accuracy and full-train metrics. They are still worth reporting, but the project improved most when evaluation moved toward validation splits, event-tolerant scores, EV, win rate, MFE, MAE, trades per day, and baseline comparison.

## What Helped Most

1. Moving from raw pivot prediction toward swing-support labels.

The fuzzy swing-support label reframed the problem from exact pivot-bar prediction to tradable setup-region prediction. This better matches the actual trading goal, because a trade does not need to catch the exact local high or low to be profitable.

2. Moving from dual binary heads to a single multi-class model.

The single long/neutral/short model improved the structure of the prediction problem. The April 11 run reached 72.34% multi-class test accuracy and produced usable side-specific probability outputs.

3. Using the 1-minute confirmation order policy.

The largest jump in trade quality came from not trading raw 10-minute setup probabilities directly. Confirmation improved EV and win rate by requiring price action to validate the model signal.

4. Using body-and-close confirmation.

Body-and-close confirmation made the order policy more conservative by requiring the 1-minute candle body and close to confirm the move. The best saved body-and-close row produced +0.3407 ATR EV per trade, which is not the top row overall but is strong enough to remain a major validation path.

5. Using asymmetric thresholds.

Long and short setups behave differently. The best saved ATR rows use different long and short thresholds, and the May 3 live-style verification also supports asymmetric thresholds with long 0.35 and short 0.65.

6. Removing leakage and forcing causal indicators.

The DPO feature from `pandas_ta` was identified as dangerous because its default centered calculation can shift future information backward. Removing this hurt some backtest metrics, but it made the research more honest and deployable.

7. Using trade-level metrics instead of only model metrics.

The project improved when evaluation moved from accuracy alone to EV, win rate, MFE, MAE, trades/day, and baseline comparison.

## What Did Not Work As Well

1. Triple-barrier probabilities.

The triple-barrier probability path looked promising before leakage fixes, but after removing non-causal features it became much less useful. One saved TB validation run had long AUC 0.5290 and short AUC 0.5136, with thresholds near zero and recall near 1.0. That means the model was mostly predicting the base rate rather than separating useful entries.

2. Meta entry model.

The meta entry model did not become a strong entry source. The April 8 entry run had validation AUC 0.5448 for long entries and 0.5321 for short entries. F1 was 0.3444 long and 0.3603 short. Those numbers are too weak to justify using the meta entry layer as the primary live signal.

3. Meta model as a full replacement.

The meta exit model was stronger than the meta entry model, with validation AUC around 0.828 for long exits and 0.834 for short exits. However, a good exit model cannot fully rescue a weak entry model. The current system therefore uses the `swing_support_single` XGBoost setup model as the primary entry probability source.

4. Shift1 as the main path.

Shift1 was mixed. The Shift1 validation run had high recall but weaker accuracy and macro F1 than the primary non-shift model. Some policy rows were respectable, but the newer competition file still favors non-shift setup-area policies for the best saved ATR rows.

5. Simple support/resistance breakout entries.

The S/R breakout baseline was consistently negative. The best lookback-24 baseline row still produced about -0.3380 ATR EV per trade with weak long and short win rates.

6. Feature selection alone.

Early experiments showed that GA plus XGBoost feature selection was useful but not sufficient by itself. In the early 343-feature comparison, XGBoost on all features reached 61.45% test accuracy, while GA-XGBoost reached 60.00%. The real gains came later from label design and order policy improvements.

## Early Classification Baseline

The early classification baseline is still worth keeping because it shows how far the project moved from simple feature/model comparisons into trading-system evaluation.

| Model | Test accuracy |
|---|---:|
| XGBoost on OHLCV only | 42.00% |
| XGBoost on all features | 61.45% |
| GA-XGBoost | 60.00% |
| GA-RF | 41.82% |
| GA-ET | 42.00% |
| GA-SVM | 58.18% |
| GA-KNN | 42.91% |

The more important baseline now is trade-level performance. Against the simple support/resistance breakout baseline, the current model plus 1-minute confirmation policy moved from negative expected value to positive expected value.

## Methodology To Formalize

The final paper should add a formal methodology section that explains:

- Data sources and date ranges.
- Intraday timeframe construction.
- Feature engineering and feature selection.
- Label construction for swing-support setups.
- Train/validation/test split logic.
- Leakage controls and causal indicator rules.
- Threshold selection.
- Order-policy evaluation.
- Difference between ATR scoreboards and live-decision replay return units.
- Why results are reported against both model metrics and trade metrics.

## Live Trading Dashboard Figure

Add a figure section here for the live trading dashboard screenshot.

Suggested caption: "Live trading dashboard used to monitor SPY setup probabilities, current position state, order-policy decisions, broker status, and trade-management rules during live or paper-trading sessions."

The final paper should use this figure to show that the research system is not only a static backtest. It has a live monitoring layer that connects model probabilities, order policy state, Alpaca execution status, and risk controls.

## Non-ML Engineering Work Still Needed

The machine-learning model is only one part of the trading system. Additional work still needed includes:

- Alpaca API reliability, including reconnect behavior, order-status reconciliation, duplicate-order prevention, and clear handling of paper versus live mode.
- Quote and fill modeling for options, including bid/ask spread, liquidity, slippage, partial fills, and realistic order types.
- Feature-engineering cleanup, including a formal registry of causal features and removal of any indicator that can accidentally look ahead.
- Runtime data validation, including missing-bar detection, stale quote checks, market-hours handling, and corporate-action or calendar edge cases.
- Risk controls, including max daily loss, max position size, max number of trades, cooldowns after losses, and kill-switch behavior.
- Logging and audit trails, including a clear record of every signal, rejected signal, order attempt, fill, exit, and dashboard state.

## Next Research Steps

1. Keep the single `swing_support_single` model as the primary entry source.
2. Continue validating the body-and-close order policy family, especially the best saved non-shift setup-area body-and-close rows.
3. Add option flow and GEX behavior as contextual features or filters, then test whether they improve entry quality without overfitting.
4. Add more of Luke's references and convert the literature section into a consistent citation format.
5. Write the formal methodology section described above.
6. Add the live trading dashboard figure section.
7. Add the non-ML engineering section covering Alpaca, execution, feature engineering, and runtime safety.

## Future Research

Multi-ticker swing extension: The current repo already contains processed 30-minute feature files for many tickers. A future research path is to adapt the swing-support framework beyond SPY and test whether the same labeling and order-policy ideas work across equities, ETFs, leveraged ETFs, and sector instruments.

Earnings guidance: Another future section should study whether earnings dates, guidance revisions, analyst estimate changes, and post-earnings drift can improve swing setup quality. This likely belongs more to swing trading than intraday SPY trading, but it may become useful for the multi-ticker system.

Momentum expansion: A smaller future research path is to test whether momentum expansion signals can improve trade selection after the setup model fires. This should be treated as an add-on filter rather than a replacement for the current setup model.

Option flow and GEX behavior: Option flow, gamma exposure, dealer positioning proxies, and unusual volume may help explain when SPY moves have enough force to continue. These features should be added carefully because they can be noisy, vendor-dependent, and easy to overfit.

## Limitations

The research is still in progress. Several results are from saved backtests and replays rather than live trading. The latest full-fit model has no validation split because it is intended for deployment. The broad replay summaries and the phase-4 ATR scoreboards use different units, so they should not be mixed without care. Some high-performing filters may be overfit until confirmed with walk-forward testing. Options execution also introduces spread, fill quality, and liquidity risks that are not fully captured by underlying-price evaluation.

## Working Conclusion

The project has moved from a broad idea about using machine learning for trading into a more specific and testable architecture. The current evidence suggests that the best path is not a more complex model by itself. The best path is a clean causal feature set, a swing-support XGBoost setup model, side-specific thresholds, and a disciplined 1-minute confirmation order policy. The biggest lesson so far is that model accuracy is only the first layer. The trading system improves when the model is judged by whether it produces entries with positive expected value against a concrete baseline.

## Literature References

- Kumbure, M. M., Lohrmann, C., Luukka, P., and Porras, J. (2022). *Machine learning techniques and data for stock market forecasting: A literature review*. Expert Systems with Applications, 197, 116659.
- Rezaei, A., Abdellatif, I., and Umar, A. (2025). *Towards Economic Sustainability: A Comprehensive Review of Artificial Intelligence and Machine Learning Techniques in Improving the Accuracy of Stock Market Movements*. International Journal of Financial Studies, 13(1), 28.
- *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*.
- GA plus XGBoost feature-selection research reference used in the original notes.
- Recent hybrid-model references in the original notes: CNN/BiLSTM, LSTM/GRU, multi-timeframe LSTM, sentiment-fusion models, and deep reinforcement learning for trading.

These references are included here as a compact carry-forward from the existing project notes. The final paper should convert this section into one citation style and add complete bibliographic formatting for every cited source.

## Data And Artifact Sources

The summary above was grounded primarily in the following saved artifacts:

- `Data/models/ga_xgboost/10min/training_run_summary_20260411T075130Z_ga_xgboost_train.json`
- `Data/models/ga_xgboost/10min/training_run_summary_20260413T074850Z_ga_xgboost_train.json`
- `Data/models/ga_xgboost/model_competition_phase4/competition_best_scoreboards.csv`
- `Data/models/ga_xgboost/model_competition_phase4/nonshift_setup_area_l0.42_s0.20_lag2_h16_tp1.5_sl1.0/best_phase4_trigger_scoreboard.csv`
- `Data/models/ga_xgboost/model_competition_phase4_focused/nonshift_setup_area_l0.42_s0.15_lag2_h12_tp1.0_sl0.8/best_phase4_trigger_scoreboard.csv`
- `Data/models/ga_xgboost/10min/analysis/phase4_1m_sr_baseline/phase4_trigger_scoreboard.csv`
- `Data/inference/spy/10min/setup/verification_current_best_2026-05-03.md`
- `Data/inference/spy/10min/setup/verify_current_2024plus_narrow_summary.csv`
- `Data/inference/spy/10min/setup/probability_normalization_experiment_2024plus_selected_full_summary.csv`
- `Data/models/ga_xgboost/10min_shift1/training_run_summary.json`
- `Data/models/meta_xgboost/10min/entry/training_run_summary.json`
- `Data/models/meta_xgboost/10min/exit/training_run_summary.json`

## Notes For Final Paper

- Separate predictive metrics from trading metrics.
- Keep the early classification baseline, but make clear that the trade-level baseline is now more important.
- Add a dedicated explanation of why DPO and other non-causal features were removed.
- Add one figure for the model pipeline and one figure for the order-policy pipeline.
- Add the live dashboard screenshot and explain how it connects research outputs to runtime monitoring.
- Keep probability normalization as an experimental calibration note, not a current headline result.
