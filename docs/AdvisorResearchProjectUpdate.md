# Capstone Research Project Update: Machine Learning Trading System Direction

Author: Luke Thompson  
Program: Artificial Intelligence capstone project  
Draft purpose: advisor update and revised project-scope proposal  
Date: June 3, 2026

## Short Advisor Message

I wanted to check whether my current project direction is approved and also give an update on what I have learned from the latest live-testing results. My original project focused on an intraday SPY trading agent using machine learning to identify long, short, and neutral setup conditions from 10-minute and 1-minute market data. The research pipeline is working from an AI/ML perspective, but the live trading evidence suggests that the SPY daytrader is not currently performing well enough as a trading system. Backtests showed promising results, but live performance has been weak, especially after realistic option execution, bid/ask spread, timing delay, and directional correctness were evaluated.

Because of that, I would like to propose shifting the main applied system from the single-symbol SPY daytrader to a multi-ticker swing trading model. The multi-ticker swing model uses the same core logic as the SPY daytrader, but expands it across multiple tickers and moves it to a higher timeframe. Instead of predicting only short-term SPY intraday moves, it applies the same machine-learning setup-detection idea to a broader universe of stocks and ETFs using 30-minute swing labels, cross-ticker features, probability outputs, and live inference. This version has shown stronger practical results, especially on long/call-side trades.

I can still include the SPY daytrader as an important baseline and negative/limited-result case study. That may actually strengthen the final report because it shows that I evaluated the model beyond backtest metrics and learned where the data and execution assumptions failed. The revised project would therefore focus on the same AI pipeline, model design, feature engineering, labeling, validation, and inference system, while presenting the multi-ticker swing model as the better-performing application of that same approach.

## Recommended Project Framing

The strongest final capstone framing is:

**Machine Learning for Multi-Ticker Swing Trade Signal Detection and AI-Assisted Trade Selection**

This framing keeps the original AI research question intact: can a machine-learning system learn tradable market setup patterns from price, volatility, trend, and context features? It also lets the report honestly discuss the SPY intraday model as an initial experiment whose backtest-to-live gap revealed important limitations.

The revised project would include three layers:

1. **SPY intraday daytrader as the original baseline**
   - Single-symbol SPY setup model.
   - 10-minute machine-learning probabilities.
   - 1-minute confirmation order policy.
   - Useful backtest results but weak live results.
   - Key lesson: candle-only intraday SPY features may not contain enough live alpha once option execution is modeled.

2. **Multi-ticker swing model as the primary working system**
   - Same core setup-detection logic as the SPY daytrader.
   - Expanded from one ticker to many tickers.
   - Moved from intraday scalping to a higher 30-minute swing timeframe.
   - Uses soft swing labels, engineered technical/context features, XGBoost classification, selected features, probability thresholds, inference, and trade/risk policy.
   - Current evidence is stronger on long/call-side trades than on puts/shorts.

3. **Higher-timeframe momentum/catalyst/pivot system as an extension**
   - Uses 4-hour and daily context to identify larger swing opportunities.
   - Adds momentum expansion, catalyst detection, theme/context features, and pivot/base-reclaim structure.
   - Still under development, but it is a natural research extension if completed and validated before the final report.

## Why The SPY Daytrader Is Not The Best Final Headline

The SPY daytrader is still valuable academically, but it is probably not the best final project headline if the goal is to showcase a strong AI system.

The main issue is not that the model pipeline failed technically. The issue is that the live trading environment is more difficult than the simplified backtest:

- SPY options are sensitive to bid/ask spread, theta decay, IV changes, and fill timing.
- The saved backtests often evaluated underlying-price movement rather than the exact live option contract economics.
- Live directional correctness appears close to noise in recent measurements.
- A 1-minute/10-minute candle-only feature set may not be enough for SPY/SPX options day trading without options flow, order book, gamma exposure, or futures order-flow data.

This is still a legitimate research result. It shows that a model can look promising in backtests while failing under realistic live constraints. However, if the project is meant to demonstrate the applied AI pipeline and show a working empirical result, the multi-ticker swing system is a better final artifact because the same ML approach has performed better in real/paper trading.

## Why The Multi-Ticker Swing Model Is A Stronger Capstone Project

The multi-ticker swing model is a better capstone candidate because it preserves the same core AI problem while improving the practical trading setup:

- It uses the same model-plus-policy architecture as the SPY daytrader.
- It generalizes the approach from one ticker to a multi-asset universe.
- It uses higher-timeframe 30-minute data, which is less sensitive to tiny execution errors than 1-minute option scalping.
- It allows cross-sectional signal comparison, not just single-symbol prediction.
- It applies the same ML tools to more tickers, a higher timeframe, and a wider opportunity set.
- It has shown better real-world results on call-side swing trades than the SPY intraday model has shown live.

This version does not require a fundamentally different ML pipeline than the SPY daytrader. It showcases the same artificial intelligence coursework skills, but in a setting where the empirical results are currently better:

- Feature engineering from OHLCV, trend, volatility, range expansion, relative strength, time, and market-context features.
- Label engineering using soft swing zones instead of exact pivot prediction.
- Supervised learning with an XGBoost multi-class classifier.
- Hyperparameter configuration and model selection.
- Feature selection and feature-importance analysis.
- Train/test separation and validation against baseline trading rules.
- Probability-based inference rather than hard-coded rules alone.
- Live/paper trading integration with audit logs, model probabilities, entry decisions, and risk controls.
- Post-hoc error analysis comparing backtest assumptions with live trade outcomes.

## Preliminary Multi-Ticker Performance Evidence

The main reason I am proposing the multi-ticker swing model as the final applied system is that it appears to be directionally stronger than the SPY daytrader. The SPY daytrader's recent live directional correctness appears close to 50%, which means it is not clearly separating long and short opportunities in live trading. By contrast, the multi-ticker swing backtest and recent live/paper audit results show better directional behavior, especially on long/call setups.

Current multi-ticker evidence to summarize in the report:

- In the saved multi-ticker swing backtest grouping, long-side trades showed about **62.6% directional win rate** across **4,690 long trades**.
- Short-side trades showed about **60.0% directional win rate** across **4,584 short trades** in the same saved grouped backtest, but the live options implementation has been weaker on puts/shorts.
- Across all grouped long and short backtest trades, the directional win rate was about **62.3%**, which is materially stronger than the roughly noise-level live directional behavior observed in the SPY daytrader.
- In recent live/paper analysis from May 28-29, **fresh closed calls averaged about +40.6% option return** while the underlying stocks moved about **+1.27%** in the predicted direction.
- A broader recent fresh-entry slice showed options averaging about **+28.1%** versus about **+1.22%** for the equivalent stock move.
- The long-only capital-normalized audit showed long/call option returns beating stock percentage returns across long-only slices, with recent fresh calls averaging about **+36.2%** versus **+1.40%** for stock.
- The main weakness is not the long/call signal. The main weakness has been puts/shorts, restored/overnight option positions, decay, and execution/routing. This is why the current practical policy direction is calls-only or long-biased until the short-side option policy is improved.

This comparison supports the revised project framing: the original SPY daytrader can be presented as an important baseline and failure-analysis case, while the multi-ticker swing model can be presented as the better-performing generalized application of the same machine-learning pipeline.

### Hold Time And Example Trades

The May 28 through June 1 multi-ticker swing audit logs were rebuilt and analyzed for holding period. Across **157 closed logged trades**, the average hold time was about **151.2 minutes** and the median hold time was about **121.9 minutes**. For the cleaner **38 fresh entries** opened during the audit window, the average hold time was about **125.0 minutes**. Calls averaged about **164.3 minutes**, while puts averaged about **136.9 minutes**.

Top three logged trades by option PnL:

| Ticker | Direction | Entry time (EDT) | Close time (EDT) | Hold | Option PnL | Option return | Underlying signed return | Exit reason |
|---|---:|---|---|---:|---:|---:|---:|---|
| SNOW | Long/call | 2026-05-28 08:05:29 | 2026-05-28 09:35:12 | 89.7 min | +$3,060 | +493.6% | +2.9% | option_take_profit |
| IBM | Long/call | 2026-05-29 08:09:13 | 2026-05-29 09:50:23 | 101.2 min | +$816 | +315.1% | +5.4% | option_take_profit |
| MSFT | Long/call | 2026-05-28 08:05:29 | 2026-05-28 10:15:02 | 129.6 min | +$755 | +361.2% | +3.2% | option_take_profit |

Top three fresh entries by option PnL:

| Ticker | Direction | Entry time (EDT) | Close time (EDT) | Hold | Option PnL | Option return | Underlying signed return | Exit reason |
|---|---:|---|---|---:|---:|---:|---:|---|
| AMD | Long/call | 2026-05-29 14:31:14 | 2026-05-29 15:55:07 | 83.9 min | +$370 | +18.1% | +1.5% | trail |
| TSLA | Long/call | 2026-05-29 11:05:11 | 2026-05-29 12:31:32 | 86.3 min | +$346 | +121.8% | +1.2% | option_profit_trail |
| CRWV | Long/call | 2026-05-29 11:05:26 | 2026-05-29 15:50:06 | 284.7 min | +$216 | +229.8% | +3.7% | expiration_itm_cutoff |

These examples help show the practical difference between the SPY daytrader and the multi-ticker swing system: the multi-ticker trades are not ultra-short scalps. They are generally held for one to three hours, giving the model's higher-timeframe signal more time to work and reducing dependence on perfect 1-minute timing.

## Higher-Timeframe Swing Extension

The higher-timeframe system could become either the final project enhancement or a future-work section, depending on how much validation is completed before the report deadline.

The planned higher-timeframe system adds:

- 4-hour and daily momentum expansion features.
- Pivot and base-reclaim structure.
- Catalyst features from news, SEC filings, analyst actions, biotech/FDA events, options snapshots, short-volume spikes, and macro events.
- Theme/sector leadership context.
- Cross-sectional ranking rather than only binary direction prediction.
- A separate option-worthiness or large-move router for deciding when a signal is strong enough for calls.

This system may ultimately be the most interesting research direction, but it is less finished than the multi-ticker swing model. If the deadline is about 1.5 months away, I would present it as an advanced extension unless it reaches clean validation quickly.

## AI And ML Components To Highlight In The Final Report

The final report should emphasize the AI/ML process more than the trading outcome:

1. **Data processing**
   - Intraday and higher-timeframe bar construction.
   - Multi-ticker feature matrices.
   - Missing-data handling and timestamp alignment.
   - Live inference data validation.

2. **Feature engineering**
   - Trend, volatility, ATR, range expansion, relative strength, market regime, and time-based features.
   - Cross-asset context features.
   - Catalyst, news, and event-derived features for the extension system.

3. **Label engineering**
   - Swing-support and soft swing-zone labels.
   - Forward-return and expansion-survival labels.
   - Trade-aware labels designed around whether a move is large enough to matter.

4. **Machine-learning models**
   - XGBoost classification for setup detection.
   - Cross-sectional ranking/regression for momentum expansion.
   - Embedding and clustering methods for catalyst/narrative classification.
   - Potential meta-models for combining technical, catalyst, theme, and social/context scores.

5. **Model training**
   - Hyperparameter selection.
   - Feature selection.
   - Train/test and walk-forward-style validation.
   - Class imbalance and probability thresholding.

6. **Inference and decision logic**
   - Live probability generation.
   - Rule-based confirmation policy.
   - Calls-only option routing and risk controls.
   - Audit logging and model-output inspection.

7. **Evaluation**
   - Classification metrics such as accuracy, precision, recall, F1, AUC, and log loss.
   - Trading metrics such as win rate, expected value, profit factor, MFE, MAE, drawdown, and trade frequency.
   - Live-vs-backtest discrepancy analysis.

## Proposed Final Project Decision

My recommendation is to ask the advisor for approval to slightly revise the project scope:

**Original:** AI-based intraday SPY daytrading agent.  
**Revised:** AI-based multi-ticker swing trading signal system, with the SPY intraday agent included as the initial baseline and limitation study.

This is not a complete project change. It is a generalization of the same research idea:

- Same core AI question.
- Same model-plus-policy design.
- Same supervised learning approach.
- Same trading-domain application.
- Broader ticker universe.
- Higher timeframe.
- Stronger evidence from live/paper trading.

## What I Think The Advisor Is Likely To Prefer

If the advisor mainly cares about research honesty, he might say it is acceptable to submit the SPY daytrader report and explain that it did not work live. A failed model can still be a valid capstone if the methodology, evaluation, and conclusions are strong.

However, if the goal is to showcase the AI skills learned in the degree program while also presenting a system with better real-life results, the multi-ticker swing model is probably the better final project. It uses the same machine-learning pipeline as the SPY daytrader: feature engineering, labels, model training, hyperparameters, validation, inference, live deployment, error analysis, and iteration from a weak live baseline to a better-performing generalized model.

The safest proposal is therefore:

**Use the multi-ticker swing system as the final project focus, keep the SPY daytrader as the baseline/first experiment, and describe the higher-timeframe momentum/catalyst model as an extension if it is not fully validated in time.**

## Suggested Email Paragraph

I wanted to check whether my current project direction is approved and also give a quick update. My original project focused on an ML-based intraday SPY trading agent. The machine-learning pipeline is functioning, but recent live testing suggests the SPY daytrader is not performing well under realistic option-execution conditions. Because of that, I am considering shifting the main applied system to a multi-ticker swing trading model that uses the same core logic as the SPY daytrader, but expands it across multiple tickers and a higher 30-minute timeframe. This version still showcases the AI components of the project, including feature engineering, label design, XGBoost model training, hyperparameter tuning, probability-based inference, and live/paper evaluation, and it currently appears more promising in practice. I would still include the SPY daytrader as the original baseline and discuss why its backtest results did not translate well to live trading. Would this revised framing be acceptable for the final capstone report?
