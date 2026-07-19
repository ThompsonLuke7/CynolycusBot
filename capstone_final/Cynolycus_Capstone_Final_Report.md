# Engineering an Auditable AI-Assisted Market Decision-Support and Trading Platform

Luke Thompson  
Master of Engineering in Artificial Intelligence  
July 2026

## Abstract

Financial markets combine noisy observations, changing regimes, and real-time execution constraints; a useful artificial-intelligence signal must survive all three. This capstone began as an intraday SPY system intended to identify short-term swing opportunities and reduce discretionary decision-making. Through repeated experiments, it developed into CynolycusBot, a modular market-analysis and trading platform that joins causal data processing, cross-sectional ranking, news and thematic context, historical simulation, live inference, broker integration, persistent audit records, and an interactive dashboard. The strongest model evidence came from walk-forward out-of-fold evaluation with a 21-day embargo. Momentum and meta-upside rankers concentrated much larger fixed-horizon returns in their highest-ranked candidates than in the broader universe. Policy audits then translated that separation into narrower execution rules; the final frozen 4-hour momentum configuration reached a 77.4% win rate, while the long-only higher-timeframe policy reached 56.9% and improved expected value per simulated trade from 1.45% to 5.01%. These figures are historical, fixed-notional simulations, not expected account returns. The present platform processes live data, generates inference features, supports paper trading, and contains separately gated broker-order paths. Its central result is an integrated and inspectable engineering system; equally important, the project established a disciplined process for converting promising experiments into more selective operational capability.

## 1. Introduction

Turning a market observation into a dependable decision is harder than predicting whether one price will rise. The signal must arrive on time, remain meaningful as volatility changes, and pass through execution controls without becoming stale or unsafe. Financial series are also nonstationary; relationships that appear durable in one interval can weaken abruptly in the next. Reviews of artificial intelligence in financial forecasting reach a similar conclusion: model choice matters, but data construction, feature design, and evaluation discipline often matter more [1], [2].

The project's first objective was intentionally narrow. It sought to identify the beginning of an intraday SPY swing, connect that inference to streaming data, and eventually automate part of the trading workflow. The motivation was practical: discretionary decisions are limited by attention, timing, and emotion. An early prototype already joined bars, technical features, model inference, broker authentication, guardrails, and a web interface; however, it still treated one instrument and one prediction task as the center of the problem.

That assumption changed. Exact pivot labels proved brittle, sequence experiments did not consistently generalize, and a proximal policy optimization study demanded more interaction data and a richer reward environment than the project could reliably supply [10]. Those outcomes were useful. Each one reduced ambiguity, exposed a design constraint, and directed effort toward a formulation that better matched the available evidence.

CynolycusBot consequently evolved into a set of specialist modules operating across a broad equity universe and several horizons. The final system includes a 30-minute swing pipeline, 4-hour momentum and higher-timeframe (HTF) rankers, a contextual meta-ranker, news and thematic analysis, backtesting, live feature generation, broker routing, shared execution controls, persistent state, and dashboard monitoring. Figure 1 presents the current architecture.

The capstone's principal contribution is therefore broader than stock-price prediction. It demonstrates how data engineering, machine learning, experimental design, real-time software, and risk-aware operations can be assembled into one decision-support platform; it also shows how an ambitious prototype can improve through deliberate testing instead of being frozen around its first encouraging result.

## 2. Methods

### 2.1 Architecture, data, and causal features

CynolycusBot is organized as specialist pipelines connected through shared services. External and stored sources provide bars, quotes, option chains, news, filings, scheduled events, short-volume data, macroeconomic context, and local research artifacts. The shared layer maintains market calendars, symbol universes, completed-bar aggregation, caches, readiness stamps, and job guards; Appendix H lists the data sources and external services used during development.

Each strategy builds a purpose-specific feature set, then returns ranked candidates to the operating layer. Momentum, HTF swing, and the meta-ranker feed a common 4-hour execution engine. That engine applies module-specific gates, checks freshness, chooses an equity or option route, sizes the intended position, records the decision, and reconciles local state with the broker. A combined server shares one Alpaca stream, schedules isolated jobs, exposes HTTP and WebSocket interfaces, and supervises data refreshes. This separation made individual experiments replaceable without dismantling the platform around them.

Feature pipelines combine trend, momentum, volatility, average true range, volume, dollar liquidity, relative strength, distance from recent highs, market regime, and lagged daily or weekly context. Rolling and exponentially weighted calculations use only information available at the decision time; higher-timeframe features are shifted before being mapped to intraday rows. Incomplete forward windows are excluded. These rules sound modest, yet they transformed the data layer from a collection of indicators into a repeatable causal process.

### 2.2 Targets, models, and contextual intelligence

The target formulation changed with the engineering problem. The SPY pipeline moved from exact pivots and triple-barrier variants toward fuzzy swing zones followed by deterministic one-minute confirmation. The 30-minute multi-ticker system used multiclass swing labels; the 4-hour modules emphasized forward expansion, persistence, drawdown, and cross-sectional relevance. Rather than require one model to predict every row correctly, ranking asked a narrower question: which candidates are strongest now?

XGBoost and LightGBM became the principal tabular model families because they handle nonlinear thresholds, missing values, and heterogeneous inputs efficiently [3], [4]. Learning-to-rank measures also matched the product decision more closely than global accuracy [5]. The meta layer combined specialist out-of-fold scores with liquidity, regime, theme, news, macroeconomic, and event context. News processing used semantic representations and FinBERT sentiment [6]; dynamic themes clustered article and company-profile embeddings with HDBSCAN, then assigned human-readable taxonomy metadata [7]. Context enriched the operator's view, while independently evaluated rankers remained the foundation for order selection.

The literature review intentionally covered a wider design space than the final implementation. It included inverted Transformers, genetic feature selection with XGBoost, attention-based recurrent models, price-and-news ensembles, multi-timescale networks, sentiment fusion, deep reinforcement learning, transductive LSTM, portfolio ensembles, Monte Carlo integration, and loss-landscape analysis [11]-[21]. These sources influenced experimentation and interpretation; they are not presented as components that all reached production status.

### 2.3 Validation, simulation, and policy refinement

Evaluation became progressively stricter. Major rankers produced chronological walk-forward out-of-fold (OOF) predictions with a 21-day embargo; policy choices were fixed from pre-test folds and validation before a one-shot frozen test. Artifact fingerprints and locked headline metrics prevent ordinary data refreshes from silently rewriting the evidence. This mattered because repeated backtest selection can make a configuration look stronger than the underlying relationship [8]. Random search remained useful for exploration [9], but it did not replace temporal separation.

The audit was an improvement mechanism, not a postmortem. It showed that the HTF model's lowest-ranked names were poor short candidates; the policy therefore removed shorts, retained high-conviction longs, widened exits, and reduced breadth. Momentum also became more selective. OOF top-K returns measured signal concentration, while event-driven simulations applied take-profit, stop-loss, maximum-hold, conviction, and ranking gates. A random-top-K control ran through the same engine, separating what the execution rules contributed from what the learned ordering added.

Historical outputs use a fixed $1,000 notional per signal, a $100,000 display base, unconstrained concurrent positions, and profit-and-loss booking at exit. The final 4-hour study includes a 20-basis-point sensitivity; other headline comparisons omit complete cost and slippage models. Consequently, the plotted curves describe additive strategy P&L under the test convention, not compounded account growth.

### 2.4 Live operation and MEng application

Operational testing made execution quality a first-class concern. Option spreads, decay, expiration, restored state, rejected orders, and partial fills can overwhelm a correct directional idea; broker-authoritative reconciliation, managed-position ledgers, entry-readiness checks, exit fallbacks, and signal-to-order audits were therefore incorporated into shared services. A memory failure during concurrent universe-wide jobs produced another concrete redesign: single-flight guards, resource thresholds, staggered schedules, subprocess isolation, and supervised recovery.

The MEng curriculum contributed directly to feature engineering, class weighting, embeddings, clustering, model comparison, evaluation design, software architecture, and deployment planning. Generative-AI assistants accelerated coding and debugging; the author retained responsibility for hypotheses, experimental choices, validation, and interpretation. The result is not a notebook demonstration. It is a working research-to-operation pipeline with explicit boundaries between evidence, policy, and execution.

## 3. Results

Walk-forward ranking produced the clearest model-level evidence. From November 14, 2022, through May 14, 2026, the momentum ranker's highest score decile averaged a 4.59% fixed-horizon close return, compared with 1.47% for the full ranked universe. Meta-upside showed the same pattern from September 16, 2024, through May 14, 2026: 6.28% in its top decile versus 1.90% overall. Figure 2 visualizes both results. Momentum's top ten candidates per 4-hour bar averaged 6.69%, compared with 1.37% outside the top ten; meta-upside reached 9.33% versus 1.78% across 762,932 OOF rows.

The clean frozen-policy backtests converted that ranking ability into a striking cumulative comparison. Under the fixed-notional convention, HTF produced approximately $336,000 in additive P&L, momentum produced $93,000, and the 30-minute swing system produced $18,000; SPY buy-and-hold gained about $29,000 on the same $100,000 display base. Figure 3 shows the complete curves and their different test windows. The chart is compelling, but its interpretation must remain precise: capital was not constrained across simultaneous trades, so the strategy lines are evidence of historical selection and policy behavior rather than attainable account returns.

Audit-informed redesign improved the two strongest 4-hour policies again. Momentum moved from 3,876 baseline trades at a 74.7% win rate and 2.39% expected value per trade to 1,496 more selective trades at 77.4% and 3.76%; reported maximum drawdown improved from 15.3% to 9.7%. HTF changed more dramatically. Its long-only policy raised frozen-test win rate from 39.0% to 56.9%, expected value from 1.45% to 5.01%, and profit factor from 1.49 to 1.89 across 2,723 simulated trades. Figure 4 makes the policy changes visible.

Randomized ranking controls clarified the source of those gains. Momentum's random top-five control achieved a 72.27% win rate but only 0.733% average trade, compared with 74.74% and 2.387% for the learned ranking. HTF's random top-20 control recorded 34.98% and 0.136%, versus 38.99% and 1.450%. The execution policy supplied much of the consistency; the model contributed the larger opportunity magnitude.

Appendices E and F show intentionally selected high-performing trades from the one-shot frozen window. They illustrate what the policies were designed to capture, including a +37.47% momentum take-profit winner and a +114.37% HTF example; they do not represent an average outcome. Operationally, the platform has also demonstrated live-data ingestion, feature generation, inference, paper trading, brokerage connectivity, audit logging, and dashboard delivery. Realized live profitability has not been established.

## 4. Discussion

The project became strongest when it stopped treating the market as one universal prediction problem. Cross-sectional ranking created a more useful decision surface; the model could concentrate on a small group of candidates instead of being equally confident everywhere. OOF decile lift, top-K separation, frozen equity comparisons, and randomized controls all support that choice. More importantly, the platform can now carry a score through causal data preparation, policy gates, broker reconciliation, persistent state, and operator monitoring.

The audit also produced a positive engineering lesson. Removing HTF shorts was not an admission that the system failed; it was a decision to stop forcing symmetry where the data showed none. The same logic narrowed momentum entries and preserved winners for a horizon consistent with the signal. Every discarded path reduced wasted complexity, and each retained rule gained a clearer empirical reason to exist.

Several limits still define the next stage. Model-family and seed selection were not nested inside every walk-forward split; some historical inputs require stronger point-in-time reconstruction. The simulator does not yet enforce portfolio capital, buying power, concurrency, full option liquidity, fill probability, or complete costs. Paper and live histories remain too short for a profitability claim. These constraints are substantial, although they no longer obscure the project's direction.

The roadmap is concrete. Persistent services and data should move to Google Cloud Platform so collection, inference, monitoring, and recovery can operate continuously. A 24/7 deployment should add immutable snapshots, transactional state, atomic updates, continuous integration, and long-duration soak testing; frozen momentum and long-only HTF policies should remain in shadow or paper mode while that record grows. Portfolio-level exposure limits, realistic slippage, option spreads, decay, and partial fills come next. Only after those gates are satisfied should a small, deliberately capped real-money allocation be considered.

Overall, the capstone demonstrates graduate-level applied AI by unifying machine learning, data engineering, experimental discipline, real-time systems, risk-aware execution, and human-facing monitoring. The results are promising; the platform behind them is the larger achievement. CynolycusBot now has a defensible signal foundation, an explicit audit trail, and a credible path from supervised research system to persistent decision-support service.

## Bibliography

[1] M. M. Kumbure, C. Lohrmann, P. Luukka, and J. Porras, "Machine learning techniques and data for stock market forecasting: A literature review," *Expert Systems with Applications*, vol. 197, art. 116659, 2022, doi: 10.1016/j.eswa.2022.116659.

[2] A. Rezaei, I. Abdellatif, and A. Umar, "Towards economic sustainability: A comprehensive review of artificial intelligence and machine learning techniques in improving the accuracy of stock market movements," *International Journal of Financial Studies*, vol. 13, no. 1, art. 28, 2025, doi: 10.3390/ijfs13010028.

[3] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2016, pp. 785-794, doi: 10.1145/2939672.2939785.

[4] G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in *Advances in Neural Information Processing Systems 30*, 2017.

[5] C. J. C. Burges et al., "Learning to rank using gradient descent," in *Proc. 22nd Int. Conf. Machine Learning*, 2005, pp. 89-96, doi: 10.1145/1102351.1102363.

[6] D. Araci, "FinBERT: Financial sentiment analysis with pre-trained language models," *arXiv:1908.10063*, 2019, doi: 10.48550/arXiv.1908.10063.

[7] L. McInnes, J. Healy, and S. Astels, "hdbscan: Hierarchical density based clustering," *Journal of Open Source Software*, vol. 2, no. 11, art. 205, 2017, doi: 10.21105/joss.00205.

[8] D. H. Bailey, J. M. Borwein, M. Lopez de Prado, and Q. J. Zhu, "The probability of backtest overfitting," *Journal of Computational Finance*, vol. 20, pp. 39-69, 2017, doi: 10.21314/JCF.2016.322.

[9] J. Bergstra and Y. Bengio, "Random search for hyper-parameter optimization," *Journal of Machine Learning Research*, vol. 13, pp. 281-305, 2012.

[10] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," *arXiv:1707.06347*, 2017, doi: 10.48550/arXiv.1707.06347.

[11] Y. Liu et al., "iTransformer: Inverted Transformers are effective for time series forecasting," in *Proc. 12th Int. Conf. Learning Representations*, 2024.

[12] K. K. Yun, S. W. Yoon, and D. Won, "Prediction of stock price direction using a hybrid GA-XGBoost algorithm with a three-stage feature engineering process," *Expert Systems with Applications*, vol. 186, art. 115716, 2021, doi: 10.1016/j.eswa.2021.115716.

[13] Z. Liu, "Improving stock price forecasting with M-A-BiLSTM: A novel approach," *Frontiers in Applied Mathematics and Statistics*, vol. 11, art. 1588202, 2025, doi: 10.3389/fams.2025.1588202.

[14] Y. Li and Y. Pan, "A novel ensemble deep learning model for stock prediction based on stock prices and news," *International Journal of Data Science and Analytics*, vol. 13, no. 2, pp. 139-149, 2022, doi: 10.1007/s41060-021-00279-9.

[15] Y. Hao and Q. Gao, "Predicting the trend of stock market index using the hybrid neural network based on multiple time scale feature learning," *Applied Sciences*, vol. 10, no. 11, art. 3961, 2020, doi: 10.3390/app10113961.

[16] M. K. Daradkeh, "A hybrid data analytics framework with sentiment convergence and multi-feature fusion for stock trend prediction," *Electronics*, vol. 11, no. 2, art. 250, 2022, doi: 10.3390/electronics11020250.

[17] A. L. Awad, S. M. Elkaffas, and M. W. Fakhr, "Stock market prediction using deep reinforcement learning," *Applied System Innovation*, vol. 6, no. 6, art. 106, 2023, doi: 10.3390/asi6060106.

[18] A. Peivandizadeh et al., "Stock market prediction with transductive long short-term memory and social media sentiment analysis," *IEEE Access*, vol. 12, pp. 87110-87130, 2024, doi: 10.1109/ACCESS.2024.3399548.

[19] M. L. Narayana et al., "Ensemble time series models for stock price prediction and portfolio optimization with sentiment analysis," *Journal of Intelligent Information Systems*, vol. 63, pp. 1079-1103, 2025, doi: 10.1007/s10844-025-00928-6.

[20] A. Deep, "Advanced financial market forecasting: Integrating Monte Carlo simulations with ensemble machine learning models," *Quantitative Finance and Economics*, vol. 8, no. 2, pp. 286-314, 2024, doi: 10.3934/QFE.2024011.

[21] H. Li, Z. Xu, G. Taylor, C. Studer, and T. Goldstein, "Visualizing the loss landscape of neural nets," in *Advances in Neural Information Processing Systems 31*, 2018.

\pagebreak

# Appendix A. Current System Architecture

**Figure 1. Current architecture of the AI-assisted market decision-support and trading platform.** Specialist models share data-readiness, execution, broker-reconciliation, audit, scheduling, and dashboard services; research and contextual modules remain distinct from operational order paths.

![Figure 1. Current system architecture](figures/Figure_1_Architecture.png)

\pagebreak

# Appendix B. Best Walk-Forward Ranker Signals

**Figure 2. Two strongest walk-forward ranker examples.** Momentum and meta-upside produced substantially larger fixed-horizon mean close returns in their highest score deciles than in their full ranked universes. Predictions were generated out of fold with a 21-day embargo; overlapping outcomes make these signal-quality plots, not tradable equity curves.

![Figure 2. Best walk-forward ranker signals](figures/Figure_2_Best_Ranker_Signals.png)

\pagebreak

# Appendix C. Frozen Backtest Comparison

**Figure 3. Clean frozen-test policy curves compared with SPY buy-and-hold.** Policies were selected on validation and frozen on test. The strategy lines book additive profit and loss from fixed $1,000 signals with unconstrained concurrency; they are not compounded portfolio returns. Different modules begin on different test dates, as labeled in the source figure.

![Figure 3. Frozen backtest comparison](figures/Figure_3_Frozen_Backtest_Comparison.png)

\pagebreak

# Appendix D. Audit-Informed Policy Redesign

**Figure 4. Pre-test-selected redesign improved both major 4-hour modules on their one-shot frozen tests.** Momentum became more selective; HTF removed negative-expected-value short entries and adopted a long-only, high-conviction policy. Values are fixed-notional historical simulations, not expected account returns.

![Figure 4. Audit-informed policy redesign](figures/Figure_4_Audit_Informed_Policy_Redesign.png)

\pagebreak

# Appendix E. Momentum Frozen-Test Trade Examples

**Figure 5. Two intentionally selected high-performing momentum trades from the one-shot frozen test.** The examples show a +37.47% take-profit winner and a +17.72% time-stop winner. They illustrate the opportunity profile captured by the policy; they do not represent the average trade or imply comparable future outcomes.

![Figure 5. Momentum frozen-test trade examples](figures/Figure_5_Momentum_Frozen_Trade_Examples.png)

\pagebreak

# Appendix F. HTF Frozen-Test Trade Examples

**Figure 6. Two intentionally selected high-performing HTF trades from the one-shot frozen test.** The long-only policy captured a +114.37% take-profit example and a +43.87% time-stop example. These are selected historical winners, included to make the strategy behavior concrete rather than to characterize the outcome distribution.

![Figure 6. HTF frozen-test trade examples](figures/Figure_6_HTF_Frozen_Trade_Examples.png)

\pagebreak

# Appendix G. Dashboard Screenshot Placeholder

**Figure 7. Operational dashboard placeholder.** Replace this panel with an author-provided screenshot of the combined web server before submission. The final capture should show live-data status, ranked candidates, and audit or operator controls without exposing credentials, account identifiers, or sensitive brokerage information.

![Figure 7. Dashboard screenshot placeholder](figures/Figure_7_Dashboard_Screenshot_Placeholder.png)

\pagebreak

# Appendix H. Data Sources and External Services

**Table A1. Data sources and external services used by the project.** Items are grouped by role; inclusion does not imply equal coverage or maturity across every module.

| Category | Sources and services |
|---|---|
| Market and brokerage | Alpaca; Schwab; Polygon; Yahoo Finance / yfinance; CBOE |
| Company, regulatory, and market structure | SEC EDGAR; FINRA; NASDAQ; Financial Modeling Prep; Finnhub |
| Macroeconomic and public data | Federal Reserve; FRED; U.S. Treasury; OpenFDA; ClinicalTrials.gov; USAspending.gov |
| News, events, and social context | Google News RSS; TradingView; PullPush; Reddit |
| AI-assisted development and taxonomy | OpenAI; Anthropic Claude |
