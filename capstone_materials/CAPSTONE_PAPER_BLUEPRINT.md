# Capstone Paper Blueprint

This is an evidence-to-section outline, not final prose. The recommended paper frame is an applied-engineering evolution: from SPY prototype to a modular, audited cross-sectional market platform, with the reproducibility correction as a central contribution.

## Working title options

1. **Engineering an Auditable AI-Assisted Market Ranking and Trading Platform**
2. **From Intraday Prediction to Cross-Sectional Ranking: An Applied AI Trading-System Engineering Study**
3. **CynolycusBot: Iterative Design, Validation, and Operation of a Multi-Strategy AI Market Platform**

Avoid titles that promise profitability or production readiness.

## Abstract evidence slots

- **Problem:** discretionary monitoring and execution across noisy, non-stationary equity markets; need causal, auditable decisions.
- **Methods:** time-indexed pipelines, tree/sequence/RL experiments, cross-sectional boosted-tree rankers, NLP/theme/context features, walk-forward OOF, frozen-test policy evaluation, shared execution and dashboards.
- **Strong results:** embargoed momentum/HTF/meta top-K separation; frozen-test policy results under explicitly fixed-notional assumptions; reproducibility audit's quantified bias correction.
- **Honest outcome:** operational paper/live capability but no sustained-profitability claim; paper-option and confluence negative results shaped redesign.
- **Contribution:** integrated engineering process and evidence discipline, not one novel financial model.

## 1. Introduction

### Practical motivation

- Human monitoring limits, emotion and inconsistent execution.
- Why “predict every bar” is the wrong operational objective; ranking rare opportunities plus confirmation/risk is more tractable.
- Constraints: latency, stale data, changing regimes, leakage, option microstructure and broker safety.

### Project scope and contribution

- Initial SPY intraday prototype.
- Evolution to specialist signals and shared execution.
- Applied capstone framing: design, integration, evaluation, operation and lessons.

### Candidate Figure 1

Condensed evolution timeline from [CAPSTONE_TIMELINE.md](CAPSTONE_TIMELINE.md).

## 2. Methods

### 2.1 System architecture

- Data sources: Alpaca, Schwab, SEC/news/events/FINRA/CBOE/local caches.
- Shared bars/universe/calendar; specialist feature/model layers; meta layer; execution/audit/UI.
- Research/paper/live boundaries and dry-run/live flags.

**Candidate Figure 2:** Mermaid architecture from [CAPSTONE_RESEARCH_DOSSIER.md](CAPSTONE_RESEARCH_DOSSIER.md).

### 2.2 Data engineering and time correctness

- Event versus availability versus signal/order/fill time.
- UTC/Eastern/session handling, previous-day daily context, as-of joins, rolling windows.
- Raw/processed/model/audit stores.
- Known limitations: current universe/metadata, calendar history, same-day catalyst leakage.

**Candidate Table 1:** source × frequency × availability × module × point-in-time caveat.

### 2.3 Evolution of labels and features

- Exact pivots/triple barrier → fuzzy setup zones.
- 30m swing labels and 4H momentum/HTF forward outcomes.
- Meta quality/upside labels.
- Technical, volatility, liquidity, relative-strength, regime, theme, news and event features.
- Class weights, soft labels and cross-sectional ranks.

### 2.4 AI/ML methods

- GA-XGBoost baseline and why boosted trees fit heterogeneous tabular data.
- Sequence/RL experiments and why they were not retained as primary.
- XGBoost/LightGBM family/seed competitions and ranking metrics.
- BGE/FinBERT/clustering/LLM taxonomy; clarify that LLM does not trade.
- Rule-based confirmation/execution alongside learned rankings.

### 2.5 Experimental design

- Chronological splits and original weaknesses.
- 21-day-embargoed walk-forward OOF.
- Validation-selected/frozen-test policy correction.
- Non-overlap assessment to reduce outcome dependence.
- Benchmark and transaction-cost conventions.
- Explicitly state non-nested family/seed selection and missing portfolio constraints.

**Candidate Figure 3:** selection-bias correction (`research/capstone/figures/fig04_selection_bias_correction.png`).

### 2.6 Operational engineering

- Shared WebSocket and bounded queues.
- Shared 4H execution with dependency injection.
- Readiness, heavy-job guard, position reconciliation, audit ledgers and exit fallback.
- Supervisor/watchdog/memory incident response.
- Credential/profile boundary and security incident caveat.

**Candidate Figure 4:** live-operation sequence diagram.

## 3. Results

### 3.1 Predictive/ranking evidence

Lead with non-overlapping and WF-OOF results, not accuracy:

- Momentum top-K and 78-window non-overlap assessment.
- HTF top-K and 49-window non-overlap; explain weak short/global correlation.
- Meta-upside versus meta-quality comparison.
- Swing classifier precision/recall and clean policy—more modest result.

**Candidate Figure 5:** `fig08_oof_decile_lift.png`.

**Candidate Figure 6:** `fig09_meta_calibration.png`.

**Candidate Table 2:** OOF/frozen results with effective sample, benchmark and caveats.

### 3.2 Policy/backtest evidence

- Clean frozen swing, momentum and HTF policies.
- State fixed-notional/unconstrained concurrency before numbers.
- Use 20 bp net sensitivity from the later 4H study.
- Do not present additive returns as account CAGR.

**Candidate Figures 7–9:** locked equity, drawdown, regime plots; captions must repeat portfolio limitations.

### 3.3 Execution and operational findings

- Meta rank-dropout versus horizon/target/scale-out.
- Paper option loss despite fresh-call underlying direction.
- Shared execution/readiness tests and operational capability.

**Candidate Figure 10:** `fig11_meta_exit_policy.png`.

**Candidate Figure 11:** `fig12_paper_sessions.png`.

### 3.4 Negative/null results

- SPY realistic result and weak exact-target/meta-entry variants.
- Theme ML marginal/regime-sensitive.
- Forward guidance near-random AUC.
- Catalyst claims currently compromised/conflicted.
- 524-pair confluence null result and power floor.

This subsection is essential evidence of graduate-level experimental judgment.

## 4. Discussion

### Efficacy

- What is actually supported: cross-sectional top-K ranking evidence and an integrated operating platform.
- Why rank quality and execution quality are separate.
- Why a good fixed-horizon label does not imply captured live return.

### Engineering lessons

- Causality and availability outrank apparent accuracy.
- Complexity must earn its operational cost.
- Frozen-test and provenance locks change conclusions.
- Broker-authoritative state and recovery are part of AI-system correctness.
- Negative results prevent false deployment confidence.

### Limitations

- Non-nested model selection; no embargo on some base splits.
- Survivorship/current metadata.
- Overlapping outcomes/effective sample.
- Catalyst point-in-time defects.
- Costs/slippage/options/concurrency/capital.
- Short paper/live histories and operational incidents.
- Mutable data and environment reproducibility.

### Responsible interpretation

- No profitability/production claim.
- Human-facing dashboards show ranks/audits but calibration is imperfect.
- LLM role limited to taxonomy/context.
- Research/paper/live distinction.

## 5. Future work

Order by roadmap importance:

1. Immutable point-in-time datasets and nested/purged model selection.
2. Capital-constrained portfolio and realistic option execution simulator.
3. Extended shadow-paper evaluation of frozen policies.
4. Catalyst availability rebuild and mature final holdout.
5. Longer dealer/CBOE/social histories; later confluence retest.
6. Environment lock, CI, state atomicity and load/soak testing.
7. Credential remediation verification and formal operations runbook.

## 6. Conclusion

One paragraph later: emphasize applied engineering mastery, platform evolution, corrected evidence, and bounded outcome. Do not conclude that the system beats the market live.

## Bibliography plan

Primary-source topic groups:

- financial backtest overfitting, purged/embargoed CV and multiple testing;
- gradient boosting, learning-to-rank and calibration;
- event/triple-barrier labeling and time-series dependence;
- market microstructure, costs, option spreads/decay and execution;
- financial language models, embeddings, semantic retrieval and clustering;
- concept drift/non-stationarity;
- real-time event-driven systems, resilience and observability;
- explainable/responsible AI in financial decision support.

Course texts/readings should be added only after the author supplies course mappings.

## Appendices

- Full evidence index and results catalog.
- Expanded timeline/commit table.
- Model feature/label definitions and split diagrams.
- Commands and test results.
- System architecture/data-flow diagrams.
- Detailed caveat taxonomy.
- Reproduction instructions and artifact hashes.

## Ten-page compression strategy for the later writing phase

If constrained to ten pages, prioritize: problem/evolution (1 page), architecture/method (3 pages), evaluation design (1.5 pages), strongest results plus correction (2 pages), discussion/limitations (1.5 pages), conclusion/future work (1 page). Move inventories, detailed metrics, tests and full timeline to appendices. Use 4–6 figures/tables total, not all 12 existing figures.
