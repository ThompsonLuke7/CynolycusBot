# Capstone Final Evidence Audit

This audit maps consequential report claims, reported metrics, and appendix visuals to repository evidence. Repository paths are relative to `CynolycusBot` unless stated otherwise.

## Project claims and quantitative results

| Report claim | Supporting evidence | Qualification |
|---|---|---|
| The project evolved from an intraday SPY prototype into a modular multi-horizon platform. | `capstone_materials/CAPSTONE_RESEARCH_DOSSIER.md`; `CAPSTONE_TIMELINE.md`; `README.md`; `LIVING_SUMMARY.md` | Timeline and current-artifact supported. |
| Shared data-readiness, execution, audit, scheduling, broker-reconciliation, and dashboard services connect specialist modules. | `core/live_4h_exec.py`; `core/live_readiness.py`; `core/live_job_guard.py`; `UI/combined_server.py`; `UI/shared_stream.py` | Current-code verified; Figure 1 is a simplified synthesis. |
| Live data ingestion, feature generation, inference, paper trading, broker paths, audit records, and dashboard delivery are implemented. | `UI/combined_server.py`; `UI/shared_stream.py`; `core/live_4h_exec.py`; `Data/inference/`; `LIVING_SUMMARY.md` | Capability evidence only; not sustained realized live returns or production reliability. |
| Momentum OOF top decile averaged 4.59% versus 1.47% for its ranked universe. | `research/capstone/figures/fig08_oof_decile_lift.png`; momentum `oof_preds.parquet`; `research/capstone/results_lock.json` | Walk-forward OOF with 21-day embargo; overlapping fixed-horizon returns, not an equity curve. |
| Meta-upside OOF top decile averaged 6.28% versus 1.90% overall. | `research/capstone/figures/fig08_oof_decile_lift.png`; meta-upside `oof_preds.parquet`; `research/capstone/results_lock.json` | Same OOF interpretation; Figure 2 is a faithful crop of the locked plot. |
| Momentum top ten averaged 6.69% versus 1.37% outside the top ten. | `research/capstone/results_lock.json`; `capstone_materials/CAPSTONE_RESULTS_CATALOG.md` | Fixed 25-bar 4-hour horizon; 29,335 top-ranked rows; outcomes overlap. |
| Meta-upside top ten averaged 9.33% versus 1.78% outside the top ten. | `research/capstone/results_lock.json`; `signals/meta_context/meta_ranker/models/upside/eval_metrics.json` | 762,932 OOF rows; ranking evidence, not account return. |
| Clean frozen curves ended near +$336k HTF, +$93k momentum, +$18k 30-minute swing, and +$29k SPY. | `research/capstone/figures/fig01_equity_curves.png`; `research/capstone/figures/README.md`; `CAPSTONE_RESULTS_CATALOG.md` | Additive $1,000-per-signal bookkeeping with unconstrained concurrency; module windows differ. |
| Momentum final frozen policy: 1,496 trades, 77.4% win rate, 3.76% EV/trade, PF 1.63, and -9.7% reported maximum drawdown. | `research/capstone/ev_optimization_4h.md`; `CAPSTONE_RESULTS_CATALOG.md` | Configuration fixed on pre-test folds and validation before one test read. |
| HTF final long-only policy: 2,723 trades, 56.9% win rate, 5.01% EV/trade, PF 1.89, and -15.2% reported maximum drawdown. | `research/capstone/ev_optimization_4h.md`; `CAPSTONE_RESULTS_CATALOG.md` | Four-hour HTF, not 30-minute swing; includes a 20-bp sensitivity. |
| Removing HTF shorts improved win rate by 17.9 points and EV/trade from 1.45% to 5.01%. | `research/capstone/ev_optimization_4h.md` | Baseline was two-sided; final policy is long-only top-five/high-conviction. |
| Random controls indicate policy explains much of hit rate while learned ranking improves selected magnitude. | `research/capstone/baselines/random_k_seeds.csv`; `CAPSTONE_RESULTS_CATALOG.md` | Ten seeds used the same execution engine and frozen exits. |
| Selected momentum trade examples are NVTS +37.47% and NKTR +17.72%. | `backtests/ev_experiments_4h/momentum_final_example_trades.png`; `research/capstone/ev_optimization_4h.md` | Deliberately selected winners from the one-shot frozen test; not typical outcomes. |
| Selected HTF trade examples are POET +114.37% and WGS +43.87%. | `backtests/ev_experiments_4h/htf_final_example_trades.png`; `research/capstone/ev_optimization_4h.md` | Same selection caveat; shown to illustrate policy behavior. |
| Future work includes GCP storage/deployment, continuous operation, extended shadow/paper testing, then only conditional limited real-money use. | Author direction; `CAPSTONE_RESEARCH_DOSSIER.md` roadmap; `CAPSTONE_QUESTIONS_FOR_AUTHOR.md` | Presented only as future work. |

## Figure and appendix provenance

| Item | Source | Transformation |
|---|---|---|
| Figure 1, architecture | Current code and dossier architecture map | New simplified diagram; no performance claims added. |
| Figure 2, ranker signals | `research/capstone/figures/fig08_oof_decile_lift.png` | Cropped to the two signals discussed in the report; values unchanged. |
| Figure 3, frozen backtests | `research/capstone/figures/fig01_equity_curves.png` | Reused without changing data or annotations. |
| Figure 4, policy redesign | `research/capstone/ev_optimization_4h.md` and baseline artifacts | New comparison diagram using documented frozen metrics. |
| Figures 5-6, trade examples | `backtests/ev_experiments_4h/*_final_example_trades.png` | Internal experiment-name banner removed; axes, labels, entry/exit marks, and values preserved. |
| Figure 7, dashboard | Author-replacement placeholder | Not evidence; explicitly labeled for replacement and redaction. |
| Appendix H, data sources | `capstone_materials/CAPSTONE_QUESTIONS_FOR_AUTHOR.md`, answer to Question 28 | Consolidated names only; maturity is not implied. |

## InvestingResearchProject.docx source trace

Every research source or linked paper listed in `docs/InvestingResearchProject.docx` appears in the final bibliography. Metadata were checked against DOI/Crossref, publisher, conference, DOAJ, PubMed Central, or repository records.

| InvestingResearchProject label | Final bibliography entry | Verification identifier |
|---|---:|---|
| Stock-market forecasting literature review | [1] | doi:10.1016/j.eswa.2022.116659 |
| Economic-sustainability AI/ML review | [2] | doi:10.3390/ijfs13010028 |
| iTransformer | [11] | ICLR 2024 / arXiv:2310.06625 |
| GA + XGBoost | [12] | doi:10.1016/j.eswa.2021.115716 |
| M-A-BiLSTM | [13] | doi:10.3389/fams.2025.1588202 |
| LSTM + GRU / linked PMC paper | [14] | PMCID:PMC8446482; doi:10.1007/s41060-021-00279-9 |
| Multi-timeframe LSTM | [15] | doi:10.3390/app10113961 |
| Sentiment convergence | [16] | doi:10.3390/electronics11020250 |
| Deep reinforcement learning | [17] | doi:10.3390/asi6060106 |
| Transductive LSTM | [18] | IEEE 10528270; doi:10.1109/ACCESS.2024.3399548 |
| Ensemble with sentiment analysis | [19] | doi:10.1007/s10844-025-00928-6 |
| Monte Carlo integration | [20] | doi:10.3934/QFE.2024011 |
| Loss-landscape visualization | [21] | NeurIPS 2018 / arXiv:1712.09913 |

No external publication is used to substantiate project performance. All reported project metrics trace to repository artifacts listed above.
