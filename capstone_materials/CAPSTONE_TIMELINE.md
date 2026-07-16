# Detailed Project Timeline

Dates are Git commit dates or dated artifacts; they are approximate development periods, not guaranteed experiment execution timestamps. Git contains 453 commits and many merges/generated artifacts, so primary code/results are cited where possible.

## 2025-11: foundation

- **2025-11-25 — Repository and Schwab authentication.** `8a26d82` initialized the project; `9ab2cf3` added Schwab API authentication. `fe66d5c` (Nov 30) adopted `schwab-py`. Goal: establish market/broker access. Limitation: no mature predictive/evaluation system yet.

## 2025-12: first models and first methodological correction

- **Dec 1–8 — Daily feature/model prototypes.** `b3d4d92` added daily data/features; `6f7c6d8` MABiLSTM; `b72e01c` GA-XGBoost; `6df3f97` broad `pandas_ta`; `fbd4e2a` comparisons/custom time-series features; `7b38ce3` preprocessing/label changes.
- **Dec 9–10 — Leakage correction.** `47d9132` and `5c374d1` explicitly removed/fixed leakage and added a leakage-testing path. This is the first evidence-backed hypothesis→audit→redesign loop.
- **Dec 16–31 — Label and computation refinement.** Label changes, pivot sequence changes, feature caching, and VMD log-return features. Motivation: produce more useful targets and reduce repeated computation.

## 2026-01: intraday generalization and model proliferation

- **Jan 1–9 — From one daily series to ticker/timeframe pipelines.** Alpaca and PPO scaffolding (`bc8cbc9`), ticker-generalized pipeline (`33f0584`), 1-hour SPY and dynamic plots (`e072200`), continuation labels/swing state (`6d70d79`), multi-timeframe concatenation (`1f8e543`), and split/training export work.
- **Jan 10 — Market-session correctness.** `93b63df` fixed out-of-regular-hours data, noting that prior plots and training were wrong. This is an important data-engineering failure and correction.
- **Jan 10–21 — Model/feature iteration.** GPU XGBoost support, support/resistance, MABiLSTM refinements, MFE/MAE/exhaustion labels, scaling modules and GA fitness changes.
- **Jan 22 onward — Reinforcement-learning execution.** `0ca20a2` created the trading environment and RL components. Subsequent commits changed OOF data, NaN handling, rewards, action behavior and artifacts. Result: technically substantial experiment, but unstable and later superseded by deterministic confirmation policies.
- **Jan 26–31 — iTransformer/quantile research.** Sequence model, masking, mixed precision, quantile losses and calibration. Surviving code is classified as legacy/experimental in `docs/PROJECT_STATUS.md`.

## 2026-02 to mid-March: first full SPY operating system

- **Feb 1–3 — Triple barrier and PPO continuation.** Triple-barrier labels entered XGBoost/RL; later evidence shows near-chance validation on this formulation.
- **Feb 4 — Alpaca streaming.** `3cab745` added WebSocket streaming and the live runner.
- **Feb 9–10 — Safety and monitoring.** `ec7b0d3` added order-policy guardrails; `cbd5c5e` added the web UI.
- **Feb 16 onward — 1-minute execution and context.** `2528b32` added 1-minute execution, followed by VIX/cross-time features and many agent/policy experiments.
- **Evidence-based outcome.** The system became end-to-end—data, features, inference, options/order policy, replay/live monitoring—but model/reward instability and weak realistic results prevented a production claim.

## late March to April 13: setup detection plus price confirmation

- **Late Mar — Fuzzy setup labels.** `4fc800a` introduced fuzzy swing-setup labeling rather than exact pivot prediction.
- **Early Apr — Trigger-policy comparison.** `a917f33` selected a trigger-policy direction. The durable architecture became a 10-minute multiclass GA-XGBoost setup detector plus a deterministic 1-minute body/close confirmation and ATR management policy.
- **Why redesign:** exact timing labels, triple barrier, meta-entry and simple breakouts were weak or unstable; model metrics alone did not translate to trades. `docs/ResearchPaperSummarySoFar.md` preserves long/short AUCs near 0.529/0.514 for a triple-barrier attempt and meta-entry AUC near 0.545/0.532.
- **Apr 11–13 — SPY artifacts.** Validation and full-fit setup artifacts were saved. Later capstone framing treats SPY as the baseline/limitation study rather than the strongest current system.

## Apr 16 to May 18: expansion beyond SPY

- **Apr 16 — Multi-ticker swing.** `2088503` created a broader 30-minute swing research track and expanded dashboard/audit behavior.
- **May — Cross-sectional model and live evolution.** `27baf7e` added multi-ticker GA-XGB; `09ac327` moved live inference to 5-minute files; `b7afdb7` added a 124-feature SPY artifact.
- **May 18 — Momentum expansion.** `f2bfd3c` launched broader 4-hour continuation/ranking work. Strategic motivation: select rare strong names cross-sectionally rather than repeatedly predict one index.

## May 19–31: multi-signal platform and negative-result-driven redesign

- **Themes.** `821f193` began theme rotation; `4f2a7f1`, `59a8188`, `615c497` expanded backtests, labels and diagnostics. Tests found top-three themes failed in the 2025 Q2 rebound, top-five/cash-off was stronger, and weak themes were rebound candidates rather than reliable shorts. The pipeline was later moved to `themes/theme_expansion_legacy/`, though its maps remain live dependencies.
- **Catalysts/events/meta.** `2364756` scaffolded news, scheduled events, forward guidance and meta context; `3038f34` added earnings/SEC enrichment; `4073699` live news scoring. Early corpus labels called “10d” effectively represented roughly 50 trading days on 4H bars, transcripts were sparse, and high-alpha categories were under-covered. Collection/body/horizon/OOF logic was redesigned.
- **Social attention.** `6217f9c` added Reddit enrichment and analysis; current repository has code/tests but not enough historical evidence for an edge.
- **Paper option forensics.** May 28–29 ledgers found fresh calls could rise sharply while the overall restored/closed option book lost money. Spreads, puts, expiration/decay, restored state and exit behavior—not only direction—became first-class engineering concerns.

## June 1–10: broader sources, stronger models, and architecture reorganization

- **Jun 1 — Market context sources.** `73a2344` added CBOE options snapshots, FINRA short-volume history and company profiles.
- **Jun 2–6 — HTF/momentum/meta training.** Cross-sectional 4H bundles and OOF evaluation were added. A Jun 6 report identified strong OOF ranking for momentum/HTF, weak theme ML, and a degenerate one-tree HTF final export. The current 2026-06-15 LightGBM HTF artifact supersedes that exact export failure, but the incident motivated safer final-fit logic.
- **Jun 8–9 — News model and leakage correction.** Body extraction, source quality, buzz and trajectory models improved the pipeline; a `predict` versus `predict_proba` evaluation bug was corrected. Expanding-window OOF news predictions replaced in-sample predictions for training-slice meta features.
- **Jun 10 — Dynamic taxonomy.** `712c540` implemented documents→embedding→HDBSCAN→summary→LLM naming→discovery→relationships→memberships→meta features. LLM output is taxonomy metadata, not a trading decision.
- **Jun 10 — Repository reorganization.** `8dbd648` moved code into `core/`, `signals/`, `themes/`, `strategies/`. `docs/REPO_CLEANUP.md` records about 4,747 generated artifacts and OAuth token files historically committed. This is both a software-maturity milestone and a security/reproducibility failure.

## June 10–28: operational integration and incident response

- **Swing startup/restore problems.** Sessions sometimes missed scans; restored broker positions dominated marked losses; shared streaming over roughly 925 symbols backlogged and dropped bars.
- **Dealer positioning, Jun 11–12.** Schwab chain parsing, gamma exposure walls/magnets, D0/D1 views, simulated trading, dashboards and optional Alpaca paper routing were added. Short history prevents a performance claim.
- **Jun 15 — Overnight polling fix.** Dealer runner was changed to sleep until next market open, with focused tests.
- **Jun 20–28 — Consolidation.** `9a81b6e` reduced redundant structure. `450c93b` and related work added HTF live runner, more dashboards, calendar/readiness/watchdog/expiry safety. The project became an integrated multi-strategy operating surface.

## July 6–7: reliability and statistical restraint

- **Jul 6 — WSL memory failure.** The combined server stopped under RAM/swap pressure while child catch-up/meta/news jobs ran outside the process-local memory keeper. Response: isolate/stagger heavy jobs, add memory/live-window guards and supervised recovery.
- **Jul 7 — Confluence null result.** `research/confluence_discovery_2026-07-07.md` tested 524 signal pairs. No pair met FDR certification; best q=0.56; power analysis estimated a +7–10 pp detection floor. The test window was consumed and nothing was wired live.

## July 12–14: reproducibility audit and headline correction

- **Jul 12 — Leakage/reproducibility audit and lock.** `b9579fe`, `8b75062`, `208e2ee` documented risks, generated `research/capstone/results_lock.json`, and added regression tests.
- **Jul 13 — Loader and selection fixes.** `3815e1b` fixed the raw-bar timestamp/index loader that silently skipped tickers. `018566b` moved policy selection to validation and froze test.
- **Measured correction.** Momentum ret/DD fell from a same-split 44.6× to 6.05× (~7.4× inflation); HTF from 41.4× to 17.9× (~2.3×); swing stale combined PnL win rate 61.3% became 48.7% on clean frozen test. These are central capstone lessons.
- **Jul 14 — Figures and 4H policy study.** `f879d01` fixed configurations on pre-test folds/validation and read frozen test once; `4753f9d` generated 12 provenance-labeled figures.

## July 15: shared audited execution architecture

- **Shared infrastructure.** `bac3f9b` added readiness gates, heavy-job guard, shared 4H execution/exit and signal auditing.
- **Module migrations.** Momentum (`729273b`), HTF (`c6cffe3`) and meta (`cc3c395`) moved onto the shared engine.
- **Swing safety.** `8c24efc` added batched cross-sectional scans/calibrated gates/dealer filter; `90e6b7c` added market-sell fallback after exit-limit exhaustion.
- **Dealer/monitoring expansion.** `25f4478`, `09d83e3` added Dealer Ranker, Amethyst dashboard, hub wiring, out-of-band data refresh and readiness-on-start.
- **Current direction.** Several specialist rankers feed shared, readiness-gated, audited execution rather than one monolithic SPY agent.

## Audit-day findings, July 15

- Non-slow capstone/shared-core validation: 31 pass, 2 fail; mutable SPY daily benchmark drifted from 1,496 to 1,498 rows.
- Advertised safe smoke suite: 110 pass, 2 fail; unconditional default earnings-result loading contaminated catalyst fixture inputs with 550 records.
- No broker, network, live-order, retraining, or destructive data command was run.

## Chronological interpretation

The evidence supports a repeated cycle: practical hypothesis → model/pipeline implementation → richer metrics → causal or operational failure → narrower claims and redesigned infrastructure. The most meaningful evolution is not “a model became more accurate”; it is movement from unstable single-asset prediction toward cross-sectional ranking, point-in-time/OOS auditing, shared execution, and explicit operational safety.
