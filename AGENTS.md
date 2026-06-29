# CynolycusBot Agent Instructions

## Mission

CynolycusBot is a research-first quantitative trading system for identifying and trading high-quality equity momentum and expansion opportunities. The system includes swing, theme, catalyst/news, market-regime, meta-ranking, and future intraday/dealer-positioning modules.

Prioritize:

1. Correctness and reproducibility.
2. Prevention of lookahead bias, leakage, and survivorship bias.
3. Risk-adjusted performance, robustness, and interpretability.
4. Reusable research infrastructure over one-off backtest optimizations.
5. Clear separation between research, paper trading, and live trading.

## Working style

* Complete the requested task end-to-end when feasible. Do not stop after partial investigation or ask unnecessary questions.
* Be concise and efficient in progress updates and final responses.
* Inspect the codebase and existing project context before proposing or implementing changes.
* Prefer the simplest correct solution that fits existing architecture and conventions.
* Make focused changes only. Do not introduce unrelated refactors, dependency upgrades, formatting churn, speculative features, or broad redesigns.
* Use KISS, DRY, YAGNI, separation of concerns, and the principle of least surprise.
* Favor small, composable functions with explicit inputs, outputs, and responsibilities.
* Fail fast with clear errors rather than silently continuing with invalid, missing, stale, or misaligned data.

## Continuity and roadmap

* Before substantial work, read `LIVING_SUMMARY.md` at the repository root if it exists.

* Treat `LIVING_SUMMARY.md` as the cross-session project handoff. It should preserve durable decisions, active work, open questions, commands run, files changed, validation results, and next steps.

* At the end of every substantive response, append a concise entry to `LIVING_SUMMARY.md` using:

  `{YYYY-MM-DD HH:MM ET} {agent: Codex or Claude} {area(s)}`
  `Brief narrative of discussion, decisions, changes, validation, and next step.`

* Keep each entry to a maximum of 3 lines.

* Do not include secrets, API keys, raw logs, large command output, private account details, or transient noise.

* Do not write generic summaries for trivial questions with no project relevance.

* Follow the active roadmap. Only recommend work that supports the current roadmap; label anything else as **Optional / Outside Roadmap**.

## Investigation and planning

* Read relevant files before editing them.
* Before creating code, search for existing modules, utilities, configuration, tests, schemas, feature pipelines, plotting helpers, and established patterns.
* Reuse existing project abstractions when they fit. Do not duplicate logic or create competing implementations.
* Run independent file reads, searches, status checks, and inspections in parallel when possible.
* Keep dependent work sequential: inspect before edit, edit before test, and validate assumptions before building on them.
* For multi-step, cross-module, risky, or ambiguous work, create a short task list before editing and update it as work progresses.
* Skip formal planning for small, obvious, low-risk changes.
* Ask questions only when the answer materially changes scope, data assumptions, trading behavior, safety, or implementation design. Otherwise state the assumption briefly and proceed.

## Data integrity and time correctness

* Treat all market, news, options, social, and macro data as time-indexed information.
* Every feature must be available at the exact decision timestamp. Never use future data, revised values unavailable at the time, or data that leaks from the target window.
* Preserve and validate timestamps, time zones, trading calendars, session boundaries, symbol mappings, source metadata, and data freshness.
* Explicitly distinguish:

  * event time,
  * observation/availability time,
  * signal generation time,
  * order submission time,
  * fill time,
  * and evaluation time.
* Do not silently forward-fill, backfill, interpolate, deduplicate, drop rows, adjust prices, or impute missing values without documenting the rule and its impact.
* Preserve raw data as immutable. Store cleaned, normalized, engineered, and model-ready data separately with versioned transformations.
* Validate joins, resampling, merges, rolling windows, and train/test boundaries for leakage and alignment errors.
* Handle corporate actions, delistings, symbol changes, splits, dividends, halts, missing bars, and market holidays explicitly where relevant.
* Never overwrite raw market data, historical labels, experiment outputs, or live trading logs without explicit approval.

## Research and backtesting standards

* Research code must be reproducible: record data sources, universe definition, date ranges, feature versions, configuration, random seeds, model versions, and evaluation assumptions.
* Preserve fixed train/validation/test boundaries unless the task explicitly changes the experiment design.
* Never tune against the final test set. Use validation or walk-forward procedures for model selection.
* Compare changes against a named baseline using the same universe, date range, execution assumptions, and evaluation protocol.
* Report meaningful metrics appropriate to the module, including where relevant:

  * CAGR or total return,
  * Sharpe/Sortino,
  * max drawdown,
  * win rate,
  * expectancy,
  * trade count,
  * turnover,
  * holding period,
  * exposure,
  * benchmark-relative performance,
  * and performance by market regime.
* Include transaction costs, slippage, liquidity constraints, position sizing, execution delay, and realistic fill assumptions when relevant.
* Do not present a backtest improvement as meaningful without checking robustness across time periods, regimes, tickers/themes, and reasonable parameter variation.
* Flag small sample sizes, concentrated returns, unstable parameters, data gaps, and potential overfitting.
* Prefer out-of-sample validation, walk-forward testing, ablation tests, and sensitivity analysis over adding more complexity.
* Distinguish clearly between correlation, predictive signal, and tradable edge.

## Model and feature engineering standards

* Build features only from information available at decision time.
* Keep feature definitions, labels, transformations, and target horizons explicit and versioned.
* Validate target construction independently before model training.
* Guard against leakage through normalization, ranking, feature selection, cross-validation, and rolling calculations.
* Fit scalers, encoders, imputers, feature selectors, and model parameters on training data only, then apply them to validation/test/live data.
* Prefer interpretable baselines before complex models.
* Add model complexity only when it produces robust, validated improvement over the baseline.
* For every model change, report feature set changes, label definition, train/validation/test dates, class balance, key hyperparameters, and comparative results.
* Preserve parity between research and live inference pipelines. Do not allow live code to calculate features differently from validated research code.

## Trading and risk controls

* Keep research, paper trading, and live trading strictly separated by configuration, storage, credentials, and execution paths.
* Default to paper mode unless live trading is explicitly requested.
* Never place, modify, or cancel live orders without explicit confirmation in the current session.
* Validate symbol, side, quantity, order type, price/stop values, account mode, buying power, and risk limits before any order action.
* Treat broker/API responses as authoritative; handle rejected, partial, delayed, duplicated, or stale orders safely.
* Do not assume a backtest signal is executable in live trading.
* Log signal time, inputs, model/version, order intent, execution response, fills, and exit rationale for all paper and live trades.
* Make risk limits explicit and configurable. Do not hardcode account-specific values unless requested.

## Architecture and code quality

* Preserve separation between:

  * data ingestion and normalization,
  * feature engineering,
  * labeling,
  * research/backtesting,
  * model training,
  * inference/ranking,
  * portfolio construction,
  * execution,
  * reporting/visualization,
  * and configuration.
* Keep modules focused and interfaces narrow.
* Prefer dependency injection or configuration-driven dependencies over hidden global state.
* Avoid circular dependencies, duplicate pipelines, magic constants, and implicit mutable state.
* Add or update tests when behavior changes.
* Refactor small local issues only when directly relevant to the requested change; do not broaden scope into cleanup work.

## Plotting and reporting

* Before creating a plot, inspect the shared plotting directory and reuse existing utilities, especially for price-versus-time charts.
* Plots must make time ranges, symbols, benchmarks, entry/exit markers, signals, and axes unambiguous.
* Use consistent project conventions for dates, time zones, indicators, legends, labels, and saved output locations.
* Do not create charts that hide drawdowns, exclude inactive periods, cherry-pick dates, or otherwise overstate results.
* Clearly label whether a result is in-sample, validation, out-of-sample, paper-trading, or live-trading.

## Verification

* Treat an unverified edit as a hypothesis, not a completed fix.
* After each code change, run the most specific relevant verification available:

  1. Tests covering the changed behavior.
  2. Targeted lint, typecheck, build, or static analysis.
  3. A focused execution, smoke test, or data sanity check when automated tests are unavailable.
* For data or model changes, validate schema, row counts, timestamps, null rates, feature availability, label alignment, and output distributions as applicable.
* For backtest changes, run a baseline comparison and inspect representative trades/signals, not only aggregate metrics.
* If verification fails, investigate and fix it when feasible. Do not suppress, ignore, or misrepresent failures.
* Before finalizing, run a post-pass: “What did I not finish?” Check for incomplete requested work, untested edits, broken assumptions, regression risk, leakage risk, accidental scope creep, and missing documentation or summary updates.
* Do not claim a result is fixed, complete, profitable, robust, or ready for live use unless the relevant validation supports that claim.

## Tool behavior

* Batch independent tool calls to reduce unnecessary round trips.
* Use the least invasive tool or command that answers the question.
* Avoid expensive, destructive, networked, or broad commands unless necessary.
* Use external documentation or research when unfamiliar APIs, broker behavior, libraries, or market-data semantics require it; do not search externally when local code already answers the question.
* If using a GPU, explicitly notify the user. If not using a GPU, do not mention it.
* When providing a PowerShell command for the user to run, keep it on one line.
* Write stock tickers without a dollar-sign prefix: use `SPY`, not `$SPY`.

## Final response

Keep final responses concise and decision-useful. Include:

* What changed or what was found.
* Files changed.
* Validation performed and results.
* Important assumptions, data limitations, risks, failures, or remaining work.
* The next roadmap-aligned step, if one is needed.
* Clearly label any non-roadmap suggestion as **Optional / Outside Roadmap**.
