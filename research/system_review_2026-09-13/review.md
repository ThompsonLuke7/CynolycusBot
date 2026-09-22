# System architecture, profitability, and monthly code review

Review date: 2026-09-13 ET. Reviewer: Codex.

Follow-up: the [September 16 reassessment](../system_review_2026-09-16/reassessment.md) supersedes the severity/scope judgments for F01–F11 below. It distinguishes active defects from conditional paths, superseded research, and intentional model compatibility, and supplies remediation designs and additional controls. This original review is retained as the historical record.

**The main problem is an incomplete connection between predictive evidence, an executable trade, and trustworthy account outcomes. There are also reproducible execution and research bugs. Fixing those is necessary; it will not, by itself, establish profitability.**

The system already has momentum features, relative strength, catalysts, dealer levels, setup detection, confirmation, and multiple exit engines. Another model or setup vocabulary is unlikely to be the highest-value first change. We need to establish whether one specific model, trading one specific policy with finite capital, makes money after implementation costs.

## Scope and evidence

Reviewed the August 13–September 13 changes, active runners, shared execution and risk paths, model training/inference seams, nightly jobs, and recent experiment/daily-report evidence. The initial repository inventory contained 34 commits in that window and about 440 changed tracked paths relative to the pre-window baseline. This is a risk-focused review, not a claim to have inspected every line of every UI, generated asset, and experiment.

The review began at `1fd906f` with substantial pre-existing working-tree edits. Another process committed those edits as `77c5737` during the review and continued work on the horizon report. No pre-existing changes were reverted. The accompanying manifest records the reviewed source hashes. No broker requests, orders, deployment, policy changes, model training, or historical artifact rewrites were performed by this review.

Evidence classes:

- **Reproduced:** eleven offline demonstrations using fake brokers, synthetic bars, actual functions, and temporary files. These assert the defective behavior; their successful execution means the defect was reproduced, not fixed.
- **Verified by source/artifact:** live paths, training selection, saved model family/objective, and caller wiring.
- **Previously reported:** performance figures from existing research and daily reports. Those studies were read and checked against their implementation where relevant, but their full datasets/backtests were not rerun here.
- **Proposed:** the ten modifications below. Their profitability benefit is a hypothesis, not a promised return.

## What actually runs

The main flow is:

`market/news/options sources → shared bars + universe + context → per-module features/models → candidates and ranks → module entry rules → instrument/quantity selection → readiness/policy checks → broker orders → fills and ownership → exits → ledgers/dashboards`

There are several parallel implementations of the latter half of this flow. A shared data layer and combined UI do not make this one coordinated portfolio strategy.

| Component | Active decision path | Entry and exit implications |
|---|---|---|
| Data and universe | Alpaca/Schwab bars and quotes; shared 1m/1h/4h/1d caches; news/catalyst processing; market/sector context; theme membership; dealer snapshots; nightly readiness | Refresh success, feature availability, signal age, and price freshness are different facts. A nightly success stamp does not guarantee a current underlying price for an intraday stop. Universe snapshots were recently added; they cannot reconstruct historical membership before their first capture. |
| Momentum Expansion | Native XGBoost, currently `binary:logistic`, with `is_strong_setup` relevance in the saved training metadata; live feature panel and candidate filters; current top-N is 3; 1h entry triggers | Shared mixed option/share execution with a nominal per-entry budget. The older `MomentumOptionPolicy`/capital/campaign configuration is not a reliable description of every shared-runner action. Trace the active `run_pass` and `ExecPolicy`. |
| HTF Swing | Live matrix `htf_score` produced by `HTFSwingScorer`, which loads native XGBoost; top-10 and rank/liquidity eligibility by CLI default | Uses the shared 4H execution policy. Research OOF output can come from the competition's LightGBM winner, creating a material research/live seam. |
| Meta Ranker | Separate upside and quality native boosters; `s_combo` is their average within-bar percentile rank; top-10 plus quality/liquidity/entry gates | Main orders use `DecisionCoordinator → policy → ExecutionGateway → journal/adapter`. Required state/DB failures have blocked exits. Its accepted-order callback lacks the shared path's full unfilled-exit protection. |
| 30m Swing | Long/short ranking artifacts and calibrated probabilities; 30m scanning with 5m confirmation/management; fresh catalysts can adjust ranking | Long-only execution gate; option expiry floor is 21 calendar days. Its own position manager handles underlying stops, trailing/no-progress logic, fill verification, restoration, and sibling ownership. It is a separate policy, not the 4H tail-rider. |
| SPY intraday | 10m model/setup calculations with faster confirmation and option management; several stacked probability/feature layers | Separate adaptive option exit policy. Recent records report severe loop delay. The new matrix cache is locally present and tested structurally; live latency and multi-row catch-up parity remain distinct acceptance criteria. |
| Dealer positioning / Amethyst | Dealer maps, gamma/structural features and signal policies, with an optional separate executor | Source quality and option-expression assumptions need independent validation. These features are not automatically proven trade predictors. |
| Dealer Ranker | Heuristic cross-sectional dealer rankings; currently defaults to equity routing | Keeps its own 20% take-profit, 50% trim, 25-bar horizon, 50% stop and 35% trail configuration. Historical losses largely involved its earlier option route. Changing expression does not validate its equity policy. |
| Intraday Structure | Rules, not a trained ML model: candidates → detected/armed → confirmed/running → target/invalidated/closed | Existing opening and catalyst source implementations, price confirmation, levels, runway and invalidation rules. Current execution config enables paper options, including 0DTE before the cutoff; some docstrings still say off/0DTE-excluded. Its broker exit path has a reproduced ownership-loss bug. |
| Between-bar risk | Separate subprocess, nominally every 300 seconds; checks hard stops and expiry without advancing 4H holding counters | Uses cached 4H closes for option underlying stops. Has a lock that the normal state writers do not acquire. Main Meta governance is bypassed by this script's raw `execute_plan` call. |

The shared 4H defaults are a 30% gain trim of 16%, 53 managed-bar horizon, no trailing stop, 39% share stop, and a 1.5-entry-range-unit underlying stop for options when the basis is available; otherwise the premium stop applies. `underlying_basis` calls the rolling mean of high-minus-low “ATR”; it does not include previous-close gaps as true range does. Also, 53 bars at two bars/session is approximately 26.5 sessions, not the roughly 21 days quoted in several comments. The counter actually advances on management passes, so cadence and reruns matter.

Do not assume the `ExecPolicy.roll_trading_days=15` dataclass default is the effective monthly-contract floor: the Momentum, HTF, and Meta CLIs explicitly default that argument to 5 and pass it into the policy. Effective runtime parameters need a single persisted policy version.

## Why the visible winners do not settle the profitability question

The user's observation is useful: multiweek expansions occur, and the system should measure how early it discovers and ranks them. The previous callout audit actually found several names already captured in some form, including CRWD, MRVL, NBIS, SMCI, SMR, and TEM. Other names arrived late or never entered the active candidate set. That points to several different failure stages, not one missing indicator. See [the callout audit](../daily_live_reports/2026-08-27_callout_architecture_audit.md).

But a future chart high is an outcome, not a causal exit rule. A candidate that eventually rises 30% may first fall substantially, tie up capital, expire an option, or incur several failed entries. “It eventually recovered” is not enough to validate keeping every loser. Measure the unsuccessful lookalikes selected by exactly the same rule, with the same available capital and clock.

Likewise, win rate is only one part of expectancy. An illustrative 80% win rate with +1R winners and −5R losers has expectancy `0.8 × 1 − 0.2 × 5 = −0.2R` before costs. We do not have a complete, independently timestamped ledger of the Discord traders' entries, stops, exits, failures and position sizes, so this review neither verifies nor dismisses their claimed performance. The proper comparison is prospective and includes every qualifying callout, not only examples supplied after a move.

Options introduce an additional decision. A correctly predicted stock move can still produce an option loss because premium also depends on time, volatility, strike and other inputs. A longer expiry or tighter spread is not sufficient evidence of positive expected value. [OIC explains these option-price inputs](https://www.optionseducation.org/referencelibrary/faq/option-price-behavior).

## What the recent experiments support—and what they do not

| Evidence | Useful implication | Limit on the conclusion |
|---|---|---|
| Rank-depth work found stronger shallow Momentum selections and later reduced top-N to 3 | There is a reason to test selective deployment rather than repeatedly buying ten names | Strength varies by year; recent live samples and OOF comparisons disagree in places. Relative ordering can improve while absolute returns remain poor. |
| Horizon/factor study reports Momentum top-3 excess growing with holding period | A multiweek holding experiment is justified | Overlapping trade returns are not a finite-capital equity curve. Survivor bias, winner/seed selection and exact deployed-model identity remain relevant. |
| The continuation report says the top 5% contribute 74% of return | Tail preservation, diversification and capital occupancy matter | That is reported sample attribution, not a validated rule for holding every position. The ML continuation branch has the day-6 leakage reproduced below. The simple day-5 price rule does not use those leaking volume/up-day features. |
| Option studies repeatedly failed to replicate promising small cells and show substantial friction | The option wrapper deserves its own promotion gate; shares are the cleaner first experiment | Trade bars remain trades, not contemporaneous executable quotes. Better derivative/underlying correlation is a useful check, not a complete quote-quality certificate. |
| Meta label comparison favors a dense ranking target over `meta_good` | A corrected Meta retrain is a sensible challenger | The old embargo is wrong. The live strategy combines two boosters, so a one-label comparison does not isolate the deployed ensemble. |
| Intraday recall study found no useful separation from its rejection gate, with weak or adverse selection against random minutes | Redesign/evaluate the existing entry rule before adding a meta-label classifier | The report itself notes insufficient matched time-of-day/recent-move controls and few confirmations. It does not prove every intraday setup class lacks edge. |
| Theme membership churn and leader/follower studies | Stable grouping may help discover related candidates | The successful grouping study used backfilled membership, unlike the highly unstable live grouping. It cannot be promoted as causal live evidence yet. |

Relevant reports: [rank depth/options](../execution_quality/23_rank_depth_and_options.md), [horizon experiments](../execution_quality/24_horizon_thesis_experiments.md), [label comparison](../execution_quality/13_label_bakeoff_results.md), [intraday recall](../execution_quality/20_recall_test.md).

Three interpretations in prior notes should be narrowed:

1. **Within-bar comparisons are not immune to survivorship.** Removing failed/delisted companies can change which candidates are ranked and how different policies behave. Sharing a survivor-only universe reduces some confounds; it does not establish unbiased differences.
2. **A near-1 Meta combo score does not prove probability saturation.** `s_combo` is a percentile-rank average. Top-ranked names will naturally cluster near one. Inspect each underlying booster's raw outputs before diagnosing saturation.
3. **Underlying percentage return and percentage of option premium are different units.** A 4–8 percentage-point stock ranking advantage cannot be subtracted directly from a 35% premium hurdle. Evaluate the actual contract's dollar P&L under aligned quotes, sizing and holding rules. Similarly, flat average stock return plus negative average option return is not a clean isolated estimate of theta.

## Ten modifications, ranked by expected practical benefit

This ordering is judgment based on the inspected failure modes. It is aligned with execution reliability, validated research, the pending Meta retrain, and the existing discovery/theme work. It does not recommend ten simultaneous strategy changes.

| Priority | Modification | Why it is worth doing | Required evidence before promotion |
|---|---|---|---|
| **1** | **Make fills drive one durable order/position lifecycle across direct, deferred, Meta and intraday execution.** Persist intent, owner, broker order ID, cumulative fills and remaining quantity; reconcile cancellations; book incremental fills exactly once. | Directly addresses orphaned positions, duplicate exposure, disappearing exits, and inaccurate P&L. Extend the existing gateway/journal/ownership infrastructure rather than invent another one. | Replays of partial fills, accepted-but-unfilled orders, cancel races, expiry/exercise, timeout and restart. Broker quantities, cash and module allocations reconcile; unknown outcomes stay unknown. |
| **2** | **Make the risk loop independent of model freshness and current on price.** Use timestamped underlying quotes/trades for current stops, immutable entry basis, explicit stale-price behavior, and a shared lock or single state writer. Give verified reduce-only exits a durable degraded-operation path. | A five-minute timer provides little protection when it keeps reading an old 4H close, ignores dead exit orders, or loses concurrent state writes. | Stop-crossing-to-order latency under forced stale-cache/model/DB outages; no duplicate sells; ownership preserved. Exits still require authoritative held quantity, account identity and working-order checks. |
| **3** | **Finish SPY runtime parity and isolate expensive inference from execution servicing.** Validate the new cache across one-bar and multi-bar catch-up, restart, session rollover and cold start; persist the effective config. | Recorded 92–152 minute delays can invalidate otherwise good signals. Faster computation is only useful if it produces the intended model inputs. | P50/P95/P99 signal age and processing time, stale-entry rejections, and input/prediction parity against an offline reference. Healthy operation must finish inside the strategy's decision interval; exits remain responsive under load. |
| **4** | **Repair research validation and bind every experiment to the exact deployed model lineage.** Purge by label end/availability times; select family/seed on validation only; fix checkpoint leakage; regenerate matching OOF; version universe, features, labels, model, costs and policy. | Prevents selecting changes that only look better because evaluation leaked or tested another model. This is a prerequisite to spending on the next retrain. | Automated future-data perturbation invariance, no label overlap at boundaries, no test metrics in selection, matching OOF/live model family and objective, and explicit survivorship limits. |
| **5** | **Build one finite-capital, shares-only Momentum top-3 baseline.** Use the actual deployed score, timestamp-valid eligibility and executable entry delay. Compare with its current policy and a simple momentum baseline on identical data. | Isolates stock selection from option decay/spreads and creates an interpretable first system to improve. Top-3 is already configured; this is not another proposed top-K tweak. | Daily equity curve with cash, maximum positions, duplicate-name rules, slippage, costs, holding occupancy and unfilled entries. Report return, drawdown, Sharpe/Sortino, expectancy, exposure and SPY-relative results by year/regime. Do not equate mean trade return with portfolio return. |
| **6** | **Allocate risk across the whole account, including reservations for pending orders.** Set per-position planned loss, total open risk, gross exposure, sector/theme concentration and option-premium-at-risk limits. | Multiple nominal $5,000 entries are not equal risk and separate module budgets do not prevent account-wide buying-power starvation. Reducing K does not justify multiplying each position's size. | Equal-capital/equal-risk replay versus current allocation, correlated-shock scenarios, partial fills and simultaneous orders. Buying power remains available for obligations and planned entries without loosening loss limits. |
| **7** | **Separate multiweek stock theses from intraday and option exit policies.** Pre-register a small exit comparison using identical entries: current rule, simple time holds, and one structural/volatility rule. Express expiry and holding constraints in actual sessions, not inconsistent bar/day comments. | Addresses the user’s multiweek-expansion thesis without importing a stock backtest's thresholds into option premium. | A frozen, out-of-sample finite-capital comparison including drawdown, return per invested dollar, capital occupancy, gaps, winner capture and loser retention. The current tail-rider/harvester evidence is not sufficient to flip defaults. Premium rules need real option-path evaluation separately. |
| **8** | **Run the pending Meta retrain as a controlled challenger after #4.** Compare the existing two-head combo with a dense rank target and a simple composite baseline; ablate the unstable theme block. Grade executable returns as well as ranking metrics. | Meta has specific evidence of weak label alignment, but an improved MFE score is not the same as better realized trading. | Corrected nested walk-forward selection, exact train/validation/test boundaries, features/labels/class balance/hyperparameters, raw score diagnostics and net portfolio comparison. Promote in shadow before allocating more capital. |
| **9** | **Maintain persistent expansion watchlists using existing discovery sources and stable cohorts.** Track discovery time, consolidation, acceleration, break/retest and invalidation across sessions. Evaluate opening/catalyst sources already implemented; stabilize theme membership or compare trailing-correlation groups. | Some moves are missed upstream or forgotten between scans. This targets early discovery and continuity without building yet another setup detector. | Prospective candidate recall, lead time, false candidates and incremental returns versus the existing scanner. All cohort memberships must exist at observation time. Backfilled leader/follower results do not qualify as this evidence. |
| **10** | **Create a prospective callout benchmark and enforce a promotion gate for every sleeve.** Record complete timestamped callouts and the bot's candidate/rank/trigger/fill/exit state; compare with matched controls and an executable copy baseline. Keep unproven variants in shadow or bounded paper experiments. | Converts examples into diagnostic labels: never seen, ranked too low, late trigger, unsuitable contract, no fill, premature exit, or valid avoided loser. Stops weak modules from receiving more resources just because their charts look convincing. | All qualifying callouts over a pre-specified period, including misses and edits, plus original entry/stop/exit/contract details. Match controls on ticker, time of day, recent move and volatility. Promotion requires independent evidence after costs, a loss limit and reliable lifecycle telemetry. |

For #10, existing supplied examples can seed the taxonomy immediately. A historical win-rate claim remains unanswerable where original posting times or the complete callout population are unavailable. That is a data limitation, not grounds to discard the examples.

## Code review findings

Severity **P1** here means a high-priority loss-control, accounting, or research-validity defect; it does not mean every scenario was observed in the real account. No P0 account emergency is asserted from this offline review.

### F01 — P1: partial exits are treated as full closes

**Location:** `core/live_4h_exec.py:1341`, `:1756`, `:1798`.

`poll_exit_fill_price` treats any positive `filled_avg_price` as completion. `execute_plan` then skips restoration and `record_exit_realized_pnl` multiplies the submitted quantity, not the broker's filled quantity. A reproduced order selling 10 contracts with only 2 filled books $1,000 instead of $200 and leaves the remaining 8 without managed ownership.

The price-only poll predates the month; the August 16 ownership/fill-handling change relies on it. Correct the consumer contract to include status, cumulative filled quantity and residual quantity. Apply the same rule to trims and late fills, with incremental accounting and restart idempotency. Alpaca distinguishes partial fills from complete fills and pending cancellation from confirmed cancellation. [Broker order lifecycle](https://docs.alpaca.markets/us/docs/orders-at-alpaca).

### F02 — P1: accepted-but-unfilled exits still lose their durable lifecycle in other paths

**Locations:** `strategies/intraday_structure/execution.py:684`, `:720`; `core/live_4h_exec.py:2356`; `signals/meta_context/meta_ranker/live_runner.py:1179`.

Intraday unconditionally writes a close and removes `_open[setup_id]` after an accepted sell even if no fill exists. The deferred flush clears accepted orders from the retry queue and writes a null-P&L close without adding `exit_pending`. Meta's accepted callback similarly writes a close without restoring ownership on an unfilled full exit.

Intraday and deferred cases were reproduced. Meta is verified by callback inspection; its gateway acceptance is not a universal fill guarantee, particularly for equities. A working broker order may still fill later, but the owner/lifecycle evidence needed to track it has been lost. Use the same fill-state protocol as F01 everywhere.

Intraday entered in `866fa00` and its ladder changed in `a92913f`. The deferred close-record addition is in `7e23dc7`; Meta's accepted-sell behavior predates the window and remains in the path modified this month.

### F03 — P1: the frequent underlying-stop check accepts arbitrarily stale 4H closes

**Locations:** `core/live_risk_pass.py:195`, `:297`; `core/live_4h_exec.py:226`; `scripts/live_risk_pass.py:114`.

The default `underlying_fn` reads the shared 4H parquet and returns a close without its timestamp. No caller supplies a current quote in the live risk script. In the reproduction, a week-old close of 100 suppresses an option stop despite an 80% premium loss; supplying a current underlying value of 90 triggers the correct 97 stop. A valid cached basis disables the fallback premium stop even when the current underlying observation is stale.

Introduced into this risk path in `bcd767b`. Retain the entry basis, obtain a current timestamped underlying observation, and specify how risk management behaves during quote outages. This is a concrete correctness issue independent of whether 1.5 ATR is a good threshold.

### F04 — P1: canceled/expired pending exits suppress risk management until the slower runner revisits

**Location:** `core/live_risk_pass.py:272`.

Any `exit_pending` dictionary is assumed to describe a working order. The risk pass never asks the broker for that order's status. A canceled order on a position below its stop yields `exit_order_already_resting`, no status read, and no retry. The new `_order_is_working` check in `build_mixed_plan` fixes the slower 4H path only; a daily Dealer pass can leave the gap much longer than five minutes.

Reproduced. Reconcile the pending order before skipping; preserve genuinely working orders, resize/retry dead residuals, and retain unresolved states on read failure.

### F05 — P1: the risk lock does not exclude normal state writers

**Locations:** `scripts/live_risk_pass.py:105`; `core/live_risk_pass.py:387`; `strategies/momentum_expansion/live/runner.py:78`.

Repository search finds production acquisition of `module_state_lock` only in the risk script, while normal runners read/write the same files without acquiring it. The separate heavy-data-job lock is not this lock. The reproduction executes Momentum's actual `_save_state` while the risk lock is held, then saves the earlier risk snapshot; the newly opened position disappears.

The new risk writer was added during the window. Atomic rename prevents torn writes, not lost updates. All writers must share one read-modify-write lock or use versioned transactions/single ownership. Lock keys and state must include account mode. Acquire current broker facts consistently with that transaction.

### F06 — P1: entry ladder continues after an unconfirmed cancellation

**Locations:** `core/live_4h_exec.py:1689`, `:1719`.

`_cancel_order_quietly` swallows cancellation errors, and the next rung submits the full original quantity. The reproduction leaves three potentially working orders totaling 30 contracts for an intended 10-contract entry. A successful cancel request also does not establish terminal cancellation; partial fills require reduced replacement quantity.

Introduced in `7e23dc7`. Current relevance is the opt-in legacy entry ladder, particularly Dealer option routing; its new default equity route reduces that exposure but does not repair the helper. Require terminal cancel/reconciliation evidence before repricing, and stop on ambiguous submission outcomes.

### F07 — P1: missing order evidence fabricates a complete loss

**Location:** `core/live_4h_exec.py:2037`–`:2049`.

When a position is absent and the order lookup fails, `resolve_settled_exit` records the entire basis as `expired_worthless`. This branch also applies to equities. A synthetic equity close with an order timeout books an invented −$1,000. For options, absence could also be exercise, external closure or an incomplete reconciliation; none proves worthless expiration.

Introduced in `7e23dc7`. Keep the outcome unresolved until fills or broker activity establish settlement. Reconcile exercise/assignment and corporate actions separately. Do not reconstruct P&L from missingness.

### F08 — P1 research: the old label comparison's embargo is materially too short

**Location:** `scripts/label_bakeoff/run_bakeoff.py:189`–`:198`.

`Timedelta(hours=4 * 60)` is ten calendar days, not 60 trading bars. With two bars each business session, the fixture has a 16-bar distance from the final training row to first validation row; even a 25-bar training label crosses the boundary. The report still advertises a 60-bar embargo.

Added in the monthly research work; known in the newer horizon report but not fixed in this reusable harness. Purge by each label's actual endpoint, rerun the affected comparisons and supersede their claims. The newer `run_horizon_grid.split_bars` uses indexed decision bars and is not accused of this same error.

### F09 — P1 research: day-5 continuation features read day 6

**Location:** `scripts/horizon_thesis/build_decision_panel.py:97`–`:100`; consumer `scripts/horizon_thesis/continuation_model.py:45`.

The extra leading `shift(-1)` makes `cp_up_day_share` and `cp_volume_ratio` cover sessions 2–6, while checkpoint price/target use the close of session 5. Changing only day-6 volume changes the purported day-5 volume ratio from 1.0 to 200.8. This was reproduced using the actual `ticker_frame` implementation and synthetic daily bars.

Present in the new horizon work committed as `77c5737`. Correct the window and rerun the ML continuation arm. This does not invalidate the simple day-5 price rule or the rank-falsification arm merely because they share the panel; identify actual column consumers.

### F10 — P1 research, inherited: family and seed selection use the test set

**Locations:** `signals/meta_context/meta_ranker/colab_competition.py:613`, `:690`, `:753`; copied competition helpers in the training exports.

The primary metric is chosen from `test_ndcg_at_*`, `test_precision_at_*`, or `test_spearman`. Family means and best seed are selected using those metrics. A reproduction supplies opposite validation/test winners; the helper chooses the test winner. Installed Momentum/HTF metadata both identify `test_ndcg_at_10` as the primary selection metric.

This predates the month, but it is directly relevant to the pending retrain and confidence in existing artifacts. Walk-forward refitting after globally choosing the winner does not make family/seed selection itself point-in-time. Use inner validation to select and preserve an untouched outer evaluation period. Report existing “test” results as selection-used, not untouched holdout evidence.

### F11 — P1 research/live parity, inherited: research winner, baseline and deployed model are not always the same

**Locations:** `strategies/multi_ticker_swing_htf/models/colab/htf_swing_train_colab.py:130`, `:154`; `strategies/multi_ticker_swing_htf/inference/scorer.py:25`; `scripts/label_bakeoff/run_bakeoff.py:56`; saved model metadata/objectives.

HTF training emits OOF scores for the overall winning family, but separately exports the best XGBoost run for live compatibility. The installed winner is `lgbm_classifier` seed 46; live loads `htf_swing_xgb.json`. The Meta research builder ingests HTF OOF, while its live updater uses `HTFSwingScorer`. That is a documented family mismatch in the pipeline, not merely different time windows. Exact historical artifact provenance should be re-established before quantifying its P&L effect.

Momentum's installed native booster has `binary:logistic` objective and its saved winner is `xgb_classifier` seed 45, trained with `is_strong_setup` relevance. Recent label-study `M0_current` fits a regression composite; it is a label-family baseline, not a faithful reconstruction of that installed classifier. The same distinction matters when reading claims that the current model's label has already won the experiment.

Verified from source and local artifacts, without loading/training models. Produce OOF for the exact selected deployable family/objective and name experimental baselines accurately. Do not swap live HTF to LightGBM based solely on its test-selected win.

## Additional architectural gaps and resolved findings

- **Meta has two exit authorization paths.** `scripts/live_risk_pass.py` includes Meta and invokes raw shared execution, while the normal runner uses governance. The static “no bypass” tests scan the normal entry points and miss this script. Treat this as an explicit migration/degraded-exit design decision: a durable, auditable reduce-only path is useful, but an undocumented bypass with shared paper/live state is not a complete design.
- **Trim state advances before broker confirmation.** `build_mixed_plan` sets `trimmed=True` and reduces saved size when planning. Unfilled trims do not restore that state. Fold this into F01/F02's fill-driven lifecycle and test failed, partial and delayed trims.
- **Risk thresholds are copied.** `scripts/live_risk_pass.MODULES` hardcodes values separately from runner CLI arguments. A custom runner policy can disagree with its risk pass. Effective policy version, account mode and entry/exit semantics should be persisted with each position.
- **Source freshness needs more than one timestamp.** Preserve bar start, bar completion, availability, signal, submit, fill and evaluation time. Offline full bars labeled by start cannot automatically be treated as fully known at that start.
- **Default pytest discovery omits several critical paths.** `pyproject.toml` does not include `core/tests`, `signals/meta_context/meta_ranker/tests`, or the SPY strategy tests in its default list. The explicit targeted invocation used here includes affected paths. Expand CI discovery alongside the meaningful regression cases; a green default suite cannot cover omitted code.
- **The stale standalone HTF feature file is not evidence of stale live HTF inference.** Live HTF uses the shared Momentum feature build through the live Meta matrix. The standalone stale artifact affects research that reads it. The concurrent horizon-report correction now makes this distinction too.
- **Recent fixes already present:** the gateway journal-receipt shape fix, broker `extended_hours` compatibility, Meta outage exit deferral, working-exit check in the slower mixed planner, intraday buying-power guard, universe snapshots, corporate-action screening, and SPY cache work. They are not presented as still-unfixed bugs. The selected relevant tests passed; their deployment/runtime state was not verified here.

## Validation and concrete next step

**255 targeted existing tests passed**, with one dependency deprecation warning, across shared exits/queues/risk, underlying stops, broker adapter compatibility, the new cache helpers, intraday execution, Meta no-bypass/gateway and governed-outage handling. No database integration or full performance replay was run. The separate **11 defect demonstrations all reproduced** their expected observations. See [the reproduction script](reproduce_findings.py), [its captured results](reproduction_results.json), and [the validation manifest](validation.json) for exact scope.

The next work package should be **F01–F07 plus shared state ownership**, with tests asserting the correct invariants rather than the currently defective behavior. Then fix F08–F11 before using a new retrain or exit study to make allocation decisions. Establish the Momentum shares baseline and its finite-capital comparison before broadening the strategy search.

The immediate success criterion is that every held unit remains owned, every intended exit retains a durable status, every realized dollar comes from broker evidence, and every model comparison describes the exact policy/model actually evaluated. Once those hold, a negative experiment is useful evidence about the strategy rather than a mixture of strategy weakness and implementation failure.
