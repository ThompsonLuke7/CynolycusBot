# Second review of the eleven findings

Reviewed September 16, 2026, against local HEAD `77c57371302119ea6e7316b3cce981b48e86a0f8`.

**The underlying problems largely stand, but the original “eleven high-priority issues” label was too broad.** Six findings concern reachable execution/accounting defects; one is conditional on an optional execution route; three concern research validity; the last combines an intentional compatibility choice with a missing research/deployment contract. None of this establishes how much of the account's losses these defects caused.

This document supersedes the severity and scope assessments in the [September 13 review](../system_review_2026-09-13/review.md). All 21 source fingerprints from that review still matched at the start of this reassessment. “Current” means current in this checkout; the remote running process and deployed artifacts were not inspected.

The work completed here is **verification and remediation design**, plus executable offline evidence. Production fixes, retraining, historical ledger correction, and deployment are not performed. No fix can honestly be guaranteed harmless before its integration and regression tests run.

| ID | Second verdict | Priority and minimal solution |
|---|---|---|
| F01 | Confirmed: a partial fill can be booked as the entire exit. | **P1 execution:** reconcile filled quantity and cumulative proceeds; retain ownership of the remainder. |
| F02 | Confirmed, with corrected and expanded scope. Deferred ownership survives; pending-order tracking does not. HTF also has the gap. | **P1 execution:** use the same exit reconciler in shared execution, HTF, Meta, deferred flushes, and Intraday. |
| F03 | Confirmed freshness defect; profitability/policy-neutrality claims require qualification. | **P1 risk:** separate entry anchor from timestamped current price; explicitly validate intrabar stop behavior. |
| F04 | Confirmed in the fast risk pass; slower runner already handles known dead orders. | **P1 risk:** distinguish working, terminal, and unknown orders before retrying. |
| F05 | Confirmed missing mutual exclusion; actual account incident not established. | **P1 ownership:** lock each complete state transaction across every writer. |
| F06 | Confirmed helper defect, conditional exposure. Default Dealer equity route does not use it. | **P2 conditional:** reconcile cancellation and partial fills before replacing an entry. Required before enabling that option route. |
| F07 | Confirmed fabricated settlement, including an equity counterexample. | **P1 accounting:** keep absent positions unresolved until broker fills/activities establish an outcome. |
| F08 | Confirmed in old bake-off; already acknowledged and avoided by the newer horizon harness. | **P2 research:** retire or repair the old splitter; label affected conclusions superseded. |
| F09 | Confirmed in two checkpoint features, not every continuation policy. | **P2 research:** remove the extra forward shift, version the panel, rerun affected ML arms before reuse. |
| F10 | Confirmed inherited test-based selection in canonical and exported harnesses. | **P1 research gate:** select on inner validation, then evaluate untouched outer periods. |
| F11 | Narrowed: intentional XGBoost export; unverified cross-family OOF/live equivalence and mislabeled baseline. | **P2 model contract:** produce deployable-family OOF and explicit lineage; do not switch live models merely to match the research winner. |

## F01 — Quantity, not price, establishes how much closed

**Evidence.** [Shared exit execution](../../core/live_4h_exec.py:1341), [price-only polling](../../core/live_4h_exec.py:1756), and [ledger recording](../../core/live_4h_exec.py:1798) ignore cumulative filled quantity when deciding a full exit completed. A two-contract fill out of ten, entry 1 and exit 2, records 1,000 rather than 200 and leaves the planned full-exit owner absent. Controls confirm that immediate full fills work and that zero-fill accepted full exits are already preserved in this shared path. The defect is specifically the partial-fill branch, not an assertion that all sells lose ownership.

**Solution.** Add a bounded order-observation function returning broker ID, raw status, cumulative filled quantity, cumulative average fill price, and observation time. Reuse the existing `BrokerOrder` vocabulary in [execution/broker.py](../../core/nervous_system/execution/broker.py), through adapters that need no database connection. Keep the old price helper for compatibility while migrating its lifecycle consumers.

Persist how much of each order has already been accounted for. Given new cumulative quantity `q` and average price `p`, use `delta_quantity = q - previous_quantity` and `delta_proceeds = q*p - previous_proceeds`; incremental realized P&L is `(delta_proceeds - basis*delta_quantity)*multiplier`. Multiplying an incremental quantity by the latest cumulative average is wrong when later fills have different prices. Preserve module ownership for unfilled units; account net holdings are a ceiling, not evidence of which module owns them.

**Architecture protection.** Separate realized fill events from the completed logical-exit projection, so ten fills do not become ten “trades” in win-rate reports. Use durable idempotent evidence keyed by account/order/fill or cumulative checkpoint, then project state and existing ledger views. Replaying after a crash must repair either projection without double booking. Trims must track requested versus filled trim quantity and only complete the trim once that intent has filled; planning must not prematurely set the completed flag.

**Acceptance tests.** Zero/full/partial fills, partial then cancel, multiple prices, repeated observations, out-of-order observations, crash between journal/state/ledger writes, delayed trim, two modules holding one OCC. Keep existing full-fill and zero-fill behavior.

## F02 — Several consumers bypass the existing unfilled-exit safeguard

**Evidence.** Reproduced Intraday's `_open` removal and phantom close at [execution.py:684](../../strategies/intraday_structure/execution.py:684). Reproduced the actual Meta callback with its real `gateway_verdict`: `SUBMITTED` enters the accepted branch and writes a null-P&L close while ownership stays removed; `REJECTED` and `AMBIGUOUS` correctly retain ownership. The gateway's acceptance is therefore not a fill guarantee. Its durable execution journal can still hold order evidence; the broken part is the strategy book/ledger projection, not total disappearance of every record.

The second review also reproduced this in [HTF's independent `_execute`](../../strategies/multi_ticker_swing_htf/live/runner.py:464). A fix confined to shared `execute_plan` would miss it. In [the deferred flush](../../core/live_4h_exec.py:2356), the original managed position remains, and `last_flush` retains the submitted ID, but no `exit_pending` is attached and accepted orders leave the retry queue. This is loss of an active reconciliation lifecycle, not immediate loss of the deferred position's ownership.

**Solution.** Adapt all five consumers to the F01 reconciler. Transition deferred intent from queued to submitted/pending only after the order identity is durably linked to its owner. Poll/reconcile accepted orders rather than submitting them again. Retain Intraday's setup ownership and original exit reason while pending; call reconciliation on runner ticks even after the setup has reached a terminal signal state. Existing expiry-only polling is not sufficient for next-week contracts.

**Architecture protection.** Preserve Meta's `DecisionCoordinator → policy → ExecutionGateway` route; put reconciliation behind its callback instead of substituting direct broker submission. Preserve Intraday's own-fill sizing, broker ceiling, lock and rejection backoff. Use shared transition logic with small runner adapters, not a wholesale merger of strategy engines.

**Acceptance tests.** The same zero/partial/full/delayed-fill contract for each of the five consumers, restart with pending intent, refusal and ambiguity controls, queue replay, and a closed Intraday setup with a still-working exit.

## F03 — A current stop and an entry anchor need different data contracts

**Evidence.** [The risk evaluator](../../core/live_risk_pass.py:297) calls `underlying_basis(ticker)` for its current observation. [That helper](../../core/live_4h_exec.py:226) returns a cached 4H close without a timestamp; [the script](../../scripts/live_risk_pass.py:114) injects no fresher source. The default [shared refresher window](../../UI/shared_data_refresher.py:53) starts at 13:45 ET; the [risk supervisor](../../UI/combined_server.py:828) runs during RTH at a 300-second default. Repeatedly checking that cache does not ensure a current morning price.

The synthetic old-close counterexample stands. Controls show the intended premium fallback works when the underlying observation is unavailable, and a fresh price above the underlying stop correctly suppresses the premium stop. **An 80% premium loss alone is not proof that this underlying-based policy should exit.** The defect is treating an arbitrarily old price as current.

**Solution.** Preserve `u_entry` and `u_atr` as immutable entry anchors. Inject a separate `CurrentUnderlyingObservation` with price, event timestamp, observed timestamp and feed. Reuse the existing timestamped quote access in [fetch_intraday.py](../../core/API/Alpaca_API/market_data/fetch_intraday.py:212), with an injected client/cache and bounded batched fetching rather than constructing a client for every position. Validate actual feed entitlement, timestamp age, session, finite prices and quote quality before using it. Encode price convention and max age explicitly.

**Architecture protection.** Do not change historical feature generation or make live quotes the source of the entry ATR. When no usable current observation exists, record degraded status and follow an explicit fallback policy; initially preserve the existing unavailable-underlying premium fallback, with valid premium evidence, rather than adding another loss threshold. Calendar-driven expiry flatten remains independent of quote freshness. Do not silently substitute delayed SIP or IEX and call them equivalent.

**Correction to policy reasoning.** The source claims more frequent fixed-stop checks are “strictly better” and parity-neutral. That is not established: an intrabar breach followed by recovery can produce a different exit from a close-only check. Current-price plumbing is a correctness fix for the stated fast-risk purpose; its return impact still requires replay/shadow comparison at the intended cadence. Keep trails, trims and model/bar counters unchanged.

**Acceptance tests.** Fresh/stale/future timestamps, overnight/weekend sessions, missing or crossed quotes, fallback behavior, expiry without quotes, fixed entry anchors, and a breach/recovery path distinguishing intrabar from close-only policy.

## F04 — A pending flag is not a broker status

**Evidence.** [Fast risk](../../core/live_risk_pass.py:272) skips accepted, partially filled, canceled and expired orders identically, without reading their status. This is correct for a working reservation but wrong for a known terminal order with residual holdings. The [slow runner](../../core/live_4h_exec.py:599) already distinguishes known working/dead statuses.

**Solution.** Reconcile before the skip. Working orders reserve their remaining quantity. Known terminal orders first account for fills, then permit only an unreserved, module-owned residual to be reconsidered under the existing exit policy. Preserve the old exit reason as audit lineage; an already queued mandatory exit must not become an entry opportunity. Follow replacement IDs before declaring an order chain dead.

**Architecture protection.** Do **not** copy `_order_is_working` unchanged: it returns false on lookup failure. A counterexample verifies that behavior. Use `WORKING / TERMINAL / UNKNOWN`; unknown retains the claim and reservation, retries observation on later ticks, and produces a visible aged-reconciliation alert. It must not authorize a duplicate sell. Bound per-pass I/O so uncertainty in one order cannot freeze every position.

**Acceptance tests.** Working and pending-cancel orders do not stack; canceled/expired partial orders expose only their residual; a timeout is not a cancellation; restarts and replacement chains retain identity. Ensure the script persists reconciliation-only changes, not just newly planned orders.

## F05 — Lock the transaction, not just the write

**Evidence.** [The risk script](../../scripts/live_risk_pass.py:105) acquires `module_state_lock`. Normal Momentum, HTF, Meta and Dealer writers do not. The actual Momentum writer succeeds while the risk lock is held, and a subsequent stale risk save erases its addition. The lock itself works: a cooperating second acquisition is excluded. The combined server uses independently scheduled risk and module subprocesses; the heavy data-job guard does not serialize these state transactions. This proves a reachable race, not that a particular historical loss resulted from it.

**Solution.** Move/re-export the existing lock and atomic persistence helpers through a dependency-light state module. Acquire the same lock before loading execution state; obtain broker facts inside that transaction; reconcile, plan, submit and persist before release. Cover normal runners, pending-open/exit flushes and direct/manual execution entry points. Do expensive scoring/data refresh before locking, then reload and revalidate the execution plan under the lock. A lock only inside `_save_state` still allows stale read-modify-write loss.

**Architecture protection.** Keep per-order persistence inside the outer lock without recursively acquiring another file lock. Use unique temporary files plus atomic replace. Keep existing paper state paths during the first compatible migration; every writer must agree on one lock identity. Paper/live path separation must move locks, states, queue paths and ownership readers together—changing only lock keys makes shared-state races worse. A contended tick should be skipped visibly, with bounded runner I/O and lock-age monitoring so the risk pass cannot be starved by a hung process.

**Acceptance tests.** Multiprocess runner/risk interleavings, concurrent flush and manual run, crash while writing, no lost positions, broker snapshot fetched after lock acquisition, no nested-lock deadlock, and consistent account/mode separation.

## F06 — An attempted cancel is not a confirmed cancel

**Evidence.** [The entry ladder](../../core/live_4h_exec.py:1655) advances after a swallowed cancellation error, and after a cancellation response that remains pending. Both produce three submissions of ten for a ten-contract intent in the fake broker. First-rung fill correctly stops escalation. Even a confirmed cancellation after two filled contracts still causes the next rung to request ten rather than eight. Thirty is potential submitted exposure, not proof all thirty would actually fill; buying power or broker rejection can limit it.

**Reachability correction.** The only production opt-in found is [Dealer](../../strategies/dealer_positioning/live_ranked_options.py:982), which defaults to `--route equity`. The combined-server launcher does not override that default. This is not a defect in every module's current entry path.

**Solution.** Preserve ladder prices/dwell settings. After a cancel request, reconcile terminal status and cumulative fills before any replacement; submit only the remaining intent quantity. Persist linked order identities and an intent-level maximum. A late full fill ends the ladder; ambiguous cancellation ends this pass and retains a pending claim for reconciliation.

**Architecture protection.** Raising an ordinary submission exception is insufficient: `execute_plan` can then call `drop_failed_entry` and discard a possibly live buy. Return/persist a structured pending outcome through the F01 entry adapter instead. Keep default single-shot entries and sell ladders unaffected. Do not copy Intraday's private cancel helper wholesale; its contract is different.

**Acceptance tests.** Confirmed zero-fill cancel, partial cancel, cancel/fill race, pending cancel, timeout, missing order ID and restart. Across all linked orders, filled plus still-reserved quantity must not exceed the intent.

## F07 — Absence is not evidence of worthless expiration

**Evidence.** [Settlement](../../core/live_4h_exec.py:2011) books `-basis*qty*multiplier` whenever it cannot recover a fill. The reproduced equity timeout becomes a fabricated −1,000 `expired_worthless` row. For options, a missing contract can reflect exercise, external closure or unsettled evidence. Alpaca supplies distinct fill and non-trade activity evidence; paper non-trade activities may only appear the following day. [Official options activities documentation](https://docs.alpaca.markets/us/docs/options-trading).

**Solution.** Return an explicit reconciliation outcome: filled, verified expiry, exercise/assignment, externally closed with evidence, or unresolved. A missing order/position read produces unresolved and no realized amount. Retain a durable reconciliation record until order fills or matching account activities explain the disposition. Exercise must link resulting underlying exposure and basis; it is not a zero-price sale of the option.

**Architecture protection.** Change both callers. A test proves that simply returning `False` still makes [the slow planner](../../core/live_4h_exec.py:575) drop ownership/history, whereas the risk pass retains it. Use a pending-reconciliation state excluded from new-entry eligibility and speculative sells; do not pretend an absent position is confirmed held. Allow delayed paper activity arrival. Preserve historical logs; subsequent evidence produces explicit correction records and a reconciled view, not overwritten history.

Some existing tests explicitly assert “gone + expired order = worthless option.” Those are evidence of the old assumption, not evidence that it is financially valid; replace their fixtures with verified settlement activity and retain a legitimate worthless-expiry control.

**Acceptance tests.** Equity lookup failure, genuine zero-value expiry, exercised call, external sale, partial fill plus later expiry, delayed paper activities, incomplete positions snapshot, and repeated reconciliation without duplicate P&L.

## F08 — Old embargo error; newer implementation already differs

**Evidence.** [Old splitter](../../scripts/label_bakeoff/run_bakeoff.py:189): `4*60` clock hours is ten calendar days. A two-bar/business-day fixture yields only sixteen bars from the last training observation to first validation observation, allowing a 25-bar target to overlap. [The newer splitter](../../scripts/horizon_thesis/run_horizon_grid.py:199) was independently exercised and separates boundaries by seventy indexed decision bars. The newer research report already acknowledges the old defect. Neither the newer splitter nor production execution should be described as having this particular clock-hour bug.

**Solution.** Retire the old splitter or replace it with a common purged split utility. Preserve predeclared decision-time cutoffs; remove training rows whose actual label end/availability reaches the next evaluation block, and similarly purge validation before test. Carry endpoint metadata from label construction, including per-symbol missing sessions. A global indexed-bar gap is a useful guard, but is not a substitute for actual endpoint checks on sparse series. Enforce sufficient remaining rows and intact timestamp groups.

**Architecture protection.** Keep target/model changes separate from splitter repair. Write newly versioned experiment artifacts with old results explicitly superseded. A test set already used for choosing research directions cannot be made untouched by rerunning it; use a new frozen prospective window or appropriate nested evaluation for the eventual promotion claim.

**Acceptance tests.** Weekends, holidays, missing ticker bars, multiple label horizons, no boundary label overlap, and unchanged predefined cutoffs. No reason to retrain live models solely because this obsolete study had a bug.

## F09 — Day-six information really reaches the day-five features

**Evidence.** [Two expressions](../../scripts/horizon_thesis/build_decision_panel.py:97) include an extra leading `shift(-1)`. Perturbing only day-six price/volume changes both `cp_up_day_share` and `cp_volume_ratio`; checkpoint price uses days one through five. The [ML continuation feature list](../../scripts/horizon_thesis/continuation_model.py:45) consumes both. Its price-only policies do not.

**Solution.** For both features use `series.rolling(CHECKPOINT).mean().shift(-(CHECKPOINT - 1))`. This gives entry day through checkpoint day, matching `cp_close`; preserve the strictly pre-entry volume denominator. An isolated AST prototype applied only in the review process passes day-six perturbation invariance and a hand-calculated volume-window check. Existing return and price-checkpoint columns are unchanged by that prototype.

**Architecture protection.** Version/rebuild the decision panel and rerun only affected ML results and their dependent conclusions. Keep previous outputs for comparison. The panel can legitimately contain future targets; the feature availability boundary is what was wrong. The reported ML arm already underperformed, so this leak does not prove its removal will improve profitability. Do not retract unaffected price-rule/rank results on this basis.

**Acceptance tests.** Perturb all post-checkpoint prices/volumes without changing checkpoint features; perturb an in-window day and see the intended response; verify the pre-entry denominator, final incomplete windows and unchanged label columns.

## F10 — Selection uses test metrics, including in the canonical export source

**Evidence.** The actual `choose_best` functions change winners when only test metrics change, with validation held fixed. This is reproduced for [the canonical harness](../../strategies/model_training/colab_competition.py:690), the Meta copy and an HTF export. Family averaging also uses the same test metric. Export scripts copy the canonical `strategies/model_training` harness; fixing only the originally cited Meta copy would leave future bundles defective. Local June model metadata names `test_ndcg_at_10` as selection metric.

**Qualification.** Such a block can intentionally be used as a selection set, but its score then cannot be presented as independent final-test evidence. Weight refitting in subsequent walk-forward folds is out of sample at the fold level; choosing the family/seed using later global test outcomes still contaminates the overall selection process. These are inherited research issues, not evidence that a current order was mispriced.

**Solution.** Compute validation ranking and Spearman metrics for every family; choose family and seed using those metrics only. Merely renaming the selector prefix is insufficient: `train_one_family` currently computes ranking metrics only on test. Fail explicitly if required selection metrics are missing rather than silently using test or first-row fallback. Use timestamp-grouped, label-purged inner training/validation within each outer evaluation period; either select inside each outer fold or fix the model specification using a genuinely earlier development period. Early stopping stays within inner validation. Test data becomes reporting-only.

**Architecture protection.** Fix the canonical harness, regenerate versioned export bundles, and record the selected metric and harness fingerprint. Preserve installed models until corrected challenger evidence exists. This avoids combining a methodology repair with an unvalidated production model change.

**Acceptance tests.** Changing test labels or test metrics cannot alter the selected model, seed, parameters or early stopping; changing validation can. Assert non-overlapping target windows and timestamp groups. Run a small CPU competition after implementation, then the controlled research evaluation; this review did not train models.

## F11 — Distinguish intentional compatibility from an unvalidated data contract

**Evidence.** [HTF trainer](../../strategies/multi_ticker_swing_htf/models/colab/htf_swing_train_colab.py:130) intentionally generates OOF from the overall winner, then independently exports the best XGBoost family for the native live scorer. Local metadata identifies LightGBM classifier seed 46; the deployed-format JSON is XGBoost `binary:logistic`. The research Meta builder reads HTF OOF, while its live updater uses `HTFSwingScorer`. However, the OOF parquet contains no family/seed/run fingerprint. Code plus neighboring metadata strongly suggest a family mismatch; they do not authenticate the exact historical lineage of every stored score.

Momentum's saved native objective is also `binary:logistic`, while `M0_current` in the label studies is a newly trained regression composite. It is a valid candidate label baseline; the comment “as deployed today” is inaccurate if interpreted as an exact replay of the installed classifier. That naming issue alone does not invalidate the whole label comparison.

**Solution.** Select a deployable model specification using corrected F10 validation. Generate fold-fitted OOF from that same family/objective/label/feature/preprocessing specification; export its final fit for live use. “Same model” here means the same learning specification, not identical fold weights. Keep unrestricted research winners as separately named challengers. Attach bundle IDs/hashes and train/validation/label availability metadata to OOF, native model, feature manifest and downstream Meta training. Require compatible lineage at promotion. Rename `M0_current` to a composite-regression baseline and add an exact deployed-specification classifier baseline for comparisons making that claim.

**Architecture protection.** Do not hot-swap live HTF to LightGBM. Changing base-score distributions requires regeneration of the Meta research matrix and validation/retraining of dependent Meta models as one challenger bundle. First establish provenance; if retained training records establish that a particular OOF already matches the live specification, retract the mismatch for that artifact instead of retraining it unnecessarily.

**Acceptance tests.** Manifest mismatch fails before promotion, same feature frame produces matching native/reference inference for one fitted artifact, OOF availability precedes each fold decision, and downstream training/live base-score specifications match. Quantify score/rank drift separately; this review establishes no P&L impact.

## Implementation order and validation boundary

1. **State ownership first:** F05 transaction boundaries, then the shared order-observation/reconciliation contract for F01/F02/F04/F07. Roll out per runner with replay/restart tests; keep the governed Meta submission boundary. Repair F06 through the same contract before using its option route.
2. **Current risk observations:** F03, keeping entry anchors and strategy thresholds fixed; compare intrabar behavior in replay/shadow before changing the policy's validation claims.
3. **Trustworthy research:** F09 feature repair and F08 splitter retirement/repair, then F10 selection and F11 deployable lineage before the pending Meta challenger retrain.

Use the existing execution journal for governed execution and a compatible durable local evidence adapter for legacy runners; do not make every emergency exit dependent on a newly introduced database service. Persist the same facts, with adapters for current state/ledger formats. Migrate old pending records without inventing fill counts. An idempotent reducer must survive replay after every persistence boundary before it replaces current bookkeeping.

Broker semantics were checked against [Alpaca's order lifecycle](https://docs.alpaca.markets/us/docs/orders-at-alpaca): partial fill, full fill and pending cancellation are distinct. This supports the order-state corrections, not an inference about how often these scenarios occurred in this account.

The accompanying [offline reassessment tests](test_reassessment.py) include defective-behavior observations, positive controls, actual runner-function bodies with injected boundaries, local artifact checks, and one isolated feature-math prototype. Their green result means the reassessment is reproducible, **not that production is fixed**. They are deliberately outside normal production test discovery. Some runner bodies are AST-extracted to avoid application bootstraps; remote runtime, database integration, fresh feed entitlement and end-to-end crash recovery remain implementation/deployment gates.

See [validation.json](validation.json) for exact commands, counts, fingerprints and limitations. Existing passing tests include outdated settlement assumptions and therefore cannot establish correctness on their own. No production source, broker state, model, raw dataset or historical experiment output was changed by this review.
