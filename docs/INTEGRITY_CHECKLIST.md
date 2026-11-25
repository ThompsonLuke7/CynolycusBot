# Logic and data-integrity checklist

Defects that a test suite does not catch, because each piece is locally correct
and the failure lives in how the pieces meet: a state that is never left, a clock
that measures the wrong interval, an input that is quietly months old. Every
class below reached the live paper stack and was found only when someone looked
at a chart. Use this in the daily review, in the weekly refresh, and when
reviewing any change that adds a state machine, a cache, a fallback or a
subprocess.

## Run the audit first

```bash
.venv/bin/python -m scripts.integrity_audit                    # latest session
.venv/bin/python -m scripts.integrity_audit --date 2026-11-23  # a given session
```

Read-only, about 5 s, exit 1 on any FAIL. Report every WARN and FAIL in the
review's Operations And Bugs section. The audit for 2026-11-23 is the reference
output: it flags the stuck setups, the wrong `bars_held` clock, the undated
universe `px` and the matrix-refresh failures.

## Defect classes

| # | Class | What it looked like | Audit check | Rule |
|---|---|---|---|---|
| 1 | **State entered, never left** | QQQ short ARMED from 11-12 to 11-23 at a 694 pivot (price 650); 102 such setups across 53 tickers, some from July. Ids are unique per ticker/direction/detector, so each one blocked its slot. | `intraday_carried_setups`, `intraday_stuck_abstainer` | Every non-terminal state needs an exit on age, on session change and on impossible geometry. A refusal that repeats every bar means stuck, not selective. |
| 2 | **Revival depends on a rare event** | Closed setups came back only when their candidate re-registered (about once a session): 6% re-armed the same day. | `intraday_rearm_rate` | Revival runs on the path that runs most often (each bar), not on an incidental event. |
| 3 | **A clock that measures the wrong interval** | "Too old to enter" counted an empty shell's whole pre-detection life (221 refusals); `bars_held` was 24 bars for a 3-minute hold. | `ledger_bars_held_clock` | Name each threshold's start event (detection, entry) and store a counter for that event. Do not reuse a lifetime counter. |
| 4 | **Undated reference data** | `shared_universe.csv` `px` came from a filter frozen at 2026-06-01, with no date: median 19.7% off. QQQ's +7% gap read as −11.7%. | `universe_px_freshness` | Every reference value carries an as-of date, and the consumer checks it against the decision session. Skip a stale value; never substitute an older one. |
| 5 | **Cache not current at decision time** | Meta's 14:20 run found a matrix still on Friday's bar. Gaps are measured from the 1d bar the 22:15 readiness run writes. | `bars_1d_freshness`, `meta_matrix_full_bar` | State which cache a decision needs and how fresh, and check it inside the decision path (fail fast), not only in a report. |
| 6 | **A fallback that drops the invariant** | No greeks → the picker ranked the whole ±10% band by open interest and bought QQQ 590P with spot at 652 (9.5% OTM). | `option_moneyness` | A fallback must keep the invariant the primary path enforces (here: near the money). Log which path was taken. |
| 7 | **Silent failure** | The refresher discarded the matrix step's stdout and stderr: three FAILED runs, no cause. | `refresher_steps` | A failed subprocess logs its exit code and output tail. "FAILED" alone is not a log line. |
| 8 | **Dependency with no preflight** | The Docker daemon was down after a restart, so the state store (55432) was down all day; Meta lost 13 actions. | `state_store` | Check external dependencies at launch, try to start them, and say loudly when they are down. |
| 9 | **Independent detectors, one bet** | Breakout and VWAP-reclaim confirmed QQQ short on the same bar (11-23 replay). | none (executor now refuses `ticker_direction_already_open`) | Exposure limits apply per ticker and direction, not per signal id. |
| 10 | **Positions with no owner** | TWST leftover expired at the broker (−$4,280) and was booked nowhere; flagged as an orphan every 5 minutes for days. | manual: grep `orphan` in the server log | An orphan has an owner by end of day, or a documented manual decision. |
| 11 | **An as-of check against the wrong clock** | The dealer snapshot was compared with the bar's OPEN; the poller writes ~48 s into each minute, so during RTH almost every live snapshot was "from the future" and 41 of 102 index-ETF setups (10-01..11-23) decided with no dealer levels. | `decision_context` (index-ETF decisions without dealer levels) | State the decision moment (a bar is decided at its close) and compare every input's timestamp with that moment, in one place. |
| 12 | **A limit that only warns** | `options_max_age_minutes` (24 h) produced a `dealer_snapshot_stale` warning but the levels were used anyway: Monday-open setups decided on Friday's walls (put 600, call 610, QQQ at 650). | `decision_context` (snapshot age is recorded) | A configured limit is enforced where the value is consumed. If the limit is wrong, change the limit. |
| 13 | **Long-lived keys suffer most from state bugs** | 22 of 28 setup slots on SPY/QQQ/IWM/DIA/GLD/SLV were stuck (some since 08-28), because watchlist ETFs never leave the candidate list, while single names rotate and get fresh slots. The module had barely been allowed to trade indices. | `intraday_carried_setups` | When judging a module per instrument class, first check that each class had the same chance to trade. |

## Questions for any new rule, state machine or data source

1. Can this state be entered and never left? List its exits. Is each one reachable, and how often?
2. What event brings it back (revival, reset, re-seed)? How often does that event fire in a live session?
3. For each threshold, what does the clock start on? Does it survive a restart (persisted), and does it reset at the right event?
4. Does any input lack an as-of timestamp? What does the consumer do when that timestamp is old?
5. What happens across a session boundary, a server restart, a half-day, or a cache that is a day behind?
6. If the primary path fails, what does the fallback do to the invariant? Is the path taken logged?
7. If this runs as a subprocess or thread, where do its errors go?
8. Can two independent signals take the same exposure?
9. Replay one real session through the change (old code vs new) and inspect what it actually does. Aggregate tests are not enough: the 11-23 replay showed 0 QQQ entries before the lifecycle fix and 30 after.

## Model-data alignment (`scripts/alignment_audit.py`)

The checks above cover plumbing. A second class is a model fed inputs unlike its
training data: every piece runs, nothing errors, and the module's P&L silently
stops testing its model. Run `.venv/bin/python -m scripts.alignment_audit` after
the integrity audit.

| Class | What it looked like (found 2026-11-23/24) | Check |
|---|---|---|
| Live recomputes a feature differently | 30m swing: stream-built bars, a 500-bar window for daily features; 169 of 190 features off research | `parity` (nightly fresh-pull recompute) |
| Scoring a bar before it closes | SPY day trader scored each 10m bar one minute in since ~09-12; momentum's 4H bars held partial hours | `parity` |
| A static id served as a default | `market_cap_bucket` 0 for every name (30m, Meta): live read a CSV with 212 rows | `drift: stuck` |
| Context frozen at an old date | 30m daily context frozen at 06-03; treasury yields at 07-24 (opt-in fetch nobody ran) | `drift: frozen` |
| A source changes definition | TradingView rows merged into the FOMC/NFP/CPI calendar: 33 events/day vs <= 1 in training | `drift: out_of_range`, `econ_calendar` |
| A loader reads the wrong row kind | earnings loader read nightly snapshots only once both kinds shared a file: 35% coverage vs 95% | `drift: null_jump`, `earnings_forward` |
| A counter beyond its training range | SPY `day_id` grows every session | `drift: out_of_range` |

Rule: before trusting a module's results, its parity must PASS and its drift
flags must be explained (or shown to carry no model gain).

## Adding a check

Add it to `scripts/integrity_audit.py` (one function returning `Check`, plus a
`CHECKS` entry), give it a regression case in `scripts/tests/test_integrity_audit.py`
built from the real trace, and add a row to the table above.
