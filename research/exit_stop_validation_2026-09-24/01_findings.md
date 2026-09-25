# Step 1 — the underlying stop works, and it is being silently bypassed 1,122 times

2026-09-24. Script: `01_underlying_stop_validation.py`. Output:
`data/underlying_stop_validation.json`. Source: `Data/inference/*/closed_trades.jsonl`
(543 closed trades, 2026-07-09 .. 2026-09-23) + `logs/`.

## First, the question that prompted this: yes, the stop IS on the underlying now

`core/live_4h_exec.py:306-318` is an **if/elif**. For an option with a usable underlying
basis the premium stop is **not evaluated at all**:

```python
u_stop = underlying_stop_level(policy, u_entry, u_atr) if route == "option" else None
if u_stop is not None and u_now is not None:
    if float(u_now) <= u_stop:
        return "exit", f"underlying_stop_-{policy.underlying_stop_atr:g}atr"
elif policy.stop_loss and gain is not None and gain <= -policy.stop_loss:
    return "exit", f"stop_-{int(policy.stop_loss * 100)}%"
```

`−39%` premium survives only for **equity** and for **options whose basis could not be
established** (deliberate fail-safe: "a half-known basis must never widen a stop silently").
`core/live_risk_pass.py:163-169` mirrors it correctly. It has fired **34 times** live.

## A. It does what it was designed to do

Underlying move at the moment the stop fired, live modules only (`dealer_ranker` excluded as
deprecated), n=19 with a resolvable basis:

| | baseline (42 premium stops, 07-17..08-18) | **now (underlying stop)** |
|---|---|---|
| median underlying move at the stop | **−3.1%** | **−6.50%** |
| in ATR units | ≈ −0.5 ATR | **−2.13 ATR** |
| share firing with underlying down < 2% | **42.9%** (18 of 42) | **10.5%** |

**The noise-exit problem is fixed: 42.9% → 10.5%.** That was the whole point of the change and
it is delivered.

Two details worth recording:
* It fires at a median **−2.13 ATR**, not −1.5 — about 0.6 ATR of overshoot, because the
  checks happen on a sampling cadence and the price gaps through the level.
* **The cost is real.** When it fires, the premium is already at a median **−65.6%** against
  the old premium stop's −40.2%. Live modules: 25 underlying stops, **−$69,249**, mean −$2,770.
  That is the arithmetic of a wider stop, not a defect — but it means the change trades more
  loss-per-stop for far fewer false stops.

## B. Aggregate direction is good, and confounded

Option route, live modules only:

| period | n | total | **per trade** | median | win rate | median runs held |
|---|---|---|---|---|---|---|
| pre 2026-08-18 | 33 | −$104,600 | **−$3,170** | −$2,310 | 15% | 3 |
| post | 248 | −$90,428 | **−$419** | −$70 | 20% | 9 |

A **7.6x improvement in loss per trade**. But this is a before/after on a live book, not a
controlled comparison: n=33 vs 248, the hold roughly tripled, the book's module composition
changed, and the tape moved. **Direction only — do not attribute the magnitude to this change.**

The full counterfactual ("would the premium stop have closed this earlier, and at what
price?") needs a historical option **premium path**, which the 2026-07 retraction established
we do not have. Every premium figure here is the mark at the exit actually taken.

---

# C. THE BUG — the fix is disabled whenever IEX cannot quote the underlying

`core/risk_prices.py:33-52`. The between-bar risk pass — which its own docstring calls
*"the path that actually fires most stops"* — reads underlying quotes from the **IEX** feed
(`RISK_UNDERLYING_FEED`, default `iex`; the account's `APCA_API_DATA_FEED` is also `iex`) and
requires a two-sided quote **no more than 60 seconds old**. When that fails it returns `None`
and logs, verbatim:

```
<TICKER> current underlying unavailable/invalid (feed=iex); premium fallback active
```

`u_now is None` then sends `live_risk_pass.py` down the `elif` and **the −39% premium stop
fires** — the exact behaviour the 2026-08-18 change exists to prevent.

**Scope in the logs: 1,122 fallback events across 18 tickers.**

| NTSK | CRWD | PURR | ASST | DELL | AXTI | MRNA | HL | SMTC | DINO | GFS | others |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 199 | 199 | 127 | 122 | 120 | 115 | 67 | 67 | 46 | 35 | 13 | ≤ 4 each |

Note CRWD, DELL and NTAP are large caps — so this is **not** only an illiquidity problem. IEX
carries a small share of US volume and frequently has no fresh two-sided quote for a given
symbol at a given moment.

**The measured consequence.** 13 of 47 option hard stops since 2026-08-18 (**27.7%**) fired on
premium instead of the underlying, for **−$30,919**. They split into two groups:

| group | n | u_entry recorded? | reading |
|---|---|---|---|
| entered **before** 2026-08-18 (CRWV, AAOI, LITE, AXTI, NBIS, CRDO) | 6 | no | correct fail-safe on legacy positions — one-time, expected |
| entered **after**, `multi_ticker_swing_htf` (WULF, MRNA, PURR, SMCI, APLD, EIX, HL) | 7 | **yes** | **the bug**: a valid basis existed and the premium stop fired anyway. −$18,971 |

PURR and MRNA between them account for 194 fallback events, and both appear in that second
group. `decision_gain` is null on 6 of the 7, consistent with the exit coming from the risk
pass rather than the 4H engine.

## C.1 Options for the fix — NOT APPLIED, this is a live risk parameter

| option | cost | note |
|---|---|---|
| **switch `RISK_UNDERLYING_FEED=sip`** | a paid Alpaca data subscription | the account is on `iex`; SIP would largely remove the gap but is not currently entitled |
| **fall back to the most recent underlying BAR close** instead of to the premium stop | small code change in `risk_prices.py` / the caller | keeps the stop underlying-referenced, just laggier. A 4-hour-stale underlying price still says far more about "is it below entry −1.5 ATR" than the premium does. `build_mixed_plan` already reads bars via `ufn(tkr, entry_bar)`, so the machinery exists |
| **HOLD when the quote is unavailable** | smallest change | an unavailable quote is not evidence of an adverse move; the 4H pass catches it next bar and `expiring_before_next_session` still covers expiry. Risk: a genuinely collapsing option goes unstopped between bars |

My recommendation is the **bar-close fallback**, with a distinct log/exit-reason string so the
substitution stays measurable. But this changes when live stops fire, so it needs your explicit
go-ahead — I have not touched it.

## C.2 A smaller instrumentation gap

9 of 34 `underlying_stop` rows carry a **null `u_entry`** in the closed-trade record even
though the rule could not have fired without a basis. The basis is not being persisted into the
ledger on every path. Harmless to trading, but it makes this audit inexact — worth a one-line
fix so the next validation is exact rather than 26-of-34.
