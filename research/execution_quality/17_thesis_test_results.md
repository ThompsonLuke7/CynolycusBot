# Thesis test — measured on real option prices

2026-09-02. `scripts/thesis_test/{fetch_contract_paths,validate_and_simulate}.py`

## Did we already have the data? No — but Alpaca does

`Data/inference/spy/option_marks/` is **empty**: the mark-capture code exists but
has never written a file. There is no stored option price history in the repo.

However, Alpaca serves **historical daily option bars** for expired contracts
(`/v1beta1/options/bars`, with contract lookup requiring `status="inactive"` —
the default silently returns nothing for a past expiry, which reads identically
to "this name has no chain"). So the whole test can be run on **measured** prices
with nothing modelled.

## Validation first (non-negotiable)

206 historical 4H entries → a ~35–45 DTE monthly call each → real daily bars.

| | |
|---|---|
| entries with option bars | 58 / 206 |
| **usable after validation** | **39** |
| rejected: low coverage | 9 |
| rejected: too few bars | 6 |
| rejected: corr too low | 4 |
| median coverage | 91% of trading days |
| **median corr(option return, underlying return)** | **+0.778** |
| median stale share | 0% |

+0.778 is the number that makes this study valid where the 2026-07 one was not
(**+0.09** there). Nineteen contracts were thrown out rather than averaged in.

## Result

Mean/median return per trade, and total $ per $1,000 deployed, on the 39 usable
contracts:

| policy | median | **mean** | win% | p90 | **$/1k** |
|---|---|---|---|---|---|
| OPTION 5d, −39% stop *(live policy)* | −39.0% | −7.4% | 26% | 80.8% | **−2,874** |
| OPTION 5d, no stop | −22.7% | +4.4% | 38% | 97.3% | **+1,707** |
| SHARES 5d | −2.8% | −1.8% | 38% | 15.6% | −705 |
| OPTION 10d, −39% stop | −39.0% | −8.1% | 26% | 70.3% | −3,158 |
| **OPTION 10d, no stop** | −5.4% | **+7.9%** | 46% | 125.6% | **+3,092** |
| SHARES 10d | +2.1% | +1.8% | 56% | 24.0% | +696 |
| OPTION 15d, no stop | −58.6% | −0.5% | 36% | 182.0% | −179 |
| SHARES 15d | +1.3% | +2.9% | 54% | 24.8% | +1,138 |
| OPTION 20d, no stop | −65.4% | −6.2% | 31% | 150.6% | −2,419 |
| SHARES 20d | +5.2% | +6.1% | 64% | 31.7% | **+2,360** |

## What this says

**1. The stop is the single biggest destroyer, exactly as the hold-length
diagnosis implied.** The live −39% premium stop loses $2,874–$4,463 per $1,000 at
every horizon. Removing it turns the 10-day cell from −$3,158 into **+$3,092** on
the same entries and the same contracts. That is a swing of over $6,000 per
$1,000 deployed, from one policy parameter.

This is consistent with the mechanical finding: **62% of live exits are stops,
firing at a median of 2.0 trading days**, and a −39% premium stop on a
7.8x-levered contract triggers on roughly a **5% underlying move** — about 3.2x
too tight for a hold whose median adverse excursion is −16%.

**2. Your thesis is right about direction and about options — and wrong about the
length.** Options beat shares decisively at 5 and 10 days (+$1,707 / +$3,092 vs
−$705 / +$696), driven by exactly the right tail you described: p90 of +97% and
+126%. But by 15–20 days theta wins: options go to −$179 and −$2,419 while shares
climb to +$1,138 and +$2,360.

**The sweet spot measured here is ~10 trading days (two weeks) with no premium
stop** — inside your stated 1–3 week band, at its short end.

**3. Horizon and instrument point in opposite directions.** If you hold 15–20
days, shares are the better expression. If you express in options, hold ~10 days.
The current system does the worst available combination: option expression, short
hold, and a stop that fires before either can work.

## Limits — read these before acting

* **n = 39.** Cell-to-cell differences are not reliable. What is robust is the
  ordering: live-stop ≪ no-stop, and 10d ≥ 5d > 15d > 20d for options.
* The entries are the ones the **current, weak ranker** chose. A better ranker
  would change the tail; a worse one too.
* Daily bars mean intraday stop paths are approximated by the daily low, which
  makes stop-based policies look slightly *worse* than reality.
* One ~2-month sample, one regime.
* Fills are bar closes, not quotes — no spread is charged on entry or exit. Real
  option spreads on these names are wide, so every option row above is optimistic
  by roughly one half-spread each way.

The last point matters most: **the +$3,092 at 10d is a gross figure**, and option
spreads on small caps can be 10–30% of mid. Confirming this properly needs the
same replay against bid/ask, which is what the (currently unused) mark-capture
infrastructure is for.

---

# Addendum — where the −39% stop came from, and the hold/stop interaction

## Provenance: it was never validated on options

Traced through the log:

* The default came from `research/capstone/exit_policy_cross_module.csv`, policy
  **"id4 tail-rider"**, which set `stop_loss .50 -> .39`, `horizon_bars 25 -> 53`,
  `trail_stop .35 -> None` (2026-07-19). id4 won that comparison cleanly on all
  three modules.
* **That harness is shares-only.** The same log entry records it explicitly:
  *"SHARES ONLY (4H stock OHLC; no option-premium path anywhere) … the deployed
  50% stop binds on 1 of 1430 stock trades — inert on shares, binds constantly on
  decaying OTM premium … so this harness CANNOT evaluate it."*
* The stop was therefore tuned where it is **inert** (1 in 1,430) and then applied
  to leveraged option premium where it binds on roughly **half** of trades.
* Prior MFE work pointed the same way and was never acted on for options:
  *"winners' pre-peak MAE median only −4.3% → tight trails/stops cut runners not
  risk"* (2,267 val entries), and separately *"wide 5-ATR stops beat tight 2-ATR"*.
* Live evidence agreed: nine option stops labelled −39% realised a mean of
  **−57.1%**, worst −88.5%.

So this result does not contradict the earlier work. **It is the first actual
measurement of the thing the earlier work said its harness could not measure.**

## The stop and the horizon are one decision, not two

Total $ per $1,000 deployed, 39 usable contracts:

| hold | −39% (live) | −50% | −60% | −75% | none | SHARES |
|---|---|---|---|---|---|---|
| 5d | −2,874 | −4,725 | +140 | +1,875 | +1,707 | −705 |
| **8d** | −824 | −3,613 | +3,590 | +4,332 | **+5,110** | −836 |
| 10d | −3,158 | −5,579 | +1,256 | +2,886 | +3,092 | +696 |
| 13d | −3,110 | −6,630 | +2,925 | +2,232 | +2,657 | +910 |
| 15d | −4,463 | −8,093 | +498 | −308 | −179 | +1,138 |
| 20d | −4,341 | −7,053 | −483 | −2,565 | −2,419 | +2,360 |
| 25d | −4,643 | −7,153 | −699 | −2,165 | −68 | **+2,495** |

**The live policy is `horizon_bars = 53` (~21 trading days) with a −39% stop —
the 20–25d row of the worst column.**

Two things follow that matter more than any single cell:

1. **Removing the stop alone would not fix it.** With the horizon left at 53 bars
   you land in the 20–25d/no-stop cell: −$2,419 and −$68. The stop and the
   horizon have to move together.
2. **Options and shares swap places at ~13–15 days.** Options win the 5–13d band;
   shares win from 15d out, and win biggest at 25d (+$2,495). That is a cleaner
   statement than "options are bad" or "options are good".

## Do not over-fit this grid

**−50% is worse than −39% at every horizon.** That is non-monotonic and cannot be
a real effect — it is n=39 talking. Read the grid as a *shape*, not a lookup
table:

* very tight premium stops on leveraged options are destructive
* long holds on options are destructive (theta)
* the good region is a moderate hold (~8–13 days) with a loose or absent stop
* shares are the better expression once you want to hold past ~15 days

Any specific number from it — 8 days, −60%, −75% — is inside the noise.

## Still unmodelled: the spread

Entries and exits are struck at bar closes with **no spread charged**. Real option
spreads on these names run 10–30% of mid, and the round trip is charged twice.
A +$3,092 gross figure could plausibly be halved. Nothing here should go live
before the same replay is run against bid/ask.
