# DTE selection and the puts-vs-calls asymmetry

**Author:** Claude, 2026-07-28. Script: `scripts/experiment_dte_and_put_call_regime.py`.
Sample: 557 of the 575 real live option fills (18 lack underlying bars).

**Valid-data discipline:** neither study reprices an option. Both use the real recorded
fills plus **underlying daily bars** (dense and reliable) and the point-in-time
market-regime panel. Nothing here depends on the retracted option-bar marks.

---

## 1. Is DTE adaptive? No — it is hardcoded to the nearest expiry

`strategies/multi_ticker_swing/live/runner.py`:

```
_MIN_DTE_DAYS = 0    # allow the nearest listed expiry, including 0DTE/1DTE weeklies
nearest_expiry = min(_contract_expiry(c) for c in tradable ...)
```

Selection is `min(listed expiry)` with a floor of zero. **No regime input, no adaptation to
expected holding time, no relationship to how long the signal's move typically takes.**
Result: median DTE at entry is **2 days**, and **56% of entries are at ≤2 DTE**.

The market-regime module exists and is point-in-time correct, but **nothing in the option
path consumes it.** That is the gap.

---

## 2. The move happens — just not in two days

Favorable underlying excursion available after entry, in the trade's own direction:

| horizon | median MFE | p75 | share reaching +10% |
|---|---:|---:|---:|
| 1–2 days *(what we buy)* | **2.4%** | 4.7% | **4%** |
| 3 days | 3.3% | 6.4% | 13% |
| 5 days | 4.2% | 7.7% | 18% |
| 10 days | 5.6% | 12.1% | 30% |
| 20 days | 8.9% | 19.2% | 46% |
| 30 days | 11.5% | 21.4% | **55%** |
| 45 days | 14.9% | 26.6% | 63% |

**A 2-day expiry captures 2.4% of a move that reaches 11.5% by 30 days.** Only 4% of trades
see a +10% underlying move within 2 days; 55% do within 30.

Extending the horizon:

| extend to | additional favorable excursion (median) | share gaining >2pp more |
|---|---:|---:|
| 10 days | **+1.9pp** | 49% |
| 20 days | **+5.4pp** | 67% |
| 30 days | **+7.8pp** | 71% |

**Even the losers were closed too early:**

| cohort | MFE by 2d | by 10d | by 30d |
|---|---:|---:|---:|
| winners (n=209) | 4.4% | 8.3% | **16.1%** |
| **losers (n=348)** | 1.4% | 3.8% | **9.2%** |

The trades we lost money on went on to make a **9.2% median favorable move** — after we were
out of them. This is the single clearest actionable finding in the whole project: the thesis
was usually right, the clock was wrong.

**Caveat:** more DTE costs more premium, and this study does not price that. It establishes
that the *move exists* to be captured, not that a 30-DTE contract would have been profitable.
Given the cost reference (`11_...md`) — spread is ~fixed in cents, so pricier longer-dated
contracts carry a *lower* percentage toll — the trade-off looks favorable, but it is untested.

---

## 3. Puts: three separate problems, and it is not a regime fluke

Baseline: calls n=258, 45% win, **+$5,827**. Puts n=299, 31% win, **−$32,928**.

### (a) The underlying moves less, and stops extending

| side | MFE 2d | 10d | 20d | 30d |
|---|---:|---:|---:|---:|
| calls | 2.6% | 9.4% | 16.1% | **18.7%** |
| puts | 2.2% | 4.1% | 6.5% | **8.3%** |

Downside moves in this book are **smaller and they stall**. Call excursion compounds 2.6% → 18.7%;
put excursion flattens at 8.3%. Buying puts fights the book's natural drift.

### (b) Puts pay less than calls for the *identical* underlying move — this is skew

Conditioning on how far the underlying actually moved in the option's favor within 2 days:

| underlying move | calls median return | puts median return | gap |
|---|---:|---:|---:|
| 0–2% | −36% | **−52%** | −16pp |
| 2–5% | +7% | **−11%** | −18pp |
| 5–10% | +27% | **+12%** | −15pp |

A consistent **15–18pp penalty at every move size.** Same directional accuracy, materially worse
payoff — that is put skew: you pay more premium for the same delta.

### (c) The headline is tail-driven, but the median put still loses

| | value |
|---|---:|
| puts total | −$32,928 |
| worst 10 put trades | **−$24,026 (73% of the loss)** |
| puts excluding worst 10 | −$8,902 over n=289 |
| median put trade | −$19 |
| median call trade | −$4 |

So the catastrophic headline is 10 trades. Excluding them puts are *mildly* negative. But the
median put still loses more than the median call, so there is a systematic component underneath
the tail.

### (d) Regime does not explain it

| regime at entry | side | n | win | total P&L | underlying MFE 2d |
|---|---|---:|---:|---:|---:|
| risk-OFF | calls | 99 | 35% | +$708 | 1.57% |
| risk-OFF | **puts** | 111 | 29% | **−$13,772** | **2.29%** |
| mid | calls | 77 | 40% | +$2,322 | 3.39% |
| mid | puts | 99 | 39% | −$4,308 | 2.83% |
| risk-ON | calls | 82 | **60%** | +$2,797 | 3.58% |
| risk-ON | puts | 89 | 26% | −$14,848 | 1.18% |

**Puts lose in all three regimes**, so this is not a regime artifact. The risk-OFF row is the
most damning: puts had *better* underlying movement than calls (2.29% vs 1.57%) and still lost
$13,772 while calls made $708. When direction is on your side and you still lose, the problem is
cost and time decay, not the signal.

Calls in risk-ON are the standout cell (60% win rate) — the one genuinely favorable combination
found.

---

## 4. Recommendations

1. **Replace `min(expiry)` with an adaptive DTE rule.** This is the highest-value change available.
   The move needs 10–30 days; we buy 2. A floor of ~14–21 DTE, or DTE scaled to the signal's
   expected holding time, is the obvious first version.
2. **Feed the regime panel into the option path.** It exists, it is point-in-time correct, and
   nothing consumes it. At minimum: calls in risk-ON is the one strong cell.
3. **Stop buying puts, or raise their bar substantially.** Three independent problems stack —
   smaller/stalling downside moves, a 15–18pp skew penalty at equal accuracy, and losses in every
   regime. Express short exposure in shares instead.
4. **Do not read this as "longer DTE is proven profitable."** It proves the *move exists* beyond
   our expiry. Whether a longer-dated contract captures it net of premium is untested and needs
   forward-captured marks.
