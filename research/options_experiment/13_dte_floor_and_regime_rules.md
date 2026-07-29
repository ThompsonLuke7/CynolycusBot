# DTE floor (as a rule), regime routing rules, and the put-timescale test

**Author:** Claude, 2026-07-28. Script: `scripts/experiment_dte_rule_and_regime_routing.py`.
557 real fills + dense underlying bars + point-in-time regime panel. No option repricing.

---

## 1. Don't train a model for DTE — the data says a flat floor is correct

You didn't want an ML model for expiry selection. The data agrees, for a concrete reason:
**nothing observable at signal time predicts how fast the move arrives.**

Correlation with days-to-reach-+10%:

| observable | r |
|---|---:|
| atr_pct | −0.219 |
| rv20 | −0.155 |
| liquidity_stress_rv20_z | +0.135 |
| spy_rv20_z | +0.135 |
| risk_appetite_iwm_spy_z | −0.126 |
| *every other regime factor* | < 0.13 |

Best predictor is |r| = 0.22 — higher-ATR names move a bit faster, which is intuitive but far too
weak to size a per-trade expiry from. **A per-trade adaptive DTE is not supportable. A flat floor is.**

### How long the move actually takes

67% of trades (372/557) eventually reach a +10% favorable underlying move within 60 days.
For those: **median 10 days, p25 3, p75 22, p90 38.**

### Floor candidates

| DTE floor | share of achievable +10% moves captured |
|---|---:|
| **2d (current)** | **19%** |
| 7d | 41% |
| 14d | 60% |
| **21d** | **74%** |
| 30d | 83% |
| 45d | 95% |

The current 2-day floor captures **19%** of the moves that actually occur. Going to 21 days
captures 74%; 30 days captures 83%.

**Recommendation: a flat floor of 21–30 DTE.** 21d is the efficiency knee (74% capture); 30d buys
another 9pp. Below 14d you are systematically buying less time than the move needs.

**Untested caveat, stated plainly:** longer-dated contracts cost more premium, and this study
cannot price that (option marks are unusable historically — see `10_RETRACTION...`). What it
establishes is that the *move exists* to be captured. The cost direction is favorable — spread is
roughly fixed in cents, so a pricier longer-dated contract carries a *lower* percentage toll
(`11_option_cost_reference.md`) — but net profitability is unproven.

---

## 2. Regime routing rules — one works, modestly, and it costs volume

Simple pre-stated rules applied to the real call fills:

| rule | n | win rate | total P&L | $/trade |
|---|---:|---:|---:|---:|
| **calls only when `risk_appetite_z > 0`** | **82** | **60%** | $2,797 | **$34** |
| calls when risk-ON and low stress | 82 | 60% | $2,797 | $34 |
| calls only when `breadth_z > 0` | 113 | 54% | $441 | $4 |
| calls only when `liquidity_stress_z < 0` | 258 | 45% | $5,827 | $23 |
| *(all calls, no rule)* | 258 | 45% | $5,827 | $23 |

**Findings:**

- **`risk_appetite_z > 0` is the only rule that helps**: win rate 45% → **60%**, per-trade P&L
  $23 → $34. But it cuts trade count by 68% (258 → 82), so *total* P&L falls to $2,797. Whether
  that is an improvement depends entirely on whether the freed capital is redeployed.
- **`liquidity_stress_z < 0` is inert** — it matches all 258 trades. Liquidity stress was negative
  throughout this sample, so the variable has no discriminating power here. A rule built on it
  would be untested in the regime that matters.
- **`breadth_z > 0` is actively harmful**: it keeps 113 trades but only $441 of P&L, i.e. it
  filters *out* the profitable ones.
- `risk_appetite_z > 0.5` matches **zero** trades — the factor's range in this window is narrow, so
  aggressive thresholds are untestable.

**Honest assessment:** n=82 on a single module's book over ~4 months is thin evidence. The
direction is plausible (buy calls when risk appetite is positive) and it agrees with the regime
tercile split found earlier, but I would treat it as a *candidate* rule to shadow-track, not one to
deploy on this evidence alone.

---

## 3. Puts and timescale — your intuition is half right, and the half that's wrong matters

Median favorable underlying excursion, calls vs puts:

| horizon | calls | puts | put/call |
|---|---:|---:|---:|
| 5d | 5.3% | 3.3% | 0.63 |
| 10d | 9.4% | 4.1% | 0.43 |
| 20d | 16.1% | 6.5% | **0.40** |
| 30d | 18.7% | 8.3% | 0.44 |
| 45d | 20.3% | 11.7% | 0.58 |
| **60d** | 21.3% | **13.5%** | **0.63** |
| 90d | 21.4% | 14.0% | 0.65 |
| 120d | 21.4% | 14.0% | 0.65 |
| 180d | 21.4% | 14.0% | 0.65 |

Share of put trades whose underlying eventually fell >20%:

| horizon | share |
|---|---:|
| 30d | 13% |
| **60d** | **33%** |
| 90d | 36% |
| 120d+ | 36% (flat) |

**You're right that the timescale is wrong.** Put excursion improves substantially from 30d to 60d
(8.3% → 13.5%), and the fraction of names falling >20% nearly triples (13% → 33%). On a 2-day
clock that is entirely invisible. The put/call ratio recovers from its 0.40 trough at 20d to 0.65
by 90d.

**But it plateaus hard at 60–90 days and never catches calls.** Beyond 90 days there is literally
no additional downside excursion in this sample — 14.0% at 90d, 120d and 180d alike.

**The important limitation:** this measures the *swing module's* put entries held longer. It does
**not** test your actual MSTR thesis, which is a different **entry criterion** — buying puts at a
thematic/cycle exhaustion point, not at a 30-minute swing signal. A −80% cycle move over many
months would show up as sustained excursion growth past 90 days, and there is none here, because
these entries are not selecting cycle tops. They are selecting short-horizon swing setups that
happen to be short.

So: your intuition likely needs a **different signal**, not a longer hold on this one. That is a
genuinely separate research question — a theme/cycle-exhaustion detector — and nothing here argues
against it.

---

## 4. Decisions and next steps

- **Shorts → shares** (user decision, accepted). Puts as a hedge instrument rather than a
  directional expression. Supported by everything in `12_...md`: smaller and stalling downside
  excursion, a 15–18pp skew penalty at equal directional accuracy, and losses across all regimes.
- **Implement a flat 21–30 DTE floor** replacing `_MIN_DTE_DAYS = 0` / `min(expiry)`.
- **Shadow-track `risk_appetite_z > 0` for call routing** rather than deploying it — n=82.
- **Do not build an ML DTE selector.** Nothing predicts move speed (best |r| = 0.22).
- **Open question worth its own project:** a thematic/cycle-exhaustion signal for the short side,
  which is what the MSTR example actually describes.
