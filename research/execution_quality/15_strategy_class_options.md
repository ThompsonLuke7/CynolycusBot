# Label and strategy classes beyond cross-sectional ranking

2026-09-01.

## Correction that changes the whole picture

Qlib's benchmark "Annualized Return" column is **`excess_return_with_cost`** =
`portfolio return − benchmark return − transaction costs`. It is **excess over the
index, after costs** — not total return.

So the table I showed reads:

| model | excess over index, after cost | IR |
|---|---|---|
| LightGBM (Alpha158) | **+9.0%** | 1.02 |
| HIST (Alpha360) | **+9.9%** | 1.37 |
| XGBoost (Alpha158) | +7.8% | 0.91 |

That is index **+9%**, not "9% total". Against SPY's ~10% total at Sharpe ~0.4–0.5,
a strategy earning the index plus 9% of uncorrelated excess at IR ~1.0 is roughly
double the risk-adjusted return — and the excess component can be levered because
it is not market beta. My earlier presentation implied these models barely match
buy-and-hold. They do not; they beat it substantially. The misreading was caused
by how I presented the table.

## Label classes other than cross-sectional ranking

Cross-sectional ranking ("which of these 3,000 names is best right now") is one
choice among several, and it is the one most exposed to weak IC, because it asks
the model to be right about *relative* ordering across a huge, noisy universe.

**1. Meta-labelling (López de Prado).** The primary model is a RULE that decides
*direction and entry*; the ML model only predicts **whether to take the trade and
how large**. Target is binary: did this specific rule-triggered setup hit its
target before its stop.

This is the best structural fit for this repo by a wide margin. It uses the
intraday engine as the primary model — a rules engine already believed in and
independently observed to work — and asks ML only the narrow question it is good
at. The sample is small and dense rather than huge and noisy, and a modest AUC on
"will this setup work" converts directly into position sizing, which is where
Kelly-style gains come from.

**2. Time-series (per-asset) rather than cross-sectional.** Predict each name's
forward move against *its own* history rather than against 3,000 peers. Different
sample construction, different label, avoids the cross-sectional noise floor.

**3. Event-conditional labels.** Only label bars that follow a catalyst — earnings,
gap, halt-resume, news spike. Far fewer rows, far higher signal density. The repo
already has the catalyst and news infrastructure to define the conditioning event.

**4. Triple-barrier labelling.** Path-dependent: label by which of {profit target,
stop, time limit} is hit first. Respects the actual exit policy instead of
measuring a horizon the trade never holds to. Worth knowing this is *not* what the
current labels do, despite the earlier assumption that it was.

**5. Duration / hazard models.** Predict *time until the move dies* rather than its
size. Directly addresses the finding that the exits give back the entire MFE.

**6. Regime-conditional models.** Separate models per regime rather than one model
with regime features. Costly in data, but the regime panel now exists to define
the split.

## What the external evidence says about where a small account can win

* Microcaps are "potentially one of the most alpha-rich but execution-constrained
  areas of public markets" — thin analyst coverage, slow information discovery.
* Institutions structurally cannot go there: a $5B plan would need a ~$100M
  microcap allocation to move the needle, which the free float will not absorb.
* Short-term reversal strategies are "the most constrained by trading costs" and
  small-cap signals "decay faster".

Read together: **the edges that do not scale are exactly the ones a small account
can access, and they are short-horizon and small-cap.** That is an argument for
the intraday engine and against trying to out-rank institutions on a 3,000-name
multi-week cross-section, which is the game they are best at and most capitalised
for.

The counterweight, stated plainly: day-trading base rates are brutal. Taiwan,
every trade on the national exchange 1992–2006 — under 1% of day traders were
reliably profitable net of fees. Brazil, index futures — of those who persisted
beyond 300 sessions, 97% lost money and 1.1% out-earned minimum wage. "I see it
working all the time" is one of the most survivorship-biased observations
available. The correct conclusion is not "it is impossible", it is "the edge has
to be demonstrated on your own fills, not inferred from other people's".

## Sources

- [Qlib metric definitions — excess_return_with_cost](https://qlib.readthedocs.io/en/latest/component/report.html)
- [Qlib TopkDropoutStrategy / benchmark config](https://qlib.readthedocs.io/en/latest/component/strategy.html)
- [Microcaps — Factor Spreads, Structural Biases, and the Institutional Imperative (OSAM)](https://www.osam.com/Commentary/microcaps-factor-spreads-structural-biases-and-the-institutional-imperative)
- [Day Traders Lose Money and Keep Trading: Evidence from Taiwan (Barber, Lee, Liu, Odean)](https://www.tradicted.com/research/barber-learning-2020/)
- [The Profitability of Day Traders](https://www.researchgate.net/publication/228289199_The_Profitability_of_Day_Traders)
- [Gu, Kelly & Xiu, RFS 33(5)](https://academic.oup.com/rfs/article/33/5/2223/5758276)
