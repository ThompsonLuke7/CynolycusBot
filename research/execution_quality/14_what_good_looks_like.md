# What "good" actually looks like — external benchmarks

2026-08-31. External research, with sources.

## 1. Is IC 0.04–0.10 good? Yes. It is the state of the art.

Microsoft's **Qlib** publishes a standardised benchmark: ~30 models, 20 random
seeds each, identical data (CSI300), identical evaluation. This is the cleanest
apples-to-apples "what does a good model score" reference that exists publicly.

**Alpha158** (engineered features — the closest analogue to our matrix):

| model | IC | ICIR | ann. return | IR |
|---|---|---|---|---|
| **XGBoost** | **0.0498** | 0.3779 | 7.8% | 0.91 |
| LightGBM | 0.0448 | 0.3660 | 9.0% | 1.02 |
| TRA (best) | 0.0440 | 0.3535 | 7.2% | 1.08 |
| MLP | 0.0376 | 0.2846 | 9.0% | 1.14 |
| TFT | 0.0358 | 0.2160 | 8.5% | 0.81 |
| LSTM | 0.0318 | 0.2367 | 3.8% | 0.56 |
| GRU | 0.0315 | 0.2450 | 3.4% | 0.52 |
| **Transformer** | **0.0264** | 0.2053 | 2.7% | 0.40 |

**Alpha360** (raw sequences — where a sequence model should win):

| model | IC | ann. return |
|---|---|---|
| HIST (best overall) | 0.0522 | 9.9% |
| GRU | 0.0493 | 7.2% |
| LSTM | 0.0448 | 6.5% |
| LightGBM | 0.0400 | 5.6% |
| XGBoost | 0.0394 | 3.4% |
| **Transformer** | **0.0114** | **−2.7%** |

Three things follow directly.

**IC ~0.05 is the ceiling of published practice, not a floor.** The best model in
the whole benchmark is 0.0522. The academic literature agrees: "IC values
encountered in practice are less than one tenth", and IC ≥ 0.05 is described as
significant predictive power.

**A vanilla XGBoost is a top-3 model on engineered features.** Your architecture
choice was right, and it is empirically supported rather than a compromise.

**The Transformer is the WORST model on both datasets** — 0.0264 and 0.0114,
against XGBoost's 0.0498 and 0.0394, and it loses money on Alpha360. That is the
direct answer to "should we try a transformer": on this task, on standardised
data, it is not close. Sequence models (GRU/LSTM/HIST) only beat trees on
**Alpha360**, i.e. on *raw price sequences*, not on engineered feature rows. If
you ever want a sequence model, the input has to change first — feeding a
transformer our 106 engineered features is feeding it the dataset it loses on.

**What IC 0.05 actually pays.** Note the return columns: 3–10% annualised with
IR 0.4–1.4. That is what a genuinely state-of-the-art cross-sectional equity
signal buys — on a well-constructed, low-cost, high-breadth book. Not 100% a year.

Caveat on comparability: our bake-off IC was measured against forward **MFE**
(a maximum, which is positively biased and more volatility-linked) rather than
forward **return**. Ours is therefore optimistic relative to a Qlib-style IC, and
"our 0.10 beats their 0.052" is not a claim I would make. Same order of
magnitude is the honest reading.

## 2. Where the money is actually made at IC ~0.05

Gu, Kelly & Xiu (2020), *Review of Financial Studies* — the canonical ML
asset-pricing paper, US equities 1957–2016:

* ensemble ML out-of-sample **Sharpe 0.45–0.61** vs 0.35 for regression benchmarks
* a long-short **decile spread** on neural-net predictions: annualised OOS
  Sharpe **1.35 value-weighted / 1.45 equal-weighted**
* trees and neural nets win, via nonlinear interactions
* the dominant signals across all methods are **momentum, liquidity, volatility**

Note the mechanism: the Sharpe comes from a **long-short decile spread across
thousands of names**, not from concentrated directional bets. That is how a
0.05 IC is monetised — breadth and cheapness, not conviction.

## 3. What the recent intraday/cost literature says

This is the part most relevant to us:

* a 2026 study finds a "significant disconnect between model prediction accuracy
  and actual net returns" — marginal accuracy gains from complex models **cannot
  offset transaction frictions** in high-frequency trading
* **"simple signal filtering rules designed based on transaction costs deliver
  far better profitability improvements than iterative optimization of deep
  learning architectures"**
* strategies with impressive single-path backtests show returns "extremely
  unevenly distributed across time intervals with weak statistical significance"

That is our result, arrived at independently: our signal is normal-strength and
the option wrapper charged 22.8% against a 4.5% edge. The literature's advice —
spend the effort on cost-aware filtering, not architecture — is precisely what
the Stage-5 counterfactuals and the instrument addendum concluded.

## 4. So what should change, concretely

**Not the model family.** XGBoost is at the benchmark frontier for this data shape.

**Not more architecture.** The transformer is measurably worse here, and the
cost literature says architecture is the wrong lever.

Worth trying, in rough order of expected value:

1. **Harvest the existing signal in shares.** Everything above says a 0.05-IC
   signal pays via breadth and low cost. We already hold ~144 concurrent
   positions, so breadth is there; the wrapper is what is killing it.
2. **Long-short, not long-only.** Every benchmark result above is a decile
   *spread*. We rank a universe and only ever buy the top. Half the signal —
   the short leg — is currently discarded.
3. **A sequence model on raw sequences, if at all.** GRU beat trees on Alpha360
   (0.0493 vs 0.0394). That is a real result, but it requires a raw-bar input
   pipeline, not our feature matrix. A genuine project, not a swap.
4. **Ensembling across families.** Gu/Kelly/Xiu's gains come substantially from
   ensembles; the competition harness already trains several families, so this
   is mostly a matter of combining rather than picking a single winner.
5. **The intraday engine is a different, legitimate strategy class.** Nothing in
   the cross-sectional literature governs it, and the cost literature actively
   favours it: short holds, cheap instruments, cost-aware rules. It should be
   evaluated on its own terms once it has real fills.

## Sources

- [Qlib benchmark results (Microsoft)](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md)
- [Gu, Kelly & Xiu, *Empirical Asset Pricing via Machine Learning*, RFS 33(5)](https://academic.oup.com/rfs/article/33/5/2223/5758276) · [author PDF](https://dachxiu.chicagobooth.edu/download/ML.pdf)
- [The Fundamental Law of Active Management: Redux (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S0927539817300543)
- [Information Coefficient as a Performance Measure of Stock Selection Models (arXiv 2010.08601)](https://arxiv.org/pdf/2010.08601)
- [Research on ML High-Frequency Trading Strategies Under Transaction Cost](https://ojs.shiharr.com/index.php/eaou/article/view/1672)
- [The Expected Returns on Machine-Learning Strategies (AFA)](https://afajof.org/management/viewp.php?n=75544)
- [Information Coefficient (IC) — FE Training](https://www.fe.training/free-resources/portfolio-management/information-coefficient-ic/)
