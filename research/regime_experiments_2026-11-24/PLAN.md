# Regime experiments: where regime belongs, how fine-grained it must be, and what it is missing

**Pre-registered 2026-11-24 15:19 ET**, before any result below was computed. E0 has run; everything
else is designed, not run. Each experiment lists its own question, data, method, control,
outputs, decision rule and where it runs.

## 0. The architecture these experiments test

- **One shared market state, published by the nervous system.** It has two clocks:
  - the **daily regime**: today, the 7-signal vote (`signals/market_regime/regime_vote.py`), known
    at the prior close;
  - an **intraday session state**: trend vs range day, volatility expansion and event shocks,
    refreshed during the session. E5 decides how often.
- **Each module owns a regime profile:** how *its* order policy responds to the shared state, given
  its holding period and goal. The state is shared; the response is not. Illustrative only — E7
  decides the actual mappings:

  | Module | Style | What the regime may change |
  |---|---|---|
  | Intraday structure | 0DTE/1DTE, minutes–hours | side bias, hold/exit rule, size (0DTE always allowed) |
  | SPY daytrader | 10-min bars, intraday | long/short probability thresholds (already built, off) |
  | Momentum (4H) | multi-day breakouts | entry threshold, size, max hold |
  | HTF swing | multi-day swings | entry threshold, size |
  | Meta | confluence, multi-day | size (multipliers live since 2026-11-23), threshold |
  | Dealer ranker | positioning levels | side bias, size |

- **Models keep one simple target each.** The regime lives in the policy unless E6 shows a
  model's own target is predicted better with it.

## 1. Regime factor inventory: what we have, what is missing, how far back

| Family | What it captures | In the live vote? | Daily history available |
|---|---|---|---|
| F1 Trend | price direction and persistence | yes (QQQ fast, QQQ<SMA50, SPY EMA20/50) | S&P 1928, Nasdaq-100 1985 |
| F2 Volatility | realised and implied vol, VIX term structure | realised only (spy_rv20_z) | realised 1928; VIX 1990; VIX3M 2006 |
| F3 Credit / liquidity | spreads, funding stress | yes (HYG/IEI) | HYG 2007; **Baa spread 1986 and HY OAS 1996 need a free FRED key** |
| F4 Rates / policy | 10y level and change, curve, Fed path | **no** | yields 1962, 3m bill 1960; fed funds via FRED |
| F5 Dollar / commodities | USD stress, oil shocks | **no** | dollar 1971; oil 2000 (1986 via FRED) |
| F6 Breadth / participation / correlation | narrow vs broad market, everything-moves-together | yes (11-sector breadth) | sector SPDRs 1998; RSP 2003 |
| F7 Sentiment / positioning | news tone, put/call, dealer gamma, social | **no** | news 2026-05; dealer 2026-07 (85 days); short history only |
| F8 Event calendar | FOMC, CPI, NFP, earnings season, OpEx | **no** | in repo FOMC 2022+ (earlier dates are public) |
| F9 Reaction to news | "good news sold" (beats that fall) | **no** | earnings dates in repo; short history |

The user's AI example (strong earnings, falling prices, fearful narrative) is F9, and it is
measured from prices: how the market *reacts* to good news. It is the leading candidate for a
"why" factor that separates similar-looking states.

## 2. Experiments

### E0 Data foundation (done)
- **Script:** `e0_fetch_long_history.py`, local, about 1 minute.
- **What it fetches:** 24 Yahoo series (indexes, VIX, yields, dollar, oil, HYG/IEF, sector SPDRs,
  RSP). Six FRED series are added when `FRED_API_KEY` is set. The key is free, and FRED's keyless
  CSV endpoint is refused from this network.
- **Rules:** the session in progress is dropped; raw downloads are immutable (one dated folder per
  run); FRED releases carry an explicit availability lag.
- **Fitness checks (run 2026-11-24 15:17 ET):** all passed.
  - Crash-day returns match: 1987-10-19 −20.5%, 2008-10-15 −9.0%, 2020-03-16 −12.0%.
  - Nasdaq-100 vs Alpaca QQQ daily returns correlate 0.9987 (2,736 days).
  - Yield and VIX ranges are sane.
- **Output:** `Data/research/regime_40y/panel_20261124T1517.parquet` (1927-12-30..2026-11-23).

### E1 Does a regime state describe different futures? (40 years, local)
- **Question:** do the vote's states have different forward *distributions*, mainly drawdown and
  volatility, consistently across decades?
- **Data:** Nasdaq-100 1986–2026 (primary, the AI/tech tape) and S&P 500 1962–2026 (secondary).
- **States:** a long-history vote from the F1/F2/F4 inputs that exist for the whole span.
  - Votes: fast trend, below SMA50, EMA20<EMA50, RV20 z>1, VIX z>1 (1990+), curve 10y–3m inverted.
  - With FRED: Baa spread z>0 (1986+).
  - Its agreement with the live 7-vote on 2021+ is reported. Thresholds are the round values from
    the live vote and are not tuned.
- **Targets** (all from the close of t):
  - worst drawdown over the next 20 sessions (primary);
  - realised vol over the next 20 sessions;
  - 20-session return (reported; direction is the hardest target).
- **Method:** state-conditional distributions with a circular 20-session block bootstrap, run
  separately for each decade.
- **Control:** a circular-shift placebo of the state series (1,000 draws).
- **Decision:** a state definition is *useful* if forward drawdown or vol differ between states in
  at least 4 of 5 decades beyond the placebo's 95th percentile. Direction is not required.

### E2 A general regime policy on 40 years (local)
- **Question:** which index-level policy survives every crash in the sample? The crashes are 1987,
  1990, 2000–02, 2007–09, 2011, 2015–16, 2018, 2020, 2022, 2025 and 2026.
- **Books** on the index, deciding at the close and trading the next open, 2 bp per unit turnover:
  - buy-and-hold;
  - long/flat when the vote is 4 or more;
  - size by regime with the live multipliers (1.0 / 0.8 / 0.5 / 0.25);
  - 15% volatility target;
  - the classic SMA200 rule.
- **Reported:** CAGR, Sharpe, MDD, time invested, a per-crash table and per-decade results.
- **Control:** a circular-shift placebo of each state series.
- **Decision:** adopt as the default general policy if MDD improves in at least 3 of every 4
  crashes, Sharpe is not below buy-and-hold beyond its confidence interval, and the placebo
  percentile on MDD is above 90%.

### E3 Are we missing a factor? Factor completeness (40 years where coverage allows, local)
- **Question:** does any factor family add out-of-sample information about the primary target (a
  20-session drawdown of 5% or worse) beyond price and volatility alone? Is any family the one we
  are missing?
- **Method:**
  - Walk forward on an expanding window with a yearly refit, Nasdaq-100 1990–2026 (training from
    1986).
  - Model: logistic regression with the scaler fit on training data only; a shallow LightGBM as a
    secondary check.
  - Arms: a baseline of F1+F2 realised; the baseline plus each family alone; all families; and
    leave-one-family-out.
  - Families with short history are run on their own span only, and the span is stated.
- **Control:** for each family, its features block-permuted in 60-session blocks (100 draws). This
  gives the null for Δ log-loss and doubles as the MDE.
- **Episode audit:** for every drawdown of 10% or more, which families moved *before* the price vote
  flagged, and where the baseline's error was largest.
- **Decision:** a family is *crucial and missing* if it improves out-of-sample log-loss beyond the
  permuted control's 95th percentile in at least 2 of the last 3 decades. A family that helps only
  in specific episodes is labelled *context* (it explains, but does not forecast).

### E4 Context factors with short history (2024–2026, local, descriptive plus forward collection)
- **"Good news sold" (F9):**
  - Measure: each day, the mean 1- and 2-day return relative to SPY of tickers with a positive
    earnings surprise or a positive catalyst (FinBERT).
  - A negative value means buyers are exhausted.
  - Test the user's case: did AI-name beats get sold before the index vote flagged on 10-16? Repeat
    for 2024-07 and 2025-02 if the data reach.
- **Other short-history families:** news-tone balance, dealer gamma sign on SPY/QQQ (85 snapshots),
  and CBOE put/call.
- **Decision:** with this few episodes nothing is certified. A factor that led the vote in at least
  2 of 3 episodes enters the forward shadow (E8) as a logged candidate.

### E5 How fine-grained must the intraday regime be? (local)
- **Why intraday at all:**
  - scheduled events (FOMC 14:00 and the 14:30 press conference; CPI and NFP at 08:30, so they are
    known at the open);
  - unscheduled news (tariffs, geopolitics);
  - trend-day vs range-day identity, which is often visible by 10:30;
  - volatility expansion and breadth thrusts.
  
  None of these needs a minute-by-minute *macro* regime. They need a session state that can flip on
  evidence.
- **Data:** SPY/QQQ 1-minute SIP bars 2016–2026 (raw); VIXY 1-minute for intraday vol; FOMC dates
  2016–2021 added from the Fed's published schedule.
- **Targets** from checkpoint t (10:00, 10:30, 11:00, 12:00, 14:30): rest-of-day realised range and
  rest-of-day return sign.
- **Arms:**
  - (a) the prior-close daily regime only;
  - (b) plus a session state at t: return since open, side of VWAP, opening-range break,
    range-so-far / ATR, VIXY change;
  - (c) the session state refreshed every 30 minutes;
  - (d) every 5 minutes;
  - (e) event-triggered refresh (FOMC/CPI days).
- **Decision:** use the coarsest refresh that captures at least 80% of the information gain of
  (d). Add event-triggered refreshes only if FOMC/CPI-day flips predict the rest of the day beyond
  what the schedule alone gives.

### E6 Model vs policy: does the model itself need the regime? (Colab for training)
- **For each ML module** (Momentum, SPY daytrader, Meta), with the label fixed and the existing
  walk-forward folds, **validation folds only** (the Stage 2/3 test set stays sealed until about
  2027-06):
  - (A) base features only;
  - (B) base plus the vote inputs;
  - (C) separate models per regime state.
  
  Momentum, Meta and the 30m swing already carry the regime-table columns, so their arm (A) is the
  ablation that removes them.
- **Scoring:** within each regime (IC or AUC, plus calibration), against a block-shuffled regime
  control.
- **Decision:** regime stays out of the model unless (B) or (C) beats (A) within regimes by more
  than the shuffled control in the validation folds. Expected answer: (C) is starved, with about
  five bear spells in training.

### E7 Policy replay: does a regime-aware order policy beat the current one? (local)
- **The same signals, run through two policies:** the current one and a regime-aware one. The
  rules are fixed before replay:
  - **Intraday structure:** extend `research/regime_switch_2026-11-20/08` to the full regime state.
    In RISK_OFF or CRISIS, shorts hold to their structural target or EOD and longs exit on the
    current fast rule. Otherwise the current policy applies.
  - **SPY daytrader:** its built-in regime thresholds on its out-of-fold signal frame
    (`phase4_signal_frame.parquet`), on vs off.
  - **Momentum / HTF / Meta:** threshold and size by regime on validation backtests
    (`family_backtest.py`) and the decision-state archive.
- **Control:** the same rules on block-shuffled regime days. The regime must beat a random
  assignment of the same rules.
- **Decision:** a module adopts its regime profile in paper only if it beats both the current
  policy and the shuffled control on validation. That leads to E8, not straight to live.

### E8 Forward shadow (live paper, ongoing)
- **What it logs:** every live decision is recorded under the current policy and every candidate
  regime profile (a counterfactual ledger), plus the E4 candidate factors.
- **Read date:** set when E7 finishes. The reading is the later of 40 sessions or 10 sessions in
  RISK_OFF/CRISIS, with the MDE stated at the read.

## 3. Order, compute and what is needed from the user

| Step | Runs where | Rough time | Needs |
|---|---|---|---|
| E0 | local | 1 min | done; **a free FRED API key** adds credit before 2007 and macro |
| E1, E2 | local | minutes | E0 |
| E3 | local | ~30 min | E0 (+ FRED) |
| E5 | local | ~1 h (1-minute fetch + replay) | Alpaca SIP |
| E7 | local | ~1 h | E5 for intraday; existing OOF frames |
| E6 | Colab | per module | E7 result first, so training is spent only where policy alone falls short |
| E4, E8 | local, ongoing | – | decision archive, shadow ledger |

Each script is written as it is run, with its pre-registered spec in the docstring, and results go
in this folder's README.
