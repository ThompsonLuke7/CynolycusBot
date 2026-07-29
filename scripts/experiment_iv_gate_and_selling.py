"""Two symmetric tests: (a) gate BUYING on option cheapness, (b) SELL premium instead.

(a) IV gate. Every result so far says the problem is that we buy EXPENSIVE convexity:
    the better the signal, the pricier the option. So gate entries on cheapness.
    Cheapness metric = iv_rv_premium = implied vol at entry / trailing realized vol.
    IV is backed out of the contract's own real market price (not modeled), realized
    vol is Yang-Zhang on the underlying's trailing 20 daily bars. <1 means the option
    is priced BELOW how much the stock has actually been moving.

(b) Premium selling. If buyers lose ~29% of premium, does the seller win? This is NOT
    just a sign flip:
      * the seller pays the same bid/ask spread,
      * "capital" for a seller is BUYING POWER (Reg-T ~20% of underlying notional),
        not the premium, so the same dollar edge sits on a much larger base,
      * the buyer's 360% winner is the seller's catastrophic loss. Tail risk is the
        whole question for short premium, so worst-case and CVaR are reported, and
        max loss is reported as unbounded where it genuinely is.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.options_lab.pricing import implied_vol, risk_free_rate
from research.options_lab.surface import yang_zhang_vol

REPO = Path(__file__).resolve().parents[1]
LR = REPO / "research/options_experiment/data/let_it_run.parquet"
P3 = REPO / "research/options_experiment/data/phase3_counterfactual.parquet"
UNDER = REPO / "Data/shared/bars/1d"
OUT = REPO / "research/options_experiment/data/iv_gate_selling.parquet"

ROUND_TRIP_SPREAD = 0.2556
COMMISSION = 0.65
REG_T_SHORT_CALL = 0.20  # 20% of underlying notional -- standard naked-short approximation


def parse_osi(sym: str):
    """OSI: ROOT + YYMMDD + C/P + strike*1000 (8 digits)."""
    body = sym[-15:]
    exp = pd.Timestamp(f"20{body[0:2]}-{body[2:4]}-{body[4:6]}", tz="UTC")
    right = body[6]
    strike = int(body[7:]) / 1000.0
    return exp, right, strike


def main() -> None:
    lr = pd.read_parquet(LR)
    lr = lr[lr.policy == "hold_to_expiry"].copy()
    p3 = pd.read_parquet(P3)
    meta = (p3[(p3.strategy == "long_call_atm") & (p3.sizing_mode == "matched_notional")]
            [["trade_id", "entry_ts", "entry_px_underlying", "expiry_date", "atr_at_entry"]]
            .drop_duplicates("trade_id"))
    d = lr.merge(meta, on="trade_id", how="inner")
    d["entry_ts"] = pd.to_datetime(d.entry_ts, utc=True)
    print(f"trades: {len(d)}")

    rows = []
    fails: dict[str, int] = {}
    ubcache: dict[str, pd.DataFrame] = {}
    for r in d.itertuples(index=False):
        try:
            if r.ticker not in ubcache:
                p = UNDER / f"{r.ticker}.parquet"
                if not p.exists():
                    ubcache[r.ticker] = pd.DataFrame()
                else:
                    b = pd.read_parquet(p)
                    if "timestamp" not in b.columns:
                        b = b.reset_index()
                    b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True)
                    ubcache[r.ticker] = b.sort_values("timestamp")
            ub = ubcache[r.ticker]
            if ub.empty:
                continue
            # the lookahead guard requires the frame to end at/before asof
            hist = ub[ub.timestamp <= r.entry_ts]
            if len(hist) < 30:
                continue
            rv = yang_zhang_vol(hist, asof=r.entry_ts, window=20)
            if rv is None or not np.isfinite(rv) or rv <= 0:
                continue
            exp = pd.Timestamp(r.expiry_date, tz="UTC")
            T = max((exp - r.entry_ts).days, 1) / 365.0
            S = float(r.entry_px_underlying)

            # CHEAPNESS without needing the strike.
            # The Phase 3 parquet does not carry the selected strike, and assuming
            # K=S produces nonsense IV whenever the matched contract was not truly
            # ATM (verified: FLNC priced 2.85 on a 5.785 spot at 24 DTE -- deep ITM,
            # not ATM). So instead of inverting to IV, compare the premium to the
            # move the stock actually makes over the contract's life:
            #     expected_move = S * realized_vol * sqrt(T)
            #     cheapness     = premium / expected_move
            # <1 means you are paying less than a typical move; >1 means you need an
            # unusually large move just to break even. Model-light and strike-free.
            exp_move = S * rv * np.sqrt(T)
            if not np.isfinite(exp_move) or exp_move <= 0:
                continue
            cheapness = float(r.entry_px) / exp_move
            iv = np.nan  # not identifiable without the strike; deliberately not guessed

            # ---- buyer side (already computed) ----
            buy_capital = float(r.capital)
            buy_pnl = float(r.net_pnl)

            # ---- seller side: same contract, opposite sign ----
            gross_seller = -float(r.gross_pnl)
            # seller pays the spread too
            spread_cost = (float(r.entry_px) + float(r.exit_px)) * 100.0 * (ROUND_TRIP_SPREAD / 2.0)
            sell_pnl = gross_seller - spread_cost - 2 * COMMISSION
            sell_bp = REG_T_SHORT_CALL * S * 100.0   # naked short call margin approximation

            rows.append(dict(
                trade_id=r.trade_id, ticker=r.ticker, week_key=r.week_key, score=r.score,
                iv=iv, rv=rv, cheapness=cheapness, exp_move=exp_move, dte=int(T * 365),
                buy_pnl=buy_pnl, buy_capital=buy_capital,
                sell_pnl=sell_pnl, sell_bp=sell_bp,
            ))
        except Exception as e:
            fails[type(e).__name__ + ": " + str(e)[:60]] = fails.get(
                type(e).__name__ + ": " + str(e)[:60], 0) + 1
            continue

    if fails:
        print("skipped rows by reason:", dict(sorted(fails.items(), key=lambda x: -x[1])[:5]))
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"priced {len(out)} trades -> {OUT}")
    if out.empty:
        return

    print(f"\ncheapness (premium / expected move): median {out.cheapness.median():.2f} "
          f"(>1 = needs a bigger-than-typical move just to break even)")

    print("\n=== (a) BUYING calls, bucketed by option CHEAPNESS (iv/rv) ===")
    out["bucket"] = pd.qcut(out.cheapness, 5,
                            labels=["Q1 cheapest", "Q2", "Q3", "Q4", "Q5 priciest"], duplicates="drop")
    g = out.groupby("bucket", observed=True).apply(lambda x: pd.Series({
        "n": len(x), "median_cheapness": x.cheapness.median(),
        "buy_roc_%": 100 * x.buy_pnl.sum() / x.buy_capital.sum(),
        "win_rate": (x.buy_pnl > 0).mean(),
    }), include_groups=False)
    print(g.round(2).to_string())

    print("\n=== (b) SELLING those same calls (naked short, Reg-T BP) ===")
    g2 = out.groupby("bucket", observed=True).apply(lambda x: pd.Series({
        "n": len(x),
        "sell_pnl_total": x.sell_pnl.sum(),
        "sell_return_on_BP_%": 100 * x.sell_pnl.sum() / x.sell_bp.sum(),
        "win_rate": (x.sell_pnl > 0).mean(),
        "worst_trade": x.sell_pnl.min(),
        "cvar5_%ofBP": 100 * x.sell_pnl.nsmallest(max(int(len(x) * .05), 1)).mean() / x.sell_bp.mean(),
    }), include_groups=False)
    print(g2.round(2).to_string())

    print("\n=== seller tail risk, all trades ===")
    w = out.sell_pnl
    print(f"  win rate {(w > 0).mean():.1%} | median +${w.median():,.0f} | "
          f"mean ${w.mean():,.0f}")
    print(f"  WORST single trade ${w.min():,.0f} vs median win ${w[w > 0].median():,.0f} "
          f"-> one worst loss wipes out {abs(w.min()) / max(w[w > 0].median(), 1):,.0f} median wins")
    print(f"  total ${w.sum():,.0f} on ${out.sell_bp.sum():,.0f} BP = "
          f"{100 * w.sum() / out.sell_bp.sum():+.2f}% return on buying power")
    print("  NOTE: naked short calls have UNBOUNDED loss; Reg-T BP is an approximation and a "
          "real broker would raise margin as the position moved against you.")


if __name__ == "__main__":
    main()
