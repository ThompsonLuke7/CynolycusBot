"""Earnings beat/miss x price reaction -> drift over the following quarter (2026-09-29).

The user's question: "if the report was good but the stock tanked because it was priced in, does it
go up later?" The forward_guidance module tried to answer this on 550 mega-cap events. Its test
AUC was 0.51 on 83 events, and its "guidance" text was SEC cover pages. This study uses every
reported quarter in the liquid-1000 instead.

Event     a reported quarter in ticker_earnings_calendar, date d. The pre-open/after-close
          time is not recorded.
Reaction  EAR = close(d+1) / close(d-1) - 1 minus SPY over the same window. This covers both
          a pre-open report and an after-close report.
Entry     the open of d+2, when the reaction is fully known. Exit at the close of session
          d+1+H, for H = 21 / 63 / 126. Excess = stock minus SPY over the identical window.
Groups    surprise: beat > +2%, miss < -2%, inline. Reaction: up > +2%, down < -2%, flat.
Adjusted  each event's excess minus the mean excess of ALL events in the same calendar month.
          This strips the market/size drift that every stock shares (2019-26 equal-weight lagged SPY).
          CIs bootstrap over event months.
Caveats   eps_estimate is Yahoo's stored consensus and may not be point-in-time, so
          beat/miss can be misclassified near zero; the +-2% band limits that. The universe is
          today's tickers. There are ~9 groups x 3 horizons, so read single cells at |t| > 3.

    .venv/bin/python research/long_horizon_discount_2026-09-25/07_earnings_reaction.py
"""
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = spec_from_file_location("rb", HERE / "05_rotation_backtest.py")
rb = module_from_spec(_spec)
_spec.loader.exec_module(rb)

CAL = rb.pb.CAL
HOLDS = (21, 63, 126)
BAND = 2.0          # percent, for both surprise and reaction
SEED = 11


def month_ci(adj: pd.Series, months: pd.Series, n=2000) -> tuple[float, float]:
    g = pd.DataFrame({"a": adj.values, "m": months.values}).groupby("m")["a"].agg(["sum", "count"])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(g), (n, len(g)))
    boots = g["sum"].values[idx].sum(1) / g["count"].values[idx].sum(1)
    return tuple(np.percentile(boots, [2.5, 97.5]))


def build_events() -> pd.DataFrame:
    p = pd.read_parquet(rb.pb.OUT, columns=["date", "ticker", "dv_rank"])
    O, C = rb.price_matrices(sorted(set(p["ticker"]) | {"SPY"}))
    cal = C.index
    ev = pd.read_parquet(CAL)
    ev = ev[ev["reported_eps"].notna() & ev["surprise_pct"].notna()].copy()
    ev["date"] = pd.to_datetime(ev["date"]).dt.tz_localize(None).dt.normalize()
    ev = ev[(ev["date"] >= "2017-01-01") & ev["ticker"].isin(C.columns)]
    ev = ev.drop_duplicates(["ticker", "date"]).sort_values("date")
    # liquid-1000 membership from the most recent weekly panel date (within 7 days)
    ev = pd.merge_asof(ev, p.sort_values("date"), on="date", by="ticker", direction="backward",
                       tolerance=pd.Timedelta(days=7)).dropna(subset=["dv_rank"])
    i0 = cal.searchsorted(ev["date"].values, side="left")      # report session (next session if d is not one)
    ok = (i0 >= 1) & (i0 + 2 < len(cal))
    ev, i0 = ev[ok].copy(), i0[ok]
    col = {t: j for j, t in enumerate(C.columns)}
    j = ev["ticker"].map(col).values
    s = col["SPY"]
    Cv, Ov = C.values, O.values
    ev["ear"] = (Cv[i0 + 1, j] / Cv[i0 - 1, j] - Cv[i0 + 1, s] / Cv[i0 - 1, s]) * 100
    ent = i0 + 2
    for h in HOLDS:
        ex = ent + h - 1
        valid = ex < len(cal)
        exi = np.where(valid, ex, 0)
        r = Cv[exi, j] / Ov[ent, j] - 1
        rs = Cv[exi, s] / Ov[ent, s] - 1
        ev[f"xs_{h}"] = np.where(valid, r - rs, np.nan)
    ev["surprise"] = ev["surprise_pct"].clip(-200, 200)
    ev["s_grp"] = np.select([ev.surprise > BAND, ev.surprise < -BAND], ["beat", "miss"], "inline")
    ev["r_grp"] = np.select([ev.ear > BAND, ev.ear < -BAND], ["up", "down"], "flat")
    ev["month"] = ev["date"].dt.to_period("M")
    ev["entry_date"] = cal[ent]
    ev["known_date"] = cal[ent - 1]     # the reaction window closes at this session's close
    return ev.dropna(subset=["ear"])


def group_table(ev: pd.DataFrame, keys: list[str], h: int) -> pd.DataFrame:
    x = ev.dropna(subset=[f"xs_{h}"]).copy()
    x["adj"] = x[f"xs_{h}"] - x.groupby("month")[f"xs_{h}"].transform("mean")
    rows = []
    for k, g in x.groupby(keys):
        lo, hi = month_ci(g["adj"], g["month"])
        se = (hi - lo) / 3.92
        rows.append({**dict(zip(keys, k if isinstance(k, tuple) else (k,))), "n": len(g),
                     "raw_xs_spy": g[f"xs_{h}"].mean(), "adj": g["adj"].mean(), "lo": lo, "hi": hi,
                     "t": g["adj"].mean() / se if se > 0 else np.nan,
                     "hit_vs_month": (g["adj"] > 0).mean()})
    return pd.DataFrame(rows)


def fmt(df: pd.DataFrame) -> str:
    out = df.copy()
    for c in ("raw_xs_spy", "adj", "lo", "hi"):
        out[c] = out[c].map(lambda v: f"{v:+.2%}")
    out["t"] = out["t"].map(lambda v: f"{v:+.1f}")
    out["hit_vs_month"] = out["hit_vs_month"].map(lambda v: f"{v:.0%}")
    return out.to_string(index=False)


def main() -> None:
    ev = build_events()
    lines = [f"=== earnings reaction drift | {len(ev):,} events, {ev.ticker.nunique()} tickers, "
             f"{ev.date.min().date()}..{ev.date.max().date()} | entry = open of report+2 sessions ==="]
    lines.append(f"groups: beat {(ev.s_grp == 'beat').mean():.0%} / inline {(ev.s_grp == 'inline').mean():.0%} "
                 f"/ miss {(ev.s_grp == 'miss').mean():.0%};  reaction up {(ev.r_grp == 'up').mean():.0%} "
                 f"/ flat {(ev.r_grp == 'flat').mean():.0%} / down {(ev.r_grp == 'down').mean():.0%}")
    lines.append("adj = excess vs SPY minus the same-month all-event mean; CI bootstraps event months")
    for h in HOLDS:
        lines.append(f"\n--- hold {h} sessions: surprise x reaction ---")
        lines.append(fmt(group_table(ev, ["s_grp", "r_grp"], h)))
    ev["ear_q"] = pd.qcut(ev["ear"], 5, labels=["Q1 worst", "Q2", "Q3", "Q4", "Q5 best"])
    for h in (63, 126):
        lines.append(f"\n--- hold {h}: reaction quintile (all events) ---")
        lines.append(fmt(group_table(ev, ["ear_q"], h)))
    for h in (63,):
        ev["yr"] = ev["date"].dt.year
        t = group_table(ev[ev.s_grp == "beat"], ["r_grp", "yr"], h)
        lines.append(f"\n--- hold {h}: BEAT events by reaction and year (stability) ---")
        lines.append(t.pivot(index="yr", columns="r_grp", values="adj").map(lambda v: f"{v:+.1%}").to_string())
    txt = "\n".join(lines)
    print(txt)
    (HERE / "07_results.txt").write_text(txt)


if __name__ == "__main__":
    main()
