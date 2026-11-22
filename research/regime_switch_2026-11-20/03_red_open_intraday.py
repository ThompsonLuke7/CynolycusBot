"""User's trade: "when the open is flashing red, short QQQ (or buy puts) and sell an hour later."

Pre-registered 2026-11-20 ~16:30 ET, before any result was computed.

Data   Alpaca SIP 5-minute bars (bar-start timestamps), QQQ and SPY, 2016-01-04..2026-11-20
       (Data/research/regime_switch_2026-11-20/bars_5m, fetched 2026-11-20 21:24 UTC).
       Regular session only (09:30 <= bar start < 16:00 ET); days with fewer than 78 bars
       (half days, gaps) are dropped. Underlying only: no option is priced.
Signal (known at 10:00 ET)
       opening drive OD = close of the 09:55 bar / open of the 09:30 bar - 1
       red open   OD <= -0.5%      green open   OD >= +0.5%   (mirror test)
Trade  enter at 10:00 (close of the 09:55 bar): SHORT on a red open, LONG on a green open.
       Primary exit 11:00 (close of the 10:55 bar). Secondary exits 12:00, 13:00, 16:00 (close).
       Bracket variant: +1.0% target / -0.5% stop / 11:00 time exit; when both levels sit inside one
       5-minute bar, the stop is assumed hit first (conservative).
Measured per trade: return, best favorable move inside the hour (perfect-exit ceiling), worst
       adverse move inside the hour, and whether the favorable move reached 0.5% / 1.0%.
Splits (all known before the open): QQQ D1 (close < SMA50) and D0 (fast) state at the prior close,
       from 01_regime_study.py; high-vol = prior-close QQQ 20d realised vol in the top third of its
       own trailing 252-session values; by year; the 2026-10-15..11-20 selloff (descriptive only).
Control: the same entry/exit on every day (drift), and red opens vs that.
Inference: one trade per day; t = mean / (sd / sqrt(n)), MDE ~= 2.8 x SE. No costs (QQQ spread ~1 bp).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BARS5 = REPO / "Data" / "research" / "regime_switch_2026-11-20" / "bars_5m"
OUT = HERE / "out"
TH = 0.005
EXITS = {"11:00": "10:55", "12:00": "11:55", "13:00": "12:55", "16:00": "15:55"}
TARGET, STOP = 0.010, 0.005

_spec = importlib.util.spec_from_file_location("rs01", HERE / "01_regime_study.py")
rs01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs01)


def sessions(ticker: str) -> dict[pd.Timestamp, pd.DataFrame]:
    d = pd.read_parquet(BARS5 / f"{ticker}.parquet")
    ts = pd.to_datetime(d["timestamp"]).dt.tz_convert("America/New_York")
    d = d.assign(ts=ts, day=ts.dt.normalize().dt.tz_localize(None), hm=ts.dt.strftime("%H:%M"))
    d = d[(d["hm"] >= "09:30") & (d["hm"] < "16:00")]
    if d.duplicated(["ts"]).any():
        raise ValueError(f"{ticker}: duplicate 5m bars")
    out = {}
    for day, g in d.groupby("day"):
        if len(g) == 78 and g["hm"].iloc[0] == "09:30" and g["hm"].iloc[-1] == "15:55":
            out[day] = g.set_index("hm")[["open", "high", "low", "close"]]
    return out


def trade(g: pd.DataFrame, side: int) -> dict:
    """side +1 long, -1 short; entry at 10:00 = close of the 09:55 bar."""
    e = g.at["09:55", "close"]
    rec = {"entry": e}
    for name, bar in EXITS.items():
        rec[f"ret_{name}"] = side * (g.at[bar, "close"] / e - 1)
    w = g.loc["10:00":"10:55"]
    if side < 0:
        fav, adv = e / w["low"].min() - 1, w["high"].max() / e - 1
    else:
        fav, adv = w["high"].max() / e - 1, 1 - w["low"].min() / e
    rec.update(fav_1h=fav, adv_1h=adv, hit05=fav >= 0.005, hit10=fav >= 0.010)
    # bracket: walk bars, stop first when both inside one bar
    br = None
    for _, b in w.iterrows():
        up, dn = b["high"] / e - 1, b["low"] / e - 1
        gain, loss = (up, -dn) if side > 0 else (-dn, up)
        if loss >= STOP:
            br = -STOP
            break
        if gain >= TARGET:
            br = TARGET
            break
    rec["ret_bracket"] = br if br is not None else rec["ret_11:00"]
    return rec


def summarize(df: pd.DataFrame, label: str) -> dict:
    r = df["ret_11:00"]
    n = len(r)
    se = r.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    row = {"group": label, "n": n, "mean_1h": r.mean(), "median_1h": r.median(), "win_1h": (r > 0).mean(),
           "t_1h": r.mean() / se if se else np.nan, "mde_1h": 2.8 * se,
           "hit05": df["hit05"].mean(), "hit10": df["hit10"].mean(),
           "med_fav": df["fav_1h"].median(), "med_adv": df["adv_1h"].median(),
           "mean_bracket": df["ret_bracket"].mean(), "win_bracket": (df["ret_bracket"] > 0).mean()}
    for name in ("12:00", "13:00", "16:00"):
        row[f"mean_{name}"] = df[f"ret_{name}"].mean()
    return row


def main() -> None:
    q = rs01.load("QQQ")
    st = rs01.states(q)
    rv = q["close"].pct_change().rolling(20).std() * np.sqrt(252)
    rv_hi = rv > rv.rolling(252, min_periods=252).quantile(2 / 3)
    prev = pd.DataFrame({"D1": st["D1_sma50"], "D0": st["D0_fast"], "rv_hi": rv_hi.where(rv.rolling(252).count() >= 252)})
    prev = prev.shift(1)  # state known at the PRIOR close
    # the 11-20 session has no daily bar in 01's data: its prior close is 11-19, which is the last row
    prev.loc[pd.Timestamp("2026-11-20")] = [st["D1_sma50"].iloc[-1], st["D0_fast"].iloc[-1],
                                            float(rv_hi.iloc[-1])]

    allrows = []
    for sym in ("QQQ", "SPY"):
        ss = sessions(sym)
        for day, g in ss.items():
            od = g.at["09:55", "close"] / g.at["09:30", "open"] - 1
            if day not in prev.index or pd.isna(prev.at[day, "D1"]):
                continue
            base = {"sym": sym, "day": day, "od": od, **prev.loc[day].to_dict()}
            allrows.append({**base, "kind": "short_all", **trade(g, -1)})
            if od <= -TH:
                allrows.append({**base, "kind": "short_red", **trade(g, -1)})
            if od >= TH:
                allrows.append({**base, "kind": "long_green", **trade(g, +1)})
    t = pd.DataFrame(allrows)
    t.to_csv(OUT / "red_open_trades.csv", index=False)

    rows = []
    for sym in ("QQQ", "SPY"):
        for kind in ("short_red", "long_green", "short_all"):
            k = t[(t.sym == sym) & (t.kind == kind)]
            rows.append({"sym": sym, "kind": kind, **summarize(k, "all")})
            for col in ("D1", "D0", "rv_hi"):
                for val, name in ((1.0, "DOWN" if col != "rv_hi" else "HIGHVOL"), (0.0, "UP" if col != "rv_hi" else "normal")):
                    sub = k[k[col] == val]
                    if len(sub) > 5:
                        rows.append({"sym": sym, "kind": kind, **summarize(sub, f"{col}={name}")})
            ep = k[(k.day >= "2026-10-15") & (k.day <= "2026-11-20")]
            if len(ep):
                rows.append({"sym": sym, "kind": kind, **summarize(ep, "selloff 10-15..11-20")})
    pd.DataFrame(rows).to_csv(OUT / "red_open_summary.csv", index=False)
    k = t[(t.sym == "QQQ") & (t.kind == "short_red")].copy()
    k["year"] = pd.to_datetime(k["day"]).dt.year
    k.groupby(["year", "D1"]).agg(n=("ret_11:00", "size"), mean_1h=("ret_11:00", "mean"),
                                  win=("ret_11:00", lambda s: (s > 0).mean())).to_csv(OUT / "red_open_by_year.csv")
    print(f"days QQQ={t[(t.sym=='QQQ') & (t.kind=='short_all')].shape[0]} red={len(k)}")


if __name__ == "__main__":
    main()
