"""Variant of 03, added AFTER seeing 03's results (prompted by the user's wording "the open is flashing red",
which may mean a gap down rather than a red first half hour). Treat as exploratory, not pre-registered.

Gap = 09:30 open / prior session's 15:55-bar close - 1 <= -0.5%. Short at 09:35 (close of the 09:30 bar),
exit 10:35 (close of the 10:30 bar) = "an hour later"; also 16:00. Same data, splits and stats as 03.
"""
import importlib.util
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
def _load(name, file):
    s = importlib.util.spec_from_file_location(name, HERE / file); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
m03 = _load("m03", "03_red_open_intraday.py"); rs01 = m03.rs01
q = rs01.load("QQQ"); st = rs01.states(q)
prev = st[["D1_sma50"]].shift(1); prev.loc[pd.Timestamp("2026-11-20")] = [st["D1_sma50"].iloc[-1]]
rows = []
for sym in ("QQQ", "SPY"):
    ss = m03.sessions(sym); days = sorted(ss)
    for a, b in zip(days[:-1], days[1:]):
        g, pc = ss[b], ss[a].at["15:55", "close"]
        gap = g.at["09:30", "open"] / pc - 1
        if gap > -0.005 or b not in prev.index or pd.isna(prev.at[b, "D1_sma50"]):
            continue
        e = g.at["09:30", "close"]; w = g.loc["09:35":"10:30"]
        rows.append({"sym": sym, "day": b, "gap": gap, "D1": prev.at[b, "D1_sma50"],
                     "ret_1h": -(g.at["10:30", "close"] / e - 1), "ret_close": -(g.at["15:55", "close"] / e - 1),
                     "fav": e / w["low"].min() - 1, "adv": w["high"].max() / e - 1})
t = pd.DataFrame(rows); t.to_csv(HERE / "out" / "gap_down_trades.csv", index=False)
def s(x, lab):
    r = x.ret_1h; se = r.std(ddof=1) / np.sqrt(len(r))
    return {"group": lab, "n": len(r), "mean_1h%": 100 * r.mean(), "win_1h": (r > 0).mean(), "t": r.mean() / se,
            "mean_close%": 100 * x.ret_close.mean(), "fav>=0.5%": (x.fav >= .005).mean(), "fav>=1%": (x.fav >= .01).mean(),
            "adv>=0.5%": (x.adv >= .005).mean()}
out = []
for sym in ("QQQ", "SPY"):
    k = t[t.sym == sym]
    out += [{"sym": sym, **s(k, "all")}, {"sym": sym, **s(k[k.D1 == 1], "D1 DOWN")}, {"sym": sym, **s(k[k.D1 == 0], "D1 UP")},
            {"sym": sym, **s(k[k.day >= "2026-10-15"], "selloff 10-15..11-20")}]
o = pd.DataFrame(out); o.to_csv(HERE / "out" / "gap_down_summary.csv", index=False)
pd.set_option("display.width", 200); print(o.round(3).to_string(index=False))
print(t[(t.sym == "QQQ") & (t.day >= "2026-10-15")].round(4).to_string(index=False))
