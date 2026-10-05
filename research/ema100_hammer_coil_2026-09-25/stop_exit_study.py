"""Same setups as event_study.py, but traded the way a discretionary trader would:
entry at open t+1, hard stop just under the signal-day low, and either a fixed time exit
or a trailing exit, instead of a blind fixed-horizon hold.

Stop fill: if a bar opens below the stop, fill at the open (gap-through); else at the stop.
Exits:
  time15  : stop, else close of day t+15
  trail20 : stop, else first close below EMA20 once price has closed back above EMA20
            (the trail is armed only after that, since entries sit below EMA20), max 40 sessions
R = return / initial risk ((entry - stop) / entry). Raw returns, no SPY adjustment, no
costs. Compare every arm against arm C under identical rules (the control).

  PYTHONPATH=. .venv/bin/python research/ema100_hammer_coil_2026-09-25/stop_exit_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from event_study import BARS, OUT, TOP_N, boot_ci, features, load  # noqa: E402

STOP_BUFFER = 0.002
MAX_HOLD = 40
N_SEED = 11


def simulate(b: pd.DataFrame, f: pd.DataFrame, i: int, mode: str) -> tuple[float, float, int] | None:
    o, h, l, c = (b[k].values for k in ("open", "high", "low", "close"))
    e20 = f["e20"].values
    if i + 1 >= len(b):
        return None
    entry = o[i + 1]
    stop = l[i] * (1 - STOP_BUFFER)
    if not (entry > stop > 0):
        return None
    risk = (entry - stop) / entry
    last = min(len(b) - 1, i + (15 if mode == "time15" else MAX_HOLD))
    armed = False
    for j in range(i + 1, last + 1):
        if j > i + 1 and o[j] <= stop:
            return o[j] / entry - 1, risk, j - i
        if l[j] <= stop:
            return stop / entry - 1, risk, j - i
        if mode == "trail20" and armed and c[j] < e20[j]:
            return c[j] / entry - 1, risk, j - i
        armed = armed or c[j] > e20[j]
    if last < i + (15 if mode == "time15" else MAX_HOLD):
        return None  # window not complete
    return c[last] / entry - 1, risk, last - i


def main() -> None:
    frames = []
    for p in sorted(BARS.glob("*.parquet")):
        b = load(p.stem)
        if b is None or len(b) < 260:
            continue
        f = features(b)
        f["leader"] = b["close"] / b["close"].shift(126) - 1 > 0.40
        f["ticker"] = p.stem
        f["i"] = np.arange(len(f))
        frames.append((p.stem, b, f))
    panel = pd.concat([f.dropna(subset=["e200", "dv60"]) for _, _, f in frames]).reset_index()
    panel["dv_rank"] = panel.groupby("date")["dv60"].rank(ascending=False)
    panel = panel[panel.dv_rank <= TOP_N]
    C = panel.uptrend & panel.touch100
    arms = {
        "C uptrend+touch100": C,
        "D C+hammer": C & panel.hammer,
        "F C+hammer+coil": C & panel.hammer & panel.coil,
        "G C+leader": C & panel.leader,
        "H F+leader": C & panel.hammer & panel.coil & panel.leader,
    }
    bars = {t: (b, f) for t, b, f in frames}
    rng = np.random.default_rng(N_SEED)
    rows, trades = [], []
    for name, m in arms.items():
        ev = panel.loc[m, ["date", "ticker", "i"]]
        for mode in ("time15", "trail20"):
            rec = []
            for d, t, i in ev.itertuples(index=False):
                b, f = bars[t]
                r = simulate(b, f, int(i), mode)
                if r is None or max(abs(b["close"].pct_change().iloc[int(i) + 1:int(i) + 1 + r[2]]).max(), 0) > 0.35:
                    continue
                rec.append((d, t, *r))
            x = pd.DataFrame(rec, columns=["date", "ticker", "ret", "risk", "days"])
            x["R"] = x.ret / x.risk
            lo, hi = boot_ci(x.ret, x.date.dt.to_period("M"), rng)
            rows.append({"arm": name, "exit": mode, "n": len(x), "mean_ret_%": 100 * x.ret.mean(),
                         "ci_lo_%": 100 * lo, "ci_hi_%": 100 * hi, "win_%": 100 * (x.ret > 0).mean(),
                         "mean_R": x.R.mean(), "median_R": x.R.median(), "avg_risk_%": 100 * x.risk.mean(),
                         "avg_days": x.days.mean(), "top5pct_share_of_pnl": x.ret.nlargest(max(1, len(x) // 20)).sum() / x.ret.sum() if x.ret.sum() > 0 else np.nan})
            x.assign(arm=name, exit=mode).pipe(trades.append)
    res = pd.DataFrame(rows).round(2)
    res.to_csv(OUT / "stop_exit_results.csv", index=False)
    pd.concat(trades).to_csv(OUT / "stop_exit_trades.csv", index=False)
    print(res.to_string(index=False))
    t = pd.concat(trades)
    print("\nAMD trades:")
    print(t[t.ticker == "AMD"].tail(12).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
