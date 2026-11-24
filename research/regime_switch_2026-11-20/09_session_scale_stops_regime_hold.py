"""Session-scale stops and a regime-aware hold for intraday structure: same entries, different exits.

The questions (user, 2026-11-23):
  1. Stops and exits scaled to session-level volatility and levels instead of one-minute noise.
  2. A regime-aware hold: let shorts run in a bearish state, cut longs fast.

Pre-registered 2026-11-23 about 22:25 ET, before any of these exits was computed. 08 (same folder) already
showed: initial stops (scaled by a 14 x 1-minute ATR) are hit 77% of the time; on 7-vote BEAR days holding
shorts to the close beat the engine's exit (+0.27% vs +0.01%, t 1.1) and holding longs cost the most.

Exits (gross, on the underlying, from the ledger entry price; flat at 15:55 ET):
  ACTUAL        the engine's own exit (ledger gross_return).
  SESS_k        stop at entry -/+ k x ATR15, held to 15:55 otherwise; k in {1.0, 1.5, 2.0}, k=1.5 primary.
                ATR15 = mean true range of the session's COMPLETED 15-minute bars before the entry minute
                (built from the module's own 1-minute archive, 09:30 ET onward, at least 2 bars).
  LEVEL         stop just beyond the session extreme set before entry (low for longs, high for shorts)
                plus a 0.25 x ATR15 buffer, but never wider than 3 x ATR15; held to 15:55 otherwise.
  REGIME        BEAR day and short -> SESS_1.5; every other setup -> ACTUAL. This is the user's rule.
A stop fills at the stop, or at the bar's open when it opened through it. Entries before ATR15 exists
(about 10:00 ET) or after 15:25 ET are excluded from the non-ACTUAL exits and counted.

Samples
  IN-SAMPLE      sessions through 2026-11-20. REGIME was suggested by 08 on these same setups, so its
                 in-sample number is descriptive by construction, not evidence.
  OUT-OF-SAMPLE  sessions from 2026-11-23 on, appended as the module trades. The pre-registered test is
                 REGIME minus ACTUAL on these sessions, session-clustered, one-sided at p < 0.05, plus
                 REGIME net of 12 bp > 0.
Reported per direction x BEAR cell: n, sessions, mean/median gross, win rate, stop rate, net of 12 bp,
session-clustered t and MDE (~2.8 x SE). Limits: as 08 (one selloff, overlapping setups counted
separately, IEX bars sparse on thin names, which shrinks ATR15 there).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OOS_START = pd.Timestamp("2026-11-23")
KS = (1.0, 1.5, 2.0)
LEVEL_BUFFER, LEVEL_CAP = 0.25, 3.0

_spec = importlib.util.spec_from_file_location("cf08", HERE / "08_intraday_structure_exit_counterfactual.py")
cf08 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cf08)
ET, COST = cf08.ET, cf08.COST


def regime_votes(sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """05's 7-vote state; where the two QQQ-daily votes are unavailable (01's QQQ series ends
    2026-11-19), BEAR is decided from the known votes when they settle it, else left NaN."""
    rs05 = cf08.rs05
    try:
        return rs05.regime_votes(sessions)
    except ValueError:
        pass
    full = rs05.VOTES
    rs05.VOTES = tuple(x for x in full if not x.startswith("q_"))
    try:
        v = rs05.regime_votes(sessions)
    finally:
        rs05.VOTES = full
    q = v[["q_d0", "q_d1"]]
    lo = v["votes"] + q.fillna(0).sum(axis=1)
    hi = lo + q.isna().sum(axis=1)
    v["votes"] = lo
    v["BEAR"] = np.where(lo >= rs05.BEAR_MIN, 1.0, np.where(hi < rs05.BEAR_MIN, 0.0, np.nan))
    return v


def atr15_before(b: pd.DataFrame, entry_ts: pd.Timestamp) -> float | None:
    """Mean true range of completed 15-minute bars from 09:30 ET up to the entry minute."""
    day_open = entry_ts.tz_convert(ET).normalize() + pd.Timedelta(hours=9, minutes=30)
    pre = b[(b.index >= day_open.tz_convert("UTC")) & (b.index < entry_ts)]
    if pre.empty:
        return None
    q = pre.tz_convert(ET).resample("15min", origin=day_open, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    q = q[q.index + pd.Timedelta(minutes=15) <= entry_ts.tz_convert(ET)]
    if len(q) < 2:
        return None
    prev = q["close"].shift()
    tr = pd.concat([q["high"] - q["low"], (q["high"] - prev).abs(), (q["low"] - prev).abs()], axis=1).max(axis=1)
    return float(tr.iloc[1:].mean()) if len(tr) > 2 else float((q["high"] - q["low"]).mean())


def hold_with_stop(path: pd.DataFrame, side: float, stop: float, eod_close: float | None) -> tuple[float | None, bool]:
    for _, r in path.iterrows():
        if side > 0 and r["low"] <= stop:
            return min(stop, float(r["open"])), True
        if side < 0 and r["high"] >= stop:
            return max(stop, float(r["open"])), True
    return eod_close, False


def counterfactuals(s: pd.Series, b: pd.DataFrame) -> dict:
    side = 1.0 if s["direction"] == "long" else -1.0
    e_ts, e_px = s["entry_ts"], float(s["entry_price"])
    if e_ts.tz_convert(ET).strftime("%H:%M") > cf08.LAST_ENTRY_HM:
        return {"excluded": "late_entry"}
    atr = atr15_before(b, e_ts)
    if not atr or atr <= 0:
        return {"excluded": "no_atr15"}
    day = e_ts.tz_convert(ET).normalize()
    eod_ts = (day + pd.Timedelta(hours=15, minutes=55)).tz_convert("UTC")
    x = cf08.bar_near(b, eod_ts)
    if x is None:
        return {"excluded": "no_eod_bar"}
    eod = float(x["close"])
    path = b[(b.index > e_ts) & (b.index <= eod_ts)]
    # Reference only (08's EOD, no stop), to reconcile with 08 on this subset; not a candidate policy.
    rec: dict = {"atr15_pct": atr / e_px, "ret_EOD": side * (eod / e_px - 1)}
    for k in KS:
        px, hit = hold_with_stop(path, side, e_px - side * k * atr, eod)
        rec[f"ret_SESS_{k}"], rec[f"stop_SESS_{k}"] = side * (px / e_px - 1), hit
    day_open = (day + pd.Timedelta(hours=9, minutes=30)).tz_convert("UTC")
    pre = b[(b.index >= day_open) & (b.index <= e_ts)]
    extreme = float(pre["low"].min()) if side > 0 else float(pre["high"].max())
    level_stop = extreme - side * LEVEL_BUFFER * atr
    widest = e_px - side * LEVEL_CAP * atr
    level_stop = max(level_stop, widest) if side > 0 else min(level_stop, widest)
    if (side > 0 and level_stop >= e_px) or (side < 0 and level_stop <= e_px):
        level_stop = e_px - side * LEVEL_BUFFER * atr  # entry sits beyond the extreme: buffer only
    px, hit = hold_with_stop(path, side, level_stop, eod)
    rec["ret_LEVEL"], rec["stop_LEVEL"] = side * (px / e_px - 1), hit
    rec["level_risk_atr"] = abs(e_px - level_stop) / atr
    return rec


def summarize(t: pd.DataFrame, label: str) -> pd.DataFrame:
    exits = ["ACTUAL", "EOD", *(f"SESS_{k}" for k in KS), "LEVEL", "REGIME"]
    out = []
    for d in ("long", "short", "all"):
        for name, m in (("all", t["BEAR"].notna()), ("BEAR", t["BEAR"] == 1), ("not BEAR", t["BEAR"] == 0)):
            sub = t[m & ((t["direction"] == d) if d != "all" else True)]
            both = sub.dropna(subset=[f"ret_{e}" for e in exits])
            for e in exits:
                row = {"sample": label, "direction": d, "cell": name, "exit": e, **cf08.cell(both, f"ret_{e}")}
                stop_col = f"stop_{e}"
                if stop_col in both:
                    row["stop_rate"] = float(both[stop_col].mean()) if len(both) else np.nan
                out.append(row)
        # the pre-registered statistic: REGIME minus ACTUAL on the same setups, session-clustered
        sub = t[(t["direction"] == d) if d != "all" else t["direction"].notna()].dropna(subset=["ret_REGIME", "ret_ACTUAL"])
        diff = (sub["ret_REGIME"] - sub["ret_ACTUAL"]).groupby(sub["session"]).mean()
        if len(diff) >= 3 and diff.std(ddof=1) > 0:
            se = diff.std(ddof=1) / np.sqrt(len(diff))
            out.append({"sample": label, "direction": d, "cell": "all", "exit": "REGIME-ACTUAL",
                        "n": len(sub), "sessions": len(diff), "gross_mean": diff.mean(),
                        "t_sessions": diff.mean() / se, "mde_sessions": 2.8 * se})
    return pd.DataFrame(out)


def main() -> None:
    a = pd.DataFrame([json.loads(x) for x in cf08.LEDGER.open()])
    a = a[a["entry_time"].notna()].copy()
    a["entry_ts"] = pd.to_datetime(a["entry_time"], utc=True)
    a["session"] = a["entry_ts"].dt.tz_convert(ET).dt.normalize().dt.tz_localize(None)
    v = regime_votes(pd.DatetimeIndex(sorted(a["session"].unique())))
    a = a.join(v[["votes", "BEAR"]], on="session")
    a["ret_ACTUAL"] = a["gross_return"]
    rows, skipped = [], 0
    for day, g in a.groupby("session"):
        p = cf08.ARCHIVE / f"bars_{day:%Y%m%d}.jsonl"
        if not p.exists():
            skipped += len(g)
            continue
        bars = cf08.load_day(p, set(g["ticker"]))
        for idx, s in g.iterrows():
            b = bars.get(s["ticker"])
            rows.append({"idx": idx, **(counterfactuals(s, b) if b is not None and len(b) else {"excluded": "no_bars"})})
    cf = pd.DataFrame(rows).set_index("idx")
    t = a.join(cf, how="inner")
    bear_short = (t["BEAR"] == 1) & (t["direction"] == "short")
    t["ret_REGIME"] = np.where(bear_short, t["ret_SESS_1.5"], t["ret_ACTUAL"])
    t["stop_REGIME"] = np.where(bear_short, t["stop_SESS_1.5"], np.nan)
    keep = [c for c in t.columns if c.startswith(("ret_", "stop_", "atr15", "level_risk")) or c in (
        "session", "ticker", "direction", "setup_type", "entry_time", "exit_reason", "BEAR", "votes", "excluded")]
    t[keep].to_csv(OUT / "exit_policy_09_trades.csv", index=False)
    res = pd.concat([summarize(t[t["session"] < OOS_START], "in-sample"),
                     summarize(t[t["session"] >= OOS_START], "out-of-sample")], ignore_index=True)
    res.to_csv(OUT / "exit_policy_09_summary.csv", index=False)
    print(f"setups={len(a)} archived={len(t)} no_archive={skipped} excluded={t['excluded'].value_counts().to_dict()}")
    print(f"ATR15 median {t['atr15_pct'].median():.4%} of price; LEVEL median risk {t['level_risk_atr'].median():.2f} ATR15")
    pd.set_option("display.width", 220)
    show = res[res["cell"].isin(["BEAR", "not BEAR", "all"])][
        ["sample", "direction", "cell", "exit", "n", "sessions", "gross_mean", "gross_median", "win",
         "stop_rate", "net_mean", "t_sessions", "mde_sessions"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
