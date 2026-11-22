"""Regime-switch study: does a simple trend state tell us when to stop buying dips and start selling rips?

Pre-registered 2026-11-20 ~16:15 ET, before any result was computed. Do not add detectors,
thresholds or horizons after seeing results; a follow-up goes in a new numbered script.

Data   Alpaca SIP daily bars, split+dividend adjusted, 2016-01-04..2026-11-19
       (Data/research/regime_switch_2026-11-20/bars_1d, fetched 2026-11-20 21:05 UTC;
       the 11-20 session is excluded because it was incomplete at fetch time).
Scope  Index/sector ETFs only: QQQ (primary), SPY, SMH. Says nothing about single names,
       about option P&L (no option is priced here) or about any live module.
Timing State uses closes <= t (known after the close of t). Trades enter at the open of t+1.
       An h-session hold exits at the close of t+h.

Detectors (state = DOWN when true), computed on the instrument itself:
  D0 fast       close < EMA20 and EMA20 < EMA20 five sessions earlier
  D1 sma50      close < SMA50   (the filter in the user's personal trade review)
  D2 ema20_50   EMA20 < EMA50   (spy_trend_state in Data/shared/market_regime)
  D3 sma200     close < SMA200  (slow textbook baseline)

Events
  dip  close-to-close return on day t <= -1.5%
  rip  close-to-close return on day t >= +1.0%  (the bounce a put buyer would sell into)

Outcomes from the t+1 open, h in {5, 10, 20}: return to the close of t+h, MAE (lowest low /
entry - 1) and MFE (highest high / entry - 1).

Tests
  T1 forward return on all days, DOWN vs UP.
  T2 dip buys, DOWN vs UP, and versus the same state's all-day mean.
  T3 rip shorts (sign-flipped return) in DOWN, with UP as the control.
  T4 detection on every QQQ close-to-close drawdown >= 10%: sessions from the peak to the
     first DOWN flag, share of the eventual drawdown already lost at the flag, and false
     flags (DOWN runs that start outside any >= 10% drawdown) per year.
  T5 underlying-only books on each instrument: long always / long-or-flat / long-or-short
     (1x), position decided at the close and applied from the next open, 2 bp per unit of
     turnover, marked daily, no interest on cash and no borrow fee. Placebo: the same state
     series circularly shifted by a random offset (>= 60 sessions), 1,000 draws. That keeps
     the share of DOWN days and the run lengths and destroys the timing.

Inference: forward windows overlap, so mean differences get a circular 20-session block
bootstrap (2,000 draws). MDE ~= 2.8 x bootstrap SE (80% power, two-sided 5%). Independent
bear episodes are few (about five), so T4 lists them one by one.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
BARS = REPO / "Data" / "research" / "regime_switch_2026-11-20" / "bars_1d"
OUT = Path(__file__).resolve().parent / "out"
INSTRUMENTS = ("QQQ", "SPY", "SMH")
DETECTORS = ("D0_fast", "D1_sma50", "D2_ema20_50", "D3_sma200")
HORIZONS = (5, 10, 20)
LAST_SESSION = pd.Timestamp("2026-11-19")
DIP, RIP = -0.015, 0.010
COST_BP = 2.0
BOOT_DRAWS, BLOCK = 2000, 20
PLACEBO_DRAWS, MIN_SHIFT = 1000, 60
RNG = np.random.default_rng(20261120)


def load(ticker: str) -> pd.DataFrame:
    d = pd.read_parquet(BARS / f"{ticker}.parquet")
    d["date"] = pd.to_datetime(d["timestamp"]).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    d = d.set_index("date")[["open", "high", "low", "close"]].sort_index()
    if d.index.duplicated().any():
        raise ValueError(f"{ticker}: duplicate session dates")
    d = d.loc[:LAST_SESSION]
    if d.index[-1] != LAST_SESSION:
        raise ValueError(f"{ticker}: last session {d.index[-1].date()} != {LAST_SESSION.date()}")
    if d[["open", "high", "low", "close"]].isna().any().any() or (d[["open", "close"]] <= 0).any().any():
        raise ValueError(f"{ticker}: missing or non-positive prices")
    return d


def states(d: pd.DataFrame) -> pd.DataFrame:
    c = d["close"]
    ema20, ema50 = c.ewm(span=20, adjust=False).mean(), c.ewm(span=50, adjust=False).mean()
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    s = pd.DataFrame(index=d.index)
    s["D0_fast"] = (c < ema20) & (ema20 < ema20.shift(5))
    s["D1_sma50"] = c < sma50
    s["D2_ema20_50"] = ema20 < ema50
    s["D3_sma200"] = c < sma200
    # warm-up: EMAs need ~3x span to forget the seed; SMA200 needs 200 sessions
    s = s.astype(float)
    s.iloc[:200] = np.nan
    return s


def forward(d: pd.DataFrame) -> pd.DataFrame:
    """Outcomes for a decision at the close of t: enter at open t+1, exit at close t+h."""
    o, h, l, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
    n = len(d)
    out = {}
    for hz in HORIZONS:
        ret, mae, mfe = (np.full(n, np.nan) for _ in range(3))
        for t in range(n - hz):
            e = o[t + 1]
            ret[t] = c[t + hz] / e - 1
            mae[t] = l[t + 1 : t + hz + 1].min() / e - 1
            mfe[t] = h[t + 1 : t + hz + 1].max() / e - 1
        out[f"ret{hz}"], out[f"mae{hz}"], out[f"mfe{hz}"] = ret, mae, mfe
    f = pd.DataFrame(out, index=d.index)
    f["r1"] = d["close"].pct_change()
    return f


def block_boot_diff(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Mean(x[a]) - mean(x[b]) and its circular block-bootstrap SE (a, b boolean masks)."""
    n = len(x)
    nb = int(np.ceil(n / BLOCK))
    diffs = np.empty(BOOT_DRAWS)
    for i in range(BOOT_DRAWS):
        starts = RNG.integers(0, n, nb)
        idx = (starts[:, None] + np.arange(BLOCK)[None, :]).ravel()[:n] % n
        xa, xb = x[idx][a[idx]], x[idx][b[idx]]
        diffs[i] = (xa.mean() if len(xa) else np.nan) - (xb.mean() if len(xb) else np.nan)
    return float(np.nanmean(x[a]) - np.nanmean(x[b])), float(np.nanstd(diffs))


def group_row(name: str, x: pd.Series, mask: pd.Series, hz: int, f: pd.DataFrame, sign: float = 1.0) -> dict:
    m = mask & x.notna()
    r = sign * x[m]
    mae = (f[f"mae{hz}"] if sign > 0 else -f[f"mfe{hz}"])[m]  # adverse excursion for this side
    return {"group": name, "n": int(m.sum()), "mean": r.mean(), "median": r.median(),
            "win": (r > 0).mean(), "med_adverse": mae.median(), "p10": r.quantile(0.10), "p90": r.quantile(0.90)}


def t1_t3(sym: str, d: pd.DataFrame, s: pd.DataFrame, f: pd.DataFrame) -> list[dict]:
    rows = []
    valid = s["D1_sma50"].notna()
    dip, rip = (f["r1"] <= DIP) & valid, (f["r1"] >= RIP) & valid
    for det in DETECTORS:
        down = (s[det] == True) & valid  # noqa: E712  (NaN-safe)
        up = (s[det] == False) & valid  # noqa: E712
        for hz in HORIZONS:
            x = f[f"ret{hz}"]
            ok = x.notna().to_numpy()
            xv = x.to_numpy()
            base = {"sym": sym, "detector": det, "h": hz}
            # T1
            dlt, se = block_boot_diff(xv, (down.to_numpy() & ok), (up.to_numpy() & ok))
            rows.append({**base, "test": "T1_all_days", **group_row("DOWN", x, down, hz, f),
                         "up_mean": x[up].mean(), "diff_vs_ref": dlt, "se": se})
            # T2 dip buys
            for st_name, st in (("DOWN", down), ("UP", up)):
                a = (dip & st).to_numpy() & ok
                dlt, se = block_boot_diff(xv, a, st.to_numpy() & ok)
                rows.append({**base, "test": f"T2_dip_buy_{st_name}", **group_row(st_name, x, dip & st, hz, f),
                             "diff_vs_ref": dlt, "se": se})
            # T3 rip shorts (positive = short made money)
            for st_name, st in (("DOWN", down), ("UP", up)):
                a = (rip & st).to_numpy() & ok
                dlt, se = block_boot_diff(-xv, a, st.to_numpy() & ok)
                rows.append({**base, "test": f"T3_rip_short_{st_name}", **group_row(st_name, x, rip & st, hz, f, -1.0),
                             "diff_vs_ref": dlt, "se": se})
    return rows


def drawdown_episodes(c: pd.Series, thresh: float = 0.10) -> list[dict]:
    """Peak-to-trough episodes on closes whose depth >= thresh; recovery = new high."""
    eps, peak_i, trough_i = [], 0, 0
    v = c.to_numpy()
    for i in range(1, len(v)):
        if v[i] >= v[peak_i]:
            if v[trough_i] / v[peak_i] - 1 <= -thresh:
                eps.append((peak_i, trough_i, i))
            peak_i = trough_i = i
        elif v[i] < v[trough_i]:
            trough_i = i
    if v[trough_i] / v[peak_i] - 1 <= -thresh:
        eps.append((peak_i, trough_i, None))
    out = []
    for p, t, r in eps:
        out.append({"peak": c.index[p], "trough": c.index[t], "recovered": None if r is None else c.index[r],
                    "depth": v[t] / v[p] - 1, "peak_i": p, "trough_i": t, "end_i": len(v) - 1 if r is None else r})
    return out


def t4(sym: str, d: pd.DataFrame, s: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    c = d["close"]
    eps = drawdown_episodes(c)
    rows, fa = [], []
    in_ep = np.zeros(len(c), bool)
    for e in eps:
        in_ep[e["peak_i"] : e["end_i"] + 1] = True
    years = (c.index[-1] - c.index[200]).days / 365.25
    for det in DETECTORS:
        st = s[det]
        start = (st == True) & (st.shift(1) == False)  # noqa: E712
        starts = np.flatnonzero(start.to_numpy())
        false_flags = int(sum(1 for i in starts if not in_ep[i]))
        fa.append({"sym": sym, "detector": det, "down_share": float((st == True).mean()),  # noqa: E712
                   "flips_per_year": len(starts) / years, "false_flags_per_year": false_flags / years})
        for e in eps:
            if e["peak_i"] < 200:
                continue
            w = st.iloc[e["peak_i"] : e["trough_i"] + 1]
            hit = np.flatnonzero((w == True).to_numpy())  # noqa: E712
            row = {"sym": sym, "detector": det, "peak": e["peak"].date(), "trough": e["trough"].date(),
                   "depth": e["depth"], "sessions_peak_to_trough": e["trough_i"] - e["peak_i"]}
            if len(hit):
                fi = e["peak_i"] + hit[0]
                flag_px, peak_px, tr_px = c.iloc[fi], c.iloc[e["peak_i"]], c.iloc[e["trough_i"]]
                row.update(flag_date=c.index[fi].date(), sessions_to_flag=int(hit[0]),
                           lost_at_flag=flag_px / peak_px - 1, share_of_dd_lost=(flag_px / peak_px - 1) / e["depth"],
                           remaining_after_flag=tr_px / flag_px - 1)
            else:
                row.update(flag_date=None, sessions_to_flag=None, lost_at_flag=None, share_of_dd_lost=None,
                           remaining_after_flag=None)
            rows.append(row)
    return rows, fa


def book(d: pd.DataFrame, down: pd.Series, mode: str) -> pd.Series:
    """Daily returns. Position p decided at close t applies from the open of t+1."""
    if mode == "long":
        p = pd.Series(1.0, index=d.index)
    elif mode == "long_flat":
        p = (~down.astype(bool)).astype(float)
    elif mode == "long_short":
        p = np.where(down.astype(bool), -1.0, 1.0)
        p = pd.Series(p, index=d.index)
    else:
        raise ValueError(mode)
    o, c = d["open"], d["close"]
    r_on = o / c.shift(1) - 1          # close s-1 -> open s, held by the position set at open s-1
    r_id = c / o - 1                   # open s -> close s, held by the position set at open s
    p_new, p_old = p.shift(1), p.shift(2)
    cost = (p_new - p_old).abs() * COST_BP / 1e4
    r = (1 + p_old * r_on) * (1 + p_new * r_id) - 1 - cost
    return r.dropna()


def stats(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    vol = r.std() * np.sqrt(252)
    return {"cagr": eq.iloc[-1] ** (1 / yrs) - 1, "vol": vol, "sharpe": r.mean() * 252 / vol if vol else np.nan,
            "mdd": (eq / eq.cummax() - 1).min(), "total": eq.iloc[-1] - 1}


def t5(sym: str, d: pd.DataFrame, s: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    rows, yearly = [], []
    sub = d.iloc[200:]
    base = book(sub, pd.Series(False, index=sub.index), "long")
    rows.append({"sym": sym, "detector": "-", "mode": "long", **stats(base), "switches_per_year": 0.0,
                 "placebo_sharpe_pct": None, "placebo_mdd_pct": None})
    for det in DETECTORS:
        st = s[det].iloc[200:].astype(bool)
        sw = float((st != st.shift(1)).iloc[1:].sum()) / (len(st) / 252)
        for mode in ("long_flat", "long_short"):
            r = book(sub, st, mode)
            real = stats(r)
            sh, md = [], []
            arr = st.to_numpy()
            for _ in range(PLACEBO_DRAWS):
                k = int(RNG.integers(MIN_SHIFT, len(arr) - MIN_SHIFT))
                ps = stats(book(sub, pd.Series(np.roll(arr, k), index=st.index), mode))
                sh.append(ps["sharpe"]); md.append(ps["mdd"])
            rows.append({"sym": sym, "detector": det, "mode": mode, **real, "switches_per_year": sw,
                         "placebo_sharpe_pct": float((np.array(sh) < real["sharpe"]).mean()),
                         "placebo_mdd_pct": float((np.array(md) < real["mdd"]).mean()),
                         "placebo_sharpe_median": float(np.median(sh))})
            for y, g in r.groupby(r.index.year):
                yearly.append({"sym": sym, "detector": det, "mode": mode, "year": y, "ret": (1 + g).prod() - 1,
                               "long_ret": (1 + base.loc[g.index]).prod() - 1})
    return rows, yearly


def current_episode(data: dict[str, pd.DataFrame], st: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    q = data["QQQ"]["close"]
    for sym in INSTRUMENTS:
        c = data[sym]["close"]
        recent = c.loc["2026-09-01":]
        pk = recent.idxmax()
        for det in DETECTORS:
            w = st[sym][det].loc[pk:]
            hit = w[w == True]  # noqa: E712
            fd = hit.index[0] if len(hit) else None
            rows.append({"sym": sym, "detector": det, "peak": pk.date(), "peak_close": c[pk],
                         "flag_date": None if fd is None else fd.date(),
                         "own_lost_at_flag": None if fd is None else c[fd] / c[pk] - 1,
                         "qqq_at_flag": None if fd is None else q[fd],
                         "own_move_flag_to_1119": None if fd is None else c.iloc[-1] / c[fd] - 1,
                         "state_1119": bool(st[sym][det].iloc[-1])})
    return rows


def main() -> None:
    OUT.mkdir(exist_ok=True)
    data = {s: load(s) for s in INSTRUMENTS}
    idx = data["QQQ"].index
    for s, d in data.items():
        if not d.index.equals(idx):
            raise ValueError(f"{s}: session calendar differs from QQQ")
    st = {s: states(d) for s, d in data.items()}
    fw = {s: forward(d) for s, d in data.items()}
    r13, r4, fa, r5, yr = [], [], [], [], []
    for s in INSTRUMENTS:
        r13 += t1_t3(s, data[s], st[s], fw[s])
        a, b = t4(s, data[s], st[s]); r4 += a; fa += b
        a, b = t5(s, data[s], st[s]); r5 += a; yr += b
    pd.DataFrame(r13).to_csv(OUT / "t1_t3_conditional.csv", index=False)
    pd.DataFrame(r4).to_csv(OUT / "t4_detection.csv", index=False)
    pd.DataFrame(fa).to_csv(OUT / "t4_flip_rates.csv", index=False)
    pd.DataFrame(r5).to_csv(OUT / "t5_books.csv", index=False)
    pd.DataFrame(yr).to_csv(OUT / "t5_yearly.csv", index=False)
    pd.DataFrame(current_episode(data, st)).to_csv(OUT / "current_episode.csv", index=False)
    meta = {"first_session": str(idx[0].date()), "last_session": str(idx[-1].date()), "sessions": len(idx),
            "first_scored_session": str(idx[200].date()), "bars_dir": str(BARS.relative_to(REPO)),
            "bars_fetched_at_utc": json.loads((BARS / "_manifest.json").read_text())["runs"][-1]["fetched_at_utc"]}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
