"""E2: a general index-level regime policy on 40 years (PLAN.md, pre-registered 2026-11-24 15:19 ET).

Books on the index (primary Nasdaq-100 1986+, secondary S&P 500 1962+), all starting the first day
every book's input exists:
  bh            buy-and-hold
  longflat_v4   flat when the long-history vote (scaled to 7) is >= 4, else long
  size_regime   live multipliers by state: RISK_ON/NEUTRAL 1.0, DETERIORATING 0.8, RISK_OFF 0.5,
                CRISIS 0.25
  voltarget15   min(1, 15% / trailing RV20 of the index) (no leverage)
  sma200        long when close > SMA200, else flat
Uninvested capital earns the 3-month bill (prior day's ^IRX / 252). 2 bp per unit turnover.

Execution, changed from the plan BEFORE running (2026-11-24 21:35 ET): the plan said "trade the
next open", but Yahoo index opens before 2000 (NDX) and 2008 (S&P) are mostly copies of the prior
close (E0 check: 54-99% of rows per year). The weight decided at the close of t is therefore
traded at the close of t+1, one session later and conservative. The next-open variant is run as a
check where opens are real (NDX 2001+, S&P 2008+).

Index returns are price only (no dividends): this understates every book's return by the dividend
yield times its time invested, so it slightly favours the defensive books relative to bh (which is
always invested). Secondary check: S&P total-return index 1988+ (^SP500TR).

Crashes (peak and trough found inside each fixed search window, per index): 1987, 1990, 2000-02,
2007-09, 2011, 2015-16, 2018, 2020, 2022, 2025, 2026 (ongoing, through the last session).
Per crash: each book's max drawdown inside [peak, trough] vs bh's. The trough is the window's
low and the peak the high before it (fixed 2026-11-24 21:40 ET after the first run took the
window's high first, which turned NDX 2020 into a 3-day June dip and dropped 2025).

Added after the first run (2026-11-24 21:40 ET), because every book passed the rule and part of
that is mechanical (any book invested < 100% beats bh in a crash): a MATCHED-EXPOSURE control per
book, a constant weight equal to the book's mean exposure with the rest in bills. Reported as
crash wins and full-sample MDD/Sharpe vs that control; not part of the pre-registered decision.

Control: circular shift of each book's weight series (>= 250 sessions, 1,000 draws); MDD and
Sharpe percentile of the real book. Sharpe difference vs bh: 95% CI from a circular block
bootstrap of daily excess-return pairs (60-session blocks, 1,000 draws).

Decision (fixed): a book becomes the default general policy if (i) its crash MDD beats bh's in at
least 3 of every 4 crashes in the sample, (ii) the upper end of its Sharpe-minus-bh CI is >= 0,
and (iii) its full-sample MDD placebo percentile is > 90.

    .venv/bin/python research/regime_experiments_2026-11-24/e2_general_policy.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regime_common as rc  # noqa: E402

COST = 0.0002
DRAWS = 1000
SEED = 20261124
CRASH_WINDOWS = {
    "1987": ("1987-06-01", "1987-12-31"), "1990": ("1990-06-01", "1990-12-31"),
    "2000-02": ("2000-01-01", "2002-12-31"), "2007-09": ("2007-07-01", "2009-06-30"),
    "2011": ("2011-04-01", "2011-12-31"), "2015-16": ("2015-06-01", "2016-06-30"),
    "2018": ("2018-08-01", "2018-12-31"), "2020": ("2020-01-01", "2020-06-30"),
    "2022": ("2021-11-01", "2022-12-31"), "2025": ("2025-01-01", "2025-06-30"),
    "2026": ("2026-09-01", "2026-12-31"),
}
# secondary, S&P only (reported, not part of the decision)
SPX_EARLY_CRASHES = {
    "1966": ("1966-01-01", "1966-12-31"), "1968-70": ("1968-11-01", "1970-07-31"),
    "1973-74": ("1973-01-01", "1974-12-31"), "1980": ("1980-01-01", "1980-04-30"),
    "1980-82": ("1980-11-01", "1982-08-31"),
}
BOOKS = ("bh", "longflat_v4", "size_regime", "voltarget15", "sma200")


def target_weights(fr: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    w = pd.DataFrame(index=fr.index)
    w["bh"] = 1.0
    w["longflat_v4"] = np.where(fr["vote7"] >= 4, 0.0, 1.0)
    w["size_regime"] = fr["state"].map(rc.LIVE_SIZE).astype(float)
    w["voltarget15"] = np.minimum(1.0, 0.15 / fr["trail_rv20"])
    sma200 = close.rolling(200, min_periods=200).mean()
    w["sma200"] = (close > sma200).astype(float).where(sma200.notna())
    ok = fr["vote7"].notna() & fr["trail_rv20"].gt(0) & sma200.notna()
    return w[ok]


def run_book(sig: np.ndarray, r: np.ndarray, rf: np.ndarray) -> np.ndarray:
    """sig[t] decided at close t, traded at close t+1, held over day t+2."""
    pos = np.concatenate([[np.nan, np.nan], sig[:-2]])
    pos[:2] = sig[0]
    trade = np.abs(np.diff(pos, append=pos[-1]))  # traded at close t: pos[t+1]-pos[t]
    return pos * r + (1 - pos) * rf - COST * trade


def run_book_next_open(sig: np.ndarray, gap: np.ndarray, intra: np.ndarray, rf: np.ndarray) -> np.ndarray:
    """sig[t] decided at close t, traded at the open of t+1."""
    old = np.concatenate([[sig[0]], sig[:-1]])  # position carried into day t's open
    new = old.copy()
    new[1:] = sig[:-1]
    old[1:] = np.concatenate([[sig[0]], sig[:-2]])
    return (1 + old * gap) * (1 + new * intra + (1 - new) * rf) - 1 - COST * np.abs(new - old)


def metrics(ret: np.ndarray, rf: np.ndarray, pos: np.ndarray | None = None) -> dict:
    eq = np.cumprod(1 + ret)
    years = len(ret) / 252
    ex = ret - rf
    out = {"cagr": eq[-1] ** (1 / years) - 1, "vol": ret.std() * np.sqrt(252),
           "sharpe": ex.mean() / ex.std() * np.sqrt(252) if ex.std() > 0 else np.nan,
           "mdd": (eq / np.maximum.accumulate(eq) - 1).min()}
    if pos is not None:
        out["invested"] = float(np.mean(pos))
        out["turnover_per_yr"] = float(np.abs(np.diff(pos)).sum() / years)
    return out


def mdd(ret: np.ndarray) -> float:
    eq = np.cumprod(1 + ret)
    return float((eq / np.maximum.accumulate(eq) - 1).min())


def crash_table(close: pd.Series, rets: pd.DataFrame, windows: dict) -> pd.DataFrame:
    rows = []
    for name, (a, b) in windows.items():
        c = close.loc[a:b]
        if len(c) < 20 or c.index[0] < rets.index[0]:
            continue
        trough = c.idxmin()  # trough first: the crash's low, then the high before it
        peak = c.loc[:trough].idxmax()
        if trough <= peak:
            continue
        seg = rets.loc[peak:trough].iloc[1:]
        row = {"crash": name, "peak": str(peak.date()), "trough": str(trough.date()),
               "index_dd": c.loc[trough] / c.loc[peak] - 1, "sessions": len(seg)}
        for book in rets.columns:
            row[f"{book}_mdd"] = mdd(seg[book].to_numpy())
        rows.append(row)
    return pd.DataFrame(rows)


def block_ci_sharpe_diff(a: np.ndarray, b: np.ndarray, rng, block: int = 60) -> tuple[float, float]:
    n = len(a)
    nb = int(np.ceil(n / block))
    d = np.empty(DRAWS)
    for k in range(DRAWS):
        idx = ((rng.integers(0, n, nb)[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]
        x, y = a[idx], b[idx]
        d[k] = (x.mean() / x.std() - y.mean() / y.std()) * np.sqrt(252)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def evaluate(target: str, start: str, early: dict | None = None, close_override: pd.Series | None = None) -> dict:
    fr, panel_name = rc.build_frame(target)
    fr = fr.loc[start:]
    p, _ = rc.latest_panel()
    close = (close_override if close_override is not None else p[target]).reindex(fr.index)
    w = target_weights(fr, close)
    close = close.reindex(w.index)
    r = close.pct_change().fillna(0.0).to_numpy()
    rf = (p["tbill3m"].shift(1).reindex(w.index) / 100 / 252).to_numpy()
    if np.isnan(rf).any():
        raise ValueError(f"{target}: {int(np.isnan(rf).sum())} sessions lack a bill rate")
    rets = pd.DataFrame({b: run_book(w[b].to_numpy(), r, rf) for b in BOOKS}, index=w.index)
    rets["bh"] = r  # always fully invested, no trades
    rng = np.random.default_rng(SEED)
    summary = {}
    for b in BOOKS:
        sig = w[b].to_numpy()
        pos = np.concatenate([[sig[0]] * 2, sig[:-2]])
        m = metrics(rets[b].to_numpy(), rf, pos)
        if b != "bh":
            n = len(sig)
            shifts = rng.integers(250, n - 250, DRAWS)
            null_mdd, null_sh = np.empty(DRAWS), np.empty(DRAWS)
            for k, s in enumerate(shifts):
                x = run_book(np.roll(sig, s), r, rf)
                null_mdd[k] = mdd(x)
                ex = x - rf
                null_sh[k] = ex.mean() / ex.std() * np.sqrt(252)
            m["mdd_placebo_pct"] = float((null_mdd < m["mdd"]).mean() * 100)
            m["sharpe_placebo_pct"] = float((null_sh < m["sharpe"]).mean() * 100)
            m["sharpe_minus_bh_ci"] = block_ci_sharpe_diff(rets[b].to_numpy() - rf, rets["bh"].to_numpy() - rf, rng)
        summary[b] = m
    crashes = crash_table(close, rets, CRASH_WINDOWS)
    for b in BOOKS[1:]:
        wins = int((crashes[f"{b}_mdd"] > crashes["bh_mdd"]).sum())
        s = summary[b]
        s["crash_wins"] = f"{wins}/{len(crashes)}"
        s["ADOPT"] = bool(wins >= 0.75 * len(crashes) and s["sharpe_minus_bh_ci"][1] >= 0
                          and s["mdd_placebo_pct"] > 90)
        const = s["invested"]
        mret = pd.DataFrame({"m": const * r + (1 - const) * rf}, index=w.index)
        mc = crash_table(close, mret, CRASH_WINDOWS)
        s["matched_exposure"] = const
        s["matched_mdd"], s["matched_sharpe"] = mdd(mret["m"].to_numpy()), metrics(mret["m"].to_numpy(), rf)["sharpe"]
        s["crash_wins_vs_matched"] = f"{int((crashes[f'{b}_mdd'].to_numpy() > mc['m_mdd'].to_numpy()).sum())}/{len(crashes)}"
        crashes[f"{b}_matched_mdd"] = mc["m_mdd"].to_numpy()
    dec = []
    for d, g in rets.groupby(rc.decade_of(rets.index).to_numpy()):
        rfd = rf[rets.index.isin(g.index)]
        for b in BOOKS:
            dec.append({"decade": d, "book": b, **metrics(g[b].to_numpy(), rfd)})
    out = {"panel": panel_name, "first": str(w.index[0].date()), "last": str(w.index[-1].date()),
           "books": summary, "crashes": crashes, "decades": pd.DataFrame(dec)}
    if early:
        out["early_crashes"] = crash_table(close, rets, early)
    return out, w, r, rf, close


def next_open_check(target: str, start: str, w: pd.DataFrame, rf: np.ndarray) -> pd.DataFrame:
    raw = sorted((rc.PANEL_DIR / "raw").glob("*"))[-1] / f"yahoo_{target}.parquet"
    ohlc = pd.read_parquet(raw).reindex(w.index)
    sel = (w.index >= pd.Timestamp(start))
    o, c = ohlc["open"].to_numpy(), ohlc["close"].to_numpy()
    gap = o / np.concatenate([[np.nan], c[:-1]]) - 1
    intra = c / o - 1
    sel &= np.isfinite(gap) & np.isfinite(intra)
    rows = []
    for b in BOOKS[1:]:
        sig = w[b].to_numpy()[sel]
        ro = run_book_next_open(sig, gap[sel], intra[sel], rf[sel])
        rc_ = run_book(sig, (c / np.concatenate([[np.nan], c[:-1]]) - 1)[sel], rf[sel])
        rows.append({"book": b, "from": start, **{f"next_open_{k}": v for k, v in metrics(ro, rf[sel]).items()},
                     **{f"next_close_{k}": v for k, v in metrics(rc_, rf[sel]).items()}})
    return pd.DataFrame(rows)


def main() -> int:
    rc.OUT.mkdir(exist_ok=True)
    results = {}
    tables = {}
    for target, start, early, open_from in (("ndx", "1986-01-01", None, "2001-01-01"),
                                             ("spx", "1962-01-01", SPX_EARLY_CRASHES, "2008-01-01")):
        out, w, r, rf, close = evaluate(target, start, early)
        results[target] = {"first": out["first"], "last": out["last"], "books": out["books"]}
        tables[f"{target}_crashes"] = out["crashes"]
        tables[f"{target}_decades"] = out["decades"]
        if early:
            tables[f"{target}_early_crashes"] = out["early_crashes"]
        tables[f"{target}_next_open"] = next_open_check(target, open_from, w, rf)
    p, _ = rc.latest_panel()
    tr, *_ = evaluate("spx", "1988-01-04", None, close_override=p["spx_tr"])
    results["spx_total_return_1988"] = {"first": tr["first"], "books": tr["books"]}
    tables["spx_tr_crashes"] = tr["crashes"]
    for k, t in tables.items():
        t.to_csv(rc.OUT / f"e2_{k}.csv", index=False)
    (rc.OUT / "e2_summary.json").write_text(json.dumps(results, indent=2, default=str))
    pd.set_option("display.width", 250)
    for target in ("ndx", "spx", "spx_total_return_1988"):
        print(f"== {target} {results[target]['first']}..")
        print(pd.DataFrame(results[target]["books"]).T.to_string())
    for k in ("ndx_crashes", "spx_crashes", "spx_early_crashes", "ndx_next_open", "spx_next_open"):
        print(f"== {k}")
        print(tables[k].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
