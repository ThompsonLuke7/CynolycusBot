"""One decision panel shared by the falsification, continuation and gate experiments.

Rows are (4H decision bar x ticker) from the Meta research matrix, which carries
momentum's and HTF's walk-forward OUT-OF-FOLD scores (leak-free, 21-day embargo)
back to 2022-11 -- the same source `scripts/rank_depth/oof_replication.py` uses.
Each row gets, from the daily bar cache:

  entry           the first session strictly after the decision bar; entry at ITS OPEN
  fwdret_{h}      open-to-close return over h sessions, h in 5/10/15/20/30
  mfe_{h}/mae_{h} favourable / adverse excursion from the entry open over h sessions
  day-5 path      ret/mfe/mae/up-day share/close-vs-high/volume ratio at the checkpoint
  rem_{c}_{h}     return from the day-5 close to the day-h close (continuation target)
  factors         past 20/60/120d return, 20d realised vol, log dollar volume,
                  distance to 252d high, 60d beta to SPY -- all strictly prior to entry
  regime          the matrix's own PIT regime panel, constant within a bar
  spyret_{h}      SPY's return over the same window (for market-relative work)

Every forward window is checked against the non-organic corporate-action flags
(the daily cache is UNADJUSTED) and flagged rows are marked, not dropped, so each
experiment can decide.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DATA = REPO / "research/execution_quality/data"
MATRIX = REPO / "signals/meta_context/meta_ranker/meta_ranker_matrix_research.parquet"
BARS_1D = REPO / "Data/shared/bars/1d"
FLAGS = DATA / "corporate_action_flags.parquet"
OUT = DATA / "decision_panel.parquet"

HOLDS = [5, 10, 15, 20, 30]
CHECKPOINT = 5
MAX_H = max(HOLDS)
MIN_XS = 200
BETA_WINDOW = 60
REGIME_COLS = ["regime_spy_trend", "regime_spy_ret_20", "regime_vix_z", "regime_vix_high",
               "breadth_z", "sector_dispersion_z", "spy_rv20_z", "risk_appetite_z",
               "liquidity_stress_z", "credit_risk_z"]


def load_panel(require: list[str] | None = None, drop_flagged: bool = True) -> pd.DataFrame:
    """Shared reader for the experiments. Drops corporate-action-flagged windows by default."""
    p = pd.read_parquet(OUT)
    if drop_flagged:
        p = p[~p["ca_flagged"]]
    if require:
        p = p.dropna(subset=require)
    return p


def ticker_frame(ticker: str, spy_ret: pd.Series | None = None) -> pd.DataFrame | None:
    """Daily forward/path/factor columns for one ticker. `spy_ret` is indexed by session_date."""
    path = BARS_1D / f"{ticker}.parquet"
    if not path.exists():
        return None
    d = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if len(d) < 150 + MAX_H:
        return None
    d["session_date"] = (pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert("America/New_York")
                         .dt.normalize().dt.tz_localize(None))
    d = d.sort_values("session_date").reset_index(drop=True)
    o, h, lo_, c, v = (d[x].to_numpy(float) for x in ("open", "high", "low", "close", "volume"))
    out: dict[str, object] = {"ticker": ticker, "session_date": d["session_date"].to_numpy()}

    def fwd_max(arr, n):
        return pd.Series(arr).rolling(n).max().shift(-(n - 1)).to_numpy()

    def fwd_min(arr, n):
        return pd.Series(arr).rolling(n).min().shift(-(n - 1)).to_numpy()

    def fwd_close(n):
        return pd.Series(c).shift(-(n - 1)).to_numpy()

    r = pd.Series(c).pct_change()
    with np.errstate(divide="ignore", invalid="ignore"):
        for n in HOLDS:
            out[f"fwdret_{n}"] = np.where(o > 0, fwd_close(n) / o - 1.0, np.nan)
            out[f"mfe_{n}"] = np.where(o > 0, fwd_max(h, n) / o - 1.0, np.nan)
            out[f"mae_{n}"] = np.where(o > 0, fwd_min(lo_, n) / o - 1.0, np.nan)
        # day-5 checkpoint path: everything here is known by that session's close
        cp_close = fwd_close(CHECKPOINT)
        cp_high = fwd_max(h, CHECKPOINT)
        out["cp_ret"] = np.where(o > 0, cp_close / o - 1.0, np.nan)
        out["cp_mfe"] = out[f"mfe_{CHECKPOINT}"]
        out["cp_mae"] = out[f"mae_{CHECKPOINT}"]
        out["cp_close_vs_high"] = np.where(cp_high > 0, cp_close / cp_high - 1.0, np.nan)
        up = pd.Series((c > pd.Series(c).shift(1).to_numpy()).astype(float))
        out["cp_up_day_share"] = up.shift(-1).rolling(CHECKPOINT).mean().shift(-(CHECKPOINT - 1)).to_numpy()
        vol20 = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
        cp_vol = pd.Series(v).shift(-1).rolling(CHECKPOINT).mean().shift(-(CHECKPOINT - 1)).to_numpy()
        out["cp_volume_ratio"] = np.where(vol20 > 0, cp_vol / vol20, np.nan)
        for n in HOLDS:
            if n > CHECKPOINT:
                out[f"rem_{CHECKPOINT}_{n}"] = np.where(cp_close > 0, fwd_close(n) / cp_close - 1.0, np.nan)
        # factors, strictly prior to the entry session's open
        prev_c = pd.Series(c).shift(1)
        for n in (20, 60, 120):
            out[f"past_ret_{n}"] = (prev_c / pd.Series(c).shift(1 + n) - 1.0).to_numpy()
        out["past_vol_20"] = r.shift(1).rolling(20).std().to_numpy()
        out["log_dollar_vol_20"] = np.log1p(pd.Series(c * v).shift(1).rolling(20).mean()).to_numpy()
        out["dist_252_high"] = (prev_c / pd.Series(h).shift(1).rolling(252, min_periods=60).max() - 1.0).to_numpy()

    if spy_ret is None:
        out["beta_60"] = np.nan
        out["ret_1d"] = r.to_numpy()
    else:
        x = r.shift(1)
        y = pd.Series(out["session_date"]).map(spy_ret).shift(1)
        mx = x.rolling(BETA_WINDOW).mean()
        my = y.rolling(BETA_WINDOW).mean()
        cov = (x * y).rolling(BETA_WINDOW).mean() - mx * my
        var = (y * y).rolling(BETA_WINDOW).mean() - my * my
        out["beta_60"] = np.where(var > 0, cov / var, np.nan)
    return pd.DataFrame(out)


def main() -> None:
    t0 = time.time()
    spy = ticker_frame("SPY")
    if spy is None:
        raise SystemExit("SPY daily bars missing; the panel needs them for beta and benchmarks")
    spy_ret = pd.Series(spy["ret_1d"].to_numpy(), index=spy["session_date"])
    spy_fwd = spy[["session_date"] + [f"fwdret_{n}" for n in HOLDS]].rename(
        columns={f"fwdret_{n}": f"spyret_{n}" for n in HOLDS})

    cols = ["mom_score", "htf_score"] + REGIME_COLS
    m = pd.read_parquet(MATRIX, columns=cols).reset_index().dropna(subset=["mom_score"])
    size = m.groupby("timestamp")["ticker"].transform("size")
    m = m[size >= MIN_XS].copy()
    m["decision_day"] = m["timestamp"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    m["xs_size"] = m.groupby("timestamp")["ticker"].transform("size")
    print(f"matrix rows {len(m):,} bars {m['timestamp'].nunique():,} "
          f"({m['decision_day'].min().date()} .. {m['decision_day'].max().date()})", flush=True)

    tickers = sorted(m["ticker"].unique())
    frames = []
    for i, t in enumerate(tickers, 1):
        f = ticker_frame(t, spy_ret)
        if f is not None:
            frames.append(f)
        if i % 250 == 0:
            print(f"  daily frames {i}/{len(tickers)} ({time.time() - t0:.0f}s)", flush=True)
    ft = pd.concat(frames, ignore_index=True).merge(spy_fwd, on="session_date", how="left")
    print(f"  daily rows {len(ft):,} over {ft['ticker'].nunique()} tickers ({time.time() - t0:.0f}s)", flush=True)

    # Join each decision to the FIRST session strictly after it (no look-ahead).
    m["_key"] = m["decision_day"] + pd.Timedelta(days=1)
    m = m.sort_values("_key")
    ft = ft.sort_values("session_date")
    panel = pd.merge_asof(m, ft, left_on="_key", right_on="session_date", by="ticker",
                          direction="forward", tolerance=pd.Timedelta(days=7)).drop(columns=["_key"])
    panel = panel.rename(columns={"session_date": "entry_session"})
    panel["rank_mom"] = panel.groupby("timestamp")["mom_score"].rank(ascending=False, method="first")
    panel["rank_htf"] = panel.groupby("timestamp")["htf_score"].rank(ascending=False, method="first")

    flags = pd.read_parquet(FLAGS)
    flags = flags[~flags["organic"]]
    span = np.timedelta64(int(np.ceil(MAX_H * 7 / 5)) + 4, "D")
    bad = np.zeros(len(panel), bool)
    day = panel["entry_session"].to_numpy()
    tick = panel["ticker"].to_numpy()
    for t, gg in flags.groupby("ticker"):
        dates = np.sort(pd.to_datetime(gg["date"]).to_numpy())
        msk = tick == t
        if msk.any():
            st = day[msk]
            bad[msk] = np.searchsorted(dates, st + span, side="right") > np.searchsorted(dates, st, side="left")
    panel["ca_flagged"] = bad
    print(f"  corporate-action flagged rows: {int(bad.sum()):,} ({bad.mean():.3%})")

    for c in panel.select_dtypes("float64").columns:
        panel[c] = panel[c].astype("float32")
    panel.to_parquet(OUT, index=False)
    print(f"wrote {OUT} rows={len(panel):,} cols={len(panel.columns)} "
          f"fwdret_30 present {panel['fwdret_30'].notna().mean():.1%}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
