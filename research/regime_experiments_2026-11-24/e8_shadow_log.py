"""E8: forward shadow ledger for the regime candidates (PLAN.md E8). Research only: it never touches
orders, the nervous system or any live table; it only reads them.

One row per completed session, APPEND-ONLY: a date already in the log is never rewritten (a forward
shadow must keep what was knowable when it was logged; later revisions or refits would leak).
Run after E0 has fetched a fresh panel:

    .venv/bin/python research/regime_experiments_2026-11-24/e0_fetch_long_history.py
    PYTHONPATH=. .venv/bin/python -W ignore research/regime_experiments_2026-11-24/e8_shadow_log.py

Logged per session t (all known at the close of t, used for t+1):
  live 7-vote count/state    from Data/shared/market_regime/daily_regime.parquet (what live used)
  LH vote / state            regime_common (40-year proxy)
  hmm_p_turbulent            HMM-R K=2 on Nasdaq-100 returns; parameters fit once per calendar year
                             on data before 1 January and frozen (Data/research/regime_shadow/)
  vol_target_w               min(1, 15% / NDX RV20)
  candidates                 sector mean pairwise 60d corr (+ trailing-252 percentile), VIX z,
                             dollar 20d return, GNS / BNB / GNS_tech (E4 definitions) + weak flags
                             (GNS became an E8 candidate 2026-11-25 on refreshed earnings data)
  profile weights            index_live_mult, index_vol_target, index_hmm_size, index_hmm_longflat,
                             momentum_riskoff_up (1.5 in live RISK_OFF/CRISIS else 1.0; a hypothesis
                             from E7-M, logged forward only)
Read rule (PLAN.md E8): at the later of 40 logged sessions or 10 sessions in RISK_OFF/CRISIS,
with the MDE stated at the read.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3_factor_completeness as e3  # noqa: E402
import e4_good_news_sold as e4  # noqa: E402
import e9_regime_model as e9  # noqa: E402
import regime_common as rc  # noqa: E402

SHADOW = rc.REPO / "Data" / "research" / "regime_shadow"
LOG = SHADOW / "shadow_log.parquet"
DAILY_REGIME = rc.REPO / "Data" / "shared" / "market_regime" / "daily_regime.parquet"


def hmm_params(year: int, r: pd.Series) -> e9.GaussianHMM:
    path = SHADOW / f"hmm_r2_params_{year}.json"
    if path.exists():
        d = json.loads(path.read_text())
        m = e9.GaussianHMM(2, 0)
        m.mu, m.var, m.A, m.pi = (np.array(d[k]) for k in ("mu", "var", "A", "pi"))
        m.z_mu, m.z_sd = np.array(d["z_mu"]), np.array(d["z_sd"])
        return m
    X = r[r.index < f"{year}-01-01"].to_numpy()[:, None]
    z_mu, z_sd = X.mean(axis=0), X.std(axis=0)
    best = None
    for s in range(5):
        m = e9.GaussianHMM(2, e9.SEED + s)
        m.init_random((X - z_mu) / z_sd)
        ll = m.fit((X - z_mu) / z_sd)
        if best is None or ll > best[0]:
            best = (ll, m)
    m = best[1]
    m.z_mu, m.z_sd = z_mu, z_sd
    path.write_text(json.dumps({"mu": m.mu.tolist(), "var": m.var.tolist(), "A": m.A.tolist(), "pi": m.pi.tolist(),
                                "z_mu": z_mu.tolist(), "z_sd": z_sd.tolist(), "fit_through": str(year - 1),
                                "loglik": best[0]}))
    return m


def build_row() -> dict:
    p, panel_name = rc.latest_panel()
    t = p["ndx"].dropna().index[-1]
    fr, _ = rc.build_frame("ndx")
    live = pd.read_parquet(DAILY_REGIME)
    live["date"] = pd.to_datetime(live["date"])
    from signals.market_regime.regime_vote import classify_frame

    lv = classify_frame(live.set_index("date")).loc[:t].tail(1)
    r = (np.log(p["ndx"]).diff() * 100).dropna()
    m = hmm_params(t.year, r)
    post = m.filtered(((r.to_numpy()[:, None]) - m.z_mu) / m.z_sd)
    hi = m.order()[-1]
    p_turb = float(post[-1, hi])
    F, fam, _, _ = e3.build_features("ndx")
    corr = F["sec_avg_corr60"]
    rx, _ = e4.reactions()
    sess = p.loc["2016-01-01":].index
    gns = e4.daily_factor(rx, sess, rx["surprise"] >= 5, 30)
    bnb = e4.daily_factor(rx, sess, rx["surprise"] <= -5, 15)
    sec = pd.read_parquet(e4.SECTORS).sort_values("snapshot_month").drop_duplicates("ticker", keep="last")
    tech = set(sec.loc[sec["sector_etf"] == "XLK", "ticker"])
    gns_tech = e4.daily_factor(rx, sess, (rx["surprise"] >= 5) & rx["ticker"].isin(tech), 10)

    def pct_rank(s: pd.Series) -> float:
        if not np.isfinite(s.get(t, np.nan)):
            return np.nan  # never rank a stale last-valid value as today's
        w = s.loc[:t].dropna().iloc[-252:]
        return float((w < w.iloc[-1]).mean()) if len(w) >= 252 else np.nan

    live_state = str(lv["regime"].iloc[-1]) if len(lv) else "UNKNOWN"
    vt = float(min(1.0, 0.15 / fr["trail_rv20"].loc[t]))
    row = {
        "date": t, "logged_at": pd.Timestamp.now(tz="America/New_York"), "panel": panel_name,
        "live_vote_date": lv.index[-1] if len(lv) else pd.NaT,
        "live_vote_count": float(lv["regime_vote_count"].iloc[-1]) if len(lv) else np.nan,
        "live_state": live_state,
        "lh_vote7": float(fr["vote7"].loc[t]), "lh_state": str(fr["state"].loc[t]),
        "hmm_p_turbulent": p_turb, "vol_target_w": vt,
        "sector_corr60": float(corr.loc[t]), "sector_corr60_pct252": pct_rank(corr),
        "vix_z": float(rc.zscore(p["vix"]).loc[t]), "dollar_ret20": float(np.log(p["dxy"]).diff(20).loc[t]),
        "gns": float(gns.loc[t]), "gns_pct252": pct_rank(gns), "bnb": float(bnb.loc[t]), "bnb_pct252": pct_rank(bnb),
        "gns_tech": float(gns_tech.loc[t]), "gns_tech_pct252": pct_rank(gns_tech),
        "index_live_mult": rc.LIVE_SIZE.get(live_state, np.nan), "index_vol_target": vt,
        "index_hmm_size": 1 - 0.75 * p_turb, "index_hmm_longflat": float(p_turb <= 0.5),
        "momentum_riskoff_up": 1.5 if live_state in ("RISK_OFF", "CRISIS") else 1.0,
    }
    row["gns_weak"] = bool(row["gns_pct252"] < 0.20) if np.isfinite(row["gns_pct252"]) else None
    row["gns_tech_weak"] = bool(row["gns_tech_pct252"] < 0.20) if np.isfinite(row["gns_tech_pct252"]) else None
    return row


def main() -> int:
    SHADOW.mkdir(parents=True, exist_ok=True)
    row = build_row()
    log = pd.read_parquet(LOG) if LOG.exists() else pd.DataFrame()
    if len(log) and (pd.to_datetime(log["date"]) == row["date"]).any():
        print(f"{row['date'].date()} already logged; append-only, nothing written")
        return 0
    if row["live_vote_date"] is pd.NaT or pd.isna(row["live_vote_date"]) or row["live_vote_date"] < row["date"]:
        print(f"warning: live vote is from {row['live_vote_date']}, panel session {row['date'].date()}", flush=True)
    out = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    tmp = LOG.with_suffix(".tmp.parquet")
    out.to_parquet(tmp)
    tmp.replace(LOG)
    print(json.dumps({k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in row.items()}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
