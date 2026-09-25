"""Step 8 — would admitting $2-5M ADV names improve the momentum top-3?

Uses the walk-forward OOF predictions of the live model family
(models/expansion_v1/oof_preds.parquet; every score is out-of-fold) and mirrors the
live selection: momentum_candidate_mask on the matrix features, then top-3 by score
per 4H bar.

Point-in-time admission from the daily bars, on the live weekly cadence
(07_pool_admission.py): history >= 200 bars, $1 <= close <= $1000, and
  core  30d ADV >= $5M                 (today's rule)
  band  $2M <= 30d ADV < $5M           (the proposed extension)

Compared per 4H bar, paired:
  A  top-3 from core only             (live)
  B  top-3 from core + band           (proposed)
plus the ranker's lift INSIDE each slice (top-3 minus slice mean, and minus an
ATR-quintile-matched random draw, since top-k on noise buys a vol tilt; see the
top-k null-control memory).

Outcome: fwd_close_return / fwd_max_return, the 25-bar (~10 session) label columns
carried on the OOF rows. Costs: an assumed extra 20bp round trip on band picks
(wider spreads), reported net and gross.

BIAS, stated first: the matrix ticker set is TODAY's pool, so a name that sat in the
$2-5M band historically is one that later grew liquid enough to be admitted. That
survivorship FAVOURS the band. A null or negative B-A is therefore robust; a positive
one is an upper bound. Stats use a 25-bar moving-block bootstrap (labels overlap).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from strategies.momentum_expansion.inference.candidate_filter import momentum_candidate_mask  # noqa: E402

OOF = REPO / "strategies/momentum_expansion/models/expansion_v1/oof_preds.parquet"
MATRIX = REPO / "strategies/momentum_expansion/data/processed/training_matrix_4h.parquet"
BARS = REPO / "Data/shared/bars/1d"
FILTER_COLS = ["low_price_flag", "dollar_vol_pctile_252", "dist_to_52w_high_atr",
               "xsec_near_high_rank", "rs_spy_20", "xsec_ret_20_rank", "range_pos_20",
               "atr_pct_14"]
K, BLOCK, BOOT, SEED = 3, 25, 2000, 7
BAND_COST = 0.0020


def admission(tickers) -> pd.DataFrame:
    rows = []
    for t in tickers:
        p = BARS / f"{t}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["timestamp", "close", "volume"])
        d["date"] = (pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert("America/New_York")
                     .dt.normalize().dt.tz_localize(None))
        d = d.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
        adv = (d["close"] * d["volume"]).rolling(30).mean()
        ok = (np.arange(1, len(d) + 1) >= 200) & d["close"].between(1.0, 1000.0)
        d["cls"] = np.select([ok & (adv >= 5e6), ok & (adv >= 2e6)], ["core", "band"], "out")
        # weekly rebuild: last session of week W decides week W+1
        wk = d["date"].dt.to_period("W-SUN")
        last = d.groupby(wk)["cls"].last().shift(1)
        d["cls_w"] = wk.map(last).fillna("out")
        rows.append(pd.DataFrame({"ticker": t, "date": d["date"], "cls": d["cls_w"]}))
    return pd.concat(rows, ignore_index=True)


def block_boot(x: np.ndarray, rng) -> tuple[float, float, float]:
    n = len(x)
    if n < BLOCK * 2:
        return float("nan"), float("nan"), float("nan")
    starts = rng.integers(0, n - BLOCK + 1, size=(BOOT, int(np.ceil(n / BLOCK))))
    idx = (starts[..., None] + np.arange(BLOCK)).reshape(BOOT, -1)[:, :n]
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def main() -> None:
    o = pd.read_parquet(OOF, columns=["score", "fwd_close_return", "fwd_max_return"]).reset_index()
    import pyarrow.parquet as pq
    have = [c for c in FILTER_COLS if c in pq.ParquetFile(MATRIX).schema_arrow.names]
    m = pd.read_parquet(MATRIX, columns=have).reset_index()
    x = o.merge(m, on=["timestamp", "ticker"], how="inner")
    print(f"oof {len(o):,}  joined {len(x):,}  filter cols {have}", flush=True)
    x = x[momentum_candidate_mask(x).to_numpy()].copy()
    x["date"] = (x["timestamp"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None))
    adm = admission(sorted(x["ticker"].unique()))
    x = x.merge(adm, on=["ticker", "date"], how="left")
    x["cls"] = x["cls"].fillna("out")
    x = x[x["cls"].isin(["core", "band"])].dropna(subset=["fwd_close_return", "score"])
    print(f"candidate rows: core {int((x.cls == 'core').sum()):,}  band {int((x.cls == 'band').sum()):,}  "
          f"bars {x.timestamp.nunique():,}", flush=True)
    x["atr_q"] = x.groupby("timestamp")["atr_pct_14"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) if len(s) >= 5 else 0)

    rng = np.random.default_rng(SEED)
    rec = []
    for ts, g in x.groupby("timestamp", sort=True):
        core = g[g.cls == "core"]
        if len(core) < K:
            continue
        a = core.nlargest(K, "score")
        b = g.nlargest(K, "score")
        band = g[g.cls == "band"]
        r = {"ts": ts, "n_core": len(core), "n_band": len(band),
             "A": a.fwd_close_return.mean(), "B": b.fwd_close_return.mean(),
             "B_net": (b.fwd_close_return - np.where(b.cls == "band", BAND_COST, 0)).mean(),
             "A_max": a.fwd_max_return.mean(), "B_max": b.fwd_max_return.mean(),
             "band_in_B": int((b.cls == "band").sum()),
             "core_mean": core.fwd_close_return.mean()}
        # ATR-matched random control for A: same atr quintiles as the picks
        ctrl = [core[core.atr_q == q].fwd_close_return.sample(1, random_state=rng.integers(1e9)).iloc[0]
                for q in a.atr_q if (core.atr_q == q).any()]
        r["A_ctrl"] = np.mean(ctrl) if ctrl else np.nan
        if len(band) >= K:
            bt = band.nlargest(K, "score")
            r["band_top"] = bt.fwd_close_return.mean()
            r["band_mean"] = band.fwd_close_return.mean()
            bc = [band[band.atr_q == q].fwd_close_return.sample(1, random_state=rng.integers(1e9)).iloc[0]
                  for q in bt.atr_q if (band.atr_q == q).any()]
            r["band_ctrl"] = np.mean(bc) if bc else np.nan
        rec.append(r)
    d = pd.DataFrame(rec).sort_values("ts").reset_index(drop=True)

    out = {"bars": len(d), "window": [str(d.ts.min()), str(d.ts.max())],
           "band_share_of_B_picks": float(d.band_in_B.sum() / (K * len(d))),
           "bars_where_B_differs": float((d.band_in_B > 0).mean())}
    stats = {}
    for name, series in {
        "A_top3 (live, core only)": d.A, "B_top3 (core+band) gross": d.B,
        "B_top3 net of band cost": d.B_net,
        "B - A gross": d.B - d.A, "B - A net": d.B_net - d.A,
        "B - A fwd_max": d.B_max - d.A_max,
        "A lift vs core mean": d.A - d.core_mean,
        "A lift vs ATR-matched random": d.A - d.A_ctrl,
        "band top3 lift vs band mean": (d.band_top - d.band_mean).dropna(),
        "band top3 lift vs ATR-matched random": (d.band_top - d.band_ctrl).dropna(),
        "band top3 - core top3 (same bars)": (d.band_top - d.A)[d.band_top.notna()],
    }.items():
        s = series.dropna().to_numpy()
        mu, lo, hi = block_boot(s, rng)
        stats[name] = {"n_bars": len(s), "mean": mu, "ci95": [lo, hi]}
        print(f"  {name:<40} n={len(s):>5}  mean {mu:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    out["stats"] = stats
    print(json.dumps({k: v for k, v in out.items() if k != "stats"}, indent=2, default=str))
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "adv_band_topk.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
