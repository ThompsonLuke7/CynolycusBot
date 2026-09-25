"""Step 7 — what admits a name to the momentum pool, and do we miss moves by admitting late?

Replays the three data-driven admission rules of
strategies/momentum_expansion/data/universe.py::score_universe point-in-time from the
shared daily bars, on the live cadence (rebuilt weekly on Sunday from data through the
prior session, held for the week):

    history   >= min_history_days (200) daily bars
    price     min_price (1) <= close <= max_price (1000)
    liquidity 30-day mean(close * volume) >= min_avg_dollar_vol ($5M)

NOT modelled: membership of the candidate pool (TOS export + swing universe + promoted
discovery names). That is today's list, so every result here is an UPPER bound on what
the three rules alone would admit.

Outcome, measured from the NEXT session's open (decision at close t, entry t+1):
    tail   max(high[t+1..t+10]) / open[t+1] - 1 >= 4 * ATR14_t / close_t   (EVENT_R = 4)
    fwd20  close[t+20] / open[t+1] - 1
Windows containing a non-organic corporate-action gap are masked (bars are UNADJUSTED).

Survivorship: Data/shared/bars/1d is today's file set (almost no delisted names), so
absolute rates are optimistic; the comparisons between admission states share the bias.
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
from core.corporate_actions import mask_windows  # noqa: E402

BARS = REPO / "Data/shared/bars/1d"
W0, W1 = pd.Timestamp("2021-01-01"), pd.Timestamp("2026-08-20")
MIN_HIST, MIN_PX, MAX_PX, MIN_ADV = 200, 1.0, 1000.0, 5e6
EVENT_R, HOLD = 4.0, 20
ALT_ADV = (1e6, 2e6, 3e6)
ALT_HIST = (60, 120)


def one(path: Path) -> pd.DataFrame | None:
    try:
        d = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close", "volume"])
    except Exception:
        return None
    if len(d) < 40:
        return None
    d["date"] = (pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert("America/New_York")
                 .dt.normalize().dt.tz_localize(None))
    d = d.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    c, v = d["close"], d["volume"]
    d["nhist"] = np.arange(1, len(d) + 1)
    d["adv30"] = (c * v).rolling(30).mean()
    tr = pd.concat([d["high"] - d["low"], (d["high"] - c.shift()).abs(),
                    (d["low"] - c.shift()).abs()], axis=1).max(axis=1)
    d["atr_pct"] = tr.rolling(14).mean() / c
    o1 = d["open"].shift(-1)
    hmax = d["high"][::-1].rolling(10, min_periods=10).max()[::-1].shift(-1)
    d["mfe10"] = hmax / o1 - 1
    d["fwd20"] = c.shift(-HOLD) / o1 - 1
    d["ret20_back"] = c / c.shift(20) - 1
    d["tail"] = d["mfe10"] >= EVENT_R * d["atr_pct"]
    d["ca_mask"] = mask_windows(d, HOLD + 1).to_numpy()
    d["ticker"] = path.stem
    return d


def weekly_hold(d: pd.DataFrame, col: str) -> pd.Series:
    """Evaluate `col` on the last session of each week, apply it to the NEXT week."""
    wk = d["date"].dt.to_period("W-SUN")
    last = d.groupby(wk)[col].last().shift(1)
    return wk.map(last).astype("boolean").fillna(False).astype(bool)


def main() -> None:
    frames = []
    files = sorted(BARS.glob("*.parquet"))
    for i, p in enumerate(files, 1):
        d = one(p)
        if d is None:
            continue
        ok_px = d["close"].between(MIN_PX, MAX_PX)
        d["r_hist"] = d["nhist"] >= MIN_HIST
        d["r_px"] = ok_px
        d["r_adv"] = d["adv30"] >= MIN_ADV
        d["elig_daily"] = d["r_hist"] & d["r_px"] & d["r_adv"]
        d["elig"] = weekly_hold(d, "elig_daily")
        for a in ALT_ADV:
            d[f"elig_adv{int(a / 1e6)}m"] = weekly_hold(
                d.assign(_e=d["r_hist"] & ok_px & (d["adv30"] >= a)), "_e")
        for h in ALT_HIST:
            d[f"elig_hist{h}"] = weekly_hold(
                d.assign(_e=(d["nhist"] >= h) & ok_px & d["r_adv"]), "_e")
        # status of the rule that blocks, as of the weekly rebuild (priority: hist, px, adv)
        for r in ("r_hist", "r_px", "r_adv"):
            d[r + "_w"] = weekly_hold(d, r)
        frames.append(d[(d["date"] >= W0) & (d["date"] <= W1) & ~d["ca_mask"]
                        & d["mfe10"].notna() & d["atr_pct"].notna()])
        if i % 500 == 0:
            print(f"  {i}/{len(files)}", flush=True)
    x = pd.concat(frames, ignore_index=True)
    x["status"] = np.select(
        [x["elig"], ~x["r_hist_w"], ~x["r_px_w"], ~x["r_adv_w"]],
        ["eligible", "blocked_history", "blocked_price", "blocked_adv"], "blocked_other")
    out: dict = {"rows": len(x), "tickers": int(x["ticker"].nunique()),
                 "window": [str(W0.date()), str(W1.date())]}

    # 1) where do tail events sit?
    t = (x.groupby("status").agg(rows=("tail", "size"), events=("tail", "sum"),
                                 med_adv=("adv30", "median"), med_fwd20=("fwd20", "median"))
         .assign(event_rate=lambda g: g["events"] / g["rows"],
                 share_of_events=lambda g: g["events"] / g["events"].sum()))
    base = t.loc["eligible", "event_rate"]
    t["lift_vs_eligible"] = t["event_rate"] / base
    out["by_status"] = t.round(5).reset_index().to_dict("records")
    print(t.round(4).to_string())

    # 2) ADV-blocked tails: how illiquid were they when the move started?
    adv_b = x[(x["status"] == "blocked_adv") & x["tail"]]
    bins = [0, 5e5, 1e6, 2e6, 3e6, 5e6]
    out["adv_blocked_tail_by_adv"] = (pd.cut(adv_b["adv30"], bins).value_counts(sort=False)
                                      .rename(str).to_dict())
    out["adv_blocked_rows_by_adv"] = (pd.cut(x.loc[x["status"] == "blocked_adv", "adv30"], bins)
                                      .value_counts(sort=False).rename(str).to_dict())

    # 3) admission episodes: flips into eligibility after >= 20 ineligible sessions
    x = x.sort_values(["ticker", "date"])
    prev_out = (x.groupby("ticker")["elig"]
                .transform(lambda s: (~s).rolling(20, min_periods=20).sum().shift(1)))
    adm = x[x["elig"] & (prev_out == 20)].copy()
    adm["blocked_by_before"] = np.where(
        x.groupby("ticker")["r_hist_w"].shift(5).loc[adm.index].eq(False), "history",
        np.where(x.groupby("ticker")["r_adv_w"].shift(5).loc[adm.index].eq(False), "adv", "other"))
    elig_day = x[x["elig"]].groupby("date").agg(b_fwd20=("fwd20", "median"),
                                                 b_back=("ret20_back", "median"),
                                                 b_tail=("tail", "mean"))
    adm = adm.join(elig_day, on="date")
    a = (adm.groupby("blocked_by_before")
         .agg(n=("ticker", "size"),
              med_ret20_before=("ret20_back", "median"), base_ret20_before=("b_back", "median"),
              med_fwd20_after=("fwd20", "median"), base_fwd20_after=("b_fwd20", "median"),
              mean_fwd20_after=("fwd20", "mean"),
              tail_rate_after=("tail", "mean"), base_tail=("b_tail", "mean")))
    out["admissions"] = a.round(4).reset_index().to_dict("records")
    print("\nadmission episodes\n", a.round(4).to_string())

    # 4) counterfactual floors: what would a looser rule add?
    cf = {}
    for col in [f"elig_adv{int(a_ / 1e6)}m" for a_ in ALT_ADV] + [f"elig_hist{h}" for h in ALT_HIST]:
        add = x[x[col] & ~x["elig"]]
        cf[col] = {"added_rows": len(add), "added_events": int(add["tail"].sum()),
                   "added_event_rate": float(add["tail"].mean()) if len(add) else None,
                   "lift_vs_eligible": float(add["tail"].mean() / base) if len(add) else None,
                   "added_med_fwd20": float(add["fwd20"].median()) if len(add) else None,
                   "added_mean_fwd20": float(add["fwd20"].mean()) if len(add) else None,
                   "added_med_adv": float(add["adv30"].median()) if len(add) else None,
                   "share_of_all_blocked_events": float(add["tail"].sum() / max(1, x.loc[~x["elig"], "tail"].sum()))}
    out["counterfactual"] = cf
    out["eligible_med_fwd20"] = float(x.loc[x["elig"], "fwd20"].median())
    out["eligible_mean_fwd20"] = float(x.loc[x["elig"], "fwd20"].mean())
    out["eligible_event_rate"] = float(base)
    print("\ncounterfactual\n", pd.DataFrame(cf).T.round(4).to_string())

    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "pool_admission.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {HERE / 'data' / 'pool_admission.json'}")


if __name__ == "__main__":
    main()
