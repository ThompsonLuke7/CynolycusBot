"""Stage 1 -- right-tail RETRIEVAL audit. No model is fitted here.

Question (thread's Experiment A): of the extreme forward moves that actually happen, what
share does CynolycusBot's funnel surface BEFORE they happen, and does loosening the funnel
buy real retrieval or only a higher firing rate?

Two layers, because they fail for different reasons (00_preregistration.md section 4):

  L1  UNIVERSE   is the name in the momentum pool at all?   population = the daily cache
  L2  GATE       among pool rows, does momentum_candidate_mask fire?

The headline is LIFT (precision / base rate), never recall: a rule firing on X% of rows
catches X% of events for free. `recall - fires` is printed next to it.

Window: matrix start .. 2025-08-07 (through VALIDATION end). The test block
(2025-09-26 onward) is never read here.

Events are frozen in 00_preregistration.md section 5.1:
  P1  rmfe_10 >= 4R     base 1.44%     (primary)
  P2  mfe_10  >= 25%    base 4.07%
  P3  rmfe_20 top 1% within decision bar   base 0.95%
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import panel as P  # noqa: E402
from strategies.momentum_expansion.config.momentum_config import (  # noqa: E402
    MOMENTUM_CANDIDATE_FILTER_CONFIG as GATE_CFG,
)
from strategies.momentum_expansion.inference.candidate_filter import (  # noqa: E402
    momentum_candidate_mask,
)

BARS_1D = REPO / "Data/shared/bars/1d"
MATRIX = REPO / "strategies/momentum_expansion/data/processed/training_matrix_4h.parquet"
FLAGS = REPO / "research/execution_quality/data/corporate_action_flags.parquet"
SHARED_UNIVERSE = REPO / "Data/shared/universe/shared_universe.csv"
DATA = HERE / "data"
POP = DATA / "stage1_population.parquet"

WINDOW_END = pd.Timestamp("2025-08-07")   # validation end; test is never touched
# L2 is bounded below by the matrix's own start (2020-09-08). L1 uses everything the daily
# cache has before WINDOW_END, which reaches back to 2018; the two windows are reported
# separately rather than forced to match, and L1 is re-reported on the L2 window as a check.
L2_WINDOW_START = pd.Timestamp("2020-09-08")
HOLDS = [10, 20]
LIQ_FLOORS = [0.0, 1e6, 5e6, 25e6]
GATE_KEYS = ["exclude_low_price", "min_dollar_vol_pctile_252", "max_dist_to_52w_high_atr",
             "min_xsec_near_high_rank", "min_rs_spy_20", "min_xsec_ret_20_rank",
             "min_range_pos_20"]
# Fully permissive value for each key: the setting that cannot exclude anything.
GATE_OPEN = {"exclude_low_price": False, "min_dollar_vol_pctile_252": 0.0,
             "max_dist_to_52w_high_atr": 1e9, "min_xsec_near_high_rank": 0.0,
             "min_rs_spy_20": -1e9, "min_xsec_ret_20_rank": 0.0, "min_range_pos_20": 0.0}


# ------------------------------------------------------------------ population

def _ticker_rows(ticker: str) -> pd.DataFrame | None:
    """One row per session: entry at THIS session's open, forward MFE from it.

    Column-trimmed restatement of build_decision_panel.ticker_frame's forward block and
    panel._ticker_atr -- same formulas, but this runs over 4,104 tickers instead of 1,080
    and the panel's 57 columns would be ~6M x 57 in memory (the box has a prior OOM on
    record). Everything here is either strictly prior to the open, or strictly forward.
    """
    path = BARS_1D / f"{ticker}.parquet"
    if not path.exists():
        return None
    d = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if len(d) < 60 + max(HOLDS):
        return None
    sess = (pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert("America/New_York")
            .dt.normalize().dt.tz_localize(None))
    d = d.assign(session_date=sess).sort_values("session_date").reset_index(drop=True)
    o, h, lo, c, v = (d[x].to_numpy(float) for x in ("open", "high", "low", "close", "volume"))
    pc = pd.Series(c).shift(1)
    tr = pd.concat([pd.Series(h) - pd.Series(lo), (pd.Series(h) - pc).abs(),
                    (pd.Series(lo) - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / P.ATR_WINDOW, adjust=False).mean()
    out = {"ticker": ticker, "session_date": d["session_date"].to_numpy(),
           "atr_pct": (atr / pd.Series(c).replace(0, np.nan)).shift(1).to_numpy(),
           "dollar_vol_20": pd.Series(c * v).shift(1).rolling(20).mean().to_numpy(),
           "px_prior": pc.to_numpy()}
    with np.errstate(divide="ignore", invalid="ignore"):
        for n in HOLDS:
            fmax = pd.Series(h).rolling(n).max().shift(-(n - 1)).to_numpy()
            out[f"mfe_{n}"] = np.where(o > 0, fmax / o - 1.0, np.nan)
    f = pd.DataFrame(out)
    return f[f["session_date"] <= WINDOW_END + pd.Timedelta(days=5)]


def build_population() -> pd.DataFrame:
    files = sorted(BARS_1D.glob("*.parquet"))
    frames = []
    for i, p in enumerate(files, 1):
        f = _ticker_rows(p.stem)
        if f is not None:
            frames.append(f)
        if i % 500 == 0:
            print(f"  population {i}/{len(files)}", flush=True)
    pop = pd.concat(frames, ignore_index=True)
    pop = pop[pop["session_date"] <= WINDOW_END].dropna(subset=["atr_pct"])
    unit = (P.K_RISK * pop["atr_pct"]).where(pop["atr_pct"] >= P.MIN_ATR_PCT)
    for n in HOLDS:
        pop[f"rmfe_{n}"] = pop[f"mfe_{n}"] / unit
    # corporate-action guard: same forward span logic as build_decision_panel
    flags = pd.read_parquet(FLAGS)
    flags = flags[~flags["organic"]]
    span = np.timedelta64(int(np.ceil(max(HOLDS) * 7 / 5)) + 4, "D")
    bad = np.zeros(len(pop), bool)
    day, tick = pop["session_date"].to_numpy(), pop["ticker"].to_numpy()
    for t, gg in flags.groupby("ticker"):
        dates = np.sort(pd.to_datetime(gg["date"]).to_numpy())
        msk = tick == t
        if msk.any():
            st = day[msk]
            bad[msk] = (np.searchsorted(dates, st + span, side="right")
                        > np.searchsorted(dates, st, side="left"))
    pop["ca_flagged"] = bad
    for c in pop.select_dtypes("float64").columns:
        pop[c] = pop[c].astype("float32")
    DATA.mkdir(exist_ok=True)
    pop.to_parquet(POP, index=False)
    return pop


# ---------------------------------------------------------------------- events

# each event needs its own validity mask: a NaN outcome must be EXCLUDED, not silently
# counted as a non-event (that would deflate every base rate).
EVENT_SOURCE = {"P1": "rmfe_10", "P2": "mfe_10", "P3": "rmfe_20"}


def add_events(pop: pd.DataFrame) -> pd.DataFrame:
    pop["P1"] = pop["rmfe_10"] >= 4.0
    pop["P2"] = pop["mfe_10"] >= 0.25
    rank = pop.groupby("session_date")["rmfe_20"].rank(pct=True, ascending=False)
    pop["P3"] = rank <= 0.01
    for ev, src in EVENT_SOURCE.items():
        pop[f"{ev}_valid"] = pop[src].notna()
    return pop


def score(pop: pd.DataFrame, fire: pd.Series, label: str, event: str) -> dict:
    ok = pop[f"{event}_valid"].to_numpy(bool) & fire.notna().to_numpy(bool)
    ev = pop[event].to_numpy(bool)[ok]
    fr = fire.to_numpy(bool)[ok]
    n = len(ev)
    base = ev.mean() if n else np.nan
    fires = fr.mean() if n else np.nan
    recall = fr[ev].mean() if ev.any() else np.nan
    prec = ev[fr].mean() if fr.any() else np.nan
    return dict(config=label, event=event, n=int(n), events=int(ev.sum()),
                base_rate=float(base), fires=float(fires), recall=float(recall),
                precision=float(prec), lift=float(prec / base) if base else np.nan,
                excess_pp=float((recall - fires) * 100))


def table(rows: list[dict], title: str) -> None:
    df = pd.DataFrame(rows)
    print(f"\n-- {title} --")
    for e, g in df.groupby("event", sort=True):
        print(f"\n  event {e}  (n={g['n'].iloc[0]:,}  events={g['events'].iloc[0]:,}  "
              f"base={g['base_rate'].iloc[0]:.3%})")
        s = g[["config", "fires", "recall", "precision", "lift", "excess_pp"]].copy()
        for c in ("fires", "recall", "precision"):
            s[c] = (s[c] * 100).round(2)
        s["lift"] = s["lift"].round(3)
        s["excess_pp"] = s["excess_pp"].round(2)
        print(s.to_string(index=False))


# ------------------------------------------------------------------------ main

def main() -> None:
    out: dict = {}
    if POP.exists():
        print(f"reading cached population {POP}")
        pop = pd.read_parquet(POP)
    else:
        print("building population over the full daily cache ...", flush=True)
        pop = build_population()
    pop = add_events(pop)

    pool_matrix = set(pd.read_parquet(MATRIX, columns=["atr_pct_14"])
                      .reset_index()["ticker"].unique())
    su = pd.read_csv(SHARED_UNIVERSE)
    pool_shared = set(su.loc[su["in_momentum_candidate"].fillna(False).astype(bool), "ticker"])

    guarded = pop[~pop["ca_flagged"]]
    out["population"] = {
        "rows_raw": int(len(pop)), "rows_guarded": int(len(guarded)),
        "ca_flagged_share": float(pop["ca_flagged"].mean()),
        "tickers": int(pop["ticker"].nunique()),
        "span": [str(pop["session_date"].min().date()), str(pop["session_date"].max().date())],
        "pool_matrix_tickers": len(pool_matrix), "pool_shared_tickers": len(pool_shared),
        "pool_overlap": len(pool_matrix & pool_shared),
    }
    print("\n" + "=" * 78 + "\nSTAGE 1 POPULATION\n" + "=" * 78)
    print(json.dumps(out["population"], indent=2))

    # ---------------------------------------------------------------- L1
    print("\n" + "=" * 78 + "\nL1 -- UNIVERSE retrieval (population = daily cache)\n" + "=" * 78)
    print("BIAS WARNING: pool membership is TODAY's list; no point-in-time universe exists\n"
          "before 2026-09-10 (core/shared_universe/universe.py:236). Today's pool includes\n"
          "names partly selected BECAUSE they later did well, so this is biased IN THE\n"
          "POOL'S FAVOUR. A poor result here is informative; a good one is not.\n")
    l1: list[dict] = []
    for floor in LIQ_FLOORS:
        sub = guarded[guarded["dollar_vol_20"].fillna(0) >= floor]
        in_m = sub["ticker"].isin(pool_matrix)
        in_s = sub["ticker"].isin(pool_shared)
        for ev in ("P1", "P2", "P3"):
            l1.append(score(sub, in_m, f"matrix pool, liq>={floor:,.0f}", ev))
            l1.append(score(sub, in_s, f"shared universe, liq>={floor:,.0f}", ev))
    out["L1"] = l1
    table(l1, "L1 universe retrieval")

    print("\n-- are the events OUTSIDE the pool tradeable? (guarded, no liq floor) --")
    comp = []
    for ev in ("P1", "P2", "P3"):
        e = guarded[guarded[ev]]
        for name, m in (("in pool", e["ticker"].isin(pool_matrix)),
                        ("outside pool", ~e["ticker"].isin(pool_matrix))):
            g = e[m]
            comp.append(dict(event=ev, group=name, n=int(len(g)),
                             med_dollar_vol=float(g["dollar_vol_20"].median()),
                             med_px=float(g["px_prior"].median()),
                             med_atr_pct=float(g["atr_pct"].median()),
                             share_dv_over_5m=float((g["dollar_vol_20"] >= 5e6).mean())))
    out["L1_composition"] = comp
    cdf = pd.DataFrame(comp)
    cdf["med_dollar_vol"] = cdf["med_dollar_vol"].round(0)
    print(cdf.to_string(index=False))

    # ---------------------------------------------------------------- L2
    print("\n" + "=" * 78 + "\nL2 -- candidate GATE retrieval (population = pool rows)\n" + "=" * 78)
    # the FEATURE columns momentum_candidate_mask reads -- not the CONFIG key names
    gcols = ["low_price_flag", "dollar_vol_pctile_252", "dist_to_52w_high_atr",
             "xsec_near_high_rank", "rs_spy_20", "xsec_ret_20_rank", "range_pos_20"]
    g4h = pd.read_parquet(MATRIX, columns=gcols).reset_index()
    g4h["decision_day"] = (g4h["timestamp"].dt.tz_convert("America/New_York")
                           .dt.normalize().dt.tz_localize(None))
    g4h = g4h[g4h["decision_day"] <= WINDOW_END]
    print(f"gate rows (4H) {len(g4h):,} over {g4h['ticker'].nunique()} tickers "
          f"{g4h['decision_day'].min().date()} .. {g4h['decision_day'].max().date()}")

    configs: list[tuple[str, dict]] = [("CURRENT (live)", {})]
    for k in GATE_KEYS:
        configs.append((f"open: {k}", {k: GATE_OPEN[k]}))
    for frac in (0.25, 0.5, 0.75):
        step = {}
        for k in GATE_KEYS:
            cur, opn = GATE_CFG[k], GATE_OPEN[k]
            if isinstance(cur, bool):
                step[k] = cur if frac < 0.75 else opn
            elif k == "max_dist_to_52w_high_atr":
                step[k] = cur * (1.0 + 3.0 * frac)
            elif k == "min_rs_spy_20":
                step[k] = cur - 0.5 * frac
            else:
                step[k] = cur * (1.0 - frac)
        configs.append((f"joint ladder {frac:.0%} open", step))
    configs.append(("FULLY OPEN", dict(GATE_OPEN)))

    # one fire-flag per (ticker, decision_day): the gate fires if ANY 4H bar of the day
    # fires, which is how the live runner behaves (a bar fires and the module acts).
    fire_by_cfg: dict[str, pd.DataFrame] = {}
    for label, override in configs:
        m = momentum_candidate_mask(g4h, cfg=override)
        agg = (g4h.assign(_f=m.to_numpy()).groupby(["ticker", "decision_day"])["_f"]
               .any().rename(label).reset_index())
        fire_by_cfg[label] = agg

    # L2 must be bounded BELOW by the matrix's own start, or rows that simply predate the
    # gate's existence are miscounted as "the gate did not evaluate them".
    pool_rows = guarded[guarded["ticker"].isin(pool_matrix)
                        & (guarded["session_date"] >= L2_WINDOW_START)].copy()
    pool_rows = pool_rows.sort_values("session_date")
    print(f"\nL2 window {L2_WINDOW_START.date()} .. {WINDOW_END.date()} "
          f"(the matrix's own span); pool rows in window {len(pool_rows):,}")

    def gate_join(agg: pd.DataFrame, col: str) -> pd.Series:
        """Map a gate decision on day D to the entry session strictly after D.

        Returns a Series aligned to pool_rows: True = fired, False = evaluated and
        rejected, NaN = the gate had NO 4H row for that (ticker, day) at all. The
        third case must stay distinct: "the module was not looking" is a different
        retrieval loss from "the module looked and said no", and collapsing them
        into False would silently credit the gate with rows it never saw.
        """
        a = agg.rename(columns={col: "fire"}).copy()
        a["_key"] = a["decision_day"] + pd.Timedelta(days=1)
        j = pd.merge_asof(a.sort_values("_key"),
                          pool_rows[["ticker", "session_date"]].sort_values("session_date"),
                          left_on="_key", right_on="session_date", by="ticker",
                          direction="forward", tolerance=pd.Timedelta(days=7))
        j = j.dropna(subset=["session_date"])
        # several decision days can land on one session (holidays); fired if any did
        j = j.groupby(["ticker", "session_date"])["fire"].any()
        idx = pd.MultiIndex.from_arrays([pool_rows["ticker"].to_numpy(),
                                         pool_rows["session_date"].to_numpy()])
        return pd.Series(j.reindex(idx).to_numpy(), index=pool_rows.index, dtype="object")

    base_fire = gate_join(fire_by_cfg["CURRENT (live)"], "CURRENT (live)")
    observed = base_fire.notna()
    print(f"\npool rows {len(pool_rows):,}; gate evaluated on {observed.sum():,} "
          f"({observed.mean():.2%}). The remainder had no 4H matrix row for that "
          f"(ticker, day) and are reported as COVERAGE loss, not as a gate rejection.")
    for ev in ("P1", "P2", "P3"):
        v = pool_rows[f"{ev}_valid"].to_numpy(bool)
        e = pool_rows[ev].to_numpy(bool) & v
        seen = observed.to_numpy(bool)
        print(f"  {ev}: {e.sum():,} events in pool; {(e & seen).sum():,} "
              f"({(e & seen).sum() / max(1, e.sum()):.2%}) on a day the gate evaluated")
    out["L2_coverage"] = {"pool_rows": int(len(pool_rows)),
                          "gate_evaluated": int(observed.sum()),
                          "gate_evaluated_share": float(observed.mean())}

    l2: list[dict] = []
    obs_rows = pool_rows[observed.to_numpy(bool)]
    for label, agg in fire_by_cfg.items():
        fire = gate_join(agg, label)[observed.to_numpy(bool)].astype(bool)
        for ev in ("P1", "P2", "P3"):
            l2.append(score(obs_rows, fire, label, ev))
    out["L2"] = l2
    table(l2, "L2 gate retrieval (pool rows where the gate was evaluated)")

    (DATA / "stage1_retrieval.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {DATA / 'stage1_retrieval.json'}")


if __name__ == "__main__":
    main()
