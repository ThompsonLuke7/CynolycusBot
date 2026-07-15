"""Cross-signal confluence search (research only — nothing here touches live modules).

Mines 2-signal (and optionally 3-signal) interactions across ORTHOGONAL signal axes
(technical / theme / news-catalyst / calendar / short-flow / regime) on the
point-in-time dataset built by build_dataset.py, under the house 60/20/20
temporal-split discipline:

  * condition thresholds are fit on TRAIN only,
  * mining + significance run on TRAIN (monthly-block t-test, BH-FDR),
  * selection requires VAL replication,
  * TEST is touched exactly once, via the separate `confirm` stage on an explicit
    shortlist file. Never iterate against `confirm` output.

Statistics are deliberately conservative for overlapping 25-bar forward windows:
significance uses per-month deltas (block structure), and sample size is reported
both as raw rows and as de-overlapped "episodes" (contiguous per-ticker runs).

Interaction (confluence) is distinguished from additive stacking:
  * delta_pp        — precision of joint cell minus BEST single marginal (practical gain)
  * log_odds_inter  — logit(pJ) - [logit(pA) + logit(pB) - logit(base)]  (> 0 means
                      super-additive, i.e. a genuine interaction, not just stacking)

Usage:
  .venv/bin/python scripts/confluence_discovery/search.py mine \
      --target meta_upside [--min-liq 0.4] [--triples] [--out-dir research/confluence]
  .venv/bin/python scripts/confluence_discovery/search.py confirm \
      --target meta_upside --shortlist research/confluence/shortlist_meta_upside.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "research/confluence/confluence_dataset.parquet"

OUTCOME_COLS = ["fwd_max_return", "fwd_max_alpha", "fwd_close_return", "fwd_max_drawdown",
                "trade_quality", "meta_good", "meta_upside"]
FORWARD_BANNED = ["trend_persistence", "earnings_in_fwd_window", "meta_label"]

TRAIN_FRAC, VAL_FRAC = 0.6, 0.2
EPISODE_GAP_BARS = pd.Timedelta(days=3)   # new episode if per-ticker gap exceeds this

# ---------------------------------------------------------------------------
# Condition library. Each entry: name -> (axis, spec) where spec is either
#   ("expr", fn(df) -> bool Series)                      fixed rule, no fitting
#   ("qhi", column, q)   value >= train-quantile(q)      fit on TRAIN only
#   ("qlo", column, q)   value <= train-quantile(q)      fit on TRAIN only
# NaN always evaluates to False.
# ---------------------------------------------------------------------------
CONDITIONS: dict[str, tuple[str, tuple]] = {
    # --- technical (setup) ---
    "mom_top10": ("tech", ("expr", lambda d: d["mom_xs_rank"] >= 0.90)),
    "mom_top20": ("tech", ("expr", lambda d: d["mom_xs_rank"] >= 0.80)),
    "htf_top20": ("tech", ("expr", lambda d: d["htf_xs_rank"] >= 0.80)),
    "mom_htf_agree": ("tech", ("expr", lambda d: (d["mom_xs_rank"] >= 0.80) & (d["htf_xs_rank"] >= 0.80))),
    "near_52w_high": ("tech", ("expr", lambda d: d["near_52w_high"] == 1)),
    "breakout20": ("tech", ("expr", lambda d: d["breakout_20"] == 1)),
    "compressed": ("tech", ("expr", lambda d: d["is_compressed_5_20"] == 1)),
    "rvol_hot": ("tech", ("qhi", "rvol_20", 0.80)),
    "atr_expanding": ("tech", ("qhi", "atr_expand_14_60", 0.80)),
    "volume_spike": ("tech", ("expr", lambda d: d["volume_spike_20"] == 1)),
    # --- theme rotation state ---
    "theme_hot": ("theme", ("qhi", "theme_heat_score", 0.80)),
    "theme_cold": ("theme", ("qlo", "theme_heat_score", 0.20)),
    "theme_rank_top10": ("theme", ("expr", lambda d: d["primary_theme_rank"] <= 10)),
    "theme_accel": ("theme", ("qhi", "theme_acceleration", 0.80)),
    "theme_new": ("theme", ("qhi", "theme_newness_score", 0.80)),
    "theme_broad": ("theme", ("qhi", "theme_breadth", 0.80)),
    "theme_crowded": ("theme", ("qhi", "theme_crowding_frac", 0.80)),
    # --- news / catalyst ---
    "news_hot": ("news", ("qhi", "news_catalyst_score", 0.90)),
    "news_any": ("news", ("expr", lambda d: d["news_catalyst_count"].fillna(0) > 0)),
    "news_none": ("news", ("expr", lambda d: d["news_catalyst_count"].fillna(0) == 0)),
    "news_breaking": ("news", ("expr", lambda d: d["news_breaking_count"].fillna(0) > 0)),
    "news_bull": ("news", ("qhi", "news_bull_alignment", 0.80)),
    "guidance_pos": ("news", ("qhi", "fg_guidance_score", 0.80)),
    # --- calendar ---
    "pre_earnings3d": ("calendar", ("expr", lambda d: d["is_pre_earnings_3d"] == 1)),
    "post_earnings3d": ("calendar", ("expr", lambda d: d["is_post_earnings_3d"] == 1)),
    "earnings_far": ("calendar", ("expr", lambda d: d["days_to_earnings"] > 15)),
    "macro_next3d": ("calendar", ("expr", lambda d: d["macro_high_impact_next_3d_count"] >= 1)),
    # --- short flow (FINRA daily short-sale volume) ---
    "short_spike": ("flow", ("expr", lambda d: d["short_ratio_z20"] >= 1.5)),
    "short_low": ("flow", ("expr", lambda d: d["short_ratio_z20"] <= -1.5)),
    "short_pct_high": ("flow", ("expr", lambda d: d["short_ratio_pct63"] >= 0.90)),
    "short_pct_low": ("flow", ("expr", lambda d: d["short_ratio_pct63"] <= 0.10)),
    "short_vol_surge": ("flow", ("qhi", "short_vol_surge20", 0.90)),
    # --- market regime (conditioners) ---
    "spy_uptrend": ("regime", ("expr", lambda d: d["regime_spy_trend"] > 0)),
    "spy_downtrend": ("regime", ("expr", lambda d: d["regime_spy_trend"] < 0)),
    "vix_calm": ("regime", ("expr", lambda d: d["regime_vix_z"] <= 0)),
    "vix_stress": ("regime", ("expr", lambda d: d["regime_vix_z"] >= 1)),
    "rates_up5d": ("regime", ("qhi", "treasury_10y_change_5d", 0.80)),
    "rates_down5d": ("regime", ("qlo", "treasury_10y_change_5d", 0.20)),
}

MIN_MONTH_ROWS = 10          # month contributes to the block test only above this
LOGIT_CLIP = 1e-4


def logit(p: np.ndarray | float) -> np.ndarray | float:
    p = np.clip(p, LOGIT_CLIP, 1 - LOGIT_CLIP)
    return np.log(p / (1 - p))


def load_universe(target: str, min_liq: float, include_etfs: bool) -> pd.DataFrame:
    df = pd.read_parquet(DATASET).reset_index()
    df = df.drop(columns=[c for c in FORWARD_BANNED if c in df.columns])
    n0 = len(df)
    df = df[df[target].notna() & df["fwd_close_return"].notna()]
    df = df[df["dollar_vol_pctile_252"].fillna(0) >= min_liq]
    df = df[df["low_price_flag"].fillna(1) == 0]
    if not include_etfs:
        df = df[df["is_etf"].fillna(0) == 0]
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)
    print(f"universe: {len(df):,}/{n0:,} rows after label/liquidity>= {min_liq}/price/etf filters; "
          f"{df['ticker'].nunique()} tickers, {df['timestamp'].min().date()} .. {df['timestamp'].max().date()}")
    return df


def temporal_split(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """House convention (family_backtest.compute_test_cutoff): row-fraction cutoffs
    over timestamp-sorted label-valid rows."""
    ts = df["timestamp"]
    val_cut = ts.iloc[int(len(ts) * TRAIN_FRAC)]
    test_cut = ts.iloc[int(len(ts) * (TRAIN_FRAC + VAL_FRAC))]
    return val_cut, test_cut


def build_condition_masks(df: pd.DataFrame, train_mask: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Boolean matrix of all conditions. Quantile thresholds are fit on TRAIN only."""
    masks = {}
    fitted = {}
    tr = df[train_mask]
    for name, (axis, spec) in CONDITIONS.items():
        kind = spec[0]
        if kind == "expr":
            s = spec[1](df)
        else:
            _, col, q = spec
            thr = float(tr[col].quantile(q))
            fitted[name] = {"col": col, "q": q, "thr": thr}
            s = (df[col] >= thr) if kind == "qhi" else (df[col] <= thr)
        masks[name] = s.fillna(False).to_numpy(dtype=bool)
    out = pd.DataFrame(masks, index=df.index)
    # Drop degenerate conditions (e.g. a train quantile that collapses to the min value
    # and selects ~everything): they add no information and only inflate the test count.
    tr_cov = out[train_mask.to_numpy()].mean()
    degenerate = tr_cov[(tr_cov > 0.90) | (tr_cov == 0.0)].index.tolist()
    if degenerate:
        print(f"dropping degenerate conditions (train coverage >90% or 0): {degenerate}")
        out = out.drop(columns=degenerate)
    return out, fitted


def n_episodes(sub: pd.DataFrame) -> int:
    """De-overlapped sample size: contiguous per-ticker selection runs."""
    if sub.empty:
        return 0
    s = sub.sort_values(["ticker", "timestamp"])
    gap = s.groupby("ticker")["timestamp"].diff()
    return int(((gap.isna()) | (gap > EPISODE_GAP_BARS)).sum())


def cell_stats(df: pd.DataFrame, mask: np.ndarray, target: str) -> dict:
    sub = df[mask]
    if sub.empty:
        return {"n": 0}
    return {
        "n": len(sub),
        "n_episodes": n_episodes(sub),
        "n_days": sub["timestamp"].dt.date.nunique(),
        "p": sub[target].mean(),
        "mean_alpha": sub["fwd_max_alpha"].mean(),
        "mean_ret": sub["fwd_close_return"].mean(),
        "win_rate": (sub["fwd_close_return"] > 0).mean(),
        "mean_dd": sub["fwd_max_drawdown"].mean(),
    }


def monthly_block_test(df: pd.DataFrame, joint: np.ndarray, mA: np.ndarray, mB: np.ndarray | None,
                       target: str) -> tuple[float, float, float, int]:
    """Per-month delta of joint precision vs best marginal; one-sided t-test.

    Returns (t, p_one_sided, pos_frac, n_months).
    """
    y = df[target].to_numpy()
    months = df["month"].to_numpy()
    deltas = []
    for m in np.unique(months):
        in_m = months == m
        jm = joint & in_m
        if jm.sum() < MIN_MONTH_ROWS:
            continue
        pj = y[jm].mean()
        pa = y[mA & in_m].mean() if (mA & in_m).sum() >= MIN_MONTH_ROWS else np.nan
        pb = y[mB & in_m].mean() if mB is not None and (mB & in_m).sum() >= MIN_MONTH_ROWS else np.nan
        best = np.nanmax([pa, pb])
        if np.isnan(best):
            continue
        deltas.append(pj - best)
    deltas = np.asarray(deltas, dtype=float)
    if len(deltas) < 4 or deltas.std(ddof=1) == 0:
        return np.nan, np.nan, np.nan, len(deltas)
    t, p_two = sstats.ttest_1samp(deltas, 0.0)
    p_one = p_two / 2 if t > 0 else 1 - p_two / 2
    return float(t), float(p_one), float((deltas > 0).mean()), len(deltas)


def eval_combo(df: pd.DataFrame, target: str, joint: np.ndarray,
               marg_masks: list[np.ndarray], base_p: float) -> dict:
    """Full stat block for a joint cell vs its marginals on one split."""
    st_j = cell_stats(df, joint, target)
    if st_j["n"] == 0:
        return {"n": 0}
    marg_ps, marg_alphas = [], []
    for mm in marg_masks:
        stm = cell_stats(df, mm, target)
        marg_ps.append(stm.get("p", np.nan))
        marg_alphas.append(stm.get("mean_alpha", np.nan))
    best_marg = np.nanmax(marg_ps) if marg_ps else np.nan
    best_marg_alpha = np.nanmax(marg_alphas) if marg_alphas else np.nan
    out = {
        "n": st_j["n"], "n_episodes": st_j["n_episodes"], "n_days": st_j["n_days"],
        "p_joint": st_j["p"], "p_base": base_p,
        "lift": st_j["p"] / base_p if base_p > 0 else np.nan,
        "best_marginal_p": best_marg,
        "delta_pp": (st_j["p"] - best_marg) * 100,
        "mean_alpha": st_j["mean_alpha"],
        "best_marginal_alpha": best_marg_alpha,
        "delta_alpha": st_j["mean_alpha"] - best_marg_alpha,
        "mean_ret": st_j["mean_ret"],
        "win_rate": st_j["win_rate"], "mean_dd": st_j["mean_dd"],
    }
    if len(marg_masks) == 2:
        out["log_odds_inter"] = float(
            logit(st_j["p"]) - (logit(marg_ps[0]) + logit(marg_ps[1]) - logit(base_p)))
    return out


def bh_fdr(p: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted q-values (NaN-safe)."""
    q = pd.Series(np.nan, index=p.index)
    valid = p.dropna().sort_values()
    n = len(valid)
    if n == 0:
        return q
    ranked = valid.rank(method="first")
    adj = (valid * n / ranked)[::-1].cummin()[::-1]
    q.loc[adj.index] = adj.clip(upper=1.0)
    return q


# ---------------------------------------------------------------------------
def stage_mine(args) -> None:
    df = load_universe(args.target, args.min_liq, args.include_etfs)
    val_cut, test_cut = temporal_split(df)
    print(f"split cutoffs: val={val_cut}  test={test_cut}  (TEST NOT TOUCHED in mine stage)")
    is_train = (df["timestamp"] < val_cut).to_numpy()
    is_val = ((df["timestamp"] >= val_cut) & (df["timestamp"] < test_cut)).to_numpy()

    masks, fitted = build_condition_masks(df, pd.Series(is_train, index=df.index))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"fitted_thresholds_{args.target}.json").write_text(json.dumps(
        {"val_cut": str(val_cut), "test_cut": str(test_cut), "min_liq": args.min_liq,
         "target": args.target, "thresholds": fitted}, indent=2, default=str))

    tr, va = df[is_train], df[is_val]
    base_tr, base_va = tr[args.target].mean(), va[args.target].mean()
    print(f"train: {len(tr):,} rows {tr['month'].nunique()} months base={base_tr:.4f} | "
          f"val: {len(va):,} rows {va['month'].nunique()} months base={base_va:.4f}")

    # Leakage tripwire: any condition column suspiciously correlated with the outcome.
    print("leakage tripwire (train |corr| with fwd_max_alpha > 0.15):")
    checked = set()
    for name, (_, spec) in CONDITIONS.items():
        col = spec[1] if spec[0] != "expr" else None
        if col and col not in checked:
            checked.add(col)
            c = tr[[col, "fwd_max_alpha"]].corr().iloc[0, 1]
            if abs(c) > 0.15:
                print(f"  !! {col}: corr={c:.3f} — investigate before trusting")

    # ---- singles report -------------------------------------------------
    singles = []
    for name in masks.columns:
        m = masks[name].to_numpy()
        st_tr = cell_stats(tr, m[is_train], args.target)
        st_va = cell_stats(va, m[is_val], args.target)
        singles.append({
            "condition": name, "axis": CONDITIONS[name][0],
            "n_train": st_tr.get("n", 0), "p_train": st_tr.get("p", np.nan),
            "lift_train": st_tr.get("p", np.nan) / base_tr,
            "mean_alpha_train": st_tr.get("mean_alpha", np.nan),
            "n_val": st_va.get("n", 0), "p_val": st_va.get("p", np.nan),
            "lift_val": st_va.get("p", np.nan) / base_va,
        })
    singles_df = pd.DataFrame(singles).sort_values("lift_train", ascending=False)
    singles_df.to_csv(out_dir / f"singles_{args.target}.csv", index=False)
    print(f"wrote singles_{args.target}.csv ({len(singles_df)} conditions)")

    # ---- pair mining on TRAIN -------------------------------------------
    names = list(masks.columns)
    rows = []
    for i, a in enumerate(names):
        ax_a = CONDITIONS[a][0]
        ma = masks[a].to_numpy()
        for b in names[i + 1:]:
            ax_b = CONDITIONS[b][0]
            if ax_a == ax_b:
                continue  # require orthogonal axes
            mb = masks[b].to_numpy()
            joint = ma & mb
            jt = joint & is_train
            if jt.sum() < args.min_n:
                continue
            ev = eval_combo(tr, args.target, jt[is_train], [ma[is_train], mb[is_train]], base_tr)
            t, p1, pos_frac, n_m = monthly_block_test(tr, jt[is_train], ma[is_train], mb[is_train], args.target)
            # higher-power continuous outcome: monthly delta of mean forward alpha
            t_a, p1_a, pos_a, _ = monthly_block_test(tr, jt[is_train], ma[is_train], mb[is_train], "fwd_max_alpha")
            ev.update({"A": a, "B": b, "axis_A": ax_a, "axis_B": ax_b,
                       "t_month": t, "p_one": p1, "month_pos_frac": pos_frac, "n_months": n_m,
                       "t_month_alpha": t_a, "p_one_alpha": p1_a, "month_pos_frac_alpha": pos_a})
            # val stats (used only for SELECTION, not for iterating thresholds)
            evv = eval_combo(va, args.target, joint[is_val], [ma[is_val], mb[is_val]], base_va)
            ev.update({f"val_{k}": v for k, v in evv.items()})
            rows.append(ev)
    res = pd.DataFrame(rows)
    if res.empty:
        print("no pairs met the minimum sample size — nothing to report")
        return
    res["q_fdr"] = bh_fdr(res["p_one"])
    res["q_fdr_alpha"] = bh_fdr(res["p_one_alpha"])
    res = res.sort_values(["q_fdr", "p_one"])
    res.to_csv(out_dir / f"pairs_{args.target}.csv", index=False)
    print(f"wrote pairs_{args.target}.csv ({len(res)} pairs tested)")

    common_gates = (
        (res["n"] >= args.min_n)
        & (res["n_episodes"] >= args.min_episodes)
        & (res["n_months"] >= args.min_months)
        & (res["val_n"] >= args.min_n_val)
    )
    surv_bin = res[
        common_gates
        & (res["q_fdr"] <= args.fdr)
        & (res["delta_pp"] > 0)
        & (res["month_pos_frac"] >= args.min_month_pos)
        & (res["val_delta_pp"] > 0)
    ].assign(family="binary_precision")
    surv_alpha = res[
        common_gates
        & (res["q_fdr_alpha"] <= args.fdr)
        & (res["delta_alpha"] > 0)
        & (res["month_pos_frac_alpha"] >= args.min_month_pos)
        & (res["val_delta_alpha"] > 0)
    ].assign(family="mean_alpha")
    survivors = pd.concat([surv_bin, surv_alpha]).drop_duplicates(subset=["A", "B"], keep="first")
    survivors.to_csv(out_dir / f"shortlist_{args.target}.csv", index=False)
    print(f"TRAIN+VAL survivors: binary={len(surv_bin)} alpha={len(surv_alpha)} "
          f"-> shortlist_{args.target}.csv ({len(survivors)} unique)")
    if not survivors.empty:
        cols = ["family", "A", "B", "n", "n_episodes", "p_joint", "lift", "delta_pp", "delta_alpha",
                "log_odds_inter", "q_fdr", "q_fdr_alpha", "month_pos_frac", "month_pos_frac_alpha",
                "val_n", "val_delta_pp", "val_delta_alpha"]
        print(survivors[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---- triples (optional): extend surviving pairs with a third axis ----
    if args.triples and not survivors.empty:
        trows = []
        for _, r in survivors.iterrows():
            pj = masks[r["A"]].to_numpy() & masks[r["B"]].to_numpy()
            used_axes = {r["axis_A"], r["axis_B"]}
            for c in names:
                if CONDITIONS[c][0] in used_axes:
                    continue
                mc = masks[c].to_numpy()
                joint = pj & mc
                jt = joint & is_train
                if jt.sum() < args.min_n:
                    continue
                # baseline for a triple is the surviving PAIR (does C add anything?)
                ev = eval_combo(tr, args.target, jt[is_train], [pj[is_train], mc[is_train]], base_tr)
                t, p1, pos_frac, n_m = monthly_block_test(tr, jt[is_train], pj[is_train], None, args.target)
                ev.update({"A": r["A"], "B": r["B"], "C": c,
                           "t_month": t, "p_one": p1, "month_pos_frac": pos_frac, "n_months": n_m})
                evv = eval_combo(va, args.target, joint[is_val], [pj[is_val], mc[is_val]], base_va)
                ev.update({f"val_{k}": v for k, v in evv.items()})
                trows.append(ev)
        tres = pd.DataFrame(trows)
        if not tres.empty:
            tres["q_fdr"] = bh_fdr(tres["p_one"])
            tres = tres.sort_values(["q_fdr", "p_one"])
            tres.to_csv(out_dir / f"triples_{args.target}.csv", index=False)
            print(f"wrote triples_{args.target}.csv ({len(tres)} triples tested)")


# ---------------------------------------------------------------------------
def stage_confirm(args) -> None:
    """Single TEST-set confirmation of an explicit shortlist. Run ONCE per study."""
    df = load_universe(args.target, args.min_liq, args.include_etfs)
    val_cut, test_cut = temporal_split(df)
    is_train = (df["timestamp"] < val_cut).to_numpy()
    is_test = (df["timestamp"] >= test_cut).to_numpy()
    masks, _ = build_condition_masks(df, pd.Series(is_train, index=df.index))

    te = df[is_test]
    base_te = te[args.target].mean()
    print(f"TEST: {len(te):,} rows {te['month'].nunique()} months "
          f"{te['timestamp'].min().date()} .. {te['timestamp'].max().date()} base={base_te:.4f}")

    short = pd.read_csv(args.shortlist)
    rows = []
    for _, r in short.iterrows():
        parts = [r["A"], r["B"]] + ([r["C"]] if "C" in short.columns and pd.notna(r.get("C")) else [])
        joint = np.logical_and.reduce([masks[p].to_numpy() for p in parts])
        marg = [masks[p].to_numpy()[is_test] for p in parts]
        ev = eval_combo(te, args.target, joint[is_test], marg, base_te)
        n_days_te = te["timestamp"].dt.date.nunique()
        sel_days = te[joint[is_test]]["timestamp"].dt.date.nunique() if ev.get("n", 0) else 0
        ev.update({"combo": " & ".join(parts),
                   "trades_per_day": ev.get("n", 0) / max(n_days_te, 1),
                   "active_day_frac": sel_days / max(n_days_te, 1),
                   "exp_net_ret": ev.get("mean_ret", np.nan) - args.cost_bps / 1e4})
        rows.append(ev)
    out = pd.DataFrame(rows)
    out_path = Path(args.out_dir) / f"test_confirmation_{args.target}.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    if not out.empty:
        cols = ["combo", "n", "n_episodes", "p_joint", "p_base", "lift", "best_marginal_p",
                "delta_pp", "log_odds_inter", "mean_alpha", "mean_ret", "exp_net_ret",
                "win_rate", "mean_dd", "trades_per_day"]
        print(out[[c for c in cols if c in out.columns]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--target", default="meta_upside", choices=["meta_upside", "meta_good"])
    common.add_argument("--min-liq", type=float, default=0.40)
    common.add_argument("--include-etfs", action="store_true")
    common.add_argument("--out-dir", default=str(REPO / "research/confluence"))

    m = sub.add_parser("mine", parents=[common], help="search on TRAIN, select with VAL")
    m.add_argument("--min-n", type=int, default=300)
    m.add_argument("--min-episodes", type=int, default=40)
    m.add_argument("--min-months", type=int, default=6)
    m.add_argument("--min-month-pos", type=float, default=0.625)
    m.add_argument("--min-n-val", type=int, default=60)
    m.add_argument("--fdr", type=float, default=0.10)
    m.add_argument("--triples", action="store_true")

    c = sub.add_parser("confirm", parents=[common], help="single TEST confirmation of a shortlist")
    c.add_argument("--shortlist", required=True)
    c.add_argument("--cost-bps", type=float, default=30.0)

    args = ap.parse_args()
    if args.stage == "mine":
        stage_mine(args)
    else:
        stage_confirm(args)


if __name__ == "__main__":
    main()
