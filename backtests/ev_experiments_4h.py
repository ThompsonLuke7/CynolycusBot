"""
EV / win-rate improvement experiments for the two 4H modules
(momentum_expansion, multi_ticker_swing_htf).

Motivation (research/capstone/leakage_audit.md §0.6): after removing the
test-window selection bias, the honest baselines are

  momentum  (xgb_classifier s45, k5  tp2 sl4 h75): WR 74.7%  EV +2.39%/trade  PF 1.53
  htf_swing (lgbm_classifier s46, k20 tp5 sl2 h25): WR 39.0%  EV +1.45%/trade  PF 1.49

This script asks, WITHOUT re-touching the frozen test window, where EV and
win rate can be improved. All exploratory work runs on walk-forward OOF
scores restricted to timestamps strictly BEFORE each strategy's test cutoff,
split into 3 sequential folds so we reward policies that are robust across
sub-periods instead of lucky in one. Selection order:

  oof-ev       E1  score->EV gradient + big-mover capture rates (fast, pandas)
  policy-sweep E2  vectorized exit/top-K/conviction-gate sweep on pre-test OOF,
                   ranked by cross-fold robustness (expectancy + win rate)
  val-check    E3  re-simulate shortlisted configs with the DEPLOYED model's
                   scores on the validation window (exact fb.simulate engine)
  frozen-test  E4  one-shot exact simulation of the final config on the test
                   window — run once, after E2/E3 have fixed the choice

Execution semantics mirror family_backtest exactly: entry at next 4H open
after the signal bar, TP/SL from ATR(14) at the signal bar, same-bar SL
priority, time-stop at max_hold, fixed $1k notional (sizing-neutral).
P&L here has NO transaction costs; net-of-cost EV is reported as
avg_trade_pct minus a 20bp round-trip haircut column (`avg_net20bp`).

Usage:
  PYTHONPATH=. .venv/bin/python backtests/ev_experiments_4h.py --exp oof-ev --strategy all
  PYTHONPATH=. .venv/bin/python backtests/ev_experiments_4h.py --exp policy-sweep --strategy momentum
  PYTHONPATH=. .venv/bin/python backtests/ev_experiments_4h.py --exp val-check --strategy momentum
  PYTHONPATH=. .venv/bin/python backtests/ev_experiments_4h.py --exp frozen-test --strategy momentum \
      --top-k 5 --tp 4 --sl 3 --hold 50 --conviction-z 1.0
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.capstone.family_backtest_clean import compute_cutoffs, load_window
from scripts.capstone.reproduce_results import _read_oof
from strategies.momentum_expansion.backtest import family_backtest as fb
from strategies.momentum_expansion.backtest.run_family_compare import STRATEGIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ev_experiments_4h")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "backtests" / "ev_experiments_4h"

SPECS = {
    "momentum": dict(
        oof=ROOT / "strategies/momentum_expansion/models/expansion_v1/oof_preds.parquet",
        fwd_up="fwd_max_return", fwd_dn="fwd_max_drawdown",
        baseline=dict(top_k=5, tp=2.0, sl=4.0, hold=75, conviction_z=None),
    ),
    "htf": dict(
        oof=ROOT / "strategies/multi_ticker_swing_htf/models/oof_preds.parquet",
        fwd_up="fwd_best_high_return", fwd_dn="fwd_worst_low_return",
        baseline=dict(top_k=20, tp=5.0, sl=2.0, hold=25, conviction_z=None),
    ),
}

K_MAX = 20                      # widest selectivity explored
N_FOLDS = 3                     # sequential pre-test folds for robustness
MIN_TRADES_PER_FOLD = 150       # combos thinner than this are noise, drop
COST_HAIRCUT = 0.002            # 20bp round-trip sensitivity column
GRID = dict(
    tp=[2.0, 3.0, 4.0, 5.0, 6.0],
    sl=[2.0, 3.0, 4.0, 5.0],
    hold=[25, 50, 75],
    top_k=[3, 5, 10, 20],
    conviction_z=[None, 1.0, 2.0],   # cross-sectional z-score gate at signal time
)


def _cfg(strategy: str) -> fb.StrategyConfig:
    spec = STRATEGIES[strategy]
    return fb.StrategyConfig.from_manifest(
        spec["name"], spec["models_dir"], spec["matrix_path"],
        allow_short=spec["allow_short"], forward_window=spec["forward_window"],
    )


def _low_price_ok(cfg: fb.StrategyConfig, cutoff: pd.Timestamp) -> pd.DataFrame:
    """(timestamp, ticker) rows passing the live low-price gate, pre-cutoff."""
    t = fb._read_reset(cfg.matrix_path, ["timestamp", "ticker", "low_price_flag"],
                       [("timestamp", "<", cutoff.to_pydatetime())])
    t["timestamp"] = pd.to_datetime(t["timestamp"], utc=True)
    t["ticker"] = t["ticker"].astype(str)
    flag = pd.to_numeric(t["low_price_flag"], errors="coerce").fillna(1.0)
    return t.loc[flag <= 0.0, ["timestamp", "ticker"]]


def load_pretest_oof(strategy: str) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """OOF rows strictly before the test cutoff, low-price gate applied."""
    cfg = _cfg(strategy)
    train_end, test_start = compute_cutoffs(cfg)
    df = _read_oof(SPECS[strategy]["oof"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ticker"] = df["ticker"].astype(str)
    df = df[df["timestamp"] < test_start].copy()
    ok = _low_price_ok(cfg, test_start)
    n0 = len(df)
    df = df.merge(ok, on=["timestamp", "ticker"], how="inner")
    logger.info("[%s] pre-test OOF rows=%d (low-price dropped %d)  span %s -> %s",
                strategy, len(df), n0 - len(df), df["timestamp"].min(), df["timestamp"].max())
    return df.reset_index(drop=True), train_end, test_start


# ---------------------------------------------------------------------------
# E1: score -> EV gradient and big-mover capture (label-space, no execution)
# ---------------------------------------------------------------------------

RANK_BUCKETS = [(1, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 10 ** 9)]
MOVER_THRESHOLDS = [0.10, 0.15, 0.20]
CAPTURE_KS = [5, 10, 20, 50]


def _bucket_table(df: pd.DataFrame, rank_col: str, fwd_close: pd.Series,
                  fwd_up: pd.Series, fwd_dn: pd.Series) -> pd.DataFrame:
    rows = []
    for lo, hi in RANK_BUCKETS:
        m = (df[rank_col] >= lo) & (df[rank_col] <= hi)
        if not m.any():
            continue
        rows.append({
            "rank_bucket": f"{lo}-{hi if hi < 10**9 else 'rest'}",
            "n": int(m.sum()),
            "ev_close_pct": round(fwd_close[m].mean() * 100, 3),
            "median_close_pct": round(fwd_close[m].median() * 100, 3),
            "win_close": round((fwd_close[m] > 0).mean(), 4),
            "avg_fwd_up_pct": round(fwd_up[m].mean() * 100, 3),
            "avg_fwd_dn_pct": round(fwd_dn[m].mean() * 100, 3),
        })
    return pd.DataFrame(rows)


def _capture_table(df: pd.DataFrame, rank_col: str, mover: pd.Series) -> pd.DataFrame:
    rows = []
    base = mover.mean()
    for thr_k in CAPTURE_KS:
        top = df[rank_col] <= thr_k
        both = (top & mover).sum()
        rows.append({
            "top_k": thr_k,
            "capture_recall": round(both / max(mover.sum(), 1), 4),
            "precision": round(both / max(top.sum(), 1), 4),
            "lift_vs_base": round((both / max(top.sum(), 1)) / base, 2) if base > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def exp_oof_ev(strategy: str) -> None:
    spec = SPECS[strategy]
    df, _, _ = load_pretest_oof(strategy)
    df["rank_long"] = df.groupby("timestamp")["score"].rank(ascending=False, method="first")

    print(f"\n=== {strategy}: LONG side — EV by cross-sectional score rank (pre-test OOF) ===")
    tbl = _bucket_table(df, "rank_long", df["fwd_close_return"], df[spec["fwd_up"]], df[spec["fwd_dn"]])
    print(tbl.to_string(index=False))
    tbl.to_csv(OUT / f"{strategy}_e1_long_rank_ev.csv", index=False)

    print(f"\n=== {strategy}: LONG big-mover capture (mover = {spec['fwd_up']} >= thr) ===")
    for thr in MOVER_THRESHOLDS:
        mover = df[spec["fwd_up"]] >= thr
        cap = _capture_table(df, "rank_long", mover)
        cap.insert(0, "mover_thr", thr)
        print(f"-- movers >= {thr:.0%} (base rate {mover.mean():.4f})")
        print(cap.to_string(index=False))
        cap.to_csv(OUT / f"{strategy}_e1_long_capture_{int(thr*100)}.csv", index=False)

    if STRATEGIES[strategy]["allow_short"]:
        df["rank_short"] = df.groupby("timestamp")["score"].rank(ascending=True, method="first")
        print(f"\n=== {strategy}: SHORT side — EV by rank (fwd returns sign-flipped) ===")
        tbl = _bucket_table(df, "rank_short", -df["fwd_close_return"],
                            -df[spec["fwd_dn"]], -df[spec["fwd_up"]])
        print(tbl.to_string(index=False))
        tbl.to_csv(OUT / f"{strategy}_e1_short_rank_ev.csv", index=False)
        for thr in MOVER_THRESHOLDS:
            mover = df[spec["fwd_dn"]] <= -thr
            cap = _capture_table(df, "rank_short", mover)
            cap.insert(0, "mover_thr", -thr)
            print(f"-- short movers <= -{thr:.0%} (base rate {mover.mean():.4f})")
            print(cap.to_string(index=False))
            cap.to_csv(OUT / f"{strategy}_e1_short_capture_{int(thr*100)}.csv", index=False)


# ---------------------------------------------------------------------------
# E2: vectorized path-based policy sweep on pre-test OOF, cross-fold robustness
# ---------------------------------------------------------------------------

B_MAX = max(GRID["hold"]) + 1   # bars of path stored per signal
BIG = 10 ** 6


def build_signals(df: pd.DataFrame, allow_short: bool) -> pd.DataFrame:
    """Top-K_MAX longs (and bottom-K_MAX shorts) per timestamp with conviction z."""
    g = df.groupby("timestamp")["score"]
    mu, sd = g.transform("mean"), g.transform("std")
    df = df.assign(cs_z=(df["score"] - mu) / sd.replace(0.0, np.nan))
    long_rank = g.rank(ascending=False, method="first")
    out = df.loc[long_rank <= K_MAX, ["timestamp", "ticker", "score", "cs_z"]].copy()
    out["rank"] = long_rank[long_rank <= K_MAX]
    out["direction"] = 1
    if allow_short:
        short_rank = g.rank(ascending=True, method="first")
        sh = df.loc[short_rank <= K_MAX, ["timestamp", "ticker", "score", "cs_z"]].copy()
        sh["rank"] = short_rank[short_rank <= K_MAX]
        sh["direction"] = -1
        sh["cs_z"] = -sh["cs_z"]          # conviction = signed extremeness
        out = pd.concat([out, sh], ignore_index=True)
    return out.rename(columns={"cs_z": "conviction_z"})


class Paths:
    """Per-signal forward bar paths in ATR units (vectorized policy evaluation).

    exc_fav/exc_adv: favorable/adverse excursion of bar extremes vs entry, in
    ATR-at-signal units; -inf past the end of available data so TP/SL can
    never trigger there. close_ret is the unsigned (close-entry)/entry path,
    last real bar repeated (a time-stop past data end exits on the last bar,
    like fb._simulate_signal's `last = min(...)`).
    """

    def __init__(self, signals: pd.DataFrame, cache: fb.BarCache):
        chunks = []
        arrs = {k: [] for k in ("exc_fav", "exc_adv", "close_ret", "exit_ts", "atr_frac", "n_avail", "direction")}
        offs = np.arange(B_MAX)
        for tkr, grp in signals.groupby("ticker", sort=False):
            bars = cache.get(tkr)
            if bars is None or len(bars["ts"]) < 20:
                continue
            ts = bars["ts"]
            sig_i = np.searchsorted(ts, grp["timestamp"].values.astype("datetime64[ns]").astype(np.int64),
                                    side="right") - 1
            entry_i = sig_i + 1
            with np.errstate(invalid="ignore"):
                atr0 = np.where(sig_i >= 0, bars["atr"][np.maximum(sig_i, 0)], np.nan)
                entry = np.where(entry_i < len(ts), bars["open"][np.minimum(entry_i, len(ts) - 1)], np.nan)
            ok = (sig_i >= 0) & (entry_i < len(ts)) & np.isfinite(atr0) & (atr0 > 0) \
                & np.isfinite(entry) & (entry > 0)
            if not ok.any():
                continue
            grp = grp.loc[ok]
            entry_i, atr0, entry = entry_i[ok], atr0[ok], entry[ok]
            idx = np.minimum(entry_i[:, None] + offs[None, :], len(ts) - 1)
            hi, lo, cl = bars["high"][idx], bars["low"][idx], bars["close"][idx]
            n_avail = np.minimum(len(ts) - entry_i, B_MAX)
            beyond = offs[None, :] >= n_avail[:, None]
            d = grp["direction"].to_numpy()[:, None]
            e, a = entry[:, None], atr0[:, None]
            fav = np.where(d > 0, (hi - e) / a, (e - lo) / a)
            adv = np.where(d > 0, (e - lo) / a, (hi - e) / a)
            fav[beyond] = -np.inf
            adv[beyond] = -np.inf
            arrs["exc_fav"].append(fav.astype(np.float32))
            arrs["exc_adv"].append(adv.astype(np.float32))
            arrs["close_ret"].append(((cl - e) / e).astype(np.float32))
            arrs["exit_ts"].append(bars["ts"][idx])
            arrs["atr_frac"].append((atr0 / entry).astype(np.float32))
            arrs["n_avail"].append(n_avail)
            arrs["direction"].append(grp["direction"].to_numpy())
            chunks.append(grp)
        self.meta = pd.concat(chunks, ignore_index=True)
        self.exc_fav = np.vstack(arrs["exc_fav"])
        self.exc_adv = np.vstack(arrs["exc_adv"])
        self.close_ret = np.vstack(arrs["close_ret"])
        self.exit_ts = np.vstack(arrs["exit_ts"])
        self.atr_frac = np.concatenate(arrs["atr_frac"])
        self.n_avail = np.concatenate(arrs["n_avail"])
        self.direction = np.concatenate(arrs["direction"])
        logger.info("paths built: %d signals x %d bars", len(self.meta), B_MAX)

    def eval_policy(self, rows: np.ndarray, tp: float, sl: float, hold: int) -> dict:
        """Exact outcome of (tp, sl, hold) for the row subset. Same-bar SL priority."""
        F = self.exc_fav[rows, : hold + 1]
        A = self.exc_adv[rows, : hold + 1]
        i_tp = np.where((F >= tp).any(1), (F >= tp).argmax(1), BIG)
        i_sl = np.where((A >= sl).any(1), (A >= sl).argmax(1), BIG)
        last_j = np.minimum(hold, self.n_avail[rows] - 1)
        j = np.minimum(np.minimum(i_tp, i_sl), last_j)
        sl_hit = (i_sl <= i_tp) & (i_sl <= last_j)
        tp_hit = (i_tp < i_sl) & (i_tp <= last_j)
        af = self.atr_frac[rows]
        d = self.direction[rows]
        cr = self.close_ret[rows, j] * d
        pnl = np.where(sl_hit, -sl * af, np.where(tp_hit, tp * af, cr))
        return dict(pnl=pnl, exit_reason=np.where(sl_hit, "sl", np.where(tp_hit, "tp", "time")),
                    exit_ts=self.exit_ts[rows, j], bars_held=j)


def _light_metrics(pnl: np.ndarray, reason: np.ndarray, bars_held: np.ndarray) -> dict:
    pos = pnl[pnl > 0].sum()
    neg = -pnl[pnl <= 0].sum()
    return {
        "trades": int(len(pnl)),
        "win_rate": round(float((pnl > 0).mean()), 4),
        "avg_trade_pct": round(float(pnl.mean()) * 100, 4),
        "avg_net20bp": round(float(pnl.mean() - COST_HAIRCUT) * 100, 4),
        "median_trade_pct": round(float(np.median(pnl)) * 100, 4),
        "profit_factor": round(float(pos / neg), 3) if neg > 0 else float("inf"),
        "tp_rate": round(float((reason == "tp").mean()), 3),
        "sl_rate": round(float((reason == "sl").mean()), 3),
        "time_rate": round(float((reason == "time").mean()), 3),
        "avg_bars_held": round(float(bars_held.mean()), 2),
    }


def exp_policy_sweep(strategy: str) -> None:
    from itertools import product
    df, _, test_start = load_pretest_oof(strategy)
    allow_short = STRATEGIES[strategy]["allow_short"]
    signals = build_signals(df, allow_short)
    paths = Paths(signals, fb.BarCache())

    m = paths.meta
    lo, hi = m["timestamp"].min(), test_start
    edges = pd.date_range(lo, hi, periods=N_FOLDS + 1)
    fold_id = np.searchsorted(edges.values[1:-1], m["timestamp"].values, side="right")
    logger.info("[%s] folds: %s", strategy, [str(e.date()) for e in edges])

    results = []
    sides = ["both", "long_only"] if allow_short else ["long_only"]
    for top_k, cz, side in product(GRID["top_k"], GRID["conviction_z"], sides):
        base = (m["rank"].to_numpy() <= top_k)
        if cz is not None:
            base &= (m["conviction_z"].to_numpy() >= cz)
        if side == "long_only":
            base &= (m["direction"].to_numpy() == 1)
        for tp, sl, hold in product(GRID["tp"], GRID["sl"], GRID["hold"]):
            rec = dict(top_k=top_k, conviction_z=cz if cz is not None else 0.0,
                       side=side, tp=tp, sl=sl, hold=hold)
            ok = True
            for f in range(N_FOLDS):
                rows = np.flatnonzero(base & (fold_id == f))
                if len(rows) < MIN_TRADES_PER_FOLD:
                    ok = False
                    break
                out = paths.eval_policy(rows, tp, sl, hold)
                lm = _light_metrics(out["pnl"], out["exit_reason"], out["bars_held"])
                rec.update({f"f{f}_{k}": v for k, v in lm.items()
                            if k in ("trades", "win_rate", "avg_trade_pct", "avg_net20bp", "profit_factor")})
            if not ok:
                continue
            rec["ev_min_fold"] = min(rec[f"f{f}_avg_trade_pct"] for f in range(N_FOLDS))
            rec["ev_mean"] = round(np.mean([rec[f"f{f}_avg_trade_pct"] for f in range(N_FOLDS)]), 4)
            rec["wr_min_fold"] = min(rec[f"f{f}_win_rate"] for f in range(N_FOLDS))
            rec["wr_mean"] = round(np.mean([rec[f"f{f}_win_rate"] for f in range(N_FOLDS)]), 4)
            results.append(rec)

    res = pd.DataFrame(results)
    for col, asc in (("ev_min_fold", False), ("wr_min_fold", False)):
        res[f"rank_{col}"] = res[col].rank(ascending=asc, method="min")
    res.to_csv(OUT / f"{strategy}_e2_policy_sweep.csv", index=False)

    show = ["top_k", "conviction_z", "side", "tp", "sl", "hold", "ev_mean", "ev_min_fold",
            "wr_mean", "wr_min_fold", "f0_trades", "f1_trades", "f2_trades",
            "f0_avg_net20bp", "f1_avg_net20bp", "f2_avg_net20bp"]
    print(f"\n=== {strategy}: top 12 by WORST-fold expectancy (robust EV) ===")
    print(res.sort_values("ev_min_fold", ascending=False).head(12)[show].to_string(index=False))
    print(f"\n=== {strategy}: top 12 by WORST-fold win rate, EV>0 net cost in every fold ===")
    net_pos = res[[f"f{f}_avg_net20bp" for f in range(N_FOLDS)]].min(axis=1) > 0
    print(res[net_pos].sort_values("wr_min_fold", ascending=False).head(12)[show].to_string(index=False))

    b = SPECS[strategy]["baseline"]
    row = res[(res.top_k == b["top_k"]) & (res.tp == b["tp"]) & (res.sl == b["sl"])
              & (res.hold == b["hold"]) & (res.conviction_z == 0.0)]
    print(f"\n=== {strategy}: current clean-baseline policy on the same folds ===")
    print(row[show].to_string(index=False) if not row.empty else "  (baseline fell below trade floor)")


# ---------------------------------------------------------------------------
# E3/E4: exact-engine evaluation of shortlisted configs (deployed model scores)
# ---------------------------------------------------------------------------

def _deployed(cfg: fb.StrategyConfig) -> tuple[str, int]:
    em = json.loads((cfg.models_dir / "eval_metrics.json").read_text())
    family = em["winner_family"]
    return family, fb.best_seed_per_family(cfg.models_dir)[family]


def _exact_eval(strategy: str, which: str, configs: list[dict]) -> pd.DataFrame:
    """Exact fb.simulate run of each config on the val or test window."""
    cfg = _cfg(strategy)
    train_end, test_start = compute_cutoffs(cfg)
    dfw = load_window(cfg, which, train_end, test_start)
    family, seed = _deployed(cfg)
    dfw["score"] = fb.score_family(dfw, cfg, family, seed)
    g = dfw.groupby("timestamp")["score"]
    dfw["conviction_z"] = (dfw["score"] - g.transform("mean")) / g.transform("std").replace(0.0, np.nan)
    cache = fb.BarCache()
    rows = []
    for c in configs:
        scored = dfw[["timestamp", "ticker", "score", "conviction_z"]].copy()
        allow_short = cfg.allow_short and c.get("side", "both") != "long_only"
        sig = fb.select_signals(scored[["timestamp", "ticker", "score"]], int(c["top_k"]), allow_short)
        if c.get("conviction_z"):
            sig = sig.merge(scored[["timestamp", "ticker", "conviction_z"]], on=["timestamp", "ticker"])
            sig = sig[sig["conviction_z"] * sig["direction"] >= float(c["conviction_z"])].drop(columns="conviction_z")
        trades = fb.simulate(sig, cache, tp_mult=float(c["tp"]), sl_mult=float(c["sl"]), max_hold=int(c["hold"]))
        met = fb.metrics(trades)
        met["avg_net20bp"] = round(met["avg_trade_pct"] - COST_HAIRCUT * 100, 4) if met["trades"] else np.nan
        rows.append({**{k: c.get(k) for k in ("label", "top_k", "conviction_z", "side", "tp", "sl", "hold")},
                     "window": which, "family": family, "seed": seed, **met})
    return pd.DataFrame(rows)


def exp_val_check(strategy: str, configs_path: Path | None) -> None:
    if configs_path and configs_path.exists():
        configs = json.loads(configs_path.read_text())
    else:
        raise SystemExit(f"--configs JSON required for val-check (shortlist from E2): {configs_path}")
    res = _exact_eval(strategy, "val", configs)
    res.to_csv(OUT / f"{strategy}_e3_val_check.csv", index=False)
    print(f"\n=== {strategy}: exact-engine VAL-window check (deployed model scores) ===")
    print(res.to_string(index=False))


def exp_frozen_test(strategy: str, config: dict) -> None:
    res = _exact_eval(strategy, "test", [config])
    res.to_csv(OUT / f"{strategy}_e4_frozen_test.csv", index=False)
    print(f"\n=== {strategy}: ONE-SHOT frozen-test result (do not iterate on this) ===")
    print(res.to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", choices=["oof-ev", "policy-sweep", "val-check", "frozen-test"], required=True)
    p.add_argument("--strategy", choices=["momentum", "htf", "all"], required=True)
    p.add_argument("--configs", type=Path, help="val-check: JSON list of shortlisted configs")
    p.add_argument("--label", default="candidate")
    p.add_argument("--top-k", type=int)
    p.add_argument("--tp", type=float)
    p.add_argument("--sl", type=float)
    p.add_argument("--hold", type=int)
    p.add_argument("--conviction-z", type=float, default=None)
    p.add_argument("--side", choices=["both", "long_only"], default="both")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    strategies = ["momentum", "htf"] if args.strategy == "all" else [args.strategy]
    for s in strategies:
        if args.exp == "oof-ev":
            exp_oof_ev(s)
        elif args.exp == "policy-sweep":
            exp_policy_sweep(s)
        elif args.exp == "val-check":
            exp_val_check(s, args.configs)
        elif args.exp == "frozen-test":
            cfg = dict(label=args.label, top_k=args.top_k, tp=args.tp, sl=args.sl,
                       hold=args.hold, conviction_z=args.conviction_z, side=args.side)
            if any(cfg[k] is None for k in ("top_k", "tp", "sl", "hold")):
                raise SystemExit("frozen-test needs --top-k --tp --sl --hold")
            exp_frozen_test(s, cfg)


if __name__ == "__main__":
    main()
