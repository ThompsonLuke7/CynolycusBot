"""
Guided (round-2) search over the Meta Ranker 4H SHARE exit policy.

Builds on exit_policy_random_search.py (round 1) — SAME val/test split so all
results are directly comparable:
  VAL  2025-07-01 -> 2026-01-15   (search here)
  TEST 2026-01-15 -> 2026-05-15   (frozen; only val-selected winners, once)

What "smarter" means here, from the round-1 correlations + MFE path analysis:
  - horizon dominates mean/total return (winners peak at median bar 48 -> the
    25-bar deployed horizon exits half of them early);
  - looser trail/stop beat tighter everywhere (tight trails cut runners:
    winners' pre-peak MAE median is only -4.3%, but tight trails still bind);
  - losers separate early: 12-bar return predicts 50-bar return (spearman .46),
    and the swing module already trades a "no-progress" exit live.
So this search (a) samples three hypothesis regions instead of the whole cube,
(b) adds a NO-PROGRESS exit (np_bars/np_ret) the round-1 harness didn't have,
and (c) locally perturbs round-1's best configs.

The extended simulator is validated against backtest_exits.simulate() before
searching: with np disabled it must reproduce round-1 numbers exactly.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/capstone/exit_policy_guided_search.py [--seed 7]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "signals/meta_context/meta_ranker"))
import backtest_exits as be  # noqa: E402

VAL_START = pd.Timestamp("2025-07-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-15", tz="UTC")
TEST_END = pd.Timestamp("2026-05-15", tz="UTC")

ROUND1_VAL = REPO / "research/capstone/exit_policy_random_search_val.csv"
OUT_VAL = REPO / "research/capstone/exit_policy_guided_search_val.csv"
OUT_TEST = REPO / "research/capstone/exit_policy_guided_search_test.csv"

_BAR_CACHE: dict[str, pd.DataFrame | None] = {}


def _bars(ticker: str) -> pd.DataFrame | None:
    if ticker not in _BAR_CACHE:
        _BAR_CACHE[ticker] = be._ticker_path(ticker, None)
    return _BAR_CACHE[ticker]


def simulate_ext(member: pd.DataFrame, *, stop=None, target=None, scale_frac=1.0,
                 trail=None, grace=0, horizon=None,
                 np_bars=None, np_ret=None) -> dict:
    """backtest_exits.simulate() + optional no-progress exit.

    np_bars/np_ret: at bar np_bars after entry, if close-return < np_ret,
    exit the whole remaining position at that close. Checked AFTER stop/trail/
    target on the same bar (protective exits take precedence), mirroring the
    ordering of the original rules.
    """
    rets, holds = [], []
    for ticker, g in member.groupby("ticker"):
        g = g.sort_values("timestamp")
        bars = _bars(ticker)
        if bars is None:
            continue
        m = g.set_index("timestamp")["in_top"].reindex(bars.index).fillna(False).astype(bool).values
        close = bars["close"].values
        high = bars["high"].values
        low = bars["low"].values
        n = len(bars)
        i = 0
        while i < n - 1:
            if not m[i]:
                i += 1
                continue
            entry = close[i]
            if entry <= 0:
                i += 1
                continue
            peak = entry
            realized = 0.0
            remaining = 1.0
            trimmed = False
            out = 0
            j = i + 1
            exit_ret = None
            while j < n and (j - i) <= be.MAX_HOLD:
                peak = max(peak, high[j])
                lo_ret = low[j] / entry - 1
                hi_ret = high[j] / entry - 1
                if stop is not None and lo_ret <= -stop:
                    exit_ret = -stop
                    break
                if trail is not None and low[j] <= peak * (1 - trail):
                    exit_ret = peak * (1 - trail) / entry - 1
                    break
                if target is not None and not trimmed and hi_ret >= target:
                    if scale_frac >= 1.0:
                        exit_ret = target
                        break
                    realized += scale_frac * target
                    remaining = 1.0 - scale_frac
                    trimmed = True
                # no-progress: cut dead positions at the checkpoint bar
                if np_bars is not None and (j - i) == np_bars and not trimmed \
                        and (close[j] / entry - 1) < np_ret:
                    exit_ret = close[j] / entry - 1
                    break
                out = out + 1 if not m[j] else 0
                if grace is not None and out > grace:
                    exit_ret = close[j] / entry - 1
                    break
                if horizon is not None and (j - i) >= horizon:
                    exit_ret = close[j] / entry - 1
                    break
                j += 1
            if exit_ret is None:
                jj = min(j, n - 1)
                exit_ret = close[jj] / entry - 1
            total = realized + remaining * exit_ret
            rets.append(total)
            holds.append(min(j, n - 1) - i)
            i = min(j, n - 1) + 1
    r = np.array(rets)
    h = np.array(holds)
    if len(r) == 0:
        return {}
    return {
        "n": len(r), "mean": r.mean(), "median": np.median(r), "win": (r > 0).mean(),
        "std": r.std(), "ret_std": r.mean() / r.std() if r.std() else 0,
        "avg_hold": h.mean(), "ret_per_bar": (r / np.maximum(h, 1)).mean(),
        "total_ret": r.sum(),
    }


def _load_member_window(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_parquet(be.SCORED).dropna(subset=["s_combo"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["rk"] = df.groupby("timestamp")["s_combo"].rank(ascending=False, method="first")
    df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
    df["in_top"] = df["rk"] <= be.TOPK
    return df[["timestamp", "ticker", "in_top"]]


PARAM_KEYS = ("stop", "trail", "target", "scale_frac", "horizon", "grace", "np_bars", "np_ret")


def _clip(v, lo, hi):
    return None if v is None else round(float(min(max(v, lo), hi)), 2)


def sample_regions(rng: np.random.Generator, n_per: int) -> list[dict]:
    cfgs: list[dict] = []
    # A) "compounder": long ride for the bar-48 peaks, protective floor, small trim
    for _ in range(n_per):
        cfgs.append(dict(
            stop=None if rng.random() < 0.2 else _clip(rng.uniform(0.15, 0.40), 0.10, 0.75),
            trail=None if rng.random() < 0.5 else _clip(rng.uniform(0.35, 0.50), 0.10, 0.50),
            target=_clip(rng.uniform(0.25, 0.50), 0.05, 0.50),
            scale_frac=_clip(rng.uniform(0.10, 0.40), 0.05, 1.0),
            horizon=int(rng.integers(45, 61)), grace=None,
            np_bars=None if rng.random() < 0.4 else int(rng.integers(8, 17)),
            np_ret=float(np.round(rng.uniform(-0.05, 0.02), 3)),
        ))
    # B) "efficiency" around round-1 id 68: short cap, full exit at modest target
    for _ in range(n_per):
        cfgs.append(dict(
            stop=_clip(rng.uniform(0.20, 0.40), 0.10, 0.75),
            trail=_clip(rng.uniform(0.30, 0.50), 0.10, 0.50),
            target=_clip(rng.uniform(0.06, 0.15), 0.05, 0.50),
            scale_frac=_clip(rng.uniform(0.85, 1.00), 0.05, 1.0),
            horizon=int(rng.integers(8, 21)), grace=None,
            np_bars=None if rng.random() < 0.5 else int(rng.integers(4, 13)),
            np_ret=float(np.round(rng.uniform(-0.03, 0.02), 3)),
        ))
    # C) user hypothesis: small early trim, loose trail on the runner, long ride
    for _ in range(n_per):
        cfgs.append(dict(
            stop=_clip(rng.uniform(0.15, 0.35), 0.10, 0.75),
            trail=_clip(rng.uniform(0.30, 0.50), 0.10, 0.50),
            target=_clip(rng.uniform(0.08, 0.15), 0.05, 0.50),
            scale_frac=_clip(rng.uniform(0.20, 0.50), 0.05, 1.0),
            horizon=int(rng.integers(45, 61)), grace=None,
            np_bars=None if rng.random() < 0.4 else int(rng.integers(8, 17)),
            np_ret=float(np.round(rng.uniform(-0.05, 0.02), 3)),
        ))
    for c in cfgs:
        if c["np_bars"] is None:
            c["np_ret"] = None
    return cfgs


def perturb_seeds(rng: np.random.Generator, seeds: list[dict], n_per: int) -> list[dict]:
    out = []
    for s in seeds:
        for _ in range(n_per):
            c = dict(s)
            if c.get("stop") is not None:
                c["stop"] = _clip(c["stop"] + rng.normal(0, 0.06), 0.10, 0.75)
            if c.get("trail") is not None:
                c["trail"] = _clip(c["trail"] + rng.normal(0, 0.05), 0.10, 0.50)
            if c.get("target") is not None:
                c["target"] = _clip(c["target"] + rng.normal(0, 0.04), 0.05, 0.50)
            c["scale_frac"] = _clip((c.get("scale_frac") or 1.0) + rng.normal(0, 0.10), 0.05, 1.0)
            if c.get("horizon") is not None:
                c["horizon"] = int(np.clip(c["horizon"] + rng.integers(-6, 7), 8, 60))
            # occasionally bolt a no-progress rule onto a round-1 shape
            if c.get("np_bars") is None and rng.random() < 0.35:
                c["np_bars"] = int(rng.integers(8, 17))
                c["np_ret"] = float(np.round(rng.uniform(-0.05, 0.02), 3))
            c["grace"] = None
            out.append(c)
    return out


def run_all(member: pd.DataFrame, configs: list[dict], label: str) -> pd.DataFrame:
    rows, t0 = [], time.time()
    for i, cfg in enumerate(configs):
        s = simulate_ext(member, **cfg)
        if s:
            rows.append({**{k: cfg.get(k) for k in PARAM_KEYS}, **s})
        if (i + 1) % 50 == 0:
            print(f"  [{label}] {i + 1}/{len(configs)}  ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    val_member = _load_member_window(VAL_START, VAL_END)
    test_member = _load_member_window(VAL_END, TEST_END)

    # ---- validate the extended simulator against round-1 numbers ----
    r1 = pd.read_csv(ROUND1_VAL)
    for pid in (1, 4, 68):
        row = r1[r1["policy_id"] == pid].iloc[0]
        cfg = {k: (None if pd.isna(row[k]) else row[k]) for k in
               ("stop", "trail", "target", "scale_frac", "horizon", "grace")}
        if cfg["horizon"] is not None:
            cfg["horizon"] = int(cfg["horizon"])
        s = simulate_ext(val_member, **cfg)
        for col in ("n", "mean", "median", "win", "ret_per_bar"):
            got, want = s[col], row[col]
            assert abs(got - want) < 1e-9, f"pid {pid} {col}: {got} != {want}"
    print("extended simulator validated: reproduces round-1 val rows exactly (pids 1/4/68)")

    # ---- build config list ----
    reference = [
        dict(stop=None, trail=None, target=None, scale_frac=1.0, horizon=None, grace=0,
             np_bars=None, np_ret=None),                                     # current live
        dict(stop=0.50, trail=0.35, target=0.20, scale_frac=0.5, horizon=25, grace=None,
             np_bars=None, np_ret=None),                                     # deployed
    ]
    elig = r1[(r1["policy_id"] >= 2) & (r1["n"] >= 20)]
    seed_ids = set()
    for obj in ("mean", "median", "win", "ret_per_bar"):
        seed_ids.update(elig.nlargest(3, obj)["policy_id"].tolist())
    elig = elig.assign(total_ret=elig["n"] * elig["mean"])
    seed_ids.update(elig.nlargest(3, "total_ret")["policy_id"].tolist())
    seeds = []
    for pid in sorted(seed_ids):
        row = r1[r1["policy_id"] == pid].iloc[0]
        s = {k: (None if pd.isna(row[k]) else row[k]) for k in
             ("stop", "trail", "target", "scale_frac", "horizon", "grace")}
        if s["horizon"] is not None:
            s["horizon"] = int(s["horizon"])
        s["np_bars"] = None
        s["np_ret"] = None
        seeds.append(s)
    print(f"round-1 seeds for local perturbation: {sorted(seed_ids)}")

    configs = reference + sample_regions(rng, 60) + perturb_seeds(rng, seeds, 12)
    print(f"running {len(configs)} configs on VAL...")
    val_df = run_all(val_member, configs, "val")
    val_df.insert(0, "policy_id", [f"g{i}" for i in range(len(val_df))])
    val_df.to_csv(OUT_VAL, index=False)
    print(f"saved {OUT_VAL} ({len(val_df)} rows)")

    # ---- winner selection (val only), then one frozen-test pass ----
    eligible = val_df[val_df["n"] >= 20]
    winners: dict[str, str] = {}
    for obj in ("mean", "median", "win", "ret_per_bar", "total_ret", "ret_std"):
        winners[obj] = eligible.loc[eligible[obj].idxmax(), "policy_id"]
    winner_ids = sorted(set(winners.values()) | {"g0", "g1"})
    print(f"\nval winners: {winners}")

    test_rows = []
    for pid in winner_ids:
        rec = val_df[val_df["policy_id"] == pid].iloc[0]
        cfg = {k: (None if pd.isna(rec[k]) else rec[k]) for k in PARAM_KEYS}
        for ik in ("horizon", "np_bars"):
            if cfg[ik] is not None:
                cfg[ik] = int(cfg[ik])
        s = simulate_ext(test_member, **cfg)
        if not s:
            continue
        won_by = ",".join(o for o, p in winners.items() if p == pid) or "reference"
        test_rows.append({"policy_id": pid, "won_by": won_by, **cfg, **s})
    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(OUT_TEST, index=False)
    print(f"saved {OUT_TEST} ({len(test_df)} rows)\n")

    hdr = (f"{'id':>5} {'won_by':24} {'stop':>5} {'trail':>5} {'tgt':>5} {'scl':>5} {'hz':>4} "
           f"{'npB':>4} {'npR':>6} | {'val_mean':>8} {'tst_mean':>8} | {'val_med':>8} {'tst_med':>8} "
           f"| {'val_win':>7} {'tst_win':>7} | {'val_rpb':>8} {'tst_rpb':>8} | {'val_tot':>8} {'tst_tot':>8}")
    print(hdr)
    for rec in test_df.to_dict("records"):
        v = val_df[val_df["policy_id"] == rec["policy_id"]].iloc[0]
        print(f"{rec['policy_id']:>5} {rec['won_by']:24} {str(rec['stop']):>5} {str(rec['trail']):>5} "
              f"{str(rec['target']):>5} {str(rec['scale_frac']):>5} {str(rec['horizon']):>4} "
              f"{str(rec['np_bars']):>4} {str(rec['np_ret']):>6} "
              f"| {v['mean']:>8.4f} {rec['mean']:>8.4f} | {v['median']:>8.4f} {rec['median']:>8.4f} "
              f"| {v['win']:>7.4f} {rec['win']:>7.4f} | {v['ret_per_bar']:>8.5f} {rec['ret_per_bar']:>8.5f} "
              f"| {v['total_ret']:>8.1f} {rec['total_ret']:>8.1f}")


if __name__ == "__main__":
    main()
