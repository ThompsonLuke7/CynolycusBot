"""
Two-sleeve exit policy: tail-rider (id4) + harvester (g284), with an
approximate options-spread overlay on the harvester sleeve.

Sleeve assignment (val-selected, test-frozen, same discipline as the other
exit-policy scripts): mom_xs_rank at ENTRY predicts tail potential
(Spearman +0.33 vs 60-bar MFE on val — stronger than the earlier win/loss
screen's +0.16). Top-tercile mom_xs_rank at entry -> tail-rider sleeve (id4
params). Everyone else -> harvester sleeve (g284 params).

Options-spread overlay — READ THIS BEFORE TRUSTING THE NUMBERS:
This repo has NO historical options-chain data (no strikes, no IV surface, no
greeks) anywhere. The payoff functions below are a structural approximation:
linear interpolation between "worthless" and "max width" based on the
underlying's return relative to assumed strikes, using DISCLOSED, UNVALIDATED
assumptions for debit/credit as a fraction of spread width. This ignores time
decay curvature, IV changes, and assumes roughly linear spot-to-spread-value
sensitivity, which is only approximately true near the strikes. Treat this as
a relative-comparison sketch (does a spread wrapper help or hurt the
harvester's return SHAPE), not a tradeable P&L estimate. A real backtest needs
actual chain data.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/capstone/two_sleeve_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "signals/meta_context/meta_ranker"))
import backtest_exits as be  # noqa: E402

VAL_START = pd.Timestamp("2025-07-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-15", tz="UTC")
TEST_END = pd.Timestamp("2026-05-15", tz="UTC")

TAIL_POLICY = dict(stop=0.39, trail=None, target=0.30, scale_frac=0.16, horizon=53, grace=None)
HARVEST_POLICY = dict(stop=0.59, trail=None, target=0.07, scale_frac=1.0, horizon=60, grace=None)

MATRIX = REPO / "signals/meta_context/meta_ranker/meta_ranker_matrix.parquet"
SPLIT_FEATURE = "mom_xs_rank"

_BAR_CACHE: dict[str, pd.DataFrame | None] = {}


def _bars(ticker: str) -> pd.DataFrame | None:
    if ticker not in _BAR_CACHE:
        _BAR_CACHE[ticker] = be._ticker_path(ticker, None)
    return _BAR_CACHE[ticker]


def _load_member_window(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_parquet(be.SCORED).dropna(subset=["s_combo"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["rk"] = df.groupby("timestamp")["s_combo"].rank(ascending=False, method="first")
    df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
    df["in_top"] = df["rk"] <= be.TOPK
    return df[["timestamp", "ticker", "in_top"]]


def two_sleeve_trades(member: pd.DataFrame, feat_lookup: dict, split_thresh: float,
                       tail_cfg: dict, harvest_cfg: dict) -> list[dict]:
    """Same exit mechanic as backtest_exits.simulate(), but the policy applied to
    each trade is chosen AT ENTRY from that entry's own feature value."""
    out = []
    for ticker, g in member.groupby("ticker"):
        g = g.sort_values("timestamp")
        bars = _bars(ticker)
        if bars is None:
            continue
        m = g.set_index("timestamp")["in_top"].reindex(bars.index).fillna(False).astype(bool).values
        close, high, low = bars["close"].values, bars["high"].values, bars["low"].values
        n = len(bars)
        i = 0
        while i < n - 1:
            if not m[i]:
                i += 1
                continue
            entry = close[i]
            entry_ts = bars.index[i]
            if entry <= 0:
                i += 1
                continue
            fval = feat_lookup.get((entry_ts, ticker))
            sleeve = "tail" if (fval is not None and fval >= split_thresh) else "harvest"
            cfg = tail_cfg if sleeve == "tail" else harvest_cfg
            stop, trail, target, scale_frac, horizon, grace = (cfg.get(k) for k in
                ("stop", "trail", "target", "scale_frac", "horizon", "grace"))
            scale_frac = scale_frac if scale_frac is not None else 1.0
            peak = entry
            realized = 0.0
            remaining = 1.0
            trimmed = False
            out_ct = 0
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
                out_ct = out_ct + 1 if not m[j] else 0
                if grace is not None and out_ct > grace:
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
            out.append(dict(ticker=ticker, entry_ts=entry_ts, sleeve=sleeve,
                             ret=total, bars_held=min(j, n - 1) - i, feat=fval))
            i = min(j, n - 1) + 1
    return out


# ---- approximate spread payoff models (see module docstring caveats) ----

def bull_call_debit_spread(underlying_ret: float, *, width_pct: float, debit_frac: float = 0.45) -> float:
    """Long call ~ATM (K1=entry), short call at K2=entry*(1+width_pct).
    Linear intrinsic-value interpolation between K1 and K2, capped both sides."""
    frac_of_width = np.clip(underlying_ret / width_pct, 0.0, 1.0)
    return (frac_of_width - debit_frac) / debit_frac


def bull_put_credit_spread(underlying_ret: float, *, short_buffer: float = 0.05,
                            protect_width: float = 0.15, credit_frac: float = 0.30) -> float:
    """Short put at K1=entry*(1-short_buffer), long put at K2=entry*(1-short_buffer-protect_width).
    Full credit kept if price stays above K1; linear loss to max loss at/below K2."""
    width = protect_width
    loss_frac = np.clip((-short_buffer - underlying_ret) / protect_width, 0.0, 1.0)
    credit = credit_frac * width
    max_loss = width - credit
    pnl = credit - loss_frac * width
    return pnl / max_loss if max_loss > 0 else 0.0


def main() -> None:
    val_member = _load_member_window(VAL_START, VAL_END)
    test_member = _load_member_window(VAL_END, TEST_END)

    matrix = pd.read_parquet(MATRIX)
    feat_series = matrix[SPLIT_FEATURE]
    feat_lookup = feat_series.to_dict()  # (timestamp, ticker) -> value

    # ---- pick the split threshold on VAL only (top-tercile of mom_xs_rank) ----
    val_vals = []
    for ticker, g in val_member[val_member["in_top"]].groupby("ticker"):
        for ts in g["timestamp"]:
            v = feat_lookup.get((ts, ticker))
            if v is not None:
                val_vals.append(v)
    split_thresh = float(np.quantile(val_vals, 2 / 3))
    print(f"split feature: {SPLIT_FEATURE}  val top-tercile threshold: {split_thresh:.4f}  "
          f"(n={len(val_vals)} entry-feature values on val)")

    for label, member in (("VAL", val_member), ("TEST (frozen)", test_member)):
        trades = two_sleeve_trades(member, feat_lookup, split_thresh, TAIL_POLICY, HARVEST_POLICY)
        df = pd.DataFrame(trades)
        print(f"\n=== {label}: {len(df)} trades ({(df['sleeve']=='tail').sum()} tail / "
              f"{(df['sleeve']=='harvest').sum()} harvest) ===")

        harv = df[df["sleeve"] == "harvest"].copy()
        harv["call_spread_ret"] = harv["ret"].apply(
            lambda r: bull_call_debit_spread(r, width_pct=HARVEST_POLICY["target"]))
        harv["put_spread_ret"] = harv["ret"].apply(bull_put_credit_spread)

        def stats(s: pd.Series) -> dict:
            return dict(n=len(s), mean=s.mean(), median=s.median(), win=(s > 0).mean(),
                        total=s.sum(), std=s.std())

        tail = df[df["sleeve"] == "tail"]
        rows = {
            "tail-rider sleeve (id4, shares)": stats(tail["ret"]),
            "harvest sleeve (g284, shares, naked)": stats(harv["ret"]),
            "harvest sleeve (call debit spread, approx)": stats(harv["call_spread_ret"]),
            "harvest sleeve (put credit spread, approx)": stats(harv["put_spread_ret"]),
        }
        print(f"{'sleeve':44} {'n':>5} {'mean':>8} {'median':>8} {'win':>7} {'total':>8}")
        for name, s in rows.items():
            print(f"{name:44} {s['n']:5d} {s['mean']*100:7.2f}% {s['median']*100:7.2f}% "
                  f"{s['win']*100:6.1f}% {s['total']*100:7.1f}%")

        # portfolio-level: tail sleeve (shares) + harvest sleeve under each variant
        print("\n  combined portfolio (tail id4-shares + harvest variant):")
        for variant, col in [("naked shares", "ret"), ("call debit spread", "call_spread_ret"),
                              ("put credit spread", "put_spread_ret")]:
            combined = pd.concat([tail["ret"], harv[col]])
            s = stats(combined)
            print(f"    harvest={variant:20s} n={s['n']:4d} mean={s['mean']*100:6.2f}% "
                  f"median={s['median']*100:6.2f}% win={s['win']*100:5.1f}% total={s['total']*100:7.1f}%")

        if label.startswith("TEST"):
            df.to_csv(REPO / "research/capstone/two_sleeve_test_trades.csv", index=False)
            harv.to_csv(REPO / "research/capstone/two_sleeve_harvest_spreads_test.csv", index=False)
        else:
            df.to_csv(REPO / "research/capstone/two_sleeve_val_trades.csv", index=False)


if __name__ == "__main__":
    main()
