"""Validation-only sensitivity study for early exits that never improve.

This does not modify any live policy.  It starts from the frozen id4 policy and
tests two pre-declared close-based fail-safes after N completed 4H bars:

* ``no_up_close``: no close is higher than the immediately previous close.
* ``never_profitable_close``: no close is higher than the entry close.

Existing hard-stop, trailing-stop, and target mechanics keep priority.  A
candidate that looks useful here still needs a fresh forward validation before
it can be considered for paper/live use.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/capstone"))
import exit_policy_segmentation as eps  # noqa: E402


WINDOWS = (2, 3, 4, 5, 6, 8, 10)
RULES = ("no_up_close", "never_profitable_close")
OUT_DIR = REPO / "research/capstone/never_profitable_exit"


def _qualifies(rule: str, entry: float, closes: np.ndarray) -> bool:
    """Return whether every completed close still satisfies the fail-safe."""
    if rule == "no_up_close":
        # closes includes the entry close as element zero.
        return not bool(np.any(closes[1:] > closes[:-1]))
    if rule == "never_profitable_close":
        return not bool(np.any(closes[1:] > entry))
    raise ValueError(f"unknown rule: {rule}")


def simulate(member: pd.DataFrame, *, rule: str | None = None,
             fail_after_bars: int | None = None, **cfg: float | int | None) -> pd.DataFrame:
    """Simulate frozen id4 mechanics with an optional completed-bar fail-safe.

    The rule is evaluated only at the close of bar ``fail_after_bars``.  It is
    therefore available at that decision time and never consults later bars.
    """
    if (rule is None) != (fail_after_bars is None):
        raise ValueError("rule and fail_after_bars must be supplied together")
    if fail_after_bars is not None and fail_after_bars < 1:
        raise ValueError("fail_after_bars must be positive")

    stop, trail, target, scale_frac, horizon, grace = (cfg.get(k) for k in
        ("stop", "trail", "target", "scale_frac", "horizon", "grace"))
    scale_frac = float(scale_frac if scale_frac is not None else 1.0)
    rows: list[dict[str, object]] = []
    for ticker, g in member.groupby("ticker"):
        bars = eps._bars(ticker)
        if bars is None:
            continue
        g = g.sort_values("timestamp")
        in_top = (g.set_index("timestamp")["in_top"]
                  .reindex(bars.index).fillna(False).astype(bool).to_numpy())
        close = bars["close"].to_numpy(dtype=float)
        high = bars["high"].to_numpy(dtype=float)
        low = bars["low"].to_numpy(dtype=float)
        i, n = 0, len(bars)
        while i < n - 1:
            if not in_top[i] or close[i] <= 0:
                i += 1
                continue
            entry = close[i]
            peak, realized, remaining = entry, 0.0, 1.0
            trimmed, out_ct, exit_ret, exit_reason = False, 0, None, "data_end"
            j = i + 1
            while j < n and (j - i) <= eps.be.MAX_HOLD:
                peak = max(peak, high[j])
                lo_ret, hi_ret = low[j] / entry - 1, high[j] / entry - 1
                if stop is not None and lo_ret <= -float(stop):
                    exit_ret, exit_reason = -float(stop), "hard_stop"
                    break
                if trail is not None and low[j] <= peak * (1 - float(trail)):
                    exit_ret, exit_reason = peak * (1 - float(trail)) / entry - 1, "trailing_stop"
                    break
                if target is not None and not trimmed and hi_ret >= float(target):
                    if scale_frac >= 1.0:
                        exit_ret, exit_reason = float(target), "target"
                        break
                    realized += scale_frac * float(target)
                    remaining, trimmed = 1.0 - scale_frac, True
                # The candidate is an end-of-completed-bar rule, after
                # established intrabar protection/profit-taking mechanics.
                if (rule is not None and j - i >= int(fail_after_bars)
                        and _qualifies(rule, entry, close[i:j + 1])):
                    exit_ret, exit_reason = close[j] / entry - 1, rule
                    break
                out_ct = out_ct + 1 if not in_top[j] else 0
                if grace is not None and out_ct > int(grace):
                    exit_ret, exit_reason = close[j] / entry - 1, "rank_drop"
                    break
                if horizon is not None and j - i >= int(horizon):
                    exit_ret, exit_reason = close[j] / entry - 1, "horizon"
                    break
                j += 1
            if exit_ret is None:
                j = min(j, n - 1)
                exit_ret = close[j] / entry - 1
            rows.append({
                "ticker": ticker, "entry_ts": bars.index[i],
                "ret": realized + remaining * exit_ret, "bars_held": j - i,
                "exit_reason": exit_reason,
            })
            i = min(j, n - 1) + 1
    return pd.DataFrame(rows)


def summarize(trades: pd.DataFrame) -> dict[str, float]:
    ret = trades["ret"]
    hold = trades["bars_held"].clip(lower=1)
    losers = ret[ret < 0]
    return {
        "n": len(trades), "mean": ret.mean(), "median": ret.median(),
        "win": (ret > 0).mean(), "total": ret.sum(),
        "p05": ret.quantile(.05), "p10": ret.quantile(.10),
        "loser_mean": losers.mean(), "worst": ret.min(),
        "avg_hold": hold.mean(), "ret_per_bar": (ret / hold).mean(),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Pre-declared validation-only window.  Do not read the frozen test split
    # while screening this family of candidate exits.
    streams = {m: eps.load_stream(m) for m in ("momentum", "htf", "meta")}
    frames, summary_rows = [], []
    for module, stream in streams.items():
        member = eps.member_from_stream(stream, eps.VAL_START, eps.VAL_END)
        variants: list[tuple[str, str | None, int | None]] = [("baseline_id4", None, None)]
        variants += [(f"{rule}_{bars}bars", rule, bars) for rule in RULES for bars in WINDOWS]
        for name, rule, bars in variants:
            tr = simulate(member, rule=rule, fail_after_bars=bars, **eps.POLICIES["id4"])
            tr["module"], tr["variant"], tr["rule"], tr["fail_after_bars"] = module, name, rule, bars
            frames.append(tr)
            row = {"module": module, "variant": name, "rule": rule, "fail_after_bars": bars}
            row.update(summarize(tr))
            row["fail_safe_exits"] = int((tr["exit_reason"] == rule).sum()) if rule else 0
            summary_rows.append(row)
            print(f"{module:9s} {name:34s} n={len(tr):4d} mean={row['mean']:+.4f} p05={row['p05']:+.4f}")
    summary = pd.DataFrame(summary_rows)
    trades = pd.concat(frames, ignore_index=True)
    summary.to_csv(OUT_DIR / "validation_summary.csv", index=False)
    trades.to_csv(OUT_DIR / "validation_trades.csv", index=False)
    print(f"saved {len(summary)} variants and {len(trades)} trade rows to {OUT_DIR}")


if __name__ == "__main__":
    main()
