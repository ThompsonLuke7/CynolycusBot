"""Risk overlays on the top-300 momentum sleeve, built from controls the system already has (2026-10-05).

Part 5 concluded that selection is not where the remaining gain is; the weakness is the -53% drawdown.
This adapts the existing risk machinery to the monthly sleeve and measures each piece on the point-in-time
universe with 05's conventions (decide at the close, trade the next open, daily marks, 4 monthly phases,
20 bp round trip). Stops are tested on daily CLOSES and exit at the next open.

Reused as-is:
  research/portfolio_lab/covariance.ledoit_wolf_asof   causal shrunk covariance (window 120, min 60)
  research/portfolio_lab/sizing.correlation_penalty    down-weight names correlated with the book (strength 1.0)
  research/portfolio_lab/sizing.portfolio_vol_scale    ex-ante vol of a weight vector
  SizingConfig.per_position_cap_pct = 0.12             cap on one name inside the sleeve
  core/live_4h_exec ExecPolicy stop_loss / trail_stop  the same two stop shapes: from entry, and from peak

Arms and parameters, fixed before the first run (none tuned on these results):
 sleeve alone (100% in the sleeve; unused or stopped capital sits in cash at 0%):
  base         equal-weight top 20 by 12-1 momentum among the 300 most liquid (= 05 mom_top300)
  corr_w       correlation-penalised weights in momentum-rank order, 12% cap
  vol20/vol30  scale the sleeve down so its ex-ante vol is at most 20% / 30% (never above 100% invested)
  stop10/20    exit a name after a close 10% / 20% below its entry (Han-Zhou-Zhu use 10% on monthly momentum)
  trail25      exit a name after a close 25% below its peak since entry
  dd25         hold cash for the month if the always-invested sleeve is >= 25% below its own peak
 blend (70% SPY + 30% sleeve; unused or stopped capital goes to SPY):
  b_base       fixed 70/30 (= 05 core_sat_300)
  b_corr       corr_w sleeve
  b_volman     sleeve share = 30% x (median past ex-ante vol / current), clipped to [10%, 50%]
  b_stop20     stop20 on the sleeve names
  b_stack      corr_w + vol-managed share + stop20 together
Verdict column: each arm's PAIRED active return against its own baseline (base or b_base).
 allocation dial (added after the first run, when no overlay helped): the plain sleeve at 10% .. 100% of the
  account with the rest in SPY. It is the same two assets in different proportions, not a new rule.

    .venv/bin/python research/long_horizon_discount_2026-09-25/16_risk_overlays.py
"""
from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from research.portfolio_lab.covariance import ledoit_wolf_asof  # noqa: E402
from research.portfolio_lab.sizing import SizingConfig, correlation_penalty, portfolio_vol_scale  # noqa: E402

_spec = spec_from_file_location("rb", HERE / "05_rotation_backtest.py")
rb = module_from_spec(_spec)
_spec.loader.exec_module(rb)
pb = rb.pb
CFG = SizingConfig()
N_TOP, N_LIQ, BLEND = 20, 300, 0.30
SLEEVE_ARMS = ["base", "corr_w", "vol20", "vol30", "stop10", "stop20", "trail25", "dd25"]
BLEND_ARMS = ["b_base", "b_corr", "b_volman", "b_stop20", "b_stack"]
DIAL = [0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00]
DIAL_ARMS = [f"share{int(x * 100)}" for x in DIAL]


def corr_weights(names: list[str], corr: pd.DataFrame | None) -> dict[str, float]:
    """Equal base x correlation penalty, added in momentum-rank order, then the 12% position cap."""
    raw, book = {}, {}
    for t in names:
        raw[t] = correlation_penalty(t, book, corr, strength=CFG.corr_penalty_strength)
        book[t] = None
    w = pd.Series(raw) / sum(raw.values())
    for _ in range(20):   # cap and hand the excess to the uncapped names
        over = w > CFG.per_position_cap_pct + 1e-12
        if not over.any():
            break
        excess = (w[over] - CFG.per_position_cap_pct).sum()
        w[over] = CFG.per_position_cap_pct
        w[~over] += excess * w[~over] / w[~over].sum()
    return w.to_dict()


def simulate(weights: list[dict], entries: list[int], O: np.ndarray, C: np.ndarray, col: dict, end: int,
             stop: float | None = None, trail: float | None = None, park: str = "CASH",
             tp: float | None = None) -> tuple[np.ndarray, float]:
    """05's simulate plus per-name exits. A name that breaches its stop, or reaches its take-profit `tp` (added
    2026-10-07 for 19), is sold at the next open and the proceeds are held in `park` until the next rebalance.
    Returns (close-marked equity, share of name-periods exited early)."""
    vals = np.full(end + 1, np.nan)
    V, w_old, n_pos, n_stopped = 1.0, None, 0, 0
    pk = col[park]
    for k, (w, e) in enumerate(zip(weights, entries)):
        nxt = entries[k + 1] if k + 1 < len(entries) else None
        last = nxt - 1 if nxt is not None else end
        if w_old is None:
            V *= 1 - rb.COST_RT / 2
        else:
            V *= 1 - 0.5 * sum(abs(w.get(x, 0) - w_old.get(x, 0)) for x in set(w) | set(w_old)) * rb.COST_RT
        keys = list(w)
        idx = np.array([col[x] for x in keys])
        wt = np.array([w[x] for x in keys])
        rel = C[e:last + 1, idx] / O[e, idx]
        nopen = O[nxt, idx] / O[e, idx] if nxt is not None else None
        parked = np.zeros(len(keys))
        for i, name in enumerate(keys):
            if name in ("SPY", "CASH") or (stop is None and trail is None and tp is None):
                continue
            n_pos += 1
            path = rel[:, i]
            hit = np.zeros(len(path), bool)
            if stop is not None:
                hit |= path <= 1 - stop
            if trail is not None:
                hit |= path <= np.maximum.accumulate(np.maximum(path, 1.0)) * (1 - trail)
            if tp is not None:
                hit |= path >= 1 + tp
            if not hit.any():
                continue
            x = e + int(np.argmax(hit)) + 1           # exit at the open after the breaching close
            if x > last:
                continue                              # breached on the period's last close: the rebalance exits it
            n_stopped += 1
            exit_val = O[x, idx[i]] / O[e, idx[i]] * (1 - rb.COST_RT)   # sell the name, buy the parking asset
            rel[x - e:, i] = exit_val * C[x:last + 1, pk] / O[x, pk]
            if nopen is not None:
                nopen[i] = exit_val * O[nxt, pk] / O[x, pk]
            parked[i] = 1.0
        vals[e:last + 1] = V * rel @ wt
        if nxt is not None:
            V *= nopen @ wt
            dw = wt * nopen / (nopen @ wt)
            w_old = {}
            for name, x, p in zip(keys, dw, parked):
                key = park if p else name
                w_old[key] = w_old.get(key, 0.0) + x
    return vals[entries[0]:], (n_stopped / n_pos if n_pos else 0.0)


def main() -> None:
    pb.BARS, pb.OUT = pb.PIT_BARS, pb.PIT_OUT
    p = pd.read_parquet(pb.OUT, columns=["date", "ticker", "mom_12_1", "dv_rank"])
    p = p[(p["date"] >= rb.START - pd.Timedelta(days=7)) & (p["dv_rank"] <= N_LIQ)].dropna(subset=["mom_12_1"])
    O_df, C_df = rb.price_matrices(sorted(set(p["ticker"]) | set(rb.BENCH)),
                                   pb.PIT_OUT.parent / "px_open_pit.parquet", pb.PIT_OUT.parent / "px_close_pit.parquet")
    C_df = C_df.ffill()
    O_df = O_df.fillna(C_df.shift(1))
    O_df["CASH"] = C_df["CASH"] = 1.0
    O_df = O_df[C_df.columns]
    cal, col = C_df.index, {t: i for i, t in enumerate(C_df.columns)}
    O, C, end = O_df.values, C_df.values, len(cal) - 1
    dec = pd.DatetimeIndex(sorted(p.loc[p["date"] >= rb.START, "date"].unique()))
    picks = {d: g.nlargest(N_TOP, "mom_12_1")["ticker"].tolist() for d, g in p.groupby("date")}

    # per-decision sleeve weights and ex-ante vol (covariance uses closes through the decision date)
    ew, cw, vol_ew, vol_cw = {}, {}, {}, {}
    for d in dec:
        names = picks[d]
        e = cal.searchsorted(d, side="right")
        asof = cal[e] if e <= end else d + pd.Timedelta(days=1)
        cov = ledoit_wolf_asof(C_df[names], asof=asof)
        ew[d] = {t: 1 / len(names) for t in names}
        cw[d] = corr_weights(names, cov.corr if cov is not None else None)
        for store, w in ((vol_ew, ew[d]), (vol_cw, cw[d])):   # target 1.0 with no clamp -> scale = 1 / vol
            store[d] = 1.0 / portfolio_vol_scale(w, cov.cov if cov is not None else None, target_vol=1.0, max_scale=1e9)
    vol_ew, vol_cw = pd.Series(vol_ew), pd.Series(vol_cw)
    med_ew, med_cw = vol_ew.expanding().median().shift(1), vol_cw.expanding().median().shift(1)  # past decisions only

    def scaled(w: dict, share: float, rest: str) -> dict:
        out = {t: x * share for t, x in w.items()}
        out[rest] = out.get(rest, 0.0) + 1 - share
        return out

    def managed_share(d, vol, med) -> float:
        return BLEND if not np.isfinite(med[d]) else float(np.clip(BLEND * med[d] / vol[d], 0.10, 0.50))

    k = rb.FREQS["monthly"]
    rets = {a: [] for a in SLEEVE_ARMS + BLEND_ARMS + DIAL_ARMS + ["spy"]}
    stats = {a: [] for a in SLEEVE_ARMS + BLEND_ARMS + DIAL_ARMS}
    extra = {a: [] for a in SLEEVE_ARMS + BLEND_ARMS + DIAL_ARMS}
    for ph in range(k):
        ds = dec[ph::k]
        entries = [i for i in (cal.searchsorted(d, side="right") for d in ds) if i <= end]
        ds = ds[:len(entries)]
        idx = cal[entries[0]:]
        b, _ = rb.simulate([{"SPY": 1.0}], entries[:1], O, C, col, end)
        v_base, _ = simulate([ew[d] for d in ds], entries, O, C, col, end)
        ref, _ = rb.simulate([ew[d] for d in ds], entries, O, C, col, end)
        if not np.allclose(v_base, ref):
            raise AssertionError("overlay simulator disagrees with 05's simulate on the no-overlay arm")
        dd = pd.Series(v_base / np.maximum.accumulate(v_base) - 1, index=idx)
        in_dd = [bool(dd.loc[:d].iloc[-1] <= -0.25) if (dd.index <= d).any() else False for d in ds]
        plans = {
            "base": ([ew[d] for d in ds], {}),
            "corr_w": ([cw[d] for d in ds], {}),
            "vol20": ([scaled(ew[d], min(1.0, 0.20 / vol_ew[d]), "CASH") for d in ds], {}),
            "vol30": ([scaled(ew[d], min(1.0, 0.30 / vol_ew[d]), "CASH") for d in ds], {}),
            "stop10": ([ew[d] for d in ds], {"stop": 0.10}),
            "stop20": ([ew[d] for d in ds], {"stop": 0.20}),
            "trail25": ([ew[d] for d in ds], {"trail": 0.25}),
            "dd25": ([{"CASH": 1.0} if out else ew[d] for d, out in zip(ds, in_dd)], {}),
            "b_base": ([scaled(ew[d], BLEND, "SPY") for d in ds], {}),
            "b_corr": ([scaled(cw[d], BLEND, "SPY") for d in ds], {}),
            "b_volman": ([scaled(ew[d], managed_share(d, vol_ew, med_ew), "SPY") for d in ds], {}),
            "b_stop20": ([scaled(ew[d], BLEND, "SPY") for d in ds], {"stop": 0.20, "park": "SPY"}),
            "b_stack": ([scaled(cw[d], managed_share(d, vol_cw, med_cw), "SPY") for d in ds], {"stop": 0.20, "park": "SPY"}),
        }
        plans |= {a: ([scaled(ew[d], x, "SPY") for d in ds], {}) for a, x in zip(DIAL_ARMS, DIAL)}
        rets["spy"].append(pd.Series(np.diff(np.r_[1.0, b]) / np.r_[1.0, b][:-1], index=idx))
        for a, (ws, kw) in plans.items():
            v, stopped = simulate(ws, entries, O, C, col, end, **kw)
            stats[a].append(rb.stats(v, b, idx))
            rets[a].append(pd.Series(np.diff(np.r_[1.0, v]) / np.r_[1.0, v][:-1], index=idx))
            invested = np.mean([sum(x for t, x in w.items() if t not in ("SPY", "CASH")) for w in ws])
            extra[a].append((invested, stopped))

    spy = pd.concat(rets["spy"], axis=1)
    s0 = pd.concat(rets["base"], axis=1).mean(axis=1)
    sp = spy.mean(axis=1)
    lines = [f"=== risk overlays on top-{N_TOP} momentum in the liquid-{N_LIQ} | point-in-time, monthly, 4 phases | "
             f"{dec.min().date()}..{cal[end].date()} ==="]
    lines.append(f"annualised daily-return mean / vol:  SPY {sp.mean() * 252:+.1%} / {sp.std() * np.sqrt(252):.1%}   "
                 f"sleeve {s0.mean() * 252:+.1%} / {s0.std() * np.sqrt(252):.1%}   correlation {s0.corr(sp):.2f}   "
                 f"sleeve ex-ante vol median {vol_ew.median():.0%} (range {vol_ew.min():.0%}-{vol_ew.max():.0%})")
    lines.append(f"{'arm':9s} {'CAGR med [min, max]':>24s} {'vol':>6s} {'Sharpe':>6s} {'MDD med/worst':>14s} "
                 f"{'vs SPY /yr [95% CI]':>24s} {'vs own baseline /yr [CI]':>26s} {'in sleeve':>9s} {'stopped':>7s}")
    for group, baseline in ((SLEEVE_ARMS, "base"), (BLEND_ARMS, "b_base"), (DIAL_ARMS, "share30")):
        base_r = pd.concat(rets[baseline], axis=1)
        for a in group:
            df = pd.DataFrame(stats[a])
            r = pd.concat(rets[a], axis=1)
            act = (r - spy).dropna().mean(axis=1)
            lo, hi = rb.block_ci(act)
            pair = (r - base_r).dropna().mean(axis=1)
            plo, phi = rb.block_ci(pair) if a != baseline else (0.0, 0.0)
            inv, stp = np.mean([x[0] for x in extra[a]]), np.mean([x[1] for x in extra[a]])
            lines.append(
                f"{a:9s} {df.cagr.median():+7.1%} [{df.cagr.min():+6.1%},{df.cagr.max():+6.1%}] "
                f"{r.mean(axis=1).std() * np.sqrt(252):6.1%} {df.sharpe.median():6.2f} {df.mdd.median():+6.1%}/{df.mdd.min():+6.1%} "
                f"{act.mean() * 252:+7.1%} [{lo:+6.1%},{hi:+6.1%}] {pair.mean() * 252:+8.1%} [{plo:+6.1%},{phi:+6.1%}] "
                f"{inv:9.0%} {stp:7.0%}")
        lines.append("")
    df = pd.DataFrame(stats["base"])
    lines.append(f"spy       {df.bcagr.median():+7.1%} {'':17s} {sp.std() * np.sqrt(252):6.1%} {df.bsharpe.median():6.2f} "
                 f"{pd.Series([((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min() for x in rets['spy']]).median():+6.1%}")
    yr = pd.DataFrame({a: (1 + rets[a][0]).groupby(rets[a][0].index.year).prod() - 1
                       for a in ["spy", "base", "stop20", "vol30", "dd25", "b_base", "b_volman", "b_stop20", "b_stack"]})
    lines.append("\ncalendar-year return, phase 0:\n" + yr.map(lambda v: f"{v:+.0%}").to_string())
    txt = "\n".join(lines)
    print(txt)
    (HERE / "16_results.txt").write_text(txt)


if __name__ == "__main__":
    main()
