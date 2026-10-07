"""The last 12 months of the top-300 momentum sleeve, month by month: returns, holdings and a chart (2026-10-06).

A BACKTEST replay on the point-in-time universe, not the forward record (that is shadow/ledger.csv, which
starts 2026-10-02). It uses the same rule and conventions as 05/16: every 4 weeks, at the Friday close, take the
20 highest 12-1 momentum names among the 300 most liquid stocks; trade at the next open; equal weight;
20 bp round trip; daily close marks.

Schedule: the 13 decisions on the live record's own 4-week grid (scripts/shadow/momentum_shadow_ledger.ANCHOR),
so month 12 ends where the forward ledger begins. One year is very sensitive to WHICH weeks the rebalances fall
on, so the other three possible weekly offsets are reported next to it as a range.

Lines:
  sleeve      the 20 leaders, re-picked every 4 weeks (month 0's leaders are the starting portfolio)
  month0_hold month 0's 20 leaders bought once and held all year with no rotation. The gap between this
              and `sleeve` is what the monthly rotation added
  blend70_30 / blend50_50   SPY + sleeve, reset to the target split every 4 weeks
  spy         buy and hold

    .venv/bin/python research/long_horizon_discount_2026-09-25/17_last_12_months.py
Outputs: 17_results.txt, 17_last_12_months.png
"""
from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from core.shared_plotting.figures import save_figure  # noqa: E402
from core.shared_plotting.theme import apply_mpl_defaults, style_figure  # noqa: E402
from scripts.shadow.momentum_shadow_ledger import ANCHOR, CADENCE_WEEKS  # noqa: E402

_spec = spec_from_file_location("rb", HERE / "05_rotation_backtest.py")
rb = module_from_spec(_spec)
_spec.loader.exec_module(rb)
pb = rb.pb
N_TOP, N_LIQ, N_PERIODS, STEP = 20, 300, 13, 4
LABELS = {"sleeve": "momentum sleeve (rotated every 4 weeks)", "month0_hold": "month-0 leaders, held all year",
          "blend50_50": "50% SPY + 50% sleeve", "blend70_30": "70% SPY + 30% sleeve", "spy": "SPY buy & hold"}


def main() -> None:
    pb.BARS, pb.OUT = pb.PIT_BARS, pb.PIT_OUT
    p = pd.read_parquet(pb.OUT, columns=["date", "ticker", "mom_12_1", "dv_rank"])
    dates = pd.DatetimeIndex(sorted(p["date"].unique()))
    p = p[(p["date"] >= dates[-(N_PERIODS * STEP + 8)]) & (p["dv_rank"] <= N_LIQ)].dropna(subset=["mom_12_1"])
    all_picks = {d: g.nlargest(N_TOP, "mom_12_1")[["ticker", "mom_12_1"]] for d, g in p.groupby("date")}
    O_df, C_df = rb.price_matrices(sorted({t for g in all_picks.values() for t in g["ticker"]} | set(rb.BENCH)),
                                   pb.PIT_OUT.parent / "px_open_pit.parquet", pb.PIT_OUT.parent / "px_close_pit.parquet")
    C_df = C_df.ffill()
    O_df = O_df.fillna(C_df.shift(1))
    O_df["CASH"] = C_df["CASH"] = 1.0
    O_df = O_df[C_df.columns]
    cal, col = C_df.index, {t: i for i, t in enumerate(C_df.columns)}
    O, C, end = O_df.values, C_df.values, len(cal) - 1

    # weekly offsets of the 4-week grid; offset 0 is the live ledger's schedule
    anchor_monday = pd.Timestamp(ANCHOR) - pd.Timedelta(days=ANCHOR.weekday())
    weeks_before = ((anchor_monday - (dates - pd.to_timedelta(dates.weekday, unit="D"))).days // 7)
    tradable = np.array([cal.searchsorted(d, side="right") <= end for d in dates])
    phase_lines = []
    for off in range(CADENCE_WEEKS):
        d_off = dates[(weeks_before % CADENCE_WEEKS == off) & (weeks_before > 0) & tradable][-N_PERIODS:]
        es = [int(cal.searchsorted(d, side="right")) for d in d_off]
        w = [{t: 1 / N_TOP for t in all_picks[d]["ticker"]} for d in d_off]
        v = rb.simulate(w, es, O, C, col, end)[0]
        b = rb.simulate([{"SPY": 1.0}], es[:1], O, C, col, end)[0]
        h = rb.simulate(w[:1], es[:1], O, C, col, end)[0]
        phase_lines.append(f"  first decision {d_off[0].date()} ({len(v)} sessions){' <- live schedule, charted' if off == 0 else '':28s} "
                           f"sleeve {v[-1] - 1:+6.1%}  max drawdown {(v / np.maximum.accumulate(v) - 1).min():+6.1%}  "
                           f"month-0 leaders held {h[-1] - 1:+6.1%}  SPY {b[-1] - 1:+6.1%}")
        if off == 0:
            dec = d_off
    picks = {d: all_picks[d] for d in dec}
    names = sorted({t for g in picks.values() for t in g["ticker"]})
    entries = [int(cal.searchsorted(d, side="right")) for d in dec]
    ew = [{t: 1 / N_TOP for t in picks[d]["ticker"]} for d in dec]
    plans = {
        "sleeve": (ew, entries),
        "month0_hold": (ew[:1], entries[:1]),
        "blend50_50": ([{"SPY": 0.5, **{t: 0.5 * w for t, w in x.items()}} for x in ew], entries),
        "blend70_30": ([{"SPY": 0.7, **{t: 0.3 * w for t, w in x.items()}} for x in ew], entries),
        "spy": ([{"SPY": 1.0}], entries[:1]),
    }
    idx = cal[entries[0]:]
    eq = pd.DataFrame({a: rb.simulate(ws, es, O, C, col, end)[0] for a, (ws, es) in plans.items()}, index=idx)

    # per-period, per-name returns: this entry open -> next entry open (the last period -> the last close)
    bounds = entries + [None]
    per_name = pd.DataFrame(index=names, columns=range(N_PERIODS), dtype=float)
    period_rows = []
    for k, d in enumerate(dec):
        e, nxt = bounds[k], bounds[k + 1]
        held = picks[d]["ticker"].tolist()
        exit_px = O[nxt, [col[t] for t in held]] if nxt is not None else C[end, [col[t] for t in held]]
        r = exit_px / O[e, [col[t] for t in held]] - 1
        per_name.loc[held, k] = r
        spy_exit = O[nxt, col["SPY"]] if nxt is not None else C[end, col["SPY"]]
        prev = set(picks[dec[k - 1]]["ticker"]) if k else set()
        period_rows.append({"month": k, "decision": d.date(), "entry": cal[e].date(),
                            "exit": (cal[nxt] if nxt is not None else cal[end]).date(),
                            "sleeve": r.mean(), "spy": spy_exit / O[e, col["SPY"]] - 1,
                            "added": sorted(set(held) - prev) if k else [], "dropped": sorted(prev - set(held)),
                            "best": held[int(np.argmax(r))], "best_r": r.max(),
                            "worst": held[int(np.argmin(r))], "worst_r": r.min()})
    per = pd.DataFrame(period_rows)

    lines = [f"=== last 12 months of the top-{N_TOP} momentum sleeve in the liquid-{N_LIQ} | BACKTEST, point-in-time universe | "
             f"{idx[0].date()} open .. {idx[-1].date()} close | 20 bp round trip ==="]
    lines.append(f"{'':38s} {'12m return':>10s} {'max drawdown':>12s} {'worst day':>9s}")
    for a in ("sleeve", "month0_hold", "blend50_50", "blend70_30", "spy"):
        v = eq[a]
        day = v.pct_change().min()
        lines.append(f"{LABELS[a]:38s} {v.iloc[-1] - 1:+10.1%} {(v / v.cummax() - 1).min():+12.1%} {day:+9.1%}")
    lines.append("\nthe same 12 months on each of the four possible weekly schedules (all marked to the same last close):")
    lines += phase_lines
    lines.append("\nper 4-week period on the live schedule (sleeve and SPY are before costs here; the totals above include costs):")
    lines.append(f"{'mo':>2s} {'entry':>10s} {'exit':>10s} {'sleeve':>8s} {'SPY':>7s} {'diff':>7s} {'kept':>4s}  best / worst / changes")
    for r in per.itertuples():
        kept = N_TOP - len(r.dropped) if r.month else N_TOP
        lines.append(f"{r.month:2d} {r.entry} {r.exit} {r.sleeve:+8.1%} {r.spy:+7.1%} {r.sleeve - r.spy:+7.1%} {kept:4d}  "
                     f"{r.best} {r.best_r:+.0%} / {r.worst} {r.worst_r:+.0%}"
                     + (f" | in: {' '.join(r.added)} | out: {' '.join(r.dropped)}" if r.month else ""))
    lines.append(f"\nperiods the sleeve beat SPY: {(per.sleeve > per.spy).sum()} of {len(per)} | names held at some point: "
                 f"{len(names)} | average names replaced per rebalance: {per.dropped.map(len)[1:].mean():.1f} of {N_TOP}")
    contrib = (per_name / N_TOP).sum(axis=1).sort_values()
    months = per_name.notna().sum(axis=1)
    lines.append("largest contributors (sum of 1/20 x period return; months held):  "
                 + "  ".join(f"{t} {contrib[t]:+.1%} ({months[t]})" for t in contrib.index[::-1][:8]))
    lines.append("largest detractors:  " + "  ".join(f"{t} {contrib[t]:+.1%} ({months[t]})" for t in contrib.index[:8]))
    lines.append(f"month-0 leaders ({dec[0].date()}): " + " ".join(picks[dec[0]]["ticker"]))
    lines.append(f"final-month leaders ({dec[-1].date()}): " + " ".join(picks[dec[-1]]["ticker"]))
    txt = "\n".join(lines)
    print(txt)
    (HERE / "17_results.txt").write_text(txt)

    # ---- chart: performance on top, holdings grid below ----
    first = per_name.notna().idxmax(axis=1)
    order = pd.DataFrame({"first": first, "months": months}).sort_values(["first", "months"], ascending=[True, False]).index
    grid = per_name.loc[order] * 100
    theme = apply_mpl_defaults(font_size=10)
    fig = plt.figure(figsize=(15, 6.2 + 0.235 * len(order)))
    gs = fig.add_gridspec(2, 1, height_ratios=[5.2, 0.235 * len(order)], hspace=0.09)
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    colors = {"sleeve": theme.blue, "month0_hold": theme.amber, "blend50_50": theme.purple,
              "blend70_30": theme.bull, "spy": theme.neutral}
    for a in ("sleeve", "month0_hold", "blend50_50", "blend70_30", "spy"):
        ax1.plot(eq.index, (eq[a] - 1) * 100, color=colors[a], lw=2.4 if a in ("sleeve", "spy") else 1.5,
                 label=f"{LABELS[a]}  {eq[a].iloc[-1] - 1:+.0%}")
    for e in entries:
        ax1.axvline(cal[e], color=theme.grid, lw=0.6, alpha=0.6)
    ax1.axhline(0, color=theme.muted_text, lw=0.6)
    ax1.set_ylabel("return since the month-0 entry (%)")
    ax1.set_title(f"Top-{N_TOP} momentum among the {N_LIQ} most liquid stocks: the last 12 months "
                  f"({idx[0].date()} to {idx[-1].date()})\nBACKTEST replay on the point-in-time universe, after 20 bp "
                  "round-trip costs. Vertical lines mark the 4-weekly rebalances.")
    ax1.legend(loc="upper left", fontsize=9.5, facecolor=theme.axes_bg, labelcolor=theme.text)

    cmap = LinearSegmentedColormap.from_list("pnl", [theme.bear, theme.axes_bg, theme.bull])
    cmap.set_bad(theme.figure_bg)
    ax2.imshow(np.ma.masked_invalid(grid.values), aspect="auto", cmap=cmap, vmin=-30, vmax=30, interpolation="nearest")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.values[i, j]
            if np.isfinite(v):
                ax2.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=6.8, color=theme.text)
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels([f"{t} *" if t in set(picks[dec[0]]["ticker"]) else t for t in order], fontsize=7.6)
    ax2.set_xticks(range(N_PERIODS))
    ax2.set_xticklabels([f"mo {k}\n{cal[e].strftime('%b %d')}\n{per.sleeve[k]:+.0%}" for k, e in enumerate(entries)], fontsize=8)
    ax2.set_xlabel("4-week holding period: month number, entry date, sleeve return that period.   "
                   "Cell = that stock's % return while held (blank = not held).   * = a month-0 leader")
    style_figure(fig, [ax1, ax2], theme)
    ax2.grid(False)
    ax2.set_xticks(np.arange(-0.5, N_PERIODS), minor=True)
    ax2.set_yticks(np.arange(-0.5, len(order)), minor=True)
    ax2.grid(which="minor", color=theme.figure_bg, linewidth=1.2)
    ax2.tick_params(which="minor", length=0)
    out = save_figure(fig, HERE / "17_last_12_months.png", close=True)
    print(out)


if __name__ == "__main__":
    main()
