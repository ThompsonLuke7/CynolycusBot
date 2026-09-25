"""Fundamentals as a QUALITY GATE (not ML inputs) on the simple rules.

quality = revenue growing (rev_yoy > 10% and TTM > 10%), gross margin >= 40%,
profitable or improving op margin. Thresholds are set by the growth-company brief and
were NOT tuned on these results. Same test years, metrics and 6-month block CIs as 03.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = spec_from_file_location("wf", HERE / "03_walkforward.py"); wf = module_from_spec(spec); spec.loader.exec_module(wf)
p = pd.read_parquet(wf.PANEL)
p = p[p["date"].dt.year >= 2019].copy()
q = ((p["rev_yoy"] > 0.10) & (p["rev_ttm_yoy"] > 0.10) & (p["gm_q"] >= 0.40)
     & ((p["opm_q"] > 0) | (p["opm_yoy_chg"] > 0)))
hand = ((p["min_low_vs_sma200_20"] > -0.05) & (p["tmo_min_10"] <= -9) & (p["days_since_tmo_turn"] <= 5))
comp = ((p["pct_above_sma200_252"] >= 0.75) & (p["sma200_slope_252"] > 0) & p["min_low_vs_ema200_20"].between(-0.05, 0.05))
mom = p["mom_12_1"].fillna(-9)
arms = {
    "quality_only (mom tiebreak)": q.astype(float) + 1e-6 * mom,
    "mom_12_1": mom,
    "quality x mom_12_1": q.astype(float) * 10 + mom,
    "hand_rule": hand.astype(float) + 1e-6 * mom,
    "quality x hand_rule": (q & hand).astype(float) * 2 + hand.astype(float) + 1e-6 * mom,
    "compounder_dip": comp.astype(float) + 1e-6 * mom,
    "quality x compounder_dip": (q & comp).astype(float) * 2 + comp.astype(float) + 1e-6 * mom,
}
p_all = p
out = []
for h in (126, 63):
    y = f"fwd_xs_{h}"
    p = p_all[p_all[y].notna()].copy()  # recent dates have no completed label yet
    spy = (p[f"fwd_ret_{h}"] - p[y]).groupby(p["date"]).median()
    out.append(f"\n=== hold {h} | quality share of universe {q.mean():.1%} ===")
    out.append(f"{'arm':28s} {'n_flag/date':>11s} {'top20 xs':>9s} {'[95% 6mo-block]':>20s} {'CAGR':>7s} {'SPY':>7s} {'MDD':>7s} {'worst yr':>9s}")
    for name, s in arms.items():
        s = s.loc[p.index]
        p["s"] = s
        m = wf.per_date_metrics(p, "s", y)
        ci = wf.block_ci(m["top20"])
        pf = wf.portfolio(p, "s", h, spy)
        flag = (s >= 1).groupby(p["date"]).sum().mean()
        worst = m["top20"].groupby(m.index.year).mean().min()
        out.append(f"{name:28s} {flag:11.1f} {m['top20'].mean():+9.2%} [{ci[0]:+6.2%}, {ci[1]:+6.2%}] "
                   f"{pf['cagr']:+7.1%} {pf['spy_cagr']:+7.1%} {pf['period_mdd']:+7.1%} {worst:+9.1%}")
txt = "\n".join(out); print(txt); (HERE / "04_results.txt").write_text(txt)
