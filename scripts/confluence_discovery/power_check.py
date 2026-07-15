"""Power check for the confluence search gauntlet (research tooling validation).

Plants a synthetic PURE interaction into the real dataset: a Bernoulli outcome with
base rate ~ the real target's, boosted by `boost` percentage points ONLY where the
planted pair (A & B) co-occurs — no marginal main effects. Then runs the exact same
pair-mining gauntlet (monthly block test, BH-FDR across all pairs, val replication)
and reports whether the planted pair survives.

If the gauntlet catches planted boosts at realistic effect sizes, a null result on
the real targets is evidence of absence-of-signal, not absence-of-power.

Usage:
  .venv/bin/python scripts/confluence_discovery/power_check.py \
      [--pair near_52w_high short_pct_high] [--boosts 0.03 0.05 0.10] [--seed 7]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from search import (CONDITIONS, bh_fdr, build_condition_masks, eval_combo,
                    load_universe, monthly_block_test, temporal_split)


def run_once(df, masks, is_train, is_val, target: str, min_n: int) -> pd.DataFrame:
    tr, va = df[is_train], df[is_val]
    base_tr, base_va = tr[target].mean(), va[target].mean()
    names = list(masks.columns)
    rows = []
    for i, a in enumerate(names):
        ax_a = CONDITIONS[a][0]
        ma = masks[a].to_numpy()
        for b in names[i + 1:]:
            if CONDITIONS[b][0] == ax_a:
                continue
            mb = masks[b].to_numpy()
            joint = ma & mb
            jt = joint & is_train
            if jt.sum() < min_n:
                continue
            ev = eval_combo(tr, target, jt[is_train], [ma[is_train], mb[is_train]], base_tr)
            t, p1, pos, n_m = monthly_block_test(tr, jt[is_train], ma[is_train], mb[is_train], target)
            ev.update({"A": a, "B": b, "t_month": t, "p_one": p1,
                       "month_pos_frac": pos, "n_months": n_m})
            evv = eval_combo(va, target, joint[is_val], [ma[is_val], mb[is_val]], base_va)
            ev.update({f"val_{k}": v for k, v in evv.items()})
            rows.append(ev)
    res = pd.DataFrame(rows)
    res["q_fdr"] = bh_fdr(res["p_one"])
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", nargs=2, default=["near_52w_high", "short_pct_high"])
    ap.add_argument("--boosts", nargs="+", type=float, default=[0.03, 0.05, 0.10])
    ap.add_argument("--base-rate", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-n", type=int, default=300)
    args = ap.parse_args()

    df = load_universe("meta_upside", 0.40, False)
    val_cut, test_cut = temporal_split(df)
    is_train = (df["timestamp"] < val_cut).to_numpy()
    is_val = ((df["timestamp"] >= val_cut) & (df["timestamp"] < test_cut)).to_numpy()
    masks, _ = build_condition_masks(df, pd.Series(is_train, index=df.index))

    a, b = args.pair
    planted = masks[a].to_numpy() & masks[b].to_numpy()
    print(f"planted pair: {a} & {b} | joint train n={(planted & is_train).sum():,} "
          f"val n={(planted & is_val).sum():,}")

    rng = np.random.default_rng(args.seed)
    for boost in args.boosts:
        p = np.full(len(df), args.base_rate)
        p[planted] += boost
        df["y_synth"] = rng.binomial(1, p)
        res = run_once(df, masks, is_train, is_val, "y_synth", args.min_n)
        row = res[(res["A"] == a) & (res["B"] == b)]
        if row.empty:
            row = res[(res["A"] == b) & (res["B"] == a)]
        r = row.iloc[0]
        caught = (r["q_fdr"] <= 0.10) and (r["delta_pp"] > 0) and (r["month_pos_frac"] >= 0.625) \
            and (r["val_delta_pp"] > 0)
        n_false = int(((res["q_fdr"] <= 0.10) & ~((res["A"] == a) & (res["B"] == b))
                       & ~((res["A"] == b) & (res["B"] == a))).sum())
        print(f"boost=+{boost*100:.0f}pp -> q_fdr={r['q_fdr']:.4f} delta_pp={r['delta_pp']:.2f} "
              f"month_pos={r['month_pos_frac']:.2f} val_delta_pp={r['val_delta_pp']:.2f} "
              f"| SURVIVES GAUNTLET: {caught} | other q<=0.1 pairs (false pos): {n_false}")


if __name__ == "__main__":
    main()
