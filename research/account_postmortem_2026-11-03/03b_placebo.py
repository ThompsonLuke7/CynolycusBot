"""Placebo for 03_counterfactuals: refuse the same NUMBER of entries at random.

With a negative per-trade edge, any rule that trades less loses less. A rule
only shows timing or selection skill to the extent it beats refusing the same
count of entries chosen at random (30 seeds). Random refusal draws from the
entries that actually occurred, uniformly, independent of time.

    PYTHONPATH=. .venv/bin/python research/account_postmortem_2026-11-03/03b_placebo.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location("cf", Path(__file__).with_name("03_counterfactuals.py"))
cf = importlib.util.module_from_spec(_spec)
sys.modules["cf"] = cf  # dataclasses resolve their module through sys.modules
_spec.loader.exec_module(cf)

SEEDS = 30


def main() -> None:
    c, a, ep, _, _ = cf.load()
    res = pd.read_csv(cf.OUT / "counterfactuals.csv")
    actual = a.equity_t1.iloc[-1]
    first_t0 = a.t0.iloc[0]
    candidates = [e for e in ep.index if ep.loc[e, "entry_ts"] > first_t0 and not e.endswith("@pre")]
    rng = np.random.default_rng(20261103)
    rows = []
    for _, r in res[res.entries_refused > 0].iterrows():
        n = int(r.entries_refused)
        saved = []
        for _ in range(SEEDS):
            drop = set(rng.choice(candidates, size=min(n, len(candidates)), replace=False))
            saved.append(cf.simulate(c, a, ep, accept=lambda e, s, drop=drop: e.name not in drop)["final"] - actual)
        saved = np.array(saved)
        rows.append({"rule": r.rule, "refused": n, "rule_saved": r.vs_actual,
                     "random_saved_mean": saved.mean(), "random_p05": np.percentile(saved, 5),
                     "random_p95": np.percentile(saved, 95),
                     "skill_vs_random": r.vs_actual - saved.mean(),
                     "beats_95pct": r.vs_actual > np.percentile(saved, 95)})
    out = pd.DataFrame(rows)
    out.to_csv(cf.OUT / "counterfactuals_placebo.csv", index=False)
    show = out.copy()
    for col in ("rule_saved", "random_saved_mean", "random_p05", "random_p95", "skill_vs_random"):
        show[col] = show[col].map(lambda v: f"{v:+,.0f}")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
