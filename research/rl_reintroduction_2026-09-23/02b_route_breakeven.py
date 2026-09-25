"""Stage 2 addendum -- make the route arm decision-useful, and check its own scale.

The main script's option LEVELS must not be read as a finding: IV is set to trailing
realised vol and held constant, and the 23.6pp round-trip spread was calibrated on our
ACTUAL fills, which were cheap short-dated contracts. AGENTS.md's verification rule is
explicit that a calibrated ratio must be shown scale-invariant before use outside its
regime. This script does three things instead:

  1. characterises the pick set's VOLATILITY, because the R-unit and percent metrics
     disagree by ~4x and that gap is the vol tilt;
  2. converts the spread into the number that actually decides the route -- the BREAKEVEN
     UNDERLYING MOVE, spread / leverage -- and compares it to the measured edge;
  3. reports how the breakeven moves across the IV range, i.e. whether the answer is
     stable or an artifact of one IV assumption.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import panel as P  # noqa: E402
from research.options_lab.pricing import bsm_greeks, bsm_price  # noqa: E402

EVAL_H, MIN_XS, TOP_K = 10, 200, 3
OPT_DTE, OPT_R, OPT_Q = 30, 0.045, 0.0
SPREAD_RT = 0.236
DATA = HERE / "data"


def leverage(sigma: float, moneyness: float = 1.0, dte: int = OPT_DTE) -> tuple[float, float]:
    """(elasticity, premium as a fraction of spot) for a call at S=1."""
    T = dte / 365.0
    c = float(np.asarray(bsm_price(1.0, moneyness, T, OPT_R, OPT_Q, sigma, "call"), float))
    d = float(np.asarray(bsm_greeks(1.0, moneyness, T, OPT_R, OPT_Q, sigma, "call").delta, float))
    return (d / c if c > 1e-9 else np.nan), c


def main() -> None:
    out: dict = {}
    pan = P.load(require=["r_10", "mom_score"], drop_flagged=True)
    pan = pan[pan["xs_size"] >= MIN_XS]
    _, va, _, bounds = P.split(pan)
    out["bounds"] = bounds

    r = va.groupby("timestamp")["mom_score"].rank(ascending=False, method="first")
    pick = va[r <= TOP_K]
    print("=" * 78 + "\n1. what the deployed score actually picks (VALIDATION, k=3)\n" + "=" * 78)
    comp = pd.DataFrame({
        "metric": ["atr_pct", "past_vol_20 (daily sigma)", "log_dollar_vol_20", "beta_60"],
        "picks_median": [pick["atr_pct"].median(), pick["past_vol_20"].median(),
                         pick["log_dollar_vol_20"].median(), pick["beta_60"].median()],
        "universe_median": [va["atr_pct"].median(), va["past_vol_20"].median(),
                            va["log_dollar_vol_20"].median(), va["beta_60"].median()],
    })
    comp["ratio"] = comp["picks_median"] / comp["universe_median"]
    out["pick_profile"] = comp.to_dict("records")
    print(comp.round(4).to_string(index=False))
    print(f"\nmean fwdret_{EVAL_H} picks {pick[f'fwdret_{EVAL_H}'].mean():.2%} "
          f"vs universe {va[f'fwdret_{EVAL_H}'].mean():.2%}")
    print(f"mean r_{EVAL_H}      picks {pick[f'r_{EVAL_H}'].mean():.3f}R "
          f"vs universe {va[f'r_{EVAL_H}'].mean():.3f}R")
    print("\nThe percent and R numbers disagree because the picks are a HIGH-VOL slice:\n"
          "dividing by ATR shrinks a big percent move on a volatile name. Both are true;\n"
          "the R number is the risk-adjusted one and is the pre-registered metric.")

    print("\n" + "=" * 78 + "\n2. BREAKEVEN underlying move for the option route\n" + "=" * 78)
    print("elasticity = delta * S / premium, i.e. the % option move per 1% underlying move\n"
          "at inception. breakeven = round-trip spread / elasticity.\n")
    rows = []
    for iv in (0.30, 0.50, 0.75, 1.00, 1.50, 2.00):
        for mny in (0.95, 1.00, 1.05):
            el, prem = leverage(iv, mny)
            rows.append(dict(iv=iv, moneyness=mny, premium_pct_of_spot=prem * 100,
                             elasticity=el, breakeven_underlying_pct=SPREAD_RT / el * 100))
    bt = pd.DataFrame(rows)
    out["breakeven"] = bt.to_dict("records")
    print(bt.pivot_table(index="iv", columns="moneyness",
                         values="breakeven_underlying_pct").round(2).to_string())
    print("\n(cells are the underlying % move needed over the hold just to cover the\n"
          " measured 23.6pp round-trip spread -- before any theta.)")
    print("\npremium as % of spot, same grid (this is why a flat pp spread is NOT\n"
          "scale-invariant: a 200%-IV contract costs 6x a 30%-IV one):")
    print(bt.pivot_table(index="iv", columns="moneyness",
                         values="premium_pct_of_spot").round(2).to_string())

    print("\n" + "=" * 78 + "\n3. the comparison that decides the route\n" + "=" * 78)
    iv_pick = float((pick["past_vol_20"] * np.sqrt(252.0)).median())
    el, prem = leverage(min(max(iv_pick, 0.08), 3.0))
    be = SPREAD_RT / el * 100
    # the measured EXCESS, converted to percent at the picks' own ATR
    exc_R = 0.4412   # k=3 incumbent_mom, from 02_decidability.py section A
    exc_pct = exc_R * P.K_RISK * float(pick["atr_pct"].median()) * 100
    out["decision"] = {"pick_median_iv_ann": iv_pick, "elasticity": el,
                       "premium_pct_of_spot": prem * 100,
                       "breakeven_underlying_pct": be,
                       "measured_excess_R": exc_R, "measured_excess_pct": exc_pct,
                       "clears_breakeven": bool(exc_pct > be)}
    print(f"picks' median annualised trailing vol   {iv_pick:.1%}")
    print(f"30-DTE ATM elasticity at that vol       {el:.2f}x")
    print(f"premium as a share of spot              {prem:.1%}")
    print(f"BREAKEVEN underlying move over the hold {be:.2f}%")
    print(f"measured top-3 EXCESS (k=3, R units)    {exc_R:.4f}R")
    print(f"  ... in percent at the picks' own ATR  {exc_pct:.2f}%")
    print(f"\nclears the spread on EXCESS alone?      {'YES' if exc_pct > be else 'NO'}")
    print("\nNOTE the excess is the edge over the bar's own mean -- a long option also\n"
          "collects the bar's mean move, so the relevant total is excess + bar drift, and\n"
          "the bar drift is a market-direction bet, not an edge.")

    DATA.mkdir(exist_ok=True)
    (DATA / "stage2b_route_breakeven.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {DATA / 'stage2b_route_breakeven.json'}")


if __name__ == "__main__":
    main()
