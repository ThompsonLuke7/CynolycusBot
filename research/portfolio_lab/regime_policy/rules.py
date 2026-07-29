"""The 14 pre-registered rule variants from HYPOTHESES.md. Thresholds here
must match that file exactly -- this module is the executable form of the
pre-registration, not a place to retune."""
from __future__ import annotations

import numpy as np

from research.portfolio_lab.regime_policy.engine import BASE_TOP_K, WIDE_TOP_K, Rule

ADMISSION_SIZING_RULES: list[Rule] = [
    # H1 -- trade less / stay out
    Rule("H1-liq-1.0", "H1", admit=lambda r: r.rank_in_bar <= BASE_TOP_K and not (r.liquidity_stress_z > 1.0)),
    Rule("H1-liq-1.5", "H1", admit=lambda r: r.rank_in_bar <= BASE_TOP_K and not (r.liquidity_stress_z > 1.5)),
    Rule("H1-rv-1.0",  "H1", admit=lambda r: r.rank_in_bar <= BASE_TOP_K and not (r.spy_rv20_z > 1.0)),
    Rule("H1-rv-1.5",  "H1", admit=lambda r: r.rank_in_bar <= BASE_TOP_K and not (r.spy_rv20_z > 1.5)),

    # H2 -- size small / cash is your friend (admission unchanged from baseline)
    Rule("H2-liq", "H2", admit=lambda r: r.rank_in_bar <= BASE_TOP_K,
         size_mult=lambda r: float(np.clip(1 - 0.5 * r.liquidity_stress_z, 0.25, 1.5))),
    Rule("H2-risk", "H2", admit=lambda r: r.rank_in_bar <= BASE_TOP_K,
         size_mult=lambda r: float(np.clip(1 + 0.5 * r.risk_appetite_z, 0.25, 1.75))),

    # H3 -- respect the trend / don't time the bottom
    Rule("H3-trend",   "H3", admit=lambda r: r.rank_in_bar <= BASE_TOP_K and r.spy_trend_state > 0),
    Rule("H3-riskapp", "H3", admit=lambda r: r.rank_in_bar <= BASE_TOP_K and r.risk_appetite_z > 0),
    Rule("H3-combo",   "H3", admit=lambda r: r.rank_in_bar <= BASE_TOP_K and r.spy_trend_state > 0 and r.risk_appetite_z > 0),

    # H5 -- lean in when dispersion is wide
    Rule("H5-size", "H5", admit=lambda r: r.rank_in_bar <= BASE_TOP_K,
         size_mult=lambda r: 1.5 if r.sector_dispersion_z > 1.0 else 1.0),
    Rule("H5-breadth", "H5",
         admit=lambda r: r.rank_in_bar <= BASE_TOP_K or (r.rank_in_bar <= WIDE_TOP_K and r.sector_dispersion_z > 1.0)),
]

# H4 -- stop distance rules need a per-row sl_mult applied at RESOLUTION
# time (they change the trade's exit, not just admission/sizing), so they
# are kept separate from the admit/size Rule objects above and consumed by
# engine.resolve_variable_sl_candidates.
H4_SL_MULT_FNS: dict[str, callable] = {
    "H4-step-1.0": lambda r: 3.0 if r.spy_rv20_z > 1.0 else 5.0,
    "H4-step-0.5": lambda r: 3.0 if r.spy_rv20_z > 0.5 else 5.0,
    "H4-cont":     lambda r: float(np.clip(5.0 - 1.5 * max(r.spy_rv20_z, 0.0), 2.0, 5.0)),
}

ALL_RULE_IDS = [r.rule_id for r in ADMISSION_SIZING_RULES] + list(H4_SL_MULT_FNS.keys())
assert len(ALL_RULE_IDS) == 14, f"expected 14 pre-registered rule variants, got {len(ALL_RULE_IDS)}"
