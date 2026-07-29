"""The pre-registered arm grid. See HYPOTHESES.md — this file is the code
image of that document's "Primary grid" section and must not be extended
after results are seen.
"""
from __future__ import annotations

from research.pyramid_lab.engine import NO_PYRAMID, PyramidPolicy

BASELINE_ARM = "baseline"

# trigger label -> (trigger, level)
TRIGGERS = {
    "L10": ("level", 0.10),
    "L20": ("level", 0.20),
    "L30": ("level", 0.30),
    "RESEL": ("reselect", None),
}
ADD_FRACS = {"a50": 0.50, "a100": 1.00}
MAX_ADDS = (1, 2)
RESELECT_SPACING_BARS = 6  # 2 bars/trading-day in this dataset -> 3 trading days


def primary_arms() -> dict[str, PyramidPolicy]:
    """16 pyramiding arms + the no-add baseline, all on basis='entry'."""
    arms: dict[str, PyramidPolicy] = {BASELINE_ARM: NO_PYRAMID}
    for tlabel, (trigger, level) in TRIGGERS.items():
        for alabel, frac in ADD_FRACS.items():
            for m in MAX_ADDS:
                arms[f"{tlabel}_{alabel}_m{m}"] = PyramidPolicy(
                    trigger=trigger, level=level, add_frac=frac, max_adds=m,
                    spacing_bars=RESELECT_SPACING_BARS, basis="entry",
                )
    return arms


def blended_basis_arms(arm_ids: list[str]) -> dict[str, PyramidPolicy]:
    """SECONDARY sensitivity only: the same arms with stop/take-profit/horizon
    keyed to the blended cost basis instead of the original entry price. Unlike
    the primary grid this DOES change exit timing. Counted in the expanded FDR
    family when reported."""
    prim = primary_arms()
    return {f"{a}__blended": PyramidPolicy(
        trigger=prim[a].trigger, level=prim[a].level, add_frac=prim[a].add_frac,
        max_adds=prim[a].max_adds, spacing_bars=prim[a].spacing_bars, basis="blended",
    ) for a in arm_ids if a != BASELINE_ARM}
