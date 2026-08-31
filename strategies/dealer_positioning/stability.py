"""Does a gamma level survive its own model assumptions?

Gamma is not observed. It is produced by an option-pricing model fed an implied
volatility, and the chain's IV is itself a model output. So every wall, magnet,
and flip point inherits an assumption.

This module re-derives the levels under shocked volatility surfaces and reports
how far they move. The distinction that makes it worth doing:

    A level whose *location* is stable under a +/-20% IV shock is structural.
    A level that jumps two strikes when you nudge IV was an artifact of the
    volatility you happened to feed it.

Magnitudes are expected to move a lot under an IV shock -- gamma scales roughly
inversely with vol. Location stability is the useful signal, which is why the
reported fields are about *where* the level is, not how big it is.

Gamma is recomputed from contract rows via the project's BSM engine
(``research.options_lab.pricing``), never from the ladder's ``call_gamma`` /
``put_gamma`` columns -- those are cross-expiry means and are not per-contract
gamma (see ``levels.build_gamma_ladder``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from research.options_lab import pricing
from strategies.dealer_positioning.levels import build_gamma_ladder, _core_levels_from_ladder


# Multiplicative shocks applied to the observed IV surface.
DEFAULT_IV_SHOCKS: tuple[float, ...] = (0.8, 1.0, 1.2)

# Risk-free rate used for the reprice. The shock comparison is a *relative*
# exercise -- every surface uses the same rate -- so a reasonable constant is
# sufficient and avoids a curve dependency inside a stability metric.
DEFAULT_RATE = 0.04

# Trading days per year, matching the convention in research/options_lab.
DAYS_PER_YEAR = 365.0


@dataclass(frozen=True)
class StructuralStability:
    """How far each level travels across the shocked surfaces.

    Stability scores are in [0, 1]: 1.0 means the level did not move at all,
    0.0 means it moved by at least ``tolerance_pct`` of spot. ``None`` means the
    level could not be located on at least two surfaces, which is itself a
    reason not to trust it.
    """

    shocks: tuple[float, ...]
    call_wall_stability: float | None
    put_wall_stability: float | None
    gamma_flip_stability: float | None
    magnet_stability: float | None
    node_rank_stability: float | None
    estimated_net_gex_sensitivity: float | None
    call_wall_by_shock: dict[str, float | None]
    put_wall_by_shock: dict[str, float | None]
    gamma_flip_by_shock: dict[str, float | None]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _reprice_gamma(frame: pd.DataFrame, *, spot: float, shock: float, rate: float) -> pd.DataFrame:
    """Recompute per-contract gamma with IV scaled by ``shock``.

    Rows without a usable IV or DTE keep their vendor gamma: dropping them would
    change which strikes exist between surfaces, which would show up as
    instability that is really just missing data.
    """
    work = frame.copy()
    for col in ("iv", "dte", "strike", "gamma"):
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")

    iv = work["iv"].astype(float)
    # The chain reports IV in percent (77.3 == 77.3%); anything above 5 is
    # percent rather than a fraction.
    iv = np.where(iv > 5.0, iv / 100.0, iv)
    T = work["dte"].astype(float) / DAYS_PER_YEAR
    usable = pd.notna(iv) & pd.notna(T) & (T > 0) & (iv > 0) & work["strike"].notna()
    if not usable.any():
        return work

    rights = work.get("option_type")
    rights = ["C"] * len(work) if rights is None else [str(x).upper()[:1] or "C" for x in rights]
    sub = usable.to_numpy()
    greeks = pricing.bsm_greeks(
        S=float(spot),
        K=work.loc[usable, "strike"].astype(float).to_numpy(),
        T=T[usable].to_numpy(),
        r=float(rate),
        q=0.0,
        sigma=(iv[sub] * float(shock)),
        right=[r for r, keep in zip(rights, sub) if keep],
    )
    work.loc[usable, "gamma"] = np.asarray(greeks.gamma, dtype=float)
    return work


def _stability_from_levels(values: list[float | None], *, spot: float, tolerance_pct: float) -> float | None:
    present = [float(v) for v in values if v is not None and np.isfinite(v)]
    if len(present) < 2 or spot <= 0.0:
        return None
    spread_pct = (max(present) - min(present)) / float(spot)
    return float(max(0.0, 1.0 - spread_pct / float(tolerance_pct)))


def _node_rank_stability(ladders: list[pd.DataFrame], *, top_n: int = 10) -> float | None:
    """Spearman agreement between the top gamma strikes across surfaces.

    Compares the *ranking* of strikes by absolute exposure, so it answers "do
    the same strikes matter" rather than "is the number the same".
    """
    ranked = []
    for ladder in ladders:
        if ladder.empty or "total_abs_gex" not in ladder.columns:
            continue
        top = ladder.nlargest(top_n, "total_abs_gex")
        ranked.append({float(k): i for i, k in enumerate(top["strike"].tolist())})
    if len(ranked) < 2:
        return None
    scores = []
    base = ranked[0]
    for other in ranked[1:]:
        shared = sorted(set(base) & set(other))
        if len(shared) < 3:
            continue
        a = pd.Series([base[k] for k in shared])
        b = pd.Series([other[k] for k in shared])
        corr = a.corr(b, method="spearman")
        if pd.notna(corr):
            scores.append(float(corr))
    if not scores:
        return None
    # Map [-1, 1] onto [0, 1]; a negative rank correlation is maximal instability.
    return float(max(0.0, min(1.0, (sum(scores) / len(scores) + 1.0) / 2.0)))


def assess_stability(
    frame: pd.DataFrame,
    *,
    spot: float,
    magnet_quantile: float = 0.90,
    shocks: tuple[float, ...] = DEFAULT_IV_SHOCKS,
    rate: float = DEFAULT_RATE,
    tolerance_pct: float = 0.02,
) -> StructuralStability:
    """Recompute levels under each IV shock and score how far they move.

    ``frame`` is the contract-level frame (``levels.rows_to_frame`` schema).
    ``tolerance_pct`` is the movement, as a fraction of spot, at which a level
    scores zero stability -- 2% by default.
    """
    empty = StructuralStability(
        shocks=tuple(shocks),
        call_wall_stability=None,
        put_wall_stability=None,
        gamma_flip_stability=None,
        magnet_stability=None,
        node_rank_stability=None,
        estimated_net_gex_sensitivity=None,
        call_wall_by_shock={},
        put_wall_by_shock={},
        gamma_flip_by_shock={},
    )
    if frame is None or frame.empty or spot <= 0.0:
        return empty

    call_walls: dict[str, float | None] = {}
    put_walls: dict[str, float | None] = {}
    flips: dict[str, float | None] = {}
    magnets: list[float | None] = []
    net_gex: list[float] = []
    ladders: list[pd.DataFrame] = []

    for shock in shocks:
        shocked = _reprice_gamma(frame, spot=spot, shock=shock, rate=rate)
        ladder = build_gamma_ladder(shocked, spot=spot)
        if ladder.empty:
            continue
        core = _core_levels_from_ladder(ladder, float(spot), float(magnet_quantile))
        key = f"{shock:g}"
        call_walls[key] = core["call_wall"]
        put_walls[key] = core["put_wall"]
        flips[key] = core["gamma_flip"]
        magnets.append(core["nearest_magnet"])
        net_gex.append(float(core["total_gex"]))
        ladders.append(ladder)

    if not ladders:
        return empty

    base_abs = max(abs(v) for v in net_gex) if net_gex else 0.0
    sensitivity = None
    if len(net_gex) >= 2 and base_abs > 0.0:
        sensitivity = float((max(net_gex) - min(net_gex)) / base_abs)

    return StructuralStability(
        shocks=tuple(shocks),
        call_wall_stability=_stability_from_levels(list(call_walls.values()), spot=spot, tolerance_pct=tolerance_pct),
        put_wall_stability=_stability_from_levels(list(put_walls.values()), spot=spot, tolerance_pct=tolerance_pct),
        gamma_flip_stability=_stability_from_levels(list(flips.values()), spot=spot, tolerance_pct=tolerance_pct),
        magnet_stability=_stability_from_levels(magnets, spot=spot, tolerance_pct=tolerance_pct),
        node_rank_stability=_node_rank_stability(ladders),
        estimated_net_gex_sensitivity=sensitivity,
        call_wall_by_shock=call_walls,
        put_wall_by_shock=put_walls,
        gamma_flip_by_shock=flips,
    )
