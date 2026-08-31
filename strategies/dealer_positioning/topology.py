"""Unsigned gamma topology, kept separate from the estimated signed exposure.

Why this module exists
----------------------
``levels.py`` computes two things and reports them as one set of numbers:

1. **Where option gamma sits** -- a strike ladder built from open interest and
   per-contract gamma. This is close to observable: the chain tells us the open
   interest and the greeks, so the *location* and *magnitude* of gamma
   concentration is high confidence.
2. **Which side the dealer is assumed to hold** -- ``call_gex`` is signed
   positive and ``put_gex`` negative, which is an assumption about who owns the
   open interest, not an observation. Nothing in an option chain reveals the
   dealer's net position.

Blending those into one ``net_gex`` number lets a low-confidence inference
contaminate a high-confidence measurement. This module keeps them apart:
``GammaTopology`` carries only unsigned structure, and
``EstimatedSignedExposure`` carries the inferred side with the convention that
produced it named in the artifact.

Nothing here computes new market information. Every quantity is a regrouping of
values ``levels.build_gamma_ladder`` already produces, which is why this change
requires no validation of its own -- it adds no claim.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


# The sign assumption baked into ``levels.build_gamma_ladder``: all call open
# interest is treated as dealer-long gamma and all put open interest as
# dealer-short. Version it so an artifact can never be read as convention-free.
SIGN_CONVENTION = "oi_calls_long@1"

# Distance bands (as a fraction of spot) for local gamma density.
DENSITY_BANDS = (0.01, 0.025, 0.05)

# Expiry buckets, in days to expiry. Collapsing a 0DTE gamma pile and a 60DTE
# gamma pile into one number describes two unrelated structures as if they were
# the same one.
EXPIRY_BUCKETS: tuple[tuple[str, int, float], ...] = (
    ("d0", 0, 0),
    ("d1_2", 1, 2),
    ("d3_7", 3, 7),
    ("d8_30", 8, 30),
    ("d30_plus", 31, math.inf),
)

NODE_QUANTILE = 0.90


@dataclass(frozen=True)
class GammaTopology:
    """Unsigned gamma structure: where gamma is, not who owns it.

    Every field here survives the sign question. A large gamma concentration at
    a strike is a fact about the chain regardless of which side of it dealers
    are on, which is why this object is treated as higher confidence than
    :class:`EstimatedSignedExposure`.
    """

    timestamp: str
    symbol: str
    spot: float
    total_abs_gamma: float
    gamma_density_1pct: float | None
    gamma_density_2_5pct: float | None
    gamma_density_5pct: float | None
    gamma_concentration: float | None
    gamma_entropy: float | None
    node_count: int
    nearest_node: float | None
    nearest_node_distance_pct: float | None
    first_node_above: float | None
    first_node_below: float | None
    void_above_width_pct: float | None
    void_below_width_pct: float | None
    expiry_bucket_shares: dict[str, float] = field(default_factory=dict)
    zero_dte_gamma_share: float | None = None
    short_gamma_share: float | None = None
    weekly_gamma_share: float | None = None
    gamma_term_slope: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EstimatedSignedExposure:
    """Gamma exposure after a *sign assumption*, with that assumption named.

    Field names carry the ``estimated_`` prefix deliberately. A downstream model
    that reads ``estimated_net_gex`` cannot mistake it for a measured dealer
    position the way a bare ``net_gex`` invites.
    """

    timestamp: str
    symbol: str
    spot: float
    sign_convention: str
    estimated_net_gex: float
    estimated_dealer_imbalance: float | None
    estimated_call_wall: float | None
    estimated_put_wall: float | None
    estimated_gamma_flip: float | None
    estimated_call_gex_total: float
    estimated_put_gex_total: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Shared distribution math
#
# These were private helpers in ``capture_historical_snapshots``; they are the
# canonical implementations now so the nightly snapshot path and the topology
# builder cannot drift apart.
# ---------------------------------------------------------------------------


def concentration_index(weights: pd.Series) -> float | None:
    """Herfindahl index of the weight distribution (1.0 = one strike holds all)."""
    clean = pd.to_numeric(weights, errors="coerce").fillna(0.0).abs()
    total = float(clean.sum())
    if total <= 0.0:
        return None
    shares = clean / total
    return float((shares**2).sum())


def entropy(weights: pd.Series) -> float | None:
    """Shannon entropy of the weight distribution, in nats."""
    clean = pd.to_numeric(weights, errors="coerce").fillna(0.0).abs()
    total = float(clean.sum())
    if total <= 0.0:
        return None
    shares = clean[clean > 0.0] / total
    if shares.empty:
        return None
    return float(-(shares * shares.apply(math.log)).sum())


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


def _density_share(ladder: pd.DataFrame, *, spot: float, band: float) -> float | None:
    total = float(ladder["total_abs_gex"].abs().sum())
    if total <= 0.0:
        return None
    inside = ladder[(ladder["strike"] - spot).abs() <= spot * band]
    return float(inside["total_abs_gex"].abs().sum() / total)


def _nodes(ladder: pd.DataFrame, *, quantile: float) -> pd.DataFrame:
    weights = ladder["total_abs_gex"].abs()
    if weights.empty or float(weights.max()) <= 0.0:
        return ladder.iloc[0:0]
    threshold = float(weights.quantile(quantile))
    if threshold <= 0.0:
        threshold = float(weights.max())
    return ladder[weights >= threshold]


def _largest_gap_pct(strikes: pd.Series, *, spot: float) -> float | None:
    """Widest span between consecutive nodes, as a fraction of spot.

    A wide span is a region with little gamma to slow price down -- the
    "vacuum"/"void" idea -- and it is measured without reference to sign.
    """
    ordered = sorted(float(x) for x in strikes.dropna().unique())
    if len(ordered) < 2 or spot <= 0.0:
        return None
    gaps = [b - a for a, b in zip(ordered, ordered[1:])]
    return float(max(gaps) / spot)


def build_gamma_topology(
    ladder: pd.DataFrame,
    *,
    symbol: str,
    spot: float,
    timestamp: str,
    frame: pd.DataFrame | None = None,
    node_quantile: float = NODE_QUANTILE,
) -> GammaTopology:
    """Build the unsigned view from a gamma ladder.

    ``frame`` is the optional contract-level frame (one row per contract, with a
    ``dte`` column). It is only needed for the expiry-bucket shares; the rest is
    derived from the ladder.
    """
    spot = float(spot)
    if ladder.empty or spot <= 0.0:
        return GammaTopology(
            timestamp=timestamp,
            symbol=symbol,
            spot=spot,
            total_abs_gamma=0.0,
            gamma_density_1pct=None,
            gamma_density_2_5pct=None,
            gamma_density_5pct=None,
            gamma_concentration=None,
            gamma_entropy=None,
            node_count=0,
            nearest_node=None,
            nearest_node_distance_pct=None,
            first_node_above=None,
            first_node_below=None,
            void_above_width_pct=None,
            void_below_width_pct=None,
        )

    work = ladder.copy()
    work["total_abs_gex"] = pd.to_numeric(work["total_abs_gex"], errors="coerce").fillna(0.0)
    work["strike"] = pd.to_numeric(work["strike"], errors="coerce")
    work = work.dropna(subset=["strike"])

    nodes = _nodes(work, quantile=node_quantile)
    above = nodes[nodes["strike"] > spot]
    below = nodes[nodes["strike"] < spot]
    nearest = None
    if not nodes.empty:
        idx = (nodes["strike"] - spot).abs().idxmin()
        nearest = float(nodes.loc[idx, "strike"])

    buckets = expiry_bucket_shares(frame, spot=spot) if frame is not None else {}
    zero_dte = buckets.get("d0")
    short = None
    if buckets:
        short = float(buckets.get("d0", 0.0) + buckets.get("d1_2", 0.0))
    weekly = buckets.get("d3_7")
    slope = None
    if buckets:
        long_share = float(buckets.get("d8_30", 0.0) + buckets.get("d30_plus", 0.0))
        slope = float((short or 0.0) - long_share)

    return GammaTopology(
        timestamp=timestamp,
        symbol=symbol,
        spot=spot,
        total_abs_gamma=float(work["total_abs_gex"].abs().sum()),
        gamma_density_1pct=_density_share(work, spot=spot, band=DENSITY_BANDS[0]),
        gamma_density_2_5pct=_density_share(work, spot=spot, band=DENSITY_BANDS[1]),
        gamma_density_5pct=_density_share(work, spot=spot, band=DENSITY_BANDS[2]),
        gamma_concentration=concentration_index(work["total_abs_gex"]),
        gamma_entropy=entropy(work["total_abs_gex"]),
        node_count=int(len(nodes)),
        nearest_node=nearest,
        nearest_node_distance_pct=None if nearest is None else float((nearest - spot) / spot),
        first_node_above=float(above["strike"].min()) if not above.empty else None,
        first_node_below=float(below["strike"].max()) if not below.empty else None,
        void_above_width_pct=_largest_gap_pct(above["strike"], spot=spot),
        void_below_width_pct=_largest_gap_pct(below["strike"], spot=spot),
        expiry_bucket_shares=buckets,
        zero_dte_gamma_share=zero_dte,
        short_gamma_share=short,
        weekly_gamma_share=weekly,
        gamma_term_slope=slope,
    )


def expiry_bucket_shares(frame: pd.DataFrame | None, *, spot: float) -> dict[str, float]:
    """Share of absolute gamma exposure falling in each expiry bucket.

    Uses the contract-level frame because the ladder has already collapsed
    expiries together. Returns an empty mapping when the frame carries no usable
    ``dte``, rather than inventing a single-bucket answer.
    """
    if frame is None or frame.empty or "dte" not in frame.columns:
        return {}
    work = frame.copy()
    work["dte"] = pd.to_numeric(work["dte"], errors="coerce")
    work["open_interest"] = pd.to_numeric(work.get("open_interest"), errors="coerce").fillna(0.0)
    work["gamma"] = pd.to_numeric(work.get("gamma"), errors="coerce").fillna(0.0)
    work = work.dropna(subset=["dte"])
    if work.empty:
        return {}
    work["abs_exposure"] = work["open_interest"].abs() * work["gamma"].abs() * 100.0 * float(spot)
    total = float(work["abs_exposure"].sum())
    if total <= 0.0:
        return {}
    shares: dict[str, float] = {}
    for label, low, high in EXPIRY_BUCKETS:
        sliced = work[(work["dte"] >= low) & (work["dte"] <= high)]
        shares[label] = float(sliced["abs_exposure"].sum() / total)
    return shares


# ---------------------------------------------------------------------------
# Estimated signed exposure
# ---------------------------------------------------------------------------


def build_estimated_signed_exposure(
    ladder: pd.DataFrame,
    *,
    symbol: str,
    spot: float,
    timestamp: str,
    call_wall: float | None = None,
    put_wall: float | None = None,
    gamma_flip: float | None = None,
) -> EstimatedSignedExposure:
    """Wrap the sign-dependent quantities with the convention that produced them."""
    spot = float(spot)
    if ladder.empty:
        return EstimatedSignedExposure(
            timestamp=timestamp,
            symbol=symbol,
            spot=spot,
            sign_convention=SIGN_CONVENTION,
            estimated_net_gex=0.0,
            estimated_dealer_imbalance=None,
            estimated_call_wall=call_wall,
            estimated_put_wall=put_wall,
            estimated_gamma_flip=gamma_flip,
            estimated_call_gex_total=0.0,
            estimated_put_gex_total=0.0,
        )
    net = float(pd.to_numeric(ladder["net_gex"], errors="coerce").fillna(0.0).sum())
    total_abs = float(pd.to_numeric(ladder["total_abs_gex"], errors="coerce").fillna(0.0).abs().sum())
    return EstimatedSignedExposure(
        timestamp=timestamp,
        symbol=symbol,
        spot=spot,
        sign_convention=SIGN_CONVENTION,
        estimated_net_gex=net,
        estimated_dealer_imbalance=(net / total_abs) if total_abs else None,
        estimated_call_wall=call_wall,
        estimated_put_wall=put_wall,
        estimated_gamma_flip=gamma_flip,
        estimated_call_gex_total=float(pd.to_numeric(ladder["call_gex"], errors="coerce").fillna(0.0).sum()),
        estimated_put_gex_total=float(pd.to_numeric(ladder["put_gex"], errors="coerce").fillna(0.0).sum()),
    )
