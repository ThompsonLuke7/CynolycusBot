"""Runtime assembly of the portfolio config from on-disk sector assignments.

`config/portfolio.py` is deliberately IO-free: it is imported by
`config/policy.py`, which the purity test forbids from reading files or clocks,
and its own comment says runtime assembly is responsible for keeping the
curated map in sync with the real one. This module is that runtime assembly.

Why it exists: the shipped `MVP_PORTFOLIO_CONFIG.sector_map` is the 96-name
curated map, which covers 95/2,903 of the live universe (3.3%). Under that map
the sector concentration limit buckets ~97% of positions as `UNALLOCATED`, and
`UNALLOCATED` deliberately never vetoes -- so the sector arm of the gate is
inert. `scripts/build_sector_assignments.py` fills the gap empirically.

Two deliberate refusals:

* **A weak correlation is not an assignment.** Below `min_correlation` the
  ticker is left out of the map entirely, so it buckets as `UNALLOCATED`
  (an explicit unknown) rather than being given a sector the data does not
  support. On the 2026-09-11 build, 18.4% of names fall under 0.30.
* **The curated map wins.** Where a hand assignment exists it is
  authoritative; the resolver agreed with it on only 77.9% of the 95 curated
  names, which is a reason to prefer the hand map, not to overwrite it.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from core.nervous_system.config.portfolio import MVP_PORTFOLIO_CONFIG, PortfolioConfig


logger = logging.getLogger(__name__)

DEFAULT_ASSIGNMENTS_PATH = Path("Data/shared/market_regime/sector_assignments.parquet")
# Below this, the empirical sector is not trusted enough to gate risk on.
DEFAULT_MIN_CORRELATION = 0.30


def load_sector_map(
    path: Path | str = DEFAULT_ASSIGNMENTS_PATH,
    *,
    min_correlation: float = DEFAULT_MIN_CORRELATION,
) -> dict[str, str]:
    """Read the empirical sector assignments into a ticker -> sector-ETF map.

    Returns an empty map if the file is absent, which degrades to the curated
    map rather than failing a trading process for a missing research artifact.
    """

    import pandas as pd

    path = Path(path)
    if not path.exists():
        logger.warning(
            "sector assignments not found at %s; falling back to the curated map "
            "(3.3%% universe coverage, so the sector limit will be largely inert)",
            path,
        )
        return {}

    frame = pd.read_parquet(path)
    required = {"ticker", "sector_etf", "correlation"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"sector assignments missing columns: {sorted(missing)}")

    usable = frame[frame["sector_etf"].notna()]
    # `curated` rows carry the hand assignment and are kept regardless of the
    # resolver's correlation, which describes the resolver's opinion, not the
    # hand map's confidence.
    if "curated" in usable.columns:
        strong = usable[
            usable["curated"].fillna(False)
            | (usable["correlation"].abs() >= min_correlation)
        ]
    else:
        strong = usable[usable["correlation"].abs() >= min_correlation]

    dropped = len(usable) - len(strong)
    if dropped:
        logger.info(
            "sector map: %d/%d names below |corr| %.2f left UNALLOCATED",
            dropped, len(usable), min_correlation,
        )
    return {
        str(row.ticker).upper(): str(row.sector_etf)
        for row in strong.itertuples(index=False)
    }


def portfolio_config_with_sectors(
    base: PortfolioConfig = MVP_PORTFOLIO_CONFIG,
    *,
    path: Path | str = DEFAULT_ASSIGNMENTS_PATH,
    min_correlation: float = DEFAULT_MIN_CORRELATION,
) -> PortfolioConfig:
    """`base` with its sector map widened by the empirical assignments.

    The curated entries win on conflict, and the config version records that
    the map was widened so two runs with different coverage are never
    mistaken for the same configuration.
    """

    empirical = load_sector_map(path, min_correlation=min_correlation)
    if not empirical:
        return base
    merged = {**empirical, **dict(base.sector_map)}
    version = f"{base.config_version}+sectors@{len(merged)}"
    return replace(base, sector_map=merged, config_version=version)


__all__ = [
    "DEFAULT_ASSIGNMENTS_PATH",
    "DEFAULT_MIN_CORRELATION",
    "load_sector_map",
    "portfolio_config_with_sectors",
]
