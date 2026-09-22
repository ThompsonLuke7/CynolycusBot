"""Trailing-correlation peer groups.

The lateral axis of the state registry.  Every other producer describes a level
of the hierarchy (market, sector, theme, ticker); this one describes which
names move TOGETHER.

Why correlation and not the theme taxonomy
------------------------------------------
Measured on this repo's own data
(``research/execution_quality/24_horizon_thesis_experiments.md`` §7), forward
20-session mean pairwise correlation is:

    theme groups                     0.388
    trailing-correlation clusters    0.343      <- this module
    sector (size-matched)            0.233
    size-matched random              0.217

So correlation clustering buys about **73%** of the taxonomy's edge over random
with no news, no embeddings, and no LLM -- and, critically, without the
taxonomy's **83-88% weekly membership churn** (co-membership Jaccard 0.12-0.14,
§4).  For a risk bucket, stability is the property that matters: a
concentration limit computed on a grouping that reshuffles weekly is not a
limit, it is noise.

What this is NOT for
--------------------
Ranking.  The theme block fed to the Meta ranker measured **-0.0167 rho**
[-0.0222, -0.0111] -- distinguishably worse than leaving it out (§4).  Nothing
here should reach a scoring feature matrix without its own experiment.

Point-in-time discipline
------------------------
* Correlations are fit on a TRAILING window ending at the as-of session.
* The universe comes from ``load_universe_as_of``, never today's list.
* Bars are screened for corporate actions (the cache is unadjusted, and a
  split prints as a ~4x overnight gap that would dominate a correlation).
* A liquidity floor is applied at build time, because §8 showed 87% of the
  leader/follower events were in names under $1M/day and those names carried
  the entire measured effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from core.corporate_actions import suspect_sessions
from core.nervous_system.contracts.enums import StateType
from core.nervous_system.contracts.quality import DataQualitySummary
from core.nervous_system.contracts.states import PeerGroupState


REPO = Path(__file__).resolve().parents[2]
BARS_1D = REPO / "Data/shared/bars/1d"

UTC = timezone.utc
_EASTERN = ZoneInfo("America/New_York")
_MARKET_CLOSE = time(16, 0)

PRODUCER = "signals.peer_structure.correlation_groups"
METHOD = "trailing_correlation_kmeans"
MODEL_VERSION = "peer-corr@1"
FEATURE_VERSION = "peer-corr-features@1"
CONFIG_VERSION = "peer-corr-config@1"

_NAMESPACE = uuid5(NAMESPACE_URL, "cynolycus.peer-structure.correlation-group.v1")


@dataclass(frozen=True)
class PeerGroupConfig:
    """Everything that changes the grouping, in one versioned object."""

    lookback_sessions: int = 60
    n_groups: int = 60
    min_members: int = 3
    min_dollar_volume: float = 10_000_000.0
    min_price: float = 5.0
    min_observations: int = 40
    # Groups expire rather than lingering: a stale grouping silently applied to
    # a new regime is the failure mode this guards.
    valid_for: timedelta = field(default=timedelta(days=9))
    random_seed: int = 17

    @property
    def config_version(self) -> str:
        return (
            f"{CONFIG_VERSION}:k{self.n_groups}:lb{self.lookback_sessions}"
            f":dv{int(self.min_dollar_volume)}:px{self.min_price}"
        )


def _session_close_utc(session: pd.Timestamp) -> datetime:
    """The 16:00 ET close of `session`, in UTC."""

    naive = datetime.combine(session.date(), _MARKET_CLOSE)
    return naive.replace(tzinfo=_EASTERN).astimezone(UTC)


def load_screened_returns(
    tickers: list[str],
    *,
    as_of_session: pd.Timestamp,
    config: PeerGroupConfig,
) -> pd.DataFrame:
    """Wide daily returns for names that pass the liquidity and artefact screens.

    Every bar strictly AFTER ``as_of_session`` is dropped before anything is
    computed, so the window cannot see past its own as-of date.
    """

    columns: dict[str, pd.Series] = {}
    for ticker in tickers:
        path = BARS_1D / f"{ticker}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(
            path, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        if frame.empty:
            continue
        index = (
            pd.to_datetime(frame["timestamp"], utc=True)
            .dt.tz_convert(_EASTERN)
            .dt.normalize()
            .dt.tz_localize(None)
        )
        frame = frame.set_index(index).sort_index()
        # Causal cut FIRST: everything below is computed only from bars the
        # as-of session could actually have seen.
        frame = frame.loc[frame.index <= as_of_session]
        window = frame.tail(config.lookback_sessions + 1)
        if len(window) < config.min_observations:
            continue

        close = window["close"]
        dollar_volume = (close * window["volume"]).tail(20).median()
        if not np.isfinite(dollar_volume) or dollar_volume < config.min_dollar_volume:
            continue
        if not np.isfinite(close.iloc[-1]) or close.iloc[-1] < config.min_price:
            continue

        returns = close.pct_change()
        # The 1d cache is unadjusted, so a split prints as a ~4x overnight gap.
        # Left in, one such bar dominates every correlation this name appears
        # in. Blanked rather than dropped so the date index stays aligned.
        flags = suspect_sessions(window)
        if not flags.empty:
            returns.loc[flags["idx"]] = np.nan
        columns[ticker] = returns

    if not columns:
        return pd.DataFrame()
    return pd.DataFrame(columns).sort_index()


def build_peer_groups(
    returns: pd.DataFrame,
    *,
    as_of_session: pd.Timestamp,
    config: PeerGroupConfig,
    universe_version: str,
) -> tuple[PeerGroupState, ...]:
    """Cluster the trailing correlation matrix into peer groups."""

    from sklearn.cluster import KMeans

    if returns.empty:
        return ()
    usable = returns.dropna(axis=1, thresh=config.min_observations)
    if usable.shape[1] < config.min_members:
        return ()

    correlation = usable.corr().fillna(0.0)
    tickers = list(correlation.columns)
    n_groups = max(1, min(config.n_groups, len(tickers) // config.min_members))
    labels = KMeans(
        n_clusters=n_groups, n_init=4, random_state=config.random_seed
    ).fit_predict(correlation.to_numpy())

    as_of = _session_close_utc(as_of_session)
    # Available only once the session it is built from has closed and the
    # nightly job has run; `generated_at` is set by the caller's publish step.
    available_at = as_of
    valid_until = available_at + config.valid_for

    states: list[PeerGroupState] = []
    for label in sorted(set(labels)):
        members = sorted(
            ticker for ticker, item in zip(tickers, labels) if item == label
        )
        if len(members) < config.min_members:
            continue
        block = correlation.loc[members, members].to_numpy()
        upper = block[np.triu_indices_from(block, k=1)]
        cohesion = float(np.nanmean(upper)) if upper.size else None

        group_id = f"corr{as_of_session.strftime('%Y%m%d')}g{label:03d}"
        states.append(
            PeerGroupState(
                state_id=uuid5(_NAMESPACE, f"{config.config_version}|{group_id}"),
                state_type=StateType.PEER_GROUP,
                entity_id=group_id,
                as_of=as_of,
                available_at=available_at,
                generated_at=available_at,
                valid_until=valid_until,
                source_window_start=_session_close_utc(usable.index[0]),
                source_window_end=as_of,
                schema_version=1,
                producer=PRODUCER,
                model_version=MODEL_VERSION,
                feature_version=FEATURE_VERSION,
                config_version=config.config_version,
                lineage_ids=(f"bars1d:{as_of_session.date().isoformat()}",),
                data_quality=DataQualitySummary(),
                group_id=group_id,
                method=METHOD,
                members=tuple(members),
                lookback_sessions=config.lookback_sessions,
                trailing_cohesion=cohesion,
                universe_version=universe_version,
                metrics={"member_count": float(len(members))},
            )
        )
    return tuple(states)


__all__ = [
    "PeerGroupConfig",
    "build_peer_groups",
    "load_screened_returns",
]
