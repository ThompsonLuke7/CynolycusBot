"""Study A: does price react differently at gamma-heavy strikes than at ordinary ones?

Design note -- a deviation from the registration, declared before running
--------------------------------------------------------------------------
The registration specified "matched pseudo-levels at the same distance from
spot." That control does not survive contact with the data: at the moment of a
touch, price *is* the level, so a same-distance control is the same price.

The control used instead is stronger. Every event in this study is the same kind
of event -- **price arriving at an option strike** -- and the strikes differ only
in how much gamma sits on them. That makes the comparison dose-response rather
than treated-vs-control, and it directly answers the question a trader is asking:
is a wall different from an ordinary price level, or is any strike a level?

Without this control the study would rediscover support and resistance and hand
gamma the credit for it.

Two registered conditioning variables are not computable from this archive and
are replaced, also declared before running:

* ``structure_confidence`` and ``call_wall_stability`` need per-contract gamma,
  IV and DTE. The archived ladders are already aggregated across expiries, so
  neither can be recovered. Replaced by ``gex_share`` (concentration at the
  strike) and ``level_persistence_min`` (how long the level has held that
  strike), which the series does support -- and persistence is the better
  stability proxy anyway, being measured rather than modelled.
* ``zero_dte_gamma_share`` needs per-DTE detail the ladder has collapsed.
  Replaced by ``dealer_imbalance`` as the gamma-regime conditioner.

Outcome definition
------------------
For each arrival at strike K, the approach side is the sign of (spot - K) ten
minutes earlier. Over the next 30 minutes, whichever happens first:

* **rejection** -- price returns 20bps or more back toward the approach side
* **penetration** -- price moves 20bps or more through to the far side
* **neither** -- reported, never dropped

Everything is measured from the archived one-minute spot series, so no external
price feed can disagree with what the dealer module actually saw.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data"
ARCHIVE_ROOT = REPO / "Data" / "dealer_positioning"

TOUCH_BPS = 0.0010          # arrival band: within 10bps of the strike
REARM_MINUTES = 20          # a new event needs this long away from the strike
APPROACH_LOOKBACK_MIN = 10  # how far back the approach direction is read
HORIZON_MIN = 30            # outcome window
# Symmetric by construction. An earlier asymmetric pair (25bps to reject, 15bps
# to penetrate) made penetration mechanically easier to achieve and would have
# produced a penetration-heavy result from geometry alone. Both thresholds are
# now the same distance, so neither outcome is favoured by the measurement.
REJECT_BPS = 0.0020
PENETRATE_BPS = 0.0020

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def _series(symbol: str) -> pd.DataFrame:
    path = DATA / f"level_series_{symbol.upper()}.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame["captured_at"] = pd.to_datetime(frame["captured_at"], utc=True)
    return frame.sort_values("captured_at").reset_index(drop=True)


def _strike_grid(frame: pd.DataFrame) -> float:
    """Infer the strike spacing from the levels the module itself reported."""
    values = pd.concat([frame["call_wall"], frame["put_wall"], frame["nearest_magnet"]]).dropna()
    if values.empty:
        return 1.0
    diffs = np.diff(np.sort(values.unique()))
    diffs = diffs[diffs > 0]
    return float(np.median(diffs)) if len(diffs) else 1.0


def _level_persistence(frame: pd.DataFrame, column: str) -> pd.Series:
    """Minutes the level has sat on its current strike, within the session."""
    key = frame[column]
    changed = (key != key.shift()) | (frame["session"] != frame["session"].shift())
    block = changed.cumsum()
    first = frame.groupby(block)["captured_at"].transform("min")
    return (frame["captured_at"] - first).dt.total_seconds() / 60.0


def build_events(symbol: str) -> pd.DataFrame:
    """Every arrival at a strike, tagged with what that strike was carrying."""
    frame = _series(symbol)
    if frame.empty:
        return pd.DataFrame()

    grid = _strike_grid(frame)
    frame = frame.copy()
    frame["nearest_strike"] = (frame["spot"] / grid).round() * grid
    frame["dist_bps"] = (frame["spot"] - frame["nearest_strike"]).abs() / frame["spot"]
    frame["in_band"] = frame["dist_bps"] <= TOUCH_BPS

    for col in ("call_wall", "put_wall", "nearest_magnet"):
        frame[f"persist_{col}"] = _level_persistence(frame, col)

    # Classify the strike from a snapshot taken BEFORE price arrived.
    #
    # This is the difference between a predictive question and a circular one.
    # Gamma peaks at the money, so at the instant price is sitting on strike K,
    # K tends to be whatever the levels point at -- the magnet is the nearest
    # strike to spot 31-70% of the time, and even the call wall (max call gamma
    # among strikes above spot) is drawn to K once spot is a hair below it.
    # Classifying at arrival would therefore partly encode "price is where price
    # is". Classifying from ten minutes earlier asks the question a trader
    # actually faces: this strike was a wall before I got here -- does that
    # change what happens next?
    prior = frame[["captured_at", "session", "call_wall", "put_wall", "nearest_magnet"]].copy()
    prior = prior.rename(
        columns={
            "call_wall": "prior_call_wall",
            "put_wall": "prior_put_wall",
            "nearest_magnet": "prior_magnet",
        }
    )
    lookup = frame[["captured_at", "session"]].copy()
    lookup["target"] = lookup["captured_at"] - pd.Timedelta(minutes=APPROACH_LOOKBACK_MIN)
    merged = pd.merge_asof(
        lookup.sort_values("target"),
        prior.sort_values("captured_at"),
        left_on="target",
        right_on="captured_at",
        by="session",
        direction="backward",
        suffixes=("", "_prior"),
    ).set_index(lookup.sort_values("target").index).sort_index()
    for col in ("prior_call_wall", "prior_put_wall", "prior_magnet"):
        frame[col] = merged[col]

    # One event per arrival: the first in-band row after being away long enough.
    events = []
    last_event_time: dict[float, pd.Timestamp] = {}
    for row in frame[frame["in_band"]].itertuples():
        strike = float(row.nearest_strike)
        prior = last_event_time.get(strike)
        if prior is not None and (row.captured_at - prior).total_seconds() / 60.0 < REARM_MINUTES:
            continue
        last_event_time[strike] = row.captured_at
        events.append(
            {
                "symbol": symbol.upper(),
                "captured_at": row.captured_at,
                "session": row.session,
                "strike": strike,
                "spot": float(row.spot),
                # Classified from the pre-arrival snapshot (see above).
                "is_call_wall": bool(row.prior_call_wall == strike),
                "is_put_wall": bool(row.prior_put_wall == strike),
                "is_magnet": bool(row.prior_magnet == strike),
                "has_prior_classification": bool(pd.notna(row.prior_call_wall)
                                                 or pd.notna(row.prior_put_wall)
                                                 or pd.notna(row.prior_magnet)),
                "is_call_wall_at_arrival": bool(row.call_wall == strike),
                "dealer_imbalance": row.dealer_imbalance,
                "total_abs_gamma": row.total_abs_gamma,
                "atm_iv": row.atm_iv,
                "call_wall_share": row.call_wall_share,
                "put_wall_share": row.put_wall_share,
                "magnet_share": row.magnet_share,
                "persist_call_wall": row.persist_call_wall,
                "persist_put_wall": row.persist_put_wall,
                "persist_magnet": row.persist_nearest_magnet,
            }
        )
    if not events:
        return pd.DataFrame()
    out = pd.DataFrame(events)
    # An event with no pre-arrival snapshot cannot be classified either way and
    # is dropped rather than silently counted as a plain strike.
    out = out[out["has_prior_classification"]].reset_index(drop=True)
    out["any_level"] = out[["is_call_wall", "is_put_wall", "is_magnet"]].any(axis=1)
    return _attach_outcomes(out, frame)


def _attach_outcomes(events: pd.DataFrame, series: pd.DataFrame) -> pd.DataFrame:
    """Resolve each arrival into rejection / penetration / neither."""
    spot = series[["captured_at", "spot", "session"]].copy()
    results = []
    for ev in events.itertuples():
        k = ev.strike
        t = ev.captured_at

        back = spot[
            (spot["captured_at"] <= t - pd.Timedelta(minutes=APPROACH_LOOKBACK_MIN))
            & (spot["session"] == ev.session)
        ]
        if back.empty:
            results.append({"outcome": None, "approach": 0})
            continue
        approach = float(np.sign(back["spot"].iloc[-1] - k))
        if approach == 0:
            results.append({"outcome": None, "approach": 0})
            continue

        window = spot[
            (spot["captured_at"] > t)
            & (spot["captured_at"] <= t + pd.Timedelta(minutes=HORIZON_MIN))
            & (spot["session"] == ev.session)
        ]
        if window.empty:
            results.append({"outcome": None, "approach": approach})
            continue

        # Signed distance in the approach direction: positive is back the way it
        # came, negative is through the level.
        signed = (window["spot"].to_numpy() - k) * approach / k
        rejected = np.where(signed >= REJECT_BPS)[0]
        penetrated = np.where(signed <= -PENETRATE_BPS)[0]
        first_reject = rejected[0] if len(rejected) else np.inf
        first_penetrate = penetrated[0] if len(penetrated) else np.inf

        if first_reject == np.inf and first_penetrate == np.inf:
            outcome = "neither"
        elif first_reject < first_penetrate:
            outcome = "rejection"
        else:
            outcome = "penetration"
        results.append({"outcome": outcome, "approach": approach})

    frame = pd.concat([events.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    return frame[frame["outcome"].notna()].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def cluster_bootstrap(
    frame: pd.DataFrame,
    mask: pd.Series,
    *,
    draws: int = 2000,
    seed: int = 20260826,
) -> tuple[float, float, float, int]:
    """Rejection-rate gap between two groups, resampling whole sessions.

    Events inside one session share one tape, so resampling events would treat a
    single trending afternoon as hundreds of independent observations. The unit
    of resampling is the session.

    Each cluster is reduced to four counts, and all draws are evaluated as one
    matrix product. Resampling frames draw-by-draw cost ~40ms each, which made a
    fourteen-comparison report take twenty minutes for arithmetic that takes
    milliseconds.
    """
    rng = np.random.default_rng(seed)
    work = frame[frame["outcome"].isin(["rejection", "penetration"])]
    if work.empty:
        return float("nan"), float("nan"), float("nan"), 0

    treated = mask.reindex(work.index).fillna(False).to_numpy(dtype=bool)
    rejected = (work["outcome"] == "rejection").to_numpy(dtype=bool)
    cluster = (work["symbol"].astype(str) + "|" + work["session"].astype(str)).to_numpy()

    codes, uniques = pd.factorize(cluster)
    n = len(uniques)
    if n < 5:
        return float("nan"), float("nan"), float("nan"), n

    t_rej = np.bincount(codes, weights=(treated & rejected).astype(float), minlength=n)
    t_tot = np.bincount(codes, weights=treated.astype(float), minlength=n)
    c_rej = np.bincount(codes, weights=(~treated & rejected).astype(float), minlength=n)
    c_tot = np.bincount(codes, weights=(~treated).astype(float), minlength=n)

    tt_all, ct_all = t_tot.sum(), c_tot.sum()
    if tt_all == 0 or ct_all == 0:
        return float("nan"), float("nan"), float("nan"), n
    point = float(t_rej.sum() / tt_all - c_rej.sum() / ct_all)

    picks = rng.integers(0, n, size=(draws, n))
    flat = (np.arange(draws)[:, None] * n + picks).ravel()
    mult = np.bincount(flat, minlength=draws * n).reshape(draws, n).astype(float)

    tt = mult @ t_tot
    ct = mult @ c_tot
    ok = (tt > 0) & (ct > 0)
    if not ok.any():
        return point, float("nan"), float("nan"), n
    values = (mult @ t_rej)[ok] / tt[ok] - (mult @ c_rej)[ok] / ct[ok]
    low, high = np.percentile(values, [2.5, 97.5])
    return point, float(low), float(high), n


def rate_table(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    work = frame[frame["outcome"].isin(["rejection", "penetration"])]
    out = work.groupby(by).agg(
        events=("outcome", "size"),
        rejection_rate=("outcome", lambda s: float((s == "rejection").mean())),
        sessions=("session", "nunique"),
    )
    return out.sort_values("events", ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=["SPY", "QQQ", "IWM", "GLD", "SLV"])
    parser.add_argument("--out", type=Path, default=DATA / "study_a_events.parquet")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    frames = []
    for symbol in args.symbols:
        events = build_events(symbol)
        if events.empty:
            logger.warning("%s: no events", symbol)
            continue
        logger.info("%s: %d events over %d sessions", symbol, len(events), events["session"].nunique())
        frames.append(events)
    if not frames:
        raise SystemExit("no events built")
    allev = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    allev.to_parquet(args.out, index=False)
    print(f"wrote {len(allev)} events -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
