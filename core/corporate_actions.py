"""Detect unadjusted corporate actions in the shared daily bar cache.

`Data/shared/bars/1d` is NOT corporate-action adjusted. A reverse split, a
recapitalisation, or an emergence from Chapter 11 changes the SHARE COUNT, so
the price per share jumps without anyone's account changing value -- and an
unadjusted series records that jump as a return.

The case that motivated this (2026-09-08, found while replicating the rank-depth
study over 2022-2026):

    date        close    volume
    2025-09-26   1.20   2,303,766
    2025-09-29  21.51     180,927     <- WOLF emerges from Chapter 11

Price per share x15 overnight while volume fell 13x; dollar volume was $2.76M
before and $3.26M after. Nobody made 1,390% -- the old shares were replaced. Left
in, those 16 observations moved one year's measured top-1 return from +5.5%
(median) to +47.3% (mean) and inflated the whole study's headline by 3x.

WHAT THIS DOES AND DOES NOT DO
------------------------------
It FLAGS. It does not adjust prices, and it does not label a flag "a 10:1 split".
That mirrors `core.live_4h_exec._implausible_mark_move`, whose docstring makes
the argument: at these magnitudes a tolerance loose enough to match a real split
matches everything else too, and a confident-looking wrong diagnosis is worse
than the raw ratio an operator can check against a real corporate-action source.

The research caller's correct action is to DROP the affected observations and
report how many, which `mask_windows` supports and AGENTS.md requires ("do not
silently ... drop rows ... without documenting the rule and its impact").

DISTINGUISHING A RECAPITALISATION FROM A REAL MOVE
-------------------------------------------------
A genuine explosive move -- a biotech readout, a squeeze -- comes with a volume
EXPANSION, and that tail is exactly what a momentum study is trying to measure.
Excluding it would bias results down. A share-count change instead moves price
and volume inversely, leaving dollar volume roughly intact.

So `dollar_volume_ratio` is reported alongside every flag: near 1.0 is positive
evidence of a recapitalisation, well above 1.0 suggests a real move with real
participation. It is a DIAGNOSTIC, not a silent second condition -- the flag
itself is on the price gap alone, because being conservative about what enters a
study is cheaper than being clever about what to keep.
"""
from __future__ import annotations

import pandas as pd

#: Overnight gap (|open_t / close_{t-1} - 1|) at or beyond which a session is
#: flagged. 3.0 = a 300% gap. Deliberately far looser than the live mark guard's
#: 0.70: that one protects an open position and should fire early, while this one
#: screens a research sample where a real +100% day is signal worth keeping.
SUSPECT_GAP = 3.0

#: Sessions of trailing volume used as the "normal" baseline for the diagnostic.
_VOL_WINDOW = 20


def suspect_sessions(
    df: pd.DataFrame,
    *,
    threshold: float = SUSPECT_GAP,
) -> pd.DataFrame:
    """Flag sessions whose opening gap is too large to be a price move.

    `df` needs `open`, `close` and (for the diagnostic) `volume`, ordered oldest
    first -- the layout of the shared 1d cache. Returns one row per flagged
    session with the raw ratios; an empty frame means nothing tripped.
    """
    if df.empty or not {"open", "close"} <= set(df.columns):
        return pd.DataFrame(columns=["idx", "gap", "gap_ratio", "volume_ratio",
                                     "dollar_volume_ratio"])
    prev_close = df["close"].shift(1)
    gap = df["open"] / prev_close - 1.0
    hit = gap.abs() >= threshold
    hit &= prev_close > 0
    if not hit.any():
        return pd.DataFrame(columns=["idx", "gap", "gap_ratio", "volume_ratio",
                                     "dollar_volume_ratio"])

    gap_ratio = df["open"] / prev_close
    if "volume" in df.columns:
        base_vol = df["volume"].shift(1).rolling(_VOL_WINDOW, min_periods=3).median()
        vol_ratio = df["volume"] / base_vol.replace(0, pd.NA)
    else:
        vol_ratio = pd.Series(pd.NA, index=df.index)
    out = pd.DataFrame({
        "idx": df.index[hit],
        "gap": gap[hit].to_numpy(),
        "gap_ratio": gap_ratio[hit].to_numpy(),
        "volume_ratio": vol_ratio[hit].to_numpy(),
    })
    # ~1.0 means the share count moved and the traded VALUE did not, which is
    # what a recapitalisation looks like. Much greater than 1.0 means real
    # participation arrived, i.e. probably a genuine move worth keeping.
    out["dollar_volume_ratio"] = out["gap_ratio"] * out["volume_ratio"]
    return out.reset_index(drop=True)


def mask_windows(
    df: pd.DataFrame,
    hold: int,
    *,
    threshold: float = SUSPECT_GAP,
) -> pd.Series:
    """True where a forward window of `hold` sessions CONTAINS a suspect session.

    A return measured from session i to session i+hold-1 is contaminated if any
    session strictly after the entry carries an unadjusted corporate action, so
    the whole window is masked, not just the gap day.
    """
    flags = suspect_sessions(df, threshold=threshold)
    bad = pd.Series(False, index=df.index)
    if flags.empty:
        return bad
    positions = {df.index.get_loc(i) for i in flags["idx"]}
    n = len(df)
    for pos in positions:
        # entering at j and holding `hold` sessions spans j .. j+hold-1; that
        # window contains `pos` when j is in (pos-hold, pos].
        lo = max(0, pos - hold + 1)
        bad.iloc[lo:min(pos + 1, n)] = True
    return bad
