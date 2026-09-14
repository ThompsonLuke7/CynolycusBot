"""Detect unadjusted corporate actions in the shared bar caches.

`Data/shared/bars/1d` and `Data/shared/bars/4h` are fetched RAW, not
corporate-action adjusted (verified 2026-09-10 on TENX, SION and WOLF: both
caches show the same artifacts). A split, reverse split, or recapitalisation
changes the SHARE COUNT, so the price per share jumps without anyone's account
changing value -- and an unadjusted series records that jump as a return.

    WOLF 2025-09-26 close 1.20, vol 2.30M  ->  2025-09-29 open 17.88, vol 0.18M
         (Chapter 11 emergence: old shares replaced; nobody made 1,390%)
    TENX 2026-08-07 close 13.44            ->  2026-08-10 open 1.79 (10:1 forward split)

WHAT A FLAG IS
--------------
A session whose opening gap is a factor of `SUSPECT_RATIO` (4x) or more in
EITHER direction: open / prev_close >= 4 or <= 1/4. The threshold is on the
ratio, not the arithmetic return, because a down-gap can never exceed -100% --
the first version of this module used `|gap| >= 300%` and so could not see a
forward split at all. A cache-wide rescan found 54 such down-gaps it had missed
(TENX, SION, KLAC 2026-06-03, the Vanguard ETF splits of 2026-04-21, ...).

ORGANIC VS NOT
--------------
Each flag carries `organic`: True only for an UP-gap on share volume at least
`ORGANIC_MIN_VOLUME_RATIO` (5x) its trailing median. The rule comes from the
cache itself (2026-09-10, 164 up-gap flags with a volume baseline):

    volume_ratio  <=1.0   121   share volume did not rise with a 4x+ price:
                  1-3      14   the share count shrank (reverse split/recap)
                  3-5       1
                  >=5      28   real participation arrived: a genuine move

The distribution is empty between 3.11 and 5, so any threshold in that gap
classifies identically. A genuine 4x-37x rise on 1-3x volume is not something
organic buying produces.

DOWN-gaps are never organic. A forward split multiplies share volume by roughly
its ratio (TENX 44x, SION 67x), which is exactly what a genuine -90% collapse
looks like too; nothing local separates them. The live mark guard
(`live_4h_exec._implausible_mark_move`) reaches the same conclusion. Neither
case is something to buy, and neither return is one to trust.

Missing volume (no baseline) is never organic: unresolvable means suspect.

WHAT CALLERS DO WITH IT
-----------------------
* Research: `mask_windows` drops forward windows containing a NON-organic flag
  and the caller reports how many (AGENTS.md: document every dropped row).
  Organic flags are kept -- that tail is what a momentum study measures.
* Live: `recent_corporate_action` answers "is there a non-organic flag inside
  the entry lookback?" and `live_4h_exec.build_mixed_plan` refuses NEW entries
  when there is. Held positions are the mark guard's job, not this one's.

It FLAGS. It never adjusts a price and never names a split ratio -- at these
magnitudes a tolerance loose enough to match a real split matches everything,
and a confident wrong label is worse than the raw ratio a person can check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Opening gap, as a price ratio in either direction, at or beyond which a
#: session is flagged. 4.0 = up 300% or down 75%.
SUSPECT_RATIO = 4.0

#: An up-gap on at least this multiple of trailing median share volume is a
#: genuine move. Sits in the empty band (3.11, 5] of the observed distribution.
ORGANIC_MIN_VOLUME_RATIO = 5.0

#: Sessions a non-organic flag keeps a name out of NEW entries. Set from two
#: things that agree: `ret_20` is the longest short-horizon return feature, so a
#: gap inside the last 20 sessions is inside it by construction; and measured on
#: the OOF momentum ranking (2022-11..2026-05), recap-shaped names sat at the
#: 93.5th score percentile beforehand, 98.0th in the first 5 days after, 96.8th
#: at 6-20 days, and were back at ~92nd by 21-40 days. That measurement rests on
#: only 4 names (BBAI, CORZ, SIRI, WOLF) -- treat 20 as a floor, not an optimum.
ENTRY_LOOKBACK_SESSIONS = 20

#: Sessions of trailing volume used as the "normal" baseline.
_VOL_WINDOW = 20
_FLAG_COLUMNS = ["idx", "gap_ratio", "direction", "volume_ratio",
                 "dollar_volume_ratio", "organic"]


def suspect_sessions(
    df: pd.DataFrame,
    *,
    min_ratio: float = SUSPECT_RATIO,
    organic_min_volume_ratio: float = ORGANIC_MIN_VOLUME_RATIO,
) -> pd.DataFrame:
    """Flag sessions whose opening gap is too large to be a price move.

    `df` needs `open` and `close` (and `volume` for the organic test), ordered
    oldest first. Returns one row per flag; `idx` is the row's label in `df`.
    An empty frame means nothing tripped.
    """
    if df is None or df.empty or not {"open", "close"} <= set(df.columns):
        return pd.DataFrame(columns=_FLAG_COLUMNS)
    prev_close = df["close"].shift(1)
    ratio = df["open"] / prev_close
    valid = (prev_close > 0) & (df["open"] > 0)
    hit = valid & ((ratio >= min_ratio) | (ratio <= 1.0 / min_ratio))
    if not hit.any():
        return pd.DataFrame(columns=_FLAG_COLUMNS)
    if "volume" in df.columns:
        base = df["volume"].shift(1).rolling(_VOL_WINDOW, min_periods=3).median()
        vol_ratio = df["volume"] / base.where(base > 0)
    else:
        vol_ratio = pd.Series(np.nan, index=df.index)
    out = pd.DataFrame({
        "idx": df.index[hit],
        "gap_ratio": ratio[hit].to_numpy(float),
        "direction": np.where(ratio[hit].to_numpy(float) > 1.0, "up", "down"),
        "volume_ratio": vol_ratio[hit].to_numpy(float),
    })
    # Diagnostic only: ~1 means traded value was preserved while the share count
    # moved. Reported, not used -- volume_ratio is the cleaner separator.
    out["dollar_volume_ratio"] = out["gap_ratio"] * out["volume_ratio"]
    out["organic"] = ((out["direction"] == "up")
                      & (out["volume_ratio"] >= organic_min_volume_ratio))
    return out


def mask_windows(
    df: pd.DataFrame,
    hold: int,
    *,
    min_ratio: float = SUSPECT_RATIO,
    include_organic: bool = False,
) -> pd.Series:
    """True where a forward window of `hold` sessions CONTAINS a flagged session.

    Entering at row j and holding `hold` rows spans j .. j+hold-1, so a flag at
    row p contaminates every entry in (p-hold, p]. Organic flags are left in by
    default: a real move is the return being measured, not a defect in it.
    """
    bad = pd.Series(False, index=df.index)
    flags = suspect_sessions(df, min_ratio=min_ratio)
    if not include_organic:
        flags = flags[~flags["organic"].astype(bool)]
    if flags.empty:
        return bad
    n = len(df)
    for pos in {df.index.get_loc(i) for i in flags["idx"]}:
        bad.iloc[max(0, pos - hold + 1):min(pos + 1, n)] = True
    return bad


def _bar_times(df: pd.DataFrame) -> pd.Series:
    """UTC timestamp per row, from a DatetimeIndex or a `timestamp` column."""
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index if df.index.tz is not None else df.index.tz_localize("UTC")
        return pd.Series(idx.tz_convert("UTC"), index=df.index)
    if "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"], utc=True)
    raise ValueError("bars need a DatetimeIndex or a 'timestamp' column to be dated")


def recent_corporate_action(
    df: pd.DataFrame | None,
    *,
    as_of=None,
    lookback_sessions: int = ENTRY_LOOKBACK_SESSIONS,
    min_ratio: float = SUSPECT_RATIO,
) -> dict | None:
    """The most recent NON-organic flag within the last `lookback_sessions`
    trading sessions, or None when the name is clear.

    Only bars at or before `as_of` are read, so a backtest can never be vetoed
    by a corporate action that had not happened yet. Works on any bar frequency:
    the lookback counts distinct US/Eastern session dates, not rows.
    """
    if df is None or len(df) < 2:
        return None
    times = _bar_times(df)
    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff
        keep = (times <= cutoff).to_numpy()
        df, times = df[keep], times[keep]
        if len(df) < 2:
            return None
    flags = suspect_sessions(df, min_ratio=min_ratio)
    flags = flags[~flags["organic"].astype(bool)]
    if flags.empty:
        return None
    sessions = times.dt.tz_convert("America/New_York").dt.normalize()
    window = sorted(sessions.unique())[-int(lookback_sessions):]
    flags = flags.assign(session=sessions.loc[flags["idx"]].to_numpy())
    flags = flags[flags["session"] >= window[0]]
    if flags.empty:
        return None
    last = flags.iloc[-1]
    vr = float(last["volume_ratio"])
    return {
        "session": pd.Timestamp(last["session"]).date().isoformat(),
        "sessions_ago": len(window) - 1 - window.index(last["session"]),
        "gap_ratio": round(float(last["gap_ratio"]), 4),
        "direction": str(last["direction"]),
        "volume_ratio": round(vr, 4) if np.isfinite(vr) else None,
        "lookback_sessions": int(lookback_sessions),
    }
