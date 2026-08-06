"""Limit rungs for closing an option position, from the mid down to the bid.

A market close always fills but always pays the spread. Walking down from the
mid to the bid first captures whatever price improvement is available; the
market order is the guarantee that comes *after* the ladder is exhausted, not
instead of it.

The rungs are a pure function of the observed market — no clock, no IO, no
randomness — so any exit can be reproduced from its audit record.

Note: ``strategies/multi_ticker_swing/live/position_manager._close_limit_ladder``
is an older, swing-specific variant anchored at the bid with extra branches for
liquidation reasons. It is deliberately left alone here; unifying the two is
its own change, and silently altering swing's exit pricing is not in scope.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal


TICK = Decimal("0.01")

# How far below the mid to walk when the bid is unknown. A degraded close still
# tries to earn price improvement, but without a bid there is no observed floor,
# so the walk is deliberately shallow.
_UNKNOWN_BID_FLOOR = Decimal("0.80")


def _to_tick(value: Decimal) -> Decimal:
    """Round down to a whole penny, never below the minimum offer."""

    rounded = value.quantize(TICK, rounding=ROUND_DOWN)
    return rounded if rounded >= TICK else TICK


def close_limit_ladder(
    *,
    mid: Decimal,
    bid: Decimal | None,
    attempts: int = 5,
) -> tuple[Decimal, ...]:
    """Return descending limit prices to try before falling back to a market order.

    The first rung is the mid and the last is the bid. With no bid, the walk
    still starts at the mid and steps down a bounded distance.
    """

    if not isinstance(mid, Decimal) or not mid.is_finite() or mid <= 0:
        raise ValueError("close ladder requires a positive finite mid")
    if bid is not None:
        if not isinstance(bid, Decimal) or not bid.is_finite() or bid <= 0:
            raise ValueError("close ladder requires a positive finite bid when supplied")
        if bid > mid:
            raise ValueError("a bid above the mid is a crossed market")

    # Never return an empty ladder: the caller would have nothing to submit and
    # the exit would silently not happen.
    rungs_wanted = max(1, int(attempts))
    top = _to_tick(mid)
    floor = _to_tick(bid) if bid is not None else _to_tick(mid * _UNKNOWN_BID_FLOOR)
    if floor > top:
        floor = top

    if rungs_wanted == 1 or top == floor:
        return (top,)

    span = top - floor
    steps = rungs_wanted - 1
    rungs: list[Decimal] = [top]
    for index in range(1, steps):
        rungs.append(_to_tick(top - (span * index / steps)))
    rungs.append(floor)

    # Rounding can collapse neighbouring rungs; each one must be a real new
    # offer, and the order must stay strictly descending.
    unique: list[Decimal] = []
    for rung in rungs:
        if not unique or rung < unique[-1]:
            unique.append(rung)
    return tuple(unique)


__all__ = ["TICK", "close_limit_ladder"]
