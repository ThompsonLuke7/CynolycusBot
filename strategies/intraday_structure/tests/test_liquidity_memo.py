from __future__ import annotations

from datetime import datetime, timedelta, timezone

from strategies.intraday_structure.config import LevelPolicy
from strategies.intraday_structure.levels import StructuralLevelProvider
from strategies.intraday_structure.models import Bar


NOW = datetime(2026, 8, 25, 13, 30, tzinfo=timezone.utc)


def _bars(n: int, *, last_close: float | None = None) -> list[Bar]:
    out = []
    for i in range(n):
        base = 100 + (i % 7) * 0.4
        out.append(Bar("XYZ", NOW + timedelta(minutes=i), base, base + 0.5, base - 0.5, base + 0.1, 1000 + i))
    if last_close is not None:
        b = out[-1]
        out[-1] = Bar(b.symbol, b.timestamp, b.open, max(b.high, last_close), min(b.low, last_close), last_close, b.volume)
    return out


def _levels(provider, bars):
    return [(x.price, x.level_type, x.strength, x.rejection_count) for x in provider._liquidity_levels(bars)]


def test_the_memo_returns_the_same_levels_it_would_have_computed() -> None:
    bars = _bars(90)
    memoized = StructuralLevelProvider(LevelPolicy())
    reference = StructuralLevelProvider(LevelPolicy())
    first = _levels(memoized, bars)
    second = _levels(memoized, bars)          # served from the memo
    assert first == second == _levels(reference, bars)


def test_a_new_bar_invalidates_the_memo() -> None:
    provider = StructuralLevelProvider(LevelPolicy())
    before = _levels(provider, _bars(90))
    after = _levels(provider, _bars(120))
    fresh = _levels(StructuralLevelProvider(LevelPolicy()), _bars(120))
    assert after == fresh
    assert before != after or len(before) == 0


def test_a_corrected_last_bar_invalidates_the_memo() -> None:
    # _append_bar REPLACES the last bar when a timestamp repeats, so keying on
    # the timestamp alone would serve a stale answer for a corrected bar.
    provider = StructuralLevelProvider(LevelPolicy())
    _levels(provider, _bars(90))
    corrected = _bars(90, last_close=140.0)
    assert _levels(provider, corrected) == _levels(StructuralLevelProvider(LevelPolicy()), corrected)


def test_the_caller_cannot_mutate_the_cached_levels() -> None:
    provider = StructuralLevelProvider(LevelPolicy())
    bars = _bars(90)
    first = provider._liquidity_levels(bars)
    if not first:
        return
    first[0].price = -999.0
    assert provider._liquidity_levels(bars)[0].price != -999.0


def test_two_symbols_do_not_share_one_memo_entry() -> None:
    provider = StructuralLevelProvider(LevelPolicy())
    xyz = _bars(90)
    abc = [Bar("ABC", b.timestamp, b.open * 3, b.high * 3, b.low * 3, b.close * 3, b.volume) for b in xyz]
    a = _levels(provider, xyz)
    b = _levels(provider, abc)
    assert _levels(provider, xyz) == a
    assert b == _levels(StructuralLevelProvider(LevelPolicy()), abc)
