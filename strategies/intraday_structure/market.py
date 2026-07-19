from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from strategies.intraday_structure.models import Bar, Candidate, MarketContext


class MarketContextProvider:
    """Causal context from synchronized index/sector 1-minute bars."""

    def __init__(self, max_bars: int = 600) -> None:
        self._bars: dict[str, list[Bar]] = defaultdict(list)
        self._max_bars = max_bars

    def update(self, bar: Bar) -> None:
        history = self._bars[bar.symbol]
        if history and bar.timestamp <= history[-1].timestamp:
            if bar.timestamp == history[-1].timestamp:
                history[-1] = bar
            return
        history.append(bar)
        if len(history) > self._max_bars:
            del history[:-self._max_bars]

    def bars(self, symbol: str, timestamp=None) -> list[Bar]:
        rows = self._bars.get(symbol.upper(), [])
        return list(rows if timestamp is None else [b for b in rows if b.timestamp <= timestamp])

    def context(self, candidate: Candidate, ticker_bars: Sequence[Bar]) -> MarketContext:
        ts = ticker_bars[-1].timestamp
        own = _ret(ticker_bars, 5)
        spy_bars = self.bars("SPY", ts)
        qqq_bars = self.bars("QQQ", ts)
        vix_bars = self.bars("VIXY", ts)
        sector_bars = self.bars(candidate.sector_etf or "", ts) if candidate.sector_etf else []
        spy = _normalized_direction(spy_bars)
        qqq = _normalized_direction(qqq_bars)
        sector_rs = float(np.clip((own - _ret(sector_bars, 5)) * 20.0, -1.0, 1.0)) if sector_bars else 0.0
        index_vwap_alignment = 0.5 * (_vwap_side(spy_bars) + _vwap_side(qqq_bars))
        volatility = float(np.clip(_ret(vix_bars, 10) * 10.0, -1.0, 1.0)) if vix_bars else 0.0
        alignment = float(np.clip(0.5 + 0.18 * spy + 0.18 * qqq + 0.10 * sector_rs + 0.04 * index_vwap_alignment - 0.08 * max(0.0, volatility), 0.0, 1.0))
        warnings: list[str] = []
        if not spy_bars or not qqq_bars:
            warnings.append("partial_index_context")
        return MarketContext(
            timestamp=ts, spy_direction=spy, qqq_direction=qqq,
            sector_relative_strength=sector_rs, volatility_regime=volatility,
            index_vwap_alignment=index_vwap_alignment,
            market_alignment_score=alignment, warnings=tuple(warnings),
        )

    def snapshot(self) -> dict[str, list[dict]]:
        return {symbol: [bar.to_dict() for bar in bars] for symbol, bars in self._bars.items()}

    def restore(self, raw: dict[str, list[dict]]) -> None:
        self._bars.clear()
        for symbol, rows in raw.items():
            self._bars[symbol] = [Bar.from_mapping(row) for row in rows][-self._max_bars:]


def _ret(bars: Sequence[Bar], periods: int) -> float:
    if len(bars) <= periods or bars[-periods - 1].close <= 0:
        return 0.0
    return bars[-1].close / bars[-periods - 1].close - 1.0


def _normalized_direction(bars: Sequence[Bar]) -> float:
    if len(bars) < 6:
        return 0.0
    ranges = [b.high - b.low for b in bars[-15:]]
    atr = max(float(np.mean(ranges)), bars[-1].close * 1e-6)
    return float(np.clip((bars[-1].close - bars[-6].close) / (atr * 3.0), -1.0, 1.0))


def _vwap_side(bars: Sequence[Bar]) -> float:
    if not bars:
        return 0.0
    subset = bars[-120:]
    volume = sum(b.volume for b in subset)
    if volume <= 0:
        return 0.0
    vwap = sum(((b.high + b.low + b.close) / 3.0) * b.volume for b in subset) / volume
    return 1.0 if subset[-1].close >= vwap else -1.0
