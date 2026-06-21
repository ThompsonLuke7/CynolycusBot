from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone

from strategies.dealer_positioning.config import DealerPositioningConfig
from strategies.dealer_positioning.models import DealerSignal, GammaLevels, SimTrade


@dataclass(frozen=True)
class PriceCandle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class _PendingBreak:
    side: str
    level: float
    target: float
    stop: float
    candles_held: int = 0


class DealerSignalEngine:
    def __init__(self, config: DealerPositioningConfig) -> None:
        self._config = config
        self._candles: dict[str, deque[PriceCandle]] = defaultdict(lambda: deque(maxlen=120))
        self._pending: dict[str, _PendingBreak] = {}
        self._last_signal_ts: dict[tuple[str, str], datetime] = {}

    def on_candle(self, symbol: str, candle: PriceCandle, levels: GammaLevels) -> list[DealerSignal]:
        symbol = symbol.upper()
        candles = self._candles[symbol]
        prev_close = candles[-1].close if candles else None
        candles.append(candle)

        out: list[DealerSignal] = []
        out.extend(self._magnet_signals(symbol, candle, prev_close, levels))
        out.extend(self._wall_signals(symbol, candle, levels))
        return [sig for sig in out if self._cooldown_ok(sig)]

    def _magnet_signals(
        self,
        symbol: str,
        candle: PriceCandle,
        prev_close: float | None,
        levels: GammaLevels,
    ) -> list[DealerSignal]:
        if levels.nearest_magnet is None or prev_close is None:
            return []
        current = self._pending.get(symbol)
        if current is not None:
            held = candle.close > current.level if current.side == "above" else candle.close < current.level
            if held:
                current.candles_held += 1
            else:
                self._pending.pop(symbol, None)
                return []
            if current.candles_held >= self._config.hold_candles:
                self._pending.pop(symbol, None)
                signal_type = "bullish_magnet" if current.side == "above" else "bearish_magnet"
                direction = "long" if current.side == "above" else "short"
                return [
                    DealerSignal(
                        timestamp=_iso(candle.timestamp),
                        symbol=symbol,
                        signal_type=signal_type,
                        direction=direction,
                        spot=float(candle.close),
                        level=float(current.level),
                        target=float(current.target),
                        stop=float(current.stop),
                        reason=f"held {current.side} broken magnet for {current.candles_held} candles",
                        confidence=float(min(1.0, abs(current.target - current.level) / max(1.0, current.level * 0.01))),
                    )
                ]
            return []

        magnet = float(levels.nearest_magnet)
        if (
            prev_close <= magnet < candle.close
            and levels.next_magnet_above is not None
            and levels.air_gap_above_score > self._config.air_gap_threshold
        ):
            self._pending[symbol] = _PendingBreak(
                side="above",
                level=magnet,
                target=float(levels.next_magnet_above),
                stop=magnet,
                candles_held=1,
            )
        elif (
            prev_close >= magnet > candle.close
            and levels.next_magnet_below is not None
            and levels.air_gap_below_score > self._config.air_gap_threshold
        ):
            self._pending[symbol] = _PendingBreak(
                side="below",
                level=magnet,
                target=float(levels.next_magnet_below),
                stop=magnet,
                candles_held=1,
            )
        return []

    def _wall_signals(self, symbol: str, candle: PriceCandle, levels: GammaLevels) -> list[DealerSignal]:
        candles = self._candles[symbol]
        if len(candles) < 3:
            return []
        if (
            levels.call_wall is not None
            and levels.put_wall is not None
            and self._config.max_wall_channel_width > 0
            and abs(float(levels.call_wall) - float(levels.put_wall)) > self._config.max_wall_channel_width
        ):
            return []
        prev = candles[-2]
        out: list[DealerSignal] = []
        touch_pct = self._config.wall_touch_pct

        if levels.call_wall is not None:
            wall = float(levels.call_wall)
            approached_from_below = prev.close < wall and candle.high >= wall * (1.0 - touch_pct)
            momentum_weak = candle.close <= prev.close or candle.close < candle.open
            if approached_from_below and momentum_weak:
                target = self._wall_target(levels, float(candle.close), "short")
                if target is None:
                    return out
                out.append(
                    DealerSignal(
                        timestamp=_iso(candle.timestamp),
                        symbol=symbol,
                        signal_type="call_wall_rejection",
                        direction="short",
                        spot=float(candle.close),
                        level=wall,
                        target=target,
                        stop=wall + float(self._config.wall_stop_points),
                        reason="price reached call wall from below and 1m momentum stalled",
                        confidence=0.55,
                    )
                )

        if levels.put_wall is not None:
            wall = float(levels.put_wall)
            approached_from_above = prev.close > wall and candle.low <= wall * (1.0 + touch_pct)
            selling_weak = candle.close >= prev.close or candle.close > candle.open
            if approached_from_above and selling_weak:
                target = self._wall_target(levels, float(candle.close), "long")
                if target is None:
                    return out
                out.append(
                    DealerSignal(
                        timestamp=_iso(candle.timestamp),
                        symbol=symbol,
                        signal_type="put_wall_bounce",
                        direction="long",
                        spot=float(candle.close),
                        level=wall,
                        target=target,
                        stop=wall - float(self._config.wall_stop_points),
                        reason="price reached put wall from above and selling stalled",
                        confidence=0.55,
                    )
                )
        return out

    def _wall_target(self, levels: GammaLevels, entry: float, direction: str) -> float | None:
        min_move = max(0.0, float(self._config.min_wall_target_points))
        if direction == "long":
            candidates = [levels.nearest_magnet, levels.next_magnet_above, levels.call_wall]
            valid = sorted(float(x) for x in candidates if x is not None and float(x) >= entry + min_move)
            return valid[0] if valid else None
        candidates = [levels.nearest_magnet, levels.next_magnet_below, levels.put_wall]
        valid = sorted(
            (float(x) for x in candidates if x is not None and float(x) <= entry - min_move),
            reverse=True,
        )
        return valid[0] if valid else None

    def _cooldown_ok(self, signal: DealerSignal) -> bool:
        if signal.signal_type not in self._config.enabled_signal_types:
            return False
        key = (signal.symbol, signal.signal_type)
        now = _parse_iso(signal.timestamp)
        last = self._last_signal_ts.get(key)
        if last is not None and (now - last).total_seconds() < self._config.signal_cooldown_seconds:
            return False
        self._last_signal_ts[key] = now
        return True


class DealerTradeSimulator:
    def __init__(self, config: DealerPositioningConfig) -> None:
        self._config = config
        self._open: dict[str, list[SimTrade]] = defaultdict(list)
        self._closed: list[SimTrade] = []
        self._counter = 0

    @property
    def open_trades(self) -> list[SimTrade]:
        return [trade for trades in self._open.values() for trade in trades]

    @property
    def closed_trades(self) -> list[SimTrade]:
        return list(self._closed)

    def on_signal(self, signal: DealerSignal) -> SimTrade | None:
        open_for_symbol = [t for t in self._open[signal.symbol] if t.status == "open"]
        if len(open_for_symbol) >= self._config.max_trades_per_symbol:
            return None
        self._counter += 1
        trade = SimTrade(
            trade_id=f"{signal.symbol}-{self._counter:06d}",
            opened_at=signal.timestamp,
            symbol=signal.symbol,
            signal_type=signal.signal_type,
            direction=signal.direction,
            entry=float(signal.spot),
            target=signal.target,
            stop=signal.stop,
        )
        self._open[signal.symbol].append(trade)
        return trade

    def on_candle(self, symbol: str, candle: PriceCandle) -> list[SimTrade]:
        symbol = symbol.upper()
        closed: list[SimTrade] = []
        still_open: list[SimTrade] = []
        for trade in self._open.get(symbol, []):
            if trade.status != "open":
                continue
            exit_price: float | None = None
            reason: str | None = None
            if trade.direction == "long":
                if trade.stop is not None and candle.low <= trade.stop:
                    exit_price, reason = float(trade.stop), "stop"
                elif trade.target is not None and candle.high >= trade.target:
                    exit_price, reason = float(trade.target), "target"
            else:
                if trade.stop is not None and candle.high >= trade.stop:
                    exit_price, reason = float(trade.stop), "stop"
                elif trade.target is not None and candle.low <= trade.target:
                    exit_price, reason = float(trade.target), "target"

            if exit_price is None:
                still_open.append(trade)
                continue

            trade.status = "closed"
            trade.closed_at = _iso(candle.timestamp)
            trade.exit_price = exit_price
            trade.exit_reason = reason
            mult = 1.0 if trade.direction == "long" else -1.0
            trade.pnl_points = (exit_price - trade.entry) * mult
            closed.append(trade)
            self._closed.append(trade)

        self._open[symbol] = still_open
        return closed


def _iso(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat()


def _parse_iso(raw: str) -> datetime:
    value = raw.replace("Z", "+00:00")
    return datetime.fromisoformat(value).astimezone(timezone.utc)
