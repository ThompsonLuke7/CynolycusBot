from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DealerPositioningConfig:
    """Runtime knobs for the non-ML dealer positioning module."""

    symbols: tuple[str, ...] = ("SPY", "SPX", "QQQ", "SLV", "IWM", "GLD", "SMH")
    dte_offsets: tuple[int, ...] = (0, 1, 2)
    poll_seconds: int = 60
    output_root: Path = Path("Data/dealer_positioning")
    strike_window_pct: float = 0.04
    magnet_quantile: float = 0.90
    air_gap_threshold: float = 2.0
    wall_touch_pct: float = 0.0015
    min_wall_target_points: float = 0.40
    max_wall_channel_width: float = 5.0
    wall_stop_points: float = 1.10
    max_trades_per_symbol: int = 1
    hold_candles: int = 2
    signal_cooldown_seconds: int = 15 * 60
    chain_strike_count: int = 80
    market_hours_only: bool = True
    submit_orders: bool = False
    alpaca_env_file: str = ".env#paper"
    option_order_qty: int = 1
    option_time_in_force: str = "day"
    option_order_type: str = "limit"
    write_archive_snapshots: bool = True
    enabled_signal_types: tuple[str, ...] = field(
        default=("bullish_magnet", "put_wall_bounce")
    )

    @classmethod
    def from_env(cls) -> "DealerPositioningConfig":
        """Build config from environment variables when present."""
        import os

        def _tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
            raw = os.getenv(name)
            if not raw:
                return default
            return tuple(x.strip().upper() for x in raw.split(",") if x.strip())

        def _raw_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
            raw = os.getenv(name)
            if not raw:
                return default
            return tuple(x.strip() for x in raw.split(",") if x.strip())

        def _int_tuple(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
            raw = os.getenv(name)
            if not raw:
                return default
            return tuple(int(x.strip()) for x in raw.split(",") if x.strip())

        return cls(
            symbols=_tuple("DEALER_POSITIONING_SYMBOLS", cls.symbols),
            dte_offsets=_int_tuple("DEALER_POSITIONING_DTES", cls.dte_offsets),
            poll_seconds=int(os.getenv("DEALER_POSITIONING_POLL_SECONDS", cls.poll_seconds)),
            output_root=Path(os.getenv("DEALER_POSITIONING_OUTPUT_ROOT", str(cls.output_root))),
            strike_window_pct=float(os.getenv("DEALER_POSITIONING_STRIKE_WINDOW_PCT", cls.strike_window_pct)),
            magnet_quantile=float(os.getenv("DEALER_POSITIONING_MAGNET_QUANTILE", cls.magnet_quantile)),
            air_gap_threshold=float(os.getenv("DEALER_POSITIONING_AIR_GAP_THRESHOLD", cls.air_gap_threshold)),
            wall_touch_pct=float(os.getenv("DEALER_POSITIONING_WALL_TOUCH_PCT", cls.wall_touch_pct)),
            min_wall_target_points=float(
                os.getenv("DEALER_POSITIONING_MIN_WALL_TARGET_POINTS", cls.min_wall_target_points)
            ),
            max_wall_channel_width=float(
                os.getenv("DEALER_POSITIONING_MAX_WALL_CHANNEL_WIDTH", cls.max_wall_channel_width)
            ),
            wall_stop_points=float(os.getenv("DEALER_POSITIONING_WALL_STOP_POINTS", cls.wall_stop_points)),
            max_trades_per_symbol=int(os.getenv("DEALER_POSITIONING_MAX_TRADES_PER_SYMBOL", cls.max_trades_per_symbol)),
            hold_candles=int(os.getenv("DEALER_POSITIONING_HOLD_CANDLES", cls.hold_candles)),
            signal_cooldown_seconds=int(
                os.getenv("DEALER_POSITIONING_SIGNAL_COOLDOWN_SECONDS", cls.signal_cooldown_seconds)
            ),
            chain_strike_count=int(os.getenv("DEALER_POSITIONING_CHAIN_STRIKE_COUNT", cls.chain_strike_count)),
            market_hours_only=os.getenv("DEALER_POSITIONING_MARKET_HOURS_ONLY", "1").strip().lower()
            not in {"0", "false", "no"},
            submit_orders=os.getenv("DEALER_POSITIONING_SUBMIT_ORDERS", "0").strip().lower()
            in {"1", "true", "yes"},
            alpaca_env_file=os.getenv("DEALER_POSITIONING_ALPACA_ENV_FILE", cls.alpaca_env_file),
            option_order_qty=int(os.getenv("DEALER_POSITIONING_OPTION_ORDER_QTY", cls.option_order_qty)),
            option_time_in_force=os.getenv("DEALER_POSITIONING_OPTION_TIF", cls.option_time_in_force),
            option_order_type=os.getenv("DEALER_POSITIONING_OPTION_ORDER_TYPE", cls.option_order_type),
            enabled_signal_types=_raw_tuple("DEALER_POSITIONING_ENABLED_SIGNAL_TYPES", cls.enabled_signal_types),
        )


def symbol_output_dir(config: DealerPositioningConfig, symbol: str) -> Path:
    return config.output_root / symbol.strip().upper()
