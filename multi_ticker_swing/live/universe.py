"""
Trading universe loader.

Reads multi_ticker_swing/config/trading_universe.json which is produced by the
sweep_v4 analysis. Contains per-ticker tier config and historical trade stats used
for EV-score signal ranking.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config" / "trading_universe.json"


@dataclass(frozen=True)
class TickerConfig:
    ticker: str
    tier: int
    entry_threshold: float   # directional-conditional prob threshold (0.60 or 0.70)
    sl_atr: float            # hard stop in ATR multiples (4.0 or 0.0)
    np_n_bars: int | None    # no-progress check at this many 5m bars (None = disabled)
    np_mfe_atr: float | None # exit if MFE < this × ATR at bar np_n_bars
    avg_win_pct: float       # historical avg win trade return (for EV score)
    avg_loss_pct: float      # historical avg loss trade return (negative, for EV score)
    profit_factor: float
    sharpe: float

    def ev_score(self, p_dir: float) -> float:
        """Expected trade PnL at this signal strength."""
        return p_dir * self.avg_win_pct + (1.0 - p_dir) * self.avg_loss_pct


MAX_TIER = 2  # only trade Tier 1 and Tier 2; Tier 3 (negative Sharpe) excluded


def load_universe(
    path: Path | str = _DEFAULT_PATH,
    max_tier: int = MAX_TIER,
) -> dict[str, TickerConfig]:
    """Returns {ticker: TickerConfig} for tickers up to max_tier (inclusive)."""
    data = json.loads(Path(path).read_text())
    return {
        ticker: TickerConfig(
            ticker=ticker,
            tier=int(v["tier"]),
            entry_threshold=float(v["entry_threshold"]),
            sl_atr=float(v["sl_atr"]),
            np_n_bars=int(v["np_n_bars"]) if v.get("np_n_bars") is not None else None,
            np_mfe_atr=float(v["np_mfe_atr"]) if v.get("np_mfe_atr") is not None else None,
            avg_win_pct=float(v["avg_win_pct"]),
            avg_loss_pct=float(v["avg_loss_pct"]),
            profit_factor=float(v["profit_factor"]),
            sharpe=float(v["sharpe"]),
        )
        for ticker, v in data.items()
        if int(v["tier"]) <= max_tier
    }
