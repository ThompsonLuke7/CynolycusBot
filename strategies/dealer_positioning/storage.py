from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from strategies.dealer_positioning.config import DealerPositioningConfig, symbol_output_dir
from strategies.dealer_positioning.models import DealerSignal, GammaLevels, SimTrade


class DealerPositioningWriter:
    def __init__(self, config: DealerPositioningConfig) -> None:
        self._config = config

    def write_snapshot(
        self,
        *,
        symbol: str,
        ladder: pd.DataFrame,
        levels: GammaLevels,
        open_trades: Iterable[SimTrade],
        closed_trades: Iterable[SimTrade],
    ) -> None:
        out_dir = symbol_output_dir(self._config, symbol)
        out_dir.mkdir(parents=True, exist_ok=True)
        ladder.to_csv(out_dir / "live_gamma_ladder.csv", index=False)
        (out_dir / "live_gamma_levels.json").write_text(
            json.dumps(levels.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.write_trades(symbol=symbol, open_trades=open_trades, closed_trades=closed_trades)
        if self._config.write_archive_snapshots:
            archive = out_dir / "snapshots"
            archive.mkdir(parents=True, exist_ok=True)
            safe_ts = levels.timestamp.replace(":", "").replace("+", "_").replace("-", "")
            ladder.to_csv(archive / f"gamma_ladder_{safe_ts}.csv", index=False)

    def write_trades(
        self,
        *,
        symbol: str,
        open_trades: Iterable[SimTrade],
        closed_trades: Iterable[SimTrade],
    ) -> None:
        out_dir = symbol_output_dir(self._config, symbol)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "live_trades.json").write_text(
            json.dumps(
                {
                    "open_trades": [t.to_dict() for t in open_trades],
                    "closed_trades": [t.to_dict() for t in closed_trades],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def append_signal(self, signal: DealerSignal) -> None:
        out_dir = symbol_output_dir(self._config, signal.symbol)
        out_dir.mkdir(parents=True, exist_ok=True)
        _append_csv(out_dir / "live_signal_log.csv", signal.to_dict())

    def append_trade(self, trade: SimTrade) -> None:
        out_dir = symbol_output_dir(self._config, trade.symbol)
        out_dir.mkdir(parents=True, exist_ok=True)
        _append_csv(out_dir / "live_trade_log.csv", trade.to_dict())


def _append_csv(path: Path, row: dict) -> None:
    exists = path.exists()
    fieldnames = list(row.keys())
    if exists:
        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            existing_fieldnames = list(reader.fieldnames or [])
            if existing_fieldnames and existing_fieldnames != fieldnames:
                rows = list(reader)
                fieldnames = list(dict.fromkeys([*existing_fieldnames, *fieldnames]))
                rows.append({key: row.get(key) for key in fieldnames})
                with path.open("w", newline="", encoding="utf-8") as out_fh:
                    writer = csv.DictWriter(out_fh, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                return
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
