from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from strategies.intraday_structure.config import IntradayStructureConfig
from strategies.intraday_structure.engine import IntradayStructureEngine
from strategies.intraday_structure.labels import LabelConfig, build_event_labels
from strategies.intraday_structure.models import Bar, Candidate, SetupState
from strategies.intraday_structure.options import NullOptionsProvider, OptionsProvider


@dataclass(frozen=True)
class ReplayResult:
    transitions: pd.DataFrame
    labels: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict


class EventReplay:
    def __init__(self, config: IntradayStructureConfig, *, options_provider: OptionsProvider | None = None) -> None:
        self.config = config
        self.options_provider = options_provider or NullOptionsProvider()

    def run(self, bars: pd.DataFrame, candidates: Iterable[Candidate]) -> ReplayResult:
        frame = _normalize_one_minute_bars(bars)
        pending = sorted(candidates, key=lambda c: c.available_at or c.timestamp)
        engine = IntradayStructureEngine(self.config, options_provider=self.options_provider)
        candidate_index = 0
        for row in frame.itertuples(index=False):
            while candidate_index < len(pending) and (pending[candidate_index].available_at or pending[candidate_index].timestamp) <= row.timestamp.to_pydatetime():
                engine.register_candidate(pending[candidate_index])
                candidate_index += 1
            engine.on_bar(Bar.from_mapping(row._asdict()))
        transition_frame = pd.DataFrame([transition.to_dict() for transition in engine.transitions])
        confirmed = []
        for setup in engine.setups.values():
            if setup.entry_price is None or not setup.targets or setup.invalidation is None:
                continue
            confirmed.append({
                "ticker": setup.ticker, "timestamp": setup.entry_time,
                "setup_type": setup.setup_type.value, "direction": setup.direction.value,
                "entry_price": setup.entry_price,
                "invalidation": setup.metadata.get("initial_invalidation", setup.invalidation),
                "targets": setup.targets, "pivot": setup.pivot,
                "confidence": setup.confidence,
            })
        labels = build_event_labels(
            frame, confirmed,
            config=LabelConfig(self.config.replay.label_forward_bars, self.config.replay.overlap_cooldown_bars),
        )
        trades = self._trade_frame(engine)
        return ReplayResult(transition_frame, labels, trades, _metrics(labels, trades))

    def _trade_frame(self, engine: IntradayStructureEngine) -> pd.DataFrame:
        costs = (self.config.replay.spread_bps + self.config.replay.slippage_bps) / 10_000.0
        rows = []
        for setup in engine.setups.values():
            if setup.entry_price is None:
                continue
            initial_stop = float(setup.metadata.get("initial_invalidation", setup.invalidation or setup.entry_price))
            risk = abs(setup.entry_price - initial_stop)
            sign = 1.0 if setup.direction.value == "long" else -1.0
            exit_price = float(setup.metadata.get("exit_price", setup.spot or setup.entry_price))
            gross = sign * (exit_price - setup.entry_price)
            net = gross - setup.entry_price * costs - 2.0 * self.config.replay.commission_per_share
            rows.append({
                "setup_id": setup.setup_id, "ticker": setup.ticker, "setup_type": setup.setup_type.value,
                "direction": setup.direction.value, "entry_time": setup.entry_time,
                "entry_price": setup.entry_price, "exit_time": setup.updated_at,
                "exit_price": exit_price, "final_state": setup.state.value,
                "gross_points": gross, "net_points": net, "realized_r_after_costs": net / risk if risk > 0 else np.nan,
                "mfe_points": setup.max_favorable_excursion, "mae_points": setup.max_adverse_excursion,
                "confidence": setup.confidence, "runway_score": setup.runway_score,
            })
        return pd.DataFrame(rows)


def run_ablations(
    bars: pd.DataFrame,
    candidates: Iterable[Candidate],
    config: IntradayStructureConfig,
    *,
    static_options: OptionsProvider | None = None,
    live_flow: OptionsProvider | None = None,
    full_options: OptionsProvider | None = None,
) -> pd.DataFrame:
    providers = {
        "A_price_structure": NullOptionsProvider(),
        "B_static_options": static_options or NullOptionsProvider(),
        "C_static_plus_flow": live_flow or static_options or NullOptionsProvider(),
        "D_full_context": full_options or live_flow or static_options or NullOptionsProvider(),
    }
    rows = []
    materialized = list(candidates)
    for name, provider in providers.items():
        result = EventReplay(config, options_provider=provider).run(bars, materialized)
        rows.append({"ablation": name, **result.metrics})
    return pd.DataFrame(rows)


def _normalize_one_minute_bars(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"replay bars missing columns: {sorted(missing)}")
    out = frame.copy()
    if "symbol" not in out:
        if "ticker" not in out:
            raise KeyError("replay bars require symbol or ticker")
        out["symbol"] = out["ticker"]
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values(["timestamp", "symbol"]).drop_duplicates(["symbol", "timestamp"], keep="last")
    for _symbol, group in out.groupby("symbol"):
        diffs = group["timestamp"].diff().dropna().dt.total_seconds()
        intraday = diffs[diffs < 6 * 3600]
        if not intraday.empty and float(intraday.median()) > 90.0:
            raise ValueError("event replay requires true 1-minute bars; higher-timeframe resampling is prohibited")
    return out.reset_index(drop=True)


def _metrics(labels: pd.DataFrame, trades: pd.DataFrame) -> dict:
    if labels.empty:
        return {"trade_count": int(len(trades)), "target_before_invalidation_rate": None, "win_rate": None, "average_r": None}
    wins = labels["target_before_invalidation"].astype(bool)
    return {
        "trade_count": int(len(trades)),
        "labeled_event_count": int(len(labels)),
        "target_before_invalidation_rate": float(wins.mean()),
        "win_rate": float((trades["realized_r_after_costs"] > 0).mean()) if not trades.empty else None,
        "average_r": float(trades["realized_r_after_costs"].mean()) if not trades.empty else None,
        "average_mfe": float(labels["max_favorable_excursion"].mean()),
        "average_mae": float(labels["max_adverse_excursion"].mean()),
        "average_time_to_target": float(labels["time_to_target"].dropna().mean()) if labels["time_to_target"].notna().any() else None,
    }


def write_replay_result(result: ReplayResult, output_dir: str | Path) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    result.transitions.to_csv(root / "transitions.csv", index=False)
    result.labels.to_csv(root / "event_labels.csv", index=False)
    result.trades.to_csv(root / "trades.csv", index=False)
    (root / "metrics.json").write_text(pd.Series(result.metrics).to_json(indent=2), encoding="utf-8")
