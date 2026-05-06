from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.probe_probability_normalization import (
    DEFAULT_ONE_MIN,
    DEFAULT_RUN_ROOT,
    DEFAULT_SIGNAL_FRAME,
    SIM_KW,
    _add_normalized_columns,
    _load_live_decisions,
    _rank_percentile,
    _rolling_percentile,
    _with_regime,
)
from scripts.sweep_live_thresholds_post_0401 import (
    _load_decisions_from_signal_frame,
    _load_one_min,
    _metrics,
    _run_one,
    _to_et,
)


DEFAULT_OUT = Path("Data/inference/spy/10min/setup/probability_normalization_experiment_summary.csv")


@dataclass(frozen=True)
class ShortConfig:
    name: str
    source_col: str
    threshold: float
    raw_floor: float | None = None
    max_opp_long: float | None = None
    min_short_edge: float | None = None


def _short_configs() -> list[ShortConfig]:
    configs: list[ShortConfig] = []

    for threshold in (0.20, 0.25, 0.30, 0.40, 0.65):
        configs.append(ShortConfig(name=f"raw_s{threshold:.2f}", source_col="p_enter_short", threshold=threshold))

    for source in ("p_short_fixed_history_pct", "p_short_regime_history_pct"):
        prefix = source.replace("p_short_", "").replace("_pct", "")
        for threshold in (0.75, 0.80, 0.85, 0.90, 0.95):
            configs.append(ShortConfig(name=f"{prefix}_q{threshold:.2f}", source_col=source, threshold=threshold))

    for window in (195, 780):
        source = f"p_short_rolling_{window}_pct"
        for threshold in (0.75, 0.80, 0.85, 0.90, 0.95):
            configs.append(ShortConfig(name=f"roll{window}_q{threshold:.2f}", source_col=source, threshold=threshold))

    for threshold in (0.75, 0.80, 0.85, 0.90):
        configs.append(
            ShortConfig(
                name=f"session_q{threshold:.2f}",
                source_col="p_short_session_expanding_pct",
                threshold=threshold,
            )
        )

    overlay_specs = [
        (195, 0.75, 0.20, 0.15, 0.15),
        (195, 0.80, 0.15, 0.15, 0.15),
        (195, 0.80, 0.20, None, 0.15),
        (195, 0.80, 0.20, 0.15, 0.00),
        (195, 0.80, 0.20, 0.15, 0.15),
        (195, 0.80, 0.20, 0.15, 0.25),
        (195, 0.80, 0.20, 0.08, 0.15),
        (195, 0.80, 0.25, 0.15, 0.15),
        (195, 0.85, 0.20, 0.15, 0.15),
        (780, 0.75, 0.20, 0.15, 0.15),
        (780, 0.80, 0.20, 0.15, 0.15),
        (780, 0.80, 0.25, 0.15, 0.15),
        (780, 0.85, 0.20, 0.15, 0.15),
        (780, 0.90, 0.20, 0.15, 0.15),
    ]
    for window, threshold, raw_floor, max_opp, edge in overlay_specs:
        cap = "none" if max_opp is None else f"{max_opp:.2f}"
        name = (
            f"roll{window}_q{threshold:.2f}_floor{raw_floor:.2f}"
            f"_opp{cap}_edge{edge:.2f}"
        )
        configs.append(
            ShortConfig(
                name=name,
                source_col=f"p_short_rolling_{window}_pct",
                threshold=threshold,
                raw_floor=raw_floor,
                max_opp_long=max_opp,
                min_short_edge=edge,
            )
        )

    session_overlay_specs = [
        (0.75, 0.15, 0.15, 0.15),
        (0.75, 0.20, 0.15, 0.15),
        (0.80, 0.15, 0.15, 0.15),
        (0.80, 0.20, 0.15, 0.15),
        (0.80, 0.20, 0.08, 0.15),
        (0.80, 0.20, 0.15, 0.25),
        (0.85, 0.20, 0.15, 0.15),
    ]
    for threshold, raw_floor, max_opp, edge in session_overlay_specs:
        configs.append(
            ShortConfig(
                name=(
                    f"session_q{threshold:.2f}_floor{raw_floor:.2f}"
                    f"_opp{max_opp:.2f}_edge{edge:.2f}"
                ),
                source_col="p_short_session_expanding_pct",
                threshold=threshold,
                raw_floor=raw_floor,
                max_opp_long=max_opp,
                min_short_edge=edge,
            )
        )

    return configs


def _load_signal_dataset(
    *,
    signal_frame: Path,
    one_min_path: Path,
    history_start: str,
    start: str,
    end: str,
    prob_source: str,
    windows: list[int],
    min_periods: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sim_start = _to_et(start)
    decisions = _load_decisions_from_signal_frame(
        signal_frame=signal_frame,
        prob_frame=None,
        start=history_start,
        prob_source=prob_source,
    )
    decisions = _with_regime(decisions, signal_frame)
    decisions = _add_normalized_columns(
        decisions,
        sim_start=sim_start,
        windows=windows,
        min_periods=min_periods,
    )
    decisions = _add_session_expanding_percentile(decisions, min_periods=3)
    sim = decisions[decisions["timestamp"] >= sim_start].copy().reset_index(drop=True)
    if end:
        sim_end = _to_et(end)
        sim = sim[sim["timestamp"] <= sim_end].copy().reset_index(drop=True)
    else:
        sim_end = sim["timestamp"].max()
    one_min = _load_one_min(one_min_path, start, sim_end)
    one_min = one_min[one_min["timestamp"] >= sim["timestamp"].min() - pd.Timedelta(minutes=10)].copy()
    return sim, one_min


def _load_recent_live_dataset(
    *,
    signal_frame: Path,
    one_min_path: Path,
    run_root: Path,
    history_start: str,
    live_start: str,
    live_end: str,
    prob_source: str,
    windows: list[int],
    min_periods: int,
    symbol: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sim_start = _to_et(live_start)
    history = _load_decisions_from_signal_frame(
        signal_frame=signal_frame,
        prob_frame=None,
        start=history_start,
        prob_source=prob_source,
    )
    history = _with_regime(history, signal_frame)
    history = history[history["timestamp"] < sim_start].copy()
    live = _load_live_decisions(run_root, live_start, symbol=symbol)
    if live_end:
        live = live[live["timestamp"] <= _to_et(live_end)].copy()
    if live.empty:
        return live, pd.DataFrame()

    combined = pd.concat([history, live], ignore_index=True, sort=False)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    combined["trend_regime"] = combined["trend_regime"].fillna("neutral")
    calibration = combined[combined["timestamp"] < sim_start]
    combined["p_short_fixed_history_pct"] = _rank_percentile(
        combined["p_enter_short"],
        calibration["p_enter_short"],
    )
    combined["p_short_regime_history_pct"] = combined["p_short_fixed_history_pct"]
    for window in windows:
        combined[f"p_short_rolling_{window}_pct"] = _rolling_percentile(
            combined["p_enter_short"],
            window=window,
            min_periods=min_periods,
        )
    combined = _add_session_expanding_percentile(combined, min_periods=3)

    sim = combined[combined["timestamp"] >= sim_start].copy().reset_index(drop=True)
    one_min = _load_one_min(one_min_path, live_start, sim["timestamp"].max())
    one_min = one_min[one_min["timestamp"] >= sim["timestamp"].min() - pd.Timedelta(minutes=10)].copy()
    return sim, one_min


def _add_session_expanding_percentile(decisions: pd.DataFrame, *, min_periods: int) -> pd.DataFrame:
    out = decisions.copy()
    values = pd.to_numeric(out["p_enter_short"], errors="coerce")
    scores = np.full(len(out), np.nan, dtype=float)
    sessions = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").dt.tz_convert("America/New_York").dt.date
    for _, idx in pd.Series(np.arange(len(out)), index=out.index).groupby(sessions).groups.items():
        arr = values.iloc[list(idx)].to_numpy(dtype=float)
        for pos, value in enumerate(arr):
            if not np.isfinite(value):
                continue
            hist = arr[:pos]
            hist = hist[np.isfinite(hist)]
            if hist.size < min_periods:
                continue
            scores[list(idx)[pos]] = np.searchsorted(np.sort(hist), value, side="right") / hist.size
    out["p_short_session_expanding_pct"] = scores
    return out


def _apply_short_config(decisions: pd.DataFrame, config: ShortConfig) -> pd.DataFrame:
    out = decisions.copy()
    signal = pd.to_numeric(out[config.source_col], errors="coerce")
    raw_short = pd.to_numeric(out["p_enter_short"], errors="coerce")
    raw_long = pd.to_numeric(out["p_enter_long"], errors="coerce")
    mask = signal.notna()
    if config.raw_floor is not None:
        mask &= raw_short >= float(config.raw_floor)
    if config.max_opp_long is not None:
        mask &= raw_long <= float(config.max_opp_long)
    if config.min_short_edge is not None:
        mask &= (raw_short - raw_long) >= float(config.min_short_edge)
    out["p_enter_short"] = signal.where(mask)
    return out


def _side_metrics(events: list[dict[str, Any]]) -> dict[str, float | int]:
    long_returns = np.array([float(e["return"]) for e in events if e.get("side") == "long"], dtype=float)
    short_returns = np.array([float(e["return"]) for e in events if e.get("side") == "short"], dtype=float)
    return {
        "long_sum_return": float(np.nansum(long_returns)) if long_returns.size else 0.0,
        "short_sum_return": float(np.nansum(short_returns)) if short_returns.size else 0.0,
        "long_avg_return": float(np.nanmean(long_returns)) if long_returns.size else float("nan"),
        "short_avg_return": float(np.nanmean(short_returns)) if short_returns.size else float("nan"),
        "short_win_rate": float(np.nanmean(short_returns > 0)) if short_returns.size else float("nan"),
    }


def _run_config(
    *,
    dataset: str,
    mode: str,
    config: ShortConfig,
    decisions: pd.DataFrame,
    one_min: pd.DataFrame,
) -> dict[str, Any]:
    sim_decisions = _apply_short_config(decisions, config)
    sim_decisions = sim_decisions.dropna(subset=["p_enter_long"]).reset_index(drop=True)
    long_thr = 2.0 if mode == "short_only" else 0.35
    events = _run_one(
        decisions=sim_decisions,
        one_min=one_min,
        long_thr=long_thr,
        short_thr=float(config.threshold),
        **SIM_KW,
    )
    metrics = _metrics(events)
    metrics.update(_side_metrics(events))
    metrics.update(
        {
            "dataset": dataset,
            "mode": mode,
            "config": config.name,
            "short_source": config.source_col,
            "short_threshold": config.threshold,
            "raw_floor": config.raw_floor,
            "max_opp_long": config.max_opp_long,
            "min_short_edge": config.min_short_edge,
            "decision_rows": int(len(sim_decisions)),
            "first_decision": decisions["timestamp"].min(),
            "last_decision": decisions["timestamp"].max(),
        }
    )
    return metrics


def _selected_2024_configs(rows: pd.DataFrame) -> list[str]:
    top: list[str] = []
    short_pool = rows[(rows["dataset"] == "signal_ytd") & (rows["mode"] == "short_only")].copy()
    short_pool = short_pool.sort_values(["short_sum_return", "short_avg_return"], ascending=[False, False])
    top.extend(list(short_pool["config"].head(6)))
    full_pool = rows[(rows["dataset"] == "signal_ytd") & (rows["mode"] == "full_long035")].copy()
    full_pool = full_pool.sort_values(["sum_return", "short_sum_return"], ascending=[False, False])
    for name in list(full_pool["config"].head(6)):
        if name not in top:
            top.append(name)
    for name in ("raw_s0.65", "raw_s0.30", "roll195_q0.80"):
        if name not in top:
            top.append(name)
    return top


def _filter_configs(configs: list[ShortConfig], names_text: str) -> list[ShortConfig]:
    names = [x.strip() for x in str(names_text or "").split(",") if x.strip()]
    if not names:
        return configs
    name_set = set(names)
    selected = [cfg for cfg in configs if cfg.name in name_set]
    missing = sorted(name_set - {cfg.name for cfg in selected})
    if missing:
        raise SystemExit(f"Unknown config name(s): {', '.join(missing)}")
    return selected


def _parse_modes(text: str) -> list[str]:
    modes = [x.strip() for x in str(text or "").split(",") if x.strip()]
    if not modes:
        return ["short_only", "full_long035"]
    allowed = {"short_only", "full_long035"}
    bad = sorted(set(modes) - allowed)
    if bad:
        raise SystemExit(f"Unknown mode(s): {', '.join(bad)}")
    return modes


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SPY short probability normalization experiment matrix.")
    parser.add_argument("--signal-frame", default=str(DEFAULT_SIGNAL_FRAME))
    parser.add_argument("--one-min", default=str(DEFAULT_ONE_MIN))
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--history-start", default="2020-12-01T00:00:00-05:00")
    parser.add_argument("--prob-source", default="blend", choices=["blend", "full", "test", "oof"])
    parser.add_argument("--ytd-start", default="2026-01-02T00:00:00-05:00")
    parser.add_argument("--ytd-end", default="")
    parser.add_argument("--live-start", default="2026-04-01T00:00:00-04:00")
    parser.add_argument("--live-end", default="2026-05-01T23:59:00-04:00")
    parser.add_argument("--full-start", default="2024-01-02T00:00:00-05:00")
    parser.add_argument("--full-end", default="2026-04-01T23:59:00-04:00")
    parser.add_argument("--skip-full-2024", action="store_true")
    parser.add_argument(
        "--only-2024-selected-from",
        default="",
        help="Run only the selected 2024+ robustness slice, using this prior summary to choose configs.",
    )
    parser.add_argument("--windows", default="195,780")
    parser.add_argument("--min-periods", type=int, default=80)
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--config-names", default="")
    parser.add_argument("--modes", default="short_only,full_long035")
    args = parser.parse_args()

    windows = [int(x.strip()) for x in str(args.windows).split(",") if x.strip()]
    configs = _filter_configs(_short_configs(), args.config_names)
    modes = _parse_modes(args.modes)
    rows: list[dict[str, Any]] = []

    if str(args.only_2024_selected_from or "").strip():
        prior = pd.read_csv(args.only_2024_selected_from)
        selected_names = _selected_2024_configs(prior)
        selected = [cfg for cfg in configs if cfg.name in selected_names]
        if not selected:
            raise SystemExit("No selected 2024+ configs remain after --config-names filtering.")
        signal_2024, signal_2024_1m = _load_signal_dataset(
            signal_frame=Path(args.signal_frame),
            one_min_path=Path(args.one_min),
            history_start=args.history_start,
            start=args.full_start,
            end=args.full_end,
            prob_source=args.prob_source,
            windows=windows,
            min_periods=int(args.min_periods),
        )
        for mode in modes:
            for config in selected:
                rows.append(
                    _run_config(
                        dataset="signal_2024plus_selected",
                        mode=mode,
                        config=config,
                        decisions=signal_2024,
                        one_min=signal_2024_1m,
                    )
                )
        summary = pd.DataFrame(rows)
        sort_cols = ["dataset", "mode", "sum_return", "short_sum_return", "trades"]
        summary = summary.sort_values(sort_cols, ascending=[True, True, False, False, False])
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out, index=False)
        print(f"wrote {out}")
        view_cols = [
            "dataset",
            "mode",
            "config",
            "trades",
            "long_trades",
            "short_trades",
            "sum_return",
            "long_sum_return",
            "short_sum_return",
            "avg_return",
            "short_avg_return",
            "win_rate",
            "short_win_rate",
            "decision_rows",
        ]
        for mode in summary["mode"].drop_duplicates():
            top = summary[summary["mode"] == mode].head(12)
            print(f"\n=== signal_2024plus_selected / {mode} top 12 by total return ===")
            print(top[view_cols].to_string(index=False))
        return

    signal_ytd, signal_ytd_1m = _load_signal_dataset(
        signal_frame=Path(args.signal_frame),
        one_min_path=Path(args.one_min),
        history_start=args.history_start,
        start=args.ytd_start,
        end=args.ytd_end,
        prob_source=args.prob_source,
        windows=windows,
        min_periods=int(args.min_periods),
    )
    recent_live, recent_live_1m = _load_recent_live_dataset(
        signal_frame=Path(args.signal_frame),
        one_min_path=Path(args.one_min),
        run_root=Path(args.run_root),
        history_start=args.history_start,
        live_start=args.live_start,
        live_end=args.live_end,
        prob_source=args.prob_source,
        windows=windows,
        min_periods=int(args.min_periods),
        symbol=str(args.symbol),
    )

    for dataset, decisions, one_min in (
        ("signal_ytd", signal_ytd, signal_ytd_1m),
        ("recent_live", recent_live, recent_live_1m),
    ):
        if decisions.empty or one_min.empty:
            continue
        for mode in modes:
            for config in configs:
                rows.append(
                    _run_config(
                        dataset=dataset,
                        mode=mode,
                        config=config,
                        decisions=decisions,
                        one_min=one_min,
                    )
                )

    summary = pd.DataFrame(rows)

    if not args.skip_full_2024:
        selected_names = _selected_2024_configs(summary)
        selected = [cfg for cfg in configs if cfg.name in selected_names]
        signal_2024, signal_2024_1m = _load_signal_dataset(
            signal_frame=Path(args.signal_frame),
            one_min_path=Path(args.one_min),
            history_start=args.history_start,
            start=args.full_start,
            end=args.full_end,
            prob_source=args.prob_source,
            windows=windows,
            min_periods=int(args.min_periods),
        )
        for mode in modes:
            for config in selected:
                rows.append(
                    _run_config(
                        dataset="signal_2024plus_selected",
                        mode=mode,
                        config=config,
                        decisions=signal_2024,
                        one_min=signal_2024_1m,
                    )
                )
        summary = pd.DataFrame(rows)

    sort_cols = ["dataset", "mode", "sum_return", "short_sum_return", "trades"]
    summary = summary.sort_values(sort_cols, ascending=[True, True, False, False, False])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)

    view_cols = [
        "dataset",
        "mode",
        "config",
        "trades",
        "long_trades",
        "short_trades",
        "sum_return",
        "long_sum_return",
        "short_sum_return",
        "avg_return",
        "short_avg_return",
        "win_rate",
        "short_win_rate",
        "decision_rows",
    ]
    print(f"wrote {out}")
    for dataset in summary["dataset"].drop_duplicates():
        for mode in summary.loc[summary["dataset"].eq(dataset), "mode"].drop_duplicates():
            top = summary[(summary["dataset"] == dataset) & (summary["mode"] == mode)].head(12)
            print(f"\n=== {dataset} / {mode} top 12 by total return ===")
            print(top[view_cols].to_string(index=False))


if __name__ == "__main__":
    main()
