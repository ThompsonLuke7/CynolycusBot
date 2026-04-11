from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _resolve_repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except NameError:
        return Path.cwd()


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Data.plots.plots import (  # noqa: E402
    _apply_time_ticks,
    _compute_marker_offset,
    _compute_time_ticks,
    _draw_day_lines,
    _extract_ohlc,
    _plot_candles,
    _select_plot_window,
)
from Data.retrieve_data import normalize_ticker  # noqa: E402
from Models.ga_xgboost.swing_label_weights import (  # noqa: E402
    apply_swing_pivot_zone_weights,
    build_phase3_swing_event_labels,
    keep_first_same_side_event,
)


def _resolve_dataset_dir(ticker: str, dataset_name: str) -> Path:
    slug = normalize_ticker(ticker).lower()
    return REPO_ROOT / "Data" / "processed" / slug / "datasets" / dataset_name


def _load_plot_frame_with_labels(ticker: str, dataset_name: str) -> pd.DataFrame:
    dataset_dir = _resolve_dataset_dir(ticker, dataset_name)
    plot_path = dataset_dir / "plot_frame.parquet"
    y_path = dataset_dir / "y.parquet"
    if not plot_path.exists():
        raise FileNotFoundError(f"Missing plot_frame.parquet at {plot_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing y.parquet at {y_path}")

    plot_df = pd.read_parquet(plot_path)
    y_df = pd.read_parquet(y_path)
    if plot_df.index.equals(y_df.index):
        return plot_df.join(y_df, how="left")

    common = plot_df.index.intersection(y_df.index)
    if not common.empty:
        return plot_df.loc[common].join(y_df.loc[common], how="left")

    min_len = min(len(plot_df), len(y_df))
    df = plot_df.iloc[:min_len].copy()
    labels = y_df.iloc[:min_len].copy()
    labels.index = df.index
    return df.join(labels, how="left")


def _series_to_binary(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        raise KeyError(f"Missing required label column: {col}")
    return pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(dtype=np.int64)


def plot_soft_swing_labels(
    df: pd.DataFrame,
    *,
    positive_window_bars: int,
    ambiguous_window_bars: int,
    neighbor_weight: float,
    ambiguous_weight: float,
    first_in_run_filter: bool,
    tail: int | None,
    save_path: str | Path,
) -> None:
    long_core_full = _series_to_binary(df, "long_swing_label")
    short_core_full = _series_to_binary(df, "short_swing_label")
    suppressed_run_long_full = np.zeros(len(df), dtype=bool)
    suppressed_run_short_full = np.zeros(len(df), dtype=bool)
    if first_in_run_filter:
        long_core_full, short_core_full, suppressed_run_long_full, suppressed_run_short_full = (
            keep_first_same_side_event(long_core_full, short_core_full)
        )
    long_zone_full, long_w_full, _ = apply_swing_pivot_zone_weights(
        long_core_full,
        positive_window_bars=positive_window_bars,
        ambiguous_window_bars=ambiguous_window_bars,
        neighbor_weight=neighbor_weight,
        ambiguous_weight=ambiguous_weight,
    )
    short_zone_full, short_w_full, _ = apply_swing_pivot_zone_weights(
        short_core_full,
        positive_window_bars=positive_window_bars,
        ambiguous_window_bars=ambiguous_window_bars,
        neighbor_weight=neighbor_weight,
        ambiguous_weight=ambiguous_weight,
    )

    work = df.copy()
    work["soft_long_core"] = long_core_full
    work["soft_short_core"] = short_core_full
    work["soft_long_label"] = long_zone_full
    work["soft_short_label"] = short_zone_full
    work["soft_long_weight"] = long_w_full
    work["soft_short_weight"] = short_w_full
    work["suppressed_run_long"] = suppressed_run_long_full
    work["suppressed_run_short"] = suppressed_run_short_full
    work = _select_plot_window(work, window=tail, random_window=False)

    pos, open_y, high_y, low_y, close_y = _extract_ohlc(work)
    marker_offset = _compute_marker_offset(work, high_y, low_y)
    date_index = work.index
    tick_positions, tick_labels = _compute_time_ticks(date_index, pos, max_ticks=20)

    long_core = work["soft_long_core"].to_numpy(dtype=bool)
    short_core = work["soft_short_core"].to_numpy(dtype=bool)
    long_zone = work["soft_long_label"].to_numpy(dtype=bool)
    short_zone = work["soft_short_label"].to_numpy(dtype=bool)
    long_w = work["soft_long_weight"].to_numpy(dtype=np.float32)
    short_w = work["soft_short_weight"].to_numpy(dtype=np.float32)
    long_neighbor = long_zone & ~long_core
    short_neighbor = short_zone & ~short_core
    long_ambiguous = (~long_zone) & (long_w != 1.0)
    short_ambiguous = (~short_zone) & (short_w != 1.0)
    suppressed_run_long = work["suppressed_run_long"].to_numpy(dtype=bool)
    suppressed_run_short = work["suppressed_run_short"].to_numpy(dtype=bool)

    fig, (ax_price, ax_weight) = plt.subplots(
        2,
        1,
        figsize=(18, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.5, 1.2]},
    )
    _plot_candles(ax_price, pos, open_y, high_y, low_y, close_y)

    long_y = low_y - marker_offset * 0.85
    long_neighbor_y = low_y - marker_offset * 0.55
    long_ambig_y = low_y - marker_offset * 0.25
    short_y = high_y + marker_offset * 0.85
    short_neighbor_y = high_y + marker_offset * 0.55
    short_ambig_y = high_y + marker_offset * 0.25

    if long_neighbor.any():
        ax_price.scatter(
            pos[long_neighbor],
            long_neighbor_y[long_neighbor],
            marker="^",
            s=45,
            color="#66BB6A",
            alpha=0.8,
            label=f"LONG soft neighbor label=1 weight={neighbor_weight:g}",
            zorder=3,
        )
    if long_core.any():
        ax_price.scatter(
            pos[long_core],
            long_y[long_core],
            marker="^",
            s=90,
            facecolors="none",
            edgecolors="#1B5E20",
            linewidths=1.8,
            label="LONG core label=1 weight=1",
            zorder=3.2,
        )
    if long_ambiguous.any():
        ax_price.scatter(
            pos[long_ambiguous],
            long_ambig_y[long_ambiguous],
            marker="x",
            s=38,
            color="#9E9E9E",
            alpha=0.7,
            label=f"LONG ambiguous label=0 weight={ambiguous_weight:g}",
            zorder=2.8,
        )
    if suppressed_run_long.any():
        ax_price.scatter(
            pos[suppressed_run_long],
            long_ambig_y[suppressed_run_long],
            marker=".",
            s=34,
            color="#2E7D32",
            alpha=0.45,
            label="LONG suppressed repeat",
            zorder=2.9,
        )

    if short_neighbor.any():
        ax_price.scatter(
            pos[short_neighbor],
            short_neighbor_y[short_neighbor],
            marker="v",
            s=45,
            color="#EF5350",
            alpha=0.8,
            label=f"SHORT soft neighbor label=1 weight={neighbor_weight:g}",
            zorder=3,
        )
    if short_core.any():
        ax_price.scatter(
            pos[short_core],
            short_y[short_core],
            marker="v",
            s=90,
            facecolors="none",
            edgecolors="#B71C1C",
            linewidths=1.8,
            label="SHORT core label=1 weight=1",
            zorder=3.2,
        )
    if short_ambiguous.any():
        ax_price.scatter(
            pos[short_ambiguous],
            short_ambig_y[short_ambiguous],
            marker="x",
            s=38,
            color="#616161",
            alpha=0.7,
            label=f"SHORT ambiguous label=0 weight={ambiguous_weight:g}",
            zorder=2.8,
        )
    if suppressed_run_short.any():
        ax_price.scatter(
            pos[suppressed_run_short],
            short_ambig_y[suppressed_run_short],
            marker=".",
            s=34,
            color="#C62828",
            alpha=0.45,
            label="SHORT suppressed repeat",
            zorder=2.9,
        )

    ax_weight.axhline(0.0, color="#333333", linewidth=1)
    ax_weight.step(pos, long_zone.astype(float) * long_w, where="mid", color="#2E7D32", label="LONG train label * weight")
    ax_weight.step(pos, -short_zone.astype(float) * short_w, where="mid", color="#C62828", label="-SHORT train label * weight")
    if long_ambiguous.any():
        ax_weight.scatter(pos[long_ambiguous], np.zeros(long_ambiguous.sum()), marker="x", s=26, color="#9E9E9E", alpha=0.65, label="LONG zero/low-weight ambiguous")
    if short_ambiguous.any():
        ax_weight.scatter(pos[short_ambiguous], np.zeros(short_ambiguous.sum()), marker="x", s=26, color="#616161", alpha=0.65, label="SHORT zero/low-weight ambiguous")

    _draw_day_lines((ax_price, ax_weight), tick_positions, line_color="#d0d0d0")
    _apply_time_ticks(ax_weight, tick_positions, tick_labels)

    ax_price.set_title(
        "Soft swing pivot-zone labels "
        f"(positive +/-{positive_window_bars}, ambiguous +/-{ambiguous_window_bars}, "
        f"first_in_run={first_in_run_filter})"
    )
    ax_price.set_ylabel("Price")
    ax_weight.set_ylabel("label * weight")
    ax_weight.set_xlabel("Date")
    ax_weight.set_ylim(
        min(-1.1, float(np.nanmin(-short_zone.astype(float) * short_w)) - 0.1),
        max(1.1, float(np.nanmax(long_zone.astype(float) * long_w)) + 0.1),
    )
    ax_price.grid(True, alpha=0.25)
    ax_weight.grid(True, alpha=0.25)
    ax_price.legend(loc="best", fontsize=8, ncols=2)
    ax_weight.legend(loc="best", fontsize=8, ncols=2)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    print(f"Saved plot to {save_path}")
    plt.close(fig)

    print(
        "[soft_swing] "
        f"long_core={int(long_core_full.sum())}, "
        f"long_zone={int(long_zone_full.sum())}, "
        f"long_ambiguous={(long_zone_full == 0).astype(bool).dot((long_w_full != 1.0).astype(int))}, "
        f"suppressed_run_long={int(suppressed_run_long_full.sum())}, "
        f"short_core={int(short_core_full.sum())}, "
        f"short_zone={int(short_zone_full.sum())}, "
        f"short_ambiguous={(short_zone_full == 0).astype(bool).dot((short_w_full != 1.0).astype(int))}, "
        f"suppressed_run_short={int(suppressed_run_short_full.sum())}"
    )


def plot_phase3_swing_labels(
    df: pd.DataFrame,
    *,
    positive_window_bars: int,
    ambiguous_window_bars: int,
    neighbor_weight: float,
    ambiguous_weight: float,
    search_pre_bars: int,
    search_post_bars: int,
    horizon_bars: int,
    tp_atr: float,
    sl_atr: float,
    entry_price_mode: str,
    same_leg_non_event_weight: float,
    macro_filter: bool,
    macro_min_leg_atr: float,
    macro_min_leg_bars: int,
    first_in_run_filter: bool,
    tail: int | None,
    save_path: str | Path,
    metadata_path: str | Path | None = None,
) -> None:
    long_label_full, short_label_full, long_w_full, short_w_full, meta, legs, pivots = (
        build_phase3_swing_event_labels(
            df,
            positive_window_bars=positive_window_bars,
            ambiguous_window_bars=ambiguous_window_bars,
            neighbor_weight=neighbor_weight,
            ambiguous_weight=ambiguous_weight,
            search_pre_bars=search_pre_bars,
            search_post_bars=search_post_bars,
            horizon_bars=horizon_bars,
            tp_atr=tp_atr,
            sl_atr=sl_atr,
            entry_price_mode=entry_price_mode,
            same_leg_non_event_weight=same_leg_non_event_weight,
            use_macro_filter=macro_filter,
            macro_min_leg_atr=macro_min_leg_atr,
            macro_min_leg_bars=macro_min_leg_bars,
            first_in_run_filter=first_in_run_filter,
        )
    )

    out_meta = meta.copy()
    out_meta["phase3_long_label"] = long_label_full
    out_meta["phase3_short_label"] = short_label_full
    out_meta["phase3_long_weight"] = long_w_full
    out_meta["phase3_short_weight"] = short_w_full
    out_meta["pivot_up"] = _series_to_binary(df, "pivot_up")
    out_meta["pivot_down"] = _series_to_binary(df, "pivot_down")
    out_meta["raw_long_swing_label"] = _series_to_binary(df, "long_swing_label")
    out_meta["raw_short_swing_label"] = _series_to_binary(df, "short_swing_label")
    if metadata_path is not None:
        metadata_path = Path(metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        out_meta.to_parquet(metadata_path)
        out_meta.to_csv(metadata_path.with_suffix(".csv"))
        print(f"Saved Phase 3 metadata to {metadata_path}")

    work = df.copy()
    for col in out_meta.columns:
        work[col] = out_meta[col]
    work = _select_plot_window(work, window=tail, random_window=False)

    pos, open_y, high_y, low_y, close_y = _extract_ohlc(work)
    marker_offset = _compute_marker_offset(work, high_y, low_y)
    tick_positions, tick_labels = _compute_time_ticks(work.index, pos, max_ticks=20)
    pos_lookup = {idx: pos_i for pos_i, idx in enumerate(work.index)}

    long_label = work["phase3_long_label"].to_numpy(dtype=bool)
    short_label = work["phase3_short_label"].to_numpy(dtype=bool)
    long_w = work["phase3_long_weight"].to_numpy(dtype=np.float32)
    short_w = work["phase3_short_weight"].to_numpy(dtype=np.float32)
    selected_long = work["is_selected_long_event"].to_numpy(dtype=bool)
    selected_short = work["is_selected_short_event"].to_numpy(dtype=bool)
    long_neighbor = long_label & (work["long_event_source"].to_numpy() == "neighbor")
    short_neighbor = short_label & (work["short_event_source"].to_numpy() == "neighbor")
    long_ambiguous = work["long_event_source"].to_numpy() == "ambiguous"
    short_ambiguous = work["short_event_source"].to_numpy() == "ambiguous"
    long_suppressed = work["long_event_source"].to_numpy() == "suppressed_leg"
    short_suppressed = work["short_event_source"].to_numpy() == "suppressed_leg"
    raw_long_removed = (work["raw_long_swing_label"].to_numpy(dtype=bool)) & ~long_label
    raw_short_removed = (work["raw_short_swing_label"].to_numpy(dtype=bool)) & ~short_label
    pivot_up = work["pivot_up"].to_numpy(dtype=bool)
    pivot_down = work["pivot_down"].to_numpy(dtype=bool)

    fig, (ax_price, ax_weight) = plt.subplots(
        2,
        1,
        figsize=(18, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3.7, 1.2]},
    )
    _plot_candles(ax_price, pos, open_y, high_y, low_y, close_y)

    for leg in legs:
        if leg.end_idx < 0 or leg.end_idx >= len(df):
            continue
        ts = df.index[leg.end_idx]
        if ts not in pos_lookup:
            continue
        color = "#2E7D32" if leg.direction == "down" else "#C62828"
        ax_price.axvline(pos_lookup[ts], color=color, linestyle="--", linewidth=0.9, alpha=0.24)
        ax_weight.axvline(pos_lookup[ts], color=color, linestyle="--", linewidth=0.9, alpha=0.24)

    long_y = low_y - marker_offset * 0.95
    short_y = high_y + marker_offset * 0.95
    long_neighbor_y = low_y - marker_offset * 0.60
    short_neighbor_y = high_y + marker_offset * 0.60
    long_ambig_y = low_y - marker_offset * 0.30
    short_ambig_y = high_y + marker_offset * 0.30

    if pivot_down.any():
        ax_price.scatter(pos[pivot_down], low_y[pivot_down] - marker_offset * 1.30, marker="o", s=30, facecolors="none", edgecolors="#00695C", alpha=0.65, label="zigzag low pivot", zorder=2.8)
    if pivot_up.any():
        ax_price.scatter(pos[pivot_up], high_y[pivot_up] + marker_offset * 1.30, marker="o", s=30, facecolors="none", edgecolors="#6A1B9A", alpha=0.65, label="zigzag high pivot", zorder=2.8)
    if raw_long_removed.any():
        ax_price.scatter(pos[raw_long_removed], low_y[raw_long_removed] - marker_offset * 1.70, marker="^", s=42, color="#A5D6A7", alpha=0.45, label="removed raw LONG positive", zorder=2.6)
    if raw_short_removed.any():
        ax_price.scatter(pos[raw_short_removed], high_y[raw_short_removed] + marker_offset * 1.70, marker="v", s=42, color="#EF9A9A", alpha=0.45, label="removed raw SHORT positive", zorder=2.6)
    if long_neighbor.any():
        ax_price.scatter(pos[long_neighbor], long_neighbor_y[long_neighbor], marker="^", s=45, color="#66BB6A", alpha=0.75, label=f"LONG neighbor weight={neighbor_weight:g}", zorder=3)
    if short_neighbor.any():
        ax_price.scatter(pos[short_neighbor], short_neighbor_y[short_neighbor], marker="v", s=45, color="#EF5350", alpha=0.75, label=f"SHORT neighbor weight={neighbor_weight:g}", zorder=3)
    if selected_long.any():
        ax_price.scatter(pos[selected_long], long_y[selected_long], marker="^", s=115, facecolors="none", edgecolors="#1B5E20", linewidths=2.0, label="selected LONG event", zorder=3.4)
    if selected_short.any():
        ax_price.scatter(pos[selected_short], short_y[selected_short], marker="v", s=115, facecolors="none", edgecolors="#B71C1C", linewidths=2.0, label="selected SHORT event", zorder=3.4)
    if long_ambiguous.any():
        ax_price.scatter(pos[long_ambiguous], long_ambig_y[long_ambiguous], marker="x", s=34, color="#9E9E9E", alpha=0.65, label=f"LONG ambiguous weight={ambiguous_weight:g}", zorder=2.7)
    if short_ambiguous.any():
        ax_price.scatter(pos[short_ambiguous], short_ambig_y[short_ambiguous], marker="x", s=34, color="#616161", alpha=0.65, label=f"SHORT ambiguous weight={ambiguous_weight:g}", zorder=2.7)
    if long_suppressed.any():
        ax_price.scatter(pos[long_suppressed], low_y[long_suppressed] - marker_offset * 0.12, marker=".", s=18, color="#81C784", alpha=0.35, label=f"LONG same-leg suppressed weight={same_leg_non_event_weight:g}", zorder=2.5)
    if short_suppressed.any():
        ax_price.scatter(pos[short_suppressed], high_y[short_suppressed] + marker_offset * 0.12, marker=".", s=18, color="#E57373", alpha=0.35, label=f"SHORT same-leg suppressed weight={same_leg_non_event_weight:g}", zorder=2.5)

    ax_weight.axhline(0.0, color="#333333", linewidth=1)
    ax_weight.step(pos, long_label.astype(float) * long_w, where="mid", color="#2E7D32", label="LONG phase3 label * weight")
    ax_weight.step(pos, -short_label.astype(float) * short_w, where="mid", color="#C62828", label="-SHORT phase3 label * weight")
    if long_ambiguous.any():
        ax_weight.scatter(pos[long_ambiguous], np.zeros(long_ambiguous.sum()), marker="x", s=24, color="#9E9E9E", alpha=0.65)
    if short_ambiguous.any():
        ax_weight.scatter(pos[short_ambiguous], np.zeros(short_ambiguous.sum()), marker="x", s=24, color="#616161", alpha=0.65)

    _draw_day_lines((ax_price, ax_weight), tick_positions, line_color="#d0d0d0")
    _apply_time_ticks(ax_weight, tick_positions, tick_labels)
    ax_price.set_title(
        "Phase 3 one-event-per-leg swing labels "
        f"(macro={macro_filter}, min={macro_min_leg_atr:g} ATR/{macro_min_leg_bars} bars, "
        f"search -{search_pre_bars}/+{search_post_bars}, H={horizon_bars}, "
        f"first_in_run={first_in_run_filter})"
    )
    ax_price.set_ylabel("Price")
    ax_weight.set_ylabel("label * weight")
    ax_weight.set_xlabel("Date")
    ax_weight.set_ylim(-1.1, 1.1)
    ax_price.grid(True, alpha=0.25)
    ax_weight.grid(True, alpha=0.25)
    ax_price.legend(loc="best", fontsize=7.5, ncols=3)
    ax_weight.legend(loc="best", fontsize=8, ncols=2)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    print(f"Saved plot to {save_path}")
    plt.close(fig)
    print(
        "[phase3_swing] "
        f"legs={len(legs)}, pivots={len(pivots)}, "
        f"macro_filter={macro_filter}, "
        f"macro_min_leg_atr={macro_min_leg_atr:g}, "
        f"macro_min_leg_bars={macro_min_leg_bars}, "
        f"first_in_run_filter={first_in_run_filter}, "
        f"selected_long={int(meta['is_selected_long_event'].sum())}, "
        f"selected_short={int(meta['is_selected_short_event'].sum())}, "
        f"suppressed_run_long={int(meta['is_suppressed_long_run_event'].sum())}, "
        f"suppressed_run_short={int(meta['is_suppressed_short_run_event'].sum())}, "
        f"long_zone={int(long_label_full.sum())}, "
        f"short_zone={int(short_label_full.sum())}, "
        f"long_ambiguous={int((meta['long_event_source'] == 'ambiguous').sum())}, "
        f"short_ambiguous={int((meta['short_event_source'] == 'ambiguous').sum())}, "
        f"long_suppressed={int((meta['long_event_source'] == 'suppressed_leg').sum())}, "
        f"short_suppressed={int((meta['short_event_source'] == 'suppressed_leg').sum())}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot soft swing pivot-zone labels.")
    parser.add_argument("--ticker", type=str, default="SPY")
    parser.add_argument("--dataset", "--dataset-name", dest="dataset", type=str, default="10min")
    parser.add_argument("--tail", type=int, default=300)
    parser.add_argument("--positive-window-bars", type=int, default=1)
    parser.add_argument("--ambiguous-window-bars", type=int, default=3)
    parser.add_argument("--neighbor-weight", type=float, default=0.75)
    parser.add_argument("--ambiguous-weight", type=float, default=0.0)
    parser.add_argument("--first-in-run-filter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--phase3-events", action="store_true")
    parser.add_argument("--phase3-search-pre-bars", type=int, default=6)
    parser.add_argument("--phase3-search-post-bars", type=int, default=3)
    parser.add_argument("--phase3-horizon-bars", type=int, default=12)
    parser.add_argument("--phase3-tp-atr", type=float, default=1.0)
    parser.add_argument("--phase3-sl-atr", type=float, default=0.8)
    parser.add_argument("--phase3-entry-price", type=str, choices=["close", "next_open"], default="close")
    parser.add_argument("--phase3-same-leg-non-event-weight", type=float, default=0.0)
    parser.add_argument("--phase3-macro-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--phase3-macro-min-leg-atr", type=float, default=2.0)
    parser.add_argument("--phase3-macro-min-leg-bars", type=int, default=6)
    parser.add_argument("--metadata-path", "--metadata_path", dest="metadata_path", type=str, default=None)
    parser.add_argument("--save-path", "--save_path", dest="save_path", type=str, default=None)
    args = parser.parse_args()

    df = _load_plot_frame_with_labels(args.ticker, args.dataset)
    slug = normalize_ticker(args.ticker).lower()
    save_path = args.save_path
    if save_path is None:
        name = f"phase3_soft_swing_labels_{args.dataset}.png" if args.phase3_events else f"soft_swing_labels_{args.dataset}.png"
        save_path = (
            REPO_ROOT
            / "Data"
            / "processed"
            / slug
            / "plots"
            / name
        )
    metadata_path = args.metadata_path
    if metadata_path is None and args.phase3_events:
        metadata_path = (
            REPO_ROOT
            / "Data"
            / "processed"
            / slug
            / "datasets"
            / args.dataset
            / "phase3_swing_event_metadata.parquet"
        )
    if args.phase3_events:
        plot_phase3_swing_labels(
            df,
            positive_window_bars=args.positive_window_bars,
            ambiguous_window_bars=args.ambiguous_window_bars,
            neighbor_weight=args.neighbor_weight,
            ambiguous_weight=args.ambiguous_weight,
            search_pre_bars=args.phase3_search_pre_bars,
            search_post_bars=args.phase3_search_post_bars,
            horizon_bars=args.phase3_horizon_bars,
            tp_atr=args.phase3_tp_atr,
            sl_atr=args.phase3_sl_atr,
            entry_price_mode=args.phase3_entry_price,
            same_leg_non_event_weight=args.phase3_same_leg_non_event_weight,
            macro_filter=bool(args.phase3_macro_filter),
            macro_min_leg_atr=args.phase3_macro_min_leg_atr,
            macro_min_leg_bars=args.phase3_macro_min_leg_bars,
            first_in_run_filter=bool(args.first_in_run_filter),
            tail=args.tail,
            save_path=save_path,
            metadata_path=metadata_path,
        )
    else:
        plot_soft_swing_labels(
            df,
            positive_window_bars=args.positive_window_bars,
            ambiguous_window_bars=args.ambiguous_window_bars,
            neighbor_weight=args.neighbor_weight,
            ambiguous_weight=args.ambiguous_weight,
            first_in_run_filter=bool(args.first_in_run_filter),
            tail=args.tail,
            save_path=save_path,
        )


if __name__ == "__main__":
    main()
