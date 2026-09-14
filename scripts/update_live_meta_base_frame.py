from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.API.Alpaca_API.inference.live_inference import (
    LiveGAXGBPredictor,
    build_meta_feature_frame_from_1m,
    continue_origin_relative_columns,
    recompute_prob_derivatives,
)
from core.API.Alpaca_API.runners.live_runner import _load_prefill_frame, _load_precomputed_meta_frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extend cached 10m meta feature matrix from local 1m history.")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol.")
    parser.add_argument(
        "--prefill-path",
        default="Data/raw/spy/spy_intraday_1min_runtime_rth_cache.parquet",
        help=(
            "Local 1m history file (.csv or .parquet). Defaults to the runtime RTH "
            "cache the live loop itself prefills from and keeps current; "
            "spy_intraday_1min.parquet holds only the latest session."
        ),
    )
    parser.add_argument(
        "--meta-base-frame-path",
        default="Data/inference/spy/10min/debug_matrices_warmup/spy/live_meta_matrix_on_trace_ts.parquet",
        help="Existing cached 10m meta matrix (.csv or .parquet).",
    )
    parser.add_argument(
        "--out-path",
        default=None,
        help="Optional output path. Defaults to overwriting --meta-base-frame-path.",
    )
    parser.add_argument("--tz", default="America/New_York", help="Timezone used by the meta frame.")
    parser.add_argument("--assume-tz", default="UTC", help="Timezone assumed for naive raw timestamps.")
    parser.add_argument("--lookback-days", type=int, default=20, help="1m warm-up window to recompute when appending.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Rebuild the whole matrix from full 1m history instead of appending. "
            "Appending many rows at once drifts: rows computed in one window "
            "become each other's predecessors, so a 79-row catch-up moved "
            "trend_phase_label by 3 and p_long_trend by 1.35 against a "
            "full-history build (2026-09-12). Appending ONE row, whose "
            "predecessors come from cache, matches to 4e-05."
        ),
    )
    parser.add_argument("--ga-model-root", default="Data/models/ga_xgboost/10min", help="GA-XGB model root.")
    parser.add_argument("--ga-feature-list", default=None, help="Optional GA-XGB feature list txt path.")
    parser.add_argument("--ga-dataset-name", default="10min", help="Dataset name for GA feature list fallback.")
    parser.add_argument("--ga-pivot-label-dir", default="swing", help="Pivot GA label dir.")
    parser.add_argument("--ga-tb-label-dir", default="tb", help="TB GA label dir.")
    return parser.parse_args()


def _resolve_ga_feature_list(symbol: str, dataset_name: str, provided: str | None) -> str | None:
    if provided:
        return str(provided)
    try:
        from Data.load_data import get_ticker_processed_base_dir
        from Data.retrieve_data import normalize_ticker

        ticker = normalize_ticker(symbol)
        candidate = (
            get_ticker_processed_base_dir(ticker)
            / "datasets"
            / dataset_name
            / f"features_X_{dataset_name}_tree.txt"
        )
        if candidate.exists():
            return str(candidate)
    except Exception:
        return None
    return None


def _write_frame(frame: pd.DataFrame, out_path: Path) -> None:
    """Write without `attrs`.

    `build_meta_feature_frame_from_1m` stashes a DataFrame in `frame.attrs`
    (`ga_prob_sources`), and pandas JSON-encodes `attrs` into parquet metadata:
    `TypeError: Object of type DataFrame is not JSON serializable`. Nothing reads
    those attrs back off the cached matrix.
    """
    out = frame.copy()
    out.attrs = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        out.reset_index().rename(columns={"index": "timestamp"}).to_csv(out_path, index=False)
    else:
        out.to_parquet(out_path)


def main() -> None:
    args = _parse_args()
    raw_path = Path(args.prefill_path)
    base_path = Path(args.meta_base_frame_path)
    out_path = Path(args.out_path) if args.out_path else base_path

    raw_df = _load_prefill_frame(raw_path)
    if "symbol" in raw_df.columns:
        raw_df = raw_df.sort_values(["symbol", "timestamp"]).copy()
    else:
        raw_df = raw_df.sort_values("timestamp").copy()
    raw_df = raw_df.set_index("timestamp")

    ga_feature_list = _resolve_ga_feature_list(args.symbol, args.ga_dataset_name, args.ga_feature_list)
    ga_predictor = None
    if ga_feature_list:
        ga_predictor = LiveGAXGBPredictor(
            model_root=args.ga_model_root,
            feature_list_path=ga_feature_list,
            include_pivot_probs=True,
            include_tb_probs=True,
            pivot_label_dir=args.ga_pivot_label_dir,
            tb_label_dir=args.ga_tb_label_dir,
        )

    if args.rebuild or not base_path.exists():
        # Bootstrap. Without this the scheme cannot recover from a missing
        # artifact: the file is git-ignored under Data/**, so when it went
        # missing every live bar silently fell back to rebuilding the whole
        # matrix from the 1m buffer (2026-09-10: ~55s per 10-minute bar).
        why = "--rebuild requested" if args.rebuild else f"no cached matrix at {base_path}"
        print(f"[meta-cache] {why}; building from full history.")
        full = build_meta_feature_frame_from_1m(
            raw_df, rule="10min", label="left", closed="left", tz=args.tz,
            assume_tz=args.assume_tz, ga_predictor=ga_predictor,
            include_pivot_probs=True, include_tb_probs=True,
        )
        _write_frame(full, out_path)
        print(f"[meta-cache] Bootstrapped {len(full):,} rows to {out_path} "
              f"({full.index.min()} .. {full.index.max()}).")
        return

    cached = _load_precomputed_meta_frame(base_path, tz=args.tz)
    cached_max = cached.index.max()

    overlap_start = cached_max - pd.Timedelta(days=max(1, int(args.lookback_days)))
    raw_tail = raw_df.loc[raw_df.index >= overlap_start].copy()
    if raw_tail.empty:
        raise SystemExit("No raw 1m rows available in the requested overlap window.")

    computed_tail = build_meta_feature_frame_from_1m(
        raw_tail,
        rule="10min",
        label="left",
        closed="left",
        tz=args.tz,
        assume_tz=args.assume_tz,
        ga_predictor=ga_predictor,
        include_pivot_probs=True,
        include_tb_probs=True,
    )
    # Append only what is new. The old merge replaced the whole overlap window
    # with rows computed from just that window, so every run overwrote up to
    # `lookback_days` of full-history values with short-history ones. The window
    # is warm-up for the new rows, not a rewrite of the old ones.
    fresh = computed_tail.loc[computed_tail.index > cached_max]
    if not fresh.empty:
        new_cols = [c for c in fresh.columns if c not in cached.columns]
        missing_cols = [c for c in cached.columns if c not in fresh.columns]
        if new_cols or missing_cols:
            print(f"[meta-cache] WARNING column drift: {len(new_cols)} new, "
                  f"{len(missing_cols)} missing (first: {(new_cols + missing_cols)[:5]}).")
        fresh = continue_origin_relative_columns(cached, fresh)
    merged = pd.concat([cached, fresh], axis=0).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    # Same reason as the live append path: the lag/rolling/delta family must read
    # the cached full-history predecessors, not the window's recomputed ones.
    merged = recompute_prob_derivatives(merged)

    _write_frame(merged, out_path)

    print(
        f"[meta-cache] Wrote {len(merged):,} rows to {out_path} "
        f"(cached_max={cached_max}, merged_max={merged.index.max()}, "
        f"appended={len(fresh):,}, warmup_days={args.lookback_days})."
    )


if __name__ == "__main__":
    main()
