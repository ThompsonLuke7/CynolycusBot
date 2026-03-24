from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from API.Alpaca_API.inference.live_inference import build_meta_feature_frame_from_1m
from API.Alpaca_API.runners.live_runner import _load_prefill_frame, _load_precomputed_meta_frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extend cached 10m meta feature matrix from local 1m history.")
    parser.add_argument(
        "--prefill-path",
        default="Data/raw/spy/spy_intraday_1min.parquet",
        help="Local 1m history file (.csv or .parquet).",
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
    parser.add_argument("--lookback-days", type=int, default=120, help="1m overlap window to recompute and append.")
    return parser.parse_args()


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
    )
    merged = pd.concat(
        [cached.loc[cached.index < computed_tail.index.min()], computed_tail],
        axis=0,
    ).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        merged.reset_index().rename(columns={"index": "timestamp"}).to_csv(out_path, index=False)
    else:
        merged.to_parquet(out_path)

    print(
        f"[meta-cache] Wrote {len(merged):,} rows to {out_path} "
        f"(cached_max={cached_max}, merged_max={merged.index.max()}, lookback_days={args.lookback_days})."
    )


if __name__ == "__main__":
    main()
