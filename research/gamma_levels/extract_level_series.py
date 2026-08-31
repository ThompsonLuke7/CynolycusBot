"""Collapse the one-minute gamma ladder archive into a per-snapshot level series.

One row per symbol per snapshot: spot, the estimated levels, and the structure
scalars a touch study needs to condition on.

Levels come from ``levels._core_levels_from_ladder`` -- the exact function the
live module and the nightly capture use -- rather than a reimplementation, so
the research answer describes the levels the system actually trades.

Output: ``research/gamma_levels/data/level_series_<SYMBOL>.parquet``
"""

from __future__ import annotations

import argparse
import logging
import re
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from strategies.dealer_positioning.levels import _core_levels_from_ladder

REPO = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = REPO / "Data" / "dealer_positioning"
OUT_ROOT = Path(__file__).resolve().parent / "data"

DEFAULT_SYMBOLS = ("SPY", "QQQ", "IWM", "GLD", "SLV")
MAGNET_QUANTILE = 0.90

_TS = re.compile(r"gamma_ladder_(\d{8}T\d{6})")

USE_COLS = [
    "strike", "spot", "call_oi", "put_oi", "call_volume", "put_volume",
    "call_gex", "put_gex", "net_gex", "abs_net_gex", "total_abs_gex",
    "total_vex", "call_iv", "put_iv",
]

logger = logging.getLogger(__name__)


def _stamp(path: Path) -> datetime | None:
    m = _TS.search(path.name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _one(path_str: str) -> dict | None:
    path = Path(path_str)
    stamp = _stamp(path)
    if stamp is None:
        return None
    try:
        ladder = pd.read_csv(path)
    except Exception:  # noqa: BLE001 - a corrupt snapshot is skipped, not fatal
        return None
    if ladder.empty or "strike" not in ladder.columns:
        return None
    for col in USE_COLS:
        if col not in ladder.columns:
            ladder[col] = float("nan")
        ladder[col] = pd.to_numeric(ladder[col], errors="coerce")
    ladder = ladder.dropna(subset=["strike"])
    if ladder.empty or ladder["spot"].dropna().empty:
        return None

    spot = float(ladder["spot"].dropna().iloc[0])
    if spot <= 0:
        return None
    ladder = ladder.fillna({"net_gex": 0.0, "abs_net_gex": 0.0, "total_abs_gex": 0.0,
                            "call_gex": 0.0, "put_gex": 0.0, "total_vex": 0.0})

    core = _core_levels_from_ladder(ladder, spot, MAGNET_QUANTILE)

    total_abs = float(ladder["total_abs_gex"].abs().sum())
    net = float(ladder["net_gex"].sum())

    def _share_at(level):
        if level is None or total_abs <= 0:
            return None
        row = ladder[ladder["strike"] == level]
        if row.empty:
            return None
        return float(row["total_abs_gex"].abs().sum() / total_abs)

    atm_idx = (ladder["strike"] - spot).abs().idxmin()
    ivs = [ladder.loc[atm_idx, "call_iv"], ladder.loc[atm_idx, "put_iv"]]
    ivs = [float(v) for v in ivs if pd.notna(v) and float(v) > 0]

    return {
        "captured_at": stamp,
        "symbol": path.parent.parent.name.upper(),
        "spot": spot,
        "call_wall": core["call_wall"],
        "put_wall": core["put_wall"],
        "nearest_magnet": core["nearest_magnet"],
        "gamma_flip": core["gamma_flip"],
        "estimated_net_gex": net,
        "total_abs_gamma": total_abs,
        "dealer_imbalance": (net / total_abs) if total_abs else None,
        "call_wall_share": _share_at(core["call_wall"]),
        "put_wall_share": _share_at(core["put_wall"]),
        "magnet_share": _share_at(core["nearest_magnet"]),
        "atm_iv": (sum(ivs) / len(ivs)) if ivs else None,
        "strikes": int(len(ladder)),
        "total_option_oi": float(ladder[["call_oi", "put_oi"]].sum().sum()),
    }


def extract(symbol: str, *, archive_root: Path, workers: int) -> pd.DataFrame:
    directory = Path(archive_root) / symbol.upper() / "snapshots"
    if not directory.exists():
        logger.warning("no archive for %s", symbol)
        return pd.DataFrame()
    paths = [str(p) for p in sorted(directory.glob("gamma_ladder_*.csv"))]
    logger.info("%s: %d snapshots", symbol, len(paths))
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for out in pool.map(_one, paths, chunksize=64):
            if out:
                records.append(out)
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame.from_records(records).sort_values("captured_at").reset_index(drop=True)
    frame["session"] = pd.to_datetime(frame["captured_at"]).dt.date
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE_ROOT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.out_root.mkdir(parents=True, exist_ok=True)
    for symbol in args.symbols:
        frame = extract(symbol.upper(), archive_root=args.archive_root, workers=args.workers)
        if frame.empty:
            continue
        path = args.out_root / f"level_series_{symbol.upper()}.parquet"
        frame.to_parquet(path, index=False)
        print(f"{symbol.upper()}: {len(frame)} rows, {frame['session'].nunique()} sessions -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
