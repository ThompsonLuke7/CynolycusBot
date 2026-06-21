"""Build a shared ticker universe across theme, momentum, and swing modules."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "Data" / "shared" / "universe"
SHARED_UNIVERSE_CSV = DATA_DIR / "shared_universe.csv"
MANUAL_AI_WATCHLIST = DATA_DIR / "manual_ai_watchlist.csv"
# Dynamic discovery queue (see core.shared_universe.discovery). Tickers whose
# status has been promoted by the nightly screen are folded in as eligible.
PENDING_TICKERS_CSV = DATA_DIR / "pending_tickers.csv"
PENDING_NOVEL_TICKERS_CSV = DATA_DIR / "pending_tickers_novel.csv"

THEME_MAP_V4 = REPO_ROOT / "themes" / "theme_expansion_legacy" / "data" / "theme_map_v4.csv"
THEME_ELIGIBLE = REPO_ROOT / "themes" / "theme_expansion_legacy" / "outputs" / "universe_filter.csv"
SWING_TRAIN = REPO_ROOT / "strategies" / "multi_ticker_swing" / "config" / "swing_trader_universe_v3.csv"
SWING_LIVE_JSON = REPO_ROOT / "strategies" / "multi_ticker_swing" / "config" / "trading_universe.json"


def clean_ticker(value: object) -> str:
    return str(value).strip().upper().replace("$", "")


def _read_ticker_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["ticker"])
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["ticker"])
    if "ticker" not in df.columns:
        df = df.rename(columns={df.columns[0]: "ticker"})
    df["ticker"] = df["ticker"].map(clean_ticker)
    df = df[df["ticker"] != ""].drop_duplicates("ticker").copy()
    return df


def _source_frame(tickers: Iterable[str], source_col: str) -> pd.DataFrame:
    return pd.DataFrame({"ticker": sorted({clean_ticker(t) for t in tickers if clean_ticker(t)})}).assign(
        **{source_col: True}
    )


def _load_theme_map() -> pd.DataFrame:
    df = _read_ticker_csv(THEME_MAP_V4)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "asset_type", "theme_1", "theme_2", "theme_3", "in_theme_v4"])
    cols = [c for c in ["ticker", "asset_type", "theme_1", "theme_2", "theme_3"] if c in df.columns]
    out = df[cols].copy()
    out["in_theme_v4"] = True
    return out


def _load_theme_eligibility() -> pd.DataFrame:
    df = _read_ticker_csv(THEME_ELIGIBLE)
    if df.empty or "is_eligible" not in df.columns:
        return pd.DataFrame(columns=["ticker", "is_eligible"])
    keep = ["ticker", "is_eligible"]
    for col in ["px", "avg_dollar_volume_20d", "market_cap"]:
        if col in df.columns:
            keep.append(col)
    out = df[keep].copy()
    out["is_eligible"] = out["is_eligible"].astype(bool)
    return out


def _load_manual_ai_watchlist() -> pd.DataFrame:
    df = _read_ticker_csv(MANUAL_AI_WATCHLIST)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "asset_type", "theme_1", "theme_2", "theme_3", "notes", "in_manual_ai_watchlist", "manual_eligible"])
    keep = [c for c in ["ticker", "asset_type", "theme_1", "theme_2", "theme_3", "notes"] if c in df.columns]
    out = df[keep].copy()
    out["in_manual_ai_watchlist"] = True
    out["manual_eligible"] = True
    return out


def _load_momentum_candidates() -> pd.DataFrame:
    try:
        from strategies.momentum_expansion.data.universe import get_candidate_pool, load_candidate_metadata

        meta = load_candidate_metadata()
        pool = set(get_candidate_pool())
    except Exception:
        meta = pd.DataFrame(columns=["ticker"])
        pool = set()
    if meta.empty:
        return _source_frame(pool, "in_momentum_candidate")
    meta = meta.copy()
    meta["ticker"] = meta["ticker"].map(clean_ticker)
    meta = meta[meta["ticker"] != ""].drop_duplicates("ticker")
    meta["in_momentum_candidate"] = meta["ticker"].isin(pool)
    keep = [c for c in ["ticker", "sector", "market_cap_bucket", "type", "notes", "in_momentum_candidate"] if c in meta.columns]
    return meta[keep]


def _load_swing_train() -> pd.DataFrame:
    df = _read_ticker_csv(SWING_TRAIN)
    if df.empty:
        return _source_frame([], "in_swing_train")
    out = df.copy()
    out["in_swing_train"] = True
    return out


def _load_discovered_promoted() -> pd.DataFrame:
    """Promoted rows from the dynamic discovery queue become eligible names."""
    empty = pd.DataFrame(columns=["ticker", "in_discovered", "discovered_eligible", "cap_tier", "stock_only"])
    if not PENDING_TICKERS_CSV.exists():
        return empty
    df = pd.read_csv(PENDING_TICKERS_CSV)
    if df.empty or "status" not in df.columns:
        return empty
    promoted = df[df["status"].astype(str) == "promoted"].copy()
    if promoted.empty:
        return empty
    promoted["ticker"] = promoted["ticker"].map(clean_ticker)
    promoted = promoted[promoted["ticker"] != ""].drop_duplicates("ticker")
    out = promoted[["ticker"]].copy()
    out["in_discovered"] = True
    out["discovered_eligible"] = True
    out["cap_tier"] = promoted["cap_tier"].values if "cap_tier" in promoted.columns else ""
    out["stock_only"] = promoted["stock_only"].values if "stock_only" in promoted.columns else False
    return out


def _load_swing_live() -> pd.DataFrame:
    if not SWING_LIVE_JSON.exists():
        return _source_frame([], "in_swing_live")
    data = json.loads(SWING_LIVE_JSON.read_text())
    rows = []
    for ticker, cfg in data.items():
        row = {"ticker": clean_ticker(ticker), "in_swing_live": True}
        if isinstance(cfg, dict):
            for col in ["tier", "entry_threshold", "profit_factor", "sharpe"]:
                if col in cfg:
                    row[f"swing_live_{col}"] = cfg[col]
        rows.append(row)
    return pd.DataFrame(rows).drop_duplicates("ticker") if rows else _source_frame([], "in_swing_live")


def build_shared_universe(*, out_path: Path = SHARED_UNIVERSE_CSV) -> pd.DataFrame:
    frames = {
        "theme": _load_theme_map(),
        "eligible": _load_theme_eligibility(),
        "manual_ai": _load_manual_ai_watchlist(),
        "momentum": _load_momentum_candidates(),
        "swing_train": _load_swing_train(),
        "swing_live": _load_swing_live(),
        "discovered": _load_discovered_promoted(),
    }
    tickers = sorted(set().union(*(set(df["ticker"]) for df in frames.values() if "ticker" in df.columns)))
    out = pd.DataFrame({"ticker": tickers})
    for df in frames.values():
        if not df.empty:
            out = out.merge(df, on="ticker", how="left", suffixes=("", "_dup"))
            dup_cols = [c for c in out.columns if c.endswith("_dup")]
            if dup_cols:
                out = out.drop(columns=dup_cols)

    for col in ["in_theme_v4", "in_manual_ai_watchlist", "in_momentum_candidate", "in_swing_train", "in_swing_live", "in_discovered", "is_eligible", "manual_eligible", "discovered_eligible"]:
        if col not in out.columns:
            out[col] = False
        out[col] = out[col].where(out[col].notna(), False).astype(bool)
    out["is_eligible"] = out["is_eligible"] | out["manual_eligible"] | out["discovered_eligible"]
    if "asset_type" not in out.columns:
        out["asset_type"] = ""
    if "type" in out.columns:
        out["asset_type"] = out["asset_type"].where(out["asset_type"].astype(str).str.len() > 0, out["type"])
    out["eligible_reason"] = ""
    out.loc[out["is_eligible"], "eligible_reason"] = "theme_liquidity_filter"
    out.loc[out["discovered_eligible"], "eligible_reason"] = "dynamic_discovery"
    out.loc[out["manual_eligible"], "eligible_reason"] = "manual_ai_watchlist"
    source_cols = ["in_theme_v4", "in_manual_ai_watchlist", "in_momentum_candidate", "in_swing_train", "in_swing_live", "in_discovered"]
    out["source_count"] = out[source_cols].sum(axis=1).astype(int)
    out = out.sort_values(["is_eligible", "source_count", "ticker"], ascending=[False, False, True]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def load_shared_universe(*, eligible_only: bool = True, rebuild: bool = False) -> pd.DataFrame:
    if rebuild or not SHARED_UNIVERSE_CSV.exists():
        df = build_shared_universe()
    else:
        df = pd.read_csv(SHARED_UNIVERSE_CSV)
        for col in ["in_theme_v4", "in_momentum_candidate", "in_swing_train", "in_swing_live", "in_discovered", "is_eligible"]:
            if col in df.columns:
                df[col] = df[col].astype(bool)
    if eligible_only and "is_eligible" in df.columns:
        df = df[df["is_eligible"]].copy()
    return df.reset_index(drop=True)


def shared_tickers(*, eligible_only: bool = True, rebuild: bool = False) -> list[str]:
    return load_shared_universe(eligible_only=eligible_only, rebuild=rebuild)["ticker"].astype(str).tolist()


def summarize_universe(df: pd.DataFrame) -> dict[str, int]:
    return {
        "raw_total": int(len(df)),
        "eligible": int(df["is_eligible"].sum()) if "is_eligible" in df.columns else 0,
        "theme_v4": int(df["in_theme_v4"].sum()) if "in_theme_v4" in df.columns else 0,
        "manual_ai_watchlist": int(df["in_manual_ai_watchlist"].sum()) if "in_manual_ai_watchlist" in df.columns else 0,
        "momentum_candidate": int(df["in_momentum_candidate"].sum()) if "in_momentum_candidate" in df.columns else 0,
        "swing_train": int(df["in_swing_train"].sum()) if "in_swing_train" in df.columns else 0,
        "swing_live": int(df["in_swing_live"].sum()) if "in_swing_live" in df.columns else 0,
        "discovered": int(df["in_discovered"].sum()) if "in_discovered" in df.columns else 0,
    }
