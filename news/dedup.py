"""News deduplication helpers."""

from __future__ import annotations

import re

import pandas as pd


def _norm_url(value: str) -> str:
    return re.sub(r"[?#].*$", "", str(value or "").strip().lower())


def deduplicate_news(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["_url_key"] = out["url"].map(_norm_url) if "url" in out.columns else ""
    out["_headline_key"] = (
        out["headline"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    )
    key_cols = ["ticker", "content_hash"]
    out = out.sort_values(["timestamp", "source"]).drop_duplicates(key_cols, keep="first")
    with_url = out["_url_key"].astype(str).ne("")
    out = pd.concat(
        [
            out.loc[with_url].drop_duplicates(["ticker", "_url_key"], keep="first"),
            out.loc[~with_url],
        ],
        ignore_index=True,
    )
    out = out.drop_duplicates(["ticker", "_headline_key", "timestamp"], keep="first")
    return out.drop(columns=["_url_key", "_headline_key"]).sort_values(["timestamp", "ticker"]).reset_index(drop=True)

