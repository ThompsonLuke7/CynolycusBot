"""Ticker extraction for Reddit posts."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from signals.social_attention.config import AMBIGUOUS_BARE_TICKERS, REDDIT_MENTIONS_PATH, REDDIT_POSTS_PATH, REPO_ROOT
from signals.social_attention.io import read_table, write_table

CASHTAG_RE = re.compile(r"(?<![A-Za-z0-9_])\$([A-Za-z]{1,6})(?![A-Za-z0-9_])")
BARE_TICKER_RE = re.compile(r"\b[A-Z]{1,6}\b")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9&.\- ]{1,80}")


def clean_ticker(value: object) -> str:
    return str(value or "").strip().upper().replace("$", "")


def load_universe_tickers() -> set[str]:
    tickers: set[str] = set()
    candidates = [
        REPO_ROOT / "signals" / "meta_context" / "data" / "processed" / "context_backtest_universe.csv",
        REPO_ROOT / "strategies" / "multi_ticker_swing" / "config" / "swing_trader_universe_v3.csv",
        REPO_ROOT / "strategies" / "multi_ticker_swing" / "config" / "universe.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "ticker" in df.columns:
            tickers |= {clean_ticker(t) for t in df["ticker"] if clean_ticker(t)}
    try:
        from strategies.momentum_expansion.data.universe import load_candidate_metadata

        df = load_candidate_metadata()
        if "ticker" in df.columns:
            tickers |= {clean_ticker(t) for t in df["ticker"] if clean_ticker(t)}
    except Exception:
        pass
    return tickers


def build_alias_map(universe: pd.DataFrame | None = None) -> dict[str, set[str]]:
    if universe is None or universe.empty:
        return {}
    aliases: dict[str, set[str]] = defaultdict(set)
    name_cols = [c for c in ("name", "company", "company_name", "notes") if c in universe.columns]
    if not name_cols or "ticker" not in universe.columns:
        return {}
    for _, row in universe.iterrows():
        ticker = clean_ticker(row.get("ticker"))
        if not ticker:
            continue
        for col in name_cols:
            text = str(row.get(col) or "").strip()
            if 3 <= len(text) <= 80 and not text.lower().startswith(("multi_ticker", "momentum")):
                aliases[ticker].add(text.lower())
    return dict(aliases)


def extract_ticker_hits(
    text: str,
    *,
    universe: Iterable[str],
    alias_map: dict[str, set[str]] | None = None,
    ambiguous_bare: set[str] = AMBIGUOUS_BARE_TICKERS,
) -> list[dict[str, str]]:
    universe_set = {clean_ticker(t) for t in universe}
    hits: dict[str, set[str]] = defaultdict(set)
    text = str(text or "")

    for match in CASHTAG_RE.finditer(text):
        ticker = clean_ticker(match.group(1))
        if ticker in universe_set:
            hits[ticker].add("cashtag")

    for match in BARE_TICKER_RE.finditer(text):
        ticker = clean_ticker(match.group(0))
        if ticker in universe_set and ticker not in ambiguous_bare:
            hits[ticker].add("bare")

    lower = f" {text.lower()} "
    for ticker, names in (alias_map or {}).items():
        if ticker not in universe_set:
            continue
        for name in names:
            if len(name) >= 3 and f" {name} " in lower:
                hits[ticker].add("alias")

    return [{"ticker": ticker, "methods": "|".join(sorted(methods))} for ticker, methods in sorted(hits.items())]


def extract_mentions(
    posts: pd.DataFrame | None = None,
    *,
    posts_path: Path | str = REDDIT_POSTS_PATH,
    output_path: Path | str = REDDIT_MENTIONS_PATH,
    universe: Iterable[str] | None = None,
    alias_map: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    posts = read_table(posts_path) if posts is None else posts.copy()
    if posts.empty:
        out = pd.DataFrame(columns=["post_id", "ticker", "methods"])
        return write_table(out, output_path)
    universe_set = set(universe or load_universe_tickers())
    rows: list[dict[str, object]] = []
    for _, post in posts.iterrows():
        text = " ".join(str(post.get(c) or "") for c in ("title", "selftext", "body", "text"))
        for hit in extract_ticker_hits(text, universe=universe_set, alias_map=alias_map):
            rows.append(
                {
                    "post_id": post["post_id"],
                    "ticker": hit["ticker"],
                    "methods": hit["methods"],
                }
            )
    out = pd.DataFrame(rows).drop_duplicates(["post_id", "ticker"]) if rows else pd.DataFrame(columns=["post_id", "ticker", "methods"])
    return write_table(out, output_path)

