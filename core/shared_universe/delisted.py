"""Symbols that no longer trade, so data providers can skip them before an API call.

The list lives in ``delisted_symbols.txt`` next to this module (tracked in git;
``Data/`` is not). It exists only to stop wasted requests -- Alpaca bars, news
sources, the earnings calendar, CBOE and Schwab chains. It deliberately does NOT
touch the universe, eligibility or any module score: ``shared_universe.csv``
still lists these names and their cached bars stay exactly as they are, so
everything downstream sees the same data it saw before.

Fail-open by design: a missing file means "nothing is delisted", and a malformed
line is logged and ignored rather than raised, because this check sits inside
providers the live stack depends on and a typo here must never block a fetch.

Refresh the list with ``scripts/build_delisted_symbols.py``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

DELISTED_SYMBOLS_PATH = Path(__file__).with_name("delisted_symbols.txt")

logger = logging.getLogger(__name__)

_cache: dict[Path, tuple[float, frozenset[str]]] = {}


def clean_symbol(symbol: object) -> str:
    return str(symbol).strip().upper().lstrip("$")


def parse_delisted_symbols(text: str, *, source: str = "delisted_symbols.txt") -> frozenset[str]:
    """One symbol per line; anything after ``#`` is a note."""
    symbols: set[str] = set()
    for lineno, line in enumerate(text.splitlines(), start=1):
        entry = line.split("#", 1)[0].strip()
        if not entry:
            continue
        if any(ch.isspace() for ch in entry):
            logger.error("%s:%d: expected one symbol per line, got %r -- line ignored", source, lineno, line)
            continue
        symbols.add(clean_symbol(entry))
    return frozenset(symbols)


def load_delisted_symbols(path: Path | str | None = None) -> frozenset[str]:
    """The current list, re-read only when the file changes."""
    path = Path(path) if path is not None else DELISTED_SYMBOLS_PATH
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return frozenset()
    cached = _cache.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    symbols = parse_delisted_symbols(path.read_text(encoding="utf-8"), source=str(path))
    _cache[path] = (mtime, symbols)
    return symbols


def is_delisted(symbol: object, *, path: Path | str | None = None) -> bool:
    return clean_symbol(symbol) in load_delisted_symbols(path)


def drop_delisted(
    symbols: Iterable[str],
    *,
    context: str,
    path: Path | str | None = None,
) -> list[str]:
    """``symbols`` minus the delisted ones, order preserved, with one log line per call."""
    delisted = load_delisted_symbols(path)
    kept: list[str] = []
    skipped: list[str] = []
    for symbol in symbols:
        (skipped if clean_symbol(symbol) in delisted else kept).append(symbol)
    if skipped:
        names = sorted({clean_symbol(s) for s in skipped})
        shown = ", ".join(names[:8]) + (f", +{len(names) - 8} more" if len(names) > 8 else "")
        logger.info("%s: skipping %d delisted symbol(s) before any API call (%s)", context, len(names), shown)
    return kept
