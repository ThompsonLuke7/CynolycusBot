"""Portfolio construction primitives: caps, vol targeting, correlation-aware
down-weighting, liquidity constraints, and whole-share rounding.

Every function here is a pure function of its arguments -- no I/O, no dates
parsed from disk, no default "current time". Callers (``portfolio_backtest.py``
for research, a future live wiring outside this plan) are responsible for
ensuring any ``cov``/``corr``/``adv_dollars`` passed in was itself computed
causally (see ``covariance.py``). This module cannot look ahead because it
never touches a clock or a file.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SizingConfig:
    capital: float = 100_000.0
    target_concurrent: int = 10          # slot budget: equal-weight base = capital / target_concurrent
    per_position_cap_pct: float = 0.12   # max gross $ in one name, as a fraction of capital
    per_sector_cap_pct: float | None = 0.30
    per_theme_cap_pct: float | None = 0.25
    target_portfolio_vol_annual: float = 0.15
    max_vol_scale: float = 3.0           # clamp on the vol-target multiplier both directions
    adv_participation_cap_pct: float | None = 0.05   # max position $ as a fraction of trailing $ADV
    corr_penalty_strength: float = 1.0   # 0 disables the correlation-aware down-weight


@dataclass
class OpenPosition:
    ticker: str
    dollar_size: float
    shares: int
    sector: str | None = None
    theme: str | None = None


def book_weights(book: dict[str, OpenPosition], capital: float) -> dict[str, float]:
    if capital <= 0:
        return {t: 0.0 for t in book}
    return {t: p.dollar_size / capital for t, p in book.items()}


def _book_dollars_by_group(book: dict[str, OpenPosition], attr: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for pos in book.values():
        key = getattr(pos, attr)
        if key is None:
            continue
        out[key] = out.get(key, 0.0) + pos.dollar_size
    return out


def cap_headroom(
    book: dict[str, OpenPosition], *, attr: str, group: str | None,
    cap_pct: float | None, capital: float,
) -> float:
    """Remaining $ room under a per-group (sector/theme) cap.

    Unlimited (``inf``) when the group is unknown (``None`` -- e.g. sector
    metadata is entirely "Unknown" today per plan defect D1, see
    ``portfolio_backtest.py`` module docstring) or the cap itself is unset.
    """
    if group is None or cap_pct is None:
        return float("inf")
    used = _book_dollars_by_group(book, attr).get(group, 0.0)
    return max(0.0, cap_pct * capital - used)


def correlation_penalty(
    ticker: str, book: dict[str, OpenPosition], corr: pd.DataFrame | None, *, strength: float,
) -> float:
    """Down-weight multiplier in (0, 1] from mean |correlation| of ``ticker``
    to names already in the open book.

    1.0 (no penalty) when the book is empty, ``strength`` is 0, or ``corr``
    doesn't cover this ticker/any held name (unknown correlation is treated
    as "no evidence of duplication", not as zero correlation) -- a Ledoit-Wolf
    matrix with N names always has an opinion, so this only fires open-loop
    when the covariance snapshot itself was unavailable that day.
    """
    if strength <= 0 or not book or corr is None or ticker not in corr.index:
        return 1.0
    held = [t for t in book if t != ticker and t in corr.columns]
    if not held:
        return 1.0
    mean_abs_corr = float(corr.loc[ticker, held].abs().mean())
    if not np.isfinite(mean_abs_corr):
        return 1.0
    return 1.0 / (1.0 + strength * mean_abs_corr)


def portfolio_vol_scale(
    weights: dict[str, float], cov: pd.DataFrame | None, *, target_vol: float, max_scale: float,
) -> float:
    """Scalar ``k`` s.t. uniformly scaling ``weights`` by ``k`` makes the
    annualized ex-ante vol of the book equal ``target_vol``.

    ``cov`` must already be annualized (``covariance.ledoit_wolf_asof`` does
    this). Falls back to 1.0 (no scaling) when covariance is unavailable, no
    weight overlaps it, or the book has zero/degenerate variance -- a
    disabled vol target must never silently zero out or blow up sizing.
    """
    if cov is None or not weights:
        return 1.0
    tickers = [t for t in weights if t in cov.index]
    if not tickers:
        return 1.0
    w = np.array([weights[t] for t in tickers], dtype=float)
    sub = cov.loc[tickers, tickers].to_numpy()
    var = float(w @ sub @ w)
    if not np.isfinite(var) or var <= 0:
        return 1.0
    realized_vol = float(np.sqrt(var))
    if realized_vol <= 0:
        return 1.0
    scale = target_vol / realized_vol
    if not np.isfinite(scale) or scale <= 0:
        return 1.0
    return float(np.clip(scale, 1.0 / max_scale, max_scale))


def whole_shares(dollar_size: float, price: float) -> int:
    """Whole shares nearest ``dollar_size`` at ``price``, floored at 0.

    Unlike the live baseline's ``core.live_4h_exec.shares_for_notional``
    (which floors at 1 -- a signal that clears every gate always gets SOME
    exposure), a sizing policy here must be able to legitimately size a name
    to 0 (e.g. sector cap already full, or the vol/liquidity-scaled dollar
    amount rounds under half a share) without that reading as an error.
    """
    if not price or price <= 0 or dollar_size <= 0:
        return 0
    return max(0, round(dollar_size / price))


@dataclass(frozen=True)
class CandidateSizing:
    ticker: str
    shares: int
    dollar_size: float
    diagnostics: dict = field(default_factory=dict)


def size_candidate(
    ticker: str, price: float, *, cfg: SizingConfig, book: dict[str, OpenPosition],
    sector: str | None = None, theme: str | None = None,
    corr: pd.DataFrame | None = None, adv_dollars: float | None = None,
    base_dollar_override: float | None = None,
) -> CandidateSizing:
    """Size one new entry against the current open book.

    Pipeline (each stage can only shrink the size further):
    correlation-aware base -> per-position cap -> per-sector cap ->
    per-theme cap -> liquidity (ADV participation) cap -> whole-share round.

    ``base_dollar_override`` lets equal-weight/fixed-notional callers reuse
    this same cap/liquidity pipeline with their own base size instead of
    ``capital / target_concurrent`` (e.g. equal-weight-top-N still wants
    liquidity/whole-share handling even though it skips vol targeting).
    """
    base = base_dollar_override if base_dollar_override is not None else (
        cfg.capital / max(1, cfg.target_concurrent)
    )
    diag: dict = {"base_dollar": base}

    penalty = correlation_penalty(ticker, book, corr, strength=cfg.corr_penalty_strength)
    diag["corr_penalty"] = penalty
    size = base * penalty
    diag["after_corr_penalty"] = size

    pos_cap = cfg.per_position_cap_pct * cfg.capital
    size = min(size, pos_cap)
    diag["after_position_cap"] = size

    sector_room = cap_headroom(book, attr="sector", group=sector, cap_pct=cfg.per_sector_cap_pct, capital=cfg.capital)
    size = min(size, sector_room)
    diag["after_sector_cap"] = size

    theme_room = cap_headroom(book, attr="theme", group=theme, cap_pct=cfg.per_theme_cap_pct, capital=cfg.capital)
    size = min(size, theme_room)
    diag["after_theme_cap"] = size

    if adv_dollars is not None and adv_dollars > 0 and cfg.adv_participation_cap_pct is not None:
        liq_cap = cfg.adv_participation_cap_pct * adv_dollars
        size = min(size, liq_cap)
        diag["after_liquidity_cap"] = size
        diag["adv_dollars"] = adv_dollars

    size = max(0.0, size)
    shares = whole_shares(size, price)
    dollar_size = shares * price
    diag["shares"] = shares
    diag["final_dollar"] = dollar_size
    return CandidateSizing(ticker=ticker, shares=shares, dollar_size=dollar_size, diagnostics=diag)
