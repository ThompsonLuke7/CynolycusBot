"""Broker positions that no strategy claims.

An orphan is a position the account holds that is in no module's managed state
and in no sibling's local book. Nothing sizes it, nothing stops it, nothing
exits it — it simply sits there until a human notices.

They arise two ways, and both have happened:

*Option exercise.* A long call that expires in the money is auto-exercised into
100 shares per contract. The module drops the *option* from managed at the next
run (`not_found`) and never picks up the *equity*. dealer_ranker's
TECK260724C00057000 and SU260724C00062000 did this on the 2026-07-24 expiry and
were still unowned five weeks later; an earlier batch (GRAB, U, SMCI, FIG) had
to be cleaned up by hand after the 2026-08-14 expiry.

*A lost claim.* A module's managed state stops naming a position it opened —
through a crash, a state rewrite, or a reconcile that dropped it while the
broker was not reporting it. meta_ranker's CRWV and EVH and HTF's AEVA are all
2026-07 entries whose claims vanished.

This module only *reports*. It never flattens anything, and that restraint is
deliberate: `multi_ticker_swing` once force-sold a legitimate HTF position it
had mistaken for an assignment (2026-07-21). Deciding what to do with an orphan
is a human call, made through `core.startup_queue`.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]

# Every module that keeps a `{"managed": {ticker: {...}}}` state file.
MANAGED_STATE_PATHS: dict[str, Path] = {
    "meta_ranker": REPO / "signals/meta_context/meta_ranker/live_state.json",
    "momentum_expansion": REPO / "strategies/momentum_expansion/live/momentum_live_state.json",
    "multi_ticker_swing_htf": REPO / "strategies/multi_ticker_swing_htf/live/htf_live_state.json",
    "dealer_ranker": REPO / "Data/inference/dealer_ranker/live_state.json",
}

# multi_ticker_swing keeps a different shape: a flat position list rather than a
# `managed` map. It has to be read, not skipped — swing holds most of the
# account's option contracts, and a scan blind to them would report every one
# as an orphan every five minutes until nobody read the warning any more.
SWING_BOOK_PATH = REPO / "Data/inference/multi_ticker_swing/open_positions.json"


def swing_book_symbols(path: Path | None = None) -> set[str]:
    """Symbols multi_ticker_swing currently holds, by option symbol and ticker."""

    out: set[str] = set()
    try:
        positions = json.loads(Path(path or SWING_BOOK_PATH).read_text()).get("positions") or []
    except Exception as exc:  # noqa: BLE001 - an unreadable book is not fatal
        logger.warning("orphan scan: could not read the swing book at %s (%s)",
                       path or SWING_BOOK_PATH, exc)
        return out
    for position in positions:
        if not isinstance(position, dict):
            continue
        for key in ("option_symbol", "ticker"):
            value = position.get(key)
            if value:
                out.add(str(value))
    return out


@dataclass(frozen=True)
class Orphan:
    """One unclaimed broker position."""

    symbol: str
    qty: float
    market_value: float
    unrealized_pl: float
    asset_class: str

    @property
    def is_option(self) -> bool:
        return self.asset_class == "us_option"


def managed_symbols(
    state_paths: Mapping[str, Path] | None = None,
) -> dict[str, list[str]]:
    """Map every symbol a 4H module claims to the modules claiming it.

    A missing or unreadable state file yields nothing for that module rather
    than raising: a detector that cannot run is worse than one that reports a
    little too much, and the caller is told which modules were readable.
    """

    out: dict[str, list[str]] = {}
    for module, path in (state_paths or MANAGED_STATE_PATHS).items():
        try:
            managed = json.loads(Path(path).read_text()).get("managed") or {}
        except Exception as exc:  # noqa: BLE001 - an unreadable state is not fatal
            logger.warning("orphan scan: could not read %s state at %s (%s)",
                           module, path, exc)
            continue
        for ticker, st in managed.items():
            if not isinstance(st, dict):
                continue
            symbol = st.get("occ") if st.get("route") == "option" else st.get("symbol", ticker)
            if symbol:
                out.setdefault(str(symbol), []).append(module)
    return out


def find_orphans(
    positions: Iterable[Mapping[str, Any]],
    *,
    extra_claimed: Iterable[str] = (),
    state_paths: Mapping[str, Path] | None = None,
    swing_book_path: Path | None = None,
) -> list[Orphan]:
    """Return the broker positions nothing claims, largest exposure first.

    Every book is read from disk: the four `managed` state files plus
    multi_ticker_swing's own position list. ``extra_claimed`` is for a caller
    that knows about a holding none of those files record yet. Passing an
    incomplete set makes the scan over-report, never under-report, which is the
    safe direction for a detector that only warns.
    """

    claimed = (
        set(managed_symbols(state_paths))
        | swing_book_symbols(swing_book_path)
        | {str(s) for s in extra_claimed}
    )
    orphans: list[Orphan] = []
    for position in positions:
        symbol = str(position.get("symbol") or "").strip()
        if not symbol or symbol in claimed:
            continue
        try:
            qty = float(position.get("qty") or 0)
        except (TypeError, ValueError):
            continue
        if qty == 0:
            continue
        orphans.append(
            Orphan(
                symbol=symbol,
                qty=qty,
                market_value=_number(position.get("market_value")),
                unrealized_pl=_number(position.get("unrealized_pl")),
                asset_class=str(position.get("asset_class") or "us_equity"),
            )
        )
    orphans.sort(key=lambda o: abs(o.market_value), reverse=True)
    return orphans


def log_orphans(orphans: list[Orphan], *, logger_: logging.Logger | None = None) -> None:
    """Report the scan at WARNING, or say plainly that there are none."""

    log = logger_ or logger
    if not orphans:
        log.info("orphan scan: every broker position is claimed by a module")
        return
    total_mv = sum(o.market_value for o in orphans)
    total_pl = sum(o.unrealized_pl for o in orphans)
    log.warning(
        "orphan scan: %d position(s) no module manages — $%s market value, "
        "%s unrealized. Nothing will stop or exit these; decide with "
        "core.startup_queue.",
        len(orphans), f"{total_mv:,.0f}", f"{total_pl:+,.0f}",
    )
    for o in orphans:
        log.warning("  orphan %-22s %-10s qty=%-10g mv=$%.0f unrealized=%+.0f",
                    o.symbol, o.asset_class, o.qty, o.market_value, o.unrealized_pl)


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


__all__ = [
    "Orphan",
    "MANAGED_STATE_PATHS",
    "SWING_BOOK_PATH",
    "managed_symbols",
    "swing_book_symbols",
    "find_orphans",
    "log_orphans",
]
