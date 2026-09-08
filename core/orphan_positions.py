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

*Not* an orphan: whatever the SPY daytrader is holding right now. It keeps no
``managed`` state file, so until 2026-09-01 every intraday 0DTE it opened was
reported as an orphan for the ~15 minutes it was held — a different contract
symbol every session (SPY260828P00771000 on 08-28, SPY260831C00767000 on
08-31), which read as a fresh orphan appearing daily on top of a static set of
12. ``spy_daytrader_symbols`` reads its book so the scan stops inventing one.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]


class ClaimBookUnreadable(RuntimeError):
    """A module's book could not be read, so its claims are unknown.

    Distinct from "the module claims nothing". Conflating the two is what turns
    a transient read failure into a real position that reads as unowned, and an
    unowned position is one a sibling's reconcile will adopt and liquidate.
    """


def _load_json_book(path: Path, *, what: str, attempts: int = 3) -> dict | None:
    """Read one JSON book, retrying briefly. None means the file is absent.

    These files are rewritten in place by live modules while the scan reads
    them, so a read can land mid-write and see truncated JSON. On 2026-09-02 at
    11:50 exactly that happened to multi_ticker_swing's book ("Expecting value:
    line 1 column 1") and the orphan count jumped from 12 to 15 for one scan —
    three positions swing genuinely owned, reported as unowned, because an
    unreadable book returned an empty claim set.

    Raises ClaimBookUnreadable when the file exists and still will not parse,
    so a caller using this to *veto* an adoption can fail closed instead of
    silently treating the module as holding nothing.
    """

    import time as _time

    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return json.loads(Path(path).read_text())
        except FileNotFoundError:
            return None
        except Exception as exc:  # noqa: BLE001 - retried below
            last = exc
            if attempt + 1 < attempts:
                _time.sleep(0.05 * (attempt + 1))
    raise ClaimBookUnreadable(f"could not read the {what} at {path} ({last})") from last


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

# intraday_structure keeps a third shape again: `{"open": {setup_id: {...}}}`
# keyed by setup id, not ticker, with the contract under `occ`. It was absent
# from every claim reader until 2026-09-01, and that absence is what let
# multi_ticker_swing adopt and force-liquidate six of its option positions
# within one to three minutes of entry — QQQ260901P00708000 bought 10:27:05 ET
# and sold out from under it at 10:30:08 ET. The executor never learned the
# contracts were gone and retried the exit ~190x/min for eight hours.
INTRADAY_STRUCTURE_BOOK_PATH = REPO / "Data/inference/intraday_structure/open_option_positions.json"

# The SPY daytrader writes no managed state. Its current holding is the last
# `open_long_symbol` / `open_short_symbol` it recorded in the newest live-run
# session directory's trade-events stream.
SPY_LIVE_RUNS_ROOT = REPO / "Data/inference/live_runs"
SPY_SESSION_GLOB = "*_live_spy"
# Enough of the tail to cover the last event without reading a multi-MB file on
# every risk pass; trade-events rows are well under 4 KB each.
_TAIL_BYTES = 65_536


def spy_daytrader_symbols(
    runs_root: Path | None = None,
    *,
    session_glob: str = SPY_SESSION_GLOB,
) -> set[str]:
    """Option symbols the SPY daytrader currently holds, if any.

    Reads only the tail of the newest session's ``trade-events.jsonl``: the file
    grows all session and this runs every few minutes. A session that has never
    traded, a truncated tail, or no session at all yields nothing rather than
    raising — the same over-report-never-under-report stance as the other books.
    """

    root = Path(runs_root or SPY_LIVE_RUNS_ROOT)
    try:
        sessions = sorted((p for p in root.glob(session_glob) if p.is_dir()),
                          key=lambda p: p.name)
    except Exception as exc:  # noqa: BLE001 - an unreadable run root is not fatal
        logger.warning("orphan scan: could not list SPY live runs at %s (%s)", root, exc)
        return set()
    if not sessions:
        return set()
    events = sessions[-1] / "trade-events.jsonl"
    try:
        with events.open("rb") as fh:
            fh.seek(0, 2)
            start = max(0, fh.tell() - _TAIL_BYTES)
            fh.seek(start)
            tail = fh.read().decode("utf-8", errors="ignore").splitlines()
            # A mid-file seek almost always lands inside a row, so the first
            # line is a fragment. Reading from byte 0 it is a whole record and
            # dropping it would lose the only event of a session that traded once.
            if start:
                tail = tail[1:]
    except FileNotFoundError:
        return set()
    except Exception as exc:  # noqa: BLE001 - an unreadable book is not fatal
        logger.warning("orphan scan: could not read the SPY daytrader book at %s (%s)",
                       events, exc)
        return set()
    out: set[str] = set()
    for line in reversed(tail):
        try:
            result = (json.loads(line).get("payload") or {}).get("result") or {}
        except Exception:  # noqa: BLE001 - skip a partial or malformed row
            continue
        if not isinstance(result, dict):
            continue
        for key in ("open_long_symbol", "open_short_symbol"):
            value = result.get(key)
            if value:
                out.add(str(value))
        # The newest parseable row is the current book, held or flat. Stop there
        # rather than unioning the whole tail, which would keep claiming
        # contracts the daytrader closed hours ago.
        return out
    return out


def swing_book_symbols(path: Path | None = None) -> set[str]:
    """Symbols multi_ticker_swing currently holds, by option symbol and ticker."""

    out: set[str] = set()
    book = _load_json_book(path or SWING_BOOK_PATH, what="swing book")
    positions = (book or {}).get("positions") or []
    for position in positions:
        if not isinstance(position, dict):
            continue
        for key in ("option_symbol", "ticker"):
            value = position.get(key)
            if value:
                out.add(str(value))
    return out


def intraday_structure_symbols(path: Path | None = None) -> set[str]:
    """OCC contract symbols intraday_structure currently claims.

    Its book is keyed by setup id rather than ticker, so it cannot go through
    ``managed_symbols``. Only the contract is claimed, never the bare
    underlying: this module is options-only and several of its names are index
    ETFs that every other module also trades, so claiming "QQQ" would stop a
    sibling from adopting an unrelated QQQ contract — including one of its own
    after a state loss. Same one-symbol-per-position rule ``managed_symbols``
    uses for an option-route entry.
    """

    out: set[str] = set()
    book = _load_json_book(
        path or INTRADAY_STRUCTURE_BOOK_PATH, what="intraday_structure book"
    )
    for position in ((book or {}).get("open") or {}).values():
        if not isinstance(position, dict):
            continue
        occ = position.get("occ")
        if occ:
            out.add(str(occ).strip().upper())
    return out


def claimed_symbols(
    *,
    state_paths: Mapping[str, Path] | None = None,
    swing_book_path: Path | None = None,
    spy_runs_root: Path | None = None,
    intraday_book_path: Path | None = None,
    extra_claimed: Iterable[str] = (),
    exclude: Iterable[str] = (),
    strict: bool = False,
) -> set[str]:
    """Every symbol any module currently claims, upper-cased.

    One reader, so a module added to the system gets picked up by the orphan
    scan and by every sibling's reconcile at the same time. Before this existed
    each caller kept its own list and they drifted: intraday_structure was in
    neither, which is how its contracts read as unowned to everything.

    ``exclude`` names books to skip, for a module asking "what do my *siblings*
    claim" — passing its own name keeps its own positions from looking
    sibling-owned. Valid names are the ``MANAGED_STATE_PATHS`` keys plus
    ``multi_ticker_swing``, ``spy_daytrader`` and ``intraday_structure``.

    ``strict`` decides what an unreadable book means. The default is lenient,
    which suits a detector that only warns: an incomplete set makes it
    over-report, never under-report. A caller using this to *veto* an adoption
    must pass ``strict=True`` and get a ClaimBookUnreadable, because there the
    lenient answer is actively dangerous — it says "nobody owns this" about a
    position somebody does own, and the reconcile then adopts it.
    """

    skip = {str(name).strip() for name in exclude}
    paths = dict(MANAGED_STATE_PATHS if state_paths is None else state_paths)
    for name in skip:
        paths.pop(name, None)

    def _read(fn, *args):
        try:
            return fn(*args)
        except ClaimBookUnreadable as exc:
            if strict:
                raise
            logger.warning("orphan scan: %s — treating it as claiming nothing", exc)
            return set()

    out = {s.upper() for s in _read(managed_symbols, paths)}
    if "multi_ticker_swing" not in skip:
        out |= {s.upper() for s in _read(swing_book_symbols, swing_book_path)}
    if "spy_daytrader" not in skip:
        out |= {s.upper() for s in _read(spy_daytrader_symbols, spy_runs_root)}
    if "intraday_structure" not in skip:
        out |= _read(intraday_structure_symbols, intraday_book_path)
    return out | {str(s).strip().upper() for s in extra_claimed}


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

    A missing state file yields nothing for that module rather than raising: a
    detector that cannot run is worse than one that reports a little too much.
    A state file that exists and will not parse raises ClaimBookUnreadable, so
    a caller vetoing an adoption can tell "claims nothing" from "cannot tell".
    """

    out: dict[str, list[str]] = {}
    for module, path in (
        MANAGED_STATE_PATHS if state_paths is None else state_paths
    ).items():
        managed = (_load_json_book(path, what=f"{module} state") or {}).get("managed") or {}
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
    spy_runs_root: Path | None = None,
    intraday_book_path: Path | None = None,
) -> list[Orphan]:
    """Return the broker positions nothing claims, largest exposure first.

    Every book is read from disk via ``claimed_symbols``: the four `managed`
    state files, plus multi_ticker_swing's own position list, plus whatever the
    SPY daytrader is holding, plus intraday_structure's option book.
    ``extra_claimed`` is for a caller that knows about a holding none of those
    record yet. Passing an incomplete set makes the scan over-report, never
    under-report, which is the safe direction for a detector that only warns.
    """

    claimed = claimed_symbols(
        state_paths=state_paths,
        swing_book_path=swing_book_path,
        spy_runs_root=spy_runs_root,
        intraday_book_path=intraday_book_path,
        extra_claimed=extra_claimed,
    )
    orphans: list[Orphan] = []
    for position in positions:
        symbol = str(position.get("symbol") or "").strip()
        if not symbol or symbol.upper() in claimed:
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
    "SPY_LIVE_RUNS_ROOT",
    "INTRADAY_STRUCTURE_BOOK_PATH",
    "managed_symbols",
    "swing_book_symbols",
    "spy_daytrader_symbols",
    "intraday_structure_symbols",
    "claimed_symbols",
    "find_orphans",
    "log_orphans",
]
