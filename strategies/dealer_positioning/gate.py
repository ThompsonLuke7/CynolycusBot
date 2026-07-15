"""Rule-based dealer-positioning entry gate.

A thin, deterministic overlay that the option-trading modules can consult right
before they commit a contract: given a symbol, a directional side (call/put) and
the *live* entry price, it answers whether the dealer gamma structure contradicts
the trade (buying a call straight into a call wall, or with a magnet pulling
price down, and the mirror for puts).

Design notes / guarantees:
  * NO machine learning. We do not have enough accumulated snapshot history for
    that; this is a hand-written structural veto, nothing more.
  * FAIL-OPEN. If a symbol has no captured snapshot, the file is missing, or the
    snapshot is stale, the gate returns ``allow`` so it can never block a trade
    just because dealer data is absent.
  * LIVE-PRICE AWARE. Proximity is recomputed against the caller's live entry
    price using the *absolute* strike levels (call_wall / put_wall / magnet),
    which do not move intraday, rather than the ``pct_to_*`` fields that are
    frozen to last night's spot.
  * SCOPE-MATCHED. Callers pass the expiry scope that matches the contract they
    are buying: ``daily_week`` for nearest-expiry (30m swing / 0DTE), and
    ``through_month`` / ``two_months`` for the monthly 4H modules.

The data source is the nightly ``dealer_level_summary.parquet`` written by
``scripts/capture_historical_snapshots.py``.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT_ROOT = REPO / "Data" / "dealer_positioning" / "historical_snapshots"

# Scopes, matched to the expiry a module actually buys.
SCOPE_NEAREST = "daily_week"      # 30m swing / SPY 0DTE-1DTE
SCOPE_MONTHLY = "through_month"   # Meta / HTF / Momentum (next monthly)
SCOPE_TWO_MONTHS = "two_months"
VALID_SCOPES = {SCOPE_NEAREST, SCOPE_MONTHLY, SCOPE_TWO_MONTHS}

_CALL_SIDES = {"call", "long", "buy", "1", "c"}
_PUT_SIDES = {"put", "short", "sell", "-1", "p"}


@dataclass(frozen=True)
class DealerGateConfig:
    """Thresholds for the structural veto (all fractions of entry price)."""

    # A wall this close overhead (call) / underfoot (put) is "heavy resistance".
    wall_proximity_pct: float = 0.0075
    # A magnet at least this far the *wrong* way is a "magnet pulling against us".
    magnet_adverse_pct: float = 0.005
    # Snapshots older than this many calendar days fail open (allow).
    max_age_days: int = 4

    @classmethod
    def from_env(cls) -> "DealerGateConfig":
        def _f(name: str, default: float) -> float:
            raw = os.environ.get(name)
            if raw in (None, ""):
                return default
            try:
                return float(raw)
            except (TypeError, ValueError):
                return default

        return cls(
            wall_proximity_pct=_f("DEALER_GATE_WALL_PROXIMITY_PCT", cls.wall_proximity_pct),
            magnet_adverse_pct=_f("DEALER_GATE_MAGNET_ADVERSE_PCT", cls.magnet_adverse_pct),
            max_age_days=int(_f("DEALER_GATE_MAX_AGE_DAYS", cls.max_age_days)),
        )


def gate_enabled() -> bool:
    """Master switch. OFF by default so live behaviour is unchanged until opted in.

    Set ``DEALER_GATE_ENABLED=1`` (paper first) to make the gate *enforce* its
    verdict. When disabled the modules still evaluate + audit the verdict
    (observe-only) but never change routing.
    """
    return str(os.environ.get("DEALER_GATE_ENABLED", "")).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class DealerGateVerdict:
    symbol: str
    side: str            # normalized: "call" or "put"
    scope: str
    action: str          # "allow" or "veto"
    reason: str
    has_data: bool
    stale: bool
    entry_price: float | None = None
    spot: float | None = None
    call_wall: float | None = None
    put_wall: float | None = None
    magnet: float | None = None
    gamma_flip: float | None = None
    room_to_call_wall_pct: float | None = None
    room_to_put_wall_pct: float | None = None
    magnet_offset_pct: float | None = None
    dealer_direction: str | None = None
    snapshot_date: str | None = None

    @property
    def vetoed(self) -> bool:
        return self.action == "veto"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DealerLevelStore:
    """Loads + caches the latest nightly ``dealer_level_summary.parquet``.

    Cached by file mtime so a fresh nightly write is picked up automatically
    without leaking stale data within a session.
    """

    def __init__(self, summary_path: Path | None = None, *, snapshot_root: Path = SNAPSHOT_ROOT) -> None:
        self._explicit_path = Path(summary_path) if summary_path is not None else None
        self._root = Path(snapshot_root)
        self._cache_key: tuple[str, float] | None = None
        self._index: dict[tuple[str, str], dict[str, Any]] = {}

    def latest_summary_path(self) -> Path | None:
        if self._explicit_path is not None:
            return self._explicit_path if self._explicit_path.exists() else None
        paths = sorted(self._root.glob("*/dealer_level_summary.parquet"))
        return paths[-1] if paths else None

    def _ensure_loaded(self) -> None:
        path = self.latest_summary_path()
        if path is None:
            self._cache_key = None
            self._index = {}
            return
        key = (str(path), path.stat().st_mtime)
        if key == self._cache_key:
            return
        frame = pd.read_parquet(path)
        index: dict[tuple[str, str], dict[str, Any]] = {}
        for row in frame.to_dict(orient="records"):
            symbol = str(row.get("symbol", "")).strip().upper()
            scope = str(row.get("scope", "")).strip()
            if symbol and scope:
                index[(symbol, scope)] = row
        self._cache_key = key
        self._index = index

    def get(self, symbol: str, scope: str) -> dict[str, Any] | None:
        self._ensure_loaded()
        return self._index.get((str(symbol).strip().upper(), str(scope).strip()))


# Process-wide default store (mtime-cached); callers may inject their own.
_DEFAULT_STORE = DealerLevelStore()


def _normalize_side(side: Any) -> str:
    raw = str(side).strip().lower()
    if raw in _CALL_SIDES:
        return "call"
    if raw in _PUT_SIDES:
        return "put"
    # numeric directions (e.g. sig.direction)
    try:
        return "call" if float(raw) > 0 else "put"
    except (TypeError, ValueError):
        return "call"


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN
        return None
    return out


def _snapshot_stale(snapshot_date: Any, *, now: date, max_age_days: int) -> bool:
    try:
        snap = datetime.fromisoformat(str(snapshot_date)).date()
    except (TypeError, ValueError):
        return False  # cannot parse -> do not treat as stale (fail open on allow anyway)
    return (now - snap).days > max_age_days


def evaluate_dealer_gate(
    symbol: str,
    side: Any,
    entry_price: float,
    scope: str,
    *,
    store: DealerLevelStore | None = None,
    config: DealerGateConfig | None = None,
    now: date | None = None,
) -> DealerGateVerdict:
    """Return a structural allow/veto verdict for entering ``side`` on ``symbol``.

    Fails open (``allow``) on any missing/stale/invalid data. Never raises.
    """
    store = store if store is not None else _DEFAULT_STORE
    config = config if config is not None else DealerGateConfig.from_env()
    now = now or datetime.now().date()
    norm_side = _normalize_side(side)

    base = dict(symbol=str(symbol).strip().upper(), side=norm_side, scope=scope,
                entry_price=_num(entry_price))

    price = _num(entry_price)
    if price is None or price <= 0:
        return DealerGateVerdict(**base, action="allow", reason="no_entry_price",
                                 has_data=False, stale=False)

    row = store.get(symbol, scope)
    if row is None:
        return DealerGateVerdict(**base, action="allow", reason="no_dealer_data",
                                 has_data=False, stale=False)

    snapshot_date = str(row.get("snapshot_date") or "") or None
    stale = _snapshot_stale(snapshot_date, now=now, max_age_days=config.max_age_days)

    call_wall = _num(row.get("call_wall"))
    put_wall = _num(row.get("put_wall"))
    magnet = _num(row.get("nearest_magnet"))
    if magnet is None:
        magnet = _num(row.get("magnet"))
    gamma_flip = _num(row.get("gamma_flip"))
    spot = _num(row.get("spot"))
    direction = row.get("dealer_direction")
    if isinstance(direction, str):
        direction = direction.strip() or None
    else:
        direction = None

    room_call = (call_wall - price) / price if call_wall is not None else None
    room_put = (put_wall - price) / price if put_wall is not None else None
    magnet_off = (magnet - price) / price if magnet is not None else None

    common = dict(
        **base, has_data=True, stale=stale, spot=spot, call_wall=call_wall,
        put_wall=put_wall, magnet=magnet, gamma_flip=gamma_flip,
        room_to_call_wall_pct=room_call, room_to_put_wall_pct=room_put,
        magnet_offset_pct=magnet_off, dealer_direction=direction,
        snapshot_date=snapshot_date,
    )

    if stale:
        return DealerGateVerdict(**common, action="allow", reason="stale_dealer_data")

    prox = config.wall_proximity_pct
    adverse = config.magnet_adverse_pct

    if norm_side == "call":
        # Heavy resistance: a call wall sitting just overhead (spot below it).
        if room_call is not None and 0.0 <= room_call < prox:
            return DealerGateVerdict(**common, action="veto", reason="call_wall_overhead")
        # Magnet pulling price DOWN, meaningfully below the entry.
        if magnet_off is not None and magnet_off <= -adverse:
            return DealerGateVerdict(**common, action="veto", reason="magnet_below")
        return DealerGateVerdict(**common, action="allow", reason="clear")

    # put side (mirror)
    if room_put is not None and -prox < room_put <= 0.0:
        return DealerGateVerdict(**common, action="veto", reason="put_wall_support")
    if magnet_off is not None and magnet_off >= adverse:
        return DealerGateVerdict(**common, action="veto", reason="magnet_above")
    return DealerGateVerdict(**common, action="allow", reason="clear")
