"""Real paper-broker option execution for confirmed intraday setups.

WHY. Until this, the engine had no broker code anywhere: `load_config` hard-raises
unless `paper_only`, and every closed setup was a MODELLED fill — entry at the
next bar's open, exit at the engine's own recorded price, costs from
`ReplayPolicy`. That is an honest simulation, but it has two consequences. The
engine's results are not comparable with the other six modules' real fills, and
the large 2026-08 upgrade to it can be neither credited nor blamed, because
nothing it produced has ever been priced by a market.

DESIGN

* **A sink, not a dependency.** The engine calls `on_entry`/`on_exit` through an
  optional hook, exactly as it already does for transitions, the closed-setup
  ledger and the evidence stream. The engine stays free of broker imports, and
  execution stays removable.
* **Failures never reach the engine.** Every public method swallows its own
  exceptions. A broker outage must degrade this module to the simulation it
  already was, not stop setups from being detected and ledgered.
* **The modelled ledger keeps running.** `closed_setups.jsonl` is unchanged and
  still records the modelled fill. This writes a SEPARATE `closed_trades.jsonl`
  in the shared cross-module schema. Keeping both is the point: the difference
  between them is the execution cost this engine has never been able to measure.
* **Underlying and premium, side by side.** Every row carries `u_entry`/`u_atr`
  next to the option premium, because the 2026-08 study could not answer whether
  the option wrapper was worth it without re-deriving the underlying from bars.

SIZING is notional-based and capped, and the position count is capped per
session. A rules engine that fires on hundreds of setups a day must not be able
to express that as hundreds of open contracts.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any

from core.live_4h_exec import append_closed_trade, closed_trade_record
from strategies.intraday_structure.config import ExecutionPolicy
from strategies.intraday_structure.models import SetupRecord

logger = logging.getLogger(__name__)

LEDGER_MODULE = "intraday_structure"
OPTION_MULTIPLIER = 100.0
ET = ZoneInfo("America/New_York")


def _hhmm(text: str | None) -> time | None:
    text = str(text or "").strip()
    if not text:
        return None
    hh, _, mm = text.partition(":")
    return time(int(hh), int(mm or 0))


def _f(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and out not in (float("inf"), float("-inf")) else None


class IntradayOptionExecutor:
    """Submits paper option orders for confirmed setups and ledgers the result."""

    def __init__(
        self,
        client,
        policy: ExecutionPolicy,
        *,
        select_option_fn=None,
        ledger_root: str | None = None,
        now_fn=None,
    ) -> None:
        self._client = client
        self._policy = policy
        self._ledger_root = ledger_root
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()
        self._open: dict[str, dict[str, Any]] = {}
        self._opened_this_session = 0
        self._state_path = Path(policy.state_path)
        self._dte_cutoff = _hhmm(getattr(policy, "dte_cutoff_hhmm", None))
        self._expiring_exit = _hhmm(getattr(policy, "expiring_exit_hhmm", None))
        if select_option_fn is not None:
            self._select = select_option_fn
        else:
            # Reused rather than reimplemented: the near-dated, ATM +/-10%,
            # liquidity-ranked selector the Dealer Ranker already runs in
            # production. It defaults to excluding same-day expiry; this module
            # opts in explicitly via allow_0dte, so Dealer's behaviour is
            # unchanged.
            from strategies.dealer_positioning.live_ranked_options import select_atm_option
            self._select = select_atm_option
        self._restore()

    # -- persistence ---------------------------------------------------------

    def _restore(self) -> None:
        """Reload open positions so a restart does not orphan live contracts."""
        try:
            if self._state_path.exists():
                raw = json.loads(self._state_path.read_text(encoding="utf-8"))
                self._open = dict(raw.get("open") or {})
                logger.info("intraday execution restored %d open position(s)", len(self._open))
        except Exception:  # noqa: BLE001
            logger.exception("intraday execution: could not restore open positions")
            self._open = {}

    def _persist(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(
                json.dumps({"updated": self._now().isoformat(), "open": self._open},
                           default=str, indent=1),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            logger.exception("intraday execution: could not persist open positions")

    # -- capacity ------------------------------------------------------------

    def _has_capacity(self, setup_id: str) -> tuple[bool, str]:
        if setup_id in self._open:
            return False, "already_open"
        if len(self._open) >= int(self._policy.max_concurrent_positions):
            return False, "max_concurrent_positions"
        if self._opened_this_session >= int(self._policy.max_new_positions_per_session):
            return False, "max_new_positions_per_session"
        return True, ""

    # -- entry ---------------------------------------------------------------

    def on_entry(self, setup: SetupRecord, *, spot: float | None = None,
                 atr: float | None = None) -> dict[str, Any] | None:
        """A setup just went RUNNING. Buy the contract. Never raises."""
        try:
            with self._lock:
                return self._on_entry(setup, spot=spot, atr=atr)
        except Exception:  # noqa: BLE001 - execution must never break detection
            logger.exception("intraday execution: entry failed for %s", getattr(setup, "setup_id", "?"))
            return None

    def _on_entry(self, setup, *, spot, atr):
        setup_id = str(setup.setup_id)
        ok, why = self._has_capacity(setup_id)
        if not ok:
            logger.info("intraday execution: skipped %s (%s)", setup_id, why)
            setup.metadata["execution_skip"] = why
            return None

        price = _f(spot) or _f(getattr(setup, "entry_price", None)) or _f(getattr(setup, "spot", None))
        if price is None or price <= 0:
            setup.metadata["execution_skip"] = "no_reference_price"
            return None

        direction = str(getattr(setup.direction, "value", setup.direction)).lower()
        want = self._policy.option_type
        cp = ("call" if direction == "long" else "put") if want == "auto" else want

        min_dte = self._min_dte_now()
        contract, reason = self._select(
            self._client, setup.ticker, price, option_type=cp,
            min_dte=min_dte, max_dte=int(self._policy.max_dte),
            allow_0dte=bool(self._policy.allow_0dte) and min_dte == 0,
        )
        if not contract:
            setup.metadata["execution_skip"] = f"no_contract({reason})"
            logger.info("intraday execution: no contract for %s (%s)", setup.ticker, reason)
            return None

        occ = str(contract.get("occ") or contract.get("symbol"))
        premium = _f(contract.get("limit")) or _f(contract.get("mid")) or _f(contract.get("ask"))
        if premium is None or premium <= 0:
            setup.metadata["execution_skip"] = "no_priced_contract"
            return None
        qty = int(max(1, min(int(self._policy.max_contracts),
                             self._policy.target_notional // (premium * OPTION_MULTIPLIER))))

        resp = self._client.submit_option_order(
            symbol=occ, qty=qty, side="buy", order_type="limit",
            time_in_force="day", limit_price=premium,
        )
        fill = _f((resp or {}).get("filled_avg_price"))
        rec = {
            "setup_id": setup_id,
            "ticker": setup.ticker,
            "direction": direction,
            "occ": occ,
            "option_type": cp,
            "qty": qty,
            "limit_price": premium,
            "entry_order_id": str((resp or {}).get("id", "")) or None,
            "entry_submitted_at": (resp or {}).get("submitted_at") or self._now().isoformat(),
            "entry_filled_at": (resp or {}).get("filled_at"),
            "entry_fill_price": fill,
            "entry_filled_qty": _f((resp or {}).get("filled_qty")),
            # The underlying leg, captured at entry so the option result can be
            # compared against the move it was a bet on.
            "u_entry": price,
            "u_atr": _f(atr) or _f(getattr(setup, "risk_points", None)),
            "expiry": contract.get("expiry"),
            "dte_at_entry": self._dte_of(contract.get("expiry")),
            "strike": contract.get("strike"),
            "setup_type": str(getattr(setup.setup_type, "value", setup.setup_type)),
        }
        self._open[setup_id] = rec
        self._opened_this_session += 1
        setup.metadata["execution_occ"] = occ
        setup.metadata["execution_entry_order_id"] = rec["entry_order_id"]
        self._persist()
        logger.info("intraday execution: BUY %s x%d for %s @ %.2f", occ, qty, setup_id, premium)
        return rec

    def _dte_of(self, expiry) -> int | None:
        try:
            return (date.fromisoformat(str(expiry)) - self._now().astimezone(ET).date()).days
        except (TypeError, ValueError):
            return None

    def _min_dte_now(self, now_et: datetime | None = None) -> int:
        """0 early in the session, 1 after the roll cutoff.

        Buying a same-day contract in the last hours of the session is buying
        the part of the move that has already happened plus the part theta is
        about to take. After the cutoff the next session is the cheaper bet, and
        it is also the one this engine can still manage tomorrow.
        """
        if not self._policy.allow_0dte:
            return max(1, int(self._policy.min_dte))
        if self._dte_cutoff is None:
            return int(self._policy.min_dte)
        now_et = now_et or self._now().astimezone(ET)
        return 0 if now_et.timetz().replace(tzinfo=None) < self._dte_cutoff else 1

    # -- same-day expiry flatten ---------------------------------------------

    def maybe_flatten_expiring(self, now_et: datetime | None = None) -> list[dict[str, Any]]:
        """Close every position expiring today once the cutoff passes.

        This is the whole reason 0DTE is safe to hold here. An expiring long
        option left open past the close is assignment/exercise risk on a
        position this engine never intended to own overnight, and the SPY
        daytrader already learned that a stuck close can let one ride into
        expiry — so this runs on a clock, not on a setup event.

        Idempotent: a position closed here leaves the open book, so a later call
        in the same session does nothing.
        """
        try:
            with self._lock:
                return self._maybe_flatten_expiring(now_et)
        except Exception:  # noqa: BLE001
            logger.exception("intraday execution: expiring flatten failed")
            return []

    def _maybe_flatten_expiring(self, now_et):
        if self._expiring_exit is None:
            return []
        now_et = now_et or self._now().astimezone(ET)
        if now_et.timetz().replace(tzinfo=None) < self._expiring_exit:
            return []
        today = now_et.date()
        closed = []
        for setup_id, pos in list(self._open.items()):
            expiry = pos.get("expiry")
            try:
                exp = date.fromisoformat(str(expiry)) if expiry else None
            except ValueError:
                exp = None
            if exp is None or exp > today:
                continue
            rec = self._close_position(setup_id, pos, exit_reason="expiring_flatten",
                                       u_exit=None, urgent=True)
            if rec is not None:
                closed.append(rec)
        if closed:
            logger.info("intraday execution: flattened %d expiring position(s)", len(closed))
        return closed

    # -- exit ----------------------------------------------------------------

    def on_exit(self, setup: SetupRecord, *, exit_reason: str | None = None,
                spot: float | None = None) -> dict[str, Any] | None:
        """A setup reached a terminal state. Sell, and write the ledger row."""
        try:
            with self._lock:
                return self._on_exit(setup, exit_reason=exit_reason, spot=spot)
        except Exception:  # noqa: BLE001
            logger.exception("intraday execution: exit failed for %s", getattr(setup, "setup_id", "?"))
            return None

    def _on_exit(self, setup, *, exit_reason, spot):
        setup_id = str(setup.setup_id)
        pos = self._open.get(setup_id)
        if pos is None:
            return None            # nothing was ever bought for this setup
        reason = str(exit_reason or setup.metadata.get("exit_reason") or "closed")
        u_exit = _f(spot) if spot is not None else _f(getattr(setup, "spot", None))
        return self._close_position(
            setup_id, pos, exit_reason=reason, u_exit=u_exit,
            modelled_entry=_f(getattr(setup, "entry_price", None)),
            modelled_exit=_f(setup.metadata.get("exit_price")),
        )

    def _close_position(self, setup_id, pos, *, exit_reason, u_exit,
                        modelled_entry=None, modelled_exit=None, urgent=False):
        """Sell the contract and write one ledger row. Shared by both exit paths."""
        occ, qty = pos["occ"], int(pos["qty"])
        try:
            resp = self._client.submit_option_order(
                symbol=occ, qty=qty, side="sell",
                # An expiring contract must actually get out, and a passive limit
                # into a thin book is how one rides into expiry. The SPY
                # daytrader hit exactly that; market is the right order here.
                order_type="market", time_in_force="day",
            ) or {}
        except Exception as exc:  # noqa: BLE001
            # The position is still held. Keep it open so the next terminal
            # event (or a restart) retries rather than losing track of it — an
            # unowned contract is how a sibling module ends up liquidating it.
            logger.warning("intraday execution: exit submit failed for %s: %s", occ, exc)
            pos["last_exit_error"] = str(exc)[:200]
            self._persist()
            return None

        exit_fill = _f(resp.get("filled_avg_price"))
        entry_px = _f(pos.get("entry_fill_price")) or _f(pos.get("limit_price"))
        realized = (round((exit_fill - entry_px) * OPTION_MULTIPLIER * qty, 2)
                    if (exit_fill is not None and entry_px) else None)

        record = closed_trade_record(
            module=LEDGER_MODULE,
            bar=self._now().isoformat(),
            ticker=pos["ticker"],
            order_symbol=occ,
            route="option",
            qty=qty,
            exit_reason=exit_reason,
            entry_avg_price=entry_px,
            exit_fill_price=exit_fill,
            realized_pnl=realized,
            entry_state=pos,
            order_id=str(resp.get("id", "")) or None,
            exit_submitted_at=resp.get("submitted_at"),
        )
        # Intraday-specific context, and the modelled counterpart. The gap
        # between the modelled prices and the realized option result is the
        # execution cost this engine could never see.
        record["setup_id"] = setup_id
        record["setup_type"] = pos.get("setup_type")
        record["direction"] = pos.get("direction")
        record["option_type"] = pos.get("option_type")
        record["expiry"] = pos.get("expiry")
        record["dte_at_entry"] = pos.get("dte_at_entry")
        record["u_exit"] = u_exit
        record["modelled_entry_price"] = modelled_entry
        record["modelled_exit_price"] = modelled_exit
        record["urgent_exit"] = bool(urgent)
        append_closed_trade(LEDGER_MODULE, record, self._ledger_root)

        self._open.pop(setup_id, None)
        self._persist()
        logger.info("intraday execution: SELL %s x%d for %s reason=%s pnl=%s",
                    occ, qty, setup_id, exit_reason, realized)
        return record


    # -- introspection -------------------------------------------------------

    @property
    def open_positions(self) -> dict[str, dict[str, Any]]:
        return dict(self._open)

    def reset_session(self) -> None:
        """Clear the per-session new-position counter (open positions persist)."""
        self._opened_this_session = 0
