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

from core.live_4h_exec import (
    append_closed_trade,
    closed_trade_record,
    poll_exit_fill_price,
    submit_option_exit_with_ladder,
)
from strategies.intraday_structure.config import ExecutionPolicy
from strategies.intraday_structure.models import SetupRecord

logger = logging.getLogger(__name__)

LEDGER_MODULE = "intraday_structure"
OPTION_MULTIPLIER = 100.0
ET = ZoneInfo("America/New_York")

# Retry discipline for a failing exit. Before this the flatten was driven by the
# runner clock with no backoff and no cap, and `_close_position` kept the
# position on failure "so the next event retries" — which on 2026-09-01 meant
# ~190 rejected submissions a minute for eight straight hours against four
# contracts the account no longer held. Attempts are counted per position and
# spaced; past the cap the position is quarantined for a human rather than
# retried forever.
EXIT_RETRY_BASE_SECONDS = 30.0
EXIT_RETRY_MAX_SECONDS = 900.0
EXIT_MAX_ATTEMPTS = 8


def _hhmm(text: str | None) -> time | None:
    text = str(text or "").strip()
    if not text:
        return None
    hh, _, mm = text.partition(":")
    return time(int(hh), int(mm or 0))


def _cancel_rejection_means_filled(exc: Exception) -> bool:
    """True when a rejected cancel is the broker saying the order already filled.

    Alpaca answers `422 {"code":42210000,"message":"order is already in
    \\"filled\\" state"}`. Matching on the text is unlovely, but the alternative
    is discarding the only positive evidence we get in this race.
    """
    text = str(exc).lower()
    return "filled" in text and ("already" in text or "state" in text)


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
        """Reload open positions so a restart does not orphan live contracts.

        The reload is then checked against the broker. A restart used to be no
        help at all for a stuck book: `maybe_flatten_expiring` fires for any
        `expiry <= today`, and an expiry in the *past* still satisfies that, so
        the 2026-09-01 contracts would have resumed their retry loop on every
        subsequent startup until someone edited the state file by hand.
        """
        try:
            if self._state_path.exists():
                raw = json.loads(self._state_path.read_text(encoding="utf-8"))
                self._open = dict(raw.get("open") or {})
                logger.info("intraday execution restored %d open position(s)", len(self._open))
        except Exception:  # noqa: BLE001
            logger.exception("intraday execution: could not restore open positions")
            self._open = {}
        if not self._open:
            return
        try:
            held = self._held_contracts()
            if held is not None:
                dropped = self._drop_unheld(held)
                if dropped:
                    logger.warning(
                        "intraday execution: startup reconcile released %d stale claim(s): %s",
                        len(dropped), ", ".join(dropped),
                    )
        except Exception:  # noqa: BLE001 - never block startup on a broker call
            logger.exception("intraday execution: startup broker reconcile failed")

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

    # -- broker truth --------------------------------------------------------

    def _held_contracts(self) -> dict[str, float] | None:
        """Contract -> qty the account actually holds, or None if unknowable.

        None and {} mean different things and must not be conflated: None is "the
        broker did not answer", which is never grounds for dropping a claim, and
        {} is "the broker answered and you hold nothing".
        """
        try:
            positions = self._client.get_positions()
        except Exception as exc:  # noqa: BLE001 - a broker outage is not a fill
            logger.warning("intraday execution: could not read broker positions (%s)", exc)
            return None
        if not isinstance(positions, list):
            return None
        held: dict[str, float] = {}
        for position in positions:
            if not isinstance(position, dict):
                continue
            if str(position.get("asset_class") or "") != "us_option":
                continue
            symbol = str(position.get("symbol") or "").strip().upper()
            qty = _f(position.get("qty"))
            if symbol and qty:
                held[symbol] = qty
        return held

    def _drop_unheld(self, held: dict[str, float]) -> list[str]:
        """Release claims on contracts the account does not hold. Returns them.

        This is the loop-killer. Whatever the reason a contract left the account
        — an entry that never filled, a sibling module's reconcile adopting and
        liquidating it, a manual close — continuing to submit sells for it is
        never right, and the broker reads those sells as *opening* a naked
        short, which is why they came back 403 "not eligible to trade uncovered
        option contracts" rather than as a plain "no position" error.
        """
        dropped = []
        for setup_id, pos in list(self._open.items()):
            occ = str(pos.get("occ") or "").strip().upper()
            if not occ or held.get(occ):
                continue
            dropped.append(occ)
            self._open.pop(setup_id, None)
            logger.warning(
                "intraday execution: releasing claim on %s (setup %s) — the account "
                "does not hold it. entry_filled_qty=%s last_exit_error=%s",
                occ, setup_id, pos.get("entry_filled_qty"),
                str(pos.get("last_exit_error"))[:120],
            )
        if dropped:
            self._persist()
        return dropped

    def reconcile_with_broker(self) -> list[str]:
        """Drop claims on contracts the account no longer holds. Never raises."""
        try:
            with self._lock:
                held = self._held_contracts()
                if held is None:
                    return []
                return self._drop_unheld(held)
        except Exception:  # noqa: BLE001
            logger.exception("intraday execution: broker reconcile failed")
            return []

    # -- retry discipline ----------------------------------------------------

    @staticmethod
    def _exit_backoff_seconds(attempts: int) -> float:
        return min(EXIT_RETRY_BASE_SECONDS * (2 ** max(0, attempts - 1)),
                   EXIT_RETRY_MAX_SECONDS)

    def _exit_retry_blocked(self, pos: dict[str, Any]) -> str | None:
        """Why this position must not be re-submitted right now, if so."""
        attempts = int(pos.get("exit_attempts") or 0)
        if attempts >= EXIT_MAX_ATTEMPTS:
            return "quarantined"
        last = pos.get("last_exit_attempt_at")
        if not last:
            return None
        try:
            since = (self._now() - datetime.fromisoformat(str(last))).total_seconds()
        except (TypeError, ValueError):
            return None
        return "backoff" if since < self._exit_backoff_seconds(attempts) else None

    def _record_exit_failure(self, pos: dict[str, Any], occ: str, exc: Exception) -> None:
        attempts = int(pos.get("exit_attempts") or 0) + 1
        pos["exit_attempts"] = attempts
        pos["last_exit_attempt_at"] = self._now().isoformat()
        pos["last_exit_error"] = str(exc)[:200]
        if attempts >= EXIT_MAX_ATTEMPTS:
            pos["quarantined"] = True
            logger.error(
                "intraday execution: QUARANTINED %s after %d failed exit attempts — "
                "no further orders will be submitted for it. Last error: %s. "
                "This needs a human; decide via core.startup_queue.",
                occ, attempts, str(exc)[:200],
            )
        else:
            logger.warning(
                "intraday execution: exit submit failed for %s (attempt %d/%d, "
                "next in %.0fs): %s",
                occ, attempts, EXIT_MAX_ATTEMPTS,
                self._exit_backoff_seconds(attempts), str(exc)[:200],
            )

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
        # A submit response is not a fill. Reading `filled_avg_price` straight
        # off it left every position in this book recording
        # `entry_filled_qty: 0.0` even when the buy had filled seconds later, so
        # the module could not tell a filled position from an unfilled one, and
        # every closed row it ever wrote had `realized_pnl: null`.
        fill, filled_qty = self._settle_entry(resp, occ)
        if fill is None:
            # Nothing was bought (or nothing we can prove was bought). Claiming
            # it anyway is what produces sells the broker reads as naked shorts.
            raced = self._cancel_quietly(resp, occ, qty)
            if raced > 0:
                # It filled as we cancelled. Own it — an unclaimed filled
                # position is an orphan nothing stops, sizes or exits.
                filled_qty = raced
                fill = _f((resp or {}).get("filled_avg_price")) or premium
                logger.warning(
                    "intraday execution: %s filled %g against the cancel — claiming it",
                    occ, raced,
                )
            else:
                setup.metadata["execution_skip"] = "entry_unfilled"
                logger.info("intraday execution: entry for %s did not fill; not claiming it", occ)
                return None
        rec = {
            "setup_id": setup_id,
            "ticker": setup.ticker,
            "direction": direction,
            "occ": occ,
            "option_type": cp,
            # The size we actually own, not the size we asked for: a partial
            # fill sold at the requested qty is the same naked-short rejection
            # by another route.
            "qty": int(filled_qty),
            "requested_qty": qty,
            "limit_price": premium,
            "entry_order_id": str((resp or {}).get("id", "")) or None,
            "entry_submitted_at": (resp or {}).get("submitted_at") or self._now().isoformat(),
            "entry_filled_at": (resp or {}).get("filled_at"),
            "entry_fill_price": fill,
            "entry_filled_qty": filled_qty,
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
        logger.info("intraday execution: BUY %s x%d filled @ %.2f for %s (limit %.2f)",
                    occ, int(filled_qty), fill, setup_id, premium)
        return rec

    def _settle_entry(self, resp, occ: str) -> tuple[float | None, float]:
        """(fill price, filled qty) once the broker confirms, else (None, 0).

        Polls briefly rather than trusting the submit response. Paper fills
        settle in well under a second; the timeout only bounds the pathological
        case so the detection loop never stalls behind a broker call.
        """
        import time as _time

        resp = resp or {}
        fill = _f(resp.get("filled_avg_price"))
        qty = _f(resp.get("filled_qty")) or 0.0
        if fill and qty:
            return fill, qty
        order_id = str(resp.get("id", "")).strip()
        if not order_id or not hasattr(self._client, "get_order"):
            return (fill, qty) if (fill and qty) else (None, 0.0)
        deadline = _time.monotonic() + float(getattr(self._policy, "entry_fill_timeout_s", 3.0))
        while _time.monotonic() < deadline:
            _time.sleep(0.25)
            try:
                cur = self._client.get_order(order_id) or {}
            except Exception:  # noqa: BLE001 - a poll failure is not a fill
                continue
            fill = _f(cur.get("filled_avg_price"))
            qty = _f(cur.get("filled_qty")) or 0.0
            if fill and qty:
                return fill, qty
            if str(cur.get("status", "")).lower() in {
                "canceled", "cancelled", "rejected", "expired", "done_for_day",
            }:
                break
        return None, 0.0

    def _cancel_quietly(self, resp, occ: str, requested_qty: float) -> float:
        """Cancel an entry that never filled. Returns any quantity that filled anyway.

        Cancellation races the book: an order can fill in the moment between the
        poll giving up and the cancel landing. Re-reading the order afterwards is
        what stops that becoming an unclaimed position — which is precisely the
        orphan this module was creating by other means.

        Every branch that returns 0.0 is a decision to walk away from a position
        the account may hold, so none of them may be reached on a *failure* to
        find out. On 2026-09-02 the cancel of SPY260902C00765000 came back
        `422 order is already in "filled" state` — the broker stating plainly
        that four contracts had been bought — and that answer was thrown away in
        favour of a follow-up read whose result was 0. The module logged "did
        not fill; not claiming it", `live_risk_pass` flagged the contract as an
        orphan three times, and fourteen minutes later the SPY daytrader's
        broker reconcile adopted all four and liquidated them at 1.45/1.46
        against a 2.04 basis, for -$235.
        """
        order_id = str((resp or {}).get("id", "")).strip()
        if not order_id or not hasattr(self._client, "cancel_order"):
            return 0.0
        # A rejected cancel is evidence about the order, not noise. "Already
        # filled" is the broker telling us we own it.
        filled_per_broker = False
        try:
            self._client.cancel_order(order_id)
        except Exception as exc:  # noqa: BLE001
            filled_per_broker = _cancel_rejection_means_filled(exc)
            logger.info(
                "intraday execution: could not cancel unfilled entry %s (%s)%s",
                occ, exc, " — the broker says it FILLED" if filled_per_broker else "",
            )

        qty = 0.0
        if hasattr(self._client, "get_order"):
            try:
                qty = _f((self._client.get_order(order_id) or {}).get("filled_qty")) or 0.0
            except Exception:  # noqa: BLE001 - a failed read is not proof of nothing
                qty = 0.0
        if qty > 0 or not filled_per_broker:
            return qty

        # The broker said filled and the order read did not confirm a size. Ask
        # the account. Clamped to what we asked for, because this contract may
        # also be held by a sibling module and only our own order is ours.
        held = self._held_contracts()
        if held is not None and held.get(occ):
            claimed = min(float(held[occ]), float(requested_qty))
            logger.warning(
                "intraday execution: %s not confirmed by the order read, but the "
                "account holds %g — claiming %g",
                occ, held[occ], claimed,
            )
            return claimed
        if held is None:
            # Unknowable, and the broker already said it filled. Claiming the
            # requested size keeps the position managed; over-claiming is
            # corrected by `_close_position`'s broker check on the way out,
            # whereas under-claiming leaves an orphan nothing exits.
            logger.warning(
                "intraday execution: %s reported FILLED by the broker but the "
                "account could not be read — claiming the requested %g",
                occ, requested_qty,
            )
            return float(requested_qty)
        return 0.0

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
        # Size from OUR OWN FILL, recorded off our own entry order id, never
        # from the broker's position. The account is shared and Alpaca nets
        # option positions by symbol, so the broker's size is the sum across
        # every module holding this contract; only our order tells us which part
        # is ours.
        occ = pos["occ"]
        qty = int(_f(pos.get("entry_filled_qty")) or pos.get("qty") or 0)
        if qty <= 0:
            logger.warning("intraday execution: %s has no recorded fill quantity — "
                           "releasing the claim rather than guessing a size", occ)
            self._open.pop(setup_id, None)
            self._persist()
            return None

        blocked = self._exit_retry_blocked(pos)
        if blocked:
            return None

        # Do not submit a sell for something the account does not hold. Every
        # rejection in the 2026-09-01 loop was this: the contract was gone, so
        # Alpaca read the sell as opening a naked short. Checking costs one
        # positions call per attempt, which the backoff above already bounds.
        held = self._held_contracts()
        if held is not None and not held.get(occ.upper()):
            logger.warning(
                "intraday execution: %s (setup %s) is no longer held at the broker — "
                "releasing the claim instead of submitting an exit",
                occ, setup_id,
            )
            self._open.pop(setup_id, None)
            self._persist()
            return None
        if held is not None:
            # The broker total is a CEILING, never the size. It can only ever be
            # smaller than our own fill if part of our position has gone (expiry,
            # assignment, a manual close); submitting more than exists is a
            # guaranteed rejection, so clamp. It being LARGER just means a
            # sibling module is long the same contract, which is fine and
            # deliberately not a reason to sell more.
            broker_qty = int(abs(held.get(occ.upper(), qty)))
            if broker_qty < qty:
                logger.warning(
                    "intraday execution: %s our fill was %d but the account holds only "
                    "%d in total — selling %d", occ, qty, broker_qty, broker_qty,
                )
                qty = broker_qty
                pos["entry_filled_qty"] = float(broker_qty)
                pos["qty"] = broker_qty
            elif broker_qty > qty:
                logger.info(
                    "intraday execution: %s account holds %d against our fill of %d "
                    "(a sibling module is long the same contract) — selling only ours",
                    occ, broker_qty, qty,
                )

        try:
            # The shared 4H exit policy: market first (best fill when there IS a
            # book), then a descending limit ladder. A bare market order is
            # rejected outright outside 09:30-16:00 ET for options
            # ("options market orders are only allowed during market hours"),
            # which is what turned every after-hours retry into a hard failure.
            resp = submit_option_exit_with_ladder(
                self._client, symbol=occ, qty=qty,
                reason=exit_reason, full_exit=True,
            ) or {}
        except Exception as exc:  # noqa: BLE001
            self._record_exit_failure(pos, occ, exc)
            self._persist()
            return None

        exit_fill = _f(resp.get("filled_avg_price")) or poll_exit_fill_price(self._client, resp)
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
