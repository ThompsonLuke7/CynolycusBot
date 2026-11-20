"""Schwab trade history for the personal page: raw transactions -> fills -> round-trip episodes.

Read-only. Schwab is called through two GET endpoints, ``get_account_numbers`` and
``get_transactions`` (types TRADE and RECEIVE_AND_DELIVER), in 364-day windows. The
endpoint serves history back to the account's first trade (verified 2026-11-16: 2019-11
onward), despite schwab-py's docstring saying 60 days.

Storage (Data/personal/, gitignored):
  transactions.jsonl  append-only raw transactions, account number masked to its last 3
                      digits. A record whose content changes on a refetch is appended
                      again; the newest copy of an activityId wins. Nothing is rewritten.
  symbol_map.json     optional {"CUSIP or retired symbol": "TICKER"} used only to find
                      chart bars (e.g. "00165C104": "AMC"). Episode keys stay raw.

Fill rules (each documented because each changes P&L or the episode boundaries):
  * Cash = the security leg's ``cost`` (signed, multiplier included); fees = the record's
    CURRENCY legs. Their sum equals Schwab's ``netAmount`` for single-security trades.
  * Option expirations ("Removed/Removal ... expiration") close at price 0. Their time is
    the expiry date at 16:00 ET, not the Monday Schwab posts them.
  * Other RECEIVE_AND_DELIVER security legs are corporate actions, applied per account and
    ET date as a group: same-symbol legs net (a split adds shares at no cash); one symbol
    closing while another opens moves the open episode to the new symbol (a reverse split
    into a new CUSIP, a SPAC merger). Anything else is applied leg by leg and noted.
  * Cash-alternative sweeps (CASH_EQUIVALENT) are not trades and are ignored.
  * Pre-2023 TD Ameritrade records have no positionEffect; opening/closing is inferred from
    the running quantity. Where Schwab does report it, the inference is checked against it.

An episode runs from flat to flat in one position key (an option contract, or a stock/ETF
symbol). Its P&L is the sum of its fills' net cash (fees included): realized, exact, and
independent of any mark. Open episodes are marked with the current Schwab market value.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from core.personal_portfolio.data import ET, personal_dir
from core.personal_portfolio.holdings import _get_json, mask_account_number, parse_occ_symbol

logger = logging.getLogger(__name__)
STORE_NAME = "transactions.jsonl"
WINDOW_DAYS = 364
BACKFILL_MAX_WINDOWS = 15        # 15 years
BACKFILL_EMPTY_STOP = 3          # consecutive empty windows that end a backfill
INCREMENTAL_OVERLAP_DAYS = 45    # late-posted records (expirations, corrections) land inside this
QTY_EPS = 1e-9
LEGACY_OPT_RE = re.compile(r"^(?P<root>[A-Z0-9.]{1,6})_(?P<mm>\d{2})(?P<dd>\d{2})(?P<yy>\d{2})"
                           r"(?P<cp>[CP])(?P<strike>\d+(?:\.\d+)?)$")
ADJUSTED_ROOT_RE = re.compile(r"^([A-Z][A-Z.]*)\d$")
_KIND = {"OPTION": "option", "EQUITY": "equity", "COLLECTIVE_INVESTMENT": "etf"}
_EXPIRY_CLOSE = time(16, 0)


# ---------- fetch ----------
def _windows(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    out, we = [], end
    while we > start:
        ws = max(start, we - timedelta(days=WINDOW_DAYS))
        out.append((ws, we))
        we = ws
    return out


def fetch_transactions(client, *, start: datetime | None, end: datetime) -> list[dict]:
    """Raw TRADE + RECEIVE_AND_DELIVER records for every linked account, account numbers masked.

    ``start=None`` backfills: windows walk back from ``end`` until BACKFILL_EMPTY_STOP
    consecutive windows return nothing (or BACKFILL_MAX_WINDOWS is reached).
    """
    types = [client.Transactions.TransactionType.TRADE, client.Transactions.TransactionType.RECEIVE_AND_DELIVER]
    accounts = _get_json(client.get_account_numbers(), "get_account_numbers")
    if not isinstance(accounts, list):
        raise ValueError("Schwab account numbers response is not a list")
    out: list[dict] = []
    for acct in accounts:
        masked, acct_hash = mask_account_number(acct.get("accountNumber")), acct.get("hashValue")
        if not acct_hash:
            raise ValueError(f"Schwab account {masked} has no hashValue")
        if start is None:
            spans, empty, we = [], 0, end
            for _ in range(BACKFILL_MAX_WINDOWS):
                spans.append((we - timedelta(days=WINDOW_DAYS), we))
                we -= timedelta(days=WINDOW_DAYS)
        else:
            spans, empty = _windows(start, end), 0
        for ws, we in spans:
            rows = _get_json(client.get_transactions(acct_hash, start_date=ws, end_date=we, transaction_types=types),
                             "get_transactions")
            if not isinstance(rows, list):
                raise ValueError(f"Schwab transactions response is not a list: {type(rows).__name__}")
            for r in rows:
                out.append({**r, "accountNumber": masked})
            empty = 0 if rows else empty + 1
            if start is None and empty >= BACKFILL_EMPTY_STOP:
                break
    return out


# ---------- raw store ----------
def store_path(base: Path | None = None) -> Path:
    return personal_dir(base) / STORE_NAME


def _canon(rec: dict) -> str:
    return json.dumps(rec, sort_keys=True, separators=(",", ":"), default=str)


def load_raw(base: Path | None = None) -> list[dict]:
    """Stored records, newest copy per activityId, in file order."""
    path = store_path(base)
    if not path.exists():
        return []
    latest: dict[Any, dict] = {}
    with path.open() as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                latest[rec.get("activityId")] = rec
    return list(latest.values())


def append_raw(records: Iterable[dict], base: Path | None = None) -> int:
    """Append records that are new or changed. Returns how many lines were written."""
    path = store_path(base)
    have = {r.get("activityId"): _canon(r) for r in load_raw(base)}
    new = []
    for rec in records:
        if "activityId" not in rec:
            raise ValueError("Schwab transaction without activityId")
        c = _canon(rec)
        if have.get(rec["activityId"]) != c:
            new.append(c)
            have[rec["activityId"]] = c
    if new:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write("".join(line + "\n" for line in new))
    return len(new)


def sync_transactions(client, now: datetime, base: Path | None = None, full: bool = False) -> dict:
    """Backfill on first use (or ``full``), else refetch from the newest stored record minus an overlap."""
    stored = load_raw(base)
    end = now.astimezone(timezone.utc)
    if full or not stored:
        start = None
    else:
        newest = max(_parse_time(r["time"]) for r in stored if r.get("time"))
        start = newest - timedelta(days=INCREMENTAL_OVERLAP_DAYS)
    fetched = fetch_transactions(client, start=start, end=end)
    written = append_raw(fetched, base)
    return {"mode": "backfill" if start is None else "incremental", "fetched": len(fetched),
            "written": written, "stored": len(load_raw(base)), "at": end.isoformat()}


def load_symbol_map(base: Path | None = None) -> dict[str, str]:
    path = personal_dir(base) / "symbol_map.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected an object of symbol -> ticker")
    return {str(k).strip().upper(): str(v).strip().upper() for k, v in raw.items()}


# ---------- normalization ----------
def _parse_time(value: str) -> datetime:
    text = str(value)
    if re.search(r"[+-]\d{4}$", text):
        text = text[:-2] + ":" + text[-2:]
    out = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if out.tzinfo is None:
        raise ValueError(f"transaction time without a zone: {value!r}")
    return out.astimezone(timezone.utc)


def _f(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def parse_option_symbol(symbol: str) -> dict | None:
    """OCC ("SPY   260918C00450000") or TD Ameritrade legacy ("SPY_091826C450") -> contract fields."""
    occ = parse_occ_symbol(symbol)
    if occ:
        return occ
    m = LEGACY_OPT_RE.match(str(symbol or "").strip())
    if not m:
        return None
    expiry = date(2000 + int(m["yy"]), int(m["mm"]), int(m["dd"]))
    return {"root": m["root"], "expiry": expiry.isoformat(),
            "put_call": "CALL" if m["cp"] == "C" else "PUT", "strike": float(m["strike"])}


def option_key(underlying: str, expiry: str, put_call: str, strike: float) -> str:
    return f"{underlying} {expiry} {put_call[0]} {strike:g}"


def _option_fields(inst: dict) -> dict:
    parsed = parse_option_symbol(inst.get("symbol", "")) or {}
    und = str(inst.get("underlyingSymbol") or parsed.get("root") or "").strip().upper()
    expiry = parsed.get("expiry")
    if inst.get("expirationDate"):
        expiry = _parse_time(inst["expirationDate"]).astimezone(ET).date().isoformat()
    pc = str(inst.get("putCall") or parsed.get("put_call") or "").upper()
    strike = _f(inst.get("strikePrice"))
    strike = strike if strike is not None else parsed.get("strike")
    if not (und and expiry and pc in ("CALL", "PUT") and strike is not None):
        raise ValueError(f"cannot identify option contract {inst.get('symbol')!r}")
    # optionDeliverables is NOT point-in-time: Schwab rewrites it to today's terms (an expired 2025 KVUE
    # call lists KMB shares after the later merger). An adjusted option at trade time has an OCC root with
    # a digit suffix (HIMZ1); its bars come from the root without the digit.
    adj = ADJUSTED_ROOT_RE.match(und)
    return {"underlying": und, "expiry": expiry, "put_call": pc, "strike": float(strike),
            "multiplier": _f(inst.get("optionPremiumMultiplier")) or 100.0, "standard_deliverable": adj is None,
            "bars_symbol": adj[1] if adj else und, "key": option_key(und, expiry, pc, float(strike))}


def _is_expiration(desc: str) -> bool:
    return "EXPIRATION" in desc.upper()


def normalize_fills(raw: list[dict]) -> tuple[list[dict], list[dict]]:
    """(fills, issues). One fill per security leg; issues list records that could not be used."""
    fills, issues = [], []
    for rec in raw:
        if rec.get("status") not in (None, "VALID"):
            issues.append({"activity_id": rec.get("activityId"), "issue": f"status {rec.get('status')}"})
            continue
        typ = rec.get("type")
        items = rec.get("transferItems") or []
        sec = [t for t in items if (t.get("instrument") or {}).get("assetType") in _KIND]
        fees = sum(_f(t.get("cost")) or 0.0 for t in items if (t.get("instrument") or {}).get("assetType") == "CURRENCY")
        if typ == "TRADE" and not sec:
            issues.append({"activity_id": rec.get("activityId"), "time": rec.get("time"),
                           "net_amount": _f(rec.get("netAmount")),
                           "issue": "trade record without a security leg (cash not attributed to any episode)"})
            continue
        if typ not in ("TRADE", "RECEIVE_AND_DELIVER") or not sec:
            continue
        t_rec = _parse_time(rec["time"])
        desc = str(rec.get("description") or "")
        weights = [abs(_f(t.get("cost")) or 0.0) for t in sec]
        weights = [w / sum(weights) for w in weights] if sum(weights) else [1 / len(sec)] * len(sec)
        for t, share in zip(sec, weights):
            inst = t["instrument"]
            kind = _KIND[inst["assetType"]]
            qty = _f(t.get("amount"))
            if qty is None or abs(qty) < QTY_EPS:
                continue
            fill = {"activity_id": rec.get("activityId"), "order_id": rec.get("orderId"),
                    "account": rec.get("accountNumber"), "kind": kind, "symbol": str(inst.get("symbol") or "").strip(),
                    "qty": qty, "price": _f(t.get("price")), "cash": _f(t.get("cost")) or 0.0,
                    "fees": fees * share,
                    "effect": t.get("positionEffect"), "time": t_rec, "description": desc or inst.get("description")}
            if kind == "option":
                fill.update(_option_fields(inst))
            else:
                fill.update(underlying=fill["symbol"], key=fill["symbol"], multiplier=1.0)
            if typ == "TRADE":
                fill["source"] = "trade"
            elif kind == "option" and _is_expiration(desc):
                fill.update(source="expiration", price=0.0, cash=0.0,
                            time=datetime.combine(date.fromisoformat(fill["expiry"]), _EXPIRY_CLOSE, ET)
                            .astimezone(timezone.utc))
            else:
                fill["source"] = "corp_action"
            fill["time_known"] = fill["time"].astimezone(ET).time() != time(0, 0)
            fills.append(fill)
    fills.sort(key=lambda f: (f["time"], str(f["activity_id"])))
    return fills, issues


# ---------- episodes ----------
def _episode_id(account: str, key: str, first_activity: Any) -> str:
    return hashlib.sha1(f"{account}|{key}|{first_activity}".encode()).hexdigest()[:12]


class _Ep:
    def __init__(self, fill: dict):
        self.account, self.key, self.kind = fill["account"], fill["key"], fill["kind"]
        self.first = fill
        self.qty = 0.0
        self.max_abs = 0.0
        self.fills: list[dict] = []
        self.symbols = [fill["symbol"]]
        self.notes: list[str] = []
        self.incomplete = False

    def add(self, fill: dict, role: str) -> None:
        self.fills.append({**fill, "role": role})


def _sign(x: float) -> int:
    return (x > QTY_EPS) - (x < -QTY_EPS)


def _apply_fill(open_eps: dict, closed: list, fill: dict, checks: dict) -> None:
    k = (fill["account"], fill["key"])
    ep = open_eps.get(k)
    if ep is None:
        ep = open_eps[k] = _Ep(fill)
        if fill.get("effect") == "CLOSING":
            ep.incomplete = True
            ep.notes.append("first fill in history closes a position: its opening predates the record")
    old, new = ep.qty, ep.qty + fill["qty"]
    if _sign(old) and _sign(new) and _sign(new) != _sign(old):
        part = -old / fill["qty"]                        # share of the fill that flattens the position
        a = {**fill, "qty": -old, "cash": fill["cash"] * part, "fees": fill["fees"] * part}
        b = {**fill, "qty": new, "cash": fill["cash"] * (1 - part), "fees": fill["fees"] * (1 - part),
             "activity_id": f"{fill['activity_id']}-flip"}
        ep.notes.append("one fill crossed zero; it was split into a close and a new open")
        _apply_fill(open_eps, closed, a, checks)
        _apply_fill(open_eps, closed, b, checks)
        return
    role = "entry" if abs(new) > abs(old) + QTY_EPS else "exit"
    if fill.get("effect") in ("OPENING", "CLOSING"):
        checks["effect_checked"] += 1
        if (fill["effect"] == "OPENING") != (role == "entry"):
            checks["effect_mismatch"] += 1
            ep.notes.append(f"Schwab tagged a fill {fill['effect']} but the running quantity says {role}")
    ep.add(fill, role)
    ep.qty = new
    ep.max_abs = max(ep.max_abs, abs(new))
    if fill["symbol"] not in ep.symbols:
        ep.symbols.append(fill["symbol"])
    if abs(new) < QTY_EPS:
        closed.append(ep)
        del open_eps[k]


def _apply_corp_group(open_eps: dict, closed: list, group: list[dict], checks: dict) -> None:
    net: dict[tuple, dict] = {}
    for f in group:
        k = (f["account"], f["key"])
        if k in net:
            net[k] = {**net[k], "qty": net[k]["qty"] + f["qty"]}
        else:
            net[k] = dict(f)
    legs = [f for f in net.values() if abs(f["qty"]) > QTY_EPS]
    closing = [f for f in legs if f["qty"] < 0]
    opening = [f for f in legs if f["qty"] > 0]
    desc = "; ".join(sorted({str(f.get("description") or "corporate action") for f in group}))
    if len(closing) == 1 and len(opening) == 1:
        old_k = (closing[0]["account"], closing[0]["key"])
        ep = open_eps.get(old_k)
        if ep is not None and abs(ep.qty + closing[0]["qty"]) < QTY_EPS:
            new_f = opening[0]
            del open_eps[old_k]
            ep.notes.append(f"corporate action {ep.key} -> {new_f['key']} ({desc}): {ep.qty:g} -> {new_f['qty']:g}")
            ep.add({**closing[0], "cash": 0.0, "fees": 0.0, "price": None}, "corp_action")
            ep.add({**new_f, "cash": 0.0, "fees": 0.0, "price": None}, "corp_action")
            ep.key, ep.qty = new_f["key"], new_f["qty"]
            ep.max_abs = max(ep.max_abs, abs(ep.qty))
            if new_f["symbol"] not in ep.symbols:
                ep.symbols.append(new_f["symbol"])
            target = (new_f["account"], new_f["key"])
            if target in open_eps:
                raise ValueError(f"corporate action moves {old_k} onto an already-open position {target}")
            open_eps[target] = ep
            return
    for f in legs:
        k = (f["account"], f["key"])
        ep = open_eps.get(k)
        if ep is None:
            ep = open_eps[k] = _Ep(f)
            ep.incomplete = True
            ep.notes.append(f"shares received by corporate action with no cost in history ({desc})")
        else:
            ep.notes.append(f"corporate action on {f['key']}: {f['qty']:+g} shares, no cash ({desc})")
        ep.add({**f, "cash": 0.0, "fees": 0.0, "price": None}, "corp_action")
        ep.qty += f["qty"]
        ep.max_abs = max(ep.max_abs, abs(ep.qty))
        if abs(ep.qty) < QTY_EPS:
            closed.append(ep)
            del open_eps[k]
        elif _sign(ep.qty) != _sign(ep.first["qty"]):
            ep.incomplete = True
            ep.notes.append("corporate action flipped the position's sign; not understood")
            checks["corp_unhandled"] += 1


def build_episodes(fills: list[dict], symbol_map: dict[str, str] | None = None) -> tuple[list[dict], dict]:
    """Flat-to-flat episodes per (account, key). Returns (episodes, checks)."""
    symbol_map = symbol_map or {}
    open_eps: dict[tuple, _Ep] = {}
    closed: list[_Ep] = []
    checks = {"effect_checked": 0, "effect_mismatch": 0, "corp_unhandled": 0}
    i = 0
    while i < len(fills):
        f = fills[i]
        if f["source"] != "corp_action":
            _apply_fill(open_eps, closed, f, checks)
            i += 1
            continue
        day = f["time"].astimezone(ET).date()
        j = i
        while j < len(fills) and fills[j]["source"] == "corp_action" and fills[j]["account"] == f["account"] \
                and fills[j]["time"].astimezone(ET).date() == day:
            j += 1
        _apply_corp_group(open_eps, closed, fills[i:j], checks)
        i = j
    episodes = [_finish(ep, "closed", symbol_map) for ep in closed] + \
               [_finish(ep, "open", symbol_map) for ep in open_eps.values()]
    episodes.sort(key=lambda e: e["open_time"])
    return episodes, checks


def _finish(ep: _Ep, status: str, symbol_map: dict[str, str]) -> dict:
    first = ep.first
    entries = [f for f in ep.fills if f["role"] == "entry"]
    exits = [f for f in ep.fills if f["role"] == "exit"]
    mult = first.get("multiplier") or 1.0
    opened = sum(abs(f["qty"]) for f in entries)
    closed_q = sum(abs(f["qty"]) for f in exits)
    cost = -sum(f["cash"] for f in entries)                    # >0 for a long, <0 for a short (credit)
    proceeds = sum(f["cash"] for f in exits)
    fees = sum(f["fees"] for f in ep.fills)
    net_cash = sum(f["cash"] + f["fees"] for f in ep.fills)

    def avg(rows):
        px = [(f["price"], abs(f["qty"])) for f in rows if f["price"] is not None]
        q = sum(w for _, w in px)
        return sum(p * w for p, w in px) / q if q else None

    open_t = ep.fills[0]["time"]
    close_t = ep.fills[-1]["time"] if status == "closed" else None
    last_exit = exits[-1] if exits else None
    if status == "open":
        exit_reason = "open"
    elif last_exit is None:
        exit_reason = "corp_action"
    elif last_exit["source"] == "expiration":
        exit_reason = "expired"
    else:
        exit_reason = "closed"
    und = first.get("underlying") or first["symbol"]
    chart = symbol_map.get(und) or first.get("bars_symbol") or und
    out = {
        "id": _episode_id(ep.account, first["key"], first["activity_id"]),
        "account": ep.account, "key": ep.key, "entry_key": first["key"], "kind": first["kind"],
        "symbol": first["symbol"], "symbols": ep.symbols, "description": first.get("description"),
        "underlying": und, "chart_symbol": chart,
        "chart_symbol_mapped": chart != und,
        "put_call": first.get("put_call"), "strike": first.get("strike"), "expiry": first.get("expiry"),
        "multiplier": mult, "standard_deliverable": first.get("standard_deliverable", True),
        "direction": "long" if first["qty"] > 0 else "short", "status": status, "exit_reason": exit_reason,
        "open_time": open_t.isoformat(), "close_time": close_t.isoformat() if close_t else None,
        "open_date": open_t.astimezone(ET).date().isoformat(),
        "close_date": close_t.astimezone(ET).date().isoformat() if close_t else None,
        "hold_days": (close_t - open_t).total_seconds() / 86400 if close_t else None,
        "n_entries": len(entries), "n_exits": len(exits), "opened_qty": opened, "closed_qty": closed_q,
        "max_qty": ep.max_abs, "open_qty": ep.qty if status == "open" else 0.0,
        "avg_entry": avg(entries), "avg_exit": avg(exits),
        "cost": cost, "proceeds": proceeds, "fees": fees, "net_cash": net_cash,
        "pnl": net_cash if status == "closed" else None,
        "return_pct": (net_cash / abs(cost)) if status == "closed" and cost else None,
        "dte_at_entry": ((date.fromisoformat(first["expiry"]) - open_t.astimezone(ET).date()).days
                         if first.get("expiry") else None),
        "incomplete": ep.incomplete, "notes": list(dict.fromkeys(ep.notes)),
        "fills": [{"time": f["time"].isoformat(), "role": f["role"], "qty": f["qty"], "price": f["price"],
                   "cash": f["cash"], "fees": f["fees"], "source": f["source"], "symbol": f["symbol"],
                   "time_known": f["time_known"], "order_id": f.get("order_id")} for f in ep.fills],
    }
    if first["kind"] == "option" and not out["standard_deliverable"]:
        out["notes"].append(f"adjusted option (root {und}): it does not deliver 100 shares of {chart}, "
                            f"so the strike is not comparable to {chart}'s price")
    return out


# ---------- marks and reconciliation ----------
def position_key(pos: dict) -> str | None:
    """The episode key of a holdings position (core.personal_portfolio.holdings.normalize_position)."""
    if pos.get("kind") == "option":
        if not (pos.get("underlying") and pos.get("expiry") and pos.get("put_call") and pos.get("strike") is not None):
            return None
        return option_key(pos["underlying"], pos["expiry"], pos["put_call"], float(pos["strike"]))
    if pos.get("kind") in ("equity", "etf"):
        return pos.get("symbol")
    return None


def attach_marks(episodes: list[dict], holdings: dict | None) -> dict:
    """Mark open episodes at Schwab's market value and reconcile them against current positions.

    Mutates open episodes (mark_value, unrealized, held_qty). Returns the reconciliation.
    """
    held: dict[tuple, dict] = {}
    for acct in (holdings or {}).get("accounts") or []:
        for p in acct.get("positions") or []:
            k = position_key(p)
            if k:
                held[(acct["account"], k)] = p
    rec = {"checked": holdings is not None, "matched": [], "qty_mismatch": [], "held_without_history": [],
           "open_not_held": []}
    seen = set()
    for ep in episodes:
        if ep["status"] != "open":
            continue
        k = (ep["account"], ep["key"])
        p = held.get(k)
        if p is None:
            rec["open_not_held"].append({"key": ep["key"], "qty": ep["open_qty"]})
            ep.update(mark_value=None, unrealized=None, held_qty=None, status="unresolved",
                      exit_reason="unresolved", incomplete=True)
            ep["notes"].append(f"history shows {ep['open_qty']:g} still open but Schwab no longer holds it: "
                               "a closing record is missing from Schwab's history")
            continue
        seen.add(k)
        mv = _f(p.get("market_value"))
        ep.update(held_qty=p.get("quantity"), mark_value=mv,
                  unrealized=(ep["net_cash"] + mv) if mv is not None else None,
                  unrealized_pct=((ep["net_cash"] + mv) / abs(ep["cost"])) if mv is not None and ep["cost"] else None)
        if abs((p.get("quantity") or 0.0) - ep["open_qty"]) > 1e-6:
            rec["qty_mismatch"].append({"key": ep["key"], "history_qty": ep["open_qty"], "held_qty": p.get("quantity")})
        else:
            rec["matched"].append(ep["key"])
    if holdings is not None:
        rec["held_without_history"] = sorted(k[1] for k in held if k not in seen)
    rec["ok"] = bool(holdings is not None and not rec["qty_mismatch"] and not rec["held_without_history"]
                     and not rec["open_not_held"])
    return rec
