#!/usr/bin/env python
"""Restate a managed position across a stock split the broker already applied.

ENHA went 10-for-1 reverse on 2026-10-29. The broker turned 3,436 shares at
$1.28 into 343.6 at $12.80; HTF's state kept 3,436 at $1.28 with a last mark of
$1.19, so every pass saw a +500-985% one-bar move and the mark-anomaly guard
(correctly) refused to evaluate the position at all, stop included.

The split is read from the broker's corporate-actions feed, never typed in. A
claim is restated only when exactly one split falls after its last mark and the
restated share count of every claimant adds up to the broker position. Share
fields are multiplied by new_rate/old_rate; per-share prices (entry, fill,
underlying basis, ATR, last mark) are divided by it; ratios (unrealized and peak
gain) are unchanged. The refusal record (`mark_anomaly`) is removed. Signal and
order audits stay as they were sent.

Each state file is copied to a timestamped `.bak` and every change is appended
to `<module>/ledger_repairs.jsonl`. Default is a dry run; pass --apply to write.

    PYTHONPATH=. .venv/bin/python scripts/apply_split_to_claim.py --symbol ENHA [--apply]
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.live_state import load_state, module_state_lock, save_state  # noqa: E402
from core.orphan_positions import MANAGED_STATE_PATHS  # noqa: E402

REASON = "split_adjusted"
SHARE_FIELDS = ("shares", "entry_filled_qty", "remaining_qty", "trim_remaining_qty")
PRICE_FIELDS = ("entry_avg_price", "entry_fill_price", "u_entry", "u_atr", "u_last", "last_mark_price")


def splits_after(actions: dict, symbol: str, after: date) -> list[dict]:
    """Every forward/reverse split of `symbol` with an ex-date after `after`."""

    found = []
    for kind in ("reverse_splits", "forward_splits"):
        for row in (actions.get("corporate_actions") or {}).get(kind) or []:
            if str(row.get("symbol", "")).upper() != symbol:
                continue
            ex = date.fromisoformat(str(row["ex_date"]))
            if ex > after:
                found.append({"kind": kind, "ex_date": ex.isoformat(),
                              "ratio": float(row["new_rate"]) / float(row["old_rate"]), "id": row.get("id")})
    return found


def restate(claim: dict, ratio: float) -> dict:
    """The claim as it reads after a split of `ratio` new shares per old share."""

    out = dict(claim)
    for key in SHARE_FIELDS:
        if isinstance(out.get(key), (int, float)) and not isinstance(out.get(key), bool):
            out[key] = round(float(out[key]) * ratio, 6)
    for key in PRICE_FIELDS:
        if isinstance(out.get(key), (int, float)) and not isinstance(out.get(key), bool):
            out[key] = round(float(out[key]) / ratio, 6)
    out.pop("mark_anomaly", None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--root", default=str(REPO / "Data/inference"))
    ap.add_argument("--env-file", default=".env#PAPER")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    symbol = args.symbol.strip().upper()

    claims = {}
    for module, path in MANAGED_STATE_PATHS.items():
        st = load_state(path)["managed"].get(symbol)
        if isinstance(st, dict) and st.get("route", "equity") == "equity":
            claims[module] = st
    if not claims:
        print(f"no module claims {symbol}")
        return 1

    from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient

    client = AlpacaOptionsClient(env_file=args.env_file)
    position = client.get_position(symbol)
    broker_qty = float((position or {}).get("qty") or 0.0)
    last_marks = [datetime.fromisoformat(str(c.get("last_mark_bar") or c["entry_bar"]).replace(" ", "T")).date()
                  for c in claims.values()]
    since = min(last_marks)
    actions = client.get_corporate_actions(symbols=symbol, start=since.isoformat(),
                                           end=date.today().isoformat())
    splits = splits_after(actions, symbol, since)
    if len(splits) != 1:
        print(f"{symbol}: {len(splits)} splits after {since} ({splits}); restate by hand")
        return 1
    ratio = splits[0]["ratio"]
    restated = {m: restate(c, ratio) for m, c in claims.items()}
    total = sum(float(c["shares"]) for c in restated.values())
    print(f"{symbol}: {splits[0]['kind']} ex {splits[0]['ex_date']} ratio {ratio:g}; "
          f"broker {broker_qty:g}, claims {sum(float(c['shares']) for c in claims.values()):g} -> {total:g}")
    if not math.isclose(total, broker_qty, abs_tol=1e-6):
        print("  restated claims do not add up to the broker position; nothing changed")
        return 1
    for module in claims:
        before = {k: claims[module].get(k) for k in SHARE_FIELDS + PRICE_FIELDS if k in claims[module]}
        after = {k: restated[module].get(k) for k in before}
        print(f"  {module}: {before} -> {after}")
    if not args.apply:
        print("DRY RUN — nothing written. Re-run with --apply to write.")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for module, original in claims.items():
        path = MANAGED_STATE_PATHS[module]
        with module_state_lock(module) as acquired:
            if not acquired:
                print(f"  {module}: state is locked by a running pass; skipped, run again")
                continue
            state = load_state(path)
            if state["managed"].get(symbol) != original:
                print(f"  {module} {symbol}: changed since it was read; skipped")
                continue
            backup = path.with_suffix(f"{path.suffix}.{stamp}.bak")
            shutil.copy2(path, backup)
            state["managed"][symbol] = restated[module]
            save_state(path, state)
            record = {"repaired_at": stamp, "reason": REASON, "module": module, "ticker": symbol,
                      "split": splits[0], "broker_qty": broker_qty, "state_backup": backup.name,
                      "from": {k: original.get(k) for k in SHARE_FIELDS + PRICE_FIELDS if k in original},
                      "to": {k: restated[module].get(k) for k in SHARE_FIELDS + PRICE_FIELDS if k in original}}
            with (Path(args.root) / module / "ledger_repairs.jsonl").open("a") as fh:
                fh.write(json.dumps(record, default=str, allow_nan=False) + "\n")
            print(f"  wrote {path} (backup {backup.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
