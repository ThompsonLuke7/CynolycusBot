from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import pandas as pd


LIVE_RUNS = [
    Path("Data/inference/live_runs/20260415_080043_live_spy"),
    Path("Data/inference/live_runs/20260416_074041_live_spy"),
    Path("Data/inference/live_runs/20260417_045215_live_spy"),
]


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _ts(value: Any) -> pd.Timestamp | None:
    if value in (None, "None", ""):
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("America/New_York")


def _option_direction(symbol: str) -> str:
    match = re.search(r"\d{6}([CP])\d{8}$", symbol or "")
    if not match:
        return "unknown"
    return "long" if match.group(1) == "C" else "short"


def _flatten_order(order: dict[str, Any], *, source: str) -> dict[str, Any] | None:
    status = str(order.get("status") or "").lower()
    if status != "filled":
        return None
    symbol = str(order.get("symbol") or "")
    filled_qty = float(order.get("filled_qty") or order.get("qty") or 0)
    price = order.get("filled_avg_price")
    if price in (None, "None", ""):
        return None
    return {
        "id": order.get("id"),
        "symbol": symbol,
        "exposure": _option_direction(symbol),
        "side": order.get("side"),
        "qty": filled_qty,
        "price": float(price),
        "submitted_at": _ts(order.get("submitted_at")),
        "filled_at": _ts(order.get("filled_at")),
        "source": source,
    }


def main() -> None:
    orders: dict[str, dict[str, Any]] = {}
    contexts: dict[str, dict[str, Any]] = {}
    decision_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []

    for run in LIVE_RUNS:
        for rec in _rows(run / "broker-state.jsonl"):
            state = rec.get("payload", {}).get("state", {})
            for order in state.get("recent_orders") or []:
                flat = _flatten_order(order, source=str(run))
                if flat and flat["id"] not in orders:
                    orders[flat["id"]] = flat

        for rec in _rows(run / "order-policy.jsonl"):
            payload = rec.get("payload", {})
            result = payload.get("result", {})
            state = payload.get("policy_state", {})
            event = result.get("event")
            ts = _ts(payload.get("timestamp"))
            policy_rows.append(
                {
                    "run": run.name,
                    "timestamp": ts,
                    "event": event,
                    "close": result.get("close"),
                    "long_reason": state.get("long_decision_reason"),
                    "short_reason": state.get("short_decision_reason"),
                    "long_thr": state.get("meta_intrabar_long_setup_threshold"),
                    "short_thr": state.get("meta_intrabar_short_setup_threshold"),
                    "long_active": state.get("long_intrabar_intent_active"),
                    "short_active": state.get("short_intrabar_intent_active"),
                    "long_ref": state.get("long_intrabar_entry_ref"),
                    "short_ref": state.get("short_intrabar_entry_ref"),
                    "open_long_symbol": state.get("open_long_symbol"),
                    "open_short_symbol": state.get("open_short_symbol"),
                }
            )
            for wrapper in result.get("orders") or []:
                response = wrapper.get("response") or {}
                verification = response.get("verification") or {}
                order = verification.get("order") or response.get("response") or {}
                flat = _flatten_order(order, source=str(run))
                if flat:
                    orders[flat["id"]] = flat
                    contexts[flat["id"]] = {
                        "policy_event": event,
                        "policy_ts": ts,
                        "long_reason": state.get("long_decision_reason"),
                        "short_reason": state.get("short_decision_reason"),
                    }

        for rec in _rows(run / "decision-10m.jsonl"):
            payload = rec.get("payload", {})
            bar = payload.get("bar", {})
            state = payload.get("policy_state", {})
            decision_rows.append(
                {
                    "run": run.name,
                    "timestamp": _ts(payload.get("timestamp")),
                    "close": bar.get("close"),
                    "p_long": bar.get("p_enter_long"),
                    "p_short": bar.get("p_enter_short"),
                    "policy_long_thr": state.get("meta_intrabar_long_setup_threshold"),
                    "policy_short_thr": state.get("meta_intrabar_short_setup_threshold"),
                    "long_reason": state.get("long_decision_reason"),
                    "short_reason": state.get("short_decision_reason"),
                    "long_active": state.get("long_intrabar_intent_active"),
                    "short_active": state.get("short_intrabar_intent_active"),
                }
            )

    order_df = pd.DataFrame(orders.values()).sort_values("filled_at")
    if not order_df.empty:
        ctx_df = pd.DataFrame([{"id": k, **v} for k, v in contexts.items()])
        if not ctx_df.empty:
            order_df = order_df.merge(ctx_df, on="id", how="left")

    open_lots: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    trades: list[dict[str, Any]] = []
    for row in order_df.to_dict("records"):
        if row["side"] == "buy":
            open_lots[row["symbol"]].append(row)
        elif row["side"] == "sell" and open_lots[row["symbol"]]:
            buy = open_lots[row["symbol"]].popleft()
            qty = min(float(buy["qty"]), float(row["qty"]))
            trades.append(
                {
                    "symbol": row["symbol"],
                    "exposure": row["exposure"],
                    "entry_time": buy["filled_at"],
                    "exit_time": row["filled_at"],
                    "entry": buy["price"],
                    "exit": row["price"],
                    "qty": qty,
                    "pnl_dollars": (row["price"] - buy["price"]) * 100.0 * qty,
                    "return_pct": (row["price"] / buy["price"] - 1.0) * 100.0 if buy["price"] else float("nan"),
                    "exit_event": row.get("policy_event"),
                    "exit_long_reason": row.get("long_reason"),
                    "exit_short_reason": row.get("short_reason"),
                }
            )

    open_positions = []
    for lots in open_lots.values():
        open_positions.extend(lots)

    print("\nFILLED ORDERS")
    print(
        order_df[
            ["filled_at", "symbol", "exposure", "side", "qty", "price", "policy_event", "long_reason", "short_reason"]
        ].to_string(index=False)
        if not order_df.empty
        else "none"
    )

    trade_df = pd.DataFrame(trades)
    print("\nPAIRED TRADES")
    if trade_df.empty:
        print("none")
    else:
        print(trade_df.to_string(index=False))
        print(
            "\nSUMMARY",
            {
                "trades": int(len(trade_df)),
                "pnl_dollars": float(trade_df["pnl_dollars"].sum()),
                "avg_return_pct": float(trade_df["return_pct"].mean()),
                "win_rate": float((trade_df["return_pct"] > 0).mean()),
            },
        )

    if open_positions:
        print("\nOPEN LOTS")
        print(
            pd.DataFrame(open_positions)[
                ["filled_at", "symbol", "exposure", "side", "qty", "price", "policy_event", "long_reason", "short_reason"]
            ].to_string(index=False)
        )

    decisions = pd.DataFrame(decision_rows).sort_values("timestamp")
    if not decisions.empty:
        print("\nDAILY PROBA / POLICY THRESHOLD SUMMARY")
        decisions["day"] = decisions["timestamp"].dt.strftime("%Y-%m-%d")
        for day, g in decisions.groupby("day"):
            print(
                day,
                {
                    "bars": int(len(g)),
                    "max_long": float(pd.to_numeric(g["p_long"], errors="coerce").max()),
                    "max_short": float(pd.to_numeric(g["p_short"], errors="coerce").max()),
                    "long>=.35": int((pd.to_numeric(g["p_long"], errors="coerce") >= 0.35).sum()),
                    "short>=.65": int((pd.to_numeric(g["p_short"], errors="coerce") >= 0.65).sum()),
                    "policy_long_thr_values": sorted(set(g["policy_long_thr"].dropna().astype(float).round(3))),
                    "policy_short_thr_values": sorted(set(g["policy_short_thr"].dropna().astype(float).round(3))),
                },
            )
            crosses = g[(pd.to_numeric(g["p_long"], errors="coerce") >= 0.35) | (pd.to_numeric(g["p_short"], errors="coerce") >= 0.65)]
            if not crosses.empty:
                print(
                    crosses[
                        ["timestamp", "close", "p_long", "p_short", "policy_long_thr", "policy_short_thr", "long_reason", "short_reason"]
                    ].to_string(index=False)
                )


if __name__ == "__main__":
    main()
