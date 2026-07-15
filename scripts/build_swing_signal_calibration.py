from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.multi_ticker_swing.live.signal_policy import DEFAULT_CALIBRATION_PATH, score_bucket_for


DEFAULT_AUDIT_ROOT = Path("UI/swing_audit")
MODULE = "multi_ticker_swing"


def build_calibration(audit_root: Path, *, min_count: int = 5) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(audit_root.glob("swing_session_*.jsonl")):
        rows.extend(_closed_trade_rows(path))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["key"]].append(row)

    buckets: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        if len(items) < min_count:
            continue
        returns = [item["return"] for item in items if math.isfinite(item["return"])]
        option_returns = [item["option_return"] for item in items if item["option_return"] is not None]
        holding = [item["holding_minutes"] for item in items if item["holding_minutes"] is not None]
        if not returns:
            continue
        bucket = {
            "key": key,
            "module": MODULE,
            "side": items[0]["side"],
            "regime": items[0]["regime"],
            "score_bucket": items[0]["score_bucket"],
            "count": len(returns),
            "win_rate": sum(1 for value in returns if value > 0.0) / len(returns),
            "avg_return": sum(returns) / len(returns),
            "max_drawdown": _max_drawdown(returns),
            "avg_holding_minutes": (sum(holding) / len(holding)) if holding else None,
            "option_return_estimate": (sum(option_returns) / len(option_returns)) if option_returns else None,
        }
        buckets.append(bucket)

    return {
        "source": "swing_audit_position_closed",
        "audit_root": str(audit_root),
        "closed_trades": len(rows),
        "min_count": int(min_count),
        "buckets": buckets,
    }


def _closed_trade_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    last_signal_by_ticker: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                event = json.loads(line)
            except Exception:
                continue
            event_type = event.get("type")
            payload = event.get("payload") or {}
            if event_type == "signal":
                ticker = str(payload.get("ticker") or "").upper()
                if ticker:
                    last_signal_by_ticker[ticker] = payload
                continue
            if event_type != "position_closed":
                continue
            ticker = str(payload.get("ticker") or "").upper()
            row = _row_from_close(
                event_ts=event.get("ts"),
                payload=payload,
                fallback_signal=last_signal_by_ticker.get(ticker),
            )
            if row is not None:
                rows.append(row)
    return rows


def _row_from_close(
    *,
    event_ts: str | None,
    payload: dict[str, Any],
    fallback_signal: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    direction = _as_int(payload.get("direction"), 0)
    side = "long" if direction > 0 else "short" if direction < 0 else None
    if side is None:
        return None
    signal_policy = ((payload.get("option_entry_meta") or {}).get("signal_policy") or {})
    score_bucket = signal_policy.get("score_bucket")
    regime = signal_policy.get("regime")
    if not score_bucket:
        p_dir = _nested(payload, "option_entry_meta", "signal_policy", "inputs", "p_dir")
        if p_dir is None and fallback_signal is not None:
            p_dir = fallback_signal.get("p_dir")
        score_bucket = score_bucket_for(_as_float(p_dir, float("nan")))
    if not regime:
        regime = _infer_regime_from_payload(payload, fallback_signal=fallback_signal)
    if not score_bucket or score_bucket == "p_dir_nan":
        return None

    pnl = _as_float(payload.get("exit_pnl_pct"), _as_float(payload.get("pnl_pct"), float("nan")))
    if not math.isfinite(pnl):
        return None
    option_return = _option_return(payload)
    holding_minutes = _holding_minutes(payload.get("entry_time"), event_ts)
    key = f"{MODULE}|{side}|{regime}|{score_bucket}"
    return {
        "key": key,
        "side": side,
        "regime": regime,
        "score_bucket": score_bucket,
        "return": float(pnl),
        "option_return": option_return,
        "holding_minutes": holding_minutes,
    }


def _infer_regime_from_payload(payload: dict[str, Any], *, fallback_signal: dict[str, Any] | None = None) -> str:
    policy = ((payload.get("option_entry_meta") or {}).get("risk_profile_policy") or {})
    if not policy and fallback_signal is not None:
        policy = fallback_signal.get("risk_profile_policy") or {}
    profile = policy.get("profile")
    if profile in {"aggressive", "balanced", "defensive"}:
        return str(profile)
    qqq = _as_float(policy.get("qqq_ret_16"), 0.0)
    if qqq <= -0.01:
        return "defensive"
    if qqq <= 0.0:
        return "balanced"
    return "aggressive"


def _option_return(payload: dict[str, Any]) -> float | None:
    entry = _as_float(payload.get("option_entry_price"), float("nan"))
    last = _as_float(payload.get("option_last_price"), float("nan"))
    if not (math.isfinite(entry) and math.isfinite(last) and entry > 0):
        return None
    return (last - entry) / entry


def _holding_minutes(entry_time: str | None, event_ts: str | None) -> float | None:
    try:
        start = datetime.fromisoformat(str(entry_time).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(event_ts).replace("Z", "+00:00"))
    except Exception:
        return None
    return max(0.0, (end - start).total_seconds() / 60.0)


def _max_drawdown(returns: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for value in returns:
        equity += float(value)
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    return float(max_dd)


def _nested(obj: dict[str, Any], *keys: str) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _as_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def main() -> int:
    parser = argparse.ArgumentParser(description="Build multi-ticker swing EV calibration buckets from audit JSONL closes.")
    parser.add_argument("--audit-root", default=str(DEFAULT_AUDIT_ROOT))
    parser.add_argument("--out", default=str(DEFAULT_CALIBRATION_PATH))
    parser.add_argument("--min-count", type=int, default=5)
    args = parser.parse_args()

    out = Path(args.out)
    result = build_calibration(Path(args.audit_root), min_count=args.min_count)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(result['buckets'])} buckets from {result['closed_trades']} closed trades to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
