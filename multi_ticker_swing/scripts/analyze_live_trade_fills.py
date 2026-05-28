from __future__ import annotations

import argparse
import csv
import json
import math
import urllib.parse
import urllib.request
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from API.Alpaca_API.core.config import AlpacaConfig
from API.Alpaca_API.options.options_api import OptionsClientConfig


ET = ZoneInfo("America/New_York")
OPTION_MULTIPLIER = 100.0


def _parse_ts(value: Any) -> pd.Timestamp | pd.NaT:
    if value is None or value == "":
        return pd.NaT
    return pd.to_datetime(value, utc=True, errors="coerce")


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def parse_occ(symbol: str) -> dict[str, Any]:
    text = str(symbol or "").strip().upper()
    if len(text) < 15:
        return {"underlying": text, "is_option": False}
    root = text[:-15]
    yymmdd = text[-15:-9]
    cp = text[-9:-8]
    strike_raw = text[-8:]
    if not root or cp not in {"C", "P"} or not yymmdd.isdigit() or not strike_raw.isdigit():
        return {"underlying": text, "is_option": False}
    year = 2000 + int(yymmdd[:2])
    exp = f"{year:04d}-{int(yymmdd[2:4]):02d}-{int(yymmdd[4:6]):02d}"
    return {
        "underlying": root,
        "is_option": True,
        "expiration": exp,
        "option_type": cp,
        "strike": int(strike_raw) / 1000.0,
    }


def request_json(url: str, key: str, secret: str, params: dict[str, Any]) -> Any:
    clean = {k: v for k, v in params.items() if v not in (None, "")}
    if clean:
        url = f"{url}?{urllib.parse.urlencode(clean)}"
    req = urllib.request.Request(
        url,
        headers={
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw else []


def fetch_fills(env_file: str, after: str, until: str | None, max_pages: int) -> list[dict[str, Any]]:
    cfg = AlpacaConfig.from_env(env_file)
    trading_base = OptionsClientConfig.from_env().trading_base_url.rstrip("/")
    url = f"{trading_base}/v2/account/activities/FILL"
    fills: list[dict[str, Any]] = []
    page_token: str | None = None
    seen_tokens: set[str] = set()
    for _ in range(max_pages):
        params = {
            "after": after,
            "until": until,
            "direction": "asc",
            "page_size": 100,
            "page_token": page_token,
        }
        page = request_json(url, cfg.key_id, cfg.secret_key, params)
        if not page:
            break
        if not isinstance(page, list):
            raise RuntimeError(f"Unexpected activities payload: {type(page).__name__}")
        fills.extend(page)
        last_id = str(page[-1].get("id") or "").strip()
        if not last_id or last_id in seen_tokens:
            break
        seen_tokens.add(last_id)
        page_token = last_id
        if len(page) < 100:
            break
    return fills


def normalize_fills(raw: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in raw:
        symbol = str(item.get("symbol") or "").upper()
        parsed = parse_occ(symbol)
        side = str(item.get("side") or "").lower()
        qty = _as_float(item.get("qty"))
        price = _as_float(item.get("price"))
        ts = _parse_ts(item.get("transaction_time") or item.get("date"))
        if not symbol or side not in {"buy", "sell"} or not math.isfinite(qty) or qty <= 0:
            continue
        rows.append(
            {
                "activity_id": item.get("id"),
                "order_id": item.get("order_id"),
                "symbol": symbol,
                "underlying": parsed.get("underlying"),
                "is_option": bool(parsed.get("is_option")),
                "expiration": parsed.get("expiration"),
                "option_type": parsed.get("option_type"),
                "strike": parsed.get("strike"),
                "side": side,
                "qty": qty,
                "price": price,
                "amount": _as_float(item.get("net_amount")),
                "transaction_time": ts,
                "date": item.get("date"),
                "raw": item,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["transaction_time", "activity_id"]).reset_index(drop=True)
    return df


def pair_option_round_trips(fills: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lots: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    closed: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    option_fills = fills[fills["is_option"]].copy()
    for row in option_fills.to_dict("records"):
        symbol = row["symbol"]
        qty_left = float(row["qty"])
        if row["side"] == "buy":
            lots[symbol].append({**row, "qty_left": qty_left})
            continue
        while qty_left > 1e-9 and lots[symbol]:
            lot = lots[symbol][0]
            close_qty = min(qty_left, float(lot["qty_left"]))
            pnl = (float(row["price"]) - float(lot["price"])) * close_qty * OPTION_MULTIPLIER
            entry_ts = _parse_ts(lot["transaction_time"])
            exit_ts = _parse_ts(row["transaction_time"])
            parsed = parse_occ(symbol)
            closed.append(
                {
                    "symbol": symbol,
                    "ticker": parsed.get("underlying"),
                    "option_type": parsed.get("option_type"),
                    "expiration": parsed.get("expiration"),
                    "strike": parsed.get("strike"),
                    "qty": close_qty,
                    "entry_time": entry_ts,
                    "exit_time": exit_ts,
                    "entry_price_option": float(lot["price"]),
                    "exit_price_option": float(row["price"]),
                    "pnl_dollars": pnl,
                    "pnl_pct_option": (float(row["price"]) / float(lot["price"]) - 1.0)
                    if float(lot["price"]) > 0
                    else np.nan,
                    "holding_minutes": (exit_ts - entry_ts).total_seconds() / 60.0
                    if pd.notna(entry_ts) and pd.notna(exit_ts)
                    else np.nan,
                    "entry_order_id": lot.get("order_id"),
                    "exit_order_id": row.get("order_id"),
                    "entry_activity_id": lot.get("activity_id"),
                    "exit_activity_id": row.get("activity_id"),
                }
            )
            qty_left -= close_qty
            lot["qty_left"] = float(lot["qty_left"]) - close_qty
            if lot["qty_left"] <= 1e-9:
                lots[symbol].popleft()
        if qty_left > 1e-9:
            unmatched.append({**row, "qty_left": qty_left, "reason": "sell_without_prior_buy"})
    for symbol, q in lots.items():
        for lot in q:
            unmatched.append({**lot, "reason": "open_buy_lot"})
    return pd.DataFrame(closed), pd.DataFrame(unmatched)


def load_audit(log_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    opens: list[dict[str, Any]] = []
    closes: list[dict[str, Any]] = []
    bars: list[dict[str, Any]] = []
    signals: list[dict[str, Any]] = []
    for path in sorted(log_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                typ = event.get("type")
                payload = event.get("payload") or {}
                ts = _parse_ts(event.get("ts"))
                if typ == "signal":
                    signals.append({"signal_ts": ts, **payload})
                elif typ == "position_opened":
                    opens.append({"audit_ts": ts, **payload})
                elif typ in {"position_closed", "position_close_failed", "position_close_abandoned", "broker_position_missing"}:
                    closes.append({"audit_ts": ts, "audit_type": typ, **payload})
                elif typ == "position_bar_5m":
                    pos = payload.get("position") or {}
                    bar = payload.get("bar") or {}
                    bars.append(
                        {
                            "audit_ts": ts,
                            "ticker": payload.get("ticker") or pos.get("ticker"),
                            "entry_time": pos.get("entry_time"),
                            "entry_price": pos.get("entry_price"),
                            "direction": pos.get("direction"),
                            "option_symbol": pos.get("option_symbol"),
                            "underlying_close": bar.get("close"),
                            "underlying_high": bar.get("high"),
                            "underlying_low": bar.get("low"),
                            "option_last_price": pos.get("option_last_price"),
                            "option_best_price": pos.get("option_best_price"),
                            "pnl_pct_underlying_mark": pos.get("pnl_pct"),
                            "bars_held": pos.get("bars_held"),
                        }
                    )
    opens_df = pd.DataFrame(opens)
    closes_df = pd.DataFrame(closes)
    bars_df = pd.DataFrame(bars)
    signals_df = pd.DataFrame(signals)
    for df in (opens_df, closes_df, bars_df, signals_df):
        for col in ("entry_time", "audit_ts", "signal_ts"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    if not bars_df.empty and not opens_df.empty and "option_symbol" in opens_df.columns:
        key_cols = ["ticker", "entry_time"]
        open_map = opens_df[key_cols + ["option_symbol"]].dropna(subset=key_cols).drop_duplicates(key_cols)
        bars_df = bars_df.drop(columns=["option_symbol"], errors="ignore").merge(
            open_map,
            on=key_cols,
            how="left",
        )
    return opens_df, closes_df, bars_df, signals_df


def enrich_trades(
    trades: pd.DataFrame,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    bars: pd.DataFrame,
    signals: pd.DataFrame,
) -> pd.DataFrame:
    if trades.empty:
        return trades
    out = trades.copy()
    if not opens.empty:
        open_cols = [
            "option_symbol",
            "ticker",
            "direction",
            "entry_price",
            "atr_at_entry",
            "tier",
            "sl_price",
            "audit_ts",
        ]
        open_ctx = opens[[c for c in open_cols if c in opens.columns]].rename(
            columns={
                "option_symbol": "symbol",
                "entry_price": "entry_price_underlying",
                "audit_ts": "audit_open_ts",
            }
        )
        open_ctx = open_ctx.sort_values("audit_open_ts").drop_duplicates("symbol", keep="first")
        out = out.merge(open_ctx, on="symbol", how="left", suffixes=("", "_audit"))
    if not closes.empty:
        close_ctx = closes.sort_values("audit_ts").drop_duplicates("option_symbol", keep="last")
        close_ctx = close_ctx.rename(columns={"option_symbol": "symbol"})
        keep = [
            "symbol",
            "audit_type",
            "exit_reason",
            "exit_price",
            "exit_pnl_pct",
            "bars_held",
            "option_best_price",
            "option_last_price",
            "audit_ts",
        ]
        out = out.merge(close_ctx[[c for c in keep if c in close_ctx.columns]], on="symbol", how="left")
    if not bars.empty:
        mark = bars.copy()
        mark["option_last_price"] = pd.to_numeric(mark["option_last_price"], errors="coerce")
        mark["pnl_pct_underlying_mark"] = pd.to_numeric(mark["pnl_pct_underlying_mark"], errors="coerce")
        stats = (
            mark.groupby("option_symbol")
            .agg(
                option_mark_max=("option_last_price", "max"),
                option_mark_min=("option_last_price", "min"),
                underlying_mark_max=("pnl_pct_underlying_mark", "max"),
                underlying_mark_min=("pnl_pct_underlying_mark", "min"),
                mark_count=("option_symbol", "size"),
            )
            .rename_axis("symbol")
            .reset_index()
        )
        out = out.merge(stats, on="symbol", how="left")
    if not signals.empty:
        sig = signals.copy()
        sig["signal_day"] = sig["signal_ts"].dt.tz_convert(ET).dt.date
        signal_rows = []
        for idx, trade in out.iterrows():
            day = trade["entry_time"].tz_convert(ET).date() if pd.notna(trade["entry_time"]) else None
            pool = sig[
                (sig["ticker"].astype(str).str.upper() == str(trade["ticker"]).upper())
                & (sig["signal_day"] == day)
                & (sig["signal_ts"] <= trade["entry_time"])
            ]
            if "direction" in out.columns and pd.notna(trade.get("direction")):
                pool = pool[pd.to_numeric(pool["direction"], errors="coerce") == float(trade["direction"])]
            if pool.empty:
                signal_rows.append({"_idx": idx})
            else:
                last = pool.sort_values("signal_ts").iloc[-1]
                signal_rows.append(
                    {
                        "_idx": idx,
                        "signal_ts": last.get("signal_ts"),
                        "p_dir": last.get("p_dir"),
                        "ev_score": last.get("ev_score"),
                        "ref_high": last.get("ref_high"),
                        "ref_low": last.get("ref_low"),
                        "signal_atr": last.get("atr"),
                    }
                )
        sig_ctx = pd.DataFrame(signal_rows).set_index("_idx")
        out = out.join(sig_ctx)
    out["entry_time_et"] = out["entry_time"].dt.tz_convert(ET)
    out["exit_time_et"] = out["exit_time"].dt.tz_convert(ET)
    out["entry_date"] = out["entry_time_et"].dt.date
    out["entry_hour"] = out["entry_time_et"].dt.hour
    out["entry_30m"] = out["entry_time_et"].dt.strftime("%H:") + (
        (out["entry_time_et"].dt.minute // 30) * 30
    ).astype(str).str.zfill(2)
    out["dte_at_entry"] = (
        pd.to_datetime(out["expiration"], errors="coerce")
        - out["entry_time_et"].dt.tz_localize(None).dt.normalize()
    ).dt.days
    out["win"] = out["pnl_dollars"] > 0
    out["option_mfe_pct_mark"] = out["option_mark_max"] / out["entry_price_option"] - 1.0
    out["option_mae_pct_mark"] = out["option_mark_min"] / out["entry_price_option"] - 1.0
    return out


def enrich_open_lots(
    unmatched: pd.DataFrame,
    opens: pd.DataFrame,
    bars: pd.DataFrame,
    signals: pd.DataFrame,
) -> pd.DataFrame:
    if unmatched.empty:
        return unmatched
    open_lots = unmatched[unmatched.get("reason").eq("open_buy_lot")].copy()
    if open_lots.empty:
        return open_lots
    rows = []
    for row in open_lots.to_dict("records"):
        parsed = parse_occ(row.get("symbol"))
        rows.append(
            {
                "symbol": row.get("symbol"),
                "ticker": parsed.get("underlying"),
                "option_type": parsed.get("option_type"),
                "expiration": parsed.get("expiration"),
                "strike": parsed.get("strike"),
                "qty_open": row.get("qty_left", row.get("qty")),
                "entry_time": _parse_ts(row.get("transaction_time")),
                "entry_price_option": row.get("price"),
                "entry_order_id": row.get("order_id"),
                "entry_activity_id": row.get("activity_id"),
                "reason": row.get("reason"),
            }
        )
    out = pd.DataFrame(rows)
    if not opens.empty:
        open_cols = [
            "option_symbol",
            "ticker",
            "direction",
            "entry_price",
            "atr_at_entry",
            "tier",
            "sl_price",
            "audit_ts",
        ]
        open_ctx = opens[[c for c in open_cols if c in opens.columns]].rename(
            columns={
                "option_symbol": "symbol",
                "entry_price": "entry_price_underlying",
                "audit_ts": "audit_open_ts",
            }
        )
        open_ctx = open_ctx.sort_values("audit_open_ts").drop_duplicates("symbol", keep="first")
        out = out.merge(open_ctx, on="symbol", how="left", suffixes=("", "_audit"))
    if not bars.empty:
        mark = bars.copy()
        mark["option_last_price"] = pd.to_numeric(mark["option_last_price"], errors="coerce")
        mark["pnl_pct_underlying_mark"] = pd.to_numeric(mark["pnl_pct_underlying_mark"], errors="coerce")
        mark["bars_held"] = pd.to_numeric(mark["bars_held"], errors="coerce")
        stats = (
            mark.groupby("option_symbol")
            .agg(
                option_mark_last=("option_last_price", "last"),
                option_mark_max=("option_last_price", "max"),
                option_mark_min=("option_last_price", "min"),
                underlying_mark_last=("pnl_pct_underlying_mark", "last"),
                underlying_mark_max=("pnl_pct_underlying_mark", "max"),
                underlying_mark_min=("pnl_pct_underlying_mark", "min"),
                bars_seen=("option_symbol", "size"),
                bars_held_last=("bars_held", "last"),
            )
            .rename_axis("symbol")
            .reset_index()
        )
        out = out.merge(stats, on="symbol", how="left")
    if not signals.empty:
        sig = signals.copy()
        sig["signal_day"] = sig["signal_ts"].dt.tz_convert(ET).dt.date
        signal_rows = []
        for idx, lot in out.iterrows():
            day = lot["entry_time"].tz_convert(ET).date() if pd.notna(lot["entry_time"]) else None
            pool = sig[
                (sig["ticker"].astype(str).str.upper() == str(lot["ticker"]).upper())
                & (sig["signal_day"] == day)
                & (sig["signal_ts"] <= lot["entry_time"])
            ]
            if "direction" in out.columns and pd.notna(lot.get("direction")):
                pool = pool[pd.to_numeric(pool["direction"], errors="coerce") == float(lot["direction"])]
            if pool.empty:
                signal_rows.append({"_idx": idx})
            else:
                last = pool.sort_values("signal_ts").iloc[-1]
                signal_rows.append(
                    {
                        "_idx": idx,
                        "signal_ts": last.get("signal_ts"),
                        "p_dir": last.get("p_dir"),
                        "ev_score": last.get("ev_score"),
                        "ref_high": last.get("ref_high"),
                        "ref_low": last.get("ref_low"),
                        "signal_atr": last.get("atr"),
                    }
                )
        sig_ctx = pd.DataFrame(signal_rows).set_index("_idx")
        out = out.join(sig_ctx)
    out["entry_time_et"] = out["entry_time"].dt.tz_convert(ET)
    out["entry_date"] = out["entry_time_et"].dt.date
    out["entry_hour"] = out["entry_time_et"].dt.hour
    out["entry_30m"] = out["entry_time_et"].dt.strftime("%H:") + (
        (out["entry_time_et"].dt.minute // 30) * 30
    ).astype(str).str.zfill(2)
    out["dte_at_entry"] = (
        pd.to_datetime(out["expiration"], errors="coerce")
        - out["entry_time_et"].dt.tz_localize(None).dt.normalize()
    ).dt.days
    out["option_open_mark_pnl_pct"] = out["option_mark_last"] / out["entry_price_option"] - 1.0
    out["option_mfe_pct_mark"] = out["option_mark_max"] / out["entry_price_option"] - 1.0
    out["option_mae_pct_mark"] = out["option_mark_min"] / out["entry_price_option"] - 1.0
    out["open_mark_pnl_dollars"] = (
        (out["option_mark_last"] - out["entry_price_option"])
        * pd.to_numeric(out["qty_open"], errors="coerce")
        * OPTION_MULTIPLIER
    )
    return out


def summarize_open_group(df: pd.DataFrame, by: str, min_n: int = 3) -> pd.DataFrame:
    if df.empty or by not in df.columns:
        return pd.DataFrame()
    g = (
        df.groupby(by, dropna=False)
        .agg(
            n=("symbol", "size"),
            marked_pnl=("open_mark_pnl_dollars", "sum"),
            avg_mark_pct=("option_open_mark_pnl_pct", "mean"),
            avg_mfe=("option_mfe_pct_mark", "mean"),
            avg_mae=("option_mae_pct_mark", "mean"),
            avg_underlying_last=("underlying_mark_last", "mean"),
            avg_underlying_mfe=("underlying_mark_max", "mean"),
            avg_underlying_mae=("underlying_mark_min", "mean"),
            median_bars_seen=("bars_seen", "median"),
        )
        .reset_index()
    )
    return g[g["n"] >= min_n].sort_values("marked_pnl", ascending=False)


def summarize_group(df: pd.DataFrame, by: str, min_n: int = 3) -> pd.DataFrame:
    if df.empty or by not in df.columns:
        return pd.DataFrame()
    g = (
        df.groupby(by, dropna=False)
        .agg(
            n=("pnl_dollars", "size"),
            net_pnl=("pnl_dollars", "sum"),
            avg_pnl=("pnl_dollars", "mean"),
            win_rate=("win", "mean"),
            avg_option_pct=("pnl_pct_option", "mean"),
            median_hold_min=("holding_minutes", "median"),
            avg_underlying_exit_pct=("exit_pnl_pct", "mean"),
            avg_option_mfe=("option_mfe_pct_mark", "mean"),
            avg_option_mae=("option_mae_pct_mark", "mean"),
        )
        .reset_index()
    )
    return g[g["n"] >= min_n].sort_values("net_pnl", ascending=False)


def cluster_trades(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "p_dir",
        "ev_score",
        "atr_at_entry",
        "entry_price_underlying",
        "holding_minutes",
        "dte_at_entry",
        "entry_hour",
        "option_mfe_pct_mark",
        "option_mae_pct_mark",
    ]
    keep = [c for c in cols if c in df.columns]
    work = df[keep].apply(pd.to_numeric, errors="coerce")
    good = work.replace([np.inf, -np.inf], np.nan).dropna()
    if len(good) < 12 or len(keep) < 3:
        return pd.DataFrame()
    k = min(5, max(2, len(good) // 15))
    scaler = StandardScaler()
    x = scaler.fit_transform(good)
    labels = KMeans(n_clusters=k, n_init=20, random_state=7).fit_predict(x)
    clustered = df.loc[good.index].copy()
    clustered["cluster"] = labels
    return (
        clustered.groupby("cluster")
        .agg(
            n=("pnl_dollars", "size"),
            net_pnl=("pnl_dollars", "sum"),
            avg_pnl=("pnl_dollars", "mean"),
            win_rate=("win", "mean"),
            avg_p_dir=("p_dir", "mean"),
            avg_ev=("ev_score", "mean"),
            avg_hold_min=("holding_minutes", "mean"),
            avg_dte=("dte_at_entry", "mean"),
            avg_mfe=("option_mfe_pct_mark", "mean"),
            avg_mae=("option_mae_pct_mark", "mean"),
        )
        .reset_index()
        .sort_values("net_pnl", ascending=False)
    )


def write_report(
    out_dir: Path,
    raw_fills: pd.DataFrame,
    trades: pd.DataFrame,
    unmatched: pd.DataFrame,
    open_lots: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    open_summaries: dict[str, pd.DataFrame],
) -> None:
    def _markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
        show = df.head(max_rows).copy()
        if show.empty:
            return ""
        for col in show.columns:
            if pd.api.types.is_float_dtype(show[col]):
                show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
            else:
                show[col] = show[col].map(lambda x: "" if pd.isna(x) else str(x))
        headers = list(show.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in show.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(str(x).replace("\n", " ") for x in row) + " |")
        return "\n".join(lines)

    lines: list[str] = []
    lines.append("# Multi-Ticker Swing Live Fill Analysis")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Coverage")
    lines.append(f"- Fills loaded: {len(raw_fills):,}")
    lines.append(f"- Closed option round trips paired FIFO: {len(trades):,}")
    lines.append(f"- Unmatched/open option lots: {len(unmatched):,}")
    lines.append(f"- Open buy lots analyzed: {len(open_lots):,}")
    if not trades.empty:
        lines.append(
            f"- Realized option PnL: ${trades['pnl_dollars'].sum():,.2f}; "
            f"win rate {trades['win'].mean():.1%}; avg/trade ${trades['pnl_dollars'].mean():,.2f}"
        )
        lines.append(
            f"- Date span: {trades['entry_time_et'].min()} to {trades['exit_time_et'].max()}"
        )
    lines.append("")
    for name, df in summaries.items():
        if df.empty:
            continue
        lines.append(f"## {name}")
        lines.append(_markdown_table(df))
        lines.append("")
    if not open_lots.empty:
        lines.append("## Open Lots Snapshot")
        cols = [
            "ticker",
            "symbol",
            "entry_time_et",
            "entry_price_option",
            "option_mark_last",
            "open_mark_pnl_dollars",
            "option_open_mark_pnl_pct",
            "option_mfe_pct_mark",
            "option_mae_pct_mark",
            "underlying_mark_last",
            "bars_seen",
        ]
        show = open_lots[[c for c in cols if c in open_lots.columns]].sort_values(
            "open_mark_pnl_dollars", ascending=True
        )
        lines.append(_markdown_table(show, max_rows=30))
        lines.append("")
    for name, df in open_summaries.items():
        if df.empty:
            continue
        lines.append(f"## Open Lots - {name}")
        lines.append(_markdown_table(df))
        lines.append("")
    report = out_dir / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--after", default="2026-05-01")
    parser.add_argument("--until", default=None)
    parser.add_argument("--max-pages", type=int, default=50)
    parser.add_argument("--audit-dir", default="UI/swing_audit")
    parser.add_argument("--out-dir", default="Data/analysis/multi_ticker_swing_live")
    parser.add_argument("--skip-fetch", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "alpaca_fills_raw.json"
    if args.skip_fetch and raw_path.exists():
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    else:
        raw = fetch_fills(args.env_file, args.after, args.until, args.max_pages)
        raw_path.write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")

    fills = normalize_fills(raw)
    trades, unmatched = pair_option_round_trips(fills)
    opens, closes, bars, signals = load_audit(Path(args.audit_dir))
    trades = enrich_trades(trades, opens, closes, bars, signals)
    open_lots = enrich_open_lots(unmatched, opens, bars, signals)

    fills.drop(columns=["raw"], errors="ignore").to_csv(out_dir / "alpaca_fills_normalized.csv", index=False)
    trades.to_csv(out_dir / "paired_option_trades.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    unmatched.drop(columns=["raw"], errors="ignore").to_csv(out_dir / "unmatched_option_lots.csv", index=False)
    open_lots.to_csv(out_dir / "open_option_lots_enriched.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    summaries = {
        "Ticker Filter Candidates": summarize_group(trades, "ticker", min_n=3),
        "Entry Time Buckets": summarize_group(trades, "entry_30m", min_n=3),
        "Direction": summarize_group(trades, "direction", min_n=3),
        "Option Type": summarize_group(trades, "option_type", min_n=3),
        "Exit Reason": summarize_group(trades, "exit_reason", min_n=3),
        "DTE At Entry": summarize_group(trades, "dte_at_entry", min_n=3),
        "Cluster Summary": cluster_trades(trades),
    }
    open_summaries = {
        "Ticker Timing": summarize_open_group(open_lots, "ticker", min_n=3),
        "Entry Time Buckets": summarize_open_group(open_lots, "entry_30m", min_n=3),
        "Direction": summarize_open_group(open_lots, "direction", min_n=3),
        "DTE At Entry": summarize_open_group(open_lots, "dte_at_entry", min_n=3),
    }
    for name, df in summaries.items():
        if not df.empty:
            safe = name.lower().replace(" ", "_")
            df.to_csv(out_dir / f"{safe}.csv", index=False)
    for name, df in open_summaries.items():
        if not df.empty:
            safe = "open_" + name.lower().replace(" ", "_")
            df.to_csv(out_dir / f"{safe}.csv", index=False)
    write_report(out_dir, fills, trades, unmatched, open_lots, summaries, open_summaries)
    print(
        f"fills={len(fills)} closed_trades={len(trades)} "
        f"unmatched={len(unmatched)} open_lots={len(open_lots)} out={out_dir}"
    )


if __name__ == "__main__":
    main()
