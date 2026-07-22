"""Dealer positioning dashboard and session controller."""
from __future__ import annotations

import argparse
import calendar
import json
import logging
import math
import queue as queue_mod
import threading
import time
import traceback
from collections import OrderedDict, deque
from datetime import date, datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from strategies.dealer_positioning.config import DealerPositioningConfig

logger = logging.getLogger(__name__)
_ET = ZoneInfo("America/New_York")
# chain_grid cache: bounded (LRU-evicted) + time-boxed so a long-running
# dashboard session doesn't serve the first-of-day chain forever, and a
# multi-symbol batch (capture_historical_snapshots) doesn't pin every chain
# fetched that run in RAM.
_CHAIN_CACHE_TTL_SECONDS = 120.0
_CHAIN_CACHE_MAX_ENTRIES = 8


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, (list, tuple, set, deque)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, datetime):
        return obj.astimezone(timezone.utc).isoformat()
    return str(obj)


class EventBroker:
    def __init__(self) -> None:
        self._subs: list[queue_mod.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue_mod.Queue:
        q: queue_mod.Queue = queue_mod.Queue(maxsize=1000)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue_mod.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, event: dict) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue_mod.Full:
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except Exception:
                    pass


class DealerDashboardStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = "idle"
        self._error: str | None = None
        self._started_at: str | None = None
        self._stopped_at: str | None = None
        self._snapshot: dict[str, Any] = {}
        self._events: deque[dict] = deque(maxlen=300)

    def reset(self) -> None:
        with self._lock:
            self._state = "running"
            self._error = None
            self._started_at = _utc_iso()
            self._stopped_at = None
            self._snapshot = {}
            self._events.clear()

    def set_state(self, state: str, *, error: str | None = None) -> None:
        with self._lock:
            self._state = state
            self._error = error
            if state in {"stopped", "error"}:
                self._stopped_at = _utc_iso()

    def update_snapshot(self, snapshot: dict[str, Any]) -> None:
        with self._lock:
            self._snapshot = snapshot

    def append_event(self, event: dict) -> None:
        with self._lock:
            self._events.appendleft(event)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": self._state,
                "error": self._error,
                "started_at": self._started_at,
                "stopped_at": self._stopped_at,
                "runner": _json_safe(self._snapshot),
                "events": list(self._events),
            }


class DealerDashboardApp:
    def __init__(self, *, bar_queue: queue_mod.Queue | None = None) -> None:
        self.store = DealerDashboardStore()
        self.broker = EventBroker()
        self._bar_queue = bar_queue
        # value: (fetched_at monotonic ts, chain payload)
        self._chain_cache: "OrderedDict[tuple[str, str], tuple[float, dict[str, Any]]]" = OrderedDict()
        self._runner = None
        self._thread: threading.Thread | None = None
        self._meta_thread: threading.Thread | None = None
        self._meta_stop = threading.Event()
        self._lock = threading.Lock()

    def snapshot(self) -> dict:
        snap = self.store.snapshot()
        snap["session_alive"] = self.is_running()
        return snap

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._thread and self._thread.is_alive())

    def start(self, payload: dict) -> dict:
        if self.is_running():
            raise RuntimeError("Dealer positioning session already running.")
        from strategies.dealer_positioning.runner import DealerPositioningRunner

        config = DealerPositioningConfig.from_env()
        symbols = payload.get("symbols")
        if symbols:
            if isinstance(symbols, str):
                parsed_symbols = tuple(x.strip().upper() for x in symbols.split(",") if x.strip())
            else:
                parsed_symbols = tuple(str(x).strip().upper() for x in symbols if str(x).strip())
            config = DealerPositioningConfig(**{**config.__dict__, "symbols": parsed_symbols})
        if payload.get("poll_seconds"):
            config = DealerPositioningConfig(**{**config.__dict__, "poll_seconds": int(payload["poll_seconds"])})
        if "submit_orders" in payload:
            config = DealerPositioningConfig(**{**config.__dict__, "submit_orders": bool(payload["submit_orders"])})
        if payload.get("alpaca_env_file"):
            config = DealerPositioningConfig(**{**config.__dict__, "alpaca_env_file": str(payload["alpaca_env_file"])})

        self.store.reset()
        runner = DealerPositioningRunner(config=config, event_sink=self._on_event, bar_queue=self._bar_queue)
        self._runner = runner
        self.store.update_snapshot(runner.snapshot())

        self._meta_stop.clear()
        self._meta_thread = threading.Thread(target=self._meta_loop, daemon=True, name="DealerPositioningMeta")
        self._meta_thread.start()

        def _run() -> None:
            try:
                runner.start()
            except Exception as exc:
                err = f"runner_crashed: {exc}"
                logger.error("%s\n%s", err, traceback.format_exc(limit=12))
                self.store.set_state("error", error=err)
                self._publish({"type": "error", "ts": _utc_iso(), "payload": {"error": err}})
            finally:
                self._meta_stop.set()
                if self.store.snapshot()["state"] != "error":
                    self.store.set_state("stopped")
                self._publish({"type": "session_ended", "ts": _utc_iso(), "payload": {}})

        thread = threading.Thread(target=_run, daemon=True, name="DealerPositioningRunner")
        with self._lock:
            self._thread = thread
        thread.start()
        self._publish({"type": "session_started", "ts": _utc_iso(), "payload": {"config": runner.snapshot()["config"]}})
        return self.snapshot()

    def stop(self) -> dict:
        runner = self._runner
        if runner is not None:
            runner.stop()
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout=10)
        self._meta_stop.set()
        if self._meta_thread is not None:
            self._meta_thread.join(timeout=2)
        self.store.set_state("stopped")
        return self.snapshot()

    def chain_grid(
        self,
        *,
        symbol: str,
        dte_offsets: tuple[int, ...] | None = None,
        strike_window_pct: float | None = None,
        expiration_scope: str | None = None,
        ref_date: date | None = None,
    ) -> dict:
        """Fetch a one-off read-only option-chain grid for an arbitrary ticker."""
        from strategies.dealer_positioning.chain import parse_schwab_option_chain, trading_expiration_buckets
        from strategies.dealer_positioning.levels import compute_gamma_levels
        from strategies.dealer_positioning.schwab_adapter import SchwabDealerDataClient

        clean_symbol = "".join(ch for ch in str(symbol or "").upper().strip() if ch.isalnum() or ch in {".", "-"})
        if not clean_symbol:
            raise ValueError("symbol is required")
        config = DealerPositioningConfig.from_env()
        updates: dict[str, Any] = {"symbols": (clean_symbol,)}
        if dte_offsets is not None:
            updates["dte_offsets"] = tuple(int(x) for x in dte_offsets)
        if strike_window_pct is not None:
            updates["strike_window_pct"] = float(strike_window_pct)
        config = DealerPositioningConfig(**{**config.__dict__, **updates})

        now = datetime.now(timezone.utc)
        today_et = now.astimezone(_ET).date()
        ref_date = ref_date or today_et
        query_ref_date = max(ref_date, today_et)
        client = SchwabDealerDataClient(config)
        scope = _clean_expiration_scope(expiration_scope)
        if scope:
            chain_key = (clean_symbol, query_ref_date.isoformat())
            cached = self._chain_cache.get(chain_key)
            now_mono = time.monotonic()
            if cached is not None and (now_mono - cached[0]) <= _CHAIN_CACHE_TTL_SECONDS:
                chain = cached[1]
                self._chain_cache.move_to_end(chain_key)
            else:
                chain = client.get_option_chain(
                    clean_symbol,
                    query_ref_date,
                    from_date=query_ref_date,
                    to_date=query_ref_date + timedelta(days=120),
                )
                self._chain_cache[chain_key] = (now_mono, chain)
                self._chain_cache.move_to_end(chain_key)
                while len(self._chain_cache) > _CHAIN_CACHE_MAX_ENTRIES:
                    self._chain_cache.popitem(last=False)
            available_expirations = _available_expirations_from_chain(chain)
            expirations = _select_expirations_for_scope(available_expirations, scope, ref_date=ref_date)
            if not expirations:
                raise RuntimeError(f"option chain had no expirations for scope {scope}")
            expiration_buckets = {expiration: idx for idx, expiration in enumerate(expirations)}
        else:
            expiration_buckets = trading_expiration_buckets(ref_date, config.dte_offsets)
            expirations = tuple(sorted(expiration_buckets))
            chain = client.get_option_chain(clean_symbol, ref_date)
            available_expirations = _available_expirations_from_chain(chain) or expirations
        spot, rows = parse_schwab_option_chain(
            chain,
            symbol=clean_symbol,
            timestamp=now,
            expiration_buckets=expiration_buckets,
        )
        if spot is None:
            spot = client.get_quote_price(clean_symbol)
        if spot is None:
            raise RuntimeError("missing spot price from option chain and quote")
        rows = [
            row for row in rows
            if abs(float(row.strike) - float(spot)) <= float(spot) * float(config.strike_window_pct)
        ]
        if not rows:
            raise RuntimeError("option chain had no rows inside configured strike window")

        ladder, levels = compute_gamma_levels(
            rows,
            symbol=clean_symbol,
            spot=float(spot),
            magnet_quantile=config.magnet_quantile,
        )
        grid_rows = _ladder_grid_rows(ladder, levels.to_dict())
        return {
            "symbol": clean_symbol,
            "ref_date_et": ref_date.isoformat(),
            "chain_query_date_et": query_ref_date.isoformat(),
            "fetched_at": now.isoformat(),
            "spot": float(spot),
            "dte_offsets": list(config.dte_offsets),
            "strike_window_pct": float(config.strike_window_pct),
            "expiration_scope": scope or "dte",
            "expiration_scope_label": _expiration_scope_label(scope),
            "available_expirations": list(available_expirations),
            "expirations": list(expirations),
            "levels": levels.to_dict(),
            "rows": grid_rows,
        }

    def _meta_loop(self) -> None:
        while not self._meta_stop.wait(2.0):
            runner = self._runner
            if runner is None:
                continue
            try:
                snap = runner.snapshot()
                self.store.update_snapshot(snap)
                self._publish({"type": "meta_tick", "ts": _utc_iso(), "payload": snap})
            except Exception:
                pass

    def _on_event(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {"type": event_type, "ts": _utc_iso(), "payload": payload}
        self.store.append_event(event)
        if self._runner is not None:
            self.store.update_snapshot(self._runner.snapshot())
        self._publish(event)

    def _publish(self, event: dict) -> None:
        self.broker.publish(event)


def _ladder_grid_rows(ladder, levels: dict[str, Any]) -> list[dict[str, Any]]:
    if ladder.empty:
        return []
    frame = ladder.sort_values("strike").reset_index(drop=True).copy()
    strikes = [float(x) for x in frame["strike"].tolist()]
    gamma_flip_row = _nearest_level_strike(strikes, levels.get("gamma_flip"))
    rows: list[dict[str, Any]] = []
    for raw in frame.to_dict(orient="records"):
        strike = float(raw.get("strike"))
        tags = _strike_tags(strike, levels, gamma_flip_row)
        raw["tags"] = tags
        raw["zone"] = _strike_zone(tags)
        rows.append(raw)
    return rows


def _clean_expiration_scope(scope: str | None) -> str | None:
    clean = str(scope or "").strip().lower().replace("-", "_")
    if not clean or clean in {"dte", "dtes"}:
        return None
    aliases = {
        "next": "next_expiration",
        "next_expiration": "next_expiration",
        "expiration": "next_expiration",
        "weekly": "next_expiration",
        "next_weekly": "next_expiration",
        "daily": "daily_week",
        "daily_week": "daily_week",
        "through_week": "daily_week",
        "week": "daily_week",
        "through_month": "through_month",
        "month": "through_month",
        "monthly": "through_month",
        "next_monthly": "through_month",
        "two_months": "two_months",
        "next_two_months": "two_months",
        "2_months": "two_months",
        "two": "two_months",
        "three_months": "two_months",
        "next_three_months": "two_months",
        "3_months": "two_months",
        "three": "two_months",
        "three_exp": "two_months",
        "three_expirations": "two_months",
    }
    if clean not in aliases:
        raise ValueError(f"unsupported expiration scope: {scope}")
    return aliases[clean]


def _expiration_scope_label(scope: str | None) -> str:
    return {
        None: "Trading-day DTE",
        "next_expiration": "Next expiration",
        "daily_week": "Through next weekly expiration",
        "through_month": "Through next monthly expiration",
        "two_months": "Through next two months",
    }.get(scope, str(scope))


def _available_expirations_from_chain(chain: dict[str, Any]) -> tuple[str, ...]:
    expirations: set[str] = set()
    if isinstance(chain.get("options"), list):
        for item in chain["options"]:
            if isinstance(item, dict):
                raw = item.get("expirationDate") or item.get("expiration")
                if _parse_date(raw):
                    expirations.add(str(raw)[:10])
    for side_key in ("callExpDateMap", "putExpDateMap"):
        side_map = chain.get(side_key) or {}
        if not isinstance(side_map, dict):
            continue
        for raw_key in side_map:
            raw_date = str(raw_key).split(":")[0]
            if _parse_date(raw_date):
                expirations.add(raw_date)
    return tuple(sorted(expirations))


def _select_expirations_for_scope(
    expiration_dates: list[str] | tuple[str, ...],
    scope: str,
    *,
    ref_date: date,
) -> tuple[str, ...]:
    dates = sorted(d for d in (_parse_date(x) for x in expiration_dates) if d is not None and d >= ref_date)
    if not dates:
        return ()
    scope = _clean_expiration_scope(scope) or "next_expiration"
    if scope == "next_expiration":
        return (dates[0].isoformat(),)
    if scope == "two_months":
        cutoff = _add_months(ref_date, 2)
        selected = [d for d in dates if d <= cutoff]
        if not selected:
            selected = [dates[0]]
        return tuple(d.isoformat() for d in selected)

    if scope == "daily_week":
        target = _next_weekly_expiration(dates)
        if target is None:
            target = dates[0]
        selected = [d for d in dates if d <= target]
        return tuple(d.isoformat() for d in selected)

    if scope == "through_month":
        target = _next_monthly_expiration(dates)
        if target is None:
            target = dates[min(2, len(dates) - 1)]
        return tuple(d.isoformat() for d in dates if d <= target)

    raise ValueError(f"unsupported expiration scope: {scope}")


def _next_weekly_expiration(expiration_dates: list[date]) -> date | None:
    friday_non_monthly = [d for d in expiration_dates if d.weekday() == 4 and not _is_monthly_expiration(d)]
    if friday_non_monthly:
        return friday_non_monthly[0]
    fridays = [d for d in expiration_dates if d.weekday() == 4]
    return fridays[0] if fridays else None


def _next_monthly_expiration(expiration_dates: list[date]) -> date | None:
    monthlies = [d for d in expiration_dates if _is_monthly_expiration(d)]
    return monthlies[0] if monthlies else None


def _is_monthly_expiration(value: date) -> bool:
    if value.weekday() != 4:
        return False
    return 15 <= value.day <= 21


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _add_months(value: date, months: int) -> date:
    month_index = value.month - 1 + int(months)
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _nearest_level_strike(strikes: list[float], level: Any) -> float | None:
    try:
        target = float(level)
    except (TypeError, ValueError):
        return None
    if not strikes:
        return None
    return min(strikes, key=lambda s: abs(s - target))


def _same_level(strike: float, level: Any) -> bool:
    try:
        return abs(float(strike) - float(level)) < 1e-9
    except (TypeError, ValueError):
        return False


def _strike_tags(strike: float, levels: dict[str, Any], gamma_flip_row: float | None) -> list[str]:
    tags: list[str] = []
    if _same_level(strike, levels.get("put_wall")):
        tags.append("floor")
        tags.append("put wall")
    if _same_level(strike, levels.get("call_wall")):
        tags.append("ceiling")
        tags.append("call wall")
    if _same_level(strike, levels.get("nearest_magnet")):
        tags.append("magnet")
    if _same_level(strike, levels.get("next_magnet_above")):
        tags.append("magnet above")
    if _same_level(strike, levels.get("next_magnet_below")):
        tags.append("magnet below")
    if _same_level(strike, levels.get("vega_wall")):
        tags.append("vega wall")
    if _same_level(strike, levels.get("next_vega_wall_above")):
        tags.append("vega above")
    if _same_level(strike, levels.get("next_vega_wall_below")):
        tags.append("vega below")
    if gamma_flip_row is not None and _same_level(strike, gamma_flip_row):
        tags.append("gamma flip")
    return list(dict.fromkeys(tags))


def _strike_zone(tags: list[str]) -> str:
    if "floor" in tags:
        return "floor"
    if "ceiling" in tags:
        return "ceiling"
    if "magnet" in tags or "magnet above" in tags or "magnet below" in tags:
        return "magnet"
    if "vega wall" in tags or "vega above" in tags or "vega below" in tags:
        return "vega"
    if "gamma flip" in tags:
        return "flip"
    return ""


class DealerDashboardHTTPServer(ThreadingHTTPServer):
    app: DealerDashboardApp


class DealerDashboardHandler(BaseHTTPRequestHandler):
    server_version = "CynolycusDealerDashboard/1.0"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        if self.path.startswith(("/api/events", "/api/state")):
            return
        super().log_message(format, *args)

    def _app(self) -> DealerDashboardApp:
        server = self.server
        if not isinstance(server, DealerDashboardHTTPServer):
            raise RuntimeError("Handler attached to unexpected server type.")
        return server.app

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8")) if raw else {}

    def _write_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(_json_safe(payload)).encode("utf-8")
        try:
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _write_text(self, body: str, *, status: int = HTTPStatus.OK, content_type: str = "text/plain") -> None:
        blob = body.encode("utf-8")
        try:
            self.send_response(int(status))
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(blob)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(blob)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def do_GET(self) -> None:  # noqa: N802
        from urllib.parse import urlparse

        parsed = urlparse(self.path)
        app = self._app()
        if parsed.path in {"/", "/index.html", "/dealer"}:
            index_path = Path(__file__).resolve().parent / "dealer_positioning_index.html"
            if not index_path.exists():
                self._write_text("Missing UI/dealer_positioning_index.html", status=HTTPStatus.NOT_FOUND)
                return
            from UI.ui_chrome import NAV_HTML
            html = index_path.read_text(encoding="utf-8").replace("<!--CYNO_NAV-->", NAV_HTML)
            self._write_text(html, content_type="text/html")
            return
        if parsed.path == "/static/cynolycus_theme.css":
            from UI.ui_chrome import serve_theme_css
            serve_theme_css(self)
            return
        if parsed.path == "/api/state":
            self._write_json(app.snapshot())
            return
        if parsed.path == "/api/chain-grid":
            from urllib.parse import parse_qs

            params = parse_qs(parsed.query)
            symbol = (params.get("symbol") or [""])[0]
            dte_raw = (params.get("dtes") or [""])[0]
            dtes = None
            if dte_raw.strip():
                dtes = tuple(int(x.strip()) for x in dte_raw.split(",") if x.strip())
            window_raw = (params.get("window_pct") or [""])[0]
            window_pct = float(window_raw) if window_raw.strip() else None
            scope = (params.get("scope") or [""])[0] or None
            try:
                self._write_json(
                    app.chain_grid(
                        symbol=symbol,
                        dte_offsets=dtes,
                        strike_window_pct=window_pct,
                        expiration_scope=scope,
                    )
                )
            except Exception as exc:
                self._write_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if parsed.path == "/api/events":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = app.broker.subscribe()
            try:
                hello = {"type": "hello", "ts": _utc_iso(), "payload": app.snapshot()}
                self.wfile.write(f"data: {json.dumps(_json_safe(hello))}\n\n".encode("utf-8"))
                self.wfile.flush()
                while True:
                    try:
                        event = q.get(timeout=12.0)
                    except queue_mod.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    self.wfile.write(f"data: {json.dumps(_json_safe(event))}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                app.broker.unsubscribe(q)
            return
        self._write_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        from urllib.parse import urlparse

        app = self._app()
        parsed = urlparse(self.path)
        try:
            payload = self._read_json_body()
            if parsed.path == "/api/start":
                self._write_json({"ok": True, "state": app.start(payload)})
                return
            if parsed.path == "/api/stop":
                self._write_json({"ok": True, "state": app.stop()})
                return
            self._write_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._write_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    parser = argparse.ArgumentParser(description="Run dealer positioning dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()
    app = DealerDashboardApp()
    server = DealerDashboardHTTPServer((args.host, args.port), DealerDashboardHandler)
    server.app = app
    logger.info("Dealer positioning dashboard: http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        app.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
