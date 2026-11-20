"""Personal portfolio page: the user's Schwab account, read-only, with evidence-based flags.

No orders. Schwab is read through GET endpoints only (core.personal_portfolio.holdings),
and only while this page is open: ``/api/portfolio`` triggers a background sync when the
last one is older than ``holdings_ttl``. ``/api/state`` (polled by the hub) never calls
Schwab and returns no balances. Watchlist edits are the only writes, to
Data/personal/watchlist.csv.

Trade history (``/api/history``) reads Schwab transactions through GET only, on its own slower
throttle (``history_ttl``), rebuilds round-trip episodes, and fills a bar cache for the charts
(core.personal_portfolio.trade_history / trade_bars / trade_chart / trade_analysis).

Because the page shows a real-money account, requests must carry a loopback Host header
(blocks DNS rebinding) and writes must be JSON (blocks cross-site form posts).
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from core.personal_portfolio import analysis, data, trade_analysis, trade_bars, trade_chart, trade_history, trade_timing
from core.personal_portfolio.holdings import fetch_holdings
from UI.ui_chrome import NAV_HTML, THEME_LINK, serve_theme_css

logger = logging.getLogger(__name__)
DEFAULT_PORT = 8778
MAX_BODY = 4096
_LOOPBACK = {"127.0.0.1", "localhost", "::1", "[::1]"}


def _token_status() -> dict:
    try:
        from core.schwab_token_status import schwab_token_status

        st = schwab_token_status()
        return {"message": st.message, "severity": st.severity,
                "expires_at": st.expires_at.isoformat() if st.expires_at else None}
    except Exception as exc:  # noqa: BLE001 - token status is advisory
        return {"message": f"token status unavailable: {exc}", "severity": "warning", "expires_at": None}


def _sync_transactions_live(now: datetime, base: Path | None, full: bool) -> dict:
    from core.API.Schwab_API.schwab_client import SchwabClient

    return trade_history.sync_transactions(SchwabClient().client, now, base, full)


def _held_symbols(holdings: dict | None) -> list[str]:
    out = set()
    for acct in (holdings or {}).get("accounts") or []:
        for p in acct.get("positions") or []:
            sym = p.get("underlying") if p.get("kind") == "option" else p.get("symbol") if p.get("kind") in ("equity", "etf") else None
            if sym:
                out.add(sym)
    return sorted(out)


class PersonalDashboardApp:
    def __init__(self, *, base: Path | None = None, holdings_fetcher: Callable[..., dict] = fetch_holdings,
                 bars_refresher: Callable[..., dict] = data.refresh_bars,
                 earnings_refresher: Callable[..., dict] = data.refresh_earnings,
                 momentum_loader: Callable[[], dict] = data.load_momentum_context,
                 themes_loader: Callable[[list[str]], dict] = data.load_themes,
                 token_status: Callable[[], dict] = _token_status,
                 transactions_syncer: Callable[..., dict] = _sync_transactions_live,
                 daily_refresher: Callable[..., dict] = trade_bars.refresh_daily,
                 intraday_refresher: Callable[..., dict] = trade_bars.refresh_intraday,
                 chart_fetch: Callable | None = None,
                 holdings_ttl: float = 60.0, history_ttl: float = 900.0,
                 clock: Callable[[], datetime] | None = None) -> None:
        self.base = base
        self._fetch, self._bars, self._earnings = holdings_fetcher, bars_refresher, earnings_refresher
        self._momentum, self._themes, self._token = momentum_loader, themes_loader, token_status
        self.holdings_ttl = holdings_ttl
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._report: dict | None = None
        self._status: dict[str, Any] = {"state": "never", "phase": None, "error": None, "synced_at": None}
        self._last_start: float | None = None
        self._market_key: tuple | None = None
        self._market_status: dict = {}
        self._last_holdings: dict | None = None
        self._tx_sync, self._daily, self._intraday = transactions_syncer, daily_refresher, intraday_refresher
        self._chart_fetch, self.history_ttl = chart_fetch, history_ttl
        self._hist_thread: threading.Thread | None = None
        self._hist_last_start: float | None = None
        self._hist_report: dict | None = None
        self._hist_episodes: list[dict] = []
        self._hist_stats: dict | None = None
        self._hist_status: dict[str, Any] = {"state": "never", "phase": None, "error": None, "synced_at": None}

    # ---- read ----
    def state(self) -> dict:
        """Cheap summary for the hub and the nav. No network, no balances."""
        with self._lock:
            rep, st = self._report, dict(self._status)
        out = {"config": {"mode": "personal", "tradeable": False, "read_only": True}, **st}
        if rep:
            out.update(n_positions=len(rep["positions"]), n_watchlist=len(rep["watchlist"]),
                       n_flags={s: sum(f["severity"] == s for f in rep["flags"]) for s in ("high", "warn", "info")})
        return out

    def portfolio(self, force: bool = False) -> dict:
        self._ensure_sync(force)
        with self._lock:
            return {"status": dict(self._status), "report": self._report, "token": self._token()}

    # ---- watchlist ----
    def add_watch(self, ticker: str, note: str = "") -> dict:
        """One ticker, or several separated by commas/spaces (a pasted list). All are validated first."""
        tickers = [data.normalize_ticker(t) for t in re.split(r"[\s,;]+", str(ticker or "").strip()) if t]
        if not tickers:
            raise ValueError("ticker required")
        if len(tickers) > 100:
            raise ValueError("at most 100 tickers at once")
        for t in dict.fromkeys(tickers):
            data.add_to_watchlist(t, note, self.base)
        return self.portfolio(force=True)

    def remove_watch(self, ticker: str) -> dict:
        data.remove_from_watchlist(ticker, self.base)
        return self.portfolio(force=True)

    # ---- sync ----
    def _ensure_sync(self, force: bool) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            fresh = self._last_start is not None and time.monotonic() - self._last_start < self.holdings_ttl
            if fresh and not force:
                return
            self._last_start = time.monotonic()
            self._status.update(state="syncing", phase="schwab", error=None)
            self._thread = threading.Thread(target=self._sync, daemon=True, name="personal-portfolio-sync")
            self._thread.start()

    def _build(self, holdings: dict, watch: list[dict], overrides: dict, cfg: data.PersonalConfig,
               tickers: list[str], now: datetime) -> dict:
        with self._lock:
            hist_stats = self._hist_stats
        rep = analysis.build_report(holdings, data.load_bars(tickers, self.base), data.load_earnings(self.base),
                                    self._momentum(), self._themes(tickers), watch, overrides, cfg, now, hist_stats)
        rep["data"]["refresh"] = self._market_status
        return analysis.sanitize(rep)

    def _publish(self, rep: dict, phase: str | None) -> None:
        with self._lock:
            self._report = rep
            self._status.update(state="syncing" if phase else "ready", phase=phase,
                                synced_at=rep["fetched_at"], error=None)

    def _sync(self) -> None:
        try:
            now = self._clock()
            cfg, overrides = data.load_config(self.base), data.load_look_through_overrides(self.base)
            watch = data.read_watchlist(self.base)
            holdings = self._fetch(extra_symbols=[w["ticker"] for w in watch] + ["SPY"])
            with self._lock:
                self._last_holdings = holdings
            req = analysis.required_tickers(holdings, watch, overrides)
            key = (tuple(req["bars"]), data.last_completed_session(now))
            stale = key != self._market_key
            self._publish(self._build(holdings, watch, overrides, cfg, req["bars"], now), "market data" if stale else None)
            if stale:
                bars = self._bars(req["bars"], now, self.base)
                earn = self._earnings(req["earnings"], now, self.base)
                self._market_status = {"at": now.isoformat(), "bars": bars, "earnings": earn,
                                       "errors": sorted(t for t, s in bars.items() if s.startswith("error"))}
                self._market_key = key
                self._publish(self._build(holdings, watch, overrides, cfg, req["bars"], now), None)
            self._write_logs(now, ((holdings.get("quotes") or {}).get("SPY") or {}).get("mark"))
        except Exception as exc:  # noqa: BLE001 - keep the last good report on screen
            logger.exception("Personal portfolio sync failed")
            with self._lock:
                self._status.update(state="error", phase=None, error=str(exc)[:500])

    def _write_logs(self, now: datetime, spy_mark: float | None) -> None:
        with self._lock:
            rep = self._report
        if not rep:
            return
        day = now.astimezone(data.ET).date().isoformat()
        if rep["totals"]["net_liq"]:
            data.upsert_equity_log(day, rep["totals"]["net_liq"], spy_mark, self.base)
        if rep["data"]["bars_current"]:
            data.append_flag_log(analysis.flag_log_records(rep), day, self.base)

    # ---- trade history ----
    def history(self, force: bool = False, full: bool = False) -> dict:
        self._ensure_history(force or full, full)
        with self._lock:
            return {"status": dict(self._hist_status), "report": self._hist_report}

    def history_chart(self, episode_id: str | None = None, ticker: str | None = None, timeframe: str = "1d") -> dict:
        with self._lock:
            episodes = self._hist_episodes
        if episode_id:
            ep = next((e for e in episodes if e["id"] == episode_id), None)
            if ep is None:
                raise KeyError(f"unknown trade {episode_id!r}")
            if timeframe not in ("1d", "intraday"):
                raise ValueError("timeframe must be 1d or intraday")
            return trade_chart.episode_chart(ep, timeframe, self._clock(), self.base, self._chart_fetch)
        if ticker:
            return trade_chart.ticker_chart(data.normalize_ticker(ticker), episodes, self.base)
        raise ValueError("id or ticker required")

    def _ensure_history(self, force: bool, full: bool) -> None:
        with self._lock:
            if self._hist_thread and self._hist_thread.is_alive():
                return
            fresh = self._hist_last_start is not None and time.monotonic() - self._hist_last_start < self.history_ttl
            if fresh and not force:
                return
            self._hist_last_start = time.monotonic()
            self._hist_status.update(state="syncing", phase="schwab history", error=None)
            self._hist_thread = threading.Thread(target=self._history_sync, args=(full,), daemon=True,
                                                 name="personal-history-sync")
            self._hist_thread.start()

    def _hist_publish(self, episodes: list[dict], meta: dict, phase: str | None) -> None:
        rows = [{k: v for k, v in e.items() if k != "fills"} for e in episodes]
        stats = trade_timing.comeback_stats(episodes)
        rep = analysis.sanitize({"generated_at": self._clock().isoformat(), "episodes": rows, "comeback": stats,
                                 "summary": trade_analysis.summarize(trade_analysis.resolved(episodes)), **meta})
        with self._lock:
            self._hist_report, self._hist_episodes, self._hist_stats = rep, episodes, stats
            self._hist_status.update(state="syncing" if phase else "ready", phase=phase,
                                     synced_at=rep["generated_at"], error=None)

    def _history_sync(self, full: bool) -> None:
        try:
            now = self._clock()
            tx = self._tx_sync(now, self.base, full)
            fills, issues = trade_history.normalize_fills(trade_history.load_raw(self.base))
            episodes, checks = trade_history.build_episodes(fills, trade_history.load_symbol_map(self.base))
            with self._lock:
                holdings = self._last_holdings
            if holdings is None:
                holdings = self._fetch(extra_symbols=[])
            rec = trade_history.attach_marks(episodes, holdings)
            meta = {"transactions": tx, "issues": issues, "checks": checks, "reconciliation": rec,
                    "first_trade": episodes[0]["open_date"] if episodes else None}
            trade_analysis.enrich(episodes, self.base, earnings=data.load_earnings(self.base))
            self._hist_publish(episodes, meta, "daily bars")
            extra = sorted(set([w["ticker"] for w in data.read_watchlist(self.base)] + _held_symbols(holdings) + ["SPY"]))
            plan = trade_bars.bar_plan(episodes, extra, now)
            daily = self._daily(plan, now, self.base)
            meta["bars"] = {"daily": daily, "daily_errors": sorted(s for s, v in daily.items() if v != "current"
                                                                   and v != "fetched"), "chart_symbols": sorted(plan)}
            # Earnings dates once per traded stock never fetched before (past report dates do not change;
            # held and watchlist names are refreshed daily by the portfolio sync).
            have = set(data.load_earnings(self.base)["ticker"])
            stocks = sorted({e["chart_symbol"] for e in episodes if trade_analysis.underlying_class(e) == "stock"} - have)
            meta["earnings_fetch"] = self._earnings(stocks, now, self.base) if stocks else {}
            trade_analysis.enrich(episodes, self.base, earnings=data.load_earnings(self.base))
            self._hist_publish(episodes, meta, "intraday bars")
            intraday = self._intraday(episodes, now, self.base)
            meta["bars"]["intraday_errors"] = sorted(k for k, v in intraday.items() if v.startswith("error"))
            trade_analysis.enrich(episodes, self.base, earnings=data.load_earnings(self.base))
            self._hist_publish(episodes, meta, None)
        except Exception as exc:  # noqa: BLE001 - keep the last good history on screen
            logger.exception("Personal trade-history sync failed")
            with self._lock:
                self._hist_status.update(state="error", phase=None, error=str(exc)[:500])


PAGE = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Personal Portfolio — Cynolycus</title>__THEME__
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
main{max-width:1500px;margin:auto;padding:18px 16px}h1{margin:0;font-size:24px}h2{font-size:15px;margin:0}
.top{display:flex;justify-content:space-between;align-items:flex-end;gap:12px;flex-wrap:wrap;margin-bottom:12px}
.sub{color:var(--muted)}.small{font-size:11px;color:var(--muted)}
.banner{padding:9px 12px;border-radius:7px;margin:8px 0;border:1px solid var(--border);background:var(--panel)}
.banner.error{border-color:#5a1f1f;background:#2a1212}.banner.warning{border-color:#5a4a1f;background:#2a2412}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px;margin:12px 0}
.tile{background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);padding:10px 12px}
.tile .k{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.4px}.tile .v{font-size:20px;font-weight:600;font-variant-numeric:tabular-nums;margin-top:2px}
.tile .n{font-size:11px;color:var(--muted);margin-top:2px}
section.card{margin:12px 0}.card-body{padding:12px}.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:6px 8px;border-bottom:1px solid var(--border);text-align:left;white-space:nowrap;vertical-align:top}
th{background:var(--panel2);color:var(--muted);font-weight:600}.num{text-align:right;font-variant-numeric:tabular-nums}
.flag{display:flex;gap:10px;padding:8px 12px;border-bottom:1px solid var(--border);align-items:flex-start}
.sev{flex:none;font-size:10px;font-weight:700;text-transform:uppercase;padding:2px 7px;border-radius:9px;letter-spacing:.3px}
.sev.high{background:#3a1010;color:var(--red)}.sev.warn{background:#3a2d10;color:var(--yellow)}.sev.info{background:#1f3a4a;color:var(--blue)}
.tag{display:inline-block;font-size:10px;padding:1px 6px;border-radius:8px;background:var(--panel2);border:1px solid var(--border);color:var(--muted);margin-right:3px}
.tag.on{color:var(--green);border-color:#1f5a33}.tag.bad{color:var(--red);border-color:#5a1f1f}
form.watch{display:flex;gap:8px;flex-wrap:wrap;padding:10px 12px;border-bottom:1px solid var(--border)}
form.watch input[name=ticker]{width:110px;text-transform:uppercase}form.watch input[name=note]{flex:1;min-width:160px}
input[type=text]{background:var(--panel2);color:var(--text);border:1px solid var(--border);border-radius:5px;padding:6px 10px;font:inherit}
.evidence{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:800px){.evidence{grid-template-columns:1fr}}
.evidence ul{margin:6px 0;padding-left:18px}.evidence li{margin:4px 0}
.x{background:none;border:none;color:var(--muted);padding:0 4px;cursor:pointer}.x:hover{color:var(--red);background:none}
[data-tip]{cursor:help}th[data-tip],.k[data-tip],.card-head [data-tip]{text-decoration:underline dotted;text-decoration-color:var(--muted);text-underline-offset:3px}
select{background:var(--panel2);color:var(--text);border:1px solid var(--border);border-radius:5px;padding:5px 8px;font:inherit}
.hfilters{display:flex;gap:10px 14px;flex-wrap:wrap;padding:10px 12px;border-bottom:1px solid var(--border);align-items:center}
.hfilters label{display:flex;gap:6px;align-items:center;color:var(--muted);font-size:12px}.hfilters input{width:110px;text-transform:uppercase}
.chartbox{height:260px;margin:0 12px 10px}#tradechart{height:440px}.charthead{display:flex;gap:10px;flex-wrap:wrap;align-items:center;padding:10px 12px 6px}
.charthead .sp{flex:1}.seg button{padding:4px 10px;font-size:12px}.seg button.on{border-color:var(--accent);color:var(--accent)}
tr.clk{cursor:pointer}tr.clk:hover td{background:var(--panel2)}tr.sel td{background:var(--panel2);box-shadow:inset 0 0 0 9999px rgba(145,221,197,.06)}
.tscroll{max-height:440px;overflow:auto}.tscroll thead th{position:sticky;top:0;z-index:1}
.lnk{background:none;border:none;color:var(--accent);padding:0 4px;cursor:pointer;font-size:11px}.lnk:hover{text-decoration:underline;background:none}
#tipbox{position:fixed;z-index:200;display:none;max-width:340px;padding:8px 10px;border-radius:7px;background:#1b262c;border:1px solid #33444b;color:var(--text);font-size:12px;line-height:1.45;font-weight:400;text-transform:none;letter-spacing:0;white-space:normal;box-shadow:0 6px 18px rgba(0,0,0,.45);pointer-events:none}
</style></head><body>__NAV__<main>
<div class=top><div><h1>Personal Portfolio</h1>
<div class=sub>Schwab · real money · read-only: this page never places, changes, or cancels orders</div></div>
<div class=controls><span id=sync class=small>Loading…</span><button id=refresh>Refresh now</button></div></div>
<div id=banners></div>
<div id=tiles class=tiles></div>
<section class=card><div class=card-head><span data-tip=sec_flags>Review flags</span></div><div id=flags><div class=card-body>Loading…</div></div></section>
<section class=card><div class=card-head>Holdings</div><div class=scroll><table><thead><tr><th>Symbol</th><th>Kind</th><th class=num>Qty</th><th class=num data-tip=h_avg>Avg cost</th><th class=num data-tip=h_mark>Mark</th><th class=num data-tip=h_mv>Market value</th><th class=num data-tip=h_pct>% net liq</th><th class=num data-tip=day_pl>Day P/L</th><th class=num data-tip=open_pl>Open P/L</th><th data-tip=h_counts>Counts as</th></tr></thead><tbody id=holdings></tbody></table></div></section>
<section class=card><div class=card-head>Options</div><div class=scroll><table><thead><tr><th>Contract</th><th class=num data-tip=o_dte>DTE</th><th class=num data-tip=o_und>Underlying</th><th class=num data-tip=o_money>Moneyness</th><th class=num data-tip=o_delta>Delta</th><th class=num data-tip=o_theta>Theta $/day</th><th class=num data-tip=o_be>Move to breakeven</th><th class=num data-tip=o_spread>Spread</th><th class=num data-tip=o_ex>Exercise cost</th><th data-tip=o_earn>Earnings before expiry</th><th class=num data-tip=open_pl>Open P/L</th></tr></thead><tbody id=options></tbody></table></div></section>
<section class=card><div class=card-head><span data-tip=sec_look>Look-through exposure by underlying</span></div><div class=scroll><table><thead><tr><th>Underlying</th><th class=num data-tip=i_exp>Price exposure</th><th class=num data-tip=i_pct>% net liq</th><th class=num data-tip=i_cap>Capital</th><th data-tip=i_via>Held via</th><th class=num data-tip=i_beta>Beta SPY</th><th class=num data-tip=i_vol>Vol 63d</th><th class=num data-tip=i_sma>vs 200d avg</th><th data-tip=i_mom>Momentum (liquid 300)</th><th data-tip=i_earn>Earnings</th><th data-tip=i_theme>Theme (dynamic)</th></tr></thead><tbody id=instruments></tbody></table></div><div id=risknote class="card-body small"></div></section>
<section class=card><div class=card-head><span data-tip=sec_watch>Watchlist</span></div>
<form class=watch id=watchform><input type=text name=ticker placeholder="Ticker(s): paste a list, e.g. NVDA, AMD TSM" maxlength=600 required style="width:260px"><input type=text name=note placeholder="Note (optional)" maxlength=200><button class=primary type=submit>Add</button><span id=watcherr class=bad></span></form>
<div class=scroll><table><thead><tr><th>Ticker</th><th class=num data-tip=w_mark>Mark</th><th class=num data-tip=w_today>Today</th><th class=num data-tip=i_sma>vs 200d avg</th><th data-tip=i_mom>Momentum (liquid 300)</th><th class=num data-tip=i_beta>Beta SPY</th><th class=num data-tip=i_vol>Vol 63d</th><th data-tip=i_earn>Earnings</th><th data-tip=i_theme>Theme (dynamic)</th><th>Note</th><th></th></tr></thead><tbody id=watchlist></tbody></table></div></section>
<section class=card id=history><div class=card-head><span data-tip=sec_history>Trade history</span> <span id=hsync class=small></span></div>
<div class=hfilters>
<label>Type <select id=f_type><option value=all>All</option><option value=option>Options</option><option value=stock>Stocks &amp; ETFs</option></select></label>
<label>Side <select id=f_side><option value=all>Calls &amp; puts</option><option value=CALL>Calls</option><option value=PUT>Puts</option></select></label>
<label>Outcome <select id=f_outcome><option value=all>All</option><option value=win>Winners</option><option value=loss>Losers</option><option value=open>Open</option><option value=unresolved>Unresolved</option></select></label>
<label>Year <select id=f_year><option value=all>All</option></select></label>
<label>Underlying <input type=text id=f_sym list=f_symlist placeholder="any"><datalist id=f_symlist></datalist></label>
<span id=f_group_chip></span><span class=sp style="flex:1"></span>
<button id=h_refresh title="Fetch new Schwab transactions and missing bars">Refresh history</button></div>
<div id=htiles class=tiles style="margin:12px"></div>
<div class=scroll style="padding:0 12px 10px"><div class=small style="margin-bottom:4px"><span data-tip=sec_decomp>Where the option P&amp;L went (filtered, resolved option trades)</span></div><table><thead><tr><th>What happened</th><th class=num>Trades</th><th class=num>Net P&amp;L</th><th class=num>Median stock move by exit</th><th class=num>Median best move while held</th><th>Points at</th></tr></thead><tbody id=hdecomp></tbody></table></div>
<div class=small style="padding:0 12px" data-tip=sec_pnlcurve>Cumulative realized P&amp;L of the filtered trades, by close date</div>
<div id=pnlchart class=chartbox></div>
<div class=charthead><b id=ct_title data-tip=sec_tradechart>Trade chart</b><span id=ct_sub class=small>Click a trade in the table, or "chart" on a watchlist name.</span><span class=sp></span>
<span class=seg id=ct_tf><button data-tf=1d class=on>Daily</button><button data-tf=intraday>Intraday</button><button data-tf=ticker>All trades on this name</button></span></div>
<div id=tradechart class=chartbox></div>
<div id=ct_notes class="small" style="padding:0 12px 8px"></div>
<div id=ct_review class=card-body style="padding:4px 12px 8px;font-size:12px"></div>
<div id=ct_fills class=scroll style="padding:0 12px 10px"></div>
<div class="scroll tscroll"><table><thead><tr><th data-sort=open_time>Opened</th><th data-sort=key>Contract / symbol</th><th data-sort=b_type>Type</th><th class=num data-sort=max_qty>Qty</th><th class=num data-tip=ht_entry data-sort=avg_entry>Entry</th><th class=num data-tip=ht_exit data-sort=avg_exit>Exit</th><th class=num data-tip=ht_pnl data-sort=pnl>P&amp;L</th><th class=num data-tip=ht_ret data-sort=return_pct>Return</th><th class=num data-tip=ht_held data-sort=hold_days>Held</th><th class=num data-tip=ht_dte data-sort=dte_at_entry>DTE at entry</th><th data-sort=b_exit>Exit</th><th class=num data-tip=ht_move data-sort=under_move_dir>Underlying move</th><th class=num data-tip=ht_expiry data-sort=hold_to_expiry_pnl>If held to expiry</th><th></th></tr></thead><tbody id=htable></tbody></table></div>
<div class=card-body><div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap"><b data-tip=sec_breakdown>Breakdown</b><label class=small>group by <select id=f_group></select></label><span class=small>Closed trades with both ends in the record, after the filters above. Click a row to filter the table to it.</span></div>
<div class=scroll style="margin-top:8px"><table><thead><tr><th>Group</th><th class=num>Trades</th><th class=num data-tip=h_winrate>Win rate</th><th class=num data-tip=h_pnl>Net P&amp;L</th><th class=num data-tip=h_exp>Avg P&amp;L</th><th class=num data-tip=ht_ci>95% range of avg</th><th class=num data-tip=ht_ret>Median return</th><th class=num data-tip=h_pf>Profit factor</th><th class=num>Avg win</th><th class=num>Avg loss</th></tr></thead><tbody id=hbreak></tbody></table></div></div>
<div id=hnotes class="card-body small"></div></section>
<section class=card><div class=card-head><span data-tip=sec_sleeve>Research momentum sleeve</span></div><div id=sleeve class=card-body></div></section>
<section class=card><div class=card-head>What the evidence supports</div><div class="card-body evidence">
<div><b>Measured and usable here</b><ul>
<li><b>Momentum sleeve.</b> The top 20 by 12-1 momentum among the 300 most liquid stocks, rebalanced monthly, beat SPY by ~27%/yr in the 2019-2026 backtest that includes delisted names (95% CI +6 to +46). Its drawdowns reached −53 to −57%. As a 30% sleeve next to 70% SPY: Sharpe 1.04 and SPY-like drawdown. <span class=small>research/long_horizon_discount_2026-09-25</span></li>
<li><b>Shrugged-off earnings miss.</b> An EPS miss beyond −2% with a reaction within ±2% of SPY lagged SPY by 2.8% over the next 126 sessions (t = −4.3). It is a reason to review a name, not a short signal.</li>
<li><b>Exposure and sizing mechanics.</b> Look-through leverage, delta exposure, correlation, and option exercise cash are arithmetic, not forecasts. In the 2026 paper post-mortem, a per-name cap was the one control that beat random refusal. <span class=small>research/account_postmortem_2026-11-03</span></li>
</ul></div>
<div><b>Tested and not supported</b><ul>
<li>Entry/exit timing from dips, TMO turns, 200-day pullbacks, or near 52-week lows: no measurable edge; buying near the 52-week low underperformed.</li>
<li>ML rankers and fundamentals filters over 63-126 days: did not beat simple momentum; fundamentals made drawdowns worse.</li>
<li>Regime/trend overlays and drawdown caps: no better than refusing the same number of trades at random.</li>
<li>The momentum sleeve's forward shadow started 2026-10-05 and is short; see its record above before leaning on it.</li>
</ul><div class=small>Flags here inform review. They are not buy or sell instructions, and nothing on this page trades.</div></div>
</div></section>
<div id=footer class=small></div>
</main><script>
const $=s=>document.querySelector(s), esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=(v,d=0)=>v==null?'—':Number(v).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d});
const usd=(v,d=0)=>v==null?'—':(v<0?'−$':'$')+num(Math.abs(v),d);
const sgn=(v,d=0)=>v==null?'—':'<span class="'+(v>0?'pos':v<0?'neg':'')+'">'+(v>0?'+':v<0?'−':'')+'$'+num(Math.abs(v),d)+'</span>';
const pct=(v,d=1,signed=false)=>v==null?'—':(signed&&v>0?'+':'')+(v<0?'−':'')+num(Math.abs(v*100),d)+'%';
const spct=(v,d=1)=>v==null?'—':'<span class="'+(v>0?'pos':v<0?'neg':'')+'">'+pct(v,d,true)+'</span>';
const when=v=>v?new Date(v).toLocaleString():'—';
const TIPS={
 net_liq:'Net liquidation: what the account is worth if every position were sold at Schwab\'s current marks and any margin loan repaid (cash + positions − debt).',
 day_pl:'Day P/L: today\'s change in the value of your open positions at Schwab\'s marks. Deposits and withdrawals are not included.',
 open_pl:'Open P/L: current market value minus what you paid, for positions you still hold. It is unrealized: nothing is locked in until you sell.',
 avail:'Available funds: cash Schwab says you can use for new trades right now without new borrowing. Exercising an option draws on this.',
 gross:'Gross exposure: total price exposure across everything you hold, divided by net liquidation. At 1×, a 1% move in all your underlyings moves the account about 1%; at 3.6×, about 3.6%. Leveraged ETFs count at their leverage, options at their delta.',
 beta_exp:'SPY-beta exposure: how strongly the account has moved with the S&P 500. Each underlying\'s price exposure × its beta to SPY, summed, divided by net liquidation. 14.7× means a 1% SPY move has typically come with about a 14.7% account move. It is a small-move estimate: option deltas change as prices move.',
 bad_day:'Bad day (5%): replays the last year of daily price moves on today\'s positions. One day in 20 lost at least this much. Not a forecast: it assumes today\'s holdings and last year\'s market.',
 worst_day:'Worst day, 1y: the single worst day in that same replay, and when it happened.',
 n_eff:'Independent bets: how many unrelated positions the book behaves like once you account for how the underlyings move together. Five names that always move together count as about one.',
 h_avg:'Average price paid per share. For options it is per option share; one contract is 100 shares.',
 h_mark:'Schwab\'s current mark: the bid/ask midpoint for options, near the last trade for stocks and ETFs.',
 h_mv:'Market value: quantity × mark (× 100 for options). What the position is worth now.',
 h_pct:'Market value ÷ net liquidation: the share of the account\'s money in this position. This is money in, not price exposure.',
 h_counts:'What the position really tracks. A 2× ETF counts as 2× its stock, but only after checking that the two actually move together (corr and β shown). An option counts as its delta-equivalent dollar amount of the underlying (Δ $).',
 o_dte:'Days to expiry, in calendar days.',
 o_und:'The underlying\'s current price, from Schwab\'s option quote.',
 o_money:'Moneyness: how far the underlying is from the strike. Positive = in the money (for a call, price above strike). Negative = out of the money: worthless at expiry if it stays there.',
 o_delta:'Delta: about how much the option price moves per share for a $1 move in the underlying. Also a rough market-implied chance of finishing in the money (0.19 ≈ 19%).',
 o_theta:'Theta $/day: value lost each day from time passing alone, if nothing else changes. It speeds up as expiry gets closer.',
 o_be:'Move to breakeven: how much the underlying must move by expiry for the option to be worth what you paid (strike + premium for a call, strike − premium for a put).',
 o_spread:'Spread: (ask − bid) ÷ mark. Selling at the bid gives up about half of this compared with the mark.',
 o_ex:'Exercise cost: cash needed to take the shares if the option is exercised (strike × 100 × contracts). Schwab auto-exercises options $0.01 or more in the money at expiry unless you close them first.',
 o_earn:'The underlying reports earnings before this option expires. Expect a big move and an implied-volatility drop (IV crush) that cuts option prices.',
 i_exp:'Price exposure (also called price sensitivity or delta-adjusted notional): the dollar size of your bet on this underlying. 1% of it is roughly your P/L for a 1% move. ETF: market value × leverage. Option: delta × 100 × contracts × underlying price.',
 i_pct:'Price exposure ÷ net liquidation. 173% means a 1% move in this name moves the account about 1.7%.',
 i_cap:'Capital: the money actually in those positions (their market value). For a long option this is the most you can lose; its price exposure can be far larger.',
 i_via:'The positions that create this exposure.',
 i_beta:'Beta to SPY over the last year of daily returns: how many percent this name has moved, on average, per 1% SPY move.',
 i_vol:'Volatility: annualized standard deviation of daily returns over the last 63 sessions (about 3 months). 100% ≈ a typical daily move of ±6%; 30% ≈ ±2%.',
 i_sma:'Last close vs the 200-day average price. Descriptive only: the research found no timing edge in 200-day pullback rules.',
 i_mom:'12-1 momentum: the return from 12 months ago to 1 month ago, ranked among the 300 most liquid US stocks at the last research decision. "In sleeve" = one of the 20 names the research list holds. Ranks refresh every 4 weeks.',
 i_earn:'Next report date (Yahoo). "Shrugged miss" = EPS missed by more than 2% but the stock moved less than 2% vs SPY over two days; in research, such names lagged SPY by 2.8% over about 6 months.',
 i_theme:'Label from the dynamic theme pipeline. Context only: labels change weekly and some are wrong. Concentration on this page uses return correlation instead.',
 w_mark:'Live Schwab mark (falls back to the last daily close).',
 w_today:'Change today vs yesterday\'s close, from Schwab.',
 sec_flags:'Checks, not trade instructions. Limits come from Data/personal/config.json. Defaults: one name 25%, a correlated cluster 50%, gross exposure 1.5× net liquidation; options are flagged inside 7 days of expiry.',
 sec_look:'Every position translated into the stock or fund it really tracks, so a 2× ETF and an option on the same name add up.',
 sec_sleeve:'The research momentum list: the 20 stocks with the strongest 12-1 momentum among the 300 most liquid, re-picked every 4 weeks and tracked in a paper-only forward test since 2026-10-05. It is a long-horizon study, separate from the Momentum dashboard\'s short-term 4-hour scanner.',
 sec_watch:'Names you are watching, with the same research context as your holdings. Nothing here places trades.',
 sev_high:'High: act-soon items, such as more than double a limit, an option expiring within a day, or exercise cash above available funds.',
 sev_warn:'Warn: over a configured limit, or a known structural risk.',
 sev_info:'Info: context worth knowing.',
 sec_history:'Every Schwab trade since the account opened, grouped into round trips: a position from the first buy until it is flat again (scale-ins and partial exits stay in one trade). P&L is the actual cash in and out, fees included, so it does not depend on option price data. Read-only: Schwab transactions are fetched with GET requests.',
 sec_pnlcurve:'Running total of realized P&L for the trades that pass the filters, dated when each trade closed. Open and unresolved trades are left out.',
 sec_tradechart:'Raw (unadjusted) price bars, so markers sit at the prices actually paid. Stock fills are drawn at the fill price. Option fills are drawn at the underlying\'s price when the fill happened (the open of the intraday bar holding the fill); the premium is in the marker text and the fill table. Option price history is not charted: option trade prints are too sparse to be a fair price line.',
 sec_breakdown:'Splits the filtered trades into groups and compares them. Small groups swing a lot: look at the 95% range, not just the average, before treating a difference as real.',
 h_trades:'Round trips that pass the filters. Resolved = closed with both the entry and the exit in Schwab\'s record. Open = still held. Unresolved = a record is missing, so their P&L is unknown and they are left out of every number.',
 h_winrate:'Share of resolved trades that made money after fees.',
 h_pnl:'Sum of realized P&L (cash out minus cash in, fees included) over resolved trades.',
 h_exp:'Average realized P&L per resolved trade (expectancy). Positive means the trades made money on average.',
 h_avgwl:'Average P&L of the winning trades and of the losing trades. Win rate × average win must beat loss rate × average loss for the trades to make money.',
 h_pf:'Profit factor: total won ÷ total lost. Above 1 the trades made money; 1.5+ is solid; below 1 they lost.',
 h_hold:'Median time from the first entry fill to the final exit, in calendar days.',
 h_expired:'Option trades that were held until they expired worthless, and their combined loss.',
 h_open:'Positions still held, valued at Schwab\'s current marks: cash paid so far vs what they are worth now. Unrealized until sold.',
 ht_entry:'Average entry price: per share for stocks, per option share for options (× 100 per contract). Quantity-weighted across scale-in fills.',
 ht_exit:'Average exit price, quantity-weighted. An expiration counts as an exit at 0.',
 ht_pnl:'Realized P&L in dollars, fees included. For open trades: unrealized P&L at Schwab\'s marks, in italics.',
 ht_ret:'P&L ÷ cash spent opening the trade. −100% = the whole premium was lost.',
 ht_held:'Calendar days from first entry to final exit. "<1d" = closed the same day.',
 ht_dte:'Days to expiry when the option was opened (calendar days).',
 ht_move:'How far the underlying moved in the trade\'s favor between entry and exit: up for stocks and calls, down for puts. Measured at the fill times (stocks: fill prices; options: the underlying when the fill happened).',
 ht_expiry:'Exact what-if: the P&L if every contract you sold early had been held to expiry instead. At expiry an option is worth exactly its intrinsic value at the closing price, so no model is involved.',
 ht_model:'The 1/2/5-more-sessions, repo-policy, take-profit and roll figures price the option with Black-Scholes using the volatility implied by your own fills. Checked against your real exit fills it lands within a median 13-20% (about 30% when an earnings report sat in between, where IV crush makes it run high). Treat them as estimates, not P&L.',
 sec_decomp:'Every resolved option trade, sorted by what the stock did. Never worked = the stock never got 3% your way and ended against you (entry or thesis). Gave it back = it got at least 3% your way, then reversed by your exit (trade management). Stock went your way but the option lost = the premium, time decay or an IV drop ate a correct call (instrument).',
 ht_ci:'95% bootstrap range of the average P&L (4,000 resamples of the trades in the group). If it spans 0, this history cannot tell whether the group makes or loses money on average.',
 flag_name_concentration:'One underlying\'s price exposure is above the per-name limit. A per-name cap was the one control that beat random refusal in the 2026 paper post-mortem.',
 flag_cluster_concentration:'Names whose daily returns correlate at 0.6 or more are grouped. Together they exceed the cluster limit, so they tend to win or lose together.',
 flag_gross_exposure:'Total price exposure across all names is above the gross limit.',
 flag_leveraged_decay:'Daily-reset leveraged ETFs lose value in choppy markets even when the stock goes nowhere: about (L² − L) × volatility² ÷ 2 per year, in log terms.',
 flag_look_through_unverified:'The fund says it is leveraged, but it could not be matched to an underlying whose returns it tracks, so it counts as itself.',
 flag_shrugged_miss:'The last report missed EPS by more than 2%, yet the stock moved less than 2% vs SPY. In research, such names lagged SPY by 2.8% over the next 126 sessions (t = −4.3).',
 flag_earnings_soon:'The underlying reports within the next 7 days.',
 flag_itm_into_expiry:'In the money near expiry: if held, Schwab auto-exercises it and you must pay for (or deliver) the shares.',
 flag_itm_short_assignment:'A short option in the money near expiry can be assigned.',
 flag_otm_near_expiry:'Out of the money near expiry: worthless at expiry unless the underlying moves at least this much.',
 flag_earnings_before_expiry:'Earnings land before expiry: expect a big move and an implied-volatility drop (IV crush).',
 flag_wide_spread:'The bid/ask spread is wide relative to the price, so closing costs a noticeable share of the position.',
 flag_exposure_unknown:'Schwab returned no delta or price, so those positions are left out of exposure and risk.',
 flag_losing_option_late:'A long option worth half or less of what you paid, with 14 or fewer days left. In your own history such options rarely came back by expiry; the flag quotes your numbers. A review prompt, not an instruction.'};
const tipbox=document.createElement('div');tipbox.id='tipbox';tipbox.setAttribute('role','tooltip');document.body.appendChild(tipbox);
function showTip(el){let t=el.dataset.tip.split(' ').map(k=>TIPS[k]).filter(Boolean).join(' ');if(!t)return;tipbox.textContent=t;tipbox.style.display='block';
 let r=el.getBoundingClientRect(),w=tipbox.offsetWidth,h=tipbox.offsetHeight,x=Math.min(Math.max(8,r.left),innerWidth-w-8),y=r.bottom+6;
 if(y+h>innerHeight-8)y=Math.max(8,r.top-h-6);tipbox.style.left=x+'px';tipbox.style.top=y+'px'}
function hideTip(){tipbox.style.display='none'}
document.addEventListener('mouseover',e=>{let el=e.target.closest('[data-tip]');el?showTip(el):hideTip()});
document.addEventListener('focusin',e=>{let el=e.target.closest('[data-tip]');el?showTip(el):hideTip()});
document.addEventListener('focusout',hideTip);document.addEventListener('keydown',e=>{if(e.key==='Escape')hideTip()});
addEventListener('scroll',hideTip,true);
document.querySelectorAll('[data-tip]').forEach(el=>el.tabIndex=0);
let timer;
(function(){let b=document.getElementById('cyno-live-badge');if(b){b.textContent='real money · read-only';b.className='live-badge live';}let p=document.getElementById('cyno-performance');if(p)p.style.display='none';})();
function tile(k,v,n,tip){return '<div class=tile><div class=k'+(tip?' data-tip="'+tip+'" tabindex=0':'')+'>'+esc(k)+'</div><div class=v>'+v+'</div>'+(n?'<div class=n>'+n+'</div>':'')+'</div>'}
function momTag(m){if(!m)return '<span class=small>ranking unavailable</span>';if(!m.ranked)return '<span class=small>not ranked (not a stock in the universe)</span>';
 if(!m.in_liquid_300)return '<span class=tag>outside liquid 300</span><span class=small>liquidity #'+m.dv_rank+' · 12-1 '+pct(m.mom_12_1,0,true)+'</span>';
 return (m.in_sleeve?'<span class="tag on">in sleeve</span>':'<span class=tag>not in sleeve</span>')+'<span class=small>#'+m.rank_in_300+' of 300 · 12-1 '+pct(m.mom_12_1,0,true)+'</span>'}
function earnCell(e){if(!e)return '<span class=small>n/a</span>';let out=[];if(e.next_earnings)out.push('next '+esc(e.next_earnings)+' ('+e.days_to_next+'d)');
 if(e.shrugged_miss)out.push('<span class="tag bad">shrugged miss</span>');else if(e.pending_report)out.push('<span class=small>report '+esc(e.pending_report)+' not in data yet</span>');
 else if(e.last_report&&e.last_report.reaction_group)out.push('<span class=small>last: '+esc(e.last_report.surprise_group)+' '+num(e.last_report.surprise_pct,1)+'%, reaction '+num(e.last_report.reaction_pct,1)+'%</span>');
 else if(e.last_report)out.push('<span class=small>last '+esc(e.last_report.date)+': '+esc(e.last_report.surprise_group)+' '+num(e.last_report.surprise_pct,1)+'%, reaction pending</span>');
 return out.join('<br>')||'<span class=small>—</span>'}
function render(d){let r=d.report,s=d.status,tk=d.token||{};let b=[];
 if(tk.severity&&tk.severity!=='info')b.push('<div class="banner '+(tk.severity==='error'?'error':'warning')+'">'+esc(tk.message)+'</div>');
 if(s.error)b.push('<div class="banner error">Last sync failed: '+esc(s.error)+(r?' · showing the previous snapshot':'')+'</div>');
 if(r&&!r.data.bars_current)b.push('<div class="banner warning">Daily bars end '+esc(r.data.bars_through||'never')+' (expected '+esc(r.data.bars_expected)+')'+(s.phase==='market data'?'; refreshing now…':'')+'. Price-based columns may be stale.</div>');
 if(r&&r.data.missing_bars.length)b.push('<div class=banner>No daily bars for '+esc(r.data.missing_bars.join(', '))+'.</div>');
 $('#banners').innerHTML=b.join('');
 $('#sync').textContent=(s.state==='syncing'?'Syncing ('+(s.phase||'')+')… ':'')+(r?'Schwab snapshot '+when(r.fetched_at):'No snapshot yet');
 if(!r){$('#flags').innerHTML='<div class=card-body>'+(s.state==='error'?'Could not load the account.':'Reading the Schwab account…')+'</div>';return;}
 let t=r.totals,h=r.risk.hist_sim,acct=r.accounts.map(a=>a.type+' '+a.account).join(' · ');
 let bal=r.accounts.reduce((o,a)=>{for(let k in a.balances){if(a.balances[k]!=null)o[k]=(o[k]||0)+a.balances[k]}return o},{});
 $('#tiles').innerHTML=[tile('Net liquidation',usd(t.net_liq),esc(acct),'net_liq'),tile('Day P/L',sgn(t.day_pl),'positions, Schwab marks','day_pl'),tile('Open P/L',sgn(t.open_pl),'vs cost basis','open_pl'),
  tile('Available funds',usd(bal.available_funds??bal.cash_balance),'cash '+usd(bal.cash_balance)+' · margin bal. '+usd(bal.margin_balance),'avail'),
  tile('Gross exposure',t.gross_exposure==null?'—':num(t.gross_exposure,2)+'×','price exposure ÷ net liq','gross'),
  tile('SPY-beta exposure',t.beta_spy_pct==null?'—':num(t.beta_spy_pct,1)+'×','≈ '+usd(t.beta_spy_dollars)+' of SPY; 1% SPY ≈ '+usd(t.beta_spy_dollars==null?null:t.beta_spy_dollars/100)+(t.no_beta.length?' · no beta: '+esc(t.no_beta.join(', ')):''),'beta_exp'),
  tile('Bad day (5%)',h?usd(h.q05):'—',h?pct(h.q05_pct,1)+' of net liq · 1 day in 20':'needs 60+ days of shared history','bad_day'),
  tile('Worst day, 1y',h?usd(h.worst):'—',h?esc(h.worst_date)+' · '+pct(h.worst_pct,1):'','worst_day'),
  tile('Independent bets',r.risk.n_eff==null?'—':num(r.risk.n_eff,1),'correlation-adjusted count','n_eff')].join('');
 $('#flags').innerHTML=r.flags.length?r.flags.map(f=>'<div class=flag><span class="sev '+f.severity+'" tabindex=0 data-tip="sev_'+esc(f.severity)+' flag_'+esc(f.code)+'">'+f.severity+'</span><span>'+esc(f.text)+'</span></div>').join(''):'<div class=card-body>No flags.</div>';
 $('#holdings').innerHTML=r.positions.map(p=>{let lt=p.look_through,via=p.kind==='option'?'Δ '+usd(p.exposure)+' of '+esc(p.instrument):(lt&&lt.verified?num(lt.leverage,0)+'× '+esc(lt.underlying)+' <span class=small>(corr '+num(lt.corr,2)+', β '+num(lt.beta,2)+')</span>':(p.instrument?esc(p.instrument):'cash-like'));
  return '<tr><td><b>'+esc(p.symbol)+'</b><div class=small>'+esc(p.description||'')+'</div></td><td>'+esc(p.kind)+'</td><td class=num>'+num(p.quantity,p.quantity%1?2:0)+'</td><td class=num>'+usd(p.average_price,2)+'</td><td class=num>'+usd(p.mark,2)+'</td><td class=num>'+usd(p.market_value)+'</td><td class=num>'+pct(p.weight)+'</td><td class=num>'+sgn(p.day_pl)+'</td><td class=num>'+sgn(p.open_pl)+'<div class=small>'+pct(p.open_pl_pct,0,true)+'</div></td><td>'+via+'</td></tr>'}).join('')||'<tr><td colspan=10>No positions.</td></tr>';
 $('#options').innerHTML=r.options.map(o=>'<tr><td><b>'+esc(o.underlying)+' '+esc(o.expiry)+' '+num(o.strike,2)+' '+esc(o.put_call)+'</b><div class=small>'+num(o.quantity)+' contract(s) · IV '+pct(o.iv==null?null:o.iv/100,0)+'</div></td><td class=num>'+(o.dte??'—')+'</td><td class=num>'+usd(o.underlying_price,2)+'</td><td class=num>'+spct(o.moneyness)+'</td><td class=num>'+num(o.delta,2)+'</td><td class=num>'+sgn(o.theta_per_day)+'</td><td class=num>'+pct(o.move_to_breakeven,1,true)+'</td><td class=num>'+pct(o.spread_pct,0)+'</td><td class=num>'+usd(o.exercise_notional)+'</td><td>'+(o.earnings_before_expiry?'<span class="tag bad">'+esc(o.next_earnings)+'</span>':'<span class=small>no</span>')+'</td><td class=num>'+sgn(o.open_pl)+'</td></tr>').join('')||'<tr><td colspan=11>No options.</td></tr>';
 $('#instruments').innerHTML=r.instruments.map(i=>'<tr><td><b>'+esc(i.instrument)+'</b></td><td class=num>'+usd(i.exposure)+'</td><td class=num>'+pct(i.pct)+'</td><td class=num>'+usd(i.capital)+'</td><td>'+esc(i.sources.join(', '))+'</td><td class=num>'+num(i.beta_spy,2)+'</td><td class=num>'+pct(i.vol,0)+'</td><td class=num>'+spct(i.vs_sma200,0)+'</td><td>'+momTag(i.momentum)+'</td><td>'+earnCell(i.earnings)+'</td><td class=small>'+esc(i.theme||'—')+'</td></tr>').join('')||'<tr><td colspan=11>No exposure.</td></tr>';
 let cl=r.risk.clusters.filter(c=>c.length>1);
 $('#risknote').innerHTML='Price exposure = market value × verified leverage, or option delta × 100 × contracts × underlying price: the dollar move for a 100% move in the underlying, so 1% of it is the P/L of a 1% move. Capital = market value of those positions, the most a long option can lose. '
  +(cl.length?'Correlated clusters (daily corr ≥ '+r.config.cluster_corr+'): '+cl.map(c=>esc(c.join(' + '))).join('; ')+'. ':'No correlated clusters. ')
  +(h?'Risk tiles replay '+h.n_days+' sessions ('+esc(h.start)+' to '+esc(h.end)+') on today\'s exposures, delta-linear with each long option\'s loss capped at its premium; no gamma, decay, or IV moves, so option upside is understated.'+(h.excluded.length?' Left out for short history: '+esc(h.excluded.join(', '))+'.':''):'');
 $('#watchlist').innerHTML=r.watchlist.map(w=>'<tr><td><b>'+esc(w.ticker)+'</b>'+(w.held?' <span class="tag on">held</span>':'')+(w.look_through&&w.look_through.verified?'<div class=small>'+num(w.look_through.leverage,0)+'× '+esc(w.look_through.underlying)+'</div>':'')+'</td><td class=num>'+usd(w.mark??w.close,2)+'</td><td class=num>'+spct(w.net_pct==null?null:w.net_pct/100)+'</td><td class=num>'+spct(w.vs_sma200,0)+'</td><td>'+momTag(w.momentum)+'</td><td class=num>'+num(w.beta_spy,2)+'</td><td class=num>'+pct(w.vol,0)+'</td><td>'+earnCell(w.earnings)+'</td><td class=small>'+esc(w.theme||'—')+'</td><td class=small>'+esc(w.note)+'</td><td><button class=lnk data-chart="'+esc(w.ticker)+'">chart</button><button class=x title="Remove" data-t="'+esc(w.ticker)+'">×</button></td></tr>').join('')||'<tr><td colspan=11>Empty. Add a ticker above.</td></tr>';
 document.querySelectorAll('button.x[data-t]').forEach(el=>el.onclick=()=>post('/api/watchlist/remove',{ticker:el.dataset.t}));
 document.querySelectorAll('button.lnk[data-chart]').forEach(el=>el.onclick=()=>{H.sel=null;H.ticker=el.dataset.chart;H.tf='ticker';hChart();$('#history').scrollIntoView({behavior:'smooth'})});
 let m=r.momentum,rec=m.record||{},cum=rec.cumulative||{};
 $('#sleeve').innerHTML=m.decision_date?('<div>Decision '+esc(m.decision_date)+' (next '+esc(rec.next_decision||'?')+'): '+m.sleeve.map(x=>'<span class=tag>'+esc(x)+'</span>').join(' ')+'</div>'
  +'<div class=small style="margin-top:6px">Forward shadow since 2026-10-05, marked through '+esc(rec.marked_through||'?')+': top-300 sleeve '+esc(cum.mom_top300??'?')+'%, 70/30 blend '+esc(cum.core_sat_300??'?')+'%, SPY '+esc(cum.spy??'?')+'%. Ranks use bars up to the decision date; 12-1 momentum skips the latest month by design.</div>'):'Momentum ranking unavailable.';
 $('#footer').innerHTML='Daily bars through '+esc(r.data.bars_through||'—')+' (Alpaca SIP, split+dividend adjusted; Data/personal/bars_1d). Earnings: Yahoo via yfinance; consensus may not be point-in-time. Themes are dynamic labels that churn weekly and can be wrong. Flags are logged once per day to Data/personal/flag_log.jsonl so they can be tested later.';
}
async function load(force){clearTimeout(timer);try{let resp=await fetch('/api/portfolio'+(force?'?force=1':''),{cache:'no-store'});let d=await resp.json();if(!resp.ok)throw Error(d.error||'request failed');render(d);
 timer=setTimeout(load,d.status.state==='syncing'||d.status.state==='never'?2000:60000);}catch(e){$('#banners').innerHTML='<div class="banner error">'+esc(e.message)+'</div>';timer=setTimeout(load,15000)}}
async function post(path,body){$('#watcherr').textContent='';try{let resp=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});let d=await resp.json();if(!resp.ok)throw Error(d.error||'request failed');render(d);clearTimeout(timer);timer=setTimeout(load,2000)}catch(e){$('#watcherr').textContent=e.message}}
$('#watchform').onsubmit=e=>{e.preventDefault();let f=e.target;post('/api/watchlist',{ticker:f.ticker.value,note:f.note.value}).then(()=>{if(!$('#watcherr').textContent)f.reset()})};
// ---------- trade history ----------
const H={rep:null,sel:null,ticker:null,tf:'1d',sort:{k:'open_time',d:-1},gsel:null,timer:null};
const HG={b_type:'Type',b_year:'Year opened',b_dte:'DTE at entry',b_hold:'Holding time',b_moneyness:'Moneyness at entry',b_trend:'Trend at entry (50-day avg)',b_extension:'Distance from 20-day high',b_into_entry:'Move since prior close',b_entry_time:'Entry time (ET)',b_spy:'SPY regime at entry',b_size:'Position size',b_class:'Underlying class',b_earnings:'Earnings',b_entry_style:'Entry style (dip / momentum)',b_exit_trigger:'Exit trigger',b_exit:'Exit',b_scaled:'Scaling',chart_symbol:'Underlying'};
$('#f_group').innerHTML=Object.entries(HG).map(([k,v])=>'<option value="'+k+'">'+esc(v)+'</option>').join('');
const isRes=e=>e.status==='closed'&&!e.incomplete&&e.pnl!=null;
const hv=(e,k)=>k==='pnl'?(e.pnl??e.unrealized):(k in e?e[k]:(e.ctx||{})[k]);
function hFilter(useGroup){let r=H.rep?H.rep.episodes:[],ty=$('#f_type').value,sd=$('#f_side').value,oc=$('#f_outcome').value,yr=$('#f_year').value,sy=$('#f_sym').value.trim().toUpperCase(),g=$('#f_group').value;
 return r.filter(e=>{if(ty==='option'&&e.kind!=='option')return false;if(ty==='stock'&&e.kind==='option')return false;if(sd!=='all'&&e.put_call!==sd)return false;
  if(oc==='win'&&!(isRes(e)&&e.pnl>0))return false;if(oc==='loss'&&!(isRes(e)&&e.pnl<=0))return false;if(oc==='open'&&e.status!=='open')return false;
  if(oc==='unresolved'&&!(e.incomplete||e.status==='unresolved'))return false;if(yr!=='all'&&e.open_date.slice(0,4)!==yr)return false;
  if(sy&&e.chart_symbol!==sy&&e.underlying!==sy&&!String(e.symbol).startsWith(sy))return false;
  if(useGroup&&H.gsel!=null&&(e[g]??'n/a')!==H.gsel)return false;return true})}
function stats(rows){let p=rows.map(e=>e.pnl),n=p.length;if(!n)return{n:0};let w=p.filter(x=>x>0),l=p.filter(x=>x<=0),sum=a=>a.reduce((s,x)=>s+x,0),med=a=>{if(!a.length)return null;let b=[...a].sort((x,y)=>x-y),m=b.length>>1;return b.length%2?b[m]:(b[m-1]+b[m])/2};
 let ret=rows.map(e=>e.return_pct).filter(x=>x!=null),hold=rows.map(e=>e.hold_days).filter(x=>x!=null);
 return{n,wins:w.length,wr:w.length/n,total:sum(p),avg:sum(p)/n,aw:w.length?sum(w)/w.length:null,al:l.length?sum(l)/l.length:null,pf:sum(l)<0?sum(w)/-sum(l):null,mret:med(ret),mhold:med(hold)}}
function rng(seed){return()=>{seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function bootCI(xs){if(xs.length<5)return null;let r=rng(7),m=[];for(let b=0;b<4000;b++){let s=0;for(let i=0;i<xs.length;i++)s+=xs[(r()*xs.length)|0];m.push(s/xs.length)}m.sort((a,b)=>a-b);return[m[100],m[3899]]}
function htile(k,v,n,tip){return '<div class=tile><div class=k data-tip="'+tip+'" tabindex=0>'+esc(k)+'</div><div class=v>'+v+'</div>'+(n?'<div class=n>'+n+'</div>':'')+'</div>'}
const days=d=>d==null?'—':d<1?'<1d':num(d,0)+'d';
function hRender(){let rep=H.rep;if(!rep)return;let all=hFilter(true),res=all.filter(isRes),op=all.filter(e=>e.status==='open'),un=all.filter(e=>e.incomplete||e.status==='unresolved'),s=stats(res);
 let ex=res.filter(e=>e.exit_reason==='expired'),exs=ex.reduce((a,e)=>a+e.pnl,0),ur=op.reduce((a,e)=>a+(e.unrealized||0),0);
 $('#htiles').innerHTML=[htile('Trades',num(res.length),op.length+' open · '+un.length+' unresolved','h_trades'),htile('Win rate',s.n?pct(s.wr,0):'—',s.n?s.wins+' of '+s.n:'','h_winrate'),
  htile('Net realized P&L',s.n?sgn(s.total):'—','fees included','h_pnl'),htile('Avg P&L / trade',s.n?sgn(s.avg):'—','expectancy','h_exp'),
  htile('Avg win / avg loss',s.n?usd(s.aw)+' / '+usd(s.al):'—','','h_avgwl'),htile('Profit factor',s.pf==null?'—':num(s.pf,2),'won ÷ lost','h_pf'),
  htile('Median hold',days(s.mhold),'median return '+pct(s.mret,0,true),'h_hold'),htile('Expired worthless',num(ex.length),ex.length?sgn(exs)+' total':'','h_expired'),
  htile('Open positions',num(op.length),op.length?sgn(ur)+' unrealized':'','h_open')].join('');
 let oc=res.filter(e=>e.kind==='option'&&e.ctx&&e.ctx.under_move_dir!=null),DC=[['right & paid',e=>e.ctx.under_move_dir>0&&e.pnl>0,'it worked'],['stock went your way, option still lost',e=>e.ctx.under_move_dir>0&&e.pnl<=0,'instrument: premium, decay, IV'],['was right during the trade, gave it back by exit',e=>e.ctx.under_move_dir<=0&&e.pnl<=0&&(e.ctx.mfe_dir||0)>=0.03,'exit / trade management'],['never worked',e=>e.ctx.under_move_dir<=0&&e.pnl<=0&&(e.ctx.mfe_dir||0)<0.03,'entry / thesis'],['stock went against you, option still paid',e=>e.ctx.under_move_dir<=0&&e.pnl>0,'luck / IV']];
 const medOf=a=>{a=a.filter(x=>x!=null).sort((x,y)=>x-y);return a.length?(a.length%2?a[a.length>>1]:(a[a.length/2-1]+a[a.length/2])/2):null};
 $('#hdecomp').innerHTML=oc.length?DC.map(([lab,fn,at])=>{let gr=oc.filter(fn);return '<tr><td>'+esc(lab)+'</td><td class=num>'+gr.length+' <span class=small>('+pct(gr.length/oc.length,0)+')</span></td><td class=num>'+sgn(gr.reduce((a,e)=>a+e.pnl,0))+'</td><td class=num>'+spct(medOf(gr.map(e=>e.ctx.under_move_dir)),1)+'</td><td class=num>'+spct(medOf(gr.map(e=>e.ctx.mfe_dir)),1)+'</td><td class=small>'+esc(at)+'</td></tr>'}).join(''):'<tr><td colspan=6 class=small>No resolved option trades with underlying bars in this filter.</td></tr>';
 let g=$('#f_group').value;$('#f_group_chip').innerHTML=H.gsel!=null?'<span class="tag on">'+esc(HG[g])+': '+esc(H.gsel)+' <button class=x id=gclear title="Clear">×</button></span>':'';
 let gc=$('#gclear');if(gc)gc.onclick=()=>{H.gsel=null;hRender()};
 pnlCurve(res);
 let k=H.sort.k,d=H.sort.d,rows=[...all].sort((a,b)=>{let x=hv(a,k),y=hv(b,k);if(x==null&&y==null)return 0;if(x==null)return 1;if(y==null)return -1;return(x<y?-1:x>y?1:0)*d});
 $('#htable').innerHTML=rows.map(e=>{let c=e.ctx||{},openp=e.status==='open',pl=openp?e.unrealized:e.pnl,rt=openp?e.unrealized_pct:e.return_pct;
  let lbl=e.kind==='option'?esc(e.underlying)+' '+esc(e.expiry)+' '+num(e.strike,e.strike%1?1:0)+(e.put_call==='CALL'?'C':'P'):esc(e.symbol);
  let note=(e.notes||[]).length?'<span title="'+esc(e.notes.join(' · '))+'">⚠</span>':'';
  return '<tr class="clk'+(H.sel===e.id?' sel':'')+'" data-id="'+e.id+'"><td>'+esc(e.open_date)+'</td><td><b>'+lbl+'</b>'+(e.direction==='short'?' <span class=tag>short</span>':'')+'</td><td>'+esc(e.b_type)+'</td><td class=num>'+num(e.max_qty,e.max_qty%1?2:0)+'</td><td class=num>'+usd(e.avg_entry,2)+'</td><td class=num>'+(openp?'<span class=small>open</span>':usd(e.avg_exit,2))+'</td><td class=num>'+(pl==null?'—':openp?'<i>'+sgn(pl)+'</i>':sgn(pl))+'</td><td class=num>'+spct(rt,0)+'</td><td class=num>'+days(e.hold_days)+'</td><td class=num>'+(e.dte_at_entry??'—')+'</td><td class=small>'+esc(e.b_exit||'')+'</td><td class=num>'+spct(c.under_move_dir,1)+'</td><td class=num>'+(c.hold_to_expiry_pnl==null?'—':sgn(c.hold_to_expiry_pnl))+'</td><td>'+note+'</td></tr>'}).join('')||'<tr><td colspan=14>No trades match.</td></tr>';
 document.querySelectorAll('#htable tr[data-id]').forEach(tr=>tr.onclick=()=>{H.sel=tr.dataset.id;H.ticker=null;if(H.tf==='ticker')H.tf='1d';hChart();document.querySelectorAll('#htable tr.sel').forEach(x=>x.classList.remove('sel'));tr.classList.add('sel')});
 let base=hFilter(false).filter(isRes),groups={};base.forEach(e=>{let key=e[g]??'n/a';(groups[key]=groups[key]||[]).push(e)});
 $('#hbreak').innerHTML=Object.entries(groups).sort((a,b)=>b[1].length-a[1].length).map(([key,gr])=>{let t=stats(gr),ci=bootCI(gr.map(e=>e.pnl));
  return '<tr class="clk'+(H.gsel===key?' sel':'')+'" data-g="'+esc(key)+'"><td><b>'+esc(key)+'</b></td><td class=num>'+t.n+'</td><td class=num>'+pct(t.wr,0)+'</td><td class=num>'+sgn(t.total)+'</td><td class=num>'+sgn(t.avg)+'</td><td class=num>'+(ci?usd(ci[0])+' to '+usd(ci[1]):'<span class=small>n&lt;5</span>')+'</td><td class=num>'+spct(t.mret,0)+'</td><td class=num>'+(t.pf==null?'—':num(t.pf,2))+'</td><td class=num>'+usd(t.aw)+'</td><td class=num>'+usd(t.al)+'</td></tr>'}).join('')||'<tr><td colspan=10>No resolved trades match.</td></tr>';
 document.querySelectorAll('#hbreak tr[data-g]').forEach(tr=>tr.onclick=()=>{H.gsel=H.gsel===tr.dataset.g?null:tr.dataset.g;hRender()});
 let rc=rep.reconciliation||{},tx=rep.transactions||{},bars=rep.bars||{},iss=rep.issues||[],ch=rep.checks||{};
 $('#hnotes').innerHTML='Source: Schwab transactions (TRADE + RECEIVE_AND_DELIVER) from '+esc(rep.first_trade||'—')+', '+num(tx.stored)+' records stored in Data/personal/transactions.jsonl (append-only; last '+esc(tx.mode||'')+' fetch '+when(tx.at)+'). '
  +(rc.checked?(rc.ok?'Open positions reconcile exactly with Schwab ('+(rc.matched||[]).length+' matched). ':'Reconciliation: '+(rc.matched||[]).length+' open positions match Schwab'+((rc.qty_mismatch||[]).length?'; quantity differs for '+esc(rc.qty_mismatch.map(x=>x.key).join(', ')):'')+((rc.held_without_history||[]).length?'; held with no history: '+esc(rc.held_without_history.join(', ')):'')+((rc.open_not_held||[]).length?'; '+rc.open_not_held.length+' unresolved (history says open, Schwab does not hold them)':'')+'. '):'Not reconciled (no holdings snapshot). ')
  +(ch.effect_checked?'Opening/closing inferred from running quantity agrees with Schwab\'s tag on '+num(ch.effect_checked-ch.effect_mismatch)+' of '+num(ch.effect_checked)+' fills. ':'')
  +(iss.length?'Records not attributed to any trade: '+esc(iss.map(i=>(i.time||'').slice(0,10)+' '+(i.net_amount!=null?usd(i.net_amount,2):'')+' ('+i.issue+')').join('; '))+'. ':'')
  +((bars.daily_errors||[]).length?'Daily bars missing for: '+esc(bars.daily_errors.join(', '))+'. ':'')
  +'Charts use raw Alpaca SIP bars (Data/personal/history_bars); entry features use split-adjusted bars and only data known before the entry fill.';
}
let PC=null,TC=null;
const chartOpts=()=>({autoSize:true,layout:{background:{type:'solid',color:'#11191e'},textColor:'#94a4a9'},grid:{vertLines:{color:'rgba(148,164,169,.07)'},horzLines:{color:'rgba(148,164,169,.07)'}},rightPriceScale:{borderColor:'#263238'},timeScale:{borderColor:'#263238',timeVisible:true,secondsVisible:false},crosshair:{mode:0}});
function pnlCurve(res){if(!window.LightweightCharts){$('#pnlchart').innerHTML='<div class=small>Chart library did not load (needs internet for unpkg.com).</div>';return}
 if(!PC){let c=LightweightCharts.createChart($('#pnlchart'),{...chartOpts(),timeScale:{borderColor:'#263238'}});PC={c,s:c.addBaselineSeries({baseValue:{type:'price',price:0},topLineColor:'#26a69a',bottomLineColor:'#ef5350',topFillColor1:'rgba(38,166,154,.25)',topFillColor2:'rgba(38,166,154,.02)',bottomFillColor1:'rgba(239,83,80,.02)',bottomFillColor2:'rgba(239,83,80,.25)',priceFormat:{type:'price',precision:0,minMove:1}})}}
 let byDay={};res.forEach(e=>{byDay[e.close_date]=(byDay[e.close_date]||0)+e.pnl});let cum=0,pts=Object.keys(byDay).sort().map(d=>({time:d,value:(cum+=byDay[d])}));
 PC.s.setData(pts);PC.c.timeScale().fitContent()}
function tcFresh(){if(TC){TC.c.remove();TC=null}let c=LightweightCharts.createChart($('#tradechart'),chartOpts());
 let candles=c.addCandlestickSeries({upColor:'#26a69a',downColor:'#ef5350',borderVisible:false,wickUpColor:'#26a69a',wickDownColor:'#ef5350',priceLineVisible:false});
 let vol=c.addHistogramSeries({priceFormat:{type:'volume'},priceScaleId:'vol',lastValueVisible:false,priceLineVisible:false});c.priceScale('vol').applyOptions({scaleMargins:{top:.86,bottom:0}});
 let pt=col=>c.addLineSeries({color:col,lineVisible:false,pointMarkersVisible:true,pointMarkersRadius:5,lastValueVisible:false,priceLineVisible:false,crosshairMarkerVisible:false});
 TC={c,candles,vol,entry:pt('#26a69a'),exit:pt('#ef5350')};return TC}
function reviewHtml(e){let c=e.ctx||{},L=[],M=v=>v==null?'—':sgn(v),P=(v,d=1)=>spct(v,d);
 if(c.feature_note)return '<span class=small>Timing review unavailable: '+esc(c.feature_note)+'.</span>';
 if(c.entry_style!=null||c.entry_pctile!=null)L.push('<b>Entry</b> ('+esc(c.entry_style||'—')+'): in the '+(c.entry_pctile==null?'—':num(c.entry_pctile*100,0))+'th percentile of the ±5-session range (0 = the low); within 5 sessions it went '+P(c.entry_wait_5d)+' against/for you at its best; the stock had already moved '+P(c.entry_prerun_5d)+' off its 5-session extreme. 10-session move before entry '+P(c.ret_10d_dir)+(c.days_to_earnings!=null?'; next earnings '+c.days_to_earnings+'d away':'')+'.');
 if(c.exit_trigger!=null||c.exit_pctile!=null)L.push('<b>Exit</b> ('+esc(c.exit_trigger||e.exit_reason)+'): '+(c.exit_pctile==null?'':'in the '+num(c.exit_pctile*100,0)+'th percentile of the ±5-session range (100 = the high); ')+'move since prior close '+P(c.exit_day_move)+', vs best seen in trade '+P(c.exit_off_peak)+'. After the exit: 1d '+P(c.post_exit_1d)+', 2d '+P(c.post_exit_2d)+', 10d '+P(c.post_exit_10d)+' (best in 10d '+P(c.post_exit_best_10d)+', worst '+P(c.post_exit_worst_10d)+').');
 if(e.kind==='option'&&e.status==='closed')L.push('<b>Other exits</b>: actual '+M(e.pnl)+' · held to expiry <b>'+M(c.hold_to_expiry_pnl)+'</b> (exact, '+(c.expiry_close!=null?'stock closed '+usd(c.expiry_close,2)+' at expiry':'no expiry bar')+') · 1 more session '+M(c.hold_plus_1d_pnl)+' · 2 more '+M(c.hold_plus_2d_pnl)+' · 5 more '+M(c.hold_plus_5d_pnl)+' · repo live exit policy '+M(c.repo_policy_pnl)+(c.repo_policy_exit?' ('+esc(c.repo_policy_exit)+')':'')+' · take profit at +50% '+M(c.take_profit_50_pnl)+' · roll proceeds into a 30-day ATM option '+M(c.roll_all_pnl)+'. <span class=small data-tip=ht_model tabindex=0>Model estimates except held-to-expiry; IV at entry '+(c.iv_entry==null?'—':pct(c.iv_entry,0))+'.</span>');
 return L.length?'<div style="display:grid;gap:4px">'+L.map(x=>'<div>'+x+'</div>').join('')+'</div>':''}
async function hChart(){document.querySelectorAll('#ct_tf button').forEach(b=>b.classList.toggle('on',b.dataset.tf===H.tf));
 let ep=H.sel&&H.rep?H.rep.episodes.find(e=>e.id===H.sel):null,url;
 if(H.tf==='ticker'){let t=H.ticker||(ep&&ep.chart_symbol);if(!t)return;H.ticker=t;url='/api/history/chart?ticker='+encodeURIComponent(t)}
 else{if(!ep)return;url='/api/history/chart?id='+ep.id+'&tf='+H.tf}
 $('#ct_title').textContent=ep&&H.tf!=='ticker'?(ep.kind==='option'?ep.underlying+' '+ep.expiry+' '+ep.strike+' '+ep.put_call:ep.symbol):H.ticker+': all trades';
 $('#ct_sub').innerHTML=ep&&H.tf!=='ticker'?esc(ep.open_date)+' → '+esc(ep.close_date||'open')+' · '+(ep.pnl!=null?sgn(ep.pnl)+' ('+pct(ep.return_pct,0,true)+')':ep.unrealized!=null?'<i>'+sgn(ep.unrealized)+' unrealized</i>':'P&L unknown')+(ep.kind==='option'?' · bars: '+esc(ep.chart_symbol):''):'Loading…';
 $('#ct_notes').textContent='Loading bars…';$('#ct_fills').innerHTML='';$('#ct_review').innerHTML='';
 if(!window.LightweightCharts){$('#ct_notes').textContent='Chart library did not load (needs internet for unpkg.com).';return}
 try{let resp=await fetch(url,{cache:'no-store'}),d=await resp.json();if(!resp.ok)throw Error(d.error||'request failed');
  let t=tcFresh();if(d.error){$('#ct_notes').textContent=d.error;return}
  t.candles.setData(d.bars.map(b=>({time:b.time,open:b.open,high:b.high,low:b.low,close:b.close})));
  t.vol.setData(d.bars.map(b=>({time:b.time,value:b.volume,color:b.close>=b.open?'rgba(38,166,154,.28)':'rgba(239,83,80,.28)'})));
  t.candles.setMarkers(d.markers||[]);t.entry.setData((d.points||{}).entry||[]);t.exit.setData((d.points||{}).exit||[]);
  (d.price_lines||[]).forEach(p=>t.candles.createPriceLine({price:p.price,color:p.color,lineWidth:1,lineStyle:p.style,axisLabelVisible:true,title:p.title}));
  if(d.visible)t.c.timeScale().setVisibleRange(d.visible);else t.c.timeScale().fitContent();
  if(H.tf==='ticker')$('#ct_sub').textContent=(d.n_episodes||0)+' trade(s) on '+d.symbol+' · daily bars, raw prices';
  let tfl={'1d':'daily bars','15Min':'15-minute bars','1Hour':'1-hour bars'}[d.timeframe]||d.timeframe;
  $('#ct_notes').textContent=[(H.tf!=='ticker'?tfl+', raw prices, times ET. ':'')+'▲ buy · ▼ sell · ● expiry/corporate action; dots = fill price (stocks) or underlying at the fill (options).',...(d.notes||[])].join(' · ');
  $('#ct_review').innerHTML=H.tf!=='ticker'&&ep?reviewHtml(ep):'';
  if(d.fills)$('#ct_fills').innerHTML='<table><thead><tr><th>Time (ET)</th><th>Fill</th><th class=num>Qty</th><th class=num>Price</th><th class=num>Cash</th><th class=num>Fees</th><th class=num>Underlying at fill</th></tr></thead><tbody>'+d.fills.map(f=>'<tr><td>'+esc(f.time_et)+'</td><td>'+esc(f.source==='expiration'?'expired':f.role)+'</td><td class=num>'+num(f.qty,f.qty%1?2:0)+'</td><td class=num>'+usd(f.price,2)+'</td><td class=num>'+sgn(f.cash,2)+'</td><td class=num>'+usd(f.fees,2)+'</td><td class=num>'+usd(f.under_price,2)+' <span class=small>'+esc(f.under_source||'')+'</span></td></tr>').join('')+'</tbody></table>';
 }catch(e){$('#ct_notes').textContent=e.message}}
document.querySelectorAll('#ct_tf button').forEach(b=>b.onclick=()=>{H.tf=b.dataset.tf;hChart()});
document.querySelectorAll('th[data-sort]').forEach(th=>{th.style.cursor='pointer';th.onclick=()=>{let k=th.dataset.sort;H.sort=H.sort.k===k?{k,d:-H.sort.d}:{k,d:-1};hRender()}});
['#f_type','#f_side','#f_outcome','#f_year'].forEach(s=>$(s).onchange=hRender);$('#f_sym').oninput=hRender;$('#f_group').onchange=()=>{H.gsel=null;hRender()};
async function hLoad(force,full){clearTimeout(H.timer);try{let resp=await fetch('/api/history'+(full?'?full=1':force?'?force=1':''),{cache:'no-store'}),d=await resp.json();if(!resp.ok)throw Error(d.error||'request failed');
 let s=d.status,first=!H.rep&&d.report;H.rep=d.report||H.rep;
 $('#hsync').textContent=(s.state==='syncing'?'syncing ('+(s.phase||'')+')… ':'')+(s.error?'last sync failed: '+s.error+' ':'')+(H.rep?'built '+when(H.rep.generated_at):'');
 if(H.rep){if(first||$('#f_year').options.length<2){let yrs=[...new Set(H.rep.episodes.map(e=>e.open_date.slice(0,4)))].sort().reverse();$('#f_year').innerHTML='<option value=all>All</option>'+yrs.map(y=>'<option>'+y+'</option>').join('');
  $('#f_symlist').innerHTML=[...new Set(H.rep.episodes.map(e=>e.chart_symbol))].sort().map(t=>'<option value="'+esc(t)+'">').join('')}hRender();
  if(first){let m=location.hash.match(/^#(trade|ticker)=([\w.\-]+)/);if(m){if(m[1]==='trade'){H.sel=m[2];H.tf='1d'}else{H.ticker=m[2].toUpperCase();H.tf='ticker'}hChart();hRender()}}}
 H.timer=setTimeout(()=>hLoad(false),s.state==='syncing'||s.state==='never'?2500:300000)}catch(e){$('#hsync').textContent=e.message;H.timer=setTimeout(()=>hLoad(false),30000)}}
$('#h_refresh').onclick=()=>hLoad(true);
hLoad(false);
$('#refresh').onclick=()=>load(true);load(false);
</script></body></html>'''


class PersonalHTTPServer(ThreadingHTTPServer):
    app: PersonalDashboardApp
    bind_host: str


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        logger.debug(fmt, *args)

    def _send(self, body: bytes, status: int = 200, ctype: str = "application/json") -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _json(self, payload: Any, status: int = 200) -> None:
        self._send(json.dumps(analysis.sanitize(payload), allow_nan=False, default=str).encode(), status)

    def _host_ok(self) -> bool:
        if self.server.bind_host not in _LOOPBACK:
            return True   # explicitly bound to a LAN address; the operator chose to expose it
        header = self.headers.get("Host") or ""
        host = header.split("]")[0] + "]" if header.startswith("[") else header.rsplit(":", 1)[0]
        return host in _LOOPBACK

    def do_GET(self):  # noqa: N802
        if not self._host_ok():
            self._json({"error": "forbidden host"}, HTTPStatus.FORBIDDEN)
            return
        url = urlparse(self.path)
        if url.path == "/static/cynolycus_theme.css":
            serve_theme_css(self)
            return
        if url.path == "/":
            self._send(PAGE.replace("__NAV__", NAV_HTML).replace("__THEME__", THEME_LINK).encode(),
                       ctype="text/html; charset=utf-8")
            return
        try:
            q = {k: v[-1] for k, v in parse_qs(url.query or "").items()}
            if url.path == "/api/state":
                self._json(self.server.app.state())
            elif url.path == "/api/portfolio":
                self._json(self.server.app.portfolio(force=q.get("force") == "1"))
            elif url.path == "/api/history":
                self._json(self.server.app.history(force=q.get("force") == "1", full=q.get("full") == "1"))
            elif url.path == "/api/history/chart":
                self._json(self.server.app.history_chart(q.get("id"), q.get("ticker"), q.get("tf", "1d")))
            else:
                self._json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
        except KeyError as exc:
            self._json({"error": str(exc).strip("'\"")}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Personal dashboard request failed")
            self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self):  # noqa: N802
        if not self._host_ok():
            self._json({"error": "forbidden host"}, HTTPStatus.FORBIDDEN)
            return
        if (self.headers.get("Content-Type") or "").split(";")[0].strip() != "application/json":
            self._json({"error": "JSON body required"}, HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise ValueError("body too large")
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("JSON object required")
            path = urlparse(self.path).path
            if path == "/api/watchlist":
                self._json(self.server.app.add_watch(body.get("ticker", ""), body.get("note", "")))
            elif path == "/api/watchlist/remove":
                self._json(self.server.app.remove_watch(body.get("ticker", "")))
            elif path == "/api/refresh":
                self._json(self.server.app.portfolio(force=True))
            else:
                self._json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, KeyError) as exc:
            self._json({"error": str(exc).strip("'\"")}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Personal dashboard write failed")
            self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def make_server(host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                app: PersonalDashboardApp | None = None) -> PersonalHTTPServer:
    server = PersonalHTTPServer((host, port), Handler)
    server.app = app or PersonalDashboardApp()
    server.bind_host = host
    return server


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    server = make_server(args.host, args.port)
    logger.info("Personal portfolio: http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
