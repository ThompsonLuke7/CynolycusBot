"""Read-only tradable-universe watchlist over the shared universe and local outputs.

Prices/volumes are dated daily bars, scores are the latest per-ticker Meta matrix
rows, and news comes from the existing Library index. No broker calls or orders.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd

from UI import news_library
from UI.ui_chrome import NAV_HTML, THEME_LINK, serve_theme_css

logger = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parents[1]
UNIVERSE = REPO / "Data/shared/universe/shared_universe.csv"
DISCOVERY = REPO / "Data/shared/universe/pending_tickers.csv"
MATRIX = REPO / "signals/meta_context/meta_ranker/meta_ranker_matrix.parquet"
BARS = REPO / "Data/shared/bars/1d"
DEFAULT_PORT = 8777
_SCORE_COLUMNS = ["mom_score", "htf_score", "news_catalyst_score", "theme"]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if value is pd.NA or isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _number(value: Any) -> float | None:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    return n if math.isfinite(n) else None


def _string(value: Any) -> str | None:
    return None if pd.isna(value) or str(value).strip() in {"", "nan", "None"} else str(value).strip()


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _stamp(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
        return st.st_mtime_ns, st.st_size
    except OSError:
        return None


def _latest_scores(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(path).schema.names)
    cols = [c for c in _SCORE_COLUMNS if c in available]
    frame = pd.read_parquet(path, columns=cols).reset_index()
    if not {"ticker", "timestamp"}.issubset(frame.columns):
        raise ValueError("Meta matrix has no ticker/timestamp index")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("ticker", keep="last")
    out: dict[str, dict] = {}
    for row in frame.to_dict("records"):
        ticker = str(row["ticker"]).upper()
        out[ticker] = {
            "score_at": row["timestamp"].isoformat(),
            "momentum_score": _number(row.get("mom_score")),
            "htf_score": _number(row.get("htf_score")),
            "catalyst_score": _number(row.get("news_catalyst_score")),
            "matrix_theme": _string(row.get("theme")),
        }
    return out


def _daily_quote(path: Path) -> dict:
    if not path.exists():
        return {}
    frame = pd.read_parquet(path, columns=["timestamp", "close", "volume"])
    if frame.empty:
        return {}
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    if frame.empty:
        return {}
    latest = frame.iloc[-1]
    tail = frame.tail(20)
    return {
        "price": _number(latest["close"]),
        "price_at": pd.Timestamp(latest["timestamp"]).isoformat(),
        "volume": _number(latest["volume"]),
        "avg_volume_20d": _number(tail["volume"].mean()),
        "avg_dollar_volume_20d": _number((tail["close"] * tail["volume"]).mean()),
        "volume_days": len(tail),
    }


def _load_rows(universe: Path, matrix: Path, bars: Path, discovery: Path) -> list[dict]:
    frame = pd.read_csv(universe, dtype={"ticker": str}, low_memory=False)
    if "ticker" not in frame or "is_eligible" not in frame:
        raise ValueError("Shared universe lacks ticker/is_eligible columns")
    scores = _latest_scores(matrix)
    caps: dict[str, dict] = {}
    if discovery.exists():
        discovered = pd.read_csv(discovery, usecols=["ticker", "market_cap", "last_seen"])
        discovered = discovered.dropna(subset=["market_cap"]).drop_duplicates("ticker", keep="last")
        caps = {str(r["ticker"]).upper(): {"market_cap": _number(r["market_cap"]),
                                               "market_cap_at": _string(r["last_seen"])}
                for r in discovered.to_dict("records")}
    rows = []
    for rec in frame.to_dict("records"):
        ticker = str(rec["ticker"]).strip().upper()
        if not ticker or ticker == "NAN":
            continue
        row = {
            "ticker": ticker,
            "eligible": _bool(rec.get("is_eligible")),
            "cap_bucket": _string(rec.get("market_cap_bucket")) or _string(rec.get("cap_tier")) or "Unknown",
            "market_cap": _number(rec.get("market_cap")),
            "market_cap_at": None,
            "sector": _string(rec.get("sector")),
            "theme": next((_string(rec.get(k)) for k in ("theme_1", "theme_2", "theme_3") if _string(rec.get(k))), None),
            "momentum_universe": _bool(rec.get("in_momentum_candidate")),
            "swing_universe": _bool(rec.get("in_swing_live")),
            "discovered": _bool(rec.get("in_discovered")),
            "eligibility_reason": _string(rec.get("eligible_reason")),
        }
        if row["market_cap"] is None:
            row.update(caps.get(ticker, {}))
        row.update(scores.get(ticker, {}))
        if not row["theme"]:
            row["theme"] = row.get("matrix_theme")
        row.update(_daily_quote(bars / f"{ticker}.parquet"))
        rows.append(row)
    return rows


class UniverseDashboardApp:
    def __init__(self, *, universe: Path = UNIVERSE, matrix: Path = MATRIX, bars: Path = BARS,
                 discovery: Path = DISCOVERY) -> None:
        self.universe, self.matrix, self.bars, self.discovery = universe, matrix, bars, discovery
        self._lock = threading.Lock()
        self._warm_lock = threading.Lock()
        self._warm_thread: threading.Thread | None = None
        self._cache: tuple[float, tuple, list[dict]] | None = None

    def _rows(self) -> list[dict]:
        # Daily bars refresh in place; the directory mtime changes on replacement.
        # The bar directory mtime changes as individual ticker files are
        # refreshed, including during a warmup. Use the TTL below for bars;
        # treating the directory mtime as identity invalidates a just-built
        # cache and leaves the UI stuck in its loading state.
        signature = (_stamp(self.universe), _stamp(self.matrix), _stamp(self.discovery))
        with self._lock:
            if self._cache and self._cache[1] == signature and time.monotonic() - self._cache[0] < 300:
                return self._cache[2]
        # Do the expensive parquet/bar scan outside the lock. State polling and
        # the search/facet endpoints must remain responsive while this warms.
        rows = _load_rows(self.universe, self.matrix, self.bars, self.discovery)
        with self._lock:
            self._cache = (time.monotonic(), signature, rows)
            return rows

    def _cached_rows(self) -> list[dict] | None:
        signature = (_stamp(self.universe), _stamp(self.matrix), _stamp(self.discovery))
        with self._lock:
            if self._cache and self._cache[1] == signature:
                return self._cache[2]
        return None

    def _warm(self) -> None:
        with self._warm_lock:
            if self._warm_thread and self._warm_thread.is_alive():
                return
            def run() -> None:
                try:
                    self._rows()
                except Exception:
                    logger.exception("Universe cache refresh failed")
            self._warm_thread = threading.Thread(target=run, daemon=True,
                                                 name="universe-cache-refresh")
            self._warm_thread.start()

    def state(self) -> dict:
        signature = (_stamp(self.universe), _stamp(self.matrix), _stamp(self.discovery))
        cached = self._cache
        if cached is None:
            self._warm()
            return {"config": {"mode": "universe", "tradeable": False},
                    "building": True, "universe": None}
        if cached[1] != signature or time.monotonic() - cached[0] >= 300:
            self._warm()
        try:
            rows = cached[2]
            return {"ts": datetime.now(timezone.utc).isoformat(), "config": {"mode": "universe", "tradeable": False},
                    "universe": {"total": len(rows), "eligible": sum(r["eligible"] for r in rows),
                                 "priced": sum(r.get("price") is not None for r in rows)}}
        except Exception as exc:
            logger.warning("Universe state failed: %s", exc)
            return {"config": {"mode": "universe", "tradeable": False}, "error": str(exc)}

    def search(self, q: dict[str, list[str]]) -> dict:
        def one(key: str) -> str:
            return (q.get(key) or [""])[0].strip()

        rows = self._cached_rows()
        if rows is None:
            self._warm()
            return {"building": True, "message": "Loading tradable universe…",
                    "rows": [], "total": 0, "offset": 0, "limit": 100}
        needle = one("q").upper()
        theme = one("theme").lower()
        cap = one("cap")
        module = one("module")
        eligible = one("eligible")
        numeric_filters = [("min_price", "price", True), ("max_price", "price", False),
                           ("min_cap", "market_cap", True),
                           ("min_volume", "avg_volume_20d", True),
                           ("min_dollar_volume", "avg_dollar_volume_20d", True)]
        for key, field, minimum in numeric_filters:
            raw = one(key)
            if not raw:
                continue
            n = _number(raw)
            if n is None or n < 0:
                raise ValueError(f"Invalid {key}")
            rows = [r for r in rows if r.get(field) is not None and
                    (r[field] >= n if minimum else r[field] <= n)]
        if needle:
            rows = [r for r in rows if needle in r["ticker"] or needle in (r.get("sector") or "").upper()]
        if theme:
            rows = [r for r in rows if theme in (r.get("theme") or "").lower()]
        if cap:
            rows = [r for r in rows if r["cap_bucket"] == cap]
        if module == "momentum":
            rows = [r for r in rows if r["momentum_universe"]]
        elif module == "swing":
            rows = [r for r in rows if r["swing_universe"]]
        elif module not in {"", "all"}:
            raise ValueError("Invalid module")
        if eligible == "yes":
            rows = [r for r in rows if r["eligible"]]
        elif eligible == "no":
            rows = [r for r in rows if not r["eligible"]]
        elif eligible not in {"", "all"}:
            raise ValueError("Invalid eligible filter")
        total = len(rows)
        sort_fields = {"ticker", "price", "avg_volume_20d", "market_cap",
                       "momentum_score", "htf_score", "catalyst_score"}
        sort = one("sort") or "ticker"
        if sort not in sort_fields:
            raise ValueError("Invalid sort")
        descending = one("direction") == "desc"
        if one("direction") not in {"", "asc", "desc"}:
            raise ValueError("Invalid direction")
        rows = sorted(rows, key=lambda r: (r.get(sort) is None,
                      -(r[sort]) if descending and sort != "ticker" and r.get(sort) is not None
                      else r[sort] if r.get(sort) is not None else "", r["ticker"]))
        if sort == "ticker" and descending:
            rows.reverse()
        try:
            limit = max(1, min(200, int(one("limit") or 100)))
            offset = max(0, int(one("offset") or 0))
        except ValueError as exc:
            raise ValueError("Invalid page") from exc
        return {"rows": rows[offset:offset + limit], "total": total, "offset": offset, "limit": limit,
                "source": {"universe_at": datetime.fromtimestamp(self.universe.stat().st_mtime, timezone.utc).isoformat(),
                           "matrix_at": datetime.fromtimestamp(self.matrix.stat().st_mtime, timezone.utc).isoformat() if self.matrix.exists() else None}}

    def facets(self) -> dict:
        rows = self._cached_rows()
        if rows is None:
            self._warm()
            return {"building": True, "caps": [], "themes": []}
        return {"caps": sorted({r["cap_bucket"] for r in rows}),
                "themes": sorted({r["theme"] for r in rows if r.get("theme")})}

    def detail(self, ticker: str) -> dict:
        symbol = ticker.strip().upper()
        row = next((r for r in self._rows() if r["ticker"] == symbol), None)
        if row is None:
            raise KeyError(symbol)
        fields = ("timestamp", "headline", "source", "url", "catalyst_family",
                  "record_catalyst_score", "predicted_direction", "tone")
        current = news_library.index_is_current()
        news = {"rows": [], "total": 0}
        if current:
            news = news_library.search(ticker=symbol, limit=8)
            news["rows"] = [{k: rec.get(k) for k in fields} for rec in news["rows"]]
        elif news_library.INDEX_PATH.exists():
            # A live catalyst write can invalidate the stamp between nightly
            # builds. Read the last built index without rebuilding it inside
            # the trading server; mark the result stale in the UI.
            frame = pd.read_parquet(news_library.INDEX_PATH, columns=list(fields),
                                    filters=[("ticker", "==", symbol)])
            frame = frame.sort_values("timestamp", ascending=False)
            news = {"total": len(frame), "rows": frame.head(8).to_dict("records")}
        return {"stock": row, "news": news, "news_index_current": current,
                "news_index_available": current or news_library.INDEX_PATH.exists()}


PAGE = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tradable Universe — Cynolycus</title>__THEME__
<style>
main{max-width:1800px;margin:auto;padding:18px}h1{margin:0 0 4px;font-size:24px}.sub{color:var(--muted);margin-bottom:16px}
.filters{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0}.filters label{display:flex;flex-direction:column;gap:3px;color:var(--muted);font-size:11px}
.filters input,.filters select{min-width:110px}.filters input[name=q]{min-width:180px}
.summary{display:flex;gap:18px;flex-wrap:wrap;margin:12px 0;color:var(--muted)}.summary b{color:var(--text)}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 380px;gap:14px}@media(max-width:1100px){.layout{grid-template-columns:1fr}}
.scroll{max-height:70vh;overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:7px 8px;border-bottom:1px solid var(--border);text-align:left;white-space:nowrap}
th{position:sticky;top:0;background:var(--panel2);z-index:1;color:var(--muted);font-weight:600}tbody tr{cursor:pointer}tbody tr:hover,tbody tr.selected{background:#20332f}
.num{text-align:right;font-variant-numeric:tabular-nums}.muted{color:var(--muted)}.card-body{padding:12px}.score{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid var(--border)}
.news{padding:9px 0;border-bottom:1px solid var(--border)}.news a{color:var(--text)}.small{font-size:11px;color:var(--muted)}.pager{display:flex;gap:8px;align-items:center;margin-top:10px}
</style></head><body>__NAV__<main><h1>Tradable Universe</h1>
<div class="sub">Shared universe watchlist · dated daily prices and volume · latest available module scores · read-only</div>
<div class="filters">
<label>Search<input name=q placeholder="ticker or sector"></label>
<label>Market cap bucket<select name=cap><option value="">All buckets</option></select></label>
<label>Theme<select name=theme><option value="">All themes</option></select></label>
<label>Module universe<select name=module><option value="">All</option><option value=momentum>Momentum</option><option value=swing>Swing live</option></select></label>
<label>Eligibility<select name=eligible><option value=yes>Eligible</option><option value=all>All</option><option value=no>Ineligible</option></select></label>
<label>Sort<select name=sort><option value=ticker>Ticker</option><option value=price>Last close</option><option value=avg_volume_20d>20d volume</option><option value=market_cap>Recorded cap</option><option value=momentum_score>Momentum score</option><option value=htf_score>HTF score</option><option value=catalyst_score>Catalyst score</option></select></label>
<label>Direction<select name=direction><option value=asc>Ascending</option><option value=desc>Descending</option></select></label>
<label>Min recorded cap $<input name=min_cap type=number min=0 step=1></label>
<label>Min price $<input name=min_price type=number min=0 step=0.01></label>
<label>Max price $<input name=max_price type=number min=0 step=0.01></label>
<label>Min 20d volume<input name=min_volume type=number min=0 step=1></label>
<label>Min 20d $ volume<input name=min_dollar_volume type=number min=0 step=1></label>
</div><div id=error class=bad></div><div id=summary class=summary></div>
<div class=layout><section class=card><div class=card-head>Universe rows</div><div class=scroll><table><thead><tr><th>Ticker</th><th>Theme</th><th>Cap bucket</th><th class=num>Recorded cap</th><th>Sector</th><th class=num>Last close</th><th class=num>Volume</th><th class=num>20d avg vol</th><th class=num>Mom</th><th class=num>HTF</th><th class=num>Catalyst</th><th>Score time</th></tr></thead><tbody id=rows></tbody></table></div><div class="card-body pager"><button id=prev>Previous</button><span id=page></span><button id=next>Next</button></div></section>
<aside class=card><div class=card-head>Stock details</div><div id=detail class=card-body>Select a ticker to see source dates, scores, and recent news.</div></aside></div>
<div class="small" style="margin-top:14px">Last close and share volume are from the daily bar cache, not live quotes. Market cap buckets are universe metadata. Recorded caps come from the discovery ledger and may be months old; hover or open a ticker for the observation date. A cap filter excludes names without a recorded value. Scores are from each ticker's latest Meta matrix row and may predate its price. Blank fields mean unavailable, not zero. News is linked to the Library index.</div>
</main><script>
const $=s=>document.querySelector(s), fmt=(v,d=0)=>v==null?'—':Number(v).toLocaleString(undefined,{maximumFractionDigits:d}), esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let offset=0, total=0, timer, selected='';
function date(v){return v?new Date(v).toLocaleString():'—'}
async function load(){let p=new URLSearchParams();document.querySelectorAll('.filters [name]').forEach(e=>{if(e.value)p.set(e.name,e.value)});p.set('offset',offset);p.set('limit',100);
try{let resp=await fetch('/api/search?'+p);let data=await resp.json();if(!resp.ok)throw Error(data.error||'Request failed');if(data.building){$('#summary').textContent=data.message||'Loading tradable universe…';$('#rows').innerHTML='<tr><td colspan=12>Loading universe data…</td></tr>';setTimeout(load,500);return;}total=data.total;$('#error').textContent='';
$('#summary').innerHTML='<span><b>'+fmt(total)+'</b> matching tickers</span><span>Universe updated '+esc(date(data.source.universe_at))+'</span><span>Score file updated '+esc(date(data.source.matrix_at))+'</span>';
$('#rows').innerHTML=data.rows.map(r=>'<tr data-ticker="'+esc(r.ticker)+'" class="'+(r.ticker===selected?'selected':'')+'"><td><b>'+esc(r.ticker)+'</b></td><td>'+esc(r.theme||'—')+'</td><td>'+esc(r.cap_bucket)+'</td><td class=num title="'+esc(r.market_cap_at||'Date unavailable')+'">'+fmt(r.market_cap)+'</td><td>'+esc(r.sector||'—')+'</td><td class=num>'+fmt(r.price,2)+'</td><td class=num>'+fmt(r.volume)+'</td><td class=num>'+fmt(r.avg_volume_20d)+'</td><td class=num>'+fmt(r.momentum_score,3)+'</td><td class=num>'+fmt(r.htf_score,3)+'</td><td class=num>'+fmt(r.catalyst_score,3)+'</td><td>'+esc(r.score_at?date(r.score_at):'—')+'</td></tr>').join('')||'<tr><td colspan=12>No matching tickers</td></tr>';
$('#page').textContent=total?(offset+1)+'–'+Math.min(offset+100,total)+' of '+total:'0 of 0';$('#prev').disabled=offset===0;$('#next').disabled=offset+100>=total;
document.querySelectorAll('#rows tr[data-ticker]').forEach(tr=>tr.onclick=()=>detail(tr.dataset.ticker));
}catch(e){$('#error').textContent=e.message}}
async function detail(t){selected=t;document.querySelectorAll('#rows tr').forEach(tr=>tr.classList.toggle('selected',tr.dataset.ticker===t));$('#detail').textContent='Loading '+t+'…';try{let resp=await fetch('/api/detail?ticker='+encodeURIComponent(t));let d=await resp.json();if(!resp.ok)throw Error(d.error);let r=d.stock;
let scores=[['Momentum',r.momentum_score],['HTF Swing',r.htf_score],['News catalyst',r.catalyst_score]].map(x=>'<div class=score><span>'+x[0]+'</span><b>'+fmt(x[1],3)+'</b></div>').join('');
let news=d.news.rows.map(n=>'<div class=news><div class=small>'+esc(date(n.timestamp))+' · '+esc(n.source||'source unknown')+' · '+esc(n.catalyst_family||'')+'</div><a href="'+esc(/^https?:\/\//.test(n.url||'')?n.url:'#')+'" target=_blank rel="noopener noreferrer">'+esc(n.headline||'Untitled')+'</a><div class=small>Model: '+esc(n.predicted_direction||'—')+' · score '+fmt(n.record_catalyst_score,3)+'</div></div>').join('');
$('#detail').innerHTML='<h2>'+esc(t)+'</h2><div class=small>'+esc(r.sector||'Sector unknown')+' · '+esc(r.theme||'Theme unknown')+' · '+esc(r.cap_bucket)+' cap bucket</div><div class=score><span>Eligible / Momentum / Swing</span><b>'+[r.eligible,r.momentum_universe,r.swing_universe].map(x=>x?'Yes':'No').join(' / ')+'</b></div><div class=score><span>Recorded market cap</span><b>'+fmt(r.market_cap)+'</b></div><div class=small>Cap observed: '+esc(r.market_cap_at||'unknown')+'</div><div class=score><span>Last close</span><b>$'+fmt(r.price,2)+'</b></div><div class=small>Price bar: '+esc(date(r.price_at))+' · Volume: '+fmt(r.volume)+' shares · 20d average: '+fmt(r.avg_volume_20d)+' shares ('+fmt(r.volume_days)+' sessions)</div><h3>Module scores</h3>'+scores+'<div class=small>Score observation: '+esc(date(r.score_at))+'</div><h3>Recent news</h3>'+(d.news_index_available?((d.news_index_current?'':'<div class=small>News index refresh pending; showing the last built snapshot.</div>')+(news||'<div class=small>No indexed records.</div>')):'<div class=small>News index unavailable. Open Library for current status.</div>')+'<p><a href="http://'+location.hostname+':8775/?ticker='+encodeURIComponent(t)+'">Open '+esc(t)+' in Library →</a></p>';
}catch(e){$('#detail').textContent=e.message}}
async function facets(){try{let f=await(await fetch('/api/facets')).json();if(f.building){setTimeout(facets,500);return;}for(let [name,values] of [['cap',f.caps||[]],['theme',f.themes||[]]]){let el=$('[name='+name+']');el.innerHTML='<option value="">All '+name+'s</option>'+values.map(v=>'<option value="'+esc(v)+'">'+esc(v)+'</option>').join('')}}catch(e){$('#error').textContent=e.message}}
document.querySelectorAll('.filters [name]').forEach(e=>e.addEventListener(e.tagName==='SELECT'?'change':'input',()=>{clearTimeout(timer);offset=0;timer=setTimeout(load,250)}));$('#prev').onclick=()=>{offset=Math.max(0,offset-100);load()};$('#next').onclick=()=>{offset+=100;load()};facets();load();
</script></body></html>'''


class UniverseHTTPServer(ThreadingHTTPServer):
    app: UniverseDashboardApp


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        logger.debug(fmt, *args)

    def _send(self, body: bytes, status: int = 200, ctype: str = "application/json") -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _json(self, payload: Any, status: int = 200) -> None:
        self._send(json.dumps(_json_safe(payload), allow_nan=False, default=str).encode(), status)

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        q = parse_qs(urlparse(self.path).query)
        if path == "/static/cynolycus_theme.css":
            serve_theme_css(self)
            return
        if path == "/":
            self._send(PAGE.replace("__NAV__", NAV_HTML).replace("__THEME__", THEME_LINK).encode(), ctype="text/html; charset=utf-8")
            return
        try:
            if path == "/api/state":
                self._json(self.server.app.state())
            elif path == "/api/search":
                self._json(self.server.app.search(q))
            elif path == "/api/facets":
                self._json(self.server.app.facets())
            elif path == "/api/detail":
                self._json(self.server.app.detail((q.get("ticker") or [""])[0]))
            else:
                self._json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, KeyError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            logger.exception("Universe request failed")
            self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def make_server(host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                app: UniverseDashboardApp | None = None) -> UniverseHTTPServer:
    server = UniverseHTTPServer((host, port), Handler)
    server.app = app or UniverseDashboardApp()
    return server


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    server = make_server(args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
