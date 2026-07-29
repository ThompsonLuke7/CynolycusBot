"""Library dashboard — searchable news/catalyst archive with a price overlay.

Read-only browser over every signal record the pipelines have collected (see
``UI/news_library.py`` for the stores and how they are deduped). Search by ticker
or headline text, narrow with the catalyst/source/relation filters, and read the
results most-recent-first against a daily price chart of the same ticker, so news
flow can be eyeballed against price action.

No account, no orders, no toggles — this dashboard never touches a broker.

Endpoints:
  GET  /                    → HTML page
  GET  /api/state           → index status + performance badge payload
  GET  /api/search?…        → records page (ticker/query/family/source/relation/dates)
  GET  /api/facets?ticker=  → distinct filter values, ticker-scoped when given
  GET  /api/tickers?q=      → ticker suggestions for the search box
  GET  /api/price?ticker=…  → daily bars + per-day news counts for the chart
"""
from __future__ import annotations

import json
import logging
import math
import subprocess
import sys
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from UI import news_library
from UI.ui_chrome import NAV_HTML, THEME_LINK, serve_theme_css

logger = logging.getLogger(__name__)

DEFAULT_TICKER = "AAPL"
DEFAULT_DAYS = 365


def _json_safe(o: Any) -> Any:
    if o is None or isinstance(o, (bool, int, str)):
        return o
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, (list, tuple)):
        return [_json_safe(x) for x in o]
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    return str(o)


class LibraryDashboardApp:
    """Query layer over the news library index.

    The index build reads the full 425MB record store and peaks near 650MB, so it
    runs as a SUBPROCESS rather than in-process — the combined server is the
    pandas-heavy process that already OOM-killed once at the WSL cap, and a
    browse-only page must not be what pushes it there again. Until the build
    lands, requests answer with ``building`` instead of blocking a request
    thread.
    """

    def __init__(self, *, default_ticker: str = DEFAULT_TICKER) -> None:
        self.default_ticker = default_ticker.upper()
        self._build_thread: threading.Thread | None = None
        self._build_error: str | None = None
        self._lock = threading.Lock()

    # -- index lifecycle -------------------------------------------------
    def _building(self) -> bool:
        t = self._build_thread
        return bool(t and t.is_alive())

    def _start_build(self) -> None:
        with self._lock:
            if self._building():
                return

            def _run() -> None:
                argv = [sys.executable, "-m", "UI.news_library", "--force"]
                logger.info("News library index: building via %s", " ".join(argv))
                try:
                    proc = subprocess.run(argv, cwd=str(REPO), capture_output=True, text=True)
                except Exception as exc:  # noqa: BLE001
                    self._build_error = str(exc)
                    logger.error("News library index build failed: %s", exc)
                    return
                if proc.returncode != 0:
                    self._build_error = (proc.stderr or "").strip()[-500:] or "build failed"
                    logger.error("News library index build exited %d: %s",
                                 proc.returncode, self._build_error)
                else:
                    self._build_error = None
                    logger.info("News library index build complete.")

            self._build_thread = threading.Thread(
                target=_run, daemon=True, name="news-library-index-build")
            self._build_thread.start()

    def _index_ready(self) -> dict | None:
        """None when the index is usable; otherwise a status payload to return."""
        if news_library.index_is_current():
            return None
        self._start_build()
        return {
            "building": True,
            "error": self._build_error,
            "message": "Building the news library index (one-off after each data refresh)…",
            "rows": [], "total": 0, "bars": [], "events": [],
        }

    # -- API -------------------------------------------------------------
    def state(self) -> dict:
        st = news_library.status()
        st["building"] = self._building()
        st["build_error"] = self._build_error
        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "config": {"mode": "library", "tradeable": False,
                       "default_ticker": self.default_ticker},
            "index": st,
        }

    def search(self, q: dict[str, list[str]]) -> dict:
        pending = self._index_ready()
        if pending:
            return pending

        def one(key: str) -> str | None:
            v = (q.get(key) or [""])[0].strip()
            return v or None

        try:
            limit = int((q.get("limit") or ["100"])[0])
            offset = int((q.get("offset") or ["0"])[0])
        except ValueError:
            limit, offset = 100, 0
        ticker = one("ticker")
        try:
            out = news_library.search(
                ticker=ticker.upper() if ticker else None,
                query=one("q"), family=one("family"),
                source=one("source"), origin=one("origin"), relation=one("relation"),
                start=one("start"), end=one("end"), limit=limit, offset=offset,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Library search failed: %s", exc)
            return {"error": f"search_failed: {exc}", "rows": [], "total": 0}
        out["building"] = False
        return out

    def facets(self, q: dict[str, list[str]]) -> dict:
        pending = self._index_ready()
        if pending:
            return pending
        try:
            return {"facets": news_library.facets((q.get("ticker") or [""])[0])}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"facets_failed: {exc}", "facets": {}}

    def tickers(self, q: dict[str, list[str]]) -> dict:
        pending = self._index_ready()
        if pending:
            return pending
        try:
            return {"tickers": news_library.tickers((q.get("q") or [""])[0], limit=40)}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"tickers_failed: {exc}", "tickers": []}

    def price(self, q: dict[str, list[str]]) -> dict:
        """Daily bars plus the matching per-day news counts for the overlay.

        The events are counted from the SAME filtered query the table shows, so a
        marker on the chart always corresponds to rows the user can scroll to —
        the chart never implies coverage the table cannot back up.
        """
        pending = self._index_ready()
        if pending:
            return pending
        ticker = ((q.get("ticker") or [""])[0] or self.default_ticker).strip().upper()
        try:
            days = int((q.get("days") or [str(DEFAULT_DAYS)])[0])
        except ValueError:
            days = DEFAULT_DAYS
        try:
            out = news_library.price_series(ticker, days=days)
        except Exception as exc:  # noqa: BLE001
            out = {"ticker": ticker, "bars": [], "error": f"price_failed: {exc}"}

        events: list[dict] = []
        if out.get("bars"):
            try:
                import pandas as pd

                def one(key: str) -> str | None:
                    v = (q.get(key) or [""])[0].strip()
                    return v or None

                # Clip the user's date filter to the plotted bar range: an event
                # outside it has nowhere to sit on the x-axis, and widening past
                # the user's filter would put markers on the chart that the
                # table below does not list.
                bar_start = pd.Timestamp(out["bars"][0]["t"])
                user_start = news_library._norm_ts(one("start"))
                start_ts = max(bar_start, user_start) if user_start is not None else bar_start
                agg = news_library.event_days(
                    ticker=ticker, query=one("q"), family=one("family"),
                    source=one("source"), origin=one("origin"), relation=one("relation"),
                    start=start_ts, end=one("end"),
                    color_by=(one("color_by") or "catalyst_family"),
                )
                events = agg["days"]
                out["classes"] = agg["classes"]
                out["color_by"] = agg["color_by"]
                out["events_total"] = agg["total"]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Library event overlay failed: %s", exc)
                out["events_error"] = str(exc)
        out["events"] = events
        out.setdefault("classes", [])
        out["building"] = False
        return out


_PAGE = r"""<!doctype html><html><head><meta charset=utf-8><title>Library</title>
__THEME_LINK__
<style>
/* Chart roles. The suite is dark-only (ui_chrome pins color-scheme:dark), so
   these are the palette's DARK steps, validated against this surface (#0d1117).
   The three --cls-* marker hues are palette slots 1-3 and are the MOST that
   clear the all-pairs colourblind + normal-vision floors here: every candidate
   4th hue fails (magenta/aqua dE 1.6 deutan, yellow/orange 10.6 normal,
   red/orange 7.1 normal). Anything past three folds into --cls-other.
   Candles stay neutral on purpose: this chart's subject is news-vs-price, and
   green/red candles would put five competing hues on one small plot. Direction
   is carried by hollow(up)/filled(down) instead, which is also CVD-safe. */
.viz{--cls-1:#3987e5;--cls-2:#d95926;--cls-3:#199e70;--cls-other:#6e7681;
  --candle:#8b949e;--candle-dim:#6e7681;
  --grid:#2c2c2a;--axis:#383835;--ink-muted:#898781}
.wrap{margin:12px 16px 40px}
h2{margin:12px 0 4px;font-size:15px}
.filters{display:flex;gap:8px;flex-wrap:wrap;align-items:flex-end;margin:10px 0}
.filters label{display:flex;flex-direction:column;gap:3px;font-size:11px;color:var(--muted);
  text-transform:uppercase;letter-spacing:.3px;font-weight:700}
.filters input,.filters select{min-width:120px}
.filters input.tick{min-width:96px;text-transform:uppercase;font-weight:700}
.filters input.q{min-width:240px}
.chart-card{margin:10px 0;position:relative}
.chart-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.chart-head .spacer{flex:1}
.chart-head select{padding:3px 6px;font-size:11px;text-transform:none;letter-spacing:0}
.chart-head button{padding:3px 8px;font-size:11px}
#chart{width:100%;height:400px;display:block;background:var(--bg);cursor:crosshair;
  touch-action:none;user-select:none}
.legend{display:flex;gap:14px;align-items:center;padding:6px 12px;font-size:11px;
  color:var(--muted);flex-wrap:wrap}
.legend .key{display:inline-flex;align-items:center;gap:6px}
.legend .candle{width:7px;height:13px;border:1.5px solid var(--ink-muted);border-radius:1px}
.legend .candle.down{background:var(--ink-muted)}
.legend .dot{width:10px;height:10px;border-radius:50%;box-shadow:0 0 0 2px var(--bg)}
.hint{padding:0 12px 7px;font-size:11px;color:var(--ink-muted)}
.hint kbd{background:var(--panel2);border:1px solid var(--border);border-radius:3px;
  padding:0 4px;font-family:inherit;font-size:10px}
#tip{position:absolute;pointer-events:none;display:none;z-index:9;max-width:340px;
  background:var(--panel);border:1px solid var(--border);border-radius:6px;padding:7px 9px;
  font-size:11px;line-height:1.4;box-shadow:0 6px 20px rgba(0,0,0,.5)}
#tip .th{font-weight:700;color:var(--text);margin-bottom:3px}
#tip li{margin:2px 0 0;color:var(--muted)}
#tip ul{margin:4px 0 0;padding-left:14px}
table{border-collapse:collapse;width:100%;margin:6px 0;font-size:12px}
td,th{border-bottom:1px solid var(--border);padding:5px 8px;text-align:left;vertical-align:top}
th{background:var(--panel2);position:sticky;top:34px;font-size:11px;text-transform:uppercase;
  letter-spacing:.3px;color:var(--muted);font-weight:700}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
/* Everything right of this divider is HINDSIGHT — realized after the record —
   so it is visually demoted and can never be read as a decision-time signal.
   Only the first hindsight column carries the divider. */
td.fwd,th.fwd{color:var(--ink-muted)}
td.fwd0,th.fwd0{border-left:2px dashed var(--border)}
th.fwd{cursor:help}
th[title]{cursor:help}
/* status colours ship WITH the word, never as the only cue */
.oc{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.3px;
  padding:1px 6px;border-radius:9px}
.oc-good{background:#0f3a1f;color:#0ca30c}
.oc-bad{background:#3a1010;color:#d03b3b}
.oc-flat{background:var(--panel2);color:var(--muted)}
td.ts{white-space:nowrap;color:var(--muted);font-variant-numeric:tabular-nums}
tr.hl td{background:#1d2b3a}
.hd{color:var(--text)}
.status{margin:6px 0;font-size:12px}
.pager{display:flex;gap:8px;align-items:center;margin:10px 0}
</style></head><body class=viz>
__NAV_HTML__
<div class=wrap>
<h2>Library <span class=pill>read-only</span> <span id=idx class=muted></span></h2>
<div class=muted>Every news &amp; catalyst record the pipelines collected, searchable by ticker or
headline and plotted against daily price so news flow can be read against price action.</div>

<div class=filters>
  <label>Ticker<input class=tick id=f_ticker list=tickerlist placeholder=AAPL></label>
  <datalist id=tickerlist></datalist>
  <label>Headline contains<input class=q id=f_q placeholder="e.g. FDA approval"></label>
  <label>Catalyst family<select id=f_family></select></label>
  <label>Source<select id=f_source></select></label>
  <label>Relation<select id=f_relation></select></label>
  <label>Origin<select id=f_origin></select></label>
  <label>From<input type=date id=f_start></label>
  <label>To<input type=date id=f_end></label>
  <label>Chart range<select id=f_days>
    <option value=90>3M</option><option value=180>6M</option>
    <option value=365 selected>1Y</option><option value=1095>3Y</option>
    <option value=0>All</option></select></label>
  <button class=primary id=go>Search</button>
  <button id=clear>Clear</button>
</div>

<div class="card chart-card">
  <div class="card-head chart-head">
    <span id=chartTitle>Price &amp; news</span>
    <span class=spacer></span>
    <label>colour markers by
      <select id=f_colorby>
        <option value=catalyst_family selected>catalyst family</option>
        <option value=source>source</option>
        <option value=relation_type>relation</option>
        <option value=origin>origin</option>
      </select></label>
    <button id=resetView>Reset view</button>
  </div>
  <canvas id=chart></canvas>
  <div id=tip></div>
  <div class=legend id=legend></div>
  <div class=hint>Scroll to zoom time (anchored on the cursor) ·
    <kbd>Shift</kbd>+scroll to pan · scroll over the price axis to zoom price ·
    drag to pan · double-click to reset. Marker size = records that day.</div>
</div>

<h2>Records <span id=count class=muted></span></h2>
<div class=status id=status></div>
<table id=tbl></table>
<div class=pager>
  <button id=prev>&larr; Newer</button><button id=next>Older &rarr;</button>
  <span class=muted id=pageinfo></span>
</div>
</div>
<script>
var STATE={offset:0,limit:100,bars:[],events:[],classes:[],
           colorBy:'catalyst_family',hl:null};
var $=function(id){return document.getElementById(id)};

function params(extra){
  var p=new URLSearchParams();
  var m={ticker:'f_ticker',q:'f_q',family:'f_family',source:'f_source',
         relation:'f_relation',origin:'f_origin',start:'f_start',end:'f_end'};
  for(var k in m){var v=($(m[k]).value||'').trim(); if(v) p.set(k,k==='ticker'?v.toUpperCase():v);}
  if(extra) for(var k2 in extra) p.set(k2,extra[k2]);
  return p;
}
function esc(s){return (s==null?'':String(s)).replace(/[&<>"]/g,function(c){
  return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});}
function fmtTs(s){if(!s)return '-';
  try{return new Date(s).toLocaleString('en-US',{timeZone:'America/New_York',
    year:'numeric',month:'short',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false});}
  catch(e){return s;}}
function dayET(s){try{var d=new Date(s);
  return new Intl.DateTimeFormat('en-CA',{timeZone:'America/New_York'}).format(d);}catch(e){return '';}}
function num(v,d){return (v==null||isNaN(v))?'':Number(v).toFixed(d==null?3:d);}
function pct(v,sign){if(v==null||isNaN(v))return '';
  var n=Number(v), cls=n>0?'pos':(n<0?'neg':'');
  return '<span class="'+cls+'">'+(sign&&n>0?'+':'')+n.toFixed(2)+'%</span>';}
/* good/bad/neutral always ships the WORD, never colour alone */
function outcomeCell(o){if(!o)return '';
  var cls=o==='good'?'oc-good':(o==='bad'?'oc-bad':'oc-flat');
  return '<span class="oc '+cls+'">'+esc(o)+'</span>';}
function dirCell(d){if(!d)return '';
  var cls=d==='bullish'?'oc-good':(d==='bearish'?'oc-bad':'oc-flat');
  return '<span class="oc '+cls+'">'+esc(d)+'</span>';}
function toneCell(t){if(!t)return '';
  var cls=t==='positive'?'oc-good':(t==='negative'?'oc-bad':'oc-flat');
  return '<span class="oc '+cls+'">'+esc(t)+'</span>';}
function maxCell(up,dn){
  if((up==null||isNaN(up))&&(dn==null||isNaN(dn)))return '';
  var u=(up==null||isNaN(up))?null:Number(up), d=(dn==null||isNaN(dn))?null:Number(dn);
  return '<span class=pos>'+(u==null?'-':'+'+u.toFixed(1)+'%')+'</span>'
    +'<span class=muted> / </span><span class=neg>'+(d==null?'-':d.toFixed(1)+'%')+'</span>';}

/* ---------- records table ---------- */
async function loadRecords(){
  var p=params({limit:STATE.limit,offset:STATE.offset});
  $('status').textContent='Searching…';
  var r=await (await fetch('/api/search?'+p.toString())).json();
  if(r.building){$('status').textContent=r.message||'Building index…';
    $('tbl').innerHTML='';setTimeout(loadRecords,4000);return;}
  if(r.error){$('status').textContent=r.error;$('tbl').innerHTML='';return;}
  $('status').textContent='';
  $('count').textContent=r.total.toLocaleString()+' matching record'+(r.total===1?'':'s');
  var head='<tr><th>Time (ET)</th><th>Ticker</th><th>Headline</th><th>Source</th>'
    +'<th>Family</th><th>Relation</th>'
    +'<th class=num title="The deployed catalyst model\'s score for THIS headline '
    +'(news_catalyst_per_record, refreshed nightly).">Score</th>'
    +'<th title="Decision-time direction from the multiclass trajectory model, using '
    +'the same thresholds build_news_signal.py aggregates on: bullish if '
    +'p_bull_steady+p_bull_volatile >= 0.50, bearish if p_crash_stayed >= 0.30.">'
    +'Predicted</th>'
    +'<th title="FinBERT tone of the headline text — a top-3 feature of the model '
    +'above. Tone is what the text SAYS, not what the model expects price to do.">'
    +'Tone</th>'
    +'<th class=num title="The per-(ticker, day) news_catalyst_score the live meta '
    +'ranker trades on — the same number the system saw. Scores the DAY, not this '
    +'single headline.">Day</th>'
    +'<th class="num fwd fwd0" title="HINDSIGHT. Realized return of the next session '
    +'after this record.">1d move &#9888;</th>'
    +'<th class="num fwd" title="HINDSIGHT. Percentile of that 1d move within THIS '
    +'ticker\'s own news history — ranked per ticker because a 3% move means very '
    +'different things for a biotech and a mega-cap.">Rank &#9888;</th>'
    +'<th class="fwd" title="HINDSIGHT. good / bad / neutral from the realized 1d '
    +'move against a fixed &plusmn;2% band. A display convention, not a model output.">'
    +'Outcome &#9888;</th>'
    +'<th class="num fwd" title="HINDSIGHT. Best gain and worst drawdown reached over '
    +'the label horizon after this record.">Max +/&minus; &#9888;</th></tr>';
  $('tbl').innerHTML=head+(r.rows||[]).map(function(x){
    var h=esc(x.headline||'(no headline)');
    var hd=x.url?('<a href="'+esc(x.url)+'" target=_blank rel=noopener>'+h+'</a>'):h;
    var sub=x.catalyst_subtype?(' <span class=muted>· '+esc(x.catalyst_subtype)+'</span>'):'';
    return '<tr data-day="'+dayET(x.timestamp)+'">'
      +'<td class=ts>'+fmtTs(x.timestamp)+'</td>'
      +'<td><b>'+esc(x.ticker)+'</b></td>'
      +'<td class=hd>'+hd+'</td>'
      +'<td class=muted>'+esc(x.source||'')+' <span class=muted>('+esc(x.origin||'')+')</span></td>'
      +'<td class=muted>'+esc(x.catalyst_family||'')+sub+'</td>'
      +'<td class=muted>'+esc(x.relation_type||'')+'</td>'
      +'<td class=num>'+num(x.record_catalyst_score)+'</td>'
      +'<td>'+dirCell(x.predicted_direction)+'</td>'
      +'<td>'+toneCell(x.tone)+'</td>'
      +'<td class=num>'+num(x.day_catalyst_score)+'</td>'
      +'<td class="num fwd fwd0">'+pct(x.move_1d_pct,true)+'</td>'
      +'<td class="num fwd">'+(x.move_rank_pct==null?'':Math.round(x.move_rank_pct))+'</td>'
      +'<td class="fwd">'+outcomeCell(x.outcome)+'</td>'
      +'<td class="num fwd">'+maxCell(x.max_favorable_pct,x.max_adverse_pct)+'</td></tr>';
  }).join('');
  var from=r.total?STATE.offset+1:0, to=Math.min(STATE.offset+STATE.limit,r.total);
  $('pageinfo').textContent=from+'–'+to+' of '+r.total.toLocaleString();
  $('prev').disabled=STATE.offset<=0; $('next').disabled=to>=r.total;
  applyHighlight();
}

/* ---------- price + news chart ----------
   VIEW is the viewport: [i0,i1] is a fractional index window into the bar array
   (horizontal), yLo/yHi the price window (vertical). yAuto refits price to the
   visible bars on every pan/zoom until the user zooms price explicitly, which is
   what makes plain horizontal scrolling feel right. */
var VIEW={i0:null,i1:null,yLo:null,yHi:null,yAuto:true};
var SCALE=null, DRAG=null, HOVER=null;
var MIN_BARS=4;   /* tightest horizontal zoom */

function resetView(){VIEW={i0:null,i1:null,yLo:null,yHi:null,yAuto:true};draw();}

async function loadChart(){
  var t=($('f_ticker').value||'').trim().toUpperCase();
  if(!t){STATE.bars=[];STATE.events=[];STATE.classes=[];
    $('chartTitle').textContent='Price & news';resetView();renderLegend('Enter a ticker to plot price.');return;}
  var p=params({days:$('f_days').value,color_by:$('f_colorby').value});
  var r=await (await fetch('/api/price?'+p.toString())).json();
  if(r.building){setTimeout(loadChart,4000);return;}
  STATE.bars=r.bars||[];STATE.events=r.events||[];STATE.classes=r.classes||[];
  STATE.colorBy=r.color_by||$('f_colorby').value;
  $('chartTitle').textContent=t+' — daily candles & matching news';
  var n=STATE.events.length;
  var note=r.error?r.error:(r.events_error?('overlay failed: '+r.events_error)
    :(n?(n.toLocaleString()+' news day'+(n===1?'':'s')+' · '
        +(r.events_total||0).toLocaleString()+' records in view'):'no matching news in this range'));
  resetView();          /* a new ticker/range always starts fully zoomed out */
  renderLegend(note);
}

/* Marker colour by class. Only the server-selected top classes get a hue; the
   4th+ fold into the neutral "Other" — see event_days() for why three is the cap. */
function classColor(name){
  var cs=getComputedStyle(document.body);
  if(name==='Other') return cs.getPropertyValue('--cls-other').trim();
  var i=STATE.classes.findIndex(function(c){return c.name===name;});
  if(i<0||i>2) return cs.getPropertyValue('--cls-other').trim();
  return cs.getPropertyValue('--cls-'+(i+1)).trim();
}
function renderLegend(note){
  var html='<span class=key><span class="candle"></span><span class="candle down"></span> '
    +'Daily candle (hollow = up)</span>';
  STATE.classes.forEach(function(c){
    html+='<span class=key><span class=dot style="background:'+classColor(c.name)+'"></span> '
      +esc(c.name)+' <span class=muted>('+c.count.toLocaleString()+')</span></span>';
  });
  if(note) html+='<span class="key muted">'+esc(note)+'</span>';
  $('legend').innerHTML=html;
  draw();
}

/* ---- viewport helpers ---- */
function clampView(){
  var n=STATE.bars.length; if(!n) return;
  if(VIEW.i0==null||VIEW.i1==null){VIEW.i0=0;VIEW.i1=n-1;return;}
  var span=Math.min(Math.max(VIEW.i1-VIEW.i0,MIN_BARS-1),n-1);
  if(VIEW.i0<0){VIEW.i0=0;}
  VIEW.i1=VIEW.i0+span;
  if(VIEW.i1>n-1){VIEW.i1=n-1;VIEW.i0=VIEW.i1-span;}
  if(VIEW.i0<0)VIEW.i0=0;
}
function visibleRange(){
  clampView();
  return [Math.max(0,Math.floor(VIEW.i0)),Math.min(STATE.bars.length-1,Math.ceil(VIEW.i1))];
}
function autoPrice(){
  var r=visibleRange(),bars=STATE.bars,lo=Infinity,hi=-Infinity;
  for(var i=r[0];i<=r[1];i++){if(bars[i].l<lo)lo=bars[i].l;if(bars[i].h>hi)hi=bars[i].h;}
  if(!isFinite(lo)||!isFinite(hi)){lo=0;hi=1;}
  var pad=((hi-lo)||Math.abs(hi)||1)*0.10;
  return [lo-pad,hi+pad*1.6];   /* extra headroom: markers sit above the highs */
}

function draw(){
  var c=$('chart'), box=c.getBoundingClientRect(), dpr=window.devicePixelRatio||1;
  var w=Math.max(360,box.width), h=c.clientHeight||400;
  c.width=w*dpr; c.height=h*dpr;
  var g=c.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,w,h);
  var cs=getComputedStyle(document.body);
  var COL={grid:cs.getPropertyValue('--grid').trim(),
           axis:cs.getPropertyValue('--axis').trim(),
           ink:cs.getPropertyValue('--ink-muted').trim(),
           candle:cs.getPropertyValue('--candle').trim(),
           candleDim:cs.getPropertyValue('--candle-dim').trim(),
           bg:cs.getPropertyValue('--bg').trim()};
  var bars=STATE.bars;
  if(!bars.length){g.fillStyle=COL.ink;g.font='12px system-ui';
    g.fillText('No daily bars for this ticker.',16,h/2);SCALE=null;return;}

  var pad={l:8,r:62,t:12,b:24};
  clampView();
  var i0=VIEW.i0,i1=VIEW.i1,plotW=w-pad.l-pad.r,plotH=h-pad.t-pad.b;
  var lo,hi;
  if(VIEW.yAuto){var ap=autoPrice();lo=ap[0];hi=ap[1];}else{lo=VIEW.yLo;hi=VIEW.yHi;}
  if(!(hi>lo)){hi=lo+1;}
  var x=function(i){return pad.l+((i-i0)/((i1-i0)||1))*plotW;};
  var iAt=function(px){return i0+((px-pad.l)/plotW)*((i1-i0)||1);};
  var y=function(v){return pad.t+(1-(v-lo)/(hi-lo))*plotH;};
  var pAt=function(py){return hi-((py-pad.t)/plotH)*(hi-lo);};

  /* recessive grid + right-hand price axis */
  g.font='10px system-ui';g.textBaseline='middle';
  for(var k=0;k<=5;k++){
    var v=lo+(hi-lo)*k/5, yy=y(v);
    g.strokeStyle=COL.grid;g.lineWidth=1;
    g.beginPath();g.moveTo(pad.l,Math.round(yy)+.5);g.lineTo(w-pad.r,Math.round(yy)+.5);g.stroke();
    g.fillStyle=COL.ink;g.textAlign='left';g.fillText(v.toFixed(2),w-pad.r+6,yy);
  }
  /* price-axis gutter is its own hit zone (vertical zoom) — mark it */
  g.strokeStyle=COL.axis;g.beginPath();
  g.moveTo(w-pad.r+.5,pad.t);g.lineTo(w-pad.r+.5,h-pad.b);g.stroke();
  g.beginPath();g.moveTo(pad.l,h-pad.b+.5);g.lineTo(w-pad.r,h-pad.b+.5);g.stroke();

  /* time axis: evenly spaced session labels from the VISIBLE window */
  var vr=visibleRange();
  g.textBaseline='top';g.fillStyle=COL.ink;
  var ticks=Math.max(2,Math.min(6,Math.floor(plotW/110)));
  for(var ti=0;ti<=ticks;ti++){
    var idx=Math.round(vr[0]+(vr[1]-vr[0])*ti/ticks);
    if(idx<0||idx>=bars.length)continue;
    var px=x(idx); if(px<pad.l-2||px>w-pad.r+2)continue;
    g.textAlign=ti===0?'left':(ti===ticks?'right':'center');
    g.fillText(bars[idx].t.slice(0,10),Math.max(pad.l,Math.min(px,w-pad.r)),h-pad.b+5);
  }

  /* ---- candles. Neutral by design (see the CSS note): hollow = up, filled =
     down, so direction survives colourblindness and never competes with the
     categorical news markers. ---- */
  var step=plotW/((i1-i0)||1);
  var cw=Math.max(1,Math.min(14,step*0.68));
  var thin=cw<2.5;   /* too dense for bodies — draw as a wick-only bar chart */
  for(var i=vr[0];i<=vr[1];i++){
    var b=bars[i],xx=x(i);
    if(xx<pad.l-cw||xx>w-pad.r+cw)continue;
    var up=b.c>=b.o, col=up?COL.candle:COL.candleDim;
    g.strokeStyle=col;g.fillStyle=col;g.lineWidth=1;
    g.beginPath();g.moveTo(Math.round(xx)+.5,y(b.h));g.lineTo(Math.round(xx)+.5,y(b.l));g.stroke();
    if(thin)continue;
    var yo=y(Math.max(b.o,b.c)), yc=y(Math.min(b.o,b.c)), bh=Math.max(1,yc-yo);
    if(up){g.lineWidth=1.2;g.strokeRect(Math.round(xx-cw/2)+.5,Math.round(yo)+.5,Math.round(cw),Math.round(bh));}
    else{g.fillRect(Math.round(xx-cw/2),Math.round(yo),Math.round(cw),Math.round(bh));}
  }

  /* ---- news markers, above the session high so they never hide a candle ---- */
  var byDay={};bars.forEach(function(b,i){byDay[b.t.slice(0,10)]=i;});
  var maxN=1;STATE.events.forEach(function(e){if(e.count>maxN)maxN=e.count;});
  var pts=[];
  STATE.events.forEach(function(e){
    var i=byDay[e.day];
    if(i==null){ /* news on a non-session day (weekend/holiday) → next session */
      for(var j=0;j<bars.length;j++){if(bars[j].t.slice(0,10)>=e.day){i=j;break;}} }
    if(i==null||i<vr[0]-1||i>vr[1]+1)return;
    var px=x(i); if(px<pad.l-12||px>w-pad.r+12)return;
    var r=4+4.5*Math.sqrt(Math.min(e.count,maxN)/maxN);
    pts.push({x:px,y:Math.max(pad.t+r,y(bars[i].h)-r-5),r:r,e:e,bar:bars[i],i:i});
  });
  pts.forEach(function(p){
    g.beginPath();g.arc(p.x,p.y,p.r,0,Math.PI*2);
    g.fillStyle=classColor(p.e.klass);g.fill();
    g.lineWidth=2;g.strokeStyle=COL.bg;g.stroke();   /* 2px surface ring */
    if(STATE.hl&&STATE.hl===p.e.day){g.lineWidth=2;g.strokeStyle='#e6edf3';
      g.beginPath();g.arc(p.x,p.y,p.r+3.5,0,Math.PI*2);g.stroke();}
  });

  SCALE={x:x,y:y,iAt:iAt,pAt:pAt,pad:pad,w:w,h:h,lo:lo,hi:hi,i0:i0,i1:i1,pts:pts,plotW:plotW};

  /* crosshair drawn last so it sits above everything */
  if(HOVER) drawCrosshair(g,COL);
}

function drawCrosshair(g,COL){
  var s=SCALE; if(!s)return;
  var mx=HOVER.x,my=HOVER.y;
  if(mx<s.pad.l||mx>s.w-s.pad.r||my<s.pad.t||my>s.h-s.pad.b)return;
  g.save();
  g.strokeStyle='rgba(230,237,243,.25)';g.lineWidth=1;g.setLineDash([3,4]);
  g.beginPath();g.moveTo(Math.round(mx)+.5,s.pad.t);g.lineTo(Math.round(mx)+.5,s.h-s.pad.b);g.stroke();
  g.beginPath();g.moveTo(s.pad.l,Math.round(my)+.5);g.lineTo(s.w-s.pad.r,Math.round(my)+.5);g.stroke();
  g.setLineDash([]);
  /* price readout in the axis gutter */
  var price=s.pAt(my),txt=price.toFixed(2);
  g.font='700 10px system-ui';g.textBaseline='middle';g.textAlign='left';
  var tw=g.measureText(txt).width+8;
  g.fillStyle='#e6edf3';g.fillRect(s.w-s.pad.r+2,my-8,tw,16);
  g.fillStyle='#0d1117';g.fillText(txt,s.w-s.pad.r+6,my+1);
  g.restore();
}

/* ---------- hover: tooltip over markers and candles ---------- */
function nearest(mx,my){
  if(!SCALE)return null;
  var best=null,bd=Infinity;
  SCALE.pts.forEach(function(p){
    var d=Math.hypot(p.x-mx,p.y-my);
    if(d<Math.max(p.r+9,15)&&d<bd){bd=d;best=p;}   /* hit target > mark */
  });
  if(best)return {kind:'news',p:best};
  var i=Math.round(SCALE.iAt(mx));
  if(i<0||i>=STATE.bars.length)return null;
  return {kind:'bar',i:i,b:STATE.bars[i]};
}
function showTip(ev){
  var c=$('chart'),rect=c.getBoundingClientRect();
  var mx=ev.clientX-rect.left,my=ev.clientY-rect.top;
  HOVER={x:mx,y:my};
  /* cursor tells you which zoom axis you're on */
  c.style.cursor=DRAG?'grabbing':(SCALE&&mx>SCALE.w-SCALE.pad.r?'ns-resize':'crosshair');
  var hit=nearest(mx,my),tip=$('tip');
  draw();
  if(!hit||!SCALE){tip.style.display='none';return;}
  var html,anchorX;
  if(hit.kind==='news'){
    var e=hit.p.e,b=hit.p.bar;
    var bd=Object.keys(e.breakdown||{}).map(function(k){
      return '<span style="color:'+classColor(k)+'">&#9679;</span> '+esc(k)+' '+e.breakdown[k];
    }).join(' &middot; ');
    html='<div class=th>'+esc(e.day)+' &middot; C '+b.c.toFixed(2)
      +' <span class=muted>('+((b.c-b.o)/(b.o||1)*100).toFixed(2)+'% on the day)</span></div>'
      +'<div class=muted>'+e.count+' matching record'+(e.count===1?'':'s')+'</div>'
      +(bd?'<div style="margin-top:3px">'+bd+'</div>':'')+'<ul>'
      +e.headlines.map(function(x){return '<li>'+esc((x.headline||'').slice(0,110))
        +' <span class=muted>('+esc(x.source||'')+')</span></li>';}).join('')
      +(e.count>e.headlines.length?'<li class=muted>+'+(e.count-e.headlines.length)+' more…</li>':'')
      +'</ul>';
    anchorX=hit.p.x;
  }else{
    var bb=hit.b,chg=((bb.c-bb.o)/(bb.o||1)*100);
    html='<div class=th>'+esc(bb.t.slice(0,10))+'</div><div class=muted>'
      +'O '+bb.o.toFixed(2)+' &middot; H '+bb.h.toFixed(2)+' &middot; L '+bb.l.toFixed(2)
      +' &middot; <b style="color:var(--text)">C '+bb.c.toFixed(2)+'</b>'
      +' &middot; '+chg.toFixed(2)+'%</div>';
    anchorX=SCALE.x(hit.i);
  }
  tip.innerHTML=html;tip.style.display='block';
  var tw=tip.offsetWidth,left=anchorX+14;
  if(left+tw>SCALE.w) left=anchorX-tw-14;
  tip.style.left=Math.max(4,left)+'px';
  tip.style.top=Math.max(4,Math.min(my+12,SCALE.h-tip.offsetHeight-4))+'px';
}
function applyHighlight(){
  document.querySelectorAll('#tbl tr[data-day]').forEach(function(tr){
    tr.classList.toggle('hl',!!STATE.hl&&tr.getAttribute('data-day')===STATE.hl);
  });
}

/* ---------- filters ---------- */
async function loadFacets(){
  var t=($('f_ticker').value||'').trim().toUpperCase();
  var r=await (await fetch('/api/facets?ticker='+encodeURIComponent(t))).json();
  if(r.building){setTimeout(loadFacets,4000);return;}
  var f=r.facets||{};
  [['f_family','catalyst_family','any family'],['f_source','source','any source'],
   ['f_relation','relation_type','any relation'],['f_origin','origin','any origin']]
  .forEach(function(spec){
    var el=$(spec[0]),cur=el.value;
    el.innerHTML='<option value="">'+spec[2]+'</option>'
      +(f[spec[1]]||[]).map(function(v){return '<option>'+esc(v)+'</option>';}).join('');
    if(cur&&(f[spec[1]]||[]).indexOf(cur)>=0) el.value=cur;
  });
}
async function loadTickers(){
  var t=($('f_ticker').value||'').trim().toUpperCase();
  var r=await (await fetch('/api/tickers?q='+encodeURIComponent(t))).json();
  if(r.building)return;
  $('tickerlist').innerHTML=(r.tickers||[]).map(function(v){
    return '<option value="'+esc(v)+'">';}).join('');
}
function run(){STATE.offset=0;STATE.hl=null;loadRecords();loadChart();}

$('go').addEventListener('click',run);
$('clear').addEventListener('click',function(){
  ['f_ticker','f_q','f_start','f_end'].forEach(function(i){$(i).value='';});
  ['f_family','f_source','f_relation','f_origin'].forEach(function(i){$(i).value='';});
  run();loadFacets();});
['f_family','f_source','f_relation','f_origin','f_days'].forEach(function(i){
  $(i).addEventListener('change',run);});
['f_q','f_start','f_end'].forEach(function(i){
  $(i).addEventListener('keydown',function(e){if(e.key==='Enter')run();});});
$('f_ticker').addEventListener('change',function(){loadFacets();loadTickers();run();});
$('f_ticker').addEventListener('keydown',function(e){if(e.key==='Enter'){loadFacets();run();}});
$('f_ticker').addEventListener('input',loadTickers);
$('prev').addEventListener('click',function(){
  STATE.offset=Math.max(0,STATE.offset-STATE.limit);loadRecords();});
$('next').addEventListener('click',function(){STATE.offset+=STATE.limit;loadRecords();});

/* ---------- chart interaction ----------
   wheel over the plot      → zoom TIME, anchored on the bar under the cursor
   Shift + wheel            → pan time
   wheel over the price axis→ zoom PRICE, anchored on the price under the cursor
   drag                     → pan (time always; price too once price is pinned)
   double-click / Reset     → back to the full range with auto price          */
var canvas=$('chart');

canvas.addEventListener('wheel',function(e){
  if(!SCALE||!STATE.bars.length)return;
  e.preventDefault();
  var rect=canvas.getBoundingClientRect(), mx=e.clientX-rect.left, my=e.clientY-rect.top;
  var d=e.deltaY||e.deltaX||0; if(!d)return;
  var f=Math.exp(d*0.0015);

  if(mx>SCALE.w-SCALE.pad.r){                       /* price axis → vertical zoom */
    var lo=SCALE.lo,hi=SCALE.hi,anchor=SCALE.pAt(Math.max(SCALE.pad.t,Math.min(my,SCALE.h-SCALE.pad.b)));
    VIEW.yAuto=false;
    VIEW.yLo=anchor-(anchor-lo)*f;
    VIEW.yHi=anchor+(hi-anchor)*f;
    if(!(VIEW.yHi>VIEW.yLo)){VIEW.yHi=VIEW.yLo+1e-6;}
    draw();return;
  }
  var n=STATE.bars.length, span=SCALE.i1-SCALE.i0;
  if(e.shiftKey){                                   /* shift → horizontal pan */
    var shift=(d/300)*span;
    VIEW.i0=SCALE.i0+shift; VIEW.i1=SCALE.i1+shift;
    clampView();draw();return;
  }
  /* horizontal zoom centred on the cursor's bar */
  var a=SCALE.iAt(mx), newSpan=Math.max(MIN_BARS-1,Math.min(n-1,span*f));
  var frac=span>0?(a-SCALE.i0)/span:0.5;
  VIEW.i0=a-frac*newSpan; VIEW.i1=VIEW.i0+newSpan;
  clampView();draw();
},{passive:false});

canvas.addEventListener('mousedown',function(e){
  if(!SCALE)return;
  var rect=canvas.getBoundingClientRect();
  DRAG={x:e.clientX,y:e.clientY,i0:SCALE.i0,i1:SCALE.i1,lo:SCALE.lo,hi:SCALE.hi,
        moved:false,px:e.clientX-rect.left,py:e.clientY-rect.top};
  canvas.style.cursor='grabbing';
});
window.addEventListener('mousemove',function(e){
  if(!DRAG||!SCALE)return;
  var dx=e.clientX-DRAG.x, dy=e.clientY-DRAG.y;
  if(Math.abs(dx)>3||Math.abs(dy)>3) DRAG.moved=true;
  var span=DRAG.i1-DRAG.i0, di=-(dx/SCALE.plotW)*span;
  VIEW.i0=DRAG.i0+di; VIEW.i1=DRAG.i1+di;
  if(!VIEW.yAuto){                                  /* price pinned → drag it too */
    var pp=(DRAG.hi-DRAG.lo)/(SCALE.h-SCALE.pad.t-SCALE.pad.b), dp=dy*pp;
    VIEW.yLo=DRAG.lo+dp; VIEW.yHi=DRAG.hi+dp;
  }
  clampView();draw();
});
window.addEventListener('mouseup',function(){
  if(DRAG){canvas.style.cursor='crosshair';}
  DRAG=null;
});
canvas.addEventListener('mousemove',showTip);
canvas.addEventListener('mouseleave',function(){
  HOVER=null;$('tip').style.display='none';draw();});
canvas.addEventListener('click',function(ev){
  if(DRAG&&DRAG.moved)return;                       /* a pan is not a click */
  var rect=canvas.getBoundingClientRect();
  var hit=nearest(ev.clientX-rect.left,ev.clientY-rect.top);
  if(hit&&hit.kind==='news'){
    STATE.hl=(STATE.hl===hit.p.e.day)?null:hit.p.e.day;
    applyHighlight();draw();
    var row=document.querySelector('#tbl tr.hl');
    if(row) row.scrollIntoView({block:'center',behavior:'smooth'});
  }});
canvas.addEventListener('dblclick',function(e){e.preventDefault();resetView();});
$('resetView').addEventListener('click',resetView);
$('f_colorby').addEventListener('change',loadChart);
window.addEventListener('resize',draw);

async function status(){
  var s=await (await fetch('/api/state')).json();
  var i=s.index||{};
  $('idx').textContent=i.building?'building index…'
    :(i.rows?(i.rows.toLocaleString()+' records · '+i.tickers.toLocaleString()
      +' tickers · indexed '+fmtTs(i.built_at)):'index not built');
  var badge=$('cyno-live-badge');
  if(badge){badge.textContent='library';badge.className='live-badge';}
}
$('f_ticker').value='__DEFAULT_TICKER__';
status();setInterval(status,30000);
loadFacets();loadTickers();run();
</script></body></html>"""


def _render_page(default_ticker: str) -> str:
    return (_PAGE
            .replace("__THEME_LINK__", THEME_LINK)
            .replace("__NAV_HTML__", NAV_HTML)
            .replace("__DEFAULT_TICKER__", default_ticker))


class LibraryHTTPServer(ThreadingHTTPServer):
    app: LibraryDashboardApp


class LibraryHandler(BaseHTTPRequestHandler):
    server_version = "CynolycusLibrary/1.0"

    def log_message(self, fmt, *a):  # noqa: A002
        return

    def _app(self) -> LibraryDashboardApp:
        return self.server.app  # type: ignore[attr-defined]

    def _send(self, body: bytes, status=HTTPStatus.OK, ctype="application/json; charset=utf-8"):
        try:
            self.send_response(int(status))
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _json(self, payload: Any) -> None:
        self._send(json.dumps(_json_safe(payload)).encode("utf-8"))

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        path, q = parsed.path, parse_qs(parsed.query)
        app = self._app()
        if path == "/" or path.startswith("/index"):
            self._send(_render_page(app.default_ticker).encode("utf-8"),
                       ctype="text/html; charset=utf-8")
        elif path == "/static/cynolycus_theme.css":
            serve_theme_css(self)
        elif path == "/api/state":
            self._json(app.state())
        elif path == "/api/search":
            self._json(app.search(q))
        elif path == "/api/facets":
            self._json(app.facets(q))
        elif path == "/api/tickers":
            self._json(app.tickers(q))
        elif path == "/api/price":
            self._json(app.price(q))
        else:
            self._send(b'{"error":"not_found"}', status=HTTPStatus.NOT_FOUND)


def make_server(host: str, port: int, app: LibraryDashboardApp) -> LibraryHTTPServer:
    srv = LibraryHTTPServer((host, port), LibraryHandler)
    srv.daemon_threads = True
    srv.app = app
    return srv


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the Library dashboard standalone.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8775)
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    app = LibraryDashboardApp(default_ticker=args.ticker)
    srv = make_server(args.host, args.port, app)
    print(f"Library dashboard: http://{args.host}:{args.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
