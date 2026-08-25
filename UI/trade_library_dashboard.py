"""Trade library dashboard — every completed position, drawn on its own price chart.

Companion to the news/catalyst library (``UI/library_dashboard.py``): same
read-only, no-broker contract, but the subject is the book's own trades. Pick a
position and its entry, trims, and exit are marked on the underlying's 4H
candles, with the underlying stop level drawn where the position had one — so a
stop that fired while the underlying was flat is visible rather than inferred.

Trims are drawn as hollow markers because they are partial: a
``take_profit_+30%`` leg books a gain and leaves the rest of the position open.
Reading them as closed trades is what made the ledger look like a 46% win rate
when the fully-closed round-trip rate was 16%.

Endpoints:
  GET  /                      → HTML page
  GET  /api/state             → hub status payload
  GET  /api/positions?…       → positions (module/ticker/status/route/outcome/reason/date filters)
  GET  /api/position?id=…     → one position + its underlying bars + markers
  GET  /api/facets            → distinct filter values
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from UI import trade_library
from UI.ui_chrome import NAV_HTML, THEME_LINK, serve_theme_css

logger = logging.getLogger(__name__)
DEFAULT_PORT = 8776
CHART_PAD_BARS = 12          # context drawn either side of the position's own window


class TradeLibraryApp:
    """Serves the reconstructed position book, never stale.

    Every endpoint asks for the current fold; ``trade_library`` reuses the last
    one only while the ledger and live-state files are byte-for-byte unchanged,
    so a runner closing a position is reflected on the very next request. The
    "re-read everything per request, it is only a few milliseconds" version of
    this class was true standalone and false in the combined_server process,
    where the same call took 1.2-2.1s against the hub's 2.5s poll timeout.
    """

    def positions(self, q: dict[str, list[str]]) -> dict[str, Any]:
        rows = trade_library.build_positions()
        full = trade_library.summary(rows)

        def one(k: str) -> str:
            return (q.get(k) or [""])[0].strip()

        module, ticker, status = one("module"), one("ticker").upper(), one("status")
        route, outcome, reason = one("route"), one("outcome"), one("reason")
        start, end = one("start"), one("end")
        if module:
            rows = [r for r in rows if r["module"] == module]
        if ticker:
            rows = [r for r in rows if ticker in r["ticker"]]
        if status:
            rows = [r for r in rows if r["status"] == status]
        if route:
            rows = [r for r in rows if r["route"] == route]
        if reason:
            rows = [r for r in rows if reason in (r["final_reason"] or "")
                    or any(reason in x for x in r["reasons"])]
        if outcome == "win":
            rows = [r for r in rows if r["realized"] > 0]
        elif outcome == "loss":
            rows = [r for r in rows if r["realized"] < 0]
        if start:
            rows = [r for r in rows if (r["last_sell"] or "") >= start]
        if end:
            rows = [r for r in rows if (r["last_sell"] or "") <= end + "T23:59:59+00:00"]
        return {"positions": rows, "summary": trade_library.summary(rows),
                "book": full, "health": trade_library.ledger_health()}

    def state(self) -> dict[str, Any]:
        """Hub status payload. Read-only, so this is never anything but ready."""
        rows = trade_library.build_positions()
        return {"book": trade_library.summary(rows),
                "health": trade_library.ledger_health(),
                "generated_at": datetime.now(timezone.utc).isoformat()}

    def facets(self) -> dict[str, Any]:
        rows = trade_library.build_positions()
        reasons = sorted({r for p in rows for r in p["reasons"] if r})
        return {"modules": sorted({p["module"] for p in rows}),
                "routes": sorted({p["route"] for p in rows}),
                "reasons": reasons,
                "tickers": sorted({p["ticker"] for p in rows})}

    def position(self, q: dict[str, list[str]]) -> dict[str, Any]:
        pid = (q.get("id") or [""])[0]
        tf = (q.get("tf") or ["4h"])[0]
        rows = trade_library.build_positions()
        pos = next((p for p in rows if p["id"] == pid), None)
        if pos is None:
            return {"error": "no such position", "id": pid}
        series = trade_library.price_series(pos["ticker"], timeframe=tf, days=365)
        bars = series.get("bars", [])
        # Window the chart to the position's own life plus context on both sides,
        # so the reader sees what happened AFTER the exit — that is the whole
        # point of reviewing a stop that fired on noise.
        stamps = [s for s in [pos["entry_bar"], pos["first_sell"], pos["last_sell"]] if s]
        if bars and stamps:
            lo, hi = min(stamps), max(stamps)
            i0 = max(0, _idx(bars, lo) - CHART_PAD_BARS)
            i1 = min(len(bars), _idx(bars, hi) + CHART_PAD_BARS + 1)
            bars = bars[i0:i1]
        markers = []
        if pos["entry_bar"]:
            markers.append({"t": pos["entry_bar"], "kind": "entry",
                            "label": f"entry {pos['route']}", "price": None})
        for leg in pos["legs"]:
            markers.append({"t": leg["ts"], "kind": "trim" if leg["is_trim"] else "exit",
                            "label": leg["reason"], "price": None,
                            "pnl": leg["pnl"], "qty": leg["qty"]})
        stop = None
        if pos.get("u_entry") and pos.get("u_atr"):
            # Only positions the runner anchored carry a basis; older ones never
            # had one and must not be drawn with an invented line.
            stop = {"level": round(pos["u_entry"] - 1.5 * pos["u_atr"], 4),
                    "u_entry": pos["u_entry"], "u_atr": pos["u_atr"]}
        return {"position": pos, "bars": bars, "markers": markers,
                "underlying_stop": stop, "error": series.get("error"),
                "timeframe": tf}


def _idx(bars: list[dict], stamp: str) -> int:
    lo, hi = 0, len(bars)
    while lo < hi:
        mid = (lo + hi) // 2
        if bars[mid]["t"] < stamp:
            lo = mid + 1
        else:
            hi = mid
    return lo


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>Trade Library — Cynolycus</title>__THEME__
<style>
body{margin:0;font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}
.wrap{padding:12px;max-width:1600px;margin:0 auto}
.card{background:var(--panel);border:1px solid #2b3440;border-radius:6px;margin:10px 0}
.card-head{padding:7px 10px;border-bottom:1px solid #2b3440;font-size:11px;
  text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
  display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.card-body{padding:10px}
.filters{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
input,select{background:var(--panel2);color:var(--text);border:1px solid #2b3440;
  border-radius:4px;padding:4px 7px;font:inherit;font-size:12px}
button{background:var(--panel2);color:var(--text);border:1px solid #2b3440;
  border-radius:4px;padding:4px 10px;font:inherit;font-size:12px;cursor:pointer}
button:hover{border-color:#4b5563}
.stats{display:flex;gap:18px;flex-wrap:wrap;font-size:12px}
.stat b{display:block;font-size:16px;font-weight:600}
.muted{color:var(--muted)}
.good{color:var(--green)} .bad{color:#f85149}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{padding:4px 7px;text-align:left;border-bottom:1px solid #222b36;white-space:nowrap}
th{color:var(--muted);font-weight:500;font-size:11px;text-transform:uppercase;
  letter-spacing:.05em;position:sticky;top:0;background:var(--panel)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tbody tr{cursor:pointer}
tbody tr:hover{background:#1d2b3a}
tbody tr.sel{background:#23344a}
.scroll{max-height:460px;overflow:auto}
.pill{display:inline-block;padding:1px 6px;border-radius:9px;font-size:10px;
  border:1px solid #35414f;color:var(--muted)}
.pill.open{border-color:#8957e5;color:#a371f7}
#chart{width:100%;height:420px;display:block;background:var(--bg)}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:var(--muted);padding:6px 10px}
.legend i{display:inline-block;width:9px;height:9px;margin-right:4px;vertical-align:middle}
.warn{background:#3d2a12;border:1px solid #6b4a1c;color:#e3b341;
  padding:6px 10px;border-radius:5px;font-size:11px;margin:8px 0}
.legs{font-size:11px;margin-top:8px}
.legs td,.legs th{padding:3px 7px}
</style></head><body>
__NAV__
<div class=wrap>
  <div class=card>
    <div class=card-head>Trade Library — completed positions
      <span class=spacer style=flex:1></span>
      <span class=muted id=gen></span></div>
    <div class=card-body>
      <div class=filters>
        <input id=f_ticker placeholder=ticker size=8>
        <select id=f_module><option value="">all modules</option></select>
        <select id=f_status><option value="">open + closed</option>
          <option value=closed>closed only</option><option value=open>open (trimmed)</option></select>
        <select id=f_route><option value="">all routes</option>
          <option value=option>option</option><option value=equity>equity</option></select>
        <select id=f_outcome><option value="">win + loss</option>
          <option value=win>winners</option><option value=loss>losers</option></select>
        <select id=f_reason><option value="">all exit reasons</option></select>
        <input id=f_start type=date title="last activity on/after">
        <input id=f_end type=date title="last activity on/before">
        <button onclick=reset()>reset</button>
      </div>
      <div id=health></div>
      <div class=stats id=stats style=margin-top:10px></div>
    </div>
  </div>

  <div class=card>
    <div class=card-head><span id=chartTitle>Select a position</span>
      <span style=flex:1></span>
      <select id=f_tf onchange=loadPos(SEL)><option value=4h>4H bars</option>
        <option value=1d>daily bars</option></select></div>
    <canvas id=chart></canvas>
    <div class=legend>
      <span><i style="background:#58a6ff"></i>entry</span>
      <span><i style="border:1px solid #3fb950;background:transparent"></i>trim (partial — position stays open)</span>
      <span><i style="background:#f85149"></i>full exit</span>
      <span><i style="background:#d29922"></i>underlying stop level (entry − 1.5×ATR)</span>
    </div>
    <div class=card-body id=legs></div>
  </div>

  <div class=card>
    <div class=card-head>positions <span id=count class=muted></span></div>
    <div class="card-body scroll" style=padding:0>
      <table><thead><tr>
        <th>module</th><th>ticker</th><th>route</th><th></th><th>entry</th><th>last sell</th>
        <th class=num>hold d</th><th class=num>entry px</th><th class=num>exit/mark</th>
        <th class=num>realized $</th><th class=num>open %</th><th>exit reason</th>
      </tr></thead><tbody id=rows></tbody></table>
    </div>
  </div>
</div>
<script>
var $=function(i){return document.getElementById(i)};
var POS=[],SEL=null,CUR=null;
function money(v){if(v==null)return '';var s=(v<0?'-':'')+'$'+Math.abs(v).toLocaleString(undefined,{maximumFractionDigits:0});
  return '<span class="'+(v>0?'good':(v<0?'bad':''))+'">'+s+'</span>'}
function d(s){return s?s.slice(0,10):''}
function num(v,p){return v==null?'':Number(v).toFixed(p==null?2:p)}

async function load(){
  var p=new URLSearchParams();
  [['ticker','f_ticker'],['module','f_module'],['status','f_status'],['route','f_route'],
   ['outcome','f_outcome'],['reason','f_reason'],['start','f_start'],['end','f_end']]
   .forEach(function(x){var v=$(x[1]).value.trim();if(v)p.set(x[0],v)});
  var r=await (await fetch('/api/positions?'+p.toString())).json();
  POS=r.positions||[];
  var s=r.summary||{},b=r.book||{};
  $('gen').textContent='book: '+b.closed+' closed / '+b.open+' open';
  $('stats').innerHTML=
     '<div class=stat><span class=muted>positions</span><b>'+s.positions+'</b></div>'
    +'<div class=stat><span class=muted>closed</span><b>'+s.closed+'</b></div>'
    +'<div class=stat><span class=muted>round-trip win rate</span><b>'+(s.win_rate==null?'—':s.win_rate+'%')+'</b></div>'
    +'<div class=stat><span class=muted>realized (closed)</span><b>'+money(s.realized_closed)+'</b></div>'
    +'<div class=stat><span class=muted>booked on still-open trims</span><b>'+money(s.realized_on_open_trims)+'</b></div>';
  var h=(r.health||{});
  var w=[];
  (h.gaps||[]).forEach(function(g){w.push('ledger gap in '+g.module+': '+(g.detail||g.event))});
  if(h.rows_missing_pnl)w.push(h.rows_missing_pnl+' sell leg(s) have no realized P&L recorded and are excluded from totals.');
  $('health').innerHTML=w.length?('<div class=warn>'+w.join('<br>')+'</div>'):'';
  render();
}
function render(){
  $('count').textContent='('+POS.length+')';
  $('rows').innerHTML=POS.map(function(p,i){
    var openPill=p.status==='open'?' <span class="pill open">open</span>':'';
    return '<tr data-i='+i+(SEL===p.id?' class=sel':'')+'>'
      +'<td class=muted>'+p.module.replace(/_/g,' ')+'</td>'
      +'<td><b>'+p.ticker+'</b></td><td class=muted>'+p.route+'</td><td>'+openPill+'</td>'
      +'<td>'+d(p.entry_bar)+'</td><td>'+d(p.last_sell)+'</td>'
      +'<td class=num>'+num(p.hold_days,1)+'</td>'
      +'<td class=num>'+num(p.entry_price)+'</td><td class=num>'+num(p.exit_price)+'</td>'
      +'<td class=num>'+money(p.realized)+'</td>'
      +'<td class=num>'+(p.open_ret_pct==null?'':num(p.open_ret_pct,1)+'%')+'</td>'
      +'<td class=muted>'+(p.final_reason||p.reasons.join(' + '))+'</td></tr>'}).join('');
  [].forEach.call($('rows').querySelectorAll('tr'),function(tr){
    tr.onclick=function(){SEL=POS[+tr.dataset.i].id;render();loadPos(SEL)}});
}
async function loadPos(id){
  if(!id)return;
  var r=await (await fetch('/api/position?id='+encodeURIComponent(id)+'&tf='+$('f_tf').value)).json();
  CUR=r;
  var p=r.position||{};
  $('chartTitle').textContent=p.ticker+' — '+p.module.replace(/_/g,' ')+' · '+p.route
    +' · '+(p.status==='open'?'still open':'closed '+d(p.last_sell));
  $('legs').innerHTML='<table class=legs><thead><tr><th>when</th><th>action</th><th class=num>qty</th>'
    +'<th class=num>price</th><th class=num>realized</th><th>order id</th></tr></thead><tbody>'
    +(p.legs||[]).map(function(l){return '<tr><td>'+l.ts.slice(0,16).replace('T',' ')+'</td>'
      +'<td>'+(l.is_trim?'TRIM (partial)':'EXIT')+' — '+l.reason+'</td>'
      +'<td class=num>'+num(l.qty,0)+'</td><td class=num>'+num(l.price)+'</td>'
      +'<td class=num>'+money(l.pnl)+'</td><td class=muted>'+(l.order_id||'').slice(0,8)+'</td></tr>'}).join('')
    +'</tbody></table>'
    +(r.error?'<div class=warn>'+r.error+'</div>':'');
  draw();
}
function draw(){
  var c=$('chart'),box=c.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  c.width=box.width*dpr;c.height=box.height*dpr;
  var g=c.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);
  g.clearRect(0,0,box.width,box.height);
  if(!CUR||!(CUR.bars||[]).length){g.fillStyle='#8b949e';g.font='12px monospace';
    g.fillText(CUR&&CUR.error?CUR.error:'no bars for this position',12,24);return}
  var bars=CUR.bars,L=52,R=12,T=12,B=22,W=box.width-L-R,H=box.height-T-B;
  var lo=Infinity,hi=-Infinity;
  bars.forEach(function(b){lo=Math.min(lo,b.l);hi=Math.max(hi,b.h)});
  var stop=CUR.underlying_stop;
  if(stop){lo=Math.min(lo,stop.level);hi=Math.max(hi,stop.level)}
  var pad=(hi-lo)*0.08||1;lo-=pad;hi+=pad;
  var y=function(v){return T+H-(v-lo)/(hi-lo)*H};
  var cw=W/bars.length, x=function(i){return L+i*cw+cw/2};
  g.strokeStyle='#222b36';g.fillStyle='#8b949e';g.font='10px monospace';g.lineWidth=1;
  for(var k=0;k<=4;k++){var v=lo+(hi-lo)*k/4,yy=y(v);
    g.beginPath();g.moveTo(L,yy);g.lineTo(L+W,yy);g.stroke();
    g.fillText(v.toFixed(2),6,yy+3)}
  bars.forEach(function(b,i){
    var up=b.c>=b.o,px=x(i);
    g.strokeStyle=up?'#3fb950':'#f85149';g.fillStyle=g.strokeStyle;
    g.beginPath();g.moveTo(px,y(b.h));g.lineTo(px,y(b.l));g.stroke();
    var bw=Math.max(1,cw*0.6),h=Math.abs(y(b.o)-y(b.c));
    g.fillRect(px-bw/2,Math.min(y(b.o),y(b.c)),bw,Math.max(1,h))});
  if(stop){
    g.strokeStyle='#d29922';g.setLineDash([5,4]);g.beginPath();
    g.moveTo(L,y(stop.level));g.lineTo(L+W,y(stop.level));g.stroke();g.setLineDash([]);
    g.fillStyle='#d29922';g.font='10px monospace';
    g.fillText('underlying stop '+stop.level.toFixed(2)+'  (entry '+stop.u_entry.toFixed(2)
      +' − 1.5×ATR '+stop.u_atr.toFixed(2)+')',L+4,y(stop.level)-4)}
  (CUR.markers||[]).forEach(function(m){
    var i=idxAt(bars,m.t);if(i<0)return;
    var px=x(i),b=bars[i];
    var col=m.kind==='entry'?'#58a6ff':(m.kind==='trim'?'#3fb950':'#f85149');
    var py=m.kind==='entry'?y(b.l)+14:y(b.h)-14;
    g.strokeStyle=col;g.fillStyle=col;g.lineWidth=1.5;
    g.beginPath();g.arc(px,py,5,0,6.284);
    if(m.kind==='trim'){g.stroke()}else{g.fill()}
    g.lineWidth=1;g.beginPath();g.moveTo(px,py);g.lineTo(px,y(b.c));g.stroke();
    g.fillStyle=col;g.font='9px monospace';
    var lbl=m.kind==='trim'?'trim':(m.kind==='entry'?'entry':m.label);
    g.fillText(lbl,px+7,py+3)});
  g.fillStyle='#8b949e';g.font='10px monospace';
  g.fillText(bars[0].t.slice(0,10),L,box.height-6);
  var last=bars[bars.length-1].t.slice(0,10);
  g.fillText(last,L+W-g.measureText(last).width,box.height-6);
}
function idxAt(bars,t){
  if(!t)return -1;
  var lo=0,hi=bars.length-1,best=-1;
  for(var i=0;i<bars.length;i++){if(bars[i].t<=t)best=i;else break}
  return best>=0?best:0;
}
function reset(){['f_ticker','f_module','f_status','f_route','f_outcome','f_reason','f_start','f_end']
  .forEach(function(i){$(i).value=''});load()}
['f_ticker','f_module','f_status','f_route','f_outcome','f_reason','f_start','f_end']
  .forEach(function(i){$(i).addEventListener('change',load)});
$('f_ticker').addEventListener('input',function(){clearTimeout(window._t);window._t=setTimeout(load,250)});
window.addEventListener('resize',draw);
(async function(){
  var f=await (await fetch('/api/facets')).json();
  (f.modules||[]).forEach(function(m){var o=document.createElement('option');
    o.value=m;o.textContent=m.replace(/_/g,' ');$('f_module').appendChild(o)});
  (f.reasons||[]).forEach(function(m){var o=document.createElement('option');
    o.value=m;o.textContent=m;$('f_reason').appendChild(o)});
  await load();
  if(POS.length){SEL=POS[0].id;render();loadPos(SEL)}
})();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    app: TradeLibraryApp = TradeLibraryApp()
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # noqa: A003 - quiet the stdlib access log
        logger.debug("trade-library %s", fmt % args)

    def _send(self, body: bytes, status=HTTPStatus.OK, ctype="application/json"):
        # A client that goes away mid-response is normal HTTP, not an
        # application fault: the hub polls every dashboard's /api/state on a 5s
        # timer with a 2.5s timeout, and every one it drops lands here as a
        # BrokenPipeError. Every other dashboard already swallows it (see the
        # 2026-07-20 audit and UI/tests/test_dashboard_broken_pipe_guard.py);
        # this one was written without the guard and flooded the console.
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _json(self, payload: Any, status=HTTPStatus.OK):
        self._send(json.dumps(payload, default=str).encode(), status)

    def do_GET(self):  # noqa: N802 - stdlib interface
        parsed = urlparse(self.path)
        path, q = parsed.path, parse_qs(parsed.query)
        try:
            if path == "/":
                page = PAGE.replace("__THEME__", THEME_LINK).replace("__NAV__", NAV_HTML)
                self._send(page.encode(), ctype="text/html; charset=utf-8")
            elif path == "/static/cynolycus_theme.css":
                serve_theme_css(self)
            elif path == "/api/state":
                self._json(self.app.state())
            elif path == "/api/positions":
                self._json(self.app.positions(q))
            elif path == "/api/position":
                self._json(self.app.position(q))
            elif path == "/api/facets":
                self._json(self.app.facets())
            else:
                self._json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001 - a dashboard must not take the server down
            logger.exception("trade-library request failed: %s", path)
            self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def make_server(host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                app: TradeLibraryApp | None = None) -> ThreadingHTTPServer:
    Handler.app = app or TradeLibraryApp()
    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Trade library dashboard (read-only).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(f"Trade library: http://{args.host}:{args.port}/")
    make_server(args.host, args.port).serve_forever()


if __name__ == "__main__":
    main()
