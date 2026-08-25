"""Shared UI chrome for the Cynolycus dashboards.

Single source of truth for the look-and-feel that every combined-server
dashboard shares: the palette, base typography, the common component classes
(.pill / button variants / .card / .controls), and the top navigation bar.

Usage:
  * Static HTML pages: add ``THEME_LINK`` in ``<head>`` (after removing the
    page's inline ``:root``) and paste ``<!--CYNO_NAV-->`` at the top of
    ``<body>`` (handlers replace it with ``NAV_HTML``).
  * Python-rendered pages: embed ``THEME_LINK`` / ``NAV_HTML`` in the markup.
  * Every handler answers ``GET /static/cynolycus_theme.css`` via
    ``serve_theme_css(handler)``.
"""
from __future__ import annotations

# Port map shared by the nav + hub fan-out.
NAV_PORTS: list[tuple[str, int]] = [
    ("Hub", 8764),
    ("SPY Intraday", 8765),
    ("Swing", 8766),
    ("HTF Swing", 8771),
    ("Momentum", 8770),
    ("Amethyst", 8772),
    ("Dealer", 8768),
    ("Dealer Ranker", 8773),
    ("Intraday Structure", 8774),
    ("Meta Ranker", 8769),
    ("Library", 8775),
    ("Trades", 8776),
]

THEME_CSS = """:root{
  color-scheme: dark;
  --bg:#0d1117; --panel:#161b22; --panel2:#1f2630; --panel-2:#1f2630;
  --text:#e6edf3; --muted:#8b949e;
  --green:#3fb950; --good:#3fb950;
  --red:#f85149;   --bad:#f85149;
  --yellow:#d29922; --warn:#d29922; --gold:#d29922;
  --blue:#58a6ff;  --accent:#58a6ff;
  --border:#30363d; --line:#30363d;
  --radius:6px;
}

*{box-sizing:border-box}
body{
  margin:0; background:var(--bg); color:var(--text);
  font-family:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-size:13px; line-height:1.45;
}
a{color:var(--blue);text-decoration:none}
a:hover{text-decoration:underline}

/* shared top navigation */
.cyno-nav{
  display:flex; align-items:center; gap:4px; flex-wrap:wrap;
  padding:6px 14px; background:#0a0e14; border-bottom:1px solid var(--border);
  font-size:12px; position:sticky; top:0; z-index:50;
}
.cyno-nav .brand{font-weight:700;letter-spacing:.4px;margin-right:10px;color:var(--text)}
.cyno-nav a{
  color:var(--muted); padding:5px 10px; border-radius:6px;
  border:1px solid transparent;
}
.cyno-nav a:hover{color:var(--text); background:var(--panel); text-decoration:none}
.cyno-nav a.active{color:var(--text); background:var(--panel2); border-color:var(--border)}
.cyno-nav .spacer{flex:1}
.cyno-nav .live-badge{
  padding:3px 9px; border-radius:10px; font-weight:700; font-size:11px;
  text-transform:uppercase; letter-spacing:.4px;
  background:#0f3a1f; color:var(--green); border:1px solid #1f5a33;
}
.cyno-nav .live-badge.live{background:#3a1010; color:var(--red); border-color:#5a1f1f}
.cyno-nav .performance-badge{color:var(--muted);font-variant-numeric:tabular-nums;font-size:11px}

/* shared components (already the swing/dealer vocabulary) */
.pill{padding:2px 8px;border-radius:10px;background:#30363d;color:var(--muted);
  font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.3px}
.pill.idle,.pill.stopped{background:#2d1010;color:var(--red)}
.pill.warming,.pill.stopping{background:#3a2d10;color:var(--yellow)}
.pill.running,.pill.ready{background:#0f3a1f;color:var(--green)}
.pill.error{background:#2d1010;color:var(--red)}
.pill.paper{background:#1f3a4a;color:var(--blue)}
.pill.live{background:#3a1010;color:var(--red)}

button,select,input[type="number"],input[type="text"]{
  background:var(--panel2);color:var(--text);border:1px solid var(--border);
  border-radius:5px;padding:6px 10px;font:inherit}
button:hover{background:#2c3340;cursor:pointer}
button.primary{background:#1f6feb;border-color:#1f6feb}
button.primary:hover{filter:brightness(1.1)}
button.danger{background:#6e1f1f;border-color:#6e1f1f}
button.danger:hover{background:#8a2929}
button:disabled{opacity:.45;cursor:not-allowed}

.controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.card{background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden}
.card-head{padding:8px 12px;background:var(--panel2);border-bottom:1px solid var(--border);
  color:var(--muted);font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.4px}
.muted{color:var(--muted)}
.pos,.good{color:var(--green)} .neg,.bad{color:var(--red)}
"""

THEME_LINK = '<link rel="stylesheet" href="/static/cynolycus_theme.css">'

# JS-driven nav. Builds the nav links from a fixed port map. Works
# identically in static + server-rendered pages.
_PORTS_JS = ",".join(f'["{n}",{p}]' for n, p in NAV_PORTS)
NAV_HTML = """<nav class="cyno-nav" id="cyno-nav"></nav>
<script>
(function(){
  var PORTS=[__PORTS__];
  var here=location.port||"80", host=location.hostname;
  var html='<span class="brand">CYNOLYCUS</span>';
  PORTS.forEach(function(p){
    var active=(String(p[1])===String(here))?' class="active"':'';
    html+='<a'+active+' href="http://'+host+':'+p[1]+'/">'+p[0]+'</a>';
  });
  html+='<span class="spacer"></span>'
      +'<span class="performance-badge" id="cyno-performance">performance: loading</span>'
      +'<span class="live-badge" id="cyno-live-badge">paper</span>';
  var el=document.getElementById('cyno-nav');
  if(el){el.innerHTML=html;}
  // Every dashboard already exposes /api/state. Use only its module-scoped
  // account book here; never show a shared account's P/L as this module's P/L.
  function setPerformance(s){
    var out=(s&&s.performance)||{}, a=(s&&s.account)||{};
    var upl=out.open_unrealized_pnl;
    if(upl==null && Array.isArray(a.positions)) upl=a.positions.reduce(function(n,p){return n+(Number(p.upl)||Number(p.unrealized_pl)||0);},0);
    var el=document.getElementById('cyno-performance'); if(!el) return;
    if(out.closed_trades){ el.textContent='tracked: $'+Number(out.tracked_pnl||0).toLocaleString(undefined,{maximumFractionDigits:0})+' · '+out.closed_trades+' closed'; }
    else if(upl!=null){ el.textContent='open P/L: $'+Number(upl).toLocaleString(undefined,{maximumFractionDigits:0})+' · exits not yet ledgered'; }
    else { el.textContent='performance: no module ledger'; }
  }
  function refreshPerformance(){fetch('/api/state',{cache:'no-store'}).then(function(r){return r.ok?r.json():null;}).then(setPerformance).catch(function(){});}
  refreshPerformance(); setInterval(refreshPerformance,10000);
})();
</script>""".replace("__PORTS__", _PORTS_JS)


def serve_theme_css(handler) -> None:
    """Answer ``GET /static/cynolycus_theme.css`` from any BaseHTTPRequestHandler.

    Swallows a mid-write client disconnect for the same reason every
    dashboard's own ``_send`` does: a closed tab is not a server fault, and an
    escaping traceback here would be attributed to whichever dashboard happened
    to serve the stylesheet.
    """
    body = THEME_CSS.encode("utf-8")
    try:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/css; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        handler.wfile.write(body)
    except (BrokenPipeError, ConnectionResetError):
        handler.close_connection = True
