"""Shared UI chrome for the Cynolycus dashboards.

Single source of truth for the look-and-feel that every combined-server
dashboard shares: the palette(s), base typography, the common component
classes (.pill / button variants / .card / .controls), the top navigation
bar, and the swappable theme system.

Themes are pure CSS-variable sets keyed off ``:root[data-theme="…"]``. The
selected theme name is stored in ``localStorage['cyno-theme']`` and applied on
every page by the nav bootstrap script, so the choice follows you across all
dashboards. The "anime" theme additionally pulls imagery the user drops into
``UI/static/themes/anime/`` (``bg.jpg`` page background, ``corner.png`` corner
art) — the app never sources or ships those images, it just displays whatever
files are present.

Usage:
  * Static HTML pages: add ``THEME_LINK`` in ``<head>`` (after removing the
    page's inline ``:root``) and paste ``<!--CYNO_NAV-->`` at the top of
    ``<body>`` (handlers replace it with ``NAV_HTML``).
  * Python-rendered pages: embed ``THEME_LINK`` / ``NAV_HTML`` in the markup.
  * Every handler answers ``GET /static/cynolycus_theme.css`` via
    ``serve_theme_css(handler)`` and ``GET /static/themes/<t>/<f>`` via
    ``serve_theme_asset(handler, path)``.
"""
from __future__ import annotations

import mimetypes
from pathlib import Path

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
    ("Meta Ranker", 8769),
]

# Selectable themes: (key, display name). "default" is the base palette.
THEMES: list[tuple[str, str]] = [
    ("default", "Cynolycus Dark"),
    ("anime", "Anime"),
]

_THEMES_DIR = Path(__file__).resolve().parent / "static" / "themes"

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

/* ---- Anime theme: dark sakura palette + user-supplied imagery ---- */
:root[data-theme="anime"]{
  --bg:#14101a; --panel:#211a2e; --panel2:#2e2440; --panel-2:#2e2440;
  --text:#fbe9f5; --muted:#c4a7d4;
  --green:#6ee7a8; --good:#6ee7a8;
  --red:#ff7aa8;   --bad:#ff7aa8;
  --yellow:#ffcf6e; --warn:#ffcf6e; --gold:#ffcf6e;
  --blue:#ff8fc8;  --accent:#ff8fc8;
  --border:#4a3a63; --line:#4a3a63;
  --radius:14px;
}
/* background image is set at runtime from whatever the user dropped in
   UI/static/themes/anime/ (see the nav bootstrap script) */
[data-theme="anime"] body{
  background-size:cover; background-position:center; background-attachment:fixed;
}
[data-theme="anime"] .card,[data-theme="anime"] .cyno-nav{
  backdrop-filter:blur(3px);
  background:color-mix(in srgb, var(--panel) 82%, transparent);
}
[data-theme="anime"] button.primary{background:#d6478f;border-color:#d6478f}
/* corner art (shown only when the user dropped UI/static/themes/anime/corner.png) */
#cyno-pinup{display:none;position:fixed;right:8px;bottom:8px;max-height:38vh;
  max-width:30vw;z-index:5;pointer-events:none;opacity:.92;
  filter:drop-shadow(0 6px 18px rgba(0,0,0,.5));border-radius:12px}
[data-theme="anime"] #cyno-pinup[src]{display:block}

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

# JS-driven nav + theme bootstrap. Applies the stored theme as early as
# possible, builds the nav links from a fixed port map, and (anime theme) wires
# the optional corner image. Works identically in static + server-rendered pages.
_PORTS_JS = ",".join(f'["{n}",{p}]' for n, p in NAV_PORTS)
NAV_HTML = """<img id="cyno-pinup" alt="">
<nav class="cyno-nav" id="cyno-nav"></nav>
<script>
(function(){
  // apply stored theme everywhere; anime pulls whatever images the user
  // dropped into UI/static/themes/anime/ (first = background, last = corner art)
  function applyTheme(t){
    t=t||'default';
    document.documentElement.dataset.theme=t;
    var img=document.getElementById('cyno-pinup');
    if(t!=='anime'){
      if(document.body) document.body.style.backgroundImage='';
      if(img) img.removeAttribute('src');
      return;
    }
    fetch('/static/themes/anime/__list__').then(function(r){return r.json();}).then(function(files){
      files=(files||[]).filter(function(f){return /\\.(jpe?g|png|webp|gif|avif|bmp)$/i.test(f);});
      if(!files.length) return;
      var base='/static/themes/anime/';
      if(document.body) document.body.style.backgroundImage=
        "linear-gradient(rgba(20,16,26,.86),rgba(20,16,26,.93)), url('"+base+encodeURIComponent(files[0])+"')";
      if(img) img.src=base+encodeURIComponent(files[files.length>1?files.length-1:0]);
    }).catch(function(){});
  }
  window.cynoApplyTheme=applyTheme;
  applyTheme(localStorage.getItem('cyno-theme'));
  window.addEventListener('storage',function(e){ if(e.key==='cyno-theme') applyTheme(e.newValue); });

  var PORTS=[__PORTS__];
  var here=location.port||"80", host=location.hostname;
  var html='<span class="brand">CYNOLYCUS</span>';
  PORTS.forEach(function(p){
    var active=(String(p[1])===String(here))?' class="active"':'';
    html+='<a'+active+' href="http://'+host+':'+p[1]+'/">'+p[0]+'</a>';
  });
  html+='<a href="http://'+host+':8764/themes">Themes</a>'
      +'<span class="spacer"></span>'
      +'<span class="live-badge" id="cyno-live-badge">paper</span>';
  var el=document.getElementById('cyno-nav');
  if(el){el.innerHTML=html;}
})();
</script>""".replace("__PORTS__", _PORTS_JS)


def _picker_page() -> str:
    cards = "".join(
        f'<div class="card theme-card" data-theme-key="{key}">'
        f'<div class="card-head">{name}</div>'
        f'<div class="preview prev-{key}"><span class="swatch s1"></span>'
        f'<span class="swatch s2"></span><span class="swatch s3"></span>'
        f'<span class="swatch s4"></span></div>'
        f'<div class="pick"><button class="primary" onclick="pick(\'{key}\')">Use this theme</button>'
        f'<span class="cur" id="cur-{key}"></span></div></div>'
        for key, name in THEMES
    )
    extra = """
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px;margin:14px 18px}
.preview{height:90px;display:flex}
.swatch{flex:1}
.prev-default .s1{background:#0d1117}.prev-default .s2{background:#1f2630}
.prev-default .s3{background:#58a6ff}.prev-default .s4{background:#3fb950}
.prev-anime .s1{background:#14101a}.prev-anime .s2{background:#2e2440}
.prev-anime .s3{background:#ff8fc8}.prev-anime .s4{background:#6ee7a8}
.pick{padding:12px;display:flex;align-items:center;gap:10px}
.cur{color:var(--green);font-weight:700;font-size:12px}
.note{margin:8px 18px;color:var(--muted);max-width:760px}
"""
    return (
        "<!doctype html><html><head><meta charset=utf-8><title>Themes</title>"
        + THEME_LINK
        + f"<style>{extra}</style></head><body>"
        + NAV_HTML
        + "<div class=grid>" + cards + "</div>"
        + "<p class=note>The <b>Anime</b> theme displays images you place in "
          "<code>UI/static/themes/anime/</code>: <code>bg.jpg</code> (page background) "
          "and <code>corner.png</code> (corner art). Drop your own files there; "
          "nothing ships by default.</p>"
        + """<script>
function pick(t){localStorage.setItem('cyno-theme',t);
  if(window.cynoApplyTheme)window.cynoApplyTheme(t);else document.documentElement.dataset.theme=t;
  mark();}
function mark(){var c=localStorage.getItem('cyno-theme')||'default';
  document.querySelectorAll('.cur').forEach(function(e){e.textContent='';});
  var el=document.getElementById('cur-'+c);if(el)el.textContent='✓ active';}
mark();
</script></body></html>"""
    )


PICKER_HTML = _picker_page()


def serve_theme_css(handler) -> None:
    """Answer ``GET /static/cynolycus_theme.css`` from any BaseHTTPRequestHandler."""
    body = THEME_CSS.encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "text/css; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif", ".bmp"}


def serve_theme_asset(handler, path: str) -> None:
    """Serve a user-supplied theme image from ``UI/static/themes/<theme>/<file>``.

    Special case: ``.../<theme>/__list__`` returns a JSON array of the image
    filenames present in that theme folder, so the UI auto-discovers whatever
    the user dropped in (no fixed filenames required).

    Path-traversal safe; 404 when the file is absent (so a theme with no images
    simply shows no imagery).
    """
    import json as _json

    rel = path[len("/static/themes/"):]
    if rel.endswith("/__list__"):
        theme = rel[: -len("/__list__")]
        tdir = (_THEMES_DIR / theme).resolve()
        files: list[str] = []
        if _THEMES_DIR.resolve() in tdir.parents or tdir == (_THEMES_DIR / theme).resolve():
            if tdir.is_dir():
                files = sorted(p.name for p in tdir.iterdir()
                               if p.is_file() and p.suffix.lower() in _IMG_EXTS)
        body = _json.dumps(files).encode("utf-8")
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        handler.wfile.write(body)
        return

    target = (_THEMES_DIR / rel).resolve()
    if _THEMES_DIR.resolve() not in target.parents or not target.is_file():
        handler.send_response(404)
        handler.send_header("Content-Length", "0")
        handler.end_headers()
        return
    ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    data = target.read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)
