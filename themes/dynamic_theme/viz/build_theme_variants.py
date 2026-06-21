"""Generate 5 visually distinct variations of the 3D theme explorer.

Reuses the exact same graph payload as ``build_theme_explorer`` (themes, links,
ticker memberships) but renders each through a different visual *style* — node
depiction, background, lighting, link treatment, and overall mood. Each style is
written to its own self-contained HTML file, plus an ``index.html`` gallery that
links all five.

    python -m themes.dynamic_theme.viz.build_theme_variants

Outputs (themes/dynamic_theme/viz/variants/):
    theme_explorer_nebula.html        glowing additive orbs, deep-space gradient
    theme_explorer_synthwave.html     neon wireframe nodes on a retro grid
    theme_explorer_blueprint.html     crisp schematic rings, technical blue
    theme_explorer_constellation.html minimalist star-points, hair-thin links
    theme_explorer_aurora.html        frosted translucent spheres, soft lighting
    index.html                        gallery linking the five
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from themes.dynamic_theme.viz.build_theme_explorer import build_graph_payload

logger = logging.getLogger(__name__)

_OUT_DIR = Path(__file__).resolve().parent / "variants"


# --------------------------------------------------------------------------- #
# Style presets. Each entry fully describes one aesthetic. The JS template      #
# reads these as `const STYLE = {...}` and branches its rendering on them.      #
# --------------------------------------------------------------------------- #
STYLES: dict[str, dict] = {
    "nebula": {
        "label": "Nebula",
        "tagline": "Glowing additive orbs drifting through a deep-space haze.",
        "bodyClass": "dark",
        "accent": "#7aa2ff",
        "bggrad": "radial-gradient(ellipse at 32% 18%, #1b1145 0%, #0a0a26 42%, #03020c 100%)",
        "bg": {"transparent": True},
        "nodeKind": "glow",
        "charge": -210,
        "linkDist": {"base": 40, "var": 50},
        "link": {"opacity": 0.85, "width": 1.0, "widthActive": 2.8, "particles": 5,
                 "particleWidth": 2.6, "particleSpeed": 0.006, "curvature": 0.18,
                 "idleOpacity": 0.30},
        "stars": {"n": 2400, "color": "#9fb4e0", "size": 1.9, "opacity": 0.55},
        "grid": None,
        "lights": None,
        "fog": None,
    },
    "synthwave": {
        "label": "Synthwave",
        "tagline": "Neon wireframe nodes hovering over a retro horizon grid.",
        "bodyClass": "dark",
        "accent": "#ff4fd8",
        "bggrad": "linear-gradient(180deg, #1a0633 0%, #2b0a4d 38%, #5b0f53 64%, #11041f 100%)",
        "bg": {"transparent": True},
        "nodeKind": "wire",
        "charge": -180,
        "linkDist": {"base": 34, "var": 42},
        "linkMono": "#ff4fd8",
        "linkMonoActive": "#27e7ff",
        "link": {"opacity": 0.9, "width": 1.2, "widthActive": 3.0, "particles": 4,
                 "particleWidth": 2.6, "particleSpeed": 0.01, "curvature": 0.0,
                 "idleOpacity": 0.22},
        "stars": {"n": 700, "color": "#ff8ad8", "size": 2.0, "opacity": 0.5},
        "grid": {"size": 2600, "div": 46, "center": "#27e7ff", "line": "#7a1f86",
                 "y": -220, "opacity": 0.55},
        "lights": None,
        "fog": None,
    },
    "blueprint": {
        "label": "Blueprint",
        "tagline": "Crisp schematic rings and straight technical wiring.",
        "bodyClass": "dark",
        "accent": "#5fd0ff",
        "bggrad": "linear-gradient(180deg, #0a2a52 0%, #07223f 100%)",
        "bg": {"solid": "#0a2647"},
        "nodeKind": "ring",
        "charge": -240,
        "linkDist": {"base": 44, "var": 46},
        "linkMono": "#7fd6ff",
        "linkMonoActive": "#d6f1ff",
        "link": {"opacity": 0.8, "width": 1.0, "widthActive": 2.2, "particles": 0,
                 "particleWidth": 2.0, "particleSpeed": 0.006, "curvature": 0.0,
                 "idleOpacity": 0.28},
        "stars": None,
        "grid": {"size": 3000, "div": 60, "center": "#2f6da3", "line": "#1c4a78",
                 "y": -260, "opacity": 0.6},
        "lights": None,
        "fog": None,
    },
    "constellation": {
        "label": "Constellation",
        "tagline": "Minimalist star-points joined by hair-thin lines.",
        "bodyClass": "dark",
        "accent": "#cdd9f2",
        "bggrad": "radial-gradient(ellipse at 50% 40%, #05060d 0%, #020308 70%, #000000 100%)",
        "bg": {"solid": "#010206"},
        "nodeKind": "point",
        "charge": -260,
        "linkDist": {"base": 50, "var": 60},
        "linkMono": "#aab8d8",
        "linkMonoActive": "#ffffff",
        "link": {"opacity": 0.7, "width": 0.5, "widthActive": 1.4, "particles": 0,
                 "particleWidth": 1.6, "particleSpeed": 0.004, "curvature": 0.0,
                 "idleOpacity": 0.14},
        "stars": {"n": 3200, "color": "#cdd9f2", "size": 1.4, "opacity": 0.65},
        "grid": None,
        "lights": None,
        "fog": {"color": "#01020a", "density": 0.0011},
    },
    "aurora": {
        "label": "Aurora Glass",
        "tagline": "Frosted translucent spheres lit by soft aurora light.",
        "bodyClass": "light",
        "accent": "#2563eb",
        "bggrad": "linear-gradient(155deg, #d7f5ee 0%, #cfe3ff 38%, #e8d9ff 70%, #ffe1ef 100%)",
        "bg": {"transparent": True},
        "nodeKind": "glass",
        "charge": -200,
        "linkDist": {"base": 42, "var": 48},
        "link": {"opacity": 0.55, "width": 2.2, "widthActive": 4.2, "particles": 3,
                 "particleWidth": 3.0, "particleSpeed": 0.005, "curvature": 0.22,
                 "idleOpacity": 0.22},
        "stars": None,
        "grid": None,
        "lights": {"ambient": 0.85, "key": "#ffffff", "keyI": 0.7,
                   "fill": "#a7d8ff", "fillI": 0.5, "rim": "#ffc4e6", "rimI": 0.4},
        "fog": None,
    },
}


def _palette_css(style: dict) -> str:
    return (
        f"html,body{{background:{style['bggrad']} !important;}}\n"
        f":root{{--accent:{style['accent']};}}\n"
        f"body.dark{{--accent:{style['accent']};}}"
    )


def build_html(payload: dict, key: str, style: dict) -> str:
    body_class = "dark" if style["bodyClass"] == "dark" else ""
    return (
        _HTML_TEMPLATE
        .replace("__GRAPH_DATA__", json.dumps(payload, separators=(",", ":")))
        .replace("__STYLE_JSON__", json.dumps(style, separators=(",", ":")))
        .replace("__PALETTE_CSS__", _palette_css(style))
        .replace("__BODY_CLASS__", body_class)
        .replace("__TITLE__", f"Theme Explorer — {style['label']}")
    )


def build_index(keys_styles: list[tuple[str, dict]]) -> str:
    cards = "\n".join(
        f"""    <a class="card {key}" href="theme_explorer_{key}.html">
      <div class="swatch"></div>
      <div class="meta"><h2>{s['label']}</h2><p>{s['tagline']}</p></div>
    </a>"""
        for key, s in keys_styles
    )
    swatches = "\n".join(
        f"  .card.{key} .swatch{{background:{s['bggrad']};}}" for key, s in keys_styles
    )
    return _INDEX_TEMPLATE.replace("__CARDS__", cards).replace("__SWATCHES__", swatches)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    payload = build_graph_payload()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, style in STYLES.items():
        out = _OUT_DIR / f"theme_explorer_{key}.html"
        out.write_text(build_html(payload, key, style), encoding="utf-8")
        print(f"  {style['label']:14} -> {out.relative_to(_OUT_DIR.parents[3])}")
    index = _OUT_DIR / "index.html"
    index.write_text(build_index(list(STYLES.items())), encoding="utf-8")
    print(
        f"\nWrote {len(STYLES)} variants + gallery\n"
        f"  themes: {payload['n_themes']}  links: {payload['n_links']}  tickers: {payload['n_tickers']}\n"
        f"  open {index} (or any variant) directly in a browser."
    )


_INDEX_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Theme Explorer — Variations</title>
<style>
  *{box-sizing:border-box;}
  body{margin:0;min-height:100%;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
    background:radial-gradient(ellipse at 30% 10%,#10182e,#070a14 60%,#03040a);color:#dce5f5;padding:48px 32px;}
  h1{font-size:22px;letter-spacing:.5px;margin:0 0 4px;}
  .sub{color:#8a97b4;font-size:13px;margin-bottom:28px;}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:18px;max-width:1100px;}
  .card{display:flex;flex-direction:column;text-decoration:none;color:inherit;border:1px solid #23314c;
    border-radius:14px;overflow:hidden;background:#0d1220;transition:transform .12s,border-color .12s,box-shadow .12s;}
  .card:hover{transform:translateY(-3px);border-color:#5db0ff;box-shadow:0 10px 30px rgba(10,20,50,.5);}
  .swatch{height:150px;}
  .meta{padding:14px 16px 16px;}
  .meta h2{margin:0 0 5px;font-size:15px;}
  .meta p{margin:0;font-size:12.5px;color:#8a97b4;line-height:1.5;}
__SWATCHES__
</style></head><body>
  <h1>Theme Explorer — 5 Variations</h1>
  <div class="sub">Same data, five renderings. Click one to open it.</div>
  <div class="grid">
__CARDS__
  </div>
</body></html>
"""


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>__TITLE__</title>
<style>
  :root { --bg:#eef2f8; --panel:rgba(255,255,255,.96); --panel2:#f4f7fc; --line:#d2dbe8;
          --txt:#1c2740; --dim:#64708a; --accent:#2563eb; --hover:#eaf0fb; --chip:#eef3fb;
          --chiptext:#3a465e; --rowborder:#e7edf6; --connbg:#eaeff8; --connhover:#e0e8f5;
          --desc:#3a465e; --overlay:rgba(255,255,255,.92); --kbd:#e8eef8; --selrow:#dde9fc;
          --bggrad:radial-gradient(ellipse at 30% 15%, #ffffff 0%, #e9eef7 55%, #dde5f1 100%); }
  body.dark { --bg:#070b14; --panel:rgba(13,18,30,.93); --panel2:#0e1320; --line:#23314c;
          --txt:#dce5f5; --dim:#8a97b4; --accent:#5db0ff; --hover:#16223a; --chip:#15203a;
          --chiptext:#bcc9e2; --rowborder:#141d2c; --connbg:#121b2c; --connhover:#1b273e;
          --desc:#c5d1e6; --overlay:rgba(10,15,26,.85); --kbd:#1b2740; --selrow:#1d2c49;
          --bggrad:radial-gradient(ellipse at 30% 20%, #0d1530 0%, #070a14 50%, #03040a 100%); }
  * { box-sizing:border-box; }
  html,body { margin:0; height:100%; background:var(--bg); color:var(--txt); overflow:hidden;
      font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; background:var(--bggrad); }
  #graph { position:absolute; inset:0 560px 0 0; }
  #side { position:absolute; top:0; right:0; bottom:0; width:560px; background:var(--panel);
      backdrop-filter:blur(8px); border-left:1px solid var(--line); display:flex; flex-direction:column; }
  #side header { padding:14px 16px 12px; border-bottom:1px solid var(--line); }
  .hrow { display:flex; align-items:center; justify-content:space-between; gap:10px; }
  .hbtns { display:flex; gap:8px; align-items:center; }
  #side header h1 { margin:0; font-size:15px; letter-spacing:.4px; }
  .badge { font-size:10px; text-transform:uppercase; letter-spacing:.7px; color:var(--accent);
      border:1px solid var(--accent); border-radius:20px; padding:2px 9px; }
  .btn { background:var(--chip); border:1px solid var(--line); color:var(--txt); font-size:12px;
      padding:6px 12px; border-radius:7px; cursor:pointer; white-space:nowrap; }
  .btn:hover { border-color:var(--accent); color:var(--accent); }
  #pins { display:none; border-bottom:1px solid var(--line); background:color-mix(in srgb, var(--accent) 6%, transparent); max-height:32%; overflow-y:auto; }
  #pins.show { display:block; }
  .pinhdr { font-size:10.5px; text-transform:uppercase; letter-spacing:.6px; color:var(--accent); padding:10px 16px 5px; display:flex; justify-content:space-between; }
  .pinhdr .none { color:var(--dim); cursor:pointer; text-transform:none; letter-spacing:0; }
  .pinhdr .none:hover { color:var(--accent); }
  .pinrow { display:flex; align-items:center; gap:9px; padding:5px 16px; font-size:12.5px; }
  .pinrow:hover { background:var(--hover); }
  .pinrow.off { opacity:.5; }
  .pinrow .box { display:inline-flex; width:15px; height:15px; border:1px solid var(--accent); border-radius:4px; align-items:center; justify-content:center; font-size:10px; color:var(--bg); cursor:pointer; }
  .pinrow.on .box { background:var(--accent); }
  .pinrow .pn { flex:1; cursor:pointer; } .pinrow .pn:hover { color:var(--accent); }
  .pinrow .ct { color:var(--dim); font-size:10px; }
  .pinrow .rm { color:var(--dim); cursor:pointer; padding:0 3px; } .pinrow .rm:hover { color:#e0455e; }
  #side header .meta { color:var(--dim); font-size:11px; margin-top:4px; }
  #search { width:100%; margin-top:10px; padding:9px 11px; background:var(--panel2); border:1px solid var(--line); border-radius:8px; color:var(--txt); font-size:13px; }
  #search:focus { outline:none; border-color:var(--accent); }
  #tickercard { display:none; padding:11px 16px; border-bottom:1px solid var(--line); background:color-mix(in srgb, var(--accent) 8%, transparent); }
  #tickercard.show { display:block; }
  #tickercard h3 { margin:0 0 6px; font-size:13px; } #tickercard h3 b { color:var(--accent); }
  #list { flex:1; overflow-y:auto; }
  .row { padding:9px 16px; border-bottom:1px solid var(--rowborder); cursor:pointer; font-size:13px; }
  .row:hover { background:var(--hover); }
  .row.sel { background:var(--selrow); box-shadow:inset 3px 0 0 var(--accent); }
  .row .nm { color:var(--txt); }
  .row .ct { color:var(--dim); font-size:11px; float:right; }
  .row .pt { color:var(--dim); font-size:11px; display:block; margin-top:2px; }
  .dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:7px; vertical-align:middle; box-shadow:0 0 0 1px rgba(0,0,0,.18); }
  #detail { border-top:1px solid var(--line); max-height:58%; overflow-y:auto; padding:0 16px 18px; display:none; background:var(--panel2); }
  #detail.show { display:block; }
  #detail h2 { font-size:14px; margin:14px 0 4px; word-break:break-word; }
  #detail .sub { color:var(--dim); font-size:11px; margin-bottom:8px; }
  #detail .desc { font-size:12px; line-height:1.55; color:var(--desc); margin-bottom:10px; }
  .sec { font-size:10.5px; text-transform:uppercase; letter-spacing:.7px; color:var(--accent); margin:13px 0 6px; }
  .conn { font-size:12px; padding:5px 8px; margin-bottom:4px; border-radius:6px; cursor:pointer; background:var(--connbg); display:flex; justify-content:space-between; gap:8px; align-items:center; }
  .conn:hover { background:var(--connhover); }
  .conn .rel { color:var(--dim); font-size:10px; white-space:nowrap; }
  .chips { display:flex; flex-wrap:wrap; gap:5px; }
  .chip { font-size:11px; padding:3px 8px; background:var(--chip); border:1px solid var(--line); border-radius:11px; color:var(--chiptext); cursor:pointer; }
  .chip:hover { border-color:var(--accent); color:var(--accent); }
  .chip .sc { color:var(--dim); margin-left:5px; }
  #legend, #help { position:absolute; left:14px; background:var(--overlay); border:1px solid var(--line); border-radius:9px; padding:10px 13px; font-size:11px; box-shadow:0 2px 12px rgba(20,30,60,.18); }
  #legend { bottom:14px; } #help { top:14px; color:var(--dim); max-width:360px; line-height:1.7; }
  #help b { color:var(--txt); }
  #legend b { display:block; margin-bottom:6px; color:var(--dim); text-transform:uppercase; letter-spacing:.5px; }
  #legend .li { display:flex; align-items:center; gap:8px; margin:3px 0; }
  #legend .sw { width:20px; height:3px; border-radius:2px; }
  kbd { background:var(--kbd); border:1px solid var(--line); border-radius:4px; padding:1px 5px; font-size:10px; color:var(--txt); }
  /* ---- per-variant palette ---- */
__PALETTE_CSS__
</style>
</head>
<body class="__BODY_CLASS__">
<div id="graph"></div>
<div id="help">
  <b>Fly:</b> <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> move ·
  <kbd>Space</kbd> up · <kbd>Shift</kbd> down · <kbd>Q</kbd>/<kbd>E</kbd> roll · drag = look · scroll = zoom<br/>
  <b>Click</b> a node/row/connection to pin &amp; highlight. Hover anything to preview. Click a ticker to jump to its themes.
</div>
<div id="legend"></div>
<div id="side">
  <header>
    <div class="hrow"><h1>Theme Explorer</h1>
      <div class="hbtns"><span class="badge" id="stylebadge"></span><button id="emergingbtn" class="btn">Emerging: on</button><button id="reset" class="btn" title="Clear highlights">⟲ Reset</button></div>
    </div>
    <div class="meta" id="meta"></div>
    <input id="search" placeholder="Search a theme or a ticker (e.g. AAPL)…" autocomplete="off" spellcheck="false" />
  </header>
  <div id="pins"></div>
  <div id="tickercard"></div>
  <div id="list"></div>
  <div id="detail"></div>
</div>

<script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
<script src="https://unpkg.com/3d-force-graph@1.73.4/dist/3d-force-graph.min.js"></script>
<script>
const DATA = __GRAPH_DATA__;
const STYLE = __STYLE_JSON__;
const REL_COLORS = { correlated:'#6f7dff', supply_chain:'#28c08a', drives_demand:'#ff9e2c',
                     enables:'#ff6fb0', competes_with:'#ff5d5d', emerging_from:'#ffd166', related:'#8aa0c0' };
function hexA(hex,a){ const m=hex.replace('#',''); const r=parseInt(m.substr(0,2),16),
  g=parseInt(m.substr(2,2),16), b=parseInt(m.substr(4,2),16); return `rgba(${r},${g},${b},${a})`; }
// transparent version of any CSS colour (hsl() nodes or #hex links) — canvas
// addColorStop throws on an unparseable colour, so handle both formats.
function clearColor(c){ if(c[0]==='#') return hexA(c,0);
  if(c.startsWith('hsl(')) return c.replace('hsl(','hsla(').replace(')',',0)');
  if(c.startsWith('rgb(')) return c.replace('rgb(','rgba(').replace(')',',0)');
  return 'rgba(0,0,0,0)'; }
function esc(s){ return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

// ---------- indices ----------
const byId = {};
DATA.nodes.forEach(n => { n.neighbors = []; n.linksByOther = {}; byId[n.id] = n; });
DATA.links.forEach(l => {
  const s = byId[l.source], t = byId[l.target]; if (!s || !t) return;
  if (!s.linksByOther[t.id]) s.neighbors.push(t);
  if (!t.linksByOther[s.id]) t.neighbors.push(s);
  (s.linksByOther[t.id] = s.linksByOther[t.id] || []).push({ ...l, dir:'out' });
  (t.linksByOther[s.id] = t.linksByOther[s.id] || []).push({ ...l, dir:'in' });
});
function parentColor(p){ let h=0; for(let i=0;i<p.length;i++) h=(h*31+p.charCodeAt(i))>>>0;
  return `hsl(${h%360}, ${62+(h>>3)%22}%, ${55+(h>>6)%10}%)`; }
DATA.nodes.forEach(n => n.color = n.status==='provisional' ? '#ffd166' : parentColor(n.parent));
const themesOf = tk => (DATA.ticker_index[tk] || []).map(x => x.theme);

const maxMcap = Math.max(1, ...DATA.nodes.map(n => n.market_cap || 0));
const nodeSize = n => n.status==='provisional'
  ? 6 + 8 * Math.max(0.15, n.emerging_score || 0)
  : 3.2 + 16 * Math.sqrt((n.market_cap || 0) / maxMcap);
function fmtCap(v){ v = v || 0;
  if (v >= 1e12) return '$' + (v/1e12).toFixed(2) + 'T';
  if (v >= 1e9)  return '$' + (v/1e9).toFixed(1) + 'B';
  if (v >= 1e6)  return '$' + (v/1e6).toFixed(0) + 'M';
  return v ? '$' + v.toFixed(0) : 'n/a'; }

// ---------- node textures (sprite-based kinds) ----------
const _tex = {};
function _cv(){ const c = document.createElement('canvas'); c.width = c.height = 128; return c; }
function orbTexture(hex){ const k='orb'+hex; if(_tex[k]) return _tex[k];
  const c=_cv(), g=c.getContext('2d');
  const grd=g.createRadialGradient(64,64,4,64,64,52);
  grd.addColorStop(0,hex); grd.addColorStop(0.78,hex); grd.addColorStop(1,clearColor(hex));
  g.fillStyle=grd; g.beginPath(); g.arc(64,64,52,0,Math.PI*2); g.fill();
  g.lineWidth=6; g.strokeStyle='rgba(120,132,156,0.75)';
  g.beginPath(); g.arc(64,64,49,0,Math.PI*2); g.stroke();
  return _tex[k]=new THREE.CanvasTexture(c); }
function glowTexture(hex){ const k='glow'+hex; if(_tex[k]) return _tex[k];
  const c=_cv(), g=c.getContext('2d');
  const grd=g.createRadialGradient(64,64,2,64,64,62);
  grd.addColorStop(0,'rgba(255,255,255,0.95)'); grd.addColorStop(0.18,hex);
  grd.addColorStop(0.45,clearColor(hex).replace(/,0\)$/,',0.55)'));
  grd.addColorStop(1,clearColor(hex));
  g.fillStyle=grd; g.fillRect(0,0,128,128);
  return _tex[k]=new THREE.CanvasTexture(c); }
function pointTexture(hex){ const k='pt'+hex; if(_tex[k]) return _tex[k];
  const c=_cv(), g=c.getContext('2d');
  const grd=g.createRadialGradient(64,64,1,64,64,46);
  grd.addColorStop(0,'#ffffff'); grd.addColorStop(0.22,hex); grd.addColorStop(0.55,clearColor(hex));
  g.fillStyle=grd; g.fillRect(0,0,128,128);
  return _tex[k]=new THREE.CanvasTexture(c); }
function ringTexture(hex){ const k='ring'+hex; if(_tex[k]) return _tex[k];
  const c=_cv(), g=c.getContext('2d');
  g.lineWidth=9; g.strokeStyle=hex; g.beginPath(); g.arc(64,64,46,0,Math.PI*2); g.stroke();
  g.lineWidth=2; g.strokeStyle='rgba(255,255,255,0.55)'; g.beginPath(); g.arc(64,64,46,0,Math.PI*2); g.stroke();
  const grd=g.createRadialGradient(64,64,2,64,64,30);
  grd.addColorStop(0,clearColor(hex).replace(',0)',',0.20)')); grd.addColorStop(1,clearColor(hex));
  g.fillStyle=grd; g.beginPath(); g.arc(64,64,40,0,Math.PI*2); g.fill();
  return _tex[k]=new THREE.CanvasTexture(c); }

function makeNode(n){
  const col=n.color, sz=nodeSize(n), K=STYLE.nodeKind; let o, mat;
  if (K==='wire' || K==='crystal' || K==='glass'){
    let geo;
    if (K==='glass') geo=new THREE.SphereGeometry(sz*1.05,26,18);
    else geo=new THREE.IcosahedronGeometry(sz*1.1, K==='crystal'?2:1);
    if (K==='wire') mat=new THREE.MeshBasicMaterial({color:col, wireframe:true, transparent:true, opacity:0.95});
    else if (K==='crystal') mat=new THREE.MeshStandardMaterial({color:col, roughness:0.4, metalness:0.18, flatShading:true, transparent:true, opacity:0.97});
    else mat=new THREE.MeshStandardMaterial({color:col, roughness:0.12, metalness:0.0, transparent:true, opacity:0.72});
    o=new THREE.Mesh(geo,mat); o.__base=1;
  } else {
    const tex = K==='glow'?glowTexture(col) : K==='point'?pointTexture(col) : K==='ring'?ringTexture(col) : orbTexture(col);
    const blend = (K==='glow'||K==='point') ? THREE.AdditiveBlending : THREE.NormalBlending;
    mat=new THREE.SpriteMaterial({map:tex, transparent:true, blending:blend, depthWrite:false, opacity:0.96});
    o=new THREE.Sprite(mat);
    const f = K==='point'?1.5 : K==='glow'?3.0 : K==='ring'?2.4 : 2.2;
    o.__base = Math.max(K==='point'?4:6, sz*f);
  }
  o.__mat=mat; n.__obj=o; return o;
}

// ---------- graph ----------
const elGraph = document.getElementById('graph');
let selected = null, hovered = null, tickerSet = null, previewSet = null;
let showEmerging = true;
const pins = [], enabled = new Set();

const Graph = ForceGraph3D()(elGraph)
  .graphData({ nodes: DATA.nodes, links: DATA.links })
  .showNavInfo(false)
  .nodeThreeObject(makeNode)
  .nodeVisibility(n => showEmerging || n.status!=='provisional')
  .linkVisibility(l => showEmerging || !String(typeof l.source==='object'?l.source.id:l.source).startsWith('emerging::'))
  .nodeLabel(n => `<div style="font:13px sans-serif;color:#fff;padding:5px 8px;
      background:rgba(8,12,20,.95);border:1px solid #2a3650;border-radius:7px">
      <b>${esc(n.label||n.id)}${n.status==='provisional'?' · EMERGING':''}</b><br><span style="color:#8a98b3">${esc(n.parent)} · ${fmtCap(n.market_cap)} · ${n.primary_count} tickers · ${n.neighbors.length} links</span></div>`)
  .linkColor(colorLink)
  .linkCurvature(STYLE.link.curvature || 0)
  .linkWidth(l => isActive(l) ? STYLE.link.widthActive : STYLE.link.width)
  .linkOpacity(STYLE.link.opacity)
  .linkDirectionalParticles(l => isActive(l) ? STYLE.link.particles : 0)
  .linkDirectionalParticleWidth(STYLE.link.particleWidth)
  .linkDirectionalParticleSpeed(STYLE.link.particleSpeed)
  .onNodeClick(n => pinTheme(n.id, true))
  .onNodeHover(n => { hovered = n; elGraph.style.cursor = n ? 'pointer' : 'grab'; refresh(); });

Graph.backgroundColor(STYLE.bg.solid || 'rgba(0,0,0,0)');
Graph.d3Force('charge').strength(STYLE.charge);
Graph.d3Force('link').distance(l => STYLE.linkDist.base + STYLE.linkDist.var * (1 - (l.strength || 0.5)));

// ---------- scene extras (lights / grid / fog) per style ----------
const scene = Graph.scene();
if (STYLE.lights){
  scene.add(new THREE.AmbientLight(0xffffff, STYLE.lights.ambient));
  const k=new THREE.DirectionalLight(new THREE.Color(STYLE.lights.key), STYLE.lights.keyI); k.position.set(1,1,1); scene.add(k);
  const f=new THREE.DirectionalLight(new THREE.Color(STYLE.lights.fill), STYLE.lights.fillI); f.position.set(-1,-0.6,-0.8); scene.add(f);
  if (STYLE.lights.rim){ const r=new THREE.DirectionalLight(new THREE.Color(STYLE.lights.rim), STYLE.lights.rimI); r.position.set(0,-1,0.6); scene.add(r); }
}
if (STYLE.grid){
  const grid=new THREE.GridHelper(STYLE.grid.size, STYLE.grid.div, new THREE.Color(STYLE.grid.center), new THREE.Color(STYLE.grid.line));
  grid.position.y = STYLE.grid.y; grid.material.transparent=true; grid.material.opacity=STYLE.grid.opacity; scene.add(grid);
}
if (STYLE.fog){ scene.fog = new THREE.FogExp2(new THREE.Color(STYLE.fog.color), STYLE.fog.density); }

function hlSet(){ return previewSet || tickerSet || enabled; }
function endpoints(l){ return [ typeof l.source==='object'?l.source.id:l.source,
                                typeof l.target==='object'?l.target.id:l.target ]; }
function isActive(l){
  const [s,t] = endpoints(l);
  if (previewSet || tickerSet) { const S = previewSet||tickerSet; return S.has(s) && S.has(t); }
  if (enabled.size && (enabled.has(s) || enabled.has(t))) return true;
  if (hovered && (s===hovered.id || t===hovered.id)) return true;
  return false;
}
function colorLink(l){
  if (STYLE.linkMono) return isActive(l) ? (STYLE.linkMonoActive || STYLE.linkMono) : hexA(STYLE.linkMono, STYLE.link.idleOpacity);
  const base = REL_COLORS[l.relationship]||REL_COLORS.related;
  return isActive(l) ? base : hexA(base, STYLE.link.idleOpacity);
}

function refresh(){
  const HL = hlSet(), hov = hovered ? hovered.id : null, any = HL.size > 0 || hov;
  DATA.nodes.forEach(n => {
    const o = n.__obj; if (!o) return;
    let st = 'normal';
    if (any) {
      const active = HL.has(n.id) || n.id === hov;
      let neigh = false;
      if (!active) {
        if (hov && byId[hov].linksByOther[n.id]) neigh = true;
        if (!neigh) for (const id of HL) { if (byId[id].linksByOther[n.id]) { neigh = true; break; } }
      }
      st = active ? 'active' : neigh ? 'neighbor' : 'dim';
    }
    n.__state = st;
    const m = o.__mat, baseOp = STYLE.nodeKind==='glass' ? 0.72 : 0.96;
    if (st==='active'){ m.opacity = 1.0; o.__mul = 1.55; }
    else if (st==='neighbor'){ m.opacity = Math.min(1, baseOp+0.04); o.__mul = 1.15; }
    else if (st==='dim'){ m.opacity = STYLE.nodeKind==='glass'?0.18:0.30; o.__mul = 0.9; }
    else { m.opacity = baseOp; o.__mul = 1.0; }
  });
  Graph.linkColor(colorLink)
       .linkWidth(l => isActive(l) ? STYLE.link.widthActive : STYLE.link.width)
       .linkDirectionalParticles(l => isActive(l) ? STYLE.link.particles : 0);
}

// pulse active nodes
let _t0 = performance.now();
(function pulse(){
  const dt = (performance.now() - _t0) / 1000;
  DATA.nodes.forEach(n => { const o = n.__obj; if (!o) return;
    const p = n.__state==='active' ? (1 + 0.12*Math.sin(dt*4)) : 1;
    o.scale.setScalar(o.__base * (o.__mul||1) * p); });
  requestAnimationFrame(pulse);
})();

// ---------- pin / highlight model ----------
function pinTheme(id, focus){
  const n = byId[id]; if (!n) return;
  if (!pins.includes(id)) pins.push(id);
  enabled.add(id); clearTickerCard(); previewSet = null; selected = n;
  if (focus) flyTo(n);
  refresh(); renderPins(); renderDetail(n); markList();
}
function togglePin(id){ if (enabled.has(id)) enabled.delete(id); else enabled.add(id);
  if (!pins.includes(id)) pins.push(id); refresh(); renderPins(); markList(); }
function unpin(id){ enabled.delete(id); const i = pins.indexOf(id); if (i>=0) pins.splice(i,1);
  refresh(); renderPins(); markList(); }
function resetAll(){ pins.length = 0; enabled.clear(); tickerSet = null; previewSet = null;
  selected = null; hovered = null; document.getElementById('search').value = ''; filter = '';
  clearTickerCard(); refresh(); renderPins(); renderDetail(null); renderList(); }

function selectTicker(tk){
  pins.length = 0; enabled.clear(); selected = null; hovered = null; previewSet = null;
  document.getElementById('search').value = tk; filter = tk;
  showTickerCard(tk); renderPins(); renderDetail(null); renderList();
}

// ---------- camera ----------
function flyTo(n){
  if (!n || n.x === undefined) return;
  const eu = new THREE.Euler(pitch,yaw,roll,'YXZ');
  const back = new THREE.Vector3(0,0,1).applyEuler(eu).multiplyScalar(nodeSize(n) + 110);
  const pos = new THREE.Vector3(n.x+back.x, n.y+back.y, n.z+back.z);
  const tmp = cam.position.clone(); cam.position.copy(pos); setLookAt(n); const ty=yaw, tp=pitch;
  cam.position.copy(tmp); fly = { pos, yaw: ty, pitch: tp };
}

// ---------- pinned panel ----------
const elPins = document.getElementById('pins');
function renderPins(){
  if (!pins.length){ elPins.className=''; elPins.innerHTML=''; return; }
  elPins.className = 'show';
  elPins.innerHTML =
    `<div class="pinhdr"><span>Highlighted themes (${enabled.size}/${pins.length})</span><span class="none" data-act="clear">clear all</span></div>` +
    pins.map(id => { const n = byId[id]; const on = enabled.has(id);
      return `<div class="pinrow ${on?'on':'off'}" data-id="${esc(id)}">
        <span class="box" data-act="toggle">${on?'✓':''}</span>
        <span class="dot" style="background:${n.color}"></span>
        <span class="pn" data-act="focus">${esc(id)}</span>
        <span class="ct">${fmtCap(n.market_cap)} · ${n.neighbors.length}🔗</span>
        <span class="rm" data-act="remove" title="remove">✕</span></div>`; }).join('');
  elPins.querySelector('[data-act="clear"]').onclick = resetAll;
  [...elPins.querySelectorAll('.pinrow')].forEach(r => {
    const id = r.dataset.id, n = byId[id];
    r.querySelector('[data-act="toggle"]').onclick = e => { e.stopPropagation(); togglePin(id); };
    r.querySelector('[data-act="remove"]').onclick = e => { e.stopPropagation(); unpin(id); };
    r.querySelector('[data-act="focus"]').onclick = () => { enabled.add(id); selected = n;
      flyTo(n); refresh(); renderPins(); renderDetail(n); markList(); };
    r.onmouseenter = () => { hovered = n; refresh(); };
    r.onmouseleave = () => { if (hovered===n){ hovered = null; refresh(); } };
  });
}
document.getElementById('reset').onclick = resetAll;
document.getElementById('emergingbtn').onclick = () => {
  showEmerging = !showEmerging;
  document.getElementById('emergingbtn').textContent = `Emerging: ${showEmerging?'on':'off'}`;
  Graph.nodeVisibility(n => showEmerging || n.status!=='provisional')
    .linkVisibility(l => showEmerging || !String(typeof l.source==='object'?l.source.id:l.source).startsWith('emerging::'));
};

// ---------- sidebar list + search ----------
const elList = document.getElementById('list'), elDetail = document.getElementById('detail'), elCard = document.getElementById('tickercard');
document.getElementById('stylebadge').textContent = STYLE.label;
document.getElementById('meta').textContent =
  `${DATA.n_themes} themes · ${DATA.n_emerging||0} emerging · ${DATA.n_links} links · ${DATA.n_tickers} tickers · ${DATA.generated_at.replace('T',' ')}`;

let filter = '';
function tickerHit(q){ const U=q.toUpperCase(); return DATA.ticker_index[U] ? U : null; }
function matchTheme(n, f){ return (n.label||n.id).toLowerCase().includes(f) || n.parent.toLowerCase().includes(f)
    || n.members.some(m => m.ticker.toLowerCase().includes(f)); }
function renderList(){
  const f = filter.toLowerCase();
  const rows = DATA.nodes.filter(n => !f || matchTheme(n,f)).sort((a,b)=>a.id.localeCompare(b.id));
  elList.innerHTML = rows.map(n => `
    <div class="row" data-id="${esc(n.id)}">
      <span class="ct">${fmtCap(n.market_cap)}</span>
      <span class="nm"><span class="dot" style="background:${n.color}"></span>${esc(n.label||n.id)}${n.status==='provisional'?' · emerging':''}</span>
      <span class="pt">${esc(n.parent)} · ${n.primary_count} tickers${n.pending_count?' · '+n.pending_count+' pending':''} · ${n.neighbors.length} connections</span>
    </div>`).join('') || '<div class="row" style="color:var(--dim)">no matches</div>';
  [...elList.querySelectorAll('.row[data-id]')].forEach(r => {
    const n = byId[r.dataset.id];
    r.onclick = () => pinTheme(n.id, true);
    r.onmouseenter = () => { hovered = n; refresh(); };
    r.onmouseleave = () => { if (hovered===n){ hovered = null; refresh(); } };
  });
  markList();
}
function markList(){ [...elList.querySelectorAll('.row')].forEach(r =>
  r.classList.toggle('sel', enabled.has(r.dataset.id) || (selected && r.dataset.id===selected.id))); }

document.getElementById('search').addEventListener('input', e => {
  filter = e.target.value.trim();
  const tk = tickerHit(filter);
  if (tk) showTickerCard(tk); else clearTickerCard();
  renderList();
});

function showTickerCard(tk){
  const themes = DATA.ticker_index[tk];
  tickerSet = new Set(themes.map(t => t.theme)); previewSet = null;
  selected = null; hovered = null;
  elCard.className = 'show';
  elCard.innerHTML = `<h3><b>${esc(tk)}</b> belongs to ${themes.length} themes — hover to preview, click to focus</h3><div class="chips">` +
    themes.map(t => `<span class="chip" data-id="${esc(t.theme)}">${esc(t.theme)}<span class="sc">${t.score}</span></span>`).join('') + `</div>`;
  [...elCard.querySelectorAll('.chip[data-id]')].forEach(c => { const id = c.dataset.id;
    c.onclick = () => pinTheme(id, true);
    c.onmouseenter = () => { hovered = byId[id]; refresh(); };
    c.onmouseleave = () => { if (hovered===byId[id]){ hovered = null; refresh(); } };
  });
  refresh();
}
function clearTickerCard(){ if(elCard.className){ elCard.className=''; elCard.innerHTML=''; } if(tickerSet){ tickerSet=null; refresh(); } }

// ---------- detail panel ----------
function renderDetail(n){
  if (!n){ elDetail.className=''; elDetail.innerHTML=''; return; }
  const conns = n.neighbors.map(o => ({ other:o, links:n.linksByOther[o.id] }))
    .sort((a,b) => Math.max(...b.links.map(l=>l.strength)) - Math.max(...a.links.map(l=>l.strength)));
  const connHtml = conns.map(c => {
    const tags = c.links.map(l => `<span class="rel" style="color:${REL_COLORS[l.relationship]||REL_COLORS.related}">${l.dir==='out'?'→':'←'} ${esc(l.relationship)} ${l.strength.toFixed(2)}</span>`).join(' ');
    return `<div class="conn" data-id="${esc(c.other.id)}"><span>${esc(c.other.id)}</span><span>${tags}</span></div>`;
  }).join('') || '<div class="sub">no graph connections</div>';
  const sibs = (n.siblings||[]).map(s => `<span class="chip" data-id="${esc(s)}">${esc(s)}</span>`).join('');
  const related = (n.related||[]).map(r => byId[r]
      ? `<span class="chip" data-id="${esc(r)}">${esc(r)}</span>` : `<span class="chip" style="cursor:default">${esc(r)}</span>`).join('');
  const members = n.members.map(m => `<span class="chip" data-tk="${esc(m.ticker)}">${esc(m.ticker)}<span class="sc">${m.score}</span></span>`).join('');

  elDetail.className = 'show';
  elDetail.innerHTML = `
    <h2><span class="dot" style="background:${n.color}"></span> ${esc(n.label||n.id)}${n.status==='provisional'?' · EMERGING':''}</h2>
    <div class="sub">parent: <b style="color:var(--desc)">${esc(n.parent)}</b> · <b style="color:var(--desc)">${fmtCap(n.market_cap)}</b> total cap · ${n.primary_count} primary / ${n.total_count} total tickers</div>
    <div class="desc">${esc(n.description) || '<i>no description</i>'}</div>
    ${n.status==='provisional' ? `<div class="sub">emerging score: <b>${(n.emerging_score||0).toFixed(3)}</b> · closest: <b>${esc(n.closest_theme||'n/a')}</b> · 5d breadth: <b>${((n.breadth_5d||0)*100).toFixed(0)}%</b></div>` : ''}
    <div class="sec">Connections (${conns.length})</div>${connHtml}
    ${sibs ? `<div class="sec">Sibling themes — same parent (${n.siblings.length})</div><div class="chips">${sibs}</div>` : ''}
    ${related ? `<div class="sec">Related (labelled)</div><div class="chips">${related}</div>` : ''}
    <div class="sec">Top tickers (${n.primary_count})</div><div class="chips">${members || '<span class="sub">none</span>'}</div>
    ${n.pending_members&&n.pending_members.length&&n.status!=='provisional' ? `<div class="sec">Pending expansion (${n.pending_count})</div><div class="chips">${n.pending_members.map(m=>`<span class="chip" data-tk="${esc(m.ticker)}">${esc(m.ticker)}<span class="sc">${m.score}</span></span>`).join('')}</div>` : ''}`;

  [...elDetail.querySelectorAll('.conn[data-id],.chip[data-id]')].forEach(c => { const id = c.dataset.id;
    c.onclick = () => pinTheme(id, true);
    c.onmouseenter = () => { hovered = byId[id]; refresh(); };
    c.onmouseleave = () => { if (hovered===byId[id]){ hovered = null; refresh(); } };
  });
  [...elDetail.querySelectorAll('.chip[data-tk]')].forEach(c => { const tk = c.dataset.tk;
    c.onclick = () => selectTicker(tk);
    c.onmouseenter = () => { previewSet = new Set(themesOf(tk)); refresh(); };
    c.onmouseleave = () => { previewSet = null; refresh(); };
  });
  elDetail.scrollTop = 0;
}

// ---------- starfield ----------
if (STYLE.stars){ (function stars(){
  const N = STYLE.stars.n, pos = new Float32Array(N*3);
  for (let i=0;i<N;i++){ const r = 1600 + Math.random()*2600, th = Math.random()*Math.PI*2, ph = Math.acos(2*Math.random()-1);
    pos[i*3]=r*Math.sin(ph)*Math.cos(th); pos[i*3+1]=r*Math.sin(ph)*Math.sin(th); pos[i*3+2]=r*Math.cos(ph); }
  const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos,3));
  scene.add(new THREE.Points(geo, new THREE.PointsMaterial({ color:new THREE.Color(STYLE.stars.color), size:STYLE.stars.size, sizeAttenuation:false, transparent:true, opacity:STYLE.stars.opacity })));
})(); }

// ---------- custom fly controls ----------
const cam = Graph.camera();
const ctrls = Graph.controls(); ctrls.enabled = false; ctrls.update = function(){};
const dom = Graph.renderer().domElement;
let yaw = 0, pitch = 0, roll = 0, speed = 6, fly = null, interacted = false;
const keys = {};
cam.position.set(0, 0, 460);
function setLookAt(target){
  const d = new THREE.Vector3(target.x-cam.position.x, target.y-cam.position.y, target.z-cam.position.z).normalize();
  yaw = Math.atan2(-d.x, -d.z); pitch = Math.asin(Math.max(-1,Math.min(1,d.y))); }
dom.addEventListener('pointerdown', e => { if(e.button!==0) return; dom.__drag=true; dom.__px=e.clientX; dom.__py=e.clientY; interacted=true; elGraph.style.cursor='grabbing'; });
window.addEventListener('pointerup', () => { dom.__drag=false; elGraph.style.cursor='grab'; });
window.addEventListener('pointermove', e => { if(!dom.__drag) return; fly=null;
  yaw -= (e.clientX - dom.__px) * 0.0026; pitch -= (e.clientY - dom.__py) * 0.0026;
  pitch = Math.max(-1.45, Math.min(1.45, pitch)); dom.__px = e.clientX; dom.__py = e.clientY; });
dom.addEventListener('wheel', e => { e.preventDefault(); interacted=true;
  const eu = new THREE.Euler(pitch,yaw,roll,'YXZ');
  cam.position.add(new THREE.Vector3(0,0,-1).applyEuler(eu).multiplyScalar(-e.deltaY*0.6)); fly=null;
}, { passive:false });
window.addEventListener('keydown', e => { if(e.target.tagName==='INPUT') return;
  keys[e.code]=true; interacted=true; if(['Space','KeyW','KeyA','KeyS','KeyD'].includes(e.code)) e.preventDefault(); });
window.addEventListener('keyup', e => { keys[e.code]=false; });
(function flyLoop(){
  if (keys.KeyQ) { roll += 0.025; fly=null; } if (keys.KeyE) { roll -= 0.025; fly=null; }
  const eu = new THREE.Euler(pitch, yaw, roll, 'YXZ');
  const fwd = new THREE.Vector3(0,0,-1).applyEuler(eu), right = new THREE.Vector3(1,0,0).applyEuler(eu), up = new THREE.Vector3(0,1,0);
  const v = new THREE.Vector3();
  if (keys.KeyW) v.add(fwd); if (keys.KeyS) v.sub(fwd);
  if (keys.KeyD) v.add(right); if (keys.KeyA) v.sub(right);
  if (keys.Space) v.add(up); if (keys.ShiftLeft||keys.ShiftRight) v.sub(up);
  if (v.lengthSq() > 0) { v.normalize().multiplyScalar(speed); cam.position.add(v); fly=null; }
  if (fly) { cam.position.lerp(fly.pos, 0.14); yaw += (fly.yaw - yaw) * 0.14; pitch += (fly.pitch - pitch) * 0.14;
    if (cam.position.distanceTo(fly.pos) < 1.2) fly = null; }
  cam.quaternion.setFromEuler(new THREE.Euler(pitch, yaw, roll, 'YXZ'));
  requestAnimationFrame(flyLoop);
})();

// ---------- legend / init ----------
document.getElementById('legend').innerHTML =
  `<b>${esc(STYLE.label)}</b>` +
  (STYLE.linkMono
    ? `<div class="li"><span class="sw" style="background:${STYLE.linkMonoActive||STYLE.linkMono}"></span>connection</div>`
    : Object.entries(REL_COLORS).filter(([k])=>k!=='related')
        .map(([k,v]) => `<div class="li"><span class="sw" style="background:${v}"></span>${k}</div>`).join('')) +
  '<div class="li" style="margin-top:6px;color:var(--dim)">node size = total market cap · colour = parent group</div>';

renderList(); renderPins(); elGraph.style.cursor = 'grab';
window.addEventListener('resize', () => Graph.width(elGraph.clientWidth).height(elGraph.clientHeight));
let framed = false;
Graph.onEngineStop(() => { if(!framed && !interacted && !fly && !selected){
  cam.position.set(0,0,Math.max(420, DATA.n_themes*7)); yaw=0; pitch=0; } framed = true; });
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
