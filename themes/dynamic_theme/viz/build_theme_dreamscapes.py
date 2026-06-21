"""Alternate-universe renderings of the theme map — the trippy cousins of
``build_theme_variants``.

These don't just reskin the force graph; they reimagine the *metaphor*. Each
reuses the same data + interaction engine (sidebar, search, fly camera) but
layers on a live WebGL plasma backdrop, hue-cycling "living" nodes that breathe,
flowing spores, and — for the mandala — a golden-angle sacred-geometry layout in
place of physics.

    python -m themes.dynamic_theme.viz.build_theme_dreamscapes

Outputs (themes/dynamic_theme/viz/dreamscapes/):
    mycelium.html       a fungal "wood-wide-web": breathing bioluminescent spores
    sacred_spiral.html  themes on a rotating golden-angle phyllotaxis mandala
    liquid_light.html   melting lava-lamp blobs that bleed into one another
    index.html          gallery

This module performs a handful of surgical string injections on the shared
``_HTML_TEMPLATE`` from build_theme_variants so the proven interaction code stays
the single source of truth. The injection anchors are asserted at import-build
time; if the base template changes shape, generation fails loudly.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from themes.dynamic_theme.viz.build_theme_explorer import build_graph_payload
from themes.dynamic_theme.viz.build_theme_variants import _HTML_TEMPLATE, _INDEX_TEMPLATE

logger = logging.getLogger(__name__)

_OUT_DIR = Path(__file__).resolve().parent / "dreamscapes"


# --------------------------------------------------------------------------- #
# Surgical injections into the shared template.                                #
# --------------------------------------------------------------------------- #
_ANCHOR_GRAPHDIV = '<div id="graph"></div>'
_INJECT_GRAPHDIV = '<canvas id="bgfx"></canvas>\n<div id="graph"></div>'

_ANCHOR_MAKENODE = """function makeNode(n){
  const col=n.color, sz=nodeSize(n), K=STYLE.nodeKind; let o, mat;
  if (K==='wire' || K==='crystal' || K==='glass'){"""
_INJECT_MAKENODE = """// soft white halo — tinted live via material.color so nodes can hue-cycle.
function softTexture(big){ const k='soft'+(big?1:0); if(_tex[k]) return _tex[k];
  const c=_cv(), g=c.getContext('2d');
  const grd=g.createRadialGradient(64,64,1,64,64,63);
  grd.addColorStop(0,'rgba(255,255,255,1)');
  grd.addColorStop(big?0.5:0.30,'rgba(255,255,255,0.5)');
  grd.addColorStop(1,'rgba(255,255,255,0)');
  g.fillStyle=grd; g.fillRect(0,0,128,128);
  return _tex[k]=new THREE.CanvasTexture(c); }
function _hash(s){ let h=0; for(let i=0;i<s.length;i++) h=(h*31+s.charCodeAt(i))>>>0; return h; }
function makeNode(n){
  const col=n.color, sz=nodeSize(n), K=STYLE.nodeKind; let o, mat;
  if (K==='spore' || K==='blob'){
    mat=new THREE.SpriteMaterial({map:softTexture(K==='blob'), transparent:true,
        blending:THREE.AdditiveBlending, depthWrite:false, opacity:K==='blob'?0.5:0.92});
    o=new THREE.Sprite(mat);
    o.__base=Math.max(8, sz*(K==='blob'?6.8:3.4));
    if (n.__hue===undefined) n.__hue=(_hash(n.parent)%360)/360;
    n.__seed=Math.random();
    o.__mat=mat; n.__obj=o; return o;
  }
  if (K==='wire' || K==='crystal' || K==='glass'){"""

_ANCHOR_GRAPHCFG = """Graph.backgroundColor(STYLE.bg.solid || 'rgba(0,0,0,0)');
Graph.d3Force('charge').strength(STYLE.charge);
Graph.d3Force('link').distance(l => STYLE.linkDist.base + STYLE.linkDist.var * (1 - (l.strength || 0.5)));"""
_INJECT_GRAPHCFG = _ANCHOR_GRAPHCFG + """

// ----- sacred-geometry layout: pin nodes on a golden-angle phyllotaxis disc -----
if (STYLE.layout === 'phyllotaxis'){
  // Keep the forces running (they keep the engine ticking + syncing object
  // positions) but pin every node via fx/fy/fz so they hold the mandala.
  Graph.d3Force('charge').strength(-4); Graph.d3Force('center', null);
  const order=[...DATA.nodes].sort((a,b)=> a.parent.localeCompare(b.parent) || (b.market_cap-a.market_cap));
  const N=order.length, GA=Math.PI*(3-Math.sqrt(5)), SC=30;
  order.forEach((n,i)=>{ const r=SC*Math.sqrt(i+0.6), a=i*GA;
    // set x/y/z directly (objects read these) AND fx/fy/fz (pins) — with forces
    // removed the sim may not tick to copy fx->x, so position explicitly.
    n.fx=n.x=r*Math.cos(a); n.fy=n.y=r*Math.sin(a); n.fz=n.z=70*Math.sin(r*0.010);
    n.__radHue=i/N; });
  if (Graph.d3ReheatSimulation) Graph.d3ReheatSimulation();
}

// ----- live WebGL plasma backdrop (domain-warped fbm) -----
let _bg=null;
function initBg(){
  const cv=document.getElementById('bgfx'); if(!cv || !STYLE.shader) return;
  const gl=cv.getContext('webgl'); if(!gl) return;
  const vs='attribute vec2 p;void main(){gl_Position=vec4(p,0.,1.);}';
  const fs=`precision highp float;uniform vec2 u_res;uniform float u_time;
    uniform vec3 c1,c2,c3;uniform float u_scale,u_speed,u_warp;
    float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
    float noise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);
      float a=hash(i),b=hash(i+vec2(1,0)),c=hash(i+vec2(0,1)),d=hash(i+vec2(1,1));
      return mix(mix(a,b,f.x),mix(c,d,f.x),f.y);}
    float fbm(vec2 p){float v=0.,a=.5;for(int i=0;i<5;i++){v+=a*noise(p);p*=2.03;a*=.5;}return v;}
    void main(){vec2 uv=(gl_FragCoord.xy-0.5*u_res)/u_res.y;
      float t=u_time*u_speed;
      vec2 q=vec2(fbm(uv*u_scale+t),fbm(uv*u_scale-t+5.2));
      vec2 r=vec2(fbm(uv*u_scale+u_warp*q+t*0.7),fbm(uv*u_scale+u_warp*q-t*0.6));
      float f=fbm(uv*u_scale+u_warp*r);
      vec3 col=mix(c1,c2,clamp(f*1.7,0.,1.));
      col=mix(col,c3,clamp(length(q)*0.9,0.,1.));
      col+=0.05*sin(vec3(0.,2.,4.)+u_time*0.3+f*6.0);
      gl_FragColor=vec4(col,1.);}`;
  function sh(t,s){const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);return o;}
  const prog=gl.createProgram();
  gl.attachShader(prog,sh(gl.VERTEX_SHADER,vs));
  gl.attachShader(prog,sh(gl.FRAGMENT_SHADER,fs));
  gl.linkProgram(prog); gl.useProgram(prog);
  const buf=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,buf);
  gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,3,-1,-1,3]),gl.STATIC_DRAW);
  const loc=gl.getAttribLocation(prog,'p'); gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc,2,gl.FLOAT,false,0,0);
  const U=k=>gl.getUniformLocation(prog,k);
  const u={res:U('u_res'),time:U('u_time'),c1:U('c1'),c2:U('c2'),c3:U('c3'),
           scale:U('u_scale'),speed:U('u_speed'),warp:U('u_warp')};
  const hx=h=>{const m=h.replace('#','');return [parseInt(m.substr(0,2),16)/255,
      parseInt(m.substr(2,2),16)/255,parseInt(m.substr(4,2),16)/255];};
  function resize(){cv.width=cv.clientWidth; cv.height=cv.clientHeight; gl.viewport(0,0,cv.width,cv.height);}
  window.addEventListener('resize',resize); resize();
  _bg={gl,u,cv,c:[hx(STYLE.shader.c1),hx(STYLE.shader.c2),hx(STYLE.shader.c3)]};
}
function renderBg(t){ if(!_bg) return; const {gl,u,c,cv}=_bg;
  if(cv.width!==cv.clientWidth||cv.height!==cv.clientHeight){
    cv.width=cv.clientWidth; cv.height=cv.clientHeight; gl.viewport(0,0,cv.width,cv.height); }
  gl.uniform2f(u.res,cv.width,cv.height); gl.uniform1f(u.time,t);
  gl.uniform3fv(u.c1,c[0]); gl.uniform3fv(u.c2,c[1]); gl.uniform3fv(u.c3,c[2]);
  gl.uniform1f(u.scale,STYLE.shader.scale); gl.uniform1f(u.speed,STYLE.shader.speed);
  gl.uniform1f(u.warp,STYLE.shader.warp);
  gl.drawArrays(gl.TRIANGLES,0,3); }
initBg();"""

_ANCHOR_PULSE = """let _t0 = performance.now();
(function pulse(){
  const dt = (performance.now() - _t0) / 1000;
  DATA.nodes.forEach(n => { const o = n.__obj; if (!o) return;
    const p = n.__state==='active' ? (1 + 0.12*Math.sin(dt*4)) : 1;
    o.scale.setScalar(o.__base * (o.__mul||1) * p); });
  requestAnimationFrame(pulse);
})();"""
_INJECT_PULSE = """let _t0 = performance.now();
// `interacted` is declared later with `let`; reading it during this IIFE's
// synchronous first run would hit the temporal dead zone, so go through a
// try/catch helper. Keeping the IIFE (vs a deferred rAF) means sprites get
// scaled on frame zero rather than depending on rAF timing.
function _userIdle(){ try { return !interacted; } catch(e){ return true; } }
(function pulse(){
  const t = (performance.now() - _t0) / 1000;
  if (STYLE.shader) renderBg(t);
  const HUE = STYLE.hue, BR = STYLE.breatheAll || 0;
  DATA.nodes.forEach(n => { const o = n.__obj; if (!o) return;
    const breathe = BR ? (1 + BR*Math.sin(t*1.05 + (n.__seed||0)*6.283)) : 1;
    const act = n.__state==='active' ? (1 + 0.16*Math.sin(t*4)) : 1;
    o.scale.setScalar(o.__base * (o.__mul||1) * breathe * act);
    if (HUE && o.__mat && o.__mat.color){
      const base = (HUE.by==='radius' ? n.__radHue : n.__hue) || 0;
      const h = (((base + t*HUE.speed) % 1) + 1) % 1;
      const lift = n.__state==='active' ? 0.16 : 0;
      o.__mat.color.setHSL(h, HUE.sat, (n.__state==='dim' ? HUE.light*0.45 : HUE.light) + lift);
    }
  });
  if (STYLE.autoOrbit && _userIdle()) scene.rotation[STYLE.orbitAxis||'y'] += STYLE.autoOrbit;
  requestAnimationFrame(pulse);
})();"""


_ANCHOR_FIT = "window.addEventListener('resize', () => Graph.width(elGraph.clientWidth).height(elGraph.clientHeight));"
_INJECT_FIT = _ANCHOR_FIT + """
// The force-graph can latch onto the container size before CSS layout settles
// (the extra synchronous init below shifts the timing), leaving the WebGL canvas
// sized wrong so the 3D scene projects outside the visible pane. Re-fit after layout.
function _fitGraph(){ Graph.width(elGraph.clientWidth).height(elGraph.clientHeight);
  if (_bg && _bg.cv){ _bg.cv.width=_bg.cv.clientWidth; _bg.cv.height=_bg.cv.clientHeight;
    _bg.gl.viewport(0,0,_bg.cv.width,_bg.cv.height); } }
requestAnimationFrame(_fitGraph); setTimeout(_fitGraph,200); setTimeout(_fitGraph,800);"""


def _base_template() -> str:
    t = _HTML_TEMPLATE
    for anchor, inject, name in (
        (_ANCHOR_GRAPHDIV, _INJECT_GRAPHDIV, "graph-div"),
        (_ANCHOR_MAKENODE, _INJECT_MAKENODE, "makeNode"),
        (_ANCHOR_GRAPHCFG, _INJECT_GRAPHCFG, "graph-config"),
        (_ANCHOR_PULSE, _INJECT_PULSE, "pulse"),
        (_ANCHOR_FIT, _INJECT_FIT, "fit"),
    ):
        if t.count(anchor) != 1:
            raise RuntimeError(f"dreamscape injection anchor '{name}' matched {t.count(anchor)} times")
        t = t.replace(anchor, inject)
    return t


_DREAM_TEMPLATE = _base_template()


# --------------------------------------------------------------------------- #
# Styles. New keys vs. variants: shader, hue, breatheAll, layout, autoOrbit.    #
# --------------------------------------------------------------------------- #
STYLES: dict[str, dict] = {
    "mycelium": {
        "label": "Mycelium",
        "tagline": "A breathing fungal wood-wide-web — bioluminescent spores drifting on living threads.",
        "bodyClass": "dark",
        "accent": "#54f0c8",
        "bggrad": "radial-gradient(ellipse at 40% 30%, #07211f 0%, #0a0a1e 60%, #05030c 100%)",
        "bg": {"transparent": True},
        "nodeKind": "spore",
        "charge": -160,
        "linkDist": {"base": 36, "var": 60},
        "linkMono": "#3affc0",
        "linkMonoActive": "#aaffe9",
        "link": {"opacity": 0.7, "width": 1.1, "widthActive": 2.6, "particles": 6,
                 "particleWidth": 2.4, "particleSpeed": 0.0035, "curvature": 0.45,
                 "idleOpacity": 0.20},
        "stars": {"n": 900, "color": "#6affd0", "size": 1.6, "opacity": 0.4},
        "grid": None, "lights": None,
        "fog": {"color": "#05121a", "density": 0.0012},
        "hue": {"by": "parent", "sat": 0.72, "light": 0.6, "speed": 0.018},
        "breatheAll": 0.13,
        "layout": "force",
        "autoOrbit": 0.0006,
        "shader": {"c1": "#04120f", "c2": "#2a0f5e", "c3": "#0fb88e",
                   "scale": 2.2, "speed": 0.05, "warp": 3.5},
    },
    "sacred_spiral": {
        "label": "Sacred Spiral",
        "tagline": "Every theme set on a golden-angle phyllotaxis mandala, slowly turning, rainbow rippling outward.",
        "bodyClass": "dark",
        "accent": "#c9a8ff",
        "bggrad": "radial-gradient(ellipse at 50% 50%, #1a0b3a 0%, #0c0524 55%, #04020f 100%)",
        "bg": {"transparent": True},
        "nodeKind": "spore",
        "charge": -160,
        "linkDist": {"base": 36, "var": 60},
        "linkMono": "#b48bff",
        "linkMonoActive": "#ffffff",
        "link": {"opacity": 0.55, "width": 0.8, "widthActive": 2.0, "particles": 0,
                 "particleWidth": 1.8, "particleSpeed": 0.004, "curvature": 0.6,
                 "idleOpacity": 0.16},
        "stars": None, "grid": None, "lights": None, "fog": None,
        "hue": {"by": "radius", "sat": 0.78, "light": 0.62, "speed": 0.03},
        "breatheAll": 0.08,
        "layout": "phyllotaxis",
        "autoOrbit": 0.0016,
        "orbitAxis": "z",
        "shader": {"c1": "#0a0524", "c2": "#3a1a6e", "c3": "#e0a64f",
                   "scale": 1.6, "speed": 0.03, "warp": 2.4},
    },
    "liquid_light": {
        "label": "Liquid Light",
        "tagline": "A melting lava-lamp where themes lose their edges and bleed into pools of colour.",
        "bodyClass": "dark",
        "accent": "#ff7ad9",
        "bggrad": "radial-gradient(ellipse at 35% 40%, #2a0636 0%, #12041f 55%, #050109 100%)",
        "bg": {"transparent": True},
        "nodeKind": "blob",
        "charge": -90,
        "linkDist": {"base": 28, "var": 38},
        "linkMono": "#ff7ad9",
        "linkMonoActive": "#ffd0f2",
        "link": {"opacity": 0.0, "width": 0.0, "widthActive": 0.0, "particles": 0,
                 "particleWidth": 1.0, "particleSpeed": 0.004, "curvature": 0.0,
                 "idleOpacity": 0.0},
        "stars": None, "grid": None, "lights": None,
        "fog": {"color": "#0a0312", "density": 0.0009},
        "hue": {"by": "parent", "sat": 0.85, "light": 0.6, "speed": 0.05},
        "breatheAll": 0.18,
        "layout": "force",
        "autoOrbit": 0.0008,
        "shader": {"c1": "#16031f", "c2": "#a3128a", "c3": "#ff8a3c",
                   "scale": 1.9, "speed": 0.07, "warp": 4.2},
    },
}


def _palette_css(style: dict) -> str:
    return (
        f"html,body{{background:{style['bggrad']} !important;}}\n"
        f":root{{--accent:{style['accent']};}}\n"
        f"body.dark{{--accent:{style['accent']};}}\n"
        "#bgfx{position:absolute;inset:0 560px 0 0;z-index:0;display:block;}\n"
        "#graph{z-index:1;}\n"
        "#help,#legend{z-index:5;}\n"
        "#side{z-index:6;}"
    )


def build_html(payload: dict, style: dict) -> str:
    body_class = "dark" if style["bodyClass"] == "dark" else ""
    return (
        _DREAM_TEMPLATE
        .replace("__GRAPH_DATA__", json.dumps(payload, separators=(",", ":")))
        .replace("__STYLE_JSON__", json.dumps(style, separators=(",", ":")))
        .replace("__PALETTE_CSS__", _palette_css(style))
        .replace("__BODY_CLASS__", body_class)
        .replace("__TITLE__", f"Theme Dreamscape — {style['label']}")
    )


def build_index(keys_styles: list[tuple[str, dict]]) -> str:
    cards = "\n".join(
        f"""    <a class="card {key}" href="{key}.html">
      <div class="swatch"></div>
      <div class="meta"><h2>{s['label']}</h2><p>{s['tagline']}</p></div>
    </a>"""
        for key, s in keys_styles
    )
    swatches = "\n".join(
        f"  .card.{key} .swatch{{background:{s['bggrad']};}}" for key, s in keys_styles
    )
    return (
        _INDEX_TEMPLATE
        .replace("Theme Explorer — Variations", "Theme Dreamscapes")
        .replace("Theme Explorer — 5 Variations", "Theme Dreamscapes — Alternate Universes")
        .replace("Same data, five renderings. Click one to open it.",
                 "The same theme map, reimagined. Click one to fall in.")
        .replace("__CARDS__", cards)
        .replace("__SWATCHES__", swatches)
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    payload = build_graph_payload()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, style in STYLES.items():
        out = _OUT_DIR / f"{key}.html"
        out.write_text(build_html(payload, style), encoding="utf-8")
        print(f"  {style['label']:14} -> {out.relative_to(_OUT_DIR.parents[3])}")
    index = _OUT_DIR / "index.html"
    index.write_text(build_index(list(STYLES.items())), encoding="utf-8")
    print(
        f"\nWrote {len(STYLES)} dreamscapes + gallery\n"
        f"  themes: {payload['n_themes']}  links: {payload['n_links']}  tickers: {payload['n_tickers']}\n"
        f"  open {index} (or any dreamscape) directly in a browser."
    )


if __name__ == "__main__":
    main()
