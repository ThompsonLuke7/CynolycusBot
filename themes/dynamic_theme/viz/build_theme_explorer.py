"""Build an interactive 3D theme explorer from the dynamic-theme outputs.

Reads the latest theme registry, relationships, and ticker memberships and writes
a single self-contained HTML file (data embedded inline) that opens directly in a
browser — no server required.

Re-run after the dynamic theme pipeline regenerates its outputs:

    python -m themes.dynamic_theme.viz.build_theme_explorer

Output: themes/dynamic_theme/viz/theme_explorer.html
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[3]
_OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
_REGISTRY_PATH = _OUTPUTS / "theme_registry.parquet"
_RELATIONSHIPS_PATH = _OUTPUTS / "theme_relationships.parquet"
_MEMBERSHIP_PATH = _OUTPUTS / "ticker_theme_membership.parquet"
_PENDING_REGISTRY_PATH = _OUTPUTS / "pending_theme_registry.parquet"
_PENDING_MEMBERSHIP_PATH = _OUTPUTS / "pending_theme_membership.parquet"
_PENDING_PROFILES_PATH = _OUTPUTS / "pending_theme_profiles.parquet"
_ASSIGNMENTS_CSV = _OUTPUTS / "ticker_theme_assignments.csv"
_PROFILES_PATH = _REPO / "signals" / "news" / "data" / "processed" / "ticker_profiles.parquet"
_HTML_OUT = Path(__file__).resolve().parent / "theme_explorer.html"

_MAX_MEMBERS = 80
_TOP_K_PER_TICKER = 6


def _safe_list(val) -> list[str]:
    if isinstance(val, list):
        return [str(v) for v in val]
    if isinstance(val, str) and val.strip():
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except Exception:
            pass
    return []


def _latest(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns and df["date"].notna().any():
        return df[df["date"] == df["date"].max()].copy()
    return df


def _membership_from_parquet(valid_themes: set[str]) -> dict[str, list[tuple[str, float]]]:
    if _MEMBERSHIP_PATH.exists():
        mem = _latest(pd.read_parquet(_MEMBERSHIP_PATH))
        mem = mem[mem["theme"].isin(valid_themes)]
        out: dict[str, list[tuple[str, float]]] = {}
        for ticker, grp in mem.groupby("ticker"):
            ranked = grp.sort_values("membership_score", ascending=False)
            out[str(ticker)] = [
                (str(t), float(s))
                for t, s in zip(ranked["theme"][:_TOP_K_PER_TICKER], ranked["membership_score"][:_TOP_K_PER_TICKER])
            ]
        return out

    if _ASSIGNMENTS_CSV.exists():
        asg = pd.read_csv(_ASSIGNMENTS_CSV)
        out = {}
        theme_cols = [c for c in asg.columns if c.startswith("theme_")]
        for _, row in asg.iterrows():
            pairs = []
            for tc in theme_cols:
                theme = row.get(tc)
                if isinstance(theme, str) and theme in valid_themes:
                    pairs.append((theme, float(row.get(tc.replace("theme_", "score_"), 0.0) or 0.0)))
            out[str(row["ticker"])] = pairs
        return out

    return {}


def _market_caps() -> dict[str, float]:
    """Return {ticker: marketCap} from the latest Yahoo profile snapshot."""
    if not _PROFILES_PATH.exists():
        return {}
    df = pd.read_parquet(_PROFILES_PATH, columns=["ticker", "snapshot_date", "marketCap"])
    df = df.dropna(subset=["ticker", "marketCap"])
    if "snapshot_date" in df.columns:
        df = df.sort_values("snapshot_date").drop_duplicates("ticker", keep="last")
    caps = {str(t).upper(): float(c) for t, c in zip(df["ticker"], df["marketCap"]) if c > 0}
    if _PENDING_PROFILES_PATH.exists():
        pending = pd.read_parquet(_PENDING_PROFILES_PATH)
        if {"ticker", "marketCap"}.issubset(pending.columns):
            for ticker, cap in zip(pending["ticker"], pending["marketCap"]):
                if pd.notna(cap) and float(cap) > 0:
                    caps[str(ticker).upper()] = float(cap)
    return caps


def build_graph_payload() -> dict:
    registry = _latest(pd.read_parquet(_REGISTRY_PATH))
    # Two clusters can get the SAME Claude label → duplicate theme names. Collapse
    # to one node per name (keep the highest-confidence row) so the graph doesn't
    # render a disconnected twin floating away from the cluster.
    registry = registry.sort_values("confidence", ascending=False).drop_duplicates("theme_name", keep="first")

    relationships = _latest(pd.read_parquet(_RELATIONSHIPS_PATH))

    valid_themes = set(registry["theme_name"].astype(str))
    ticker_memberships = _membership_from_parquet(valid_themes)
    caps = _market_caps()

    primary_members: dict[str, list[tuple[str, float]]] = {}
    total_count: dict[str, int] = {}
    theme_mcap: dict[str, float] = {}
    for ticker, pairs in ticker_memberships.items():
        for rank, (theme, score) in enumerate(pairs):
            total_count[theme] = total_count.get(theme, 0) + 1
            if rank == 0:
                primary_members.setdefault(theme, []).append((ticker, score))
                theme_mcap[theme] = theme_mcap.get(theme, 0.0) + caps.get(ticker.upper(), 0.0)

    siblings: dict[str, list[str]] = {}
    for _parent, grp in registry.groupby(registry["parent_theme"].fillna("uncategorized")):
        names = sorted(str(x) for x in grp["theme_name"])
        for n in names:
            siblings[n] = [m for m in names if m != n]

    nodes = []
    node_by_name = {}
    for _, r in registry.iterrows():
        name = str(r["theme_name"])
        prim = sorted(primary_members.get(name, []), key=lambda x: -x[1])
        node = {
                "id": name,
                "label": name,
                "status": "established",
                "parent": str(r.get("parent_theme") or "") or "uncategorized",
                "description": str(r.get("description") or ""),
                "confidence": float(r.get("confidence") or 0.0),
                "related": [x for x in _safe_list(r.get("related_themes")) if x],
                "siblings": siblings.get(name, []),
                "primary_count": len(primary_members.get(name, [])),
                "total_count": total_count.get(name, 0),
                "market_cap": round(theme_mcap.get(name, 0.0)),
                "members": [{"ticker": t, "score": round(s, 3)} for t, s in prim[:_MAX_MEMBERS]],
                "pending_members": [],
                "pending_count": 0,
            }
        nodes.append(node)
        node_by_name[name] = node

    # De-dupe links too (collapsed names can create duplicate edges).
    seen, links = set(), []
    for _, r in relationships.iterrows():
        src, tgt = str(r["source"]), str(r["target"])
        key = (src, tgt, str(r.get("relationship")))
        if src in valid_themes and tgt in valid_themes and src != tgt and key not in seen:
            seen.add(key)
            links.append({"source": src, "target": tgt,
                          "relationship": str(r.get("relationship") or "related"),
                          "strength": float(r.get("strength") or 0.5)})

    ticker_index = {
        t: [{"theme": th, "score": round(sc, 3)} for th, sc in pairs]
        for t, pairs in ticker_memberships.items()
    }

    emerging_count = 0
    if _PENDING_REGISTRY_PATH.exists() and _PENDING_MEMBERSHIP_PATH.exists():
        pending_registry = _latest(pd.read_parquet(_PENDING_REGISTRY_PATH))
        pending_membership = _latest(pd.read_parquet(_PENDING_MEMBERSHIP_PATH))
        pending_by_key = {
            key: group.sort_values("membership_score", ascending=False)
            for key, group in pending_membership.groupby("theme_key")
        }
        for _, row in pending_registry.iterrows():
            theme_type = str(row.get("theme_type") or "")
            theme_name = str(row.get("theme_name") or "")
            members = pending_by_key.get(str(row.get("theme_key")), pd.DataFrame())
            member_rows = [
                {"ticker": str(t), "score": round(float(s), 3)}
                for t, s in zip(members.get("ticker", []), members.get("membership_score", []))
            ]
            if theme_type == "extension" and theme_name in node_by_name:
                node = node_by_name[theme_name]
                node["pending_members"] = member_rows[:_MAX_MEMBERS]
                node["pending_count"] = len(member_rows)
                node["emerging_score"] = float(row.get("emerging_score") or 0.0)
                for member in member_rows:
                    ticker_index.setdefault(member["ticker"], []).append(
                        {"theme": theme_name, "score": member["score"], "status": "pending"}
                    )
                continue
            if theme_type != "provisional":
                continue

            node_id = f"emerging::{theme_name}"
            market_cap = round(sum(caps.get(member["ticker"].upper(), 0.0) for member in member_rows))
            nodes.append(
                {
                    "id": node_id,
                    "label": theme_name,
                    "status": "provisional",
                    "parent": str(row.get("parent_theme") or "emerging_market_theme"),
                    "description": str(row.get("description") or ""),
                    "confidence": float(row.get("emerging_score") or 0.0),
                    "emerging_score": float(row.get("emerging_score") or 0.0),
                    "closest_theme": str(row.get("closest_theme") or ""),
                    "closest_similarity": float(row.get("closest_similarity") or 0.0),
                    "breadth_5d": float(row.get("breadth_5d") or 0.0),
                    "avg_return_5d": float(row.get("avg_return_5d") or 0.0),
                    "volume_acceleration": float(row.get("avg_volume_acceleration") or 0.0),
                    "related": [],
                    "siblings": [],
                    "primary_count": len(member_rows),
                    "total_count": len(member_rows),
                    "pending_count": len(member_rows),
                    "market_cap": market_cap,
                    "members": member_rows[:_MAX_MEMBERS],
                    "pending_members": member_rows[:_MAX_MEMBERS],
                }
            )
            emerging_count += 1
            closest = str(row.get("closest_theme") or "")
            if closest in valid_themes:
                links.append(
                    {
                        "source": node_id,
                        "target": closest,
                        "relationship": "emerging_from",
                        "strength": max(0.35, float(row.get("closest_similarity") or 0.0)),
                    }
                )
            for member in member_rows:
                ticker_index.setdefault(member["ticker"], []).append(
                    {"theme": node_id, "score": member["score"], "status": "provisional"}
                )

    return {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "n_themes": len(nodes),
        "n_links": len(links),
        "n_tickers": len(ticker_index),
        "n_emerging": emerging_count,
        "nodes": nodes,
        "links": links,
        "ticker_index": ticker_index,
    }


def build_html(payload: dict) -> str:
    return _HTML_TEMPLATE.replace("__GRAPH_DATA__", json.dumps(payload, separators=(",", ":")))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    payload = build_graph_payload()
    _HTML_OUT.write_text(build_html(payload), encoding="utf-8")
    print(
        f"Wrote {_HTML_OUT}\n"
        f"  themes: {payload['n_themes']}  links: {payload['n_links']}  tickers: {payload['n_tickers']}\n"
        f"  open it directly in a browser (no server needed)."
    )


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Theme Explorer — 3D</title>
<style>
  /* light theme (default vars) — body.dark overrides for dark mode */
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
  .hbtns { display:flex; gap:8px; }
  #side header h1 { margin:0; font-size:15px; letter-spacing:.4px; }
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
</style>
</head>
<body class="dark">
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
      <div class="hbtns"><button id="emergingbtn" class="btn">Emerging: on</button><button id="themebtn" class="btn">☀ Light</button><button id="reset" class="btn" title="Clear selections and reset camera">⌂ Reset view</button></div>
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
const REL_COLORS = { correlated:'#6f7dff', supply_chain:'#28c08a', drives_demand:'#ff9e2c',
                     enables:'#ff6fb0', competes_with:'#ff5d5d', emerging_from:'#ffd166', related:'#8aa0c0' };
function hexA(hex,a){ const m=hex.replace('#',''); const r=parseInt(m.substr(0,2),16),
  g=parseInt(m.substr(2,2),16), b=parseInt(m.substr(4,2),16); return `rgba(${r},${g},${b},${a})`; }
// transparent version of any CSS colour (node colours are hsl(), link colours are #hex).
// hexA() only parses #hex, so feeding it an hsl() string yields an unparseable colour and
// canvas addColorStop() throws — which would kill every orb sprite. Handle both formats.
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

// ---------- orb sprites (solid disc + crisp ring → readable on light & dark) ----------
const _tex = {};
function orbTexture(hex){
  if (_tex[hex]) return _tex[hex];
  const c = document.createElement('canvas'); c.width = c.height = 128;
  const g = c.getContext('2d');
  const grd = g.createRadialGradient(64,64,4, 64,64,52);
  grd.addColorStop(0.0, hex); grd.addColorStop(0.78, hex); grd.addColorStop(1.0, clearColor(hex));
  g.fillStyle = grd; g.beginPath(); g.arc(64,64,52,0,Math.PI*2); g.fill();
  g.lineWidth = 6; g.strokeStyle = 'rgba(120,132,156,0.75)';
  g.beginPath(); g.arc(64,64,49,0,Math.PI*2); g.stroke();
  const t = new THREE.CanvasTexture(c); _tex[hex] = t; return t;
}
function makeNode(n){
  const mat = new THREE.SpriteMaterial({ map: orbTexture(n.color), transparent:true,
      blending: THREE.NormalBlending, depthWrite:false, opacity:0.96 });
  const s = new THREE.Sprite(mat);
  const base = nodeSize(n) * 2.2; s.scale.set(base, base, 1);
  s.__base = base; s.__mat = mat; n.__obj = s; return s;
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
  .linkWidth(l => isActive(l) ? 2.6 : 1.0)
  .linkOpacity(0.9)
  .linkDirectionalParticles(l => isActive(l) ? 4 : 0)
  .linkDirectionalParticleWidth(2.4)
  .linkDirectionalParticleSpeed(0.006)
  .onNodeClick(n => pinTheme(n.id, true))
  .onNodeHover(n => { hovered = n; elGraph.style.cursor = n ? 'pointer' : 'grab'; refresh(); });

Graph.d3Force('charge').strength(-230);
Graph.d3Force('link').distance(l => 36 + 46 * (1 - (l.strength || 0.5)));

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
function colorLink(l){ const base = REL_COLORS[l.relationship]||REL_COLORS.related;
  return isActive(l) ? base : hexA(base, 0.34); }

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
    const m = o.__mat;
    if (st==='active'){ m.opacity = 1.0; o.__mul = 1.55; }
    else if (st==='neighbor'){ m.opacity = 0.96; o.__mul = 1.15; }
    else if (st==='dim'){ m.opacity = 0.34; o.__mul = 0.9; }
    else { m.opacity = 0.96; o.__mul = 1.0; }
  });
  Graph.linkColor(colorLink).linkWidth(l => isActive(l)?2.6:1.0).linkDirectionalParticles(l => isActive(l)?4:0);
}

// pulse active orbs
let _t0 = performance.now();
(function pulse(){
  const dt = (performance.now() - _t0) / 1000;
  DATA.nodes.forEach(n => { const o = n.__obj; if (!o) return;
    const p = n.__state==='active' ? (1 + 0.12*Math.sin(dt*4)) : 1;
    const s = o.__base * (o.__mul||1) * p; o.scale.set(s, s, 1); });
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
  resetView(); clearTickerCard(); refresh(); renderPins(); renderDetail(null); renderList(); }

// jump to a ticker: deselect everything, highlight the themes it belongs to
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

// ---------- theme toggle ----------
const BG = { light:'#eef2f8', dark:'#070b14' };
let theme = 'dark';
function applyTheme(t){ theme = t; document.body.classList.toggle('dark', t==='dark');
  Graph.backgroundColor(BG[t]); document.getElementById('themebtn').textContent = t==='dark'?'☀ Light':'☾ Dark'; }
document.getElementById('themebtn').onclick = () => applyTheme(theme==='dark'?'light':'dark');

// ---------- sidebar list + search ----------
const elList = document.getElementById('list'), elDetail = document.getElementById('detail'), elCard = document.getElementById('tickercard');
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

  // theme chips + connection rows: click pins, hover previews the node
  [...elDetail.querySelectorAll('.conn[data-id],.chip[data-id]')].forEach(c => { const id = c.dataset.id;
    c.onclick = () => pinTheme(id, true);
    c.onmouseenter = () => { hovered = byId[id]; refresh(); };
    c.onmouseleave = () => { if (hovered===byId[id]){ hovered = null; refresh(); } };
  });
  // ticker chips: click jumps to that ticker's themes, hover previews them
  [...elDetail.querySelectorAll('.chip[data-tk]')].forEach(c => { const tk = c.dataset.tk;
    c.onclick = () => selectTicker(tk);
    c.onmouseenter = () => { previewSet = new Set(themesOf(tk)); refresh(); };
    c.onmouseleave = () => { previewSet = null; refresh(); };
  });
  elDetail.scrollTop = 0;
}

// ---------- starfield (dark mode only ambience; subtle on light) ----------
(function stars(){
  const N = 1400, pos = new Float32Array(N*3);
  for (let i=0;i<N;i++){ const r = 1600 + Math.random()*2600, th = Math.random()*Math.PI*2, ph = Math.acos(2*Math.random()-1);
    pos[i*3]=r*Math.sin(ph)*Math.cos(th); pos[i*3+1]=r*Math.sin(ph)*Math.sin(th); pos[i*3+2]=r*Math.cos(ph); }
  const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos,3));
  Graph.scene().add(new THREE.Points(geo, new THREE.PointsMaterial({ color:0x9fb4e0, size:1.8, sizeAttenuation:false, transparent:true, opacity:0.5 })));
})();

// ---------- custom fly controls ----------
const cam = Graph.camera();
const ctrls = Graph.controls(); ctrls.enabled = false; ctrls.update = function(){};
const dom = Graph.renderer().domElement;
let yaw = 0, pitch = 0, roll = 0, speed = 6, fly = null, interacted = false;
const keys = {};
function defaultCameraDistance(){ return Math.max(420, DATA.n_themes * 7); }
function resetView(){
  fly = null; yaw = 0; pitch = 0; roll = 0;
  cam.position.set(0, 0, defaultCameraDistance());
}
resetView();
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
  '<b>Relationship</b>' + Object.entries(REL_COLORS).filter(([k])=>k!=='related')
    .map(([k,v]) => `<div class="li"><span class="sw" style="background:${v}"></span>${k}</div>`).join('') +
  '<div class="li" style="margin-top:6px;color:var(--dim)">orb size = total market cap · colour = parent group</div>';

applyTheme('dark');
renderList(); renderPins(); elGraph.style.cursor = 'grab';
window.addEventListener('resize', () => Graph.width(elGraph.clientWidth).height(elGraph.clientHeight));
let framed = false;
Graph.onEngineStop(() => { if(!framed && !interacted && !fly && !selected){
  resetView(); } framed = true; });
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
