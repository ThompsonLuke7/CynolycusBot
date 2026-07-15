"""Amethyst options-location dashboard.

Read-only visual surface for daily price bars plus dealer-positioning levels.

Endpoints:
  GET /             -> Amethyst dashboard
  GET /api/state    -> lightweight status
  GET /api/view     -> bars + dealer levels for a symbol
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from UI.ui_chrome import NAV_HTML, THEME_LINK, serve_theme_asset, serve_theme_css

logger = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parents[1]
BARS_1D = REPO / "Data" / "shared" / "bars" / "1d"

GridLoader = Callable[[str, str, float], dict[str, Any]]


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, datetime):
        return obj.astimezone(timezone.utc).isoformat()
    if hasattr(obj, "item"):
        try:
            return _json_safe(obj.item())
        except Exception:
            pass
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass
    return str(obj)


def _clean_symbol(symbol: str) -> str:
    clean = "".join(ch for ch in str(symbol or "").upper().strip() if ch.isalnum() or ch in {".", "-"})
    if not clean:
        raise ValueError("symbol is required")
    return clean


def _clean_scope(scope: str | None) -> str:
    clean = str(scope or "daily_week").strip().lower().replace("-", "_")
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
    return aliases.get(clean, "daily_week")


class AmethystDashboardApp:
    def __init__(
        self,
        *,
        bars_root: Path = BARS_1D,
        grid_loader: GridLoader | None = None,
    ) -> None:
        self.bars_root = Path(bars_root)
        self._grid_loader = grid_loader or self._load_grid_from_dealer
        self._last_view: dict[str, Any] | None = None

    def state(self) -> dict[str, Any]:
        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "state": "ready",
            "bars_root": str(self.bars_root),
            "last_symbol": (self._last_view or {}).get("symbol"),
            "last_error": (self._last_view or {}).get("error"),
        }

    def view(
        self,
        *,
        symbol: str,
        days: int = 180,
        expiration_scope: str = "daily_week",
        window_pct: float = 0.50,
    ) -> dict[str, Any]:
        clean_symbol = _clean_symbol(symbol)
        scope = _clean_scope(expiration_scope)
        days = max(20, min(int(days), 520))
        window_pct = max(0.01, min(float(window_pct), 1.00))
        bars = self._load_daily_bars(clean_symbol, days=days)
        grid: dict[str, Any] = {"rows": [], "levels": {}, "error": None}
        try:
            grid = self._grid_loader(clean_symbol, scope, window_pct)
        except Exception as exc:
            grid = {"rows": [], "levels": {}, "error": str(exc)}
        payload = {
            "symbol": clean_symbol,
            "days": days,
            "expiration_scope": scope,
            "window_pct": window_pct,
            "bars": bars,
            "grid": grid,
            "levels": grid.get("levels") or {},
            "rows": grid.get("rows") or [],
            "error": grid.get("error"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._last_view = payload
        return payload

    def _load_daily_bars(self, symbol: str, *, days: int) -> list[dict[str, Any]]:
        import pandas as pd

        path = self.bars_root / f"{symbol}.parquet"
        if not path.exists():
            return []
        df = pd.read_parquet(path)
        if "timestamp" not in df.columns:
            df = df.reset_index().rename(columns={"index": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
        df = df.drop_duplicates("timestamp").sort_values("timestamp").tail(days)
        keep = ["timestamp", "open", "high", "low", "close", "volume"]
        out = []
        for row in df[[c for c in keep if c in df.columns]].to_dict(orient="records"):
            out.append(
                {
                    "timestamp": row["timestamp"].isoformat(),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume") or 0.0),
                }
            )
        return out

    def _load_grid_from_dealer(self, symbol: str, expiration_scope: str, window_pct: float) -> dict[str, Any]:
        from UI.dealer_positioning_dashboard import DealerDashboardApp

        return DealerDashboardApp().chain_grid(
            symbol=symbol,
            strike_window_pct=window_pct,
            expiration_scope=expiration_scope,
        )


_PAGE = r"""<!doctype html><html><head><meta charset=utf-8><title>Amethyst</title>
__THEME_LINK__
<style>
:root{
  --am-bg:#06070d; --am-panel:#0b0f18; --am-panel2:#111827;
  --am-line:#1d2330; --am-text:#edf2ff; --am-muted:#768095;
  --am-purple:#a855f7; --am-mag:#f7c843; --am-red:#ff3d54; --am-green:#31d17c;
  --am-pink:#ff4fbf; --am-blue:#6ee7ff;
}
body{background:radial-gradient(circle at 58% 42%,rgba(88,28,135,.18),transparent 28%),
  linear-gradient(180deg,#070911,#05060a);color:var(--am-text);overflow:auto}
.shell{display:grid;grid-template-columns:170px minmax(580px,1fr) 470px;height:calc(100vh - 36px);min-height:720px}
.rail{background:#0a0f18;border-right:1px solid #171d29;padding:18px 14px;overflow:hidden}
.rail .brand{display:flex;align-items:center;gap:9px;font-weight:800;letter-spacing:3px;font-size:14px}
.gem{width:24px;height:24px;border-radius:8px;background:linear-gradient(135deg,#2e1065,#a855f7 48%,#f0abfc);
  box-shadow:0 0 22px rgba(168,85,247,.45);transform:rotate(45deg)}
.rail small{display:block;color:#596273;margin:2px 0 28px 34px;letter-spacing:2px;text-transform:uppercase}
.navsec{color:#596273;font-size:10px;letter-spacing:2px;text-transform:uppercase;margin:18px 8px 9px}
.railitem{display:flex;align-items:center;gap:10px;color:#6f788b;padding:10px 11px;border-radius:7px;font-weight:700}
.railitem.active{color:#fff;background:linear-gradient(90deg,rgba(168,85,247,.82),rgba(244,63,94,.46));box-shadow:0 0 22px rgba(168,85,247,.18)}
.main{display:grid;grid-template-rows:auto auto 1fr;background:#070911;min-width:0;min-height:0}
.top{display:flex;align-items:center;gap:10px;padding:8px 14px;border-bottom:1px solid #141a25;background:#0b0f18;flex-wrap:wrap}
input,button,select{background:#101722;color:#e5ecff;border:1px solid #1d2635;border-radius:7px;padding:8px 10px;font:inherit}
button{font-weight:800;cursor:pointer}button.primary{background:linear-gradient(90deg,#7c3aed,#db2777);border:0}
.tickerbox{display:flex;align-items:center;gap:8px;background:#0d131d;border:1px solid #1a2230;border-radius:8px;padding:7px 11px}
.quote{font-weight:900;background:#0f1621;border:1px solid #1d2635;border-radius:8px;padding:9px 13px;min-width:140px}
.quote .chg{font-size:12px;margin-left:8px}.pos{color:var(--am-purple)}.neg{color:var(--am-red)}
.toolbar{margin-left:auto;display:flex;gap:8px;align-items:center}.chip{padding:7px 12px;background:#111827;border:1px solid #1d2635;border-radius:8px;color:#8791a5;font-weight:800}.chip.hot{color:#fff;background:#4c1d95;box-shadow:0 0 16px rgba(168,85,247,.25)}
.scopebar{display:flex;gap:6px;align-items:center;flex-wrap:wrap}.scope-chip{padding:7px 10px;background:#0d131d;border:1px solid #1d2635;border-radius:7px;color:#8190aa;font-weight:900;font-size:12px}.scope-chip.hot{color:#fff;background:linear-gradient(90deg,#5b21b6,#9d174d);box-shadow:0 0 14px rgba(168,85,247,.20)}
.insightbar{border-bottom:1px solid #141a25;background:#080c14;padding:8px 18px;color:#b8c4da;font-size:12px;line-height:1.45}.insightbar strong{color:#fff}.insightbar .warn{color:#ffd866}.insightbar .scope{color:#d8b4fe;font-weight:800}
.stage{position:relative;padding:14px 18px 18px;min-width:0;min-height:0}
.chart-wrap{position:relative;height:100%;border-left:1px solid #111827;border-right:1px solid #111827}
#chart{width:100%;height:100%;display:block;background:linear-gradient(180deg,#070a11,#05070c);cursor:grab;touch-action:none}
.watermark{position:absolute;inset:0;display:grid;place-items:center;pointer-events:none;opacity:.08;font-size:min(12vw,160px);font-weight:900;letter-spacing:14px;color:#c084fc;filter:blur(.2px)}
.legend{position:absolute;top:14px;right:24px;display:flex;gap:13px;color:#788398;font-size:12px;font-weight:800}
.chart-tools{position:absolute;left:16px;top:14px;display:flex;gap:8px;align-items:center}.mini{padding:5px 8px;border-radius:6px;background:#111827;border:1px solid #222b3b;color:#aeb8cc;font-size:11px;font-weight:900}
.readout{position:absolute;left:16px;bottom:14px;background:rgba(13,19,29,.86);border:1px solid #232d3e;border-radius:6px;color:#dbeafe;padding:5px 8px;font:700 11px ui-monospace,monospace;pointer-events:none;min-width:96px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;box-shadow:0 0 10px currentColor}
.side{background:#080b12;border-left:1px solid #171d29;display:grid;grid-template-rows:auto auto minmax(0,1fr);min-width:0;min-height:0}
.side-head{display:flex;align-items:center;gap:10px;padding:10px 14px;border-bottom:1px solid #171d29}
.panel{border-bottom:1px solid #171d29;padding:12px 14px;min-width:0}
.level-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}
.tile{background:#0d131d;border:1px solid #1b2432;border-radius:7px;padding:9px;min-height:56px}
.tile .k{color:#748096;font-size:10px;text-transform:uppercase;letter-spacing:1px}.tile .v{font-size:18px;font-weight:900;margin-top:4px}.tile .s{color:#657086;font-size:11px}
.matrix{overflow-y:auto;overflow-x:hidden;padding:0}.matrix table{border-collapse:collapse;width:100%;table-layout:fixed;font-variant-numeric:tabular-nums;font-size:12px}
.matrix th{position:sticky;top:0;background:#0d131d;color:#687489;text-align:right;padding:8px 6px;border-bottom:1px solid #1a2230;z-index:1;white-space:nowrap}
.matrix td{text-align:right;padding:7px 6px;border-bottom:1px solid #111827;color:#c9d3e8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.matrix th:first-child,.matrix td:first-child{text-align:left;position:sticky;left:0;background:#090d14;z-index:2;width:58px}
.matrix th:nth-child(2),.matrix td:nth-child(2){text-align:center;width:108px;white-space:normal;overflow:visible}
.matrix .spot-row td{box-shadow:inset 0 1px 0 rgba(255,255,255,.45),inset 0 -1px 0 rgba(255,255,255,.45);background:rgba(255,255,255,.06)}
.matrix .spot-row td:first-child::after{content:' SPOT';display:inline-block;margin-left:6px;padding:1px 5px;border-radius:7px;background:#f3f4f6;color:#111827;font-size:9px;font-weight:900}
.matrix .above-spot td:first-child{color:#d8b4fe}.matrix .below-spot td:first-child{color:#ff7384}
.tag{display:block;width:max-content;max-width:96px;margin:2px auto;border-radius:8px;padding:2px 6px;font-size:9px;font-weight:900;text-transform:uppercase;letter-spacing:.5px;line-height:1.15;overflow:hidden;text-overflow:ellipsis}
.t-floor{background:rgba(49,209,124,.18);color:#7bf7ad}.t-ceiling{background:rgba(255,61,84,.18);color:#ff7384}
.t-magnet{background:rgba(247,200,67,.20);color:#ffd866}.t-vega{background:rgba(168,85,247,.22);color:#d8b4fe}.t-flip{background:rgba(255,255,255,.13);color:#fff}
tr.floor td{background:rgba(49,209,124,.10)}tr.ceiling td{background:rgba(255,61,84,.10)}tr.magnet td{background:rgba(247,200,67,.10)}tr.vega td{background:rgba(168,85,247,.12)}
.matrix tr.heat td{border-bottom-color:rgba(255,255,255,.08)}
.status{color:#778297;font-size:12px;min-width:220px}.err{color:#ff7384}
@media(max-width:1100px){.shell{grid-template-columns:1fr;height:auto}.rail{display:none}.side{min-height:620px}.main{min-height:720px}.toolbar{margin-left:0}.stage{height:640px}}
</style></head><body>
__NAV_HTML__
<div class=shell>
  <aside class=rail>
    <div class=brand><span class=gem></span><span>AMETHYST</span></div><small>night vision</small>
    <div class=navsec>Workspace</div>
    <div class="railitem active">Strike Matrix</div>
  </aside>
  <section class=main>
    <div class=top>
      <div class=tickerbox><span style="color:#5f6a7f;font-weight:900">Search</span><input id=symbol value=SPY maxlength=12></div>
      <div class=quote><span id=qSym>SPY</span> <span id=qPx>-</span><span id=qChg class=chg></span></div>
      <button class=primary id=load>Load</button>
      <input id=days type=number min=30 max=520 value=180 title="Daily bars">
      <input id=window type=number min=.01 max=1 step=.01 value=.50 title="Strike window pct">
      <div class=scopebar>
        <button class="scope-chip hot" data-scope=daily_week>Through Week</button>
        <button class=scope-chip data-scope=through_month>Through Month</button>
        <button class=scope-chip data-scope=two_months>2 Months</button>
      </div>
      <div class=toolbar>
        <button class="chip hot" data-mode=gex>GEX</button>
        <button class=chip data-mode=vex>VEX</button>
        <button class=chip data-mode=matrix>Matrix</button>
      </div>
    </div>
    <div id=insight class=insightbar>Load a symbol to summarize dealer levels and price context.</div>
    <div class=stage>
      <div class=chart-wrap><canvas id=chart></canvas><div class=watermark>AMETHYST</div>
      <div class=chart-tools><button class=mini id=resetView>Reset</button></div><div id=chartReadout class=readout>Price -</div>
      <div class=legend><span style="color:var(--am-mag)"><i class=dot></i>Magnet</span><span style="color:var(--am-purple)"><i class=dot></i>Above</span><span style="color:var(--am-red)"><i class=dot></i>Below</span></div></div>
    </div>
  </section>
  <aside class=side>
    <div class=side-head><strong>Strike Matrix</strong><span id=status class=status>ready</span></div>
    <div class=panel><div id=levels class=level-grid></div></div>
    <div class="matrix panel"><table><thead id=matrixHead></thead><tbody id=rows></tbody></table></div>
  </aside>
</div>
<script>
const C={floor:'#31d17c',ceiling:'#ff3d54',magnet:'#f7c843',vega:'#a855f7',flip:'#ffffff',grid:'#141a25',text:'#dce6ff',muted:'#626d81',purple:'#a855f7',red:'#ff3d54'};
let DATA=null, MODE='gex', SCOPE='daily_week', VIEW={auto:true,yMin:null,yMax:null}, LAST_SCALE=null, DRAG=null, HOVER_PRICE=null;
const TAG_HELP={
  floor:'Put wall below spot. Often watched as possible support/liquidity; not a standalone buy signal.',
  'put wall':'Put wall below spot. Often watched as possible support/liquidity; not a standalone buy signal.',
  ceiling:'Call wall above spot. Often watched as possible resistance/supply; not a standalone short signal.',
  'call wall':'Call wall above spot. Often watched as possible resistance/supply; not a standalone short signal.',
  magnet:'High absolute gamma exposure. Price may pin or be pulled toward it when nearby, but direction still needs price confirmation.',
  'magnet above':'Nearest high-gamma magnet above spot.',
  'magnet below':'Nearest high-gamma magnet below spot.',
  'vega wall':'High vega exposure. This is a volatility/hedging pressure zone, especially around IV changes.',
  'vega above':'Nearest high-vega wall above spot.',
  'vega below':'Nearest high-vega wall below spot.',
  'gamma flip':'Approximate strike where net gamma changes sign; hedging behavior can change around this level.'
};
function fmt(n,d=2){return n==null||Number.isNaN(Number(n))?'-':Number(n).toLocaleString(undefined,{maximumFractionDigits:d});}
function money(n){if(n==null||Number.isNaN(Number(n)))return '-';let a=Math.abs(Number(n));let s=Number(n)<0?'-':'';if(a>=1e9)return s+'$'+fmt(a/1e9,1)+'B';if(a>=1e6)return s+'$'+fmt(a/1e6,1)+'M';if(a>=1e3)return s+'$'+fmt(a/1e3,1)+'K';return s+'$'+fmt(a,0);}
function esc(s){return String(s??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));}
function levelPrice(v){if(v==null)return null;let n=Number(v);return Number.isFinite(n)&&n>0?n:null;}
function pctFrom(price,spot){return Number.isFinite(price)&&Number.isFinite(spot)&&spot?((price-spot)/spot*100):null;}
function scopeName(){return ({next_expiration:'Next Expiration',daily_week:'Through Week',through_month:'Through Month',two_months:'2 Months'}[SCOPE]||SCOPE);}
function levelList(levels){
  const specs=[['put_wall','Floor','floor'],['call_wall','Ceiling','ceiling'],['nearest_magnet','Magnet','magnet'],['next_magnet_above','Magnet +','magnet'],['next_magnet_below','Magnet -','magnet'],['vega_wall','Vega Wall','vega'],['gamma_flip','Gamma Flip','flip']];
  let seen=new Set(), out=[]; specs.forEach(s=>{let v=levelPrice(levels[s[0]]); if(v!=null){let k=s[2]+':'+v.toFixed(4); if(!seen.has(k)){seen.add(k); out.push({key:s[0],label:s[1],price:v,kind:s[2]});}}});
  if(MODE==='vex') return out.filter(x=>x.kind==='vega'||x.kind==='flip');
  if(MODE==='gex') return out.filter(x=>x.kind!=='vega');
  return out;
}
async function load(){
  const sym=document.getElementById('symbol').value.trim().toUpperCase()||'SPY';
  const qs=new URLSearchParams({symbol:sym,days:document.getElementById('days').value,scope:SCOPE,window_pct:document.getElementById('window').value});
  document.getElementById('status').textContent='loading '+sym+'...';
  let r=await fetch('/api/view?'+qs.toString(),{cache:'no-store'}); DATA=await r.json(); resetView(false); renderAll();
}
function renderAll(){
  let bars=DATA.bars||[], levels=DATA.levels||{}, rows=DATA.rows||[];
  document.getElementById('qSym').textContent=DATA.symbol||'-';
  let spot=Number(levels.spot), mark=Number.isFinite(spot)?spot:null;
  if(bars.length||mark!=null){let last=bars[bars.length-1]||{}, prev=bars[bars.length-2]||last, price=mark!=null?mark:Number(last.close), base=mark!=null?Number(last.close||price):Number(prev.close||last.close||price), ch=base?(price/base-1)*100:0;document.getElementById('qPx').textContent='$'+fmt(price,2);let e=document.getElementById('qChg');e.textContent=(ch>=0?'+':'')+fmt(ch,2)+'%';e.className='chg '+(ch>=0?'pos':'neg');}
  let grid=DATA.grid||{}, gridErr=grid.error, exp=(grid.expirations||[]).join(', ');
  document.getElementById('status').innerHTML=gridErr?'<span class=err>'+gridErr+'</span>':((rows.length||0)+' strikes'+(exp?' | '+exp:''));
  renderLevels(levels); renderInsight(levels,bars,grid); renderRows(rows); drawChart();
}
function levelTip(name, price, spot){
  let p=levelPrice(price), rel=pctFrom(p,Number(spot)), side=rel==null?'':(rel>=0?' above spot':' below spot')+' ('+(rel>=0?'+':'')+fmt(rel,1)+'%)';
  let base=TAG_HELP[String(name).toLowerCase()]||TAG_HELP[name]||'Dealer-derived level from the selected option expirations.';
  return base+(p!=null?' Strike '+fmt(p,2)+side+'.':'');
}
function tile(k,v,s,cls,tip){return '<div class=tile title="'+esc(tip||'')+'"><div class=k>'+esc(k)+'</div><div class="v '+(cls||'')+'">'+esc(v)+'</div><div class=s>'+esc(s||'')+'</div></div>'}
function renderLevels(x){
  const spot=Number(x.spot);
  document.getElementById('levels').innerHTML=[
    tile('Spot',fmt(x.spot),'chain mark','', 'Current spot/mark from the option chain or quote. The candles are local daily bars, so the last candle can differ intraday.'),
    tile('Floor',fmt(x.put_wall),'put wall','pos',levelTip('floor',x.put_wall,spot)),
    tile('Ceiling',fmt(x.call_wall),'call wall','neg',levelTip('ceiling',x.call_wall,spot)),
    tile('Magnet',fmt(x.nearest_magnet),'nearest pull','',levelTip('magnet',x.nearest_magnet,spot)),
    tile('Gamma Flip',fmt(x.gamma_flip),Number(x.total_gex)>=0?'positive net':'negative net','',levelTip('gamma flip',x.gamma_flip,spot)),
    tile('Vega Wall',fmt(x.vega_wall),'nearest VEX','',levelTip('vega wall',x.vega_wall,spot))
  ].join('');
}
function renderInsight(levels,bars,grid){
  const spot=Number(levels.spot), items=[
    ['Floor',levels.put_wall,'floor'],['Ceiling',levels.call_wall,'ceiling'],['Magnet',levels.nearest_magnet,'magnet'],
    ['Magnet above',levels.next_magnet_above,'magnet'],['Magnet below',levels.next_magnet_below,'magnet'],
    ['Vega wall',levels.vega_wall,'vega wall'],['Vega above',levels.next_vega_wall_above,'vega above'],['Vega below',levels.next_vega_wall_below,'vega below'],
    ['Gamma flip',levels.gamma_flip,'gamma flip']
  ].map(x=>[x[0],levelPrice(x[1]),x[2]]).filter(x=>x[1]!=null);
  if(!Number.isFinite(spot)){document.getElementById('insight').innerHTML='No chain spot yet; dealer levels will summarize after the grid loads.';return;}
  const above=items.filter(x=>Number(x[1])>spot).sort((a,b)=>Number(a[1])-Number(b[1])).slice(0,3);
  const below=items.filter(x=>Number(x[1])<spot).sort((a,b)=>Number(b[1])-Number(a[1])).slice(0,3);
  const same=items.filter(x=>Math.abs(Number(x[1])-spot)<1e-9);
  const clusters={}; items.forEach(x=>{let k=Number(x[1]).toFixed(4);(clusters[k]||(clusters[k]=[])).push(x[0]);});
  const stacked=Object.entries(clusters).filter(([k,v])=>v.length>1).map(([k,v])=>v.join(' + ')+' at '+fmt(k,2)).slice(0,2);
  const phrase=arr=>arr.length?arr.map(x=>x[0]+' '+fmt(x[1],2)+' ('+(pctFrom(Number(x[1]),spot)>=0?'+':'')+fmt(pctFrom(Number(x[1]),spot),1)+'%)').join('; '):'none nearby';
  const latest=bars&&bars.length?bars[bars.length-1]:null, latestClose=latest?Number(latest.close):null;
  let stale='';
  if(latest&&Number.isFinite(latestClose)){let diff=pctFrom(spot,latestClose); stale=' <span class=warn>Candles are daily bars: latest close '+fmt(latestClose,2)+' vs chain spot '+fmt(spot,2)+' ('+(diff>=0?'+':'')+fmt(diff,1)+'%).</span>';}
  let bias='Watch confirmation: ';
  if(below.length&&(!above.length||Math.abs(spot-Number(below[0][1]))<Math.abs(Number(above[0][1])-spot))) bias+='nearest pressure is below, so weakness can target '+below[0][0].toLowerCase()+' '+fmt(below[0][1],2)+'.';
  else if(above.length) bias+='nearest pressure is above, so strength can target '+above[0][0].toLowerCase()+' '+fmt(above[0][1],2)+'.';
  else bias+='no nearby dealer level stands out in this scope.';
  document.getElementById('insight').innerHTML='<strong>'+esc(DATA.symbol||'')+'</strong> <span class=scope>'+esc(scopeName())+'</span>: spot '+fmt(spot,2)+'. Above: '+esc(phrase(above))+'. Below: '+esc(phrase(below))+'. '+esc(bias)+' '+(stacked.length?'Stacked levels: '+esc(stacked.join('; '))+'. ':'')+'Levels are context, not a forecast.'+stale;
}
function tagHtml(tags){let all=(tags||[]);return all.map(raw=>{let t=String(raw);return '<span title="'+esc(TAG_HELP[t.toLowerCase()]||'Dealer level tag at this strike.')+'" class="tag t-'+(t.includes('floor')?'floor':t.includes('ceiling')||t.includes('call wall')?'ceiling':t.includes('magnet')?'magnet':t.includes('vega')?'vega':'flip')+'">'+esc(t)+'</span>';}).join('');}
function nearestSpotStrike(rows, spot){
  if(!rows.length||!Number.isFinite(Number(spot))) return null;
  return rows.reduce((best,r)=>Math.abs(Number(r.strike)-spot)<Math.abs(Number(best.strike)-spot)?r:best, rows[0]).strike;
}
function matrixColumns(){
  if(MODE==='vex') return [
    ['strike','Strike',r=>fmt(r.strike,2)],
    ['tags','Tags',r=>tagHtml(r.tags)],
    ['total_vex','Total VEX',r=>money(r.total_vex)],
    ['call_vex','Call VEX',r=>money(r.call_vex)],
    ['put_vex','Put VEX',r=>money(r.put_vex)],
    ['oi','OI',r=>fmt((Number(r.call_oi)||0)+(Number(r.put_oi)||0),0)]
  ];
  if(MODE==='matrix') return [
    ['strike','Strike',r=>fmt(r.strike,2)],
    ['tags','Tags',r=>tagHtml(r.tags)],
    ['net_gex','Net GEX',r=>money(r.net_gex)],
    ['total_abs_gex','Total GEX',r=>money(r.total_abs_gex)],
    ['total_vex','VEX',r=>money(r.total_vex)],
    ['call_oi','Call OI',r=>fmt(r.call_oi,0)],
    ['put_oi','Put OI',r=>fmt(r.put_oi,0)]
  ];
  return [
    ['strike','Strike',r=>fmt(r.strike,2)],
    ['tags','Tags',r=>tagHtml(r.tags)],
    ['net_gex','Net GEX',r=>money(r.net_gex)],
    ['total_abs_gex','Total GEX',r=>money(r.total_abs_gex)],
    ['call_gex','Call GEX',r=>money(r.call_gex)],
    ['put_gex','Put GEX',r=>money(r.put_gex)]
  ];
}
function heatMetric(r){
  if(MODE==='vex') return Math.abs(Number(r.total_vex)||0);
  if(MODE==='matrix') return Math.abs(Number(r.total_abs_gex)||0);
  return Math.abs(Number(r.net_gex)||0);
}
function heatStyle(r,maxMag){
  const mag=maxMag?Math.min(1,Math.max(0,heatMetric(r)/maxMag)):0;
  const a=mag>0?(0.018+0.56*Math.pow(mag,0.82)).toFixed(3):'0';
  const net=Number(r.net_gex)||0;
  const color=MODE==='vex'?'168,85,247':(net<0?'255,61,84':'168,85,247');
  return 'background-color:rgba('+color+','+a+');';
}
function renderRows(rows){
  rows=(rows||[]).slice().sort((a,b)=>Number(b.strike)-Number(a.strike));
  const spot=Number((DATA.levels||{}).spot), spotStrike=nearestSpotStrike(rows,spot), cols=matrixColumns();
  const maxMag=Math.max(0,...rows.map(heatMetric));
  document.getElementById('matrixHead').innerHTML='<tr>'+cols.map(c=>'<th>'+c[1]+'</th>').join('')+'</tr>';
  document.getElementById('rows').innerHTML=rows.map(r=>{
    let cls=['heat',r.zone||'', Number(r.strike)>spot?'above-spot':'below-spot', Number(r.strike)===Number(spotStrike)?'spot-row':''].join(' ');
    let tip=(r.tags||[]).length?' title="'+esc('Strike '+fmt(r.strike,2)+': '+(r.tags||[]).join(', '))+'"':'';
    let style=heatStyle(r,maxMag);
    return '<tr class="'+cls+'"'+tip+' data-spot="'+(Number(r.strike)===Number(spotStrike)?'1':'0')+'">'+cols.map(c=>'<td style="'+style+'">'+c[2](r)+'</td>').join('')+'</tr>';
  }).join('');
  setTimeout(()=>{let el=document.querySelector('#rows tr[data-spot="1"]'); if(el) el.scrollIntoView({block:'center'});}, 0);
}
function drawChart(){
  const canvas=document.getElementById('chart'), box=canvas.getBoundingClientRect(), dpr=window.devicePixelRatio||1;
  canvas.width=Math.max(600,box.width*dpr); canvas.height=Math.max(420,box.height*dpr); const ctx=canvas.getContext('2d'); ctx.scale(dpr,dpr);
  let w=box.width,h=box.height,bars=(DATA&&DATA.bars)||[], levels=levelList((DATA&&DATA.levels)||{}); ctx.clearRect(0,0,w,h);
  ctx.fillStyle='#070911';ctx.fillRect(0,0,w,h); if(!bars.length){LAST_SCALE=null;ctx.fillStyle=C.muted;ctx.fillText('No local daily bars found for '+((DATA&&DATA.symbol)||''),30,40);return;}
  const pad={l:42,r:96,t:34,b:34}; let vals=[]; bars.forEach(b=>{vals.push(b.high,b.low)}); levels.forEach(l=>vals.push(l.price)); let spot=Number((DATA.levels||{}).spot); if(Number.isFinite(spot)) vals.push(spot); let autoLo=Math.min(...vals), autoHi=Math.max(...vals), span=Math.max(.01,autoHi-autoLo); autoLo-=span*.08; autoHi+=span*.08;
  let lo=VIEW.auto||!Number.isFinite(Number(VIEW.yMin))||!Number.isFinite(Number(VIEW.yMax))?autoLo:Number(VIEW.yMin), hi=VIEW.auto||!Number.isFinite(Number(VIEW.yMin))||!Number.isFinite(Number(VIEW.yMax))?autoHi:Number(VIEW.yMax); if(hi<=lo){hi=lo+.01;} LAST_SCALE={pad,w,h,lo,hi};
  const x=i=>pad.l+i*Math.max(2,(w-pad.l-pad.r)/(Math.max(1,bars.length-1))); const y=v=>pad.t+(hi-v)/(hi-lo)*(h-pad.t-pad.b);
  ctx.strokeStyle=C.grid;ctx.lineWidth=1;ctx.font='12px ui-monospace,monospace';ctx.fillStyle=C.muted;
  for(let i=0;i<6;i++){let yy=pad.t+i*(h-pad.t-pad.b)/5;ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(w-pad.r,yy);ctx.stroke();let val=hi-i*(hi-lo)/5;ctx.fillText(fmt(val,2),w-pad.r+12,yy+4);}
  spreadLevels(levels,y,pad,h).forEach(l=>drawLevel(ctx,l,pad,w,y,l.drawY));
  const cw=Math.max(3,Math.min(9,(w-pad.l-pad.r)/bars.length*.58));
  bars.forEach((b,i)=>{let xx=x(i), up=b.close>=b.open, col=up?C.purple||'#a855f7':C.red||'#ff3d54';ctx.strokeStyle=col;ctx.fillStyle=col;ctx.globalAlpha=.9;ctx.beginPath();ctx.moveTo(xx,y(b.high));ctx.lineTo(xx,y(b.low));ctx.stroke();let yo=y(Math.max(b.open,b.close)), yc=y(Math.min(b.open,b.close));ctx.fillRect(xx-cw/2,yo,cw,Math.max(2,yc-yo));ctx.globalAlpha=1;});
  spot=(DATA.levels||{}).spot||bars[bars.length-1].close; ctx.strokeStyle='rgba(255,255,255,.45)';ctx.setLineDash([4,5]);ctx.beginPath();ctx.moveTo(pad.l,y(spot));ctx.lineTo(w-pad.r,y(spot));ctx.stroke();ctx.setLineDash([]);label(ctx,'SPOT '+fmt(spot,2),w-pad.r+18,y(spot),'#f3f4f6','#111827');drawHoverPriceLine(ctx,pad,w,h,y);document.getElementById('chartReadout').textContent=Number.isFinite(Number(HOVER_PRICE))?'Y '+fmt(HOVER_PRICE,2):'Price -';
}
function spreadLevels(levels,y,pad,h){
  const groups={}; levels.forEach(l=>{let k=Number(l.price).toFixed(4);(groups[k]||(groups[k]=[])).push(l);});
  let out=[]; Object.values(groups).forEach(group=>{let n=group.length, gap=14; group.forEach((l,i)=>{let yy=y(l.price)+(i-(n-1)/2)*gap; yy=Math.max(pad.t+8,Math.min(h-pad.b-8,yy)); out.push({...l,drawY:yy});});});
  return out;
}
function drawLevel(ctx,l,pad,w,y,yyOverride){
  let col=C[l.kind]||C.magnet, yy=Number.isFinite(yyOverride)?yyOverride:y(l.price);ctx.save();ctx.shadowColor=col;ctx.shadowBlur=l.kind==='magnet'?18:10;ctx.strokeStyle=col;ctx.lineWidth=l.kind==='magnet'?4:2;ctx.globalAlpha=l.kind==='flip'?.65:.85;ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(w-pad.r,yy);ctx.stroke();ctx.shadowBlur=0;if(l.kind==='magnet'){for(let x=pad.l+18;x<w-pad.r;x+=42){ctx.fillStyle=col;ctx.beginPath();ctx.arc(x,yy,2.4,0,Math.PI*2);ctx.fill();}}ctx.restore();label(ctx,l.label+' '+fmt(l.price,2),w-pad.r+18,yy,col,'#0b0f18');
}
function drawHoverPriceLine(ctx,pad,w,h,y){
  const price=Number(HOVER_PRICE);
  if(!Number.isFinite(price)) return;
  const yy=y(price);
  if(yy<pad.t||yy>h-pad.b) return;
  ctx.save();
  ctx.strokeStyle='rgba(216,180,254,.82)';
  ctx.lineWidth=1;
  ctx.setLineDash([6,6]);
  ctx.beginPath();
  ctx.moveTo(pad.l,yy);
  ctx.lineTo(w-pad.r,yy);
  ctx.stroke();
  ctx.setLineDash([]);
  label(ctx,fmt(price,2),w-pad.r+18,yy,'#d8b4fe','#0b0f18');
  ctx.restore();
}
function label(ctx,text,x,y,fg,bg){ctx.font='700 11px ui-monospace,monospace';let m=ctx.measureText(text),hh=19;ctx.fillStyle=fg;ctx.shadowColor=fg;ctx.shadowBlur=8;roundRect(ctx,x,y-hh/2,m.width+12,hh,5);ctx.fill();ctx.shadowBlur=0;ctx.fillStyle=bg;ctx.fillText(text,x+6,y+4);}
function roundRect(ctx,x,y,w,h,r){ctx.beginPath();ctx.moveTo(x+r,y);ctx.arcTo(x+w,y,x+w,y+h,r);ctx.arcTo(x+w,y+h,x,y+h,r);ctx.arcTo(x,y+h,x,y,r);ctx.arcTo(x,y,x+w,y,r);ctx.closePath();}
function setMode(mode){
  MODE=mode;
  document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('hot',b.dataset.mode===MODE));
  if(DATA) renderAll();
}
function setScope(scope){
  SCOPE=scope;
  document.querySelectorAll('[data-scope]').forEach(b=>b.classList.toggle('hot',b.dataset.scope===SCOPE));
  load();
}
function priceAt(clientY){
  if(!LAST_SCALE) return null;
  const rect=document.getElementById('chart').getBoundingClientRect(), py=clientY-rect.top, s=LAST_SCALE, span=s.h-s.pad.t-s.pad.b;
  return s.hi-(py-s.pad.t)/span*(s.hi-s.lo);
}
function resetView(redraw=true){VIEW={auto:true,yMin:null,yMax:null}; if(redraw&&DATA)drawChart();}
function bindChartInteractions(){
  const canvas=document.getElementById('chart'), readout=document.getElementById('chartReadout');
  canvas.addEventListener('wheel',e=>{
    if(!DATA||!LAST_SCALE)return; e.preventDefault();
    const anchor=priceAt(e.clientY); if(anchor==null)return;
    const factor=Math.exp(e.deltaY*0.0012), lo=LAST_SCALE.lo, hi=LAST_SCALE.hi;
    VIEW={auto:false,yMin:anchor-(anchor-lo)*factor,yMax:anchor+(hi-anchor)*factor}; drawChart();
  },{passive:false});
  canvas.addEventListener('mousedown',e=>{if(!LAST_SCALE)return;DRAG={y:e.clientY,lo:LAST_SCALE.lo,hi:LAST_SCALE.hi};canvas.style.cursor='grabbing';});
  window.addEventListener('mouseup',()=>{DRAG=null;canvas.style.cursor='grab';});
  window.addEventListener('mousemove',e=>{
    if(DRAG&&LAST_SCALE){let spanPx=LAST_SCALE.h-LAST_SCALE.pad.t-LAST_SCALE.pad.b, pp=(DRAG.hi-DRAG.lo)/spanPx, delta=(e.clientY-DRAG.y)*pp;VIEW={auto:false,yMin:DRAG.lo+delta,yMax:DRAG.hi+delta};drawChart();return;}
    let p=priceAt(e.clientY); if(p!=null){HOVER_PRICE=p;if(readout)readout.textContent='Y '+fmt(p,2);drawChart();}
  });
  canvas.addEventListener('mouseleave',()=>{if(!DRAG){HOVER_PRICE=null;if(readout)readout.textContent='Price -';drawChart();}});
}
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>setMode(b.dataset.mode));
document.querySelectorAll('[data-scope]').forEach(b=>b.onclick=()=>setScope(b.dataset.scope));
document.getElementById('resetView').onclick=()=>resetView(true);
document.getElementById('load').onclick=load;document.getElementById('symbol').addEventListener('keydown',e=>{if(e.key==='Enter')load();});window.addEventListener('resize',()=>DATA&&drawChart());bindChartInteractions();load();
</script></body></html>""".replace("__THEME_LINK__", THEME_LINK).replace("__NAV_HTML__", NAV_HTML)


class AmethystHTTPServer(ThreadingHTTPServer):
    app: AmethystDashboardApp


class AmethystHandler(BaseHTTPRequestHandler):
    server_version = "CynolycusAmethyst/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A002
        if self.path.startswith("/api/"):
            return
        super().log_message(fmt, *args)

    def _app(self) -> AmethystDashboardApp:
        return self.server.app  # type: ignore[attr-defined]

    def _send(self, body: bytes, status=HTTPStatus.OK, ctype="application/json; charset=utf-8") -> None:
        self.send_response(int(status))
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Any, status=HTTPStatus.OK) -> None:
        self._send(json.dumps(_json_safe(payload), allow_nan=False).encode("utf-8"), status=status)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html", "/amethyst"}:
            self._send(_PAGE.encode("utf-8"), ctype="text/html; charset=utf-8")
            return
        if parsed.path == "/static/cynolycus_theme.css":
            serve_theme_css(self)
            return
        if parsed.path.startswith("/static/themes/"):
            serve_theme_asset(self, parsed.path)
            return
        if parsed.path == "/api/state":
            self._send_json(self._app().state())
            return
        if parsed.path == "/api/view":
            params = parse_qs(parsed.query)
            try:
                payload = self._app().view(
                    symbol=(params.get("symbol") or ["SPY"])[0],
                    days=int((params.get("days") or ["180"])[0]),
                    expiration_scope=(params.get("scope") or ["daily_week"])[0],
                    window_pct=float((params.get("window_pct") or ["0.50"])[0]),
                )
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(payload)
            return
        self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)


def make_server(host: str, port: int, app: AmethystDashboardApp) -> AmethystHTTPServer:
    srv = AmethystHTTPServer((host, port), AmethystHandler)
    srv.daemon_threads = True
    srv.app = app
    return srv


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    parser = argparse.ArgumentParser(description="Run the Amethyst dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8772)
    args = parser.parse_args()

    server = make_server(args.host, args.port, AmethystDashboardApp())
    logger.info("Amethyst dashboard: http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
