"""Cynolycus hub — a single overview screen for the combined-server dashboards.

Serves on its own port (default 8764) and links to / monitors / controls every
dashboard. Each module card shows a paper/real-money indicator + its own
real-money toggle (the account choice is per-module — there is no confusing
single global switch). The header shows aggregated account stats.

  GET  /                 → HTML overview (auto-refreshing cards + Start All)
  GET  /api/state        → fan-out: each dashboard's status + account, fetched
                           server-side (avoids cross-port CORS in the browser)
  POST /api/start        → start one dashboard ({"key": "swing", "live": false})
  POST /api/stop         → stop one dashboard ({"key": "swing"})
  POST /api/start-all    → start every startable dashboard
                           ({"live_map": {"swing": false, ...}})

Orders default to the paper account; real money only when a card's toggle is on.
"""
from __future__ import annotations

import json
import logging
import math
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from UI.ui_chrome import NAV_HTML, THEME_LINK, serve_theme_css
from UI.performance import module_performance

logger = logging.getLogger(__name__)
# Snapshot reads must stay snappy — a slow module should not stall the hub page.
_TIMEOUT = 2.5
# Control POSTs (start/stop) can legitimately block for a while: the swing
# session, for one, warms up ~900 tickers synchronously before /api/start
# returns. Using the read timeout here reported those slow-but-successful starts
# as "timed out" and dropped the socket mid-response (BrokenPipe on the module).
_CONTROL_TIMEOUT = 120.0


class _Dash:
    """Descriptor for one downstream dashboard."""

    def __init__(self, key: str, name: str, port: int, *, startable: bool,
                 stoppable: bool, tradeable: bool, start_path: str | None,
                 start_body: Callable[[bool], dict], adapt: Callable[[dict], dict]) -> None:
        self.key = key
        self.name = name
        self.port = port
        self.startable = startable
        self.stoppable = stoppable
        self.tradeable = tradeable  # has a paper/real-money account choice
        self.start_path = start_path
        self.start_body = start_body
        self.adapt = adapt


def _truthy(*vals: Any) -> bool:
    return any(bool(v) for v in vals)


def _f(v: Any) -> float | None:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "item"):
        try:
            return _json_safe(obj.item())
        except Exception:  # noqa: BLE001
            pass
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:  # noqa: BLE001
            return str(obj)
    return str(obj)


def _positions_upl(positions: list[dict]) -> float:
    tot = 0.0
    for p in positions or []:
        tot += _f(p.get("upl")) or _f(p.get("unrealized_pl")) or 0.0
    return tot


# ---- per-dashboard /api/state adapters -> {state, detail, account_type?, acct?} ----
def _adapt_swing(s: dict) -> dict:
    st = s.get("status") or {}
    alive = _truthy(s.get("session_alive"))
    state = st.get("state") or ("running" if alive else "idle")
    pos = s.get("positions") or []
    out = {"state": state, "detail": f"{len(pos)} open · {st.get('universe_size', 0)} universe",
           "live_available": bool(s.get("live_available")),
           "account_type": "real money" if st.get("real_account_policy") else "paper"}
    if pos:
        out["acct"] = {"n_positions": len(pos), "upl": _positions_upl(pos)}
    return out


def _adapt_spy(s: dict) -> dict:
    st = s.get("status") or {}
    alive = _truthy(s.get("session_alive"))
    state = st.get("state") or s.get("status_message") or ("running" if alive else "idle")
    cfg = s.get("config") or {}
    broker = s.get("broker_state") or {}
    pos = broker.get("positions") or []
    out = {"state": str(state), "detail": f"mode {cfg.get('runner_mode', '-')}",
           "live_available": True, "account_type": "paper"}
    eq = _f(broker.get("equity") or broker.get("account_equity"))
    if eq is not None or pos:
        out["acct"] = {"equity": eq, "n_positions": len(pos), "upl": _positions_upl(pos)}
    return out


def _adapt_dealer(s: dict) -> dict:
    state = s.get("state") or ("running" if _truthy(s.get("session_alive")) else "idle")
    return {"state": str(state), "detail": "analytics"}


def _adapt_dealer_ranker(s: dict) -> dict:
    cfg = s.get("config") or {}
    acct = s.get("account") or {}
    scan = s.get("scan") or {}
    state = "running" if _truthy(s.get("loop_running")) else "ready"
    out = {"state": state, "detail": f"{len(scan.get('picks', []))} ranks · {cfg.get('workers', '-')} workers",
           "live_available": bool(cfg.get("live_available")),
           "account_type": "real money" if cfg.get("live") else "paper"}
    if not acct.get("error"):
        out["acct"] = {"equity": _f(acct.get("equity")),
                       "n_positions": acct.get("n_positions"),
                       "upl": _positions_upl(acct.get("positions") or []),
                       "account_n_positions": acct.get("account_n_positions"),
                       "account_upl": _f(acct.get("account_upl"))}
    return out


def _adapt_amethyst(s: dict) -> dict:
    state = s.get("state") or "ready"
    last = s.get("last_symbol") or "-"
    err = s.get("last_error")
    detail = f"night vision · last {last}"
    if err:
        detail = f"{detail} · {err}"
    return {"state": str(state), "detail": detail}


def _adapt_meta(s: dict) -> dict:
    cfg = s.get("config") or {}
    acct = s.get("account") or {}
    state = "running" if _truthy(s.get("loop_running")) else "ready"
    out = {"state": state, "detail": f"{cfg.get('mode', 'equity')} · {acct.get('n_positions', 0)} positions",
           "live_available": bool(cfg.get("live_available")),
           "account_type": "real money" if cfg.get("live") else "paper"}
    if not acct.get("error"):
        out["acct"] = {"equity": _f(acct.get("equity")),
                       "n_positions": acct.get("n_positions"),
                       "upl": _positions_upl(acct.get("positions") or []),
                       "account_n_positions": acct.get("account_n_positions"),
                       "account_upl": _f(acct.get("account_upl"))}
    return out


def _adapt_momentum(s: dict) -> dict:
    cfg = s.get("config") or {}
    acct = s.get("account") or {}
    scan = s.get("scan") or {}
    state = "running" if _truthy(s.get("loop_running")) else "ready"
    out = {"state": state, "detail": f"{len(scan.get('picks', []))} signals",
           "live_available": bool(cfg.get("live_available")),
           "account_type": "real money" if cfg.get("live") else "paper"}
    if not acct.get("error"):
        out["acct"] = {"equity": _f(acct.get("equity")),
                       "n_positions": acct.get("n_positions"),
                       "upl": _positions_upl(acct.get("positions") or []),
                       "account_n_positions": acct.get("account_n_positions"),
                       "account_upl": _f(acct.get("account_upl"))}
    return out


def _adapt_htf(s: dict) -> dict:
    scan = s.get("scan") or {}
    age = scan.get("age_days")
    return {"state": "ready" if scan.get("picks") else "idle",
            "detail": f"{len(scan.get('picks', []))} scored · {age}d old" if age is not None
            else f"{len(scan.get('picks', []))} scored"}


def _adapt_intraday_structure(s: dict) -> dict:
    active = len(s.get("active_signals") or [])
    candidates = int(s.get("candidate_count") or 0)
    return {
        "state": "running" if s.get("running") else "idle",
        "detail": f"{active} active setups · {candidates} candidates · paper-only",
    }


class HubDashboardApp:
    def __init__(self, *, host: str = "127.0.0.1", port_spy: int = 8765,
                 port_swing: int = 8766, port_dealer: int = 8768,
                 port_meta: int = 8769, port_momentum: int = 8770,
                 port_htf: int = 8771, port_amethyst: int = 8772,
                 port_dealer_ranker: int = 8773,
                 port_intraday_structure: int | None = None) -> None:
        self.host = host
        self.dashboards: list[_Dash] = [
            _Dash("spy", "SPY Intraday", port_spy, startable=True, stoppable=True, tradeable=True,
                  start_path="/api/start",
                  start_body=lambda live: {"runner_mode": "live", "live": live,
                                           "enable_option_orders": True, "simulate_orders": False},
                  adapt=_adapt_spy),
            _Dash("swing", "Swing", port_swing, startable=True, stoppable=True, tradeable=True,
                  start_path="/api/start",
                  start_body=lambda live: {"max_entries": 5, "live": live},
                  adapt=_adapt_swing),
            _Dash("htf", "HTF Swing", port_htf, startable=False, stoppable=False, tradeable=False,
                  start_path=None, start_body=lambda live: {}, adapt=_adapt_htf),
            _Dash("momentum", "Momentum", port_momentum, startable=True, stoppable=False, tradeable=True,
                  start_path="/api/run-loop", start_body=lambda live: {}, adapt=_adapt_momentum),
            _Dash("amethyst", "Amethyst", port_amethyst, startable=False, stoppable=False,
                  tradeable=False, start_path=None, start_body=lambda live: {}, adapt=_adapt_amethyst),
            _Dash("dealer", "Dealer Positioning", port_dealer, startable=True, stoppable=True,
                  tradeable=False, start_path="/api/start", start_body=lambda live: {}, adapt=_adapt_dealer),
            _Dash("dealer_ranker", "Dealer Ranker", port_dealer_ranker, startable=True, stoppable=False,
                  tradeable=True, start_path="/api/run-loop", start_body=lambda live: {},
                  adapt=_adapt_dealer_ranker),
            _Dash("meta", "Meta Ranker", port_meta, startable=True, stoppable=False, tradeable=True,
                  start_path="/api/run-loop", start_body=lambda live: {}, adapt=_adapt_meta),
        ]
        if port_intraday_structure is not None:
            self.dashboards.append(
                _Dash("intraday_structure", "Intraday Structure", port_intraday_structure,
                      startable=False, stoppable=False, tradeable=False,
                      start_path=None, start_body=lambda live: {}, adapt=_adapt_intraday_structure)
            )

    def _by_key(self, key: str) -> _Dash:
        for d in self.dashboards:
            if d.key == key:
                return d
        raise KeyError(f"unknown dashboard: {key}")

    def _request(self, port: int, path: str, *, method: str = "GET",
                 body: dict | None = None, timeout: float = _TIMEOUT) -> dict:
        url = f"http://{self.host}:{port}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        return json.loads(raw or b"{}")

    # ---- read ----
    def snapshot(self) -> dict:
        cards = []
        equity: float | None = None
        attributed_positions = 0   # sum of each module's OWN book
        attributed_upl = 0.0
        broker_positions: int | None = None   # true shared-account totals (once)
        broker_upl: float | None = None

        # Fan out to every dashboard CONCURRENTLY. Done sequentially this blocked
        # the hub's own /api/state for up to (2.5s × N) — long enough that the
        # page rendered blank — and one slow module stalled all the others.
        def _fetch(d: _Dash) -> tuple[str, dict | None, str | None]:
            try:
                return d.key, self._request(d.port, "/api/state"), None
            except Exception as exc:  # noqa: BLE001
                return d.key, None, str(exc)

        with ThreadPoolExecutor(max_workers=max(1, len(self.dashboards))) as ex:
            results = {k: (s, e) for k, s, e in ex.map(_fetch, self.dashboards)}

        for d in self.dashboards:
            entry = {"key": d.key, "name": d.name,
                     "url": f"http://{self.host}:{d.port}/", "port": d.port,
                     "startable": d.startable, "stoppable": d.stoppable,
                     "tradeable": d.tradeable, "up": False, "state": "down",
                     "detail": "", "live_available": False,
                     "account_type": None, "error": None}
            s, err = results.get(d.key, (None, "no_result"))
            if err is not None or s is None:
                entry["error"] = err or "unreachable"
            else:
                try:
                    entry.update(d.adapt(s))
                    acct = entry.get("acct") or {}
                    entry["performance"] = module_performance(
                        d.key if d.key != "htf" else "multi_ticker_swing_htf",
                        open_upl=_f(acct.get("upl")),
                    )
                    entry["up"] = True
                    acct = entry.pop("acct", None)
                    if acct:
                        if acct.get("equity") is not None and equity is None:
                            equity = acct["equity"]  # shared account → take first reported
                        attributed_positions += int(acct.get("n_positions") or 0)
                        attributed_upl += _f(acct.get("upl")) or 0.0
                        # Broker-wide totals describe the shared account, so take them
                        # once (first module that reports them) instead of summing.
                        if broker_positions is None and acct.get("account_n_positions") is not None:
                            broker_positions = int(acct.get("account_n_positions") or 0)
                            broker_upl = _f(acct.get("account_upl")) or 0.0
                except Exception as exc:  # noqa: BLE001
                    entry["error"] = str(exc)
            cards.append(entry)
        return {"dashboards": cards,
                "totals": {"equity": equity,
                           "open_positions": attributed_positions,
                           "unrealized_pl": round(attributed_upl, 2),
                           "account_positions": broker_positions,
                           "account_unrealized_pl": (round(broker_upl, 2)
                                                     if broker_upl is not None else None)}}

    # ---- control ----
    def start_one(self, key: str, live: bool) -> dict:
        d = self._by_key(key)
        if not d.startable or d.start_path is None:
            return {"key": key, "ok": False, "error": "not_startable"}
        if d.key in ("meta", "momentum", "dealer_ranker"):  # set the account before firing the loop
            try:
                self._request(d.port, "/api/set-live", method="POST", body={"live": live},
                              timeout=_CONTROL_TIMEOUT)
            except Exception as exc:  # noqa: BLE001
                return {"key": key, "ok": False, "error": f"set-live: {exc}"}
        try:
            res = self._request(d.port, d.start_path, method="POST", body=d.start_body(bool(live)),
                                timeout=_CONTROL_TIMEOUT)
            return {"key": key, "ok": True, "result": res}
        except Exception as exc:  # noqa: BLE001
            return {"key": key, "ok": False, "error": str(exc)}

    def stop_one(self, key: str) -> dict:
        d = self._by_key(key)
        if not d.stoppable:
            return {"key": key, "ok": False, "error": "not_stoppable"}
        try:
            res = self._request(d.port, "/api/stop", method="POST", body={}, timeout=_CONTROL_TIMEOUT)
            return {"key": key, "ok": True, "result": res}
        except Exception as exc:  # noqa: BLE001
            return {"key": key, "ok": False, "error": str(exc)}

    def start_all(self, live_map: dict) -> dict:
        live_map = live_map or {}
        # Start All is for continuous live sessions only. Meta/Momentum /api/run-loop
        # is a one-shot 4H order pass, so launching it at boot/open can trade stale
        # cached signals and kick off expensive work. Scheduled passes handle those.
        startable = [d for d in self.dashboards if d.startable and d.key not in ("meta", "momentum", "dealer_ranker")]
        skipped = [
            {"key": d.key, "ok": True, "skipped": True, "reason": "scheduled_4h_loop"}
            for d in self.dashboards
            if d.startable and d.key in ("meta", "momentum", "dealer_ranker")
        ]
        return {"results": [self.start_one(d.key, bool(live_map.get(d.key))) for d in startable] + skipped}


_PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Cynolycus Hub</title>
__THEME_LINK__
<style>
.wrap{margin:16px 18px}
h1{font-size:18px;margin:0 0 4px}
.totals{display:flex;gap:18px;flex-wrap:wrap;margin:8px 0 4px;font-size:13px}
.totals b{font-size:16px}
.bar{display:flex;align-items:center;gap:12px;margin:10px 0 16px;flex-wrap:wrap}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:14px}
.card .body{padding:12px;display:grid;gap:10px}
.row{display:flex;align-items:center;justify-content:space-between;gap:8px}
.detail{color:var(--muted);font-size:12px}
.tog{display:inline-flex;align-items:center;gap:6px;color:var(--muted);font-size:12px}
.warn{color:var(--yellow);font-size:12px}
.acct{font-size:12px}
.perf{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;font-size:11px}
.perf span{background:var(--panel2);padding:6px;border-radius:5px;text-align:center;color:var(--muted)}
.perf b{display:block;font-size:13px;color:var(--text)}
</style></head><body>
__NAV_HTML__
<div class=wrap>
<h1>Cynolycus Overview</h1>
<div class=detail>Each module submits to the PAPER account unless its own "real money" toggle is on.</div>
<div class=totals id=totals></div>
<div class=bar>
  <button class=primary onclick=startAll()>▶ Start All</button>
  <span id=msg class=muted></span>
</div>
<div class=grid id=grid></div>
</div>
<script>
var liveState={};  // per-card real-money intent, persisted in localStorage
try{liveState=JSON.parse(localStorage.getItem('cyno-hub-live')||'{}');}catch(e){}
var staticDashboards=[
  {key:'spy',name:'SPY Intraday',url:'http://'+location.hostname+':8765/',startable:true,stoppable:true,tradeable:true,up:false,state:'down',detail:'waiting for state'},
  {key:'swing',name:'Swing',url:'http://'+location.hostname+':8766/',startable:true,stoppable:true,tradeable:true,up:false,state:'down',detail:'waiting for state'},
  {key:'htf',name:'HTF Swing',url:'http://'+location.hostname+':8771/',startable:false,stoppable:false,tradeable:false,up:false,state:'down',detail:'waiting for state'},
  {key:'momentum',name:'Momentum',url:'http://'+location.hostname+':8770/',startable:true,stoppable:false,tradeable:true,up:false,state:'down',detail:'waiting for state'},
  {key:'amethyst',name:'Amethyst',url:'http://'+location.hostname+':8772/',startable:false,stoppable:false,tradeable:false,up:false,state:'down',detail:'waiting for state'},
  {key:'dealer',name:'Dealer Positioning',url:'http://'+location.hostname+':8768/',startable:true,stoppable:true,tradeable:false,up:false,state:'down',detail:'waiting for state'},
  {key:'dealer_ranker',name:'Dealer Ranker',url:'http://'+location.hostname+':8773/',startable:true,stoppable:false,tradeable:true,up:false,state:'down',detail:'waiting for state'},
  {key:'meta',name:'Meta Ranker',url:'http://'+location.hostname+':8769/',startable:true,stoppable:false,tradeable:true,up:false,state:'down',detail:'waiting for state'}
];
function setLive(key,v){liveState[key]=v;localStorage.setItem('cyno-hub-live',JSON.stringify(liveState));tick();}
function f(n,d){if(n==null)return '-';return Number(n).toLocaleString(undefined,{maximumFractionDigits:(d==null?2:d)});}
function render(s){
  let t=s.totals||{};
  var acctPos=(t.account_positions==null)?'':' <span class=muted>/ '+f(t.account_positions,0)+' in account</span>';
  document.getElementById('totals').innerHTML=
    '<div>Account equity (shared) <b>$'+f(t.equity,0)+'</b></div>'+
    '<div>Open positions (attributed) <b>'+f(t.open_positions,0)+'</b>'+acctPos+'</div>'+
    '<div>Unrealized P/L (attributed) <b class="'+((t.unrealized_pl||0)>=0?'pos':'neg')+'">$'+f(t.unrealized_pl)+'</b></div>';
  let g=document.getElementById('grid');
  g.innerHTML=(s.dashboards||[]).map(function(d){
    var intend=!!liveState[d.key];
    var acctType=d.tradeable?(d.account_type||(intend?'real money':'paper')):null;
    var indicator=acctType?('<span class="pill '+(acctType==='real money'?'live':'paper')+'">'+acctType+'</span>'):'';
    var pillState=d.up?(d.state||'idle'):'error';
    var toggle=d.tradeable?('<label class=tog title="Off: paper. On: real-money account.">'+
       '<input type=checkbox '+(intend?'checked':'')+' '+(d.live_available?'':'disabled')+
       ' onchange="setLive(\\''+d.key+'\\',this.checked)"> real money</label>'+
       (intend&&!d.live_available?'<div class=warn>real money not configured — runs paper</div>':''):'';
    var startBtn=d.startable?('<button class=primary onclick="ctl(\\''+d.key+'\\',\\'start\\')">'+
       (d.key==='meta'||d.key==='momentum'||d.key==='dealer_ranker'?'Run':'Start')+'</button>'):'';
    var stopBtn=d.stoppable?('<button class=danger onclick="ctl(\\''+d.key+'\\',\\'stop\\')">Stop</button>'):'';
    var p=d.performance||{};
    var tracked=p.closed_trades>0;
    var perf='<div class=perf><span><b class="'+((p.tracked_pnl||0)>=0?'pos':'neg')+'">'+(tracked?'$'+f(p.tracked_pnl):'—')+'</b>tracked P/L</span>'+
      '<span><b>'+f(p.win_rate,1)+(p.win_rate==null?'':'%')+'</b>win rate</span>'+
      '<span><b>'+f(p.closed_trades,0)+'</b>closed</span></div>';
    return '<div class=card><div class=card-head>'+d.name+'</div><div class=body>'+
      '<div class=row><span class="pill '+pillState+'">'+(d.up?(d.state||'idle'):'down')+'</span>'+
      indicator+'<a href="'+d.url+'" target=_blank>open ↗</a></div>'+
      '<div class=detail>'+(d.up?(d.detail||''):(d.error||'unreachable'))+'</div>'+
      perf+
      toggle+
      '<div class=row>'+startBtn+stopBtn+'</div>'+
      '</div></div>';
  }).join('');
}
async function tick(){
  let s;
  try{
    let resp=await fetch('/api/state');
    if(!resp.ok) throw new Error('HTTP '+resp.status);
    s=await resp.json();
  }
  catch(e){
    document.getElementById('msg').textContent='hub state unavailable, retrying...';
    render({dashboards:staticDashboards,totals:{}});
    return;
  }
  document.getElementById('msg').textContent='';
  render(s);
}
async function ctl(key,action){
  document.getElementById('msg').textContent=action+' '+key+'...';
  var live=!!liveState[key];
  if(action==='start'&&live&&!confirm('Start '+key+' on the REAL-MONEY account?')){
    document.getElementById('msg').textContent='';return;}
  var body=action==='start'?{key:key,live:live}:{key:key};
  let r=await(await fetch('/api/'+action,{method:'POST',body:JSON.stringify(body)})).json();
  document.getElementById('msg').textContent=(r.ok?'ok: ':'failed: ')+key+(r.error?(' — '+r.error):'');
  tick();
}
async function startAll(){
  var anyLive=Object.keys(liveState).some(function(k){return liveState[k];});
  if(anyLive&&!confirm('Start All — some modules are set to REAL MONEY. Continue?'))return;
  document.getElementById('msg').textContent='starting all...';
  let r=await(await fetch('/api/start-all',{method:'POST',body:JSON.stringify({live_map:liveState})})).json();
  let bad=(r.results||[]).filter(function(x){return !x.ok;});
  document.getElementById('msg').textContent=bad.length?('issues: '+bad.map(function(x){return x.key+'('+x.error+')';}).join(', ')):'all started';
  tick();
}
tick();setInterval(tick,5000);
</script></body></html>""".replace("__THEME_LINK__", THEME_LINK).replace("__NAV_HTML__", NAV_HTML)


class HubHTTPServer(ThreadingHTTPServer):
    app: HubDashboardApp


class HubHandler(BaseHTTPRequestHandler):
    server_version = "CynolycusHub/1.0"

    def log_message(self, fmt, *a):  # noqa: A002
        return

    def _app(self) -> HubDashboardApp:
        return self.server.app  # type: ignore[attr-defined]

    def _send(self, body: bytes, status=HTTPStatus.OK, ctype="application/json; charset=utf-8"):
        self.send_response(int(status))
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b""
        return json.loads(raw or b"{}")

    def do_GET(self):  # noqa: N802
        if self.path == "/" or self.path.startswith("/index"):
            self._send(_PAGE.encode("utf-8"), ctype="text/html; charset=utf-8")
        elif self.path == "/static/cynolycus_theme.css":
            serve_theme_css(self)
        elif self.path.startswith("/api/state"):
            self._send_json(self._app().snapshot())
        else:
            self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self):  # noqa: N802
        app = self._app()
        try:
            payload = self._body()
        except Exception as exc:  # noqa: BLE001
            self._send_json({"ok": False, "error": f"bad_json: {exc}"},
                            status=HTTPStatus.BAD_REQUEST)
            return
        if self.path.startswith("/api/start-all"):
            res = app.start_all(payload.get("live_map") or {})
        elif self.path.startswith("/api/start"):
            res = app.start_one(str(payload.get("key")), bool(payload.get("live")))
        elif self.path.startswith("/api/stop"):
            res = app.stop_one(str(payload.get("key")))
        else:
            self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
            return
        self._send_json(res)

    def _send_json(self, payload: Any, status=HTTPStatus.OK) -> None:
        self._send(
            json.dumps(_json_safe(payload), allow_nan=False).encode("utf-8"),
            status=status,
        )


def make_server(host: str, port: int, app: HubDashboardApp) -> HubHTTPServer:
    srv = HubHTTPServer((host, port), HubHandler)
    srv.daemon_threads = True
    srv.app = app
    return srv
