"""Dealer Ranker dashboard — near-close dealer ranking + ATM options pass.

Serves on its own port and is wired into the combined server. Manual runs submit
to the paper account by default; the real-money account requires this module's
own live toggle.
"""
from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient
from UI.ui_chrome import NAV_HTML, THEME_LINK, serve_theme_css

logger = logging.getLogger(__name__)
RANKING_ROOT = REPO / "Data/dealer_positioning/rankings"
RANKINGS = REPO / "Data/dealer_positioning/rankings/dealer_swing_rankings_latest.parquet"
RANKING_HISTORY = REPO / "Data/dealer_positioning/rankings/dealer_swing_rankings_history.parquet"
STATE = REPO / "Data/inference/dealer_ranker/live_state.json"
RUNNER = REPO / "strategies/dealer_positioning/live_ranked_options.py"
SCAN_TTL = 30.0
ACCT_TTL = 8.0
RANKING_TTL = 30.0


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


def _ranking_path_date(path: Path) -> str | None:
    stem = path.stem.replace("dealer_swing_rankings_", "")
    if stem in {"latest", "history"} or len(stem) != 8 or not stem.isdigit():
        return None
    return f"{stem[:4]}-{stem[4:6]}-{stem[6:]}"


def _ranking_paths(root: Path = RANKING_ROOT) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted(Path(root).glob("dealer_swing_rankings_*.parquet")):
        date_key = _ranking_path_date(path)
        if date_key:
            out[date_key] = path
    return out


def _coerce_date_key(value: Any) -> str:
    try:
        return pd.Timestamp(value).date().isoformat()
    except Exception:
        return str(value)


def _as_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def _as_float(value: Any, digits: int | None = None) -> float | None:
    try:
        if pd.isna(value):
            return None
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return round(out, digits) if digits is not None else out


def _row_value(row: pd.Series, key: str, default: Any = None) -> Any:
    return row[key] if key in row.index else default


def _ranking_row(row: pd.Series, *, rank_col: str, score_col: str, direction_col: str | None = None) -> dict:
    return {
        "rank": _as_int(_row_value(row, rank_col)),
        "ticker": str(_row_value(row, "symbol", "")),
        "score": _as_float(_row_value(row, score_col), 2),
        "direction": str(_row_value(row, direction_col, "-")) if direction_col else "-",
        "structural_rank": _as_int(_row_value(row, "dealer_swing_rank")),
        "shift_rank": _as_int(_row_value(row, "dealer_change_intensity_rank")),
        "bullish_shift_rank": _as_int(_row_value(row, "dealer_change_bullish_rank")),
        "bearish_shift_rank": _as_int(_row_value(row, "dealer_change_bearish_rank")),
        "shift_bias": _as_float(_row_value(row, "dealer_change_direction_bias"), 3),
        "gamma_flip_change": _as_float(_row_value(row, "avg_gamma_flip_change_1d"), 2),
        "call_wall_change": _as_float(_row_value(row, "avg_call_wall_change"), 2),
        "put_wall_change": _as_float(_row_value(row, "avg_put_wall_change"), 2),
        "magnet_change": _as_float(_row_value(row, "avg_magnet_change"), 2),
    }


def _top_rows(frame: pd.DataFrame, *, rank_col: str, score_col: str, direction_col: str | None, top: int) -> list[dict]:
    if frame.empty or rank_col not in frame.columns:
        return []
    ranked = frame.sort_values(rank_col).head(max(1, int(top)))
    return [_ranking_row(row, rank_col=rank_col, score_col=score_col, direction_col=direction_col) for _, row in ranked.iterrows()]


def _cross_day_rows(current: pd.DataFrame, previous: pd.DataFrame, *, top: int) -> list[dict]:
    if current.empty or previous.empty:
        return []
    cur_cols = [
        "symbol",
        "dealer_swing_rank",
        "dealer_swing_potential_score",
        "dealer_change_intensity_rank",
        "dealer_change_intensity_score",
        "dealer_change_bullish_rank",
        "dealer_change_bearish_rank",
        "dealer_change_direction",
        "dealer_change_direction_bias",
    ]
    keep_cur = [c for c in cur_cols if c in current.columns]
    keep_prev = [c for c in cur_cols if c in previous.columns]
    merged = current[keep_cur].merge(previous[keep_prev], on="symbol", how="left", suffixes=("", "_prev"))
    for col in ("dealer_swing_rank", "dealer_change_intensity_rank"):
        if col in merged.columns and f"{col}_prev" in merged.columns:
            merged[f"{col}_improvement"] = pd.to_numeric(merged[f"{col}_prev"], errors="coerce") - pd.to_numeric(merged[col], errors="coerce")
    if "dealer_swing_potential_score" in merged.columns and "dealer_swing_potential_score_prev" in merged.columns:
        merged["dealer_swing_score_delta"] = pd.to_numeric(merged["dealer_swing_potential_score"], errors="coerce") - pd.to_numeric(
            merged["dealer_swing_potential_score_prev"], errors="coerce"
        )
    if "dealer_change_intensity_score" in merged.columns and "dealer_change_intensity_score_prev" in merged.columns:
        merged["dealer_shift_score_delta"] = pd.to_numeric(merged["dealer_change_intensity_score"], errors="coerce") - pd.to_numeric(
            merged["dealer_change_intensity_score_prev"], errors="coerce"
        )

    if "dealer_swing_rank_improvement" not in merged.columns:
        return []
    movers = merged.dropna(subset=["dealer_swing_rank_improvement"]).copy()
    movers = movers.sort_values(["dealer_swing_rank_improvement", "dealer_shift_score_delta"], ascending=[False, False]).head(max(1, int(top)))
    rows = []
    for _, row in movers.iterrows():
        rows.append(
            {
                "ticker": str(row["symbol"]),
                "structural_rank": _as_int(_row_value(row, "dealer_swing_rank")),
                "prev_structural_rank": _as_int(_row_value(row, "dealer_swing_rank_prev")),
                "structural_rank_improvement": _as_float(_row_value(row, "dealer_swing_rank_improvement"), 0),
                "structural_score": _as_float(_row_value(row, "dealer_swing_potential_score"), 2),
                "structural_score_delta": _as_float(_row_value(row, "dealer_swing_score_delta"), 2),
                "shift_rank": _as_int(_row_value(row, "dealer_change_intensity_rank")),
                "prev_shift_rank": _as_int(_row_value(row, "dealer_change_intensity_rank_prev")),
                "shift_rank_improvement": _as_float(_row_value(row, "dealer_change_intensity_rank_improvement"), 0),
                "shift_score": _as_float(_row_value(row, "dealer_change_intensity_score"), 2),
                "shift_score_delta": _as_float(_row_value(row, "dealer_shift_score_delta"), 2),
                "shift_direction": str(_row_value(row, "dealer_change_direction", "-")),
                "shift_bias": _as_float(_row_value(row, "dealer_change_direction_bias"), 3),
            }
        )
    return rows


class DealerRankerDashboardApp:
    def __init__(
        self,
        *,
        env_file: str = ".env#PAPER",
        live_env_file: str | None = None,
        top_k: int = 10,
        workers: int = 8,
        contracts: int = 1,
    ) -> None:
        self.paper_env_file = env_file
        self.live_env_file = live_env_file
        self.top_k = int(top_k)
        self.workers = int(workers)
        self.contracts = int(contracts)
        self._live = False
        self._client = AlpacaOptionsClient(env_file=self.env_file)
        self._scan_cache: tuple[float, dict, float | None] | None = None
        self._ranking_cache: tuple[float, dict, tuple[tuple[str, float | None], ...]] | None = None
        self._broker_cache: tuple[float, tuple[dict, list]] | None = None
        self._lock = threading.Lock()
        self._running = False

    @property
    def env_file(self) -> str:
        return self.live_env_file if (self._live and self.live_env_file) else self.paper_env_file

    def set_live(self, live: bool) -> dict:
        want = bool(live) and bool(self.live_env_file)
        if want != self._live:
            self._live = want
            self._client = AlpacaOptionsClient(env_file=self.env_file)
            self._broker_cache = None
        return {"live": self._live, "available": bool(self.live_env_file)}

    def _state_file(self) -> dict:
        try:
            return json.loads(STATE.read_text())
        except Exception:
            return {"managed": {}, "history": []}

    def _scan(self) -> dict:
        try:
            mtime = RANKINGS.stat().st_mtime
        except OSError:
            mtime = None
        if self._scan_cache and self._scan_cache[2] == mtime and time.time() - self._scan_cache[0] < SCAN_TTL:
            return self._scan_cache[1]
        try:
            frame = pd.read_parquet(RANKINGS).sort_values("dealer_swing_rank").head(self.top_k)
            out = {
                "ranking_path": str(RANKINGS),
                "mtime": datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat() if mtime else None,
                "picks": [
                    {
                        "rank": int(r.dealer_swing_rank),
                        "ticker": r.symbol,
                        "score": round(float(r.dealer_swing_potential_score), 2),
                        "direction": str(getattr(r, "dealer_direction", "-")),
                        "vacuum": round(float(getattr(r, "avg_vacuum_component", 0.0)), 3),
                        "pinning": round(float(getattr(r, "avg_pinning_room_component", 0.0)), 3),
                        "max_scope": round(float(getattr(r, "max_scope_score", 0.0)), 3),
                    }
                    for _, r in frame.iterrows()
                ],
            }
        except Exception as exc:  # noqa: BLE001
            out = {"error": f"ranking_scan_failed: {exc}", "picks": []}
        self._scan_cache = (time.time(), out, mtime)
        return out

    def _ranking_dates(self) -> dict:
        paths = _ranking_paths()
        dates = sorted(paths)
        latest = dates[-1] if dates else None
        mtimes = {
            date_key: datetime.fromtimestamp(paths[date_key].stat().st_mtime, tz=timezone.utc).isoformat()
            for date_key in dates
            if paths[date_key].exists()
        }
        return {
            "dates": dates,
            "latest_date": latest,
            "history_path": str(RANKING_HISTORY),
            "history_exists": RANKING_HISTORY.exists(),
            "mtimes": mtimes,
            "names": {
                "structural": "Structural Setup",
                "shift": "Structure Shift",
                "bullish_shift": "Bullish Structure Shift",
                "bearish_shift": "Bearish Structure Shift",
            },
        }

    def _load_rankings_for_date(self, date_key: str | None) -> tuple[str | None, pd.DataFrame, Path | None]:
        paths = _ranking_paths()
        if not paths:
            return None, pd.DataFrame(), None
        selected = date_key if date_key in paths else sorted(paths)[-1]
        frame = pd.read_parquet(paths[selected])
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        return selected, frame, paths[selected]

    def rankings_view(self, *, date_key: str | None = None, compare_date: str | None = None, top: int = 15) -> dict:
        top = max(1, min(50, int(top)))
        signature_items: list[tuple[str, float | None]] = []
        for key, path in _ranking_paths().items():
            try:
                signature_items.append((key, path.stat().st_mtime))
            except OSError:
                signature_items.append((key, None))
        signature = tuple(signature_items)
        cache_key = {
            "date": date_key,
            "compare_date": compare_date,
            "top": top,
            "signature": signature,
        }
        if (
            self._ranking_cache
            and time.time() - self._ranking_cache[0] < RANKING_TTL
            and self._ranking_cache[1].get("_cache_key") == cache_key
        ):
            return {k: v for k, v in self._ranking_cache[1].items() if k != "_cache_key"}

        dates_meta = self._ranking_dates()
        selected, frame, path = self._load_rankings_for_date(date_key)
        dates = dates_meta["dates"]
        compare = compare_date if compare_date in dates else None
        if selected and not compare:
            prior = [d for d in dates if d < selected]
            compare = prior[-1] if prior else None
        _, compare_frame, compare_path = self._load_rankings_for_date(compare) if compare else (None, pd.DataFrame(), None)

        out = {
            **dates_meta,
            "selected_date": selected,
            "compare_date": compare,
            "ranking_path": str(path) if path else None,
            "compare_path": str(compare_path) if compare_path else None,
            "top": top,
            "structural": _top_rows(
                frame,
                rank_col="dealer_swing_rank",
                score_col="dealer_swing_potential_score",
                direction_col="dealer_direction",
                top=top,
            ),
            "structure_shift": _top_rows(
                frame,
                rank_col="dealer_change_intensity_rank",
                score_col="dealer_change_intensity_score",
                direction_col="dealer_change_direction",
                top=top,
            ),
            "bullish_shift": _top_rows(
                frame,
                rank_col="dealer_change_bullish_rank",
                score_col="dealer_change_direction_bias",
                direction_col="dealer_change_direction",
                top=top,
            ),
            "bearish_shift": _top_rows(
                frame,
                rank_col="dealer_change_bearish_rank",
                score_col="dealer_change_direction_bias",
                direction_col="dealer_change_direction",
                top=top,
            ),
            "cross_day_movers": _cross_day_rows(frame, compare_frame, top=top) if compare else [],
        }
        cached = dict(out)
        cached["_cache_key"] = cache_key
        self._ranking_cache = (time.time(), cached, signature)
        return out

    def _fetch_broker(self) -> tuple[dict, list]:
        now = time.time()
        if self._broker_cache and now - self._broker_cache[0] < ACCT_TTL:
            return self._broker_cache[1]
        account_raw = self._client.get_account()
        positions_raw = self._client.get_positions() or []
        account = {
            "equity": float(account_raw.get("equity", 0) or 0),
            "cash": float(account_raw.get("cash", 0) or 0),
            "buying_power": float(account_raw.get("buying_power", 0) or 0),
        }
        positions = [
            {
                "symbol": p["symbol"],
                "qty": float(p["qty"]),
                "avg_entry": float(p.get("avg_entry_price", 0) or 0),
                "current": float(p.get("current_price", 0) or 0),
                "mv": float(p.get("market_value", 0) or 0),
                "upl": float(p.get("unrealized_pl", 0) or 0),
                "upl_pct": round(float(p.get("unrealized_plpc", 0) or 0) * 100, 2),
            }
            for p in positions_raw
        ]
        self._broker_cache = (now, (account, positions))
        return account, positions

    def _account(self, managed_state: dict) -> dict:
        symbols = set()
        for ticker, st in (managed_state or {}).items():
            symbols.add(str(ticker).upper())
            if isinstance(st, dict):
                symbols.update(str(st.get(k, "")).upper() for k in ("occ", "symbol") if st.get(k))
        try:
            account, positions = self._fetch_broker()
            book = [p for p in positions if str(p["symbol"]).upper() in symbols]
            return {
                "equity": account["equity"],
                "cash": account["cash"],
                "buying_power": account["buying_power"],
                "n_positions": len(book),
                "positions": sorted(book, key=lambda x: -x["upl"]),
                "account_n_positions": len(positions),
                "account_upl": round(sum(p["upl"] for p in positions), 2),
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": f"account_failed: {exc}", "positions": []}

    def snapshot(self) -> dict:
        state = self._state_file()
        managed = state.get("managed", {})
        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "config": {
                "env": self.env_file,
                "top_k": self.top_k,
                "workers": self.workers,
                "contracts": self.contracts,
                "live": self._live,
                "live_available": bool(self.live_env_file),
            },
            "scan": self._scan(),
            "ranking_dates": self._ranking_dates(),
            "account": self._account(managed),
            "managed": sorted(managed.keys()),
            "history": list(reversed(state.get("history", [])))[:20],
            "loop_running": self._running,
        }

    def run_loop(self, payload: dict | None = None) -> dict:
        payload = payload or {}
        top_k = int(payload.get("top_k") or self.top_k)
        workers = int(payload.get("workers") or self.workers)
        contracts = int(payload.get("contracts") or self.contracts)
        refresh_chain = bool(payload.get("refresh_chain", True))
        with self._lock:
            if self._running:
                return {"started": False, "reason": "already_running"}
            self._running = True

        def _go() -> None:
            argv = [
                sys.executable,
                str(RUNNER),
                "--top-k",
                str(top_k),
                "--contracts",
                str(contracts),
                "--workers",
                str(workers),
                "--submit",
            ]
            if refresh_chain:
                argv.append("--refresh-chain")
            if self._live:
                argv.append("--live")
            try:
                subprocess.run(argv, cwd=str(REPO), env={**os.environ, "PYTHONPATH": str(REPO)})
            except Exception as exc:  # noqa: BLE001
                logger.error("dealer ranker loop failed: %s", exc)
            finally:
                self._scan_cache = None
                self._broker_cache = None
                self._running = False

        threading.Thread(target=_go, daemon=True, name="dealer-ranker-loop").start()
        return {
            "started": True,
            "live": self._live,
            "top_k": top_k,
            "workers": workers,
            "contracts": contracts,
            "refresh_chain": refresh_chain,
        }


_PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Dealer Ranker</title>
__THEME_LINK__
<style>
.wrap{margin:14px 18px}
h2{margin:14px 0 6px;font-size:15px}
table{border-collapse:collapse;width:100%;margin:8px 0}
td,th{border:1px solid var(--border);padding:4px 8px;text-align:right}th{background:var(--panel2)}
td:first-child,th:first-child,td.t,th.t{text-align:left}
.controls{margin:6px 0}
.tog{display:inline-flex;align-items:center;gap:6px;color:var(--muted);font-size:12px}
.rank-grid{display:grid;grid-template-columns:repeat(2,minmax(360px,1fr));gap:14px;align-items:start}
@media(max-width:980px){.rank-grid{grid-template-columns:1fr}}
.mini{font-size:11px;color:var(--muted)}
</style></head><body>
__NAV_HTML__
<div class=wrap>
<h2>Dealer Ranker <span id=cfg class=pill></span> <span id=age class=muted></span></h2>
<div class=controls>
  <button class=primary onclick=runLoop()>Run scan + SUBMIT</button>
  <label>Top <input id=topk type=number min=1 max=25 value=10 style="width:64px"></label>
  <label>Workers <input id=workers type=number min=1 max=24 value=8 style="width:64px"></label>
  <label>Contracts <input id=contracts type=number min=1 max=50 value=1 style="width:64px"></label>
  <label class=tog><input type=checkbox id=refresh checked> refresh chains</label>
  <label class=tog title="Off: submit on PAPER. On: real-money account.">
    <input type=checkbox id=live-toggle onchange=setLive(this.checked)> Real money account</label>
  <span id=msg class=muted></span>
</div>
<h2>Account</h2><div id=acct></div>
<h2>Execution Top Ranks</h2><table id=scan></table>
<h2>Daily Ranking Research <span id=rankMeta class=muted></span></h2>
<div class=controls>
  <label>Date <select id=rankDate onchange=loadRankings()></select></label>
  <label>Compare <select id=compareDate onchange=loadRankings()></select></label>
  <label>Rows <input id=rankRows type=number min=5 max=50 value=15 style="width:64px" onchange=loadRankings()"></label>
</div>
<div class=rank-grid>
  <div><h2>Structural Setup</h2><table id=structRank></table></div>
  <div><h2>Structure Shift</h2><table id=shiftRank></table></div>
  <div><h2>Bullish Structure Shift</h2><table id=bullRank></table></div>
  <div><h2>Bearish Structure Shift</h2><table id=bearRank></table></div>
</div>
<h2>Cross-Day Movers</h2><table id=moversRank></table>
<h2>Open positions</h2><table id=pos></table>
<h2>Recent runs</h2><div id=hist class=muted></div>
</div>
<script>
function f(n,d=2){return n==null?'-':Number(n).toLocaleString(undefined,{maximumFractionDigits:d})}
let rankDatesKey='';
function esc(x){return String(x==null?'-':x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function optHtml(dates,selected){return (dates||[]).map(d=>'<option value="'+esc(d)+'" '+(d===selected?'selected':'')+'>'+esc(d)+'</option>').join('')}
function renderRankTable(id,rows,mode){
let head='<tr><th>#</th><th class=t>ticker</th><th>score</th><th>dir</th><th>struct #</th><th>shift #</th><th>bias</th></tr>';
document.getElementById(id).innerHTML=head+(rows||[]).map(p=>'<tr><td>'+f(p.rank,0)+'</td><td class=t>'+esc(p.ticker)+'</td><td>'+f(p.score,2)+'</td><td>'+esc(p.direction)+'</td><td>'+f(p.structural_rank,0)+'</td><td>'+f(p.shift_rank,0)+'</td><td>'+f(p.shift_bias,3)+'</td></tr>').join('');
}
function renderMovers(rows){
let head='<tr><th class=t>ticker</th><th>setup #</th><th>prev</th><th>Δrank</th><th>Δscore</th><th>shift #</th><th>Δshift</th><th>shift dir</th></tr>';
document.getElementById('moversRank').innerHTML=head+(rows||[]).map(p=>'<tr><td class=t>'+esc(p.ticker)+'</td><td>'+f(p.structural_rank,0)+'</td><td>'+f(p.prev_structural_rank,0)+'</td><td>'+f(p.structural_rank_improvement,0)+'</td><td>'+f(p.structural_score_delta,2)+'</td><td>'+f(p.shift_rank,0)+'</td><td>'+f(p.shift_rank_improvement,0)+'</td><td>'+esc(p.shift_direction)+'</td></tr>').join('');
}
async function loadRankings(){
let d=document.getElementById('rankDate').value;
let c=document.getElementById('compareDate').value;
let top=Number(document.getElementById('rankRows').value||15);
let q='?top='+encodeURIComponent(top)+(d?'&date='+encodeURIComponent(d):'')+(c?'&compare='+encodeURIComponent(c):'');
let r=await(await fetch('/api/rankings'+q)).json();
document.getElementById('rankMeta').textContent=(r.selected_date||'-')+(r.compare_date?' vs '+r.compare_date:'')+' · history '+(r.history_exists?'saved':'missing');
if((r.dates||[]).join('|')!==rankDatesKey){
  rankDatesKey=(r.dates||[]).join('|');
  document.getElementById('rankDate').innerHTML=optHtml(r.dates,r.selected_date);
  document.getElementById('compareDate').innerHTML='<option value="">previous</option>'+optHtml(r.dates,r.compare_date);
}
renderRankTable('structRank',r.structural,'structural');
renderRankTable('shiftRank',r.structure_shift,'shift');
renderRankTable('bullRank',r.bullish_shift,'bull');
renderRankTable('bearRank',r.bearish_shift,'bear');
renderMovers(r.cross_day_movers);
}
async function tick(){let s=await(await fetch('/api/state')).json();
let live=!!s.config.live;
document.getElementById('cfg').textContent=(live?'real money':'paper')+' · '+s.config.contracts+'x';
document.getElementById('cfg').className='pill '+(live?'live':'paper');
let lt=document.getElementById('live-toggle');lt.checked=live;lt.disabled=!s.config.live_available;
let badge=document.getElementById('cyno-live-badge');
if(badge){badge.textContent=live?'real money':'paper';badge.className='live-badge'+(live?' live':'');}
document.getElementById('age').textContent='rank file '+(s.scan.mtime||'-')+(s.loop_running?' · running':'');
let a=s.account;document.getElementById('acct').innerHTML=a.error?('<span class=neg>'+a.error+'</span>'):
('equity $'+f(a.equity,0)+' · cash $'+f(a.cash,0)+' · BP $'+f(a.buying_power,0)+' · '+a.n_positions+' dealer-ranker positions');
document.getElementById('scan').innerHTML='<tr><th>#</th><th class=t>ticker</th><th>score</th><th>dir</th><th>vacuum</th><th>pinning</th><th>max scope</th></tr>'+
(s.scan.picks||[]).map(p=>'<tr><td>'+p.rank+'</td><td class=t>'+p.ticker+'</td><td>'+p.score+'</td><td>'+p.direction+'</td><td>'+p.vacuum+'</td><td>'+p.pinning+'</td><td>'+p.max_scope+'</td></tr>').join('');
document.getElementById('pos').innerHTML='<tr><th class=t>symbol</th><th>qty</th><th>entry</th><th>current</th><th>mkt val</th><th>unreal P/L</th><th>%</th></tr>'+
(a.positions||[]).map(p=>{let c=p.upl>=0?'pos':'neg';return '<tr><td class=t>'+p.symbol+'</td><td>'+f(p.qty,0)+'</td><td>'+f(p.avg_entry)+'</td><td>'+f(p.current)+'</td><td>$'+f(p.mv,0)+'</td><td class='+c+'>$'+f(p.upl,0)+'</td><td class='+c+'>'+p.upl_pct+'%</td></tr>'}).join('');
document.getElementById('hist').innerHTML=(s.history||[]).map(h=>h.ts+' — '+(h.profile||'paper')+' — '+(h.orders||0)+' orders').join('<br>')||'none yet';}
async function setLive(v){await fetch('/api/set-live',{method:'POST',body:JSON.stringify({live:v})});tick();}
async function runLoop(){document.getElementById('msg').textContent='launching...';
let body={top_k:Number(document.getElementById('topk').value||10),workers:Number(document.getElementById('workers').value||8),contracts:Number(document.getElementById('contracts').value||1),refresh_chain:document.getElementById('refresh').checked};
let r=await(await fetch('/api/run-loop',{method:'POST',body:JSON.stringify(body)})).json();
document.getElementById('msg').textContent=r.started?('running ('+(r.live?'real money':'paper')+')'):('skipped: '+r.reason);}
tick();loadRankings();setInterval(tick,5000);
</script></body></html>""".replace("__THEME_LINK__", THEME_LINK).replace("__NAV_HTML__", NAV_HTML)


class DealerRankerHTTPServer(ThreadingHTTPServer):
    app: DealerRankerDashboardApp


class DealerRankerHandler(BaseHTTPRequestHandler):
    server_version = "CynolycusDealerRanker/1.0"

    def log_message(self, fmt, *a):  # noqa: A002
        return

    def _app(self) -> DealerRankerDashboardApp:
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
            self._send(json.dumps(_json_safe(self._app().snapshot())).encode("utf-8"))
        elif self.path.startswith("/api/rankings"):
            query = parse_qs(urlparse(self.path).query)
            date_key = (query.get("date") or [None])[0]
            compare_date = (query.get("compare") or [None])[0]
            try:
                top = int((query.get("top") or ["15"])[0])
            except ValueError:
                top = 15
            payload = self._app().rankings_view(date_key=date_key, compare_date=compare_date, top=top)
            self._send(json.dumps(_json_safe(payload)).encode("utf-8"))
        else:
            self._send(b'{"error":"not_found"}', status=HTTPStatus.NOT_FOUND)

    def do_POST(self):  # noqa: N802
        if self.path.startswith("/api/set-live"):
            res = self._app().set_live(bool(self._body().get("live")))
            self._send(json.dumps(res).encode("utf-8"))
            return
        if self.path.startswith("/api/run-loop"):
            res = self._app().run_loop(self._body())
            self._send(json.dumps(res).encode("utf-8"))
            return
        self._send(b'{"error":"not_found"}', status=HTTPStatus.NOT_FOUND)


def make_server(host: str, port: int, app: DealerRankerDashboardApp) -> DealerRankerHTTPServer:
    srv = DealerRankerHTTPServer((host, port), DealerRankerHandler)
    srv.daemon_threads = True
    srv.app = app
    return srv
