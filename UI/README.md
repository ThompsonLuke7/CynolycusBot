# Live / Replay Dashboard

Run the dashboard server:

```powershell
python -m UI.live_dashboard --host 127.0.0.1 --port 8765
```

Open:

`http://127.0.0.1:8765`

Use the `Start` button to launch a session. The UI supports:

- `Mode = live`: runs the live runner stream
- `Mode = replay`: replays historical 1m bars through the same inference path

Replay options:

- set `Replay Data Path` (CSV/Parquet 1m bars)
- optional `Replay Start` / `Replay End`
- optional `Replay Sleep` to slow playback
- optional `Replay Max Bars` to cap run length

When `Mode = replay` and option orders are enabled, orders are forced to simulated payloads (no broker submission).

The UI streams:

- live 1m bars + 15m closes
- agent actions + agent state
- option policy state
- broker positions + recent orders (when option orders are enabled)

Execution filter defaults:

- entry confirmation: `1` consecutive 15m bar while flat
- exit/flip confirmation: `2` consecutive 15m bars while in-position
- if signal re-aligns with current position before confirmation, pending exit/flip is canceled

Use `Stop` in the UI to stop the live session. Use `Ctrl+C` in the terminal to stop the dashboard server.

## Combined-server operations workspace

The combined-server hub is at `http://127.0.0.1:8764/` (or your configured
hub port). It shares a charcoal/teal visual system with the module dashboards
and the architecture atlas.

The redesign prioritizes account summary → module availability → individual
module details. A persistent directory replaces the hub's crowded top navigation;
search and Trading / Research & data / Needs attention filters narrow the cards.
Live previews are opt-in and are unloaded when disabled. Polls preserve focused
card controls. Failed polls retain the last snapshot with a stale notice and
pause controls until connectivity returns. If every module is unavailable,
position and P/L totals display as unknown rather than zero.

Module permissions and ports come from the server's descriptors. Real-money
intent remains per module and requires the existing confirmation when starting.
Start all sessions retains its continuous-session scope; it skips scheduled
4H passes and cannot start the governed Meta path.

The System atlas link opens `/architecture/`, which serves **only** the validated
Public build. Build it before using that route:

```bash
.venv/bin/python scripts/build_architecture_atlas.py
```

The atlas uses a quiet graph canvas, labeled tools, an optional ambient backdrop,
and a mobile outline. Search, flow filters, presentation mode, large text, and
saved component links remain available. Local Full remains a separate artifact.
Restart the hub through your normal server lifecycle after deploying Python or
HTML changes; do not restart an active trading server solely for a stylesheet.

Implementation: `hub_static/` holds the hub HTML/CSS/JS, `hub_dashboard.py` serves
it, `ui_chrome.py` defines shared module styling, and `architecture_atlas/static/`
holds the atlas. No external font or UI CDN is needed.

Focused verification:

```bash
.venv/bin/python -m pytest UI/tests/test_hub_dashboard.py UI/tests/test_architecture_atlas_build.py UI/tests/test_architecture_atlas_content.py UI/tests/test_dashboard_broken_pipe_guard.py -q
```

Optional browser verification requires Playwright plus Chromium:

```bash
.venv/bin/python -m pytest UI/tests/test_hub_ui_browser.py -q
```

Browser checks use fixture snapshots and intercept all downstream requests.
`CYNO_UI_SCREENSHOTS=/tmp/cyno-ui-review` saves fixture-only desktop/mobile images.

## Personal portfolio (Schwab, read-only)

`http://127.0.0.1:8778/` shows the user's own Schwab account next to the research
evidence. It never places, changes, or cancels orders: `core/personal_portfolio/holdings.py`
calls only `get_accounts(fields=positions)` and `get_quotes`, and a test fails if order code
appears in the package. Account numbers are masked to the last 3 digits.

- Schwab is read only while the page is open (at most once per 60 s, or on "Refresh now").
  The hub polls `/api/state`, which never calls Schwab and returns counts, not balances. The
  hub card has no `acct` block, so these positions never enter the paper-account totals.
- Leveraged ETFs are looked through to their underlying when the fund description names one
  AND the two return series agree (corr >= 0.85, beta within 30% of the stated leverage).
  Options count as delta x 100 x contracts x underlying price.
- Flags: name / correlated-cluster / gross concentration, leveraged-ETF volatility drag,
  options near expiry (exercise cash vs available funds, move to breakeven, earnings before
  expiry, wide spreads), shrugged-off earnings misses (research 07), earnings within 7 days.
  Momentum rank comes from the saved shadow-ledger ranking
  (`Data/research/shadow_cross_sections/`, refreshed each 4-weekly decision).
- Files (gitignored) under `Data/personal/`: `watchlist.csv`, optional `config.json`
  (thresholds; unknown keys fail) and `look_through.json` overrides, `bars_1d/`,
  `earnings.parquet`, `flag_log.jsonl` (one row per name per day, for testing the flags
  later), `equity_log.csv`.
- Requests need a loopback Host header and writes need a JSON body. Disable with
  `--no-personal`; change the port with `--port-personal`.
- **Trade history** (`/api/history`, at most every 15 min or on "Refresh history"): Schwab
  `get_transactions` (GET; TRADE + RECEIVE_AND_DELIVER, back to the account's first trade) into
  append-only `transactions.jsonl`, rebuilt into flat-to-flat round trips with cash P&L
  (`trade_history.py`; expirations close at 0, splits/reverse splits/mergers carry the trade,
  open trades reconcile against current positions). Charts (`trade_chart.py`) use raw daily and
  per-trade 15Min/1Hour bars so fills sit at the price paid; option fills are drawn at the
  underlying's price at the fill. Filters, breakdowns, and entry features (`trade_analysis.py`,
  pre-entry data only) run in the page. `#trade=<id>` / `#ticker=XYZ` deep-link a chart.
  Retired symbols that Schwab reports as CUSIPs map to tickers in `symbol_map.json`.
  `trade_timing.py` adds the timing review under the chart (entry style and exit trigger known at the
  time; entry/exit percentile, post-exit moves) and option what-ifs: hold-to-expiry is exact
  (intrinsic at the expiry close); +N sessions, the repo's live 4H exit policy, take-profit and roll
  are Black-Scholes estimates from IV backed out of the user's own fills (median error ~13-20%).
  Trades whose bars disagree with the fills (misaligned split adjustment, wrong security) are skipped.
- Flag `losing_option_late` (config `late_loss_frac`/`late_loss_dte`, default 50% / 14 days): a long option at or
  below half its cost late in life, quoting the user's own comeback odds from `trade_timing.comeback_stats`.
  The report's registered hypotheses (H1-H7, registered 2026-11-18) re-score on trades opened after that date.
  Report: `.venv/bin/python -m core.personal_portfolio.trade_review [--reconcile]` ->
  `Data/personal/reports/`.
- The watchlist form accepts a pasted list (commas/spaces). Schwab's API has no watchlist
  endpoint, so app watchlists (e.g. "looking for entries") must be pasted in once.

Standalone: `PYTHONPATH=. .venv/bin/python -m UI.personal_dashboard --port 8778`

Tests: `.venv/bin/python -m pytest core/personal_portfolio/tests UI/tests/test_personal_dashboard.py -q`

## Decision replay (read-only)

`http://127.0.0.1:8779/` replays a position bar by bar next to what its module saw. Pick a
position (open or closed, including open positions with no sell yet), or use **any ticker** to
replay any module, ticker and date window, e.g. a name the model scored but never bought. It
never writes and never contacts a broker (`UI/decision_replay.py`, `UI/decision_replay_dashboard.py`).

- **No look-ahead.** Frame i means "bar i has closed": later bars stay hidden unless "show
  future bars (hindsight)" is ticked. Every decision and fill sits on the bar during which it
  happened. Archived decisions use their wall-clock `decision_ts`. Audit rows log no time, so
  they sit where the bar they scored ended, and a later run re-scoring that bar sits on the next
  bar (late, never early). The bar a decision scored is drawn as "model's last bar". Chart
  timeframes shorter than a module's decision bar are not offered (no 1h for the 4H modules).
- **Evidence per position** (shown in the picker):
  - `archive`: `Data/research/decision_state/` (2026-11-13 on). Every run's score, score rank
    among all candidates, each gate, action and picks, plus the full feature vector. Each value
    shows its percentile among that run's candidates and its change since the entry decision.
  - `audit`: `live_signal_audit.jsonl` (2026-07 on). Top-K membership, score and rank for picks
    only, routing, and planned orders with their disposition. This is the only evidence for
    older positions, and it has no features.
  - `transitions`: Intraday Structure setup states with their evidence, on 1m bars from
    `Data/archive/intraday_1m/` (about 30 sessions).
  - `fills only`: SPY and 30m swing before their archive hooks were live.
- **Position at the cursor:**
  - The contract, for options: the OCC symbol the order went out with, days to expiry, moneyness
    against the recorded underlying anchor, premium in and out, contracts and cost.
  - Gain against the entry reference (the share fill, or the recorded underlying anchor for
    options) and the high and low against it so far.
  - The module's own `runs_held` against the horizon, and the distance to each exit level.
  - For Intraday Structure, the engine's closed-setup record: the level it traded, the
    invalidation (initial, and where it had trailed to), targets with their level types,
    expected and realized R.
  - Levels apply only from entry to exit, and are drawn from the cursor rightwards, so they
    never cover the bars. A level named in the exit reason (`stop_-39%`) is the one that fired.
    Other levels come from `scripts/live_risk_pass.py` and are labelled "current risk-pass
    policy". Option premium is never replayed: the bar cache holds the underlying only, and
    option prints are not marks.
- **Chart tools:** wheel to zoom, drag to pan, double-click or `0` to fit, `+`/`-` to zoom.
  "Follow cursor" scrolls the view during playback. Overlays are toggleable chips (saved per
  browser), and series tagged "model input" are inputs of that module's model.
  - Bar charts (1h/4h/1d) use momentum's own indicator code from `feature_matrix_4h.py`
    (EMA 10/20/50/100, ATR14, 20- and 60-bar and 52-week ceilings and floors, RSI, ADX, MACD
    hist), computed on the full cached history.
  - **Model-implied levels (×):** the EMAs and ceilings back-solved from an archived momentum
    row (e.g. EMA20 = close − `ema_dist_20` × ATR14), drawn at the bar it scored, with a
    "model saw vs chart" table. If a × sits off the line, the model scored a different bar
    than the one now in the cache.
  - 1m charts (Intraday Structure) show the engine's own `compute_features` at every
    regular-hours bar (VWAP, EMA9/20, opening range, prior-day range, micro swings).
    `/api/engine` adds the engine's clustered level map at the cursor
    (`StructuralLevelProvider` on its own 780-bar history), drawn right of the cursor and
    listed with each cluster's merged level types. Options-derived levels are not rebuilt,
    because they need the dealer snapshot the engine held at the time.
  - Lower pane: any recomputed indicator, or any archived live feature at each decision.
- Deep links: `#id=<position id>&f=<frame>&pane=<series>` or `#module=..&ticker=..&start=..&end=..`.
  Keys: left/right step, shift+left/right jump between decisions, space plays. The page is
  `UI/decision_replay_static/replay.html`, read per request.
- Disable with `--no-replay`; change the port with `--port-replay`.

Standalone: `PYTHONPATH=. .venv/bin/python -m UI.decision_replay_dashboard --port 8779`

Tests: `.venv/bin/python -m pytest UI/tests/test_decision_replay.py -q`
