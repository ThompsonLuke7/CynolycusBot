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
