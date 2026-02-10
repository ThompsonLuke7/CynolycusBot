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

Use `Stop` in the UI to stop the live session. Use `Ctrl+C` in the terminal to stop the dashboard server.
