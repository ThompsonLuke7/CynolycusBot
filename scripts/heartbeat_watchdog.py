#!/usr/bin/env python3
"""
Live-trading heartbeat watchdog.

Catches the failure mode the auto-restart can't: a combined_server that is still
*alive* but has stopped doing work (hung stream, deadlock) during market hours,
or one that died while the supervisor is also down. The 2026-06-26 OOM stayed
undetected until end of day precisely because nothing was watching.

It watches the freshest *live audit write* — the swing session JSONL and the
intraday SPY broker-state JSONL, both of which the server appends to roughly
every 30-60s while running. During regular trading hours (09:30-16:00 ET,
weekdays) it alerts if:

  * no audit file has been written for STALE_MIN minutes, OR
  * no `combined_server` process is running,

but only *after* it has seen at least one fresh write since startup (so it never
nags when you simply haven't started the server, or on a market holiday when the
server idles).

Separately, and at ANY hour, it alerts when a `combined_server` it has already
seen running disappears. The RTH gate is right for a hung-but-alive server —
that only costs anything while the market is open — but a process that stops
outside RTH takes whatever it was running down with it. On 2026-08-20 the server
went quiet at 22:43 ET with no exit code, no OOM signature and no alert; the
22:15 data-readiness job died with it, its catch-up then lost the heavy-job lock
race, and Meta skipped five live entries the next afternoon on a stale readiness
stamp. "It was running and now it is not" needs no market-hours qualifier.

Alerts go to: this log (always), a Windows toast via powershell.exe (best-effort
under WSL), and an optional webhook (LIVE_ALERT_WEBHOOK, e.g. Slack/Discord).
Alerts are de-duplicated by ALERT_COOLDOWN so you get one ping, not a flood.

Config (all via env, sensible defaults):
  STALE_MIN=8           minutes without a write before it's "stale"
  CHECK_INTERVAL=60     seconds between checks
  ALERT_COOLDOWN=900    seconds between repeat alerts for the same condition
  RTH_START=09:30  RTH_END=16:00   trading window (ET)
  LIVE_ALERT_WEBHOOK=   optional POST url (JSON {"text": ...})

Stdlib only. Run standalone or let run_live_server.sh launch it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
ET = ZoneInfo("America/New_York")

STALE_MIN = float(os.environ.get("STALE_MIN", "8"))
CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL", "60"))
ALERT_COOLDOWN = int(os.environ.get("ALERT_COOLDOWN", "900"))
WEBHOOK = os.environ.get("LIVE_ALERT_WEBHOOK", "").strip()


def _parse_hhmm(value: str, default: dtime) -> dtime:
    try:
        hh, mm = value.split(":")
        return dtime(int(hh), int(mm))
    except Exception:
        return default


RTH_START = _parse_hhmm(os.environ.get("RTH_START", "09:30"), dtime(9, 30))
RTH_END = _parse_hhmm(os.environ.get("RTH_END", "16:00"), dtime(16, 0))

# Files the live server appends to while running.
WATCH_GLOBS = [
    ("swing", REPO_ROOT / "UI" / "swing_audit", "swing_session_*.jsonl"),
    ("spy", REPO_ROOT / "Data" / "inference" / "live_runs", "*/broker-state.jsonl"),
]


def log(msg: str) -> None:
    print(f"{datetime.now(ET):%Y-%m-%d %H:%M:%S} [watchdog] {msg}", flush=True)


def in_rth(now: datetime) -> bool:
    if now.weekday() >= 5:  # Sat/Sun
        return False
    return RTH_START <= now.timetz().replace(tzinfo=None) <= RTH_END


def freshest_write_age_secs() -> float | None:
    """Age (seconds) of the most recently modified watched audit file, or None
    if no watched file exists yet."""
    newest: float | None = None
    for _name, base, pattern in WATCH_GLOBS:
        if not base.exists():
            continue
        for path in base.glob(pattern):
            try:
                mt = path.stat().st_mtime
            except OSError:
                continue
            if newest is None or mt > newest:
                newest = mt
    if newest is None:
        return None
    return max(0.0, time.time() - newest)


def server_running() -> bool:
    try:
        out = subprocess.run(
            ["pgrep", "-f", "UI.combined_server"],
            capture_output=True, text=True, timeout=10,
        )
        return out.returncode == 0 and bool(out.stdout.strip())
    except Exception:
        return True  # if we can't tell, don't cry wolf


def _windows_toast(title: str, message: str) -> None:
    """Best-effort Windows toast/notification from WSL via powershell.exe."""
    ps = (
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, "
        "ContentType=WindowsRuntime] > $null; "
        "$t=[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('CynolycusBot'); "
        f"$x=[Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent("
        "[Windows.UI.Notifications.ToastTemplateType]::ToastText02); "
        f"$x.GetElementsByTagName('text')[0].AppendChild($x.CreateTextNode('{title}')) > $null; "
        f"$x.GetElementsByTagName('text')[1].AppendChild($x.CreateTextNode('{message}')) > $null; "
        "$t.Show([Windows.UI.Notifications.ToastNotification]::new($x))"
    )
    try:
        subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps],
                       capture_output=True, timeout=15)
    except Exception:
        pass


def _webhook(text: str) -> None:
    if not WEBHOOK:
        return
    try:
        data = json.dumps({"text": text}).encode()
        req = urllib.request.Request(WEBHOOK, data=data,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as exc:
        log(f"webhook post failed: {exc}")


def alert(message: str) -> None:
    log(f"ALERT: {message}")
    _windows_toast("CynolycusBot live-server ALERT", message)
    _webhook(f":rotating_light: CynolycusBot: {message}")


def main() -> int:
    log(f"watchdog up: stale>{STALE_MIN}m during {RTH_START:%H:%M}-{RTH_END:%H:%M} ET, "
        f"check every {CHECK_INTERVAL}s, cooldown {ALERT_COOLDOWN}s, "
        f"webhook={'set' if WEBHOOK else 'off'}")
    seen_fresh_today: str | None = None  # date string we last saw a fresh write
    last_alert_ts = 0.0
    # Has this watchdog ever seen the server process alive? Only then can its
    # absence mean "it stopped" rather than "it was never started".
    seen_server_running = False
    last_disappeared_alert_ts = 0.0

    while True:
        now = datetime.now(ET)
        today = now.strftime("%Y-%m-%d")

        # A server that WAS running and now is not has stopped, and that is worth
        # knowing at any hour — the RTH gate below only makes sense for the
        # hung-but-alive case. On 2026-08-20 the combined_server's last log line
        # was 22:43:21 ET with no exit code, no OOM signature and no alert; the
        # supervisor started a fresh instance at 23:45. Outside RTH, so nothing
        # fired. The cost landed the next morning: the 22:15 data-readiness job
        # died with it, its catch-up lost the heavy-job lock race, and Meta's
        # 14:20 run skipped five live entries on a stale stamp.
        running_now = server_running()
        if running_now:
            seen_server_running = True
        elif seen_server_running:
            if time.time() - last_disappeared_alert_ts >= ALERT_COOLDOWN:
                alert(f"combined_server DISAPPEARED at {now:%Y-%m-%d %H:%M} ET "
                      f"(it was running earlier in this watchdog session). "
                      f"Check for a silent stop; anything it was running died with it.")
                last_disappeared_alert_ts = time.time()
            # Re-arm: a supervised relaunch should alert again if it stops again.
            seen_server_running = False

        if in_rth(now):
            age = freshest_write_age_secs()
            running = running_now
            if age is not None and age <= STALE_MIN * 60:
                seen_fresh_today = today  # server is clearly alive today

            # Only alert if we have evidence the server ran today, then went quiet.
            armed = (seen_fresh_today == today)
            stale = (age is None or age > STALE_MIN * 60)
            if armed and (stale or not running):
                if time.time() - last_alert_ts >= ALERT_COOLDOWN:
                    if not running:
                        alert(f"combined_server NOT RUNNING during RTH ({now:%H:%M} ET). "
                              f"Last audit write {('%.0f' % (age/60)) if age else '?'}m ago.")
                    else:
                        alert(f"combined_server appears HUNG: no audit write for "
                              f"{age/60:.0f}m during RTH ({now:%H:%M} ET).")
                    last_alert_ts = time.time()
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
