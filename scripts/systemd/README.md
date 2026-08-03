# Keeping the live server alive

The supervisor (`scripts/run_live_server.sh`) restarts `combined_server` when it
exits. It cannot restart *itself*. That gap is what actually costs trading days:

| Date | What died | Restart ladder fired? |
|---|---|---|
| 2026-07-29 ~18:53 ET | whole process tree | no — supervisor died too |
| 2026-07-30 ~02:55 ET | whole process tree | no |
| 2026-07-30 11:25 ET | whole process tree | no |

In every case there was no `[supervisor] combined_server exited rc=…` line: the
supervisor was killed alongside its child, so the loop it was sitting in never
got to run. The 2026-07-30 session lost the afternoon 4H signal runs, the dealer
ranker, the shadow tracker, the nightly data pipeline and the account snapshot —
and left 121 positions unmanaged with two armed trailing stops.

There are two distinct failure modes and they need two distinct fixes.

## Layer 1 — the terminal (fixed by systemd)

Started from a VS Code terminal, the supervisor is a child of that terminal's
session. Closing the terminal, closing VS Code, or dropping the WSL session
sends it `SIGHUP`.

`systemd` is PID 1 in this distro (`/etc/wsl.conf` has `systemd=true`), so a
user unit is outside every terminal's process tree and starts on VM boot.

```bash
scripts/systemd/install.sh          # install + enable (does not start)
sudo loginctl enable-linger "$USER" # once: run the user manager without a login session
systemctl --user start cynolycus-live
```

`enable-linger` is not optional. Without it the user systemd manager only exists
while you are logged in, which reintroduces exactly the dependency being removed.

Day-to-day:

```bash
systemctl --user status  cynolycus-live
systemctl --user restart cynolycus-live
journalctl --user -u cynolycus-live -f        # supervisor lines only
tail -F logs/live_server/server_$(date +%Y%m%d).log   # everything
```

To go back to running it by hand, `systemctl --user stop cynolycus-live` first —
two supervisors would fight over the same ports and the same broker account.

## Layer 2 — the WSL VM itself (needs Windows)

When the WSL2 VM crashes or shuts down, everything inside it is gone, systemd
included. **Nothing inside WSL can fix this** — the VM has to be started from
the Windows side.

WSL also shuts the VM down on its own once the last session closes
(`vmIdleTimeout`, default ~60s), so "no terminal open" eventually means "no VM".

Create a Windows Scheduled Task that pokes the distro on a timer. Starting the
VM is enough: systemd + linger then bring the service up. In an **Administrator
PowerShell** on Windows:

```powershell
schtasks /Create /TN "CynolycusWSLKeepAlive" /SC MINUTE /MO 5 /RU "$env:USERNAME" /RL HIGHEST /F /TR "wsl.exe -d Ubuntu -u luket --exec /bin/true"
```

Then add a boot trigger so a Windows restart also brings it back:

```powershell
schtasks /Create /TN "CynolycusWSLBoot" /SC ONSTART /RU "$env:USERNAME" /RL HIGHEST /F /TR "wsl.exe -d Ubuntu -u luket --exec /bin/true"
```

`/bin/true` is deliberate: the task's only job is to make sure the VM exists.
Everything about *what runs* stays in the systemd unit, in this repo, under
version control.

Verify from Windows:

```powershell
schtasks /Query /TN "CynolycusWSLKeepAlive" /V /FO LIST
```

Optionally stop the VM idling out at all, in `C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
vmIdleTimeout=-1
```

## What each layer catches

| Failure | Inner loop | systemd unit | Windows task |
|---|---|---|---|
| `combined_server` crashes / OOM | yes | — | — |
| supervisor killed, VM alive | no | yes | — |
| terminal / VS Code closed | no | yes | — |
| WSL VM crashed or idled out | no | no | yes |
| Windows rebooted | no | no | yes (ONSTART) |

## Still worth knowing

The crash cause itself is unresolved. `scripts/wsl_crash_logger.sh` streams guest
`dmesg` to `C:\Users\<you>\wsl_crashlog` precisely because a WSL2 VM crash leaves
no host-side trace and wipes the guest kernel log on restart. Check that
directory after the next event — restarting faster is not the same as knowing
why it fell over.
