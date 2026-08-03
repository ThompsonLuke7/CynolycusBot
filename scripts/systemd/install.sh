#!/bin/bash
#
# Install the live server as a systemd *user* service so it survives the
# terminal it was started from, and starts by itself when the WSL VM boots.
#
# Idempotent: safe to re-run after editing the template.
#
# Usage:
#   scripts/systemd/install.sh              # install + enable, do not start
#   scripts/systemd/install.sh --start      # install + enable + start now
#   scripts/systemd/install.sh --uninstall  # stop, disable, remove the unit
#
# `enable-linger` needs sudo once; everything else is per-user and does not.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UNIT_NAME="cynolycus-live.service"
TEMPLATE="$REPO_ROOT/scripts/systemd/${UNIT_NAME}.template"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT_PATH="$UNIT_DIR/$UNIT_NAME"

START_NOW=0
UNINSTALL=0
for arg in "$@"; do
  case "$arg" in
    --start) START_NOW=1 ;;
    --uninstall) UNINSTALL=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [ "$UNINSTALL" = "1" ]; then
  systemctl --user stop "$UNIT_NAME" 2>/dev/null || true
  systemctl --user disable "$UNIT_NAME" 2>/dev/null || true
  rm -f "$UNIT_PATH"
  systemctl --user daemon-reload
  echo "Removed $UNIT_PATH"
  echo "Linger left enabled; disable with: sudo loginctl disable-linger $USER"
  exit 0
fi

if [ ! -f "$TEMPLATE" ]; then
  echo "Template not found: $TEMPLATE" >&2
  exit 1
fi

# Fail fast rather than installing a unit that cannot possibly work.
if [ "$(ps -p 1 -o comm=)" != "systemd" ]; then
  echo "PID 1 is not systemd. Add the following to /etc/wsl.conf, then run" >&2
  echo "'wsl.exe --shutdown' from Windows and reopen the distro:" >&2
  echo "" >&2
  echo "  [boot]" >&2
  echo "  systemd=true" >&2
  exit 1
fi

mkdir -p "$UNIT_DIR"
sed "s|@REPO_ROOT@|$REPO_ROOT|g" "$TEMPLATE" > "$UNIT_PATH"
echo "Wrote $UNIT_PATH"

systemctl --user daemon-reload
systemctl --user enable "$UNIT_NAME"
echo "Enabled $UNIT_NAME (starts on VM boot)"

# Without linger, the user manager only runs while a login session exists — so
# the service would still be tied to "is a terminal open", which is the whole
# problem we are removing.
if [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != "yes" ]; then
  echo ""
  echo "Linger is OFF. The service will not start at VM boot without it."
  echo "Run once (needs sudo):"
  echo ""
  echo "  sudo loginctl enable-linger $USER"
  echo ""
fi

if [ "$START_NOW" = "1" ]; then
  systemctl --user restart "$UNIT_NAME"
  sleep 2
  systemctl --user --no-pager status "$UNIT_NAME" | head -20
else
  echo ""
  echo "Not started. When you are ready to hand the running server over:"
  echo "  # stop whatever is running in your terminal first (Ctrl-C), then:"
  echo "  systemctl --user start $UNIT_NAME"
fi

echo ""
echo "Useful:"
echo "  systemctl --user status  $UNIT_NAME"
echo "  systemctl --user restart $UNIT_NAME"
echo "  systemctl --user stop    $UNIT_NAME"
echo "  journalctl --user -u $UNIT_NAME -f      # supervisor lines"
echo "  tail -F $REPO_ROOT/logs/live_server/server_\$(date +%Y%m%d).log   # full detail"
