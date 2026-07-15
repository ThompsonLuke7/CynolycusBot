#!/bin/bash
#
# WSL2 crash forensics logger.
#
# WHY: the live server dies daily to a WSL2 *VM* crash (Windows logs nothing —
# not sleep, not power, not Hyper-V; Windows stays up for weeks). The fault is
# internal to the Linux guest, but the guest's dmesg/journal is WIPED when the VM
# restarts, so we never see the cause. This logger streams the kernel ring buffer
# and periodic memory stats to the NTFS side (/mnt/c), which SURVIVES a guest
# crash — so the next crash leaves its dying words behind (OOM kill? panic? hung
# task?). Read the files after the next crash to get the root cause.
#
# Output (on the Windows filesystem, survives VM death):
#   C:\Users\<user>\wsl_crashlog\dmesg.log   — full kernel ring buffer + live tail
#   C:\Users\<user>\wsl_crashlog\mem.log     — timestamped free/load/top-RSS every 20s
#
# Usage:  nohup scripts/wsl_crash_logger.sh >/dev/null 2>&1 &
# (run_live_server.sh starts it automatically alongside the server.)
set -uo pipefail

WINUSER="$(cmd.exe /c 'echo %USERNAME%' 2>/dev/null | tr -d '\r\n' || echo luket)"
OUT="/mnt/c/Users/${WINUSER}/wsl_crashlog"
mkdir -p "$OUT" 2>/dev/null || OUT="/mnt/c/Users/luket/wsl_crashlog"
mkdir -p "$OUT"
DMESG_LOG="$OUT/dmesg.log"
MEM_LOG="$OUT/mem.log"

stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }

# 1) Snapshot the current ring buffer, then follow new kernel messages. --follow
#    keeps appending; the last lines before a crash land on NTFS and persist.
{
  echo "===== dmesg logger started $(stamp) (kernel $(uname -r)) ====="
  dmesg -T 2>/dev/null || dmesg 2>/dev/null
} >> "$DMESG_LOG"
# `dmesg --follow` blocks; run it in the background so we can also log memory.
( dmesg -T --follow 2>/dev/null || dmesg --follow 2>/dev/null ) >> "$DMESG_LOG" &
DMESG_PID=$!
trap 'kill "$DMESG_PID" 2>/dev/null' EXIT INT TERM

# 2) Memory/pressure sampler — the leading crash hypothesis is a guest OOM under
#    the market-open ingestion surge, so capture what RSS/cache/swap looked like.
echo "===== mem logger started $(stamp) (cap $(grep -i memtotal /proc/meminfo | awk '{print int($2/1024)" MB"}')) =====" >> "$MEM_LOG"
while true; do
  {
    echo "--- $(stamp) load:$(cut -d' ' -f1-3 /proc/loadavg) ---"
    free -m | sed 's/^/  /'
    # PSI memory pressure (if available) — nonzero 'some avg10' = real stalling.
    [ -r /proc/pressure/memory ] && echo "  psi: $(head -1 /proc/pressure/memory)"
    echo "  top-RSS:"
    ps -eo rss,comm --sort=-rss 2>/dev/null | head -6 | sed 's/^/    /'
  } >> "$MEM_LOG"
  sleep 20
done
