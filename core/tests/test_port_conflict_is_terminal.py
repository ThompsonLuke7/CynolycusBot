"""A port conflict must be a distinct, terminal exit — not a generic failure.

The supervisor's backoff assumes a restart can eventually succeed. Against a
combined_server that already owns the dashboard sockets it never can, so the
loop just gets slower: on 2026-08-24 a manual restart at 23:49 left the old
instance holding the ports and scripts/run_live_server.sh fast-failed nine
times between 23:53 and 00:17 before settling into a 300s retry it could not
win. See research/daily_live_reports/2026-08-25.md.
"""
from __future__ import annotations

import re
import socket
from pathlib import Path

import pytest

from UI import combined_server

SUPERVISOR = Path(__file__).resolve().parents[2] / "scripts" / "run_live_server.sh"


def test_a_bound_port_exits_with_the_port_conflict_code() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        port = held.getsockname()[1]

        with pytest.raises(SystemExit) as excinfo:
            combined_server._assert_ports_free("127.0.0.1", {"probe": port})

    assert excinfo.value.code == combined_server.EX_PORT_CONFLICT
    # Never 1: the supervisor cannot tell that apart from a crash.
    assert combined_server.EX_PORT_CONFLICT != 1


def test_free_ports_do_not_raise() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    combined_server._assert_ports_free("127.0.0.1", {"probe": port})


def test_the_supervisor_agrees_on_the_code_and_stops_on_it() -> None:
    """The two halves are in different languages; only a test keeps them paired."""

    script = SUPERVISOR.read_text(encoding="utf-8")

    declared = re.search(r"^_EX_PORT_CONFLICT=(\d+)$", script, re.MULTILINE)
    assert declared, "run_live_server.sh must declare _EX_PORT_CONFLICT"
    assert int(declared.group(1)) == combined_server.EX_PORT_CONFLICT

    # It has to actually break the loop, not just log differently.
    branch = re.search(
        r'if \[ "\$rc" -eq "\$_EX_PORT_CONFLICT" \]; then(.+?)\n  fi',
        script,
        re.DOTALL,
    )
    assert branch, "the supervisor must branch on the port-conflict code"
    assert "break" in branch.group(1)
