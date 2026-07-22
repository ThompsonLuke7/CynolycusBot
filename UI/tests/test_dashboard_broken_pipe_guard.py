"""Every dashboard's plain JSON/text write path must swallow a client
mid-write disconnect instead of letting a full traceback escape into the
server log.

Regression coverage for the 2026-07-20 audit: ~940 BrokenPipeError /
ConnectionResetError tracebacks were logged in one session (dealer_ranker_dashboard
880, meta_ranker_dashboard 762, live_dashboard 294, htf_dashboard 10,
amethyst_dashboard 9, dealer_positioning_dashboard 2) -- all from a browser tab
closing or polling past its timeout mid-response, which is normal HTTP server
behavior, not an application bug. `dealer_ranker_dashboard.py`'s `_send` already
caught this; the same guard is now applied consistently to the other five.
"""
from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler

import pytest

from UI.amethyst_dashboard import AmethystHandler
from UI.dealer_positioning_dashboard import DealerDashboardHandler
from UI.dealer_ranker_dashboard import DealerRankerHandler
from UI.htf_dashboard import HTFHandler
from UI.live_dashboard import DashboardHandler
from UI.meta_ranker_dashboard import MetaRankerHandler


class _FakeWfile:
    def write(self, data):
        raise BrokenPipeError(32, "Broken pipe")

    def flush(self):
        pass


def _bare_handler(cls):
    """Build a handler instance with just enough state for send_response/
    send_header/end_headers to run, without a real socket/server."""
    h = BaseHTTPRequestHandler.__new__(cls)
    h.wfile = _FakeWfile()
    h.request_version = "HTTP/1.1"
    h.protocol_version = "HTTP/1.0"
    h.close_connection = False
    h._headers_buffer = []
    h.requestline = "GET / HTTP/1.1"
    h.client_address = ("127.0.0.1", 12345)
    h.log_message = lambda *a, **k: None
    return h


@pytest.mark.parametrize(
    "cls,method,call",
    [
        (DealerRankerHandler, "_send", lambda h: h._send(b"{}")),
        (MetaRankerHandler, "_send", lambda h: h._send(b"{}")),
        (HTFHandler, "_send", lambda h: h._send(b"{}")),
        (AmethystHandler, "_send", lambda h: h._send(b"{}")),
        (DashboardHandler, "_write_json", lambda h: h._write_json({})),
        (DashboardHandler, "_write_text", lambda h: h._write_text("hi")),
        (DealerDashboardHandler, "_write_json", lambda h: h._write_json({})),
        (DealerDashboardHandler, "_write_text", lambda h: h._write_text("hi")),
    ],
)
def test_write_path_swallows_broken_pipe(cls, method, call):
    h = _bare_handler(cls)
    call(h)  # must not raise
    assert h.close_connection is True


@pytest.mark.parametrize("cls", [DealerRankerHandler, MetaRankerHandler, HTFHandler, AmethystHandler])
def test_send_still_sets_headers_on_success(cls):
    class _OkWfile:
        def __init__(self):
            self.written = b""

        def write(self, data):
            self.written += data

    h = BaseHTTPRequestHandler.__new__(cls)
    h.wfile = _OkWfile()
    h.request_version = "HTTP/1.1"
    h.protocol_version = "HTTP/1.0"
    h.close_connection = False
    h._headers_buffer = []
    h.requestline = "GET / HTTP/1.1"
    h.client_address = ("127.0.0.1", 12345)
    h.log_message = lambda *a, **k: None
    h._send(b'{"ok": true}', status=HTTPStatus.OK)
    assert b'{"ok": true}' in h.wfile.written
    assert h.close_connection is False
