"""An unattended process must never be offered an interactive Schwab login.

2026-08-24: a re-auth was started a second time, which moved the (freshly
minted, perfectly good) token aside and was then interrupted before the browser
round-trip completed, leaving no ``schwab_token.json``. The next
``SchwabClient()`` — built inside a combined_server worker thread with no stdin
— fell through to schwab-py's ``input('Redirect URL> ')`` and died with
``EOFError``, twelve frames deep. The 2026-07-22 variant of the same fall-through
printed eight simultaneous prompts and blocked every Schwab job for a day.
"""

from __future__ import annotations

import builtins

import pytest

from core.API.Schwab_API import schwab_client


@pytest.fixture(autouse=True)
def _no_cached_client(monkeypatch):
    monkeypatch.setattr(schwab_client, "_cached_client", None)


def test_missing_token_raises_instead_of_prompting(monkeypatch, tmp_path):
    def _explode(*args, **kwargs):
        raise AssertionError("the interactive flow must not run unattended")

    monkeypatch.setattr(schwab_client, "TOKEN_PATH", tmp_path / "schwab_token.json")
    monkeypatch.setattr(schwab_client, "client_from_manual_flow", _explode)

    with pytest.raises(RuntimeError) as exc:
        schwab_client._create_or_load_client()

    assert "--reauth" in str(exc.value), "the error must name the fix"


def test_existing_token_still_loads_from_file(monkeypatch, tmp_path):
    token_path = tmp_path / "schwab_token.json"
    token_path.write_text("{}")
    sentinel = object()
    monkeypatch.setattr(schwab_client, "TOKEN_PATH", token_path)
    monkeypatch.setattr(
        schwab_client, "client_from_token_file", lambda **kwargs: sentinel
    )

    assert schwab_client._create_or_load_client() is sentinel


def test_reauth_command_survives_direct_script_execution(monkeypatch):
    """``schwab_client.py`` is run as a script, where ``core`` is not importable."""
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "core.schwab_token_status":
            raise ModuleNotFoundError("No module named 'core'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    # The point is that it degrades to the literal instead of raising; both
    # branches yield the same command string, so only the no-raise matters.
    assert "--reauth" in schwab_client._reauth_command()
