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
    seen = {}
    monkeypatch.setattr(schwab_client, "TOKEN_PATH", token_path)
    monkeypatch.setenv("SCHWAB_API_KEY", "key-from-env")
    monkeypatch.setenv("SCHWAB_APP_SECRET", "secret-from-env")
    monkeypatch.setattr(
        schwab_client, "client_from_token_file", lambda **kwargs: seen.update(kwargs) or sentinel
    )

    assert schwab_client._create_or_load_client() is sentinel
    assert (seen["api_key"], seen["app_secret"]) == ("key-from-env", "secret-from-env")


def test_credentials_parse_the_env_file_format(monkeypatch, tmp_path):
    """Spaces around '=', quotes and trailing comments, as in the repo .env."""
    monkeypatch.delenv("SCHWAB_API_KEY", raising=False)
    monkeypatch.delenv("SCHWAB_APP_SECRET", raising=False)
    env = tmp_path / ".env"
    env.write_text('OTHER=1\nSCHWAB_API_KEY = "abc123"          # App Key\n'
                   "SCHWAB_APP_SECRET = 'xyz789'    # App Secret\n")
    assert schwab_client._app_credentials(env) == ("abc123", "xyz789")
    monkeypatch.setenv("SCHWAB_API_KEY", "override")
    assert schwab_client._app_credentials(env) == ("override", "xyz789")


def test_missing_credentials_fail_fast_and_keep_the_token(monkeypatch, tmp_path):
    monkeypatch.delenv("SCHWAB_API_KEY", raising=False)
    monkeypatch.delenv("SCHWAB_APP_SECRET", raising=False)
    monkeypatch.setattr(schwab_client, "ENV_PATH", tmp_path / "missing.env")
    token_path = tmp_path / "schwab_token.json"
    token_path.write_text("{}")
    monkeypatch.setattr(schwab_client, "TOKEN_PATH", token_path)
    with pytest.raises(RuntimeError, match="SCHWAB_API_KEY and SCHWAB_APP_SECRET"):
        schwab_client._create_or_load_client()
    with pytest.raises(RuntimeError, match="SCHWAB_API_KEY"):
        schwab_client._create_or_load_client(force_manual=True)
    assert token_path.exists(), "a failed reauth must not move the good token aside"


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


@pytest.fixture
def _live_token(monkeypatch, tmp_path):
    token = tmp_path / "schwab_token.json"
    token.write_text('{"live": 1}')
    monkeypatch.setattr(schwab_client, "TOKEN_PATH", token)
    monkeypatch.setenv("SCHWAB_API_KEY", "k")
    monkeypatch.setenv("SCHWAB_APP_SECRET", "s")
    return token


@pytest.mark.parametrize("failure", [KeyboardInterrupt, EOFError, RuntimeError])
def test_abandoned_reauth_leaves_the_live_token_in_place(monkeypatch, _live_token, failure):
    """2026-11-13 00:12: --reauth moved the token aside, the login was not finished, and every
    Schwab job (and the personal page) failed until the still-valid token was restored by hand."""
    def _login(**kwargs):
        open(kwargs["token_path"], "w").write("partial")
        raise failure()

    monkeypatch.setattr(schwab_client, "client_from_manual_flow", _login)
    with pytest.raises(failure):
        schwab_client._create_or_load_client(force_manual=True)
    assert _live_token.read_text() == '{"live": 1}'
    assert sorted(p.name for p in _live_token.parent.iterdir()) == ["schwab_token.json"]   # no pending, no backup


def test_login_that_writes_nothing_keeps_the_live_token(monkeypatch, _live_token):
    monkeypatch.setattr(schwab_client, "client_from_manual_flow", lambda **kwargs: object())
    with pytest.raises(RuntimeError, match="without writing a token"):
        schwab_client._create_or_load_client(force_manual=True)
    assert _live_token.read_text() == '{"live": 1}'


def test_successful_reauth_swaps_in_the_new_token_and_backs_up_the_old(monkeypatch, _live_token):
    loaded = {}
    sentinel = object()

    def _login(**kwargs):
        assert kwargs["token_path"] != str(_live_token), "the login must not write the live file"
        open(kwargs["token_path"], "w").write('{"new": 1}')
        return object()

    monkeypatch.setattr(schwab_client, "client_from_manual_flow", _login)
    monkeypatch.setattr(schwab_client, "client_from_token_file", lambda **kwargs: loaded.update(kwargs) or sentinel)
    assert schwab_client._create_or_load_client(force_manual=True) is sentinel
    assert _live_token.read_text() == '{"new": 1}' and loaded["token_path"] == str(_live_token)
    backups = list(_live_token.parent.glob("schwab_token.invalid_*.json"))
    assert len(backups) == 1 and backups[0].read_text() == '{"live": 1}'
    assert not list(_live_token.parent.glob("*.pending.json"))
    assert schwab_client._cached_client is sentinel
