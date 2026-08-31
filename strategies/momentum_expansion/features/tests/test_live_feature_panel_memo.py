"""The per-ticker memo must save the rebuild without ever changing the answer.

Context (2026-08-24): `build_ticker_features_4h` is ~129ms/ticker and 93% of a
panel build, while reading both parquets is ~10ms. A full 2709-name universe
therefore costs ~6 minutes standalone and measured 1018.2s on the contended
live server -- and the momentum dashboard paid for 41 of them in one session,
every one producing a bit-identical answer, because the bar caches only change
when new bars land. The memo keys each ticker's trimmed feature rows on a
content hash of the exact frames handed to the builder.

"Never changes the answer" is the whole point, so these tests pin both halves:
a hit is byte-identical to a rebuild, and anything that could alter the result
(new bars, revised bars, changed context, different trim) misses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.momentum_expansion.config.momentum_config import MIN_4H_BARS
from strategies.momentum_expansion.features import live_feature_panel_4h as mod
from strategies.momentum_expansion.features.live_feature_panel_4h import (
    build_live_feature_panel_4h,
    clear_feature_panel_memo,
)

N_BARS = MIN_4H_BARS + 20
TICKERS = ["AAA", "BBB", "CCC"]


def _bars(seed: int, n: int = N_BARS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    close = np.clip(100 + np.cumsum(rng.normal(0, 1, n)), 5, None)
    return pd.DataFrame({
        "open": close + rng.normal(0, 0.3, n),
        "high": close + rng.uniform(0, 1, n),
        "low": close - rng.uniform(0, 1, n),
        "close": close,
        "volume": rng.uniform(1e5, 1e6, n),
    }, index=idx)


@pytest.fixture(autouse=True)
def _clean_memo():
    clear_feature_panel_memo()
    yield
    clear_feature_panel_memo()


@pytest.fixture()
def env(monkeypatch):
    """Loaders over mutable synthetic bars, plus a call counter on the builder."""
    store = {t: _bars(seed=i) for i, t in enumerate(TICKERS)}
    calls: list[str] = []
    real = mod.build_ticker_features_4h

    def counting(*, ticker, df_4h, df_1d, ctx_4h):
        calls.append(ticker)
        return real(ticker=ticker, df_4h=df_4h, df_1d=df_1d, ctx_4h=ctx_4h)

    monkeypatch.setattr(mod, "build_ticker_features_4h", counting)

    def build(**kw):
        kw.setdefault("tickers", TICKERS)
        kw.setdefault("tail", 1)
        kw.setdefault("ctx_4h", {})
        return build_live_feature_panel_4h(
            load_4h_bars=lambda t: store[t], load_1d_bars=lambda t: None, **kw)

    return type("Env", (), {"store": store, "calls": calls, "build": staticmethod(build)})


def test_second_build_is_a_hit_and_is_identical(env):
    first = env.build()
    assert env.calls == TICKERS
    assert first.memo_hits == 0

    env.calls.clear()
    second = env.build()

    assert env.calls == [], "rebuilt tickers whose bars had not changed"
    assert second.memo_hits == len(TICKERS)
    pd.testing.assert_frame_equal(first.panel, second.panel)


def test_new_bar_rebuilds_only_the_ticker_that_moved(env):
    env.build()
    env.calls.clear()

    env.store["BBB"] = _bars(seed=1, n=N_BARS + 1)    # one fresh bar for BBB
    result = env.build()

    assert env.calls == ["BBB"]
    assert result.memo_hits == 2
    assert result.tickers_built == 3


def test_a_revised_bar_mid_history_is_caught(env):
    """A mtime/last-timestamp key would miss this; a content hash must not."""
    env.build()
    env.calls.clear()

    revised = env.store["AAA"].copy()
    revised.iloc[len(revised) // 2, revised.columns.get_loc("close")] += 0.01
    env.store["AAA"] = revised
    env.build()

    assert env.calls == ["AAA"], "a mid-history bar revision was served from the memo"


def test_changed_context_invalidates_every_ticker(env):
    """ctx_4h feeds every ticker's features, so it has to key every entry."""
    env.build(ctx_4h={"SPY": _bars(seed=99)})
    env.calls.clear()

    env.build(ctx_4h={"SPY": _bars(seed=100)})

    assert env.calls == TICKERS


def test_different_tail_is_a_different_entry(env):
    env.build(tail=1)
    env.calls.clear()

    result = env.build(tail=3)

    assert env.calls == TICKERS, "a tail=1 entry was served to a tail=3 caller"
    rows_per_ticker = result.panel.groupby(level="ticker").size()
    assert set(rows_per_ticker) == {3}


def test_a_hit_cannot_be_poisoned_by_its_caller(env):
    """Callers concat and post-process the rows; the stored frame must be safe."""
    first = env.build()
    first.panel.iloc[0, 0] = -12345.0          # scribble on the returned panel

    env.calls.clear()
    second = env.build()

    assert second.memo_hits == len(TICKERS)
    assert second.panel.iloc[0, 0] != -12345.0


def test_build_failures_are_never_memoized(env, monkeypatch):
    """A transient build error must not pin a ticker out of the panel forever."""
    boom = {"AAA"}

    real = mod.build_ticker_features_4h

    def flaky(*, ticker, df_4h, df_1d, ctx_4h):
        env.calls.append(ticker)
        if ticker in boom:
            raise RuntimeError("transient")
        return real(ticker=ticker, df_4h=df_4h, df_1d=df_1d, ctx_4h=ctx_4h)

    monkeypatch.setattr(mod, "build_ticker_features_4h", flaky)

    first = env.build()
    assert first.tickers_skipped == ["AAA"]

    boom.clear()                                # the failure clears up
    env.calls.clear()
    second = env.build()

    assert "AAA" in env.calls, "the failed ticker was memoized as a failure"
    assert second.tickers_skipped == []
    assert "AAA" in second.panel.index.get_level_values("ticker")


def test_untrimmed_results_are_not_stored(env):
    """Full-history rows are ~3.2MB/ticker -- the batch path must not fill RAM."""
    env.build(tail=None)
    assert len(mod._MEMO) == 0

    env.build(tail=1)
    assert set(mod._MEMO) == set(TICKERS)


def test_memo_holds_one_entry_per_ticker(env):
    """Bounded by construction: re-keying replaces, it does not accumulate."""
    for n in range(4):
        env.store["AAA"] = _bars(seed=0, n=N_BARS + n)
        env.build()
    assert len(mod._MEMO) == len(TICKERS)
