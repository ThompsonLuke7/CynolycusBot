from __future__ import annotations

import pandas as pd

from scripts.discord_ledger.score_ace_contract_alerts import first_touch, osi, score

ALERT = {'alert_id': 'a', 'symbol': 'SPY', 'strike': 765.0, 'right': 'put', 'expiry': '2026-11-02', 'premium': 2.2,
         'stop': 2.0, 'flags': [], 'posted_at_et': '2026-10-29T15:12:40-04:00', 'posted_at_utc': '2026-10-29T19:12:40Z'}


def bars(start: str, rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(rows), freq='1min', tz='UTC')
    return pd.DataFrame(rows, index=idx, columns=['open', 'high', 'low', 'close'])


def test_osi_maps_share_class_and_index_roots() -> None:
    assert osi(ALERT) == 'SPY261102P00765000'
    assert osi({**ALERT, 'symbol': 'BRK.B', 'strike': 550.0, 'right': 'call', 'expiry': '2026-10-30'}) == 'BRKB261030C00550000'


def test_same_bar_stop_and_target_counts_as_the_stop() -> None:
    path = bars('2026-10-29T19:15:00Z', [(2.2, 4.5, 1.9, 4.0)])
    assert first_touch(path, 2.0, 4.4)[0] == 'stop_same_bar_as_target'


def test_entry_waits_one_minute_and_mfe_uses_later_bars_only() -> None:
    opt = bars('2026-10-29T19:13:00Z', [(9.0, 9.0, 9.0, 9.0), (2.3, 2.4, 2.25, 2.3), (2.3, 4.0, 2.3, 3.9)])
    out = score(ALERT, opt, None, pd.Timestamp('2026-10-30T12:00:00Z'))
    assert out['entry_time_utc'].startswith('2026-10-29T19:14') and out['entry_proxy'] == 2.3
    assert round(out['mfe_pct_vs_entry'], 2) == round(100 * (4.0 / 2.3 - 1), 2)
    assert out['status'] == 'censored_unexpired' and out['final_basis'] == 'last_option_print'
    assert round(out['rule_a_return_pct'], 2) == round(100 * (3.9 / 2.3 - 1), 2)


def test_first_fill_through_the_stated_stop_is_not_scored_as_a_stop_exit() -> None:
    opt = bars('2026-10-29T19:14:00Z', [(1.9, 1.9, 1.8, 1.85), (1.85, 2.5, 1.8, 2.4)])
    out = score(ALERT, opt, None, pd.Timestamp('2026-10-30T12:00:00Z'))
    assert out['rule_scoring'] == 'unscorable_entry_at_or_below_stated_stop' and 'rule_a_return_pct' not in out


def test_expired_contract_settles_on_underlying_intrinsic() -> None:
    opt = bars('2026-11-02T14:00:00Z', [(2.3, 2.3, 2.3, 2.3), (2.3, 2.35, 2.25, 2.3), (2.3, 2.31, 2.29, 2.3)])
    und = bars('2026-11-02T20:58:00Z', [(760.0, 760.0, 760.0, 760.0), (759.0, 759.0, 759.0, 759.0)])
    out = score({**ALERT, 'stop': None, 'posted_at_utc': '2026-11-02T13:58:00Z'}, opt, und, pd.Timestamp('2026-11-03T12:00:00Z'))
    assert out['final_basis'] == 'expiry_intrinsic_from_underlying' and out['final_value'] == 6.0
