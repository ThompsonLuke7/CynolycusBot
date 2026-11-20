import numpy as np
import pandas as pd

from scripts.discord_ledger.score_ace_lifecycles import control_entries, osi_for, policy_values

ET = 'America/New_York'


def bars(prices, start='2026-03-02 09:31'):
    idx = pd.date_range(pd.Timestamp(start, tz=ET), periods=len(prices), freq='1min').tz_convert('UTC')
    p = np.asarray(prices, dtype=float)
    return pd.DataFrame({'open': p, 'high': p, 'low': p, 'close': p}, index=idx)


def test_osi_maps_spx_to_weeklies():
    assert osi_for({'symbol': 'SPX', 'expiry': '2026-03-02', 'option_type': 'put', 'strike': 6800}) == 'SPXW260302P06800000'
    assert osi_for({'symbol': 'SPY', 'expiry': '2026-03-06', 'option_type': 'call', 'strike': 681.5}) == 'SPY260306C00681500'


def test_stated_exits_trim_half_then_exit_rest_one_minute_after_each_post():
    opt = bars([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 0.5])
    t = opt.index
    expiry_t = t[-1] + pd.Timedelta(hours=1)
    v = policy_values(opt, t[0], 1.0, [('trim', t[1]), ('exit', t[3])], 0.0, expiry_t, index=False)
    assert v['stated_exits'] == 0.5 * 2.0 + 0.5 * 3.0  # trim fills at t[2], exit at t[4]
    assert v['hold_to_expiry'] == 0.0
    assert v['double_or_expiry'] == 2.0
    assert v['flat_same_session'] == 0.5


def test_unexpired_remainder_is_censored_but_a_full_exit_is_not():
    opt = bars([1.0, 1.2, 1.4])
    t = opt.index
    expiry_t = t[-1] + pd.Timedelta(days=5)
    assert policy_values(opt, t[0], 1.0, [], None, expiry_t, index=False)['stated_exits'] is None
    full = policy_values(opt, t[0], 1.0, [('exit', t[0] + pd.Timedelta(seconds=30))], None, expiry_t, index=False)
    assert full['stated_exits'] == 1.4


def test_control_entries_stay_in_session_and_away_from_the_real_entry():
    opt = bars([1.0] * 390, start='2026-03-02 09:30')
    entry_t = opt.index[100]
    picks = control_entries(opt, entry_t, np.random.default_rng(0))
    assert len(picks) == 20
    local = [t.tz_convert(ET) for t, _ in picks]
    assert all(pd.Timestamp('09:31').time() <= x.time() <= pd.Timestamp('15:30').time() for x in local)
    assert all(abs(t - entry_t) >= pd.Timedelta(minutes=5) for t, _ in picks)
