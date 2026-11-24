"""13:00 ET half-days: the calendar knows them, and the live modules' end-of-day
rules fire the same distance before the real close (Fri 2026-11-27 is one).

The early-close list was checked against Alpaca's /v2/calendar on 2026-11-23:
every session 2018-2027 whose close is not 16:00 closes at 13:00, on exactly the
dates below.
"""
from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from core.calendar import (
    close_relative_time,
    early_closes,
    is_market_open_now,
    session_close_time,
)

ET = ZoneInfo("America/New_York")
HALF_DAY = date(2026, 11, 27)      # day after Thanksgiving
THANKSGIVING = date(2026, 11, 26)
EVE = date(2026, 11, 25)           # normal session before the holiday

ALPACA_EARLY_CLOSES_2018_2027 = [
    "2018-07-03", "2018-11-23", "2018-12-24", "2019-07-03", "2019-11-29",
    "2019-12-24", "2020-11-27", "2020-12-24", "2021-11-26", "2022-11-25",
    "2023-07-03", "2023-11-24", "2024-07-03", "2024-11-29", "2024-12-24",
    "2025-07-03", "2025-11-28", "2025-12-24", "2026-11-27", "2026-12-24",
    "2027-11-26",
]


def _et(d: date, hh: int, mm: int) -> datetime:
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=ET)


# -- calendar -----------------------------------------------------------------

def test_early_closes_match_alpaca_calendar():
    got = sorted(str(d) for y in range(2018, 2028) for d in early_closes(y))
    assert got == ALPACA_EARLY_CLOSES_2018_2027


def test_session_close_time():
    assert session_close_time(HALF_DAY) == time(13, 0)
    assert session_close_time(EVE) == time(16, 0)


def test_close_relative_time_shifts_only_on_half_days():
    assert close_relative_time(time(15, 40), EVE) == time(15, 40)
    assert close_relative_time(time(15, 40), HALF_DAY) == time(12, 40)
    assert close_relative_time(time(13, 0), HALF_DAY) == time(10, 0)
    assert close_relative_time(time(9, 45), HALF_DAY) == time(9, 30)  # never before the open


def test_market_open_respects_early_close_and_holiday():
    assert is_market_open_now(_et(HALF_DAY, 12, 59))
    assert not is_market_open_now(_et(HALF_DAY, 13, 0))
    assert not is_market_open_now(_et(THANKSGIVING, 12, 0))
    assert is_market_open_now(_et(EVE, 15, 59))


# -- SPY daytrader policy -----------------------------------------------------

def _spy_policy(submit_orders: bool = False, **over):
    from strategies.spy_intraday.Policy.order_policy import OptionOrderPolicy, OptionOrderPolicyConfig

    cfg = OptionOrderPolicyConfig(
        submit_orders=submit_orders, max_live_entry_lag_sec=0.0, use_wall_clock_entry_cutoff=False, **over
    )
    return OptionOrderPolicy(cfg)


def test_spy_expiry_skips_thanksgiving_and_rolls_early_on_the_half_day():
    pol = _spy_policy(dte_cutoff_hhmm="13:00")
    assert pol._resolve_expiration(_et(EVE, 14, 0)) == HALF_DAY          # not the closed 11-26
    assert pol._resolve_expiration(_et(HALF_DAY, 9, 45)) == HALF_DAY     # 0DTE before 10:00
    assert pol._resolve_expiration(_et(HALF_DAY, 10, 30)) == date(2026, 11, 30)
    assert pol._resolve_expiration(_et(EVE, 12, 0)) == EVE               # normal day unchanged


def test_spy_new_entry_cutoff_moves_with_the_close():
    pol = _spy_policy(option_new_entry_cutoff_hhmm="15:00")
    assert pol._new_entries_allowed(local_ts=_et(HALF_DAY, 11, 55))
    assert not pol._new_entries_allowed(local_ts=_et(HALF_DAY, 12, 0))
    assert pol._new_entries_allowed(local_ts=_et(EVE, 14, 55))


def test_spy_expiring_flatten_fires_before_the_half_day_close():
    pol = _spy_policy(submit_orders=True, expiring_position_exit_hhmm="15:40")
    submitted = []

    class _Client:
        def cancel_order(self, order_id):
            return {"id": order_id, "status": "canceled"}

        def submit_option_order(self, **kwargs):
            submitted.append(kwargs)
            return {"id": "m1", "status": "accepted", "filled_qty": "1", "filled_avg_price": "0.05"}

    pol._client = _Client()
    pol.sync_from_broker = lambda **_kw: {}  # type: ignore[method-assign]
    pol._long_contracts = 1
    pol._long_symbol = "SPY261127C00700000"
    assert pol._maybe_force_close_expiring_positions(local_ts=_et(HALF_DAY, 12, 39), logger=lambda *_: None) is None
    assert pol._maybe_force_close_expiring_positions(local_ts=_et(HALF_DAY, 12, 40), logger=lambda *_: None)
    assert submitted and submitted[0]["symbol"] == "SPY261127C00700000"


# -- Multi-ticker swing -------------------------------------------------------

def test_multi_ticker_expiry_reference_date_on_the_half_day():
    from strategies.multi_ticker_swing.live.runner import _entry_contract_ref_date

    assert _entry_contract_ref_date(_et(HALF_DAY, 9, 45)) == HALF_DAY
    assert _entry_contract_ref_date(_et(HALF_DAY, 10, 0)) == date(2026, 12, 1)  # Friday-late rule
    assert _entry_contract_ref_date(_et(date(2026, 11, 20), 12, 0)) == date(2026, 11, 20)


def test_multi_ticker_close_cutoffs_on_the_half_day():
    from strategies.multi_ticker_swing.live.position_manager import (
        _past_expiring_itm_close_cutoff,
        _should_defer_trail_exit,
    )

    assert not _past_expiring_itm_close_cutoff(_et(HALF_DAY, 12, 44))
    assert _past_expiring_itm_close_cutoff(_et(HALF_DAY, 12, 45))
    assert not _past_expiring_itm_close_cutoff(_et(EVE, 12, 45))
    assert _past_expiring_itm_close_cutoff(_et(EVE, 15, 45))
    assert _should_defer_trail_exit({"timestamp": _et(HALF_DAY, 12, 55)})
    assert not _should_defer_trail_exit({"timestamp": _et(EVE, 12, 55)})


# -- Dealer positioning -------------------------------------------------------

def test_dealer_runner_market_hours_and_next_open():
    from strategies.dealer_positioning.runner import _is_market_hours, _next_market_open

    assert not _is_market_hours(_et(THANKSGIVING, 12, 0))
    assert _is_market_hours(_et(HALF_DAY, 13, 0))
    assert not _is_market_hours(_et(HALF_DAY, 13, 1))
    assert _next_market_open(_et(EVE, 17, 0)) == _et(HALF_DAY, 9, 30)


def test_near_close_schedule_moves_with_the_close_and_skips_the_holiday():
    from UI.nightly_scheduler import next_run_at

    assert next_run_at(_et(HALF_DAY, 9, 0), 15, 15, close_relative=True) == _et(HALF_DAY, 12, 15)
    assert next_run_at(_et(EVE, 16, 0), 15, 15, close_relative=True) == _et(HALF_DAY, 12, 15)
    assert next_run_at(_et(HALF_DAY, 12, 30), 15, 15, close_relative=True) == _et(date(2026, 11, 30), 15, 15)
    # Default scheduling is unchanged (still weekday-only, holiday-blind).
    assert next_run_at(_et(EVE, 16, 0), 15, 15) == _et(THANKSGIVING, 15, 15)


def test_dealer_capture_boundary_is_session_relative():
    from core.nervous_system.config.freshness import META_4H_1620_PROFILE
    from core.nervous_system.context.requirements import _profile_boundary

    assert _profile_boundary(META_4H_1620_PROFILE, _et(HALF_DAY, 16, 20)) == _et(HALF_DAY, 11, 20)
    assert _profile_boundary(META_4H_1620_PROFILE, _et(EVE, 16, 20)) == _et(EVE, 14, 20)


# -- Bar filters: thin post-market prints after a 13:00 close are not RTH -------

def test_multi_ticker_session_windows_end_at_the_half_day_close():
    from strategies.multi_ticker_swing.live.session import (
        is_regular_trading_time,
        should_check_confirmation,
        should_scan_after_30m_close,
    )

    assert is_regular_trading_time(_et(HALF_DAY, 12, 59))
    assert not is_regular_trading_time(_et(HALF_DAY, 13, 0))
    assert is_regular_trading_time(_et(EVE, 15, 59))
    assert should_check_confirmation(_et(HALF_DAY, 12, 55))
    assert not should_check_confirmation(_et(HALF_DAY, 13, 30))
    assert not should_scan_after_30m_close(_et(HALF_DAY, 12, 55))
    assert should_scan_after_30m_close(_et(EVE, 15, 30))


def test_spy_regular_session_bar_filter_and_last_minute():
    from core.API.Alpaca_API.runners.live_runner import _bar_in_regular_session, _bar_is_session_last_minute

    def bar(d, hh, mm):
        return {"timestamp": _et(d, hh, mm).isoformat()}

    assert _bar_in_regular_session(bar(HALF_DAY, 12, 59))
    assert not _bar_in_regular_session(bar(HALF_DAY, 13, 0))
    assert _bar_in_regular_session(bar(EVE, 15, 59))
    assert _bar_is_session_last_minute(bar(HALF_DAY, 12, 59))
    assert not _bar_is_session_last_minute(bar(HALF_DAY, 15, 59))
    assert _bar_is_session_last_minute(bar(EVE, 15, 59))


def test_stream_watchdog_window_follows_the_session():
    from UI.shared_stream import SharedBarStream

    window = SharedBarStream._is_rth_watch_window
    assert window(_et(HALF_DAY, 13, 5))
    assert not window(_et(HALF_DAY, 13, 6))
    assert not window(_et(THANKSGIVING, 12, 0))
    assert window(_et(EVE, 16, 5))


# -- Thanksgiving (full holiday): recognized, nothing trades, nothing breaks ----

def test_thanksgiving_is_a_closed_day_and_sessions_skip_it():
    from core.calendar import is_trading_day, latest_session_date, next_trading_day, prev_trading_day

    assert not is_trading_day(THANKSGIVING)
    assert latest_session_date(_et(THANKSGIVING, 12, 0)) == EVE
    assert next_trading_day(EVE) == HALF_DAY
    assert prev_trading_day(HALF_DAY) == EVE


def test_trading_jobs_skip_thanksgiving_but_data_jobs_still_run():
    from UI.nightly_scheduler import next_run_at

    wed_close = _et(EVE, 16, 30)
    # 4H loops, pre-open flushes and the startup queue: next run is Friday.
    assert next_run_at(wed_close, 14, 20, trading_days_only=True) == _et(HALF_DAY, 14, 20)
    assert next_run_at(wed_close, 9, 35, trading_days_only=True) == _et(HALF_DAY, 9, 35)
    # Data jobs keep the weekday schedule (nightly news still collected Thursday).
    assert next_run_at(_et(THANKSGIVING, 0, 5), 16, 45) == _et(THANKSGIVING, 16, 45)


def test_trading_jobs_are_wired_trading_days_only():
    """Audit the combined server's call sites, like test_broker_snapshot_schedule."""
    import ast
    import inspect

    import UI.combined_server as combined_server

    tree = ast.parse(inspect.getsource(combined_server))
    flags: dict[str, dict[str, object]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        kws = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        if node.func.id == "_schedule_loop" and isinstance(kws.get("label"), ast.Constant):
            label = kws["label"].value
        elif node.func.id == "NightlyScheduler" and isinstance(kws.get("name"), ast.JoinedStr):
            label = "".join(p.value for p in kws["name"].values if isinstance(p, ast.Constant))
        else:
            continue
        flags[label] = {k: v.value for k, v in kws.items() if isinstance(v, ast.Constant)}
    for label in ("HTF Swing", "Momentum", "Meta pre-open flush", "HTF pre-open flush",
                  "Momentum pre-open flush", "Startup queue (market hours)", "meta-ranker-scheduler-"):
        assert flags[label].get("trading_days_only") is True, label
    assert flags["Dealer Ranker"].get("close_relative") is True


def test_thursday_readiness_job_skips_and_friday_gate_passes(tmp_path):
    import json

    from core.live_readiness import readiness_status

    stamp = tmp_path / "latest_success.json"
    stamp.write_text(json.dumps({
        "completed_at_utc": _et(EVE, 23, 30).isoformat(), "job": "nightly_data_readiness",
        "status": "success", "version": 1,
    }))
    assert readiness_status(path=stamp, now=_et(THANKSGIVING, 22, 15), for_next_session=True)[0]
    assert readiness_status(path=stamp, now=_et(HALF_DAY, 9, 0))[0]
    assert not readiness_status(path=stamp, now=_et(HALF_DAY, 22, 15), for_next_session=True)[0]


def test_wednesday_deferred_entries_survive_to_friday_flush():
    from core.live_4h_exec import pending_entry_bar_is_stale

    wed_bar = "2026-11-25 14:00:00-05:00"
    assert not pending_entry_bar_is_stale(wed_bar, now=_et(HALF_DAY, 9, 35))
    assert pending_entry_bar_is_stale(wed_bar, now=_et(date(2026, 11, 30), 9, 35))
