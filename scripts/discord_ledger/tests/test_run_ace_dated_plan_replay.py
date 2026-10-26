import pandas as pd

from scripts.discord_ledger.run_ace_dated_plan_replay import first_plan_trigger, next_minute_open, plot_dated_cases, timing_comparison


def _sessions(day="2026-10-07", opening=291, low=290, high=294):
    start = pd.Timestamp(f"{day}T13:30Z")
    frame = pd.DataFrame([{"timestamp": start, "decision_at_utc": start + pd.Timedelta(minutes=5),
                           "open": opening, "low": low, "high": high, "close": opening + 1}])
    return [(day, frame, {"prior_close": 302.56, "source_session_file": "fixture"})]


def _plan():
    return {"symbol": "ARM", "message_id": "plan", "available_at_utc": "2026-10-06T05:26Z", "entry_zone": [298, 298]}


def test_gap_through_does_not_manufacture_traded_price_or_fill():
    assert first_plan_trigger(_plan(), _sessions()) is None
    observed = first_plan_trigger(_plan(), _sessions(), gap_aware=True)
    assert observed["trigger_kind"] == "session_open_gap_crossing"
    assert observed["traded_through_zone"] is False
    assert observed["confirmed_fill_price"] is None
    assert observed["decision_at_utc"] == pd.Timestamp("2026-10-07T13:35Z").isoformat()


def test_full_path_first_hit_is_not_reset_on_a_later_notification_day():
    sessions = _sessions("2026-10-06", 299, 297, 300) + _sessions("2026-10-07", 299, 297, 300)
    observed = first_plan_trigger(_plan(), sessions)
    assert observed["session_date_et"] == "2026-10-06"


def test_candle_started_before_plan_availability_cannot_trigger():
    plan = {**_plan(), "available_at_utc": "2026-10-07T13:33Z"}
    assert first_plan_trigger(plan, _sessions(opening=299, low=297, high=300)) is None


def test_timing_comparison_retains_negative_early_notification_error():
    note = {"symbol": "ARM", "message_id": "hit", "plan_message_id": "plan",
            "kind": "underlying_zone_tag_claim", "available_at_utc": "2026-10-07T13:39:02Z"}
    result = timing_comparison(note, first_plan_trigger(_plan(), _sessions(), gap_aware=True), "gap")
    assert -5 < result["signed_model_minus_notification_minutes"] < -4
    assert result["matched"]["5"] is True
    assert result["source_message_ids"] == ["plan", "hit"]


def test_underlying_follower_reference_waits_until_rth_and_next_minute():
    raw = pd.DataFrame({"symbol": ["MU"] * 4,
                        "timestamp": ["2026-10-07T13:29Z", "2026-10-07T13:30Z", "2026-10-07T13:31Z", "2026-10-07T20:00Z"],
                        "open": [99, 100, 101, 105]})
    assert next_minute_open(raw, "MU", pd.Timestamp("2026-10-07T05:26Z"))["price_reference"] == 100
    assert next_minute_open(raw, "MU", pd.Timestamp("2026-10-07T13:30:30Z"))["price_reference"] == 101
    assert next_minute_open(raw, "MU", pd.Timestamp("2026-10-07T20:00Z")) is None


def test_candles_and_clock_markers_use_same_datetime_coordinates(tmp_path, monkeypatch):
    import core.shared_plotting as plotting
    original = plotting.plot_candles_from_frame
    called = []

    def checked(*args, **kwargs):
        assert kwargs["compressed"] is False
        result = original(*args, **kwargs)
        assert min(result.x) > 10000  # matplotlib dates, not zero-based candle positions
        called.append(True)
        return result

    monkeypatch.setattr(plotting, "plot_candles_from_frame", checked)
    sessions = _sessions(opening=299, low=297, high=300)
    note = {"symbol": "ARM", "message_id": "hit", "plan_message_id": "plan",
            "kind": "underlying_zone_tag_claim", "available_at_utc": "2026-10-07T13:39:02Z"}
    plot_dated_cases(tmp_path, [note], {"plan": _plan()}, {"ARM": sessions}, [], [])
    assert called == [True]
    assert (tmp_path / "dated_notification_and_policy_paths.png").exists()
