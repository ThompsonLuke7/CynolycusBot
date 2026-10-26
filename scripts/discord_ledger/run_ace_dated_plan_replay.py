"""Score dated, typed October plans without giving alert clocks to the engine.

The conditional arm consumes prepublished chart zones, not an independently
recreated watchlist. Range-touch and gap-crossing are separate observations,
neither is a fill. Existing entry/position policies run continuously in parallel
as diagnostics. Source clocks are consulted only after paths are generated.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from scripts.discord_ledger.build import digest, read_jsonl, write_jsonl
from scripts.discord_ledger.run_ace_sequence_replay import (
    Policy, completed_candles, one_to_one_matches, prepare_daily_bars,
    replay_symbol, session_bounds, session_daily_features, utc,
)

DAYS = ("2026-10-06", "2026-10-07", "2026-10-08")


def first_plan_trigger(plan: dict, sessions: list, *, gap_aware: bool = False,
                       target: float | None = None) -> dict | None:
    """First completed-candle observation over the full path after publication.

    Never choose the alert's session, use its timestamp, or impute missing bars.
    A gap crossing is not a trade through the zone, even if it tags a threshold.
    """
    zone = [target, target] if target is not None else plan["entry_zone"]
    lo, hi = zone
    for day, frame, daily in sessions:
        start, _ = session_bounds(day)
        for bar in frame.to_dict("records"):
            if utc(bar["timestamp"]) < utc(plan["available_at_utc"]):
                continue
            touch = bar["low"] <= hi and bar["high"] >= lo
            gap = (bar["timestamp"] == start and
                   min(daily["prior_close"], bar["open"]) <= hi and
                   max(daily["prior_close"], bar["open"]) >= lo and not touch)
            if not touch and not (gap_aware and gap):
                continue
            return {"symbol": plan["symbol"], "decision_at_utc": bar["decision_at_utc"].isoformat(),
                    "bar_start_utc": bar["timestamp"].isoformat(), "session_date_et": day,
                    "trigger_kind": "session_open_gap_crossing" if gap else "traded_range_intersection",
                    "traded_through_zone": bool(touch), "price_reference": bar["close"],
                    "zone": zone, "source_message_ids": [plan["message_id"]],
                    "source_session_file": daily["source_session_file"],
                    "confirmed_fill_price": None}
    return None


def timing_comparison(notification: dict, trigger: dict | None, variant: str) -> dict:
    delta = None if trigger is None else (utc(trigger["decision_at_utc"]) -
                                         utc(notification["available_at_utc"])).total_seconds() / 60
    return {"symbol": notification["symbol"], "notification_message_id": notification["message_id"],
            "notification_kind": notification["kind"], "notification_at_utc": notification["available_at_utc"],
            "plan_message_id": notification["plan_message_id"], "variant": variant,
            "source_message_ids": [notification["plan_message_id"], notification["message_id"]],
            "trigger": trigger, "signed_model_minus_notification_minutes": delta,
            "matched": {str(t): delta is not None and abs(delta) <= t for t in (5, 10, 15)},
            "status": "trigger_not_observed" if trigger is None else "descriptive_notification_timing_not_fill",
            "confidence": "high"}


def next_minute_open(raw: pd.DataFrame, symbol: str, ready: pd.Timestamp) -> dict | None:
    frame = raw[raw.symbol.eq(symbol)].copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    local_day = ready.tz_convert("America/New_York").date().isoformat()
    start, end = session_bounds(local_day)
    frame = frame[(frame.timestamp >= max(start, ready)) & (frame.timestamp < end)].sort_values("timestamp")
    if frame.empty:
        return None
    b = frame.iloc[0]
    return {"bar_start_utc": b.timestamp.isoformat(), "price_reference": float(b.open),
            "price_basis": "underlying_next_observed_minute_open_benchmark_not_quote_or_fill"}


def plot_dated_cases(out: Path, notifications: list[dict], plans: dict,
                     symbol_sessions: dict, comparisons: list[dict], events: list[dict]) -> None:
    """Every scored notification, full RTH session, exact clock markers in ET."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_ace")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from core.shared_plotting import plot_candles_from_frame, save_figure, setup_datetime_axis, to_mpl_time
    fig, axes = plt.subplots(len(notifications), 1, figsize=(16, 4 * len(notifications)), squeeze=False)
    for ax, note in zip(axes[:, 0], notifications):
        day = utc(note["available_at_utc"]).tz_convert("America/New_York").date().isoformat()
        frame = next(f for d, f, _ in symbol_sessions[note["symbol"]] if d == day)
        plot_candles_from_frame(ax, frame, time_col="decision_at_utc", compressed=False, label=False)
        setup_datetime_axis(ax, date_format="%H:%M")
        plan = plans[note["plan_message_id"]]
        lo, hi = (note["target"], note["target"]) if note.get("target") else plan["entry_zone"]
        ax.axhspan(lo, hi, alpha=.15, color="#d97706")
        ax.axhline(hi, color="#d97706", linewidth=1, label=f"Prepublished {'target' if note.get('target') else 'zone'} {lo:g}–{hi:g}")
        ax.axvline(to_mpl_time([note["available_at_utc"]])[0], color="#7c3aed", linestyle="--", label="ACE notification publication")
        for comparison in comparisons:
            if comparison["notification_message_id"] != note["message_id"] or not comparison["trigger"]:
                continue
            trigger = comparison["trigger"]
            ax.axvline(to_mpl_time([trigger["decision_at_utc"]])[0], color="#0891b2" if comparison["variant"] == "range_only" else "#dc2626",
                       alpha=.65, linestyle=":" if comparison["variant"] == "range_only" else "-.", label=comparison["variant"])
        for event in events:
            if event["policy"] != "published_entry_zone" or event["symbol"] != note["symbol"] or utc(event["decision_at_utc"]).tz_convert("America/New_York").date().isoformat() != day:
                continue
            marker, color = {"entry": ("^", "#0891b2"), "trim": ("s", "#d97706"), "exit": ("x", "#dc2626")}[event["kind"]]
            ax.scatter(to_mpl_time([event["decision_at_utc"]]), [event["price_reference"]], marker=marker, color=color,
                       s=65, zorder=4, label=f"Reaction policy {event['kind']} (not a fill)")
        start, end = session_bounds(day)
        ax.set_xlim(to_mpl_time([start, end]))
        ax.set_title(f"{note['symbol']} | {day} | {note['kind']} | source {note['message_id']} | full session")
        ax.set_ylabel("Underlying price")
        ax.set_xlabel("Completed five-minute candles, Eastern time; missing minutes not filled")
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=8, loc="best")
        ax.grid(alpha=.15)
    save_figure(fig, out / "dated_notification_and_policy_paths.png", dpi=130, close=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--bars-dir", type=Path, required=True)
    parser.add_argument("--daily-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    evidence = read_jsonl(args.evidence)
    plans = {r["message_id"]: r for r in evidence if r.get("kind") is None and "targets" in r}
    notifications = [r for r in evidence if r.get("kind") in {"underlying_zone_tag_claim", "underlying_target_tag_claim"}]
    symbol_sessions, coverage, daily_inputs = {}, [], []
    raw = {day: pd.read_parquet(args.bars_dir / f"{day}.parquet") for day in DAYS}
    for symbol in sorted({r["symbol"] for r in evidence} - {"GEV"}):
        daily = prepare_daily_bars(pd.read_parquet(args.daily_dir / f"{symbol}.parquet"))
        sessions = []
        for day in DAYS:
            path = args.bars_dir / f"{day}.parquet"
            candles, cov = completed_candles(raw[day], symbol, day)
            feat = session_daily_features(daily, day, candles, str(path))
            if feat["daily_feature_status"] != "ok":
                raise ValueError(f"Unusable daily input: {symbol} {day} {feat['daily_feature_status']}")
            sessions.append((day, candles, feat))
            coverage.append(cov)
            daily_inputs.append(dict(feat, symbol=symbol, session_date_et=day))
        symbol_sessions[symbol] = sessions
    comparisons = []
    for note in notifications:
        plan = plans[note["plan_message_id"]]
        if utc(plan["available_at_utc"]) >= utc(note["available_at_utc"]):
            raise ValueError("Future plan cannot label an earlier notification")
        for gap_aware in (False, True):
            trigger = first_plan_trigger(plan, symbol_sessions[plan["symbol"]], gap_aware=gap_aware,
                                         target=note.get("target"))
            comparisons.append(timing_comparison(note, trigger, "range_or_session_gap" if gap_aware else "range_only"))
    policies = [Policy("published_entry_zone", "published", 2),
                Policy("sma5_touch_reaction_swing", "daily", 2, ("sma_5",)),
                Policy("ema21_touch_reaction_swing", "daily", 2, ("ema_21",)),
                Policy("intraday_retest_reaction", "intraday")]
    model_plans = [dict(p, source_message_ids=[p["message_id"]]) for p in plans.values() if p.get("entry_zone")]
    events, trades, traces = [], [], []
    for symbol, sessions in symbol_sessions.items():
        for policy in policies:
            es, ts, tr = replay_symbol(symbol, sessions, policy, model_plans)
            events.extend(es); trades.extend(ts); traces.extend(tr)
    labels = [{"label_id": n["message_id"], "kind": "entry", "symbol": n["symbol"],
               "direction_sign": 1, "decision_at_utc": n["available_at_utc"],
               "session_date_et": utc(n["available_at_utc"]).tz_convert("America/New_York").date().isoformat(),
               "source_message_ids": [n["message_id"]], "label_role": "zone_notification_not_actual_entry"}
              for n in notifications if n["kind"] == "underlying_zone_tag_claim"]
    entry_diagnostics = {}
    for policy in policies:
        candidates = [e for e in events if e["policy"] == policy.name and e["kind"] == "entry"]
        entry_diagnostics[policy.name] = {"generated_entries": len(candidates),
            "zone_notification_timing_overlap": {str(t): one_to_one_matches(labels, candidates, t) for t in (5, 10, 15)}}
    mu = [r for r in evidence if r["symbol"] == "MU"]
    buy = next(r for r in mu if r["kind"] == "option_buy_reference")
    trim = next(r for r in mu if r["kind"] == "partial_exit_claim_with_mark")
    follower, marks = [], []
    for r in mu:
        if r.get("option_mark") is not None:
            marks.append({"message_id": r["message_id"], "available_at_utc": r["available_at_utc"],
                          "option_mark": r["option_mark"], "buy_reference": buy["buy_reference"],
                          "reference_to_mark_change_pct": 100 * (r["option_mark"] / buy["buy_reference"] - 1),
                          "source_message_ids": [buy["message_id"], r["message_id"]],
                          "realized_return_pct": None, "reported_claim_pct": r.get("claimed_gain_pct")})
    for seconds in (30, 60, 300):
        entry = next_minute_open(raw["2026-10-07"], "MU", utc(buy["available_at_utc"]) + pd.Timedelta(seconds=seconds))
        exit_ref = next_minute_open(raw["2026-10-07"], "MU", utc(trim["available_at_utc"]) + pd.Timedelta(seconds=seconds))
        follower.append({"symbol": "MU", "delay_seconds": seconds, "hypothetical_entry": entry,
                         "partial_exit_time_benchmark": exit_ref,
                         "gross_underlying_change_pct": None if entry is None or exit_ref is None else
                             100 * (exit_ref["price_reference"] / entry["price_reference"] - 1),
                         "source_message_ids": [buy["message_id"], trim["message_id"]],
                         "source_session_file": str(args.bars_dir / "2026-10-07.parquet"),
                         "assumptions": "Read premarket intent; hypothetically enter underlying at first RTH minute open and compare to next minute open after partial-exit publication plus delay. No option fill, realized P&L, costs, quantities, or actual ACE entry inferred."})
    positions = {t["position_id"]: t for t in trades}
    management_valid = all(e["position_id"] in positions and
        utc(e["decision_at_utc"]) > utc(positions[e["position_id"]]["entry_at_utc"])
        for e in events if e["kind"] in {"trim", "exit"})
    if not management_valid:
        raise ValueError("Management precedes or lacks its own position entry")
    for name, rows in (("notification_timing", comparisons), ("algorithm_events", events),
                       ("algorithm_positions", trades), ("state_transitions", traces),
                       ("coverage", coverage), ("daily_features", daily_inputs),
                       ("mu_reference_marks", marks), ("mu_underlying_follower_benchmarks", follower)):
        write_jsonl(args.out / f"{name}.jsonl", rows)
    result = {"schema": "ace_dated_plan_replay_v1", "research_only": True,
              "universe_role": "source_watchlist_supplied_not_independent_candidate_selection",
              "session_dates_et": list(DAYS), "notification_count": len(notifications),
              "notification_matches": {variant: {str(t): sum(r["matched"][str(t)] for r in comparisons
                  if r["variant"] == variant) for t in (5, 10, 15)}
                  for variant in ("range_only", "range_or_session_gap")},
              "entry_diagnostics": entry_diagnostics,
              "actual_option_entry_clock_labels": 0, "confirmed_full_exit_labels": 0,
              "joint_actual_entry_exit_recall": None,
              "management_position_and_time_check": management_valid,
              "null_test_status": "descriptive_four_notification_case_study_not_an_inferential_null_test",
              "session_mde_pp": None,
              "mde_status": "Not estimable for this selected-case descriptive study; do not interpret zero timing overlaps as a rejected strategy. The unchanged historical timing benchmark has 28.7–30.6pp MDE for its two principal policies.",
              "power_note": "Three zone notifications on three sessions plus one target; too few for a timing edge inference. No statistical rejection or claimed superiority.",
              "script_sha256": digest(Path(__file__)), "evidence_sha256": digest(args.evidence),
              "dependency_code_sha256": {str(p): digest(p) for p in [
                  Path("scripts/discord_ledger/run_ace_sequence_replay.py"),
                  Path("scripts/discord_ledger/run_ace_state_machine_experiment.py"),
                  Path("core/calendar/us_market_calendar.py")]},
              "source_bar_sha256": {str(args.bars_dir / f"{d}.parquet"): digest(args.bars_dir / f"{d}.parquet") for d in DAYS},
              "daily_source_sha256": {str(args.daily_dir / f"{s}.parquet"): digest(args.daily_dir / f"{s}.parquet") for s in symbol_sessions},
              "vendor_publication_time_certified": False,
              "variant_status": "Fixed existing reaction policies; gap-aware notification semantics added after ARM inspection, exploratory not held-out."}
    (args.out / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = ["# ACE dated October plan replay", "",
             "The updated export resolves publication clocks in Eastern time. Plans were available before all scored notifications. The engine scans October 6–8 continuously; it is not given the notification date or clock. Chart levels are supplied by ACE, so this tests notification mechanics, not autonomous watchlist/level discovery.", "",
             "| Event | ACE publication (ET) | First range-only observation (ET) | First gap-aware observation (ET) | Gap-aware timing error |",
             "| --- | --- | --- | --- | ---: |"]
    local = lambda t: "not observed" if t is None else utc(t).tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M:%S")
    for note in notifications:
        rows = [r for r in comparisons if r["notification_message_id"] == note["message_id"]]
        clocks = [local(r["trigger"]["decision_at_utc"]) if r["trigger"] else "not observed" for r in rows]
        err = rows[1]["signed_model_minus_notification_minutes"]
        lines.append(f"| {note['symbol']} {note['kind']} [{note['message_id']}] | {local(note['available_at_utc'])} | {clocks[0]} | {clocks[1]} | {'not observed' if err is None else f'{err:+.2f} min'} |")
    lines += ["", "The gap-aware observer matches all three zone notifications within one five-minute candle and ALAB's TP1 notification within two. Range-only misses ARM at the one/two/three-candle tolerance: the session opened below 298 and traded back through it later. No 298 fill is inferred. This gap-aware variant was added after inspecting that discrepancy and needs new examples, not a success claim based on these same cases.", "",
              "## Entry and exit policy confluence", "",
              "Zone touches are not confirmation entries. These unchanged reaction policies own their entry, stop, targets, trims and exits through a position ID. Below are overlaps with zone-notification clocks, not recall against actual option entry clocks.", "",
              "| Reaction policy | All generated entries (six symbols, three sessions) | ±5m zone-clock overlaps | ±10m | ±15m |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for name, row in entry_diagnostics.items():
        counts = [len(row["zone_notification_timing_overlap"][str(t)]) for t in (5, 10, 15)]
        lines.append(f"| {name} | {row['generated_entries']} | {counts[0]}/3 | {counts[1]}/3 | {counts[2]}/3 |")
    lines += ["", "This is a selected-case diagnostic, not an inferential null test: its MDE is not estimable, and zero overlaps do not reject the strategy. The earlier historical entry-timing benchmark's main policies had 28.7–30.6 percentage-point MDEs. New variants and population claims need additional independent cases.", "",
              "No actual option entry-time label or full-close label is established for these October chains. MU's 01:26 buy reference is outside the regular option session and 'mostly out' is partial. Joint actual-entry/full-exit recall is explicitly unavailable, not zero. All generated management events pass the owning-position/prior-entry check.", "",
              "## MU evidence and estimated underlying follower scenario", "",
              "The export's 10:41 image shows 6.25, not the user's 10:45 image's 6.45; these are kept separate. Verified later images show 7.73, 8.15 and 8.95, with a 3.28 premarket Buy reference. Reference-to-mark changes are not realized returns. The 1000% text claim has a different/unresolved basis; the reviewed marks do not establish that return.", "",
              "| Delay | Hypothetical underlying entry | Partial-exit-time price benchmark | Gross underlying change |",
              "| --- | ---: | ---: | ---: |"]
    for r in follower:
        lines.append(f"| {r['delay_seconds']} sec | {r['hypothetical_entry']['price_reference']:.2f} | {r['partial_exit_time_benchmark']['price_reference']:.2f} | {r['gross_underlying_change_pct']:+.2f}% |")
    lines += ["", "This is an explicitly hypothetical first-RTH-entry/partial-publication-exit underlying benchmark, not ACE's realized trade or a tradable option fill estimate. Quantities, option execution, bid/ask, spread, cost, and realized P&L remain unestimated. Historical SIP OHLCV is used without missing-minute interpolation; vendor availability vintage is not independently certified.", "",
              "## Next testable variant", "",
              "Use the now-resolved prepublished plans as separate labels for candidate choice, support selection, threshold notification, confirmation entry and management. Test an independently generated SMA5/EMA21 plus prior-pivot support planner on new dated examples; retain gap-through as a separate armed state, not an assumed fill. Freeze selection before scoring. Do not impose BULL-only cycle or divergence vetoes from these conflicting panels. The private cycle/expected-move formulas remain unidentified; this hypothesis family is not exhausted."]
    plot_dated_cases(args.out, notifications, plans, symbol_sessions, comparisons, events)
    lines += ["", "[Full-session candles, exact source clocks and position-linked reaction signals](dated_notification_and_policy_paths.png). See the canonical JSONL for per-event source IDs, model position IDs, full state transitions and price references."]
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: result[k] for k in ("notification_matches", "management_position_and_time_check", "actual_option_entry_clock_labels")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
