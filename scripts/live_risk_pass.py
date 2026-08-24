"""Between-bar risk pass for the 4H modules (paper by default).

Runs the non-model half of each module's exit policy — hard stop and expiry
flatten — on a short cadence instead of waiting for the next 4H bar. See
``core.live_risk_pass`` for why that breaks no research/live parity and for the
rules it deliberately refuses to evaluate.

  # dry run across every module (no orders)
  PYTHONPATH=. .venv/bin/python scripts/live_risk_pass.py
  # actually submit on paper
  PYTHONPATH=. .venv/bin/python scripts/live_risk_pass.py --submit
  # one module only
  PYTHONPATH=. .venv/bin/python scripts/live_risk_pass.py --module dealer_ranker --submit
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient
from core.live_4h_exec import ExecPolicy, execute_plan
from core.live_risk_pass import (
    RiskPassConfig,
    evaluate_risk_exits,
    load_state,
    module_state_lock,
    save_state,
)
from signals.meta_context.meta_ranker.options_exec import equity_order_tif

logger = logging.getLogger("live_risk_pass")
_ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class ModuleSpec:
    """Where a module keeps state, and the exit thresholds it actually runs.

    These MUST mirror each runner's argparse defaults. A risk pass that stops at
    a different level from its own 4H runner is worse than no risk pass: it
    would fire exits the module never asked for. Dealer Ranker is the one that
    differs (50% stop, 20% take-profit) because it kept its pre-2026-07 policy.
    """

    module: str
    state_path: Path
    policy: ExecPolicy


MODULES: dict[str, ModuleSpec] = {
    "meta_ranker": ModuleSpec(
        module="meta_ranker",
        state_path=REPO / "signals/meta_context/meta_ranker/live_state.json",
        policy=ExecPolicy(take_profit=0.30, scale_frac=0.16, horizon_bars=53,
                          stop_loss=0.39, trail_stop=None),
    ),
    "momentum_expansion": ModuleSpec(
        module="momentum_expansion",
        state_path=REPO / "strategies/momentum_expansion/live/momentum_live_state.json",
        policy=ExecPolicy(take_profit=0.30, scale_frac=0.16, horizon_bars=53,
                          stop_loss=0.39, trail_stop=None),
    ),
    "multi_ticker_swing_htf": ModuleSpec(
        module="multi_ticker_swing_htf",
        state_path=REPO / "strategies/multi_ticker_swing_htf/live/htf_live_state.json",
        policy=ExecPolicy(take_profit=0.30, scale_frac=0.16, horizon_bars=53,
                          stop_loss=0.39, trail_stop=None),
    ),
    "dealer_ranker": ModuleSpec(
        module="dealer_ranker",
        state_path=REPO / "Data/inference/dealer_ranker/live_state.json",
        policy=ExecPolicy(take_profit=0.20, scale_frac=0.5, horizon_bars=25,
                          stop_loss=0.50, trail_stop=0.35),
    ),
}


def build_pos_info(client) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in client.get_positions() or []:
        try:
            out[str(p["symbol"]).upper()] = {
                "qty": int(float(p["qty"])),
                "avg_entry": float(p.get("avg_entry_price", 0) or 0),
                "current": float(p.get("current_price", 0) or 0),
            }
        except Exception:  # noqa: BLE001
            continue
    return out


def run_module(spec: ModuleSpec, *, client, pos_info: dict, cfg: RiskPassConfig,
               now_et: dt.datetime, submit: bool) -> dict:
    """One module's risk pass. Returns a small summary for logging."""
    with module_state_lock(spec.module) as acquired:
        if not acquired:
            return {"module": spec.module, "skipped": "state_locked"}

        state = load_state(spec.state_path)
        managed = state.get("managed", {}) or {}
        if not managed:
            return {"module": spec.module, "managed": 0, "orders": 0}

        res = evaluate_risk_exits(
            client, module=spec.module, managed=managed, pos_info=pos_info,
            policy=spec.policy, now_et=now_et, cfg=cfg)

        for sym, side, qty, reason, route in res.plan:
            print(f"  {spec.module:24s} {side.upper():4s} {qty:>6} {sym:<22} [{reason}]")

        if res.plan and submit:
            execute_plan(
                client, plan=res.plan, limits={}, submit=True,
                equity_tif_fn=equity_order_tif,
                new_managed=res.new_managed, exit_context=res.exit_context,
                module=spec.module, pos_lookup=pos_info,
                bar=now_et.isoformat(),
            )

        # Only persist when this pass actually changed something. A read-only
        # tick must not rewrite state the 4H runner owns. Clearing an entry's
        # unconfirmed flag counts: it is the whole point of settling it here
        # rather than waiting for the next 4H bar, and an in-place mutation that
        # is never written back would settle nothing.
        if submit and (res.plan or res.settled or res.confirmed_entries):
            state["managed"] = res.new_managed
            save_state(spec.state_path, state)

        return {
            "module": spec.module,
            "managed": len(managed),
            "orders": len(res.plan),
            "settled": len(res.settled),
            "confirmed_entries": len(res.confirmed_entries),
            "anomalies": len(res.anomalies),
            "skipped_positions": len(res.skipped),
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--module", choices=sorted(MODULES), action="append",
                    help="Limit to one module (repeatable). Default: all.")
    ap.add_argument("--submit", action="store_true",
                    help="Actually place orders. Without it this is a dry run.")
    ap.add_argument("--live", action="store_true",
                    help="Use the LIVE account instead of paper. Refused unless "
                         "--i-understand-live is also given.")
    ap.add_argument("--i-understand-live", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--no-hard-stop", action="store_true",
                    help="Disable the hard-stop rule (leaves expiry flatten on).")
    ap.add_argument("--no-expiry-flatten", action="store_true",
                    help="Disable the expiry flatten (leaves the hard stop on).")
    ap.add_argument("--trailing-stop", action="store_true",
                    help="OPT-IN: evaluate the trailing stop between bars. Path-dependent — "
                         "it ratchets off the observed peak, so a faster cadence triggers "
                         "earlier than the 4H calibration. Backtest at this cadence first.")
    ap.add_argument("--take-profit-trim", action="store_true",
                    help="OPT-IN: evaluate the take-profit trim between bars. Same caveat.")
    ap.add_argument("--expiry-cutoff", default="15:45",
                    help="ET time after which an option on its last tradable session is "
                         "flattened (default 15:45, matching the swing sleeve).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.live and not args.i_understand_live:
        print("ERROR: --live requires --i-understand-live.", file=sys.stderr)
        return 2

    try:
        hh, mm = (int(x) for x in str(args.expiry_cutoff).split(":", 1))
    except Exception:  # noqa: BLE001
        print(f"ERROR: bad --expiry-cutoff {args.expiry_cutoff!r}; want HH:MM", file=sys.stderr)
        return 2

    cfg = RiskPassConfig(
        hard_stop=not args.no_hard_stop,
        expiry_flatten=not args.no_expiry_flatten,
        trailing_stop=bool(args.trailing_stop),
        take_profit_trim=bool(args.take_profit_trim),
        expiry_cutoff_et=(hh, mm),
    )
    if not (cfg.hard_stop or cfg.expiry_flatten or cfg.trailing_stop or cfg.take_profit_trim):
        print("ERROR: every rule is disabled; nothing to do.", file=sys.stderr)
        return 2

    now_et = dt.datetime.now(_ET)
    specs = [MODULES[m] for m in (args.module or sorted(MODULES))]

    profile = "LIVE" if args.live else "PAPER"
    client = AlpacaOptionsClient(env_file=f".env#{profile}")
    pos_info = build_pos_info(client)

    mode = profile.lower() if profile == "PAPER" else profile
    print(f"risk pass {now_et:%Y-%m-%d %H:%M:%S %Z} [{mode}/"
          f"{'SUBMIT' if args.submit else 'dry-run'}] "
          f"rules: stop={cfg.hard_stop} expiry={cfg.expiry_flatten} "
          f"trail={cfg.trailing_stop} tp={cfg.take_profit_trim}")
    print(f"  broker positions: {len(pos_info)}")

    total_orders = 0
    for spec in specs:
        try:
            summary = run_module(spec, client=client, pos_info=pos_info, cfg=cfg,
                                 now_et=now_et, submit=args.submit)
        except Exception as exc:  # noqa: BLE001 - one module must not stop the rest
            logger.exception("risk pass: %s failed: %s", spec.module, exc)
            continue
        total_orders += int(summary.get("orders", 0) or 0)
        logger.info("risk pass summary: %s", summary)

    print(f"  orders {'submitted' if args.submit else 'planned'}: {total_orders}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
