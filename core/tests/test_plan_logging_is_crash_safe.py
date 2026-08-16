"""A diagnostic print must never be able to abort a trading run.

2026-08-10, Dealer Ranker: `_select_atm_option` deliberately falls back to the
strike band when an expiry has no usable greeks ("delta_pool or same_exp"), so a
selected contract legitimately arrives with delta=None. build_mixed_plan's
verbose routing line formatted it bare —

    f"delta={order.get('delta'):.2f} mid={order['mid']:.2f} ..."

— which raised `TypeError: unsupported format string passed to NoneType.__format__`
inside build_mixed_plan, propagated through main() and killed the run. UMC 20C
x60 and GFI 40C x18 had already been selected and printed. Zero orders were
submitted for the session because of a logging statement.
"""
from __future__ import annotations

from core.live_4h_exec import ExecPolicy, _fmt_num, build_mixed_plan


def test_fmt_num_formats_numbers():
    assert _fmt_num(0.3812) == "0.38"
    assert _fmt_num(2) == "2.00"
    assert _fmt_num("1.5") == "1.50"


def test_fmt_num_degrades_instead_of_raising():
    assert _fmt_num(None) == "n/a"
    assert _fmt_num("abc") == "n/a"
    assert _fmt_num(object()) == "n/a"


def _route_with(order):
    return lambda *a, **k: ("option", order, "ok")


def _plan(order, *, verbose):
    return build_mixed_plan(
        None, targets=["UMC"], managed={}, pos_info={}, bar="2026-08-10 18:00:00+00:00",
        signal_audits={}, policy=ExecPolicy(), route_fn=_route_with(order),
        ref_price_fn=lambda _t: 19.5, verbose=verbose)


_OK = {"occ": "UMC260821C00020000", "mid": 0.68, "limit": 0.70, "delta": 0.38,
       "expiry": "2026-08-21", "strike": 20.0, "open_interest": 4996}


def test_a_none_delta_still_produces_the_order():
    """The regression: the contract is tradeable, only its greek is missing."""
    res = _plan({**_OK, "delta": None}, verbose=True)
    assert [(i[0], i[1], i[3]) for i in res.plan] == [
        ("UMC260821C00020000", "buy", "entry")]
    assert res.contract_selection["UMC"]["delta"] is None


def test_a_missing_mid_does_not_crash_the_plan():
    """`order['mid']` was a bare subscript on the same line — a KeyError there
    would have failed the run just as hard as the TypeError did."""
    order = {k: v for k, v in _OK.items() if k != "mid"}
    res = _plan(order, verbose=True)
    assert len(res.plan) == 1


def test_verbose_and_quiet_agree_on_the_plan():
    """Logging must be observationally pure with respect to the orders."""
    loud = _plan({**_OK, "delta": None}, verbose=True)
    quiet = _plan({**_OK, "delta": None}, verbose=False)
    assert loud.plan == quiet.plan
    assert loud.limits == quiet.limits


def test_the_08_10_plan_survives_the_contract_that_killed_it(capsys):
    """Two good contracts then the one with no delta — all three must survive."""
    orders = {
        "UMC": {"occ": "UMC260821C00020000", "mid": 0.68, "limit": 0.70, "delta": 0.38,
                "expiry": "2026-08-21", "strike": 20.0, "open_interest": 4996},
        "GFI": {"occ": "GFI260821C00040000", "mid": 2.39, "limit": 2.45, "delta": 0.63,
                "expiry": "2026-08-21", "strike": 40.0, "open_interest": 4276},
        "TRMB": {"occ": "TRMB260821C00060000", "mid": 1.10, "limit": 1.15, "delta": None,
                 "expiry": "2026-08-21", "strike": 60.0, "open_interest": 800},
    }
    res = build_mixed_plan(
        None, targets=list(orders), managed={}, pos_info={}, bar="2026-08-10 18:00:00+00:00",
        signal_audits={}, policy=ExecPolicy(),
        route_fn=lambda _c, t, *a, **k: ("option", orders[t], "ok"),
        ref_price_fn=lambda _t: 20.0, verbose=True)
    assert len(res.plan) == 3
    out = capsys.readouterr().out
    assert "delta=0.38" in out and "delta=n/a" in out
