"""Option source-fitness gate (Task 24).

This gate exists because of a specific, documented failure. In 2026-07 an
entire options-routing study reached confident conclusions and was fully
retracted: the option "prices" were stale trade prints, and the correlation
between option returns and underlying direction was +0.09 where a long call
should be around +0.9. Two intermediate error corrections changed magnitudes
but not signs, which made the wrong answer look robust.

So the gate runs *before* any P&L is computed, and its default answer is no.
A source has to prove it can answer the question; failing to disprove fitness
is not the same as establishing it.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from core.nervous_system.contracts.replay import (
    FitnessReason,
    MarkType,
    SideFitnessMetrics,
    SourceFitnessStatus,
    SourceFitnessThresholds,
)
from core.nervous_system.replay.fitness import evaluate_source_fitness


def _metrics(**updates: object) -> SideFitnessMetrics:
    payload: dict[str, object] = {
        "option_type": "CALL",
        "mark_type": MarkType.QUOTE_BID_ASK,
        "matched_positions": 40,
        "sessions": 15,
        "valid_quote_fraction": Decimal("0.99"),
        "identical_mark_fraction": Decimal("0.01"),
        "pearson": Decimal("0.91"),
        "spearman": Decimal("0.88"),
        "max_quote_age_seconds": Decimal("30"),
        "entitlement_verified": True,
    }
    payload.update(updates)
    return SideFitnessMetrics(**payload)  # type: ignore[arg-type]


def _evaluate(*sides: SideFitnessMetrics, **updates: object):
    payload: dict[str, object] = {
        "sides": sides or (_metrics(), _metrics(option_type="PUT")),
        "thresholds": SourceFitnessThresholds(),
        "source": "alpaca",
        "feed": "opra",
        "tier": "indicative",
    }
    payload.update(updates)
    return evaluate_source_fitness(**payload)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The happy path is narrow on purpose
# ---------------------------------------------------------------------------


def test_a_healthy_two_sided_source_is_fit() -> None:
    report = _evaluate()

    assert report.status is SourceFitnessStatus.FIT_FOR_OPTION_PNL
    assert report.option_pnl_eligible is True
    assert report.reasons == ()


def test_both_correlations_are_reported_not_just_the_gating_one() -> None:
    report = _evaluate()

    call = report.side("CALL")
    assert call.pearson == Decimal("0.91")
    assert call.spearman == Decimal("0.88")


def test_a_correlation_below_the_warning_band_is_flagged_but_still_fit() -> None:
    """Between +0.70 and +0.85 the source is usable but suspicious, and saying
    so is the difference between a caveat and a silent assumption.
    """

    report = _evaluate(_metrics(pearson=Decimal("0.80")), _metrics(option_type="PUT"))

    assert report.status is SourceFitnessStatus.FIT_FOR_OPTION_PNL
    assert FitnessReason.CORRELATION_BELOW_WARNING_BAND in report.warnings


# ---------------------------------------------------------------------------
# The retraction, encoded
# ---------------------------------------------------------------------------


def test_the_retracted_studys_correlation_is_refused() -> None:
    """corr = +0.09 between a long call and its underlying. This is the exact
    number that invalidated the 2026-07 study.
    """

    report = _evaluate(_metrics(pearson=Decimal("0.09")), _metrics(option_type="PUT"))

    assert report.status is SourceFitnessStatus.SOURCE_UNFIT_FOR_OPTION_PNL
    assert FitnessReason.LOW_DERIVATIVE_CORRELATION in report.reasons
    assert report.option_pnl_eligible is False


@pytest.mark.parametrize(
    "mark_type,reason",
    [
        (MarkType.TRADE_PRINT, FitnessReason.TRADE_ONLY),
        (MarkType.LAST_PRICE, FitnessReason.TRADE_ONLY),
        (MarkType.SYNTHETIC, FitnessReason.SYNTHETIC_MARKS),
        (MarkType.FORWARD_FILLED, FitnessReason.SYNTHETIC_MARKS),
        (MarkType.INTERPOLATED, FitnessReason.SYNTHETIC_MARKS),
    ],
)
def test_a_non_quote_mark_is_never_fit(mark_type: MarkType, reason: FitnessReason) -> None:
    """A trade bar only has bars on days something traded. On an illiquid
    contract the price at an arbitrary timestamp is a stale last print, not a
    mark, and any P&L built from it is fiction. No amount of correlation
    rescues it.
    """

    report = _evaluate(
        _metrics(mark_type=mark_type, pearson=Decimal("0.99")),
        _metrics(option_type="PUT"),
    )

    assert report.status is SourceFitnessStatus.SOURCE_UNFIT_FOR_OPTION_PNL
    assert reason in report.reasons


def test_stale_marks_are_refused() -> None:
    report = _evaluate(
        _metrics(max_quote_age_seconds=Decimal("3600")), _metrics(option_type="PUT")
    )

    assert report.status is SourceFitnessStatus.SOURCE_UNFIT_FOR_OPTION_PNL
    assert FitnessReason.STALE_MARKS in report.reasons


def test_identical_entry_and_exit_marks_are_refused() -> None:
    """Positions whose entry and exit marks are the same number are the
    signature of a price series that is not moving with value.
    """

    report = _evaluate(
        _metrics(identical_mark_fraction=Decimal("0.20")), _metrics(option_type="PUT")
    )

    assert report.status is SourceFitnessStatus.SOURCE_UNFIT_FOR_OPTION_PNL
    assert FitnessReason.STALE_MARKS in report.reasons


def test_thin_quote_coverage_is_refused() -> None:
    report = _evaluate(
        _metrics(valid_quote_fraction=Decimal("0.60")), _metrics(option_type="PUT")
    )

    assert report.status is SourceFitnessStatus.SOURCE_UNFIT_FOR_OPTION_PNL
    assert FitnessReason.CROSSED_QUOTES in report.reasons


def test_an_unverified_entitlement_is_refused() -> None:
    """A 200 response is not evidence of fitness for purpose; the tier actually
    in use has to be confirmed.
    """

    report = _evaluate(
        _metrics(entitlement_verified=False), _metrics(option_type="PUT")
    )

    assert report.status is SourceFitnessStatus.SOURCE_UNFIT_FOR_OPTION_PNL
    assert FitnessReason.ENTITLEMENT_UNVERIFIED in report.reasons


# ---------------------------------------------------------------------------
# Insufficient sample is its own answer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("matched_positions", 10, FitnessReason.INSUFFICIENT_POSITIONS),
        ("sessions", 3, FitnessReason.INSUFFICIENT_SESSIONS),
    ],
)
def test_too_little_evidence_is_not_reported_as_unfit(
    field: str, value: object, reason: FitnessReason
) -> None:
    """"We cannot tell yet" and "this source is broken" are different answers,
    and collapsing them would either hide a real defect or condemn a good
    source on thin data.
    """

    report = _evaluate(_metrics(**{field: value}), _metrics(option_type="PUT"))

    assert report.status is SourceFitnessStatus.SOURCE_FITNESS_INSUFFICIENT_SAMPLE
    assert reason in report.reasons
    assert report.option_pnl_eligible is False


def test_a_real_defect_outranks_a_thin_sample() -> None:
    """A source with too few positions *and* synthetic marks is broken, not
    merely unproven. Reporting it as 'insufficient sample' would invite
    somebody to gather more of the same bad data.
    """

    report = _evaluate(
        _metrics(matched_positions=5, mark_type=MarkType.SYNTHETIC),
        _metrics(option_type="PUT"),
    )

    assert report.status is SourceFitnessStatus.SOURCE_UNFIT_FOR_OPTION_PNL


# ---------------------------------------------------------------------------
# Per-side enforcement
# ---------------------------------------------------------------------------


def test_a_failing_put_side_condemns_the_source_even_if_calls_are_perfect() -> None:
    """Thresholds are enforced per side, not in aggregate: averaging a broken
    put series against a healthy call series hides the break.
    """

    report = _evaluate(
        _metrics(pearson=Decimal("0.95")),
        _metrics(option_type="PUT", pearson=Decimal("0.10")),
    )

    assert report.status is SourceFitnessStatus.SOURCE_UNFIT_FOR_OPTION_PNL
    assert FitnessReason.LOW_DERIVATIVE_CORRELATION in report.reasons


def test_the_put_correlation_is_measured_against_the_negated_underlying() -> None:
    """A put gains when the underlying falls, so its fitness correlation is
    against the negated underlying and is expected to be positive. A raw
    negative correlation would be the healthy case reported as a failure.
    """

    report = _evaluate(
        _metrics(), _metrics(option_type="PUT", pearson=Decimal("0.90"))
    )

    assert report.status is SourceFitnessStatus.FIT_FOR_OPTION_PNL


def test_a_missing_side_is_not_silently_treated_as_passing() -> None:
    report = _evaluate(_metrics())

    assert report.status is not SourceFitnessStatus.FIT_FOR_OPTION_PNL
    assert FitnessReason.SIDE_NOT_EVALUATED in report.reasons


# ---------------------------------------------------------------------------
# Report integrity
# ---------------------------------------------------------------------------


def test_the_report_records_the_source_feed_and_tier() -> None:
    report = _evaluate()

    assert report.source == "alpaca"
    assert report.feed == "opra"
    assert report.tier == "indicative"


def test_the_threshold_hash_changes_when_a_threshold_changes() -> None:
    """A verdict is only meaningful next to the bar it was judged against."""

    strict = SourceFitnessThresholds(min_pearson=Decimal("0.80"))

    assert SourceFitnessThresholds().content_hash() != strict.content_hash()


def test_the_report_carries_the_threshold_hash_it_was_judged_against() -> None:
    thresholds = SourceFitnessThresholds()
    report = _evaluate(thresholds=thresholds)

    assert report.thresholds_hash == thresholds.content_hash()


def test_reasons_are_ordered_and_deduplicated() -> None:
    """The report is content-hashed downstream, so the same failure must always
    render identically. Asserted with several distinct reasons on purpose: a
    single-reason report is sorted no matter what the code does.
    """

    broken = _metrics(
        mark_type=MarkType.TRADE_PRINT,
        entitlement_verified=False,
        pearson=Decimal("0.09"),
        matched_positions=1,
    )
    first = _evaluate(broken, _metrics(option_type="PUT", mark_type=MarkType.TRADE_PRINT))
    second = _evaluate(broken, _metrics(option_type="PUT", mark_type=MarkType.TRADE_PRINT))

    assert len(first.reasons) >= 4, "this fixture must produce several reasons"
    assert first.reasons == second.reasons
    assert len(set(first.reasons)) == len(first.reasons)
    assert list(first.reasons) == sorted(first.reasons, key=lambda r: r.value)


def test_a_midpoint_is_not_a_fit_mark_for_executable_pnl() -> None:
    """Executable long-option P&L pays the ask on entry and receives the bid on
    exit. A midpoint is a mark-to-market basis, not an executable one, so it is
    labelled separately rather than quietly accepted here.
    """

    report = _evaluate(
        _metrics(mark_type=MarkType.MID), _metrics(option_type="PUT")
    )

    assert report.status is SourceFitnessStatus.SOURCE_UNFIT_FOR_OPTION_PNL
    assert report.option_pnl_eligible is False
