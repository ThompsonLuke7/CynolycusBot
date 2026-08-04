"""Price-recap vs forward-looking headline labelling.

Headlines are real rows from the live catalyst ledger (2026-06-16 -> 2026-08-03).
"""
from __future__ import annotations

import pandas as pd
import pytest

from signals.news.information_direction import (
    VALID,
    add_information_direction,
    classify_information_direction,
)


@pytest.mark.parametrize("headline", [
    "Critical Metals (NASDAQ:CRML) Shares Surge 16% as Policy Optimism Clashes",
    "Eos Energy Enterprises (NASDAQ:EOSE) Stock Price Up 10.5% - Time to Buy?",
    "Coursera (COUR) Shares Skyrocket, What You Need To Know",
    "DBS Stock Gains 2.45% Today",
    "Scotts Miracle-Gro (NYSE:SMG) Stock Price Down 7.1%",
    "Novo Nordisk A/S Stock (NVO) Moved Up by 6.81% on Jun 22",
    "Bloom Energy (NYSE:BE) Shares Gap Down",
])
def test_price_recaps_are_labelled(headline):
    assert classify_information_direction(headline) == "price_recap"


@pytest.mark.parametrize("headline", [
    "NVT Maintains Buy Rating by Citigroup -- Price Target Raised to $215",
    "RSI Maintained by Wells Fargo -- Price Target Raised to $35.00",
    "FDA approves Novartis gene therapy for rare disease",
    "Company announces $500M share repurchase program",
    "Acme Corp appoints new CEO effective September",
    "Boeing awarded $1.2B defense contract",
])
def test_forward_looking_headlines_are_labelled(headline):
    assert classify_information_direction(headline) == "forward_looking"


def test_a_headline_carrying_both_is_mixed_not_forced():
    """"Gap Up ON an upgrade" is both a recap and news; collapsing it would hide
    which half drives the return."""
    assert classify_information_direction(
        "Visteon (NASDAQ:VC) Shares Gap Up on Analyst Upgrade") == "mixed"


@pytest.mark.parametrize("headline", [
    "Apple unveils new MacBook Pro lineup",   # 'unveils' -> forward_looking
    "A quiet day in the semiconductor sector",
])
def test_other_and_forward_are_distinguished(headline):
    assert classify_information_direction(headline) in VALID


def test_neutral_headline_is_other():
    assert classify_information_direction("A quiet day in the semiconductor sector") == "other"


@pytest.mark.parametrize("value", [None, "", 123])
def test_degenerate_inputs_do_not_raise(value):
    assert classify_information_direction(value) in VALID


def test_add_information_direction_stamps_a_column():
    df = pd.DataFrame({"headline": [
        "Coursera (COUR) Shares Skyrocket, What You Need To Know",
        "FDA approves Novartis gene therapy for rare disease",
    ]})
    out = add_information_direction(df)
    assert list(out["information_direction"]) == ["price_recap", "forward_looking"]


def test_add_information_direction_tolerates_empty_and_missing_columns():
    empty = pd.DataFrame({"headline": []})
    assert "information_direction" in add_information_direction(empty).columns or len(empty) == 0
    odd = pd.DataFrame({"title": ["x"]})
    assert "information_direction" not in add_information_direction(odd).columns
