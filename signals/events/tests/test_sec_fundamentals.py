import pandas as pd

from signals.events.sec_earnings_releases import earnings_8ks, pick_exhibit
from signals.events.sec_fundamentals import extract_quarterly_facts


def _fact(start, end, val, filed, form="10-Q", accn="a"):
    return {"start": start, "end": end, "val": val, "filed": filed, "form": form, "accn": accn}


def _facts(tag, rows):
    return {"facts": {"us-gaap": {tag: {"units": {"USD": rows}}}}}


def test_first_reported_value_wins_over_restatement():
    cf = _facts("Revenues", [
        _fact("2024-01-01", "2024-03-31", 100.0, "2024-05-01", accn="q1"),
        # Same period re-reported (restated) a year later as a comparative.
        _fact("2024-01-01", "2024-03-31", 90.0, "2025-05-01", accn="q1_next_year"),
    ])
    out = extract_quarterly_facts(cf, "TST", "0000000001")
    row = out[out["metric"] == "revenue"].iloc[0]
    assert row["value"] == 100.0
    assert row["available_at"] == pd.Timestamp("2024-05-01")


def test_q4_derived_from_annual_and_available_at_10k_filing():
    cf = _facts("Revenues", [
        _fact("2024-01-01", "2024-03-31", 10.0, "2024-05-01"),
        _fact("2024-04-01", "2024-06-30", 20.0, "2024-08-01"),
        _fact("2024-07-01", "2024-09-30", 30.0, "2024-11-01"),
        _fact("2024-01-01", "2024-12-31", 100.0, "2025-02-20", form="10-K"),
    ])
    out = extract_quarterly_facts(cf, "TST", "0000000001")
    q4 = out[out["derived"]]
    assert len(q4) == 1
    assert q4.iloc[0]["value"] == 40.0
    assert q4.iloc[0]["available_at"] == pd.Timestamp("2025-02-20")
    assert q4.iloc[0]["period_end"] == pd.Timestamp("2024-12-31")


def test_higher_priority_tag_wins_per_period():
    cf = {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [_fact("2024-01-01", "2024-03-31", 5.0, "2024-05-01")]}},
        "RevenueFromContractWithCustomerExcludingAssessedTax": {
            "units": {"USD": [_fact("2024-01-01", "2024-03-31", 7.0, "2024-05-02")]}},
    }}}
    out = extract_quarterly_facts(cf, "TST", "0000000001")
    assert out[out["metric"] == "revenue"]["value"].tolist() == [7.0]


def test_earnings_8k_filter_and_exhibit_pick():
    filings = pd.DataFrame({
        "form": ["8-K", "8-K", "10-Q", "8-K"],
        "items": ["2.02,9.01", "5.02", "", "2.02"],
        "filingDate": ["2024-05-01", "2024-05-02", "2024-05-03", "2015-01-01"],
        "accessionNumber": ["a", "b", "c", "d"],
    })
    assert earnings_8ks(filings)["accessionNumber"].tolist() == ["a"]
    items = [{"name": "nvda-20240522.htm"}, {"name": "q1fy25pr-ex99_2.htm"},
             {"name": "q1fy25pr-ex99_1.htm"}, {"name": "R1.htm"}]
    assert pick_exhibit(items) == "q1fy25pr-ex99_1.htm"
    assert pick_exhibit([{"name": "cover.htm"}]) is None
