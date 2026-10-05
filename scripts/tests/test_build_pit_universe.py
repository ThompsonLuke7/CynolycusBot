import pandas as pd

from scripts.research_data.build_pit_universe import find_aliases, split_lives


def _bars(dates, closes, volume=1_000, trades=500):
    n = len(dates)
    vol = volume if isinstance(volume, list) else [volume] * n
    tc = trades if isinstance(trades, list) else [trades] * n
    return pd.DataFrame({"timestamp": pd.to_datetime(dates, utc=True), "close": closes, "volume": vol, "trade_count": tc})


def test_zero_volume_padding_does_not_glue_two_companies():
    # FI shape: Frank's International at $3, zero-volume placeholder bars, then Fiserv at $130.
    old = pd.bdate_range("2021-01-04", periods=40)
    pad = pd.bdate_range(old[-1] + pd.Timedelta(days=1), periods=120)
    new = pd.bdate_range(pad[-1] + pd.Timedelta(days=1), periods=40)
    df = pd.concat([_bars(old, [3.0] * 40), _bars(pad, [3.0] * 120, volume=0, trades=0), _bars(new, [130.0] * 40)])
    lives = split_lives(df)
    assert [len(g) for g in lives] == [40, 40]
    assert lives[0]["close"].max() == 3.0 and lives[1]["close"].min() == 130.0


def test_split_on_bankruptcy_jump_and_spinoff_handoff_but_not_on_a_real_crash():
    d = pd.bdate_range("2022-01-03", periods=160)
    up = split_lives(_bars(d, [0.15] * 80 + [29.0] * 80))        # OAS: new equity under the old ticker
    down = split_lives(_bars(d, [150.0] * 80 + [8.3] * 80))      # BHVN: cash deal + spin-off keeps the ticker
    crash = split_lives(_bars(d, [50.0] * 80 + [9.8] * 80))      # KOD: a genuine -80% day stays a return
    assert [len(g) for g in up] == [80, 80]
    assert [len(g) for g in down] == [80, 80]
    assert [len(g) for g in crash] == [160]


def test_alias_that_stopped_earlier_is_dropped_and_unrelated_names_kept():
    d = pd.bdate_range("2019-01-02", periods=300)
    tc = [1_000 + i for i in range(300)]
    lives = {
        "RTX": _bars(d, [70.0] * 300, trades=tc),                       # survivor carries the full history
        "UTX": _bars(d[:200], [70.0] * 200, trades=tc[:200]),           # old symbol: same security, stopped earlier
        "AAPL": _bars(d, [40.0] * 300, trades=[5_000 + 7 * i for i in range(300)]),
    }
    meta = pd.DataFrame([{"sec_id": k, "last_bar": g["timestamp"].max(), "n_bars": len(g), "source": "extra"}
                         for k, g in lives.items()])
    assert find_aliases(lives, meta) == {"UTX": "RTX"}
