"""Per-snapshot QQQ total GEX + gamma flip from archived intraday gamma ladders (as-of = snapshot timestamp)."""
import glob, sys
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from strategies.dealer_positioning.levels import _core_levels_from_ladder

def one(f):
    try:
        lad = pd.read_csv(f)
        spot = float(lad.spot.iloc[0])
        core = _core_levels_from_ladder(lad, spot, 0.9)
        return dict(ts=lad.timestamp.iloc[0], spot=spot, gex=float(lad.net_gex.sum()), flip=core.get("gamma_flip"),
                    cw=core.get("call_wall"), pw=core.get("put_wall"))
    except Exception as e:
        return None

if __name__ == "__main__":
    fs = sorted(glob.glob("Data/dealer_positioning/QQQ/snapshots/gamma_ladder_*.csv"))
    with ProcessPoolExecutor(8) as ex:
        rows = [r for r in ex.map(one, fs, chunksize=64) if r]
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df.ts, utc=True)
    df.sort_values("ts").to_csv(sys.argv[1], index=False)
    print(len(fs), len(df))
