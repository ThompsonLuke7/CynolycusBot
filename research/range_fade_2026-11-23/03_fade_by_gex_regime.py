"""Join 01's trade files to 02's causal GEX series (latest snapshot <= signal bar close, max age 10 min); split by GEX sign."""
import sys
import numpy as np
import pandas as pd

gex_path, *trade_paths = sys.argv[1:]
g = pd.read_csv(gex_path, parse_dates=["ts"]).sort_values("ts")
g["ts"] = g.ts.dt.tz_convert("America/New_York").astype("datetime64[ns, America/New_York]")
for path in trade_paths:
    tr = pd.read_csv(path)
    tr["sig"] = (pd.to_datetime(tr.date.astype(str) + " " + tr.t).dt.tz_localize("America/New_York")
                 + pd.Timedelta(minutes=1)).astype("datetime64[ns, America/New_York]")
    m = pd.merge_asof(tr.sort_values("sig"), g[["ts", "gex", "flip"]].rename(columns={"ts": "sig"}), on="sig",
                      direction="backward", tolerance=pd.Timedelta(minutes=10))
    m = m[m.gex.notna()].assign(regime=lambda x: np.where(x.gex > 0, "GEX+", "GEX-"))
    rows = []
    for (arm, reg), x in m.groupby(["arm", "regime"]):
        sd = x.bps.std(ddof=1)
        rows.append((arm, reg, len(x), x.date.nunique(), x.bps.mean(), sd / np.sqrt(len(x)), 2.8 * sd / np.sqrt(len(x)), (x.bps > 0).mean()))
    print(f"\n{path}  trades with snapshot: {len(m)}  days: {m.date.nunique()}")
    print(pd.DataFrame(rows, columns=["arm", "regime", "n", "days", "gross_bps", "se", "mde80", "win"]).round(2).to_string(index=False))
