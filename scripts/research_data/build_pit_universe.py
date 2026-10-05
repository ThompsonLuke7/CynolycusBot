"""Build the point-in-time research universe: every common stock that traded since 2016,
including delisted names and names that today's shared_universe filters dropped.

Inputs
  Data/research/pit_universe/symbol_months.parquet     monthly raw bars per symbol (probe_symbol_history)
  Data/research/pit_universe/alpaca_assets.parquet     names, for fund screening
  Data/raw/tiingo/supported_tickers_*.zip              assetType (Stock / ETF), where Tiingo knows the ticker
  Data/research/bars_1d_sip_adj/                       existing research cache (today's universe + ETFs)

Stages (run in this order)
  --select   classify every probed symbol and keep non-fund symbols whose best month averaged
             >= MIN_DV dollars/day. Symbols not yet in the research cache go to to_fetch.txt.
  fetch      .venv/bin/python -m scripts.research_data.fetch_research_bars_1d \
                 --tickers-file Data/research/pit_universe/to_fetch.txt \
                 --out-dir Data/research/bars_1d_sip_adj_extra --workers 6
  --build    write one file per SECURITY into Data/research/bars_1d_sip_adj_pit/. A symbol's bars
             are split wherever more than MAX_GAP_DAYS pass with no bar, because that is a
             reused ticker (old BBBY 2016-23 vs the new BBBY 2025+), and wherever the close moves
             more than 10x up or below 0.12x in one day (new equity after bankruptcy, spin-off handoff). Earlier lives get the ids
             SYMBOL~1, SYMBOL~2; the latest keeps SYMBOL. Lives shorter than MIN_BARS are dropped.

Known limits (documented in the study README):
  * A stock that moved to OTC keeps only its exchange history; its last exchange close is its final
    mark. SIVB ends at $106 though holders got ~0. The 05 --delist-zero bound covers that.
  * Alpaca usually keeps a renamed company's history under BOTH symbols (UTX and RTX both carry
    2016-2020). --build drops the alias: when two lives share the same daily trade_count on >= 90%
    of the shorter life's days, the one that stopped earlier is removed (kind = "alias").
    Where the new symbol starts only at the rename (OSTK -> BYON), the old life simply ends there
    and is marked at its last close, which is neutral.
  * Zero-volume bars are dropped before splitting: Alpaca pads dead periods with them, which hides the gap.
  * Split/dividend adjustment is per symbol. A later split in a reused ticker rescales the
    earlier life's price LEVEL (its returns are unaffected).
"""
from __future__ import annotations

import argparse
import logging
import re
import shutil
from pathlib import Path

import pandas as pd

from scripts.research_data.fetch_research_bars_1d import ETFS, OUT_DIR as CACHE_DIR, UNIVERSE_CSV
from scripts.research_data.probe_symbol_history import LISTED, OUT_DIR as PIT_DIR, tiingo_table

REPO = Path(__file__).resolve().parents[2]
EXTRA_DIR = REPO / "Data" / "research" / "bars_1d_sip_adj_extra"
PIT_BARS = REPO / "Data" / "research" / "bars_1d_sip_adj_pit"
MIN_DV = 5e6          # best-month average daily dollar volume; the liquid-1000 floor was $14M (2017) to $85M (2026)
MAX_GAP_DAYS = 45
MIN_BARS = 30
# A one-day close ratio outside these bounds is a different security under the same ticker, not a return:
#   > 10x    new equity after a bankruptcy (OAS +19,893%, WLL +3,342%, WOLF +1,726%) or a missed reverse split
#   < 0.12x  a spin-off / merger handoff (BHVN -94.5%, AAN, SRC, ARNC -89%)
# The worst GENUINE one-day falls in this data are -75% to -84% (KOD, PRAX, SMMT), so they are kept as returns.
# Not caught: bad adjustment factors of smaller size (NVS 2019-04-09, -82% on the Alcon spin). No momentum
# pick was held through one (checked 2026-10-03).
JUMP_UP, JUMP_DOWN = 10.0, 0.12
SESSIONS_PER_MONTH = 21
FUND_RE = re.compile(
    r"\b(ETF|ETN|ETNs|Fund|Index|ProShares|iShares|Direxion|SPDR|Invesco|VanEck|WisdomTree|Vanguard|Global X|"
    r"First Trust|iPath|VelocityShares|PowerShares|Barclays|Credit Suisse AG|UBS AG|MicroSectors|GraniteShares|"
    r"Roundhill|YieldMax|Defiance|Leverage Shares|T-Rex|Tradr|REX Shares|AXS|Volatility Shares|ARK |"
    r"Warrant|Warrants|Units?|Rights?|Preferred|Depositary Shares|Notes due|% Senior)\b", re.I)

# Exchange-traded products that neither Alpaca's asset list nor Tiingo classifies (delisted ETNs / renamed ETFs).
# Leveraged and volatility notes MUST be excluded: one VIX spike would put them in a momentum top 20.
FUND_SYMBOLS = {
    "TVIX", "TVIZ", "XIV", "ZIV", "VIIX", "VXXB", "VXZB", "UWTI", "DWTI", "UWT", "DWT", "UGAZ", "DGAZ",
    "USLV", "DSLV", "UGLD", "DGLD", "OIL", "OILB", "DTO", "DNO", "FNGU", "FNGD", "CHAD", "SPLG", "GBF", "PHB",
    "PIN", "JKD", "JKE", "JKF", "JKG", "JKH", "JKI", "JKJ", "JKK", "JKL", "RYE", "RHS", "RYT", "RTM", "RYH", "RYF",
    "RGI", "RCD", "RYU", "EWRE", "XXXX",
}
ALIAS_MIN_TRADES = 200     # daily trade_count must be at least this to serve as a match key
ALIAS_MIN_DAYS = 20
ALIAS_MIN_SHARE = 0.9

logger = logging.getLogger("build_pit_universe")


def classify(months: pd.DataFrame) -> pd.DataFrame:
    """One row per probed symbol: liquidity, life span and stock/fund kind with its evidence."""
    m = months.assign(dv=months["vwap"] * months["volume"] / SESSIONS_PER_MONTH)
    s = m.groupby("symbol").agg(first_month=("month", "min"), last_month=("month", "max"),
                                n_months=("month", "size"), peak_dv=("dv", "max")).reset_index()
    uni = pd.read_csv(UNIVERSE_CSV)
    uni["ticker"] = uni["ticker"].astype(str).str.upper()
    uni_kind = dict(zip(uni["ticker"], uni["type"].fillna("Stock").map(lambda t: "fund" if t == "ETF" else "stock")))
    tg = tiingo_table()
    tg = tg[tg["exchange"].isin(LISTED)]
    tg_kind = tg.groupby("ticker")["assetType"].agg(lambda a: "fund" if set(a) == {"ETF"} else "stock" if "Stock" in set(a) else None)
    assets = pd.read_parquet(PIT_DIR / "alpaca_assets.parquet").drop_duplicates("symbol", keep="last")
    name = dict(zip(assets["symbol"], assets["name"].fillna("")))
    cached = {p.stem for p in CACHE_DIR.glob("*.parquet")}

    def kind(sym: str) -> tuple[str, str]:
        if sym in ETFS:
            return "fund", "research_etf_list"
        if sym in FUND_SYMBOLS:
            return "fund", "manual_list"
        if sym in uni_kind:
            return uni_kind[sym], "shared_universe"
        if FUND_RE.search(name.get(sym, "")):
            return "fund", "alpaca_name"
        if tg_kind.get(sym) in ("fund", "stock"):
            return tg_kind[sym], "tiingo"
        if sym in name:
            return "stock", "alpaca_name"
        return "unknown", "none"

    k = s["symbol"].map(kind)
    s["kind"], s["kind_source"] = k.str[0], k.str[1]
    s["name"] = s["symbol"].map(name).fillna("")
    s["in_cache"] = s["symbol"].isin(cached)
    return s


def select() -> None:
    months = pd.read_parquet(PIT_DIR / "symbol_months.parquet")
    s = classify(months)
    s["selected"] = (s["kind"] != "fund") & (s["peak_dv"] >= MIN_DV)
    s.to_parquet(PIT_DIR / "symbols.parquet", index=False)
    to_fetch = sorted(s.loc[s["selected"] & ~s["in_cache"], "symbol"])
    (PIT_DIR / "to_fetch.txt").write_text("\n".join(to_fetch) + "\n")
    logger.info("probed %d symbols; kinds %s", len(s), s["kind"].value_counts().to_dict())
    logger.info("selected %d (peak >= $%.0fM/day, not a fund); %d already cached, %d to fetch",
                int(s["selected"].sum()), MIN_DV / 1e6, int((s["selected"] & s["in_cache"]).sum()), len(to_fetch))
    unk = s[s["selected"] & (s["kind"] == "unknown")].nlargest(40, "peak_dv")
    logger.info("largest UNKNOWN-kind selected symbols (review for ETNs/funds):\n%s",
                unk[["symbol", "peak_dv", "first_month", "last_month"]].to_string(index=False))


def split_lives(df: pd.DataFrame) -> list[pd.DataFrame]:
    # Alpaca pads a ticker's dead period with zero-volume placeholder bars (FI: Frank's International at $3
    # through 2021, padding, then Fiserv at $130 from 2023). Without this filter the two companies read as
    # one continuous stock with a +4,000% "return".
    df = df[(df["volume"] > 0) & (df["trade_count"] > 0)]
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    ratio = df["close"] / df["close"].shift(1)
    life = ((df["timestamp"].diff().dt.days > MAX_GAP_DAYS) | (ratio > JUMP_UP) | (ratio < JUMP_DOWN)).cumsum()
    return [g for _, g in df.groupby(life) if len(g) >= MIN_BARS]


def find_aliases(lives: dict[str, pd.DataFrame], meta: pd.DataFrame) -> dict[str, str]:
    """{dropped sec_id: kept sec_id} for lives that are the same security under two symbols."""
    keys = pd.concat([g.loc[g["trade_count"] >= ALIAS_MIN_TRADES, ["timestamp", "trade_count"]].assign(sec_id=k)
                      for k, g in lives.items()], ignore_index=True)
    n_days = keys.groupby("sec_id").size()
    shared = keys[keys.groupby(["timestamp", "trade_count"])["sec_id"].transform("size") > 1]
    pairs = shared.merge(shared, on=["timestamp", "trade_count"])
    pairs = pairs[pairs["sec_id_x"] < pairs["sec_id_y"]].groupby(["sec_id_x", "sec_id_y"]).size().rename("n").reset_index()
    pairs["share"] = pairs["n"] / pd.concat([pairs["sec_id_x"].map(n_days), pairs["sec_id_y"].map(n_days)], axis=1).min(axis=1)
    partial = pairs[(pairs["n"] >= 60) & (pairs["share"] < ALIAS_MIN_SHARE)]
    if len(partial):
        logger.info("%d pairs overlap on >= 60 days but below the alias threshold (left in, double counted "
                    "on the overlap): %s", len(partial),
                    ", ".join(f"{a}/{b}" for a, b in partial[["sec_id_x", "sec_id_y"]].head(15).values))
    dup = pairs[(pairs["n"] >= ALIAS_MIN_DAYS) & (pairs["share"] >= ALIAS_MIN_SHARE)]
    parent = {k: k for k in lives}

    def root(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for a, b in dup[["sec_id_x", "sec_id_y"]].values:
        parent[root(a)] = root(b)
    m = meta.set_index("sec_id")
    groups: dict[str, list[str]] = {}
    for k in lives:
        groups.setdefault(root(k), []).append(k)
    out = {}
    for members in groups.values():
        if len(members) > 1:   # keep the life that trades latest, then the longest, then today's-universe symbol
            keep = max(members, key=lambda k: (m.at[k, "last_bar"], m.at[k, "n_bars"], m.at[k, "source"] == "cache"))
            out.update({k: keep for k in members if k != keep})
    return out


def build() -> None:
    s = pd.read_parquet(PIT_DIR / "symbols.parquet")
    kind = dict(zip(s["symbol"], s["kind"]))
    selected = set(s.loc[s["selected"], "symbol"])
    if PIT_BARS.exists():
        shutil.rmtree(PIT_BARS)  # derived directory: always rebuilt whole from the two source caches
    PIT_BARS.mkdir(parents=True)
    rows, lives = [], {}
    sources = [(p, "cache") for p in sorted(CACHE_DIR.glob("*.parquet"))] + \
              [(p, "extra") for p in sorted(EXTRA_DIR.glob("*.parquet"))]
    for path, source in sources:
        sym = path.stem
        if source == "cache" and (sym in ETFS or kind.get(sym) == "fund"):
            shutil.copy2(path, PIT_BARS / path.name)   # benchmarks: no split, not in the stock universe
            rows.append({"sec_id": sym, "symbol": sym, "kind": "fund", "source": source})
            continue
        if kind.get(sym) == "fund" or (source == "extra" and sym not in selected):
            continue
        parts = split_lives(pd.read_parquet(path))
        for i, g in enumerate(parts):
            sec_id = sym if i == len(parts) - 1 else f"{sym}~{i + 1}"
            lives[sec_id] = g
            rows.append({"sec_id": sec_id, "symbol": sym, "kind": "stock", "source": source,
                         "first_bar": g["timestamp"].min(), "last_bar": g["timestamp"].max(), "n_bars": len(g)})
    sec = pd.DataFrame(rows)
    alias = find_aliases(lives, sec[sec["kind"] == "stock"])
    sec["alias_of"] = sec["sec_id"].map(alias)
    sec.loc[sec["alias_of"].notna(), "kind"] = "alias"
    for sec_id, g in lives.items():
        if sec_id not in alias:
            g.to_parquet(PIT_BARS / f"{sec_id}.parquet", index=False)
    end = sec["last_bar"].max()
    sec["stopped"] = sec["last_bar"] < end - pd.Timedelta(days=21)
    sec.to_parquet(PIT_DIR / "securities.parquet", index=False)
    st = sec[sec["kind"] == "stock"]
    logger.info("wrote %d stock securities (%d from today's cache, %d added; %d stopped trading; %d earlier lives of "
                "reused tickers) + %d funds; dropped %d aliases of renamed companies",
                len(st), int((st["source"] == "cache").sum()), int((st["source"] == "extra").sum()),
                int(st["stopped"].sum()), int(st["sec_id"].str.contains("~").sum()),
                int((sec["kind"] == "fund").sum()), len(alias))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.select:
        select()
    if args.build:
        build()


if __name__ == "__main__":
    main()
