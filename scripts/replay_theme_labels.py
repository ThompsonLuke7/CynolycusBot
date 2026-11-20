"""Replay step05 labeling on the latest cluster artifacts, into a scratch directory.

Labels every cluster of the most recent weekly run with the CURRENT step05 code and
compares the result with that run's registry snapshot. As in production, a cluster
whose snapshot row was carried forward goes to the labeler with that name as
current_name (--no-current-names labels everything from scratch). Reports:

  * completeness: clusters labeled, clusters that needed a one-per-call retry
  * carried names kept vs replaced (with --no-current-names: not applicable)
  * cost: calls and tokens, from the client's per-call usage log
  * agreement: exact theme-name match vs the snapshot, split by whether the
    snapshot row was freshly labeled or carried forward from the prior week
  * quality: label_regression.csv check and theme_map_v4 agreement on a
    ticker -> cluster-theme map, descriptions naming tickers from other
    clusters, and the share of low-confidence (< MIN_CLUSTER_LABEL_CONFIDENCE) labels

Reads themes/dynamic_theme/outputs/{ticker_clusters,ticker_documents,theme_registry}.parquet;
writes only under --out (default Data/research/theme_label_replay/<UTC stamp>/).
Spends API money: about ceil(clusters / batch size) calls.

Usage:
  .venv/bin/python scripts/replay_theme_labels.py [--effort low|medium|high] [--batch-size 15]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")  # before config import: it reads ANTHROPIC_API_KEY at import

import pandas as pd  # noqa: E402

from themes.dynamic_theme import label_checks as lc  # noqa: E402
from themes.dynamic_theme.config import (  # noqa: E402
    CLAUDE_MODEL,
    MIN_CLUSTER_LABEL_CONFIDENCE,
    THEME_REGISTRY_PATH,
    TICKER_CLUSTERS_PATH,
    TICKER_DOCUMENTS_PATH,
)
from themes.dynamic_theme.stages import step05_claude_labeling as s5  # noqa: E402
from themes.dynamic_theme.stages.step04_cluster_summary import build_cluster_summaries  # noqa: E402

# $ per million tokens (input, output), first-party list prices.
PRICES = {"claude-sonnet-5-5": (2.0, 10.0), "claude-sonnet-5": (2.0, 10.0), "claude-sonnet-4-6": (3.0, 15.0)}
# 3-5 capitals: 1-2 letter tickers (A, S, AI, IT) collide with ordinary words.
TICKER_TOKEN = re.compile(r"\b[A-Z]{3,5}\b")
ACRONYMS = {"USA", "LNG", "LPG", "ETF", "ETFS", "REIT", "REITS", "CEO", "GDP", "EPS", "IPO", "SPAC",
            "ESG", "API", "SAAS", "HVAC", "LED", "OEM", "OEMS", "GLP", "MRNA", "CDMO", "NASA", "FDA"}


class _UsageCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.calls: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        usage = getattr(record, "claude_usage", None)
        if usage:
            self.calls.append(usage)


def _primary(clusters: pd.DataFrame, names: dict[int, str]) -> pd.DataFrame:
    df = clusters[["ticker", "cluster_id"]].copy()
    df["primary_theme"] = df["cluster_id"].map(names)
    return df.dropna(subset=["primary_theme"])


def _foreign_mentions(rows: pd.DataFrame, members: dict[int, set[str]], universe: set[str],
                      column: str) -> int:
    count = 0
    for cid, text in zip(rows["cluster_id"], rows[column].fillna("")):
        mentioned = (set(TICKER_TOKEN.findall(str(text))) - ACRONYMS) & universe
        count += bool(mentioned - members.get(int(cid), set()))
    return count


def _quality(clusters, names, v4, cases) -> dict:
    primary = _primary(clusters, names)
    reg = lc.regression_check(primary, cases)
    return {
        "regression_match_rate": reg["match_rate"],
        "regression_scored": reg["scored"],
        "regression_forbidden": reg["forbidden"],
        "v4_agreement": lc.v4_agreement(primary, v4).get("agreement"),
        "distinct_themes": len(set(names.values())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--effort", default=s5.CLAUDE_LABEL_EFFORT, choices=["low", "medium", "high"])
    parser.add_argument("--batch-size", type=int, default=s5.CLAUDE_LABEL_BATCH_SIZE)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-current-names", action="store_true",
                        help="label every cluster from scratch instead of offering carried names")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    clusters = pd.read_parquet(TICKER_CLUSTERS_PATH)
    docs = pd.read_parquet(TICKER_DOCUMENTS_PATH)
    registry = pd.read_parquet(THEME_REGISTRY_PATH)
    run_date = pd.to_datetime(clusters["date"]).max().normalize().tz_localize(None)
    registry["date"] = pd.to_datetime(registry["date"]).dt.tz_localize(None)
    base = registry[registry["date"] == run_date]
    prior_dates = registry.loc[registry["date"] < run_date, "date"]
    if base.empty or prior_dates.empty:
        raise SystemExit(f"registry needs a snapshot at {run_date.date()} and one before it")
    prior = registry[registry["date"] == prior_dates.max()]

    summaries = build_cluster_summaries(clusters, docs_df=docs)
    carried_pairs = set(zip(prior["theme_name"], prior["description"]))
    carried_name = {int(c): n for c, n, d in zip(base["cluster_id"], base["theme_name"], base["description"])
                    if (n, d) in carried_pairs}
    if not args.no_current_names:
        summaries = [{**s, "current_name": carried_name[int(s["cluster_id"])]}
                     if int(s["cluster_id"]) in carried_name else s for s in summaries]
    out = args.out or REPO / "Data/research/theme_label_replay" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)

    collector = _UsageCollector()
    logging.getLogger("themes.dynamic_theme.client.claude_client").addHandler(collector)
    s5.CLAUDE_LABEL_EFFORT = args.effort
    s5.CLAUDE_LABEL_BATCH_SIZE = args.batch_size
    labels = s5._label_new_clusters(summaries, set(prior["theme_name"].astype(str)))

    members = clusters.groupby("cluster_id")["ticker"].apply(lambda s: set(s.astype(str))).to_dict()
    universe = set(clusters["ticker"].astype(str))
    carried = set(zip(prior["theme_name"], prior["description"]))
    base = base.assign(carried=[(n, d) in carried for n, d in zip(base["theme_name"], base["description"])],
                       placeholder=base["theme_name"].astype(str).str.fullmatch(r"cluster_-?\d+"))
    base_ids = set(base["cluster_id"].astype(int))
    rows = pd.DataFrame([
        {"cluster_id": int(s["cluster_id"]), "tickers": " ".join(s["tickers"][:12]),
         "new_name": labels.get(int(s["cluster_id"]), {}).get("theme_name"),
         "new_confidence": labels.get(int(s["cluster_id"]), {}).get("confidence"),
         "new_description": labels.get(int(s["cluster_id"]), {}).get("description")}
        for s in summaries
    ]).merge(base[["cluster_id", "theme_name", "description", "confidence", "carried", "placeholder"]]
             .rename(columns={"theme_name": "base_name", "description": "base_description",
                              "confidence": "base_confidence"}),
             on="cluster_id", how="left")
    rows.to_csv(out / "labels.csv", index=False)

    comparable = rows[rows["base_name"].notna() & ~rows["placeholder"].fillna(False) & rows["new_name"].notna()]
    same = comparable["new_name"] == comparable["base_name"]
    v4, cases = lc.load_v4_map(), lc.load_regression_cases()
    base_names = {int(c): n for c, n, p in zip(base["cluster_id"], base["theme_name"], base["placeholder"]) if not p}
    new_names = {c: lab["theme_name"] for c, lab in labels.items()}
    price_in, price_out = PRICES.get(CLAUDE_MODEL, (float("nan"), float("nan")))
    tokens_in = sum(c["input_tokens"] for c in collector.calls)
    tokens_out = sum(c["output_tokens"] for c in collector.calls)
    summary = {
        "run_date": str(run_date.date()), "model": CLAUDE_MODEL, "effort": args.effort,
        "batch_size": args.batch_size, "clusters": len(summaries),
        "labeled": len(labels), "missing": sorted(set(int(s["cluster_id"]) for s in summaries) - set(labels)),
        "calls": len(collector.calls), "min_calls": math.ceil(len(summaries) / args.batch_size),
        "input_tokens": tokens_in, "output_tokens": tokens_out,
        "est_cost_usd": round(tokens_in / 1e6 * price_in + tokens_out / 1e6 * price_out, 4),
        "carried_names": None if args.no_current_names else {
            "offered": len(carried_name),
            "kept": sum(1 for c, n in carried_name.items() if labels.get(c, {}).get("theme_name") == n),
            "replaced": {c: [n, labels[c]["theme_name"]] for c, n in carried_name.items()
                         if c in labels and labels[c]["theme_name"] != n},
        },
        "clusters_not_in_snapshot": sorted(set(int(s["cluster_id"]) for s in summaries) - base_ids),
        "name_match_vs_snapshot": {
            "all": round(float(same.mean()), 3) if len(same) else None, "n": int(len(same)),
            "fresh_rows": round(float(same[~comparable["carried"]].mean()), 3) if (~comparable["carried"]).any() else None,
            "carried_rows": round(float(same[comparable["carried"]].mean()), 3) if comparable["carried"].any() else None,
        },
        "foreign_ticker_descriptions": {
            "new": _foreign_mentions(rows.dropna(subset=["new_name"]), members, universe, "new_description"),
            # carried rows reuse a description written for last week's members
            "snapshot_fresh": _foreign_mentions(base[~base["carried"]], members, universe, "description"),
            "snapshot_carried": _foreign_mentions(base[base["carried"]], members, universe, "description"),
            "snapshot_rows_fresh_carried": [int((~base["carried"]).sum()), int(base["carried"].sum())],
        },
        "low_confidence_share": {
            "new": round(float((rows["new_confidence"] < MIN_CLUSTER_LABEL_CONFIDENCE).mean()), 3),
            "snapshot": round(float((base.loc[~base["placeholder"], "confidence"] < MIN_CLUSTER_LABEL_CONFIDENCE).mean()), 3),
        },
        "quality_new": _quality(clusters, new_names, v4, cases),
        "quality_snapshot": _quality(clusters, base_names, v4, cases),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {out}/labels.csv and summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
