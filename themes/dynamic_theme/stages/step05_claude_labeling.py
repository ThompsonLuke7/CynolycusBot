"""Step 5 — Claude theme labeling.

Labels clusters in batches (CLAUDE_LABEL_BATCH_SIZE per call, structured JSON
output) to assign:
  theme_name, parent_theme, description, related_themes, confidence
Clusters a batch fails to label are retried one per call.

Merges with any existing theme_registry.parquet (upsert by cluster_id + date).
Writes nothing when any cluster is left without a real label: a ``cluster_<id>``
placeholder aborts step08 anyway, so writing it only contaminated the registry
(2026-11-16, API credit ran out after 102 of 110 labels).

Output: theme_registry.parquet
Schema : cluster_id | theme_name | parent_theme | description | related_themes | confidence | date
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import pandas as pd

from themes.dynamic_theme.client.claude_client import ClaudeUnavailableError, call_claude_json
from themes.dynamic_theme.config import (
    CLAUDE_LABEL_BATCH_SIZE,
    CLAUDE_LABEL_EFFORT,
    CLAUDE_LABEL_MAX_OTHER_TICKERS,
    CLAUDE_LABEL_MAX_TOKENS,
    MIN_CLUSTER_LABEL_CONFIDENCE,
    THEME_REGISTRY_PATH,
    ensure_outputs,
)

logger = logging.getLogger(__name__)

# Mirrors step08's guard in canonical_theme_id: such a name is not a theme identity.
_EPHEMERAL_LABEL_RE = re.compile(r"^cluster_-?\d+$", re.IGNORECASE)


class LabelingIncompleteError(RuntimeError):
    """Some clusters ended the run with a placeholder name; the registry was not written."""


_LABEL_SYSTEM = """\
You name clusters of US-listed stocks for an equity research theme taxonomy. Each \
cluster comes from embedding the companies' business descriptions and recent news, \
so its members usually share a business line, end market or economic exposure. The \
names feed a trading system's theme features, so a precise, durable name matters \
more than a catchy one.

You receive a batch of clusters. For each you see how many members it has, its largest \
members (company name, industry and a one-line business summary), the remaining tickers, \
and a few recent headlines. Return one label per cluster:

- theme_name: snake_case, 1-4 words, naming what the members have in common \
(nuclear_energy, regional_banks, ai_infrastructure).
- parent_theme: a broader snake_case category (energy, financials, technology).
- description: one sentence on the business the members share. Describe the theme, not \
a member list: it should still read true if one member leaves next week.
- related_themes: snake_case names of economically related themes.
- confidence: 0.0-1.0, how clearly the members share one theme. Below 0.5 means a \
grab-bag with no single business; say so with the score rather than forcing a story.

Name the cluster from what its companies do, as the business summaries describe it. \
Headlines are a weak hint: they are often about one member, or about a different \
company that shares a word with a member, so never name a theme after a headline alone. \
Where the largest members and the rest disagree, follow the bulk of the members.

Clusters in a batch are independent: a neighbouring cluster is not evidence about this \
one, and a ticker belongs only to the cluster it is listed under.

Some clusters carry current_name, the name this group of companies had last week. Keep \
it, verbatim, when it still describes these members: a rename breaks the theme's weekly \
history. Replace it only when it is clearly wrong for these companies. Either way, write \
the description and confidence fresh from the current members.

Reuse a name from EXISTING THEMES verbatim when a cluster is substantially that theme: \
"regional_bank_holding_companies" and "community_regional_banks" are one theme, and a \
second name for it splits the taxonomy. Mint a new name only for a genuinely distinct \
theme. Two clusters in one batch that are the same theme get the same name.

Return exactly one label for each cluster_id in the batch. Never use a placeholder \
name such as cluster_12.
"""

_LABEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cluster_id": {"type": "integer"},
                    "theme_name": {"type": "string"},
                    "parent_theme": {"type": "string"},
                    "description": {"type": "string"},
                    "related_themes": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "cluster_id", "theme_name", "parent_theme",
                    "description", "related_themes", "confidence",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["labels"],
    "additionalProperties": False,
}


def _cluster_item(summary: dict[str, Any]) -> dict[str, Any]:
    tickers = [str(t) for t in summary.get("tickers", [])]
    item: dict[str, Any] = {"cluster_id": int(summary["cluster_id"]), "n_members": len(tickers)}
    if summary.get("current_name"):
        item["current_name"] = summary["current_name"]
    members = summary.get("members") or []
    if members:
        shown = {m["ticker"] for m in members}
        item["largest_members"] = members
        others = [t for t in tickers if t not in shown]
        if others:
            item["other_tickers"] = others[:CLAUDE_LABEL_MAX_OTHER_TICKERS]
    else:
        item["tickers"] = tickers
    item["sample_headlines"] = list(summary.get("sample_headlines") or [])
    return item


def _batch_prompt(summaries: list[dict[str, Any]], existing_names: list[str]) -> str:
    clusters = [_cluster_item(s) for s in summaries]
    return (
        f"EXISTING THEMES ({len(existing_names)}):\n{json.dumps(sorted(existing_names))}\n\n"
        f"CLUSTERS ({len(clusters)}):\n{json.dumps(clusters, indent=1, ensure_ascii=False)}\n"
    )


def _as_list(value: object) -> list:
    """Registry stores related_themes as a JSON string; accept a list too."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value or "[]")
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return list(value) if value is not None else []


def _snake(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _clean_label(item: Any) -> dict[str, Any] | None:
    """Normalize one returned label; None when it cannot be a real label."""
    try:
        cid = int(item["cluster_id"])
        confidence = float(item["confidence"])
    except (KeyError, TypeError, ValueError):
        return None
    name = _snake(item.get("theme_name", ""))
    if not name or _EPHEMERAL_LABEL_RE.fullmatch(name) or not 0.0 <= confidence <= 1.0:
        return None
    return {
        "cluster_id": cid,
        "theme_name": name,
        "parent_theme": _snake(item.get("parent_theme", "")) or "unknown",
        "description": str(item.get("description", "")).strip(),
        "related_themes": [r for r in (_snake(x) for x in item.get("related_themes") or []) if r],
        "confidence": confidence,
    }


def _label_batch(summaries: list[dict[str, Any]], existing_names: list[str]) -> dict[int, dict[str, Any]]:
    """One call for a batch of clusters -> {cluster_id: label}.

    Clusters missing from the answer, or with an unusable label, are simply
    absent; the caller retries them. ``ClaudeUnavailableError`` propagates.
    """
    wanted = {int(s["cluster_id"]) for s in summaries}
    prompt = _batch_prompt(summaries, existing_names)  # a bad summary is a bug: raise, don't retry
    try:
        result = call_claude_json(
            prompt,
            system=_LABEL_SYSTEM,
            max_tokens=CLAUDE_LABEL_MAX_TOKENS,
            effort=CLAUDE_LABEL_EFFORT,
            json_schema=_LABEL_SCHEMA,
        )
    except ClaudeUnavailableError:
        raise  # no credit / bad key: every later call would fail the same way
    except Exception as exc:
        logger.error("Claude labeling failed for cluster(s) %s: %s", sorted(wanted), exc)
        return {}
    labels: dict[int, dict[str, Any]] = {}
    items = result.get("labels", []) if isinstance(result, dict) else []
    for item in items:
        label = _clean_label(item)
        if label is None or label["cluster_id"] not in wanted or label["cluster_id"] in labels:
            logger.warning("Dropped unusable label from batch %s: %r", sorted(wanted), item)
            continue
        labels[label["cluster_id"]] = label
    return labels


def _label_new_clusters(
    summaries: list[dict[str, Any]], known_names: set[str]
) -> dict[int, dict[str, Any]]:
    """Label clusters CLAUDE_LABEL_BATCH_SIZE per call; retry leftovers one per call.

    ``known_names`` grows with each batch so later batches can reuse names minted
    by earlier ones (the model already sees every cluster within its own batch).
    """
    labels: dict[int, dict[str, Any]] = {}
    for start in range(0, len(summaries), CLAUDE_LABEL_BATCH_SIZE):
        batch = summaries[start : start + CLAUDE_LABEL_BATCH_SIZE]
        got = _label_batch(batch, sorted(known_names))
        missing = [s for s in batch if int(s["cluster_id"]) not in got]
        if missing:
            logger.warning(
                "%d of %d cluster(s) unlabeled in batch; retrying one per call",
                len(missing), len(batch),
            )
            for summary in missing:
                names = known_names | {lab["theme_name"] for lab in got.values()}
                got.update(_label_batch([summary], sorted(names)))
        for label in got.values():
            known_names.add(label["theme_name"])
        labels.update(got)
    return labels


def _label_cluster(summary: dict[str, Any], existing_names: list[str]) -> dict[str, Any]:
    """Label one cluster; ``cluster_<id>`` placeholder on failure.

    Kept for themes/dynamic_theme/emerging.py, which imports it. Note that it
    calls this with one argument, so that call has always raised TypeError and
    fallen back to its own label (no Claude call) -- left as is on purpose.
    """
    cid = int(summary["cluster_id"])
    label = _label_batch([summary], list(existing_names)).get(cid)
    if label is not None:
        return label
    return {"cluster_id": cid, "theme_name": f"cluster_{cid}", "parent_theme": "unknown",
            "description": "", "related_themes": [], "confidence": 0.0}


def label_clusters(
    cluster_summaries: list[dict[str, Any]],
    *,
    as_of: pd.Timestamp | None = None,
    relabel_all: bool = False,
) -> pd.DataFrame:
    """Label clusters and write/upsert theme_registry.parquet — *stably*.

    HDBSCAN cluster ids are not stable week to week, so naively labeling every
    cluster re-names themes that were already correct. Instead each current
    cluster is matched to the prior week's theme centroids; if the best match
    clears ``LABEL_STABILITY_THRESHOLD`` the prior label is carried forward (no
    Claude call). Only clusters with no strong prior match get (re)labeled.
    Hand-pinned seed themes are always appended.

    ``relabel_all`` skips carry-forward for one run so every cluster is labeled
    from its own members — the recovery path when prior names are known to be
    wrong (they would otherwise be carried forward indefinitely).
    """
    ensure_outputs()
    as_of = (as_of or pd.Timestamp.now(tz="UTC")).normalize().tz_localize(None)

    if not cluster_summaries:
        logger.warning("No cluster summaries provided — writing seed themes only")
        return _write_registry([], as_of)

    # Build current + prior theme centroids for stability matching. Best-effort:
    # if anything is missing (e.g. first run, no prior), fall back to labeling all.
    cur_centroids, prior_centroids = ({}, {}) if relabel_all else _stability_centroids()
    prior_meta = _latest_theme_metadata()  # theme_name -> registry row (carry-forward)
    if relabel_all:
        logger.info("relabel_all: carry-forward disabled — labeling every cluster")

    n = len(cluster_summaries)
    logger.info("Labeling %d clusters (stable carry-forward + Claude for new only)...", n)
    # Running set of names already in use this run — seeded from last week's
    # active themes, grown as each cluster is carried forward or freshly
    # labeled, so Claude can reuse an existing name instead of minting a
    # near-duplicate for gray-zone clusters (see _LABEL_SYSTEM).
    known_names = set(prior_meta.keys())

    # Propose a carry-forward for every cluster, then award each prior name to
    # a single cluster. Without the second pass several clusters keep the same
    # name and the duplicate (ticker, theme) rows abort step08.
    # A prior label under MIN_CLUSTER_LABEL_CONFIDENCE is the labeler saying "no single
    # business here". It is never carried: carrying copied one 0.38 description (written
    # for WYFI/GDS/VEON) onto four different clusters 10-05..11-03, the last an
    # AI-infrastructure cluster (NVDA, ANET, SMCI, NBIS), and its low confidence then
    # unpublished that cluster. Such clusters are labeled fresh from their own members.
    proposals: dict[int, tuple[str, float]] = {}
    low_conf_skipped = 0
    for summary in cluster_summaries:
        cid = int(summary["cluster_id"])
        match = _match_prior_theme_scored(cid, cur_centroids, prior_centroids)
        if match is not None and match[0] in prior_meta:
            if _prior_confidence(prior_meta[match[0]]) < MIN_CLUSTER_LABEL_CONFIDENCE:
                low_conf_skipped += 1
                continue
            proposals[cid] = match
    if low_conf_skipped:
        logger.info("Low-confidence prior labels not carried: %d cluster(s) will be relabeled", low_conf_skipped)
    carry_forward, displaced = _resolve_carry_forward(proposals)
    if displaced:
        logger.info(
            "Carry-forward collisions: %d cluster(s) lost a contested prior name "
            "and will be relabeled",
            displaced,
        )

    def _carried(cid: int) -> bool:
        name = carry_forward.get(cid)
        return name is not None and name in prior_meta

    # Every cluster goes to the labeler. A carried one goes with its prior name as
    # current_name: the labeler keeps it unless it is clearly wrong, and always
    # writes a fresh description and confidence. Until 2026-11-17 carried rows
    # copied last week's description and confidence verbatim: 41 of 67 carried
    # descriptions on 11-16 named tickers that had since moved to other clusters,
    # and a wrong carried name could never be corrected.
    to_label = [
        {**s, "current_name": carry_forward[int(s["cluster_id"])]} if _carried(int(s["cluster_id"])) else s
        for s in cluster_summaries
    ]
    logger.info(
        "Claude-labeling %d cluster(s) (%d with a carried name) in batches of %d",
        len(to_label), sum(1 for s in to_label if "current_name" in s), CLAUDE_LABEL_BATCH_SIZE,
    )
    try:
        labels = _label_new_clusters(to_label, known_names)
    except ClaudeUnavailableError:
        logger.error("Claude API unavailable; aborting before theme_registry.parquet is written")
        raise

    rows, kept, renamed, fresh, stale = [], 0, 0, 0, 0
    for summary in cluster_summaries:
        cid = int(summary["cluster_id"])
        label = labels.get(cid)
        prior_name = carry_forward[cid] if _carried(cid) else None
        if label is None and prior_name is not None:
            # labeler failed for this cluster: keep last week's row rather than lose the name
            pr = prior_meta[prior_name]
            label = {"theme_name": prior_name, "parent_theme": str(pr.get("parent_theme", "unknown")),
                     "description": str(pr.get("description", "")),
                     "related_themes": _as_list(pr.get("related_themes")),
                     "confidence": float(pr.get("confidence", 0.0))}
            stale += 1
            note = "carried as is (labeler failed)"
        elif label is None:  # every attempt failed; caught by the check below
            label = {"theme_name": f"cluster_{cid}", "parent_theme": "unknown",
                     "description": "", "related_themes": [], "confidence": 0.0}
            note = "UNLABELED"
        elif prior_name is None:
            fresh += 1
            note = "new"
        elif label["theme_name"] == prior_name:
            kept += 1
            note = "kept"
        else:
            renamed += 1
            note = f"renamed from {prior_name}"
        rows.append({
            "cluster_id": cid,
            "theme_name": label["theme_name"],
            "parent_theme": label["parent_theme"],
            "description": label["description"],
            "related_themes": json.dumps(label["related_themes"]),
            "confidence": float(label["confidence"]),
            "date": as_of,
        })
        logger.info("  cluster %3d → %-30s (%s, conf %.2f)", cid, label["theme_name"], note, label["confidence"])

    logger.info(
        "Labeling done: %d carried name kept, %d carried name replaced, %d new, %d carried as is",
        kept, renamed, fresh, stale,
    )
    unlabeled = sorted(
        r["cluster_id"] for r in rows if _EPHEMERAL_LABEL_RE.fullmatch(str(r["theme_name"]).strip())
    )
    if unlabeled:
        raise LabelingIncompleteError(
            f"{len(unlabeled)} cluster(s) have no real label {unlabeled}; theme_registry.parquet "
            "not written (step08 would reject these names). Rerun the weekly theme stage."
        )
    return _write_registry(rows, as_of)


def _write_registry(rows: list[dict], as_of: pd.Timestamp) -> pd.DataFrame:
    """Upsert today's cluster rows + seed themes (only those not already emergent)."""
    from themes.dynamic_theme.seed_themes import seed_registry_rows

    labeled_names = {r["theme_name"] for r in rows}
    seeds = seed_registry_rows(as_of)
    seeds = seeds[~seeds["theme_name"].isin(labeled_names)]  # gap-fill only, no dupes

    parts = ([pd.DataFrame(rows)] if rows else []) + ([seeds] if not seeds.empty else [])
    new_df = pd.concat(parts, ignore_index=True) if parts else seed_registry_rows(as_of)

    existing = _load_registry_or_empty()
    if not existing.empty:
        existing = existing[existing["date"] < as_of]
    registry = pd.concat([existing, new_df], ignore_index=True)
    registry = registry.sort_values(["date", "cluster_id"]).reset_index(drop=True)
    registry.to_parquet(THEME_REGISTRY_PATH, index=False)
    logger.info("Wrote %s  total_rows=%d (+%d seed themes added)",
                THEME_REGISTRY_PATH, len(registry), len(seeds))
    return registry


def _prior_confidence(row: dict) -> float:
    """A prior label's confidence; missing or unreadable counts as 0 (relabel rather than trust)."""
    try:
        value = float(row.get("confidence"))
    except (TypeError, ValueError):
        return 0.0
    return value if value == value else 0.0


def _latest_theme_metadata() -> dict[str, dict]:
    """{theme_name: latest registry row} for carrying parent/description forward."""
    reg = load_registry(latest_only=True)
    if reg.empty:
        return {}
    return {str(r["theme_name"]): r.to_dict() for _, r in reg.iterrows()}


def _stability_centroids() -> tuple[dict[int, "object"], dict[str, "object"]]:
    """(current cluster_id→centroid, prior theme_name→centroid). Empty on any miss."""
    try:
        import numpy as np  # noqa: F401
        from themes.dynamic_theme.config import TICKER_CLUSTERS_PATH
        from themes.dynamic_theme.stages.step02_embed import load_embeddings_matrix
        from themes.dynamic_theme.stages.step03_cluster import compute_centroids
        from themes.dynamic_theme.stages.step06_discovery import _load_prior_centroids

        tickers, matrix, _ = load_embeddings_matrix()
        cur_clusters = pd.read_parquet(TICKER_CLUSTERS_PATH)
        cur = compute_centroids(matrix, tickers, cur_clusters)
        emb_df = pd.DataFrame({"ticker": tickers, "embedding": list(matrix)})
        prior = _load_prior_centroids(_load_registry_or_empty(), emb_df)
        return cur, prior
    except Exception as exc:
        logger.warning("Stability matching unavailable (%s) — labeling all clusters", exc)
        return {}, {}


def _match_prior_theme_scored(
    cid: int, cur_centroids: dict, prior_centroids: dict
) -> tuple[str, float] | None:
    """Best prior theme_name for cluster ``cid``, with its similarity.

    The score is what makes carry-forward exclusive: when several clusters all
    match the same prior theme, the resolver needs to know which one matches it
    *best* rather than which happened to be iterated first.
    """
    from themes.dynamic_theme.config import LABEL_STABILITY_THRESHOLD
    from themes.dynamic_theme.stages.step06_discovery import _cosine_similarity

    if not prior_centroids or cid not in cur_centroids:
        return None
    best_name, best_sim = None, -1.0
    for name, pc in prior_centroids.items():
        s = _cosine_similarity(cur_centroids[cid], pc)
        if s > best_sim:
            best_name, best_sim = name, s
    if best_name and best_sim >= LABEL_STABILITY_THRESHOLD and not str(best_name).startswith("cluster_"):
        return str(best_name), float(best_sim)
    return None


def _match_prior_theme(cid: int, cur_centroids: dict, prior_centroids: dict) -> str | None:
    """Return a prior theme_name to carry forward if cluster cid matches it well."""
    match = _match_prior_theme_scored(cid, cur_centroids, prior_centroids)
    return match[0] if match else None


def _resolve_carry_forward(
    proposals: dict[int, tuple[str, float]],
) -> tuple[dict[int, str], int]:
    """Award each prior theme name to at most one cluster.

    Carry-forward matches every cluster to its nearest prior centroid
    independently, and nothing stopped two clusters from claiming the same
    name. Downstream that is not a cosmetic duplicate: membership scores are
    computed per cluster, so one (ticker, theme) pair ends up with two
    different scores and step08's immutable-history guard refuses the write.

    On 2026-08-17 this collapsed 187 clusters onto 88 names — seven of them
    called ``mortgage_reits``. Highest similarity wins the name; the clusters
    that lose fall through to a fresh label, which is the honest outcome for a
    grouping that is not actually last week's theme.

    Returns the winning ``{cluster_id: theme_name}`` and the number displaced.
    """
    claimed: dict[str, int] = {}
    # Deterministic: strongest match first, cluster id breaking exact ties.
    for cid, (name, sim) in sorted(proposals.items(), key=lambda kv: (-kv[1][1], kv[0])):
        claimed.setdefault(name, cid)
    carried = {cid: name for name, cid in claimed.items()}
    return carried, len(proposals) - len(carried)


def _load_registry_or_empty() -> pd.DataFrame:
    if THEME_REGISTRY_PATH.exists():
        try:
            return pd.read_parquet(THEME_REGISTRY_PATH)
        except Exception:
            pass
    return pd.DataFrame(
        columns=["cluster_id", "theme_name", "parent_theme", "description", "related_themes", "confidence", "date"]
    )


def load_registry(*, latest_only: bool = True) -> pd.DataFrame:
    """Load theme registry. If latest_only=True returns only the most recent date's rows."""
    df = _load_registry_or_empty()
    if latest_only and not df.empty and "date" in df.columns:
        latest = df["date"].max()
        df = df[df["date"] == latest].copy()
    return df
