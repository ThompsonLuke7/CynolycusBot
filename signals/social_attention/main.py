"""CLI for the Reddit social-attention pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from signals.social_attention.config import DEFAULT_START, DEFAULT_SUBREDDITS, ensure_dirs


def _setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper()), format="%(asctime)s %(levelname)s %(name)s %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Reddit social-attention pipeline.")
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "collect-reddit",
            "enrich-praw",
            "extract",
            "features",
            "sentiment",
            "embed",
            "cluster",
            "narrative-features",
            "labels",
            "train-lightgbm",
            "all-offline",
        ],
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=None)
    parser.add_argument("--subreddits", nargs="*", default=list(DEFAULT_SUBREDDITS))
    parser.add_argument("--query", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cpu", help="Embedding device; default keeps local runs CPU-bound.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--log", default="INFO")
    args = parser.parse_args()
    _setup_logging(args.log)
    ensure_dirs()

    if args.stage == "collect-reddit":
        if not args.end:
            raise ValueError("--end is required for collect-reddit")
        from signals.social_attention.reddit_collector import collect_reddit_history

        collect_reddit_history(
            start=args.start,
            end=args.end,
            subreddits=args.subreddits,
            q=args.query,
            resume=not args.no_resume,
        )
    elif args.stage == "enrich-praw":
        from signals.social_attention.config import REDDIT_POSTS_PATH
        from signals.social_attention.io import read_table, write_table
        from signals.social_attention.praw_enrichment import enrich_posts_with_praw

        write_table(enrich_posts_with_praw(read_table(REDDIT_POSTS_PATH), env_file=args.env_file), REDDIT_POSTS_PATH)
    elif args.stage == "extract":
        from signals.social_attention.ticker_extractor import extract_mentions

        extract_mentions()
    elif args.stage == "features":
        from signals.social_attention.attention_features import build_attention_features

        build_attention_features()
    elif args.stage == "sentiment":
        from signals.social_attention.sentiment import score_reddit_sentiment

        score_reddit_sentiment()
    elif args.stage == "embed":
        from signals.social_attention.embeddings import build_social_embeddings

        build_social_embeddings(device=args.device)
    elif args.stage == "cluster":
        from signals.social_attention.narrative_clustering import cluster_social_embeddings

        cluster_social_embeddings()
    elif args.stage == "narrative-features":
        from signals.social_attention.narrative_clustering import build_narrative_features

        build_narrative_features()
    elif args.stage == "labels":
        from signals.social_attention.labels import build_social_labels

        build_social_labels()
    elif args.stage == "train-lightgbm":
        from signals.social_attention.train_lightgbm import train_lightgbm

        print(json.dumps(train_lightgbm(force=args.force), indent=2, default=str))
    elif args.stage == "all-offline":
        from signals.social_attention.attention_features import build_attention_features
        from signals.social_attention.embeddings import build_social_embeddings
        from signals.social_attention.labels import build_social_labels
        from signals.social_attention.narrative_clustering import build_narrative_features, cluster_social_embeddings
        from signals.social_attention.sentiment import score_reddit_sentiment
        from signals.social_attention.ticker_extractor import extract_mentions

        score_reddit_sentiment()
        extract_mentions()
        build_attention_features()
        build_social_embeddings(device=args.device)
        cluster_social_embeddings()
        build_narrative_features()
        build_social_labels()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
