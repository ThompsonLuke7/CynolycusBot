"""Real-time / batch catalyst scoring using the trained XGBoost booster.

Given a freshly-collected news record (or a small batch), produce a
``catalyst_score`` ∈ [0, 1] that ranks expansion probability. This is the
inference-time path that replaces the cosine-similarity score in
``news/scoring.py``.

Flow per record:
  1. Build text (headline + summary + body).
  2. Run BGE embedding (384-dim) + FinBERT (3 scores).
  3. Apply the regex classifier from ``news/catalyst_types.classify_catalyst_types``
     to get a coarse catalyst_family / catalyst_subtype.
  4. Look up ticker profile (sector, marketCap, beta, float, avg_vol) from
     the universe profile parquet.
  5. Look up FINRA short_ratio z-score for the trade date.
  6. Assemble a feature row in the exact order of the training manifest.
  7. Run the booster ``.predict`` and return the score.

The class caches the BGE model, FinBERT model, profile lookups, FINRA lookups,
and the booster so subsequent calls are fast (~5-10 ms per record).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from news.catalyst_types import classify_catalyst_types
from news.config import (
    CBOE_OPTIONS_SUMMARY_PATH,
    FINRA_SHORT_VOLUME_PATH,
    TICKER_PROFILE_PATH,
)


DEFAULT_MODEL_PATH = Path("meta_context/models/news_catalyst_target_expansion_10pct_v2.json")
DEFAULT_MANIFEST_PATH = Path("meta_context/data/processed/catalyst_training_manifest.json")
BARS_DAILY_DIR = Path("Data/shared/bars/1d")
KAGGLE_MARKET_PRICES = Path("/home/luket/.cache/kagglehub/datasets/belbino/financial-news-sentiment-vs-market-2020-present/versions/50/market_prices.csv")


class CatalystScorer:
    """Single-record / small-batch live scorer for catalyst classification."""

    def __init__(
        self,
        *,
        model_path: Path = DEFAULT_MODEL_PATH,
        manifest_path: Path = DEFAULT_MANIFEST_PATH,
        engine: str = "auto",
        use_finbert: bool = True,
        device: str = "cpu",
    ) -> None:
        self.model_path = Path(model_path)
        self.manifest = json.loads(Path(manifest_path).read_text())
        self.feature_cols: list[str] = self.manifest["feature_columns"]
        self.family_levels: list[str] = self.manifest["family_levels"]
        self.subtype_levels: list[str] = self.manifest["subtype_levels"]
        self.sector_levels: list[str] = self.manifest["sector_levels"]
        self.bge_dim: int = int(self.manifest.get("bge_dim", 384))

        # Pick engine from file extension if auto
        if engine == "auto":
            engine = "xgboost" if str(self.model_path).endswith(".json") else "lightgbm"
        self.engine = engine

        self.booster = self._load_booster()

        # Defer BGE / FinBERT loading until first scoring call (heavy imports)
        self._bge_model = None
        self._finbert_pipeline = None
        self._use_finbert = bool(use_finbert)
        self._device = device

        # Side-channel data
        self._profiles = self._load_profiles()
        self._finra = self._load_finra_lookup()
        self._cboe = self._load_cboe_lookup()
        self._macro = self._load_macro_lookup()
        self._bars_cache: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------
    def _load_booster(self):
        if self.engine == "xgboost":
            import xgboost as xgb
            booster = xgb.XGBClassifier()
            booster.load_model(str(self.model_path))
            return booster
        elif self.engine == "lightgbm":
            import lightgbm as lgb
            return lgb.Booster(model_file=str(self.model_path))
        raise ValueError(f"unknown engine: {self.engine}")

    def _load_profiles(self) -> dict[str, dict]:
        if not Path(TICKER_PROFILE_PATH).exists():
            return {}
        df = pd.read_parquet(TICKER_PROFILE_PATH)
        cols = ["ticker", "sector", "marketCap", "floatShares", "averageVolume", "beta"]
        df = df[[c for c in cols if c in df.columns]]
        return {row["ticker"]: row.to_dict() for _, row in df.iterrows()}

    def _load_finra_lookup(self) -> pd.DataFrame:
        if not Path(FINRA_SHORT_VOLUME_PATH).exists():
            return pd.DataFrame()
        df = pd.read_parquet(FINRA_SHORT_VOLUME_PATH)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None).astype("datetime64[ns]")
        df = df.groupby(["ticker", "date"], as_index=False)[["short_volume", "total_volume"]].sum()
        df["short_ratio"] = df["short_volume"] / df["total_volume"].clip(lower=1)
        df = df.sort_values(["ticker", "date"])
        grp = df.groupby("ticker")
        df["mean_ratio"] = grp["short_ratio"].transform(lambda s: s.rolling(20, min_periods=5).mean())
        df["std_ratio"] = grp["short_ratio"].transform(lambda s: s.rolling(20, min_periods=5).std())
        df["short_z"] = (df["short_ratio"] - df["mean_ratio"]) / df["std_ratio"].replace(0, np.nan)
        return df.set_index(["ticker", "date"])[["short_ratio", "short_z"]]

    def _load_cboe_lookup(self) -> dict[str, dict]:
        if not Path(CBOE_OPTIONS_SUMMARY_PATH).exists():
            return {}
        df = pd.read_parquet(CBOE_OPTIONS_SUMMARY_PATH)
        df = df.sort_values("snapshot_date").drop_duplicates("ticker", keep="last")
        cols = ["ticker", "iv30", "iv30_change_percent", "put_call_volume_ratio"]
        df = df[[c for c in cols if c in df.columns]]
        return {row["ticker"]: row.to_dict() for _, row in df.iterrows()}

    def _load_macro_lookup(self) -> pd.DataFrame:
        """Kaggle market_prices.csv keyed by date for SPY/QQQ/VIX regime features."""
        if not KAGGLE_MARKET_PRICES.exists():
            return pd.DataFrame()
        df = pd.read_csv(KAGGLE_MARKET_PRICES)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None).astype("datetime64[ns]")
        keep = [
            "date", "spy_return_1d", "spy_return_5d", "spy_return_20d", "spy_range_pct", "spy_gap_pct",
            "qqq_return_1d", "qqq_return_5d", "qqq_return_20d",
            "vix_close", "vix_pct_chg", "market_direction",
        ]
        keep = [c for c in keep if c in df.columns]
        return df[keep].set_index("date").sort_index()

    def _bars_for(self, ticker: str) -> pd.DataFrame | None:
        """Load + cache per-ticker daily bars with pre-computed price-action features."""
        if ticker in self._bars_cache:
            return self._bars_cache[ticker]
        path = BARS_DAILY_DIR / f"{ticker}.parquet"
        if not path.exists():
            self._bars_cache[ticker] = None
            return None
        try:
            b = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close", "volume"])
        except Exception:
            self._bars_cache[ticker] = None
            return None
        b["date"] = pd.to_datetime(b["timestamp"], utc=True).dt.tz_convert(None).dt.tz_localize(None).dt.floor("D").astype("datetime64[ns]")
        b = b.sort_values("date").reset_index(drop=True)
        b["ret_1d"] = b["close"].pct_change(1)
        b["ret_5d"] = b["close"].pct_change(5)
        b["ret_20d"] = b["close"].pct_change(20)
        b["vol_20d"] = b["ret_1d"].rolling(20, min_periods=5).std()
        b["high_20d"] = b["high"].rolling(20, min_periods=5).max()
        b["dist_from_high_20d"] = b["close"] / b["high_20d"] - 1.0
        b["high_252d"] = b["high"].rolling(252, min_periods=20).max()
        b["dist_from_high_252d"] = b["close"] / b["high_252d"] - 1.0
        delta = b["ret_1d"]
        gain = delta.clip(lower=0).rolling(14, min_periods=5).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean()
        rs = gain / loss.replace(0, np.nan)
        b["rsi_14"] = 100 - 100 / (1 + rs)
        cached = b.set_index("date")[
            ["ret_1d", "ret_5d", "ret_20d", "vol_20d", "dist_from_high_20d", "dist_from_high_252d", "rsi_14"]
        ]
        self._bars_cache[ticker] = cached
        return cached

    def _ensure_bge(self):
        if self._bge_model is None:
            from news.nlp import embed_texts_bge
            self._embed_fn = embed_texts_bge
            self._bge_model = True  # sentinel
        return self._embed_fn

    def _ensure_finbert(self):
        if not self._use_finbert:
            return None
        if self._finbert_pipeline is None:
            from news.nlp import finbert_scores_batch
            self._finbert_pipeline = finbert_scores_batch
        return self._finbert_pipeline

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def score(self, records: Iterable[dict] | dict) -> np.ndarray:
        """Score one record (dict) or a list of records. Returns float array."""
        if isinstance(records, dict):
            records = [records]
        records = list(records)
        if not records:
            return np.array([], dtype=np.float32)

        df = pd.DataFrame(records)
        df["ticker"] = df["ticker"].astype(str).str.upper()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

        # Classify (regex pass — same as the offline pipeline's first stage)
        df = classify_catalyst_types(df)

        # Text → BGE
        texts = (df.get("headline", "").fillna("") + " "
                 + df.get("summary", "").fillna("") + " "
                 + df.get("body", "").fillna("")).tolist()
        embed_fn = self._ensure_bge()
        emb = embed_fn(texts)
        if emb.shape[1] != self.bge_dim:
            raise RuntimeError(f"BGE returned {emb.shape[1]} dims, expected {self.bge_dim}")

        # FinBERT scores (optional but recommended for parity with training)
        finbert_fn = self._ensure_finbert()
        if finbert_fn is not None:
            fb_rows = finbert_fn(texts)  # list of dicts {positive, negative, neutral}
            df["finbert_positive_score"] = [float(r.get("positive", 0)) for r in fb_rows]
            df["finbert_negative_score"] = [float(r.get("negative", 0)) for r in fb_rows]
            df["finbert_neutral_score"] = [float(r.get("neutral", 0)) for r in fb_rows]
        else:
            df["finbert_positive_score"] = 0.0
            df["finbert_negative_score"] = 0.0
            df["finbert_neutral_score"] = 0.0

        X = self._assemble_feature_matrix(df, emb)
        proba = self._predict(X)
        return proba

    # ------------------------------------------------------------------
    # Feature assembly
    # ------------------------------------------------------------------
    def _assemble_feature_matrix(self, df: pd.DataFrame, emb: np.ndarray) -> pd.DataFrame:
        n = len(df)
        out = pd.DataFrame(0.0, index=range(n), columns=self.feature_cols, dtype=np.float32)

        # BGE
        bge_cols = [f"bge_{i:03d}" for i in range(self.bge_dim)]
        for i, col in enumerate(bge_cols):
            if col in out.columns:
                out[col] = emb[:, i].astype(np.float32)

        # FinBERT
        for col in ("finbert_positive_score", "finbert_negative_score", "finbert_neutral_score"):
            if col in out.columns and col in df.columns:
                out[col] = df[col].astype(np.float32).values

        # catalyst_family one-hot
        fam = df["catalyst_family"].fillna("none").astype(str)
        for level in self.family_levels:
            col = f"fam_{level}"
            if col in out.columns:
                out[col] = (fam == level).astype(np.float32).values
        if "fam_other" in out.columns:
            out["fam_other"] = (~fam.isin(self.family_levels)).astype(np.float32).values

        # catalyst_subtype one-hot
        sub = df["catalyst_subtype"].fillna("none").astype(str)
        for level in self.subtype_levels:
            col = f"sub_{level}"
            if col in out.columns:
                out[col] = (sub == level).astype(np.float32).values
        if "sub_other" in out.columns:
            out["sub_other"] = (~sub.isin(self.subtype_levels)).astype(np.float32).values

        # Profile features
        for i, ticker in enumerate(df["ticker"].astype(str).values):
            prof = self._profiles.get(ticker, {})
            if "prof_marketCap_log" in out.columns:
                mc = float(prof.get("marketCap") or 0)
                out.at[i, "prof_marketCap_log"] = float(np.log1p(mc))
            if "prof_float_log" in out.columns:
                fl = float(prof.get("floatShares") or 0)
                out.at[i, "prof_float_log"] = float(np.log1p(fl))
            if "prof_avg_vol_log" in out.columns:
                av = float(prof.get("averageVolume") or 0)
                out.at[i, "prof_avg_vol_log"] = float(np.log1p(av))
            if "prof_beta" in out.columns:
                out.at[i, "prof_beta"] = float(prof.get("beta") or 1.0)
            # Sector one-hot
            sector = prof.get("sector") or "none"
            sector_col = f"sec_{sector}"
            if sector_col in out.columns:
                out.at[i, sector_col] = 1.0
            elif "sec_other" in out.columns:
                out.at[i, "sec_other"] = 1.0

        # FINRA short z-score at event date
        if not self._finra.empty:
            for i, (ticker, ts) in enumerate(zip(df["ticker"], df["timestamp"])):
                key = (ticker, ts.tz_convert("UTC").tz_localize(None).floor("D")) if ts is not pd.NaT else None
                try:
                    row = self._finra.loc[key]
                    if "finra_short_ratio" in out.columns:
                        out.at[i, "finra_short_ratio"] = float(row.get("short_ratio", 0))
                    if "finra_short_z" in out.columns:
                        out.at[i, "finra_short_z"] = float(row.get("short_z", 0) or 0)
                except KeyError:
                    pass

        # Kaggle macro features at event date (forward-fill so a weekend event uses Friday's value)
        if not self._macro.empty:
            event_dates = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).dt.floor("D").astype("datetime64[ns]")
            macro_rows = self._macro.reindex(event_dates.values, method="ffill")
            for col in self._macro.columns:
                target_col = f"macro_{col}"
                if target_col in out.columns:
                    out[target_col] = macro_rows[col].fillna(0.0).astype(np.float32).values

        # Per-ticker price-action features at event date (load + cache bars lazily)
        px_cols = ["ret_1d", "ret_5d", "ret_20d", "vol_20d",
                   "dist_from_high_20d", "dist_from_high_252d", "rsi_14"]
        if any(f"px_{c}" in out.columns for c in px_cols):
            for i, (ticker, ts) in enumerate(zip(df["ticker"], df["timestamp"])):
                bars = self._bars_for(ticker)
                if bars is None or bars.empty:
                    continue
                ev_date = ts.tz_convert("UTC").tz_localize(None).floor("D").to_datetime64()
                try:
                    row = bars.reindex([ev_date], method="ffill").iloc[0]
                except Exception:
                    continue
                for col in px_cols:
                    target_col = f"px_{col}"
                    if target_col in out.columns:
                        val = row.get(col)
                        if val is not None and not pd.isna(val):
                            out.at[i, target_col] = float(val)

        return out

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.engine == "xgboost":
            import xgboost as xgb
            # XGBoost predicts P(class=1)
            arr = self.booster.predict_proba(X)
            return arr[:, 1].astype(np.float32)
        elif self.engine == "lightgbm":
            return self.booster.predict(X).astype(np.float32)
        raise ValueError(self.engine)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_scorer: CatalystScorer | None = None


def get_default_scorer() -> CatalystScorer:
    global _default_scorer
    if _default_scorer is None:
        _default_scorer = CatalystScorer()
    return _default_scorer


def score_records(records: list[dict] | dict) -> np.ndarray:
    """One-shot scoring of a record or list of records via the default scorer."""
    return get_default_scorer().score(records)
