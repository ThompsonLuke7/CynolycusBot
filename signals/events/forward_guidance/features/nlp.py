"""Forward-looking guidance extraction, structured text features, and embeddings."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from signals.events.forward_guidance.config import (
    ALT_FINBERT_MODEL,
    DEFAULT_FINBERT_MODEL,
    DEFAULT_SEC_BERT_MODEL,
    DEFAULT_SENTENCE_MODEL,
    EMBEDDINGS_DIR,
)


FORWARD_KEYWORDS = (
    "guidance",
    "outlook",
    "expect",
    "expects",
    "expected",
    "forecast",
    "project",
    "projects",
    "projected",
    "future",
    "demand",
    "backlog",
    "orders",
    "pipeline",
    "full year",
    "fiscal year",
    "fy",
    "next quarter",
    "coming quarter",
    "margin",
    "capex",
    "capital expenditure",
)

GUIDANCE_HEADINGS = (
    "guidance",
    "business outlook",
    "financial outlook",
    "outlook",
    "forward-looking",
    "expectations",
    "fiscal year",
    "full year",
    "third quarter",
    "fourth quarter",
    "first quarter",
    "second quarter",
)

QA_HEADINGS = (
    "question-and-answer",
    "question and answer",
    "q&a",
    "questions and answers",
    "analyst question",
)

# Every field emitted by this post-earnings text extractor is research/result
# evidence, never decision-time scheduled-catalyst evidence.  Keep the
# declaration beside the producer so adapter boundaries fail closed when the
# extractor surface grows.
FORWARD_SECTION_OUTPUT_FIELDS = frozenset({"forward_guidance", "qa"})
STRUCTURED_GUIDANCE_OUTPUT_FIELDS = frozenset(
    {
        "guidance_text_chars",
        "guidance_revenue_raise_cut",
        "guidance_eps_raise_cut",
        "margin_expansion_score",
        "ai_demand_mentions",
        "backlog_order_mentions",
        "uncertainty_language",
        "confidence_language",
        "capex_expansion",
        "hiring_slowdown",
        "guidance_strength_score",
    }
)
FINBERT_OUTPUT_FIELDS = frozenset(
    {
        "finbert_positive",
        "finbert_negative",
        "finbert_neutral",
        "finbert_tone_score",
    }
)
ALT_FINBERT_OUTPUT_FIELDS = frozenset(f"alt_{name}" for name in FINBERT_OUTPUT_FIELDS)
FORWARD_GUIDANCE_EXTRACTOR_OUTPUT_FIELDS = (
    FORWARD_SECTION_OUTPUT_FIELDS
    | STRUCTURED_GUIDANCE_OUTPUT_FIELDS
    | FINBERT_OUTPUT_FIELDS
    | ALT_FINBERT_OUTPUT_FIELDS
    | {"embedding_available", "finbert_available"}
)
FORWARD_GUIDANCE_EXTRACTOR_OUTPUT_PREFIXES = ("emb_", "metric_")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s+", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _paragraphs(text: str) -> list[str]:
    chunks = re.split(r"\n\s*\n|(?<=[.!?])\s+(?=[A-Z][A-Za-z0-9,(])", normalize_text(text))
    return [c.strip() for c in chunks if c and len(c.strip()) > 20]


def _heading_matches(paragraph: str, headings: Iterable[str]) -> bool:
    p = paragraph.strip().lower()
    if len(p) > 160:
        p = p[:160]
    return any(h in p for h in headings)


def _collect_heading_window(paragraphs: list[str], headings: Iterable[str], *, max_paragraphs: int = 12) -> str:
    selected: list[str] = []
    for i, para in enumerate(paragraphs):
        if not _heading_matches(para, headings):
            continue
        selected.extend(paragraphs[i : i + max_paragraphs])
    return "\n\n".join(dict.fromkeys(selected))


def _keyword_sentences(paragraphs: list[str]) -> str:
    selected = []
    for para in paragraphs:
        low = para.lower()
        if any(k in low for k in FORWARD_KEYWORDS):
            selected.append(para)
    return "\n\n".join(selected)


def extract_forward_sections(text: str) -> dict[str, str]:
    """Return forward guidance and Q&A text only, falling back to keyword spans."""
    if not text:
        return {"forward_guidance": "", "qa": ""}
    paragraphs = _paragraphs(text)
    guidance = _collect_heading_window(paragraphs, GUIDANCE_HEADINGS)
    if not guidance:
        guidance = _keyword_sentences(paragraphs)
    qa = _collect_heading_window(paragraphs, QA_HEADINGS, max_paragraphs=20)
    return {
        "forward_guidance": normalize_text(guidance),
        "qa": normalize_text(qa),
    }


def _count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.I))


def _directional_score(pos: int, neg: int) -> float:
    denom = max(pos + neg, 1)
    return float((pos - neg) / denom)


def _near(term_pattern: str, pos_pattern: str, neg_pattern: str, text: str, *, window: int = 90) -> float:
    score = 0
    for m in re.finditer(term_pattern, text, flags=re.I):
        lo = max(0, m.start() - window)
        hi = min(len(text), m.end() + window)
        span = text[lo:hi]
        score += _count(pos_pattern, span)
        score -= _count(neg_pattern, span)
    return float(np.clip(score, -3, 3) / 3.0)


def extract_structured_guidance_features(text: str) -> dict[str, float]:
    """Create deterministic language features for guidance/reaction disagreement."""
    low = normalize_text(text).lower()
    if not low:
        return {
            "guidance_text_chars": 0.0,
            "guidance_revenue_raise_cut": 0.0,
            "guidance_eps_raise_cut": 0.0,
            "margin_expansion_score": 0.0,
            "ai_demand_mentions": 0.0,
            "backlog_order_mentions": 0.0,
            "uncertainty_language": 0.0,
            "confidence_language": 0.0,
            "capex_expansion": 0.0,
            "hiring_slowdown": 0.0,
            "guidance_strength_score": 0.0,
        }

    raise_words = r"\b(raise|raised|raising|increase|increased|higher|above|beat|strong|accelerat\w+|improv\w+)\b"
    cut_words = r"\b(cut|lower|lowered|reduce|reduced|below|weak|declin\w+|decelerat\w+|pressure|headwind)\b"

    rev_score = _near(r"\b(revenue|sales|top line|bookings)\b", raise_words, cut_words, low)
    eps_score = _near(r"\b(eps|earnings per share|profit|net income|bottom line)\b", raise_words, cut_words, low)

    margin_pos = _count(r"\b(margin expansion|expand\w+ margin|gross margin.*improv|operating margin.*improv)\b", low)
    margin_neg = _count(r"\b(margin compression|compress\w+ margin|margin pressure|gross margin.*declin)\b", low)
    margin_score = _directional_score(margin_pos, margin_neg)

    ai_mentions = _count(r"\b(ai|artificial intelligence|accelerated computing|genai|generative ai)\b", low)
    backlog_mentions = _count(r"\b(backlog|bookings|orders?|remaining performance obligation|rpo)\b", low)
    uncertainty = _count(r"\b(uncertain|uncertainty|cautious|volatility|visibility|headwinds?|challeng\w+)\b", low)
    confidence = _count(r"\b(confident|confidence|strong demand|robust demand|momentum|resilient|visibility improved)\b", low)
    capex = _count(r"\b(capex|capital expenditure|capacity expansion|expand capacity|data center investment)\b", low)
    hiring_slow = _count(r"\b(hiring slowdown|slow hiring|headcount reduction|layoff|restructuring)\b", low)

    strength = (
        0.25 * rev_score
        + 0.20 * eps_score
        + 0.15 * margin_score
        + 0.12 * np.tanh(ai_mentions / 3.0)
        + 0.10 * np.tanh(backlog_mentions / 4.0)
        + 0.10 * np.tanh(confidence / 4.0)
        + 0.08 * np.tanh(capex / 3.0)
        - 0.18 * np.tanh(uncertainty / 5.0)
        - 0.10 * np.tanh(hiring_slow / 2.0)
    )

    return {
        "guidance_text_chars": float(len(low)),
        "guidance_revenue_raise_cut": float(rev_score),
        "guidance_eps_raise_cut": float(eps_score),
        "margin_expansion_score": float(margin_score),
        "ai_demand_mentions": float(ai_mentions),
        "backlog_order_mentions": float(backlog_mentions),
        "uncertainty_language": float(uncertainty),
        "confidence_language": float(confidence),
        "capex_expansion": float(capex),
        "hiring_slowdown": float(hiring_slow),
        "guidance_strength_score": float(np.clip(strength, -1.0, 1.0)),
    }


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")


@dataclass
class EmbeddingCache:
    root: Path = EMBEDDINGS_DIR

    def path_for(self, *, event_id: str, backend: str, model_name: str, content_hash: str) -> Path:
        return self.root / backend / model_slug(model_name) / f"{event_id}_{content_hash}.npy"

    def load(self, *, event_id: str, backend: str, model_name: str, content_hash: str) -> np.ndarray | None:
        path = self.path_for(event_id=event_id, backend=backend, model_name=model_name, content_hash=content_hash)
        if not path.exists():
            return None
        return np.load(path)

    def save(
        self,
        values: np.ndarray,
        *,
        event_id: str,
        backend: str,
        model_name: str,
        content_hash: str,
    ) -> Path:
        path = self.path_for(event_id=event_id, backend=backend, model_name=model_name, content_hash=content_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, values.astype(np.float32))
        return path


def embed_sentence_transformer(
    text: str,
    *,
    model_name: str = DEFAULT_SENTENCE_MODEL,
    event_id: str | None = None,
    cache: EmbeddingCache | None = None,
) -> np.ndarray:
    """Generate a sentence-transformer embedding with optional local cache."""
    content_hash = text_hash(text)
    cache = cache or EmbeddingCache()
    if event_id:
        cached = cache.load(event_id=event_id, backend="sentence_transformers", model_name=model_name, content_hash=content_hash)
        if cached is not None:
            return cached
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("Install sentence-transformers to generate semantic guidance embeddings.") from exc
    model = SentenceTransformer(model_name, device="cpu")
    values = np.asarray(model.encode([text or ""], normalize_embeddings=True)[0], dtype=np.float32)
    if event_id:
        cache.save(values, event_id=event_id, backend="sentence_transformers", model_name=model_name, content_hash=content_hash)
    return values


def score_finbert(
    text: str,
    *,
    model_name: str = DEFAULT_FINBERT_MODEL,
    max_chunks: int = 16,
) -> dict[str, float]:
    """Run a Hugging Face financial tone classifier on CPU and average chunks."""
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError("Install transformers to generate FinBERT tone features.") from exc
    chunks = _paragraphs(text)[:max_chunks] or [text[:512]]
    clf = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, device=-1, truncation=True)
    rows = clf(chunks)
    totals = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for row in rows:
        label = str(row.get("label", "")).lower()
        score = float(row.get("score", 0.0))
        if "pos" in label or label == "label_1":
            totals["positive"] += score
        elif "neg" in label or label == "label_2":
            totals["negative"] += score
        else:
            totals["neutral"] += score
    n = max(len(rows), 1)
    return {
        "finbert_positive": totals["positive"] / n,
        "finbert_negative": totals["negative"] / n,
        "finbert_neutral": totals["neutral"] / n,
        "finbert_tone_score": (totals["positive"] - totals["negative"]) / n,
    }


def score_alt_finbert(text: str, *, model_name: str = ALT_FINBERT_MODEL) -> dict[str, float]:
    values = score_finbert(text, model_name=model_name)
    return {f"alt_{k}": v for k, v in values.items()}


def embed_transformer_mean(
    text: str,
    *,
    model_name: str = DEFAULT_SEC_BERT_MODEL,
    event_id: str | None = None,
    cache: EmbeddingCache | None = None,
) -> np.ndarray:
    """Mean-pool a Hugging Face transformer as an optional SEC-BERT backend."""
    content_hash = text_hash(text)
    cache = cache or EmbeddingCache()
    if event_id:
        cached = cache.load(event_id=event_id, backend="transformer_mean", model_name=model_name, content_hash=content_hash)
        if cached is not None:
            return cached
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install torch and transformers to generate SEC-BERT embeddings.") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    encoded = tokenizer(text or "", padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded)
    mask = encoded["attention_mask"].unsqueeze(-1).expand(output.last_hidden_state.size()).float()
    values = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    arr = values[0].detach().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    if event_id:
        cache.save(arr, event_id=event_id, backend="transformer_mean", model_name=model_name, content_hash=content_hash)
    return arr


def embedding_feature_dict(values: np.ndarray, *, prefix: str) -> dict[str, float]:
    return {f"{prefix}_{i:04d}": float(v) for i, v in enumerate(values.astype(float).ravel())}
