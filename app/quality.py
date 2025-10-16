"""Lightweight auto-QC checks for operator submissions.

The goal is to provide deterministic, dependency-light heuristics that catch
the most common failure modes (deduplication, toxicity, PII leakage) without
pulling heavyweight ML models into the deploy path. The checks are intentionally
transparent so that humans can reason about why a submission passed or failed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class QCThresholds:
    max_dedup: float = 0.85
    max_toxic: float = 0.1
    allow_pii: bool = False


_threshold_cache: Optional[QCThresholds] = None


def _load_thresholds() -> QCThresholds:
    global _threshold_cache
    if _threshold_cache is None:  # pragma: no branch
        from app.config import settings

        _threshold_cache = QCThresholds(
            max_dedup=float(getattr(settings, "QC_MAX_DEDUP_SCORE", 0.85)),
            max_toxic=float(getattr(settings, "QC_MAX_TOXICITY", 0.1)),
            allow_pii=bool(getattr(settings, "QC_ALLOW_PII", False)),
        )
    return _threshold_cache


TOXIC_LEXICON = {
    "idiot",
    "stupid",
    "hate",
    "kill",
    "moron",
    "dumb",
    "racist",
    "sexist",
    "trash",
    "nazi",
}

PII_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"[\w\.-]+@[\w\.-]+", re.IGNORECASE),
    re.compile(r"\b\+?\d{1,3}[\s-]?\(?\d{2,3}\)?[\s-]?\d{3}[\s-]?\d{2,4}\b"),
    re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),  # SSN-like patterns
)


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[\w']+", text.lower())
    return tokens


def _shingles(tokens: Iterable[str], size: int = 3) -> list[str]:
    window = []
    shingles: list[str] = []
    for token in tokens:
        window.append(token)
        if len(window) == size:
            shingles.append(" ".join(window))
            window.pop(0)
    return shingles


def _dedup_score(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    shingles = _shingles(tokens)
    if not shingles:
        return 0.0
    counts = Counter(shingles)
    total = sum(counts.values())
    unique = len(counts)
    # Higher score = more duplication. Ratio of repeated shingles.
    repeated = total - unique
    return repeated / total if total else 0.0


def _toxicity_score(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    toxic_hits = sum(1 for t in tokens if t in TOXIC_LEXICON)
    return toxic_hits / len(tokens)


def _pii_hits(text: str) -> list[str]:
    hits: list[str] = []
    for pattern in PII_PATTERNS:
        hits.extend(pattern.findall(text))
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for hit in hits:
        if hit not in seen:
            deduped.append(hit)
            seen.add(hit)
    return deduped


def run_auto_qc(raw_text: str, thresholds: Optional[QCThresholds] = None) -> Dict[str, Any]:
    """Run deterministic heuristics and return a QC report."""

    text = (raw_text or "").strip()
    tokens = _tokenize(text)

    dedup = _dedup_score(tokens)
    toxicity = _toxicity_score(tokens)
    pii = _pii_hits(text)
    token_count = len(tokens)

    active_thresholds = thresholds or _load_thresholds()

    passed = True
    reasons: list[str] = []

    if dedup > active_thresholds.max_dedup:
        passed = False
        reasons.append(f"dedup>{active_thresholds.max_dedup:.2f}")

    if toxicity > active_thresholds.max_toxic:
        passed = False
        reasons.append(f"toxicity>{active_thresholds.max_toxic:.2f}")

    if pii and not active_thresholds.allow_pii:
        passed = False
        reasons.append("pii_detected")

    quality_bonus = bool(passed and token_count > 80 and dedup < 0.2 and toxicity == 0.0)

    report: Dict[str, Any] = {
        "passed": passed,
        "reasons": reasons,
        "token_count": token_count,
        "dedup_score": round(dedup, 4),
        "toxicity_score": round(toxicity, 4),
        "pii_matches": pii,
        "quality_bonus": quality_bonus,
    }

    return report
