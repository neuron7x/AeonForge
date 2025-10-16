"""Simple reputation scoring helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from app.models import User

_thresholds_cache: List[int] | None = None


def _load_thresholds() -> List[int]:
    from app.config import settings

    return sorted({int(x) for x in settings.LEVEL_THRESHOLDS})


def _get_thresholds() -> List[int]:
    global _thresholds_cache
    if _thresholds_cache is None:
        try:
            _thresholds_cache = list(_load_thresholds())
        except ModuleNotFoundError:
            _thresholds_cache = [0, 10, 25, 45]
    return _thresholds_cache


def calculate_level(score: float) -> int:
    """Map a continuous reputation score to an integer access level."""

    thresholds = _get_thresholds()
    level = 0
    for idx, threshold in enumerate(thresholds):
        if score >= threshold:
            level = idx
    return min(level, len(thresholds) - 1)


async def refresh_user_level(session, user: "User") -> None:
    user.level = calculate_level(user.reputation_score)
    await session.flush()


async def apply_reputation_event(
    session,
    user: "User",
    *,
    accepted: bool = False,
    rejected: bool = False,
    quality_bonus: bool = False,
    late: bool = False,
) -> None:
    delta = 0.0
    if accepted:
        delta += 2.0
    if rejected:
        delta -= 1.0
    if quality_bonus:
        delta += 0.5
    if late:
        delta -= 0.2

    user.reputation_score = max(-100.0, user.reputation_score + delta)
    await refresh_user_level(session, user)
