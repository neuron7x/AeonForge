"""Information-theoretic relevance filtering utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set

import numpy as np
from numpy.typing import NDArray

State = NDArray[np.float64]


@dataclass
class SelectionExplanation:
    dimension: int
    importance: float


class InformationTheoreticFilter:
    """Greedy selector for informative state dimensions."""

    def __init__(self, state_dim: int, topk: int, *, method: str = "variance") -> None:
        if state_dim <= 0:  # pragma: no cover - defensive validation
            raise ValueError("state_dim must be positive")
        if topk <= 0 or topk > state_dim:  # pragma: no cover - defensive validation
            raise ValueError("topk must lie inside (0, state_dim]")
        if method not in {"variance", "entropy", "kl"}:  # pragma: no cover - defensive validation
            raise ValueError("method must be 'variance', 'entropy' or 'kl'")
        self.D = int(state_dim)
        self.K = int(topk)
        self.method = method

    # ------------------------------------------------------------------
    # Mutual information proxies
    # ------------------------------------------------------------------
    def _mi_variance(self, X_t: NDArray[np.float64], X_tp1: NDArray[np.float64], dims: Set[int]) -> float:  # pragma: no cover - optional helper
        delta = X_tp1 - X_t
        return float(sum(np.var(delta[:, d]) for d in dims))

    def _mi_entropy(self, X_t: NDArray[np.float64], X_tp1: NDArray[np.float64], dims: Set[int], bins: int = 16) -> float:  # pragma: no cover - optional helper
        if not dims:
            return 0.0
        delta = X_tp1[:, sorted(dims)] - X_t[:, sorted(dims)]
        hist, _ = np.histogram(delta.ravel(), bins=bins)
        probs = hist / np.sum(hist)
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs + 1e-10)))

    def _mi_kl(self, X_t: NDArray[np.float64], X_tp1: NDArray[np.float64], dims: Set[int], bins: int = 16) -> float:  # pragma: no cover - optional helper
        if not dims:
            return 0.0
        delta = X_tp1[:, sorted(dims)] - X_t[:, sorted(dims)]
        hist, _ = np.histogram(delta.ravel(), bins=bins)
        probs = hist / np.sum(hist)
        probs = np.clip(probs, 1e-10, 1.0)
        uniform = np.ones_like(probs) / probs.size
        return float(np.sum(probs * np.log(probs / uniform)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def mask(self, x_t: State, x_tp1: State, reward: float = 0.0) -> NDArray[np.bool_]:
        if x_t.shape != x_tp1.shape:  # pragma: no cover - defensive validation
            raise ValueError("state vectors must match")
        if x_t.ndim != 1:  # pragma: no cover - defensive validation
            raise ValueError("mask expects 1-D inputs")
        delta = np.abs(x_tp1 - x_t)
        idx = np.argsort(-delta)[: self.K]
        mask = np.zeros(self.D, dtype=bool)
        mask[idx] = True
        return mask

    def mask_batch(self, X_t: State, X_tp1: State, rewards: Optional[NDArray[np.float64]] = None) -> NDArray[np.bool_]:  # pragma: no cover - batch helper
        if X_t.shape != X_tp1.shape:  # pragma: no cover - defensive validation
            raise ValueError("batch shapes must match")
        if X_t.ndim != 2:  # pragma: no cover - defensive validation
            raise ValueError("mask_batch expects 2-D arrays")
        selected: Set[int] = set()
        for _ in range(self.K):
            best_dim = -1
            best_score = -float("inf")
            for dim in range(self.D):
                if dim in selected:
                    continue
                candidate = set(selected)
                candidate.add(dim)
                if self.method == "variance":
                    score = self._mi_variance(X_t, X_tp1, candidate)
                elif self.method == "entropy":
                    score = self._mi_entropy(X_t, X_tp1, candidate)
                else:
                    score = self._mi_kl(X_t, X_tp1, candidate)
                if score > best_score:
                    best_score = score
                    best_dim = dim
            if best_dim >= 0:
                selected.add(best_dim)
        mask = np.zeros(self.D, dtype=bool)
        for dim in selected:
            mask[dim] = True
        return mask

    def explain_selection(self, X_t: State, X_tp1: State, mask: NDArray[np.bool_]) -> List[SelectionExplanation]:  # pragma: no cover - explanation helper
        if mask.shape[0] != self.D:  # pragma: no cover - defensive validation
            raise ValueError("mask has wrong dimensionality")
        dims = np.where(mask)[0]
        scores: List[SelectionExplanation] = []
        for dim in dims:
            if self.method == "variance":
                delta = X_tp1[:, dim] - X_t[:, dim]
                importance = float(np.var(delta))
            elif self.method == "entropy":
                delta = X_tp1[:, dim] - X_t[:, dim]
                hist, _ = np.histogram(delta, bins=16)
                probs = hist / np.sum(hist)
                probs = probs[probs > 0]
                importance = float(-np.sum(probs * np.log(probs + 1e-10)))
            else:
                delta = X_tp1[:, dim] - X_t[:, dim]
                hist, _ = np.histogram(delta, bins=16)
                probs = hist / np.sum(hist)
                probs = np.clip(probs, 1e-10, 1.0)
                uniform = np.ones_like(probs) / probs.size
                importance = float(np.sum(probs * np.log(probs / uniform)))
            scores.append(SelectionExplanation(dimension=int(dim), importance=importance))
        scores.sort(key=lambda item: item.importance, reverse=True)
        return scores


class RelevanceFilter(InformationTheoreticFilter):
    """Backward compatible alias for the historic RelevanceFilter class."""

    def __init__(self, state_dim: int, topk: int, method: str = "variance") -> None:
        super().__init__(state_dim=state_dim, topk=topk, method=method)


__all__ = ["InformationTheoreticFilter", "RelevanceFilter", "SelectionExplanation"]
