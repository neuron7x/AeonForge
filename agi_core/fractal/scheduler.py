"""Energy-aware schedulers for fractal control graphs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

import numpy as np


@dataclass(slots=True)
class EnergyAdaptiveScheduler:
    """Allocate compute budgets across a fractal subtree.

    The scheduler maintains an exponential moving average of historical load and
    adapts the per-child allocations to favour subtrees that consistently return
    confident results.  The design keeps the interface intentionally small so it
    can be embedded inside vectorised workloads without introducing a heavy
    dependency graph.
    """

    base_budget: float = 1.0
    min_budget: float = 0.05
    max_budget: float = 1.5
    adaptation: float = 0.2
    decay: float = 0.95
    history: List[float] = field(default_factory=list)
    last_budget: float = 1.0

    def update(self, utilisation: float) -> None:
        utilisation = float(np.clip(utilisation, 0.0, 10.0))
        if self.history:
            ema = self.decay * self.history[-1] + (1.0 - self.decay) * utilisation
        else:
            ema = utilisation
        self.history.append(ema)
        budget = self.last_budget + self.adaptation * (utilisation - 1.0)
        self.last_budget = float(np.clip(budget, self.min_budget, self.max_budget))

    def allocate(self, weights: Iterable[float]) -> np.ndarray:
        weights_arr = np.asarray(list(weights), dtype=float)
        if weights_arr.size == 0:
            return np.zeros(0, dtype=float)
        weights_arr = np.maximum(weights_arr, 1e-9)
        weights_arr /= weights_arr.sum()
        allocation = self.last_budget * weights_arr
        allocation = np.clip(allocation, self.min_budget, self.max_budget)
        return allocation

    def reset(self) -> None:
        self.history.clear()
        self.last_budget = float(self.base_budget)
