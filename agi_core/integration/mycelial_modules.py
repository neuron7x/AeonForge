"""Reusable Mycelial module implementations used across configs.

The production system ships lightweight stand-ins for heavyweight services
such as LLM coordination or analytic pipelines.  The goal is to provide
modules with deterministic behaviour mirroring the :class:`DummyModule`
semantics from the test-suite while exposing richer metadata suitable for
Fractal/Mycelium experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, MutableMapping, Sequence

import numpy as np

from agi_core.integration.signal_types import ProcessResult, Signal


@dataclass(slots=True)
class LLMAgentModule:
    """Heuristic LLM agent producing embeddings for downstream modules."""

    responses: Mapping[str, Sequence[float]]
    default: Sequence[float]
    confidence: float = 0.6

    _responses: MutableMapping[str, np.ndarray] = field(init=False, repr=False)
    _default: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._responses = {
            key: np.asarray(value, dtype=float) for key, value in self.responses.items()
        }
        self._default = np.asarray(self.default, dtype=float)

    def process(self, signal: Signal, strength: float, path: Sequence[str]) -> ProcessResult:
        vector = self._responses.get(signal.type, self._default)
        scaled = np.clip(vector * max(1.0, strength), 0.0, 1.0)
        return ProcessResult(True, scaled, confidence=self.confidence)


@dataclass(slots=True)
class AnalyticWorkerModule:
    """Scoring worker translating state vectors into actionable policies."""

    weights: Sequence[float]
    bias: float = 0.0
    threshold: float = 0.05

    _weights: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._weights = np.asarray(self.weights, dtype=float)

    def process(self, signal: Signal, strength: float, path: Sequence[str]) -> ProcessResult:
        if signal.state is None:
            return ProcessResult(False, None, confidence=0.0, error="state_missing")

        score = float(np.dot(signal.state, self._weights) + self.bias)
        success = score >= self.threshold
        suggestion = np.full(self._weights.shape, np.clip(score * strength, -1.0, 1.0))
        confidence = float(min(1.0, abs(score)))
        return ProcessResult(success, suggestion, confidence=confidence)


@dataclass(slots=True)
class DelegationCriticModule:
    """Critic that moderates delegation confidence using payload metadata."""

    gain: float = 0.5

    def process(self, signal: Signal, strength: float, path: Sequence[str]) -> ProcessResult:
        payload_score = float(signal.payload.get("score", 0.0)) if signal.payload else 0.0
        confidence = max(0.0, min(1.0, payload_score * self.gain))
        vector = np.array([confidence, 1.0 - confidence])
        success = confidence >= 0.2
        return ProcessResult(success, vector, confidence=confidence)


__all__ = [
    "LLMAgentModule",
    "AnalyticWorkerModule",
    "DelegationCriticModule",
]

