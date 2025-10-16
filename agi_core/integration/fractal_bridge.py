"""Bridge utilities between fractal coordinators and mycelial integration."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable, Dict, Iterable, Optional, Protocol, Sequence

import numpy as np
import pandas as pd

from agi_core.integration.mycelium_np import MycelialIntegration
from agi_core.integration.signal_types import ProcessResult, Signal

from agi_core.fractal import FractalComposite, FractalContext, FractalCortex, FractalNode


class MemorySinkProtocol(Protocol):
    """Minimal episodic memory contract used by the bridge."""

    def add_episode(self, name: str, content: str, timestamp: pd.Timestamp) -> None:
        """Persist an episodic log entry."""


@dataclass(frozen=True)
class FractalRoleSpec:
    """Declarative description of a fractal coordinator/subagent hierarchy."""

    name: str
    local_fn: Callable[[Signal, float, FractalContext], Optional[ProcessResult]]
    children: Sequence["FractalRoleSpec"] = ()
    routing_fn: Optional[Callable[[Signal, Dict[str, FractalNode], FractalContext], Iterable[str]]] = None

    def instantiate(self) -> FractalComposite:
        node = FractalComposite(self.name, self.local_fn, routing_fn=self.routing_fn)
        for child in self.children:
            node.add_child(child.instantiate())
        return node


@dataclass(slots=True)
class BridgeStats:
    """Lightweight telemetry for hybrid execution."""

    cortex_success: int = 0
    mycelium_fallbacks: int = 0
    last_error: Optional[str] = None


class FractalMyceliumBridge:
    """Hybrid execution pipeline combining fractal and mycelial substrates."""

    def __init__(
        self,
        cortex: Optional[FractalCortex],
        mycelium: MycelialIntegration,
        entry_node: str,
        on_failure: Optional[Callable[[Signal], Optional[ProcessResult]]] = None,
        topology: Optional[FractalRoleSpec] = None,
        memory: Optional[MemorySinkProtocol] = None,
    ) -> None:
        if cortex is None and topology is None:
            raise ValueError("Either a pre-built cortex or a topology specification must be provided")
        if cortex is not None and topology is not None:
            raise ValueError("Provide either 'cortex' or 'topology', not both")

        if topology is not None:
            cortex = FractalCortex(topology.instantiate())

        assert cortex is not None
        self.cortex = cortex
        self.mycelium = mycelium
        self.entry_node = entry_node
        self.on_failure = on_failure
        self.stats = BridgeStats()
        self._memory = memory
        self._iterations = 0

    def route(self, signal: Signal, strength: float = 1.0) -> Optional[ProcessResult]:
        self._iterations += 1
        result = self.cortex.propagate(signal, strength=strength)
        used_fallback = False
        if result is not None and result.success:
            self.stats.cortex_success += 1
            self._record_iteration(signal, strength, result, used_fallback)
            return result

        self.stats.mycelium_fallbacks += 1
        used_fallback = True
        try:
            suggestion = self.mycelium.propagate(self.entry_node, signal)
            if suggestion is None and self.on_failure is not None:
                fallback = self.on_failure(signal)
                self._record_iteration(signal, strength, fallback, used_fallback)
                return fallback
            if suggestion is None:
                self._record_iteration(signal, strength, None, used_fallback)
                return None
            result = ProcessResult(True, suggestion, confidence=1.0)
            self._record_iteration(signal, strength, result, used_fallback)
            return result
        except Exception as exc:  # pragma: no cover - defensive safeguard
            self.stats.last_error = str(exc)
            if self.on_failure is not None:
                fallback = self.on_failure(signal)
                self._record_iteration(signal, strength, fallback, used_fallback)
                return fallback
            self._record_iteration(signal, strength, None, used_fallback)
            return None

    def summary(self) -> Dict[str, float]:
        total = float(self.stats.cortex_success + self.stats.mycelium_fallbacks or 1)
        return {
            "cortex_success": float(self.stats.cortex_success),
            "mycelium_fallbacks": float(self.stats.mycelium_fallbacks),
            "cortex_success_rate": (self.stats.cortex_success / total),
            "mycelium_usage_rate": (self.stats.mycelium_fallbacks / total),
            "iterations": float(self._iterations),
            "last_error": self.stats.last_error or "",
            "mycelium_history": float(len(self.mycelium.history)),
        }

    def _record_iteration(
        self,
        signal: Signal,
        strength: float,
        result: Optional[ProcessResult],
        used_fallback: bool,
    ) -> None:
        if self._memory is None:
            return

        payload = {
            "signal": signal.type,
            "strength": float(strength),
            "used_fallback": used_fallback,
            "result": self._serialise_result(result),
            "summary": self.summary(),
        }
        content = json.dumps(payload, default=self._json_default, ensure_ascii=False)
        timestamp = pd.Timestamp.now(tz="UTC")
        self._memory.add_episode("delegation_iteration", content=content, timestamp=timestamp)

    @staticmethod
    def _serialise_result(result: Optional[ProcessResult]) -> Optional[Dict[str, object]]:
        if result is None:
            return None
        suggestion = None
        if result.suggestion is not None:
            suggestion = np.asarray(result.suggestion).tolist()
        return {
            "success": bool(result.success),
            "confidence": float(result.confidence),
            "suggestion": suggestion,
            "error": result.error,
        }

    @staticmethod
    def _json_default(value: object) -> object:
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)
