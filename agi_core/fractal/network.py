"""High-level orchestration for fractal execution graphs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from agi_core.integration.signal_types import ProcessResult, Signal

from .node import FractalNode, FractalSignal


@dataclass(slots=True)
class CortexStats:
    """Telemetry summarising how the cortex utilised its resources."""

    invocations: int = 0
    propagated: int = 0
    fallbacks: int = 0


class FractalCortex:
    """Entry point for fractal execution.

    The cortex owns a hierarchy of :class:`FractalNode` objects.  Each call to
    :meth:`propagate` injects a :class:`~agi_core.integration.signal_types.Signal`
    into the root node while tracking statistics that can be surfaced to
    higher-level monitors or WANDB trackers.
    """

    def __init__(self, root: FractalNode) -> None:
        self.root = root
        self.stats = CortexStats()

    def propagate(self, signal: Signal, strength: float = 1.0) -> Optional[ProcessResult]:
        self.stats.invocations += 1
        fsig = FractalSignal(signal=signal, strength=strength, depth=0)
        result = self.root.process(fsig)
        if result is None:
            self.stats.fallbacks += 1
            return None
        if result.success:
            self.stats.propagated += 1
        else:
            self.stats.fallbacks += 1
        return result

    def summary(self) -> Dict[str, float]:
        total = float(self.stats.invocations or 1)
        return {
            "invocations": float(self.stats.invocations),
            "propagate_rate": self.stats.propagated / total,
            "fallback_rate": self.stats.fallbacks / total,
        }

    def reset_metrics(self) -> None:
        self.stats = CortexStats()
