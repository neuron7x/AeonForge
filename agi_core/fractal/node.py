"""Fractal node abstractions used by :mod:`agi_core` orchestrators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from agi_core.integration.signal_types import ProcessResult, Signal

from .scheduler import EnergyAdaptiveScheduler


@dataclass(slots=True)
class FractalSignal:
    """Wrapper for :class:`~agi_core.integration.signal_types.Signal` with metadata."""

    signal: Signal
    strength: float = 1.0
    depth: int = 0


@dataclass(slots=True)
class FractalContext:
    """Runtime context shared along a propagation path."""

    path: Tuple[str, ...]
    budget: float
    depth: int

    def descend(self, node: str, budget: float) -> "FractalContext":
        return FractalContext(path=self.path + (node,), budget=budget, depth=self.depth + 1)


class FractalNode:
    """Abstract base for recursive fractal nodes.

    Subclasses implement :meth:`_process_local` which performs the local
    transformation on the incoming :class:`FractalSignal`.  The base class handles
    routing to children, budget management and aggregation of the returned
    suggestions.
    """

    def __init__(
        self,
        name: str,
        level: int = 0,
        scheduler: Optional[EnergyAdaptiveScheduler] = None,
    ) -> None:
        self.name = name
        self.level = int(level)
        self.children: Dict[str, FractalNode] = {}
        self.scheduler = scheduler or EnergyAdaptiveScheduler()

    # Public API -----------------------------------------------------------
    def add_child(self, child: "FractalNode") -> None:
        if child.name in self.children:
            raise ValueError(f"child '{child.name}' already registered under '{self.name}'")
        self.children[child.name] = child

    def remove_child(self, name: str) -> None:
        self.children.pop(name)

    def process(self, fsig: FractalSignal, context: Optional[FractalContext] = None) -> Optional[ProcessResult]:
        context = context or FractalContext(path=(self.name,), budget=fsig.strength, depth=fsig.depth)
        local = self._process_local(fsig, context)
        if local is None:
            return None

        if not self.children:
            return local

        utilisation = 0.0
        weights: List[float] = []
        child_results: List[Tuple[float, ProcessResult]] = []
        selected = list(self._select_children(fsig, context))
        for child in selected:
            weights.append(self._child_weight(child, local))
        budgets = self.scheduler.allocate(weights)
        for (child, budget) in zip(selected, budgets):
            utilisation += float(budget)
            next_signal = FractalSignal(signal=fsig.signal, strength=float(budget), depth=context.depth + 1)
            result = child.process(next_signal, context.descend(child.name, budget))
            if result is not None:
                child_results.append((budget * max(1e-6, result.confidence), result))
        self.scheduler.update(utilisation)
        if not child_results:
            return local
        child_results.sort(key=lambda t: -t[0])
        top_conf, top_res = child_results[0]
        if top_conf > max(1e-9, local.confidence * context.budget):
            return top_res
        return local

    # Hooks ----------------------------------------------------------------
    def _process_local(self, fsig: FractalSignal, context: FractalContext) -> Optional[ProcessResult]:
        raise NotImplementedError

    def _child_weight(self, child: "FractalNode", local_result: ProcessResult) -> float:
        return max(0.1, float(local_result.confidence))

    def _select_children(self, fsig: FractalSignal, context: FractalContext) -> Sequence["FractalNode"]:
        return list(self.children.values())


class FractalLeaf(FractalNode):
    """Leaf node that delegates to a callable."""

    def __init__(self, name: str, fn: Callable[[Signal, float], ProcessResult]) -> None:
        super().__init__(name, level=0)
        self._fn = fn

    def _process_local(self, fsig: FractalSignal, context: FractalContext) -> Optional[ProcessResult]:
        return self._fn(fsig.signal, fsig.strength)

    def _select_children(self, fsig: FractalSignal, context: FractalContext) -> Sequence["FractalNode"]:
        return ()


class FractalComposite(FractalNode):
    """Composite node that aggregates local heuristics with its children."""

    def __init__(
        self,
        name: str,
        local_fn: Callable[[Signal, float, FractalContext], Optional[ProcessResult]],
        level: int = 0,
        scheduler: Optional[EnergyAdaptiveScheduler] = None,
        routing_fn: Optional[Callable[[Signal, Dict[str, "FractalNode"], FractalContext], Iterable[str]]] = None,
    ) -> None:
        super().__init__(name, level=level, scheduler=scheduler)
        self._local_fn = local_fn
        self._routing_fn = routing_fn

    def _process_local(self, fsig: FractalSignal, context: FractalContext) -> Optional[ProcessResult]:
        return self._local_fn(fsig.signal, fsig.strength, context)

    def _select_children(self, fsig: FractalSignal, context: FractalContext) -> Sequence[FractalNode]:
        if not self.children:
            return ()
        if self._routing_fn is None:
            return list(self.children.values())
        names = list(self._routing_fn(fsig.signal, self.children, context))
        return [self.children[name] for name in names if name in self.children]
