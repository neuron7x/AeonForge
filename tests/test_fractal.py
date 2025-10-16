from typing import Dict

import numpy as np
import pytest

from agi_core.fractal import (
    EnergyAdaptiveScheduler,
    FractalComposite,
    FractalContext,
    FractalCortex,
    FractalLeaf,
    FractalNode,
    FractalSignal,
)
from agi_core.integration.fractal_bridge import FractalMyceliumBridge, FractalRoleSpec
from agi_core.integration.mycelium_np import MycelialIntegration
from agi_core.integration.signal_types import ProcessResult, Signal


class DummyModule:
    def __init__(self, value: float, success: bool = True):
        self.value = value
        self.success = success

    def process(self, signal: Signal, strength: float, path):
        if not self.success:
            return ProcessResult(False, None, confidence=0.0)
        vec = np.full(2, self.value)
        return ProcessResult(True, vec, confidence=0.5)


def test_fractal_cortex_prefers_child_suggestion():
    def local_fn(signal: Signal, strength: float, context):
        return ProcessResult(True, np.zeros(2), confidence=0.1)

    def leaf_fn(signal: Signal, strength: float):
        return ProcessResult(True, np.array([1.0, 0.0]), confidence=0.9)

    root = FractalComposite("root", local_fn)
    root.add_child(FractalLeaf("leaf", leaf_fn))
    cortex = FractalCortex(root)
    sig = Signal(type="test")
    result = cortex.propagate(sig)
    assert result is not None
    assert np.allclose(result.suggestion, np.array([1.0, 0.0]))


def test_fractal_cortex_respects_energy_budget():
    calls = {"n": 0}

    def local_fn(signal: Signal, strength: float, context):
        return ProcessResult(True, np.ones(2), confidence=0.5)

    def leaf_fn(signal: Signal, strength: float):
        calls["n"] += 1
        assert strength <= 1.5
        return ProcessResult(True, np.ones(2) * strength, confidence=0.4)

    scheduler = EnergyAdaptiveScheduler(base_budget=0.5, max_budget=1.5)
    root = FractalComposite("root", local_fn, scheduler=scheduler)
    root.add_child(FractalLeaf("leaf", leaf_fn))
    cortex = FractalCortex(root)
    sig = Signal(type="budget")
    for _ in range(3):
        cortex.propagate(sig, strength=1.0)
    assert calls["n"] == 3
    summary = cortex.summary()
    assert "invocations" in summary


def test_fractal_mycelium_bridge_fallback():
    def local_fail(signal: Signal, strength: float, context):
        return ProcessResult(False, None, confidence=0.0)

    root = FractalComposite("root", local_fail)
    cortex = FractalCortex(root)
    bridge = FractalMyceliumBridge(
        cortex,
        MycelialIntegration({"entry": DummyModule(0.5)}),
        entry_node="entry",
    )
    sig = Signal(type="need_action")
    result = bridge.route(sig)
    assert result is not None
    assert result.success
    assert np.allclose(result.suggestion, np.full(2, 0.5))


def test_fractal_scheduler_and_context_helpers():
    scheduler = EnergyAdaptiveScheduler(base_budget=0.3, min_budget=0.1, max_budget=0.5)
    empty_alloc = scheduler.allocate([])
    assert empty_alloc.size == 0
    scheduler.update(0.0)
    scheduler.reset()
    assert scheduler.history == [] and scheduler.last_budget == scheduler.base_budget


def test_fractal_node_duplicate_child_and_none_local():
    class NullNode(FractalComposite):
        def __init__(self):
            super().__init__("null", lambda *_: None)

    root = FractalComposite("root", lambda s, st, ctx: ProcessResult(True, np.ones(2), confidence=0.2))
    leaf = FractalLeaf("leaf", lambda s, st: ProcessResult(True, np.ones(2), confidence=0.1))
    root.add_child(leaf)
    with pytest.raises(ValueError):
        root.add_child(leaf)
    root.remove_child("leaf")
    assert "leaf" not in root.children

    null = NullNode()
    result = null.process(FractalSignal(signal=Signal(type="null")))
    assert result is None


def test_fractal_cortex_handles_none_and_reset():
    class NoneComposite(FractalComposite):
        def __init__(self):
            super().__init__("root", lambda *_: None)

    cortex = FractalCortex(NoneComposite())
    assert cortex.propagate(Signal(type="none")) is None
    cortex.reset_metrics()
    assert cortex.stats.invocations == 0


def test_bridge_prefers_cortex_and_summary():
    def local_success(signal: Signal, strength: float, context):
        if signal.type == "test":
            return ProcessResult(True, np.array([0.2, 0.8]), confidence=0.9)
        return ProcessResult(False, None, confidence=0.0)

    cortex = FractalCortex(FractalComposite("root", local_success))
    bridge = FractalMyceliumBridge(
        cortex,
        MycelialIntegration({"entry": DummyModule(0.1)}),
        entry_node="entry",
        on_failure=lambda sig: ProcessResult(False, None, confidence=0.0),
    )
    sig = Signal(type="test")
    res1 = bridge.route(sig)
    assert res1 is not None and res1.success

    # Force fallback to exercise summary + failure callback path
    bridge.mycelium.modules["entry"] = DummyModule(0.0, success=False)
    res2 = bridge.route(Signal(type="fallback"))
    assert res2 is not None
    summary = bridge.summary()
    assert summary["cortex_success"] >= 1
    assert summary["iterations"] >= 2


def test_bridge_returns_none_without_callback():
    cortex = FractalCortex(FractalComposite("root", lambda *_: None))
    bridge = FractalMyceliumBridge(
        cortex,
        MycelialIntegration({"entry": DummyModule(0.0, success=False)}),
        entry_node="entry",
        on_failure=None,
    )
    assert bridge.route(Signal(type="none")) is None


def test_bridge_builds_topology_and_records_memory():
    def coordinator(signal: Signal, strength: float, context: FractalContext):
        return ProcessResult(False, None, confidence=0.0)

    def specialist(signal: Signal, strength: float, context: FractalContext):
        return ProcessResult(True, np.array([0.9, 0.1]), confidence=0.8)

    topology = FractalRoleSpec(
        "coordinator",
        coordinator,
        children=[FractalRoleSpec("specialist", specialist)],
    )

    class Memory:
        def __init__(self):
            self.episodes = []

        def add_episode(self, name: str, content: str, timestamp):
            self.episodes.append((name, content, timestamp))

    memory = Memory()
    bridge = FractalMyceliumBridge(
        cortex=None,
        mycelium=MycelialIntegration({"entry": DummyModule(0.2)}),
        entry_node="entry",
        topology=topology,
        memory=memory,
    )

    result = bridge.route(Signal(type="delegated"))
    assert result is not None and result.success
    assert memory.episodes
    episode_name, content, _ = memory.episodes[-1]
    assert episode_name == "delegation_iteration"
    assert "delegated" in content


def test_fractal_composite_routing_paths():
    def local_fn(signal: Signal, strength: float, context: FractalContext):
        return ProcessResult(True, np.array([strength, strength]), confidence=0.2)

    def router(signal: Signal, children: Dict[str, FractalNode], context: FractalContext):
        if signal.type == "skip":
            return []
        return children.keys()

    root = FractalComposite("root", local_fn, routing_fn=router)
    child = FractalLeaf("leaf", lambda s, st: ProcessResult(True, np.array([0.0, 1.0]), confidence=0.1))
    root.add_child(child)
    # routing removes child so local path returned
    res_skip = root.process(FractalSignal(signal=Signal(type="skip")))
    assert res_skip is not None and np.allclose(res_skip.suggestion, np.array([1.0, 1.0]))
    # default branch includes child
    res_default = root.process(FractalSignal(signal=Signal(type="use")))
    assert res_default is not None
    # explicitly inspect composite helper branches
    ctx = FractalContext(path=("root",), budget=1.0, depth=0)
    assert root._select_children(FractalSignal(signal=Signal(type="skip")), ctx) == []
    root.remove_child("leaf")
    assert root._select_children(FractalSignal(signal=Signal(type="skip")), ctx) == ()


def test_fractal_node_not_implemented():
    class BareNode(FractalNode):
        pass

    node = BareNode("bare")
    with pytest.raises(NotImplementedError):
        node.process(FractalSignal(signal=Signal(type="bare")))


def test_fractal_leaf_helpers_and_none_child_branch():
    root = FractalComposite("root", lambda s, st, ctx: ProcessResult(True, np.ones(2), confidence=0.3))
    leaf = FractalLeaf("leaf", lambda s, st: None)
    root.add_child(leaf)
    res = root.process(FractalSignal(signal=Signal(type="none")))
    assert np.allclose(res.suggestion, np.ones(2))
    ctx = FractalContext(path=("leaf",), budget=1.0, depth=1)
    assert leaf._select_children(FractalSignal(signal=Signal(type="none")), ctx) == ()


def test_fractal_node_default_select_children():
    class SimpleNode(FractalNode):
        def _process_local(self, fsig: FractalSignal, context: FractalContext):
            return ProcessResult(True, np.zeros(1), confidence=0.2)

    node = SimpleNode("root")
    child = FractalLeaf("leaf", lambda s, st: ProcessResult(True, np.ones(1), confidence=0.1))
    node.add_child(child)
    ctx = FractalContext(path=("root",), budget=1.0, depth=0)
    selected = node._select_children(FractalSignal(signal=Signal(type="default")), ctx)
    assert selected == [child]
