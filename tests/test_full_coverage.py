import json
import sys
from pathlib import Path

import numpy as np
import pytest

from agi_core.affordance.affordance_np import LinearAffordanceMap
from agi_core.devloop.replay_safe import SafeReplayBuffer
from agi_core.engine import runner_mycelial
from agi_core.integration.mycelium_np import MycelialIntegration
from agi_core.integration.signal_types import ProcessResult, Signal
from agi_core.meta.meta_bandit_np import MetaBanditController
from agi_core.meta.meta_linucb_np import LinUCBMeta
from agi_core.relevance.relevance_np import RelevanceFilter
from agi_core.utils.jsonl import JSONLLogger
from agi_core.utils.metrics import rollout_real_gap, set_global_seed
from agi_core.utils.random import RNGManager
from agi_core.utils.tracking import WBConfig, WBTracker
from agi_core.viz.gif import build_gif
from agi_core.world.linear_scm_np import LinearDynamicsSCM
from agi_core.world.neural_scm_np import NeuralDynamicsSCM


def test_linear_affordance_paths():
    aff = LinearAffordanceMap(4, 3, 1e-2)
    X = np.random.randn(50, 4)
    acts = np.random.randint(0, 3, size=50)
    succ = (np.random.rand(50) > 0.7).astype(float)
    assert not aff.conditional_fit(X, acts, succ, min_samples=200)
    X_big = np.random.randn(256, 4)
    acts_big = np.random.randint(0, 3, size=256)
    succ_big = np.ones(256)
    assert aff.conditional_fit(X_big, acts_big, succ_big, min_samples=32)
    scores = aff.score_single(np.ones(4))
    feas = aff.feasible_actions(np.ones(4), thr=0.0)
    assert scores.shape == (3,) and set(feas) == {0, 1, 2}


class _StubModule:
    def __init__(self, fail=False, result=None):
        self.fail = fail
        self.result = result
        self.calls = 0

    def process(self, signal, strength, path):
        self.calls += 1
        if self.fail:
            raise ValueError("boom")
        if self.result is None:
            return ProcessResult(False, None, 0.0)
        return self.result


def test_mycelial_integration_paths():
    sugg = ProcessResult(True, np.array([1.0, 0.0]), 0.9)
    modules = {
        "aff": _StubModule(result=sugg),
        "wm": _StubModule(),
        "bad": _StubModule(fail=True),
    }
    net = MycelialIntegration(modules, quarantine_failures=1)
    sig = Signal(type="need_action", state=np.zeros(2))
    out = net.propagate("aff", sig, max_hops=2)
    assert np.allclose(out, np.array([1.0, 0.0]))
    net.propagate("bad", sig)
    assert net.quarantine["bad"]
    net.passive_decay()
    text = net.ascii_timeline(3)
    assert "edges" in text
    net.edges["aff"]["aff"] = 0.5
    loop_sig = Signal(type="need_action", state=np.ones(2))
    net.propagate("aff", loop_sig, max_hops=4)
    net.edges["aff"]["wm"] = 1e-6
    net.propagate("aff", loop_sig, max_hops=2)
    failing = MycelialIntegration({"solo": _StubModule()}, quarantine_failures=1)
    assert failing.propagate("solo", Signal(type="noop", state=None)) is None
    modules_loop = {
        "root": _StubModule(),
        "s1": _StubModule(),
        "s2": _StubModule(),
        "a": _StubModule(),
        "b": _StubModule(),
        "end": _StubModule(result=ProcessResult(True, np.array([0.0]), 0.1)),
    }
    net_loop = MycelialIntegration(modules_loop)
    for key in net_loop.edges:
        net_loop.edges[key] = {}
    net_loop.edges["root"] = {"s1": 1.0, "s2": 1.0}
    net_loop.edges["s1"] = {"a": 1.0}
    net_loop.edges["s2"] = {"a": 1.0}
    net_loop.edges["a"] = {"b": 1.0}
    net_loop.edges["b"] = {"end": 1.0}
    net_loop.edges["end"] = {}
    net_loop.propagate("root", Signal(type="loop", state=np.ones(1)), max_hops=5)


def test_safe_replay_buffer_modes():
    rng = RNGManager(0).get("replay")
    buf = SafeReplayBuffer(10, 3, 2, rng, protect_latest=1)
    empty = buf.sample(4)
    assert all(x.size == 0 for x in empty)
    x = np.zeros(3)
    a = np.zeros(2)
    for i in range(6):
        buf.add(x + i, a + i, x + i + 1, float(i), td_error=float(i))
    xs, aa, xp, rr = buf.sample(4)
    assert xs.shape[1:] == (3,)
    buf_res = SafeReplayBuffer(10, 3, 2, RNGManager(1).get("replay"), eviction="reservoir")
    for i in range(3):
        buf_res.add(x + i, a + i, x + i + 1, float(i))
    assert buf_res.sample(2)[0].shape[0] <= 3
    buf_fifo = SafeReplayBuffer(5, 2, 1, RNGManager(2).get("replay"), eviction="fifo")
    for i in range(4):
        buf_fifo.add(np.ones(2) * i, np.ones(1) * i, np.ones(2) * i, float(i))
    take = buf_fifo.sample(3)[0]
    assert take.shape[0] == 3


class _FakeWandb:
    def __init__(self):
        self.logged = []

    def init(self, **kwargs):
        self.kwargs = kwargs
        return self

    def log(self, data):
        self.logged.append(data)

    def finish(self):
        self.logged.append("finished")


def test_wbtracker_handles_availability(monkeypatch):
    cfg = WBConfig(enabled=False)
    tracker = WBTracker(cfg)
    with tracker as t:
        t.log({"a": 1})
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    cfg = WBConfig(enabled=True, project="p", entity="e", run_name="r")
    tracker = WBTracker(cfg)
    with tracker as t:
        t.log({"a": 2}, step=3)
    assert fake.logged[-1] == "finished"
    class _FailingRun:
        def log(self, data):
            raise RuntimeError("log fail")

        def finish(self):
            raise RuntimeError("finish fail")

    class _FailingWandb:
        def init(self, **kwargs):
            return _FailingRun()

    monkeypatch.setitem(sys.modules, "wandb", _FailingWandb())
    tracker = WBTracker(WBConfig(enabled=True))
    with tracker as t:
        t.log({"oops": 1})
    class _RaisingWandb:
        def init(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "wandb", _RaisingWandb())
    tracker = WBTracker(WBConfig(enabled=True))
    with tracker as t:
        t.log({})


def test_jsonl_logger_and_metrics(tmp_path):
    path = tmp_path / "logs" / "file.jsonl"
    with JSONLLogger(str(path)) as log:
        log.log({"value": 1})
    content = path.read_text().strip()
    assert json.loads(content)["value"] == 1
    set_global_seed(123)
    gap = rollout_real_gap(np.zeros((3, 2)), np.ones((3, 2)))
    assert pytest.approx(gap, rel=1e-6) == 1.0


def test_build_gif_branches(tmp_path, monkeypatch):
    png1 = tmp_path / "a.png"
    png2 = tmp_path / "b.png"
    png1.write_bytes(b"\x89PNG\r\n\x1a\n")
    png2.write_bytes(b"\x89PNG\r\n\x1a\n")

    class _FakeImageIO:
        class v2:
            @staticmethod
            def imread(path):
                return np.zeros((2, 2, 3), dtype=np.uint8)

        @staticmethod
        def mimsave(path, imgs, duration):
            Path(path).write_bytes(b"GIF")

    monkeypatch.setitem(sys.modules, "imageio", _FakeImageIO)
    out_path = tmp_path / "out.gif"
    assert build_gif([png1, png2], out_path, fps=10)
    assert out_path.read_bytes() == b"GIF"
    monkeypatch.delitem(sys.modules, "imageio")
    assert not build_gif([png1], out_path)


def test_linear_and_neural_scm_behaviour():
    scm = LinearDynamicsSCM(2, 1, 1e-3, seed=0)
    X = np.random.randn(128, 2)
    U = np.random.randn(128, 1)
    Y = X + U @ np.array([[0.5, -0.2]])
    loss = scm.fit(X, U, Y)
    assert loss >= 0
    pred = scm.predict_next(X[0], U[0], noise=False)
    assert pred.shape == (2,)
    noisy = scm.predict_next(X[0], U[0], noise=True)
    assert noisy.shape == (2,)
    pol = lambda x, t: np.zeros(1)
    Xm, Am = scm.rollout(np.zeros(2), pol, horizon=2)
    assert Xm.shape[0] == 3 and Am.shape[0] == 2
    Xm_plan, _ = scm.rollout(np.zeros(2), pol, horizon=1, do_plan=[{0: 1.0}])
    assert Xm_plan.shape == (2, 2)

    nn = NeuralDynamicsSCM(2, 1, hidden=4, lr=1e-2, seed=1)
    Xs = np.random.randn(20, 2)
    As = np.random.randn(20, 1)
    Ys = Xs + 0.1 * As
    best = nn.fit(Xs, As, Ys, epochs=3, batch=5, patience=1)
    assert isinstance(best, float)
    y_pred = nn.predict_next(Xs[0], As[0], noise=False)
    assert y_pred.shape == (2,)
    roll = nn.rollout(np.zeros(2), lambda x, t: np.zeros(1), horizon=2)
    assert roll[0].shape[0] == 3
    nn_static = NeuralDynamicsSCM(2, 1, hidden=3, lr=1e-2, seed=2)
    flat = lambda Z: (np.zeros((Z.shape[0], nn_static.H)), np.zeros((Z.shape[0], nn_static.S)))
    nn_static._forward = flat  # type: ignore[method-assign]
    nn_static.fit(Xs, As, np.zeros_like(Ys), epochs=4, batch=4, patience=2)
    nn_empty = NeuralDynamicsSCM(2, 1, hidden=2, lr=1e-2, seed=3)
    nn_empty.fit(Xs[:2], As[:2], Ys[:2], epochs=0)


def test_linucb_and_meta_bandit_behaviour():
    meta = MetaBanditController(epsilon=1.0, seed=0, arms=2)
    arm = meta.select()
    meta.update(arm, reward=1.0)
    ctx = np.ones(3)
    lin_diag = LinUCBMeta(arms=2, d=3, mode="diag", seed=0)
    arm_diag = lin_diag.select(ctx)
    lin_diag.update(arm_diag, 0.5, ctx)
    lin_full = LinUCBMeta(arms=2, d=3, mode="full", seed=0)
    arm_full = lin_full.select(ctx)
    lin_full.update(arm_full, 0.5, ctx)


def test_relevance_filter():
    filt = RelevanceFilter(5, 2)
    mask = filt.mask(np.zeros(5), np.array([0, 1, 2, 3, 4]), 0.0)
    assert mask.sum() == 2


def test_runner_mycelial_main(tmp_path, monkeypatch):
    out_dir = tmp_path / "viz"
    def run_once(iters):
        monkeypatch.setattr(sys, "argv", [
            "runner_mycelial",
            "--iters",
            str(iters),
            "--horizon",
            "2",
            "--viz-dir",
            str(out_dir),
            "--gif-fps",
            "5",
        ])
        original_propagate = runner_mycelial.MycelialIntegration.propagate
        state = {"aff": 0, "wm": 0}

        def patched_propagate(self, source, signal, max_hops=6):
            if source == "aff" and state["aff"] == 0:
                state["aff"] += 1
                return None
            if source == "wm" and state["wm"] == 0:
                state["wm"] += 1
                return None
            return original_propagate(self, source, signal, max_hops=max_hops)

        monkeypatch.setattr(runner_mycelial.MycelialIntegration, "propagate", patched_propagate)
        runner_mycelial.main()

    def _fake_build(paths, out, fps=6):
        Path(out).write_bytes(b"GIF")
        return True

    monkeypatch.setattr(runner_mycelial, "build_gif", _fake_build)
    run_once(1)
    gif_path = out_dir / "mycelium.gif"
    assert not gif_path.exists()
    run_once(5)
    assert gif_path.exists()
    timeline_files = sorted(out_dir.glob("timeline_*.txt"))
    assert timeline_files

