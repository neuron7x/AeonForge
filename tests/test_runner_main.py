import sys

import numpy as np
import pytest

from agi_core.engine import runner
from agi_core.devloop import replay_safe


def _fake_sample(self, batch):
    calls = getattr(self, "_fake_calls", 0)
    self._fake_calls = calls + 1
    n = 32 if calls == 0 else 64
    xs = np.zeros((n, self.S))
    aa = np.zeros((n, self.A))
    xp = np.zeros((n, self.S))
    rewards = np.zeros(n)
    return xs, aa, xp, rewards


def test_runner_main_linear(tmp_path, monkeypatch):
    log_path = tmp_path / "run_linear.jsonl"
    original_rollout = runner.LinearDynamicsSCM.rollout
    call_count = {"n": 0}

    def patched_rollout(self, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("boom")
        return original_rollout(self, *args, **kwargs)

    monkeypatch.setattr(sys, "argv", [
        "runner",
        "--env",
        "tanh",
        "--iters",
        "2",
        "--horizon",
        "2",
        "--state-dim",
        "3",
        "--action-dim",
        "2",
        "--log-jsonl",
        str(log_path),
    ])
    monkeypatch.setattr(replay_safe.SafeReplayBuffer, "sample", _fake_sample)
    monkeypatch.setattr(runner.LinearDynamicsSCM, "rollout", patched_rollout)
    runner.main()
    data = log_path.read_text().strip().splitlines()
    assert len(data) == 2


def test_runner_main_neural(tmp_path, monkeypatch):
    log_path = tmp_path / "run_neural.jsonl"
    original_fit = runner.NeuralDynamicsSCM.fit

    def patched_fit(self, X, A, U, epochs=60, batch=128, patience=6):
        return original_fit(self, X, A, U, epochs=2, batch=16, patience=1)

    monkeypatch.setattr(sys, "argv", [
        "runner",
        "--env",
        "lqr",
        "--iters",
        "2",
        "--horizon",
        "2",
        "--state-dim",
        "4",
        "--action-dim",
        "2",
        "--model",
        "neural",
        "--meta",
        "bandit",
        "--log-jsonl",
        str(log_path),
    ])
    monkeypatch.setattr(replay_safe.SafeReplayBuffer, "sample", _fake_sample)
    monkeypatch.setattr(runner.NeuralDynamicsSCM, "fit", patched_fit)
    runner.main()
    assert log_path.exists()
    assert log_path.read_text().strip()


def test_runner_env_helper_branches():
    env_tanh = runner._env("tanh", 3, 2, seed=0)
    env_lqr = runner._env("lqr", 3, 2, seed=0)
    env_reacher = runner._env("reacher", 0, 0, seed=0)
    env_minigrid = runner._env("minigrid", 0, 0, seed=0)
    assert env_tanh.S == 3 and env_tanh.A == 2
    assert env_lqr.S == 3 and env_lqr.A == 2
    assert hasattr(env_reacher, "rollout")
    assert hasattr(env_minigrid, "step")
    with pytest.raises(ValueError):
        runner._env("unknown", 0, 0, seed=0)


def test_runner_greedy_policy_explores_and_exploits():
    aff = runner.LinearAffordanceMap(4, 3, 1e-2)
    x = np.zeros(4)
    aff.W.fill(-10.0)
    greedy_policy = runner.greedy(aff, 3, runner.RNGManager(0).get("policy"))
    act = greedy_policy(x, 0)
    assert act.shape == (3,)
    aff.W[:4] = 10.0
    act2 = greedy_policy(np.ones(4), 1)
    assert np.argmax(act2) == 0
