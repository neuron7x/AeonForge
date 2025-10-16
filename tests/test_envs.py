import numpy as np

from agi_core.envs.tanh_env import TanhEnv
from agi_core.envs.lqr_env import LQREnv
from agi_core.envs.reacher2d import Reacher2D
from agi_core.envs.minigrid_lite import MiniGridLite


def test_tanh_env_step_and_rollout():
    env = TanhEnv(4, 2, seed=0)
    x0 = np.zeros(4)
    a0 = np.zeros(2)
    x1, r1 = env.step(x0, a0, do_set={1: 0.5})
    assert x1.shape == (4,) and isinstance(r1, float)
    policy = lambda x, t: np.zeros(2)
    states, actions, rewards = env.rollout(np.zeros(4), policy, horizon=3, do_plan=[{0: 0.1}] * 3)
    assert states.shape == (4, 4)
    assert actions.shape == (3, 2)
    assert rewards.shape == (3,)


def test_lqr_env_step_and_rollout():
    env = LQREnv(3, 2, seed=0)
    x0 = np.zeros(3)
    a0 = np.zeros(2)
    nxt, reward = env.step(x0, a0, do_set={2: 1.0})
    assert nxt.shape == (3,) and isinstance(reward, float)
    policy = lambda x, t: np.ones(2) * 0.1
    states, actions, rewards = env.rollout(np.zeros(3), policy, horizon=2, do_plan=[{1: 0.0}, {}])
    assert states.shape == (3, 3)
    assert actions.shape == (2, 2)
    assert rewards.shape == (2,)


def test_reacher_env_step_and_rollout():
    env = Reacher2D(step_lim=5, seed=0)
    obs = env.reset()
    a = np.array([0.2, -0.2])
    obs2, r = env.step(obs, a, do_set={0: 0.0, 1: 0.0})
    assert obs2.shape == (4,) and isinstance(r, float)
    policy = lambda x, t: np.array([(-1) ** t * 0.05, 0.05])
    states, actions, rewards = env.rollout(4, policy)
    assert states.shape == (5, 4)
    assert actions.shape == (4, 2)
    assert rewards.shape == (4,)


def test_minigrid_env_actions():
    env = MiniGridLite(size=5, step_limit=6, seed=0)
    x = env.reset()
    actions = [np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0]), np.array([0, 0, 1, 0]), np.array([0, 0, 0, 1])]
    for act in actions:
        x, _ = env.step(x, act)
    x, _ = env.step(x, np.array([1, 0, 0, 0]), do_set={0: 2, 1: 3})
    policy = lambda state, t: actions[t % len(actions)]
    states, actions, rewards = env.rollout(3, policy)
    assert states.shape == (4, 4)
    assert actions.shape == (3, 4)
    assert rewards.shape == (3,)
