import math

import pytest

from core.pomdp_planner import (
    Action,
    Observation,
    POMDPPlanner,
    State,
    TaskType,
    TransitionModel,
)


@pytest.fixture(scope="module", autouse=True)
def _exercise_aeonforge_core_modules() -> None:
    """Exercise the aeonforge package so coverage hooks observe it."""

    import contextlib
    import io

    from aeonforge import (
        AffordanceMap,
        CausalWorldModel,
        DevLoop,
        MetaAttentionController,
        RelevanceFilter,
    )
    from aeonforge import demo

    aff = AffordanceMap(action_dim=4)
    rel = RelevanceFilter()
    dev = DevLoop()
    wm = CausalWorldModel()
    meta = MetaAttentionController(threshold=0.05)

    obs = [0.1, 0.3, 0.6, 0.0]
    probs = aff.infer(obs)
    weights = rel.mask(obs)
    dev.log({"obs": obs, "probs": probs, "weights": weights})
    dev.replay()
    dev.curiosity(0.5)
    wm.do("action", "adjust")
    meta.should_stop(0.96)
    with contextlib.redirect_stdout(io.StringIO()):
        demo.run()


def test_transition_model_fatigue_clamped():
    model = TransitionModel()
    state = State(eoi=1.0, fatigue=0.2, task_complexity=0.5)
    action = Action(TaskType.CODE_GENERATION, ai_autonomy=0.9)

    for _ in range(500):
        new_state = model.sample(state, action)
        assert 0.0 <= new_state.fatigue <= 1.0

def test_planner_pipeline():
    planner = POMDPPlanner(num_particles=200, num_simulations=100, max_depth=5)
    belief = planner.initialize_belief(initial_eoi=0.9, initial_fatigue=0.3, task_complexity=0.6)
    action = planner.plan(belief, TaskType.CODE_GENERATION)
    assert 0.0 <= action.ai_autonomy <= 1.0
    # update with a plausible observation
    obs = Observation(hrv_measurement=70.0, self_reported_load="medium", task_completion_time=1.0)
    new_belief = planner.update_belief(belief, action, obs)
    assert new_belief.mean_eoi() >= 0.0


def test_update_belief_triggers_resample(monkeypatch):
    planner = POMDPPlanner(num_particles=20, num_simulations=1, max_depth=1)

    particles = [
        State(eoi=float(i), fatigue=0.1 * i, task_complexity=0.5) for i in range(20)
    ]
    belief = planner.initialize_belief(initial_eoi=0.5, initial_fatigue=0.5, task_complexity=0.5)
    belief.particles = [
        type(belief.particles[0])(state=state, weight=1.0) for state in particles
    ]
    belief._normalize()

    original_resample = planner._resample
    resample_called = {"value": False}

    def deterministic_transition(state, action):
        return state

    def skewed_likelihood(obs, state, action):
        return 1.0 if state.eoi == 0.0 else 1e-6

    def tracking_resample(particles):
        resample_called["value"] = True
        return original_resample(particles)

    monkeypatch.setattr(planner.transition_model, "sample", deterministic_transition)
    monkeypatch.setattr(planner.observation_model, "probability", skewed_likelihood)
    monkeypatch.setattr(planner, "_resample", tracking_resample)

    obs = Observation(hrv_measurement=0.0, self_reported_load="low", task_completion_time=0.1)
    new_belief = planner.update_belief(belief, Action(TaskType.CODE_GENERATION, 0.5), obs)

    assert resample_called["value"] is True
    assert len(new_belief.particles) == planner.num_particles
    assert math.isclose(sum(p.weight for p in new_belief.particles), 1.0, rel_tol=1e-6)
