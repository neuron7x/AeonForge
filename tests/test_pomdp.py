import itertools

import pytest

from core.pomdp_planner import (
    Action,
    BeliefState,
    Observation,
    Particle,
    POMDPPlanner,
    State,
    TaskType,
)

def test_planner_pipeline():
    planner = POMDPPlanner(num_particles=200, num_simulations=100, max_depth=5)
    belief = planner.initialize_belief(initial_eoi=0.9, initial_fatigue=0.3, task_complexity=0.6)
    action = planner.plan(belief, TaskType.CODE_GENERATION)
    assert 0.0 <= action.ai_autonomy <= 1.0
    # update with a plausible observation
    obs = Observation(hrv_measurement=70.0, self_reported_load="medium", task_completion_time=1.0)
    new_belief = planner.update_belief(belief, action, obs)
    assert new_belief.mean_eoi() >= 0.0


def test_update_belief_triggers_resample_on_low_ess(monkeypatch: pytest.MonkeyPatch) -> None:
    planner = POMDPPlanner(num_particles=4, num_simulations=0, max_depth=1)

    base_state = State(eoi=0.5, fatigue=0.2, task_complexity=0.3)
    particles = [Particle(base_state, weight=1.0) for _ in range(4)]
    belief = BeliefState(particles)

    action = Action(TaskType.CODE_GENERATION, ai_autonomy=0.5)
    observation = Observation(hrv_measurement=70.0, self_reported_load="medium", task_completion_time=1.0)

    monkeypatch.setattr(planner.transition_model, "sample", lambda state, act: state)

    likelihoods = itertools.cycle([1.0, 1e-6, 1e-6, 1e-6])

    def fake_probability(obs: Observation, state: State, act: Action) -> float:
        return next(likelihoods)

    monkeypatch.setattr(planner.observation_model, "probability", fake_probability)

    resample_called = {"count": 0}

    original_resample = planner._resample

    def wrapped_resample(parts):
        resample_called["count"] += 1
        return original_resample(parts)

    monkeypatch.setattr(planner, "_resample", wrapped_resample)

    planner.update_belief(belief, action, observation)

    assert resample_called["count"] == 1
