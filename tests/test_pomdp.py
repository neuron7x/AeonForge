import numpy as np

import aeonforge  # noqa: F401  Ensures coverage includes main package

from core.pomdp_planner import (
    Action,
    Observation,
    POMDPPlanner,
    State,
    TaskType,
    TransitionModel,
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


def test_transition_model_fatigue_bounds():
    np.random.seed(0)
    transition_model = TransitionModel()
    state = State(eoi=0.8, fatigue=0.5, task_complexity=0.5)
    action = Action(task_type=TaskType.CODE_GENERATION, ai_autonomy=0.6)

    for _ in range(1000):
        state = transition_model.sample(state, action)
        assert 0.0 <= state.fatigue <= 1.0


def test_aeonforge_demo_run(capsys):
    aeonforge.demo.run()
    captured = capsys.readouterr()
    assert "Affordance:" in captured.out
    assert "Meta should_stop:" in captured.out
