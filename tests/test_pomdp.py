from core.pomdp_planner import POMDPPlanner, TaskType, Observation

def test_planner_pipeline():
    planner = POMDPPlanner(num_particles=200, num_simulations=100, max_depth=5)
    belief = planner.initialize_belief(initial_eoi=0.9, initial_fatigue=0.3, task_complexity=0.6)
    action = planner.plan(belief, TaskType.CODE_GENERATION)
    assert 0.0 <= action.ai_autonomy <= 1.0
    # update with a plausible observation
    obs = Observation(hrv_measurement=70.0, self_reported_load="medium", task_completion_time=1.0)
    new_belief = planner.update_belief(belief, action, obs)
    assert new_belief.mean_eoi() >= 0.0
