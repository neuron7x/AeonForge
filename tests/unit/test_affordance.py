import math
import pytest
from aeonforge import AffordanceMap

@pytest.mark.parametrize("action_dim", [1, 2, 3, 8, 16])
def test_affordance_uniform_probs_shape_sum(action_dim):
    m = AffordanceMap(action_dim=action_dim)
    out = m.infer([0.2, 0.5, 0.3])
    assert isinstance(out, list)
    assert len(out) == action_dim
    # probabilities uniform and sum ~= 1
    assert all(abs(v - out[0]) < 1e-12 for v in out)
    assert math.isclose(sum(out), 1.0, rel_tol=1e-12, abs_tol=1e-12)
