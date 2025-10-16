import math
from hypothesis import given, strategies as st
from aeonforge import AffordanceMap, RelevanceFilter, DevLoop

@given(st.integers(min_value=1, max_value=128))
def test_affordance_probabilities_sum_to_one(action_dim):
    m = AffordanceMap(action_dim=action_dim)
    out = m.infer([0.0])
    assert len(out) == action_dim
    assert math.isclose(sum(out), 1.0, rel_tol=1e-9, abs_tol=1e-9)

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), max_size=256))
def test_relevance_mask_length_and_sum(lst):
    r = RelevanceFilter()
    w = r.mask(lst)
    assert len(w) == len(lst)
    if len(lst) == 0:
        assert w == []
    else:
        assert math.isclose(sum(w), 1.0, rel_tol=1e-7, abs_tol=1e-7)

@given(st.floats(width=16))
def test_curiosity_in_0_1(x):
    d = DevLoop()
    c = d.curiosity(x)
    assert 0.0 <= c <= 1.0
