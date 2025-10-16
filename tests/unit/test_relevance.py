import math
from aeonforge import RelevanceFilter

def test_relevance_mask_normalizes_and_matches_length():
    r = RelevanceFilter()
    feats = [0.1, 2.5, -3.0, 0.0]
    w = r.mask(feats)
    assert isinstance(w, list)
    assert len(w) == len(feats)
    assert math.isclose(sum(w), 1.0, rel_tol=1e-9, abs_tol=1e-9)

def test_relevance_mask_empty_ok():
    r = RelevanceFilter()
    w = r.mask([])
    # Contract: empty in -> empty out (no crash)
    assert w == []
