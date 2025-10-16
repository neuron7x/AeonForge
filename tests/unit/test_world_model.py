from aeonforge import CausalWorldModel

def test_worldmodel_do_returns_counterfactual_dict():
    wm = CausalWorldModel()
    out = wm.do("action", "adjust")
    assert isinstance(out, dict)
    assert out.get("action") == "adjust"
    assert out.get("effect") == "counterfactual-updated"
