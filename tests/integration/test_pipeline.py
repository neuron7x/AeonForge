from aeonforge import AffordanceMap, RelevanceFilter, DevLoop, CausalWorldModel, MetaAttentionController

def test_end_to_end_pipeline():
    aff = AffordanceMap(action_dim=4)
    rel = RelevanceFilter()
    dev = DevLoop()
    wm  = CausalWorldModel()
    meta = MetaAttentionController(threshold=0.2)

    obs = [0.2, 0.7, 0.1]
    actions = aff.infer(obs)
    weights = rel.mask(actions)
    dev.log({'obs': obs, 'actions': actions, 'weights': weights})
    cf = wm.do('action', 'adjust')
    # fake score from weights (sum==1) -> pick close to 1.0 for stop
    score = sum(weights)
    assert score == 1.0
    should_stop = meta.should_stop(score)
    # with threshold=0.2 -> stop if score >= 0.8; here score=1.0 -> True
    assert should_stop is True
    assert isinstance(cf, dict)
    assert len(dev.replay()) >= 1
