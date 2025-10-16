from aeonforge import DevLoop

def test_devloop_log_replay_window5():
    d = DevLoop()
    for i in range(12):
        d.log(i)
    rp = d.replay()
    assert len(rp) == 5
    assert rp == [7,8,9,10,11]

def test_curiosity_bounds_and_monotonicity():
    d = DevLoop()
    for e in [-10, -1, 0, 0.1, 0.5, 1.0, 10]:
        c = d.curiosity(e)
        assert 0.0 <= c <= 1.0
    # Larger absolute error -> not decreasing curiosity
    assert d.curiosity(0.1) <= d.curiosity(0.5) <= d.curiosity(1.0)
