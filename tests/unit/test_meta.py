from aeonforge import MetaAttentionController

def test_meta_should_stop_threshold_logic():
    m = MetaAttentionController(threshold=0.1)  # stop if score >= 0.9
    assert m.should_stop(0.91) is True
    assert m.should_stop(0.9) is True
    assert m.should_stop(0.89) is False
