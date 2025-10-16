def test_imports():
    import src.aeonforge as core
    assert hasattr(core,'AffordanceMap') and hasattr(core,'MetaAttentionController')
