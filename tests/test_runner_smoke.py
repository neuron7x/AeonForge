import importlib

def test_runner_imports():
    # Ensure runner modules can import
    for name in ['agi_core.engine.runner', 'agi_core.engine.runner_mycelial']:
        importlib.import_module(name)
