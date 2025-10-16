from pathlib import Path

import numpy as np

from agi_core.integration.mycelial_registry import load_mycelial_modules
from agi_core.integration.signal_types import Signal


def test_load_mycelial_modules_from_configs():
    modules = load_mycelial_modules(Path("configs/mycelium"))
    assert {"coordinator_llm", "delegation_critic", "market_signal_analyst"}.issubset(modules)

    llm = modules["coordinator_llm"]
    result_llm = llm.process(Signal(type="need_action"), strength=1.0, path=("root",))
    assert result_llm.success
    assert np.isclose(result_llm.suggestion.sum(), 1.0, atol=1e-6)

    analyst = modules["market_signal_analyst"]
    result_analyst = analyst.process(
        Signal(type="score", state=np.ones(6)), strength=0.5, path=("root",)
    )
    assert result_analyst.suggestion is not None
