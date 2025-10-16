import pytest
from aeonforge import AffordanceMap

@pytest.mark.perf
def test_affordance_infer_perf(benchmark):
    m = AffordanceMap(action_dim=32)
    def run():
        for _ in range(1000):
            m.infer([0.1, 0.2, 0.3])
    benchmark(run)
