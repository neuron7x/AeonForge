import numpy as np
from agi_core.integration.mycelium_np import MycelialIntegration
from agi_core.integration.signal_types import Signal, ProcessResult

class Node:
    def process(self,sig,strength,path): return ProcessResult(True, np.array([1.0]), 0.9)

def test_mycelium():
    net=MycelialIntegration({'a':Node(),'b':Node(),'c':Node()})
    out=net.propagate('a', Signal(type='x'))
    assert out is not None
