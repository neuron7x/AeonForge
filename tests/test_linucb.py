import numpy as np
from agi_core.meta.meta_linucb_np import LinUCBMeta

def test_linucb():
    m=LinUCBMeta(arms=2,d=8,alpha=0.8,mode='diag',seed=0)
    x=np.ones(8); k=m.select(x); m.update(k,1.0,x); assert k in (0,1)
