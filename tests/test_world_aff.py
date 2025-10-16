import numpy as np
from agi_core.world.linear_scm_np import LinearDynamicsSCM
from agi_core.affordance.affordance_np import LinearAffordanceMap

def test_scm():
    S, A = 4, 3
    scm = LinearDynamicsSCM(S, A, 1e-2, seed=0)
    X = np.random.randn(128, S)
    actions = np.random.randn(128, A)
    coupling = np.random.randn(A, S) * 0.1
    Y = X + actions @ coupling
    loss = scm.fit(X, actions, Y)
    y = scm.predict_next(X[0], actions[0], False)
    assert y.shape == (S,)

def test_aff():
    S,A=6,4; aff=LinearAffordanceMap(S,A,1e-2)
    X=np.random.randn(200,S); aidx=np.random.randint(0,A,size=200); succ=(np.random.rand(200)>0.4).astype(float)
    ok=aff.conditional_fit(X,aidx,succ,128,0.5); s=aff.score_single(X[0]); assert s.shape==(A,)
