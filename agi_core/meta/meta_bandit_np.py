import numpy as np
class MetaBanditController:
    def __init__(self,epsilon:float=0.1,seed:int=0,arms:int=3): self.eps=epsilon; self.K=arms; self.rng=np.random.default_rng(seed); self.Q=np.zeros(self.K); self.N=np.zeros(self.K)+1e-6
    def select(self):
        return int(self.rng.integers(0,self.K)) if self.rng.random()<self.eps else int(np.argmax(self.Q))
    def update(self,arm:int,reward:float,ctx=None): self.N[arm]+=1; self.Q[arm]+= (reward-self.Q[arm])/self.N[arm]
