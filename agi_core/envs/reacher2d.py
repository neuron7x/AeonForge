import numpy as np
class Reacher2D:
    def __init__(self,step_lim:int=64,seed:int=0): self.rng=np.random.default_rng(seed); self.S,self.A=4,2; self.limit=step_lim; self.reset()
    def reset(self): self.t=0; self.p=self.rng.uniform(-1,1,2); self.g=self.rng.uniform(-1,1,2); return self._obs()
    def _obs(self): return np.array([self.p[0],self.p[1],self.g[0],self.g[1]],dtype=float)
    def step(self,x,a,do_set=None):
        u=np.clip(a.reshape(-1),-0.1,0.1); self.p=np.clip(self.p + u + self.rng.normal(0,0.01,2), -1.5, 1.5)
        if isinstance(do_set,dict):
            if 0 in do_set: self.p[0]=float(np.clip(do_set[0],-1.5,1.5))
            if 1 in do_set: self.p[1]=float(np.clip(do_set[1],-1.5,1.5))
        self.t+=1; r=-float(np.linalg.norm(self.p-self.g)); return self._obs(), r
    def rollout(self,horizon:int,policy):
        x=self._obs(); X=[x.copy()]; A=[]; R=[]
        for t in range(horizon):
            a=policy(x,t); x,r=self.step(x,a); X.append(x.copy()); A.append(a.copy()); R.append(float(r))
        return np.stack(X,0), np.stack(A,0), np.array(R)
