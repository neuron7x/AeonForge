import numpy as np
class MiniGridLite:
    def __init__(self,size:int=9,step_limit:int=64,seed:int=0): self.size=size; self.limit=step_limit; self.rng=np.random.default_rng(seed); self.S,self.A=4,4; self.reset()
    def reset(self): self.t=0; self.p=self.rng.integers(0,self.size,2); self.g=self.rng.integers(0,self.size,2); return self._obs()
    def _obs(self): return np.array([self.p[0],self.p[1],self.g[0],self.g[1]],dtype=float)
    def step(self,x,a,do_set=None):
        a=int(np.argmax(a))
        if a==0: self.p[0]=max(0,self.p[0]-1)
        elif a==1: self.p[0]=min(self.size-1,self.p[0]+1)
        elif a==2: self.p[1]=max(0,self.p[1]-1)
        else: self.p[1]=min(self.size-1,self.p[1]+1)
        if isinstance(do_set,dict):
            for k,v in do_set.items():
                if k in (0,1): self.p[k]=int(np.clip(v,0,self.size-1))
        self.t+=1; r=-float(np.linalg.norm(self.p-self.g)); return self._obs(), r
    def rollout(self,horizon:int,policy):
        x=self._obs(); X=[x.copy()]; A=[]; R=[]
        for t in range(horizon):
            a=policy(x,t); x,r=self.step(x,a); X.append(x.copy()); A.append(a.copy()); R.append(float(r))
        return np.stack(X,0), np.stack(A,0), np.array(R)
