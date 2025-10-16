import numpy as np
class TanhEnv:
    def __init__(self,state_dim:int,action_dim:int,noise_scale:float=0.02,seed:int=0):
        self.S,self.A=state_dim,action_dim; self.rng=np.random.default_rng(seed)
        self.W=self.rng.standard_normal((self.S,self.S))*0.2; self.U=self.rng.standard_normal((self.S,self.A))*0.2; self.noise=noise_scale
    def step(self,x,a,do_set=None):
        x=np.tanh(self.W@x.reshape(-1)+self.U@a.reshape(-1))+self.rng.normal(0.0,self.noise,self.S)
        if isinstance(do_set,dict):
            for k,v in do_set.items():
                if 0<=k<self.S: x[k]=float(v)
        r=-float(np.linalg.norm(x)); return x,r
    def rollout(self,x0,policy,horizon:int,do_plan=None):
        x=np.array(x0,dtype=float).reshape(-1); X=[x.copy()]; A=[]; R=[]
        for t in range(horizon):
            a=policy(x,t); ds=None if (do_plan is None or t>=len(do_plan)) else do_plan[t]
            x,r=self.step(x,a,ds); X.append(x.copy()); A.append(a.copy()); R.append(r)
        return np.stack(X,0), np.stack(A,0), np.array(R)
