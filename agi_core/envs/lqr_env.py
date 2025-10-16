import numpy as np
class LQREnv:
    def __init__(self,state_dim:int=5,action_dim:int=3,noise:float=0.01,seed:int=0):
        self.S,self.A=state_dim,action_dim; self.rng=np.random.default_rng(seed)
        M=self.rng.standard_normal((self.S,self.S))*0.2; self.Am=np.eye(self.S)*0.9 + (M-M.T)*0.05
        self.Bm=self.rng.standard_normal((self.S,self.A))*0.2; self.Q=np.eye(self.S); self.R=0.5*np.eye(self.A); self.noise=noise
    def step(self,x,u,do_set=None):
        x=x.reshape(-1); u=u.reshape(-1); nxt=self.Am@x + self.Bm@u + self.rng.normal(0.0,self.noise,self.S)
        if isinstance(do_set,dict):
            for k,v in do_set.items():
                if 0<=k<self.S: nxt[k]=float(v)
        r=- float(x.T@self.Q@x + u.T@self.R@u); return nxt,r
    def rollout(self,x0,policy,horizon:int,do_plan=None):
        x=np.array(x0,dtype=float).reshape(-1); X=[x.copy()]; A=[]; R=[]
        for t in range(horizon):
            a=policy(x,t); ds=None if (do_plan is None or t>=len(do_plan)) else do_plan[t]
            x,r=self.step(x,a,ds); X.append(x.copy()); A.append(a.copy()); R.append(r)
        return np.stack(X,0), np.stack(A,0), np.array(R)
