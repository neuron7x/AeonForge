import numpy as np
class LinUCBMeta:
    def __init__(self,arms:int,d:int,alpha:float=0.8,mode:str='diag',seed:int=0):
        self.K,self.d,self.alpha,self.mode=arms,d,alpha,mode; self.rng=np.random.default_rng(seed)
        if mode=='full':
            self.A=[np.eye(d) for _ in range(self.K)]; self.Ainv=[np.eye(d) for _ in range(self.K)]; self.b=[np.zeros((d,1)) for _ in range(self.K)]
        else:
            self.A=[np.ones(d) for _ in range(self.K)]; self.b=[np.zeros(d) for _ in range(self.K)]
    def _theta(self,k:int):
        return (self.Ainv[k]@self.b[k]).reshape(-1,1) if self.mode=='full' else (self.b[k]/self.A[k]).reshape(-1,1)
    def select(self,ctx:np.ndarray)->int:
        x=ctx.reshape(-1,1); vals=[]
        for k in range(self.K):
            if self.mode=='full':
                mu=float((self._theta(k).T@x).reshape(())); ucb=self.alpha*float(np.sqrt((x.T@self.Ainv[k]@x).reshape(())))
            else:
                th=self._theta(k); mu=float((th.T@x).reshape(())); ucb=self.alpha*float(np.sqrt(np.sum((x.reshape(-1)**2)/self.A[k])))
            vals.append(mu+ucb)
        return int(np.argmax(vals))
    def update(self,arm:int,reward:float,ctx:np.ndarray):
        x=ctx.reshape(-1,1)
        if self.mode=='full':
            Ainv=self.Ainv[arm]; denom=float(1.0 + (x.T@Ainv@x)); Ax=Ainv@x; self.Ainv[arm]=Ainv - (Ax@Ax.T)/denom; self.A[arm]+=x@x.T; self.b[arm]+=reward*x
        else:
            self.A[arm]+= (x.reshape(-1)**2); self.b[arm]+= reward*x.reshape(-1)
