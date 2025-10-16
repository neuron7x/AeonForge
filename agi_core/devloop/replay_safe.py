import numpy as np
class SafeReplayBuffer:
    def __init__(self,capacity:int,state_dim:int,action_dim:int,rng,eviction:str='prioritized',protect_latest:int=1024):
        self.capacity=capacity; self.S=state_dim; self.A=action_dim; self.rng=rng
        self.X_t=np.zeros((capacity,self.S)); self.A_t=np.zeros((capacity,self.A)); self.X_tp1=np.zeros((capacity,self.S)); self.R=np.zeros((capacity,)); self.td=np.zeros((capacity,))
        self.ptr=0; self.full=False; self.eviction=eviction; self.protect=protect_latest
    def add(self,x,a,xn,r,td_error:float=0.0):
        self.X_t[self.ptr]=x; self.A_t[self.ptr]=a; self.X_tp1[self.ptr]=xn; self.R[self.ptr]=r; self.td[self.ptr]=td_error
        self.ptr=(self.ptr+1)%self.capacity; self.full=self.full or self.ptr==0
    def _indices(self,n:int):
        size=self.capacity if self.full else self.ptr
        if size<=0: return np.array([],dtype=int)
        if self.eviction=='prioritized':
            w=self.td[:size]+1e-6; w[:max(0,size-self.protect)]=w[:max(0,size-self.protect)]*0.5; w=w/np.sum(w)
            return self.rng.choice(size,size=min(n,size),replace=size<n,p=w)
        elif self.eviction=='reservoir':
            return self.rng.choice(size,size=min(n,size),replace=size<n)
        else:
            return np.arange(max(0,size-min(n,size)), size)
    def sample(self,batch:int):
        idx=self._indices(batch); return self.X_t[idx], self.A_t[idx], self.X_tp1[idx], self.R[idx]
