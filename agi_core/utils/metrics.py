import numpy as np, random

def set_global_seed(s:int): random.seed(s); np.random.seed(s)

def rollout_real_gap(Xm:np.ndarray, Xe:np.ndarray)->float:
    n=min(len(Xm),len(Xe));
    return 0.0 if n==0 else float(np.sqrt(np.mean((Xm[:n]-Xe[:n])**2)))
