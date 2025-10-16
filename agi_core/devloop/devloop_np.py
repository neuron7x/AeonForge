class DevLoop:
    def __init__(self,threshold_lp:float=0.005,max_experts:int=3): self.th=threshold_lp; self.maxe=max_experts; self.experts=[]
    def learning_progress(self,recent_losses):
        if len(recent_losses)<3: return 0.0
        import numpy as np
        w=np.array([0.2,0.3,0.5]); recent=np.array(recent_losses[-3:])
        return float(np.dot(w, np.maximum(0.0, np.diff(np.concatenate([[recent[0]], recent])))))
    def maybe_grow(self,factory,train_batch,val_batch,eval_fn):
        if len(self.experts)>=self.maxe: return None
        cand=factory(); X,A,Y=train_batch
        try:
            cand.fit(X,A,Y); base=1e9 if not self.experts else eval_fn(self.experts[-1], *val_batch)
            cand_err=eval_fn(cand, *val_batch)
            if cand_err + self.th < base:
                self.experts.append(cand); return cand
        except Exception:
            return None
        return None
