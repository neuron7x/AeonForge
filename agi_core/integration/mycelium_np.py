import numpy as np, time
from typing import Dict, List, Tuple
from .signal_types import Signal, ProcessResult

class MycelialIntegration:
    def __init__(self,modules:Dict[str,object],alpha:float=0.1,decay:float=0.98,min_w:float=1e-3,max_w:float=1.0,quarantine_failures:int=3):
        self.modules=modules; self.alpha=alpha; self.decay=decay; self.min_w=min_w; self.max_w=max_w
        self.edges={k:{} for k in modules}; names=list(modules.keys())
        for i in range(len(names)):
            for j in range(len(names)):
                if i!=j: self.edges[names[i]][names[j]]=0.05
        self.fail={k:0 for k in modules}; self.quarantine={k:False for k in modules}; self.qf=quarantine_failures
        self.history=[]
    def _safe_process(self,node:str,signal:Signal,strength:float,path:List[str])->ProcessResult:
        if self.quarantine.get(node,False): return ProcessResult(False,None,0.0,f"node '{node}' quarantined")
        try:
            out=self.modules[node].process(signal,strength=strength,path=path)
            return out if isinstance(out,ProcessResult) else ProcessResult(False,None,0.0,'bad result')
        except Exception as e:
            self.fail[node]+=1
            if self.fail[node]>=self.qf: self.quarantine[node]=True
            return ProcessResult(False,None,0.0,str(e))
    def propagate(self,source:str,signal:Signal,max_hops:int=6):
        queue=[(source,1.0,[source],signal)]; suggestions=[]; seen=set(); hops=0
        while queue and hops<max_hops:
            nxt=[]
            for current,strength,path,payload in queue:
                key=(current,tuple(path[-3:]));
                if key in seen: continue
                seen.add(key)
                res=self._safe_process(current,payload,strength,path)
                if res.success and res.suggestion is not None:
                    suggestions.append((strength*max(1e-6,res.confidence),path.copy(),res.suggestion))
                    self._reinforce_path(path,success_gain=float(res.confidence))
                for nb,w in self.edges[current].items():
                    if nb in path: continue
                    ns=strength*w*self.decay
                    if ns<1e-4: continue
                    nxt.append((nb,ns,path+[nb],payload))
            queue=nxt; hops+=1
        self.passive_decay(); self.history.append((time.time(), []))
        if not suggestions: return None
        suggestions.sort(key=lambda t:-t[0]); return suggestions[0][-1]
    def _reinforce_path(self,path:List[str],success_gain:float):
        for i in range(len(path)-1):
            u,v=path[i],path[i+1]; w=self.edges[u][v]; w=min(self.max_w, w*(1.0+self.alpha*max(0.0,success_gain))); self.edges[u][v]=w
    def passive_decay(self):
        for u,nbrs in self.edges.items():
            for v in list(nbrs.keys()): nbrs[v]=max(self.min_w, nbrs[v]*self.decay)
    def ascii_timeline(self,topk:int=6)->str:
        return '\n'.join([f"{i:03d} | edges={len(self.edges)}" for i,_ in enumerate(self.history[-topk:])])
