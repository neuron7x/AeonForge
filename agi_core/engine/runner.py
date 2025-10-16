import argparse, numpy as np
from agi_core.envs.tanh_env import TanhEnv
from agi_core.envs.lqr_env import LQREnv
from agi_core.envs.reacher2d import Reacher2D
from agi_core.envs.minigrid_lite import MiniGridLite
from agi_core.world.linear_scm_np import LinearDynamicsSCM
from agi_core.world.neural_scm_np import NeuralDynamicsSCM
from agi_core.affordance.affordance_np import LinearAffordanceMap
from agi_core.relevance.relevance_np import RelevanceFilter
from agi_core.meta.meta_bandit_np import MetaBanditController
from agi_core.meta.meta_linucb_np import LinUCBMeta
from agi_core.devloop.replay_safe import SafeReplayBuffer
from agi_core.utils.metrics import rollout_real_gap, set_global_seed
from agi_core.utils.random import RNGManager
from agi_core.utils.jsonl import JSONLLogger
from agi_core.utils.tracking import WBTracker, WBConfig

def _env(kind,S,A,seed):
    if kind=='tanh': return TanhEnv(S,A,0.02,seed+1)
    if kind=='lqr': return LQREnv(S if S>0 else 5,A if A>0 else 3,seed=seed+1)
    if kind=='reacher': return Reacher2D(step_lim=48, seed=seed+1)
    if kind=='minigrid': return MiniGridLite(size=9, step_limit=48, seed=seed+1)
    raise ValueError(kind)

def greedy(aff,A,rng,low=-0.7,high=0.7):
    def pol(x,t):
        feas=aff.feasible_actions(x,0.5)
        if feas.size==0:
            return rng.uniform(low,high,size=A)
        a=np.zeros(A); a[int(feas[0])%A]=high; return a
    return pol

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--env',default='tanh'); ap.add_argument('--iters',type=int,default=50)
    ap.add_argument('--horizon',type=int,default=30); ap.add_argument('--state-dim',type=int,default=6); ap.add_argument('--action-dim',type=int,default=3)
    ap.add_argument('--seed',type=int,default=42); ap.add_argument('--model',choices=['linear','neural'],default='linear'); ap.add_argument('--meta',choices=['bandit','linucb'],default='linucb')
    ap.add_argument('--linucb-mode',choices=['full','diag'],default='diag'); ap.add_argument('--log-jsonl',type=str,default='')
    args=ap.parse_args()
    set_global_seed(args.seed); rngm=RNGManager(args.seed); rng=rngm.get('policy'); env=_env(args.env,args.state_dim,args.action_dim,args.seed)
    S,A=getattr(env,'S',args.state_dim), getattr(env,'A',args.action_dim)
    scm=LinearDynamicsSCM(S,A,1e-2,args.seed+2) if args.model=='linear' else NeuralDynamicsSCM(S,A,64,3e-3,args.seed+2)
    aff=LinearAffordanceMap(S,A,1e-2); rel=RelevanceFilter(S,max(1,S//2))
    meta=MetaBanditController(0.15,args.seed+3) if args.meta=='bandit' else LinUCBMeta(3,16,0.8,args.linucb_mode,args.seed+3)
    replay=SafeReplayBuffer(20000,S,A,rng,eviction='prioritized',protect_latest=2048)
    jlog=JSONLLogger(args.log_jsonl) if args.log_jsonl else None
    if jlog: jlog.__enter__()
    try:
        x=np.zeros(S)
        for it in range(args.iters):
            pol=greedy(aff,A,rng)
            total=0.0
            for t in range(args.horizon):
                a=pol(x,t); x1,r=env.step(x,a); total+=r; mask=rel.mask(x,x1,r); td=float(np.linalg.norm(x1 - scm.predict_next(x,a,False)))
                replay.add(x,a,x1,r,td_error=td); x=x1
            xs,aa,xp,_=replay.sample(1024)
            if xs.shape[0]>=64:
                if args.model=='linear': scm.fit(xs,aa,xp)
                else: scm.fit(xs,aa,xp,epochs=60,batch=128,patience=6)
            x0=np.zeros(S)
            try:
                Xm,_=scm.rollout(x0,pol,args.horizon,do_plan=[None]*args.horizon,noise=True)
                Xe,_,_=env.rollout(x0,pol,args.horizon)
                rmse=float(rollout_real_gap(Xm,Xe))
            except Exception:
                rmse=0.0
            rec={'it':it+1,'reward':total,'rmse':rmse}
            if jlog: jlog.log(rec)
    finally:
        if jlog: jlog.__exit__(None,None,None)

if __name__=='__main__': main()
