import argparse, numpy as np
from agi_core.envs.tanh_env import TanhEnv
from agi_core.envs.lqr_env import LQREnv
from agi_core.envs.reacher2d import Reacher2D
from agi_core.envs.minigrid_lite import MiniGridLite
from agi_core.world.linear_scm_np import LinearDynamicsSCM
from agi_core.affordance.affordance_np import LinearAffordanceMap

def greedy(aff,A):
    def pol(x,t): s=aff.score_single(x); a=np.zeros(A); a[int(s.argmax())]=1.0; return a
    return pol

def train(env,hz,iters,seed):
    rng=np.random.default_rng(seed); S,A=env.S,env.A; scm=LinearDynamicsSCM(S,A,1e-2,seed=seed+1); aff=LinearAffordanceMap(S,A,1e-2)
    rets=[]
    for _ in range(iters):
        policy=greedy(aff,A); X,U,R=env.rollout(np.zeros(S) if hasattr(env,'S') else 32, policy, hz)
        rets.append(float(np.sum(R)))
        xs,aa,xp=X[:-1],U,X[1:]; scm.fit(xs,aa,xp)
        aidx=np.argmax(U,1)%A; succ=(np.linalg.norm(xp[:,:min(2,S)],1) < np.linalg.norm(xs[:,:min(2,S)],1)).astype(float)
        aff.conditional_fit(xs,aidx,succ,min_samples=min(64,xs.shape[0]//2),min_success_rate=0.5)
    return float(np.mean(rets[-5:])), float(np.std(rets[-5:]))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--horizon',type=int,default=32); ap.add_argument('--iters-per-task',type=int,default=10); ap.add_argument('--seed',type=int,default=0); args=ap.parse_args()
    envs=[('tanh',TanhEnv(6,3,seed=args.seed+1)),('lqr',LQREnv(5,3,seed=args.seed+2)),('reacher',Reacher2D(step_lim=args.horizon, seed=args.seed+3)),('minigrid',MiniGridLite(size=9,step_limit=args.horizon,seed=args.seed+4))]
    for name,env in envs:
        m,s=train(env,args.horizon,args.iters_per_task,args.seed); print(f"{name}: {m:.3f} ± {s:.3f}")

if __name__=='__main__': main()
