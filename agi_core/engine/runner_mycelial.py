import argparse, numpy as np, os
from agi_core.integration.mycelium_np import MycelialIntegration
from agi_core.integration.signal_types import Signal, ProcessResult
from agi_core.envs.tanh_env import TanhEnv
from agi_core.world.linear_scm_np import LinearDynamicsSCM
from agi_core.affordance.affordance_np import LinearAffordanceMap
from agi_core.viz.ascii_timeline import write_timeline
from agi_core.viz.gif import build_gif

class Aff:
    def __init__(self,a): self.a=a
    def process(self,sig, strength, path):
        if sig.type=='need_action' and sig.state is not None:
            s=self.a.score_single(sig.state); o=np.zeros(self.a.A); o[int(s.argmax())]=1.0
            return ProcessResult(True,o,float(s.max()))
        return ProcessResult(False,None,0.0)
class WM:
    def __init__(self,m): self.m=m
    def process(self,sig, strength, path):
        if sig.type=='predict' and sig.state is not None and sig.payload and 'a' in sig.payload:
            x1=self.m.predict_next(sig.state, sig.payload['a'], False); return ProcessResult(True,x1,1.0)
        return ProcessResult(False,None,0.0)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--iters',type=int,default=20); ap.add_argument('--horizon',type=int,default=24); ap.add_argument('--viz-dir',type=str,default='out/viz'); ap.add_argument('--gif-fps',type=int,default=6); args=ap.parse_args()
    os.makedirs(args.viz_dir, exist_ok=True)
    env=TanhEnv(6,3,seed=0); scm=LinearDynamicsSCM(6,3,1e-2,seed=1); aff=LinearAffordanceMap(6,3,1e-2)
    net=MycelialIntegration({'aff':Aff(aff),'wm':WM(scm),'meta':object()})
    x=np.zeros(6); pngs=[]
    for it in range(args.iters):
        act = net.propagate('aff', Signal(type='need_action', state=x))
        if act is None:
            act = np.array([1.0,0.0,0.0])
        x_next = net.propagate('wm', Signal(type='predict', state=x, payload={'a':act}))
        if x_next is not None:
            x = x_next
        if (it+1)%5==0:
            tl=net.ascii_timeline(6); p=os.path.join(args.viz_dir, f'timeline_{it+1:03d}.txt'); write_timeline(p, tl)
            gp=os.path.join(args.viz_dir, f'graph_{it+1:03d}.png'); open(gp,'wb').write(b'\x89PNG\r\n\x1a\n'); pngs.append(gp)
    if pngs:
        build_gif(pngs, os.path.join(args.viz_dir,'mycelium.gif'), fps=max(1,args.gif_fps))

if __name__=='__main__': main()
