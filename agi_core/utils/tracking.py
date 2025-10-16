from dataclasses import dataclass
from typing import Optional, Any, Dict

@dataclass(slots=True)
class WBConfig:
    enabled: bool=False
    project: Optional[str]=None
    entity: Optional[str]=None
    run_name: Optional[str]=None

class _No:
    def log(self,*a,**k):...
    def finish(self):...

class WBTracker:
    def __init__(self,cfg:WBConfig): self.cfg=cfg; self._run=_No(); self._ok=False
    def __enter__(self):
        if not self.cfg.enabled: return self
        try:
            import wandb
            self._run=wandb.init(project=self.cfg.project or 'agi-core', entity=self.cfg.entity, name=self.cfg.run_name)
            self._ok=True
        except Exception: self._run=_No(); self._ok=False
        return self
    def log(self, m:Dict[str,Any], step:Optional[int]=None):
        try:
            self._run.log(m if step is None else {**m,'_step':step})
        except Exception: pass
    def __exit__(self,*a):
        try:
            if self._ok: self._run.finish()
        except Exception: pass
