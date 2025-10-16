import hashlib, numpy as np

class RNGManager:
    def __init__(self, master_seed:int): self.master=int(master_seed); self.cache={}
    def _derive(self, name:str)->int:
        h=hashlib.blake2b(f"{self.master}:{name}".encode(),digest_size=8).digest(); return int.from_bytes(h,'little')%(2**32-1)
    def get(self, name:str):
        if name not in self.cache: self.cache[name]=np.random.default_rng(self._derive(name))
        return self.cache[name]
