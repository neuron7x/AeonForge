import os, json
class JSONLLogger:
    def __init__(self, path:str): self.path=path; os.makedirs(os.path.dirname(path),exist_ok=True)
    def __enter__(self): self.f=open(self.path,'a',encoding='utf-8'); return self
    def log(self, rec:dict): self.f.write(json.dumps(rec,ensure_ascii=False)+'\n'); self.f.flush()
    def __exit__(self,*a): self.f.close()
