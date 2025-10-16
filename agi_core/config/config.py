import argparse

def load_config(path:str)->dict:
    import yaml
    with open(path,'r',encoding='utf-8') as f: return yaml.safe_load(f) or {}

def apply_config_to_argparse(ap:argparse.ArgumentParser,cfg:dict):
    for a in list(ap._actions):
        if a.dest in cfg: a.default = cfg[a.dest]
