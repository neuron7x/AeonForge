import logging, sys

def get_logger(name:str):
    l=logging.getLogger(name)
    if not l.handlers:
        h=logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s'))
        l.addHandler(h)
    l.setLevel('INFO'); return l
