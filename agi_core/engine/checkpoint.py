import os, json

def save_checkpoint_full(path:str, scm, aff, net=None, replay=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state={'scm':{},'aff':{}}
    if hasattr(scm,'Theta'): state['scm']['Theta']=getattr(scm,'Theta').tolist()
    if hasattr(aff,'W'): state['aff']['W']=getattr(aff,'W').tolist()
    if replay is not None:
        size=replay.capacity if replay.full else replay.ptr
        state['replay']={'capacity':replay.capacity,'ptr':replay.ptr,'full':replay.full,
                         'X_t': replay.X_t[:size].tolist(),'A_t':replay.A_t[:size].tolist(),'X_tp1':replay.X_tp1[:size].tolist(),'R':replay.R[:size].tolist()}
    with open(path,'w',encoding='utf-8') as f: json.dump(state,f)
