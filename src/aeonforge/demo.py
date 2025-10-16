from .affordance import AffordanceMap
from .relevance import RelevanceFilter
from .devloop import DevLoop
from .world_model import CausalWorldModel
from .meta import MetaAttentionController

def run():
    aff=AffordanceMap(); rel=RelevanceFilter(); dev=DevLoop(); wm=CausalWorldModel(); meta=MetaAttentionController()
    obs=[0.2,0.5,0.3]
    actions=aff.infer(obs); weights=rel.mask(actions); dev.log({'obs':obs,'actions':actions,'weights':weights}); cf=wm.do('action','adjust'); stop=meta.should_stop(0.92)
    print('Affordance:',actions)
    print('Relevance:',weights)
    print('DevLoop replay:',dev.replay())
    print('WorldModel do():',cf)
    print('Meta should_stop:',stop)
