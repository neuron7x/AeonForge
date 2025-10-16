class CausalWorldModel:
    def do(self, var:str, value):
        return {var: value, 'effect': 'counterfactual-updated'}
