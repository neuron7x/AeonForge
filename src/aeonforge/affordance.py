class AffordanceMap:
    def __init__(self, input_dim=128, action_dim=8):
        self.input_dim=input_dim; self.action_dim=action_dim
    def infer(self, obs):
        return [1.0/self.action_dim]*self.action_dim
