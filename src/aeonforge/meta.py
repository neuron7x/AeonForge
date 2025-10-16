class MetaAttentionController:
    def __init__(self, threshold=0.1):
        self.threshold=threshold
    def should_stop(self, score:float)->bool:
        return score >= (1.0 - self.threshold)
