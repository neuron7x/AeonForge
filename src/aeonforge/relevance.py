class RelevanceFilter:
    def mask(self, features):
        n=len(features) if hasattr(features,'__len__') else 1
        return [1.0/max(n,1)]*n
