class DevLoop:
    def __init__(self):
        self.buffer=[]
    def curiosity(self, error:float)->float:
        return max(0.0,min(1.0,abs(error)))
    def replay(self):
        return list(self.buffer)[-5:]
    def log(self,item):
        self.buffer.append(item)
