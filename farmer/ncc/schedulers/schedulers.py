import numpy as np

class Scheduler:
    def __init__(self, lr_max, lr_min, T_max):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.T_max = T_max

    def cosine_decay(epoch):
        lr = self.lr_min
        lr += 1/2*(self.lr_max-self.lr_min)*(1+np.cos(epoch/self.T_max*np.pi))
        return lr