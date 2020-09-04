import numpy as np

class Scheduler:
    def __init__(self, lr_max, lr_min, T_max, base_lr,
                step_size, step_gamma, milestones, exp_gamma):
        # CosineDecay
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.T_max = T_max

        # TODO :引数に追加
        self.base_lr = base_lr

        # StepLR
        self.step_size = step_size
        self.step_gamma = step_gamma

        # MultiStepLR
        self.milestones = milestones

        # ExponentialLR
        self.exp_gamma = exp_gamma


    def cosine_decay(self, epoch):
        lr = self.lr_min
        lr += 1/2*(self.lr_max-self.lr_min)*(1+np.cos(epoch/self.T_max*np.pi))
        return lr

    def step_lr(self, epoch):
        if (epoch != 0) and (epoch % self.step_size == 0):
            reduce_num = epoch // self.step_size
            lr = self.base_lr * (self.step_gamma ** reduce_num)
        return lr

    def multi_step_lr(self, epoch):
        if epoch in self.milestones:
            index = self.milestones.index(epoch)
            lr = self.base_lr * (self.step_gamma ** (index + 1))
        return lr

    def exponential_lr(self, epoch):
        lr = self.base_lr * (self.exp_gamma ** epoch)
        return lr

    # def cyclical_lr(self, epoch):

