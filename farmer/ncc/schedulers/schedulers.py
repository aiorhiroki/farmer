import numpy as np

class Scheduler:
    def __init__(self, cos_lr_max, cos_lr_min, T_max, base_lr,
                step_size, step_gamma, milestones, exp_gamma, cyc_lr_max, cyc_lr_min):
        self.cos_lr_max = cos_lr_max
        self.cos_lr_min = cos_lr_min
        self.T_max = T_max
        self.base_lr = base_lr
        self.step_size = step_size
        self.step_gamma = step_gamma
        self.milestones = milestones
        self.milestone_num = 0
        self.exp_gamma = exp_gamma
        self.cyc_lr_max = cyc_lr_max
        self.cyc_lr_min = cyc_lr_min

        self.milestones = []
        if milestones:
            mile_list = sorted(
                milestones.items(),
                key=lambda x: x[0]
            )
            for _, v in mile_list:
                step_epoch = int(
                    T_max * ( v / 100 )
                )
                self.milestones.append(step_epoch)


    def cosine_decay(self, epoch):
        lr = self.cos_lr_min
        lr += 1/2*(self.cos_lr_max-self.cos_lr_min)*(1+np.cos(epoch/self.T_max*np.pi))
        return lr

    def step_lr(self, epoch):
        reduce_num = epoch // self.step_size
        lr = self.base_lr * (self.step_gamma ** reduce_num)
        return lr

    def multi_step_lr(self, epoch):
        if epoch in self.milestones:
            self.milestone_num = self.milestones.index(epoch) + 1
        lr = self.base_lr * (self.step_gamma ** self.milestone_num)
        return lr

    def exponential_lr(self, epoch):
        lr = self.base_lr * (self.exp_gamma ** epoch)
        return lr

    def cyclical_lr(self, epoch):
        max_min_diff = self.cyc_lr_max - self.cyc_lr_min
        quotient = epoch // self.step_size
        remainder = epoch % self.step_size
        if quotient % 2 == 0:
            lr = self.cyc_lr_min
            lr += max_min_diff * (remainder / self.step_size)
        else:
            lr = self.cyc_lr_max
            lr -= max_min_diff * (remainder / self.step_size)
        return lr

