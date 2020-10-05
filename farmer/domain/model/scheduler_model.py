from dataclasses import dataclass
from typing import List, Dict
import numpy as np

@dataclass
class StepLR:
    step_size: int = None
    step_gamma: float = 0.1

    def func(self, base_lr, **kwargs):
        self.base_lr = base_lr
        return self.step_lr

    def step_lr(self, epoch):
        reduce_num = epoch // self.step_size
        lr = self.base_lr * (self.step_gamma ** reduce_num)
        return lr

@dataclass
class MultiStepLR:
    step_gamma: float = 0.5
    milestones: dict = None

    def func(self, base_lr, n_epoch, **kwargs):
        self.milestone_num = 0
        self.base_lr = base_lr
        milestones = []
        mile_list = sorted(
            self.milestones.items(),
            key=lambda x: x[0]
        )
        for _, v in mile_list:
            step_epoch = int(
                n_epoch * ( v / 100 )
            )
            milestones.append(step_epoch)
        self.milestones = milestones

        return self.multi_step_lr

    def multi_step_lr(self, epoch):
        if epoch in self.milestones:
            self.milestone_num = self.milestones.index(epoch) + 1
        lr = self.base_lr * (self.step_gamma ** self.milestone_num)
        return lr

@dataclass
class ExponentialLR:
    exp_gamma: float = 0.90

    def func(self, base_lr, **kwargs):
        self.base_lr = base_lr
        return self.exponential_lr

    def exponential_lr(self, epoch):
        lr = self.base_lr * (self.exp_gamma ** epoch)
        return lr

@dataclass
class CyclicalLR:
    step_size: int = None
    cyc_freq: int = None
    cyc_lr_max: float = None
    cyc_lr_min: float = None

    def func(self, n_epoch, **kwargs):
        epoch_per_freq = n_epoch / self.cyc_freq
        self.step_size = epoch_per_freq // 2
        return self.cyclical_lr

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

@dataclass
class CosineDecay:
    cosine_lr_max: float = 0.001
    cosine_lr_min: float = 0.0001

    def func(self, n_epoch, **kwargs):
        self.n_epoch = n_epoch
        return self.cosine_decay

    def cosine_decay(self, epoch):
        lr = self.cosine_lr_min
        lr += 1/2*(self.cosine_lr_max-self.cosine_lr_min)*(1+np.cos(epoch/self.n_epoch*np.pi))
        return lr
