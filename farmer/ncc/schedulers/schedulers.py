from ..schedulers import functional as F


class StepLR:
    def __init__(self, base_lr, step_size, gamma=0.1, **kwargs):
        self.base_lr = base_lr
        self.step_size = step_size
        self.gamma = gamma

    def __call__(self, epoch):
        return F.step_lr(
            epoch,
            self.base_lr,
            self.step_size,
            self.gamma
        )


class MultiStepLR:
    def __init__(self, base_lr, n_epoch, milestones, gamma=0.5, **kwargs):
        self.milestone_num = 0
        self.base_lr = base_lr
        self.gamma = gamma
        milestones_tmp = []
        mile_list = sorted(
            milestones.items(),
            key=lambda x: x[0]
        )
        for _, v in mile_list:
            step_epoch = int(
                n_epoch * ( v / 100 )
            )
            milestones_tmp.append(step_epoch)
        self.milestones = milestones_tmp

    def __call__(self, epoch):
        return F.multi_step_lr(
            epoch,
            self.base_lr,
            self.milestones,
            self.milestone_num,
            self.gamma
        )


class ExponentialLR:
    def __init__(self, base_lr, gamma=0.9, **kwargs):
        self.base_lr = base_lr
        self.gamma = gamma

    def __call__(self, epoch):
        return F.exponential_lr(
            epoch,
            self.base_lr,
            self.gamma
        )


class CyclicalLR:
    def __init__(self, n_epoch, lr_max, lr_min, cyc_freq, **kwargs):
        epoch_per_freq = n_epoch / cyc_freq
        self.step_size = epoch_per_freq / 2
        self.lr_max = lr_max
        self.lr_min = lr_min

    def __call__(self, epoch):
        return F.cyclical_lr(
            epoch,
            self.lr_max,
            self.lr_min,
            self.step_size
        )


class CosineDecay:
    def __init__(self, n_epoch, lr_max, lr_min, **kwargs):
        self.n_epoch = n_epoch
        self.lr_max = lr_max
        self.lr_min = lr_min

    def __call__(self, epoch):
        return F.cosine_decay(
            epoch,
            self.lr_min,
            self.lr_max,
            self.n_epoch
        )


class ExponentialCosineDecay:
    def __init__(self, n_epoch, base_lr, lr_min, frequency, gamma, **kwargs):
        self.n_epoch = n_epoch
        self.lr_max = base_lr
        self.lr_min = lr_min
        self.frequency = frequency
        self.gamma = gamma

    def __call__(self, epoch):
        return F.exponential_cosine_decay(
            epoch,
            self.n_epoch,
            self.lr_min,
            self.lr_max,
            self.frequency,
            self.gamma
        )
