import numpy as np


def step_lr(epoch, base_lr, step_size, gamma):
    reduce_num = epoch // step_size
    lr = base_lr * (gamma ** reduce_num)
    return lr


def multi_step_lr(epoch, base_lr, milestones, milestone_num, gamma):
    for i, milestone in enumerate(milestones):
        if epoch >= milestone:
            milestone_num = i + 1
    lr = base_lr * (gamma ** milestone_num)
    return lr


def exponential_lr(epoch, base_lr, gamma):
    lr = base_lr * (gamma ** epoch)
    return lr


def cyclical_lr(epoch, lr_max, lr_min, step_size):
    max_min_diff = lr_max - lr_min
    epoch += 1
    quotient = epoch // step_size
    remainder = epoch % step_size
    if quotient % 2 == 0:
        lr = lr_min
        lr += max_min_diff * (remainder / step_size)
    else:
        lr = lr_max
        lr -= max_min_diff * (remainder / step_size)
    return lr


def cosine_decay(epoch, lr_min, lr_max, n_epoch):
    lr = lr_min
    lr += 1 / 2 * (lr_max - lr_min) * (1 + np.cos(epoch / n_epoch * np.pi))
    return lr


def exponential_cosine_decay(epoch, n_epoch, lr_min, lr_max, frequency, gamma=0.99):
    ft = (2 * frequency - 1) * epoch
    lr = cosine_decay(ft, lr_min, lr_max, n_epoch)
    lr *= gamma ** epoch
    return lr
