import tensorflow as tf
import numpy as np


def gen(dataset, shuffle=False):
    indexes = np.arange(len(dataset))
    if shuffle:
        indexes = np.random.permutation(indexes)
    while True:
        for i in indexes:
            yield dataset[i]
