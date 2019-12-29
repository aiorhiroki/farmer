import random as rn
import multiprocessing as mp
import numpy as np
import os
import tensorflow as tf
from keras import backend as K


def set_keras_env(gpu, seed=1):
    # set random_seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)
    tf.set_random_seed(seed)

    # set gpu and cpu devices
    if gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    core_num = mp.cpu_count()
    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=core_num,
        inter_op_parallelism_threads=core_num
    )
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
