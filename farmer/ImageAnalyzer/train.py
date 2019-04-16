import sys
import os
from .utils import reporter as rp
from .utils.model import build_model, cce_dice_loss, iou_score

from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy
import tensorflow as tf # add
from keras.utils import multi_gpu_model # add
from ncc.readers import data_set_from_annotation

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import farmer.ImageAnalyzer
    __package__ = "farmer.ImageAnalyzer"


def _train(task, step):

    reporter = rp.Reporter(task)
    train = "/home/hiroki/ncc-dev/data/Phase_Classification/data/processed/annotation/outside/"
    train_path = os.path.join(train, "train_{}.csv".format(step))
    reporter.train_files, _ = data_set_from_annotation(train_path, train_path)
    # define model
    with tf.device("/cpu:0"):
        base_model = build_model(task=task,
                                    nb_classes=reporter.nb_classes,
                                    height=reporter.height,
                                    width=reporter.width,
                                    backbone=reporter.backbone
                                    )
    if step > 0:
        base_model.load_weights('outside_{}.h5'.format(step-1))
    if reporter.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = reporter.gpu
        nb_gpu = len(reporter.gpu.split(','))
        multi_gpu = nb_gpu > 1
    else:
        multi_gpu = False

    if multi_gpu:
        model = multi_gpu_model(base_model, gpus=nb_gpu)
        reporter.batch_size *= nb_gpu
        compile_and_run(task, model, reporter, multi_gpu)
        base_model.save('outside_{}.h5'.format(step))

    else:
        compile_and_run(task, base_model, reporter, multi_gpu)


def compile_and_run(task, model, reporter, multi_gpu):
    if task == 'classification':
        model.compile(reporter.optimizer, loss=categorical_crossentropy, metrics=['acc'])
    elif task == 'segmentation':
        model.compile(reporter.optimizer, loss=cce_dice_loss, metrics=[iou_score])
    else:
        raise NotImplementedError

    if multi_gpu:
        validation_data = None
        validation_steps = None
        workers = 16
        max_queue_size = 32
        use_multiprocessing = True
        callbacks = [reporter]
    else:
        validation_data = reporter.generate_batch_arrays(training=False)
        validation_steps=len(reporter.test_files)//reporter.batch_size
        workers = 1
        max_queue_size = 10
        use_multiprocessing = False
        checkpoint = ModelCheckpoint(reporter.model_dir + '/best_model.h5')
        callbacks = [reporter, checkpoint]

    model.fit_generator(
        reporter.generate_batch_arrays(),
        steps_per_epoch=len(reporter.train_files)//reporter.batch_size,
        callbacks=callbacks,
        epochs=reporter.epoch,
        validation_data=validation_data,
        validation_steps=validation_steps,
        workers=workers,
        max_queue_size=max_queue_size,
        use_multiprocessing=use_multiprocessing
    )


def classification():
    for step in range(30):
        _train('classification', step)


def segmentation():
    _train('segmentation')
