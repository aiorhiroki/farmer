import sys
import os
from .utils import reporter as rp
from .utils.model import build_model, cce_dice_loss, iou_score

from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy
import tensorflow as tf # add
from keras.utils import multi_gpu_model # add

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import farmer.ImageAnalyzer
    __package__ = "farmer.ImageAnalyzer"


def _train(task):

    reporter = rp.Reporter(task)
    checkpoint = ModelCheckpoint(reporter.model_dir + '/best_model.h5')

    # define model
    with tf.device("/cpu:0"):  # add
        base_model = build_model(task=task,
                                 nb_classes=reporter.nb_classes,
                                 height=reporter.height,
                                 width=reporter.width,
                                 backbone=reporter.backbone
                                 )
    model = multi_gpu_model(base_model, gpus=4)
    if task == 'classification':
        model.compile(reporter.optimizer, loss=categorical_crossentropy, metrics=['acc'])
    elif task == 'segmentation':
        model.compile(reporter.optimizer, loss=cce_dice_loss, metrics=[iou_score])
    else:
        raise NotImplementedError
    model.fit_generator(
        reporter.generate_batch_arrays(),
        steps_per_epoch=len(reporter.train_files)//reporter.batch_size,
        callbacks=[reporter, checkpoint],
        epochs=reporter.epoch,
        # validation_data=reporter.generate_batch_arrays(training=False),
        # validation_steps=len(reporter.test_files)//reporter.batch_size,
        workers=16,
        max_queue_size=32,
        use_multiprocessing=True
    )


def classification():
    _train('classification')


def segmentation():
    _train('segmentation')


if __name__ == '__main__':
    classification()
