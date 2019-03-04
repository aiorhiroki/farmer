import sys
import os
from .utils import reporter as rp
from .utils.model import build_model, bce_jaccard_loss, iou_score

from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy


# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import farmer.ImageAnalyzer
    __package__ = "farmer.ImageAnalyzer"


def _train(task):

    reporter = rp.Reporter(task)
    checkpoint = ModelCheckpoint(reporter.model_dir + '/best_model.h5')

    # define model
    model = build_model(task=task,
                        nb_classes=reporter.nb_classes,
                        height=reporter.height,
                        width=reporter.width,
                        backbone=reporter.backbone
                        )
    if task == 'classification':
        model.compile(reporter.optimizer, loss=categorical_crossentropy, metrics=['acc'])
    elif task == 'segmentation':
        model.compile(reporter.optimizer, loss=bce_jaccard_loss, metrics=[iou_score])
    else:
        raise NotImplementedError
    model.fit_generator(
        reporter.generate_batch_arrays(),
        steps_per_epoch=len(reporter.train_files)//reporter.batch_size,
        callbacks=[reporter, checkpoint],
        epochs=reporter.epoch,
        validation_data=reporter.generate_batch_arrays(training=False),
        validation_steps=len(reporter.test_files)//reporter.batch_size
    )


def classification():
    _train('classification')


def segmentation():
    _train('segmentation')


if __name__ == '__main__':
    classification()
