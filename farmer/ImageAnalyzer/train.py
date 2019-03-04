from .utils import reporter as rp
from .utils.model import build_model

from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy


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
    model.compile(reporter.optimizer, loss=categorical_crossentropy, metrics=['acc'])
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
