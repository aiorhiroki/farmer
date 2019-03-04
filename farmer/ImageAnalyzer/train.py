from .utils import reporter as rp
from .utils.reader import train_test_files
from .utils.model import build_model

from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy


def train(task):
    train_set, test_set = train_test_files()

    reporter = rp.Reporter(train_set, test_set, task)
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
        steps_per_epoch=len(train_set)//reporter.batch_size,
        callbacks=[reporter, checkpoint],
        epochs=reporter.epoch,
        validation_data=reporter.generate_batch_arrays(training=False),
        validation_steps=len(test_set)//reporter.batch_size
    )


if __name__ == '__main__':
    train(task='classification')
