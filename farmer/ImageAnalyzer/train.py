from utils import reporter as rp
from utils.parser import get_parser
from utils.reader import train_test_files
from utils.model import build_model

from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy


def train(args):
    train_set, test_set = train_test_files()

    reporter = rp.Reporter(train_set, test_set,
                           size=(args.width, args.height),
                           nb_categories=args.classes,
                           parser=args
                           )
    checkpoint = ModelCheckpoint(reporter.model_dir + '/best_model.h5')

    # define model
    model = build_model(task=args.task,
                        nb_classes=args.classes,
                        img_height=args.height,
                        img_width=args.width,
                        backbone=args.backbone
                        )
    model.compile(args.optimizer, loss=categorical_crossentropy, metrics=['acc'])
    model.fit_generator(
        reporter.generate_batch_arrays(),
        steps_per_epoch=len(train_set)//args.batchsize,
        callbacks=[reporter, checkpoint],
        epochs=args.epoch,
        validation_data=reporter.generate_batch_arrays(training=False),
        validation_steps=len(test_set)//args.batchsize
    )


if __name__ == '__main__':
    parse_args = get_parser().parse_args()
    train(parse_args)
