import sys
import os
import numpy as np

from .utils import reporter as rp
from .utils.model import build_model, cce_dice_loss, iou_score
from .utils.generator import ImageSequence
from ncc.callbacks import MultiGPUCheckpointCallback
from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy
import tensorflow as tf
from keras.utils import multi_gpu_model

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import farmer.ImageAnalyzer
    __package__ = "farmer.ImageAnalyzer"


def classification():
    _train('classification')


def segmentation():
    _train('segmentation')


def classification_predict():
    _predict('classification')


def segmentation_predict():
    _predict('segmentation')


def _build_model(task):
    multi_gpu = False
    reporter = rp.Reporter(task)

    with tf.device("/cpu:0"):
        base_model = build_model(
            task=task,
            nb_classes=reporter.nb_classes,
            height=reporter.height,
            width=reporter.width,
            backbone=reporter.backbone
        )

    if reporter.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = reporter.gpu
        nb_gpu = len(reporter.gpu.split(','))
        multi_gpu = nb_gpu > 1

    if reporter.model_path is not None:
        base_model.load_weights(reporter.model_path)

    if multi_gpu:
        model = multi_gpu_model(base_model, gpus=nb_gpu)
        reporter.batch_size *= nb_gpu
        return model, reporter, multi_gpu, base_model
    else:
        return base_model, reporter, multi_gpu, None


def _train(task):
    model, reporter, multi_gpu, base_model = _build_model(task)
    if task == 'classification':
        model.compile(reporter.optimizer,
                      loss=categorical_crossentropy, metrics=['acc'])
    elif task == 'segmentation':
        model.compile(reporter.optimizer, loss=cce_dice_loss,
                      metrics=[iou_score])
    else:
        raise NotImplementedError

    split_steps = int(reporter.nb_split)
    if split_steps > 1:
        reporter.validation_files = list()  # empty list
        reporter.epoch = 1  # not to learn same data during split learning
    for step in range(split_steps):
        reporter.train_files = reporter.train_files[step::split_steps]
        train_gen = ImageSequence(
            annotations=reporter.train_files,
            input_shape=(reporter.height, reporter.width),
            nb_classes=reporter.nb_classes
        )
        validation_gen = ImageSequence(
            annotations=reporter.validation_files,
            input_shape=(reporter.height, reporter.width),
            nb_classes=reporter.nb_classes
        )

        model.fit_generator(
            train_gen,
            steps_per_epoch=len(train_gen),
            callbacks=_set_callbacks(multi_gpu, reporter, step, base_model),
            epochs=reporter.epoch,
            validation_data=validation_gen,
            validation_steps=len(validation_gen),
            workers=16 if multi_gpu else 1,
            max_queue_size=32 if multi_gpu else 10,
            use_multiprocessing=multi_gpu
        )
        if multi_gpu:
            base_model.save(os.path.join(reporter.model_dir, 'last_model.h5'))


def _predict(task):
    model, reporter, multi_gpu, base_model = _build_model(task)
    test_gen = ImageSequence(
        annotations=reporter.test_files,
        input_shape=(reporter.height, reporter.width),
        nb_classes=reporter.nb_classes
    )
    prediction = model.predict_generator(
        test_gen,
        steps=len(test_gen),
        workers=16 if multi_gpu else 1,
        max_queue_size=32 if multi_gpu else 10,
        use_multiprocessing=multi_gpu,
        verbose=1
    )
    np.save(f'{reporter.model_name}.npy', prediction)


def _set_callbacks(multi_gpu, reporter, step, base_model=None):
    if step == 0:
        best_model_name = 'best_model.h5'
    else:
        best_model_name = 'best_mode_on_step%d.h5' % step

    if multi_gpu:
        checkpoint = MultiGPUCheckpointCallback(
            filepath=os.path.join(reporter.model_dir, best_model_name),
            base_model=base_model,
            save_best_only=True,
        )
    else:
        checkpoint = ModelCheckpoint(
            filepath=os.path.join(reporter.model_dir, best_model_name),
            save_best_only=True,
        )
    return [reporter, checkpoint]