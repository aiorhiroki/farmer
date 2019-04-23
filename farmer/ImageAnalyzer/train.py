import sys
import os
from .utils import reporter as rp
from .utils.model import build_model, cce_dice_loss, iou_score
from ncc.callbacks import MultiGPUCheckpointCallback
from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy
import tensorflow as tf  # add
from keras.utils import multi_gpu_model  # add

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import farmer.ImageAnalyzer
    __package__ = "farmer.ImageAnalyzer"


def _train(task):
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

    if multi_gpu:
        model = multi_gpu_model(base_model, gpus=nb_gpu)
        reporter.batch_size *= nb_gpu
        compile_and_run(task, model, reporter, multi_gpu, base_model)
        base_model.save(os.path.join(reporter.model_dir, 'last_model.h5'))
    else:
        compile_and_run(task, base_model, reporter, multi_gpu)


def compile_and_run(task, model, reporter, multi_gpu, base_model=None):
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
        reporter.test_files = list()  # empty list
        reporter.epoch = 1  # not to learn same data during split learning
    for step in range(split_steps):
        reporter.train_files = reporter.train_files[step::split_steps]

        model.fit_generator(
            reporter.generate_batch_arrays(),
            steps_per_epoch=len(reporter.train_files)//reporter.batch_size,
            callbacks=set_callbacks(multi_gpu, reporter, step, base_model),
            epochs=reporter.epoch,
            validation_data=reporter.generate_batch_arrays(training=False),
            validation_steps=len(
                reporter.validation_files)//reporter.batch_size,
            workers=16 if multi_gpu else 1,
            max_queue_size=32 if multi_gpu else 10,
            use_multiprocessing=multi_gpu
        )


def set_callbacks(multi_gpu, reporter, step, base_model=None):
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


def classification():
    _train('classification')


def segmentation():
    _train('segmentation')
