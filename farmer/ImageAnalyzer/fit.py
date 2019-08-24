import sys
import os
import numpy as np
import cv2
from sklearn import metrics
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tqdm import tqdm
from .utils import reporter as rp
from .utils.model import build_model
from .utils.image_util import ImageUtil
from .utils.generator import ImageSequence

import ncc
from .task import Task

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import farmer.ImageAnalyzer
    __package__ = "farmer.ImageAnalyzer"


def train(config):
    reporter = rp.Reporter(config)
    model, base_model = build_model(reporter)
    sequence_args = dict(
        annotations=reporter.train_files,
        input_shape=(reporter.height, reporter.width),
        nb_classes=reporter.nb_classes,
        task=reporter.task,
        batch_size=reporter.batch_size,
        augmentation=reporter.augmentation
    )
    train_gen = ImageSequence(**sequence_args)

    sequence_args.update(
        annotations=reporter.validation_files,
        augmentation=False
    )
    validation_gen = ImageSequence(**sequence_args)

    model.fit_generator(
        train_gen,
        steps_per_epoch=len(train_gen),
        callbacks=_set_callbacks(reporter.multi_gpu, reporter, base_model),
        epochs=reporter.epoch,
        validation_data=validation_gen,
        validation_steps=len(validation_gen),
        workers=16 if reporter.multi_gpu else 1,
        max_queue_size=32 if reporter.multi_gpu else 10,
        use_multiprocessing=reporter.multi_gpu
    )
    if reporter.multi_gpu:
        base_model.save(os.path.join(reporter.model_dir, 'last_model.h5'))


def classification_predict(config, save_npy=False):
    task_id = int(config['project_settings'].get('task_id'))
    reporter = rp.Reporter(config, training=False)
    model, reporter, multi_gpu, base_model = build_model(reporter)
    test_gen = ImageSequence(
        annotations=reporter.test_files,
        input_shape=(reporter.height, reporter.width),
        nb_classes=reporter.nb_classes,
        task=task_id,
        batch_size=reporter.batch_size
    )
    prediction = model.predict_generator(
        test_gen,
        steps=len(test_gen),
        workers=16 if multi_gpu else 1,
        max_queue_size=32 if multi_gpu else 10,
        use_multiprocessing=multi_gpu,
        verbose=1
    )
    true = np.array([test_data[1] for test_data in reporter.test_files])
    if save_npy:
        np.save('{}.npy'.format(reporter.model_name), prediction)
    return prediction, true


def segmentation_predict(config):
    task = 'segmentation'
    reporter = rp.Reporter(config, training=False)
    model, reporter, multi_gpu, base_model = build_model(task)
    image_util = ImageUtil(
        reporter.nb_classes,
        (reporter.height, reporter.width)
    )
    for input_file, _ in tqdm(reporter.test_files):
        file_name = os.path.basename(input_file)
        input_image = image_util.read_image(
            input_file, anti_alias=True
        )
        # need to use base model
        prediction = base_model.predict(np.expand_dims(input_image, axis=0))
        output = image_util.blend_image(
            prediction[0], image_util.current_raw_size)
        cv2.imwrite(os.path.join(reporter.image_test_dir, file_name), output)


def evaluate(config):
    task_id = int(config['project_settings'].get('task_id'))
    if task_id == Task.CLASSIFICATION:
        eval_report = classification_evaluation(config)
    elif task_id == Task.SEMANTIC_SEGMENTATION:
        eval_report = segmentation_evaluation(config)
    return eval_report


def segmentation_evaluation(config):
    task = 'segmentation'
    reporter = rp.Reporter(config, training=False)
    model, reporter, multi_gpu, base_model = build_model(task)
    iou = reporter.iou_validation(reporter.test_files, model)
    print('IoU: ', iou)


def classification_evaluation(config):
    nb_classes = int(config['project_settings'].get('nb_classes'))
    prediction, true_cls = classification_predict(config)
    prediction_cls = np.argmax(prediction, axis=1)
    true = np.eye(nb_classes, dtype=np.uint8)[true_cls]
    eval_report = metrics.classification_report(
        true_cls, prediction_cls, output_dict=True
    )
    fpr, tpr, auc = ncc.metrics.roc(
        true, prediction, nb_classes, show_plot=False
    )
    eval_report.update(
        dict(
            fpr=list(fpr['macro']),
            tpr=list(tpr['macro']),
            auc=auc['macro']
        )
    )
    return eval_report


def _set_callbacks(multi_gpu, reporter, base_model=None):
    best_model_name = 'best_model.h5'
    if multi_gpu:
        checkpoint = ncc.callbacks.MultiGPUCheckpointCallback(
            filepath=os.path.join(reporter.model_dir, best_model_name),
            base_model=base_model,
            save_best_only=True,
        )
    else:
        checkpoint = ModelCheckpoint(
            filepath=os.path.join(reporter.model_dir, best_model_name),
            save_best_only=True,
        )
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=3, verbose=1)
    plot_history = ncc.callbacks.PlotHistory(reporter.learning_dir)
    return [reporter, checkpoint, reduce_lr, plot_history]
