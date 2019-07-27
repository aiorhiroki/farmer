import sys
import os
import numpy as np
import cv2
from sklearn import metrics
from tqdm import tqdm
from .utils import reporter as rp
from .utils.model import build_model, iou_score
from .utils.model import cce_dice_loss
from .utils.image_util import ImageUtil
from .utils.generator import ImageSequence
from ncc.callbacks import MultiGPUCheckpointCallback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import categorical_crossentropy
import tensorflow as tf
from tensorflow.keras.utils import multi_gpu_model
from .task import Task

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import farmer.ImageAnalyzer
    __package__ = "farmer.ImageAnalyzer"


def _build_model(task_id, reporter):
    multi_gpu = False

    with tf.device("/cpu:0"):
        base_model = build_model(
            task=task_id,
            model_name=reporter.model_name,
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
        return base_model, reporter, multi_gpu, base_model


def train(config):
    task_id = int(config['project_settings'].get('task_id'))
    reporter = rp.Reporter(config)
    model, reporter, multi_gpu, base_model = _build_model(task_id, reporter)
    if task_id == Task.CLASSIFICATION:
        model.compile(reporter.optimizer,
                      loss=categorical_crossentropy, metrics=['acc'])
    elif task_id == Task.SEMANTIC_SEGMENTATION:
        model.compile(
            reporter.optimizer,
            loss=cce_dice_loss,
            metrics=[iou_score]
        )
    else:
        raise NotImplementedError

    np.random.shuffle(reporter.train_files)
    train_gen = ImageSequence(
        annotations=reporter.train_files,
        input_shape=(reporter.height, reporter.width),
        nb_classes=reporter.nb_classes,
        task=task_id,
        batch_size=reporter.batch_size,
        augmentation=reporter.augmentation
    )
    validation_gen = ImageSequence(
        annotations=reporter.validation_files,
        input_shape=(reporter.height, reporter.width),
        nb_classes=reporter.nb_classes,
        task=task_id,
        batch_size=reporter.batch_size
    )

    model.fit_generator(
        train_gen,
        steps_per_epoch=len(train_gen),
        callbacks=_set_callbacks(multi_gpu, reporter, base_model),
        epochs=reporter.epoch,
        validation_data=validation_gen,
        validation_steps=len(validation_gen),
        workers=16 if multi_gpu else 1,
        max_queue_size=32 if multi_gpu else 10,
        use_multiprocessing=multi_gpu
    )
    if multi_gpu:
        base_model.save(os.path.join(reporter.model_dir, 'last_model.h5'))


def classification_predict(config, save_npy=False):
    task_id = int(config['project_settings'].get('task_id'))
    reporter = rp.Reporter(config)
    model, reporter, multi_gpu, base_model = _build_model(task_id, reporter)
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
        np.save(f'{reporter.model_name}.npy', prediction)
    return prediction, true


def segmentation_predict():
    task = 'segmentation'
    model, reporter, multi_gpu, base_model = _build_model(task)
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


def segmentation_evaluation():
    task = 'segmentation'
    model, reporter, multi_gpu, base_model = _build_model(task)
    iou = reporter.iou_validation(reporter.test_files, model)
    print('IoU: ', iou)


def classification_evaluation(config):
    class_names = config['project_settings'].get('class_names')
    prediction, true_cls = classification_predict(config)
    prediction_cls = np.argmax(prediction)
    true = np.eye(len(class_names), dtype=np.uint8)[true_cls]
    eval_report = metrics.classification_report(
        true_cls, prediction_cls
    )
    fpr, tpr, thresholds = metrics.roc_curve(true, prediction)
    auc = metrics.auc(fpr, tpr)
    eval_report.update(
        dict(
            fpr=fpr, tpr=tpr, thresholds=thresholds, auc=auc
        )
    )

    return eval_report


def _set_callbacks(multi_gpu, reporter, base_model=None):
    best_model_name = 'best_model.h5'

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
