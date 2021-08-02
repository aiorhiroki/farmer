from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mmdet.core.evaluation import eval_map
from mrcnn import visualize

from imgaug import augmenters as iaa
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import cv2
import os
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml

from farmer.ncc.readers import segmentation_set
from farmer.ncc.generators import MaskrcnnDataset

"""
# command
nohup docker exec -t maskrcnn_tf2 bash -c "cd $PWD && env CUDA_VISIBLE_DEVICES=1 \
python farmer/domain/tasks/train_mrcnn.py" > train1.out &
"""

with open("example/yamls/mrcnn-instruments.yaml") as yamlfile:
    yaml_config = yaml.safe_load(yamlfile)


if yaml_config["TRAIN_DIRS"] is None and yaml_config["VAL_DIRS"] is None:

    # for artery
    """
    ima_dataset_dirs = os.listdir(yaml_config["DATASET"] + "ima")
    sra_dataset_dirs = os.listdir(yaml_config["DATASET"] + "sra")

    ima_split_line = int(0.8*len(ima_dataset_dirs))
    sra_split_line = int(0.8*len(sra_dataset_dirs))

    ima_train = ima_dataset_dirs[:ima_split_line]
    ima_val = ima_dataset_dirs[ima_split_line:]
    sra_train = sra_dataset_dirs[:sra_split_line]
    sra_val = sra_dataset_dirs[sra_split_line:]

    ima_train = [f"ima/{d}" for d in ima_train]
    ima_val = [f"ima/{d}" for d in ima_val]
    sra_train = [f"sra/{d}" for d in sra_train]
    sra_val = [f"sra/{d}" for d in sra_val]
    yaml_config["TRAIN_DIRS"] = ima_train + sra_train
    yaml_config["VAL_DIRS"] = ima_val + sra_val
    """

    # for instruments10
    """
    train_dirs = list()
    val_dirs = list()
    for instrument_name in os.listdir(yaml_config["DATASET"]):
        for cv_id in range(1, 6):
            cv = f"cv{cv_id}"
            for case in os.listdir(str(Path(yaml_config["DATASET"]) / instrument_name / cv)):
                if cv_id == 5:
                    val_dirs.append(f"{instrument_name}/{cv}/{case}")
                else:
                    train_dirs.append(f"{instrument_name}/{cv}/{case}")
    yaml_config["TRAIN_DIRS"] = train_dirs
    yaml_config["VAL_DIRS"] = val_dirs
    """

    dataset_dirs = os.listdir(yaml_config["DATASET"])
    split_line = int(0.8*len(dataset_dirs))
    yaml_config["TRAIN_DIRS"] = dataset_dirs[:split_line]
    yaml_config["VAL_DIRS"] = dataset_dirs[split_line:]

    print("Train Dirs", yaml_config["TRAIN_DIRS"])
    print("Val Dirs", yaml_config["VAL_DIRS"])

class TrainConfig(Config):
    NAME = "farmer"
    NUM_CLASSES = 1 + max(yaml_config["CLASS_IDS"].values())
    IMAGE_MAX_DIM = 640
    USE_MINI_MASK = False


class InferenceConfig(TrainConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def evaluate_det(model, dataset, config, eval_type="bbox", image_ids=None):
    RESULTS_DIR = "results/forceps"
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    nb_classes = config.NUM_CLASSES - 1
    print("class names: ", dataset.class_names)
    print("submit dir: ", submit_dir)

    image_ids = image_ids or dataset.image_ids
    annotations = list()
    det_results = list()
    for i, image_id in tqdm(enumerate(image_ids)):
        # Load image
        image, _, gt_class_id, gt_bbox, mask = modellib.load_image_gt(
            dataset_val, config, image_id)
        gt_class_id = gt_class_id - 1
        annotations.append(dict(bboxes=gt_bbox, labels=gt_class_id))
        # Run detection
        r = model.detect([image], verbose=0)[0]
        r['class_ids'] = r['class_ids'] - 1
        det_result = [list() for _ in range(nb_classes)]
        num_det = r['rois'].shape[0]
        for det_id in range(num_det):
            class_id = r['class_ids'][det_id]
            for det_cls in range(nb_classes):
                if class_id == det_cls:
                    det_result[det_cls].append(
                        list(r['rois'][det_id]) + [float(r['scores'][det_id])])
        for det_cls in range(nb_classes):
            if len(det_result[det_cls]) == 0:
                det_result[det_cls] = np.empty((0, 5))
            else:
                det_result[det_cls] = np.array(det_result[det_cls])
        det_results.append(det_result)

        # to check prediction
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=True, show_mask=True,
            title="Predictions")

        # to check groud truth
        """
        visualize.display_instances(
            image, gt_bbox, mask, gt_class_id+1,
            dataset.class_names, [1.0 for _ in range(len(gt_bbox))],
            show_bbox=True, show_mask=True,
            title="Predictions")
        """
        plt.savefig("{}/{}.png".format(
          submit_dir, os.path.basename(dataset.image_info[image_id]["id"])))

    mean_ap, _ = eval_map(det_results, annotations, iou_thr=0.75)
    print(mean_ap)


if yaml_config["TRAINING"]:
    config = TrainConfig()
    mode = "training"
else:
    config = InferenceConfig()
    mode = "inference"
config.display()

class MaskRCNNSingleThread(modellib.MaskRCNN):
    def __init__(self, mode, config, model_dir):
        super().__init__(mode, config, model_dir)

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):

        assert self.mode == "training", "Create model in training mode."
        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = DataGenerator(train_dataset, self.config, shuffle=True,
                                         augmentation=augmentation)
        val_generator = DataGenerator(val_dataset, self.config, shuffle=True)

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)
        workers = 0

        self.keras_model.fit(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=workers > 1,
        )
        self.epoch = max(self.epoch, epochs)


model = MaskRCNNSingleThread(mode=mode, config=config, model_dir="logs")
if yaml_config["TRAINING"]:
    model_path = model.get_imagenet_weights()
else:
    model_path = yaml_config["TRAINED_MODEL"]
model.load_weights(model_path, by_name=True)

train_annos = segmentation_set(yaml_config["DATASET"], yaml_config["TRAIN_DIRS"], yaml_config["IMAGE_DIR"], yaml_config["MASK_DIR"])
dataset_train = MaskrcnnDataset(train_annos, yaml_config["CLASS_IDS"], yaml_config["CLASS_NAMES"])
dataset_train.load_dataset()
dataset_train.prepare()

train_annos = segmentation_set(yaml_config["DATASET"], yaml_config["VAL_DIRS"], yaml_config["IMAGE_DIR"], yaml_config["MASK_DIR"])
dataset_val = MaskrcnnDataset(train_annos, yaml_config["CLASS_IDS"], yaml_config["CLASS_NAMES"])
dataset_val.load_dataset()
dataset_val.prepare()

augmentation = iaa.SomeOf((0, 2), [
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Multiply((0.8, 1.5)),
    iaa.GaussianBlur(sigma=(0.0, 5.0))
])

if yaml_config["TRAINING"]:
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                augmentation=augmentation,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE/10,
                epochs=160,
                augmentation=augmentation,
                layers='all')
else:
    evaluate_det(model, dataset_val, config)
